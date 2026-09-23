<#
.SYNOPSIS
    Bootstrap a machine to build and run ClashRoyaleEnv: Python 3.11, both
    venvs, the pybind11 extension, and the Catch2 suite.

.DESCRIPTION
    Each step names the failure it guards against, since most look like
    something else:

      * a non-3.11 interpreter importing the .pyd fails "DLL load failed",
        which reads like a corrupt build;
      * the post-build copy fails MSB3073 if any Python process holds the .pyd;
      * MSBuild from Git Bash has /p: and /m mangled by MSYS path translation
        (MSB1008). That is why this is a .ps1.

    Run from PowerShell, from any directory.

.PARAMETER SkipPythonInstall
    Do not invoke winget. Use when 3.11 is already present or is managed by
    something else (pyenv-win, a company image).

.PARAMETER SkipVenvs
    Do not create or populate the venvs.

.PARAMETER SkipBuild
    Do not configure or build any C++.

.PARAMETER SkipTests
    Build, but do not run the C++ or pytest suites.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\setup_dev_env.ps1

.EXAMPLE
    # Rebuild the .pyd only, on a machine already set up:
    powershell -File tools\setup_dev_env.ps1 -SkipPythonInstall -SkipVenvs -SkipTests
#>
[CmdletBinding()]
param(
    [switch]$SkipPythonInstall,
    [switch]$SkipVenvs,
    [switch]$SkipBuild,
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

function Step($n, $msg) { Write-Host "`n=== [$n] $msg" -ForegroundColor Cyan }
function Ok($msg)       { Write-Host "    OK   $msg" -ForegroundColor Green }
function Warn($msg)     { Write-Host "    WARN $msg" -ForegroundColor Yellow }
function Die($msg)      { Write-Host "    FAIL $msg" -ForegroundColor Red; exit 1 }

Write-Host "ClashRoyaleEnv dev environment setup" -ForegroundColor White
Write-Host "repo: $RepoRoot"

# ---------------------------------------------------------------------------
Step 0 "Preflight -- probe, don't inherit"
# None of the toolchain is on PATH, so probe by absolute path.

$MSBuild = 'C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe'
$CMake   = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'

if (-not (Test-Path $MSBuild)) {
    # Try the other editions.
    $alt = Get-ChildItem 'C:\Program Files\Microsoft Visual Studio\2022' -Directory -ErrorAction SilentlyContinue |
           ForEach-Object { Join-Path $_.FullName 'MSBuild\Current\Bin\MSBuild.exe' } |
           Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($alt) { $MSBuild = $alt } else { Die "MSBuild not found. Install VS 2022 with the C++ workload." }
}
Ok "MSBuild  $MSBuild"

if (-not (Test-Path $CMake)) {
    $alt = Get-Command cmake -ErrorAction SilentlyContinue
    if ($alt) { $CMake = $alt.Source } else { Die "cmake not found (VS ships one; the C++ workload installs it)." }
}
Ok "cmake    $CMake"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Die "git not found. CMakeLists.txt FetchContent-s pybind11 and Catch2 from GitHub; the build cannot start without it."
}
Ok "git      $((Get-Command git).Source)"

# FetchContent clones from github.com at configure time; a TLS-interception
# failure surfaces as a CMake configure error, not a git one.
$probe = & git ls-remote --exit-code https://github.com/pybind/pybind11.git HEAD 2>&1
if ($LASTEXITCODE -ne 0) {
    Warn "cannot reach github.com over https:"
    Warn "  $probe"
    Warn "If this is TLS interception, the fix that worked on this repo was:"
    Warn "  git config --global http.sslBackend schannel"
    Warn "Continuing -- but the CMake configure step below will fail at FetchContent."
} else {
    Ok "github.com reachable (FetchContent can fetch pybind11 + Catch2)"
}

# ---------------------------------------------------------------------------
Step 1 "Python 3.11"
# 3.11 specifically: the .pyd is ABI-specific, and on another version the
# error names a DLL rather than a version.

function Find-Py311 {
    $p = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $p) { return $p.Trim() }
    return $null
}

$Py311 = Find-Py311
if ($Py311) {
    Ok "found $Py311"
} elseif ($SkipPythonInstall) {
    Die "Python 3.11 not found and -SkipPythonInstall was passed."
} else {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Die "Python 3.11 not found and winget is unavailable. Install from https://www.python.org/downloads/release/python-3119/ and re-run."
    }
    Write-Host "    installing Python.Python.3.11 via winget (this is a system change)..."
    winget install --id Python.Python.3.11 --exact --silent --accept-package-agreements --accept-source-agreements
    # The py launcher may not see a just-installed runtime in an open shell.
    $Py311 = Find-Py311
    if (-not $Py311) { Die "winget reported success but 'py -3.11' still fails. Open a NEW PowerShell and re-run with -SkipPythonInstall." }
    Ok "installed $Py311"
}

$pyver = & $Py311 -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ($pyver.Trim() -ne '3.11') { Die "expected 3.11, got $pyver" }

# ---------------------------------------------------------------------------
Step 2 "Virtual environments"
# Two, deliberately separate: python_ai/venv usually has a training run
# against it and must not be disturbed by perception's dependencies.

$venvs = @(
    @{ Name = 'python_ai';  Path = (Join-Path $RepoRoot 'python_ai\venv');   Req = (Join-Path $RepoRoot 'python_ai\requirements.txt') },
    @{ Name = 'perception'; Path = (Join-Path $RepoRoot 'perception\.venv'); Req = (Join-Path $RepoRoot 'perception\requirements.txt') }
)

if ($SkipVenvs) {
    Warn "skipped (-SkipVenvs)"
} else {
    foreach ($v in $venvs) {
        $py = Join-Path $v.Path 'Scripts\python.exe'
        if (Test-Path $py) {
            Ok "$($v.Name): venv already exists, reusing"
        } else {
            Write-Host "    creating $($v.Name) venv..."
            & $Py311 -m venv $v.Path
            if ($LASTEXITCODE -ne 0) { Die "venv creation failed for $($v.Name)" }
        }
        & $py -m pip install --upgrade pip --quiet
        Write-Host "    installing $($v.Name) requirements (torch is ~200 MB; this is the slow step)..."
        & $py -m pip install -r $v.Req
        if ($LASTEXITCODE -ne 0) {
            Warn "pip failed for $($v.Name)."
            if ($v.Name -eq 'python_ai') {
                Warn "If torch==2.13.0 is the failure: this box has no GPU and CLAUDE.md records the"
                Warn "CPU wheel. Retry with:"
                Warn "  $py -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu"
            }
            Die "dependency install failed"
        }
        Ok "$($v.Name): $py"
    }
}

# ---------------------------------------------------------------------------
Step 3 "Configure CMake into build_python/"
# The documented MSBuild commands build build_python\*.vcxproj, and this is
# the only thing that generates it: CMakePresets.json's Ninja presets emit to
# out/build/<preset> and produce no .vcxproj.
#
# PYTHON_EXECUTABLE is explicit because FindPython otherwise picks the newest
# interpreter on the box and builds an extension 3.11 cannot load.

$BuildDir = Join-Path $RepoRoot 'build_python'

if ($SkipBuild) {
    Warn "skipped (-SkipBuild)"
} else {
    & $CMake -S $RepoRoot -B $BuildDir -G "Visual Studio 17 2022" -A x64 `
        "-DPYTHON_EXECUTABLE=$Py311" "-DPython_EXECUTABLE=$Py311" "-DPython3_EXECUTABLE=$Py311"
    if ($LASTEXITCODE -ne 0) { Die "CMake configure failed (FetchContent needs github.com -- see preflight)" }
    Ok "configured $BuildDir"

    # -----------------------------------------------------------------------
    Step 4 "Build clash_royale_env.pyd"
    # POST_BUILD copies the .pyd into python_ai/. Windows will not overwrite a
    # mapped DLL, so a live interpreter holding it turns this into MSB3073.
    $holders = Get-Process python -ErrorAction SilentlyContinue
    if ($holders) {
        Warn "python processes are running (PIDs: $($holders.Id -join ', '))."
        Warn "If one of them has imported clash_royale_env, the POST_BUILD copy fails MSB3073."
        Warn "That is Windows refusing to overwrite a mapped DLL, NOT a broken compile."
    }

    & $MSBuild (Join-Path $BuildDir 'clash_royale_env.vcxproj') /p:Configuration=Release /p:Platform=x64 /m
    if ($LASTEXITCODE -ne 0) { Die "MSBuild failed. If the error is MSB3073, close any running Python and re-run with -SkipPythonInstall -SkipVenvs." }

    $pyd = Join-Path $RepoRoot 'python_ai\clash_royale_env.pyd'
    if (-not (Test-Path $pyd)) { Die "build reported success but $pyd is absent -- the POST_BUILD copy did not run" }
    Ok "built $pyd"

    # The import is the real acceptance test: a .pyd built against the wrong
    # interpreter passes every check above and fails here.
    $probe = Join-Path $RepoRoot 'python_ai\venv\Scripts\python.exe'
    if (Test-Path $probe) {
        & $probe -c "import python_ai, clash_royale_env as e; print('    OK   import works, observation_size=%d' % e.ClashRoyaleEnv([1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8],3600).observation_size())"
        if ($LASTEXITCODE -ne 0) { Die "the .pyd built but will not import -- check that PYTHON_EXECUTABLE really was 3.11" }
    } else {
        Warn "python_ai venv absent, skipping the import check"
    }
}

# ---------------------------------------------------------------------------
Step 5 "Test suites"
if ($SkipTests) {
    Warn "skipped (-SkipTests)"
} elseif ($SkipBuild) {
    Warn "skipped (nothing was built)"
} else {
    # Exactly one case fails by design (test_navigation_wedge.cpp's
    # [!shouldfail], pinning the open collision-wedge defect) and the runner
    # exits 0. A non-zero exit or a second failure is a regression.
    #
    # A new test file needs the build run twice: the first MSBuild re-globs
    # tests/core and tests/entities and links without the new file.
    & $MSBuild (Join-Path $BuildDir 'ClashRoyaleTests.vcxproj') /p:Configuration=Release /p:Platform=x64 /m
    if ($LASTEXITCODE -ne 0) { Die "test suite failed to build" }

    $exe = Join-Path $BuildDir 'Release\ClashRoyaleTests.exe'
    if (-not (Test-Path $exe)) { $exe = Join-Path $RepoRoot 'build_python\Release\ClashRoyaleTests.exe' }
    & $exe
    if ($LASTEXITCODE -ne 0) {
        Die "ClashRoyaleTests exited $LASTEXITCODE. Expected 0 with exactly one [!shouldfail] case failing."
    }
    Ok "C++ suite exit 0 (one [!shouldfail] case failing is correct)"

    foreach ($v in $venvs) {
        $py = Join-Path $v.Path 'Scripts\python.exe'
        if ($v.Name -eq 'perception') { $tests = Join-Path $RepoRoot 'perception\tests' }
        else                          { $tests = Join-Path $RepoRoot 'python_ai\tests' }
        if ((Test-Path $py) -and (Test-Path $tests)) {
            Write-Host "`n    pytest $($v.Name)..."
            & $py -m pytest $tests -q
            if ($LASTEXITCODE -ne 0) { Warn "$($v.Name) pytest reported failures -- see output above" }
        }
    }
}

# ---------------------------------------------------------------------------
Write-Host "`nDone." -ForegroundColor White
Write-Host @"

  Interpreters (never the bare 'python' -- that is 3.14 here and cannot load the .pyd):
    python_ai/venv/Scripts/python.exe
    perception/.venv/Scripts/python.exe

  Expected suite state:
    C++         exactly one [!shouldfail] case failing, exit 0
    python_ai   no failures (a couple of skips depend on the opening shuffle)
    perception  no failures
"@
