# Builds tools/lab/out/lab_cli.exe: the Reflex Lab engine (web/lab/engine/
# lab_engine.h) compiled natively against the header-only engine, the way
# tools/audit/build.ps1 builds its instruments -- deliberately not a CMake
# target, so it cannot force a reconfigure of the solution the .pyd builds from.
#
# Usage:  powershell -NoProfile -File tools/lab/build_cli.ps1
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }

$outDir = Join-Path $PSScriptRoot "out"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

$incs = @("include\core", "include\entities", "include\rendering", "web\lab\engine") |
    ForEach-Object { "/I `"$(Join-Path $repo $_)`"" }
$src = Join-Path $PSScriptRoot "lab_cli.cpp"
$exe = Join-Path $outDir "lab_cli.exe"
$obj = Join-Path $outDir "lab_cli.obj"

$cmd = "`"$vcvars`" >nul && cl /nologo /std:c++20 /O2 /EHsc /bigobj $($incs -join ' ') `"$src`" /Fo:`"$obj`" /Fe:`"$exe`""
cmd /c $cmd
if ($LASTEXITCODE -ne 0) { throw "compile failed ($LASTEXITCODE)" }
Write-Host "built $exe"
