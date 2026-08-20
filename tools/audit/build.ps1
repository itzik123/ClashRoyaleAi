# Compiles a standalone audit instrument against the header-only engine.
#
# Deliberately NOT a CMake target: these are measurement harnesses, not part of
# the shipped build, and adding targets would force a reconfigure of the
# generated solution that the .pyd and the Catch2 suite both build from.
#
# Usage:  powershell -File tools/audit/build.ps1 bridge_audit
param(
    [Parameter(Mandatory = $true)][string]$Name
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$repo = Split-Path -Parent $PSScriptRoot | Split-Path -Parent

$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }

$src = Join-Path $PSScriptRoot "$Name.cpp"
if (-not (Test-Path $src)) { throw "no such source: $src" }

$outDir = Join-Path $PSScriptRoot "bin"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

$inc1 = Join-Path $repo "include\core"
$inc2 = Join-Path $repo "include\entities"
$inc3 = Join-Path $repo "include\rendering"
$exe = Join-Path $outDir "$Name.exe"
$obj = Join-Path $outDir "$Name.obj"

$cmd = "`"$vcvars`" >nul && cl /nologo /std:c++17 /O2 /EHsc /I `"$inc1`" /I `"$inc2`" /I `"$inc3`" `"$src`" /Fo:`"$obj`" /Fe:`"$exe`""
cmd /c $cmd
if ($LASTEXITCODE -ne 0) { throw "compile failed ($LASTEXITCODE)" }
Write-Host "built $exe"
