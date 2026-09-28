# Builds the Reflex Lab engine for the browser: web/lab/engine/engine.js +
# engine.wasm, from lab_wasm.cpp over the unchanged engine headers.
#
# Needs Emscripten (https://emscripten.org/docs/getting_started/downloads.html):
#   git clone https://github.com/emscripten-core/emsdk ; cd emsdk
#   .\emsdk install latest ; .\emsdk activate latest ; .\emsdk_env.ps1
# then, from the repo root:
#   powershell -NoProfile -File web/lab/engine/build.ps1
#   node tools/lab/parity.mjs          # must print 200/200 identical
#
# Run it from PowerShell, not Git Bash (MSYS rewrites the -s flags' paths).
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))

$emcc = (Get-Command emcc -ErrorAction SilentlyContinue).Source
if (-not $emcc -and $env:EMSDK) { $emcc = Join-Path $env:EMSDK "upstream\emscripten\emcc.bat" }
if (-not $emcc -or -not (Test-Path $emcc)) {
    throw "emcc not found. Install emsdk and run emsdk_env first (see the header of this file)."
}

$flags = @(
    "-std=c++20", "-O3", "-fexceptions", "--bind",
    "-I", (Join-Path $repo "include\core"), "-I", (Join-Path $repo "include\entities"),
    "-I", (Join-Path $repo "include\rendering"), "-I", $PSScriptRoot,
    (Join-Path $PSScriptRoot "lab_wasm.cpp"),
    "-o", (Join-Path $PSScriptRoot "engine.js"),
    "-sMODULARIZE=1", "-sEXPORT_NAME=createLabEngine", "-sENVIRONMENT=web,worker,node",
    "-sALLOW_MEMORY_GROWTH=1", "-sSTACK_SIZE=1048576"
)
& $emcc @flags
if ($LASTEXITCODE -ne 0) { throw "emcc failed ($LASTEXITCODE)" }
Get-Item (Join-Path $PSScriptRoot "engine.wasm"), (Join-Path $PSScriptRoot "engine.js") |
    ForEach-Object { "{0,-12} {1,10:N0} bytes" -f $_.Name, $_.Length }
