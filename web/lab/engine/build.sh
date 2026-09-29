#!/usr/bin/env bash
# The same build as build.ps1, for Linux/macOS and CI (.github/workflows/lab-pages.yml).
# Needs emcc on PATH (emsdk activated).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
# -fwasm-exceptions: 2.5x faster than -fexceptions (see build.ps1).
emcc -std=c++20 -O3 -fwasm-exceptions --bind \
  -I "$repo/include/core" -I "$repo/include/entities" -I "$repo/include/rendering" -I "$here" \
  "$here/lab_wasm.cpp" -o "$here/engine.js" \
  -sMODULARIZE=1 -sEXPORT_NAME=createLabEngine -sENVIRONMENT=web,worker,node \
  -sALLOW_MEMORY_GROWTH=1 -sSTACK_SIZE=1048576
ls -l "$here/engine.js" "$here/engine.wasm"
