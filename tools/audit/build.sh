#!/usr/bin/env bash
# Compiles a standalone audit instrument against the header-only engine under
# WSL g++, for machines without MSVC. See build.ps1 for why these are not CMake
# targets.
#
# Absolute timings are not comparable with build.ps1's; ratios within a run are.
#
# Usage:  bash tools/audit/build.sh engine_profile && tools/audit/bin/engine_profile
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <instrument-name>   (e.g. engine_profile, soak, bridge_audit)" >&2
    exit 2
fi

name="$1"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
src="$here/$name.cpp"

[ -f "$src" ] || { echo "no such source: $src" >&2; exit 1; }
mkdir -p "$here/bin"

g++ -std=c++17 -O2 -DNDEBUG \
    -I "$repo/include/core" \
    -I "$repo/include/entities" \
    -I "$repo/include/rendering" \
    "$src" -o "$here/bin/$name"

echo "built $here/bin/$name"
