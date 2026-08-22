#!/usr/bin/env bash
# Compiles a standalone audit instrument against the header-only engine, under
# WSL g++. The companion of build.ps1, which needs MSVC.
#
# BOTH EXIST BECAUSE THE TOOLCHAIN DIFFERS BETWEEN THE MACHINES THIS REPO IS
# WORKED ON, which CLAUDE.md documents at length: one box has a full MSVC
# install and no WSL, the other has WSL 2.6 with g++ 13.3 and no MSVC at all.
# On the second box build.ps1 cannot run, and before this file the audit
# instruments were simply unavailable there.
#
# Deliberately NOT a CMake target, for the same reason as build.ps1: these are
# measurement harnesses, and adding targets would force a reconfigure of the
# generated solution that the .pyd and the Catch2 suite both build from.
#
# NOTE ON COMPARING NUMBERS ACROSS THE TWO ROUTES: g++/glibc and MSVC/UCRT do
# not produce the same absolute timings, and the two machines' CPUs differ by
# more than the compilers do. Ratios within one run are comparable; absolute
# milliseconds across runs on different boxes are not.
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
