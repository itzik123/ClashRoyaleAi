#!/usr/bin/env bash
# One-shot setup for a fresh Ubuntu 22.04 GPU box. Idempotent -- safe to re-run.
#
# Nothing here generates build_python/; that directory is the Windows/MSVC
# route. On Linux we configure a separate build_linux/ so the two never collide
# in a shared checkout.
set -euo pipefail

REPO="${CLASH_REPO:-/opt/clash}"
PYVER=3.11

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# --- 1. system packages ----------------------------------------------------
log "system packages"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git curl ca-certificates \
    "python${PYVER}" "python${PYVER}-dev" "python${PYVER}-venv"

# --- 2. the venv -----------------------------------------------------------
# Python 3.11 ONLY. The extension is a 3.11-ABI module; on 3.12+ the import
# fails with a message that reads like a corrupt build and is a version
# mismatch. See CLAUDE.md, "Environment".
log "python ${PYVER} venv"
if [ ! -x "$REPO/.venv/bin/python" ]; then
    "python${PYVER}" -m venv "$REPO/.venv"
fi
"$REPO/.venv/bin/pip" install --quiet --upgrade pip wheel

log "torch (CUDA build -- NOT the +cpu wheel pinned in python_ai/requirements.txt)"
"$REPO/.venv/bin/pip" install --quiet \
    torch=="${CLASH_TORCH_VERSION:-2.13.0}" \
    --index-url "${CLASH_TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

log "remaining python dependencies"
"$REPO/.venv/bin/pip" install --quiet \
    numpy==2.4.6 gymnasium==1.3.0 tensorboard==2.21.0 wandb

# --- 3. the engine ---------------------------------------------------------
# -DPython_EXECUTABLE pins pybind11 to the 3.11 venv. Without it CMake finds the
# system python3 and produces an extension the venv cannot import -- the same
# class of failure as the .pyd/.so suffix bug, reached from a different angle.
log "configuring build_linux/"
cmake -S "$REPO" -B "$REPO/build_linux" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DPython_EXECUTABLE="$REPO/.venv/bin/python" \
      -DBUILD_TESTING_CLASHROYALE=ON

log "building the extension module"
# --target clash_royale_env only: the ClashRoyaleEnv executable pulls in
# main.cpp and TerminalRenderer.h, neither of which the training path uses.
cmake --build "$REPO/build_linux" --target clash_royale_env -j"$(nproc)"

log "building the Catch2 suite"
cmake --build "$REPO/build_linux" --target ClashRoyaleTests -j"$(nproc)"

# --- 4. gates --------------------------------------------------------------
log "C++ suite"
# The invariant is the SHAPE, not the count: exactly one [!shouldfail] failure
# (tests/core/test_navigation_wedge.cpp, the open collision-wedge defect) and
# exit 0. A second failure or a non-zero exit is a real regression.
"$REPO/build_linux/ClashRoyaleTests" || {
    echo "C++ suite exited non-zero -- that is a REAL regression, not the" >&2
    echo "expected [!shouldfail] case. Stopping." >&2
    exit 1
}

log "does the built module carry THIS engine?"
# The C++ suite cannot tell you this -- it links the headers directly and never
# loads the extension.
"$REPO/.venv/bin/python" "$REPO/tools/audit/verify_pyd.py"

log "python suite"
OMP_NUM_THREADS=1 "$REPO/.venv/bin/python" -m pytest "$REPO/python_ai/tests" -q

log "pre-flight gate"
OMP_NUM_THREADS=1 "$REPO/.venv/bin/python" -m python_ai.tools.validate_pipeline

# --- 5. hardware read-out --------------------------------------------------
log "sizing"
echo "physical cores : $(lscpu -p=Core,Socket | grep -cv '^#')"
echo "logical cpus   : $(nproc)"
echo "suggested CLASH_NUM_ENVS = physical cores - 2"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || \
    echo "NO GPU VISIBLE -- 87% of the wall clock is the PPO update; without a" \
         "GPU this migration buys almost nothing."
free -g | head -2

cat <<'EOF'

Next:
  1. Set CLASH_NUM_ENVS in cloud/clash.service to (physical cores - 2).
  2. Put WANDB_API_KEY (and any rclone secrets) in /opt/clash/.env, chmod 600.
  3. Re-measure where the hour goes -- do NOT inherit the laptop's 87/13 split:
       OMP_NUM_THREADS=1 .venv/bin/python -m python_ai.tools.profile_training \
           --mode async --episodes 60
  4. Dry-run the derived hyperparameters:
       .venv/bin/python cloud/launch.py --phase 1 --dry-run
  5. Install the units:  sudo cp cloud/clash*.service cloud/clash*.timer \
       /etc/systemd/system/ && sudo systemctl daemon-reload
EOF
