#!/usr/bin/env bash
# Keeps ONE training pipeline alive under systemd, across the phase-1 -> phase-2
# handoff and across restarts.
#
# TWO THINGS MAKE THIS MORE THAN `ExecStart=python train.py`
# ----------------------------------------------------------
# 1. THE LIFECYCLE TRAP. `train.py`'s `launch_pipeline2()` does
#    `subprocess.Popen(train_selfplay.py)` and then RETURNS, so `train.py` exits
#    0 while its child keeps running. Under systemd's default
#    `KillMode=control-group` that exit tears down the whole cgroup and reaps
#    the child it just launched -- a dead run with a clean exit code and nothing
#    in any log. `clash.service` sets `KillMode=process`; this script is what
#    then re-adopts the survivor.
#
# 2. THE CONFIG TRAP. `train_selfplay.py:483` constructs `Phase2Trainer()` with
#    NO cfg, so the auto-spawned child runs at `CLASH_NUM_ENVS` but with
#    `num_minibatches`/`lr`/`ppo_epochs` at their N=8 defaults -- the large-batch
#    regime `cloud/launch.py:scaled_config` exists to avoid. We retire that child
#    and relaunch phase 2 through `launch.py`.
#
#    Retiring it is safe and loses nothing: `Phase1Trainer.on_finish` calls
#    `save_checkpoint()` BEFORE `launch_pipeline2()` (train.py:472-477), so the
#    handoff checkpoint is already on disk, and the child's first act is a slow
#    `build_envs()` that writes nothing.
#
# TODO.md item 9 carries the one-line `os.execv` change that would delete both
# traps and most of this file.
set -u

REPO="${CLASH_REPO:-/opt/clash}"
PY="${CLASH_PYTHON:-$REPO/.venv/bin/python}"
PHASE2_PAT="python_ai.trainers.train_selfplay|trainers/train_selfplay.py"

cd "$REPO" || { echo "supervise: cannot cd to $REPO" >&2; exit 1; }

log() { printf '[supervise %s] %s\n' "$(date -Is)" "$*"; }

phase2_pid() { pgrep -f "$PHASE2_PAT" | head -1; }

# Block until the given pid exits, without owning it. `tail --pid` is the
# portable way to wait on a non-child process.
track() {
  log "tracking existing phase-2 process (pid $1)"
  exec tail --pid="$1" -f /dev/null
}

# --- restart path: a phase 2 that outlived a previous supervisor ------------
pid="$(phase2_pid)"
if [ -n "$pid" ]; then
  track "$pid"
fi

# --- restart path: phase 1 already finished in an earlier life -------------
if [ -f "$REPO/python_ai/model_weights_selfplay.pth" ]; then
  log "phase-2 checkpoint present -- starting pipeline 2 directly"
  exec "$PY" -u "$REPO/cloud/launch.py" --phase 2
fi

# --- normal path: run phase 1 ----------------------------------------------
log "starting pipeline 1"
"$PY" -u "$REPO/cloud/launch.py" --phase 1
rc=$?
log "pipeline 1 exited rc=$rc"

# launch.py refuses to start on a too-narrow curriculum window (rc=2). That is a
# configuration error, not a crash -- restarting would loop forever on it.
if [ "$rc" -eq 2 ]; then
  log "pipeline 1 refused to start (configuration). Not restarting."
  exit 0
fi

# Give train.py's Popen a moment to appear before deciding it did not happen.
sleep 15
pid="$(phase2_pid)"
if [ -n "$pid" ]; then
  log "handoff detected (pid $pid) -- retiring the unconfigured child"
  pkill -f "$PHASE2_PAT"
  # Wait for it to actually go, then escalate once.
  for _ in $(seq 1 20); do
    [ -z "$(phase2_pid)" ] && break
    sleep 1
  done
  [ -n "$(phase2_pid)" ] && pkill -9 -f "$PHASE2_PAT" && sleep 2
  log "relaunching pipeline 2 with the scaled config"
  exec "$PY" -u "$REPO/cloud/launch.py" --phase 2
fi

# Phase 1 died without handing off. Let Restart=always bring us back.
log "no handoff -- exiting nonzero so systemd restarts"
exit "${rc:-1}"
