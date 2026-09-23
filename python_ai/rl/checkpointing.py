"""Where a run's weights go, and which of them another run may play against.

  the live checkpoint     resumed from; carries optimizer + training state
  historical_checkpoints/ pipeline 2's PFSP opponent pool, ordered by mtime
  stage_checkpoints/      diagnostic only, one per curriculum-stage transition

Kept separate: the pool is read as a weakest-to-strongest ladder by save order,
so extra files there would change what self-play samples. Every destination is
anchored, never cwd-relative, via `weights_path` and `run_path`.
"""
import os
import tempfile
import time

import torch

import python_ai


#: On Windows `os.replace` fails while any other handle has the destination
#: open, and monitor_run.py opens the checkpoint on every check.
_REPLACE_RETRIES = 20
_REPLACE_BACKOFF_S = 0.5


def _replace_with_retry(src, dst):
    import time as _time
    for attempt in range(_REPLACE_RETRIES):
        try:
            return os.replace(src, dst)
        except PermissionError:
            if attempt == _REPLACE_RETRIES - 1:
                raise
            _time.sleep(_REPLACE_BACKOFF_S)


def atomic_save(payload, path, keep_previous=False):
    """`torch.save` that cannot leave a truncated file at `path`.

    Writes a temp file in the same directory, fsyncs it, then renames: a
    same-volume `os.replace` is atomic on Windows and POSIX, so a reader sees
    the whole old file or the whole new one. The live checkpoint is the only
    copy of a multi-day run. Snapshots use it too, since a partial file in the
    pool directory would crash phase 2 when sampled.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    # Two independent guards keep an orphan out of the pool: the leading dot
    # (glob's `*` skips dotfiles) and the suffix (the pool globs `*.pth`). A
    # file left by SIGKILL is permanent, so either alone is too fragile.
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".partial", dir=directory)
    os.close(fd)
    try:
        with open(tmp, "wb") as fh:
            torch.save(payload, fh)
            fh.flush()
            # fsync before the rename, or the rename can be durable while the
            # content is still in the page cache.
            os.fsync(fh.fileno())
        if keep_previous and os.path.exists(path):
            # One backup generation. A copy, not a rename: renaming a file
            # another process has open fails on Windows.
            import shutil
            shutil.copy2(path, path + ".prev")
        _replace_with_retry(tmp, path)
    except BaseException:
        # BaseException: KeyboardInterrupt is the likeliest interruption
        # mid-save.
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def weights_path(name):
    """A `.pth` destination, anchored on the package directory.

    An absolute name is returned untouched: an explicit CLASH_WEIGHTS is how an
    experiment arm stays off the live checkpoint.
    """
    return name if os.path.isabs(name) else os.path.join(
        python_ai.PACKAGE_DIR, name)


def run_path(name):
    """A run-artifact destination (TensorBoard runs, snapshot pools), anchored on
    the repository root.

    A different anchor from `weights_path` on purpose; `base_trainer` deletes
    `log_dir` on a non-resume start, so a wrong anchor here is destructive.
    """
    return name if os.path.isabs(name) else os.path.join(
        python_ai.REPO_ROOT, *name.split("/"))


# The run's CLASH_* configuration, stamped into every checkpoint. These
# settings change where a run writes, how often, with how many workers or from
# which seed, not what it learns; a different value on resume is reported, not
# flagged.
OPERATIONAL_SETTINGS = frozenset({
    "CLASH_WEIGHTS", "CLASH_LOGDIR", "CLASH_SAVE_EVERY", "CLASH_SEED",
    "CLASH_NUM_ENVS",
})
#: Compared on its resolved value in BaseTrainer.restore_common: names and ids
#: spell the same deck differently.
_COMPARED_ELSEWHERE = frozenset({"CLASH_DECK"})


def clash_settings(environ=None):
    """Every CLASH_* environment variable in force, as a sorted dict.

    Each is read at import, so a resumed run is configured by whatever the
    relaunching shell holds. Recording them makes a changed objective visible.
    """
    env = os.environ if environ is None else environ
    return {k: env[k] for k in sorted(env) if k.startswith("CLASH_")}


def settings_drift(saved, current):
    """(changed, operational): (key, saved, current) triples that differ.

    Unset is None on either side, so adding or removing a variable counts.
    CLASH_DECK is left to the resolved-deck check.
    """
    changed, operational = [], []
    for key in sorted(set(saved) | set(current)):
        if key in _COMPARED_ELSEWHERE:
            continue
        a, b = saved.get(key), current.get(key)
        if a == b:
            continue
        (operational if key in OPERATIONAL_SETTINGS else changed).append((key, a, b))
    return changed, operational


# Weights-only snapshots of past policies for pipeline 2's PFSP pool, saved
# throughout a run. Timestamp-prefixed because both pipelines write here on
# unrelated episode scales, and the pool is ordered by save time.
HISTORICAL_CHECKPOINT_DIR = run_path("historical_checkpoints")

# MIN_OPPONENT_AGE_EPISODES in train_selfplay.py is kept at 3x this value;
# change them together. Denser snapshots dilute every opponent's PFSP share and
# each worker's per-opponent win-rate estimate.
HISTORICAL_CHECKPOINT_INTERVAL_EPISODES = 2000

# Diagnostic-only snapshots, one per curriculum-stage transition, captured
# before the opponent gets harder, so one policy can later be replayed against
# several stages' opponents. Nothing trains against them.
STAGE_CHECKPOINT_DIR = run_path("stage_checkpoints")

def _teacher_table_size():
    from python_ai.rl.curriculum import CURRICULUM_STAGES
    return len(CURRICULUM_STAGES)


def save_stage_snapshot(net, directory, stage, episodes_completed, teacher_stage, reason):
    """Weights-only snapshot tagged with the curriculum state it was taken at.

    Never resumed from, so no optimizer state. The metadata records which
    opponent strength the policy was trained against.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"stage{stage}_ep{episodes_completed:08d}.pth")
    atomic_save({
        "model": net.state_dict(),
        "curriculum_stage": stage,
        # Stamped so a snapshot copied over a resume target is not remapped as
        # a legacy six-rung index.
        "teacher_table_size": _teacher_table_size(),
        "episodes_completed": episodes_completed,
        "teacher_stage": teacher_stage,
        "reason": reason,
    }, path)
    print(f">>> Stage snapshot saved to {path} ({reason}, teacher_stage={teacher_stage})")
    return path


def save_historical_snapshot(net, episodes_completed, pipeline,
                             directory=HISTORICAL_CHECKPOINT_DIR):
    """Bare weights into the shared PFSP pool, timestamp-prefixed.

    `pipeline` is embedded because train_selfplay's age gate matches on it.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(
        directory,
        f"{int(time.time() * 1000)}_{pipeline}_ep{episodes_completed:08d}.pth")
    atomic_save({"model": net.state_dict()}, path)
    print(f">>> Historical snapshot saved to {path}")
    return path
