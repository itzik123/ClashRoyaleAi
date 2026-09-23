"""Where a run's weights go, and which of them another run may play against.

Three destinations, deliberately separate directories:

  the live checkpoint     resumed from; carries optimizer + training state
  historical_checkpoints/ pipeline 2's PFSP opponent pool, ordered by mtime
  stage_checkpoints/      diagnostic-only, one per curriculum-stage transition

Mixing the last two would silently change which opponents self-play samples --
the pool is read as a weakest-to-strongest ladder by save order.

EVERY DESTINATION IS ANCHORED, NEVER CWD-RELATIVE (2026-08-25). All of these
used to be bare relative strings, so which directory a run was launched from
decided whether it resumed or started fresh, and whether the PFSP pool was
found at all. `weights_path` and `run_path` below are the single resolution
point; see `python_ai/tests/test_checkpoint_paths.py` for the measured failure.
"""
import os
import tempfile
import time

import torch

import python_ai


#: Retries for `os.replace` on Windows, where it fails with PermissionError while
#: ANY other handle has the destination open -- measured (audit 08): a reader
#: holding the checkpoint made the replace raise and the trainer die.
#: `monitor_run.py` opens the checkpoint on every check, so this is not rare.
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
    """`torch.save` that cannot leave a TRUNCATED file at `path`.

    THE LIVE CHECKPOINT IS THE RUN. `save_checkpoint` used to overwrite it in
    place, so an interruption during the write -- Ctrl-C, an OOM kill, a full
    disk, a power cut -- left `model_weights.pth` truncated, and it is the only
    copy. A multi-day run was destroyed at the exact moment it tried to
    preserve itself. The window is not negligible either: the payload carries
    the model AND Adam's two moment buffers, so roughly three times the
    parameter count, rewritten on every save for the life of the run.

    Write to a temp file in the SAME directory, flush it all the way to the
    disk, then rename. `os.replace` is atomic for a same-volume rename on both
    Windows and POSIX, so a reader either sees the whole old file or the whole
    new one and never a partial. Same directory is load-bearing: a rename
    across volumes is a copy, and copies are not atomic.

    The snapshot helpers use it too, for a different failure -- they mint a
    unique filename, so a torn write cannot destroy an existing file, but it
    CAN leave a partial one in a pool directory that `league.py` enumerates and
    torch.loads wholesale, crashing phase 2 on whichever reset samples it.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    # TWO independent reasons this cannot be discovered as a pool opponent:
    # the leading dot (Python's glob excludes dotfiles from `*` patterns) and
    # the suffix (the pool globs `*.pth`). Either alone would do; relying on
    # only one makes a rename of the other a silent regression, and the file
    # that survives a SIGKILL here is permanent -- the cleanup handler cannot
    # run, so an orphan would break every phase-2 run afterwards.
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".partial", dir=directory)
    os.close(fd)
    try:
        with open(tmp, "wb") as fh:
            torch.save(payload, fh)
            fh.flush()
            # fsync before the rename: the rename can otherwise be durable
            # while the CONTENT it points at is still only in the page cache,
            # which is the same truncated file by a slower route.
            os.fsync(fh.fileno())
        if keep_previous and os.path.exists(path):
            # ONE backup generation. There was a single copy of a multi-day
            # run's training state (audit 08, gap 6). A COPY, not a rename: a
            # rename of a file another process has open fails on Windows, and
            # the copy leaves the live file in place if anything below fails.
            import shutil
            shutil.copy2(path, path + ".prev")
        _replace_with_retry(tmp, path)
    except BaseException:
        # BaseException, not Exception: KeyboardInterrupt is the single most
        # likely way a training run is interrupted mid-save.
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def weights_path(name):
    """A `.pth` destination, anchored on the package directory.

    `python_ai/` is where the checkpoints and the compiled engine already live,
    and where `shipping.load_shipping_net` already resolves from -- so this
    moves nothing, it only stops `os.getcwd()` from being able to redirect it.
    An already-absolute name is returned untouched, because an explicit
    `CLASH_WEIGHTS` is a deliberate act (an experiment arm redirecting away
    from the live checkpoint) and second-guessing it would defeat the override.
    """
    return name if os.path.isabs(name) else os.path.join(
        python_ai.PACKAGE_DIR, name)


def run_path(name):
    """A run-artifact destination (TensorBoard runs, snapshot pools), anchored
    on the repository root, which is where `runs/`, `replays/` and
    `historical_checkpoints/` already sit.

    Deliberately a DIFFERENT anchor from `weights_path`. Collapsing the two
    onto one base would relocate either the checkpoints or the run artifacts,
    and `base_trainer` `shutil.rmtree`s `log_dir` on a non-resume start -- so a
    wrong anchor here is destructive, not merely untidy.
    """
    return name if os.path.isabs(name) else os.path.join(
        python_ai.REPO_ROOT, *name.split("/"))


# --------------------------------------------------------------------------
# The run's CLASH_* configuration, stamped into every checkpoint (TODO 00.9)
# --------------------------------------------------------------------------
#: Settings that change WHERE a run writes, how often, with how many workers
#: or from which seed -- not WHAT it learns. A different value on resume is
#: reported, not flagged.
OPERATIONAL_SETTINGS = frozenset({
    "CLASH_WEIGHTS", "CLASH_LOGDIR", "CLASH_SAVE_EVERY", "CLASH_SEED",
    "CLASH_NUM_ENVS",
})
#: Compared elsewhere, on its RESOLVED value: `CLASH_DECK` accepts names or ids,
#: so "hog rider,..." and "15,..." are the same deck and different strings.
#: `BaseTrainer.restore_common` compares the resolved ids and warns on its own.
_COMPARED_ELSEWHERE = frozenset({"CLASH_DECK"})


def clash_settings(environ=None):
    """Every CLASH_* environment variable in force, as a plain sorted dict.

    Stamped into the checkpoint because every one of these is read at IMPORT
    (the reward weights, gamma, the anneal, the scenario mix, the deck pool...),
    so the process that resumes a run is configured by whatever the operator's
    shell holds at relaunch -- and only the deck was recorded. A resume under a
    different CLASH_GAMMA continued silently under a different objective.
    """
    env = os.environ if environ is None else environ
    return {k: env[k] for k in sorted(env) if k.startswith("CLASH_")}


def settings_drift(saved, current):
    """(changed, operational): (key, saved, current) triples that differ.

    Unset is None on either side, so adding or removing a variable counts.
    `changed` holds the ones that alter the run; `operational` the ones in
    OPERATIONAL_SETTINGS. CLASH_DECK is left to the resolved-deck check.
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


# Historical self-play (pipeline #2, train_selfplay.py) needs a library of past
# versions of this same policy to play against, weakest to strongest -- these
# are saved here as bare weights-only snapshots (never resumed-from for further
# gradient training, so no optimizer/training-state needed) periodically
# throughout THIS training run, not just at the end, so the library already
# spans a useful weak->strong range by the time pipeline #2 starts. Filenames
# are timestamp-prefixed rather than keyed by this run's own episode count --
# train_selfplay.py sorts the shared folder by save order (mtime) to build a
# single weakest-to-strongest queue, since it saves its own new (and by then
# much stronger) snapshots into the same folder as pipeline #2 progresses, and
# those two runs' episode counters aren't on the same scale.
HISTORICAL_CHECKPOINT_DIR = run_path("historical_checkpoints")

# Lowered 5000 -> 2000 on 2026-08-09, as a RE-DENOMINATION rather than a change
# of intent. The 2026-08-07 movement-speed fix left gradient steps per hour
# unchanged but cut episodes per hour 2,873 -> 1,301, so one episode now carries
# ~2.2x more transitions and ~2.2x more policy change. 5,000 episodes had come
# to mean what ~11,000 used to; 5000 / 2.2 ~= 2,270, rounded to 2,000.
#
# Deliberately NOT 1,000, which was the other candidate: that would make the
# pool ~5x denser than the original design, and 5,000 was itself a judgement
# call rather than a measured optimum, so there is nothing to justify
# overshooting it. Two costs bound this from above -- every extra pool member
# dilutes the PFSP share of every other (which is what already forced
# DEFENSIVE_SCRIPTED_MIN_WEIGHT up to 0.8), and each worker keeps its OWN local
# per-opponent win-rate estimate, so more members means fewer games each and a
# noisier (1 - winrate)^2 weighting.
#
# MIN_OPPONENT_AGE_EPISODES in train_selfplay.py is kept at 3x this value; see
# its comment. The two are coupled and changing one alone silently changes
# which snapshots are eligible.
HISTORICAL_CHECKPOINT_INTERVAL_EPISODES = 2000

# Diagnostic-only snapshots, one per curriculum-stage transition: the exact
# policy that just cleared a stage's 80%-over-100-episodes gate, captured
# BEFORE the opponent gets harder.
#
# Deliberately a SEPARATE directory from HISTORICAL_CHECKPOINT_DIR, not extra
# files in it: that one is pipeline #2's PFSP opponent pool, ordered by mtime
# as a weakest->strongest ladder, so dropping additional checkpoints in would
# silently change which opponents self-play samples and how often. These are
# for offline analysis only and nothing ever trains against them.
#
# The motivating question, which could not be answered on the previous run
# because the checkpoints had already been deleted: the narrow policy measured
# at stage 4 (8 of 288 placement cells, 5 of 8 deck cards, one lane) was
# measured ONLY against that stage's 1.4x-elixir opponent. Whether that
# narrowness is a pathological collapse or a rational response to being
# out-elixired is decidable -- but only by replaying ONE fixed policy against
# SEVERAL stages' opponents, which requires having kept the per-stage weights.
STAGE_CHECKPOINT_DIR = run_path("stage_checkpoints")

def _teacher_table_size():
    from python_ai.rl.curriculum import CURRICULUM_STAGES
    return len(CURRICULUM_STAGES)


def save_stage_snapshot(net, directory, stage, episodes_completed, teacher_stage, reason):
    """Weights-only snapshot tagged with the curriculum state it was taken at.

    Weights only (no optimizer/training state) on purpose -- these are never
    resumed from, only loaded for offline behavioral probes, so carrying Adam's
    buffers would triple the file size for nothing. The metadata is what makes
    a probe reproducible: it records which opponent strength this policy was
    actually trained against, so a later comparison can replay it against a
    DIFFERENT multiplier and attribute the difference correctly.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"stage{stage}_ep{episodes_completed:08d}.pth")
    atomic_save({
        "model": net.state_dict(),
        "curriculum_stage": stage,
        # Stamped so a snapshot copied over a resume target is not remapped as
        # a legacy six-rung index (audit 04 C4).
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

    Timestamp-prefixed rather than keyed by episode count because BOTH pipelines
    drop snapshots into this one folder on two unrelated episode scales -- see
    HISTORICAL_CHECKPOINT_DIR. `pipeline` ("pipeline1"/"pipeline2") is still
    embedded because train_selfplay's age gate matches on it: only pipeline 2's
    own recent snapshots are too young to be eligible.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(
        directory,
        f"{int(time.time() * 1000)}_{pipeline}_ep{episodes_completed:08d}.pth")
    atomic_save({"model": net.state_dict()}, path)
    print(f">>> Historical snapshot saved to {path}")
    return path
