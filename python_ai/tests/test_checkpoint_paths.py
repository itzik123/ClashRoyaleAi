"""Checkpoint and log destinations must not depend on the working directory.

WHY THIS FILE EXISTS. `Phase1Trainer.__init__` read
`os.environ.get("CLASH_WEIGHTS", "model_weights.pth")` -- a BARE RELATIVE PATH --
and the checkpoints live in `python_ai/`, not at the repo root. So:

    launched from the repo root  -> os.path.exists() is False -> starts FRESH
    launched from python_ai/     -> os.path.exists() is True  -> RESUMES

Both are silent. A multi-day run either inherits a stale curriculum stage,
phase, entropy schedule and Adam state, or discards a good checkpoint, and
which one happens is decided by the directory somebody happened to `cd` into.
Measured 2026-08-25: `model_weights.pth` carried episode 7,063, phase
`random_opponent`, curriculum stage 4 and an already-decayed entropy
coefficient, from an engine two gameplay-affecting changes old.

`shipping.py` was already fixed for exactly this ("loaded from the package
directory the .pth files live in -- never from the caller's cwd, which is what
made a harness and a deployment able to load two different checkpoints under one
name"). The trainers were not, and `base_trainer` compounds it: `log_dir` is
also cwd-relative and line 217 `shutil.rmtree`s it on a non-resume start.

THE RULE PINNED HERE. Every destination resolves against an anchor that is
derived from `__file__`, never from `os.getcwd()`:

    *.pth                                  -> python_ai.PACKAGE_DIR
    runs/, historical_checkpoints/, ...    -> python_ai.REPO_ROOT

which is where each of those already physically lives, so nothing moves and no
existing layout changes -- the only thing that changes is that the working
directory stops being able to redirect them.
"""
import os

import pytest

import python_ai
from python_ai.rl import base_trainer, checkpointing
from python_ai.trainers import train, train_selfplay


# --------------------------------------------------------------------------
# the resolver itself, including the control that must fire
# --------------------------------------------------------------------------
def test_a_relative_name_is_anchored_and_an_absolute_one_is_left_alone():
    """The positive control. A resolver that returned its input unchanged would
    pass every 'is absolute' assertion below purely because the constants it is
    handed are already absolute -- so the relative case has to be exercised
    explicitly, and the absolute case has to be shown to survive untouched.
    """
    anchored = checkpointing.weights_path("model_weights.pth")
    assert os.path.isabs(anchored)
    assert anchored == os.path.join(python_ai.PACKAGE_DIR, "model_weights.pth")

    # an absolute override is a deliberate act and must be honoured verbatim
    explicit = os.path.join(os.sep, "tmp", "arm_b", "weights.pth")
    assert checkpointing.weights_path(explicit) == explicit

    run = checkpointing.run_path("runs/clash_royale_experiment")
    assert os.path.isabs(run)
    assert run == os.path.join(python_ai.REPO_ROOT, "runs", "clash_royale_experiment")


def test_the_two_anchors_are_different_directories():
    """Guards against 'fixing' this by collapsing both onto one base, which
    would silently relocate either the checkpoints or the TensorBoard runs.
    """
    assert python_ai.PACKAGE_DIR != python_ai.REPO_ROOT
    assert os.path.dirname(python_ai.PACKAGE_DIR) == python_ai.REPO_ROOT


@pytest.mark.parametrize("resolver", [checkpointing.weights_path,
                                      checkpointing.run_path])
def test_resolution_is_identical_from_two_different_working_directories(
        resolver, tmp_path, monkeypatch):
    """The actual failure, reproduced: same call, two `cd`s, one answer.

    Against the old bare-string behaviour the two results differed by the whole
    prefix, which is precisely how one launch resumed and another did not.
    """
    monkeypatch.chdir(python_ai.REPO_ROOT)
    from_root = resolver("model_weights.pth")

    monkeypatch.chdir(tmp_path)
    from_elsewhere = resolver("model_weights.pth")

    assert from_root == from_elsewhere
    assert os.path.isabs(from_root)
    # and it is not merely "absolute": it must not have picked up either cwd
    assert not from_root.startswith(str(tmp_path))


# --------------------------------------------------------------------------
# every destination a long run actually writes to
# --------------------------------------------------------------------------
def test_every_checkpoint_directory_is_absolute():
    for name in ("HISTORICAL_CHECKPOINT_DIR", "STAGE_CHECKPOINT_DIR"):
        value = getattr(checkpointing, name)
        assert os.path.isabs(value), f"{name} is cwd-relative: {value!r}"
        assert value.startswith(python_ai.REPO_ROOT)


def test_both_pipelines_name_absolute_weight_and_log_destinations():
    """`train.py` computes its two in `__init__` from the environment, so those
    are covered by the resolver tests above; these are the module- and
    class-level constants, which are frozen at import and cannot be fixed by a
    later `cd`.
    """
    for value in (train_selfplay.WEIGHT_PATH,
                  train_selfplay.BOOTSTRAP_FROM_PATH,
                  base_trainer.BaseTrainer.weight_path):
        assert os.path.isabs(value), f"cwd-relative weights: {value!r}"
        assert os.path.dirname(value) == python_ai.PACKAGE_DIR

    for value in (train_selfplay.Phase2Trainer.log_dir,
                  base_trainer.BaseTrainer.log_dir):
        assert os.path.isabs(value), f"cwd-relative log dir: {value!r}"
        assert value.startswith(python_ai.REPO_ROOT)


def test_the_phase1_defaults_resolve_the_same_way(monkeypatch, tmp_path):
    """Phase 1 reads CLASH_WEIGHTS / CLASH_LOGDIR at construction time.

    Constructing a `Phase1Trainer` would build eight environments, so the two
    expressions are exercised directly instead -- they are one line each and
    the point is the anchoring, not the trainer.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CLASH_WEIGHTS", raising=False)
    monkeypatch.delenv("CLASH_LOGDIR", raising=False)

    weights = checkpointing.weights_path(
        os.environ.get("CLASH_WEIGHTS", "model_weights.pth"))
    logs = checkpointing.run_path(
        os.environ.get("CLASH_LOGDIR", "runs/clash_royale_experiment"))

    assert weights == os.path.join(python_ai.PACKAGE_DIR, "model_weights.pth")
    assert logs.startswith(python_ai.REPO_ROOT)
    assert not weights.startswith(str(tmp_path))


def test_an_experiment_arm_can_still_redirect_both(monkeypatch, tmp_path):
    """The override exists so a smoke run cannot clobber the live checkpoint.
    Making the default absolute must not take that away.
    """
    arm = tmp_path / "arm_b.pth"
    monkeypatch.setenv("CLASH_WEIGHTS", str(arm))
    assert checkpointing.weights_path(os.environ["CLASH_WEIGHTS"]) == str(arm)


# --------------------------------------------------------------------------
# the PFSP pool, which reads the same constant from a third module
# --------------------------------------------------------------------------
def test_the_pfsp_pool_is_discoverable_from_any_working_directory(
        monkeypatch, tmp_path):
    """`league.discover_historical_checkpoints` globs HISTORICAL_CHECKPOINT_DIR.
    While that was relative, a run launched from the wrong directory saw an
    EMPTY pool -- and an empty pool is not an error, it is a league with only
    the four scripted bots in it. Silent, and it degrades the opponent
    distribution rather than crashing.
    """
    from python_ai.trainers import league

    monkeypatch.chdir(tmp_path)
    # a decoy that a cwd-relative glob would find instead of the real directory
    (tmp_path / "historical_checkpoints").mkdir()
    (tmp_path / "historical_checkpoints" / "999_pipeline1_ep00000042.pth").touch()

    found = league.discover_historical_checkpoints(current_episode=100000)
    assert all(not p.startswith(str(tmp_path)) for p in found), (
        "league picked up the working directory's decoy pool")


def test_train_module_no_longer_holds_a_bare_relative_default():
    """Reading the source, because the value itself is computed inside
    `__init__` and only exists on a constructed trainer. This is the assertion
    that would have failed before the fix.
    """
    src = open(train.__file__, encoding="utf-8").read()
    assert 'os.environ.get("CLASH_WEIGHTS", "model_weights.pth")' not in src \
        or "weights_path(" in src, \
        "CLASH_WEIGHTS default is still used unanchored"
