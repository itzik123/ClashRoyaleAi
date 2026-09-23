"""Checkpoint and log destinations do not depend on the working directory.

    *.pth                                  -> python_ai.PACKAGE_DIR
    runs/, historical_checkpoints/, ...    -> python_ai.REPO_ROOT

Both anchors derive from `__file__`, never `os.getcwd()`, and are where each
already lived. A cwd-relative weights path silently decided whether a run
resumed or started fresh depending on where it was launched.
"""
import os

import pytest

import python_ai
from python_ai.rl import base_trainer, checkpointing
from python_ai.trainers import train, train_selfplay


# --- the resolver, including the control that must fire ---
def test_a_relative_name_is_anchored_and_an_absolute_one_is_left_alone():
    """Positive control: a resolver returning its input unchanged would pass every
    "is absolute" check below, so the relative case is exercised explicitly and
    the absolute case shown untouched.
    """
    anchored = checkpointing.weights_path("model_weights.pth")
    assert os.path.isabs(anchored)
    assert anchored == os.path.join(python_ai.PACKAGE_DIR, "model_weights.pth")

    # An absolute override is deliberate and honoured verbatim.
    explicit = os.path.join(os.sep, "tmp", "arm_b", "weights.pth")
    assert checkpointing.weights_path(explicit) == explicit

    run = checkpointing.run_path("runs/clash_royale_experiment")
    assert os.path.isabs(run)
    assert run == os.path.join(python_ai.REPO_ROOT, "runs", "clash_royale_experiment")


def test_the_two_anchors_are_different_directories():
    """Collapsing both anchors onto one base would silently relocate the
    checkpoints or the runs.
    """
    assert python_ai.PACKAGE_DIR != python_ai.REPO_ROOT
    assert os.path.dirname(python_ai.PACKAGE_DIR) == python_ai.REPO_ROOT


@pytest.mark.parametrize("resolver", [checkpointing.weights_path,
                                      checkpointing.run_path])
def test_resolution_is_identical_from_two_different_working_directories(
        resolver, tmp_path, monkeypatch):
    """The actual failure: the same call from two working directories gives one
    answer.
    """
    monkeypatch.chdir(python_ai.REPO_ROOT)
    from_root = resolver("model_weights.pth")

    monkeypatch.chdir(tmp_path)
    from_elsewhere = resolver("model_weights.pth")

    assert from_root == from_elsewhere
    assert os.path.isabs(from_root)
    # ...and it must not have picked up either cwd.
    assert not from_root.startswith(str(tmp_path))


# --- every destination a long run writes to ---
def test_every_checkpoint_directory_is_absolute():
    for name in ("HISTORICAL_CHECKPOINT_DIR", "STAGE_CHECKPOINT_DIR"):
        value = getattr(checkpointing, name)
        assert os.path.isabs(value), f"{name} is cwd-relative: {value!r}"
        assert value.startswith(python_ai.REPO_ROOT)


def test_both_pipelines_name_absolute_weight_and_log_destinations():
    """Module- and class-level constants are frozen at import and cannot be fixed
    by a later `cd`; the ones computed in `__init__` are covered by the
    resolver tests.
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
    """Phase 1 reads CLASH_WEIGHTS / CLASH_LOGDIR at construction; building a
    trainer would build eight envs, so the two expressions are exercised
    directly.
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
    """The override lets an experiment avoid clobbering the live checkpoint."""
    arm = tmp_path / "arm_b.pth"
    monkeypatch.setenv("CLASH_WEIGHTS", str(arm))
    assert checkpointing.weights_path(os.environ["CLASH_WEIGHTS"]) == str(arm)


# --- the PFSP pool ---
def test_the_pfsp_pool_is_discoverable_from_any_working_directory(
        monkeypatch, tmp_path):
    """A cwd-relative pool directory would yield an empty pool: not an error, but
    a league of only the scripted bots.
    """
    from python_ai.trainers import league

    monkeypatch.chdir(tmp_path)
    # A decoy a cwd-relative glob would find instead of the real directory.
    (tmp_path / "historical_checkpoints").mkdir()
    (tmp_path / "historical_checkpoints" / "999_pipeline1_ep00000042.pth").touch()

    found = league.discover_historical_checkpoints(current_episode=100000)
    assert all(not p.startswith(str(tmp_path)) for p in found), (
        "league picked up the working directory's decoy pool")


def test_train_module_no_longer_holds_a_bare_relative_default():
    """Read from source, since the value exists only on a constructed trainer.
    """
    src = open(train.__file__, encoding="utf-8").read()
    assert 'os.environ.get("CLASH_WEIGHTS", "model_weights.pth")' not in src \
        or "weights_path(" in src, \
        "CLASH_WEIGHTS default is still used unanchored"


# --- torn writes ---
# The live checkpoint is the run: a bare `torch.save` interrupted mid-write
# (Ctrl-C, OOM kill, full disk) leaves it truncated, the only copy. Pool
# snapshots matter too: `league.py` loads every entry, so one truncated file
# crashes phase 2 when sampled.

def test_an_interrupted_save_leaves_the_previous_checkpoint_intact(tmp_path,
                                                                   monkeypatch):
    import torch
    from python_ai.rl import checkpointing

    dest = tmp_path / "model_weights.pth"
    torch.save({"model": {"w": torch.ones(3)}, "episodes_completed": 5_000},
               dest)

    real_save = torch.save
    calls = {"n": 0}

    def exploding_save(obj, f, *a, **kw):
        """Write a few bytes, then die: a torn write. `f` is a handle, since an
        atomic implementation opens its own temp file.
        """
        calls["n"] += 1
        f.write(b"\x80\x02}")               # a plausible pickle prefix
        raise KeyboardInterrupt("killed mid-save")

    monkeypatch.setattr(torch, "save", exploding_save)
    with pytest.raises(KeyboardInterrupt):
        checkpointing.atomic_save({"model": {"w": torch.zeros(3)}}, str(dest))
    monkeypatch.setattr(torch, "save", real_save)

    assert calls["n"] == 1, "the save was never attempted"
    restored = torch.load(dest, weights_only=False)
    assert restored["episodes_completed"] == 5_000, (
        "the previous checkpoint was destroyed by a torn write")


def test_a_successful_atomic_save_round_trips(tmp_path):
    import torch
    from python_ai.rl import checkpointing

    dest = tmp_path / "w.pth"
    checkpointing.atomic_save({"episodes_completed": 42}, str(dest))
    assert torch.load(dest, weights_only=False)["episodes_completed"] == 42


def test_an_atomic_save_leaves_no_temporary_files_behind(tmp_path):
    """A pool directory is enumerated by glob, so a leftover temp file would be
    loaded as an opponent.
    """
    import torch
    from python_ai.rl import checkpointing

    dest = tmp_path / "w.pth"
    checkpointing.atomic_save({"episodes_completed": 1}, str(dest))
    assert [p.name for p in tmp_path.iterdir()] == ["w.pth"]


def test_an_interrupted_save_leaves_no_partial_file_in_a_pool(tmp_path,
                                                              monkeypatch):
    """Nothing may appear in the directory until the payload is complete."""
    import torch
    from python_ai.rl import checkpointing

    def exploding_save(obj, f, *a, **kw):
        f.write(b"\x80\x02")
        raise RuntimeError("disk full")

    monkeypatch.setattr(torch, "save", exploding_save)
    with pytest.raises(RuntimeError):
        checkpointing.atomic_save({"model": {}}, str(tmp_path / "snap.pth"))
    assert list(tmp_path.iterdir()) == [], (
        f"a partial file survived: {[p.name for p in tmp_path.iterdir()]}")


def test_no_module_in_the_package_writes_a_checkpoint_unatomically():
    """Package-wide: any bare `torch.save` (e.g. a burst snapshot into the shared
    PFSP pool) can deposit a corrupt file another run later loads.
    """
    import pathlib
    import re

    import python_ai
    from python_ai.rl import checkpointing

    import inspect
    # `atomic_save`'s own body holds the one legitimate torch.save.
    exempt_body = inspect.getsource(checkpointing.atomic_save).splitlines()
    root = pathlib.Path(python_ai.PACKAGE_DIR)
    offenders = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(("tests/", "venv/", "archive")):
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.search(r"(?<![\w.])torch\.save\(", code) and line not in exempt_body:
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, (
        "checkpoint written without atomic_save:\n" + "\n".join(offenders))


def test_the_temp_file_can_never_be_discovered_as_a_pool_opponent(tmp_path,
                                                                  monkeypatch):
    """The in-flight temp file must not match the pool's `*.pth` glob, or a
    SIGKILL leaves a permanent broken opponent. Two things keep it out, both
    pinned: the leading dot (glob's `*` skips dotfiles) and the `.partial`
    suffix.
    """
    import torch
    from python_ai.rl import checkpointing
    from python_ai.trainers.league import discover_historical_checkpoints

    seen = {}

    def peeking_save(obj, f, *a, **kw):
        # Look at the directory while the temp file exists.
        seen["during"] = discover_historical_checkpoints(directory=str(tmp_path))
        f.write(b"\x80\x02}q\x00.")

    monkeypatch.setattr(torch, "save", peeking_save)
    checkpointing.atomic_save({"model": {}}, str(tmp_path / "snap.pth"))

    assert seen["during"] == [], (
        f"the temp file was discoverable as an opponent: {seen['during']}")
