"""Replay destinations are anchored, not cwd-relative.

Checkpoints and TensorBoard runs were anchored on 2026-08-25 because "which
directory you launched from" silently decided whether a multi-day run resumed
or started fresh. `replays/` was left behind by that pass -- and
`checkpointing.run_path`'s own docstring already CLAIMS to cover it:

    "A run-artifact destination (TensorBoard runs, snapshot pools), anchored on
     the repository root, which is where `runs/`, `replays/` and
     `historical_checkpoints/` already sit."

`run_path` was even already imported into base_trainer.py. It simply was not
used for the two `replays/` strings, so the documentation and the code
disagreed, silently and in the direction that scatters artifacts.

Observed: `historical_checkpoints/` and `runs/` sit at the repo root as
`run_path` says, `python_ai/replays/` does not exist at all, and a stale
`replays/replay_ep1000.json` sits at the repo root -- i.e. written by a run
launched from there. A smoke run launched from a scratch directory created
`replays/` in the scratch directory instead.

This matters beyond tidiness: `replays/*.json` is the input to the placement
PHASE histogram, the only cheap detector for the ConvTranspose2d checkerboard
artifact. A detector pointed at a directory the run did not write to reports a
clean board forever.
"""
import os

import pytest

import python_ai
from python_ai.rl.checkpointing import run_path


def test_run_path_anchors_on_the_repo_root_not_the_cwd(tmp_path, monkeypatch):
    before = run_path("replays")
    monkeypatch.chdir(tmp_path)
    assert run_path("replays") == before, "run_path moved with the cwd"
    assert os.path.isabs(before)
    assert before == os.path.join(python_ai.REPO_ROOT, "replays")


def test_base_trainer_builds_its_replay_paths_through_run_path():
    """A source-level check, because the alternative is running a full replay.

    Greps for a bare relative "replays/" literal, which is exactly what the two
    sites used to hold.
    """
    src_path = os.path.join(python_ai.PACKAGE_DIR, "rl", "base_trainer.py")
    with open(src_path, encoding="utf-8") as fh:
        lines = fh.readlines()

    offenders = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for literal in ('"replays"', "'replays'", '"replays/', "'replays/",
                        'f"replays/'):
            if literal in stripped and "run_path" not in stripped:
                offenders.append((i, stripped))
                break
    assert not offenders, (
        "cwd-relative replay path(s) in base_trainer.py: "
        f"{offenders} -- route them through run_path()")


@pytest.mark.slow
def test_a_replay_lands_at_the_anchor_regardless_of_cwd(tmp_path, monkeypatch):
    """The behavioural proof: run the recorder from an unrelated cwd and check
    the file appears at the ANCHOR, not under the cwd."""
    import gymnasium as gym

    from python_ai.envs import gym_wrapper
    from python_ai.rl import base_trainer
    from python_ai.rl.config import PPOConfig
    from python_ai.trainers.train import PHASE1_OPPONENT, Phase1Trainer

    anchor = tmp_path / "anchor"
    cwd = tmp_path / "elsewhere"
    anchor.mkdir()
    cwd.mkdir()

    # Re-anchor the whole repo-root-relative family into tmp_path, so the test
    # never writes into the real repository.
    monkeypatch.setattr(python_ai, "REPO_ROOT", str(anchor))
    monkeypatch.setattr(base_trainer, "run_path",
                        lambda name: os.path.join(str(anchor),
                                                  *name.split("/")))
    monkeypatch.setattr(base_trainer, "HISTORICAL_CHECKPOINT_DIR",
                        str(anchor / "hc"))
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("CLASH_WEIGHTS", str(tmp_path / "w.pth"))
    monkeypatch.setenv("CLASH_LOGDIR", str(tmp_path / "runs"))

    cfg = PPOConfig(num_envs=1, update_timestep=2, bptt_chunk=2,
                    num_minibatches=1, ppo_epochs=1,
                    save_every_episodes=10 ** 9,
                    replay_every_episodes=10 ** 9)

    class Harness(Phase1Trainer):
        def __init__(self):
            super().__init__(cfg)
            self.log_dir = str(tmp_path / "runs")

        def build_envs(self):
            def make():
                return gym_wrapper.MicroRoyaleEnv(
                    {"opponent": PHASE1_OPPONENT, "teacher_stage": 0})
            return gym.vector.SyncVectorEnv([make])

    t = Harness()
    t.setup()
    try:
        assert t.record_replay() is True
    finally:
        t.envs.close()

    landed = list((anchor / "replays").glob("*.json"))
    assert landed, f"no replay at the anchor {anchor / 'replays'}"
    assert not list(cwd.glob("replays/*.json")), (
        "a replay was written relative to the cwd")
