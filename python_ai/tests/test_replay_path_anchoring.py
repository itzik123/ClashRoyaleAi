"""Replay destinations are anchored on the repo root, not the cwd.

`replays/*.json` feeds the placement phase histogram; a detector pointed at a
directory the run did not write to reports a clean board forever.
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
    """Source-level check (the alternative is running a full replay): no bare
    relative "replays" literal outside run_path.
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
    """Behavioural proof: record from an unrelated cwd and the file appears at the
    anchor.
    """
    import gymnasium as gym

    from python_ai.envs import gym_wrapper
    from python_ai.rl import base_trainer
    from python_ai.rl.config import PPOConfig
    from python_ai.trainers.train import PHASE1_OPPONENT, Phase1Trainer

    anchor = tmp_path / "anchor"
    cwd = tmp_path / "elsewhere"
    anchor.mkdir()
    cwd.mkdir()

    # Re-anchor the repo-root-relative family into tmp_path, so nothing is
    # written to the real repo.
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
