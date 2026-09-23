"""A seeded run reproduces its opening hands, not just its weights.

The engine's opening-hand shuffle is most of an episode's variance;
`BaseTrainer.setup` must seed it through `reset(seed=...)`, or a paired A/B
cannot hold openings fixed while the run reports itself deterministic.
"""
import numpy as np
import pytest
import torch

from python_ai.rl.config import PPOConfig
from python_ai.rl.seeding import engine_seeds, worker_seeds

CFG = dict(num_envs=2, update_timestep=2, bptt_chunk=2, num_minibatches=1,
           ppo_epochs=1, save_every_episodes=10 ** 9,
           replay_every_episodes=10 ** 9)


def _build(tmp_path, monkeypatch, seed):
    import gymnasium as gym

    from python_ai.envs import gym_wrapper
    from python_ai.rl import base_trainer
    from python_ai.trainers.train import PHASE1_OPPONENT, Phase1Trainer

    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLASH_WEIGHTS", str(tmp_path / "w.pth"))
    monkeypatch.setenv("CLASH_LOGDIR", str(tmp_path / "runs"))
    monkeypatch.setattr(base_trainer, "run_path",
                        lambda name: str(tmp_path.joinpath(*name.split("/"))))
    monkeypatch.setattr(base_trainer, "HISTORICAL_CHECKPOINT_DIR",
                        str(tmp_path / "hc"))

    cfg = PPOConfig(seed=seed, **CFG)

    class H(Phase1Trainer):
        def __init__(self):
            super().__init__(cfg)
            self.log_dir = str(tmp_path / "runs")

        def build_envs(self):
            def make():
                return gym_wrapper.MicroRoyaleEnv(
                    {"opponent": PHASE1_OPPONENT, "teacher_stage": 0})
            return gym.vector.SyncVectorEnv(
                [make for _ in range(self.cfg.num_envs)])

    t = H()
    t.setup()
    return t


def _hands(t):
    obs = torch.tensor(np.asarray(t._obs, dtype=np.float32))
    return t.net.hand_card_ids(obs).tolist()


@pytest.mark.slow
def test_the_same_seed_reproduces_the_opening_hands(tmp_path, monkeypatch):
    """The regression test."""
    a = _build(tmp_path / "a", monkeypatch, 4242)
    ha = _hands(a)
    a.envs.close()

    b = _build(tmp_path / "b", monkeypatch, 4242)
    hb = _hands(b)
    b.envs.close()

    assert ha == hb, (
        f"a seeded run dealt different openings: {ha} vs {hb}. The engine's "
        "shuffle is not being seeded from the training path, so the run is "
        "only partially deterministic while reporting itself deterministic.")


@pytest.mark.slow
def test_different_seeds_still_deal_different_openings(tmp_path, monkeypatch):
    """Control: a bug pinning every run to one fixed hand would satisfy the test
    above.
    """
    a = _build(tmp_path / "a", monkeypatch, 1)
    ha = _hands(a)
    a.envs.close()

    b = _build(tmp_path / "b", monkeypatch, 2)
    hb = _hands(b)
    b.envs.close()

    assert ha != hb, "two different seeds produced identical openings"


@pytest.mark.slow
def test_an_UNSEEDED_run_is_still_random(tmp_path, monkeypatch):
    """Seeding stays opt-in: making runs deterministic by default would silently
    change every existing configuration.
    """
    seen = set()
    for i in range(3):
        t = _build(tmp_path / f"u{i}", monkeypatch, None)
        seen.add(repr(_hands(t)))
        t.envs.close()
    assert len(seen) > 1, "unseeded runs became deterministic"


@pytest.mark.slow
def test_the_two_workers_do_not_get_the_SAME_opening(tmp_path, monkeypatch):
    """Per-worker seeds stay distinct, or the vectorised envs run in lockstep and
    the batch collapses to one trajectory.
    """
    t = _build(tmp_path / "d", monkeypatch, 99)
    hands = _hands(t)
    t.envs.close()
    assert hands[0] != hands[1], (
        "both workers dealt the same opening hand -- they are sharing a seed")


# --- the seed derivation itself ---

def test_engine_seeds_are_reproducible_and_distinct():
    a = engine_seeds(7, 4)
    assert a == engine_seeds(7, 4)
    assert len(set(a)) == 4


def test_engine_seeds_are_INDEPENDENT_of_the_scenario_stream():
    """One integer for both would tie which scenario is injected to which hand is
    dealt.
    """
    assert engine_seeds(7, 4) != worker_seeds(7, 4)


def test_engine_seeds_of_none_are_all_none():
    assert engine_seeds(None, 3) == [None, None, None]
