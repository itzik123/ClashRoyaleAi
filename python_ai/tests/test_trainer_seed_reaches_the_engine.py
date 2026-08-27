"""A seeded run must reproduce its OPENING HANDS, not just its weights.

`rl/seeding.py` calls itself "One seed, every stochastic source. The single
entry point for determinism", and its own docstring lists the C++ engine's
opening-hand shuffle among the sources it accounts for. It did not reach it:

  * `seed_everything` covers torch / numpy-global / stdlib-random,
  * `worker_seeds` feeds each env a `scenario_seed`, which seeds that env's
    scenario Generator ONLY,
  * `MicroRoyaleEnv` seeds the engine exclusively from `reset(seed=...)`,
  * and `BaseTrainer.setup` called `self.envs.reset()` with NO seed.

So the engine's `mt19937` stayed on OS entropy. Measured before the fix, two
trainers built with CLASH_SEED=4242:

    network init identical : True
    opening hands run A    : [[7, 24, 6, 33], [24, 25, 6, 7]]
    opening hands run B    : [[24, 33, 25, 72], [40, 15, 25, 24]]

The run still PRINTED "Deterministic run: seed=4242". That is the damaging part:
a paired A/B could not hold the openings fixed across arms, a crash could not be
re-run, and nothing said so -- while the console asserted the opposite.

The opening hand decides what the agent is able to play, so it is not a minor
stochastic source; it is most of the episode's variance.
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
    """THE regression test."""
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
    """The control. Without it, a bug that pinned every run to one fixed hand
    would satisfy the test above perfectly."""
    a = _build(tmp_path / "a", monkeypatch, 1)
    ha = _hands(a)
    a.envs.close()

    b = _build(tmp_path / "b", monkeypatch, 2)
    hb = _hands(b)
    b.envs.close()

    assert ha != hb, "two different seeds produced identical openings"


@pytest.mark.slow
def test_an_UNSEEDED_run_is_still_random(tmp_path, monkeypatch):
    """Seeding is opt-in and must stay so: every win rate in CLAUDE.md was
    earned unseeded, and making runs deterministic by default would silently
    change what every existing configuration does."""
    seen = set()
    for i in range(3):
        t = _build(tmp_path / f"u{i}", monkeypatch, None)
        seen.add(repr(_hands(t)))
        t.envs.close()
    assert len(seen) > 1, "unseeded runs became deterministic"


@pytest.mark.slow
def test_the_two_workers_do_not_get_the_SAME_opening(tmp_path, monkeypatch):
    """Per-worker seeds must stay distinct, or the vectorized envs run in
    lockstep and the rollout's effective batch collapses to one trajectory."""
    t = _build(tmp_path / "d", monkeypatch, 99)
    hands = _hands(t)
    t.envs.close()
    assert hands[0] != hands[1], (
        "both workers dealt the same opening hand -- they are sharing a seed")


# --- the seed derivation itself -------------------------------------------

def test_engine_seeds_are_reproducible_and_distinct():
    a = engine_seeds(7, 4)
    assert a == engine_seeds(7, 4)
    assert len(set(a)) == 4


def test_engine_seeds_are_INDEPENDENT_of_the_scenario_stream():
    """Reusing one integer for both would tie WHICH SCENARIO is injected to
    WHICH HAND is dealt -- a correlation that could bias an experiment while
    looking perfectly seeded."""
    assert engine_seeds(7, 4) != worker_seeds(7, 4)


def test_engine_seeds_of_none_are_all_none():
    assert engine_seeds(None, 3) == [None, None, None]
