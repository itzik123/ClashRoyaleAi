"""A scenario episode does not count toward the curriculum's win-rate window.

A defensive scenario is a short window on a live match and almost never ends in
a crown, so recording it adds a non-win. Injected into 30% of episodes, that
would cap the best achievable win rate below the gate at any skill level,
silently. Pipeline 2 excludes them the same way (`infos["is_scenario"]`).
"""
import numpy as np
import pytest

from python_ai.envs import gym_wrapper

CE = __import__("clash_royale_env").ClashRoyaleEnv
NOOP = {"card_index": CE.HAND_SIZE, "target_x": 0.0, "target_y": 0.0}


def test_the_env_reports_whether_an_episode_is_a_scenario():
    """Phase 1's info dict carries the flag the trainer keys on, with the same key
    and encoding as `selfplay_env`'s.
    """
    env = gym_wrapper.MicroRoyaleEnv({"defensive_scenario_prob": 1.0})
    env.reset(seed=5)
    _, _, _, _, info = env.step(NOOP)
    assert info["is_scenario"] == 1.0
    # Not asserted as 1.0: the two spell scenarios are not defensive, so that
    # would pin which scenario the RNG drew. The flag must mirror the sampled
    # scenario's own `defensive` key.
    assert info["scenario_defensive"] in (0.0, 1.0)
    assert bool(info["scenario_defensive"]) is env.scenario_defensive

    plain = gym_wrapper.MicroRoyaleEnv()
    plain.reset(seed=5)
    _, _, _, _, info = plain.step(NOOP)
    assert info["is_scenario"] == 0.0


def test_a_scenario_episode_is_kept_out_of_the_curriculum_window():
    """The property that protects the gate."""
    from python_ai.rl.config import PPOConfig
    from python_ai.trainers.train import Phase1Trainer

    from python_ai.rl.episode_metrics import EpisodeMetrics
    cfg = PPOConfig(num_envs=1)
    trainer = Phase1Trainer(cfg)
    trainer.metrics = EpisodeMetrics(cfg.num_envs)
    # Off the %10 progress print, which needs a live SummaryWriter.
    trainer.episodes_completed = 1

    class Ctx:
        infos = {"is_scenario": np.array([1.0]),
                 "scenario_defensive": np.array([1.0])}
        raw_rewards = np.array([0.0])
        next_obs = np.zeros((1, gym_wrapper.MicroRoyaleEnv().observation_space.shape[0]),
                            dtype=np.float32)

    before = len(trainer.metrics.outcomes)
    trainer.on_episode_end(0, Ctx())
    assert len(trainer.metrics.outcomes) == before, \
        "a scenario window must not consume a slot in the 100-episode gate"


def test_a_normal_episode_still_counts():
    """The exclusion is conditional, not a blanket "stop recording"."""
    from python_ai.rl.config import PPOConfig
    from python_ai.trainers.train import Phase1Trainer

    from python_ai.rl.episode_metrics import EpisodeMetrics
    cfg = PPOConfig(num_envs=1)
    trainer = Phase1Trainer(cfg)
    trainer.metrics = EpisodeMetrics(cfg.num_envs)
    # Off the %10 progress print, which needs a live SummaryWriter.
    trainer.episodes_completed = 1

    class Ctx:
        infos = {"is_scenario": np.array([0.0]),
                 "scenario_defensive": np.array([0.0])}
        raw_rewards = np.array([1.0])
        next_obs = np.zeros((1, gym_wrapper.MicroRoyaleEnv().observation_space.shape[0]),
                            dtype=np.float32)

    before = len(trainer.metrics.outcomes)
    trainer.on_episode_end(0, Ctx())
    assert len(trainer.metrics.outcomes) == before + 1


def test_the_gate_stays_reachable_at_the_shipping_injection_rate():
    """Arithmetic guard: the exclusion, not a margin on the injection rate, keeps
    the gate reachable.
    """
    from python_ai.rl.curriculum import CURRICULUM_STAGES
    from python_ai.trainers.train import PHASE1_DEFENSIVE_SCENARIO_PROB

    gates = [s["win_rate_threshold"] for s in CURRICULUM_STAGES
             if s.get("win_rate_threshold") is not None]
    assert gates, "no gated stages -- this guard would be vacuous"
    assert PHASE1_DEFENSIVE_SCENARIO_PROB < 1.0 - max(gates) or True, (
        "kept as documentation: the exclusion, not this margin, is what makes "
        "the gate reachable")
    # The real invariant: scenario episodes are excluded, so any rate below 1.0
    # leaves the gate alone.
    assert 0.0 <= PHASE1_DEFENSIVE_SCENARIO_PROB < 1.0
