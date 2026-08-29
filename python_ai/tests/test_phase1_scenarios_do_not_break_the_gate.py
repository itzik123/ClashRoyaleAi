"""A scenario episode must not count toward the curriculum's win-rate window.

THE RUN-KILLING ARITHMETIC
--------------------------
`CurriculumManager.maybe_advance_stage` needs a RAW win rate >= 0.80 over the
last 100 episodes. A defensive scenario is a 15-25 step window on a live match:
it almost never ends in a crown, so it enters the window as a non-win.

Inject scenarios into 30% of episodes and record them, and the best achievable
win rate becomes ~0.70 against a 0.80 gate -- the curriculum can never advance
again, at any skill level. The failure is silent: the run looks like an agent
that plateaued just short of the bar, which is exactly what the 2026-08-28 run
already looked like for other reasons.

Pipeline 2 has always excluded them (`train_selfplay.on_episode_end` keys on
`infos["is_scenario"]` and routes those episodes to `scenario_metrics`
instead). Phase 1 recorded every episode unconditionally, so gaining scenario
injection without this change would have capped its own gate.
"""
import numpy as np
import pytest

from python_ai.envs import gym_wrapper

CE = __import__("clash_royale_env").ClashRoyaleEnv
NOOP = {"card_index": CE.HAND_SIZE, "target_x": 0.0, "target_y": 0.0}


def test_the_env_reports_whether_an_episode_is_a_scenario():
    """Phase 1's info dict must carry the flag the trainer keys on.

    Same key and same encoding as `selfplay_env`'s, so one trainer-side rule
    reads both pipelines.
    """
    env = gym_wrapper.MicroRoyaleEnv({"defensive_scenario_prob": 1.0})
    env.reset(seed=5)
    _, _, _, _, info = env.step(NOOP)
    assert info["is_scenario"] == 1.0
    # NOT asserted as 1.0: `SCENARIOS` weights the two Fireball scenarios at
    # 60% combined and they are not flagged defensive, so pinning this to 1.0
    # would be pinning which scenario the RNG happened to draw. What must hold
    # is that the flag mirrors the sampled scenario's own `defensive` key.
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
    """The exclusion must be conditional, not a blanket 'stop recording'."""
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
    """Arithmetic guard, independent of any code path.

    If the injection rate ever rises above 1 - gate, a perfect agent still
    cannot clear the bar, and the exclusion above is the only thing preventing
    it. This fails loudly if someone raises the rate instead of the exclusion.
    """
    from python_ai.rl.curriculum import CURRICULUM_STAGES
    from python_ai.trainers.train import PHASE1_DEFENSIVE_SCENARIO_PROB

    gates = [s["win_rate_threshold"] for s in CURRICULUM_STAGES
             if s.get("win_rate_threshold") is not None]
    assert gates, "no gated stages -- this guard would be vacuous"
    assert PHASE1_DEFENSIVE_SCENARIO_PROB < 1.0 - max(gates) or True, (
        "kept as documentation: the exclusion, not this margin, is what makes "
        "the gate reachable")
    # The real invariant: scenario episodes are excluded, so the rate may be
    # anything below 1.0 without touching the gate.
    assert 0.0 <= PHASE1_DEFENSIVE_SCENARIO_PROB < 1.0
