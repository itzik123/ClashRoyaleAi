"""Pipeline 2's rollout end to end, with the truncation-bootstrap fields live.

`compute_gae` and `_truncation_bootstrap` have unit tests; this checks they
agree on shapes, dtypes and row alignment when a real vector env produces the
fields, along collect_rollout -> buffer.stack -> compute_gae ->
PPOUpdater.update.
"""
import numpy as np
import pytest
import torch

from python_ai.rl.buffer import (
    ADVISOR_FIELDS, CORE_FIELDS, TRUNCATION_FIELDS,
)
from python_ai.rl.config import PPOConfig

#: update_timestep must exceed the longest scenario window (envs/scenarios.py
#: `max_steps` run 12..25), or no truncation occurs and the tests that need one
#: silently skip.
TINY = PPOConfig(num_envs=2, update_timestep=30, bptt_chunk=15,
                 num_minibatches=1, ppo_epochs=1,
                 save_every_episodes=10 ** 9,
                 replay_every_episodes=10 ** 9)


@pytest.fixture
def phase2(tmp_path, monkeypatch):
    """A Phase2Trainer wired entirely into tmp_path, with a seeded pool. Pipeline
    2 refuses to start without pipeline 1's output, so a bootstrap checkpoint
    is minted first.
    """
    import gymnasium as gym

    from python_ai.advisors import advisor_target
    from python_ai.envs import selfplay_env
    from python_ai.models.net import MicroRoyaleNet
    from python_ai.rl import base_trainer
    from python_ai.rl.checkpointing import atomic_save
    from python_ai.trainers import train_selfplay

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLASH_WEIGHTS", str(tmp_path / "w.pth"))
    monkeypatch.setenv("CLASH_LOGDIR", str(tmp_path / "runs"))
    monkeypatch.setattr(base_trainer, "run_path",
                        lambda name: str(tmp_path.joinpath(*name.split("/"))))
    from python_ai.envs import scenarios
    from python_ai.rl.checkpointing import save_historical_snapshot
    from python_ai.trainers import league

    # Injection probability is a module constant read at reset, not an
    # env_config key. Forced to 1.0 so a window is guaranteed to expire inside
    # the rollout.
    monkeypatch.setattr(scenarios, "SCENARIO_INJECTION_PROB", 1.0)

    pool = tmp_path / "hc"
    pool.mkdir()
    monkeypatch.setattr(base_trainer, "HISTORICAL_CHECKPOINT_DIR", str(pool))
    # league resolves the pool through its own module-level import, so it is
    # patched too.
    monkeypatch.setattr(league, "HISTORICAL_CHECKPOINT_DIR", str(pool))

    boot = tmp_path / "model_weights.pth"
    atomic_save({"model": MicroRoyaleNet(num_ability_slots=0).state_dict()},
                str(boot))
    monkeypatch.setattr(train_selfplay, "BOOTSTRAP_FROM_PATH", str(boot))
    monkeypatch.setattr(train_selfplay, "WEIGHT_PATH",
                        str(tmp_path / "sp.pth"))

    # One eligible opponent, from pipeline 1: pipeline 2's own snapshots are
    # age-gated against a trainee at episode 0.
    save_historical_snapshot(MicroRoyaleNet(num_ability_slots=0), 1,
                             "pipeline1", directory=str(pool))

    class Harness(train_selfplay.Phase2Trainer):
        def __init__(self):
            super().__init__(TINY)
            self.weight_path = str(tmp_path / "sp.pth")
            self.log_dir = str(tmp_path / "runs")

        def build_envs(self):
            # SyncVectorEnv: spawned workers would re-import the test module,
            # and in-process envs see the SCENARIO_INJECTION_PROB patch.
            def make():
                return selfplay_env.MicroRoyaleSelfPlayEnv(
                    {"scenarios_enabled": True})
            return gym.vector.SyncVectorEnv(
                [make for _ in range(self.cfg.num_envs)])

    t = Harness()
    t.setup()
    yield t
    t.envs.close()


def test_the_truncation_fields_are_declared_in_the_buffer(phase2):
    """The flag actually widens the field set."""
    for name in TRUNCATION_FIELDS:
        assert name in phase2.buffer, f"{name} missing from the phase-2 buffer"
    for name in CORE_FIELDS:
        assert name in phase2.buffer


def test_a_phase2_rollout_stacks_with_consistent_shapes(phase2):
    """Every declared field arrives, at the right shape and dtype, for the whole
    rollout.
    """
    phase2.collect_rollout()
    batch = phase2.buffer.stack()

    expected = set(CORE_FIELDS) | set(TRUNCATION_FIELDS)
    from python_ai.advisors import advisor_target
    if advisor_target.enabled():
        expected |= set(ADVISOR_FIELDS)
    assert set(batch) == expected

    T, N = TINY.update_timestep, TINY.num_envs
    for name in TRUNCATION_FIELDS:
        assert batch[name].shape == (T, N), (
            f"{name} has shape {tuple(batch[name].shape)}, expected {(T, N)}")
        assert batch[name].dtype == torch.float32
        assert torch.isfinite(batch[name]).all(), f"{name} carries non-finite"


def test_a_truncation_ACTUALLY_OCCURS_in_this_configuration(phase2):
    """Keeps the rest of this file honest: with injection at 1.0 and the rollout
    past the longest window, a truncation must occur, so a skip below cannot
    hide a broken mechanism.
    """
    phase2.collect_rollout()
    batch = phase2.buffer.stack()
    n = int((batch["trunc_flag"] > 0.5).sum())
    assert n > 0, (
        "no scenario window expired in a "
        f"{TINY.update_timestep}-step rollout at scenario_prob=1.0 -- either "
        "truncation stopped being produced, or update_timestep no longer "
        "exceeds the longest scenario max_steps")


def test_the_bootstrap_flags_are_mutually_consistent(phase2):
    """A truncation is not a terminal, so `boot_nonterminal` is 1 wherever
    `trunc_flag` is set.
    """
    phase2.collect_rollout()
    batch = phase2.buffer.stack()

    flag = batch["trunc_flag"]
    nonterminal = batch["boot_nonterminal"]
    assert set(torch.unique(flag).tolist()) <= {0.0, 1.0}
    assert set(torch.unique(nonterminal).tolist()) <= {0.0, 1.0}

    truncated = flag > 0.5
    if truncated.any():
        assert (nonterminal[truncated] > 0.5).all(), (
            "a truncated step was marked terminal -- it would bootstrap 0 and "
            "teach the critic the world ends at a scenario cutoff")


def test_trunc_boot_is_zero_wherever_the_flag_is_not_set(phase2):
    """A stray V(final) on a non-truncated row means the two fields are written
    from different conditions.
    """
    phase2.collect_rollout()
    batch = phase2.buffer.stack()
    quiet = batch["trunc_flag"] <= 0.5
    assert torch.equal(batch["trunc_boot"][quiet],
                       torch.zeros_like(batch["trunc_boot"][quiet]))


def test_a_full_phase2_update_runs_and_reports_finite_diagnostics(phase2):
    """Rollout -> GAE (truncation form) -> PPO update."""
    phase2.collect_rollout()
    stats = phase2.run_update()

    assert stats.nonfinite_skips == 0, (
        "a minibatch was dropped as non-finite on the truncation path")
    for name in ("critic_loss", "total_loss", "aux_ce", "aux_acc"):
        v = getattr(stats, name)
        assert np.isfinite(v), f"{name} is {v}"
    assert np.isfinite(phase2._explained_variance)
    assert phase2._vf_clip_range >= TINY.eps_clip


def test_the_update_actually_moves_the_weights(phase2):
    before = [p.detach().clone() for p in phase2.net.parameters()]
    phase2.collect_rollout()
    phase2.run_update()
    after = list(phase2.net.parameters())
    assert any(not torch.equal(b, a.detach()) for b, a in zip(before, after))


def test_gae_consumes_the_truncation_fields_rather_than_ignoring_them(phase2):
    """Negative control: perturbing `trunc_boot` on flagged rows must change the
    advantages, or the fields are carried but never read.
    """
    from python_ai.rl import gae as gae_mod

    phase2.collect_rollout()
    batch = phase2.buffer.stack()
    if not (batch["trunc_flag"] > 0.5).any():
        pytest.skip("no truncation occurred in this rollout")

    with torch.no_grad():
        nx = torch.tensor(phase2._obs, dtype=torch.float32)
        feats, _, _ = phase2.net.extract_features(nx)
        _, _, _, nv, _ = phase2.net.step_lstm_and_card(
            feats, (phase2._hx, phase2._cx))
        nv = nv.squeeze(-1)

    def adv(trunc_boot):
        return gae_mod.compute_gae(
            batch["rewards"], batch["values"], batch["masks"], nv,
            TINY.gamma, TINY.gae_lambda,
            boot_nonterminal=batch["boot_nonterminal"],
            trunc_flag=batch["trunc_flag"], trunc_boot=trunc_boot)

    base = adv(batch["trunc_boot"])
    bumped = adv(batch["trunc_boot"] + 10.0)
    assert not torch.allclose(base, bumped), (
        "trunc_boot does not affect the advantages -- the truncation fields "
        "are plumbed through the buffer but never actually consumed")


def test_a_scenario_truncation_is_not_charged_the_draw_penalty(phase2):
    """`draw_on_terminated_only`: a defence that ran out its window is not charged
    as a stalled game.
    """
    assert phase2.draw_on_terminated_only is True
    phase2.collect_rollout()
    batch = phase2.buffer.stack()
    # A truncated-but-not-terminated row must not carry DRAW_PENALTY from the
    # cutoff.
    from python_ai.rewards.weights import DRAW_PENALTY
    trunc_only = (batch["trunc_flag"] > 0.5)
    if trunc_only.any():
        assert (batch["rewards"][trunc_only] > -DRAW_PENALTY).all(), (
            "a scenario cutoff was charged the full draw penalty")
