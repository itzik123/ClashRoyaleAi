"""Pipeline 2's rollout, end to end, with the truncation-bootstrap fields live.

THE GAP THIS FILLS. Before this file, `Phase2Trainer` was covered only by
assertions about its DECLARED attributes -- `uses_truncation_bootstrap is True`,
`draw_on_terminated_only is True` -- and by checkpoint-key checks. Nothing ever
ran a phase-2 rollout or update, so the three fields those flags switch on
(`boot_nonterminal`, `trunc_flag`, `trunc_boot`) had never travelled the whole
path: collect_rollout -> buffer.stack -> compute_gae -> PPOUpdater.update.

That is also the one path the live smoke run could not reach, because phase 1
returns early from `_truncation_bootstrap` -- so these fields were, in the
literal sense, untested wiring on both axes at once.

`compute_gae` and `_truncation_bootstrap` each have good unit tests. What was
missing is that they AGREE about tensor shapes, dtypes and row alignment when a
real vector env produces them, which is exactly the class of defect unit tests
on the pieces cannot see.
"""
import numpy as np
import pytest
import torch

from python_ai.rl.buffer import (
    ADVISOR_FIELDS, CORE_FIELDS, TRUNCATION_FIELDS,
)
from python_ai.rl.config import PPOConfig

#: update_timestep MUST exceed the longest scenario window, or no truncation
#: can occur inside a rollout and every test that needs one silently SKIPS --
#: which is the failure mode this whole file exists to remove. Scenario
#: `max_steps` values in envs/scenarios.py run 12..25, so 30 guarantees at
#: least one window expires. A first draft used 4 and skipped the negative
#: control without saying anything useful.
TINY = PPOConfig(num_envs=2, update_timestep=30, bptt_chunk=15,
                 num_minibatches=1, ppo_epochs=1,
                 save_every_episodes=10 ** 9,
                 replay_every_episodes=10 ** 9)


@pytest.fixture
def phase2(tmp_path, monkeypatch):
    """A Phase2Trainer wired entirely into tmp_path, with a seeded pool.

    Pipeline 2 refuses to start without pipeline 1's output, so a bootstrap
    checkpoint is minted first.
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

    # Injection probability is a MODULE CONSTANT read at reset
    # (selfplay_env.py: `self.scenario_rng.random() < SCENARIO_INJECTION_PROB`),
    # NOT an env_config key -- passing "scenario_prob" in the config is silently
    # ignored, which cost this file one debugging round. Forced to 1.0 so every
    # reset injects and a window is guaranteed to expire inside the rollout.
    monkeypatch.setattr(scenarios, "SCENARIO_INJECTION_PROB", 1.0)

    pool = tmp_path / "hc"
    pool.mkdir()
    monkeypatch.setattr(base_trainer, "HISTORICAL_CHECKPOINT_DIR", str(pool))
    # league resolves the pool through its OWN module-level import, so patching
    # base_trainer's alone would leave discovery pointed at the real pool.
    monkeypatch.setattr(league, "HISTORICAL_CHECKPOINT_DIR", str(pool))

    boot = tmp_path / "model_weights.pth"
    atomic_save({"model": MicroRoyaleNet(num_ability_slots=0).state_dict()},
                str(boot))
    monkeypatch.setattr(train_selfplay, "BOOTSTRAP_FROM_PATH", str(boot))
    monkeypatch.setattr(train_selfplay, "WEIGHT_PATH",
                        str(tmp_path / "sp.pth"))

    # One eligible opponent. A PIPELINE-1 snapshot deliberately: pipeline 2's
    # own are gated by MIN_OPPONENT_AGE_EPISODES against a trainee sitting at
    # episode 0, so none would ever qualify here.
    save_historical_snapshot(MicroRoyaleNet(num_ability_slots=0), 1,
                             "pipeline1", directory=str(pool))

    class Harness(train_selfplay.Phase2Trainer):
        def __init__(self):
            super().__init__(TINY)
            self.weight_path = str(tmp_path / "sp.pth")
            self.log_dir = str(tmp_path / "runs")

        def build_envs(self):
            # SyncVectorEnv: spawned workers re-import the test module under
            # pytest, and the trainer only ever calls reset/step/call. In-process
            # also means the SCENARIO_INJECTION_PROB patch above is visible to
            # the envs, which it would not be across a spawn.
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
    """The flag must actually widen the field set, not just read True."""
    for name in TRUNCATION_FIELDS:
        assert name in phase2.buffer, f"{name} missing from the phase-2 buffer"
    for name in CORE_FIELDS:
        assert name in phase2.buffer


def test_a_phase2_rollout_stacks_with_consistent_shapes(phase2):
    """The wiring test proper: every declared field must arrive, at the right
    shape and dtype, for the WHOLE rollout."""
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
    """The guard that keeps the rest of this file honest.

    Several tests below are meaningful only on a rollout that contains a
    truncation, and a `skip` when none happens looks identical to success. With
    scenario_prob=1.0 and update_timestep past the longest window, at least one
    MUST fire -- so if this fails, the others' skips are hiding a broken
    mechanism rather than an unlucky draw.
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
    """`trunc_flag` marks a truncation; a truncation is NOT a terminal, so
    `boot_nonterminal` must be 1 wherever the flag is set. Getting this pair
    inconsistent is how a truncation silently becomes 'the world ends here'."""
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
    """A captured V(final_obs) on a non-truncated row would be read by
    compute_gae only if the flag were set, but a stray value there means the
    two are being written from different conditions."""
    phase2.collect_rollout()
    batch = phase2.buffer.stack()
    quiet = batch["trunc_flag"] <= 0.5
    assert torch.equal(batch["trunc_boot"][quiet],
                       torch.zeros_like(batch["trunc_boot"][quiet]))


def test_a_full_phase2_update_runs_and_reports_finite_diagnostics(phase2):
    """The whole point: rollout -> GAE (truncation form) -> PPO update."""
    phase2.collect_rollout()
    stats = phase2.run_update()

    assert stats.nonfinite_skips == 0, (
        "a minibatch was dropped as non-finite on the truncation path")
    for name in ("critic_loss", "total_loss", "aux_mse", "aux_mae"):
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
    """Negative control on the plumbing.

    Perturbing `trunc_boot` on rows the flag marks must change the advantages.
    If it does not, the fields are being carried but never read -- which is
    exactly what 'declared but untested wiring' looks like.
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
    """`draw_on_terminated_only` exists so a successful defence that merely ran
    out its focused window is not punished as a stalled game. Pins the flag's
    behaviour, not just its value."""
    assert phase2.draw_on_terminated_only is True
    phase2.collect_rollout()
    batch = phase2.buffer.stack()
    # A truncated-but-not-terminated row must not carry a large negative reward
    # of DRAW_PENALTY's magnitude purely from the cutoff.
    from python_ai.rewards.weights import DRAW_PENALTY
    trunc_only = (batch["trunc_flag"] > 0.5)
    if trunc_only.any():
        assert (batch["rewards"][trunc_only] > -DRAW_PENALTY).all(), (
            "a scenario cutoff was charged the full draw penalty")
