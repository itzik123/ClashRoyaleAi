"""The advantage-normalisation statistics come from the rows the actor loss reads.

The normalised advantages have one consumer, the actor loss, masked by
`decision` (>= 2 card arms legal). Normalising over `valid`, a superset
including forced steps, leaves the actor's rows off-centre; PPO's clip is
asymmetric in sign(A), so that offset is not a harmless baseline shift.
"""
import numpy as np
import pytest
import torch

from python_ai.rl import gae as gae_mod
from python_ai.rl.config import PPOConfig

TINY = PPOConfig(num_envs=2, update_timestep=20, bptt_chunk=10,
                 num_minibatches=1, ppo_epochs=1,
                 save_every_episodes=10 ** 9, replay_every_episodes=10 ** 9)


@pytest.fixture
def captured(tmp_path, monkeypatch):
    """Run one real update and capture the advantages handed to the updater."""
    import gymnasium as gym

    from python_ai.envs import gym_wrapper
    from python_ai.rl import base_trainer
    from python_ai.trainers.train import PHASE1_OPPONENT, Phase1Trainer

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLASH_WEIGHTS", str(tmp_path / "w.pth"))
    monkeypatch.setenv("CLASH_LOGDIR", str(tmp_path / "runs"))
    monkeypatch.setattr(base_trainer, "HISTORICAL_CHECKPOINT_DIR",
                        str(tmp_path / "hc"))

    seen = {}

    class Harness(Phase1Trainer):
        def __init__(self):
            super().__init__(TINY)
            self.log_dir = str(tmp_path / "runs")
            self.n = 0

        def build_envs(self):
            def make():
                return gym_wrapper.MicroRoyaleEnv(
                    {"opponent": PHASE1_OPPONENT, "teacher_stage": 0})
            return gym.vector.SyncVectorEnv(
                [make for _ in range(self.cfg.num_envs)])

        def run_update(self):
            batch = self.buffer.stack()
            real = self.updater.update

            def spy(b, advantages_norm, returns, *a, **k):
                seen["adv"] = advantages_norm.detach().clone()
                seen["decision"] = batch["decision"].detach().clone()
                seen["valid"] = batch["valid"].detach().clone()
                seen["returns"] = returns.detach().clone()
                return real(b, advantages_norm, returns, *a, **k)

            self.updater.update = spy
            try:
                stats = super().run_update()
            finally:
                self.updater.update = real
            self.n += 1
            return stats

        def should_stop(self):
            return self.n >= 1

        def on_finish(self):
            self.envs.close()

    Harness().run()
    assert "adv" in seen, "the update never ran"
    return seen


@pytest.mark.slow
def test_advantages_are_centred_on_the_rows_the_ACTOR_actually_reads(captured):
    """The normalised advantages are centred and unit-scaled on the decision rows.
    """
    adv, decision = captured["adv"], captured["decision"]
    rows = decision > 0.5
    if int(rows.sum()) < 2:
        pytest.skip("this rollout had no decision steps to normalize over")

    assert float(adv[rows].mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(adv[rows].std()) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.slow
def test_the_statistics_are_NOT_taken_over_valid(captured):
    """Negative control: when the row sets differ, a `valid`-normalised tensor
    cannot also be centred on the decision rows. Skipped rather than vacuous
    when they coincide.
    """
    adv = captured["adv"]
    decision, valid = captured["decision"], captured["valid"]
    if int((valid > 0.5).sum()) == int((decision > 0.5).sum()):
        pytest.skip("valid and decision coincided; nothing to distinguish")

    over_valid = float(adv[valid > 0.5].mean())
    over_decision = float(adv[decision > 0.5].mean())
    assert abs(over_valid - over_decision) > 1e-7, (
        "advantages look centred on BOTH row sets, which is impossible when "
        "they differ -- the normalization mask is probably still `valid`")


@pytest.mark.slow
def test_critic_targets_stay_RAW(captured):
    """Only advantages are normalised: normalising the critic's returns would make
    it regress a batch-dependent target.
    """
    returns = captured["returns"]
    valid = captured["valid"] > 0.5
    assert abs(float(returns[valid].mean())) > 1e-6 or \
        abs(float(returns[valid].std()) - 1.0) > 1e-3


# --- the unit-level property, without a live env ---

def test_normalize_centres_on_the_masked_rows_only():
    torch.manual_seed(0)
    adv = torch.randn(50, 4)
    mask = torch.zeros(50, 4)
    mask[:20] = 1.0                      # a strict subset with its own mean
    adv[20:] += 7.0                      # make the excluded rows very different

    out = gae_mod.normalize(adv, mask=mask)
    keep = mask > 0.5
    assert float(out[keep].mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(out[keep].std()) == pytest.approx(1.0, abs=1e-4)
    # The excluded rows are rescaled by those constants, not dropped.
    assert float(out[~keep].mean()) > 5.0


def test_a_superset_mask_leaves_the_subset_off_centre():
    """The removed failure, in miniature."""
    torch.manual_seed(0)
    adv = torch.randn(50, 4)
    decision = torch.zeros(50, 4)
    decision[:20] = 1.0
    valid = torch.ones(50, 4)
    adv[20:] += 7.0

    over_valid = gae_mod.normalize(adv, mask=valid)
    keep = decision > 0.5
    assert abs(float(over_valid[keep].mean())) > 1.0, (
        "the superset normalization should leave the subset badly off-centre")
