"""Which rows set the constants that rescale the advantages.

`rl/ppo.py`'s module docstring states the rule this pins:

    `decision`  1 where >=2 card arms were legal. The ACTOR and the ENTROPY
                terms are normalized by this, not by `valid`.

The normalized advantages have exactly ONE consumer -- `mb_adv` in the actor
loss, which is masked by `mb_decision`. So the rows that set the mean and std
must be the rows the loss reads. `run_update` used to pass `valid`, a strict
superset that also contains every FORCED step (nothing affordable), whose
advantages are real but never trained on.

MEASURED before the fix, three consecutive rollouts off a 308-episode
checkpoint (decision covered 73.7-75.1% of rows):

    centering error   -0.0312, -0.0820, -0.0073   (of a unit std)
    scale error        0.9757,  0.9492,  1.0041   (1.0 = correct)

Small, but not nothing and not free of consequence: PPO's clip is asymmetric in
sign(A), so an off-centre advantage changes WHICH samples clip and in which
direction. A pure baseline shift would be harmless in vanilla policy gradient;
under clipping it is not.
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
    """Run ONE real update and capture the advantages handed to the updater."""
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
    """The normalized advantages have one consumer and it masks by `decision`.

    Normalizing over `valid` left the actor's own rows off-centre by as much as
    0.082 of a unit std (measured). PPO's clip is asymmetric in sign(A), so that
    offset is not the free baseline shift it would be in vanilla PG.
    """
    adv, decision = captured["adv"], captured["decision"]
    rows = decision > 0.5
    if int(rows.sum()) < 2:
        pytest.skip("this rollout had no decision steps to normalize over")

    assert float(adv[rows].mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(adv[rows].std()) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.slow
def test_the_statistics_are_NOT_taken_over_valid(captured):
    """The negative control: pin that we did not simply keep the old mask.

    Whenever the two row sets genuinely differ, a `valid`-normalized tensor
    cannot also be exactly centred on the decision rows. Skipped (rather than
    silently passing) when this rollout happened to have them coincide, so the
    assertion can never be vacuous.
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
    """Only advantages are normalized. `returns` feed the critic and must keep
    their own scale -- normalizing them would make the value head regress a
    batch-dependent target."""
    returns = captured["returns"]
    valid = captured["valid"] > 0.5
    assert abs(float(returns[valid].mean())) > 1e-6 or \
        abs(float(returns[valid].std()) - 1.0) > 1e-3


# --- the unit-level property, without needing a live env ------------------

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
    # the excluded rows are still rescaled by those constants, not dropped
    assert float(out[~keep].mean()) > 5.0


def test_a_superset_mask_leaves_the_subset_off_centre():
    """The failure this fix removes, reproduced in miniature."""
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
