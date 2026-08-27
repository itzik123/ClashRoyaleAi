"""The value-clip range on a batch with no estimable spread.

`run_update` scales the critic's trust region to the batch's own return spread:

    vf_clip_range = clamp(vf_clip_std_frac * r.std(), min=eps_clip)

`Tensor.std()` is the UNBIASED estimator, so with fewer than two masked-in rows
it divides by n-1 = 0 and returns NaN -- and `torch.clamp` propagates NaN rather
than flooring it. So the guard READS as "never tighter than eps_clip" while
actually passing NaN through, which is the shape of check this project has been
bitten by repeatedly: one that cannot do the job it appears to do.

The NaN then reaches `value_clipped`, `critic_loss_per_elem`, and the total
loss, where the containment guard drops EVERY minibatch. The update survives
(that is what the guard is for) but learns nothing, and `std()` also emits a
UserWarning into a suite that is kept warning-clean.

`gae.normalize` already guards precisely this case, with precisely this
reasoning ("A batch with fewer than two masked-in rows has NO ESTIMABLE SPREAD
... Zero is the correct answer instead"). This applies the same rule to the one
statistic in that block still computing an unguarded `std()`.

Reachable only on a degenerate batch -- it needs all but one row to be a
phantom post-autoreset step -- but the cost of the guard is one comparison and
the cost of the hole is a silently dead update.
"""
import warnings

import pytest
import torch

from python_ai.rl.gae import safe_std


@pytest.mark.parametrize("n", [0, 1])
def test_no_estimable_spread_is_zero_not_nan(n):
    assert safe_std(torch.randn(n)) == 0.0


def test_it_does_not_warn_on_a_degenerate_input():
    """`std()` on n<2 emits a UserWarning; the suite is kept warning-clean."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert safe_std(torch.randn(1)) == 0.0
        assert safe_std(torch.randn(0)) == 0.0


def test_it_is_the_ordinary_unbiased_std_otherwise():
    torch.manual_seed(0)
    x = torch.randn(64)
    assert safe_std(x) == pytest.approx(float(x.std()), abs=1e-7)


def test_a_constant_batch_has_zero_spread():
    assert safe_std(torch.full((32,), 3.5)) == pytest.approx(0.0, abs=1e-6)


def test_the_clip_range_floors_at_eps_clip_instead_of_going_nan():
    """The property the caller actually depends on."""
    eps_clip = 0.2
    for n in (0, 1, 2, 64):
        torch.manual_seed(n)
        r = torch.randn(n)
        rng = torch.clamp(torch.tensor(1.0 * safe_std(r)), min=eps_clip).item()
        assert rng == rng, f"vf_clip_range is NaN at n={n}"
        assert rng >= eps_clip

def test_clamp_alone_would_NOT_have_floored_it():
    """Negative control: pins that the hazard was real and that `clamp(min=)`
    is not itself sufficient, so nobody 'simplifies' the guard away later."""
    with warnings.catch_warnings():
        # This test's whole point is to build the NaN the guard prevents, so
        # the degrees-of-freedom warning is expected here and only here.
        warnings.simplefilter("ignore", UserWarning)
        nan_std = torch.randn(1).std()      # NaN by construction
    assert nan_std != nan_std
    floored = torch.clamp(nan_std, min=0.2)
    assert floored != floored, (
        "clamp appears to floor NaN on this torch build -- if so this guard "
        "is redundant and the reasoning above needs revisiting")


@pytest.mark.slow
def test_run_update_survives_a_batch_with_almost_no_valid_rows(tmp_path,
                                                               monkeypatch):
    """End to end: an almost-entirely-phantom rollout must still produce a
    finite clip range and must not have every minibatch dropped."""
    import gymnasium as gym

    from python_ai.envs import gym_wrapper
    from python_ai.rl import base_trainer
    from python_ai.rl.config import PPOConfig
    from python_ai.trainers.train import PHASE1_OPPONENT, Phase1Trainer

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLASH_WEIGHTS", str(tmp_path / "w.pth"))
    monkeypatch.setenv("CLASH_LOGDIR", str(tmp_path / "runs"))
    monkeypatch.setattr(base_trainer, "run_path",
                        lambda name: str(tmp_path.joinpath(*name.split("/"))))
    monkeypatch.setattr(base_trainer, "HISTORICAL_CHECKPOINT_DIR",
                        str(tmp_path / "hc"))

    cfg = PPOConfig(num_envs=2, update_timestep=4, bptt_chunk=2,
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
            return gym.vector.SyncVectorEnv(
                [make for _ in range(self.cfg.num_envs)])

    t = Harness()
    t.setup()
    try:
        t.collect_rollout()
        # Force the degenerate case: one valid row in the entire batch.
        t.buffer._data["valid"] = [
            torch.zeros(cfg.num_envs) for _ in t.buffer._data["valid"]]
        t.buffer._data["valid"][0] = torch.tensor([1.0, 0.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            stats = t.run_update()
    finally:
        t.envs.close()

    rng = t._vf_clip_range
    assert rng == rng, "vf_clip_range came out NaN"
    assert rng >= cfg.eps_clip
    assert stats.nonfinite_skips == 0, (
        "every minibatch was dropped -- the NaN reached the loss")
