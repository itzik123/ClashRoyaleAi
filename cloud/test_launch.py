"""Tests for `cloud/launch.py`'s scaling rule.

DELIBERATELY NOT IN `python_ai/tests/` -- that tree is read-only by default and
this tests deployment scaffolding, not the training mechanism. Run it with:

    python_ai/venv/Scripts/python.exe -m pytest cloud -q

It imports only `python_ai.rl.config`, which pulls in `math`/`os`/`dataclasses`
and nothing else, so this suite runs on a box with no `.pyd` and no torch.

WHAT IS ACTUALLY BEING PINNED HERE. `scaled_config` encodes the argument in
CLAUDE.md's "Scaling the training loop" section, and the whole argument rests on
one property: optimizer steps per rollout must GROW with `num_envs` instead of
staying pinned at 32. A regression here would not raise -- it would quietly put
a 60-hour run in the large-batch regime.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from launch import (BASELINE_ENVS, TARGET_MINIBATCH_ROWS,  # noqa: E402
                    _rows_per_minibatch, scaled_config)
from python_ai.rl.config import PPOConfig  # noqa: E402

SCALES = [14, 16, 28, 32, 48, 64, 96, 128]


def test_baseline_is_returned_untouched():
    """N=8 must reproduce the documented baseline EXACTLY.

    Every measured number in CLAUDE.md was earned at this configuration, so a
    scaling rule that perturbs it at N=8 would silently invalidate the one
    comparison point the whole file is written against.
    """
    assert scaled_config(BASELINE_ENVS) == PPOConfig(num_envs=BASELINE_ENVS)


def _steps(cfg):
    return cfg.num_minibatches * cfg.ppo_epochs


@pytest.mark.parametrize("n", SCALES)
def test_optimizer_steps_never_regress_below_baseline(n):
    """THE load-bearing property, and the reason this module exists.

    At the stock config `num_minibatches * ppo_epochs` is 32 regardless of
    `num_envs`, so raising N buys samples while holding gradient steps flat.
    Scaling must never do WORSE than that.

    `>=` rather than `>` is deliberate and is the honest bound. Because
    rows/minibatch is pinned near 2x the baseline, `num_minibatches` only
    reaches the baseline's 8 at N=16 -- so N=14 and N=16 legitimately land on
    exactly 32 steps over ~2x the data at sqrt(2) the learning rate, which is
    the standard large-batch recipe applied conservatively. Anything BELOW 32
    is the regression this pins.
    """
    baseline_steps = _steps(PPOConfig(num_envs=BASELINE_ENVS))
    cfg = scaled_config(n)
    assert _steps(cfg) >= baseline_steps, (
        f"N={n} gives {_steps(cfg)} optimizer steps against the baseline's "
        f"{baseline_steps} -- the large-batch regime")


def test_optimizer_steps_are_monotonic_in_num_envs():
    """More parallel envs must never buy FEWER gradient steps per rollout.

    A non-monotonic rule is the subtle version of the same failure: some middle
    N would be quietly worse than both its neighbours, and nothing would say so.
    """
    steps = [_steps(scaled_config(n)) for n in SCALES]
    assert steps == sorted(steps), dict(zip(SCALES, steps))


def test_optimizer_steps_grow_strictly_once_scaling_is_real():
    """By 3x the baseline env count the extra data must be buying extra steps,
    not just bigger ones."""
    baseline_steps = _steps(PPOConfig(num_envs=BASELINE_ENVS))
    for n in [n for n in SCALES if n >= 3 * BASELINE_ENVS]:
        assert _steps(scaled_config(n)) > baseline_steps, n


@pytest.mark.parametrize("n", SCALES)
def test_minibatch_rows_stay_near_target(n):
    """Held near TARGET_MINIBATCH_ROWS, never allowed to scale with N.

    Below the target the GPU is launch-bound; above it, gradient steps stop
    growing. Both failure modes are silent.
    """
    cfg = scaled_config(n)
    rows = _rows_per_minibatch(cfg, cfg.num_minibatches)
    assert abs(rows - TARGET_MINIBATCH_ROWS) <= TARGET_MINIBATCH_ROWS // 2, (
        f"N={n} put {rows} rows in a minibatch against a target of "
        f"{TARGET_MINIBATCH_ROWS}")


@pytest.mark.parametrize("n", SCALES)
def test_num_minibatches_divides_the_segment_count_evenly(n):
    """`PPOUpdater` walks the permutation in strides of
    `n_segments // num_minibatches`, so an uneven split leaves one ragged
    trailing minibatch per epoch. Tolerated by the updater, avoidable here."""
    cfg = scaled_config(n)
    segments = (cfg.update_timestep // cfg.bptt_chunk) * cfg.num_envs
    assert segments % cfg.num_minibatches == 0


@pytest.mark.parametrize("n", SCALES)
def test_trajectory_parameters_are_never_touched(n):
    """gamma, gae_lambda, update_timestep and bptt_chunk are per-trajectory or
    structural. Nothing about running more trajectories in parallel justifies
    moving any of them, and moving gamma or gae_lambda WOULD be a change to the
    learning problem itself."""
    base, cfg = PPOConfig(num_envs=BASELINE_ENVS), scaled_config(n)
    for field in ("gamma", "gae_lambda", "eps_clip", "update_timestep",
                  "bptt_chunk", "max_grad_norm", "vf_clip_std_frac"):
        assert getattr(cfg, field) == getattr(base, field), field


@pytest.mark.parametrize("n", SCALES)
def test_learning_rate_tracks_the_minibatch_not_num_envs(n):
    """LR scales on the MINIBATCH, and the minibatch is held constant -- so the
    LR is constant too across every scaled N, and never more than the
    undiscounted sqrt bound over the baseline."""
    base, cfg = PPOConfig(num_envs=BASELINE_ENVS), scaled_config(n)
    rows = _rows_per_minibatch(cfg, cfg.num_minibatches)
    baseline_rows = _rows_per_minibatch(base, base.num_minibatches)
    sqrt_bound = base.lr * (rows / baseline_rows) ** 0.5
    assert base.lr <= cfg.lr <= sqrt_bound + 1e-12, (
        f"N={n}: lr {cfg.lr:.4e} outside [{base.lr:.4e}, {sqrt_bound:.4e}]")


@pytest.mark.parametrize("n", SCALES)
def test_config_is_constructible_and_self_consistent(n):
    """PPOConfig.__post_init__ enforces update_timestep % bptt_chunk == 0."""
    cfg = scaled_config(n)
    assert cfg.num_envs == n
    assert cfg.update_timestep % cfg.bptt_chunk == 0
    assert cfg.segments_per_rollout == (cfg.update_timestep // cfg.bptt_chunk) * n


def test_ppo_epochs_only_ever_decreases():
    """More data per rollout means less reuse of it, never more."""
    base = PPOConfig(num_envs=BASELINE_ENVS)
    epochs = [scaled_config(n).ppo_epochs for n in SCALES]
    assert all(e <= base.ppo_epochs for e in epochs)
    assert epochs == sorted(epochs, reverse=True), epochs
    assert all(e >= 1 for e in epochs)
