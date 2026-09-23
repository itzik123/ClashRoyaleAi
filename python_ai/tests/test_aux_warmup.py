"""The next-card auxiliary loss is ramped in over a from-scratch run's first
episodes.

A fresh 185-way head starts at CE ln(185) = 5.22, far above the ~ln(8) its
coefficient was sized for, and its gradient out-pulls and opposes the policy
and critic on the shared LSTM and trunk exactly when a random-init net must
find its first wins. A linear warm-up, not a removal; `aux_warmup_episodes = 0`
restores the old behaviour exactly.
"""
import copy

import pytest
import torch
import torch.optim as optim

from python_ai.rl.config import PPOConfig, aux_warmup_scale
from python_ai.rl.ppo import PPOUpdater
from python_ai.tests.test_rl_ppo import TINY, rollout  # noqa: F401  (fixture)


def test_the_schedule_ramps_linearly_and_then_holds():
    assert aux_warmup_scale(0, 2000) == 0.0
    assert aux_warmup_scale(1000, 2000) == pytest.approx(0.5)
    assert aux_warmup_scale(2000, 2000) == 1.0
    assert aux_warmup_scale(50_000, 2000) == 1.0


def test_a_zero_warmup_is_exactly_the_old_behaviour():
    assert aux_warmup_scale(0, 0) == 1.0


def test_the_default_config_warms_up():
    assert PPOConfig().aux_warmup_episodes > 0


def _grads(rollout, **kw):
    net0, batch = rollout
    net = copy.deepcopy(net0)
    captured = {}
    import python_ai.rl.ppo as ppo_mod
    real = ppo_mod.clip_and_step

    def cap(opt, params, m):
        captured.update({n: p.grad.clone() for n, p in net.named_parameters()
                         if p.grad is not None})
        return True
    ppo_mod.clip_and_step = cap
    try:
        torch.manual_seed(0)
        cfg = kw.pop("cfg", TINY)
        upd = PPOUpdater(net, optim.Adam(net.parameters(), lr=1e-4), cfg)
        T, N = batch["rewards"].shape
        batch = dict(batch)
        batch["aux_opp_played"] = torch.full((T, N), 15, dtype=torch.long)
        upd.update(batch, torch.ones(T, N), torch.zeros(T, N), vf_clip_range=0.2,
                   ent_coef_card=0.05, ent_coef_placement=0.06, coverage_coef=0.0,
                   **kw)
    finally:
        ppo_mod.clip_and_step = real
    return captured


def test_scale_zero_removes_the_aux_gradient_and_scale_one_changes_nothing(rollout):
    import dataclasses
    g_default = _grads(rollout)
    g_one = _grads(rollout, aux_scale=1.0)
    g_zero = _grads(rollout, aux_scale=0.0)
    g_nocoef = _grads(rollout, cfg=dataclasses.replace(TINY, aux_card_coef=0.0))
    # Tolerance, not torch.equal: two identical runs differ by up to ~1.5e-08
    # (multithreaded CPU backward).
    for n in g_default:
        assert torch.allclose(g_default[n], g_one[n], atol=1e-6), n
        assert torch.allclose(g_zero[n], g_nocoef[n], atol=1e-6), n
    lstm = [n for n in g_default if n.startswith("lstm")]
    assert max(float((g_default[n] - g_zero[n]).abs().max()) for n in lstm) > 1e-4, \
        "the aux term reached nothing, so this test cannot see the scale"
