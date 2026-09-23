"""Generalized Advantage Estimation, with correct bootstrapping through a
truncation.

A scenario window running out (`envs/scenarios.py`) is a truncation, not a
terminal: bootstrapping 0 would teach the critic that the world ends there.
`values[t + 1]` is no use either, since the env has already auto-reset, so the
rollout records V(final_obs) while the final observation exists. Without the
truncation arguments this reduces exactly to plain GAE
(`tests/test_rl_gae.py`).
"""
import torch


def compute_gae(rewards, values, masks, next_value, gamma, gae_lambda,
                boot_nonterminal=None, trunc_flag=None, trunc_boot=None):
    """(T, N) advantages.

    rewards, values, masks : (T, N). `masks` is 1 - done, folded with `valid`,
    and cuts the trace at every episode boundary.
    next_value : (N,) value of the state after the last stored step.
    boot_nonterminal : (T, N) coefficient on the bootstrap term. Defaults to
    `masks` (done means terminal); pipeline 2 passes 0 only on true terminals.
    trunc_flag, trunc_boot : (T, N). Where `trunc_flag` > 0.5 the next-state
    value is `trunc_boot`, the captured V(final_obs).

    The trace mask stays `masks`: a truncation ends credit propagation even
    though it bootstraps a value.
    """
    if boot_nonterminal is None:
        boot_nonterminal = masks
    horizon = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(next_value)
    for t in reversed(range(horizon)):
        base_next = next_value if t == horizon - 1 else values[t + 1]
        if trunc_flag is None:
            next_val = base_next
        else:
            next_val = torch.where(trunc_flag[t] > 0.5, trunc_boot[t], base_next)
        delta = rewards[t] + gamma * next_val * boot_nonterminal[t] - values[t]
        gae = delta + gamma * gae_lambda * masks[t] * gae
        advantages[t] = gae
    return advantages


def normalize(advantages, eps=1e-8, mask=None):
    """Batch-normalized advantages (critic targets stay raw).

    `mask`: optional (T, N) of 1/0; mean and std come from the masked-in rows,
    but every row is rescaled. The live caller passes `decision`, the rows the
    actor loss reads.

    Fewer than two rows has no estimable spread, and the unbiased std() would
    return NaN, which poisons every parameter in one step; zero is returned
    instead. A non-finite value from upstream is deliberately not swallowed:
    PPOUpdater's guard drops and reports it.
    """
    flat = advantages.reshape(-1) if mask is None else advantages[mask > 0.5]
    if flat.numel() < 2:
        return torch.zeros_like(advantages)
    return (advantages - flat.mean()) / (flat.std() + eps)


def safe_std(x):
    """`x.std()`, but 0.0 with fewer than two elements, where the unbiased
    estimator returns NaN (and clamp would pass the NaN through). Returns a
    float.
    """
    if x.numel() < 2:
        return 0.0
    return float(x.std())


def explained_variance(returns, values):
    """1 - Var(return - V) / Var(return): ~1 good, 0 no better than the mean, < 0
    worse.

    Raw critic MSE is uninterpretable alone, being bounded below by the
    returns' noise.
    """
    # Checked before var(), which would warn on n < 2; the suite runs
    # warning-clean.
    if returns.numel() < 2:
        return torch.tensor(0.0)
    var = returns.var()
    if not (var > 0):
        return torch.tensor(0.0)
    return 1.0 - (returns - values).var() / var.clamp(min=1e-8)
