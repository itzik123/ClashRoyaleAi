"""Generalized Advantage Estimation, with correct bootstrapping through a
truncation.

The two pipelines had two versions of this loop. Pipeline 2's is a strict
GENERALIZATION of pipeline 1's, not a different algorithm, and writing them as
one function is what makes that checkable rather than asserted: pass no
truncation arguments and the arithmetic reduces, term for term, to the simple
form. `tests/test_rl_gae.py` pins that equivalence.

WHY THE TRUNCATION CASE EXISTS. Pipeline 2 injects scenarios with a focused
window (`envs/scenarios.py`), and a window that runs out is a TRUNCATION, not a
terminal: no king died. Bootstrapping 0 there teaches the critic a biased "the
world ends here" value for a state the world does not end in. But
`values[t + 1]` is no use either -- at a done step the vector env has already
auto-reset, so that value belongs to the NEXT episode. The captured
V(final_obs) is the only correct next-state value, and it has to be recorded
during the rollout while the final observation still exists.
"""
import torch


def compute_gae(rewards, values, masks, next_value, gamma, gae_lambda,
                boot_nonterminal=None, trunc_flag=None, trunc_boot=None):
    """(T, N) advantages.

    rewards, values, masks : (T, N). `masks` is 1 - done, folded with `valid`,
        and cuts the trace at every episode boundary.
    next_value : (N,) value of the state after the last stored step.
    boot_nonterminal : (T, N) coefficient on the bootstrap term. Defaults to
        `masks`, which is pipeline 1's behaviour -- there, done means terminal.
        Pipeline 2 passes a mask that is 0 only on TRUE terminals, so a
        truncation still bootstraps.
    trunc_flag, trunc_boot : (T, N). Where `trunc_flag` > 0.5 the next-state
        value is taken from `trunc_boot` (the captured V(final_obs)) instead of
        from the next stored step.

    The trace mask stays `masks` in every case: a truncation ends the episode
    for the purpose of credit propagation even though it bootstraps a value.
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
    """Batch-normalized advantages. Critic targets are the RAW returns; only
    advantages are normalized, which is why this is a separate step.

    `mask` : optional (T, N) of 1/0. Where given, the mean and std are taken
        over the MASKED-IN rows only, but every row is still rescaled by them.
        Callers pass `valid`, so the phantom post-autoreset rows -- which are
        excluded from every loss term already -- stop setting the constants
        that rescale the real ones. `run_update` filters by `valid` for
        explained variance and for the value-clip range on the next two lines;
        this was the one statistic in that block still taken over the
        contaminated tensor.

    A batch with fewer than two masked-in rows has NO ESTIMABLE SPREAD, and
    `Tensor.std()` is the unbiased estimator, so it divides by n-1 = 0 and
    returns NaN. That NaN reaches the actor loss, and one non-finite element
    there turns every parameter to NaN in a single optimizer step -- measured
    2026-08-26, 32 of 32 parameter tensors, permanently. Zero is the correct
    answer instead: with no spread there is no directional information to
    scale, so every advantage is exactly its own mean.

    A non-finite value arriving from UPSTREAM is deliberately NOT swallowed
    here. Zeroing it would silently convert a numerical fault into a
    no-learning run, which is the harder failure to notice; it is left to
    propagate to `PPOUpdater`'s containment guard, which drops the step AND
    reports it.
    """
    flat = advantages.reshape(-1) if mask is None else advantages[mask > 0.5]
    if flat.numel() < 2:
        return torch.zeros_like(advantages)
    return (advantages - flat.mean()) / (flat.std() + eps)


def explained_variance(returns, values):
    """1 - Var(return - V) / Var(return): ~1 good, 0 no better than the mean,
    <0 worse than the mean.

    Exists because raw critic MSE is uninterpretable on its own -- it is bounded
    below by the irreducible noise in the returns. This project called the
    critic broken three times off a flat MSE around 0.11; explained variance at
    the same moment measured +0.64, i.e. healthy all along.
    """
    var = returns.var()
    if not (var > 0):
        return torch.tensor(0.0)
    return 1.0 - (returns - values).var() / var.clamp(min=1e-8)
