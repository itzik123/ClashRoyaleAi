"""GAE, including that pipeline 2's truncation form is a strict generalisation of
the simple one, checked against an independent reimplementation.
"""
import numpy as np
import pytest
import torch

from python_ai.rl.gae import compute_gae, explained_variance, normalize


def _reference_simple_gae(rewards, values, masks, next_value, gamma, lam):
    """The simple loop, written independently: a test that reuses the code it
    checks can only detect changes, never errors.
    """
    T, N = rewards.shape
    adv = torch.zeros_like(rewards)
    gae = torch.zeros(N)
    for t in reversed(range(T)):
        next_val = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * masks[t] - values[t]
        gae = delta + gamma * lam * masks[t] * gae
        adv[t] = gae
    return adv


@pytest.fixture
def episode():
    torch.manual_seed(7)
    T, N = 9, 3
    rewards = torch.randn(T, N)
    values = torch.randn(T, N)
    masks = (torch.rand(T, N) > 0.2).float()
    next_value = torch.randn(N)
    return rewards, values, masks, next_value


def test_matches_an_independent_implementation_of_the_simple_form(episode):
    rewards, values, masks, next_value = episode
    got = compute_gae(rewards, values, masks, next_value, 0.99, 0.9)
    want = _reference_simple_gae(rewards, values, masks, next_value, 0.99, 0.9)
    assert torch.allclose(got, want, atol=1e-6)


def test_truncation_form_reduces_to_the_simple_form_when_nothing_truncates(episode):
    """With `boot_nonterminal = masks` and no truncation flags, the extra
    machinery contributes exactly nothing.
    """
    rewards, values, masks, next_value = episode
    simple = compute_gae(rewards, values, masks, next_value, 0.99, 0.9)
    general = compute_gae(
        rewards, values, masks, next_value, 0.99, 0.9,
        boot_nonterminal=masks,
        trunc_flag=torch.zeros_like(masks),
        trunc_boot=torch.zeros_like(masks))
    assert torch.equal(simple, general)


def test_a_truncated_step_bootstraps_the_captured_value_not_the_next_episode():
    """At a done step the vector env has already auto-reset, so `values[t+1]`
    belongs to the next episode; a truncated step must bootstrap the captured
    V(final) instead.
    """
    T, N = 3, 1
    rewards = torch.zeros(T, N)
    values = torch.tensor([[0.0], [0.0], [0.0]])
    masks = torch.tensor([[1.0], [0.0], [1.0]])       # step 1 ends the episode
    boot = torch.ones(T, N)                            # not a true terminal
    trunc_flag = torch.tensor([[0.0], [1.0], [0.0]])
    trunc_boot = torch.tensor([[0.0], [5.0], [0.0]])   # the captured V(final)
    # values[2] differs from trunc_boot[1], so reaching for the next step's
    # value would change the advantage at t=1.
    values = torch.tensor([[0.0], [0.0], [99.0]])
    adv = compute_gae(rewards, values, masks, torch.zeros(N), 0.99, 0.9,
                      boot_nonterminal=boot, trunc_flag=trunc_flag,
                      trunc_boot=trunc_boot)
    assert adv[1, 0] == pytest.approx(0.99 * 5.0)


def test_a_true_terminal_bootstraps_nothing():
    """boot_nonterminal = 0 zeroes the bootstrap, so the return at a king kill is
    the reward alone.
    """
    T, N = 2, 1
    rewards = torch.tensor([[0.0], [1.0]])
    values = torch.zeros(T, N)
    masks = torch.tensor([[1.0], [0.0]])
    boot = torch.tensor([[1.0], [0.0]])                # step 1 is a terminal
    adv = compute_gae(rewards, values, masks, torch.full((N,), 42.0), 0.99, 0.9,
                      boot_nonterminal=boot,
                      trunc_flag=torch.zeros(T, N),
                      trunc_boot=torch.zeros(T, N))
    assert adv[1, 0] == pytest.approx(1.0)


def test_the_trace_is_cut_at_an_episode_boundary():
    """mask=0 stops credit propagating backwards across a reset."""
    T, N = 4, 1
    rewards = torch.tensor([[0.0], [0.0], [0.0], [10.0]])
    values = torch.zeros(T, N)
    masks = torch.tensor([[1.0], [0.0], [1.0], [1.0]])
    adv = compute_gae(rewards, values, masks, torch.zeros(N), 0.99, 0.95)
    # The reward at t=3 must not reach t=0: the boundary at t=1 blocks it.
    assert adv[0, 0] == pytest.approx(0.0)
    assert adv[2, 0] > 0.0


def test_normalize_gives_zero_mean_unit_std():
    x = torch.randn(20, 4) * 3.0 + 7.0
    z = normalize(x)
    assert float(z.mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(z.std()) == pytest.approx(1.0, abs=1e-3)


def test_explained_variance_is_1_for_a_perfect_critic_and_0_for_the_mean():
    returns = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert float(explained_variance(returns, returns.clone())) == pytest.approx(1.0)
    mean_predictor = torch.full_like(returns, float(returns.mean()))
    assert float(explained_variance(returns, mean_predictor)) == pytest.approx(0.0)


def test_explained_variance_is_negative_when_worse_than_the_mean():
    """Below 0 the critic is worse than predicting the mean, which a flat raw MSE
    cannot show.
    """
    returns = torch.tensor([1.0, 2.0, 3.0, 4.0])
    inverted = torch.tensor([4.0, 3.0, 2.0, 1.0])
    assert float(explained_variance(returns, inverted)) < 0.0


def test_explained_variance_of_a_constant_return_is_zero_not_nan():
    """Var(return) = 0 happens on a short all-draw window; NaN there would poison
    the series.
    """
    returns = torch.full((8,), 0.5)
    assert float(explained_variance(returns, torch.zeros(8))) == 0.0


# --- normalize: degenerate batches and phantom rows ---

def test_normalize_of_a_single_element_is_finite_not_nan():
    """The unbiased `std()` of one element is NaN, and a NaN advantage turns every
    parameter to NaN in one optimizer step.
    """
    out = normalize(torch.tensor([[0.5]]))
    assert torch.isfinite(out).all(), out


def test_normalize_of_a_constant_batch_is_finite_and_zero():
    """A zero-spread batch carries no direction: every advantage is exactly 0, not
    NaN or an eps-amplified spike.
    """
    out = normalize(torch.full((3, 2), 4.0))
    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.zeros_like(out))


def test_normalize_does_not_warn_on_a_degenerate_batch():
    """The suite is kept warning-clean, and `std()` warns on degrees of freedom <=
    0.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        normalize(torch.tensor([[0.5]]))


def test_normalize_takes_its_statistics_from_valid_rows_only():
    """Phantom post-autoreset rows are excluded from the loss, so they must not
    set the mean and std that rescale the real rows.
    """
    adv = torch.tensor([[1.0, 2.0], [3.0, 1000.0]])
    valid = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

    out = normalize(adv, mask=valid)

    real = out[valid > 0.5]
    assert abs(float(real.mean())) < 1e-5, real
    assert abs(float(real.std()) - 1.0) < 1e-3, real


def test_normalize_without_a_mask_is_unchanged():
    """The mask is optional: without one the arithmetic is exactly the old one.
    """
    adv = torch.randn(6, 3)
    assert torch.allclose(normalize(adv), (adv - adv.mean()) / (adv.std() + 1e-8))


def test_normalize_with_an_all_zero_mask_is_finite():
    """An update where every row is a phantom is degenerate but must not produce
    NaN.
    """
    adv = torch.randn(4, 2)
    out = normalize(adv, mask=torch.zeros(4, 2))
    assert torch.isfinite(out).all(), out
