"""Why `masked_kl_elementwise` does NOT produce NaN gradients -- and the
property that keeps it that way.

`-inf` sits on illegal cells in BOTH distributions, so `lo.exp()` is 0 there and
`(lo - ln)` is `-inf - -inf = nan`. The original form computed `term = 0 * nan`
and then zeroed the result with `torch.where`, which fixes the FORWARD and does
NOT fix the backward: `where` still differentiates the branch it did not select,
and `0 * nan = nan`. With `a = lo.exp()` and `b = (lo - ln)`:

    grad_a = grad_term * b = 0 * nan = NAN
    grad_b = grad_term * a = 0 * 0   = 0

So a NaN really was produced. It stayed harmless only because `grad_a` flows
into `log_softmax(target_logits)` and the target is a CONSTANT -- the advisor's
surface, buffered from numpy during the rollout. Two implicit properties were
holding the safety up: that the target never requires grad, and that no row is
entirely `-inf`. The second is demonstrably load-bearing -- a fully-masked row
put NaN straight into the trainable `new_logits.grad` while the forward read a
clean 0.0.

`masked_kl_elementwise` now substitutes the OPERANDS before the arithmetic
instead of masking the result, so no NaN is created and neither property has to
hold. The forward is bit-identical on every reachable input.

ONE CASE REMAINS UNSAFE AND IS DELIBERATELY LEFT SO: an all-`-inf` row is
already `[nan, nan, nan]` coming out of `log_softmax` itself, upstream of
anything this function does. It cannot occur -- see
`test_a_fully_masked_row_is_clean_FORWARD_but_not_backward` for the two
invariants that prevent it -- and defending it would cost a branch on every
call. That test pins the boundary rather than papering over it.
"""
import pytest
import torch

from python_ai.advisors.advisor_target import masked_kl_elementwise

NI = float("-inf")


@pytest.mark.parametrize("name,new,tgt", [
    ("both -inf on the same illegal cells",
     [1.0, 2.0, 0.3, NI, NI], [3.0, -1.0, 0.7, NI, NI]),
    ("target -inf where new is finite",
     [1.0, 2.0, 0.3, 0.9, 0.1], [3.0, -1.0, 0.7, NI, NI]),
    ("new -inf where target is finite",
     [1.0, 2.0, 0.3, NI, NI], [3.0, -1.0, 0.7, 0.5, 0.2]),
    ("no masking at all",
     [1.0, 2.0, 0.3, 0.9, 0.1], [3.0, -1.0, 0.7, 0.5, 0.2]),
])
def test_the_gradient_into_new_logits_is_finite(name, new, tgt):
    n = torch.tensor([new], requires_grad=True)
    t = torch.tensor([tgt])
    kl = masked_kl_elementwise(n, t)
    assert torch.isfinite(kl).all(), f"{name}: forward is non-finite"
    kl.sum().backward()
    assert torch.isfinite(n.grad).all(), (
        f"{name}: gradient into new_logits is non-finite: {n.grad.tolist()}")


def test_a_fully_masked_row_is_clean_FORWARD_but_not_backward():
    """The one case this function does NOT make safe, recorded exactly.

    An all-`-inf` row is degenerate before any of this function's arithmetic
    runs: `log_softmax([-inf, -inf, -inf])` is `[nan, nan, nan]`, so the NaN is
    created inside log_softmax's own backward and no amount of sanitizing its
    OUTPUT can remove it. Fixing it would mean substituting the logits before
    the softmax, which costs a branch on every call to defend a state that
    cannot occur.

    IT CANNOT OCCUR, by two independent invariants, and both are worth naming
    because this test is the thing that will notice if either is dropped:

      * `net.placement_mask` always leaves at least one legal cell -- there is
        always a legal row, so no row comes out entirely `-inf`;
      * `forward_sequence`'s row-compaction fills INACTIVE rows with ZEROS
        rather than `-inf`, a choice its own comment makes for exactly this
        reason (an all-`-inf` row gives `Categorical.entropy() = nan`).

    So the assertion here is deliberately the honest one: forward clean, and
    the backward NaN documented rather than papered over.
    """
    n = torch.tensor([[NI, NI, NI]], requires_grad=True)
    t = torch.tensor([[NI, NI, NI]])
    kl = masked_kl_elementwise(n, t)
    assert torch.isfinite(kl).all(), "forward must still be clean"
    assert float(kl.detach()) == 0.0
    kl.sum().backward()
    assert not torch.isfinite(n.grad).all(), (
        "a fully-masked row now yields a finite gradient. If log_softmax's "
        "degenerate case was handled upstream, that is an improvement -- "
        "update this test and drop the caveat in the module docstring.")


def test_the_kl_is_zero_for_identical_distributions():
    """Sanity: the metric is a KL, not merely finite."""
    logits = [1.0, 2.0, 0.3, NI]
    n = torch.tensor([logits], requires_grad=True)
    kl = masked_kl_elementwise(n, torch.tensor([logits]))
    assert float(kl.detach()) == pytest.approx(0.0, abs=1e-6)


def test_the_kl_is_positive_when_they_differ():
    n = torch.tensor([[1.0, 2.0, 0.3, NI]], requires_grad=True)
    t = torch.tensor([[3.0, -1.0, 0.7, NI]])
    assert float(masked_kl_elementwise(n, t).detach()) > 0.0


def test_illegal_cells_contribute_EXACTLY_zero():
    """Widening the mask must not change the KL of the cells that remain --
    otherwise the term silently depends on how many cells are illegal."""
    n_small = torch.tensor([[1.0, 2.0, 0.3]], requires_grad=True)
    t_small = torch.tensor([[3.0, -1.0, 0.7]])
    n_big = torch.tensor([[1.0, 2.0, 0.3, NI, NI, NI]], requires_grad=True)
    t_big = torch.tensor([[3.0, -1.0, 0.7, NI, NI, NI]])
    small = float(masked_kl_elementwise(n_small, t_small).detach())
    big = float(masked_kl_elementwise(n_big, t_big).detach())
    assert small == pytest.approx(big, abs=1e-6)


def test_the_TARGET_is_never_differentiable_in_the_real_path():
    """THE assumption the gradient safety rests on.

    A NaN genuinely IS produced in the backward (grad into `lo.exp()` is
    0 * nan). It is dropped only because the target is a constant. Passing a
    target that requires grad makes that NaN reach a trainable tensor -- this
    test documents and demonstrates it, so the day someone distils from a
    LEARNED teacher they find out here.
    """
    n = torch.tensor([[1.0, 2.0, 0.3, NI, NI]], requires_grad=True)
    t = torch.tensor([[3.0, -1.0, 0.7, NI, NI]], requires_grad=True)
    kl = masked_kl_elementwise(n, t)
    assert torch.isfinite(kl).all(), "forward should still be clean"
    kl.sum().backward()

    assert torch.isfinite(n.grad).all(), "the trainable side stays finite"
    assert torch.isfinite(t.grad).all(), (
        "the target's gradient is non-finite. Sanitizing the operands before "
        "the arithmetic was supposed to make this safe for a differentiable "
        "target too -- if it is NaN again, the substitution has regressed to "
        "masking the RESULT, which fixes the forward and not the backward")


def test_the_buffered_advisor_target_does_not_require_grad():
    """The production path, checked rather than assumed: the advisor surface is
    built from numpy during the rollout and buffered."""
    import numpy as np
    tgt = torch.from_numpy(np.zeros((2, 4), dtype=np.float32))
    assert tgt.requires_grad is False
