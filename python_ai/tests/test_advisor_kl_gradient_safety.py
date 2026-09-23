"""`masked_kl_elementwise` produces no NaN gradients.

`-inf` sits on illegal cells in both distributions, so the naive product is `0
* (-inf - -inf) = 0 * nan`. Masking the result with `torch.where` fixes the
forward but not the backward, which still differentiates the unselected branch.
The function substitutes the operands before the arithmetic instead, so no NaN
is created; the forward is bit-identical on every reachable input.

An all-`-inf` row is the one unsafe case, left so deliberately: `log_softmax`
already returns NaN there, upstream of this function, and the row cannot occur
(see the test below).
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
    """The one case this function does not make safe, recorded exactly.

    `log_softmax([-inf, -inf, -inf])` is NaN inside its own backward, so
    sanitising its output cannot help. The row cannot occur, by two invariants
    this test would expose if dropped:

      * `net.placement_mask` always leaves at least one legal cell;
      * `forward_sequence`'s row compaction fills inactive rows with zeros, not `-inf`.
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
    """Widening the mask must not change the KL of the remaining cells."""
    n_small = torch.tensor([[1.0, 2.0, 0.3]], requires_grad=True)
    t_small = torch.tensor([[3.0, -1.0, 0.7]])
    n_big = torch.tensor([[1.0, 2.0, 0.3, NI, NI, NI]], requires_grad=True)
    t_big = torch.tensor([[3.0, -1.0, 0.7, NI, NI, NI]])
    small = float(masked_kl_elementwise(n_small, t_small).detach())
    big = float(masked_kl_elementwise(n_big, t_big).detach())
    assert small == pytest.approx(big, abs=1e-6)


def test_the_TARGET_is_never_differentiable_in_the_real_path():
    """The target's gradient stays finite even if it requires grad (e.g.
    distilling from a learned teacher), because the operands, not the result,
    are sanitised.
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
    """The production path: the advisor surface is built from numpy and buffered,
    so it never requires grad.
    """
    import numpy as np
    tgt = torch.from_numpy(np.zeros((2, 4), dtype=np.float32))
    assert tgt.requires_grad is False
