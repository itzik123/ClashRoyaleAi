"""A head that only got WIDER must be warm-started, not thrown away.

THE INCIDENT THIS PINS (2026-08-28, phase-1 run, episode 7,609)
--------------------------------------------------------------
The cycle skip connection made `card_head`, `value_head`, `place_ctx` and
`place_ctx_hi` read `cat((hx, cycle_feat))` instead of `hx` alone, so each of
those matrices gained exactly 24 input columns. `load_state_dict_flexible`
compared shapes for equality, found four mismatches, and DISCARDED all four --
the entire card-selection policy, both placement contexts and the critic --
replacing them with fresh init while reporting a successful warm start of the
other 48 tensors.

Measured consequence: win rate fell 0.80 -> 0.15 within 80 episodes and the
curriculum demoted a stage. It cost roughly 1,500 episodes of compute.

WHY ZERO-PADDING IS EXACT, NOT AN APPROXIMATION
-----------------------------------------------
`torch.cat((hx, cycle_feat), dim=-1)` puts the OLD features first, so the old
matrix's columns keep their meaning at `[:, :old_width]` and the new features
occupy `[:, old_width:]`. With the new columns zeroed,

    cat(hx, c) @ W_new.T  ==  hx @ W_old.T + c @ 0  ==  hx @ W_old.T

so the widened layer is mathematically identical to the narrow one at load and
the new pathway learns only what it adds. This is the same device `net.py`
already uses for `place_hires[-1]` and `DilatedContextBlock`'s final 1x1, both
zero-initialised so an added branch is an exact no-op for a warm-started
checkpoint.

The rule is DIRECTIONAL and the tests below bound it from both sides: growth is
only safe along dim 1 (the layer reads more input features, appended at the
end). A matrix that grew along dim 0 has gained new OUTPUT units, where a zero
row is not a no-op -- it is a logit of 0.0 competing with the trained ones --
so that case must still be discarded.
"""
import torch

from python_ai.models.net import MicroRoyaleNet
from python_ai.models.policy_io import load_state_dict_flexible


def _narrowed_checkpoint(net, key, drop):
    """The state dict of `net` with `key` narrowed by `drop` input columns.

    Stands in for a checkpoint written before a skip connection widened that
    layer -- the real ep-7,609 case, where drop == 24.
    """
    sd = {k: v.clone() for k, v in net.state_dict().items()}
    sd[key] = sd[key][:, :-drop].clone()
    return sd


def test_widened_head_is_warm_started_by_zero_padding_not_discarded():
    net = MicroRoyaleNet()
    key = "card_head.weight"
    full_width = net.state_dict()[key].shape[1]
    old = _narrowed_checkpoint(net, key, drop=24)

    # Re-init so the target net's card_head cannot coincidentally equal the
    # checkpoint's -- otherwise "discarded" and "warm-started" look identical.
    target = MicroRoyaleNet()
    load_state_dict_flexible(target, old, "test-widened")

    got = target.state_dict()[key]
    assert got.shape[1] == full_width, "layer must keep its new width"
    torch.testing.assert_close(got[:, :-24], old[key],
                               msg="the trained columns must survive verbatim")
    assert torch.count_nonzero(got[:, -24:]) == 0, \
        "the appended columns must be zero, so the new pathway is a no-op at load"


def test_zero_padded_head_is_functionally_identical_to_the_narrow_one():
    """The property that actually matters: appended features change nothing."""
    net = MicroRoyaleNet()
    key = "card_head.weight"
    old = _narrowed_checkpoint(net, key, drop=24)

    target = MicroRoyaleNet()
    load_state_dict_flexible(target, old, "test-functional")
    w = target.state_dict()[key]

    torch.manual_seed(0)
    hx = torch.randn(3, w.shape[1] - 24)
    cycle = torch.randn(3, 24)
    bias = target.state_dict()["card_head.bias"]

    narrow = hx @ old[key].T + bias
    widened = torch.cat((hx, cycle), dim=-1) @ w.T + bias
    torch.testing.assert_close(widened, narrow)


def test_a_head_that_grew_new_OUTPUT_rows_is_still_discarded():
    """Zero-padding dim 0 would inject a 0.0 logit competing with trained ones.

    A new output unit has no trained counterpart to inherit, so unlike a new
    input column it is not a no-op. This case must keep the old behaviour.
    """
    net = MicroRoyaleNet()
    key = "card_head.weight"
    sd = {k: v.clone() for k, v in net.state_dict().items()}
    sd[key] = sd[key][:-1, :].clone()          # one FEWER output arm
    sd["card_head.bias"] = sd["card_head.bias"][:-1].clone()

    target = MicroRoyaleNet()
    fresh = target.state_dict()[key].clone()
    load_state_dict_flexible(target, sd, "test-row-growth")

    torch.testing.assert_close(target.state_dict()[key], fresh)


def test_clean_load_is_unaffected():
    """The common path must not change: an exact match still reports True."""
    net = MicroRoyaleNet()
    target = MicroRoyaleNet()
    assert load_state_dict_flexible(target, net.state_dict(), "test-clean") is True


def test_growth_still_reports_an_unclean_load():
    """Adam's per-parameter buffers are shaped like the OLD matrix.

    Returning True here would let a resuming trainer load a stale optimizer
    state against a reshaped parameter, so growth must keep signalling False
    even though nothing trained was lost.
    """
    net = MicroRoyaleNet()
    old = _narrowed_checkpoint(net, "card_head.weight", drop=24)
    target = MicroRoyaleNet()
    assert load_state_dict_flexible(target, old, "test-unclean") is False
