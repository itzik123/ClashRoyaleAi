"""A head that only got wider is warm-started, not thrown away.

`torch.cat((hx, cycle_feat))` puts the old features first, so the old columns
keep their meaning and the new ones zero-pad:

    cat(hx, c) @ W_new.T  ==  hx @ W_old.T + c @ 0  ==  hx @ W_old.T

The widened layer is identical to the narrow one at load. Only growth along dim
1 (more inputs, appended) is safe; a new output row is a 0.0 logit competing
with trained ones, so that case is still discarded.
"""
import torch

from python_ai.models.net import MicroRoyaleNet
from python_ai.models.policy_io import load_state_dict_flexible


def _narrowed_checkpoint(net, key, drop):
    """The state dict of `net` with `key` narrowed by `drop` input columns: a
    checkpoint from before a skip connection widened the layer.
    """
    sd = {k: v.clone() for k, v in net.state_dict().items()}
    sd[key] = sd[key][:, :-drop].clone()
    return sd


def test_widened_head_is_warm_started_by_zero_padding_not_discarded():
    net = MicroRoyaleNet()
    key = "card_head.weight"
    full_width = net.state_dict()[key].shape[1]
    old = _narrowed_checkpoint(net, key, drop=24)

    # A fresh target, so "discarded" and "warm-started" cannot look identical.
    target = MicroRoyaleNet()
    load_state_dict_flexible(target, old, "test-widened")

    got = target.state_dict()[key]
    assert got.shape[1] == full_width, "layer must keep its new width"
    torch.testing.assert_close(got[:, :-24], old[key],
                               msg="the trained columns must survive verbatim")
    assert torch.count_nonzero(got[:, -24:]) == 0, \
        "the appended columns must be zero, so the new pathway is a no-op at load"


def test_zero_padded_head_is_functionally_identical_to_the_narrow_one():
    """The property that matters: appended features change nothing."""
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
    """A new output unit has no trained counterpart, so zero-padding it is not a
    no-op.
    """
    net = MicroRoyaleNet()
    key = "card_head.weight"
    sd = {k: v.clone() for k, v in net.state_dict().items()}
    sd[key] = sd[key][:-1, :].clone()          # one fewer output arm
    sd["card_head.bias"] = sd["card_head.bias"][:-1].clone()

    target = MicroRoyaleNet()
    fresh = target.state_dict()[key].clone()
    load_state_dict_flexible(target, sd, "test-row-growth")

    torch.testing.assert_close(target.state_dict()[key], fresh)


def test_clean_load_is_unaffected():
    """An exact match still reports True."""
    net = MicroRoyaleNet()
    target = MicroRoyaleNet()
    assert load_state_dict_flexible(target, net.state_dict(), "test-clean") is True


def test_growth_still_reports_an_unclean_load():
    """Adam's buffers are shaped like the old matrix, so growth reports an unclean
    load even though nothing trained was lost.
    """
    net = MicroRoyaleNet()
    old = _narrowed_checkpoint(net, "card_head.weight", drop=24)
    target = MicroRoyaleNet()
    assert load_state_dict_flexible(target, old, "test-unclean") is False
