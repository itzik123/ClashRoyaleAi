"""The base CNN trunk's receptive field, pinned by gradient rather than by reading
the layer list.

Two convolutions (21 -> 16 -> 32), each followed by a 2x2 pool: 10 rows x 10
columns on the 34x18 board (3 -> 4 -> 8 -> 10), too short to span a bridge and
the tower behind it. The shipped net appends `DilatedContextBlock`s to reach
the whole board (test_net_dilated_context.py). This file builds the base trunk
explicitly with `context_dilations=()`: the blocks are zero-initialised
residuals, so a fresh shipped net would also measure 10x10 and these assertions
would pass for the wrong reason.

Also pinned: output width and receptive field are separate axes. Widening the
trunk costs throughput and changes the field not at all; depth or dilation is
what widens it. Which trunk is better is a training-run question and is not
decided here.
"""
import torch

from python_ai.engine_constants import BOARD_H, BOARD_W, N_CHANNELS, SPATIAL_SIZE
from python_ai.models.net import MicroRoyaleNet


def _rf_once(trunk, x):
    """Rows/cols of `x` that one centre output cell depends on, for one input.
    """
    x = x.clone().requires_grad_(True)
    feat = trunk(x)
    feat[0, :, feat.shape[2] // 2, feat.shape[3] // 2].sum().backward()
    g = x.grad[0].abs().sum(0)
    return (g.sum(1) > 0), (g.sum(0) > 0)


def measured_receptive_field(net, trials=8):
    """(rows, cols) of input that one centre trunk cell depends on, by gradient.

    The input must be random and the result unioned over several draws: at a
    zero pre-activation ReLU has zero gradient, so an all-zero board measures
    only the channels whose bias happens to leave them open, a smaller field
    that varies with the seed.
    """
    rows = torch.zeros(BOARD_H, dtype=torch.bool)
    cols = torch.zeros(BOARD_W, dtype=torch.bool)
    for t in range(trials):
        torch.manual_seed(1000 + t)
        x = torch.randn(1, N_CHANNELS, BOARD_H, BOARD_W).abs()
        r, c = _rf_once(net.cnn_trunk, x)
        rows |= r
        cols |= c
    return int(rows.sum()), int(cols.sum())


def test_base_trunk_receptive_field_is_what_the_architecture_implies():
    """The base trunk (no context blocks) at 10x10, exactly what the layer
    arithmetic predicts: two independent derivations agreeing.
    """
    net = MicroRoyaleNet(num_ability_slots=0, context_dilations=())
    rows, cols = measured_receptive_field(net)
    assert (rows, cols) == (10, 10), (
        f"trunk receptive field measured {rows}x{cols}, expected 10x10. If a "
        "kernel, stride, dilation or pool changed, that is fine -- but it "
        "changes what a single spatial feature can relate, so update this "
        "test deliberately rather than letting it drift.")


def test_base_trunk_alone_does_not_span_bridge_to_tower():
    """The limitation the context blocks exist to remove: bridge to enemy King is
    14 rows, the base trunk sees 10. If this fails, the blocks buy nothing.
    """
    net = MicroRoyaleNet(num_ability_slots=0, context_dilations=())
    rows, _ = measured_receptive_field(net)
    bridge_to_tower_rows = 14
    assert rows < bridge_to_tower_rows, (
        "the BASE trunk now spans bridge-to-tower on its own; if that is real "
        "then the context blocks are redundant and should be removed rather "
        "than left as cost.")


def test_widening_and_receptive_field_are_independent_axes():
    """Doubling the channel counts leaves the receptive field unchanged; dilation
    widens it. Built from raw layers, so the claim is about convolution
    arithmetic.
    """
    import torch.nn as nn

    def rf_of(trunk):
        # Random-input union, as in measured_receptive_field.
        rows = torch.zeros(BOARD_H, dtype=torch.bool)
        cols = torch.zeros(BOARD_W, dtype=torch.bool)
        for t in range(8):
            torch.manual_seed(1000 + t)
            x = torch.randn(1, N_CHANNELS, BOARD_H, BOARD_W).abs()
            r, c = _rf_once(trunk, x)
            rows |= r
            cols |= c
        return int(rows.sum()), int(cols.sum())

    def build(c1, c2, dilation=1):
        return nn.Sequential(
            nn.Conv2d(N_CHANNELS, c1, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(c1, c2, 3, padding=dilation, dilation=dilation),
            nn.ReLU(), nn.MaxPool2d(2, 2, ceil_mode=True))

    narrow = rf_of(build(16, 32))
    wide = rf_of(build(32, 64))
    assert narrow == wide, (
        f"width changed the receptive field ({narrow} vs {wide}); that "
        "contradicts convolution arithmetic and means this test is wrong")

    dilated = rf_of(build(16, 32, dilation=2))
    assert dilated[0] > narrow[0] and dilated[1] > narrow[1], (
        f"dilation did not widen the receptive field ({narrow} -> {dilated})")


def test_dilation_preserves_the_lstm_input_width():
    """With `padding=dilation` on a 3x3 the spatial dims are unchanged, so
    `lstm_input_dim` does not move. A checkpoint would still load, but its
    features would change meaning: shape-compatible is not
    behaviour-compatible.
    """
    import torch.nn as nn

    x = torch.zeros(1, N_CHANNELS, BOARD_H, BOARD_W)

    def out_dim(dilation):
        trunk = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(16, 32, 3, padding=dilation, dilation=dilation),
            nn.ReLU(), nn.MaxPool2d(2, 2, ceil_mode=True))
        return trunk(x).numel()

    assert out_dim(1) == out_dim(2), (
        "dilation changed the trunk output width; with padding=dilation on a "
        "3x3 kernel it must not, and if it does the shape-compatibility "
        "argument above is void")

    net = MicroRoyaleNet(num_ability_slots=0)
    assert net.cnn_out_dim == out_dim(1) == 1440
    assert net.lstm.input_size == net.cnn_out_dim + 64
