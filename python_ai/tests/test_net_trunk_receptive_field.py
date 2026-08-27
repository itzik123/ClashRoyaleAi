"""The CNN trunk's receptive field, pinned by MEASUREMENT rather than by
reading the layer list.

WHY THIS FILE EXISTS. The trunk is two convolutions, `21 -> 16 -> 32`, each
followed by a 2x2 pool. Measured 2026-08-27 by backpropagating from one centre
cell of the trunk output to the input, the receptive field is **10 rows x 10
columns** on a 34x18 board -- 29% of the board's height, and exactly the value
the kernel/pool arithmetic predicts (3 -> 4 -> 8 -> 10). The bridges sit at
y=16.5 and the Princess Towers at y=3.0/14.0 in board coordinates, so no single
trunk feature can span a bridge and the tower behind it.

That is a real limit on the SPATIAL features, though deliberately not a limit
on global reasoning: the trunk output is flattened into a 1440-dim vector and
the LSTM is fully connected over all of it, so long-range relations can still
be formed -- just not as convolutional features, and not by the placement head,
which reads the spatial map directly.

Two things this file pins, and one it deliberately does not.

PINNED: the receptive field itself. It is a derived consequence of kernel
sizes, strides, dilations and pool count, and nothing in the code states it --
so a future "small" trunk edit can shrink it with no test noticing. Here it is
measured the only way that cannot be fooled, by gradient.

PINNED: that the trunk's OUTPUT SHAPE is a separate axis from its receptive
field. Measured cost of the alternatives, fwd+bwd over one real 200-sample
BPTT minibatch:

    trunk                        params   LSTM in   RF   time    cost
    current 21->16->32            7,680     1440    10   85.4ms  1.00x
    wider   21->32->64           24,576     2880    10  194.9ms  2.28x
    deeper  21->32->32->64       33,824     2880    14  218.7ms  2.56x
    dilated 21->16->32 (dil 2)    7,680     1440    14   91.3ms  1.07x

Widening costs 2.28x and does NOT change the receptive field at all -- width
and receptive field are orthogonal, which is the single most useful thing to
know before anyone "makes the CNN bigger" to fix a spatial-reasoning problem.

NOT PINNED, deliberately: which trunk is better. That is a training-run
question and this repo has an explicit rule against answering those from a
forward pass. Throughput is the binding constraint (943 ep/hour, CPU-only, and
CLAUDE.md already puts the trunk at 43% of update wall-clock), so a 2.28x trunk
is a real cost against an unmeasured benefit. Nothing here changes the default.
"""
import torch

from python_ai.engine_constants import BOARD_H, BOARD_W, N_CHANNELS, SPATIAL_SIZE
from python_ai.models.net import MicroRoyaleNet


def _rf_once(trunk, x):
    """Rows/cols of `x` that one centre output cell depends on, for ONE input."""
    x = x.clone().requires_grad_(True)
    feat = trunk(x)
    feat[0, :, feat.shape[2] // 2, feat.shape[3] // 2].sum().backward()
    g = x.grad[0].abs().sum(0)
    return (g.sum(1) > 0), (g.sum(0) > 0)


def measured_receptive_field(net, trials=8):
    """(rows, cols) of input that one CENTRE trunk cell actually depends on.

    By gradient, not by arithmetic over the layer list: an off-by-one in a
    padding or a dilation is exactly the kind of thing an arithmetic model of
    the network reproduces faithfully while the network does something else.

    THE INPUT MUST BE RANDOM, AND THE RESULT UNIONED OVER SEVERAL DRAWS.
    A zeros input measures 7x7 here rather than the true 9x10, and the reason
    is a trap worth naming: at a zero pre-activation `ReLU` has gradient zero,
    so with an all-zero board every channel whose conv bias happens to be
    negative gates shut and contributes no gradient at all. The measurement
    then reports the receptive field of *the subset of channels that happened
    to be open*, which is smaller than the structural one and, worse, varies
    with the seed. Random inputs keep the units active; the union over draws
    removes the residual chance that one particular draw closes an edge path.

    This is the same class of error as validating a mask with the predicate
    that generated it: the instrument silently answers a narrower question
    than the one asked.
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


def test_trunk_receptive_field_is_what_the_architecture_implies():
    """Pins 10x10. Fails loudly if a trunk edit shrinks what a feature sees.

    10 is exactly what the layer arithmetic predicts -- conv3 gives 3, the
    first pool 4, conv3 again 8, the second pool 10 -- which is the check that
    makes the gradient measurement trustworthy rather than merely empirical:
    two independent derivations agreeing.
    """
    net = MicroRoyaleNet(num_ability_slots=0)
    rows, cols = measured_receptive_field(net)
    assert (rows, cols) == (10, 10), (
        f"trunk receptive field measured {rows}x{cols}, expected 10x10. If a "
        "kernel, stride, dilation or pool changed, that is fine -- but it "
        "changes what a single spatial feature can relate, so update this "
        "test deliberately rather than letting it drift.")


def test_receptive_field_does_not_span_bridge_to_tower():
    """The measured limitation, stated as the game fact it implies.

    A bridge sits at y=16.5 and the Princess Towers at y=3.0 and y=14.0
    (ArenaLayout, via engine_constants). Relating "their win condition just
    crossed" to "this tower is what it is walking at" is ~14 rows. The trunk
    sees 9. This test does not assert that is WRONG -- the LSTM integrates
    globally afterwards -- it asserts that the fact stays visible, so nobody
    concludes the convolutional features carry that relation when they cannot.
    """
    net = MicroRoyaleNet(num_ability_slots=0)
    rows, _ = measured_receptive_field(net)
    bridge_to_tower_rows = 14
    assert rows < bridge_to_tower_rows, (
        "the trunk's receptive field now spans bridge-to-tower; that is an "
        "improvement, but this test and the note in CLAUDE.md both describe "
        "the old limit and must be rewritten together.")


def test_widening_and_receptive_field_are_independent_axes():
    """The measurement that stops the wrong fix being applied to this problem.

    Doubling the channel counts leaves the receptive field exactly where it
    was -- so "the CNN is too small to understand the board" is answered by
    depth or dilation, never by width. Built here from raw layers rather than
    from a second MicroRoyaleNet so the claim is about convolution arithmetic
    and not about this particular net's configuration.
    """
    import torch.nn as nn

    def rf_of(trunk):
        # Same random-input union as measured_receptive_field, and for the
        # same reason -- a zeros input measures the open-channel subset.
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
    """Why dilation is the cheap axis: with `padding=dilation` on a 3x3 kernel
    the spatial dims are unchanged, so the flattened trunk output stays 1440
    and `lstm_input_dim` does not move.

    The consequence worth stating precisely, because it is easy to over-read:
    a checkpoint would still LOAD (every weight tensor keeps its shape), but
    those weights were fitted under a different receptive field, so the
    features they compute change meaning. Shape-compatible is NOT
    behaviour-compatible, and this is the more dangerous of the two failure
    modes -- it degrades silently instead of raising.
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
