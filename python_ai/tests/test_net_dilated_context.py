"""Bottleneck 1: the trunk could not span a bridge push and the tower it
threatens. Pinned by GRADIENT, never by reading the layer list.

THE MEASURED FACT THIS REPLACES. `test_net_trunk_receptive_field.py` measured
the old trunk at 10x10 on a 34x18 board (3 -> 4 -> 8 -> 10, two 3x3 convs and
two 2x2 pools) and asserted, deliberately, that it did NOT reach:

    bridge y=16.5  ->  enemy King y=30.5   = 14.0 rows
    bridge y=16.5  ->  enemy Princess y=27.0 = 10.5 rows

so no single convolutional feature could relate "their win condition just
crossed" to "this is the tower it is walking at". That file said in as many
words that if the receptive field ever grew, it and CLAUDE.md "must be
rewritten together". This is that rewrite's other half.

THE FIX. A `DilatedContextBlock` appended to `cnn_trunk`, twice, at dilations
2 and 4. It runs on the POOLED 9x5 map where a cell is worth 4 input cells, so
a 3x3 kernel at dilation d reaches 8d further in input rows for the cost of a
9x5 convolution -- the cheapest place on the whole board to buy reach.
10 + 16 + 32 = 58 >= 34, i.e. the full board.

TWO PROPERTIES THAT INTERACT, AND THE TRAP BETWEEN THEM
=======================================================
The block is a residual whose final 1x1 is ZERO-INITIALISED, so at
initialisation it computes exactly `x` and a warm-started checkpoint's trunk
features are bit-identical to what they were. That is the same device
`place_hires[-1]` already uses in this net.

But zero-init means `d(out)/d(input)` through the dilated path is also exactly
zero at initialisation -- so a gradient-measured receptive field on a FRESH net
reports 10, the identity path's value, and reports it truthfully. The
instrument is not broken; it is answering "what does this net relate RIGHT NOW"
when the question is "what CAN a trained feature relate".

That is the same class of error the file this replaces already warned about
(a zeros input measures the receptive field of whichever ReLU channels happen
to be open). So the structural measurement here activates the gate first, and
`test_the_block_is_an_exact_identity_at_initialisation` pins the other half.
Both facts are real and neither alone is the whole truth.
"""
import torch

from python_ai.engine_constants import BOARD_H, BOARD_W, N_CHANNELS
from python_ai.models.net import MicroRoyaleNet
from python_ai.tests.test_net_trunk_receptive_field import measured_receptive_field

# Bridge to enemy King, in board rows, from ArenaLayout via engine_constants.
# Stated as the game distance it must cover rather than as a target number, so
# the assertion below fails for a reason someone can act on.
BRIDGE_TO_ENEMY_KING_ROWS = 14
BRIDGE_TO_ENEMY_PRINCESS_ROWS = 11


def _activate_context_gates(net, scale=0.5, seed=0):
    """Give the zero-initialised residual gates generic non-zero weights.

    Required before any GRADIENT measurement of the structural receptive
    field: with the gate at exactly zero the dilated path contributes no
    gradient at all and the measurement reports the identity path's 10x10 --
    correctly, but for a net that has not begun training. Everything else is
    left at its own initialisation.
    """
    g = torch.Generator().manual_seed(seed)
    for mod in net.cnn_trunk:
        if hasattr(mod, "expand"):
            with torch.no_grad():
                mod.expand.weight.normal_(0.0, scale, generator=g)
                mod.expand.bias.zero_()
    return net


def test_trunk_receptive_field_spans_bridge_to_the_tower_it_threatens():
    """The bottleneck, closed. Measured by gradient on the real trunk.

    14 rows is bridge-to-enemy-King; 11 covers bridge-to-enemy-Princess. Both
    were outside a 10-row field, which is precisely why a convolutional
    feature could not represent "this push threatens that tower".
    """
    net = _activate_context_gates(MicroRoyaleNet(num_ability_slots=0))
    rows, cols = measured_receptive_field(net)
    assert rows >= BRIDGE_TO_ENEMY_KING_ROWS, (
        f"trunk receptive field is {rows} rows; it must reach at least "
        f"{BRIDGE_TO_ENEMY_KING_ROWS} to relate a bridge push (y=16.5) to the "
        "enemy King (y=30.5) as a single convolutional feature")
    assert cols >= BOARD_W // 2, (
        f"trunk receptive field is {cols} columns of {BOARD_W}; it must span "
        "at least half the board to relate one lane to the other")


def test_trunk_receptive_field_covers_the_whole_board():
    """Dilations 2 and 4 at jump 4 add 16 and 32 rows to a base of 10.

    Pinned as the exact value, because a receptive field is a derived
    consequence of kernels, strides, dilations and pool count that nothing in
    the code states -- so a later "small" trunk edit can shrink it silently.
    """
    net = _activate_context_gates(MicroRoyaleNet(num_ability_slots=0))
    assert measured_receptive_field(net) == (BOARD_H, BOARD_W)


def test_the_block_is_an_exact_identity_at_initialisation():
    """Zero-initialised gate: a warm-started checkpoint's trunk is unchanged.

    This is what makes the change strictly additive rather than a silent
    reinterpretation of every existing trunk weight. Bit-identical, not close:
    the residual adds exactly `expand(...)` and `expand.weight` is exactly 0.
    """
    net = MicroRoyaleNet(num_ability_slots=0)

    blocks = [m for m in net.cnn_trunk if hasattr(m, "expand")]
    assert blocks, "no DilatedContextBlock found in cnn_trunk"

    for block in blocks:
        inp = torch.randn(4, block.channels, net.pooled_h, net.pooled_w)
        assert torch.equal(block(inp), inp), (
            "the context block is not an exact identity at initialisation; a "
            "warm-started checkpoint's trunk features would silently change "
            "meaning")


def test_hires_branch_still_receives_the_prepool_activation():
    """`cnn_trunk[:2]` must still be conv+ReLU at full board resolution.

    Two call sites split the trunk by INDEX (`extract_features_hires` and
    `hires_features`), so appending is the only safe way to extend it -- an
    insertion at the front would hand the full-resolution placement branch a
    different tensor with no error anywhere.
    """
    net = MicroRoyaleNet(num_ability_slots=0)
    x = torch.randn(2, N_CHANNELS, BOARD_H, BOARD_W)
    hires = net.cnn_trunk[:2](x)
    assert hires.shape == (2, 16, BOARD_H, BOARD_W)
    # ...and the two halves must still compose to the whole trunk.
    assert torch.equal(net.cnn_trunk[2:](hires), net.cnn_trunk(x))


def test_trunk_output_shape_and_lstm_input_width_are_unchanged():
    """The context block must not move the LSTM's input width.

    The LSTM is 1,804,288 of the net's 1.9M parameters. A change to
    `lstm_input_dim` re-shapes that tensor and discards it on load, so buying
    receptive field at the cost of the recurrent core would be a bad trade
    made silently.
    """
    net = MicroRoyaleNet(num_ability_slots=0)
    assert (net.pooled_h, net.pooled_w) == (9, 5)
    assert net.cnn_out_dim == 1440
    assert net.lstm.input_size == net.cnn_out_dim + 64
    x = torch.randn(3, N_CHANNELS, BOARD_H, BOARD_W)
    assert net.cnn_trunk(x).shape == (3, 32, 9, 5)


def test_forward_and_backward_are_finite_through_the_new_block():
    """No NaN/Inf introduced, gradients reach the new parameters.

    The gate is zero-initialised, so this also proves the block is REACHABLE:
    a zero gate still receives gradient (its input activation is non-zero),
    even though it emits none. A block that never learned would be dead code
    dressed as an architecture change.
    """
    net = _activate_context_gates(MicroRoyaleNet(num_ability_slots=0))
    x = torch.randn(3, N_CHANNELS, BOARD_H, BOARD_W)
    out = net.cnn_trunk(x)
    assert torch.isfinite(out).all()
    out.pow(2).mean().backward()

    blocks = [m for m in net.cnn_trunk if hasattr(m, "expand")]
    assert blocks, "no DilatedContextBlock found: this test would pass vacuously"
    for block in blocks:
        for name, p in block.named_parameters():
            assert p.grad is not None, f"{name} received no gradient"
            assert torch.isfinite(p.grad).all(), f"{name} grad not finite"
            assert p.grad.abs().sum() > 0, (
                f"{name} received an all-zero gradient: the block is dead")


def test_gradient_at_initialisation_flows_only_through_the_identity_path():
    """The trap, pinned so nobody re-derives it from a confusing measurement.

    On a FRESH net the measured receptive field is still 10x10 -- not because
    the dilated path is absent but because a zero gate emits zero gradient.
    Anyone measuring the structural field must activate the gate first, and
    if this test ever fails the initialisation stopped being an identity.
    """
    fresh = MicroRoyaleNet(num_ability_slots=0)
    assert measured_receptive_field(fresh) == (10, 10)


def test_an_existing_checkpoint_warm_starts_into_an_exact_identity():
    """The whole point of the zero-init gate, end to end.

    A checkpoint saved from the PREVIOUS trunk has no `cnn_trunk.6/7.*` keys.
    `load_state_dict_flexible` warm-starts what it can and leaves the rest at
    initialisation -- which for this block is exactly zero, so the loaded net
    computes bit-identically what it computed before the architecture changed.

    Without the zero-init this would be the dangerous kind of failure the old
    receptive-field test already names: shape-compatible but not
    behaviour-compatible, degrading silently instead of raising.
    """
    from python_ai.models.policy_io import load_state_dict_flexible

    old = MicroRoyaleNet(num_ability_slots=0, context_dilations=())
    new = MicroRoyaleNet(num_ability_slots=0)
    load_state_dict_flexible(new, old.state_dict(), "warm-start test")

    x = torch.randn(4, N_CHANNELS, BOARD_H, BOARD_W)
    assert torch.equal(new.cnn_trunk(x), old.cnn_trunk(x)), (
        "a warm-started checkpoint does not reproduce its own trunk features; "
        "the context block is not an identity after load")
