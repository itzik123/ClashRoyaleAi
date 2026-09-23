"""The trunk spans a bridge push and the tower it threatens. Pinned by gradient,
never by reading the layer list.

The base trunk sees 10 rows (see test_net_trunk_receptive_field.py), while
bridge (y=16.5) to enemy King (y=30.5) is 14 rows. `DilatedContextBlock`s
appended to `cnn_trunk` (CONTEXT_DILATIONS) run on the pooled 9x5 map, where a
3x3 at dilation d reaches 8d further input rows for the cost of a 9x5
convolution, covering the whole board.

The block's final 1x1 is zero-initialised, so at init it is exactly the
identity and a warm-started trunk is bit-identical. That also means no gradient
flows through the dilated path at init: a gradient-measured field on a fresh
net truthfully reports the identity path's 10 rows. The structural measurement
therefore activates the gate first, and the identity test pins the other half.
"""
import torch

from python_ai.engine_constants import BOARD_H, BOARD_W, N_CHANNELS
from python_ai.models.net import MicroRoyaleNet
from python_ai.tests.test_net_trunk_receptive_field import measured_receptive_field

# Bridge to enemy King / Princess in board rows (ArenaLayout via
# engine_constants), stated as distances so a failure says what to fix.
BRIDGE_TO_ENEMY_KING_ROWS = 14
BRIDGE_TO_ENEMY_PRINCESS_ROWS = 11


def _activate_context_gates(net, scale=0.5, seed=0):
    """Give the zero-initialised residual gates generic non-zero weights, as
    required before a gradient measurement of the structural field. Everything
    else keeps its initialisation.
    """
    g = torch.Generator().manual_seed(seed)
    for mod in net.cnn_trunk:
        if hasattr(mod, "expand"):
            with torch.no_grad():
                mod.expand.weight.normal_(0.0, scale, generator=g)
                mod.expand.bias.zero_()
    return net


def test_trunk_receptive_field_spans_bridge_to_the_tower_it_threatens():
    """Measured by gradient on the real trunk: bridge-to-King and
    bridge-to-Princess were both outside the base 10-row field.
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
    """Pinned as the exact value: the receptive field follows from kernels,
    strides, dilations and pools, none of which states it, so a small trunk
    edit could shrink it silently.
    """
    net = _activate_context_gates(MicroRoyaleNet(num_ability_slots=0))
    assert measured_receptive_field(net) == (BOARD_H, BOARD_W)


def test_the_block_is_an_exact_identity_at_initialisation():
    """Zero-initialised gate: bit-identical, not close, so the change is strictly
    additive for existing trunk weights.
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
    """`cnn_trunk[:2]` is still conv+ReLU at full resolution. Two call sites slice
    the trunk by index, so blocks may only be appended; an insertion would hand
    the placement branch a different tensor with no error.
    """
    net = MicroRoyaleNet(num_ability_slots=0)
    x = torch.randn(2, N_CHANNELS, BOARD_H, BOARD_W)
    hires = net.cnn_trunk[:2](x)
    assert hires.shape == (2, 16, BOARD_H, BOARD_W)
    # ...and the two halves still compose to the whole trunk.
    assert torch.equal(net.cnn_trunk[2:](hires), net.cnn_trunk(x))


def test_trunk_output_shape_and_lstm_input_width_are_unchanged():
    """The block must not move the LSTM's input width: the LSTM holds nearly all
    the parameters, and a new `lstm_input_dim` discards them on load.
    """
    net = MicroRoyaleNet(num_ability_slots=0)
    assert (net.pooled_h, net.pooled_w) == (9, 5)
    assert net.cnn_out_dim == 1440
    assert net.lstm.input_size == net.cnn_out_dim + 64
    x = torch.randn(3, N_CHANNELS, BOARD_H, BOARD_W)
    assert net.cnn_trunk(x).shape == (3, 32, 9, 5)


def test_forward_and_backward_are_finite_through_the_new_block():
    """No NaN/Inf, and the block is reachable: a zero gate still receives gradient
    (its input is non-zero) though it emits none.
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
    """On a fresh net the measured field is still 10x10 because a zero gate emits
    zero gradient. If this fails, initialisation stopped being an identity.
    """
    fresh = MicroRoyaleNet(num_ability_slots=0)
    assert measured_receptive_field(fresh) == (10, 10)


def test_an_existing_checkpoint_warm_starts_into_an_exact_identity():
    """A checkpoint from the previous trunk has no context-block keys; loaded
    flexibly, the zero-initialised blocks leave it computing bit-identically
    what it did before.
    """
    from python_ai.models.policy_io import load_state_dict_flexible

    old = MicroRoyaleNet(num_ability_slots=0, context_dilations=())
    new = MicroRoyaleNet(num_ability_slots=0)
    load_state_dict_flexible(new, old.state_dict(), "warm-start test")

    x = torch.randn(4, N_CHANNELS, BOARD_H, BOARD_W)
    assert torch.equal(new.cnn_trunk(x), old.cnn_trunk(x)), (
        "a warm-started checkpoint does not reproduce its own trunk features; "
        "the context block is not an identity after load")
