"""Bottleneck 2: the scalar encoder was one `Linear(1124 -> 64)` serving four
inputs with nothing in common, and the opponent's card cycle had just been
appended to it.

WHAT WAS ACTUALLY WRONG, STATED PRECISELY. It is tempting to say a 17.6:1
compression "destroys" the cycle, and that claim does not survive contact with
the mathematics: a random projection of 370 dims into 64 preserves pairwise
structure rather well (Johnson-Lindenstrauss), so at INITIALISATION the cycle
information is largely still there. Asserting otherwise would be a test that
sounds decisive and measures nothing.

The real defect is about LEARNING, not information. In one `Linear(1124, 64)`
every output dimension is a row of a single weight matrix, and that row spans
ALL 1124 inputs at once. So the columns carrying the opponent's cycle share
their outputs with the 740 hand one-hot columns, and a gradient step taken to
improve hand encoding necessarily rewrites the same rows the cycle is read
through. The cycle is not compressed away -- it is *continuously perturbed by
something else's learning*, which is worse, because it never converges.

That is a structural property, it is decidable without a training run, and
`test_learning_about_the_hand_cannot_perturb_the_cycle_encoding` below is the
decisive test: it takes a real optimizer step on a hand-only objective and asks
whether the encoder's RESPONSE to the cycle changed. The monolithic encoder
fails it and the branched one passes it bit-identically.

THE FIX. Four semantic branches whose widths sum to exactly 64, so
`lstm_input_dim` stays 1504 and the 1,804,288-parameter LSTM -- 96% of the net
-- keeps its shape and its checkpoint:

    econ    elixir + the 4 hand costs      5   ->  8
    hand    4 one-hots, shared per-slot  740   -> 20   (Linear(185, 5) per slot)
    extra   time, both spends, 6 tower HP  9   -> 12   (an EXPANSION, not a
                                                        compression: these 9
                                                        decide who is winning
                                                        and were sharing 64
                                                        outputs with 740
                                                        one-hot dims)
    cycle   seen[185] + recency[185]     370   -> 24

The cycle branch does not flatten its 370 inputs into a dense layer. `seen` and
`recency` index the SAME card space, so both are projected through one shared
`Linear(num_card_ids, 16)`: `seen @ W` is the sum of the embeddings of the
cards they have shown, and `recency @ W` is the same sum weighted by how
recently. That is the quantity a human tracks, it is 2,960 parameters instead
of a dense layer's 8,880, and it makes the two blocks share one notion of card
identity rather than learning it twice.

A SEPARATE embedding from `card_id_embed`, deliberately. Sharing would be
cheaper and would let the cycle speak the hand's language immediately, but it
also routes cycle gradient into a component that already works and is read by
the placement head. This change is meant to be strictly additive; coupling it
to a working module is the opposite of that.
"""
import torch

from python_ai.models.net import MicroRoyaleNet


def _obs_batch(net, n=8, seed=0):
    """A batch of whole observations (spatial + scalar), shaped as the net eats."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, net.spatial_size + net.scalar_size, generator=g)


def _scalar_part(net, obs):
    return obs[:, net.spatial_size:]


def _with_cycle(net, scalar, value, seed=1):
    """Copy of `scalar` with ONLY the opponent-cycle block replaced."""
    g = torch.Generator().manual_seed(seed)
    out = scalar.clone()
    block = torch.rand(scalar.shape[0], net.cycle_block_size, generator=g) * value
    out[:, net.cycle_start:net.cycle_start + net.cycle_block_size] = block
    return out


def test_scalar_encoder_output_width_is_unchanged_so_the_lstm_is_untouched():
    """64 in total, and therefore `lstm_input_dim` 1504 and the LSTM intact.

    The branch widths are allowed to be retuned; their SUM is not, without
    knowingly discarding 1.8M trained parameters.
    """
    net = MicroRoyaleNet(num_ability_slots=0)
    scalar = _scalar_part(net, _obs_batch(net))
    assert net.scalar_mlp(scalar).shape == (8, 64)
    assert net.lstm.input_size == net.cnn_out_dim + 64 == 1504
    assert sum(p.numel() for p in net.lstm.parameters()) == 1_804_288


def test_the_scalar_encoder_reads_its_blocks_from_the_engines_own_offsets():
    """No second copy of the observation layout.

    `extra_start` and `cycle_start` are forward offsets bound from the engine
    (CLAUDE.md: locate a section by a FORWARD offset, never by subtracting from
    the end). The encoder must consume those, not restate them -- the last
    thing that restated this layout read card-recency floats as tower HP.
    """
    net = MicroRoyaleNet(num_ability_slots=0)
    enc = net.scalar_mlp
    assert enc.cycle_start == net.cycle_start
    assert enc.extra_start == net.extra_start
    assert enc.cycle_block_size == net.cycle_block_size
    # ...and the blocks must tile the scalar vector exactly, with no gap and no
    # overlap. A gap is a silently ignored input; an overlap is double-counting.
    assert net.extra_start + net.num_extra_scalars == net.cycle_start
    assert net.cycle_start + net.cycle_block_size == net.scalar_size


def test_the_cycle_block_owns_a_dedicated_slice_of_the_scalar_features():
    """Changing ONLY the opponent's cycle must move only the cycle's own dims.

    This is the property the monolithic layer could not have: there, every one
    of the 64 outputs is a function of all 1124 inputs.
    """
    net = MicroRoyaleNet(num_ability_slots=0)
    scalar = _scalar_part(net, _obs_batch(net))
    lo, hi = net.scalar_mlp.cycle_slice

    a = net.scalar_mlp(_with_cycle(net, scalar, 0.1, seed=1))
    b = net.scalar_mlp(_with_cycle(net, scalar, 0.9, seed=2))

    assert not torch.equal(a[:, lo:hi], b[:, lo:hi]), (
        "the cycle block does not reach its own output slice at all")
    outside = torch.cat([a[:, :lo], a[:, hi:]], dim=1)
    outside_b = torch.cat([b[:, :lo], b[:, hi:]], dim=1)
    assert torch.equal(outside, outside_b), (
        "changing the opponent's cycle moved outputs outside the cycle slice; "
        "the branches are not actually independent")


def test_learning_about_the_hand_cannot_perturb_the_cycle_encoding():
    """THE decisive test, and the one the monolithic encoder fails.

    Takes a real optimizer step on an objective that depends only on the hand
    branch's outputs, then asks whether the encoder's RESPONSE to the
    opponent's cycle changed. Phrased as a response difference rather than as
    "the cycle slice moved" so that the identical question can be put to an
    encoder that has no cycle slice -- which is exactly what the contrast test
    below does.
    """
    net = MicroRoyaleNet(num_ability_slots=0)
    enc = net.scalar_mlp
    scalar = _scalar_part(net, _obs_batch(net))
    lo, hi = enc.cycle_slice

    def cycle_response():
        with torch.no_grad():
            return (enc(_with_cycle(net, scalar, 0.9, seed=2))
                    - enc(_with_cycle(net, scalar, 0.1, seed=1)))

    before = cycle_response()
    opt = torch.optim.SGD(enc.parameters(), lr=0.5)
    for _ in range(3):
        opt.zero_grad()
        enc(scalar)[:, :lo].pow(2).mean().backward()
        opt.step()
    after = cycle_response()

    assert torch.equal(before, after), (
        "a gradient step taken to improve NON-cycle encoding changed how the "
        "encoder responds to the opponent's card cycle; the branches share "
        "weights and the cycle is still being overwritten by other learning")


def test_a_monolithic_scalar_layer_fails_that_same_question():
    """The contrast that makes the test above mean something.

    Without this, `test_learning_about_the_hand_cannot_perturb_the_cycle_
    encoding` could be passing for a trivial reason (a dead branch, a zeroed
    gradient) and nobody would know. Here the SAME procedure is applied to the
    encoder this change replaces, and it must come out differently -- otherwise
    the procedure is not measuring what it claims.
    """
    net = MicroRoyaleNet(num_ability_slots=0, branched_scalars=False)
    enc = net.scalar_mlp
    scalar = _scalar_part(net, _obs_batch(net))

    def cycle_response():
        with torch.no_grad():
            return (enc(_with_cycle(net, scalar, 0.9, seed=2))
                    - enc(_with_cycle(net, scalar, 0.1, seed=1)))

    before = cycle_response()
    opt = torch.optim.SGD(enc.parameters(), lr=0.5)
    for _ in range(3):
        opt.zero_grad()
        enc(scalar)[:, :20].pow(2).mean().backward()
        opt.step()
    after = cycle_response()

    assert not torch.equal(before, after), (
        "the monolithic Linear(1124, 64) did NOT change its cycle response "
        "after a hand-only step; if that is true then the premise of this "
        "whole bottleneck is wrong and the branched encoder buys nothing")


def test_every_branch_receives_gradient_and_the_output_is_finite():
    """A branch that never learns is dead code dressed as an architecture."""
    net = MicroRoyaleNet(num_ability_slots=0)
    enc = net.scalar_mlp
    scalar = _scalar_part(net, _obs_batch(net))
    out = enc(scalar)
    assert torch.isfinite(out).all()
    out.pow(2).mean().backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} grad not finite"
        assert p.grad.abs().sum() > 0, f"{name} is dead: all-zero gradient"


def test_the_branched_encoder_is_cheaper_than_the_layer_it_replaces():
    """It buys non-interference and costs fewer parameters, not more.

    The monolithic layer spends 72,000 parameters mapping 740 sparse one-hot
    dims densely into every output. Encoding each hand slot through one shared
    per-slot embedding does the same job structurally and far more cheaply.
    """
    branched = MicroRoyaleNet(num_ability_slots=0).scalar_mlp
    mono = MicroRoyaleNet(num_ability_slots=0,
                          branched_scalars=False).scalar_mlp
    nb = sum(p.numel() for p in branched.parameters())
    nm = sum(p.numel() for p in mono.parameters())
    assert nb < nm, f"branched encoder is not cheaper: {nb:,} vs {nm:,}"
