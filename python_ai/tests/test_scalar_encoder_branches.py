"""The scalar encoder is four semantic branches, not one dense layer.

The defect of a single `Linear(1124, 64)` is about learning, not information: a
random projection of the cycle into 64 dims largely preserves it, but every
output row spans all inputs, so a gradient step improving the hand encoding
rewrites the rows the cycle is read through, and the cycle never converges.
That is decidable without training, and the decisive test below takes a real
optimizer step on a hand-only objective.

Branch widths sum to 64, so the LSTM keeps its shape:

    econ    elixir + the 4 hand costs      5   ->  8
    hand    4 one-hots, shared per slot  740   -> 20   (Linear(185, 5) per slot)
    extra   time, both spends, 6 tower HP  9   -> 12   (an expansion)
    cycle   seen[185] + recency[185]     370   -> 24

`seen` and `recency` share one `Linear(num_card_ids, 16)` card projection: the
sum of the embeddings of cards shown, and the same weighted by recency. It is
separate from `card_id_embed`, so cycle gradients cannot disturb the working
placement path.
"""
import torch

from python_ai.models.net import MicroRoyaleNet


def _obs_batch(net, n=8, seed=0):
    """A batch of whole observations (spatial + scalar), as the net takes them.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, net.spatial_size + net.scalar_size, generator=g)


def _scalar_part(net, obs):
    return obs[:, net.spatial_size:]


def _with_cycle(net, scalar, value, seed=1):
    """Copy of `scalar` with only the opponent-cycle block replaced."""
    g = torch.Generator().manual_seed(seed)
    out = scalar.clone()
    block = torch.rand(scalar.shape[0], net.cycle_block_size, generator=g) * value
    out[:, net.cycle_start:net.cycle_start + net.cycle_block_size] = block
    return out


def test_scalar_encoder_output_width_is_unchanged_so_the_lstm_is_untouched():
    """The branch widths may be retuned; their sum may not, without discarding the
    LSTM.
    """
    net = MicroRoyaleNet(num_ability_slots=0)
    scalar = _scalar_part(net, _obs_batch(net))
    assert net.scalar_mlp(scalar).shape == (8, 64)
    assert net.lstm.input_size == net.cnn_out_dim + 64 == 1504
    assert sum(p.numel() for p in net.lstm.parameters()) == 1_804_288


def test_the_scalar_encoder_reads_its_blocks_from_the_engines_own_offsets():
    """The encoder consumes the engine's forward offsets rather than restating the
    layout.
    """
    net = MicroRoyaleNet(num_ability_slots=0)
    enc = net.scalar_mlp
    assert enc.cycle_start == net.cycle_start
    assert enc.extra_start == net.extra_start
    assert enc.cycle_block_size == net.cycle_block_size
    # ...and the blocks tile the scalar vector exactly: a gap is an ignored
    # input, an overlap double counts.
    assert net.extra_start + net.num_extra_scalars == net.cycle_start
    assert net.cycle_start + net.cycle_block_size == net.scalar_size


def test_the_cycle_block_owns_a_dedicated_slice_of_the_scalar_features():
    """Changing only the opponent's cycle moves only the cycle's own output dims.
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
    """The decisive test: after a real optimizer step on a hand-only objective,
    the encoder's response to the cycle is unchanged. Phrased as a response
    difference so the same question can be put to the monolithic encoder below.
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
    """The contrast: the monolithic encoder must fail the same procedure, or the
    test above could be passing for a trivial reason.
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
    """A branch that never learns is dead code. The cycle branch is exempt: it is
    detached in `forward` and trained only by its identity head (next test).
    """
    net = MicroRoyaleNet(num_ability_slots=0)
    enc = net.scalar_mlp
    scalar = _scalar_part(net, _obs_batch(net))
    out = enc(scalar)
    assert torch.isfinite(out).all()
    out.pow(2).mean().backward()
    detached = {"cycle_card.weight", "cycle_out.weight", "cycle_out.bias"}
    for name, p in enc.named_parameters():
        if name in detached:
            continue
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} grad not finite"
        assert p.grad.abs().sum() > 0, f"{name} is dead: all-zero gradient"


def test_the_cycle_branch_is_reachable_only_from_the_identity_head():
    """Gradient isolation from both sides: nothing reading the encoder output may
    reshape the cycle branch, but the identity head must reach it, and must
    stop there. One-sided, the test would pass on a dead branch.
    """
    net = MicroRoyaleNet(num_ability_slots=0)
    obs = _obs_batch(net)
    cyc_params = [net.scalar_mlp.cycle_card.weight,
                  net.scalar_mlp.cycle_out.weight,
                  net.scalar_mlp.cycle_out.bias]

    def grad_norms(loss):
        net.zero_grad()
        loss.backward()
        return [0.0 if p.grad is None else float(p.grad.norm())
                for p in cyc_params]

    # 1. Nothing reading the encoder's output may reshape the branch; the
    #    policy path is downstream of that output.
    assert grad_norms(net.scalar_mlp(_scalar_part(net, obs)).pow(2).mean()) \
        == [0.0, 0.0, 0.0]

    # 2. The skip into the heads carries no gradient path either: actor and
    #    critic read these dims without owning them.
    feats, _, _, _ = net.extract_features_hires(obs)
    skip = net._split_cycle(feats)
    assert skip.shape[-1] == net.cycle_feature_dim
    assert skip.requires_grad is False

    # 3. The identity head must reach all three, or the branch is dead.
    ident = net.predict_cycle_card(net.cycle_features(obs)).pow(2).mean()
    assert all(g > 0.0 for g in grad_norms(ident))

    # 4. ...and its gradient stops at the branch; reaching the LSTM would make
    #    it a second aux task.
    net.zero_grad()
    net.predict_cycle_card(net.cycle_features(obs)).pow(2).mean().backward()
    assert net.lstm.weight_ih.grad is None
    assert net.cnn_trunk[0].weight.grad is None


def test_the_branched_encoder_is_cheaper_than_the_layer_it_replaces():
    """Non-interference at fewer parameters: sparse one-hots through one shared
    per-slot embedding, instead of a dense map into every output.
    """
    branched = MicroRoyaleNet(num_ability_slots=0).scalar_mlp
    mono = MicroRoyaleNet(num_ability_slots=0,
                          branched_scalars=False).scalar_mlp
    nb = sum(p.numel() for p in branched.parameters())
    nm = sum(p.numel() for p in mono.parameters())
    assert nb < nm, f"branched encoder is not cheaper: {nb:,} vs {nm:,}"
