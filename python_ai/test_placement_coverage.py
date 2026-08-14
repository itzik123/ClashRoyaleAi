"""Pins the placement-gradient coverage hole and the fix for it.

Run (the .pyd is Python 3.11 only):

    python_ai/venv/Scripts/python.exe -m pytest python_ai/test_placement_coverage.py -q

WHY THESE TESTS EXIST
---------------------
`card_id_embed` is `nn.Linear(num_card_ids, 16, bias=False)`, so column c of its
weight belongs to card id c ALONE. `placement_given_card` selects only the
chosen slot's embedding, and nothing else downstream of the LSTM consumes the
others -- the card head reads `hx`, not the embeddings.

That makes the coverage hole mechanically visible rather than merely plausible:
the gradient of the loss with respect to an unchosen card's embedding column is
EXACTLY zero, in exact arithmetic, not just small. `test_unchosen_card_gets_no_
gradient` asserts the defect, and `test_coverage_pass_restores_gradient` asserts
the fix removes it. The first test is what fails on the old code path.

This is the whole root cause of the (11,0) placement collapse; see
PLACEMENT_COLLAPSE.md for the behavioural measurements.
"""
import os
import sys

import pytest
import torch
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clash_royale_env  # noqa: E402
import gym_wrapper  # noqa: E402
from model import MicroRoyaleNet  # noqa: E402
from train import PLACEMENT_COVERAGE_COEF, placement_coverage_slots  # noqa: E402

CE = clash_royale_env.ClashRoyaleEnv
L, B = 3, 2


def _fixture():
    """A tiny (L,B) chunk built from real observations, with a real hand."""
    torch.manual_seed(0)
    net = MicroRoyaleNet(num_ability_slots=0)
    deck = list(gym_wrapper.DEFAULT_DECK)
    env = CE(deck, deck, 3600)
    env.reset()
    obs = torch.tensor(env.get_observation_for_team(0), dtype=torch.float32)
    obs_seq = obs.view(1, 1, -1).expand(L, B, -1).contiguous()

    flat = obs_seq.view(L * B, -1)
    feats, embeds, spatial = net.extract_features(flat)
    feats_seq = feats.view(L, B, -1)
    embeds_seq = embeds.view(L, B, *embeds.shape[1:])
    spatial_seq = spatial.view(L, B, *spatial.shape[1:])
    # Everything affordable, so "unchosen" is a real choice and not a mask
    # artifact -- otherwise the test could pass for the wrong reason.
    card_mask = torch.ones(L, B, net.hand_size + 1, dtype=torch.bool)
    resets = torch.ones(L, B)
    hidden = (torch.zeros(B, 256), torch.zeros(B, 256))
    hand = net.hand_card_ids(flat)[0].tolist()
    return net, obs_seq, feats_seq, embeds_seq, spatial_seq, card_mask, resets, hidden, hand


def test_unchosen_card_gets_no_gradient():
    """THE DEFECT. Scoring only the chosen card starves every other card.

    Slot 0 is chosen everywhere; the assertion is that slot 1's card gets an
    exactly-zero gradient. That is what freezes an unplayed card's placement map
    at a constant cell forever.
    """
    net, obs_seq, feats_seq, embeds_seq, spatial_seq, card_mask, resets, hidden, hand = _fixture()
    chosen = torch.zeros(L, B, dtype=torch.long)

    cl, pl, values, aux, _, extra = net.forward_sequence(
        feats_seq, embeds_seq, spatial_seq, obs_seq, card_mask, chosen, resets, hidden)
    assert extra is None, "no coverage pass was requested"

    # The pre-fix loss: card log-prob + placement of the CHOSEN card only.
    loss = (Categorical(logits=cl).entropy().mean()
            + Categorical(logits=pl).entropy().mean() + values.mean())
    net.zero_grad(set_to_none=True)
    loss.backward()

    g = net.card_id_embed.weight.grad
    assert g is not None
    chosen_id, other_id = hand[0], hand[1]
    assert chosen_id != other_id
    assert g[:, chosen_id].abs().sum().item() > 0.0, "the chosen card must receive gradient"
    # Exactly zero, not merely small -- this is a structural disconnection.
    assert g[:, other_id].abs().sum().item() == 0.0, (
        "an unchosen card received placement gradient; the coverage hole this "
        "test pins has changed shape")


def test_coverage_pass_restores_gradient():
    """THE FIX. The coverage term reaches the card the actor loss cannot."""
    net, obs_seq, feats_seq, embeds_seq, spatial_seq, card_mask, resets, hidden, hand = _fixture()
    chosen = torch.zeros(L, B, dtype=torch.long)
    cover = torch.ones(L, B, dtype=torch.long)      # always slot 1

    cl, pl, values, aux, _, extra = net.forward_sequence(
        feats_seq, embeds_seq, spatial_seq, obs_seq, card_mask, chosen, resets, hidden,
        extra_card_idx_seq=cover)
    assert extra is not None and extra.shape == pl.shape

    loss = (Categorical(logits=cl).entropy().mean()
            + Categorical(logits=pl).entropy().mean() + values.mean()
            + PLACEMENT_COVERAGE_COEF * Categorical(logits=extra).entropy().mean())
    net.zero_grad(set_to_none=True)
    loss.backward()

    g = net.card_id_embed.weight.grad
    assert g[:, hand[1]].abs().sum().item() > 0.0, (
        "the coverage pass did not deliver gradient to the unchosen card")


def test_coverage_does_not_change_the_ppo_ratio():
    """The regularizer must not touch the quantity PPO is clipping.

    If the coverage pass altered the chosen action's log-prob, it would corrupt
    the ratio silently -- the exact failure mode CLAUDE.md records for masks
    that drift between rollout and update.
    """
    net, obs_seq, feats_seq, embeds_seq, spatial_seq, card_mask, resets, hidden, hand = _fixture()
    chosen = torch.zeros(L, B, dtype=torch.long)
    cover = torch.ones(L, B, dtype=torch.long)

    with torch.no_grad():
        _, pl_a, v_a, aux_a, _, _ = net.forward_sequence(
            feats_seq, embeds_seq, spatial_seq, obs_seq, card_mask, chosen, resets, hidden)
        _, pl_b, v_b, aux_b, _, extra = net.forward_sequence(
            feats_seq, embeds_seq, spatial_seq, obs_seq, card_mask, chosen, resets, hidden,
            extra_card_idx_seq=cover)

    assert torch.equal(pl_a, pl_b), "chosen-card placement logits changed"
    assert torch.equal(v_a, v_b) and torch.equal(aux_a, aux_b)


def test_coverage_slots_are_affordable_or_the_noop_fallback():
    """The sampler must never propose a slot the affordability mask forbids."""
    torch.manual_seed(1)
    hand_size = 4
    mask = torch.zeros(5, 3, hand_size + 1, dtype=torch.bool)
    mask[..., -1] = True                    # no-op always legal
    mask[0, 0, 2] = True                    # exactly one affordable card
    mask[1, :, 1] = True
    mask[2, 1, 0] = True
    idx = placement_coverage_slots(mask, hand_size)
    assert idx.shape == (5, 3)
    assert int(idx[0, 0]) == 2
    assert all(int(idx[1, b]) == 1 for b in range(3))
    assert int(idx[2, 1]) == 0
    # Rows with nothing affordable fall back to the no-op slot, which is always
    # a legal index into card_embeds (hand_size == the no-op column).
    assert int(idx[0, 1]) == hand_size
    assert idx.max().item() <= hand_size


def test_sampler_reaches_every_affordable_card():
    """Uniform over affordable slots -- otherwise coverage is itself biased."""
    torch.manual_seed(2)
    hand_size = 4
    mask = torch.zeros(400, 4, hand_size + 1, dtype=torch.bool)
    mask[..., :hand_size] = True
    idx = placement_coverage_slots(mask, hand_size)
    counts = torch.bincount(idx.view(-1), minlength=hand_size + 1)[:hand_size]
    share = counts.float() / counts.sum()
    assert share.min() > 0.20, f"a slot is being starved: {share.tolist()}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
