"""Placement coverage: keeping the placement map of an unplayed card alive.

The actor and entropy terms only reach the chosen card's placement map, so a
card the policy stops playing gets no placement gradient; its map freezes, the
frozen cell makes it worthless, and it stays unplayed. This term adds a
placement gradient for one affordable card per step, chosen or not.
`placement_coverage_slots` samples the slot; `rl.ppo` applies the term.
"""
import os

import torch

# Env-overridable so the term can be ablated with byte-identical code (0 =
# control arm).
PLACEMENT_COVERAGE_COEF = float(os.environ.get("CLASH_PLACEMENT_COVERAGE_COEF", 0.02))

def placement_coverage_slots(card_mask_seq, hand_size, slot_weights=None):
    """(L, B) long: one sampled affordable hand slot per timestep.

    Rows with nothing affordable fall back to the no-op slot; they are masked
    out of the term anyway.

    `slot_weights` (L, B, hand_size) reweights the draw, e.g. toward cards with
    an advisor rule: a uniform draw over affordable slots favours cheap cards,
    which already get actor gradient. None draws uniformly.
    """
    L, B, _ = card_mask_seq.shape
    playable = card_mask_seq[..., :hand_size].reshape(L * B, hand_size).float()
    if slot_weights is not None:
        playable = playable * slot_weights.reshape(L * B, hand_size).float()
        # If the weights zero every affordable slot, fall back to the
        # unweighted mask: weights may re-rank candidates, never remove them.
        empty = playable.sum(-1) <= 0
        if bool(empty.any()):
            base = card_mask_seq[..., :hand_size].reshape(L * B, hand_size).float()
            playable = torch.where(empty.unsqueeze(-1), base, playable)
    none = playable.sum(-1) <= 0
    # The all-zero rows get a dummy distribution so multinomial cannot raise;
    # they are overwritten below.
    probs = torch.where(none.unsqueeze(-1), torch.ones_like(playable), playable)
    idx = torch.multinomial(probs, 1).squeeze(-1)
    idx = torch.where(none, torch.full_like(idx, hand_size), idx)
    return idx.view(L, B)
