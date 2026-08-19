"""Placement-gradient COVERAGE: keeping the map of an unplayed card alive.

`placement_coverage_slots` is the sampler; `rl.ppo` applies the term. The
measured defect it closes, and why entropy alone was not enough, is in
`PLACEMENT_COVERAGE_COEF`'s comment below and in `advisors/advisor_target.py`.
"""
import os

import torch

# --- placement coverage -----------------------------------------------------
# THE BUG THIS EXISTS FOR, measured 2026-08-14 on model_weights_dist_e3.pth.
#
# Both the actor loss and the placement entropy bonus flow through
# `placement_given_card` for the card that was CHOSEN and no other. A card the
# policy has stopped playing therefore receives ZERO placement gradient from
# either term, forever. Its conditional map freezes at whatever it happened to
# be and drifts only as the shared trunk moves under it.
#
# That is a self-sustaining deadlock, not a transient: the frozen cell makes the
# card worthless, worthlessness keeps the card head from selecting it, and not
# being selected keeps the head frozen. No amount of additional training escapes
# it, which is why "train it longer" had not worked.
#
# Measured, 40 greedy episodes at 1.5x opponent elixir:
#
#   card       modal cell   modal share   plays   H(place|card)
#   Cannon       (11,0)        91.0%        24        0.098
#   Fireball     (11,0)        58.4%         2        0.141
#   Giant        (11,0)        54.1%         2        0.147
#   Mini PEKKA   (14,15)       19.0%       230        0.086
#
# Mini PEKKA has the LOWEST entropy of the four and is the most-played card, so
# entropy does not separate them -- modal-cell stability across states does. And
# the collapse is not a valuation: scored by tower HP preserved over a Cannon's
# full 300-tick life across 449 threatened states, the policy's own cell saved
# 121 HP against 396 for a RANDOM legal cell (paired -274 HP, 95% CI
# [-328, -221]). A policy cannot be correctly valuing a card it places
# significantly worse than chance.
#
# Dating it: the phase-1 net from 2026-08-09 placed Cannon at (3,15) with a 9.5%
# modal share and H=0.368 -- healthy and state-dependent. The collapse appears in
# the 2026-08-11 net, bracketing the entropy-masking change of that day
# (e16cdd7, "Measure placement entropy on real placements, not on no-ops").
# That change was CORRECT for the defect it targeted, and it had an unmeasured
# side effect: the no-op steps it stopped rewarding were the only thing holding
# open the placement maps of cards that are never played. Note the signature --
# in the pre-fix net Cannon and Giant had the two HIGHEST per-card placement
# entropies; after it they have the lowest. The rank order inverted for exactly
# the unplayed cards, which is what this mechanism predicts and little else does.
#
# The fix restores a placement gradient for affordable-but-unchosen cards
# without reintroducing the measurement bug: the REPORTED
# Entropy/Placement_Measured still averages over real placements only.
#
# Cost is real: the placement head is ~41% of update time and this runs it a
# second time. Measured wall clock is in the commit message. It is the cheapest
# correct option -- covering all 4 slots every step would be ~4x, and sampling
# one slot uniformly reaches every card ~1000 times per rollout, which is ample.
#
# Overridable from the environment ONLY so the fix can be ablated against
# itself: the A/B that justifies this term sets CLASH_PLACEMENT_COVERAGE_COEF=0
# for the control arm and leaves the default for the treatment. Both arms then
# run byte-identical code, which is the only way the comparison attributes the
# difference to the term rather than to two different scripts.
PLACEMENT_COVERAGE_COEF = float(os.environ.get("CLASH_PLACEMENT_COVERAGE_COEF", 0.02))

def placement_coverage_slots(card_mask_seq, hand_size, slot_weights=None):
    """(L,B) long: one sampled AFFORDABLE hand slot per timestep.

    Rows with nothing affordable fall back to the no-op slot. Those rows are
    masked out of the coverage term anyway (mb_decision is 0 exactly there), so
    the fallback only has to be a legal index, never a meaningful one.

    `slot_weights` (L,B,hand_size) multiplies the per-slot sampling probability,
    for concentrating the coverage budget where it is needed. Left None the
    draw is uniform over affordable slots, which is the original behaviour.

    WHY THE WEIGHTS EXIST, measured 2026-08-14. The coverage budget is one slot
    per step, and the cards it has to reach -- Cannon(3), Fireball(4), Giant(5)
    -- are exactly the ones affordability hides: the agent sits under 3 elixir
    on 65.3% of decisions, so a uniform draw over AFFORDABLE slots is biased
    toward the cheap cards, which are also the ones already receiving actor
    gradient because they are the ones being played. Weighting toward the cards
    with an advisor rule spends a scarce budget on the starved cards instead.
    """
    L, B, _ = card_mask_seq.shape
    playable = card_mask_seq[..., :hand_size].reshape(L * B, hand_size).float()
    if slot_weights is not None:
        playable = playable * slot_weights.reshape(L * B, hand_size).float()
        # A row where every AFFORDABLE slot got weight 0 would be indistinguishable
        # from a row with nothing affordable. Restore the unweighted mask there so
        # the weights can only ever re-rank, never remove, a candidate.
        empty = playable.sum(-1) <= 0
        if bool(empty.any()):
            base = card_mask_seq[..., :hand_size].reshape(L * B, hand_size).float()
            playable = torch.where(empty.unsqueeze(-1), base, playable)
    none = playable.sum(-1) <= 0
    # Uniform over the affordable slots; the all-zero rows get a valid dummy
    # distribution so multinomial cannot raise, then are overwritten below.
    probs = torch.where(none.unsqueeze(-1), torch.ones_like(playable), playable)
    idx = torch.multinomial(probs, 1).squeeze(-1)
    idx = torch.where(none, torch.full_like(idx, hand_size), idx)
    return idx.view(L, B)
