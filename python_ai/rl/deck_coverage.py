"""A floor under P(play card | card in hand), per deck card.

WHAT THIS EXISTS TO CATCH, AND WHY NOTHING ELSE DID
---------------------------------------------------
Across the 2026-08-28 phase-1 run the agent used five of its eight cards for
30,000 consecutive episodes. Every instrument in the loop read healthy:

  * `EntropyController` held `target_card = 0.35` almost exactly (0.339 ->
    0.350) and never approached its 0.5 ceiling. It measures entropy over the
    four hand SLOTS plus no-op, per decision -- which is blind to the marginal
    distribution over the DECK. A policy that no-ops ~81% of the time and
    spreads the rest over five cards sits at that target forever.
  * `Entropy/Placement_ByCard/*` only logs cards that were PLAYED, so a card at
    zero usage vanishes from the readout instead of alarming.
  * The advisor coverage term (`advisors/advisor_target.py`) does act on
    unplayed cards, but only on the three that have hand-written rules, and its
    KL to the advisor surface still sat at 0.73 after 24,000 episodes.

Measured on the final checkpoint (ep 32,484), P(play card | card in hand):

    Skeletons   0.3429      Hog Rider   0.0704      The Log    0.0091
    Ice Spirit  0.3247      Musketeer   0.0662      Cannon     0.0053
    Ice Golem   0.1868                              Fireball   0.0011

Five live cards at 0.066-0.343, three dead ones at 0.0011-0.0091, and a factor
of ~7 of empty space between the two groups.

A HINGE, NOT A TARGET -- and that distinction is the whole design
-----------------------------------------------------------------
`DECK_COVERAGE_FLOOR` sits inside that gap, so the term is EXACTLY ZERO for
every healthy card and only ever acts on a frozen one. It must not be an
entropy-style target: pushing card use toward uniform would be a strictly worse
policy (a 2.6 Hog deck genuinely should cycle Skeletons far more often than it
Fireballs), and it would fight the policy gradient everywhere instead of only
where a head has died.

It is also NOT part of the PPO objective. Like the advisor coverage term it
never touches `new_logprobs`, so the importance ratio is untouched and the
update stays a valid PPO step -- see `rl/ppo.py`'s comment at the coverage call
site for why that separation is load-bearing.

THE DENOMINATOR IS "IN HAND", NOT "AFFORDABLE"
-----------------------------------------------
`card_logits` arrive already `masked_fill(-inf)` on unaffordable arms, so an
unaffordable card contributes p = 0 to its own mean and drags it down. That is
deliberate and matches the table above, which was measured the same way -- but
it has two consequences worth knowing before reading the metric:

  * the term is slightly stronger for EXPENSIVE cards, which are unaffordable
    more often. Fireball is the most-affected and also the deadest, so the bias
    points the right way here, but it is a bias and not a neutral choice.
  * `Deck/MinCardProb` is therefore NOT the probe's `take-up`, which divides by
    affordability instead. The two answer different questions and will not
    agree; compare a run against itself, never across the two instruments.

The GRADIENT is unaffected by the dilution: a -inf arm has zero gradient, so the
push lands only on rows where the card could actually have been played.

WHAT IT CANNOT DO
-----------------
A floor keeps a card's logit off zero; it cannot make the card good. If the
placement head for that card is poor the card stays -EV and the floor merely
pays a small, bounded price to keep exploring it. The companion fix is on the
state-distribution side (defensive scenario injection in phase 1), which is
what makes the exploration worth anything.
"""
import os

import torch
import torch.nn.functional as F

#: Minimum P(play card | card in hand) below which a card is being pushed up.
#: Read off the measured separation above, not tuned on an outcome -- the same
#: discipline CLAUDE.md records for the advisor target's temperature, where a
#: value picked on the outcome put the target at 94% of maximum entropy while
#: looking busy. 0.02 is 3.3x below the weakest live card and 2.2x above the
#: strongest dead one.
DECK_COVERAGE_FLOOR = float(os.environ.get("CLASH_DECK_COVERAGE_FLOOR", 0.02))

#: Weight on the penalty. The term's magnitude is bounded by the floor itself
#: (a fully dead card contributes at most `floor / n_cards` before weighting),
#: so this is scaled to be comparable to the entropy bonus rather than to the
#: actor loss: with 4 cards and one dead, the raw term is at most 0.005.
DECK_COVERAGE_COEF = float(os.environ.get("CLASH_DECK_COVERAGE_COEF", 2.0))


def enabled():
    return DECK_COVERAGE_COEF > 0.0


def deck_coverage_penalty(card_logits, hand_ids, decision, floor=None):
    """mean over deck cards of `relu(floor - P(play card | card in hand))`.

    card_logits: (N, hand_size + 1) -- the final column is the no-op arm.
    hand_ids:    (N, hand_size) long, -1 for an empty slot.
    decision:    (N,) float; rows at 0 are padding and contribute nothing.

    Returns (penalty, min_p, n_cards). `penalty` carries gradient; `min_p` and
    `n_cards` are plain Python numbers for logging.

    The no-op arm is excluded from the coverage set deliberately: it is not a
    deck card, and flooring it would be a standing pressure to play rather than
    wait -- the opposite of what this run needs, where waiting more is already
    better under the current objective.
    """
    floor = DECK_COVERAGE_FLOOR if floor is None else floor
    hand_size = hand_ids.shape[1]

    # Softmax over ALL arms including no-op -- these are the real action
    # probabilities -- then drop the no-op column from the coverage set.
    probs = torch.softmax(card_logits, dim=-1)[:, :hand_size]

    valid = (hand_ids >= 0) & (decision.reshape(-1, 1) > 0)
    if not bool(valid.any()):
        # A minibatch of pure padding. Return a real zero that still carries a
        # grad_fn, so a caller adding this to its loss cannot get a NaN from a
        # degenerate batch -- the failure `test_rl_ppo_degenerate_batches.py`
        # exists to prevent.
        return probs.sum() * 0.0, 0.0, 0

    ids = hand_ids[valid]
    p = probs[valid]

    uniq, inv = torch.unique(ids, return_inverse=True)
    zeros = torch.zeros(uniq.numel(), dtype=p.dtype, device=p.device)
    sums = zeros.index_add(0, inv, p)
    counts = zeros.index_add(0, inv, torch.ones_like(p))
    p_card = sums / counts

    shortfall = F.relu(floor - p_card)
    # .detach() before the scalar read: min_p is a logging value, and pulling
    # it off the graph as a bare float otherwise warns and keeps the subgraph
    # alive for no reason.
    return shortfall.mean(), float(p_card.min().detach()), int(uniq.numel())
