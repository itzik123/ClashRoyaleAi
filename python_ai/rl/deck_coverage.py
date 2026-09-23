"""A floor under P(play card | card in hand), per deck card.

The penalty is relu(log(floor / p)): zero for every healthy card, acting only
on a frozen one. Log space because a hinge on p itself carries the softmax
factor p(1-p), which vanishes exactly as a card dies. A floor, not a target:
pushing usage toward uniform would be a worse policy. Like the placement
coverage term it never enters the PPO ratio.

The denominator is "in hand", not "affordable": unaffordable rows count as p =
0, which weights expensive cards a little more, and makes `Deck/MinCardProb`
differ from a probe's take-up rate. The gradient only lands where the card was
playable.

Off by default: measured harmful. A floor keeps a card's logit off zero but
cannot make its placement good, and forcing poorly placed cards into play cost
more win rate than they were worth, gated or not. Fix placement first; see
docs/DECISIONS.md.
"""
import math
import os

import torch
import torch.nn.functional as F

#: Minimum P(play card | card in hand) before a card is pushed up; sits between
#: the live cards (>= 0.066) and the dead ones (<= 0.009).
DECK_COVERAGE_FLOOR = float(os.environ.get("CLASH_DECK_COVERAGE_FLOOR", 0.02))

#: Guards log(p) for a card unaffordable across a whole minibatch.
_EPS = 1e-9

#: Weight on the penalty. 0.0: the term is off (see the module docstring).
DECK_COVERAGE_COEF = float(os.environ.get("CLASH_DECK_COVERAGE_COEF", 0.0))


def enabled():
    return DECK_COVERAGE_COEF > 0.0


def deck_coverage_penalty(card_logits, hand_ids, decision, floor=None,
                          threat=None):
    """Mean over deck cards of `relu(log(floor / P(play card | card in hand)))`.

    card_logits: (N, hand_size + 1); the last column is the no-op arm.
    hand_ids:    (N, hand_size) long, -1 for an empty slot.
    decision:    (N,) float; rows at 0 contribute nothing.
    threat:      (N,) float or None. Rows at 0 (quiet boards) contribute
    nothing; None counts every row.

    Returns (penalty, min_p, n_cards); only `penalty` carries gradient. The
    no-op arm is excluded: flooring it would be standing pressure to play
    rather than wait.
    """
    floor = DECK_COVERAGE_FLOOR if floor is None else floor
    hand_size = hand_ids.shape[1]

    # Softmax over all arms, including no-op, then drop the no-op column.
    probs = torch.softmax(card_logits, dim=-1)[:, :hand_size]

    valid = (hand_ids >= 0) & (decision.reshape(-1, 1) > 0)
    if threat is not None:
        # Threat gate: only push on boards that carry a threat.
        valid = valid & (threat.reshape(-1, 1) > 0)
    if not bool(valid.any()):
        # Pure padding: a zero that still has a grad_fn, so a degenerate batch
        # cannot produce NaN.
        return probs.sum() * 0.0, 0.0, 0

    ids = hand_ids[valid]
    p = probs[valid]

    uniq, inv = torch.unique(ids, return_inverse=True)
    zeros = torch.zeros(uniq.numel(), dtype=p.dtype, device=p.device)
    sums = zeros.index_add(0, inv, p)
    counts = zeros.index_add(0, inv, torch.ones_like(p))
    p_card = sums / counts

    # Log space; see the module docstring. clamp_min guards a card that was
    # never affordable in the minibatch.
    shortfall = F.relu(math.log(floor) - torch.log(p_card.clamp_min(_EPS)))
    return shortfall.mean(), float(p_card.min().detach()), int(uniq.numel())
