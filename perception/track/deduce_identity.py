"""Recover card identity by deduction instead of recognition.

The icon templates read our own hand correctly only 33.8% of the time (README
stage 2, 328 in-match plays over 8 recordings) and over-predict Giant at 35%
against a 12.5% prior. That is far too noisy to label a demonstration dataset.

But identity does not have to be *recognised*. Three facts make it deducible:

  * the deck is known and fixed -- 8 specific cards, no levels, no surprises
  * the cycle is strict FIFO: playing a card sends it to the back of an
    8-slot rotation and draws the front into the vacated hand slot
    (see track/cycle.py, which reproduces the engine exactly, 0 desyncs)
  * each play's COST is observable to ~0.99 confidence from the elixir ledger,
    which is a *hard* constraint on which card it could have been

So the unknown is one permutation: which of the 8 cards started in each of the
4 hand slots and the 4 queue positions. 8! = 40,320 candidates, which is
nothing. Replay the observed (slot, cost) sequence against every candidate and
keep the ones that never contradict it.

WHAT THIS CANNOT DO, stated up front: cost alone cannot separate cards that
cost the same. In DEFAULT_DECK the classes are {Archers, Minions, Cannon} at 3,
{Valkyrie, Fireball, Musketeer, Mini P.E.K.K.A} at 4, and {Giant} at 5, so pure
cost logic can at best narrow to 3!*4!*1! = 144 permutations -- one cost
pattern, every internal relabelling of it. Breaking that last tie needs a
second evidence channel (`observe` below takes optional per-play identity
likelihoods, e.g. the 33.8% templates, which compound because the cycle ties
every sighting of the same physical card together).
"""
from __future__ import annotations

from collections import deque
from itertools import permutations

# HAND_SIZE/QUEUE_SIZE were redefined here until 2026-08-24, three lines below
# a docstring insisting costs come "from the engine registry, never a second
# hardcoded copy". Same rule, same file, opposite practice.
from track.cycle import HAND_SIZE, QUEUE_SIZE, advance


class IdentityContradiction(RuntimeError):
    """No permutation survives -- an observation must be wrong."""


def _costs_from_bindings(deck):
    """Costs from the engine registry, never a second hardcoded copy."""
    import clash_royale_env  # noqa: PLC0415 -- optional, see deduce()'s `costs`
    return {c: float(clash_royale_env.get_card_info(c)["cost"]) for c in deck}


def _replay(order, plays, costs):
    """Walk one candidate permutation through the observed plays.

    `order` is (hand0..3, q0..3). Returns the identity played at each step, or
    None as soon as an observed cost contradicts the candidate.
    """
    hand = list(order[:HAND_SIZE])
    queue = deque(order[HAND_SIZE:], maxlen=QUEUE_SIZE)
    played = []
    for slot, cost in plays:
        if not (0 <= slot < HAND_SIZE):
            return None
        card = hand[slot]
        if cost is not None and costs[card] != cost:
            return None
        played.append(card)
        # `slot` is passed explicitly: a permutation may repeat an id, and
        # hand.index() would then resolve to the wrong slot.
        advance(hand, queue, card, index=slot)
    return played


def deduce(deck, plays, costs=None, hand_priors=None):
    """Identify every played card from (slot, cost) observations alone.

    deck   : the 8 card ids, order irrelevant
    plays  : [(slot_index, cost_or_None), ...] in chronological order.
             slot_index comes from readers.hand.infer_play_from_hand_change,
             which is a set-difference on consecutive hand reads and so is
             reliable even when the icon IDENTITY is not; cost comes from the
             elixir ledger. A None cost contributes no constraint.
    costs  : {card_id: cost}; derived from the bindings when omitted.
    hand_priors : optional [{card_id: log-likelihood}, ...] parallel to
             `plays`, for a soft second channel such as the icon templates.
             Candidates are ranked by the total, but never eliminated by it --
             a 33.8%-accurate reader must not be allowed to contradict the
             elixir ledger.

    Returns a dict with:
      n_candidates  surviving permutations
      identities    per play: sorted list of possible card ids (len 1 == solved)
      resolved      fraction of plays with exactly one possibility
      best          the highest-scoring candidate's identity sequence
    """
    deck = list(deck)
    if len(deck) != HAND_SIZE + QUEUE_SIZE:
        raise ValueError(f"deck must have {HAND_SIZE + QUEUE_SIZE} cards, got {len(deck)}")
    costs = costs or _costs_from_bindings(deck)

    survivors = []
    for order in permutations(deck):
        seq = _replay(order, plays, costs)
        if seq is not None:
            survivors.append(seq)
    if not survivors:
        raise IdentityContradiction(
            f"no permutation of the deck reproduces {len(plays)} observed "
            "(slot, cost) pairs -- a slot read or an elixir cost is wrong")

    identities = [sorted({s[i] for s in survivors}) for i in range(len(plays))]
    resolved = (sum(1 for p in identities if len(p) == 1) / len(identities)
                if identities else 0.0)

    best = survivors[0]
    if hand_priors:
        def score(seq):
            return sum(hand_priors[i].get(c, 0.0) for i, c in enumerate(seq))
        best = max(survivors, key=score)

    return {"n_candidates": len(survivors), "identities": identities,
            "resolved": resolved, "best": best}
