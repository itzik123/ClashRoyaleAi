"""Recover card identity by deduction instead of recognition.

The icon templates read our own hand too unreliably to label a demonstration
dataset. But identity is deducible:

  * the deck is known and fixed: 8 cards;
  * the cycle is a strict FIFO (track/cycle.py): a played card goes to the
    back and the front fills the vacated slot;
  * each play's cost is observable to ~0.99 confidence from the elixir ledger,
    a hard constraint on which card it was.

So the unknown is one permutation of the 8 cards over 4 hand slots and 4 queue
positions: 8! = 40,320 candidates. Replay the observed (slot, cost) sequence
against each and keep those that never contradict it.

Cost cannot separate cards of equal cost. In the Giant deck ({Archers, Minions,
Cannon} at 3, {Valkyrie, Fireball, Musketeer, Mini P.E.K.K.A} at 4, {Giant} at
5) cost logic narrows at best to 3!*4!*1! = 144 permutations. Breaking that tie
needs a second channel: `deduce` takes optional per-play identity likelihoods,
which compound because the cycle ties every sighting of one physical card
together.
"""
from __future__ import annotations

from collections import deque
from itertools import permutations

from track.cycle import HAND_SIZE, QUEUE_SIZE, advance


class IdentityContradiction(RuntimeError):
    """No permutation survives -- an observation must be wrong."""


def _costs_from_bindings(deck):
    """Costs from the engine registry."""
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
        # hand.index() would find the wrong slot.
        advance(hand, queue, card, index=slot)
    return played


def deduce(deck, plays, costs=None, hand_priors=None):
    """Identify every played card from (slot, cost) observations alone.

    deck   : the 8 card ids, order irrelevant
    plays  : [(slot_index, cost_or_None), ...] in chronological order.
             slot_index comes from readers.hand.infer_play_from_hand_change, a
             set-difference on consecutive hand reads, reliable even when icon
             identity is not; cost comes from the elixir ledger. A None cost
             adds no constraint.
    costs  : {card_id: cost}; derived from the bindings when omitted.
    hand_priors : optional [{card_id: log-likelihood}, ...] parallel to
    `plays`,
             a soft second channel such as the icon templates. Ranks candidates,
             never eliminates them: a noisy reader must not contradict the ledger.

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
