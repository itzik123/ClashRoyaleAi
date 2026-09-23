"""The identity solver must be sound (the truth always survives) before it is
useful: a confidently wrong label poisons a demonstration dataset silently.
"""
from collections import deque

import pytest

from perception.track.deduce_identity import (
    HAND_SIZE, QUEUE_SIZE, IdentityContradiction, deduce)

DECK = [10, 1, 41, 25, 7, 2, 6, 5]
COST = {10: 4, 1: 3, 41: 3, 25: 3, 7: 4, 2: 5, 6: 4, 5: 4}


def _simulate(order, slots):
    hand = list(order[:HAND_SIZE])
    queue = deque(order[HAND_SIZE:], maxlen=QUEUE_SIZE)
    plays, truth = [], []
    for s in slots:
        card = hand[s]
        plays.append((s, COST[card]))
        truth.append(card)
        hand[s] = queue.popleft()
        queue.append(card)
    return plays, truth


def test_truth_always_survives():
    """Soundness: the real card is never eliminated from its play's candidates."""
    order = [10, 1, 41, 25, 7, 2, 6, 5]
    slots = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2]
    plays, truth = _simulate(order, slots)
    out = deduce(DECK, plays, costs=COST)
    for card, possible in zip(truth, out["identities"]):
        assert card in possible


def test_unique_cost_resolves_exactly():
    """Giant is the only 5-cost card, so every Giant play is pinned outright."""
    order = [2, 1, 41, 25, 7, 10, 6, 5]
    plays, truth = _simulate(order, [0, 1, 2, 3, 0, 1, 2, 3])
    out = deduce(DECK, plays, costs=COST)
    for card, possible in zip(truth, out["identities"]):
        if COST[card] == 5:
            assert possible == [2]


def test_cannot_beat_the_cost_class_floor():
    """The solver's real limit: cost cannot separate equal-cost cards. 3! * 4! *
    1! = 144 relabellings are indistinguishable from (slot, cost) alone,
    however many plays are observed.
    """
    order = [10, 1, 41, 25, 7, 2, 6, 5]
    plays, _ = _simulate(order, [i % HAND_SIZE for i in range(60)])
    out = deduce(DECK, plays, costs=COST)
    assert out["n_candidates"] == 144


def test_contradiction_is_raised_not_guessed():
    """A cost no permutation can produce must fail loudly."""
    plays = [(0, 3.0), (0, 3.0), (0, 3.0), (0, 3.0), (0, 3.0), (0, 3.0)]
    with pytest.raises(IdentityContradiction):
        deduce(DECK, plays, costs=COST)


def test_missing_cost_is_allowed_and_constrains_nothing():
    order = [10, 1, 41, 25, 7, 2, 6, 5]
    plays, truth = _simulate(order, [0, 1, 2])
    relaxed = [(s, None) for s, _ in plays]
    out = deduce(DECK, relaxed, costs=COST)
    assert out["n_candidates"] > deduce(DECK, plays, costs=COST)["n_candidates"]
    for card, possible in zip(truth, out["identities"]):
        assert card in possible


def test_rejects_a_deck_that_is_not_eight_cards():
    with pytest.raises(ValueError):
        deduce(DECK[:6], [(0, 3.0)], costs=COST)
