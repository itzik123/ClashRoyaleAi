"""Tests for live/hand_tracker.py. Synthetic cycles, each pinning one FIFO rule.
No engine import: costs are injected.
"""
from __future__ import annotations

import pytest

from contracts import UNKNOWN_CARD_SIM_ID
from live.hand_tracker import CONSENSUS_WINDOW, HandTracker

DECK = (10, 1, 41, 25, 7, 2, 6, 5)
COSTS = {10: 4.0, 1: 3.0, 41: 3.0, 25: 3.0, 7: 4.0, 2: 5.0, 6: 4.0, 5: 4.0}
START = (10, 1, 41, 25)          # Valkyrie, Archers, Minions, Cannon


def seeded(hand=START):
    t = HandTracker(deck=DECK, costs=COSTS)
    for _ in range(CONSENSUS_WINDOW):
        t.update(hand, [])
    assert t.seeded
    return t


def test_unseeded_reports_unknown_rather_than_guessing():
    t = HandTracker(deck=DECK, costs=COSTS)
    t.update(START, [])
    assert not t.seeded
    assert t.as_tuple() == (UNKNOWN_CARD_SIM_ID,) * 4
    assert t.confidence == 0.0


def test_seeds_from_the_consensus_not_a_single_frame():
    """One frame of the detector is noise, so seeding waits for the window to
    fill.
    """
    t = HandTracker(deck=DECK, costs=COSTS)
    for i in range(CONSENSUS_WINDOW - 1):
        t.update(START, [])
        assert not t.seeded, i
    t.update(START, [])
    assert t.seeded and set(t.hand) == set(START)


def test_the_queue_is_the_cards_not_in_hand():
    t = seeded()
    assert set(t.queue) == set(DECK) - set(START)


def test_an_unambiguous_cost_advances_the_fifo():
    """Giant is the only 5-cost card in this deck, so a 5-drop names it."""
    t = seeded((10, 1, 2, 25))
    t.update((10, 1, 2, 25), [(5.0,)])
    assert 2 not in t.hand           # Giant left
    assert 2 in t.queue              # ...to the back of the queue
    assert len(set(t.hand)) == 4
    assert t.plays_applied == 1


def test_the_played_card_keeps_its_slot():
    """The engine refills the vacated slot rather than shifting, preserving slot
    position as the real game shows.
    """
    t = seeded((10, 1, 2, 25))
    incoming = t.queue[0]
    t.update((10, 1, 2, 25), [(5.0,)])
    assert t.hand[2] == incoming     # Giant was slot 2


def test_a_played_card_cannot_return_before_the_cycle_allows():
    """The FIFO invariant the raw detector violates."""
    t = seeded()
    played = []
    for _ in range(3):
        cost = COSTS[t.hand[0]]
        gone = t.hand[0]
        t.update(tuple(t.hand), [(cost,)])
        played.append(gone)
        assert gone not in t.hand, "returned immediately"
    assert len(set(played)) == 3


def test_ambiguity_is_counted_not_hidden():
    """Cost 3 matches Archers, Minions and Cannon; a tracker that silently guessed
    would look identical to one that knew.
    """
    t = seeded()
    t.update(START, [(3.0,)])
    assert t.ambiguous == 1


def test_a_cost_not_in_hand_is_recorded_as_desync_not_dropped():
    """The ledger is surer that something was played than we are about the hand;
    dropping the play would leave the FIFO permanently one behind with no
    signal.
    """
    t = seeded((1, 41, 25, 6))          # costs 3,3,3,4 -- no 5
    t.update((1, 41, 25, 6), [(5.0,)])
    assert t.desyncs == 1
    assert t.plays_applied == 0


def test_a_lagging_consensus_is_not_treated_as_disagreement():
    """After a play the window still describes the previous hand for about half a
    window; checking then would keep overwriting the tracker with a hand the
    game has left.
    """
    t = seeded((10, 1, 2, 25))
    t.update((10, 1, 2, 25), [(5.0,)])
    before = t.desyncs
    for _ in range(CONSENSUS_WINDOW - 2):
        t.update((10, 1, 2, 25), [])     # stale reading, still showing Giant
    assert t.desyncs == before, "stale evidence counted as a desync"


def test_every_tracked_hand_is_four_distinct_in_deck_cards():
    """The tracked hand never violates the FIFO, which the raw detector cannot
    say; it is what makes the tracked hand safe to encode.
    """
    t = seeded()
    for _ in range(12):
        cost = COSTS[t.hand[0]]
        t.update(tuple(t.hand), [(cost,)])
        assert len(set(t.hand)) == 4
        assert all(c in DECK for c in t.hand)
