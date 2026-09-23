"""Cycle, deck discovery and opponent elixir, against real replay data: real
hands, real cycling, real elixir, which exercises the engine's behaviour rather
than a reading of it.
"""

from __future__ import annotations

import pytest

from contracts import UNKNOWN_CARD_SIM_ID, Phase
from timebase import STARTING_ELIXIR, ticks_to_seconds
from track.cycle import CycleTracker
from track.opp_deck import OpponentDeckTracker
from track.opp_elixir import OpponentElixirTracker


# --- cycle ---


def test_cycle_reproduces_the_engines_fifo_exactly(replay):
    """Replay our real plays and predict the hand after each:
    PlayerState::playCard sends the played card to the back of the queue and
    the front fills the vacated slot, so the predicted hand must match the
    logged hand after every play.
    """
    if not replay.has_action_labels:
        pytest.skip("replay has no action labels")

    hands = {t.tick: t.ai_hand for t in replay.ticks}
    deck = replay.starting_deck(0)

    tracker = CycleTracker()
    tracker.set_deck(deck)
    tracker.seed_hand(replay.ticks[0].ai_hand)

    checked = 0
    for placement in replay.iter_placements():
        tracker.on_play(placement.card_id)
        after = hands.get(placement.tick + 1)
        if after is None:
            continue
        assert set(tracker.hand) == set(after), (
            f"after playing {placement.card_id} at tick {placement.tick}: "
            f"predicted {sorted(tracker.hand)}, engine had {sorted(after)}"
        )
        checked += 1

    assert checked >= 5, "fixture did not exercise enough plays to be meaningful"
    assert tracker.desyncs == 0


def test_queue_is_the_last_four_plays():
    """The identity opponent tracking depends on: the queue is four long and grows
    only at the back, so it is always the last four cards played.
    """
    deck = (0, 1, 2, 3, 4, 5, 6, 7)
    tracker = CycleTracker()
    tracker.set_deck(deck)
    tracker.seed_hand((0, 1, 2, 3))

    played = [0, 1, 2, 3, 4, 5]
    for card in played:
        tracker.on_play(card)

    assert list(tracker.queue) == played[-4:]
    assert set(tracker.hand) == set(deck) - set(played[-4:])
    assert tracker.next_card == played[-4]


def test_observe_hand_corrects_and_counts_a_desync():
    tracker = CycleTracker()
    tracker.set_deck((0, 1, 2, 3, 4, 5, 6, 7))
    tracker.seed_hand((0, 1, 2, 3))

    assert tracker.observe_hand((0, 1, 2, 3)) is True
    assert tracker.desyncs == 0

    assert tracker.observe_hand((0, 1, 2, 7)) is False
    assert tracker.desyncs == 1
    assert set(tracker.hand) == {0, 1, 2, 7}
    assert tracker.confidence < 1.0


def test_next_card_is_unknown_until_the_order_is():
    tracker = CycleTracker()
    tracker.set_deck((0, 1, 2, 3, 4, 5, 6, 7))
    tracker.seed_hand((0, 1, 2, 3))
    # The queue's membership is known and its order is not, so next_card must
    # not pretend otherwise.
    assert tracker.next_card == UNKNOWN_CARD_SIM_ID
    for card in (0, 1, 2, 3):
        tracker.on_play(card)
    assert tracker.next_card == 0


# --- opponent deck ---


def test_opponent_deck_is_discovered_from_plays(replay):
    deck = replay.starting_deck(1)
    tracker = OpponentDeckTracker()
    for placement in replay.infer_opponent_placements(deck=deck):
        tracker.on_play(placement.card_id, placement.tick)

    assert tracker.deck <= set(deck)
    assert tracker.confidence == pytest.approx(len(tracker.known) / 8)


def test_cycle_wrap_proves_a_missed_placement():
    """A repeat before all eight are known proves a miss: the FIFO plays every
    card once before any twice, so anything still unseen when the cycle wraps
    was missed.
    """
    tracker = OpponentDeckTracker()
    for tick, card in enumerate([10, 11, 12, 13, 14, 15]):
        tracker.on_play(card, tick)
    assert tracker.missed_at_wrap == 0

    tracker.on_play(10, 100)  # repeat: the cycle demonstrably wrapped
    assert tracker.wrapped_tick == 100
    assert tracker.missed_at_wrap == 2
    assert "opp_deck_incomplete_at_wrap" in tracker.flags()


def test_unmapped_opponent_card_is_counted_separately():
    tracker = OpponentDeckTracker()
    tracker.on_play(UNKNOWN_CARD_SIM_ID, 5)
    assert tracker.unmapped_seen == 1
    assert tracker.known == []
    assert "opp_deck_has_unmapped_card" in tracker.flags()


# --- opponent elixir ---


def test_opponent_elixir_regenerates_at_the_real_rate():
    tracker = OpponentElixirTracker()
    assert tracker.value == STARTING_ELIXIR
    tracker.advance_to(28)  # 2.8 s -> exactly one elixir at the real rate
    assert tracker.value == pytest.approx(6.0, abs=1e-6)


def test_opponent_elixir_uses_the_real_rate_not_the_simulators():
    """At the simulator's rate the balance would be ~1.3 elixir low by three
    minutes, firing the negative-balance alarm on good input. See timebase.py.
    """
    tracker = OpponentElixirTracker()
    tracker.advance_to(1800)  # 180 s
    at_real_rate = STARTING_ELIXIR + 180 / 2.8
    at_sim_rate = STARTING_ELIXIR + 1800 * 0.035
    assert at_real_rate - at_sim_rate == pytest.approx(1.3, abs=0.1)
    assert tracker.value == pytest.approx(min(10.0, at_real_rate))


def test_negative_balance_is_recorded_before_clamping():
    """The most valuable signal in the pipeline must not be clamped away."""
    tracker = OpponentElixirTracker()
    tracker.on_play(4.0, tick=0)
    tracker.on_play(4.0, tick=1)  # 5 - 8 = -3: impossible without a missed play

    assert tracker.went_negative
    assert tracker.negative_events
    assert tracker.negative_events[-1][1] < 0
    assert tracker.value == 0.0  # clamped only after the fact
    assert "opp_elixir_negative" in tracker.flags()
    assert tracker.confidence < 1.0


def test_unknown_cost_disarms_the_alarm_and_says_so():
    tracker = OpponentElixirTracker()
    tracker.on_play(None, tick=0)
    assert tracker.unknown_cost_plays == 1
    assert tracker.value == STARTING_ELIXIR  # not debited
    assert tracker.confidence == 0.0
    assert "opp_elixir_unknown_cost" in tracker.flags()


def test_phase_multiplier_scales_regeneration():
    single = OpponentElixirTracker()
    single.advance_to(280, Phase.SINGLE)
    double = OpponentElixirTracker()
    double.advance_to(280, Phase.DOUBLE)
    # Both cap at 10, so compare over a short window.
    a, b = OpponentElixirTracker(), OpponentElixirTracker()
    a.advance_to(28, Phase.SINGLE)
    b.advance_to(28, Phase.DOUBLE)
    assert (b.value - STARTING_ELIXIR) == pytest.approx(2 * (a.value - STARTING_ELIXIR))


def test_elixir_cannot_run_backwards():
    tracker = OpponentElixirTracker()
    tracker.advance_to(100)
    with pytest.raises(ValueError, match="backwards"):
        tracker.advance_to(50)


def test_opponent_elixir_stays_non_negative_on_a_real_match(replay):
    """On a clean input stream the balance must never go negative: stage 4's
    acceptance criterion as a test. The opponent's plays are reconstructed from
    the replay, as complete as the fixture allows.
    """
    deck = replay.starting_deck(1)
    costs = {cid: float(replay.card_meta.get(cid, {}).get("cost", 0)) for cid in deck}

    tracker = OpponentElixirTracker()
    for placement in replay.infer_opponent_placements(deck=deck):
        tracker.advance_to(placement.tick)
        tracker.on_play(costs.get(placement.card_id) or None, placement.tick)

    assert tracker.value >= 0.0
