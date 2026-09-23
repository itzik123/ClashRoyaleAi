"""Tests for live/elixir_ledger.py.

Synthetic traces, since each test pins one observed failure mode and a
recording mixes them. No engine import: costs are injected, so this runs
without the .pyd.
"""
from __future__ import annotations

import pytest

from live.elixir_ledger import DEFAULT_COSTS, ElixirLedger, decompositions


def feed(values, costs=DEFAULT_COSTS, dt=0.1):
    """Readings `dt` apart, explicitly: the ledger models regeneration between
    samples, and a wall-clock default would make every test depend on loop
    speed.
    """
    ledger = ElixirLedger(costs=costs)
    for i, v in enumerate(values):
        ledger.update(v, now=i * dt)
    return ledger


def test_a_clean_single_placement_is_counted():
    # Regen up to 7, spend 4, hold. Repeats pad the median's 3-sample window.
    ledger = feed([5, 5, 6, 6, 7, 7, 7, 3, 3, 3])
    assert ledger.cards == 1
    assert ledger.spent == pytest.approx(4.0)


def test_regen_is_not_counted_as_spend():
    ledger = feed([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    assert ledger.cards == 0
    assert ledger.spent == 0.0
    assert ledger.gained > 0


def test_an_isolated_zero_is_despiked_not_read_as_a_10_elixir_spend():
    """The observed reader failure: a lone 0 inside a stable run (CRBAB's
    `_calculate_elixir` latching onto a low-variance patch at the ROI's left
    edge). Untreated it looks like a huge spend followed by an impossible
    refill.
    """
    ledger = feed([7, 7, 7, 7, 0, 7, 7, 7])
    assert ledger.glitches >= 1
    assert ledger.cards == 0
    assert ledger.spent == 0.0


def test_two_cards_in_one_sample_are_decomposed():
    """A drop of 7 is 3+4, not a glitch: two plays can land inside one sample.
    """
    ledger = feed([10, 10, 10, 3, 3, 3])
    assert ledger.cards == 2
    assert ledger.spent == pytest.approx(7.0)


def test_a_drop_below_the_cheapest_card_is_not_a_placement():
    """1-2 elixir steps are integer quantisation noise; counting them would invent
    placements in a cumulative total. Not `unexplained` either: that is the
    normal case between two samples, and `unexplained` is reserved for a
    play-sized drop matching no legal combination.
    """
    ledger = feed([8, 8, 8, 7, 7, 7])
    assert ledger.cards == 0
    assert ledger.spent == 0.0
    assert ledger.unexplained == 0


def test_residual_is_positive_when_regen_overflows_the_cap():
    """Time sitting at the cap credits nothing, so the residual must not go
    negative, which would mean spend was invented.
    """
    ledger = feed([10, 10, 10, 10, 10, 10, 6, 6, 6])
    assert ledger.residual >= 0


def test_unreadable_frames_are_skipped_not_treated_as_zero():
    """None means the bar could not be read; treating it as 0 would fabricate a
    10-elixir spend.
    """
    ledger = feed([6, 6, 6, None, None, 6, 6, 6])
    assert ledger.cards == 0
    assert ledger.spent == 0.0


def test_decompositions_cover_one_to_three_cards():
    table = decompositions((3.0, 4.0, 5.0), max_cards=3)
    assert 3.0 in table and 5.0 in table          # one card
    assert 7.0 in table and 10.0 in table         # two
    assert 15.0 in table                          # three
    assert 2.0 not in table                       # below the cheapest
    assert 16.0 not in table                      # above three of the dearest


def test_costs_are_injected_so_a_different_deck_works():
    """Costs come from the caller, not a copy of the deck baked in here."""
    ledger = feed([9, 9, 9, 7, 7, 7], costs=(2.0, 6.0))
    assert ledger.cards == 1
    assert ledger.spent == pytest.approx(2.0)


def test_a_ledger_with_no_input_is_inert():
    ledger = ElixirLedger()
    assert ledger.spent == 0.0
    assert ledger.cards == 0
    assert ledger.residual == 0.0


# --- rate independence ---

def _trace(rate_hz, plays=((6.0, 4.0), (20.0, 3.0), (40.0, 5.0)),
           duration=60.0, regen_per_s=1.0 / 2.8, cap=10.0, start=5.0):
    """An integer-quantised elixir bar sampled at `rate_hz`, with known plays.
    Quantisation is the point: each end of a measured drop carries up to
    +/-0.5.
    """
    fine, t, elixir = 0.01, 0.0, start
    pending = list(plays)
    curve = []
    while t < duration:
        elixir = min(cap, elixir + regen_per_s * fine)
        while pending and t >= pending[0][0]:
            elixir = max(0.0, elixir - pending.pop(0)[1])
        curve.append((t, elixir))
        t += fine
    step = max(1, int(round((1.0 / rate_hz) / fine)))
    return [(t, float(int(round(e)))) for t, e in curve[::step]]


def _run(rate_hz):
    ledger = ElixirLedger()
    for t, v in _trace(rate_hz):
        ledger.update(v, now=t)
    return ledger


def test_spend_does_not_depend_on_the_sample_rate():
    """The ledger is fed at ~1.5 Hz live. `drop = last - value` would assume no
    regeneration between samples: 0.036 elixir at 10 fps, 0.24 at 1.5 Hz.
    """
    true_spent = 12.0
    for rate in (10.0, 5.0, 2.0, 1.5):
        ledger = _run(rate)
        assert ledger.cards == 3, f"{rate} Hz found {ledger.cards} cards"
        assert ledger.spent == pytest.approx(true_spent, abs=1.5), (
            f"{rate} Hz recovered {ledger.spent} of {true_spent}")


def test_regeneration_is_credited_even_while_a_card_is_played():
    """`gained` is modelled inflow, not observed rises, so a sample containing a
    placement still accrues its regen.
    """
    ledger = _run(2.0)
    assert ledger.gained > 15.0
    # gained == spent + (final - initial), within quantisation.
    assert ledger.residual == pytest.approx(0.0, abs=2.0)


def test_time_at_the_cap_is_not_credited_as_regeneration():
    """Elixir cannot accumulate past 10; crediting inflow there would make the
    residual demand a placement that never happened.
    """
    ledger = ElixirLedger()
    for i in range(20):
        ledger.update(10.0, now=i * 1.0)
    assert ledger.gained == pytest.approx(0.0, abs=1e-6)
    assert ledger.cards == 0


def test_the_nearest_decomposition_wins_not_the_cheapest():
    """With an integer bar most drops are ambiguous between adjacent costs; the
    smallest qualifying total would bias every one downward.
    """
    ledger = ElixirLedger(tolerance=1.0)
    for i, v in enumerate([9, 9, 9, 5, 5, 5]):
        ledger.update(v, now=i * 0.1)
    # A drop of ~4 must read as 4, not as the cheaper 3 that also fits.
    assert ledger.plays == [(4.0,)]


def test_a_double_elixir_multiplier_changes_the_expected_regen():
    """Regen doubles after the 2x mark; modelling it at 1x shrinks every measured
    drop.
    """
    single = ElixirLedger()
    double = ElixirLedger()
    for i in range(6):
        single.update(5.0, now=i * 1.0, multiplier=1.0)
        double.update(5.0, now=i * 1.0, multiplier=2.0)
    assert double.gained == pytest.approx(2 * single.gained, rel=1e-6)


# --- ground truth from our own plays ---

def test_a_confirmed_play_uses_the_cost_we_know_not_a_guess():
    """The bar cannot separate 3 from 4; knowing what we issued turns that into
    arithmetic.
    """
    ledger = ElixirLedger()
    ledger.record_play(3.0, now=0.0)
    for i, v in enumerate([9, 9, 9, 6, 6, 6]):
        ledger.update(v, now=i * 0.1)
    assert ledger.plays == [(3.0,)]
    assert ledger.spent == pytest.approx(3.0)


def test_an_issued_play_is_not_spend_until_the_bar_confirms_it():
    """A tap can be rejected (the elixir may be gone by the time it lands), so
    counting it at issue would invent spend.
    """
    ledger = ElixirLedger()
    ledger.record_play(4.0, now=0.0)
    assert ledger.spent == 0.0
    assert ledger.cards == 0
    assert ledger.unconfirmed_cost == pytest.approx(4.0)


def test_an_unconfirmed_play_is_written_off_not_carried_forever():
    """Otherwise one rejected tap would debit the agent's elixir for the rest of
    the match.
    """
    ledger = ElixirLedger()
    ledger.record_play(4.0, now=0.0)
    for i, v in enumerate([7] * 6):
        ledger.update(v, now=i * 2.0)          # well past the window
    assert ledger.rejected == 1
    assert ledger.unconfirmed_cost == 0.0
    assert ledger.spent == 0.0


def test_unconfirmed_cost_is_what_stops_the_double_spend():
    """The burst: the agent commits, the bar has not moved, and it commits again
    against the same stale elixir.
    """
    ledger = ElixirLedger()
    ledger.record_play(5.0, now=0.0)
    ledger.record_play(4.0, now=0.5)
    assert ledger.unconfirmed_cost == pytest.approx(9.0)


def test_two_issued_plays_confirmed_by_one_drop():
    """Both taps can land inside a single sample at this frame rate."""
    ledger = ElixirLedger()
    ledger.record_play(3.0, now=0.0)
    ledger.record_play(4.0, now=0.0)
    for i, v in enumerate([10, 10, 10, 3, 3, 3]):
        ledger.update(v, now=i * 0.1)
    assert ledger.cards == 2
    assert ledger.spent == pytest.approx(7.0)
    assert ledger.unconfirmed_cost == 0.0


def test_inference_still_works_with_nothing_pending():
    """The observer case (a recording of someone else playing) has no issue
    stream; BC extraction runs on it.
    """
    ledger = ElixirLedger()
    for i, v in enumerate([9, 9, 9, 5, 5, 5]):
        ledger.update(v, now=i * 0.1)
    assert ledger.cards == 1
    assert ledger.spent == pytest.approx(4.0)


def test_a_drop_matching_no_pending_play_falls_back_to_inference():
    """Something spent elixir we did not issue. Dropping it silently would leave
    the residual demanding a placement nobody can find.
    """
    ledger = ElixirLedger()
    ledger.record_play(3.0, now=0.0)
    for i, v in enumerate([10, 10, 10, 5, 5, 5]):   # a drop of 5, not 3
        ledger.update(v, now=i * 0.1)
    assert ledger.cards == 1
    assert ledger.spent == pytest.approx(5.0)


def test_record_play_defaults_to_the_ledgers_own_clock():
    """Time bases must not mix: readings carry frame capture time (near zero),
    `time.monotonic()` is in the hundreds of thousands. Mixed, nothing ever
    expires and the optimistic debit reads zero elixir for the rest of the
    match.
    """
    ledger = ElixirLedger()
    for i, v in enumerate([7, 7, 7]):
        ledger.update(v, now=i * 0.5)          # frame clock, near zero
    ledger.record_play(4.0)                     # no timestamp
    assert ledger.unconfirmed_cost == pytest.approx(4.0)
    # Advance the frame clock past the window; it must expire.
    for i, v in enumerate([7, 7, 7, 7]):
        ledger.update(v, now=10.0 + i * 2.0)
    assert ledger.rejected == 1
    assert ledger.unconfirmed_cost == 0.0
