"""Tests for live/elixir_ledger.py.

Synthetic traces rather than a recording fixture, because each test pins one
failure mode that was actually observed and a recording mixes them all
together. The end-to-end number the module was fitted against (29 cards,
residual +12%, over 935 frames of a real match) lives in the module docstring
and BOT_REQUESTS.md item 8.

No engine import: costs are injected, so this runs without the .pyd.
"""
from __future__ import annotations

import pytest

from live.elixir_ledger import DEFAULT_COSTS, ElixirLedger, decompositions


def feed(values, costs=DEFAULT_COSTS, dt=0.1):
    """Readings `dt` apart. Explicit, because the ledger models regeneration
    between samples and letting it default to the wall clock would make every
    test depend on how fast the loop happened to run."""
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
    """The observed reader failure: a lone 0 inside a stable run.

    CRBAB's `_calculate_elixir` takes the first window whose rolling std falls
    under a threshold, so a low-variance patch at the ROI's left edge reads ~0
    whatever the true level. Untreated this looks like a huge spend followed by
    an impossible instant refill.
    """
    ledger = feed([7, 7, 7, 7, 0, 7, 7, 7])
    assert ledger.glitches >= 1
    assert ledger.cards == 0
    assert ledger.spent == 0.0


def test_two_cards_in_one_sample_are_decomposed():
    """A drop of 7 is 3+4, not a glitch.

    Rejecting anything above the single-card maximum was the first version's
    bug and left 39% of a real match's elixir unaccounted for.
    """
    ledger = feed([10, 10, 10, 3, 3, 3])
    assert ledger.cards == 2
    assert ledger.spent == pytest.approx(7.0)


def test_a_drop_below_the_cheapest_card_is_not_a_placement():
    """1-2 elixir steps are integer quantisation noise. Counting them would
    invent placements that never happened, and the total is cumulative so the
    error would never wash out.

    It is not counted as `unexplained` either: a sub-threshold difference is
    the NORMAL case between two samples of an integer bar, so counting it would
    fire on almost every frame and tell nobody anything. `unexplained` is
    reserved for a drop large enough to be a play that still matches no legal
    combination of costs.
    """
    ledger = feed([8, 8, 8, 7, 7, 7])
    assert ledger.cards == 0
    assert ledger.spent == 0.0
    assert ledger.unexplained == 0


def test_residual_is_positive_when_regen_overflows_the_cap():
    """Conservation is one-sided: elixir regenerated while the bar sits at 10
    is invisible, so `gained` under-counts and the residual runs positive. A
    NEGATIVE residual would mean spend was invented, which is the real alarm."""
    ledger = feed([10, 10, 10, 10, 10, 10, 6, 6, 6])
    assert ledger.residual >= 0


def test_unreadable_frames_are_skipped_not_treated_as_zero():
    """None means the bar could not be read. Treating it as 0 would fabricate
    a 10-elixir spend out of a failed reading."""
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
    """Costs come from the caller, not a copy of the deck baked in here --
    the duplicated-constant drift CLAUDE.md names twice."""
    ledger = feed([9, 9, 9, 7, 7, 7], costs=(2.0, 6.0))
    assert ledger.cards == 1
    assert ledger.spent == pytest.approx(2.0)


def test_a_ledger_with_no_input_is_inert():
    ledger = ElixirLedger()
    assert ledger.spent == 0.0
    assert ledger.cards == 0
    assert ledger.residual == 0.0


# --- rate independence -------------------------------------------------------

def _trace(rate_hz, plays=((6.0, 4.0), (20.0, 3.0), (40.0, 5.0)),
           duration=60.0, regen_per_s=1.0 / 2.8, cap=10.0, start=5.0):
    """An integer-quantised elixir bar sampled at `rate_hz`, with known plays.

    Quantisation is the point: the reader returns whole elixir, so each end of
    a measured drop carries up to +/-0.5.
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
    """The regression this fix exists for. The ledger was built and scored at
    ~10 fps and is fed at ~1.5 Hz live, because the producer's rate became the
    detector's rate. Rate-blind, it recovered -48% of spend at 1.5 Hz and -82%
    at 1 Hz: `drop = last - value` silently assumed no regeneration between
    samples, which is 0.036 elixir at 10 fps and 0.24 at 1.5 Hz.
    """
    true_spent = 12.0
    for rate in (10.0, 5.0, 2.0, 1.5):
        ledger = _run(rate)
        assert ledger.cards == 3, f"{rate} Hz found {ledger.cards} cards"
        assert ledger.spent == pytest.approx(true_spent, abs=1.5), (
            f"{rate} Hz recovered {ledger.spent} of {true_spent}")


def test_regeneration_is_credited_even_while_a_card_is_played():
    """`gained` is modelled inflow, not observed rises, so a sample containing
    a placement still accrues its regen. Reading it off the bar made `gained`
    one-sided -- time spent at the 10 cap was invisible -- and the residual
    biased positive by design."""
    ledger = _run(2.0)
    assert ledger.gained > 15.0
    # gained == spent + (final - initial), within quantisation.
    assert ledger.residual == pytest.approx(0.0, abs=2.0)


def test_time_at_the_cap_is_not_credited_as_regeneration():
    """Elixir cannot accumulate past 10, so a bar sitting there gains nothing.
    Crediting it would invent inflow and make the residual demand a placement
    that never happened."""
    ledger = ElixirLedger()
    for i in range(20):
        ledger.update(10.0, now=i * 1.0)
    assert ledger.gained == pytest.approx(0.0, abs=1e-6)
    assert ledger.cards == 0


def test_the_nearest_decomposition_wins_not_the_cheapest():
    """With an integer bar most drops are ambiguous between adjacent costs.
    Taking the smallest qualifying total biased every one of them downward."""
    ledger = ElixirLedger(tolerance=1.0)
    for i, v in enumerate([9, 9, 9, 5, 5, 5]):
        ledger.update(v, now=i * 0.1)
    # A drop of ~4 must read as 4, not as the cheaper 3 that also fits.
    assert ledger.plays == [(4.0,)]


def test_a_double_elixir_multiplier_changes_the_expected_regen():
    """Regen is doubled after the 2x mark; modelling it at 1x under-counts the
    inflow and shrinks every measured drop."""
    single = ElixirLedger()
    double = ElixirLedger()
    for i in range(6):
        single.update(5.0, now=i * 1.0, multiplier=1.0)
        double.update(5.0, now=i * 1.0, multiplier=2.0)
    assert double.gained == pytest.approx(2 * single.gained, rel=1e-6)
