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


def feed(values, costs=DEFAULT_COSTS):
    ledger = ElixirLedger(costs=costs)
    for v in values:
        ledger.update(v)
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
    error would never wash out."""
    ledger = feed([8, 8, 8, 7, 7, 7])
    assert ledger.cards == 0
    assert ledger.unexplained >= 1


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
