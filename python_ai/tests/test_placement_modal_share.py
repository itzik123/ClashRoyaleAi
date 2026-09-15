"""Per-card MODAL SHARE of placements, logged every update.

CLAUDE.md is explicit that this -- not `Entropy/Placement_ByCard_Min` -- is the
conditional-collapse detector: measured 2026-08-14, the lowest-entropy card was
the HEALTHIEST (most played, modal share 19%) while a card at 91% modal share
had higher entropy. "Count how often each card's argmax cell repeats across
states, not how peaked the distribution is." Audit 08 found no modal-share
scalar logged anywhere, so the run about to start had no working detector.
"""
import numpy as np
import pytest

from python_ai.rl.placement_stats import ModalShareWindow


def test_a_card_always_placed_on_one_cell_reads_one():
    w = ModalShareWindow(window_updates=5, min_plays=10)
    w.add_update(card_ids=[25] * 30, cells=[100] * 30)
    assert w.shares() == {25: pytest.approx(1.0)}


def test_a_card_spread_evenly_reads_low():
    w = ModalShareWindow(window_updates=5, min_plays=10)
    w.add_update(card_ids=[7] * 40, cells=list(range(40)))
    assert w.shares()[7] == pytest.approx(1 / 40)


def test_a_card_with_too_few_plays_is_not_reported():
    """A card played three times reads 33%+ by construction -- noise, not collapse."""
    w = ModalShareWindow(window_updates=5, min_plays=10)
    w.add_update(card_ids=[15, 15, 15], cells=[1, 2, 3])
    assert 15 not in w.shares()


def test_the_window_forgets_old_updates():
    w = ModalShareWindow(window_updates=2, min_plays=5)
    w.add_update(card_ids=[25] * 20, cells=[9] * 20)          # collapsed
    w.add_update(card_ids=[25] * 20, cells=list(range(20)))   # recovered
    w.add_update(card_ids=[25] * 20, cells=list(range(20, 40)))
    assert w.shares()[25] == pytest.approx(1 / 40)


def test_no_op_rows_are_the_callers_job_but_negative_ids_are_ignored():
    w = ModalShareWindow(window_updates=2, min_plays=5)
    w.add_update(card_ids=[-1] * 10 + [6] * 10, cells=[0] * 10 + list(range(10)))
    assert set(w.shares()) == {6}
