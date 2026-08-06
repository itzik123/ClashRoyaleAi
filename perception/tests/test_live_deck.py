"""The deck's cost table, and why it must not be assumed.

The elixir ledger explains a drop by decomposing it into card costs. Its
default table is (3, 4, 5), which is exactly the current deck's profile -- so
the live loop worked while quietly depending on that coincidence. A deck with a
2 or a 6 would produce drops matching nothing, and those placements would leave
no trace at all.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

pytest.importorskip("clashroyalebuildabot", reason="vendored bot not importable")
pytest.importorskip("windows_capture", reason="live capture not installed")

from live.elixir_ledger import DEFAULT_COSTS, ElixirLedger  # noqa: E402
from live.mvp_loop import DECK, deck_costs  # noqa: E402


@dataclass(frozen=True)
class FakeCard:
    name: str
    cost: int


def test_the_current_deck_derives_the_costs_it_used_to_assume():
    """Behaviour-preserving today. If this ever fails, the live loop's elixir
    accounting changed meaning without anyone choosing that."""
    costs, warnings = deck_costs(DECK)
    assert costs == DEFAULT_COSTS
    assert warnings == []


def test_a_deck_with_an_unusual_cost_is_reflected_in_the_table():
    """The failure the derivation exists to prevent: 2 is not representable in
    the default (3, 4, 5) table at all."""
    costs, _ = deck_costs([FakeCard("barbarian_barrel", 2),
                           FakeCard("archers", 3)])
    assert 2.0 in costs


def test_the_cross_check_catches_a_real_registry_disagreement():
    """Not a synthetic case. The engine's registry prices Barbarian Hut at 6
    while CRBAB -- and the live game -- say 7, so the two sources genuinely
    disagree about a card that exists in both.

    It resolves to the ENGINE's number, because that is what fills the
    observation's cost scalars and what affordability_mask gates on; a ledger
    disagreeing with the mask would be worse than one disagreeing with reality.
    But it says so, because for any deck containing this card the agent's whole
    notion of affordability is off by one elixir and no amount of perception
    work would reveal it.
    """
    costs, warnings = deck_costs([FakeCard("barbarian_hut", 7)])
    assert costs == (6.0,)
    assert any("barbarian_hut" in w and "6" in w and "7" in w
               for w in warnings)


def test_costs_are_distinct_and_sorted():
    costs, _ = deck_costs([FakeCard("archers", 3), FakeCard("minions", 3),
                           FakeCard("giant", 5)])
    assert costs == (3.0, 5.0)


def test_a_card_the_engine_does_not_carry_falls_back_rather_than_vanishing():
    """Dropping it would remove a whole card's worth of explanations from the
    table -- the exact silent loss being fixed. It falls back and says so."""
    costs, warnings = deck_costs([FakeCard("not_a_real_card", 6)])
    assert 6.0 in costs
    assert any("not_a_real_card" in w for w in warnings)


def test_a_cost_disagreement_between_engine_and_crbab_is_reported():
    """Both describe the same real game. If they disagree, one is wrong about
    it, and preferring either silently would hide that."""
    _costs, warnings = deck_costs([FakeCard("archers", 99)])
    assert any("archers" in w and "99" in w for w in warnings)


def test_the_default_table_misprices_a_cost_it_cannot_represent():
    """Same trace, two cost tables -- and the damage is not what I first
    assumed. With the tolerance at 1.0 the drop does not go unexplained; it
    finds the NEIGHBOURING cost instead. A 2-cost play is booked as 3, so
    `my_elixir_spent` drifts by a full elixir every time that card is played
    and nothing anywhere reports a problem.
    """
    trace = [(i * 0.1, v) for i, v in enumerate([8, 8, 8, 6, 6, 6])]

    assumed = ElixirLedger()                       # the old default (3, 4, 5)
    for t, v in trace:
        assumed.update(v, now=t)
    assert assumed.cards == 1
    assert assumed.spent == pytest.approx(3.0), "misattributed, not dropped"

    derived = ElixirLedger(costs=(2.0, 3.0, 4.0))
    for t, v in trace:
        derived.update(v, now=t)
    assert derived.cards == 1
    assert derived.spent == pytest.approx(2.0)
