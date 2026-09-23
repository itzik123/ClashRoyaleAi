"""The deck's cost table, and why it must not be assumed. The elixir ledger
explains a drop by decomposing it into card costs; a cost missing from its
table makes drops that match nothing or snap to a neighbouring cost.
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


def test_the_current_deck_needs_costs_the_default_table_cannot_represent():
    """The 2.6 Hog Cycle has costs the ledger's (3, 4, 5) default cannot represent
    (Skeletons and Ice Spirit at 1, Ice Golem and The Log at 2), so the table
    must differ from the default. Those cards would not be dropped but snapped
    to the nearest representable cost (see
    test_the_default_table_misprices_a_cost_it_cannot_represent).
    """
    costs, warnings = deck_costs(DECK)
    assert costs == (1.0, 2.0, 3.0, 4.0)
    assert costs != DEFAULT_COSTS, (
        "the derivation is only load-bearing while these differ; if they have "
        "converged again, this test has stopped proving anything")
    assert warnings == []


def test_the_live_deck_is_the_deck_the_policy_TRAINED_on():
    """The loop is internally consistent whatever deck it holds, so a stale deck
    goes undetected. Deriving it from gym_wrapper.DEFAULT_DECK prevents that;
    pinned against the ids, not a hand-written list.
    """
    from live.mvp_loop import _training_deck_ids  # noqa: PLC0415
    from live.unit_to_card import hand_card_id_for  # noqa: PLC0415

    assert [hand_card_id_for(c.name) for c in DECK] == _training_deck_ids()


def test_a_deck_with_an_unusual_cost_is_reflected_in_the_table():
    """2 is not representable in the default (3, 4, 5) table at all."""
    costs, _ = deck_costs([FakeCard("barbarian_barrel", 2),
                           FakeCard("archers", 3)])
    assert 2.0 in costs


def test_the_cross_check_catches_a_real_registry_disagreement():
    """Not synthetic: the engine prices Barbarian Hut at 6 while CRBAB and the
    live game say 7. It resolves to the engine's number, which fills the
    observation's cost scalars and gates affordability_mask, but says so: for a
    deck with this card the agent's affordability is off by one elixir.
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
    table; it falls back and says so.
    """
    costs, warnings = deck_costs([FakeCard("not_a_real_card", 6)])
    assert 6.0 in costs
    assert any("not_a_real_card" in w for w in warnings)


def test_a_cost_disagreement_between_engine_and_crbab_is_reported():
    """Both describe the same real game; if they disagree one is wrong, and
    preferring either silently would hide it.
    """
    _costs, warnings = deck_costs([FakeCard("archers", 99)])
    assert any("archers" in w and "99" in w for w in warnings)


def test_the_default_table_misprices_a_cost_it_cannot_represent():
    """Same trace, two cost tables. With the tolerance at 1.0 a 2-cost play is not
    left unexplained but booked as 3, so `my_elixir_spent` drifts a full elixir
    per play with nothing reporting it.
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
