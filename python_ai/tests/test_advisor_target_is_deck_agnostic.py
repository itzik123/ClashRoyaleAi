"""The advisor target speaks for whatever deck is trained.

Roles come from `card_probes`, which measures what a card does; for the shipped
deck the table is unchanged.
"""
import numpy as np
import pytest

import clash_royale_env as E
from python_ai import deck as D
from python_ai.advisors import advisor_target as AT
from python_ai.advisors import card_probes

NAME = lambda c: E.get_card_info(c)["name"]  # noqa: E731


def test_the_shipped_deck_keeps_exactly_its_three_rules():
    table = AT.advisor_cards_for(list(D.SHIPPED_DECK))
    assert table == {25: "building", 7: "spell", 15: "wincon"}


@pytest.mark.parametrize("spec,expected", [
    # Royal Giant control: the targeter, the spell and the building all speak.
    ("royal giant,fisherman,hunter,electro spirit,skeletons,lightning,the log,cannon",
     {"Royal Giant": "wincon", "Lightning": "spell", "Cannon": "building"}),
    # A Tesla does the Cannon's job and gets the Cannon's rule.
    ("hog rider,musketeer,tesla,ice golem,skeletons,ice spirit,the log,fireball",
     {"Hog Rider": "wincon", "Tesla": "building", "Fireball": "spell"}),
    # Miner control: only Poison fits a rule. The Miner skips the bridge, so
    # the bridge-commit rule must not claim it.
    ("miner,poison,bomber,musketeer,valkyrie,skeletons,ice spirit,the log",
     {"Poison": "spell"}),
])
def test_roles_follow_the_deck(spec, expected):
    table = AT.advisor_cards_for(D.parse_deck(spec))
    assert {NAME(c): k for c, k in table.items()} == expected


def test_a_building_that_does_not_attack_gets_no_defensive_rule():
    deck = D.parse_deck("golem,baby dragon,mega minion,lightning,tornado,"
                        "barbarians,elixir collector,zap")
    table = AT.advisor_cards_for(deck)
    assert 99 not in table, "Elixir Collector was given the Cannon's coverage rule"
    assert table.get(19) == "wincon"


def test_a_spell_is_scored_with_its_own_measured_radius():
    """Poison (3.5) must catch a unit that Fireball (2.5) cannot reach."""
    assert card_probes.spell_effect(32)[0] > card_probes.spell_effect(7)[0]
