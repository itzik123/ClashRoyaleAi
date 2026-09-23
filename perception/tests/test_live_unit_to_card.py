"""Tests for live/unit_to_card.py, against the real engine binding and CRBAB
tables: a fixture would freeze one side and miss the drift the module exists to
catch.
"""
from __future__ import annotations

import pytest

pytest.importorskip("clashroyalebuildabot", reason="vendored bot not importable")

from live.unit_to_card import (PROJECTILE_CLASSES, UnitMappingError,  # noqa: E402
                               card_id_for, coverage, is_board_presence,
                               unit_to_card_id)


def test_every_detector_class_maps(engine):
    """A detected unit with no card cannot become an observation entry."""
    mapped, missing = coverage()
    assert missing == [], f"detector classes with no engine card: {missing}"
    assert mapped >= 97


def test_our_own_deck_resolves(engine):
    """`archer` and `minion` do not match engine names, and Archers and Minions
    are two of our cards: the case that makes the mapping necessary.
    """
    expected = {
        "archer": "Archers", "minion": "Minions", "valkyrie": "Valkyrie",
        "cannon": "Cannon", "giant": "Giant", "musketeer": "Musketeer",
        "minipekka": "Mini PEKKA",
    }
    for unit, card_name in expected.items():
        assert engine.get_card_info(card_id_for(unit))["name"] == card_name


@pytest.mark.parametrize("unit,parent", [
    ("golemite", "Golem"),
    ("lava_pup", "Lava Hound"),
    ("elixir_golem_small", "Elixir Golem"),
    ("elixir_golem_medium", "Elixir Golem"),
    ("elixir_golem_large", "Elixir Golem"),
    ("phoenix_egg", "Phoenix"),
    ("phoenix_small", "Phoenix"),
])
def test_death_spawns_map_to_their_parent_card(engine, unit, parent):
    assert engine.get_card_info(card_id_for(unit))["name"] == parent


@pytest.mark.parametrize("unit,card", [
    ("skeleton", "Skeletons"),        # over Skeleton Army, Graveyard, Witch...
    ("goblin", "Goblins"),            # over Goblin Barrel, Goblin Gang...
    ("spear_goblin", "Spear Goblins"),
    ("barbarian", "Barbarians"),      # over Barbarian Barrel, Barbarian Hut
    ("bat", "Bats"),                  # over Night Witch
    ("minion", "Minions"),            # over Minion Horde
])
def test_ambiguous_units_take_the_documented_preference(engine, unit, card):
    """One unit, several possible parents: the choice must be deliberate."""
    assert engine.get_card_info(card_id_for(unit))["name"] == card


def test_projectiles_are_not_board_presence(engine):
    """ClashEnv skips !isTargetable() entities, so a spell in flight must not
    become a spatial-channel entry.
    """
    for name in PROJECTILE_CLASSES:
        assert is_board_presence(name) is False
        card_id_for(name)          # still resolvable, just not board presence
    assert is_board_presence("giant") is True


def test_unknown_unit_raises_rather_than_defaulting(engine):
    """A wrong card id is worse than a missing one: it fills the attribute
    channels with another card's stats.
    """
    with pytest.raises(UnitMappingError):
        card_id_for("not_a_real_unit")


def test_mapping_is_stable_across_calls(engine):
    assert unit_to_card_id() is unit_to_card_id()
