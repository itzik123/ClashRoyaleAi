"""ONE win-condition resolver for the teacher and the agent's reward.

Until 2026-09-15 there were two copies and both ranked by COST ("the most
expensive building-targeter"). Measured failures (audits 05 and 07):

* the agent's `W_WIN_CONDITION_DAMAGE` term went silently dead for every deck
  whose route to a tower is a siege building, a spawning spell or a
  deploy-anywhere troop (4 of 8 plausible replacement decks);
* on a Miner deck the teacher named a 2-elixir Ice Golem, and Miner control got
  no win condition at all (a Miner targets ground, so it is not a targeter);
* LavaLoon was inverted (Lava Hound over Balloon).

The two defects found while FIXING it are pinned too, because both looked right
until the whole pool was printed:

* per-elixir ranking named the Miner over the Balloon: the probe saturates at
  one Princess (2534 HP), so every card that takes a tower reads 2534/cost and
  the cheapest wins;
* a rolling spell was probed on a cell it cannot be cast on, and Barbarian
  Barrel beat the Graveyard in graveyard_control.
"""
import pytest

import clash_royale_env as E
from python_ai.envs import gym_wrapper
from python_ai.opponents import deck_pool, teacher

NAME = lambda c: None if c is None else E.get_card_info(c)["name"]  # noqa: E731


def wincon(deck):
    return next((c for c, r in teacher.card_roles(list(deck)).items()
                 if r == "wincon"), None)


EXPECTED_POOL = {
    "hog_26_mirror": "Hog Rider",
    "dart_bait_cycle": "Goblin Barrel",
    "classic_log_bait_inferno": "Goblin Barrel",
    "graveyard_control": "Graveyard",
    "lavaloon": "Balloon",
    "miner_poison_control": "Miner",
    "xbow_30_cycle": "X-Bow",
    "mortar_cycle": "Mortar",
    "golem_beatdown": "Golem",
    "rg_fisherman_cycle": "Royal Giant",
    "royal_hogs_furnace": "Royal Hogs",
    "giant_double_dragon": "Giant",
}


@pytest.fixture(scope="module")
def pool():
    return {d.name: d for d in deck_pool.load_pool()}


@pytest.mark.parametrize("deck,expected", sorted(EXPECTED_POOL.items()))
def test_every_pool_deck_names_the_card_it_is_built_around(pool, deck, expected):
    assert NAME(wincon(pool[deck].card_ids)) == expected


def test_a_miner_beats_a_cheap_building_targeting_tank():
    deck = [6, 116, 40, 24, 72, 33, 7, 52]   # Golden Knight at slot 1, Ice Golem, Miner
    assert NAME(wincon(deck)) == "Miner"


def test_a_deck_with_no_route_to_a_tower_has_no_win_condition():
    """CONTROL that must return None, or the resolver is vacuously permissive."""
    deck = [7, 3, 29, 33, 25, 26, 27, 32]    # body-less spells and defensive buildings
    assert wincon(deck) is None


def test_the_reward_and_the_teacher_name_the_same_card(pool):
    for d in pool.values():
        assert gym_wrapper._find_win_condition(d.card_ids) == wincon(d.card_ids), d.name
    assert gym_wrapper.WIN_CONDITION_ID == wincon(gym_wrapper.DEFAULT_DECK)


def test_a_rolling_spell_is_never_probed_on_a_cell_it_cannot_be_cast_on():
    assert teacher.wincon_damage_per_elixir(101) == 0.0    # Barbarian Barrel
    assert teacher.wincon_damage_per_elixir(110) > 0.0     # Graveyard, control
