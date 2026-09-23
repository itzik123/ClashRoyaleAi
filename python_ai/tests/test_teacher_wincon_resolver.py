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


# --- TODO 00.6: a SPAWNER building is not a siege win condition --------------
# `siege_reach > 0` admitted every building whose SPAWNED bodies walk to a tower.
# Measured over 28 decks: Splashyard named Tombstone over its Graveyard, and a
# Barbarian Hut outranked the Giant beside it (6182 in a 1200-tick siege window
# against a troop's 300-tick one). The 16 pool decks are the control above.

X_BOW, MORTAR, BARB_HUT, GOBLIN_HUT, TOMBSTONE, GOBLIN_CAGE, GOBLIN_DRILL = (
    92, 93, 94, 95, 96, 97, 98)
CANNON = 25


def test_only_a_building_that_fires_at_the_tower_itself_is_siege():
    for cid in (X_BOW, MORTAR):
        assert teacher.siege_building(cid), NAME(cid)
        assert teacher.building_spawns_bodies(cid) == 0, NAME(cid)
    for cid in (BARB_HUT, TOMBSTONE, GOBLIN_CAGE, GOBLIN_HUT):
        assert not teacher.siege_building(cid), NAME(cid)
        assert teacher.building_spawns_bodies(cid) > 0, NAME(cid)
    # Deploy-anywhere is its own class, and a defensive building is no route.
    assert not teacher.siege_building(GOBLIN_DRILL)
    assert not teacher.siege_building(CANNON)


def test_splashyard_is_won_by_its_graveyard_not_its_tombstone():
    from python_ai import deck as D
    deck = D.parse_deck("graveyard,poison,baby dragon,bowler,ice wizard,"
                        "tornado,the log,tombstone")
    assert NAME(wincon(deck)) == "Graveyard"


def test_a_barbarian_hut_does_not_outrank_the_giant_beside_it():
    from python_ai import deck as D
    deck = D.parse_deck("barbarian hut,giant,musketeer,zap,fireball,knight,"
                        "archers,minions")
    assert NAME(wincon(deck)) == "Giant"


def test_a_goblin_drill_is_measured_and_played_beside_the_enemy_tower():
    """From the own siege row a Drill measured 0 tower damage in 300 ticks;
    beside the tower, 2654. It is deploy-anywhere, like the Miner."""
    import numpy as np
    from python_ai import engine_constants as EC
    assert teacher._wincon_eligible(GOBLIN_DRILL, False)
    assert teacher.wincon_damage_per_elixir(GOBLIN_DRILL) * 4.0 > 2000.0
    deck = [GOBLIN_DRILL, 3, 29, 0, 1, 24, 41, 25]
    t = teacher.UtilityTeacher(deck, team=0)
    env = E.ClashRoyaleEnv(deck, deck, 3600)
    env.seed(0)
    obs = np.asarray(env.get_observation_for_team(0), dtype=np.float32)
    cells = t._cells_for("wincon", GOBLIN_DRILL, obs)
    assert cells and all(y > EC.BRIDGE_Y for _x, y in cells), cells


def test_the_advisor_judges_a_spawner_by_behaviour_not_by_the_siege_test():
    from python_ai.advisors import card_probes
    assert card_probes.building_defends(CANNON)          # control
    assert card_probes.building_defends(GOBLIN_HUT)      # its Spear Goblins shoot
    assert not card_probes.building_defends(X_BOW)       # siege
    assert not card_probes.building_defends(GOBLIN_DRILL)  # a win condition
