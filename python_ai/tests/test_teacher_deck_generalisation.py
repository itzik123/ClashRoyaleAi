"""Can the teacher pilot a deck whose win condition is not a walking troop?

Siege buildings (Mortar, X-Bow) and body-spawning spells (Goblin Barrel,
Graveyard) are win conditions too, and each needs a cell that actually reaches
the enemy tower. Everything is derived by injection, never a card-name list.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401
import clash_royale_env as E  # noqa: E402
from python_ai import engine_constants as EC  # noqa: E402
from python_ai.opponents import teacher as T  # noqa: E402
from python_ai.opponents.deck_pool import load_pool  # noqa: E402

CE = E.ClashRoyaleEnv


@pytest.fixture(scope="module")
def pool():
    return load_pool()


@pytest.fixture(scope="module")
def by_name():
    return {E.get_card_info(c)["name"]: c for c in E.get_all_card_ids()}


def _deck(pool, name):
    return next(d for d in pool if d.name == name).card_ids


# --- the roles ---

def test_every_pool_deck_has_exactly_one_win_condition(pool):
    """A deck with no win condition cannot attack at all: every combo family bails
    on `wincon_id is None`.
    """
    missing = [d.name for d in pool
               if not any(r == "wincon"
                          for r in T.card_roles(d.card_ids).values())]
    assert missing == [], f"decks the teacher cannot attack with: {missing}"


def test_a_siege_building_becomes_the_win_condition(pool, by_name):
    """Mortar and X-Bow reach the enemy tower from the own half; Tesla, Cannon and
    Inferno Tower do not. Separated by reach.
    """
    roles = {d.name: T.card_roles(d.card_ids) for d in pool}
    assert roles["mortar_cycle"][by_name["Mortar"]] == "wincon"
    assert roles["xbow_30_cycle"][by_name["X-Bow"]] == "wincon"
    # ...and the short-ranged building in that deck is not promoted.
    assert roles["xbow_30_cycle"][by_name["Tesla"]] == "building"


def test_a_spell_that_spawns_bodies_becomes_the_win_condition(pool, by_name):
    """Goblin Barrel puts bodies on the tower. A direct-damage spell in the same
    deck also hurts a tower, so "does it hurt the tower" cannot be the
    discriminator.
    """
    for name in ("dart_bait_cycle", "classic_log_bait_inferno"):
        roles = T.card_roles(_deck(pool, name))
        assert roles[by_name["Goblin Barrel"]] == "wincon", name
        assert roles[by_name["The Log"]] == "spell", name
    log_bait = T.card_roles(_deck(pool, "classic_log_bait_inferno"))
    assert log_bait[by_name["Rocket"]] == "spell", "Rocket is a spell"


def test_decks_that_already_had_a_win_condition_are_untouched(pool, by_name):
    """The decks that resolved a win condition under the old cost ranking. Three
    moved with the damage-ranked resolver, to the card a human names the deck
    after (Miner over Wall Breakers, Balloon over Lava Hound); the rest are
    unchanged.
    """
    expected = {
        "hog_26_mirror": "Hog Rider",
        "three_musketeers_bridge": "Battle Ram",
        "rg_fisherman_cycle": "Royal Giant",
        "miner_poison_control": "Miner",
        "royal_hogs_furnace": "Royal Hogs",
        "giant_double_dragon": "Giant",
        "pekka_bridge_spam": "Battle Ram",
        "wall_breakers_cycle": "Miner",
        "golem_beatdown": "Golem",
        "lavaloon": "Balloon",
        "mega_knight_ram": "Battle Ram",
    }
    for name, card in expected.items():
        roles = T.card_roles(_deck(pool, name))
        wc = next(c for c, r in roles.items() if r == "wincon")
        assert wc == by_name[card], (
            f"{name}: wincon moved to {E.get_card_info(wc)['name']}")


# --- the cells ---

def _empty_env(deck):
    env = CE(list(deck), list(deck), 3600)
    env.reset()
    return env


def _enemy_tower_hp(env):
    return sum(max(0.0, env.get_tower_hp(1, s)) for s in (0, 1, 2))


def _damage_from(deck, card_id, x, y, ticks=1200):
    """Enemy tower HP actually lost, both sides no-op.

    step_self_play, never step (which runs the defending C++ heuristic); and
    get_tower_damage_dealt reads non-zero on an empty board, so tower HP is the
    metric.
    """
    env = _empty_env(deck)
    before = _enemy_tower_hp(env)
    env.inject(card_id, float(x), float(y), 0, -1.0, 0)
    for _ in range(ticks // 10):
        env.step_self_play(CE.HAND_SIZE, 0.0, 0.0, CE.HAND_SIZE, 0.0, 0.0, 10,
                           False, False, False, False)
    return before - _enemy_tower_hp(env)


@pytest.mark.parametrize("deck_name,card", [("mortar_cycle", "Mortar"),
                                            ("xbow_30_cycle", "X-Bow")])
def test_the_proposed_siege_cell_actually_reaches_the_enemy_tower(
        pool, by_name, deck_name, card):
    """Most of a siege building's legal cells deal exactly zero, so proposing the
    wrong row is as good as not proposing the card.
    """
    deck = _deck(pool, deck_name)
    cid = by_name[card]
    t = T.UtilityTeacher(deck, team=0, horizon_ticks=0, k_cells=3)
    t.reset()
    env = _empty_env(deck)
    obs = np.asarray(env.get_observation_for_team(0), np.float32)
    cells = t._cells_for(t.roles[cid], cid, obs)
    assert cells, f"no cell proposed for {card}"
    reach = [(x, y) for (x, y) in cells if _damage_from(deck, cid, x, y) > 0.0]
    assert reach, (f"{card}: none of the proposed cells {cells} damages the "
                   f"enemy tower")


@pytest.mark.parametrize("deck_name,card", [("dart_bait_cycle", "Goblin Barrel"),
                                            ("graveyard_control", "Graveyard")])
def test_a_spawning_spell_lands_beside_an_enemy_tower_where_it_measures_best(
        pool, by_name, deck_name, card):
    """A spawning spell is worth far more at the tower, and the spell rule aims at
    enemy troop clusters, which do not exist on a quiet board. It lands where
    the win-condition probe measured it best around the tower, which is not
    always the tower's own cell: a Graveyard there loses its Skeletons to the
    tower before they deploy. Pinned by what the engine says the cell is worth.
    """
    deck = _deck(pool, deck_name)
    cid = by_name[card]
    t = T.UtilityTeacher(deck, team=0, horizon_ticks=0, k_cells=3)
    t.reset()
    env = _empty_env(deck)
    obs = np.asarray(env.get_observation_for_team(0), np.float32)
    cells = t._cells_for(t.roles[cid], cid, obs)
    y = float(EC.princess_y(1))
    towers = [(float(EC.LEFT_LANE_X), y), (float(EC.RIGHT_LANE_X), y)]

    def nearest_tower(x, cy):
        return min(towers, key=lambda tw: np.hypot(x - tw[0], cy - tw[1]))

    def gap(x, cy):
        tx, ty = nearest_tower(x, cy)
        return np.hypot(x - tx, cy - ty)

    beside = [(x, cy) for x, cy in cells if gap(x, cy) <= 3.0]
    assert beside, f"{card} proposed {cells}, none beside an enemy Princess Tower"

    def tower_hp_lost(x, cy):
        e = T._probe_env()
        before = T._enemy_tower_hp(e)
        e.inject(cid, float(x), float(cy), 0, -1.0, -1)
        T._roll_idle(e, T.WINCON_PROBE_TICKS)
        return before - T._enemy_tower_hp(e)

    x, cy = beside[0]
    tx, ty = nearest_tower(x, cy)
    side = -1.0 if tx < float(EC.BOARD_CENTER_X) else 1.0
    around = {(tx + side * o, ty + b): tower_hp_lost(tx + side * o, ty + b)
              for o, b in T.SPELL_ATTACK_OFFSETS}
    assert tower_hp_lost(x, cy) == max(around.values()), (
        f"{card} at {(x, cy)} takes {tower_hp_lost(x, cy)}; the best cell "
        f"around that tower takes {max(around.values())}: {around}")


# --- the frame ---

@pytest.mark.parametrize("deck_name", ["mortar_cycle", "xbow_30_cycle",
                                       "dart_bait_cycle",
                                       "classic_log_bait_inferno",
                                       "graveyard_control"])
def test_the_new_win_condition_is_playable_as_TEAM_1(pool, deck_name):
    """The teacher is team 1 in training, so a cell legal only for team 0 would
    make the fix a no-op where it matters, reading as "weak teacher".
    `is_valid_placement` takes absolute y while `_cells_for` returns own-frame
    cells; this checks the composition.
    """
    deck = _deck(pool, deck_name)
    for team in (0, 1):
        t = T.UtilityTeacher(deck, team=team, horizon_ticks=0, k_cells=3)
        t.reset()
        wc = t.wincon_id
        assert wc is not None, deck_name
        env = _empty_env(deck)
        obs = np.asarray(env.get_observation_for_team(team), np.float32)
        cells = t._cells_for(t.roles[wc], wc, obs)
        legal = [(x, y) for (x, y) in cells
                 if env.is_valid_placement(wc, float(int(x)),
                                           t.to_absolute_y(float(int(y))), team)]
        assert legal, (
            f"{deck_name}: team {team} proposed {cells} for "
            f"{E.get_card_info(wc)['name']}, none legal in the absolute frame")
