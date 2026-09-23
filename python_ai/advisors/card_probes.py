"""What a card does, measured by injecting it into the engine.

`get_card_info` says nothing about what a card does once it lands (a spell's
area and damage, whether a building defends), and hardcoded card ids go missing
on any other deck. So each fact is recovered on an empty board and cached per
card id.

A spell's radius is the largest centre distance at which a stationary target
takes damage (a target left free to walk during the cast delay reads too
small). Its damage is its whole effect, read until the damage stops, so
damage-over-time spells are counted in full. The probes reproduce the registry:
Fireball 2.5 / 689, Arrows 3.5 / 369, Zap 2.5 / 192, Poison 3.5 / 736, Rocket
2.0 / 1485.
"""
import functools

import numpy as np

import clash_royale_env as E
from python_ai import engine_constants as EC

CE = E.ClashRoyaleEnv
HAND = CE.HAND_SIZE
PLANE = CE.BOARD_WIDTH * CE.BOARD_HEIGHT

#: Any legal 8-card deck; the probed card is injected, never drawn from hand.
_PROBE_DECK = [15, 6, 25, 40, 24, 72, 33, 7]
#: Open ground on the enemy half, clear of the river and the towers' reach.
_CX, _CY = 9.0, 22.0
#: A Knight (ground, single target, mid HP), held in its deploy phase so it
#: cannot walk out of the measured area.
_TARGET_ID = 0
_HOLD_TICKS = 300
#: A P.E.K.K.A.: no single spell kills it, so the damage it takes is the
#: spell's damage.
_TANK_ID = 13
_RADIUS_STEP = 0.25
_RADIUS_MAX = 6.0


def _env():
    e = CE(_PROBE_DECK, _PROBE_DECK, 3600)
    e.seed(1)
    return e


def _idle(e, ticks):
    for _ in range(max(1, ticks // 10)):
        e.step_self_play(HAND, 0.0, 0.0, HAND, 0.0, 0.0, 10)


#: A spell's effect is over once its damage has not moved for this long: longer
#: than any gap between pulses (<= 10 ticks) or cast delay (Rocket's 15).
_SETTLE_TICKS = 50


def _settled_damage(e, read):
    """Idle `e` until `read(e)` stops changing and return the final value, bounded
    by `_HOLD_TICKS` (the longest spell, Poison, finishes at 80).
    """
    value, quiet, elapsed = read(e), 0, 0
    while elapsed < _HOLD_TICKS - 10:
        _idle(e, 10)
        elapsed += 10
        now = read(e)
        quiet = quiet + 10 if now == value else 0
        value = now
        if quiet >= _SETTLE_TICKS and elapsed >= _SETTLE_TICKS:
            break
    return value


def _damage_to_enemy_target(card_id, target_id, dx):
    """Troop damage team 0 deals by casting `card_id` at the centre, with one
    enemy `target_id` held stationary `dx` tiles to the side.
    """
    e = _env()
    e.inject(target_id, _CX + dx, _CY, 1, -1.0, _HOLD_TICKS)
    e.step_self_play(HAND, 0.0, 0.0, HAND, 0.0, 0.0, 1)
    before = e.get_troop_damage_dealt(0)
    e.inject(card_id, _CX, _CY, 0, -1.0, 0)
    return _settled_damage(e, lambda env: env.get_troop_damage_dealt(0)) - before


@functools.lru_cache(maxsize=512)
def spell_effect(card_id):
    """(radius, damage) for a damaging area spell, or None.

    None for a non-spell, a spell that cannot be cast on the enemy half (a
    roller: its value is a corridor), one that puts bodies on the board (Goblin
    Barrel, Graveyard: win conditions, not answers), or one that deals no
    damage (Rage, Clone, Freeze here).
    """
    info = E.get_card_info(card_id)
    if not info["is_spell"]:
        return None
    if not _env().is_valid_placement(card_id, _CX, 30.0, 0):
        return None
    from python_ai.opponents.teacher import spell_spawns_bodies
    if spell_spawns_bodies(card_id) > 0:
        return None
    damage = _damage_to_enemy_target(card_id, _TANK_ID, 0.0)
    if damage <= 0:
        return None
    radius = 0.0
    d = 0.0
    while d <= _RADIUS_MAX + 1e-9:
        if _damage_to_enemy_target(card_id, _TARGET_ID, d) > 0:
            radius = d
        else:
            break
        d += _RADIUS_STEP
    if radius <= 0.0:
        return None
    return float(radius), float(damage)


@functools.lru_cache(maxsize=512)
def spell_tower_damage(card_id):
    """Crown Tower HP one cast of `card_id` removes, cast on team 1's left
    Princess Tower.

    Not `spell_effect`'s damage. The real game deals spells a reduced Crown
    Tower damage (15-30% of troop damage); this engine currently deals 100%
    (UPSTREAM_REQUESTS.md item 29). Measuring the tower keeps the lethal window
    right under either engine.
    """
    info = E.get_card_info(card_id)
    if not info["is_spell"]:
        return 0.0
    e = _env()
    x, y = float(EC.LEFT_LANE_X), float(EC.princess_y(1))
    if not e.is_valid_placement(card_id, x, y, 0):
        return 0.0
    before = e.get_tower_hp(1, 1)
    e.inject(card_id, x, y, 0, -1.0, 0)
    return float(before - _settled_damage(e, lambda env: env.get_tower_hp(1, 1)))


@functools.lru_cache(maxsize=64)
def roller_damage(card_id):
    """Damage a rolling spell (The Log, Barbarian Barrel) deals one body it
    sweeps, or 0.0 for anything else.

    Cast from the bridge row at an enemy P.E.K.K.A. three tiles down the
    corridor, read at first contact (a roller hits each target once, and a
    Barbarian Barrel's Barbarian would add damage later).
    """
    info = E.get_card_info(card_id)
    if not info["is_spell"] or spell_effect(card_id) is not None:
        return 0.0
    e = _env()
    x, y = _CX, float(EC.BRIDGE_Y)
    if not e.is_valid_placement(card_id, x, y, 0):
        return 0.0
    e.inject(_TANK_ID, x, y + 3.0, 1, -1.0, _HOLD_TICKS)
    e.step_self_play(HAND, 0.0, 0.0, HAND, 0.0, 0.0, 1)
    before = e.get_troop_damage_dealt(0)
    e.inject(card_id, x, y, 0, -1.0, 0)
    for _ in range(60):
        e.step_self_play(HAND, 0.0, 0.0, HAND, 0.0, 0.0, 1)
        dealt = e.get_troop_damage_dealt(0) - before
        if dealt > 0:
            return float(dealt)
    return 0.0


def damage_spell(deck):
    """(card_id, tower_damage, cost) for the deck's finishing spell, or None.

    Both spell reward terms (W_LETHAL_SPELL, W_SPELL_VALUE_START) follow this
    card. Chosen by measured tower damage (the lethal window is "tower hp <=
    damage", so a Zap + Rocket deck keys it to the Rocket), then cost, then id.
    Candidates are `spell_effect`'s damaging area spells, so a deck whose only
    spell is a roller or a spawner returns None; both terms are then zero, and
    `validate_deck` reports it.
    """
    best = None
    for cid in sorted({int(c) for c in deck}):
        if spell_effect(cid) is None:
            continue
        damage = spell_tower_damage(cid)
        if damage <= 0.0:
            continue
        cost = float(E.get_card_info(cid)["cost"])
        key = (-damage, cost, cid)
        if best is None or key < best[0]:
            best = (key, (cid, damage, cost))
    return None if best is None else best[1]


@functools.lru_cache(maxsize=512)
def building_defends(card_id):
    """Is this a defensive building: one that attacks and has no route to a tower
    of its own?

    Excludes siege buildings (`teacher.siege_building`: their place is the
    siege row), deploy-anywhere buildings (Goblin Drill is a win condition) and
    buildings that do not attack (Elixir Collector). Spawners are judged by the
    behavioural test: Goblin Hut passes; Tombstone, Barbarian Hut and Goblin
    Cage do not attack inside its window.
    """
    info = E.get_card_info(card_id)
    if not info["is_building"] or info.get("deploy_anywhere", False):
        return False
    from python_ai.opponents.teacher import siege_building
    if siege_building(card_id):
        return False
    # Behavioural, because the DPS channel is non-zero even for an Elixir
    # Collector: hold an enemy Knight four tiles in front (inside a defensive
    # building's range, outside the towers') and compare against the same board
    # without the building.
    def dealt(with_building):
        e = _env()
        if with_building:
            e.inject(card_id, _CX, 8.0, 0, -1.0, 0)
        e.inject(_TARGET_ID, _CX, 12.0, 1, -1.0, _HOLD_TICKS)
        _idle(e, 60)
        return e.get_troop_damage_dealt(0)
    return dealt(True) > dealt(False)


#: The flyer the air probe uses: a Balloon, looked up by name. A flying
#: building-targeter ignores every troop, so only an anti-air card can answer
#: it.
_AIR_TARGET_NAME = "Balloon"


@functools.lru_cache(maxsize=1)
def _air_target_id():
    return next(c for c in E.get_all_card_ids()
                if E.get_card_info(c)["name"] == _AIR_TARGET_NAME)


@functools.lru_cache(maxsize=512)
def damages_air(card_id):
    """Can this card hurt a flying unit? Behavioural, like `building_defends`.

    An enemy Balloon is held stationary, the card is played beside it (troop or
    building) or on it (spell), and the damage is compared with the same board
    without the card. The observation's anti-air channel cannot be read
    naively: a whole-plane max also sees the caster's own towers, which all
    target air.
    """
    info = E.get_card_info(card_id)
    target = _air_target_id()

    def dealt(with_card):
        e = _env()
        e.inject(target, _CX, _CY, 1, -1.0, _HOLD_TICKS)
        if with_card:
            dx = 0.0 if info["is_spell"] else 1.5
            e.inject(card_id, _CX + dx, _CY - (0.0 if info["is_spell"] else 1.0),
                     0, -1.0, 0)
        e.step_self_play(HAND, 0.0, 0.0, HAND, 0.0, 0.0, 1)
        before = e.get_troop_damage_dealt(0)
        _idle(e, 100)
        return e.get_troop_damage_dealt(0) - before
    return dealt(True) > dealt(False)


@functools.lru_cache(maxsize=512)
def walking_building_targeter(card_id):
    """A troop that walks past defenders to hit buildings (Hog, Giant, Balloon),
    as opposed to one that skips the walk (Miner) or is not a troop.
    """
    info = E.get_card_info(card_id)
    if info["is_spell"] or info["is_building"] or info.get("deploy_anywhere", False):
        return False
    e = _env()
    e.inject(card_id, _CX, 8.0, 0)
    e.step_self_play(HAND, 0.0, 0.0, HAND, 0.0, 0.0, 1)
    obs = np.asarray(e.get_observation_for_team(0), np.float32)
    return bool(float(obs[2 * PLANE:3 * PLANE].max()) > 1e-6)
