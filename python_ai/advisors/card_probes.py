"""What a card DOES, measured by injecting it into the engine.

WHY THIS MODULE EXISTS. `get_card_info` exposes cost, name, is_spell,
is_building, is_champion, is_hero, deploy_anywhere and a placement radius -- and
nothing about what the card does once it lands. The advisor layer needed exactly
that (a spell's area and damage, whether a building defends), and answered it
with three literal card ids: `CANNON_ID = 25`, `FIREBALL_ID = 7`, `HOG_ID = 15`.
Measured 2026-09-15 (audit 07, F1): on 5 of 8 plausible replacement decks none
of those ids is present, so the advisor-target coverage term -- 10% of the
placement head's training signal -- trained on ZERO cards, with nothing raising.

So the facts are recovered the way this repo already recovers archetypes and
win conditions: inject the card on an empty board and read the engine. Every
probe is cached per card id and costs milliseconds.

Values that the probes reproduce exactly, as a check that they measure what the
registry means (2026-09-15):

    Fireball   radius 2.5   damage 689      (CardRegistry: spell(7, ..., 2.5f, 689, ...))
    Arrows 3.5 / 369   Zap 2.5 / 192   Poison 3.5 / 368   Rocket 2.0 / 1485

A spell's radius is the largest CENTRE distance at which a STATIONARY target
takes damage. Stationary matters: a first version let the target walk during
the cast delay and read Fireball at 1.0.
"""
import functools

import numpy as np

import clash_royale_env as E

CE = E.ClashRoyaleEnv
HAND = CE.HAND_SIZE
PLANE = CE.BOARD_WIDTH * CE.BOARD_HEIGHT

#: Any legal 8-card deck; the probed card is injected, never drawn from hand.
_PROBE_DECK = [15, 6, 25, 40, 24, 72, 33, 7]
#: Where the probes stage their fights: open ground on the enemy half, away from
#: both the river and the towers' reach.
_CX, _CY = 9.0, 22.0
#: A Knight: ground, single target, mid HP. Held in its deploy phase for the
#: whole probe so it cannot walk out of the area being measured.
_TARGET_ID = 0
_HOLD_TICKS = 300
#: A P.E.K.K.A.: enough HP that no single spell kills it, so the damage it takes
#: IS the spell's damage.
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


def _damage_to_enemy_target(card_id, target_id, dx):
    """Troop damage team 0 deals after casting `card_id` at the centre, with
    one enemy `target_id` held stationary `dx` tiles to the side."""
    e = _env()
    e.inject(target_id, _CX + dx, _CY, 1, -1.0, _HOLD_TICKS)
    e.step_self_play(HAND, 0.0, 0.0, HAND, 0.0, 0.0, 1)
    before = e.get_troop_damage_dealt(0)
    e.inject(card_id, _CX, _CY, 0, -1.0, 0)
    _idle(e, 40)
    return e.get_troop_damage_dealt(0) - before


@functools.lru_cache(maxsize=512)
def spell_effect(card_id):
    """(radius, damage) for a DAMAGING area spell, or None.

    None for anything that is not a spell, cannot be cast on the enemy half
    (a rolling spell: its value is a corridor, not a disc), puts bodies on the
    board (Goblin Barrel, Graveyard -- those are win conditions, not answers), or
    deals no damage (Rage, Clone, Freeze in this engine).
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
def building_defends(card_id):
    """Is this a DEFENSIVE building -- one that attacks and has no route to a
    tower of its own?

    Excludes siege buildings (Mortar, X-Bow: their rule is the siege row, and a
    defensive coverage rule would pull them back to exactly the cells worth
    nothing) and buildings that do not attack (Elixir Collector).
    """
    info = E.get_card_info(card_id)
    if not info["is_building"]:
        return False
    from python_ai.opponents.teacher import siege_reach
    if siege_reach(card_id) > 0.0:
        return False
    # BEHAVIOURAL, not a channel read: the observation's DPS channel is
    # non-zero for an Elixir Collector, which never attacks. So place the
    # building, hold one enemy Knight four tiles in front of it -- inside any
    # defensive building's range and outside both Princess Towers' -- and see
    # whether it takes damage. The control is the same board with no building.
    def dealt(with_building):
        e = _env()
        if with_building:
            e.inject(card_id, _CX, 8.0, 0, -1.0, 0)
        e.inject(_TARGET_ID, _CX, 12.0, 1, -1.0, _HOLD_TICKS)
        _idle(e, 60)
        return e.get_troop_damage_dealt(0)
    return dealt(True) > dealt(False)


@functools.lru_cache(maxsize=512)
def walking_building_targeter(card_id):
    """A troop that walks past defenders to hit buildings (Hog, Giant, Balloon),
    as opposed to one that skips the walk (Miner) or is not a troop at all."""
    info = E.get_card_info(card_id)
    if info["is_spell"] or info["is_building"] or info.get("deploy_anywhere", False):
        return False
    e = _env()
    e.inject(card_id, _CX, 8.0, 0)
    e.step_self_play(HAND, 0.0, 0.0, HAND, 0.0, 0.0, 1)
    obs = np.asarray(e.get_observation_for_team(0), np.float32)
    return bool(float(obs[2 * PLANE:3 * PLANE].max()) > 1e-6)
