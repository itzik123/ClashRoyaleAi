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
registry means (2026-09-16):

    Fireball   radius 2.5   damage 689      (CardRegistry: spell(7, ..., 2.5f, 689, ...))
    Arrows 3.5 / 369   Zap 2.5 / 192   Poison 3.5 / 736   Rocket 2.0 / 1485

A spell's radius is the largest CENTRE distance at which a STATIONARY target
takes damage. Stationary matters: a first version let the target walk during
the cast delay and read Fireball at 1.0.

**A spell's damage is its WHOLE effect, read until it stops.** Until 2026-09-16
the probe idled a fixed 40 ticks and so cut every damage-over-time spell off
halfway: Poison is `spell(32, ..., 92, ...).withRepeats(8, 10)` = 736 over 80
ticks and read **368**; Goblin Curse, 43 x 6 = 258, read **129**. This
docstring then quoted the 368 as proof the probe "reproduces exactly" what the
registry means -- a check anchored on the truncated reading, so it could not
fail. `_settled_damage` now idles until the damage stops moving.
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


#: A spell's effect is over once its damage has not moved for this long. Longer
#: than any registered gap between two pulses of one spell (every `withRepeats`
#: interval is 10 ticks or less) and longer than any cast delay (Rocket's 15).
_SETTLE_TICKS = 50


def _settled_damage(e, read):
    """Idle `e` until `read(e)` stops changing, and return its final value.

    The damage-over-time fix. Bounded by `_HOLD_TICKS` because a held target is
    only held that long; no registered non-spawning spell comes near it (Poison,
    the longest, is done at 80).
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
    """Troop damage team 0 deals after casting `card_id` at the centre, with
    one enemy `target_id` held stationary `dx` tiles to the side."""
    e = _env()
    e.inject(target_id, _CX + dx, _CY, 1, -1.0, _HOLD_TICKS)
    e.step_self_play(HAND, 0.0, 0.0, HAND, 0.0, 0.0, 1)
    before = e.get_troop_damage_dealt(0)
    e.inject(card_id, _CX, _CY, 0, -1.0, 0)
    return _settled_damage(e, lambda env: env.get_troop_damage_dealt(0)) - before


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
def spell_tower_damage(card_id):
    """Crown Tower HP one cast of `card_id` removes, cast on the tower's centre.

    Team 0 casts on team 1's LEFT Princess Tower -- position from the bound
    arena, never restated -- on an otherwise empty board, and the tower's own HP
    loss is read once the effect has settled.

    THIS IS NOT `spell_effect`'s DAMAGE, and the difference is the whole reason
    it exists. The real game deals a spell's "Crown Tower damage", a published
    15-30% of its troop damage (Fireball 159 of 688, Rocket 371 of 1484, The Log
    41 of 268). This engine currently charges towers 100% -- measured for every
    registered spell on 2026-09-16, and proposed as a C++ fix in
    `perception/UPSTREAM_REQUESTS.md`. A lethal window keyed to the TROOP number
    would be correct today and silently 4x too wide the day that proposal lands,
    which is precisely the one-change-later failure this repo keeps recording.
    Measuring the tower is right under both engines.
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
    """Damage a ROLLING spell (The Log, Barbarian Barrel) deals one body it sweeps.

    `spell_effect` declines rollers -- their value is a corridor, not a disc -- so
    their damage had no derivation, and a harness restated The Log's as 240.0
    against the registry's 269 (`spell(33, "The Log", ..., 269, ...)`). Measured:
    cast from the bridge row (a roller may be cast on its own half or the river,
    no further), one enemy P.E.K.K.A. held three tiles down the corridor, and the
    damage read at FIRST CONTACT -- a roller hits each target once, and reading
    any later would add the Barbarian a Barbarian Barrel drops at the end of its
    roll. 0.0 for anything that is not a roller castable there.
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

    THE DECK'S spell, not card id 7. Two reward terms are built on this one card
    -- `W_LETHAL_SPELL` ("a tower is inside my spell's damage and I can cast it")
    and `W_SPELL_VALUE_START` (the trade ratio a cast earns) -- and both were
    keyed to Fireball, so a deck without it trained under a quietly smaller
    objective with nothing raising: measured over 12 seeded matches, a Hog deck
    carrying Rocket instead of Fireball had BOTH terms at exactly zero on all
    2,372 steps. See TODO.md item 00.3.

    **Picked by TOWER damage, not by cost or by hand order.** Both terms are
    about FINISHING: the lethal window is literally "tower hp <= damage", and a
    deck carrying Zap and Rocket must key it to the Rocket. Cost breaks a tie
    only because the cheaper of two equal spells is the one a cycle deck can
    actually reach, and the id breaks that in turn so the answer is
    deterministic. The damage is `spell_tower_damage`'s -- what the spell does
    to the thing the window is about -- not the troop damage.

    Candidates are the cards `spell_effect` accepts: damaging area spells. That
    declines rollers (a corridor, not a disc) and spawning spells (a Goblin
    Barrel is a win condition, not a finisher). So a deck whose only spell is The
    Log returns None; that is a legitimate deck, both terms then contribute
    exactly zero, and `validate_deck` says so at startup rather than letting it
    go silent.
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
    """Is this a DEFENSIVE building -- one that attacks and has no route to a
    tower of its own?

    Excludes siege buildings (Mortar, X-Bow: their rule is the siege row, and a
    defensive coverage rule would pull them back to exactly the cells worth
    nothing), deploy-anywhere buildings (a Goblin Drill is a win condition,
    played beside the enemy tower) and buildings that do not attack (Elixir
    Collector).

    "Siege" is `teacher.siege_building`, the resolver's own definition. This read
    `siege_reach > 0` until 2026-09-23, which also excluded every SPAWNER before
    the behavioural test below ever ran. Spawners are now judged by that test on
    their own merits: Goblin Hut passes (its Spear Goblins shoot); Tombstone,
    Barbarian Hut and Goblin Cage do not attack inside its 60-tick window, so
    they stay uncovered -- by measurement now, not by the siege test. TODO 00.6.
    """
    info = E.get_card_info(card_id)
    if not info["is_building"] or info.get("deploy_anywhere", False):
        return False
    from python_ai.opponents.teacher import siege_building
    if siege_building(card_id):
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


#: The flyer the air probe defends against: a Balloon, found by NAME from the
#: registry rather than by a literal id. A flying BUILDING-targeter is the case
#: only an anti-air card can answer -- a Minion chases troops, so a ground unit
#: can still distract it; a Balloon ignores every troop on the board.
_AIR_TARGET_NAME = "Balloon"


@functools.lru_cache(maxsize=1)
def _air_target_id():
    return next(c for c in E.get_all_card_ids()
                if E.get_card_info(c)["name"] == _AIR_TARGET_NAME)


@functools.lru_cache(maxsize=512)
def damages_air(card_id):
    """Can this card hurt a flying unit? BEHAVIOURAL, like `building_defends`.

    One enemy Balloon is held stationary; the card is played beside it (a troop
    or building) or on it (a spell); the Balloon's damage is read against the
    same board without the card. Measured 2026-09-23: Musketeer, Ice Spirit,
    Archers, Minions, Tesla and Fireball do; Knight, Skeletons, Ice Golem, Hog
    Rider, Cannon, Bomb Tower and The Log do not -- the real game's split.

    NOT read off the observation's anti-air channel without care: that channel
    is right per entity, but a whole-plane max also sees the caster's own
    TOWERS, which all target air (`Tower.h`), and reads 1.0 for every card --
    a probe of this very function did exactly that before it was made
    behavioural.
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
    as opposed to one that skips the walk (Miner) or is not a troop at all."""
    info = E.get_card_info(card_id)
    if info["is_spell"] or info["is_building"] or info.get("deploy_anywhere", False):
        return False
    e = _env()
    e.inject(card_id, _CX, 8.0, 0)
    e.step_self_play(HAND, 0.0, 0.0, HAND, 0.0, 0.0, 1)
    obs = np.asarray(e.get_observation_for_team(0), np.float32)
    return bool(float(obs[2 * PLANE:3 * PLANE].max()) > 1e-6)
