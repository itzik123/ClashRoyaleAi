"""`perception.contracts.GameState` -> the observation vector the policy reads.

Lives on the training side because the observation layout does; perception
emits a GameState and stops.

Everything the engine exposes is read from the binding. The per-card attribute
rows (type, flying, targets-air, damage, range, speed, max HP) are not exposed,
so they are recovered at import by injecting each card on an empty board and
diffing the observation, which cannot drift from the encoding the policy
consumes. The only literals are four unbound normalisers.

`GameState.opp_elixir_spent` is None (there is no opponent elixir bar, and
inferring it from units over-counts); writing 0.0 there measured as costing the
policy nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import clash_royale_env as engine

_E = engine.ClashRoyaleEnv

NUM_CHANNELS = _E.NUM_CHANNELS
BOARD_WIDTH = _E.BOARD_WIDTH
BOARD_HEIGHT = _E.BOARD_HEIGHT
NUM_EXTRA_SCALARS = _E.NUM_EXTRA_SCALARS
NUM_CARD_IDS = _E.NUM_CARD_IDS
HAND_SIZE = _E.HAND_SIZE
MAX_TROOP_HP = _E.MAX_TROOP_HP
MAX_BUILDING_HP = _E.MAX_BUILDING_HP
MAX_MATCH_ELIXIR = _E.MAX_MATCH_ELIXIR
#: Module level on the engine: the phase schedule belongs to the match clock,
#: not an env instance.
MAX_ELIXIR_MULTIPLIER = engine.MAX_ELIXIR_MULTIPLIER
#: The engine's tick rate (perception/timebase.py derives it); restated because
#: this module may not import perception/.
TICKS_PER_SECOND = 10.0

CH_COUNT = _E.CH_COUNT
CH_FLYING = _E.CH_FLYING
CH_ANTIAIR = _E.CH_ANTIAIR
CH_DPS = _E.CH_DPS
CH_RANGE = _E.CH_RANGE
CH_SPEED = _E.CH_SPEED

# Unbound normalisers (ClashEnv.h:110-113): the only values here that can
# silently disagree with the engine.
MAX_UNIT_DPS = 60.0
MAX_ATTACK_RANGE = 12.0
MAX_UNIT_SPEED = 1.5
MAX_CELL_UNITS = 5.0

PLANE = BOARD_WIDTH * BOARD_HEIGHT
SPATIAL_SIZE = NUM_CHANNELS * PLANE
# Plus the opponent card-cycle blocks after the extra scalars.
CYCLE_BLOCK_SIZE = _E.CYCLE_BLOCK_SIZE
SCALAR_SIZE = (1 + HAND_SIZE + HAND_SIZE * NUM_CARD_IDS + NUM_EXTRA_SCALARS
               + CYCLE_BLOCK_SIZE)
OBSERVATION_SIZE = SPATIAL_SIZE + SCALAR_SIZE

# See the module docstring.
OPP_SPEND_WHEN_UNMEASURED = 0.0

# The max_ticks the policy is trained with (gym_wrapper's default). The time
# scalar must use it, not the real game's 1800-tick regular time, or the
# deployed net's clock runs 2x fast. A constructor argument, not an engine
# constant, hence a literal.
TRAINING_MAX_TICKS = 3600.0

# Any legal deck works; it only stands an env up for probing.
_PROBE_DECK = [10, 1, 41, 25, 7, 2, 6, 5]
_PROBE_CELL = (9, 8)          # x, y: middle of our own half, away from towers


@dataclass(frozen=True)
class CardAttributes:
    """One card's observation footprint, recovered from the engine."""

    card_id: int
    name: str
    type_offset: int
    """0 melee, 1 ranged, 2 building-targeter (tank), 3 building.
    ClashEnv.h:203-210."""

    max_hp: float
    """Absolute, this card's own. GameState reports hp as a fraction of THIS,
    while the observation wants a fraction of MAX_TROOP_HP/MAX_BUILDING_HP, so
    both are needed to convert."""

    units: int
    """Bodies one placement spawns -- Minions 3, Archers 2, Knight 1. Drives
    CH_COUNT, the one channel where a Skeleton Army stops looking like one
    skeleton."""

    flying: float
    antiair: float
    dps: float
    attack_range: float
    speed: float
    """All already NORMALISED, exactly as the engine writes them, so the
    encoder never re-divides and cannot disagree about the divisor."""

    is_building: bool

    @property
    def hp_divisor(self) -> float:
        return MAX_BUILDING_HP if self.is_building else MAX_TROOP_HP


def _probe(card_id: int | None):
    env = _E(_PROBE_DECK, _PROBE_DECK)
    env.reset()
    if card_id is not None:
        env.inject(card_id, float(_PROBE_CELL[0]), float(_PROBE_CELL[1]), 0)
    # One tick: the entity exists and has not moved.
    env.step(HAND_SIZE, 0.0, 0.0, 1)
    return np.asarray(env.get_observation_for_team(0), np.float32)


def _index(channel: int, y: int, x: int) -> int:
    return channel * PLANE + y * BOARD_WIDTH + x


def build_card_table(card_ids=None) -> dict[int, CardAttributes]:
    """Recover every card's attribute row by injection and difference.

    Spells and untargetable cards (Royal Ghost, Suspicious Bush) produce no
    board entity and are skipped.
    """
    base = _probe(None)
    if card_ids is None:
        card_ids = engine.get_all_card_ids()

    table: dict[int, CardAttributes] = {}
    for card_id in card_ids:
        info = engine.get_card_info(card_id)
        if info.get("is_spell"):
            continue
        after = _probe(card_id)
        delta = after - base
        touched = np.flatnonzero(np.abs(delta) > 1e-6)
        spatial = [int(i) for i in touched if i < SPATIAL_SIZE]
        if not spatial:
            continue          # no board presence, e.g. a stealth card

        cells = {(i % PLANE) // BOARD_WIDTH * BOARD_WIDTH + (i % PLANE) % BOARD_WIDTH
                 for i in spatial}
        type_offset = next((i // PLANE for i in spatial if i // PLANE < 4), 0)
        hp_cells = [delta[i] for i in spatial if i // PLANE == type_offset]
        is_building = bool(info.get("is_building"))
        divisor = MAX_BUILDING_HP if is_building else MAX_TROOP_HP

        def channel_max(channel):
            vals = [delta[i] for i in spatial if i // PLANE == channel]
            return float(max(vals)) if vals else 0.0

        # CH_COUNT adds 1/MAX_CELL_UNITS per body, so summing it recovers the
        # body count, even across cells.
        count_total = sum(delta[i] for i in spatial if i // PLANE == CH_COUNT)

        table[card_id] = CardAttributes(
            card_id=card_id,
            name=info["name"],
            type_offset=int(type_offset),
            max_hp=float(max(hp_cells) * divisor) if hp_cells else 0.0,
            units=int(round(count_total * MAX_CELL_UNITS)),
            flying=channel_max(CH_FLYING),
            antiair=channel_max(CH_ANTIAIR),
            dps=channel_max(CH_DPS),
            attack_range=channel_max(CH_RANGE),
            speed=channel_max(CH_SPEED),
            is_building=is_building,
        )
    return table


_TABLE: dict[int, CardAttributes] | None = None


def card_table() -> dict[int, CardAttributes]:
    """Generated once per process."""
    global _TABLE
    if _TABLE is None:
        _TABLE = build_card_table()
    return _TABLE


def base_spatial() -> np.ndarray:
    """The engine's own fresh-board spatial planes: river plus six towers.

    Taken from the engine rather than rebuilt, so tower or river moves
    propagate. Towers occupy the building channels plus the count and attribute
    channels.
    """
    return _probe(None)[:SPATIAL_SIZE].copy()


@dataclass(frozen=True)
class TowerCells:
    """Where one tower lives in the observation, and what it reads at full HP."""

    indices: tuple[int, ...]
    full_hp_channel_value: float
    """The building channel's value for an undamaged tower: hp / MAX_BUILDING_HP.
    A Princess reads 0.632 (2534/4008), a King 1.0 -- so a GameState fraction,
    which is relative to the tower's OWN maximum, has to be multiplied by this
    rather than written straight in."""


def _tower_cells() -> dict[str, TowerCells]:
    """Locate the six towers in the engine's fresh observation by geometry (ally
    low rows, enemy high, King on the centre column), not by a coordinate
    table.
    """
    base = base_spatial()
    found: dict[str, TowerCells] = {}
    for ally, channel in ((True, 3), (False, 7)):
        cells = [i for i in range(channel * PLANE, (channel + 1) * PLANE)
                 if base[i] > 1e-6]
        by_col: dict[int, list[int]] = {}
        for i in cells:
            by_col.setdefault((i % PLANE) % BOARD_WIDTH, []).append(i)
        centre = BOARD_WIDTH / 2.0
        king_col = min(by_col, key=lambda c: abs(c + 0.5 - centre))
        for col, idxs in by_col.items():
            if col == king_col:
                key = "own_king" if ally else "opp_king"
            elif col < centre:
                key = "own_princess_left" if ally else "opp_princess_left"
            else:
                key = "own_princess_right" if ally else "opp_princess_right"
            found[key] = TowerCells(tuple(idxs), float(base[idxs[0]]))
    return found


_BASE: np.ndarray | None = None
_TOWERS: dict[str, TowerCells] | None = None


def encode(state, *, card_table_=None) -> np.ndarray:
    """A GameState -> one observation vector, from our side (team 0).

    Mirrors `ClashEnv::extractObservationForTeam(0)`: the HP channels assign,
    CH_COUNT accumulates, every other attribute takes the max over units in a
    cell.

    The opponent card-cycle blocks are left zero ("nothing shown yet"):
    GameState carries no play history. The live mirror should instead call
    `ClashEnv.note_played_card(1, card_id)` per opponent play and read the
    observation off the engine.
    """
    global _BASE, _TOWERS
    table = card_table_ if card_table_ is not None else card_table()
    if _BASE is None:
        _BASE = base_spatial()
        _TOWERS = _tower_cells()

    obs = np.zeros(OBSERVATION_SIZE, np.float32)
    # River and towers from the engine's fresh board, each tower scaled to its
    # observed health; a destroyed tower is zeroed.
    obs[:SPATIAL_SIZE] = _BASE
    towers = {
        "own_king": state.own_king,
        "own_princess_left": state.own_princess_left,
        "own_princess_right": state.own_princess_right,
        "opp_king": state.opp_king,
        "opp_princess_left": state.opp_princess_left,
        "opp_princess_right": state.opp_princess_right,
    }
    for key, cells in (_TOWERS or {}).items():
        tower = towers[key]
        scale = 0.0 if tower.destroyed else float(tower.hp_fraction)
        for i in cells.indices:
            obs[i] = _BASE[i] * scale

    for unit in state.units:
        attrs = table.get(unit.card_sim_id)
        if attrs is None:
            continue          # unmapped or spell
        x, y = unit.tile_x, unit.tile_y
        if not (0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT):
            continue
        ally = unit.team == 0
        side = 0 if ally else 1

        # GameState reports HP as a fraction of the card's own maximum; the
        # observation wants a fraction of the global troop/building maximum.
        absolute = unit.hp_fraction * attrs.max_hp
        obs[_index((0 if ally else 4) + attrs.type_offset, y, x)] = min(
            absolute / attrs.hp_divisor, 1.0)

        idx = _index(CH_COUNT + side, y, x)
        obs[idx] = min(obs[idx] + 1.0 / MAX_CELL_UNITS, 1.0)

        for channel, value in ((CH_FLYING, attrs.flying),
                               (CH_ANTIAIR, attrs.antiair),
                               (CH_DPS, attrs.dps),
                               (CH_RANGE, attrs.attack_range),
                               (CH_SPEED, attrs.speed)):
            j = _index(channel + side, y, x)
            obs[j] = max(obs[j], value)

    s = SPATIAL_SIZE
    obs[s] = min(state.my_elixir / 10.0, 1.0)
    for slot, card_id in enumerate(state.my_hand[:HAND_SIZE]):
        if card_id is None or card_id < 0:
            continue
        info = engine.get_card_info(card_id)
        obs[s + 1 + slot] = info["cost"] / 10.0
        onehot = s + 1 + HAND_SIZE + slot * NUM_CARD_IDS
        if 0 <= card_id < NUM_CARD_IDS:
            obs[onehot + card_id] = 1.0

    e = s + 1 + HAND_SIZE + HAND_SIZE * NUM_CARD_IDS
    # Normalised by the training env's max_ticks, which the net learned the
    # scalar against, not the real game's regular time.
    obs[e + 0] = min(state.seconds_elapsed * TICKS_PER_SECOND / TRAINING_MAX_TICKS, 1.0)
    obs[e + 1] = min(state.my_elixir_spent / MAX_MATCH_ELIXIR, 1.0)
    opp_spend = (OPP_SPEND_WHEN_UNMEASURED if state.opp_elixir_spent is None
                 else min(state.opp_elixir_spent / MAX_MATCH_ELIXIR, 1.0))
    obs[e + 2] = opp_spend
    # Tower scalars are hp / MAX_BUILDING_HP, so a full Princess reads 0.632
    # and only a King 1.0; the GameState fraction is scaled by what a full
    # tower reads.
    for offset, key in enumerate(("own_king", "own_princess_left",
                                  "own_princess_right", "opp_king",
                                  "opp_princess_left", "opp_princess_right")):
        tower = towers[key]
        full = (_TOWERS or {}).get(key)
        at_full = full.full_hp_channel_value if full else 1.0
        obs[e + 3 + offset] = (0.0 if tower.destroyed
                               else float(tower.hp_fraction) * at_full)

    # Elixir phase, from the engine's schedule. It depends on absolute elapsed
    # time, so unlike the time fraction it is not rescaled by
    # TRAINING_MAX_TICKS. The simulator never uses this file, so a scalar
    # missing here would be silently zero only in deployment.
    obs[e + 9] = (engine.elixir_multiplier_at_tick(
                      int(state.seconds_elapsed * TICKS_PER_SECOND))
                  / MAX_ELIXIR_MULTIPLIER)
    return obs
