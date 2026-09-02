"""`perception.contracts.GameState` -> the observation vector the policy reads.

WHY THIS FILE IS HERE AND NOT IN perception/
--------------------------------------------
The observation layout belongs to the training side. It has already changed
once -- 6253 -> 13606 on 2026-07-29, when NUM_CHANNELS went 9 -> 21 -- and an
encoder living on the sensor side would have broken silently on that day while
looking like a perception bug. Perception emits a GameState and stops; this is
the only place that knows what a float at index 9,731 means.

NOTHING HERE IS HARDCODED THAT THE ENGINE CAN ANSWER
-----------------------------------------------------
Channel indices, board size, scalar counts and the two HP maxima all come from
the binding. `CLAUDE.md` names duplicated engine constants as having gone stale
twice, so the rule here is: bound -> read it; unbound -> hardcode with a comment
naming the header line, the pattern `perception/geometry.py` already uses.

Four normalisers are genuinely unbound (`ClashEnv.h:110-113`) and are the only
literals below.

THE PER-CARD ATTRIBUTE TABLE IS GENERATED, NOT WRITTEN
-------------------------------------------------------
Type class, flying, targets-air, damage, range, speed and maximum HP all exist
in C++ and none are exposed. They do not need to be: injecting one card onto an
empty board and diffing the observation against an untouched env recovers the
card's whole attribute row *out of the exact encoding the policy consumes*,
which is strictly more trustworthy than a hand-transcribed table and cannot
drift from it.

Generated at import rather than committed as JSON, for the same staleness
reason.

WHAT THE SENSOR CANNOT SUPPLY
-----------------------------
`GameState.opp_elixir_spent` is None -- there is no opponent elixir bar, and
inferring spend from units appearing over-counts 2.1x
(`perception/BOT_REQUESTS.md` item 8). Measured on a frozen checkpoint over 900
episodes, writing 0.0 there costs the policy nothing: delta +0.031, 95% CI
[-0.018, +0.080]. `OPP_SPEND_WHEN_UNMEASURED` is that constant, named so it is
visible rather than an anonymous zero, and so changing it is understood to
invalidate that measurement.
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
#: Normaliser for the elixir-phase scalar. Module level on the engine, not on
#: ClashRoyaleEnv, because the phase schedule is a property of the match clock
#: rather than of an env instance -- see bindings.cpp.
MAX_ELIXIR_MULTIPLIER = engine.MAX_ELIXIR_MULTIPLIER
#: The engine's tick rate. Derived in perception/timebase.py from 132 agreeing
#: attackCooldown rows; restated here (with that source named) because this
#: module may not import from perception/, the same fallback TRAINING_MAX_TICKS
#: below uses for a genuinely unbound value.
TICKS_PER_SECOND = 10.0

CH_COUNT = _E.CH_COUNT
CH_FLYING = _E.CH_FLYING
CH_ANTIAIR = _E.CH_ANTIAIR
CH_DPS = _E.CH_DPS
CH_RANGE = _E.CH_RANGE
CH_SPEED = _E.CH_SPEED

# Unbound. ClashEnv.h:110-113. Requesting bindings is UPSTREAM_REQUESTS
# material, not blocking -- but these four are the file's only magic numbers
# and the only thing here that can silently disagree with the engine.
MAX_UNIT_DPS = 60.0
MAX_ATTACK_RANGE = 12.0
MAX_UNIT_SPEED = 1.5
MAX_CELL_UNITS = 5.0

PLANE = BOARD_WIDTH * BOARD_HEIGHT
SPATIAL_SIZE = NUM_CHANNELS * PLANE
# + the opponent card-cycle blocks appended behind the tail (item 24).
CYCLE_BLOCK_SIZE = _E.CYCLE_BLOCK_SIZE
SCALAR_SIZE = (1 + HAND_SIZE + HAND_SIZE * NUM_CARD_IDS + NUM_EXTRA_SCALARS
               + CYCLE_BLOCK_SIZE)
OBSERVATION_SIZE = SPATIAL_SIZE + SCALAR_SIZE

# See the module docstring. Not an anonymous 0.0.
OPP_SPEND_WHEN_UNMEASURED = 0.0

# The `max_ticks` the policy is TRAINED with (gym_wrapper.MicroRoyaleEnv's
# default, and ClashEnv's own constructor default). The time scalar has to be
# normalised by this and not by the real game's 1800-tick regular time, or the
# deployed net reads a clock running at twice the rate it learned. Not an
# engine constant -- it is a constructor argument -- so it lives here as a
# named literal rather than a binding lookup.
TRAINING_MAX_TICKS = 3600.0

# Deck used only to stand an env up for probing; any legal deck works.
_PROBE_DECK = [10, 1, 41, 25, 7, 2, 6, 5]
_PROBE_CELL = (9, 8)          # x, y -- middle of our own half, away from towers


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
    # One tick past the deploy delay, so the entity exists and has not moved.
    env.step(HAND_SIZE, 0.0, 0.0, 1)
    return np.asarray(env.get_observation_for_team(0), np.float32)


def _index(channel: int, y: int, x: int) -> int:
    return channel * PLANE + y * BOARD_WIDTH + x


def build_card_table(card_ids=None) -> dict[int, CardAttributes]:
    """Recover every card's attribute row by injection and difference.

    Spells and anything `isTargetable()` hides (Royal Ghost, Suspicious Bush)
    produce no board entity and are skipped -- they legitimately have no
    spatial footprint, and a caller placing one writes nothing.
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

        # CH_COUNT accumulates 1/MAX_CELL_UNITS per body, so summing it back
        # over every cell recovers the body count -- including for cards whose
        # bodies land in different cells.
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

    Taken from the engine rather than reconstructed. Both parts have moved
    before -- the river on 2026-07-29, the tower x positions on 2026-07-30 --
    and rebuilding them here would be a third copy of coordinates `CLAUDE.md`
    already records as having gone stale twice. Starting from the engine's own
    output means a future move propagates instead of diverging.

    Towers occupy the building channels (3 ally, 7 enemy) plus the count and
    attribute channels, which is 34 floats that a units-only encoder leaves at
    zero -- measured, not assumed: that was the entire round-trip difference.
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
    """Locate the six towers in the engine's own fresh observation.

    Identified by geometry rather than by a coordinate table: ally towers sit
    in the low rows, enemy in the high ones, and the King is the one on the
    centre column. That survives the towers being moved again.
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
    """A GameState -> one observation vector, from OUR side (team 0).

    Mirrors `ClashEnv::extractObservationForTeam(0)`. Two behaviours are easy
    to get wrong and are called out where they happen: the HP channels ASSIGN
    while CH_COUNT ACCUMULATES, and every other attribute takes the MAX over
    units sharing a cell.

    THE OPPONENT CARD-CYCLE BLOCKS ARE LEFT ZERO, and that is a KNOWN STUB
    rather than an oversight (item 24, 2026-08-27). `obs` is allocated at the
    engine's full `OBSERVATION_SIZE`, so the vector is always the right LENGTH
    and the net accepts it; the last `CYCLE_BLOCK_SIZE` floats simply read as
    "the opponent has shown nothing", which is what a fresh match looks like
    and is the one wrong answer that cannot mislead the policy into acting on
    a card it has not seen.

    Filling them needs a `GameState` that carries the opponent's play history,
    which it does not yet. The engine-side route already exists and is the one
    the live mirror should use: call `ClashEnv.note_played_card(1, card_id)`
    each time the placement stream reports an opponent play, and read the
    observation off the mirror rather than building it here. That is why the
    recorder is separate from `inject` -- see `ClashEnv::notePlayedCard`.
    """
    global _BASE, _TOWERS
    table = card_table_ if card_table_ is not None else card_table()
    if _BASE is None:
        _BASE = base_spatial()
        _TOWERS = _tower_cells()

    obs = np.zeros(OBSERVATION_SIZE, np.float32)
    # River and towers come from the engine's own fresh board, then each
    # tower's cells are scaled to its observed health. A destroyed tower is
    # zeroed, matching the engine, where it is simply no longer a live entity.
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
            continue          # unmapped or spell; nothing to write
        x, y = unit.tile_x, unit.tile_y
        if not (0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT):
            continue
        ally = unit.team == 0
        side = 0 if ally else 1

        # HP: GameState reports a fraction of the CARD'S own maximum, the
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
    # TIME. Must use the SAME denominator the policy was trained against, which
    # is the training env's maxTicks (3600), NOT the real game's 1800-tick
    # regular time. The engine writes `currentTick / maxTicks`, so the net has
    # only ever learned to read this scalar as "fraction of a 3600-tick match".
    #
    # This read 1800.0, i.e. the live clock ran 2x fast: at 90 real seconds the
    # net was handed 0.50 where training would have shown it 0.25. That is a
    # train/deploy mismatch on the one input TimeoutRules-style clock
    # management depends on, and it is invisible in every simulator metric
    # because the simulator never uses this file.
    #
    # TRAINING_MAX_TICKS mirrors gym_wrapper's `max_ticks` default. It is not
    # bound (it is a constructor argument, not an engine constant), so it is
    # hardcoded here with this comment naming its source -- the pattern
    # perception/geometry.py uses for genuinely unbound values.
    obs[e + 0] = min(state.seconds_elapsed * TICKS_PER_SECOND / TRAINING_MAX_TICKS, 1.0)
    obs[e + 1] = min(state.my_elixir_spent / MAX_MATCH_ELIXIR, 1.0)
    opp_spend = (OPP_SPEND_WHEN_UNMEASURED if state.opp_elixir_spent is None
                 else min(state.opp_elixir_spent / MAX_MATCH_ELIXIR, 1.0))
    obs[e + 2] = opp_spend
    # Tower scalars are hp / MAX_BUILDING_HP, NOT a fraction of the tower's own
    # maximum -- an undamaged Princess reads 0.632 (2534/4008) and only a King
    # reads 1.0. GameState carries the own-maximum fraction, so it has to be
    # scaled by what a full tower reads. Kings hid this: they ARE 4008, so both
    # conventions agree on them and only the four Princesses disagreed.
    for offset, key in enumerate(("own_king", "own_princess_left",
                                  "own_princess_right", "opp_king",
                                  "opp_princess_left", "opp_princess_right")):
        tower = towers[key]
        full = (_TOWERS or {}).get(key)
        at_full = full.full_hp_channel_value if full else 1.0
        obs[e + 3 + offset] = (0.0 if tower.destroyed
                               else float(tower.hp_fraction) * at_full)

    # ELIXIR PHASE (scalar 9, added 2026-09-02).
    #
    # This line is the whole reason the phase feature is not silently broken in
    # deployment. Nothing in the simulator uses this file, so a scalar that the
    # engine writes and this encoder does not is zero on a real screen and
    # correct in every training metric -- the identical failure mode
    # `note_played_card` exists to prevent for the cycle blocks, and the
    # identical reason the time-fraction scalar above ran 2x fast unnoticed.
    #
    # Derived from the engine's own schedule via the bound free function, so
    # the two boundary ticks and the /3.0 normaliser each have exactly one
    # definition. `seconds_elapsed * TICKS_PER_SECOND` is the same real-clock-to-tick
    # conversion the time fraction above uses (10 ticks = 1 s).
    #
    # NOTE the asymmetry with the time fraction, which is deliberate: that one
    # divides by TRAINING_MAX_TICKS (3600) because the net learned it as "a
    # fraction of a 3600-tick match", whereas the phase is a function of
    # ABSOLUTE elapsed time and means the same thing on a real screen as in the
    # simulator. No max-ticks scaling belongs here, and applying one would put
    # a live match into triple elixir at 1:30.
    obs[e + 9] = (engine.elixir_multiplier_at_tick(
                      int(state.seconds_elapsed * TICKS_PER_SECOND))
                  / MAX_ELIXIR_MULTIPLIER)
    return obs
