"""Step a perceived board forward using the engine's own dynamics.

WHY THIS EXISTS -- TWO QUESTIONS, ONE MECHANISM
-----------------------------------------------
  fidelity   The agent trains against this engine and then plays the real
             game. Any interaction the engine models differently is a lesson
             it learns wrong, and no amount of training fixes a wrong lesson.
             Stepping a REAL board forward and comparing against what really
             happened next measures that directly -- per card, per horizon --
             instead of by reading the constants and hoping.

  lookahead  End-to-end latency is ~1 s (perception staleness + actuator +
             deploy), so the board the policy acts on is always about a second
             out of date. If the engine predicts the next second better than
             the stale board does, that second is recoverable.

The second question is only worth asking if the first says the dynamics are
close, which is why this module answers both and the harness reports both.

WHY inject + step, NOT copy
---------------------------
Branching search over candidate actions needs a `GameManager` copy, and
`Board` holds `shared_ptr<Entity>`, so a copy is shallow -- that is open
problem #2 and it is genuinely blocked on a virtual `Entity::clone()`.

Single-line forward prediction needs no branching at all: reset, inject what
was seen, step, read. Every one of those is already bound. The blocker applies
to search, not to prediction, and conflating the two is what kept this
unexplored.

WHY step_self_play, NOT step
----------------------------
`ClashEnv::step` also runs `HeuristicOpponent`, which would deploy cards the
real opponent never played -- the prediction would hallucinate an enemy push
and then be judged against footage that contains none. `stepSelfPlay`
deliberately never calls `opponentTurn()`.

Any card index outside [0, 4) is a no-op in both, which is how a pure
"advance time" step is expressed.

WHAT CANNOT BE RECONSTRUCTED -- read this before trusting a number
------------------------------------------------------------------
`inject` is additive and there is no setter for anything else. Grepping
`set_*|clear|remove` across src/bindings.cpp returns nothing. So:

  unit HP     injected units spawn at FULL health. Perception measures a
              fraction and it cannot be applied.
  tower HP    always full after reset. These are extra scalars 3-8, i.e. the
              six numbers the win condition is defined on.
  elixir      always the starting value.
  match clock always zero.
  removal     an entity that perception no longer sees cannot be deleted;
              the state is rebuilt from scratch each call instead.
  the HAND    `reset()` reshuffles from an unseeded mt19937 that cannot be
              seeded or set, so two forecasts of the SAME board come back
              with different hands. Measured: forecasting one board twice
              differs in 11 floats, every one of them a hand one-hot or a
              hand cost, and in ZERO of the 12,852 spatial floats.

That last one is the trap for anyone who later feeds a forecast straight to
the policy: the board would be a prediction but the hand would be fiction,
and `affordability_mask` is built from those very scalars. The hand must be
overwritten from perception before the vector is used as a policy input.

It is also what makes cumulative stepping safe. Stepping 0.5 s and then
another 1.0 s is bit-identical to stepping 1.5 s across the whole spatial
half of the observation -- the engine's combat is deterministic and a no-op
forecast never touches the one RNG it has.

This is why the harness's primary metric is UNIT OCCUPANCY with towers
excluded. Occupancy is the one thing that survives all five gaps intact, and
it is exactly what the lookahead question is about -- where the units are.
A whole-observation distance is reported too, but it folds in the HP and tower
resets and must not be read as dynamics error.

THE MATERIALISING TICK
----------------------
Injected entities are not on the board until one update has run: measured, a
read at zero ticks returns the six towers and nothing else, and the injected
unit first appears at tick 1. So the shortest horizon this can express is one
tick, 0.1 s, and `horizon_s=0` is not a thing that can be asked for. Every
horizon below therefore already contains 100 ms of engine dynamics, which is
small but is not nothing and is not hidden.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from timebase import TICKS_PER_SECOND, ticks_to_seconds

_PYTHON_AI = Path(__file__).resolve().parent.parent / "python_ai"
if str(_PYTHON_AI) not in sys.path:
    sys.path.insert(0, str(_PYTHON_AI))

# Any index outside [0, 4) skips the play branch in stepSelfPlay -- see
# ClashEnv::stepSelfPlay's `cardIndex0 >= 0 && cardIndex0 < 4`.
NO_OP_CARD = 9

# One tick is the floor: an injected entity does not exist until an update has
# run. See "THE MATERIALISING TICK" above.
MIN_HORIZON_S = 1.0 / TICKS_PER_SECOND


def _import_engine():
    try:
        import clash_royale_env  # noqa: PLC0415
    except ImportError as exc:                              # pragma: no cover
        raise ImportError(
            "could not import clash_royale_env. The .pyd is built for Python "
            "3.11 and lives in python_ai/; run this with "
            "perception/.venv-dml/Scripts/python.exe or py -3.11."
        ) from exc
    return clash_royale_env


@dataclass(frozen=True)
class Forecast:
    """One predicted board, and what was lost building it."""

    horizon_s: float
    """Seconds of engine time stepped. Never below MIN_HORIZON_S."""

    observation: np.ndarray
    """`get_observation_for_team(0)` -- the same vector the policy consumes."""

    injected: int
    unmappable: int
    """Units perception saw but could not be injected, because the detector's
    class has no engine counterpart (`UNKNOWN_CARD_SIM_ID`). Counted rather
    than ignored: a forecast missing a third of the board is not a forecast,
    and the caller cannot tell from the observation alone."""


class SimForecaster:
    """Rebuilds a perceived board in the engine and steps it forward.

    Holds ONE env across calls. `reset()` costs 0.135 ms and is the documented
    way to clear the board, since entities cannot be removed individually.
    """

    def __init__(self, deck, engine=None):
        self._engine = engine if engine is not None else _import_engine()
        self._deck = [int(c) for c in deck]
        self._env = self._engine.ClashRoyaleEnv(self._deck, self._deck)
        self._bodies: dict[int, int] = {}

    def bodies_per_card(self, card_id) -> int:
        """How many entities ONE injection of this card creates.

        `inject` spawns a CARD; perception detects BODIES. Injecting once per
        detected body therefore multiplies every swarm card by its own body
        count -- measured on real footage, four detected spear goblins became
        twelve, and the reconstructed board had 12 occupied cells against
        perception's 7 before a single tick was stepped.

        Measured from the engine, not tabulated: `get_card_info` does not
        report it, and a hand-written table would be a second copy of an
        engine fact that changes whenever a card is rebalanced.

        Counted through the CH_COUNT channel rather than by occupied cells,
        because bodies of a swarm routinely share one cell and counting cells
        would report Minions as 1.
        """
        card_id = int(card_id)
        if card_id not in self._bodies:
            # Self-contained: resets first, so it can never be called into the
            # middle of a forecast and silently return a count contaminated by
            # whatever else was on the board.
            self._env.reset()
            self._env.step_self_play(NO_OP_CARD, 0.0, 0.0,
                                     NO_OP_CARD, 0.0, 0.0, 1)
            before = self._count_bodies()
            self._env.inject(card_id, 9.5, 12.5, 0)
            self._env.step_self_play(NO_OP_CARD, 0.0, 0.0,
                                     NO_OP_CARD, 0.0, 0.0, 1)
            self._bodies[card_id] = max(1, self._count_bodies() - before)
        return self._bodies[card_id]

    def _count_bodies(self) -> int:
        enc = _encoder()
        obs = np.asarray(self._env.get_observation_for_team(0), np.float32)
        plane = enc.PLANE
        block = obs[enc.CH_COUNT * plane:(enc.CH_COUNT + 1) * plane]
        return int(round(float(block.sum()) * enc.MAX_CELL_UNITS))

    def forecast(self, game_state, horizons_s) -> list[Forecast]:
        """Predict `game_state` forward to each horizon.

        Horizons are stepped CUMULATIVELY on one trajectory rather than
        re-simulated per horizon -- the engine is deterministic, so stepping
        1 s then another 0.5 s is bit-identical to stepping 1.5 s, and doing it
        once costs a fraction as much.
        """
        wanted = sorted({max(float(h), MIN_HORIZON_S) for h in horizons_s})
        if not wanted:
            return []

        # Group first, and resolve every body count BEFORE the reset below:
        # bodies_per_card resets the env itself, so calling it mid-injection
        # would wipe the board being built.
        groups: dict[tuple[int, int], list] = {}
        unmappable = 0
        for unit in game_state.units:
            if int(unit.card_sim_id) < 0:
                unmappable += 1
                continue
            groups.setdefault((int(unit.card_sim_id), int(unit.team)),
                              []).append(unit)
        bodies = {cid: self.bodies_per_card(cid) for cid, _team in groups}

        self._env.reset()
        injected = 0
        for (card_id, team), members in groups.items():
            # One injection per CARD, not per detected body. A card that
            # spawns three goblins must be injected once for every three
            # bodies seen, or the reconstructed board carries three times the
            # swarm the real one does.
            per_card = bodies[card_id]
            count = max(1, round(len(members) / per_card))
            # Spread the injection points across the detected bodies rather
            # than stacking them all on the first, so a swarm strung out along
            # a lane is rebuilt along that lane.
            step = max(1, len(members) // count)
            for i in range(count):
                unit = members[min(i * step, len(members) - 1)]
                # Cell CENTRE. The engine truncates with static_cast<int>, so
                # +0.5 lands in the tile perception named rather than on its
                # edge, where float error could tip it into the neighbour.
                self._env.inject(card_id,
                                 float(unit.tile_x) + 0.5,
                                 float(unit.tile_y) + 0.5,
                                 team)
                injected += 1

        out: list[Forecast] = []
        stepped = 0
        for horizon in wanted:
            ticks = max(1, int(round(horizon * TICKS_PER_SECOND)))
            if ticks > stepped:
                self._env.step_self_play(NO_OP_CARD, 0.0, 0.0,
                                         NO_OP_CARD, 0.0, 0.0,
                                         ticks - stepped)
                stepped = ticks
            out.append(Forecast(
                horizon_s=ticks_to_seconds(stepped),
                observation=np.asarray(
                    self._env.get_observation_for_team(0), np.float32),
                injected=injected,
                unmappable=unmappable))
        return out

    def measure_speed(self, card_id, min_tiles=6.0, max_ticks=200) -> float:
        """The engine's own tiles/second for one card, measured not read.

        `get_card_info` does not expose speed and the observation's CH_SPEED
        is normalised by a divisor this module would have to copy. Injecting
        into open ground and measuring displacement asks the engine directly,
        which is the rule this project already follows for every other engine
        constant.

        Measured over a LONG baseline, not a fixed short one. The observation
        is cell-quantised, so displacement carries about +/-1 tile of error
        regardless of duration: over a 3-tile walk that is 33%, over a 10-tile
        walk it is 10%. A first version used a flat 10 ticks and reported the
        Giant at 3.61 tiles/s against a true 3.0 -- the error was the
        quantisation, not the engine.

        Starts MID-BOARD, clear of every tower. An earlier version started at
        (5.5, 4.5) and worked only while troops were fast: once movement was
        corrected to real-game speed the unit no longer cleared its own left
        Princess Tower at (4, 6) within the tick cap, merged into a tower cell,
        and the measurement aborted early -- reporting the Giant at 0.17
        tiles/s against a true ~0.58. The bug was latent the whole time and
        surfaced only when the thing being measured changed.

        `max_ticks` is generous for the same reason: at real-game speed a unit
        needs roughly five times as long to cover the same ground.
        """
        self._env.reset()
        self._env.inject(int(card_id), 9.5, 8.5, 0)
        self._env.step_self_play(NO_OP_CARD, 0.0, 0.0, NO_OP_CARD, 0.0, 0.0, 1)
        first = _unit_centroid(
            np.asarray(self._env.get_observation_for_team(0), np.float32))
        if first is None:
            return float("nan")

        stepped, moved, latest = 1, 0.0, first
        while stepped < max_ticks and moved < min_tiles:
            self._env.step_self_play(NO_OP_CARD, 0.0, 0.0,
                                     NO_OP_CARD, 0.0, 0.0, 2)
            stepped += 2
            # CENTROID of the card's bodies, not "the one cell". A first
            # version required exactly one occupied cell and so returned nan
            # for every swarm -- Archers, Minions, Spear Goblins and Skeletons
            # all spawn more than one body and never satisfy it.
            cell = _unit_centroid(
                np.asarray(self._env.get_observation_for_team(0), np.float32))
            if cell is None:                     # died, or walked into a tower
                break
            latest = cell
            moved = float(np.hypot(latest[0] - first[0], latest[1] - first[1]))
        if moved <= 0.0:
            # A building does not move. That is a real answer, not a failure.
            return 0.0
        return moved / ticks_to_seconds(stepped - 1)


# --- reading boards back out ------------------------------------------------
#
# Layout knowledge lives in python_ai/perception_encoder.py and is imported,
# never restated. Channels 0-3 are team 0's four type classes and 4-7 team 1's;
# CLAUDE.md's rule about not keeping a second copy of an engine constant covers
# the layout just as much as the numbers.

def _encoder():
    import perception_encoder  # noqa: PLC0415
    return perception_encoder


def tower_cells() -> set[tuple[int, int]]:
    """The six tower cells, which must be excluded from any agreement score.

    Towers never move, so leaving them in guarantees six matching cells in
    every comparison and inflates agreement by a constant that depends only on
    how many units happen to be on the board. On a quiet board that is most of
    the score.
    """
    enc = _encoder()
    width = enc.BOARD_WIDTH
    return {(int(i) % width, int(i) // width)
            for cells in enc._tower_cells().values()
            for i in (idx % enc.PLANE for idx in cells.indices)}


def occupancy(observation, team) -> set[tuple[int, int]]:
    """Cells holding at least one unit of `team`, towers included.

    HP-independent on purpose. It is the one reading that survives the
    reconstruction gaps listed in the module docstring, so it is the only
    honest basis for "did the units end up where the engine said".
    """
    enc = _encoder()
    plane, width = enc.PLANE, enc.BOARD_WIDTH
    obs = np.asarray(observation, np.float32)
    base = 0 if int(team) == 0 else 4
    cells: set[tuple[int, int]] = set()
    for channel in range(base, base + 4):
        block = obs[channel * plane:(channel + 1) * plane]
        for i in np.nonzero(block > 1e-6)[0]:
            cells.add((int(i) % width, int(i) // width))
    return cells


def _unit_centroid(observation):
    """Mean position of team 0's non-tower cells, or None if there are none.

    A centroid rather than a single cell so swarms are measurable: Archers,
    Minions, Spear Goblins and Skeletons all spawn several bodies, and a
    "exactly one cell" rule reports nan for every one of them.
    """
    cells = occupancy(observation, 0) - tower_cells()
    if not cells:
        return None
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    return float(np.mean(xs)), float(np.mean(ys))


def agreement(predicted, actual) -> float:
    """Intersection over union of two occupancy sets.

    Association-free by construction. Perception gives units no identity --
    `UnitObservation` has no track id -- so any metric needing to say WHICH
    unit moved where would first have to solve data association, and would
    then be reporting the tracker's errors as the engine's.

    Empty vs empty is 1.0: two boards that agree nothing is there agree.
    """
    if not predicted and not actual:
        return 1.0
    union = predicted | actual
    return len(predicted & actual) / len(union) if union else 1.0
