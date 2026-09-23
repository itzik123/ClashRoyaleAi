"""Step a perceived board forward using the engine's own dynamics.

Two questions, one mechanism:

  fidelity   The agent trains in this engine and plays the real game, so any
             interaction the engine models differently is a lesson learned
             wrong. Stepping a real board forward and comparing with what
             happened next measures that per card and per horizon.
  lookahead  End-to-end latency is ~1 s, so the board the policy acts on is
             always a second old. If the engine predicts the next second
             better than the stale board does, that second is recoverable.

The second is only worth asking if the first says the dynamics are close.

Single-line prediction needs no branching, so this is reset, inject, step,
read; branching search over candidates uses `snapshot()` in
python_ai/search/realtime_search.py. It steps with step_self_play, because step
also runs HeuristicOpponent, which would deploy cards the real opponent never
played. A card index outside [0, 4) is a pure "advance time" step.

What the rebuild does not reconstruct. The engine has setters for all of these
(inject's hp and deploy_ticks, set_tower_hp, set_current_tick,
set_elixir_for_team, set_hand_for_team) and this module uses none of them yet:

  unit HP      injected units spawn at full health;
  deploy       every injected unit gets a fresh deploy second;
  tower HP     full after reset;
  clock        zero;
  elixir       the starting value;
  the hand     whatever reset() deals, so two forecasts of one board differ
               in the hand scalars (never in the spatial ones).
  removal      entities cannot be deleted, so each call rebuilds from scratch.

Consumers (tools/sim_fidelity.py, tools/measure_decoupling.py) therefore score
unit occupancy with towers excluded, which none of these gaps touch and which
is what the lookahead question is about. A whole-observation distance is
reported too but folds in the HP and tower resets. Anyone feeding a forecast to
the policy must overwrite the hand from perception first: `affordability_mask`
reads those scalars. set_hand_for_team returns a bool and can refuse; check it.

Stepping 0.5 s then 1.0 s is bit-identical to stepping 1.5 s: combat is
deterministic and a no-op forecast never touches the engine's one RNG.

Injected entities appear only after one update, so the shortest horizon is one
tick (0.1 s) and every horizon includes 100 ms of engine dynamics.
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
# ...and the repo root, so `python_ai.*` resolves. The python_ai/ entry stays:
# clash_royale_env is an unpackaged .pyd inside it.
if str(_PYTHON_AI.parent) not in sys.path:
    sys.path.insert(0, str(_PYTHON_AI.parent))

# Any index outside [0, 4) skips stepSelfPlay's play branch (`cardIndex0 >= 0
# && cardIndex0 < 4`).
NO_OP_CARD = 9

# One tick is the floor: an injected entity does not exist until an update has
# run.
MIN_HORIZON_S = 1.0 / TICKS_PER_SECOND


def _import_engine():
    """The engine bindings, preferring a fresher build over python_ai/'s copy.

    The post-build copy into python_ai/ fails (MSB3073) while a training
    process has the module mapped, so during a live run build_python/Release/
    holds the current binary and python_ai/ a stale one. This prefers the build
    output rather than overwriting python_ai/, which could disturb the training
    run.
    """
    import engine  # noqa: PLC0415
    return engine.load()


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
    """Rebuilds a perceived board in the engine and steps it forward. Holds one
    env across calls; `reset()` (0.135 ms) is the only way to clear the board.
    """

    def __init__(self, deck, engine=None):
        self._engine = engine if engine is not None else _import_engine()
        self._deck = [int(c) for c in deck]
        self._env = self._engine.ClashRoyaleEnv(self._deck, self._deck)
        self._bodies: dict[int, int] = {}

    def bodies_per_card(self, card_id) -> int:
        """How many entities one injection of this card creates.

        `inject` spawns a card; perception detects bodies. Injecting once per
        body multiplies every swarm by its own body count. Measured from the
        engine rather than tabulated (`get_card_info` does not report it), and
        counted through CH_COUNT rather than occupied cells, since a swarm's
        bodies share cells.
        """
        card_id = int(card_id)
        if card_id not in self._bodies:
            # Resets first, so a count can never be contaminated by a board
            # under construction.
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
        """Predict `game_state` forward to each horizon. Horizons are stepped
        cumulatively on one trajectory: the engine is deterministic, so this
        equals re-simulating each and costs a fraction.
        """
        wanted = sorted({max(float(h), MIN_HORIZON_S) for h in horizons_s})
        if not wanted:
            return []

        # Resolve every body count before the reset below: bodies_per_card
        # resets the env itself.
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
            # One injection per card, not per detected body, or the rebuilt
            # board carries three times the swarm.
            per_card = bodies[card_id]
            count = max(1, round(len(members) / per_card))
            # Spread injection points across the detected bodies, so a swarm
            # strung along a lane is rebuilt along it.
            step = max(1, len(members) // count)
            for i in range(count):
                unit = members[min(i * step, len(members) - 1)]
                # Cell centre. The engine truncates, so +0.5 lands in the named
                # tile rather than on an edge where float error could tip it
                # over.
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
        """The engine's own tiles/second for one card, measured by injecting into
        open ground: `get_card_info` does not expose speed, and CH_SPEED's
        divisor would be a copied constant.

        Measured over a long baseline, since cell quantisation adds about +/-1
        tile of error regardless of duration. Starts mid-board, clear of every
        tower, so a slow unit does not merge into a tower cell and abort early.
        `max_ticks` is generous for slow units.
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
            # Centroid of the card's bodies, so swarms are measurable.
            cell = _unit_centroid(
                np.asarray(self._env.get_observation_for_team(0), np.float32))
            if cell is None:                     # died, or walked into a tower
                break
            latest = cell
            moved = float(np.hypot(latest[0] - first[0], latest[1] - first[1]))
        if moved <= 0.0:
            # A building does not move: a real answer, not a failure.
            return 0.0
        return moved / ticks_to_seconds(stepped - 1)


# --- reading boards back out ---
# Layout knowledge is imported from python_ai/models/perception_encoder.py,
# never restated. Channels 0-3 are team 0's four type classes, 4-7 team 1's.

def _encoder():
    from python_ai.models import perception_encoder  # noqa: PLC0415
    return perception_encoder


def tower_cells() -> set[tuple[int, int]]:
    """The six tower cells, excluded from any agreement score: towers never move,
    so leaving them in inflates agreement by a constant that dominates on a
    quiet board.
    """
    enc = _encoder()
    width = enc.BOARD_WIDTH
    return {(int(i) % width, int(i) // width)
            for cells in enc._tower_cells().values()
            for i in (idx % enc.PLANE for idx in cells.indices)}


def occupancy(observation, team) -> set[tuple[int, int]]:
    """Cells holding at least one unit of `team`, towers included. HP-independent,
    so it survives the reconstruction gaps in the module docstring.
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
    """Mean position of team 0's non-tower cells, or None if there are none. A
    centroid, so swarms are measurable.
    """
    cells = occupancy(observation, 0) - tower_cells()
    if not cells:
        return None
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    return float(np.mean(xs)), float(np.mean(ys))


def agreement(predicted, actual) -> float:
    """Intersection over union of two occupancy sets.

    Association-free: perception gives units no identity (`UnitObservation` has
    no track id), so a metric saying which unit moved where would report the
    tracker's errors as the engine's. Empty vs empty is 1.0.
    """
    if not predicted and not actual:
        return 1.0
    union = predicted | actual
    return len(predicted & actual) / len(union) if union else 1.0
