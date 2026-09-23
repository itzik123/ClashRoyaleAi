"""Tests for forecast.py: stepping a perceived board with engine physics.

Two pin real mistakes: injected entities do not exist until one update has run,
and `inject` spawns a card while perception detects bodies.
"""
from __future__ import annotations

import pytest

from contracts import (
    UNKNOWN_CARD_SIM_ID,
    GameState,
    Phase,
    TowerObservation,
    UnitObservation,
)

# No module-level importorskip for the engine: conftest's `engine` fixture runs
# after collection, so it would skip the module even when the .pyd is fine (see
# test_perception_encoder.py).

GIANT, ARCHERS, MINIONS, MUSKETEER = 2, 1, 41, 6
DECK = [10, 1, 41, 25, 7, 2, 6, 5]
FULL = TowerObservation(hp_fraction=1.0, hp_measured=True)


@pytest.fixture(scope="module")
def forecaster(engine):
    from forecast import SimForecaster
    return SimForecaster(DECK, engine=engine)


def board(*units) -> GameState:
    return GameState(
        units=tuple(units), my_elixir=5.0, my_hand=(10, 1, 41, 25),
        seconds_elapsed=30.0, phase=Phase.SINGLE,
        own_king=FULL, own_princess_left=FULL, own_princess_right=FULL,
        opp_king=FULL, opp_princess_left=FULL, opp_princess_right=FULL)


def unit(card_id, x, y, team=0, name="") -> UnitObservation:
    return UnitObservation(card_sim_id=card_id, unit_name=name, team=team,
                           tile_x=x, tile_y=y, hp_fraction=1.0,
                           hp_measured=True)


def cells(observation, team):
    from forecast import occupancy, tower_cells
    return occupancy(observation, team) - tower_cells()


# --- the materialising tick ---

def test_a_forecast_never_returns_an_empty_board(forecaster):
    """Zero ticks reads back the six towers and nothing else: an injected entity
    is not on the board until an update has run.
    """
    from forecast import MIN_HORIZON_S
    out = forecaster.forecast(board(unit(GIANT, 9, 8)), [0.0])
    assert out[0].horizon_s == pytest.approx(MIN_HORIZON_S)
    assert cells(out[0].observation, 0), "injected unit did not materialise"


def test_horizons_below_the_floor_clamp_rather_than_vanish(forecaster):
    from forecast import MIN_HORIZON_S
    out = forecaster.forecast(board(unit(GIANT, 9, 8)), [0.0, 0.01])
    assert all(f.horizon_s >= MIN_HORIZON_S for f in out)


# --- bodies vs cards ---

def test_bodies_per_card_is_measured_not_assumed(forecaster):
    """Swarm counts come from the engine, not a table that goes stale on a
    rebalance.
    """
    assert forecaster.bodies_per_card(GIANT) == 1
    assert forecaster.bodies_per_card(ARCHERS) == 2
    assert forecaster.bodies_per_card(MINIONS) == 3


def test_a_swarm_is_not_multiplied_by_its_own_body_count(forecaster):
    """Three detected minions are one card; injecting once per body would put nine
    on the board. Counted through CH_COUNT, since swarm bodies share cells.
    """
    import numpy as np
    from python_ai.models import perception_encoder as enc

    seen = [unit(MINIONS, 9, 8, name="minion"),
            unit(MINIONS, 9, 8, name="minion"),
            unit(MINIONS, 10, 8, name="minion")]
    out = forecaster.forecast(board(*seen), [0.1])[0]
    plane = enc.PLANE
    block = np.asarray(out.observation, np.float32)[
        enc.CH_COUNT * plane:(enc.CH_COUNT + 1) * plane]
    bodies = round(float(block.sum()) * enc.MAX_CELL_UNITS)
    # Towers are in the count plane too; the three minions are the excess.
    assert out.injected == 1, "three bodies of one card injected as three cards"
    assert bodies <= 3 + 3, f"swarm multiplied: {bodies} bodies on the board"


def test_a_swarm_is_rebuilt_along_the_lane_it_was_seen_in(forecaster):
    """Six archers strung down a lane are three cards; stacking all three on the
    first body would rebuild a blob where the real board has a line.
    """
    seen = [unit(ARCHERS, 5, 6 + i, name="archer") for i in range(6)]
    out = forecaster.forecast(board(*seen), [0.1])[0]
    assert out.injected == 3
    ys = [y for _x, y in cells(out.observation, 0)]
    assert max(ys) - min(ys) >= 2, "rebuilt as a blob, not a lane"


# --- what cannot be reconstructed is reported, not hidden ---

def test_unmappable_units_are_counted_rather_than_dropped(forecaster):
    """A forecast missing part of the board is not a forecast, and the caller
    cannot tell from the observation alone.
    """
    out = forecaster.forecast(
        board(unit(GIANT, 9, 8), unit(UNKNOWN_CARD_SIM_ID, 9, 10)), [0.5])[0]
    assert out.injected == 1
    assert out.unmappable == 1


# --- the opponent must not be simulated ---

def test_the_heuristic_opponent_never_plays_during_a_forecast(forecaster):
    """`step` also runs HeuristicOpponent, which would invent an enemy push the
    footage does not contain. Two seconds is ample for the heuristic to act if
    it were running.
    """
    out = forecaster.forecast(board(unit(GIANT, 9, 8)), [2.0])[0]
    assert cells(out.observation, 1) == set(), "an enemy appeared from nowhere"


# --- dynamics ---

def test_units_actually_move(forecaster):
    """A forecast horizon must actually advance the world.

    `cells()` reads the integer grid, so movement is visible only once a unit
    crosses a cell boundary. A Giant spawned at x=9.0 sits on one and lane
    pathing sends it right (the right bridge is nearer), so x must climb a
    whole tile before this can see anything: it reaches cell (10, 9) at ~4.0 s.
    5.0 s clears the boundary from any start.
    """
    start, later = forecaster.forecast(board(unit(GIANT, 9, 8)), [0.1, 5.0])
    assert cells(start.observation, 0) != cells(later.observation, 0)


def test_stepping_cumulatively_matches_stepping_directly(forecaster):
    """Horizons are stepped on one trajectory, which is sound only because the
    engine is deterministic. Compared over the spatial half: each reset() deals
    a fresh hand, so the scalars differ.
    """
    import numpy as np
    from python_ai.models import perception_encoder as enc

    spatial = enc.PLANE * 21
    together = forecaster.forecast(board(unit(GIANT, 9, 8)), [0.5, 1.5])
    alone = forecaster.forecast(board(unit(GIANT, 9, 8)), [1.5])
    assert np.array_equal(together[-1].observation[:spatial],
                          alone[0].observation[:spatial])


def test_the_forecast_hand_is_fiction(forecaster):
    """A live trap: forecast.py rebuilds with reset(), so the hand is whatever the
    shuffle dealt. A forecast fed to the policy would pair a predicted board
    with an invented hand, and `affordability_mask` reads those scalars. The
    hand must be overwritten from perception first.
    """
    import numpy as np
    from python_ai.models import perception_encoder as enc

    spatial = enc.PLANE * 21
    seen = board(unit(GIANT, 9, 8))
    hands = {tuple(np.asarray(forecaster.forecast(seen, [0.5])[0]
                              .observation[spatial:]).tolist())
             for _ in range(6)}
    assert len(hands) > 1, "the shuffle looks seeded -- re-read this test"


# Giant's engine speed in tiles/second, from include/core/CardStats.h (none of
# these constants is bound):
#
#   SPEED_SLOW = 45 tiles/min * REAL_TILES_PER_MIN_TO_ENGINE (0.011045)
#              = 0.497025 tiles/tick of raw stat
#   * MOVEMENT_SPEED_SCALE (0.2) * 10 ticks/s = 0.994 tiles/s
#
# Updating this line acknowledges that troop speed changed.
GIANT_TILES_PER_SECOND = 0.994


def test_speed_is_measured_over_a_long_baseline(forecaster):
    """The observation is cell-quantised, so a short baseline carries ~+/-1 tile
    regardless of duration. The tolerance is that allowance: the measurement
    reads ~0.92 against the registry's 0.994.
    """
    assert forecaster.measure_speed(GIANT) == pytest.approx(
        GIANT_TILES_PER_SECOND, abs=0.15)


def test_the_engine_moves_at_roughly_real_game_speed(forecaster):
    """A guard on MOVEMENT_SPEED_SCALE. Loose, since the real-game figures are a
    measured bracket; it fails loudly if the scale is dropped and troops move
    five times too fast.
    """
    for card_id in (GIANT, MUSKETEER):
        assert 0.4 < forecaster.measure_speed(card_id) < 2.5


# --- the agreement metric ---

def test_agreement_is_one_for_identical_boards():
    from forecast import agreement
    assert agreement({(1, 2), (3, 4)}, {(1, 2), (3, 4)}) == 1.0


def test_agreement_is_zero_for_disjoint_boards():
    from forecast import agreement
    assert agreement({(1, 2)}, {(5, 6)}) == 0.0


def test_two_empty_boards_agree():
    """Both saying the board is clear is agreement; scoring it 0 would make every
    quiet frame a total failure.
    """
    from forecast import agreement
    assert agreement(set(), set()) == 1.0


def test_towers_are_excluded_from_occupancy_comparisons(forecaster):
    """Towers never move, so leaving them in adds six guaranteed matches to every
    comparison.
    """
    from forecast import occupancy, tower_cells
    out = forecaster.forecast(board(), [0.5])[0]
    assert occupancy(out.observation, 0) & tower_cells()
    assert cells(out.observation, 0) == set()
