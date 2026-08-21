"""Tests for forecast.py -- stepping a perceived board with engine physics.

Most of these pin something that was actually WRONG while this was being
built, which is the only reason worth having a test:

  * injected entities do not exist until one update has run, so a zero-tick
    forecast returned six towers and nothing else;
  * `inject` spawns a CARD while perception detects BODIES, so four detected
    spear goblins became twelve and the rebuilt board had 12 occupied cells
    against perception's 7 before any time passed.
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

# No module-level importorskip for the engine -- see the note in
# test_perception_encoder.py: conftest's `engine` fixture runs after collection,
# so an importorskip here would skip the module even when the .pyd is fine.

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


# --- the materialising tick -------------------------------------------------

def test_a_forecast_never_returns_an_empty_board(forecaster):
    """Zero ticks reads back the six towers and nothing else: an injected
    entity is not on the board until an update has run. A caller asking for
    'now' would silently get a board with no units on it."""
    from forecast import MIN_HORIZON_S
    out = forecaster.forecast(board(unit(GIANT, 9, 8)), [0.0])
    assert out[0].horizon_s == pytest.approx(MIN_HORIZON_S)
    assert cells(out[0].observation, 0), "injected unit did not materialise"


def test_horizons_below_the_floor_clamp_rather_than_vanish(forecaster):
    from forecast import MIN_HORIZON_S
    out = forecaster.forecast(board(unit(GIANT, 9, 8)), [0.0, 0.01])
    assert all(f.horizon_s >= MIN_HORIZON_S for f in out)


# --- bodies vs cards --------------------------------------------------------

def test_bodies_per_card_is_measured_not_assumed(forecaster):
    """Swarm counts come from the engine. A hand-written table would be a
    second copy of an engine fact that moves whenever a card is rebalanced."""
    assert forecaster.bodies_per_card(GIANT) == 1
    assert forecaster.bodies_per_card(ARCHERS) == 2
    assert forecaster.bodies_per_card(MINIONS) == 3


def test_a_swarm_is_not_multiplied_by_its_own_body_count(forecaster):
    """The bug this exists for. Three detected minions are ONE card; injecting
    once per detected body would put nine minions on the board.

    Counted through CH_COUNT rather than by occupied cells, because swarm
    bodies routinely share a cell and cell-counting would hide the fault.
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
    """Six archers strung down a lane are three cards, and stacking all three
    on the first body would rebuild a blob where the real board has a line."""
    seen = [unit(ARCHERS, 5, 6 + i, name="archer") for i in range(6)]
    out = forecaster.forecast(board(*seen), [0.1])[0]
    assert out.injected == 3
    ys = [y for _x, y in cells(out.observation, 0)]
    assert max(ys) - min(ys) >= 2, "rebuilt as a blob, not a lane"


# --- what cannot be reconstructed is reported, not hidden -------------------

def test_unmappable_units_are_counted_rather_than_dropped(forecaster):
    """A forecast missing part of the board is not a forecast, and the caller
    cannot tell from the observation alone."""
    out = forecaster.forecast(
        board(unit(GIANT, 9, 8), unit(UNKNOWN_CARD_SIM_ID, 9, 10)), [0.5])[0]
    assert out.injected == 1
    assert out.unmappable == 1


# --- the opponent must not be simulated -------------------------------------

def test_the_heuristic_opponent_never_plays_during_a_forecast(forecaster):
    """`step` also runs HeuristicOpponent, which would deploy cards the real
    opponent never played -- the forecast would invent an enemy push and then
    be judged against footage containing none. Two seconds is ample time for
    the heuristic to act if it were running.
    """
    out = forecaster.forecast(board(unit(GIANT, 9, 8)), [2.0])[0]
    assert cells(out.observation, 1) == set(), "an enemy appeared from nowhere"


# --- dynamics ---------------------------------------------------------------

def test_units_actually_move(forecaster):
    """A forecast horizon must actually advance the world.

    The horizon is 5.0 s, not the 2.0 s this used to use, and the reason is
    worth keeping. `cells()` reads the observation's integer grid, so this can
    only see movement once a unit crosses a CELL boundary -- and a Giant spawned
    at x=9.0 sits exactly on one.

    Which way it then steps changed on 2026-08-21. The old "walk to the closest
    enemy tower" rule tie-broke between the two Princess Towers by iteration
    order and sent it LEFT, immediately off the boundary into cell 8. Lane
    pathing sends it RIGHT, because 9.0 is 5.5 tiles from the right bridge and
    6.5 from the left -- correct, and it means x has to climb a whole tile to
    9.0 -> 10.0 before this assertion can see anything. Measured: the Giant is at
    (9.36, 8.48) at 2.0 s and reaches cell (10, 9) at 4.0 s, having moved
    normally the whole time at 0.0355/0.0484 per tick after its 10-tick deploy.

    So the old 2.0 s bound was passing on an accident of tie-break order, not on
    a property of the engine. 5.0 s clears the boundary from any start.
    """
    start, later = forecaster.forecast(board(unit(GIANT, 9, 8)), [0.1, 5.0])
    assert cells(start.observation, 0) != cells(later.observation, 0)


def test_stepping_cumulatively_matches_stepping_directly(forecaster):
    """Horizons are stepped on one trajectory rather than re-simulated per
    horizon, which is only sound because the engine is deterministic.

    Compared over the SPATIAL half only. A first version asserted the whole
    vector and failed -- correctly, but for an unrelated reason: `reset()`
    reshuffles the hand from an unseeded RNG, so the two runs differed in 11
    floats, every one a hand one-hot or hand cost, and in none of the 12,852
    spatial floats. The board really is bit-identical; the hand is fiction
    either way. See test_the_forecast_hand_is_fiction.
    """
    import numpy as np
    from python_ai.models import perception_encoder as enc

    spatial = enc.PLANE * 21
    together = forecaster.forecast(board(unit(GIANT, 9, 8)), [0.5, 1.5])
    alone = forecaster.forecast(board(unit(GIANT, 9, 8)), [1.5])
    assert np.array_equal(together[-1].observation[:spatial],
                          alone[0].observation[:spatial])


def test_the_forecast_hand_is_fiction(forecaster):
    """Pinned because it is a live trap, not a curiosity.

    The hand comes from the engine's own unseeded shuffle, which cannot be
    seeded or set. Anyone feeding a forecast to the policy gets a predicted
    BOARD with an invented HAND -- and `affordability_mask` is built from
    exactly those scalars, so the policy would be gated on cards it does not
    hold. The hand must be overwritten from perception first.
    """
    import numpy as np
    from python_ai.models import perception_encoder as enc

    spatial = enc.PLANE * 21
    seen = board(unit(GIANT, 9, 8))
    hands = {tuple(np.asarray(forecaster.forecast(seen, [0.5])[0]
                              .observation[spatial:]).tolist())
             for _ in range(6)}
    assert len(hands) > 1, "the shuffle looks seeded -- re-read this test"


def test_speed_is_measured_over_a_long_baseline(forecaster):
    """The observation is cell-quantised, so a short baseline carries ~+/-1
    tile regardless of duration. A first version used a flat 10 ticks and
    reported the Giant at 3.61 tiles/s against a true 3.0.

    The expected value is now ~0.6, not ~3.0: `MOVEMENT_SPEED_SCALE` landed on
    2026-08-07 after troop movement measured 4-5x the real game's. A Giant is
    the Slow tier, 0.3 tiles/tick before scaling, so 0.3 * 0.2 * 10 = 0.6
    tiles/s. Real-game Slow is ~0.75, and the remaining ~20% is the documented
    residual of using one flat scale (see CardStats.h).
    """
    assert forecaster.measure_speed(GIANT) == pytest.approx(0.6, abs=0.15)


def test_the_engine_moves_at_roughly_real_game_speed(forecaster):
    """A guard on the whole point of MOVEMENT_SPEED_SCALE.

    Not a tight assertion -- the real-game figures this was calibrated against
    are a measured bracket, not constants. It exists to fail loudly if the
    scale is ever dropped, which would silently return every timing the agent
    learns to being five times too fast.
    """
    for card_id in (GIANT, MUSKETEER):
        assert 0.4 < forecaster.measure_speed(card_id) < 2.5


# --- the agreement metric ---------------------------------------------------

def test_agreement_is_one_for_identical_boards():
    from forecast import agreement
    assert agreement({(1, 2), (3, 4)}, {(1, 2), (3, 4)}) == 1.0


def test_agreement_is_zero_for_disjoint_boards():
    from forecast import agreement
    assert agreement({(1, 2)}, {(5, 6)}) == 0.0


def test_two_empty_boards_agree():
    """Both saying the board is clear IS agreement. Scoring it 0 would make
    every quiet frame look like a total prediction failure."""
    from forecast import agreement
    assert agreement(set(), set()) == 1.0


def test_towers_are_excluded_from_occupancy_comparisons(forecaster):
    """Towers never move, so leaving them in adds six guaranteed matches to
    every comparison -- on a quiet board that is most of the score."""
    from forecast import occupancy, tower_cells
    out = forecaster.forecast(board(), [0.5])[0]
    assert occupancy(out.observation, 0) & tower_cells()
    assert cells(out.observation, 0) == set()
