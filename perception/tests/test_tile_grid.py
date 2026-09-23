"""Pin the tile grid to the arena rectangle measured off the real game.

A wrong fit is invisible to every other test: the detector keeps producing
plausible tiles, and the only symptom is that cards placed on our back rows
silently fail to deploy (the tap lands below the board). So the values are
pinned against the measurement that produced them, with tolerances set to what
it can resolve.

Expressed as arena edges rather than the four raw constants, so a future refit
may change TILE_WIDTH and TILE_INIT_X together as long as the board lands where
the game draws it. See constants.py for the method and `tools/deploy_zone.py
--fit-grid` to reproduce it.
"""
from __future__ import annotations

import pytest

from clashroyalebuildabot.constants import (
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    N_WIDE_TILES,
    TILE_HEIGHT,
    TILE_INIT_X,
    TILE_INIT_Y,
    TILE_WIDTH,
)
from live.actuator import engine_row_is_tappable, engine_tile_centre

# From the red deploy-zone tint, the board rectangle the game draws. Left/right
# are stable to a pixel across frames and all 616 interior columns; the river
# edge reads 569 from the 10th to the 90th percentile of those columns.
ARENA_LEFT, ARENA_RIGHT = 50.5, 668.5
RIVER_EDGE = 569.5                      # detector row boundary 15

# From tapping raw pixels with a card selected and elixir capped: accepted
# y<=980 / refused y>=982 at x=379, accepted y<=978 / refused y>=984 at x=100.
# Two columns 279 px apart, so not the King HP bar occluding it.
ARENA_BOTTOM = 981.0
ARENA_BOTTOM_TOL = 2.0

# The tint's edges are sharp, but the threshold crossing and the half-pixel
# convention are each worth about a pixel.
EDGE_TOL = 2.0


def arena_bottom() -> float:
    return DISPLAY_HEIGHT - TILE_INIT_Y


def test_board_spans_the_measured_arena_width():
    assert TILE_INIT_X == pytest.approx(ARENA_LEFT, abs=EDGE_TOL)
    right = TILE_INIT_X + N_WIDE_TILES * TILE_WIDTH
    assert right == pytest.approx(ARENA_RIGHT, abs=EDGE_TOL)


def test_kings_column_lands_on_the_display_centre():
    """Nine tile widths from the left edge is the board's centre (the Kings'
    column), so it must map to 360.

    Necessary but not sufficient: a centre-anchored fit passes this while wrong
    in scale, since the error grows outwards. Hence the edges are pinned too.
    """
    centre = TILE_INIT_X + 9.0 * TILE_WIDTH
    assert centre == pytest.approx(DISPLAY_WIDTH / 2, abs=EDGE_TOL)


def test_own_half_is_fifteen_rows_ending_at_the_river():
    """The bottom edge and the river edge must be 15 tile rows apart, or the back
    of our own half is addressed below the board.
    """
    assert arena_bottom() == pytest.approx(ARENA_BOTTOM, abs=ARENA_BOTTOM_TOL)
    river = arena_bottom() - 15 * TILE_HEIGHT
    assert river == pytest.approx(RIVER_EDGE, abs=EDGE_TOL)


def test_every_placeable_engine_row_taps_inside_the_arena():
    """Engine rows 1..15 are the own half the actuator must reach. Checked at both
    edge columns as well as the centre, since a scale error is smallest in the
    middle.
    """
    for y in range(1, 16):
        assert engine_row_is_tappable(y), f"engine row {y} is masked out"
        for x in (0, 9, N_WIDE_TILES - 1):
            tap = engine_tile_centre(x, y)
            assert ARENA_LEFT <= tap.x <= ARENA_RIGHT, (
                f"engine ({x},{y}) taps x={tap.x}, outside the board")
            assert RIVER_EDGE <= tap.y <= ARENA_BOTTOM, (
                f"engine ({x},{y}) taps y={tap.y}, outside "
                f"[{RIVER_EDGE}, {ARENA_BOTTOM}]")


def test_engine_row_zero_is_still_unreachable():
    """The engine's extra back row has no arena row and must stay masked: its tap
    lands below the board whatever the grid is.
    """
    assert not engine_row_is_tappable(0)
    assert engine_tile_centre(9, 0).y > ARENA_BOTTOM
