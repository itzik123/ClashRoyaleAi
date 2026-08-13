"""Pin the tile grid to the arena rectangle measured off the real game.

WHY THIS TEST EXISTS
--------------------
These four constants have now been refitted twice, and the 2026-08-05 fit was
wrong in both axes in a way nothing caught: every existing test passed, the
detector kept producing plausible tiles, and the only symptom was that cards
placed on our own back rows silently failed to deploy. The tap landed below the
board, the game ignored it, and the elixir ledger reported "issued but never
confirmed" -- which had three other candidate explanations.

So the values are pinned here against the measurement that produced them, with
the tolerance set to what the measurement can actually resolve rather than to
whatever the current numbers happen to be.

WHAT IS BEING PINNED IS THE BOARD, NOT THE CONSTANTS
-----------------------------------------------------
Deliberately expressed as arena EDGES rather than as the four raw numbers. Any
future refit is free to change TILE_WIDTH and TILE_INIT_X together as long as
the board still lands where the game draws it, which is the property that
actually matters and the one both wrong fits violated. Pinning the raw values
would just be a copy of the file under test.

Every figure below is measured -- see constants.py for the method and
`tools/deploy_zone.py --fit-grid` to reproduce it.
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

# From the red deploy-zone tint, which is the board rectangle drawn by the game
# itself. Left/right are stable to a pixel across frames and across all 616
# interior columns; the river edge reads 569 from the 10th to the 90th
# percentile of those columns.
ARENA_LEFT, ARENA_RIGHT = 50.5, 668.5
RIVER_EDGE = 569.5                      # detector row boundary 15

# From tapping raw pixels with a card selected and the elixir bar at its cap:
# accepted at y<=980 / refused at y>=982 (x=379), accepted y<=978 / refused
# y>=984 (x=100). Two columns 279 px apart, so not the King HP bar occluding it.
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
    """engine x 9.0 is the board's true centre, so it must map to 360.

    An independent check: the width fit above used only the two side edges.
    Note it is necessary but NOT sufficient -- both previous fits passed this
    while being 10% wrong in scale, because they were anchored at the centre
    and the error grows outwards. That is precisely why the edges are pinned
    too.
    """
    centre = TILE_INIT_X + 9.0 * TILE_WIDTH
    assert centre == pytest.approx(DISPLAY_WIDTH / 2, abs=EDGE_TOL)


def test_own_half_is_fifteen_rows_ending_at_the_river():
    """The bottom edge and the river edge must be 15 tile rows apart.

    This is the constraint the 2026-08-05 fit broke: it put the arena bottom at
    y=1003.8 when the game stops accepting taps at y=981, so the whole back of
    our own half was addressed below the board.
    """
    assert arena_bottom() == pytest.approx(ARENA_BOTTOM, abs=ARENA_BOTTOM_TOL)
    river = arena_bottom() - 15 * TILE_HEIGHT
    assert river == pytest.approx(RIVER_EDGE, abs=EDGE_TOL)


def test_every_placeable_engine_row_taps_inside_the_arena():
    """The regression, stated as the thing that actually failed.

    Engine rows 1..15 are the own half the actuator has to reach. Under the old
    grid engine row 1 tapped y=989 against a board ending at 981, and measured
    0/6 acceptance live; every other row was progressively less wrong. Rows are
    checked at both edge columns as well as the centre, because the scale error
    was smallest in the middle.
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
    """The engine's extra back row has no arena row, and must stay masked.

    It is not a legality question -- the row genuinely does not exist on screen,
    so its tap lands below the board whatever the grid is.
    """
    assert not engine_row_is_tappable(0)
    assert engine_tile_centre(9, 0).y > ARENA_BOTTOM
