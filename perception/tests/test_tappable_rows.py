"""Engine rows the actuator cannot reach must never be offered to the policy.

Engine row 0 converts to detector row -1 (TILE_Y_OFFSET = 1), a row the 32-row
arena lacks; its tap falls in the strip between the arena and the card tray, so
the game deselects the card and deploys nothing. The net's placement_mask
allows row 0 because the engine has it; the screen mapping cannot reach it.

Stated in tile units, not pixels: pixel assertions derived from the tile grid
pass even when the grid is wrong. Row 0 is unreachable under any correct grid;
`test_tile_grid.py` pins the grid itself.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clashroyalebuildabot.constants import (  # noqa: E402
    DISPLAY_HEIGHT,
    TILE_HEIGHT,
    TILE_INIT_Y,
)
from live.actuator import (  # noqa: E402
    DETECTOR_ROWS,
    engine_row_is_tappable,
    engine_tile_centre,
)
from live.adapter import TILE_Y_OFFSET  # noqa: E402

ARENA_BOTTOM = DISPLAY_HEIGHT - TILE_INIT_Y


def test_engine_row_zero_taps_below_the_arena():
    """The measured defect itself."""
    tap = engine_tile_centre(9, 0)
    assert tap.y > ARENA_BOTTOM, (
        f"engine row 0 taps at y={tap.y}, arena bottom is {ARENA_BOTTOM}. If this "
        f"now passes, TILE_Y_OFFSET changed and the mask below must be re-derived.")
    assert not engine_row_is_tappable(0)


def test_every_tappable_row_lands_inside_the_arena():
    """The property the mask exists to guarantee, over the whole board."""
    for engine_y in range(34):
        tap = engine_tile_centre(9, engine_y)
        if engine_row_is_tappable(engine_y):
            assert tap.y <= ARENA_BOTTOM, (
                f"engine row {engine_y} is marked tappable but taps at y={tap.y}, "
                f"below the arena bottom {ARENA_BOTTOM}")
            assert tap.y >= 0, f"engine row {engine_y} taps off the top at {tap.y}"


def test_tappable_rows_are_exactly_the_rows_with_a_detector_row():
    lo, hi = TILE_Y_OFFSET, TILE_Y_OFFSET + DETECTOR_ROWS
    assert [y for y in range(34) if engine_row_is_tappable(y)] == list(range(lo, hi))


def test_the_agents_own_half_loses_exactly_one_row():
    """Own-half placement rows are 0..15 and only row 0 is unreachable. Bounds the
    cost of the fix: a geometry change making the mask eat half our own half
    should fail here, not show up as a passive bot.
    """
    own_half = [y for y in range(16) if not engine_row_is_tappable(y)]
    assert own_half == [0], f"expected only engine row 0 unreachable, got {own_half}"


def test_row_one_is_reachable_and_near_the_bottom_edge():
    """Row 1 is valid, documented because it is the natural next suspect: it lands
    inside the arena, and a game rejection there would be a separate finding
    about the real deployable area.
    """
    tap = engine_tile_centre(9, 1)
    assert engine_row_is_tappable(1)
    assert tap.y < ARENA_BOTTOM
    assert ARENA_BOTTOM - tap.y < TILE_HEIGHT, (
        "row 1 should sit within one tile of the arena's bottom edge")
