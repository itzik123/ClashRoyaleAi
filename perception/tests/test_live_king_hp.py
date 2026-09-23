"""Tests for live/king_hp.py and live/board_filter.py.

Synthetic frames: the King bar is a flat-coloured rectangle at a fixed place,
and only a synthetic frame guarantees the "no bar" case. The colours are the
ones measured off real frames and quoted in live/king_hp.py, so a re-skin of
the HP bars fails these.
"""
from __future__ import annotations

import numpy as np
import pytest

from live.board_filter import BOARD_HEIGHT, BOARD_WIDTH, filter_units, on_board
from live.king_hp import (ALLY_KING_BAR_Y, ENEMY_KING_BAR_Y, KING_BAR_HEIGHT,
                          KING_BAR_X0, KING_BAR_X1, read_king_hp)

# Sampled from real frames; see live/king_hp.py.
ARENA_GREEN = (110, 170, 70)
TRACK_BROWN = (111, 94, 83)
ALLY_BLUE = (111, 208, 252)
ENEMY_PINK = (228, 56, 108)
NUMERAL_WHITE = (255, 255, 255)

WIDTH = KING_BAR_X1 - KING_BAR_X0


def _frame():
    """An arena-coloured screenshot-space frame with no HP bar drawn."""
    return np.full((652, 368, 3), ARENA_GREEN, dtype=np.uint8)


def _draw_bar(frame, ally: bool, filled_columns: int, numeral: bool = False):
    y = ALLY_KING_BAR_Y if ally else ENEMY_KING_BAR_Y
    fill = ALLY_BLUE if ally else ENEMY_PINK
    frame[y:y + KING_BAR_HEIGHT, KING_BAR_X0:KING_BAR_X1] = TRACK_BROWN
    if filled_columns:
        frame[y:y + KING_BAR_HEIGHT, KING_BAR_X0:KING_BAR_X0 + filled_columns] = fill
    if numeral:
        # The HP number is drawn on the bar from its left edge, which is what
        # made a fill-based presence test report a dying King as full.
        frame[y + 1:y + KING_BAR_HEIGHT - 1, KING_BAR_X0:KING_BAR_X0 + 8] = NUMERAL_WHITE
    return frame


@pytest.mark.parametrize("ally", [True, False])
def test_no_bar_means_full_not_destroyed(ally):
    """The inverse of the Princess convention: an absent bar means full."""
    result = read_king_hp(_frame(), ally=ally)
    assert result.bar_present is False
    assert result.fraction == 1.0


@pytest.mark.parametrize("ally", [True, False])
@pytest.mark.parametrize("columns", [10, 22, 40])
def test_fill_fraction_tracks_filled_columns(ally, columns):
    frame = _draw_bar(_frame(), ally=ally, filled_columns=columns)
    result = read_king_hp(frame, ally=ally)
    assert result.bar_present is True
    assert result.fraction == pytest.approx(columns / WIDTH, abs=1.5 / WIDTH)


@pytest.mark.parametrize("ally", [True, False])
def test_bar_with_no_resolvable_fill_is_low_not_full(ally):
    """A King at ~8% whose fill is hidden by the numeral. Reading it as 1.0 would
    tell the agent it is safe one hit from losing; the requirement is only that
    it comes out near zero.
    """
    frame = _draw_bar(_frame(), ally=ally, filled_columns=0, numeral=True)
    result = read_king_hp(frame, ally=ally)
    assert result.bar_present is True
    assert result.fraction < 0.05


def test_ally_and_enemy_bars_are_read_independently():
    """A bar on one side must not be picked up as the other side's."""
    frame = _draw_bar(_frame(), ally=False, filled_columns=20)
    assert read_king_hp(frame, ally=False).bar_present is True
    assert read_king_hp(frame, ally=True).bar_present is False


def test_rejects_a_frame_of_the_wrong_size():
    """Device pixels are not screenshot space; failing loudly beats guessing."""
    with pytest.raises(ValueError):
        read_king_hp(np.zeros((10, 10, 3), dtype=np.uint8), ally=True)


# --- board_filter ---


class _Pos:
    def __init__(self, x, y):
        self.tile_x, self.tile_y = x, y


class _Unit:
    def __init__(self, x, y):
        self.position = _Pos(x, y)


def test_on_board_bounds():
    assert on_board(0, 0)
    assert on_board(BOARD_WIDTH - 1, BOARD_HEIGHT - 1)
    assert not on_board(BOARD_WIDTH, 0)
    assert not on_board(-1, 0)
    assert not on_board(0, BOARD_HEIGHT)


def test_filter_drops_the_measured_phantom_positions():
    """(19,5) and (-2,13) are the two player avatar icons, read as Knights."""
    units = [_Unit(19, 5), _Unit(-2, 13), _Unit(9, 16), _Unit(3, 20)]
    kept, report = filter_units(units)
    assert [(u.position.tile_x, u.position.tile_y) for u in kept] == [(9, 16), (3, 20)]
    assert report.dropped == 2
    assert report.kept == 2
    assert report.dropped_fraction == pytest.approx(0.5)


def test_filter_report_is_defined_when_empty():
    kept, report = filter_units([])
    assert kept == []
    assert report.dropped_fraction == 0.0
