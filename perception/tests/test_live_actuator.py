"""Tests for live/actuator.py.

This is the module that can place a real card in a real match, and a mistake in
it is invisible downstream: a tap on the wrong tile produces a perfectly valid
GameState next frame, showing a unit somewhere nobody intended. So the geometry
is checked against ClashRoyaleBuildABot's own methods -- the ones upstream runs
against the real game -- rather than against numbers re-derived here.
"""
from __future__ import annotations

import pytest

from live.actuator import (
    AdbActuator,
    ActuationError,
    card_centre,
    engine_tile_centre,
    tile_centre,
)
from live.adapter import TILE_Y_OFFSET

pytest.importorskip("clashroyalebuildabot", reason="vendored bot not importable")


@pytest.mark.parametrize("tile_x", [0, 4, 9, 14, 17])
@pytest.mark.parametrize("tile_y", [0, 5, 15, 16, 31])
def test_tile_centre_matches_crbabs_own(tile_x, tile_y):
    """Agreement with the implementation upstream ships against the real game.

    Not a tautology: this module re-implements the mapping so the actuator does
    not import from `bot.py`, which drags in the whole bot. A drift between the
    two would put every placement in the wrong square.
    """
    from clashroyalebuildabot.bot.bot import Bot

    mine = tile_centre(tile_x, tile_y)
    theirs = Bot._get_tile_centre(tile_x, tile_y)
    assert (mine.x, mine.y) == (round(theirs[0]), round(theirs[1]))


@pytest.mark.parametrize("slot", [0, 1, 2, 3])
def test_card_centre_matches_crbabs_own(slot):
    from clashroyalebuildabot.bot.bot import Bot

    mine = card_centre(slot)
    theirs = Bot._get_card_centre(slot)
    assert (mine.x, mine.y) == (round(theirs[0]), round(theirs[1]))


def test_engine_and_detector_frames_differ_by_exactly_the_row_offset():
    """The policy emits ENGINE coordinates (18x34); the screen mapping is in
    DETECTOR coordinates (18x32). Conflating them shifts every placement by a
    row -- the bug class this project has already paid for once."""
    for tile_x in (0, 9, 17):
        for engine_y in (TILE_Y_OFFSET, 10, 32):
            assert (engine_tile_centre(tile_x, engine_y)
                    == tile_centre(tile_x, engine_y - TILE_Y_OFFSET))


def test_engine_frame_is_not_the_identity():
    """Guards against someone 'simplifying' the offset away. If these ever
    coincide, the conversion has been lost."""
    assert engine_tile_centre(9, 10) != tile_centre(9, 10)


def test_a_bad_hand_slot_raises_rather_than_tapping_somewhere():
    for slot in (-1, 4, 99):
        with pytest.raises(ActuationError):
            card_centre(slot)


def test_dry_run_is_the_default():
    """Acting must be asked for. A forgotten flag should mean the loop
    watches, not that it plays cards in a live match."""
    assert AdbActuator().dry_run is True


def test_dry_run_records_intent_without_calling_adb():
    actuator = AdbActuator(dry_run=True)
    card, target = actuator.play(2, 9, 10)
    assert actuator.taps == [card, target]
    assert card == card_centre(2)
    assert target == engine_tile_centre(9, 10)


def test_play_taps_the_card_before_the_tile():
    """Order is the placement protocol, not a preference: Clash Royale selects
    a card and then targets. Reversed, the first tap lands on the arena with no
    card held and the second selects a card that is never placed."""
    actuator = AdbActuator(dry_run=True)
    actuator.play(1, 4, 20)
    assert actuator.taps[0] == card_centre(1)
    assert actuator.taps[1] == engine_tile_centre(4, 20)


def test_taps_land_inside_the_android_display():
    """720x1280 is what `input tap` addresses. A coordinate outside it is
    silently swallowed by Android rather than reported."""
    for tile_x in range(18):
        for tile_y in range(32):
            p = tile_centre(tile_x, tile_y)
            assert 0 <= p.x < 720, (tile_x, tile_y, p)
            assert 0 <= p.y < 1280, (tile_x, tile_y, p)
    for slot in range(4):
        p = card_centre(slot)
        assert 0 <= p.x < 720 and 0 <= p.y < 1280


def test_the_hand_row_is_below_the_arena():
    """A card tap must not land on the board, or it would place rather than
    select."""
    lowest_board_row = tile_centre(9, 0).y
    assert all(card_centre(s).y > lowest_board_row for s in range(4))
