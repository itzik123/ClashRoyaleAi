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


# --- asynchrony -------------------------------------------------------------

def test_play_does_not_block_the_caller(monkeypatch, tmp_path):
    """The reason this class was rewritten. A tap costs ~410 ms on-device and a
    placement is two of them, so acting inline dragged the decision loop from
    1.0 Hz to ~0.6 Hz exactly when the agent was most active -- feeding a
    recurrent policy intervals it was never trained on.
    """
    import time

    fake_adb = tmp_path / "adb.exe"
    fake_adb.write_text("")
    actuator = AdbActuator(dry_run=False, adb=fake_adb)
    monkeypatch.setattr(actuator, "_shell", lambda script: time.sleep(0.5))
    try:
        started = time.perf_counter()
        actuator.play(0, 9, 8)
        elapsed = time.perf_counter() - started
        assert elapsed < 0.05, f"play() blocked for {elapsed * 1000:.0f} ms"
        actuator.flush()
    finally:
        actuator.close()


def test_a_placement_is_sent_as_one_unit(monkeypatch, tmp_path):
    """Both taps in one shell command. Interleaving two placements' taps would
    pair a card-select with the wrong tile tap."""
    sent = []
    fake_adb = tmp_path / "adb.exe"
    fake_adb.write_text("")
    actuator = AdbActuator(dry_run=False, adb=fake_adb)
    monkeypatch.setattr(actuator, "_shell", sent.append)
    try:
        actuator.play(2, 4, 12)
        actuator.flush()
    finally:
        actuator.close()
    assert len(sent) == 1, sent
    card, target = card_centre(2), engine_tile_centre(4, 12)
    assert f"input tap {card.x} {card.y}" in sent[0]
    assert f"input tap {target.x} {target.y}" in sent[0]
    assert sent[0].index(f"tap {card.x}") < sent[0].index(f"tap {target.x}")


def test_a_placement_arriving_mid_tap_is_dropped_not_queued(monkeypatch, tmp_path):
    """Depth one, and full means drop. A queued placement would land seconds
    after the board it was chosen for, by which point it is not a late move but
    a different one. Blocking instead would put the latency straight back on
    the decision thread."""
    import threading
    import time

    release = threading.Event()
    fake_adb = tmp_path / "adb.exe"
    fake_adb.write_text("")
    actuator = AdbActuator(dry_run=False, adb=fake_adb)
    monkeypatch.setattr(actuator, "_shell", lambda s: release.wait(2.0))
    try:
        actuator.play(0, 9, 8)
        time.sleep(0.1)                 # worker picks the first one up
        actuator.play(1, 9, 8)          # in flight -> queued
        actuator.play(2, 9, 8)          # queue full -> dropped
        assert actuator.dropped >= 1
    finally:
        release.set()
        actuator.close()


def test_a_failed_tap_does_not_kill_the_worker(monkeypatch, tmp_path):
    """A dead worker looks exactly like an agent that is still acting, while
    nothing reaches the game."""
    calls = {"n": 0}

    def flaky(script):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ActuationError("boom")

    fake_adb = tmp_path / "adb.exe"
    fake_adb.write_text("")
    actuator = AdbActuator(dry_run=False, adb=fake_adb)
    monkeypatch.setattr(actuator, "_shell", flaky)
    try:
        actuator.play(0, 9, 8)
        actuator.flush()
        actuator.play(1, 9, 8)
        actuator.flush()
        assert actuator.errors == 1
        assert calls["n"] == 2, "worker stopped after the failure"
    finally:
        actuator.close()


def test_dry_run_starts_no_worker_and_sends_nothing():
    """Dry run must stay fully synchronous: it is what the tests and the replay
    path assert against, and a thread would make `taps` racy."""
    actuator = AdbActuator(dry_run=True)
    actuator.play(0, 9, 8)
    assert actuator._worker is None
    assert len(actuator.taps) == 2
    actuator.close()


def test_taps_are_recorded_at_intent_not_at_send(monkeypatch, tmp_path):
    """So a caller can read back what it asked for without waiting on adb."""
    import threading

    release = threading.Event()
    fake_adb = tmp_path / "adb.exe"
    fake_adb.write_text("")
    actuator = AdbActuator(dry_run=False, adb=fake_adb)
    monkeypatch.setattr(actuator, "_shell", lambda s: release.wait(2.0))
    try:
        actuator.play(3, 4, 20)
        assert actuator.taps == [card_centre(3), engine_tile_centre(4, 20)]
    finally:
        release.set()
        actuator.close()
