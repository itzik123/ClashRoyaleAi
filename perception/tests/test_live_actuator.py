"""Tests for live/actuator.py.

This module can place a real card in a real match, and a wrong tile is
invisible downstream. So the geometry is checked against CRBAB's own mapping,
the one upstream runs against the real game, rather than against numbers
re-derived here.
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

from clashroyalebuildabot.constants import (  # noqa: E402
    DISPLAY_CARD_DELTA_X,
    DISPLAY_CARD_HEIGHT,
    DISPLAY_CARD_INIT_X,
    DISPLAY_CARD_WIDTH,
    DISPLAY_CARD_Y,
    DISPLAY_HEIGHT,
    TILE_HEIGHT,
    TILE_INIT_X,
    TILE_INIT_Y,
    TILE_WIDTH,
)

# Upstream's mapping, transcribed verbatim from `Bot._get_tile_centre` and
# `Bot._get_card_centre` (clashroyalebuildabot/bot/bot.py, since deleted:
# importing it pulled PyQt6, `keyboard` and the ADB emulator into a sensor that
# never opens a window).
#
# A deliberate second copy: a cross-check needs two independent expressions of
# the mapping, and deriving this side from live/actuator.py would make the test
# vacuous. It reads the same constants, so a legitimate grid re-fit
# (tools/fit_tile_grid.py) moves both sides together; only a drift in the
# arithmetic fails.
def _upstream_tile_centre(tile_x, tile_y):
    x = TILE_INIT_X + (tile_x + 0.5) * TILE_WIDTH
    y = DISPLAY_HEIGHT - TILE_INIT_Y - (tile_y + 0.5) * TILE_HEIGHT
    return x, y


def _upstream_card_centre(card_n):
    x = (
        DISPLAY_CARD_INIT_X
        + DISPLAY_CARD_WIDTH / 2
        + card_n * DISPLAY_CARD_DELTA_X
    )
    y = DISPLAY_CARD_Y + DISPLAY_CARD_HEIGHT / 2
    return x, y


@pytest.mark.parametrize("tile_x", [0, 4, 9, 14, 17])
@pytest.mark.parametrize("tile_y", [0, 5, 15, 16, 31])
def test_tile_centre_matches_crbabs_own(tile_x, tile_y):
    """Agreement with the mapping upstream ships against the real game.
    live/actuator.py expresses it independently of the reference above; a drift
    would put every placement in the wrong square.
    """
    mine = tile_centre(tile_x, tile_y)
    theirs = _upstream_tile_centre(tile_x, tile_y)
    assert (mine.x, mine.y) == (round(theirs[0]), round(theirs[1]))


@pytest.mark.parametrize("slot", [0, 1, 2, 3])
def test_card_centre_matches_crbabs_own(slot):
    mine = card_centre(slot)
    theirs = _upstream_card_centre(slot)
    assert (mine.x, mine.y) == (round(theirs[0]), round(theirs[1]))


def test_engine_and_detector_frames_differ_by_exactly_the_row_offset():
    """The policy emits engine coordinates (18x34); the screen mapping is in
    detector coordinates (18x32). Conflating them shifts every placement by a
    row.
    """
    for tile_x in (0, 9, 17):
        for engine_y in (TILE_Y_OFFSET, 10, 32):
            assert (engine_tile_centre(tile_x, engine_y)
                    == tile_centre(tile_x, engine_y - TILE_Y_OFFSET))


def test_engine_frame_is_not_the_identity():
    """Guards against "simplifying" the offset away: if these coincide, the
    conversion is lost.
    """
    assert engine_tile_centre(9, 10) != tile_centre(9, 10)


def test_a_bad_hand_slot_raises_rather_than_tapping_somewhere():
    for slot in (-1, 4, 99):
        with pytest.raises(ActuationError):
            card_centre(slot)


def test_dry_run_is_the_default():
    """Acting must be asked for: a forgotten flag means the loop watches."""
    assert AdbActuator().dry_run is True


def test_dry_run_records_intent_without_calling_adb():
    actuator = AdbActuator(dry_run=True)
    card, target = actuator.play(2, 9, 10)
    assert actuator.taps == [card, target]
    assert card == card_centre(2)
    assert target == engine_tile_centre(9, 10)


def test_play_taps_the_card_before_the_tile():
    """Order is the placement protocol: select, then target. Reversed, the first
    tap lands on the arena with no card held and the second selects a card
    never placed.
    """
    actuator = AdbActuator(dry_run=True)
    actuator.play(1, 4, 20)
    assert actuator.taps[0] == card_centre(1)
    assert actuator.taps[1] == engine_tile_centre(4, 20)


def test_taps_land_inside_the_android_display():
    """720x1280 is what `input tap` addresses; Android silently swallows a
    coordinate outside it.
    """
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
    select.
    """
    lowest_board_row = tile_centre(9, 0).y
    assert all(card_centre(s).y > lowest_board_row for s in range(4))


# --- asynchrony ---

def test_play_does_not_block_the_caller(monkeypatch, tmp_path):
    """A tap costs ~410 ms and a placement is two, so inline acting dragged the
    decision loop from 1.0 Hz to ~0.6 Hz when the agent was busiest, feeding a
    recurrent policy intervals it never trained on.
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
    """Both taps in one shell command, so two placements' taps cannot interleave.
    """
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
    """Depth one, and full means drop: a queued placement would land seconds after
    its board, and blocking would put the latency back on the decision thread.
    """
    import threading
    import time

    release = threading.Event()
    fake_adb = tmp_path / "adb.exe"
    fake_adb.write_text("")
    actuator = AdbActuator(dry_run=False, adb=fake_adb)
    monkeypatch.setattr(actuator, "_shell", lambda s: release.wait(2.0))
    try:
        actuator.play(0, 9, 8)
        time.sleep(0.1)                 # worker picks up the first
        actuator.play(1, 9, 8)          # in flight -> queued
        actuator.play(2, 9, 8)          # queue full -> dropped
        assert actuator.dropped >= 1
    finally:
        release.set()
        actuator.close()


def test_a_failed_tap_does_not_kill_the_worker(monkeypatch, tmp_path):
    """A dead worker looks like an agent still acting while nothing reaches the
    game.
    """
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
    """Dry run stays synchronous: tests and the replay path assert against `taps`,
    and a thread would make it racy.
    """
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


# --- raw evdev touch ---

def test_the_panel_is_landscape_and_the_app_is_rotated_onto_it():
    """`wm size` reports 1280x720 while the app renders 720x1280; the touch
    device's axes are the panel's, and assuming the app's frame sent a
    top-right tap to the top-left.
    """
    from live.actuator import RawTouch

    assert RawTouch(1280, 720).rotated is True
    assert RawTouch(720, 1280).rotated is False


def test_raw_coordinates_match_the_two_verified_live_taps():
    """Both confirmed against real buttons: (655,135) opened the hamburger menu,
    (430,418) the Training Camp dialog.
    """
    from live.actuator import RawTouch

    raw = RawTouch(1280, 720)
    assert raw.to_device(655, 135) == (29311, 29809)
    assert raw.to_device(430, 418) == (22067, 19569)


def test_the_screen_centre_maps_to_the_axis_centre():
    from live.actuator import RawTouch

    assert RawTouch(1280, 720).to_device(360, 640) == (16384, 16384)


def test_raw_coordinates_stay_inside_the_axis_range():
    from live.actuator import ABS_MAX, RawTouch

    raw = RawTouch(1280, 720)
    for x, y in ((0, 0), (719, 1279), (0, 1279), (719, 0)):
        ex, ey = raw.to_device(x, y)
        assert 0 <= ex <= ABS_MAX and 0 <= ey <= ABS_MAX


def test_a_placement_holds_each_contact_and_releases_it():
    """A contact needs duration: press and release in one write is a zero-length
    touch that dismissed a menu instead of pressing the button under it.
    """
    from live.actuator import RawTouch, Tap

    script = RawTouch(1280, 720).placement_script(Tap(100, 200), Tap(300, 400))
    assert script.count("sleep") == 3          # hold, gap, hold
    assert script.count("base64 -d") == 4      # press+release, twice


def test_raw_touch_is_one_round_trip_for_the_whole_placement(monkeypatch, tmp_path):
    """The holds and the gap run on-device, inside a trip already being made.
    """
    sent = []
    fake_adb = tmp_path / "adb.exe"
    fake_adb.write_text("")
    actuator = AdbActuator(dry_run=False, adb=fake_adb, raw_touch=False)
    monkeypatch.setattr(actuator, "_shell", sent.append)
    from live.actuator import RawTouch
    actuator.raw = RawTouch(1280, 720)
    try:
        actuator.play(0, 9, 8)
        actuator.flush()
    finally:
        actuator.close()
    assert len(sent) == 1


def test_it_falls_back_to_input_tap_without_the_device(tmp_path):
    """Device path, axis ranges and rotation are properties of this emulator;
    being wrong taps the wrong place rather than failing.
    """
    fake_adb = tmp_path / "adb.exe"
    fake_adb.write_text("")
    actuator = AdbActuator(dry_run=False, adb=fake_adb, raw_touch=False)
    try:
        assert actuator.raw is None
        assert actuator.backend == "input-tap"
    finally:
        actuator.close()
