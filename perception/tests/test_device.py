"""Tests for live/device.py, the phone capture crop, and the phone-mode actuator.

The invariant everything hangs on: the detector's frame and the taps go through
ONE geometry, `DeviceProfile.game_rect`. The end-to-end test paints the exact
device pixel a tap would hit, mirrors the screen the way scrcpy does (scaled,
inside a window with chrome), crops it with the real capture code, and requires
the mark to land where that tile sits in the detector's frame.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from live.device import (
    CRBAB_DISPLAY,
    DeviceError,
    DeviceProfile,
    default_game_rect,
    find_adb,
    parse_devices,
    parse_wm_size,
    require_device,
)


def test_the_crbab_space_is_crbabs_own():
    from clashroyalebuildabot.constants import DISPLAY_HEIGHT, DISPLAY_WIDTH
    assert CRBAB_DISPLAY == (DISPLAY_WIDTH, DISPLAY_HEIGHT)


# --- parsing adb -------------------------------------------------------------

@pytest.mark.parametrize("out, want", [
    ("Physical size: 1080x2400\n", (1080, 2400)),
    ("Physical size: 1440x3200\nOverride size: 1080x2400\n", (1080, 2400)),
    ("Physical size: 2400x1080\n", (1080, 2400)),          # reported landscape
])
def test_wm_size_is_portrait_and_override_wins(out, want):
    assert parse_wm_size(out) == want


def test_an_unreadable_wm_size_raises():
    with pytest.raises(DeviceError):
        parse_wm_size("error: no devices")


def test_adb_devices_lines_are_parsed_with_their_state():
    out = ("* daemon not running; starting now at tcp:5037\n"
           "* daemon started successfully\n"
           "List of devices attached\n"
           "R5CT12345   device usb:1-1 product:x model:SM_G991B device:o1s\n"
           "192.168.1.20:5555 unauthorized\n"
           "emulator-5554 offline\n")
    devs = parse_devices(out)
    assert [(d.serial, d.state) for d in devs] == [
        ("R5CT12345", "device"), ("192.168.1.20:5555", "unauthorized"),
        ("emulator-5554", "offline")]
    assert devs[0].model == "SM_G991B"


@pytest.mark.parametrize("listing, serial, needle", [
    ("List of devices attached\n", None, "no Android device"),
    ("List of devices attached\nA device\nB device\n", None, "--serial"),
    ("List of devices attached\nA unauthorized\n", None, "Allow USB debugging"),
    ("List of devices attached\nA device\n", "B", "not attached"),
])
def test_require_device_names_what_to_fix(monkeypatch, listing, serial, needle):
    import live.device as D
    monkeypatch.setattr(D, "run_adb", lambda *a, **k: listing)
    with pytest.raises(DeviceError, match=needle):
        require_device("adb", serial)


def test_find_adb_order_and_a_useful_failure(monkeypatch, tmp_path):
    import live.device as D
    for var in ("CLASH_ADB", "ANDROID_HOME", "ANDROID_SDK_ROOT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(D.shutil, "which", lambda name: None)
    monkeypatch.setattr(D, "_BLUESTACKS_ADB", tmp_path / "missing-bluestacks")
    with pytest.raises(DeviceError, match="Platform-Tools"):
        find_adb()
    sdk = tmp_path / "sdk" / "platform-tools"
    sdk.mkdir(parents=True)
    (sdk / "adb").write_text("")
    monkeypatch.setenv("ANDROID_HOME", str(tmp_path / "sdk"))
    assert find_adb() == sdk / "adb"
    env_adb = tmp_path / "custom-adb"
    env_adb.write_text("")
    monkeypatch.setenv("CLASH_ADB", str(env_adb))
    assert find_adb() == env_adb                     # CLASH_ADB beats the SDK
    explicit = tmp_path / "explicit-adb"
    explicit.write_text("")
    assert find_adb(explicit) == explicit            # an explicit path beats all


# --- geometry -----------------------------------------------------------------

def test_a_9_16_screen_is_the_whole_screen_and_maps_like_the_emulator():
    """The control: on the emulator's own 720x1280 the mapping is identity."""
    p = DeviceProfile(name="emu", display=(720, 1280))
    assert p.game_rect == (0.0, 0.0, 1.0, 1.0)
    for x, y in [(0, 0), (360, 640), (719, 1279), (123, 987)]:
        assert p.to_device(x, y) == (x, y)


def test_a_tall_phone_gets_a_centred_9_16_rect():
    assert default_game_rect((1080, 2400)) == pytest.approx((0.0, 0.1, 1.0, 0.8))
    p = DeviceProfile(name="tall", display=(1080, 2400))
    assert p.to_device(0, 0) == (0, 240)
    assert p.to_device(360, 640) == (540, 1200)


def test_a_wide_screen_gets_a_centred_9_16_rect():
    assert default_game_rect((1200, 1600)) == pytest.approx((0.125, 0.0, 0.75, 1.0))


def test_a_rect_outside_the_screen_is_refused():
    with pytest.raises(DeviceError):
        DeviceProfile(name="bad", display=(1080, 2400),
                      game_rect=(0.0, 0.5, 1.0, 0.8))


def test_a_profile_round_trips_through_its_json(tmp_path):
    p = DeviceProfile(name="mine", display=(1080, 2400), serial="R5CT",
                      game_rect=(0.0, 0.08, 1.0, 0.82), calibrated=True)
    path = p.save(tmp_path / "mine.json")
    q = DeviceProfile.load(path)
    assert (q.display, q.serial, q.game_rect, q.calibrated) == (
        p.display, p.serial, p.game_rect, True)


def test_a_missing_profile_says_how_to_make_one(tmp_path):
    with pytest.raises(DeviceError, match="--save-profile"):
        DeviceProfile.load(tmp_path / "nope.json")


# --- capture and taps agree -------------------------------------------------

def _mirror_in_window(screen: np.ndarray, scale_w: int, title_px: int = 30,
                      side_px: int = 40) -> np.ndarray:
    """What WGC returns for a scrcpy window: the screen scaled to `scale_w`
    wide, a lit title bar on top, black bars at the sides."""
    import cv2
    h = int(round(screen.shape[0] * scale_w / screen.shape[1]))
    scaled = cv2.resize(screen, (scale_w, h), interpolation=cv2.INTER_AREA)
    win = np.zeros((h + title_px, scale_w + 2 * side_px, 3), np.uint8)
    win[:title_px, :, :] = 200                      # title bar spans everything
    win[title_px:, side_px:side_px + scale_w] = scaled
    return win


@pytest.mark.parametrize("display, rect", [
    ((1080, 2400), None),                          # the guessed default
    ((1080, 2400), (0.0, 0.06, 1.0, 0.85)),        # a calibrated, non-9:16 rect
    ((1440, 3200), None),
])
def test_a_tap_target_lands_where_the_detector_sees_that_tile(display, rect):
    pytest.importorskip("cv2")
    from capture.window import PHONE_OUTPUT_SIZE, crop_phone_frame
    from live.actuator import tile_centre

    prof = DeviceProfile(name="t", display=display, game_rect=rect)
    W, H = display
    screen = np.full((H, W, 3), 90, np.uint8)       # lit, so the screen is found
    for tile in [(3, 5), (14, 20), (9, 27)]:
        t = tile_centre(*tile)
        dx, dy = prof.to_device(t.x, t.y)
        screen[max(0, dy - 6):dy + 6, max(0, dx - 6):dx + 6] = (255, 255, 255)

    window = _mirror_in_window(screen, scale_w=576)
    frame = crop_phone_frame(window, prof)
    assert (frame.shape[1], frame.shape[0]) == PHONE_OUTPUT_SIZE

    ow, oh = PHONE_OUTPUT_SIZE
    for tile in [(3, 5), (14, 20), (9, 27)]:
        t = tile_centre(*tile)
        ex = t.x * ow / CRBAB_DISPLAY[0]
        ey = t.y * oh / CRBAB_DISPLAY[1]
        y0, y1 = int(ey) - 4, int(ey) + 5
        x0, x1 = int(ex) - 4, int(ex) + 5
        assert frame[y0:y1, x0:x1].max() > 200, (
            f"tile {tile}: the tap mark is not where the detector frame puts it")


def test_the_emulator_rule_would_have_cut_a_tall_phone():
    """The control for the aspect parameter: the 9:16 default, applied to a
    20:9 screen, returns a box shorter than the screen."""
    from capture.window import find_game_rect
    screen = np.full((2400, 1080, 3), 90, np.uint8)
    window = _mirror_in_window(screen, scale_w=540)
    _x, _y, w, h = find_game_rect(window)
    assert h < round(w * 2400 / 1080) - 10
    _x, _y, w, h = find_game_rect(window, aspect=(1080, 2400))
    assert h == pytest.approx(w * 2400 / 1080, abs=2)


# --- the actuator in phone mode ----------------------------------------------

def test_phone_taps_are_sent_in_the_phones_own_pixels(monkeypatch, tmp_path):
    from live.actuator import AdbActuator, card_centre, engine_tile_centre
    fake_adb = tmp_path / "adb"
    fake_adb.write_text("")
    prof = DeviceProfile(name="tall", display=(1080, 2400), serial="R5CT")
    sent = []
    a = AdbActuator(dry_run=False, adb=fake_adb, device=prof)
    monkeypatch.setattr(a, "_shell", sent.append)
    try:
        assert a.raw is None and a.serial == "R5CT"
        a.play(1, 4, 12)
        a.flush()
    finally:
        a.close()
    card, tile = card_centre(1), engine_tile_centre(4, 12)
    cx, cy = prof.to_device(card.x, card.y)
    tx, ty = prof.to_device(tile.x, tile.y)
    assert f"input tap {cx} {cy}" in sent[0]
    assert f"input tap {tx} {ty}" in sent[0]
    assert a.taps == [card, tile]                  # the log stays CRBAB-space


def test_the_emulator_path_is_unchanged(monkeypatch, tmp_path):
    from live.actuator import AdbActuator, card_centre
    fake_adb = tmp_path / "adb"
    fake_adb.write_text("")
    a = AdbActuator(dry_run=False, adb=fake_adb, raw_touch=False)
    sent = []
    monkeypatch.setattr(a, "_shell", sent.append)
    try:
        a.play(0, 9, 8)
        a.flush()
    finally:
        a.close()
    c = card_centre(0)
    assert f"input tap {c.x} {c.y}" in sent[0]


# --- the loop's guards ------------------------------------------------------

def test_mvp_loop_refuses_to_act_through_an_uncalibrated_profile(monkeypatch, tmp_path):
    from live import mvp_loop
    path = DeviceProfile(name="p", display=(1080, 2400)).save(tmp_path / "p.json")
    monkeypatch.setattr(sys, "argv", ["mvp_loop", "--device", str(path), "--act"])
    with pytest.raises(SystemExit, match="not calibrated"):
        mvp_loop.main()


def test_mvp_loop_refuses_ensure_match_on_a_phone(monkeypatch, tmp_path):
    from live import mvp_loop
    path = DeviceProfile(name="p", display=(1080, 2400),
                         calibrated=True).save(tmp_path / "p.json")
    monkeypatch.setattr(sys, "argv",
                        ["mvp_loop", "--device", str(path), "--ensure-match"])
    with pytest.raises(SystemExit, match="emulator-only"):
        mvp_loop.main()


# --- the calibration overlay ---------------------------------------------------

def test_the_overlay_marks_the_hand_slots_where_the_taps_go():
    from PIL import Image

    sys.path.insert(0, str(__import__("conftest").PERCEPTION_ROOT / "tools"))
    from phone_check import draw_overlay
    from live.actuator import card_centre

    prof = DeviceProfile(name="tall", display=(1080, 2400))
    shot = Image.new("RGB", (540, 1200), (30, 30, 30))     # a half-scale screenshot
    out = np.asarray(draw_overlay(shot, prof))
    assert out.shape == (1200, 540, 3)
    c = card_centre(2)
    dx, dy = prof.to_device(c.x, c.y)
    ring = out[int(dy / 2) - 14:int(dy / 2) + 15, int(dx / 2) - 14:int(dx / 2) + 15]
    assert (ring == (255, 255, 0)).all(axis=2).any(), "no slot marker at the tap"
