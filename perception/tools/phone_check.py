"""Preflight for playing on a PHYSICAL phone (docs/TODO.md item P0).

    python perception/tools/phone_check.py                      # checks only
    python perception/tools/phone_check.py --save-profile myphone
    python perception/tools/phone_check.py --profile myphone --overlay out.png
    python perception/tools/phone_check.py --profile myphone --tap-at 360 80 --taps 5

Checks, in the order a fresh setup fails them:
  1. adb is found (platform-tools; see live/device.find_adb)
  2. exactly one device is attached and AUTHORIZED (or --serial picks one)
  3. its display size (`wm size`), which the profile is built from
  4. adb round-trip latency: `adb shell true`, one-shot, N times

`--save-profile NAME` writes config/devices/NAME.json with the GUESSED game
rect (live/device.default_game_rect) and `calibrated: false`.

`--overlay OUT.png` is the calibration check. It takes one screenshot of the
phone and draws, through the profile's own geometry:
  * the game rect: what the detector will see;
  * the 18x32 tile grid: where `tile_centre` taps land;
  * the four hand-slot tap points: where `card_centre` taps land.
Open a Training Camp match first. The grid must sit on the arena's own tiles
and the four dots on the four cards. If they do not, edit `game_rect` in the
profile (fractions of the screen: x, y, width, height), re-run, and set
`"calibrated": true` only once they do. `mvp_loop --act` refuses an
uncalibrated profile.

`--tap-at X Y` TAPS THE PHONE, so it is opt-in: it measures `input tap`
latency at a CRBAB-space point you choose (pick somewhere harmless, such as
the empty sky above the arena on the home screen).
"""
from __future__ import annotations

import argparse
import io
import statistics
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live.device import (  # noqa: E402
    CRBAB_DISPLAY,
    DeviceError,
    DeviceProfile,
    find_adb,
    parse_wm_size,
    require_device,
    run_adb,
)


def time_round_trips(adb: Path, serial: str | None, n: int = 10) -> list[float]:
    """Milliseconds per one-shot `adb shell true`: the floor under every tap
    that does not use a persistent shell."""
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        run_adb(adb, serial, "shell", "true")
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def screenshot(adb: Path, serial: str | None):
    """One PNG off the phone. `exec-out screencap` is 1.4-4.9 s a frame, far
    too slow for the loop (capture/window.py), and fine for a one-off
    calibration image."""
    from PIL import Image  # noqa: PLC0415
    cmd = [str(adb)] + (["-s", serial] if serial else []) + [
        "exec-out", "screencap", "-p"]
    data = subprocess.run(cmd, capture_output=True, timeout=30, check=True).stdout
    return Image.open(io.BytesIO(data)).convert("RGB")


def draw_overlay(image, profile: DeviceProfile):
    """Draw the profile's geometry onto a FULL-SCREEN screenshot at any scale.
    Returns a new image. Everything goes through `profile.to_device`, the
    function the actuator taps through, so the overlay shows the real taps."""
    from PIL import ImageDraw  # noqa: PLC0415
    from clashroyalebuildabot.constants import (  # noqa: PLC0415
        DISPLAY_HEIGHT,
        TILE_HEIGHT,
        TILE_INIT_X,
        TILE_INIT_Y,
        TILE_WIDTH,
    )

    from live.actuator import card_centre  # noqa: PLC0415

    img = image.copy()
    sx = img.width / profile.display[0]
    sy = img.height / profile.display[1]
    draw = ImageDraw.Draw(img)

    def px(x, y):
        dx, dy = profile.to_device(x, y)
        return dx * sx, dy * sy

    gx, gy, gw, gh = profile.game_rect_px()
    draw.rectangle([gx * sx, gy * sy, (gx + gw) * sx - 1, (gy + gh) * sy - 1],
                   outline=(255, 0, 255), width=3)
    # The 18x32 tile grid, the inverse of actuator.tile_centre.
    for col in range(19):
        x = TILE_INIT_X + col * TILE_WIDTH
        top = DISPLAY_HEIGHT - TILE_INIT_Y - 32 * TILE_HEIGHT
        bottom = DISPLAY_HEIGHT - TILE_INIT_Y
        draw.line([px(x, top), px(x, bottom)], fill=(0, 255, 255), width=1)
    for row in range(33):
        y = DISPLAY_HEIGHT - TILE_INIT_Y - row * TILE_HEIGHT
        draw.line([px(TILE_INIT_X, y), px(TILE_INIT_X + 18 * TILE_WIDTH, y)],
                  fill=(0, 255, 255), width=1)
    for slot in range(4):
        c = card_centre(slot)
        x, y = px(c.x, c.y)
        draw.ellipse([x - 12, y - 12, x + 12, y + 12], outline=(255, 255, 0),
                     width=4)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adb", default=None)
    ap.add_argument("--serial", default=None)
    ap.add_argument("--profile", default=None, help="profile name or path")
    ap.add_argument("--save-profile", default=None, metavar="NAME")
    ap.add_argument("--force", action="store_true",
                    help="let --save-profile overwrite a CALIBRATED profile")
    ap.add_argument("--overlay", type=Path, default=None, metavar="OUT.png")
    ap.add_argument("--tap-at", type=int, nargs=2, default=None,
                    metavar=("X", "Y"), help="CRBAB 720x1280 point; TAPS the phone")
    ap.add_argument("--taps", type=int, default=5)
    args = ap.parse_args()

    try:
        adb = find_adb(args.adb)
        print(f"[ok] adb: {adb}")
        print("     " + run_adb(adb, None, "version").splitlines()[0])
        dev = require_device(adb, args.serial)
        serial = dev.serial
        print(f"[ok] device: {serial} {dev.model}".rstrip())
        size = parse_wm_size(run_adb(adb, serial, "shell", "wm", "size"))
        print(f"[ok] display: {size[0]}x{size[1]} (portrait); CRBAB space is "
              f"{CRBAB_DISPLAY[0]}x{CRBAB_DISPLAY[1]}")
        trips = time_round_trips(adb, serial)
        print(f"[ok] adb round trip: median {statistics.median(trips):.0f} ms, "
              f"max {max(trips):.0f} ms over {len(trips)} (one-shot `shell true`)")
    except DeviceError as exc:
        print(f"[FAIL] {exc}")
        return 1

    profile = None
    if args.save_profile:
        path = Path(args.save_profile)
        existing = None
        try:
            existing = DeviceProfile.load(args.save_profile)
        except DeviceError:
            pass
        if existing is not None and existing.calibrated and not args.force:
            print(f"[FAIL] {args.save_profile!r} is calibrated; refusing to "
                  "overwrite it without --force")
            return 1
        profile = DeviceProfile(name=path.stem, display=size, serial=serial)
        out = profile.save(path if path.suffix else None)
        print(f"[ok] wrote {out}: game_rect {profile.game_rect} (GUESSED, "
              "calibrated=false). Next: --overlay.")
    elif args.profile:
        profile = DeviceProfile.load(args.profile)
        if profile.display != size:
            print(f"[WARN] profile display {profile.display} != the phone's "
                  f"{size}; re-save the profile")

    if args.overlay:
        if profile is None:
            print("[FAIL] --overlay needs --profile or --save-profile")
            return 1
        img = draw_overlay(screenshot(adb, serial), profile)
        args.overlay.parent.mkdir(parents=True, exist_ok=True)
        img.save(args.overlay)
        print(f"[ok] overlay: {args.overlay}. Grid on the arena tiles and "
              "dots on the four cards means calibrated.")

    if args.tap_at:
        if profile is None:
            print("[FAIL] --tap-at needs --profile")
            return 1
        x, y = profile.to_device(*args.tap_at)
        times = []
        for _ in range(max(1, args.taps)):
            t0 = time.perf_counter()
            run_adb(adb, serial, "shell", "input", "tap", str(x), str(y))
            times.append((time.perf_counter() - t0) * 1000.0)
        print(f"[ok] input tap at device ({x},{y}): median "
              f"{statistics.median(times):.0f} ms, max {max(times):.0f} ms. "
              "A placement is two taps plus the gap.")

    if profile is not None and not profile.calibrated:
        print("[WARN] profile is NOT calibrated: mvp_loop will watch but refuse "
              "--act")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
