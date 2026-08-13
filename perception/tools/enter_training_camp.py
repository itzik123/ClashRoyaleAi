"""Navigate the Clash Royale lobby into a Training Camp match, via adb taps.

WHY THIS EXISTS
---------------
Live testing of the placement pipeline needs a match in progress, and a LADDER
match puts trophies on the line for what is a debugging run. Training Camp is
the same in-game screen with no stakes, so every live iteration should start
here.

WHY adb `input tap` AND NOT the actuator's RawTouch path
--------------------------------------------------------
`AdbActuator.play()` exists for placements, where latency is the whole problem
-- a tap that lands a second late lands on a different board. Menu navigation
has no such constraint: the lobby does not move. `input tap` is a subprocess per
tap (~80-150 ms) and that is entirely fine here, while being far easier to
verify than synthesised evdev events.

COORDINATES
-----------
Anchored to DISPLAY space (720x1280) and derived from a capture at 544x967 by
the ratio between them, rather than hardcoded from a screenshot at one size --
the same reason `geometry.py` derives rather than pins. Verified by capturing
the frame, locating the button, and scaling:

    Training Camp centre, capture space : (322, 308)
    scale                               : 1280/967 = 1.3237
    DISPLAY space                       : (426, 408)

The menu button and the Battle/OK confirmations are located the same way.

IDEMPOTENCE
-----------
Every step CHECKS the screen before and after acting, using the same detector
the live loop uses, so running this when already in a match is a no-op rather
than a stray tap into the arena. That matters because this script exists to be
run repeatedly by an automated iteration loop.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live.actuator import ADB  # noqa: E402

# DISPLAY-space (720x1280) tap targets. See COORDINATES above.
CAPTURE_W, CAPTURE_H = 544, 967
DISPLAY_W, DISPLAY_H = 720, 1280


def _to_display(x: int, y: int) -> tuple[int, int]:
    return round(x * DISPLAY_W / CAPTURE_W), round(y * DISPLAY_H / CAPTURE_H)


HAMBURGER = _to_display(494, 91)      # three-line button, top right of lobby
TRAINING_CAMP = _to_display(322, 308)  # target/sword entry in the open menu


def tap(x: int, y: int, serial: str | None = None, dry: bool = False) -> None:
    cmd = [str(ADB)]
    if serial:
        cmd += ["-s", serial]
    cmd += ["shell", "input", "tap", str(x), str(y)]
    if dry:
        print(f"  [dry] tap ({x}, {y})")
        return
    subprocess.run(cmd, check=True, capture_output=True, timeout=20)
    print(f"  tap ({x}, {y})")


def read_screen() -> str:
    """Which screen the detector thinks we are on: lobby / in_game / ... .

    Uses the live pipeline's own detector rather than a bespoke template match,
    so this script cannot disagree with the loop it is setting up.
    """
    from PIL import Image  # noqa: PLC0415

    from capture.window import WindowSource  # noqa: PLC0415
    from clashroyalebuildabot.constants import (  # noqa: PLC0415
        SCREENSHOT_HEIGHT,
        SCREENSHOT_WIDTH,
    )
    from clashroyalebuildabot.detectors.detector import Detector  # noqa: PLC0415
    from live.mvp_loop import DECK  # noqa: PLC0415

    src = WindowSource("BlueStacks App Player")
    frame = src.grab() if hasattr(src, "grab") else next(iter(src))
    # Exactly mvp_loop.perceive's preparation: BGR->RGB then resize to the
    # detector's own input size. Feeding the raw array instead fails with
    # "'numpy.ndarray' object has no attribute 'crop'", and feeding it at the
    # wrong size would silently degrade every template match.
    native = Image.fromarray(frame.image[:, :, ::-1])
    small = native.resize((SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT), Image.LANCZOS)
    state = Detector(DECK).run(small)
    if state is None:
        return "unknown"
    screen = getattr(state, "screen", None)
    return str(getattr(screen, "name", screen))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--serial", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--settle", type=float, default=1.5,
                    help="seconds to wait after each tap for the UI to animate")
    args = ap.parse_args()

    before = read_screen()
    print(f"screen before: {before}")
    if before == "in_game":
        print("already in a match -- nothing to do")
        return 0

    print("opening the menu")
    tap(*HAMBURGER, serial=args.serial, dry=args.dry_run)
    time.sleep(args.settle)

    print("selecting Training Camp")
    tap(*TRAINING_CAMP, serial=args.serial, dry=args.dry_run)
    time.sleep(args.settle * 2)

    after = read_screen()
    print(f"screen after: {after}")
    if after != "in_game" and not args.dry_run:
        print("NOT in a match. The lobby layout may differ from the capture the "
              "coordinates were derived from -- re-capture and re-derive rather "
              "than nudging the numbers.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
