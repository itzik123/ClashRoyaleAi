"""Get the game into a Training Camp match, from wherever it currently is.

`enter_training_camp.py` already does the lobby -> Training Camp hop, but it
assumes it starts in the lobby with the menu closed, and it does not answer the
"Do you want to start a training match?" confirmation. Both assumptions break
the moment a script runs unattended: a probe that takes longer than a match
comes back to `bypass_end_of_game`, not to the lobby.

This is the unattended version -- a state machine over the screens the detector
already recognises, with an overall deadline, so an automated experiment can
say "put me in a match" and get one or a clean failure.

WHY IT NEVER TAPS THE LOBBY'S OWN CLICK POINT
---------------------------------------------
`Screens.LOBBY.click_xy` is (360, 1000): the BATTLE button, which queues a
LADDER match and puts trophies on a debugging run. The lobby is left via the
hamburger menu instead, exactly as `enter_training_camp.py` does it.

CAPTURE IS VIA adb, NOT THE WINDOW
----------------------------------
`enter_training_camp.read_screen()` grabs the BlueStacks window through WGC,
which needs the window visible and unoccluded. `adb exec-out screencap` does
not, so this keeps working while the desktop is doing something else -- which
is the whole point of an unattended loop.
"""
from __future__ import annotations

import io
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live.actuator import ADB  # noqa: E402

CAPTURE_W, CAPTURE_H = 544, 967
DISPLAY_W, DISPLAY_H = 720, 1280


def _to_display(x: int, y: int) -> tuple[int, int]:
    return round(x * DISPLAY_W / CAPTURE_W), round(y * DISPLAY_H / CAPTURE_H)


HAMBURGER = _to_display(494, 91)
TRAINING_CAMP = _to_display(322, 308)

# "Do you want to start a training match?" -> OK. Located by eye on a 720x1280
# screencap of the dialog and confirmed by the match starting; the dialog is
# fixed-size and centred, so this does not move.
CONFIRM_OK = (486, 738)


def screencap(adb: Path = ADB, serial: str | None = None) -> Image.Image:
    cmd = [str(adb)]
    if serial:
        cmd += ["-s", serial]
    raw = subprocess.run(cmd + ["exec-out", "screencap", "-p"],
                         capture_output=True, timeout=30).stdout
    return Image.open(io.BytesIO(raw)).convert("RGB")


def tap(x: int, y: int, adb: Path = ADB, serial: str | None = None) -> None:
    cmd = [str(adb)]
    if serial:
        cmd += ["-s", serial]
    subprocess.run(cmd + ["shell", "input", "tap", str(x), str(y)],
                   check=True, capture_output=True, timeout=20)


def read_screen(detector, adb: Path = ADB, serial: str | None = None):
    """(screen name, State). Uses the live pipeline's own detector."""
    from clashroyalebuildabot.constants import (  # noqa: PLC0415
        SCREENSHOT_HEIGHT,
        SCREENSHOT_WIDTH,
    )
    img = screencap(adb, serial)
    state = detector.run(img.resize((SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT),
                                    Image.LANCZOS))
    if state is None:
        return "unknown", None
    return str(getattr(state.screen, "name", state.screen)), state


def ensure_in_match(detector, adb: Path = ADB, serial: str | None = None,
                    timeout_s: float = 180.0, verbose: bool = True):
    """Block until a Training Camp match is running. Returns the in-game State.

    Idempotent: called while already in a match it reads one frame and returns.
    """
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        screen, state = read_screen(detector, adb, serial)
        if verbose and screen != last:
            print(f"  [nav] screen={screen}")
            last = screen

        if screen == "in_game":
            return state
        if screen in ("end_of_game", "bypass_end_of_game"):
            from clashroyalebuildabot.namespaces.screens import Screens  # noqa: PLC0415
            target = (Screens.END_OF_GAME if screen == "end_of_game"
                      else Screens.BYPASS_END_OF_GAME)
            tap(*target.click_xy, adb=adb, serial=serial)
            time.sleep(2.0)
            continue
        if screen == "lobby":
            tap(*HAMBURGER, adb=adb, serial=serial)
            time.sleep(1.5)
            tap(*TRAINING_CAMP, adb=adb, serial=serial)
            time.sleep(1.5)
            # The confirmation only appears sometimes (it is suppressed once
            # per session in some versions), so this tap is unconditional and
            # harmless: on the loading screen it lands on empty background.
            tap(*CONFIRM_OK, adb=adb, serial=serial)
            time.sleep(6.0)
            continue
        # unknown: a transition, a loading screen, or a dialog we do not model.
        # Waiting is right -- tapping blindly is how a script buys a chest.
        time.sleep(1.5)
    raise TimeoutError(f"could not reach a match within {timeout_s:.0f}s "
                       f"(last screen: {last})")


def main() -> int:
    from clashroyalebuildabot.detectors.detector import Detector  # noqa: PLC0415
    from clashroyalebuildabot.namespaces.cards import Cards  # noqa: PLC0415
    deck = [Cards.VALKYRIE, Cards.ARCHERS, Cards.MINIONS, Cards.CANNON,
            Cards.FIREBALL, Cards.GIANT, Cards.MUSKETEER, Cards.MINIPEKKA]
    state = ensure_in_match(Detector(deck))
    print(f"in a match: elixir {state.numbers.elixir.number}   "
          f"hand {[c.name for c in state.cards[1:5]]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
