"""Get the game into a Training Camp match, from wherever it currently is.

The single navigation path: a state machine over the screens the detector
recognises, with an overall deadline, so an unattended experiment can say "put
me in a match" and get one or a clean failure. It starts from any screen (a
probe that outlasts a match returns to `bypass_end_of_game`, not the lobby) and
answers the "start a training match?" confirmation.

It never taps the lobby's own click point: `Screens.LOBBY.click_xy` (360, 1000)
is the BATTLE button, which queues a ladder match. The lobby is left via the
hamburger menu.

Capture is via adb, not the window: WGC needs the BlueStacks window visible,
`adb exec-out screencap` does not.
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

# "Do you want to start a training match?" -> OK. Located on a 720x1280
# screencap; the dialog is fixed-size and centred.
CONFIRM_OK = (486, 738)


#: PNG signature. `raw` is scanned for it rather than assumed to start with
#: it; see decode_screencap.
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def decode_screencap(raw: bytes, returncode: int = 0,
                     stderr: bytes = b"") -> Image.Image:
    """Turn `adb exec-out screencap -p` stdout into an image.

    On the first call after the adb daemon was down, HD-Adb.exe prepends its
    "daemon not running. starting it now" banner to stdout (85 bytes ahead of
    the PNG, stderr empty, exit code 0), and PIL raises
    `UnidentifiedImageError`, which reads like a corrupt capture. It bites only
    the first call of a session, so anyone who ran `adb devices` first never
    sees it. Seeking the magic rather than stripping known banner text survives
    whatever a future adb announces.
    """
    if returncode != 0:
        raise RuntimeError(
            f"adb screencap failed (exit {returncode}): "
            f"{stderr.decode('utf-8', 'replace').strip() or 'no stderr'}")
    start = raw.find(_PNG_MAGIC)
    if start < 0:
        head = raw[:200].decode("utf-8", "replace").strip()
        raise RuntimeError(
            f"adb screencap returned {len(raw)} bytes with no PNG signature. "
            f"Is the emulator running and authorised? First bytes: {head!r}")
    return Image.open(io.BytesIO(raw[start:])).convert("RGB")


_daemon_warmed = False


def _warm_daemon(adb: Path, serial: str | None) -> None:
    """Start the adb daemon on its own, before any call whose stdout we parse.

    The start banner contaminates the first call's stdout, and a cold start
    plus device enumeration can take over 60 s, which folded into a capture
    becomes a mid-navigation TimeoutExpired. `start-server` is idempotent and
    free when warm. Does not raise: `decode_screencap` gives the better
    diagnosis with the bytes in hand.
    """
    global _daemon_warmed
    if _daemon_warmed:
        return
    cmd = [str(adb)]
    if serial:
        cmd += ["-s", serial]
    try:
        subprocess.run(cmd + ["start-server"], capture_output=True, timeout=120)
    except (subprocess.TimeoutExpired, OSError):
        pass          # the capture below produces the real diagnosis
    _daemon_warmed = True


def screencap(adb: Path = ADB, serial: str | None = None) -> Image.Image:
    _warm_daemon(adb, serial)
    cmd = [str(adb)]
    if serial:
        cmd += ["-s", serial]
    # 60 s: the first capture after the emulator boots is slow even with a warm
    # daemon.
    p = subprocess.run(cmd + ["exec-out", "screencap", "-p"],
                       capture_output=True, timeout=60)
    return decode_screencap(p.stdout, p.returncode, p.stderr)


def tap(x: int, y: int, adb: Path = ADB, serial: str | None = None) -> None:
    cmd = [str(adb)]
    if serial:
        cmd += ["-s", serial]
    subprocess.run(cmd + ["shell", "input", "tap", str(x), str(y)],
                   check=True, capture_output=True, timeout=20)


def read_screen(detector, adb: Path = ADB, serial: str | None = None):
    """(screen name, State), using the live pipeline's own detector."""
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
    """Block until a Training Camp match is running; returns the in-game State.
    Idempotent: in a match it reads one frame and returns.
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
            # The confirmation appears only sometimes, so this tap is
            # unconditional; on the loading screen it lands on empty
            # background.
            tap(*CONFIRM_OK, adb=adb, serial=serial)
            time.sleep(6.0)
            continue
        # Unknown: a transition, a loading screen or an unmodelled dialog.
        # Wait; tapping blindly is how a script buys a chest.
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
