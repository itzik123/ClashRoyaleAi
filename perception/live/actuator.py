"""Place a card on the real screen: tile coordinates -> two taps.

THE UNTESTED JOINT
------------------
Everything upstream of this has been measured against recorded frames. This
has not run at all. It is also the one stage where being wrong is invisible
from the inside: a tap landing on the wrong tile produces a perfectly valid
GameState on the next frame, showing a unit somewhere the agent did not intend,
and nothing in the pipeline flags it. That is why it is worth closing the loop
here before anything upstream gets polished further.

WHY TWO TAPS AND NOT A DRAG
---------------------------
Clash Royale places a card by selecting it in the hand and then tapping the
target. `ClashRoyaleBuildABot` does exactly that (`Bot.play_action`), and this
follows it rather than inventing a drag gesture, because the tap pair is what
upstream has running against the real game.

TAPPING IS ASYNCHRONOUS, AND HAS TO BE
--------------------------------------
A tap costs ~410 ms and a placement is two of them, so acting inline cost the
decision loop ~1.7 s -- measured live, the loop fell from 1.0 Hz to ~0.6 Hz
exactly when the agent was most active. That breaks the thing the whole
pipelined design exists to protect: the policy is recurrent and was trained
stepping once per second, so a cadence that stretches whenever it plays feeds
the LSTM intervals it never saw.

Where the time goes, measured on this emulator:

    adb shell echo          one-shot 107 ms   persistent shell   3.2 ms
    adb shell input         one-shot 413 ms   persistent shell   412 ms

So it is NOT host-side process spawn. `input` is a Java program and Android
starts an app_process for every invocation; a persistent host channel collapses
trivial commands 107 -> 3 ms and does nothing at all for `input`.

That last point stopped being true once the tap stopped being `input` -- see
below. A persistent shell IS used now, because the raw path's cost is host-side
connection setup and nothing else: 129 ms one-shot against 50 ms down an
already-open shell.

Taps therefore move OFF the decision thread. That fixed the cadence; the
latency itself is fixed below.

CUTTING THE LATENCY: RAW EVDEV, NOT sendevent
---------------------------------------------
The obvious next step was `sendevent`, to skip the JVM. Measured, it is SLOWER:

    6 sendevent calls (one tap)        362 ms
    input tap                          307 ms

Each `sendevent` is its own process, and six device-side spawns cost more than
one JVM start. The cost was never the JVM specifically -- it is process count.

So the events are written to the device directly instead, as one 144-byte
payload, which is one process rather than six:

    one-shot   `echo <b64> | base64 -d > /dev/input/event4`     129 ms
    persistent shell, same command                               50 ms

TWO THINGS THIS GOT WRONG FIRST, both found by tapping a real button:

  * The panel is LANDSCAPE. `wm size` reports 1280x720 while the app renders
    720x1280 rotated onto it, and the touch device's axes are the PANEL's. The
    first attempt assumed the app's frame and landed a tap meant for the
    top-right hamburger on the top-left profile banner instead.

  * A contact needs DURATION. Press and release in a single write is a
    zero-length touch and does not register as a tap; it dismissed a menu
    rather than pressing the button under it.

An `input tap` fallback is kept, because the device path, the axis ranges and
the rotation are all properties of this emulator.

A placement is enqueued whole. Its two taps must not interleave with another
placement's, or a card-select lands against the wrong tile tap.

COORDINATES ARE ANDROID'S, NOT THE SCREEN'S
-------------------------------------------
`adb shell input tap` takes coordinates in the Android display space
(720x1280 here), NOT in captured-frame pixels and NOT in desktop pixels. Three
spaces are in play and confusing them silently misplaces every card:

    Android display   720 x 1280   what input tap wants, and what CRBAB's
                                   TILE_INIT_*/TILE_* constants are in
    captured frame    549 x 976    what the detector and readers see
    desktop           1920 x 1020  the WGC surface, physical pixels

The conversion is CRBAB's own `_get_tile_centre` inverse, imported rather than
re-derived -- a second copy of TILE_INIT_X here is precisely the duplicated
constant CLAUDE.md names as having gone stale twice.
"""
from __future__ import annotations

import base64
import queue
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from clashroyalebuildabot.constants import (
    DISPLAY_CARD_DELTA_X,
    DISPLAY_CARD_HEIGHT,
    DISPLAY_CARD_INIT_X,
    DISPLAY_CARD_WIDTH,
    DISPLAY_CARD_Y,
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    TILE_HEIGHT,
    TILE_INIT_X,
    TILE_INIT_Y,
    TILE_WIDTH,
)

from live.adapter import TILE_Y_OFFSET

ADB = Path(r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe")

TOUCH_DEVICE = "/dev/input/event4"          # "BlueStacks Virtual Touch"
ABS_MT_POSITION_X, ABS_MT_POSITION_Y = 53, 54
EV_ABS, EV_SYN = 3, 0
SYN_REPORT, SYN_MT_REPORT = 0, 2

# The device reports both axes as 0..32767 regardless of the panel's shape.
ABS_MAX = 32767

# How long the contact is held. Press and release in one write is a
# zero-duration touch and does not register; 60 ms was verified against real
# buttons. Cheap -- it is an on-device sleep inside a round trip we are making
# anyway, not another round trip.
TOUCH_HOLD_S = 0.06

# Marks the end of a command down the persistent shell.
_SENTINEL = "__ACT_DONE__"


def _input_event(ev_type: int, code: int, value: int) -> bytes:
    """One `struct input_event`, 64-bit layout: two 8-byte timeval fields,
    then type, code, value. The kernel timestamps writes itself, so the time
    fields are zero."""
    return struct.pack("<qqHHi", 0, 0, ev_type, code, value)

# Between the card tap and the tile tap. The card has to register as selected
# before the placement lands; without a gap the second tap can be swallowed.
# Both taps go in ONE `adb shell`, so the ~410 ms `input` costs sit between
# them anyway and this is belt and braces.
TAP_GAP_S = 0.08


class ActuationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Tap:
    x: int
    y: int


def card_centre(slot: int) -> Tap:
    """Android coordinates of hand slot 0-3."""
    if not 0 <= slot <= 3:
        raise ActuationError(f"hand slot must be 0-3, got {slot}")
    x = DISPLAY_CARD_INIT_X + DISPLAY_CARD_WIDTH / 2 + slot * DISPLAY_CARD_DELTA_X
    y = DISPLAY_CARD_Y + DISPLAY_CARD_HEIGHT / 2
    return Tap(int(round(x)), int(round(y)))


def tile_centre(tile_x: int, tile_y: int) -> Tap:
    """Android coordinates of a DETECTOR-frame tile (18x32).

    Detector frame, not engine frame. The engine's board is 18x34 with an extra
    row behind each King, so anything holding engine coordinates must convert
    first -- `engine_tile_centre` does that. Mixing the two shifts every
    placement by one row, which is the bug class this project has already paid
    for once.
    """
    x = TILE_INIT_X + (tile_x + 0.5) * TILE_WIDTH
    y = DISPLAY_HEIGHT - TILE_INIT_Y - (tile_y + 0.5) * TILE_HEIGHT
    return Tap(int(round(x)), int(round(y)))


# The detector frame is 18x32; the engine frame is 18x34 (an extra row behind
# each King). engine_tile_centre subtracts TILE_Y_OFFSET to convert, so engine
# rows outside [TILE_Y_OFFSET, TILE_Y_OFFSET + DETECTOR_ROWS) have NO detector
# row and therefore no tappable pixel -- their "centre" lands outside the arena
# rectangle entirely.
DETECTOR_ROWS = 32


def engine_row_is_tappable(tile_y: int) -> bool:
    """Does this ENGINE row correspond to a real, tappable arena row?

    Measured, not assumed: with TILE_Y_OFFSET = 1, engine row 0 converts to
    detector row -1, whose centre is pixel y=1018 against an arena bottom edge
    of DISPLAY_HEIGHT - TILE_INIT_Y = 1003.81. The tap lands BELOW the arena, in
    the dead strip above the card tray, so the game silently drops the
    placement -- the card is deselected and no unit is deployed.

    That is not a hypothetical. A 180 s live match issued 25 placements, 5 of
    them on engine row 0, and reported "18 issued plays never confirmed". The
    policy is free to choose row 0 because model.py's placement_mask is built
    from the ENGINE's own bounds, which happily include it -- the engine really
    does have that row, it is simply not reachable through this screen mapping.

    Derived from the geometry rather than hardcoded to `y > 0` so it stays
    correct if TILE_Y_OFFSET is ever re-fitted. This is the same discipline
    CLAUDE.md requires of engine constants: live where derivable.
    """
    return 0 <= tile_y - TILE_Y_OFFSET < DETECTOR_ROWS


def engine_tile_centre(tile_x: int, tile_y: int) -> Tap:
    """Android coordinates of an ENGINE-frame tile (18x34).

    The policy emits engine coordinates, so this is the one a live loop wants.
    Note the offset is still unverified -- see adapter.TILE_Y_OFFSET -- so a
    systematic one-row placement error is possible and would show up here
    first, as cards landing a row nearer or further than intended.
    """
    return tile_centre(tile_x, tile_y - TILE_Y_OFFSET)


class RawTouch:
    """Screen coordinates -> a shell command that writes evdev events.

    Stateless and pure: it builds a command string and never talks to adb, so
    the whole coordinate mapping is testable without an emulator.
    """

    def __init__(self, panel_w: int, panel_h: int,
                 screen_w: int = DISPLAY_WIDTH, screen_h: int = DISPLAY_HEIGHT,
                 device: str = TOUCH_DEVICE):
        self.panel_w, self.panel_h = panel_w, panel_h
        self.screen_w, self.screen_h = screen_w, screen_h
        self.device = device
        # The app is rotated onto the panel when one is portrait and the other
        # landscape. Assuming otherwise put a tap meant for the top-right
        # corner into the top-left one.
        self.rotated = (panel_w > panel_h) != (screen_w > screen_h)

    def to_device(self, x: int, y: int) -> tuple[int, int]:
        """Screen pixel -> the device's own axis units."""
        if self.rotated:
            # Screen y runs along the panel's x, backwards; screen x along its y.
            fx = (self.panel_w - y * self.panel_w / self.screen_h) / self.panel_w
            fy = x / self.screen_w
        else:
            fx = x / self.screen_w
            fy = y / self.screen_h
        clamp = lambda f: min(max(f, 0.0), 1.0)                  # noqa: E731
        return (int(round(clamp(fx) * ABS_MAX)),
                int(round(clamp(fy) * ABS_MAX)))

    def _write(self, payload: bytes) -> str:
        return (f"echo {base64.b64encode(payload).decode()} | "
                f"base64 -d > {self.device}")

    def _press(self, x: int, y: int) -> bytes:
        ex, ey = self.to_device(x, y)
        return b"".join((
            _input_event(EV_ABS, ABS_MT_POSITION_X, ex),
            _input_event(EV_ABS, ABS_MT_POSITION_Y, ey),
            _input_event(EV_SYN, SYN_MT_REPORT, 0),
            _input_event(EV_SYN, SYN_REPORT, 0)))

    def _release(self) -> bytes:
        # An empty frame: a SYN_MT_REPORT reporting no contacts, then SYN.
        return b"".join((_input_event(EV_SYN, SYN_MT_REPORT, 0),
                         _input_event(EV_SYN, SYN_REPORT, 0)))

    def tap_script(self, x: int, y: int) -> str:
        return "; ".join((self._write(self._press(x, y)),
                          f"sleep {TOUCH_HOLD_S}",
                          self._write(self._release())))

    def placement_script(self, card: Tap, target: Tap) -> str:
        """Both taps of a placement in ONE round trip.

        The waits happen on the device, inside a trip we are making anyway, so
        the gap costs nothing extra.
        """
        return "; ".join((self.tap_script(card.x, card.y),
                          f"sleep {TAP_GAP_S}",
                          self.tap_script(target.x, target.y)))


class AdbActuator:
    """Taps via adb, off the caller's thread. Dry run only records intent.

    Dry run is the default on purpose: this module can misplace real cards in a
    real match, so acting has to be asked for explicitly rather than being what
    happens if a caller forgets a flag.
    """

    def __init__(self, dry_run: bool = True, adb: Path = ADB,
                 serial: str | None = None, raw_touch: bool = True):
        self.dry_run = dry_run
        self.adb = Path(adb)
        self.serial = serial
        self.taps: list[Tap] = []
        self.dropped = 0
        self.errors = 0
        self.last_error: BaseException | None = None
        self.raw: RawTouch | None = None
        self._proc: subprocess.Popen | None = None
        if not dry_run and not self.adb.exists():
            raise ActuationError(f"adb not found at {self.adb}")
        if raw_touch and not dry_run:
            # Probed once rather than assumed: the device path, the axis ranges
            # and the rotation are all properties of THIS emulator, and being
            # wrong about any of them taps the wrong place rather than failing.
            self.raw = self._probe_raw_touch()

        # Depth ONE, and a full queue drops rather than blocks. Both are
        # deliberate. Blocking would put the latency straight back on the
        # decision thread, and queueing would let a placement land seconds
        # after the board it was chosen for -- by which time it is not a late
        # move, it is a different and probably wrong one. A drop is counted
        # and visible; a silent late tap is neither.
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        if not dry_run:
            self._worker = threading.Thread(target=self._run, name="actuator",
                                            daemon=True)
            self._worker.start()

    # -- public ---------------------------------------------------------------

    def play(self, slot: int, tile_x: int, tile_y: int, *,
             engine_frame: bool = True) -> tuple[Tap, Tap]:
        """Queue "select this hand slot, then tap this tile". Returns at once.

        The pair is enqueued WHOLE. Interleaving two placements' taps would
        pair a card-select with the wrong tile tap and put the card somewhere
        nobody asked for.
        """
        card = card_centre(slot)
        target = (engine_tile_centre(tile_x, tile_y) if engine_frame
                  else tile_centre(tile_x, tile_y))
        # Recorded synchronously, at intent, so `taps` is deterministic for
        # callers and tests regardless of when the worker gets to it.
        self.taps.extend((card, target))
        if not self.dry_run:
            try:
                self._q.put_nowait((card, target))
            except queue.Full:
                self.dropped += 1
        return card, target

    def flush(self) -> None:
        """Block until every queued placement has actually been sent.

        For callers that must observe the RESULT of a tap -- the placement
        probe screenshots the board and would otherwise photograph it before
        the card had landed. The decision loop must never call this; blocking
        is the exact thing this class was changed to stop doing.
        """
        if self._worker is not None:
            self._q.join()

    def close(self, timeout: float = 5.0) -> None:
        """Let an in-flight placement finish, then stop the worker."""
        if self._worker is None:
            return
        self._q.join()
        self._stop.set()
        self._worker.join(timeout=timeout)
        self._close_shell()

    @property
    def pending(self) -> int:
        return self._q.qsize()

    # -- internals ------------------------------------------------------------

    def _probe_raw_touch(self) -> RawTouch | None:
        """The panel's real geometry, or None to fall back to `input tap`."""
        try:
            cmd = [str(self.adb)]
            if self.serial:
                cmd += ["-s", self.serial]
            out = subprocess.run(cmd + ["shell", f"wm size; ls {TOUCH_DEVICE}"],
                                 capture_output=True, text=True,
                                 timeout=15).stdout
            if TOUCH_DEVICE not in out:
                return None
            size = [w for w in out.replace("x", " ").split()
                    if w.isdigit()]
            if len(size) < 2:
                return None
            return RawTouch(int(size[0]), int(size[1]))
        except Exception:                                   # noqa: BLE001
            return None

    @property
    def backend(self) -> str:
        return "raw-evdev" if self.raw is not None else "input-tap"

    def _adb(self, *args: str) -> list[str]:
        cmd = [str(self.adb)]
        if self.serial:
            cmd += ["-s", self.serial]
        return cmd + list(args)

    def _open_shell(self):
        """One long-lived `adb shell`, so a placement pays no connection cost.

        Measured on this box: the same command costs 129 ms as a one-shot
        `adb shell` and 50 ms down an already-open one. That is pure host-side
        connection setup, and it is ~80 ms off every placement without changing
        a single timing constant on the device.
        """
        return subprocess.Popen(
            self._adb("shell"), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)

    def _shell(self, script: str) -> None:
        try:
            self._shell_persistent(script)
            return
        except Exception:                                   # noqa: BLE001
            # A dead or wedged shell must not cost us the placement: drop it
            # and fall back, and the next call reopens. Silently retrying
            # forever down a broken pipe would look exactly like an actuator
            # that had stopped working.
            self._close_shell()
        done = subprocess.run(self._adb("shell", script), capture_output=True,
                              text=True, timeout=20)
        if done.returncode != 0:
            raise ActuationError(
                f"adb failed ({done.returncode}): {done.stderr.strip()}")

    def _shell_persistent(self, script: str) -> None:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = self._open_shell()
        # A sentinel rather than a fixed read: the script's own output is not
        # something this can predict, and reading a fixed number of lines would
        # desynchronise the stream the first time one of them printed anything.
        self._proc.stdin.write(f"{script}; echo {_SENTINEL}\n")
        self._proc.stdin.flush()
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                raise ActuationError("adb shell closed")
            if _SENTINEL in line:
                return
        raise ActuationError("adb shell timed out")

    def _close_shell(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()
        except Exception:                                   # noqa: BLE001
            pass
        proc.terminate()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                card, target = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                # One `adb shell` for the whole placement either way: two host
                # round trips buy nothing, and with raw touch the holds and the
                # gap happen on-device inside a trip we are making anyway.
                if self.raw is not None:
                    self._shell(self.raw.placement_script(card, target))
                else:
                    self._shell(f"input tap {card.x} {card.y}; "
                                f"sleep {TAP_GAP_S}; "
                                f"input tap {target.x} {target.y}")
            except Exception as exc:                        # noqa: BLE001
                # A failed tap must not kill the worker: the loop would then
                # look like it was still acting while nothing reached the game.
                self.errors += 1
                self.last_error = exc
            finally:
                self._q.task_done()
