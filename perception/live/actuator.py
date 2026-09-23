"""Place a card on the real screen: tile coordinates -> two taps.

A tap on the wrong tile is invisible from inside the pipeline: the next frame
is a valid GameState showing a unit where the agent did not intend.

Select the card, then tap the target, as CRBAB's `Bot.play_action` does against
the real game.

Taps run off the decision thread. A tap costs ~400 ms and a placement is two,
and the policy is recurrent and was trained stepping once per second, so inline
taps would stretch its cadence exactly when it plays.

Events are written straight to the touch device as one 144-byte payload. `input
tap` starts a Java app_process per call, and `sendevent` is slower still (one
process per event); what costs is device-side process count. Down a persistent
`adb shell` the raw write is ~50 ms against ~129 ms one-shot, the difference
being host-side connection setup.

Two properties of this emulator matter: the panel is landscape (`wm size`
1280x720) while the app renders 720x1280 rotated onto it, and touch axes are
the panel's; and a contact needs duration, since press and release in one write
does not register as a tap. An `input tap` fallback is kept because the device
path, axis ranges and rotation are all emulator-specific.

A placement is enqueued whole, so its two taps never interleave with another's.

Three coordinate spaces are in play:

    Android display   720 x 1280   what taps want, and what CRBAB's
                                   TILE_INIT_*/TILE_* constants are in
    captured frame    549 x 976    what the detector and readers see
    desktop           1920 x 1020  the WGC surface, physical pixels

The conversion is the inverse of CRBAB's own `_get_tile_centre`, imported
rather than re-derived.
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

# How long the contact is held; 60 ms was verified against real buttons. An
# on-device sleep inside a round trip already being made.
TOUCH_HOLD_S = 0.06

# Marks the end of a command down the persistent shell.
_SENTINEL = "__ACT_DONE__"


def _input_event(ev_type: int, code: int, value: int) -> bytes:
    """One `struct input_event`, 64-bit layout: two 8-byte timeval fields,
    then type, code, value. The kernel timestamps writes itself, so the time
    fields are zero."""
    return struct.pack("<qqHHi", 0, 0, ev_type, code, value)

# Between the card tap and the tile tap, so the card registers as selected
# before the placement lands.
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
    """Android coordinates of a detector-frame tile (18x32).

    Not the engine frame, which is 18x34 with an extra row behind each King;
    engine coordinates go through `engine_tile_centre`. Mixing the two shifts
    every placement by a row.
    """
    x = TILE_INIT_X + (tile_x + 0.5) * TILE_WIDTH
    y = DISPLAY_HEIGHT - TILE_INIT_Y - (tile_y + 0.5) * TILE_HEIGHT
    return Tap(int(round(x)), int(round(y)))


# Engine rows outside [TILE_Y_OFFSET, TILE_Y_OFFSET + DETECTOR_ROWS) have no
# detector row, so their "centre" lands outside the arena.
DETECTOR_ROWS = 32


def engine_row_is_tappable(tile_y: int) -> bool:
    """Does this engine row correspond to a real, tappable arena row?

    The engine board is 34 rows and the arena 32, so with TILE_Y_OFFSET = 1
    engine row 0 is detector row -1: its tap lands in the strip above the card
    tray and the game drops the placement. Only row 0: engine rows 1..15 were
    verified live at 100% acceptance across columns 0, 9 and 17. Derived from
    the geometry, not hardcoded to `y > 0`, so it follows a re-fitted offset.
    """
    return 0 <= tile_y - TILE_Y_OFFSET < DETECTOR_ROWS


def engine_tile_centre(tile_x: int, tile_y: int) -> Tap:
    """Android coordinates of an engine-frame tile (18x34), which is what the
    policy emits. The offset is derived rather than measured
    (adapter.TILE_Y_OFFSET); a systematic one-row error would show up here
    first.
    """
    return tile_centre(tile_x, tile_y - TILE_Y_OFFSET)


class RawTouch:
    """Screen coordinates -> a shell command that writes evdev events. Pure:
    builds a string and never talks to adb, so the mapping is testable without
    an emulator.
    """

    def __init__(self, panel_w: int, panel_h: int,
                 screen_w: int = DISPLAY_WIDTH, screen_h: int = DISPLAY_HEIGHT,
                 device: str = TOUCH_DEVICE):
        self.panel_w, self.panel_h = panel_w, panel_h
        self.screen_w, self.screen_h = screen_w, screen_h
        self.device = device
        # The app is rotated onto the panel when one is portrait and the other
        # landscape.
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
        """Both taps of a placement in one round trip; the waits happen on-device.
        """
        return "; ".join((self.tap_script(card.x, card.y),
                          f"sleep {TAP_GAP_S}",
                          self.tap_script(target.x, target.y)))


class AdbActuator:
    """Taps via adb, off the caller's thread. Dry run, the default, only records
    intent: acting on a real match must be asked for.
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
            # Probed, not assumed: device path, axis ranges and rotation are
            # properties of this emulator, and being wrong taps the wrong place
            # rather than failing.
            self.raw = self._probe_raw_touch()

        # Depth one, and a full queue drops rather than blocks. Blocking puts
        # the latency back on the decision thread; queueing lets a placement
        # land seconds after the board it was chosen for. A drop is counted and
        # visible.
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        if not dry_run:
            self._worker = threading.Thread(target=self._run, name="actuator",
                                            daemon=True)
            self._worker.start()

    # --- public ---

    def play(self, slot: int, tile_x: int, tile_y: int, *,
             engine_frame: bool = True) -> tuple[Tap, Tap]:
        """Queue "select this hand slot, then tap this tile". Returns at once.
        Enqueued whole, so two placements' taps cannot interleave.
        """
        card = card_centre(slot)
        target = (engine_tile_centre(tile_x, tile_y) if engine_frame
                  else tile_centre(tile_x, tile_y))
        # Recorded at intent, so `taps` is deterministic regardless of when the
        # worker runs.
        self.taps.extend((card, target))
        if not self.dry_run:
            try:
                self._q.put_nowait((card, target))
            except queue.Full:
                self.dropped += 1
        return card, target

    def play_pixel(self, slot: int, px: int, py: int) -> tuple[Tap, Tap]:
        """Select a hand slot, then tap a raw screen pixel.

        Calibration only: `play()` goes through the tile conversion under test,
        so measuring where the deployable edge sits must bypass it.
        """
        card = card_centre(slot)
        target = Tap(int(px), int(py))
        self.taps.extend((card, target))
        if not self.dry_run:
            try:
                self._q.put_nowait((card, target))
            except queue.Full:
                self.dropped += 1
        return card, target

    def flush(self) -> None:
        """Block until every queued placement has been sent. For callers that
        observe a tap's result (the placement probe screenshots the board). The
        decision loop must never call this.
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

    # --- internals ---

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
        """One long-lived `adb shell`, so a placement pays no connection cost (~80
        ms per placement).
        """
        return subprocess.Popen(
            self._adb("shell"), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)

    def _shell(self, script: str) -> None:
        try:
            self._shell_persistent(script)
            return
        except Exception:                                   # noqa: BLE001
            # A dead or wedged shell must not cost the placement: drop it, fall
            # back, and reopen next call.
            self._close_shell()
        done = subprocess.run(self._adb("shell", script), capture_output=True,
                              text=True, timeout=20)
        if done.returncode != 0:
            raise ActuationError(
                f"adb failed ({done.returncode}): {done.stderr.strip()}")

    def _shell_persistent(self, script: str) -> None:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = self._open_shell()
        # A sentinel, not a fixed line count: the script's output is
        # unpredictable, and a fixed read would desynchronise the stream.
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
                # One `adb shell` for the whole placement; with raw touch the
                # holds and gap happen on-device.
                if self.raw is not None:
                    self._shell(self.raw.placement_script(card, target))
                else:
                    self._shell(f"input tap {card.x} {card.y}; "
                                f"sleep {TAP_GAP_S}; "
                                f"input tap {target.x} {target.y}")
            except Exception as exc:                        # noqa: BLE001
                # A failed tap must not kill the worker, or the loop would look
                # like it was acting while nothing reached the game.
                self.errors += 1
                self.last_error = exc
            finally:
                self._q.task_done()
