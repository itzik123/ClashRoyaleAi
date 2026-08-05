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
trivial commands 107 -> 3 ms and does nothing at all for `input`. That is why
this does not use one.

Taps therefore move OFF the decision thread rather than being made faster.
Latency is unchanged, cadence is restored. Cutting the latency itself needs
`sendevent` against /dev/input/event4 ("BlueStacks Virtual Touch",
ABS_MT_POSITION 0..32767), which skips the JVM entirely -- a much larger and
riskier change, deliberately not bundled here.

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

import queue
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

from clashroyalebuildabot.constants import (
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

from live.adapter import TILE_Y_OFFSET

ADB = Path(r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe")

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


def engine_tile_centre(tile_x: int, tile_y: int) -> Tap:
    """Android coordinates of an ENGINE-frame tile (18x34).

    The policy emits engine coordinates, so this is the one a live loop wants.
    Note the offset is still unverified -- see adapter.TILE_Y_OFFSET -- so a
    systematic one-row placement error is possible and would show up here
    first, as cards landing a row nearer or further than intended.
    """
    return tile_centre(tile_x, tile_y - TILE_Y_OFFSET)


class AdbActuator:
    """Taps via adb, off the caller's thread. Dry run only records intent.

    Dry run is the default on purpose: this module can misplace real cards in a
    real match, so acting has to be asked for explicitly rather than being what
    happens if a caller forgets a flag.
    """

    def __init__(self, dry_run: bool = True, adb: Path = ADB,
                 serial: str | None = None):
        self.dry_run = dry_run
        self.adb = Path(adb)
        self.serial = serial
        self.taps: list[Tap] = []
        self.dropped = 0
        self.errors = 0
        self.last_error: BaseException | None = None
        if not dry_run and not self.adb.exists():
            raise ActuationError(f"adb not found at {self.adb}")

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

    @property
    def pending(self) -> int:
        return self._q.qsize()

    # -- internals ------------------------------------------------------------

    def _shell(self, script: str) -> None:
        cmd = [str(self.adb)]
        if self.serial:
            cmd += ["-s", self.serial]
        cmd += ["shell", script]
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if done.returncode != 0:
            raise ActuationError(
                f"adb failed ({done.returncode}): {done.stderr.strip()}")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                card, target = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                # One `adb shell` for the pair: two host round trips buy
                # nothing when each `input` already costs ~410 ms on-device.
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
