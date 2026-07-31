"""Live frames from the emulator window, via Windows.Graphics.Capture.

WHY NOT ADB, MEASURED
---------------------
`adb exec-out screencap` is the obvious way to get native-resolution frames and
it is far too slow to be in a control loop:

    screencap -p (PNG)   2243 - 4886 ms   (10 samples)
    screencap    (raw)   1414 - 1885 ms   (6 samples)

The policy acts once per second and the detector alone costs ~800 ms of that,
so the capture budget is ~200 ms. ADB misses it by an order of magnitude.

WHY NOT A SCREEN-REGION GRAB
----------------------------
`Graphics.CopyFromScreen` reads the composited desktop, so it returns whatever
is on TOP of the emulator -- verified, it came back with an unrelated window
covering BlueStacks. That would make the machine unusable during a run and
would let any notification popup corrupt a frame.

Windows.Graphics.Capture reads the window's own surface instead. Measured
against BlueStacks while it was fully covered: frames arrive with content,
1920x1020, mean 85 ms between them. BlueStacks renders through D3D, which is
exactly the case where PrintWindow tends to return black, so this was tested
rather than assumed.

THE BUFFER IS NOT THE GAME
--------------------------
The captured surface is the whole emulator window: a title bar on top, a
toolbar strip down the right, and the portrait game pillarboxed in the middle
against black. The detector must never see the chrome -- it already invents
Knights from the two player avatars inside the arena (see board_filter.py), and
a toolbar full of icons is more of the same. So every frame is cropped to the
game rect, found per-frame rather than hardcoded, because the window can be
moved or resized under us.

DPI IS WHY THE BUFFER IS BIGGER THAN THE WINDOW
-----------------------------------------------
`GetWindowRect` reported 1536x816 while the captured surface is 1920x1020: the
display runs at 125%, and WGC hands back PHYSICAL pixels. That is 1.25x more
detail than the logical window rect suggests, and it is the reason the game
area comes out at 0.76 of native 720x1280 rather than the 0.64 the logical
rect implies. Never size anything from GetWindowRect here.

RESOLUTION LOSS IS NOT THE PROBLEM IT LOOKS LIKE
------------------------------------------------
The game renders at ~549x976 on this display against the 720x1280 that every
constant in unit_hp.py was fitted on. Measured by re-scoring the HP reader
against its 60 hand labels at each scale, the cost is inside the noise:

    scale 1.000 (720x1280)   precision 0.98   recall 0.56   F1 0.72
    scale 0.850              0.98   0.65   0.78
    scale 0.750              0.98   0.53   0.69
    scale 0.637              0.97   0.50   0.66

0.85 and 0.55 both score ABOVE native, so the spread is sampling noise on ~15
weighted positives, not a resolution effect. Downscaling preserves the
fill/track ratio, which is the only thing an HP fraction reads.
"""
from __future__ import annotations

import threading
import time
from typing import Iterator

import numpy as np

from capture.source import Frame, FrameSource

# A pillarbox pixel is not merely dark, it is black. The threshold is above
# pure black only to absorb compression noise on the surface.
LETTERBOX_MAX = 24

# The emulator preserves the Android panel's aspect, so the game rect's HEIGHT
# is derived from its measured WIDTH rather than detected. Detecting it fails:
# the title bar spans the game's own columns and is not black, so a row scan
# runs straight through it.
NATIVE_WIDTH, NATIVE_HEIGHT = 720, 1280

# Refuse to run below this rather than silently deliver a stale board. One
# decision per second is the policy's cadence, and a control loop fed frames
# slower than it acts is making decisions about the past.
DEFAULT_MIN_FPS = 2.0

# Rows are subsampled when scanning for the pillarbox. The bars are uniform
# black over hundreds of rows, so every 4th row decides the same columns for a
# quarter of the work -- and this scan runs on the hot path.
RECT_ROW_STEP = 4

# How often to re-derive the game rect even when the surface size has not
# changed. Measured at 75 ms on a 1920x1020 buffer, which is far too expensive
# per frame -- it was 74.7 of the 85 ms `read()` cost, and 29% of a recorder
# frame. A resize changes the surface shape and is caught immediately; this
# interval only covers a letterbox change at constant size, which needs the
# Android app itself to change aspect.
RECT_REVALIDATE_S = 5.0


class GameRectError(RuntimeError):
    """The game area could not be located inside the captured surface."""


def find_game_rect(buffer: np.ndarray) -> tuple[int, int, int, int]:
    """(x, y, w, h) of the pillarboxed game inside a full window capture.

    The width comes from the widest run of non-black columns in a band across
    the MIDDLE of the surface -- middle, because the title bar and the toolbar
    both span the full width and would join every run into one. The height then
    follows from the Android aspect and is anchored to the bottom of the
    surface, since the chrome is on top.
    """
    if buffer.ndim != 3 or buffer.shape[2] < 3:
        raise GameRectError(f"expected an HxWx3 image, got {buffer.shape}")
    height, width = buffer.shape[:2]
    band = buffer[int(height * 0.25):int(height * 0.75):RECT_ROW_STEP, :, :3]
    if band.size == 0:
        raise GameRectError(f"surface too small to sample: {buffer.shape}")

    lit = band.max(axis=(0, 2)) > LETTERBOX_MAX
    runs = _runs(lit)
    if not runs:
        raise GameRectError(
            "no non-black columns in the middle band -- the window is likely "
            "minimised or the emulator is not rendering")
    x0, x1 = max(runs, key=lambda r: r[1] - r[0])
    game_w = x1 - x0

    game_h = int(round(game_w * NATIVE_HEIGHT / NATIVE_WIDTH))
    rows_lit = np.flatnonzero(
        buffer[:, x0:x1, :3].max(axis=(1, 2)) > LETTERBOX_MAX)
    bottom = int(rows_lit.max()) + 1 if rows_lit.size else height
    y0 = max(0, bottom - game_h)
    return x0, y0, game_w, min(game_h, bottom - y0)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True spans of a 1-D mask, as [start, end) pairs."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


class WindowSource(FrameSource):
    """Live frames from a named window, cropped to the game area.

    Capture runs on its own thread and keeps only the LATEST frame. A control
    loop wants the current board, never a backlog: queueing frames would mean
    that after any hitch the agent acts on a board that has already changed,
    which is worse than skipping the hitch entirely.
    """

    def __init__(self, window_name: str = "BlueStacks App Player", *,
                 min_fps: float = DEFAULT_MIN_FPS, warmup_s: float = 3.0):
        try:
            from windows_capture import WindowsCapture  # noqa: PLC0415
        except ImportError as exc:                       # pragma: no cover
            raise ImportError(
                "windows-capture is required for live capture; it is in "
                "perception/requirements.txt. Recorded sources (video.py, "
                "frames.py) do not need it."
            ) from exc

        self._min_fps = min_fps
        self._lock = threading.Lock()
        self._latest: tuple[float, np.ndarray] | None = None
        self._stamps: list[float] = []
        self._index = 0
        self._closed = False
        self._t0 = time.perf_counter()
        self._rect: tuple[int, int, int, int] | None = None
        self._rect_shape: tuple[int, ...] | None = None
        self._rect_at = 0.0
        # Counts CAPTURED surfaces, not reads. WGC delivers on window updates,
        # so a caller polling faster than that will otherwise be handed the
        # same surface twice -- which reads downstream as two observations of
        # one instant, and as "nothing changed" to anything diffing frames.
        self._seq = 0
        self._last_seq = -1

        capture = WindowsCapture(cursor_capture=False, draw_border=False,
                                 window_name=window_name)

        @capture.event
        def on_frame_arrived(frame, control):          # noqa: ANN001
            now = time.perf_counter()
            with self._lock:
                # BGRA surface; drop alpha. Copy because the buffer is reused.
                self._latest = (now, frame.frame_buffer[:, :, :3].copy())
                self._seq += 1
                self._stamps.append(now)
                if len(self._stamps) > 240:
                    del self._stamps[:120]

        @capture.event
        def on_closed():                                # noqa: ANN001
            self._closed = True

        self._control = capture.start_free_threaded()
        self._await_first(warmup_s, window_name)

    def _await_first(self, timeout_s: float, window_name: str) -> None:
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            with self._lock:
                if self._latest is not None:
                    return
            time.sleep(0.02)
        self.close()
        raise TimeoutError(
            f"no frame from {window_name!r} within {timeout_s}s. The window "
            "must exist and be rendering; it does NOT need to be visible.")

    # -- FrameSource ---------------------------------------------------------

    @property
    def fps(self) -> float:
        """ACHIEVED rate, not a nominal one.

        The ABC tells a source to raise rather than guess when it is not
        constant-rate, and live capture never is -- WGC delivers on window
        updates. Reporting the measured rate is not a guess, and nothing
        downstream depends on it for timing: every Frame carries its own
        `wall_time_ms` taken at capture, so time conversion is exact whatever
        this says. It exists to be checked against `min_fps`.
        """
        with self._lock:
            stamps = list(self._stamps)
        if len(stamps) < 2:
            return 0.0
        span = stamps[-1] - stamps[0]
        return (len(stamps) - 1) / span if span > 0 else 0.0

    @property
    def size(self) -> tuple[int, int]:
        img = self._grab()[1]
        return img.shape[1], img.shape[0]

    @property
    def frame_count(self) -> int:
        return -1

    def __iter__(self) -> Iterator[Frame]:
        while not self._closed:
            stamp, image = self._grab()
            rate = self.fps
            if rate and rate < self._min_fps:
                raise RuntimeError(
                    f"capture fell to {rate:.1f} fps, below the {self._min_fps} "
                    "floor -- the loop would be acting on a stale board. "
                    "Check whether the emulator is still rendering.")
            yield Frame(index=self._index,
                        wall_time_ms=(stamp - self._t0) * 1000.0,
                        image=image)
            self._index += 1

    def read(self) -> Frame:
        """The current board, once. For a control loop that paces itself.

        May return the same surface twice if called faster than the window
        updates -- use `read_new` when duplicates would be misread as evidence.
        """
        stamp, image = self._grab()
        with self._lock:
            self._last_seq = self._seq
        frame = Frame(index=self._index,
                      wall_time_ms=(stamp - self._t0) * 1000.0, image=image)
        self._index += 1
        return frame

    def read_new(self, timeout_s: float = 1.0) -> Frame | None:
        """The next surface the window actually painted, or None on timeout.

        A recorder wants this rather than `read`: a duplicated frame is two
        entries in the timeline for one instant, which biases any rate measured
        from it and makes a frame-difference test report "no change" for a
        moment that was never observed twice.
        """
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            with self._lock:
                fresh = self._seq != self._last_seq and self._latest is not None
            if fresh:
                return self.read()
            time.sleep(0.002)
        return None

    def close(self) -> None:
        if getattr(self, "_control", None) is not None and not self._closed:
            try:
                self._control.stop()
            except Exception:                            # pragma: no cover
                pass
        self._closed = True

    # -- internals -----------------------------------------------------------

    def _grab(self) -> tuple[float, np.ndarray]:
        with self._lock:
            if self._latest is None:
                raise RuntimeError("no frame captured yet")
            stamp, buffer = self._latest
        x, y, w, h = self._game_rect(buffer)
        return stamp, np.ascontiguousarray(buffer[y:y + h, x:x + w])

    def _game_rect(self, buffer: np.ndarray) -> tuple[int, int, int, int]:
        """The cached game rect, re-derived only when it can have changed.

        Deriving it costs ~75 ms on a 1920x1020 surface, which dominated
        everything else on the hot path. It is a property of the window
        geometry, not of the frame, so it is cached against the surface shape
        (a resize changes that) with a slow revalidation behind it.
        """
        now = time.perf_counter()
        if (self._rect is None
                or self._rect_shape != buffer.shape
                or now - self._rect_at > RECT_REVALIDATE_S):
            self._rect = find_game_rect(buffer)
            self._rect_shape = buffer.shape
            self._rect_at = now
        return self._rect
