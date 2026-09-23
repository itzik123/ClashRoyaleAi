"""Live frames from the emulator window, via Windows.Graphics.Capture.

Not ADB: `adb exec-out screencap` takes 1.4-4.9 s per frame, against a capture
budget of ~200 ms (the policy acts once per second and the detector costs ~800
ms). Not a screen-region grab: `Graphics.CopyFromScreen` reads the composited
desktop and returns whatever window is on top. WGC reads the window's own
surface, so covering the emulator changes nothing; tested against BlueStacks,
which renders through D3D, where PrintWindow tends to return black. A minimized
window is not repainted, so no frames arrive.

The captured surface is the whole emulator window: title bar, a toolbar down
the right, and the portrait game pillarboxed against black. Every frame is
cropped to the game rect, found from the frame rather than hardcoded, since the
window can move or resize; the detector must never see the chrome.

WGC returns physical pixels: at 125% display scaling the surface is 1920x1020
while `GetWindowRect` reports 1536x816. Never size anything from GetWindowRect.

The game renders at ~549x976 here, against the 720x1280 unit_hp.py was fitted
on. Re-scoring the HP reader against its 60 hand labels at 1.0, 0.85, 0.75 and
0.637 scale moved F1 within sampling noise (0.66-0.78, non-monotonic):
downscaling preserves the fill/track ratio an HP fraction reads.
"""
from __future__ import annotations

import threading
import time
from typing import Iterator

import numpy as np

from capture.source import Frame, FrameSource

# A pillarbox pixel is black; the threshold only absorbs compression noise.
LETTERBOX_MAX = 24

# The emulator preserves the panel's aspect, so the game rect's height is
# derived from its measured width. Detecting it fails: the title bar spans the
# game's columns and is not black.
NATIVE_WIDTH, NATIVE_HEIGHT = 720, 1280

# Refuse to run below this rather than deliver a stale board: a loop fed frames
# slower than it acts is deciding about the past.
DEFAULT_MIN_FPS = 2.0

# Rows subsampled when scanning for the pillarbox: the bars are uniform over
# hundreds of rows, and this scan is on the hot path.
RECT_ROW_STEP = 4

# How often to re-derive the game rect at unchanged surface size. Deriving it
# costs ~75 ms on a 1920x1020 buffer, far too much per frame. A resize changes
# the surface shape and is caught immediately; this only covers the app itself
# changing aspect.
RECT_REVALIDATE_S = 5.0


class GameRectError(RuntimeError):
    """The game area could not be located inside the captured surface."""


def find_game_rect(buffer: np.ndarray) -> tuple[int, int, int, int]:
    """(x, y, w, h) of the pillarboxed game inside a full window capture.

    The width is the widest run of non-black columns in a band across the
    middle of the surface, where the title bar and toolbar do not join every
    run into one. The height follows from the Android aspect, anchored to the
    bottom, since the chrome is on top.
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
    """Live frames from a named window, cropped to the game area. Capture runs on
    its own thread and keeps only the latest frame: after a hitch the agent
    should act on the current board, not a backlog.
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
        # Counts captured surfaces, not reads. WGC delivers on window updates,
        # so a faster poller would get the same surface twice, which reads
        # downstream as two observations of one instant.
        self._seq = 0
        self._last_seq = -1

        capture = WindowsCapture(cursor_capture=False, draw_border=False,
                                 window_name=window_name)

        @capture.event
        def on_frame_arrived(frame, control):          # noqa: ANN001
            now = time.perf_counter()
            with self._lock:
                # BGRA surface; drop alpha. Copied because the buffer is
                # reused.
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
            "must be RENDERING. Being covered by other windows is fine -- "
            "that is verified and is the whole point of using WGC -- but a "
            "MINIMIZED window has no surface being repainted and delivers "
            "nothing. Restore it and try again.")

    # --- FrameSource ---

    @property
    def fps(self) -> float:
        """Achieved rate, not a nominal one. Live capture is never constant-rate,
        but nothing depends on this for timing: every Frame carries its own
        capture-time `wall_time_ms`. It exists to be checked against `min_fps`.
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
        """The current board, once, for a loop that paces itself. May return the
        same surface twice; use `read_new` when duplicates would be misread as
        evidence.
        """
        stamp, image = self._grab()
        with self._lock:
            self._last_seq = self._seq
        frame = Frame(index=self._index,
                      wall_time_ms=(stamp - self._t0) * 1000.0, image=image)
        self._index += 1
        return frame

    def read_new(self, timeout_s: float = 1.0) -> Frame | None:
        """The next surface the window actually painted, or None on timeout. For a
        recorder: a duplicate is two timeline entries for one instant, biasing
        any measured rate.
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

    # --- internals ---

    def _grab(self) -> tuple[float, np.ndarray]:
        with self._lock:
            if self._latest is None:
                raise RuntimeError("no frame captured yet")
            stamp, buffer = self._latest
        x, y, w, h = self._game_rect(buffer)
        return stamp, np.ascontiguousarray(buffer[y:y + h, x:x + w])

    def _game_rect(self, buffer: np.ndarray) -> tuple[int, int, int, int]:
        """The cached game rect, re-derived only when it can have changed: cached
        against the surface shape (a resize changes it), with a slow
        revalidation behind it.
        """
        now = time.perf_counter()
        if (self._rect is None
                or self._rect_shape != buffer.shape
                or now - self._rect_at > RECT_REVALIDATE_S):
            self._rect = find_game_rect(buffer)
            self._rect_shape = buffer.shape
            self._rect_at = now
        return self._rect
