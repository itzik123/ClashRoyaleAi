"""The abstract frame source every downstream stage reads from: one interface for
a recorded match, a directory of stills or a live window, so the pipeline is
testable without an emulator.

A frame is (index, wall_time_ms, image). `wall_time_ms` is presentation time
within the source (a video's decoder timestamp), which converts to simulator
ticks (timebase.frame_index_to_ticks). Downstream, PlacementEvent.wall_time_ms
is the same quantity for a recording and the true capture clock live, so the
live path fills it without any consumer changing. Time from a frame counter
alone holds only at constant frame rate, which video.py checks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class Frame:
    index: int
    wall_time_ms: float
    image: np.ndarray
    """BGR uint8, (H, W, 3) -- OpenCV's native order, kept rather than
    converted so no copy happens on the hot path."""

    @property
    def size(self) -> tuple[int, int]:
        """(width, height)."""
        return self.image.shape[1], self.image.shape[0]


class FrameSource(ABC):
    """Read-only, forward-only sequence of timestamped frames."""

    @property
    @abstractmethod
    def fps(self) -> float:
        """Frames per second. Must raise rather than guess if the source is not
        constant-rate: every downstream time conversion depends on it.
        """

    @property
    @abstractmethod
    def size(self) -> tuple[int, int]:
        """(width, height) of every frame from this source."""

    @property
    @abstractmethod
    def frame_count(self) -> int:
        """Total frames, or -1 if unknown (live sources)."""

    @abstractmethod
    def __iter__(self) -> Iterator[Frame]:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def sample_every(self, target_fps: float) -> Iterator[Frame]:
        """Decimate to approximately `target_fps`.

        The detection rate is configurable because the signal is ~20 placements
        a minute; the expensive detector at capture rate is waste. Decimation
        happens here once, by presentation time rather than index modulo, which
        stays correct whatever the source rate.
        """
        if target_fps <= 0:
            raise ValueError(f"sample_every: non-positive target_fps {target_fps!r}")
        interval_ms = 1000.0 / target_fps
        next_due = -1.0
        for frame in self:
            if frame.wall_time_ms + 1e-6 >= next_due:
                yield frame
                # Anchored to the frame actually taken, so a stall cannot cause
                # a catch-up burst.
                next_due = frame.wall_time_ms + interval_ms
