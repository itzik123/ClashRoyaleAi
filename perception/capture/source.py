"""The abstract frame source every downstream stage reads from.

One interface, so the pipeline cannot tell whether it is being driven by a
recorded match, a directory of stills, or (eventually) a live window. That is
not architectural neatness for its own sake -- it is the property that makes
the whole thing testable without an emulator running, which is a hard
requirement here.

FRAMES CARRY THEIR OWN TIME
---------------------------
A frame is (index, wall_time_ms, image), never a bare image. Two different
clocks matter and they are not interchangeable:

  * `wall_time_ms` is presentation time within the source -- for a video, the
    decoder's timestamp for that frame. This is what converts to simulator
    ticks (see timebase.frame_index_to_ticks) and what makes a recorded run
    reproducible.

  * PlacementEvent.wall_time_ms, downstream, is the same quantity for a
    recording and the true capture clock for a live source. Keeping the field
    on the frame means the live path can populate it correctly later without
    any consumer changing.

Deriving time from a frame counter alone works only at constant frame rate,
which is why the CFR question is answered here rather than assumed -- see
video.py.
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
        """Frames per second.

        Must raise rather than guess if the source is not constant-rate.
        Every downstream time conversion assumes this number is meaningful.
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

        This is the configurable detection rate the whole design rests on:
        the useful signal is ~20 placements a minute, so running the
        expensive detector at capture rate is pure waste. Decimation happens
        here, once, instead of each stage inventing its own skip logic.

        Decimation is by presentation TIME, not by index modulo. Index
        stepping silently changes meaning if the source rate is not what you
        think it is; time stepping stays correct either way.
        """
        if target_fps <= 0:
            raise ValueError(f"sample_every: non-positive target_fps {target_fps!r}")
        interval_ms = 1000.0 / target_fps
        next_due = -1.0
        for frame in self:
            if frame.wall_time_ms + 1e-6 >= next_due:
                yield frame
                # Anchored to the frame we actually took, not to a running
                # accumulator, so a stall in the source cannot make the
                # sampler try to "catch up" by emitting a burst.
                next_due = frame.wall_time_ms + interval_ms
