"""FrameSource backed by a video file. Build and test everything against this.

VARIABLE FRAME RATE IS CHECKED, NOT ASSUMED
-------------------------------------------
Phone screen recorders almost always produce VFR: the encoder drops frames
whenever the scene is static and the container stores a per-frame timestamp
instead of a fixed interval. The file still reports a nominal fps, and
OpenCV still hands it over, and everything looks fine.

It is not fine. With VFR, frame index is not proportional to elapsed time, so
every index-derived tick is wrong by an amount that varies with how busy the
screen was -- which correlates with exactly the moments that matter, since
placements are when the screen gets busy. The resulting clock error is not
noise, it is a bias that grows during fights.

So `probe_timing()` measures it: sample presentation timestamps across the
file and check the inter-frame deltas are actually constant. `fps` raises on
a VFR file unless the caller explicitly opts in by supplying `assume_fps`,
which forces the caller to have made the decision.

The check uses the decoder's own timestamps rather than shelling out to
ffprobe so there is no external binary dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from capture.source import Frame, FrameSource


class VideoTimingError(RuntimeError):
    """The source's frame timing is not usable for tick conversion."""


class VideoSource(FrameSource):
    """Frames from a video file, in order, with decoder timestamps.

    Parameters
    ----------
    path:
        Video file.
    assume_fps:
        Override the CFR check and force this rate. Use only when you know
        the file is VFR and have decided the error is acceptable; it is
        recorded in `timing_override` so downstream reports can say so.
    max_timing_jitter:
        Fraction of the nominal frame interval that inter-frame deltas may
        vary by before the source is called VFR. The default 0.15 is loose
        enough to tolerate millisecond-resolution timestamp quantisation on a
        30fps file (a 33.33ms interval stored as integer ms alternates
        33/34ms, which is 2% jitter) and tight enough to catch a recorder
        that is genuinely dropping frames.
    """

    def __init__(
        self,
        path: str | Path,
        assume_fps: float | None = None,
        max_timing_jitter: float = 0.15,
    ):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"no video at {self.path}")

        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise RuntimeError(
                f"OpenCV could not open {self.path}. Unsupported codec, or the "
                "file is truncated."
            )

        self._nominal_fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self._frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._size = (
            int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self._max_timing_jitter = max_timing_jitter
        self.timing_override = assume_fps
        self._timing: dict | None = None

    # -- timing ----------------------------------------------------------

    def probe_timing(self, samples: int = 240) -> dict:
        """Measure whether this file is really constant frame rate.

        Reads the first `samples` frames sequentially and looks at the
        distribution of presentation-time deltas. Sequential rather than
        seeking: seeking in a long-GOP H.264 file snaps to keyframes, so
        seek-derived timestamps would measure the keyframe interval instead
        of the frame interval and call every file VFR.

        Restores the read position afterwards, so this is safe to call before
        iterating.
        """
        if self._timing is not None:
            return self._timing

        start_pos = int(self._capture.get(cv2.CAP_PROP_POS_FRAMES))
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        stamps: list[float] = []
        for _ in range(max(samples, 2)):
            ok = self._capture.grab()
            if not ok:
                break
            stamps.append(float(self._capture.get(cv2.CAP_PROP_POS_MSEC)))

        self._capture.set(cv2.CAP_PROP_POS_FRAMES, start_pos)

        result: dict = {
            "nominal_fps": self._nominal_fps,
            "sampled_frames": len(stamps),
            "is_cfr": False,
            "measured_fps": float("nan"),
            "jitter": float("nan"),
            "reason": "",
        }

        if len(stamps) < 3:
            result["reason"] = "too few frames to assess timing"
            self._timing = result
            return result

        deltas = np.diff(np.array(stamps, dtype=np.float64))
        deltas = deltas[deltas > 0]
        if len(deltas) < 2:
            result["reason"] = "decoder reported no usable presentation timestamps"
            self._timing = result
            return result

        median = float(np.median(deltas))
        # Median absolute deviation, not standard deviation: a single dropped
        # frame produces one huge delta, and a mean-based statistic would let
        # that one outlier swamp the measurement in either direction.
        jitter = float(np.median(np.abs(deltas - median)) / median) if median > 0 else float("inf")

        result["measured_fps"] = 1000.0 / median if median > 0 else float("nan")
        result["jitter"] = jitter
        result["is_cfr"] = jitter <= self._max_timing_jitter
        if not result["is_cfr"]:
            result["reason"] = (
                f"inter-frame deltas vary by {jitter:.1%} of the median "
                f"({median:.2f}ms) -- looks variable-frame-rate"
            )

        self._timing = result
        return result

    @property
    def fps(self) -> float:
        if self.timing_override is not None:
            return float(self.timing_override)

        timing = self.probe_timing()
        if not timing["is_cfr"]:
            raise VideoTimingError(
                f"{self.path.name}: {timing['reason']}.\n"
                "Frame index cannot be converted to time on a variable-rate "
                "source, so every derived tick would be wrong by an amount "
                "that grows during fights (see this module's docstring).\n"
                "Re-record with constant frame rate (OBS: Rate Control = CBR), "
                "or transcode:\n"
                f"    ffmpeg -i {self.path.name} -vsync cfr -r 30 -c:v libx264 "
                f"-crf 16 {self.path.stem}_cfr.mp4\n"
                "or pass assume_fps=<rate> to override deliberately."
            )
        # The measured rate, not the container's nominal one. They usually
        # agree; when they do not, the measurement is what the frames
        # actually do.
        return timing["measured_fps"]

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    @property
    def frame_count(self) -> int:
        return self._frame_count

    # -- iteration -------------------------------------------------------

    def _stamp(self, index: int) -> float:
        """Presentation time of frame `index`, in milliseconds.

        Derived from the index and the frame rate, NOT read per-frame from
        the decoder -- and that is a correctness fix, not an optimisation.

        CAP_PROP_POS_MSEC is not reliably positioned relative to read(): with
        some backends it reports the time BEFORE the pending frame, with
        others AFTER the one just returned, and at least one (observed here,
        mp4v on Windows) reports 0.0 for both of the first two frames. Any of
        those silently shifts the entire timeline by a frame, which then
        shifts every placement tick by the same amount.

        Deriving from the index is exact precisely BECAUSE `fps` refuses to
        return a value until probe_timing() has confirmed the source is
        constant-rate. On a CFR source index * interval is the definition of
        presentation time; on a VFR source there is no fps to derive from and
        the property raises instead.
        """
        return index * 1000.0 / self.fps

    def __iter__(self) -> Iterator[Frame]:
        stamp_of = self._stamp  # raises here, before any decoding, if VFR
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        index = 0
        while True:
            ok, image = self._capture.read()
            if not ok:
                break
            yield Frame(index=index, wall_time_ms=stamp_of(index), image=image)
            index += 1

    def read_frame_at(self, index: int) -> Frame:
        """Random access, for calibration and tests.

        Not on the FrameSource interface on purpose: a live source cannot
        support it, and code written against random access silently stops
        working the day it is pointed at a window instead of a file.
        """
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = self._capture.read()
        if not ok:
            raise IndexError(f"no frame at index {index} in {self.path.name}")
        return Frame(index=index, wall_time_ms=self._stamp(index), image=image)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def describe(self) -> str:
        """One-paragraph summary for calibration reports and test output."""
        timing = self.probe_timing()
        rate = (
            f"{timing['measured_fps']:.3f} fps (CFR)"
            if timing["is_cfr"]
            else f"VARIABLE ({timing['reason']})"
        )
        seconds = self._frame_count / self._nominal_fps if self._nominal_fps else 0.0
        return (
            f"{self.path.name}: {self._size[0]}x{self._size[1]}, "
            f"{self._frame_count} frames, {seconds:.1f}s, {rate}"
        )
