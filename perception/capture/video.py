"""FrameSource backed by a video file. Build and test everything against this.

Screen recorders often produce variable frame rate: frames are dropped when the
scene is static, while the file still reports a nominal fps. Under VFR frame
index is not proportional to time, and the error grows exactly when the screen
is busy, i.e. during fights, when placements happen.

So `probe_timing()` checks that inter-frame presentation deltas are constant,
and `fps` raises on a VFR file unless the caller opts in with `assume_fps`. The
decoder's own timestamps are used, so there is no ffprobe dependency.
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
        Override the CFR check and force this rate, for a known-VFR file whose
        error is acceptable. Recorded in `timing_override` so reports can say so.
    max_timing_jitter:
        Fraction of the nominal frame interval inter-frame deltas may vary by
        before the source is called VFR. 0.15 tolerates millisecond timestamp
        quantisation (a 33.33 ms interval stored as 33/34 ms is 2% jitter) and
        still catches a recorder that drops frames.
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

    # --- timing ---

    def probe_timing(self, samples: int = 240) -> dict:
        """Measure whether this file really is constant frame rate, from the
        presentation-time deltas of the first `samples` frames.

        Sequential rather than seeking: seeking in a long-GOP H.264 file snaps
        to keyframes, which would measure the keyframe interval and call every
        file VFR. Restores the read position, so it is safe before iterating.
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

        array = np.array(stamps, dtype=np.float64)
        deltas = np.diff(array)
        positive = deltas[deltas > 0]
        if len(positive) < 2:
            result["reason"] = "decoder reported no usable presentation timestamps"
            self._timing = result
            return result

        median = float(np.median(positive))
        # Median absolute deviation: one dropped frame is one huge delta, which
        # would swamp a mean-based statistic.
        jitter = float(np.median(np.abs(positive - median)) / median) if median > 0 else float("inf")

        # Rate from the total span, not the median delta. Timestamps are
        # quantised to whole milliseconds, so a 33.333 ms interval arrives as
        # 33/33/34 and its median reads 30.303 fps, 1% fast: 1.8 s (18 ticks)
        # by the end of a 176 s match. The span divides one quantisation error
        # by the whole sample. Jitter still comes from the deltas.
        span_per_frame = float((array[-1] - array[0]) / (len(array) - 1))
        measured = 1000.0 / span_per_frame if span_per_frame > 0 else float("nan")

        # Prefer the container's declared rate when the measurement confirms
        # it: it is the exact number the encoder intended (30.0, or
        # 30000/1001).
        nominal = self._nominal_fps
        if nominal > 0 and abs(measured - nominal) / nominal < 0.005:
            measured = nominal

        result["measured_fps"] = measured
        result["span_fps"] = 1000.0 / span_per_frame if span_per_frame > 0 else float("nan")
        result["median_delta_ms"] = median
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
        # The measured rate, not the container's nominal one, when they
        # disagree: the measurement is what the frames do.
        return timing["measured_fps"]

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    @property
    def frame_count(self) -> int:
        return self._frame_count

    # --- iteration ---

    def _stamp(self, index: int) -> float:
        """Presentation time of frame `index`, in milliseconds.

        Derived from index and frame rate rather than read per frame:
        CAP_PROP_POS_MSEC is positioned inconsistently relative to read()
        across backends (before the pending frame, after the returned one, or
        0.0 for the first two frames with mp4v on Windows), shifting the whole
        timeline by a frame. The derivation is exact because `fps` returns only
        after probe_timing() has confirmed constant rate.
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

    def sample_every(self, target_fps: float) -> Iterator[Frame]:
        """Decimate without decoding the discarded frames.

        Overrides FrameSource.sample_every, which decodes every frame. grab()
        advances the decoder without the colour conversion and copy that
        retrieve() does, so decimation costs roughly the skip ratio. Stepping
        by index is safe here only because `fps` has confirmed constant rate.
        """
        if target_fps <= 0:
            raise ValueError(f"sample_every: non-positive target_fps {target_fps!r}")
        stride = max(1, int(round(self.fps / target_fps)))

        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        index = 0
        while True:
            if index % stride:
                if not self._capture.grab():
                    break
                index += 1
                continue
            ok, image = self._capture.read()
            if not ok:
                break
            yield Frame(index=index, wall_time_ms=self._stamp(index), image=image)
            index += 1

    def read_frame_at(self, index: int) -> Frame:
        """Random access, for calibration and tests. Not on the FrameSource
        interface: a live source cannot support it, and code relying on it
        would break when pointed at a window.
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
