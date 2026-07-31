"""FrameSource backed by a directory of stills.

Two jobs. First, calibration: a single extracted frame is the natural unit
for picking anchors, and a directory of them is the natural unit for
regression-testing readers against a handful of interesting moments (elixir
nearly full, clock at a phase boundary, a card slot greyed out) without
shipping a video for each.

Second, it is how a failing frame from a long recording gets turned into a
permanent test case: dump the frame, drop it in a directory, and the same
pipeline runs over it unchanged.

Since stills carry no timing of their own, the rate is supplied by the
caller and timestamps are synthesised from it. That is honest for the
calibration use, where the frames are independent samples rather than a
sequence, and it is exactly why this class does not pretend to probe timing
the way VideoSource does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2

from capture.source import Frame, FrameSource

_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


class FrameDirSource(FrameSource):
    """Images from a directory, in sorted filename order.

    Sorted so ordering is deterministic and reproducible across machines --
    directory iteration order is not. Zero-pad numeric filenames
    (frame_0001.png) or the tenth frame sorts before the second.
    """

    def __init__(self, directory: str | Path, fps: float = 30.0, pattern: str = "*"):
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise NotADirectoryError(f"not a directory: {self.directory}")

        self.paths = sorted(
            p for p in self.directory.glob(pattern)
            if p.suffix.lower() in _EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(
                f"no images matching {pattern!r} in {self.directory} "
                f"(looked for {', '.join(_EXTENSIONS)})"
            )

        self._fps = float(fps)
        if self._fps <= 0:
            raise ValueError(f"fps must be positive, got {fps!r}")

        probe = cv2.imread(str(self.paths[0]), cv2.IMREAD_COLOR)
        if probe is None:
            raise RuntimeError(f"could not decode {self.paths[0]}")
        self._size = (probe.shape[1], probe.shape[0])

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    @property
    def frame_count(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[Frame]:
        interval_ms = 1000.0 / self._fps
        for index, path in enumerate(self.paths):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"could not decode {path}")
            # Mixed sizes would make one calibration profile silently wrong
            # for part of the directory, which is the failure this catches.
            if (image.shape[1], image.shape[0]) != self._size:
                raise ValueError(
                    f"{path.name} is {image.shape[1]}x{image.shape[0]} but the "
                    f"directory started at {self._size[0]}x{self._size[1]} -- "
                    "one calibration profile cannot cover both."
                )
            yield Frame(index=index, wall_time_ms=index * interval_ms, image=image)

    def close(self) -> None:
        return None


class RecordingSource(FrameSource):
    """A directory written by `tools/record_match.py`: stills plus a manifest.

    Distinct from FrameDirSource, which synthesises timestamps from a nominal
    rate because stills carry no timing of their own. A live recording DOES
    carry timing -- the recorder stamps every frame at capture -- and it is
    variable-rate by nature, since Windows.Graphics.Capture delivers on window
    repaints. Feeding it through a constant-rate source would replace real
    timestamps with a fiction, and the readers that depend on the clock would
    inherit the drift.

    That matters concretely: the clock digit templates are labelled by
    `clock(t) = anchor - (t - anchor_t)`, so a wrong `t` mislabels the glyph and
    the error is baked into the template rather than showing up as a bad read.
    """

    def __init__(self, directory: str | Path):
        import json  # noqa: PLC0415

        self.directory = Path(directory)
        manifest_path = self.directory / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no manifest.json in {self.directory} -- this source is for "
                "tools/record_match.py output. For a plain directory of stills "
                "use FrameDirSource, which synthesises timestamps."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.rows = manifest["frames"]
        if not self.rows:
            raise ValueError(f"manifest in {self.directory} lists no frames")

        self._achieved_fps = float(manifest.get("achieved_fps") or 0.0)
        probe = cv2.imread(str(self.directory / self.rows[0]["file"]),
                           cv2.IMREAD_COLOR)
        if probe is None:
            raise RuntimeError(f"could not decode {self.rows[0]['file']}")
        self._size = (probe.shape[1], probe.shape[0])

    @property
    def fps(self) -> float:
        """ACHIEVED rate, recorded at capture time.

        Not a constant-rate promise -- see the class docstring. Anything doing
        time arithmetic must use each Frame's own `wall_time_ms`, which is the
        real capture stamp, never index/fps.
        """
        return self._achieved_fps

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    @property
    def frame_count(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[Frame]:
        for row in self.rows:
            path = self.directory / row["file"]
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"could not decode {path}")
            if (image.shape[1], image.shape[0]) != self._size:
                raise ValueError(
                    f"{row['file']} is {image.shape[1]}x{image.shape[0]} but "
                    f"the recording started at {self._size[0]}x{self._size[1]}"
                )
            yield Frame(index=int(row["index"]),
                        wall_time_ms=float(row["wall_time_ms"]), image=image)

    def close(self) -> None:
        return None
