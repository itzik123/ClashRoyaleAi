"""Crop clock digit templates from a recording, labelled automatically.

    perception/.venv/Scripts/python.exe perception/tools/build_digit_templates.py \
        --video "<file>.mp4" --anchor-at 60.0 --anchor-clock 2:09

Templates need labelled glyphs, and reading the clock is what they are for. One
human reading breaks the circle: the clock counts down at one second per
second, so one (video timestamp, clock value) pair fixes it everywhere:

    clock(t) = anchor_clock - (t - anchor_t)

Every digit cell in every frame is labelled for free. Each template is the
per-pixel median of its samples, discarding transition and animation frames
without detecting them. The result is verified by reading the whole recording
back: templates built from a few hundred crops must reproduce thousands of
frames, including digit combinations the anchor never showed.

Cropped from the recording rather than shipped: pre-rendered glyphs assume one
rasterisation, and a mismatch shows only as uniformly mediocre scores.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from calib.homography import load_profile  # noqa: E402
from capture.video import VideoSource  # noqa: E402
from readers.clock import (  # noqa: E402
    GLYPH_SHAPE, DigitTemplates, _split_mss_cells, normalise_glyph,
)


# During the intro flourish the clock is partly transparent and occluded;
# templates from those frames blur toward the background. The arena is fully up
# a few seconds after the countdown.
ARENA_VISIBLE_AFTER_S = 19.0


def parse_clock(text: str) -> int:
    minutes, seconds = text.split(":")
    return int(minutes) * 60 + int(seconds)


def open_source(video: Path | None, frames: Path | None):
    """A recording: a video file or a record_match.py directory. Live capture is
    variable-rate, so a frame's real capture stamp is its only honest time;
    every label is `anchor - (t - anchor_t)`, so a synthesised stamp would
    mislabel glyphs and bake the error into the templates.
    """
    if (video is None) == (frames is None):
        raise SystemExit("pass exactly one of --video or --frames")
    if video is not None:
        return VideoSource(video)
    from capture.frames import RecordingSource  # noqa: PLC0415
    return RecordingSource(frames)


def resolve_roi(profile_path: Path | None, roi: tuple[int, int, int, int] | None):
    """The clock ROI, from an explicit rectangle or a calibration profile. The
    live path takes its tile mapping from CRBAB's constants, so a profile for
    this resolution would need a fabricated homography, which would later get
    used for real. The ROI is a measured rectangle and passed as one.
    """
    if roi is not None:
        return roi
    if profile_path is None:
        raise SystemExit("pass --roi or --profile")
    found = load_profile(profile_path).rois.get("clock")
    if found is None:
        raise SystemExit("no 'clock' ROI in the profile -- run calibrate.py first")
    return found


def build(source, roi, anchor_t: float, anchor_clock: int,
          out_dir: Path, sample_fps: float = 3.0) -> DigitTemplates:
    x, y, w, h = roi
    samples: dict[str, list[np.ndarray]] = {str(d): [] for d in range(10)}

    for frame in source.sample_every(sample_fps):
        t = frame.wall_time_ms / 1000.0
        remaining = anchor_clock - (t - anchor_t)
        # Only where the clock is running and fully legible: before the anchor
        # the arena may be loading, below zero the match is over.
        if remaining <= 1 or remaining > 172 or t < ARENA_VISIBLE_AFTER_S:
            continue
        remaining_int = int(round(remaining))
        # Skip frames within 0.25 s of a tick, where the digit may be
        # mid-change and would blend two glyphs into the median.
        if abs(remaining - remaining_int) > 0.25:
            continue

        digits = f"{remaining_int // 60}{(remaining_int % 60) // 10}{remaining_int % 10}"
        patch = cv2.cvtColor(frame.image[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        for glyph, cell in zip(digits, _split_mss_cells(patch)):
            # Normalised here, not at match time, so crops of one digit from
            # cells of different widths stack coherently (see
            # readers.clock.normalise_glyph).
            samples[glyph].append(normalise_glyph(cell))

    source.close()

    missing = [g for g, v in samples.items() if len(v) < 3]
    if missing:
        raise SystemExit(
            f"too few samples for digits {missing}. A single 3-minute match "
            "shows every digit in the seconds field, but the minutes field "
            "only ever shows 0, 1 and 2 -- so digits 3-9 come from the seconds "
            "cells. Check the anchor is right."
        )

    templates = {}
    for glyph, crops in samples.items():
        templates[glyph] = np.median(np.stack(crops), axis=0).astype(np.uint8)
        print(f"  digit {glyph}: {len(crops):4d} samples")

    result = DigitTemplates(templates)
    result.save(out_dir)
    print(f"wrote {out_dir}")
    return result


def verify(source, roi, templates: DigitTemplates,
           anchor_t: float, anchor_clock: int, sample_fps: float = 2.0) -> None:
    """Read the whole recording back and score against the arithmetic."""
    from readers.clock import ClockReader  # noqa: PLC0415

    reader = ClockReader(roi, templates)

    samples: list[tuple[float, int, float]] = []
    for frame in source.sample_every(sample_fps):
        t = frame.wall_time_ms / 1000.0
        remaining = anchor_clock - (t - anchor_t)
        if remaining <= 1 or remaining > 172 or t < ARENA_VISIBLE_AFTER_S:
            continue
        reading = reader.read(frame.image)
        samples.append((t, int(reading.seconds_remaining), reading.confidence))
    source.close()

    # The anchor was read from one frame, so its sub-second phase is unknown,
    # and samples near a tick boundary would look like off-by-one misreads. The
    # phase is fitted: one scalar over the whole recording, maximising
    # agreement. It cannot mask a real error, which is wrong at every phase.
    best_correct, best_phase, best_errors = -1, 0.0, []
    for phase in np.arange(0.0, 1.0, 0.05):
        errors, correct = [], 0
        for t, got, _conf in samples:
            expected = int(np.floor(anchor_clock + phase - (t - anchor_t)))
            if got == expected:
                correct += 1
            elif len(errors) < 12:
                errors.append((t, f"{expected // 60}:{expected % 60:02d}",
                               f"{got // 60}:{got % 60:02d}"))
        if correct > best_correct:
            best_correct, best_phase, best_errors = correct, phase, errors

    total = len(samples)
    low_conf = sum(1 for _t, _g, c in samples if c < 0.5)
    print(f"\nverification: {best_correct}/{total} frames read correctly "
          f"({100.0 * best_correct / max(total, 1):.2f}%) "
          f"at fitted anchor phase +{best_phase:.2f}s")
    print(f"   low-confidence readings: {low_conf}/{total}")
    for t, want, got in best_errors:
        print(f"   t={t:7.2f}s expected {want}  read {got}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path,
                        help="a recorded video file")
    parser.add_argument("--frames", type=Path,
                        help="a tools/record_match.py output directory")
    parser.add_argument("--profile", type=Path, default=None,
                        help="calibration profile carrying a 'clock' ROI")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                        help="clock ROI directly, instead of a profile")
    parser.add_argument("--anchor-at", type=float, required=True,
                        help="video timestamp, seconds, of a frame you read by eye")
    parser.add_argument("--anchor-clock", required=True,
                        help="the clock at that frame, M:SS")
    parser.add_argument("--out", type=Path,
                        default=_ROOT / "config" / "templates" / "clock")
    args = parser.parse_args()

    anchor = parse_clock(args.anchor_clock)
    roi = resolve_roi(args.profile, tuple(args.roi) if args.roi else None)
    templates = build(open_source(args.video, args.frames), roi,
                      args.anchor_at, anchor, args.out)
    # Re-open: a FrameSource is forward-only and the build pass consumed it.
    verify(open_source(args.video, args.frames), roi, templates,
           args.anchor_at, anchor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
