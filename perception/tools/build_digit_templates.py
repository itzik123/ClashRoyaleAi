"""Crop clock digit templates from a recording, labelled automatically.

    perception/.venv/Scripts/python.exe perception/tools/build_digit_templates.py \
        --video "<file>.mp4" --anchor-at 60.0 --anchor-clock 2:09

HOW THE LABELS ARE OBTAINED WITHOUT LABELLING ANYTHING
-------------------------------------------------------
Templates need labelled glyphs, and reading the clock is the very thing they
are for -- so the obvious approach is circular.

It is broken with a single human reading. The match clock is a monotone
countdown at exactly one second per second, so ONE (video timestamp, clock
value) pair determines the clock at every other frame in the recording:

    clock(t) = anchor_clock - (t - anchor_t)

Every digit cell in every frame is then labelled for free, and each of the ten
glyphs appears many times over three minutes. The template for each is the
per-pixel MEDIAN of its samples, which discards the ones corrupted by a
transition frame or a passing animation without needing to detect them.

The result is verified by reading the whole recording back and comparing
against the same arithmetic. That check is not circular: the templates are
built from a few hundred crops and then asked to reproduce ~5000 frames,
including every digit combination the anchor frame never showed.

WHY NOT SHIP PRE-RENDERED GLYPHS
---------------------------------
They would encode an assumption about the font's rasterisation at one
specific scale. Any mismatch shows up as uniformly mediocre match scores
rather than an obvious failure -- the worst kind of calibration bug. Cropping
from the recording itself cannot have that problem.
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


# The clock is drawn during the intro flourish too, partly transparent and
# partly occluded. Templates built from those frames are blurred toward the
# background. The arena is fully up a few seconds after the countdown.
ARENA_VISIBLE_AFTER_S = 19.0


def parse_clock(text: str) -> int:
    minutes, seconds = text.split(":")
    return int(minutes) * 60 + int(seconds)


def build(video: Path, profile_path: Path, anchor_t: float, anchor_clock: int,
          out_dir: Path, sample_fps: float = 3.0) -> DigitTemplates:
    profile = load_profile(profile_path)
    roi = profile.rois.get("clock")
    if roi is None:
        raise SystemExit("no 'clock' ROI in the profile -- run calibrate.py first")

    source = VideoSource(video)
    x, y, w, h = roi
    samples: dict[str, list[np.ndarray]] = {str(d): [] for d in range(10)}

    for frame in source.sample_every(sample_fps):
        t = frame.wall_time_ms / 1000.0
        remaining = anchor_clock - (t - anchor_t)
        # Only the stretch where the clock is actually running and fully
        # legible. Before the anchor the arena may still be loading; below
        # zero the match is over.
        if remaining <= 1 or remaining > 172 or t < ARENA_VISIBLE_AFTER_S:
            continue
        remaining_int = int(round(remaining))
        # Skip frames within 0.25s of a tick, where the rendered digit may be
        # mid-change and would poison the median with a blend of two glyphs.
        if abs(remaining - remaining_int) > 0.25:
            continue

        digits = f"{remaining_int // 60}{(remaining_int % 60) // 10}{remaining_int % 10}"
        patch = cv2.cvtColor(frame.image[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        for glyph, cell in zip(digits, _split_mss_cells(patch)):
            # Normalised HERE, not at match time, so crops of the same digit
            # taken from cells of different widths stack coherently. Pooling
            # raw cells was the defect that scored 24.7% -- see
            # readers.clock.normalise_glyph.
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


def verify(video: Path, profile_path: Path, templates: DigitTemplates,
           anchor_t: float, anchor_clock: int, sample_fps: float = 2.0) -> None:
    """Read the whole recording back and score against the arithmetic."""
    from readers.clock import ClockReader

    profile = load_profile(profile_path)
    reader = ClockReader(profile.rois["clock"], templates)
    source = VideoSource(video)

    samples: list[tuple[float, int, float]] = []
    for frame in source.sample_every(sample_fps):
        t = frame.wall_time_ms / 1000.0
        remaining = anchor_clock - (t - anchor_t)
        if remaining <= 1 or remaining > 172 or t < ARENA_VISIBLE_AFTER_S:
            continue
        reading = reader.read(frame.image)
        samples.append((t, int(reading.seconds_remaining), reading.confidence))
    source.close()

    # The anchor was read by eye from ONE frame, so its sub-second phase is
    # unknown: "2:09" is on screen for a whole second and there is no way to
    # tell from the reading which part of that second the anchor frame fell
    # in. Left unmodelled, every sample near a tick boundary looks like an
    # off-by-one misread when the reader was in fact correct -- which is
    # exactly what the first version of this check reported, at every sample
    # landing on a .8s offset.
    #
    # So the phase is fitted: one scalar, over the whole recording, chosen to
    # maximise agreement. It cannot mask a real error, because a genuine
    # misread is wrong at every phase.
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
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--profile", type=Path,
                        default=_ROOT / "config" / "profile_gpg_1920x1080.json")
    parser.add_argument("--anchor-at", type=float, required=True,
                        help="video timestamp, seconds, of a frame you read by eye")
    parser.add_argument("--anchor-clock", required=True,
                        help="the clock at that frame, M:SS")
    parser.add_argument("--out", type=Path,
                        default=_ROOT / "config" / "templates" / "clock")
    args = parser.parse_args()

    anchor = parse_clock(args.anchor_clock)
    templates = build(args.video, args.profile, args.anchor_at, anchor, args.out)
    verify(args.video, args.profile, templates, args.anchor_at, anchor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
