"""Our elixir, read by sampling pixels along the bar. No OCR.

The bar is ten fixed segments in a fixed place. Counting how many are filled
and how far the partial one has advanced is a handful of array lookups --
deterministic, sub-millisecond, and exact to the pixel. OCR on the numeral
would be slower, would fail on the fractional part entirely (the number is
integer-valued while the true value is continuous), and would introduce a
model where none is needed.

HOW A SEGMENT IS SCORED
-----------------------
Each segment is sampled along a horizontal line through the middle of the
bar, and its fill is judged by colour rather than brightness. Elixir pink is
strongly saturated in a narrow hue band; the empty slot behind it is dark and
desaturated. Working in HSV and thresholding on saturation-and-value makes
the test robust to the bar's own animated shimmer, which changes brightness
noticeably without changing hue.

The partial segment gives the fraction: the proportion of its sampled columns
that read as filled is the fractional part of the elixir value.

CALIBRATION IS REQUIRED, NOT GUESSED
------------------------------------
The bar's rectangle is a per-resolution constant that has to come from a
calibration profile. There is no sensible default and a wrong one produces
confident nonsense, so this raises instead. See CalibrationProfile.rois.

WHAT THIS IS CROSS-CHECKED AGAINST
----------------------------------
readers/hand.py reads which card slots are greyed out for being unaffordable.
That is an independent observation of the same quantity: if three of four
slots are dimmed and the cheapest of them costs 4, elixir is below 4. The two
readings are compared in `cross_check`, giving a free consistency test with
no ground truth -- exactly the sort of check the pipeline needs more of, since
nothing here can be labelled by hand at scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MAX_ELIXIR_SEGMENTS = 10

# Hue window for elixir pink in OpenCV's 0-179 hue scale, plus the minimum
# saturation and value a filled pixel must reach. Deliberately wide on hue and
# strict on saturation: the shimmer animation moves value a long way and hue
# barely at all, while everything behind the bar is desaturated.
DEFAULT_HUE_RANGE = (135, 175)
DEFAULT_MIN_SATURATION = 90
DEFAULT_MIN_VALUE = 70

# Fraction of a segment's sampled columns that must read as filled before the
# segment counts as full.
SEGMENT_FULL_THRESHOLD = 0.75


class ElixirCalibrationMissing(NotImplementedError):
    """No elixir-bar ROI in the calibration profile."""


@dataclass(frozen=True)
class ElixirReading:
    value: float
    """0.0 - 10.0, continuous."""

    full_segments: int
    partial_fraction: float
    confidence: float
    per_segment_fill: tuple[float, ...]
    """Fill ratio of each segment, for diagnosing a bad ROI at a glance: a
    correctly placed ROI produces a monotone non-increasing sequence."""


class ElixirBarReader:
    """Reads the elixir bar from a calibrated ROI."""

    def __init__(
        self,
        roi: tuple[int, int, int, int] | None,
        hue_range: tuple[int, int] = DEFAULT_HUE_RANGE,
        min_saturation: int = DEFAULT_MIN_SATURATION,
        min_value: int = DEFAULT_MIN_VALUE,
    ):
        if roi is None:
            raise ElixirCalibrationMissing(
                "no 'elixir_bar' ROI in the calibration profile. The bar's "
                "pixel rectangle is resolution-specific and cannot be guessed "
                "-- a wrong rectangle reads plausible values that are entirely "
                "fictional. Run perception/tools/calibrate.py on a frame from "
                "this recording."
            )
        self.roi = roi
        self.hue_range = hue_range
        self.min_saturation = min_saturation
        self.min_value = min_value

    def read(self, frame: np.ndarray) -> ElixirReading:
        x, y, w, h = self.roi
        if y + h > frame.shape[0] or x + w > frame.shape[1]:
            raise ElixirCalibrationMissing(
                f"elixir ROI {self.roi} falls outside a "
                f"{frame.shape[1]}x{frame.shape[0]} frame -- wrong profile."
            )

        patch = frame[y:y + h, x:x + w]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

        lo = np.array([self.hue_range[0], self.min_saturation, self.min_value], np.uint8)
        hi = np.array([self.hue_range[1], 255, 255], np.uint8)
        mask = cv2.inRange(hsv, lo, hi) > 0

        # Column occupancy over the middle band of the bar's height. The
        # outer rows are where the border and any rounded corners live, and
        # including them drags every segment's ratio down by a constant that
        # varies with the ROI's vertical alignment.
        band = mask[h // 4: max(h // 4 + 1, 3 * h // 4), :]
        column_filled = band.mean(axis=0) > 0.5

        edges = np.linspace(0, w, MAX_ELIXIR_SEGMENTS + 1).astype(int)
        fills = []
        for i in range(MAX_ELIXIR_SEGMENTS):
            segment = column_filled[edges[i]:edges[i + 1]]
            fills.append(float(segment.mean()) if segment.size else 0.0)

        full = 0
        partial = 0.0
        for fill in fills:
            if fill >= SEGMENT_FULL_THRESHOLD:
                full += 1
            else:
                partial = fill
                break

        value = min(float(full) + partial, float(MAX_ELIXIR_SEGMENTS))

        # A correct reading is monotone: every segment after the partial one
        # must be empty. Violations mean the ROI is misaligned, an overlay is
        # covering part of the bar, or the hue window is wrong -- all of which
        # produce a number that looks perfectly reasonable.
        tail = fills[full + 1:]
        monotone = all(f < SEGMENT_FULL_THRESHOLD for f in tail)
        confidence = 1.0 if monotone else 0.3

        return ElixirReading(
            value=value,
            full_segments=full,
            partial_fraction=partial,
            confidence=confidence,
            per_segment_fill=tuple(round(f, 3) for f in fills),
        )


def cross_check(reading: ElixirReading, affordable_costs: list[float],
                unaffordable_costs: list[float]) -> tuple[bool, str]:
    """Check the bar against which hand slots are greyed out.

    An independent observation of the same quantity, free of charge -- see
    this module's docstring. Returns (consistent, explanation).
    """
    if affordable_costs and reading.value + 0.15 < max(affordable_costs):
        return False, (
            f"bar reads {reading.value:.2f} but a {max(affordable_costs)} card "
            "is shown as playable"
        )
    if unaffordable_costs and reading.value > min(unaffordable_costs) + 0.15:
        return False, (
            f"bar reads {reading.value:.2f} but a {min(unaffordable_costs)} card "
            "is shown as unaffordable"
        )
    return True, "consistent"
