"""Our elixir, read by sampling pixels along the bar. No OCR.

The value comes from where the fill ends: the ten segments are dividers over
one continuous bar, so the edge encodes the value to sub-segment precision,
where counting segments breaks as soon as something punches a hole in the fill.
Something does: the elixir number and the "Max: 10" caption are drawn on top of
the bar and blank whole columns. So the edge is read, from a row band the text
never reaches; the calibration profile's ROI is that band.

Colour, not brightness: elixir pink is saturated in a narrow hue band and the
trough behind it dark blue. Hue and saturation survive the bar's shimmer, which
moves value a long way and hue barely at all.

The bar's rectangle comes from a calibration profile; a wrong default would
produce confident nonsense, so this raises instead (CalibrationProfile.rois).

`cross_check` compares against readers/hand.py's dimmed slots, an independent
observation of the same quantity: three of four slots dimmed with the cheapest
costing 4 means elixir is below 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MAX_ELIXIR_SEGMENTS = 10

# Hue window for elixir pink on OpenCV's 0-179 scale, plus minimum saturation
# and value. Wide on hue, strict on saturation: the shimmer moves value, not
# hue, and everything behind the bar is desaturated.
DEFAULT_HUE_RANGE = (135, 175)
DEFAULT_MIN_SATURATION = 90
DEFAULT_MIN_VALUE = 70

# Fraction of a segment's columns that must read filled for the segment to
# count as full. Diagnostic only; the value comes from the fill edge.
SEGMENT_FULL_THRESHOLD = 0.75

# Fraction of a column's rows that must be pink for the column to count as
# filled. The ROI avoids the overlaid text, so a filled column is filled almost
# all the way through.
COLUMN_FILL_RATIO = 0.5

# Consecutive filled columns required to accept a fill edge, rejecting the
# bar's glow and antialiased rim.
EDGE_RUN_COLUMNS = 3


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
        column_filled = mask.mean(axis=0) >= COLUMN_FILL_RATIO

        # Fill edge: scanning from the right, the first run of consecutive
        # filled columns, so glow and rim do not add a few tenths to every
        # reading.
        edge = 0
        run = 0
        for index in range(w - 1, -1, -1):
            if column_filled[index]:
                run += 1
                if run >= EDGE_RUN_COLUMNS:
                    edge = index + EDGE_RUN_COLUMNS
                    break
            else:
                run = 0

        value = min(MAX_ELIXIR_SEGMENTS * edge / float(w), float(MAX_ELIXIR_SEGMENTS))

        edges = np.linspace(0, w, MAX_ELIXIR_SEGMENTS + 1).astype(int)
        fills = [
            float(column_filled[edges[i]:edges[i + 1]].mean()) if edges[i + 1] > edges[i] else 0.0
            for i in range(MAX_ELIXIR_SEGMENTS)
        ]

        # Consistency, not the measurement: left of the edge should be filled
        # and right of it empty. A violation means a misaligned ROI, an
        # overlay, or a wrong hue window, all of which still produce a
        # plausible number.
        left = column_filled[:edge]
        right = column_filled[edge:]
        left_ok = float(left.mean()) if left.size else 1.0
        right_ok = 1.0 - (float(right.mean()) if right.size else 0.0)
        confidence = round(min(1.0, left_ok * right_ok), 4)

        full = sum(1 for f in fills if f >= SEGMENT_FULL_THRESHOLD)
        return ElixirReading(
            value=value,
            full_segments=full,
            partial_fraction=value - int(value),
            confidence=confidence,
            per_segment_fill=tuple(round(f, 3) for f in fills),
        )


def cross_check(reading: ElixirReading, affordable_costs: list[float],
                unaffordable_costs: list[float]) -> tuple[bool, str]:
    """Check the bar against which hand slots are greyed out, an independent
    observation of the same quantity. Returns (consistent, explanation).
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
