"""Our elixir, read by sampling pixels along the bar. No OCR.

The bar is ten fixed segments in a fixed place. Counting how many are filled
and how far the partial one has advanced is a handful of array lookups --
deterministic, sub-millisecond, and exact to the pixel. OCR on the numeral
would be slower, would fail on the fractional part entirely (the number is
integer-valued while the true value is continuous), and would introduce a
model where none is needed.

THE VALUE COMES FROM THE FILL EDGE, NOT FROM COUNTING SEGMENTS
--------------------------------------------------------------
The ten segments are dividers drawn over one continuous bar, so the position
where the fill ends already encodes the value to sub-segment precision.
Locating that edge is both more accurate than counting full segments and far
more robust, because counting breaks the moment anything punches a hole in
the fill.

Which is exactly what happens here. The elixir NUMBER and the "Max: 10"
caption are drawn ON TOP of the bar, and they blank whole columns of it. A
segment-counting reader treats the first blanked column as the end of the
fill and reports ~0.5 elixir for the rest of the match. Measured on this
recording before the fix: 450 of 589 samples flagged low-confidence, with
values pinned near 0.5 while the real value ranged over 1-10.

The fix is twofold: read the edge rather than count, and read it from a row
band the text does not reach. Only y-rows 16-21 of the bar are clean in every
frame sampled; the ROI in the calibration profile is set to that band and
nothing else.

Colour, not brightness: elixir pink is strongly saturated in a narrow hue
band while the empty trough behind it is dark blue. Thresholding hue and
saturation in HSV survives the bar's animated shimmer, which moves value a
long way and hue almost not at all.

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

# Fraction of a segment's columns that must read as filled before the segment
# counts as full. Diagnostic only -- the value comes from the fill edge.
SEGMENT_FULL_THRESHOLD = 0.75

# Fraction of a COLUMN's rows that must be pink for the column to count as
# filled. The ROI is a narrow band chosen to be free of the overlaid text, so
# a filled column is filled almost all the way through.
COLUMN_FILL_RATIO = 0.5

# Consecutive filled columns required to accept a fill edge. Rejects the
# bar's outer glow and antialiased rim, which are a few columns wide and
# would otherwise inflate every reading.
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

        # Fill edge = the last column that is filled, ignoring isolated
        # specks past it. Scanning from the RIGHT and stopping at the first
        # run of consecutive filled columns rejects the glow and the
        # antialiased rim, which would otherwise add a few tenths of an
        # elixir at every reading.
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

        # Consistency, not the measurement: everything left of the edge should
        # be filled and everything right of it empty. A violation means the
        # ROI is misaligned, an overlay is covering the bar, or the hue window
        # is wrong -- all of which still yield a perfectly reasonable-looking
        # number, which is why this is checked rather than assumed.
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
