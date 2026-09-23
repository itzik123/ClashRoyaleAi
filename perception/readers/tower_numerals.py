"""Tower HP as the absolute number printed above each bar.

CRBAB's bar reader returns a fraction by colour-matching, and `_calculate_hp`
returns 0.0 both for an empty bar and for one it cannot match, so it cannot
tell a live tower from an unreadable one. The numeral is absolute HP, needing
no tower-level table (maxima vary by account: 1750 at level 4, 1890 at level 5,
the engine's level-9 2534), and a failed read reports low confidence rather
than 0.

The numeral's box is derived from CRBAB's own bar bbox (`NUMERAL_OFFSET`,
applied after scaling into the native frame), so a tower's position has one
definition. It must be read at native resolution: in the 368x652 detector frame
the glyphs are ~6 px tall, at the 549x976 capture ~14 px.

Segmentation is by column projection, as in `readers/clock.py`, since glyph
widths differ (a "1" is a third the width of a "0"). The digit count varies: a
tower drops through 1000 mid-match, so this returns however many digits it
finds.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from readers.clock import DigitTemplates, normalise_glyph

# (left, top, right, bottom) deltas applied to the bar's bbox in native pixels,
# `top`/`bottom` relative to the bar's top edge. The bottom stops above the
# bar: the bar is a saturated pink stripe across the ROI, and including even
# two rows of it joined every column of the projection into one run.
NUMERAL_OFFSET = (-6, -19, 6, -3)

# Below this a reading is reported as not measured: a tower of unknown HP must
# stay distinguishable from a nearly dead one, since a live tower reported
# empty tells the policy a lane is lost.
MIN_DIGIT_CONFIDENCE = 0.35

# A plausible Princess/King HP; anything outside is a misread.
HP_RANGE = (1, 9999)


@dataclass(frozen=True)
class TowerNumeralReading:
    """One tower's HP, or an explicit failure to read it."""

    value: int | None
    """Absolute HP. None when the numeral could not be read -- NOT 0, which
    would be indistinguishable from a destroyed tower."""

    digits: str
    confidence: float
    """Lowest per-digit confidence. One bad glyph makes the whole number
    wrong, so the weakest link is the reading's confidence."""

    @property
    def measured(self) -> bool:
        return self.value is not None


def numeral_roi(bar_bbox, native_size, detector_size) -> tuple[int, int, int, int]:
    """The numeral's box in native pixels, derived from CRBAB's bar box so the
    tower's position has one definition.
    """
    nw, nh = native_size
    dw, dh = detector_size
    sx, sy = nw / dw, nh / dh
    x1, y1, x2, _y2 = bar_bbox
    dl, dt, dr, db = NUMERAL_OFFSET
    return (max(0, int(x1 * sx) + dl),
            max(0, int(y1 * sy) + dt),
            min(nw, int(x2 * sx) + dr),
            min(nh, int(y1 * sy) + db))


def ink_channel(patch: np.ndarray) -> np.ndarray:
    """Per-pixel minimum across colour channels, not luminance.

    The numerals are near-white on grass. In luminance grass (~169) and digits
    (~230) are both bright and land on the same side of Otsu, merging the
    digits into one blob. The minimum channel separates them: grass is
    saturated green with low blue (~90), a near-white glyph is high in every
    channel (~220). Any saturated background (grass, the pink bar, a blue roof)
    drops out the same way.
    """
    if patch.ndim == 3:
        return patch.min(axis=2).astype(np.uint8)
    return patch


def split_digits(patch: np.ndarray, min_height_frac: float = 0.45) -> list[np.ndarray]:
    """Split a numeral patch into digit cells by column projection.

    Returns however many it finds; asserting a count would fail on exactly the
    damaged frames being measured. Runs shorter than `min_height_frac` of the
    patch are dropped, removing the bar's edge and stray highlights without a
    width rule (which would also drop a narrow "1").
    """
    patch = ink_channel(patch)
    if patch.size == 0:
        return []
    h, w = patch.shape
    _score, ink = cv2.threshold(patch, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    columns = (ink > 0).sum(axis=0)
    active = columns > max(1, h // 8)

    runs: list[tuple[int, int]] = []
    start = None
    for x in range(w):
        if active[x]:
            if start is None:
                start = x
        elif start is not None:
            runs.append((start, x))
            start = None
    if start is not None:
        runs.append((start, w))

    cells = []
    for a, b in runs:
        rows = np.where((ink[:, a:b] > 0).any(axis=1))[0]
        if rows.size and (rows[-1] - rows[0] + 1) >= h * min_height_frac:
            cells.append(patch[:, a:b])
    return cells


class TowerNumeralReader:
    """Reads a tower's absolute HP from the numeral above its bar."""

    def __init__(self, templates: DigitTemplates):
        self.templates = templates

    def read(self, native_frame, bar_bbox, detector_size) -> TowerNumeralReading:
        # The RGB frame is passed through: split_digits needs colour to
        # separate near-white glyphs from saturated backgrounds (see
        # ink_channel).
        gray = native_frame
        nh, nw = gray.shape[:2]
        x1, y1, x2, y2 = numeral_roi(bar_bbox, (nw, nh), detector_size)
        if x2 <= x1 or y2 <= y1:
            return TowerNumeralReading(None, "", 0.0)

        cells = split_digits(gray[y1:y2, x1:x2])
        if not cells:
            return TowerNumeralReading(None, "", 0.0)

        digits, confidences = [], []
        for cell in cells:
            glyph, score = self.templates.classify(cell)
            digits.append(glyph)
            confidences.append(score)

        text = "".join(digits)
        worst = min(confidences) if confidences else 0.0
        if not text.isdigit() or worst < MIN_DIGIT_CONFIDENCE:
            return TowerNumeralReading(None, text, worst)
        value = int(text)
        if not (HP_RANGE[0] <= value <= HP_RANGE[1]):
            # A number outside any real tower's range is a segmentation failure
            # that happened to produce digits; reporting it would look
            # measured.
            return TowerNumeralReading(None, text, worst)
        return TowerNumeralReading(value, text, worst)


def normalise_for_templates(cell: np.ndarray, shape) -> np.ndarray:
    """Exposed so a template builder and the reader normalise identically."""
    return normalise_glyph(cell, shape)
