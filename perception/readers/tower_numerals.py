"""Tower HP as the ABSOLUTE number printed above each bar.

WHY THIS EXISTS
---------------
`live/adapter.py` already argues for it and the live runs proved the point:
CRBAB's bar reader returns a FRACTION by colour-matching, and `_calculate_hp`
returns 0.0 both when the bar reads empty and when it cannot match the colours
at all. Measured on one Training Camp trial, over 77 consecutive frames:

    both of OUR princess towers          0.00 for all 77 frames (towers alive)
    the tower a Fireball actually hit    1.00 -> 0.00 -> 0.62 -> 0.67 -> 1.00

None of that second sequence was the Fireball. A reader that cannot tell a
live tower from an unreadable one is unusable as a clock, and it is what
blocked measuring the real game's spell delay.

The numeral has none of those problems. It is absolute HP, so no tower-level
table is needed -- maxima vary by account (measured 1750 at level 4, 1890 at
level 5, against the engine's level-9 2534), which makes a fraction wrong by
~30% and by a different factor per player. And "unreadable" is distinguishable
from "low", because a failed match reports low confidence rather than 0.

WHERE THE NUMERAL IS
--------------------
Derived from CRBAB's own bar bbox rather than calibrated separately, so there
is ONE definition of where a tower is and this cannot drift away from it. The
numeral sits directly ABOVE the bar: `NUMERAL_OFFSET` is applied to the bar
box after scaling it from detector space into the native frame.

It must be read at NATIVE resolution. In the 368x652 detector frame the whole
bar is 40x10 px and the glyphs inside it are ~6 px tall -- unreadable. At the
549x976 capture the same digits are ~14 px, which is comfortably above what
`normalise_glyph` needs.

SEGMENTATION IS BY INK, AND THE DIGIT COUNT VARIES
--------------------------------------------------
Same lesson `readers/clock.py` already paid for: fixed fractional splits clip
strokes and let neighbours bleed across boundaries, because glyph widths
differ -- a "1" is about a third the width of a "0". Column-projection
segmentation tracks that automatically.

Unlike the clock there is no colon to reject, but there IS a variable digit
count: a tower reads 2446 at full health and 887 after damage, and the count
changes mid-match as it drops through 1000. So this returns however many
digits it finds instead of asserting three.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from readers.clock import DigitTemplates, normalise_glyph

# Applied to the bar's bbox once it is in NATIVE pixels: (left, top, right,
# bottom) deltas, where `top`/`bottom` are relative to the bar's TOP edge.
#
# The bottom edge stops ABOVE the bar, and that is load-bearing rather than
# tidy. A first version ended at bar_top + 2, which included two pixels of the
# bar itself -- and the bar is a saturated pink stripe spanning the full ROI
# width. Otsu then put the bar and the digits on the same side of the
# threshold, so the column projection saw one unbroken run and segmentation
# returned a SINGLE cell for "2030". The digits were plainly legible in the
# crop; it was the bar underneath them that destroyed the projection.
NUMERAL_OFFSET = (-6, -19, 6, -3)

# Below this a reading is reported as not measured rather than guessed. A
# tower whose HP is unknown must stay distinguishable from one that is nearly
# dead: extra scalars 3-8 are tower HP, and a live tower reported as empty
# tells the policy a lane is already lost.
MIN_DIGIT_CONFIDENCE = 0.35

# A plausible Princess/King HP. Anything outside it is a misread, not a tower:
# the lowest real tower is well above 100 and the highest King is under 10000.
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
    """The numeral's box in native pixels, from CRBAB's bar box.

    Derived rather than calibrated so the tower's position has one definition.
    A separate calibration entry would be a second copy of the same fact and
    would drift from it the first time the detector's layout moved.
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
    """Per-pixel MINIMUM across colour channels, not luminance.

    The single most important line in this module, and it was found by the
    reader failing on a crop where the digits were plainly legible by eye.

    The numerals are near-white on ARENA GRASS. In luminance the grass reads
    ~169 and the digits ~230 -- both bright, both on the same side of Otsu, so
    the column projection saw one unbroken blob and "2030" segmented into a
    single cell. Converting to grey throws away exactly the information that
    separates them.

    The minimum channel does not: grass is a saturated green, so its blue is
    low (~90), while a near-white glyph is high in every channel (~220). Any
    saturated background -- grass, the pink HP bar, a blue tower roof -- drops
    out for the same reason, which is why this is more robust than picking one
    channel that happens to work on grass.
    """
    if patch.ndim == 3:
        return patch.min(axis=2).astype(np.uint8)
    return patch


def split_digits(patch: np.ndarray, min_height_frac: float = 0.45) -> list[np.ndarray]:
    """Split a numeral patch into digit cells by column projection.

    Returns however many it finds. The digit COUNT is not fixed here: a tower
    reads 2446 at full health and 887 after damage, and asserting a count
    would fail on exactly the frames where damage is what is being measured.

    Runs shorter than `min_height_frac` of the patch are dropped. That removes
    the bar's own edge and any stray highlight without needing a width rule,
    which is what `clock.py` found necessary to stop a narrow "1" being
    filtered out alongside the noise.
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
        # The RGB frame is passed through to split_digits, which needs the
        # colour to separate near-white glyphs from saturated backgrounds --
        # see ink_channel. Converting here would destroy that.
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
            # A number outside any real tower's range is a segmentation
            # failure that happened to produce digits -- reporting it would be
            # worse than reporting nothing, because it looks measured.
            return TowerNumeralReading(None, text, worst)
        return TowerNumeralReading(value, text, worst)


def normalise_for_templates(cell: np.ndarray, shape) -> np.ndarray:
    """Exposed so a template builder and the reader normalise identically.

    They must: templates cropped one way and matched another is the failure
    `clock.py` scored 24.7% on.
    """
    return normalise_glyph(cell, shape)
