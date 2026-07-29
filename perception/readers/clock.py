"""The match clock, read by digit template matching. Not Tesseract.

The clock is a fixed-position, fixed-font, four-glyph string over an alphabet
of eleven symbols (0-9 and the colon). A general OCR engine is built for
arbitrary text in arbitrary fonts and pays for that generality with a
segmentation stage, a language model, and tens of milliseconds per call --
all of it wasted here, and all of it capable of failing in ways a fixed
template simply cannot.

Nearest-neighbour over eleven cropped templates is a handful of array
operations, gives a usable confidence for free (the margin between the best
and second-best match), and cannot hallucinate a character outside the
alphabet.

TEMPLATES ARE BUILT FROM THE RECORDING, NOT SHIPPED
---------------------------------------------------
The glyphs are extracted once from a frame of the actual capture, at the
actual resolution, and cached in the calibration profile's directory.
Shipping pre-rendered templates would mean shipping an assumption about the
font's rasterisation at one specific scale, and any mismatch shows up as
systematically low match scores rather than as an obvious failure.

Until they exist, `read` raises. The clock reading is what the entire
timeline's resync depends on, so a guess here would corrupt every timestamp
downstream, not just this one field.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DIGIT_ALPHABET = "0123456789"

# Below this normalised-correlation score the best match is not trusted. A
# correct glyph against its own template scores well above 0.9; genuine
# confusions (8 vs 6, 3 vs 8) still score high, which is why the MARGIN below
# matters more than the absolute score.
MIN_TEMPLATE_SCORE = 0.60

# Required gap between best and second-best. This is the real discriminator:
# a partially occluded or motion-blurred digit matches several templates
# almost equally well, and that ambiguity is exactly what should lower
# confidence rather than being resolved arbitrarily.
MIN_TEMPLATE_MARGIN = 0.05


class ClockCalibrationMissing(NotImplementedError):
    """No clock ROI or no digit templates. See this module's docstring."""


@dataclass(frozen=True)
class ClockReading:
    seconds_remaining: float
    text: str
    confidence: float
    per_digit_confidence: tuple[float, ...]


# Canonical glyph size (height, width) every crop is normalised to before
# matching. See normalise_glyph.
GLYPH_SHAPE = (26, 18)


def normalise_glyph(cell: np.ndarray, shape: tuple[int, int] = GLYPH_SHAPE) -> np.ndarray:
    """Tightly crop a digit to its own ink and rescale to a canonical box.

    NOT cosmetic -- without it the templates are unusable, and the way they
    fail is instructive. The clock's three cells have different widths and the
    glyph sits at a different offset inside each, so pooling crops of the same
    digit from different cells averages several misaligned copies of it. The
    first version of this reader did exactly that and scored 24.7% on a
    recording where the correct answer was known for every frame.

    Cropping to the ink's bounding box removes cell width, glyph position and
    stroke scale in one step, so a "5" from the minutes cell and a "5" from
    the seconds cell become the same picture.

    ASPECT RATIO IS PRESERVED, and that is not a detail. A "1" is about a
    third the width of a "0" at the same height, and that proportion is the
    single most reliable thing distinguishing it. Rescaling the bounding box
    to a fixed width stretches the 1 into a fat bar that matches 3, 7 and 9
    almost as well as itself -- measured here as a persistent 1->3 confusion
    that survived two other fixes. So the glyph is scaled by HEIGHT and then
    centre-padded to the canonical width, leaving the 1 narrow.

    Otsu rather than a fixed threshold: the clock sits on a wooden banner
    whose brightness varies with the arena skin, while the digits are always
    the brightest thing in the cell.
    """
    if cell.size == 0:
        return np.zeros(shape, np.uint8)
    _score, ink = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(ink)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        if w >= 2 and h >= 4:
            cell = cell[y:y + h, x:x + w]

    target_h, target_w = shape
    scale = target_h / cell.shape[0]
    new_w = max(1, min(target_w, int(round(cell.shape[1] * scale))))
    scaled = cv2.resize(cell, (new_w, target_h), interpolation=cv2.INTER_AREA)

    out = np.zeros(shape, np.uint8)
    left = (target_w - new_w) // 2
    out[:, left:left + new_w] = scaled
    return out


class DigitTemplates:
    """Glyph templates cropped from the capture itself."""

    def __init__(self, templates: dict[str, np.ndarray]):
        missing = set(DIGIT_ALPHABET) - set(templates)
        if missing:
            raise ClockCalibrationMissing(
                f"digit templates missing for {sorted(missing)}. Build a full "
                "set with perception/tools/build_digit_templates.py from a "
                "recording that shows every digit -- a partial set silently "
                "misreads the digits it lacks as whichever glyph it does have."
            )
        self.templates = {k: v for k, v in templates.items()}
        shapes = {v.shape for v in self.templates.values()}
        if len(shapes) != 1:
            raise ClockCalibrationMissing(
                f"digit templates have inconsistent shapes {shapes} -- they "
                "must all be cropped to the same cell size."
            )
        self.shape = shapes.pop()

    @classmethod
    def load(cls, directory: Path) -> DigitTemplates:
        directory = Path(directory)
        index = directory / "digits.json"
        if not index.exists():
            raise ClockCalibrationMissing(
                f"no digit templates at {directory}. The clock is the anchor "
                "for every timestamp in the pipeline, so this is not "
                "defaulted -- build them from the recording first."
            )
        meta = json.loads(index.read_text(encoding="utf-8"))
        templates = {}
        for glyph, filename in meta["glyphs"].items():
            image = cv2.imread(str(directory / filename), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ClockCalibrationMissing(f"could not read template {filename}")
            templates[glyph] = image
        return cls(templates)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        glyphs = {}
        for glyph, image in self.templates.items():
            filename = f"digit_{glyph}.png"
            cv2.imwrite(str(directory / filename), image)
            glyphs[glyph] = filename
        (directory / "digits.json").write_text(
            json.dumps({"glyphs": glyphs, "shape": list(self.shape)}, indent=2),
            encoding="utf-8",
        )

    def classify(self, cell: np.ndarray) -> tuple[str, float]:
        """Best-matching glyph and a confidence from the match margin."""
        cell = normalise_glyph(cell, self.shape)
        scores = {
            glyph: float(cv2.matchTemplate(cell, template, cv2.TM_CCOEFF_NORMED).max())
            for glyph, template in self.templates.items()
        }
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best, best_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best_score - runner_up

        if best_score < MIN_TEMPLATE_SCORE or margin < MIN_TEMPLATE_MARGIN:
            # Reported, not resolved. A low-margin digit is genuinely
            # ambiguous and MatchClock rejects implausible corrections on
            # exactly this basis -- see its resync().
            return best, max(0.0, min(1.0, margin / MIN_TEMPLATE_MARGIN)) * 0.5
        return best, min(1.0, 0.5 + margin)


class ClockReader:
    """Reads M:SS from a calibrated ROI using digit templates."""

    def __init__(self, roi: tuple[int, int, int, int] | None, templates: DigitTemplates | None):
        if roi is None:
            raise ClockCalibrationMissing(
                "no 'clock' ROI in the calibration profile -- run "
                "perception/tools/calibrate.py."
            )
        if templates is None:
            raise ClockCalibrationMissing(
                "no digit templates supplied -- see this module's docstring."
            )
        self.roi = roi
        self.templates = templates

    def read(self, frame: np.ndarray) -> ClockReading:
        x, y, w, h = self.roi
        patch = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)

        # M:SS -- three digit cells, with the colon's column skipped rather
        # than classified. The colon never changes, so matching it would only
        # ever add a way to fail.
        cells = _split_mss_cells(patch)
        digits, confidences = [], []
        for cell in cells:
            glyph, confidence = self.templates.classify(cell)
            digits.append(glyph)
            confidences.append(confidence)

        text = f"{digits[0]}:{digits[1]}{digits[2]}"
        seconds = int(digits[0]) * 60 + int(digits[1]) * 10 + int(digits[2])

        # A seconds field above 59 is arithmetically impossible, so it is
        # proof of a misread rather than a surprising value -- worth
        # collapsing confidence for even when the per-digit margins looked
        # fine.
        tens = int(digits[1])
        plausible = tens <= 5
        confidence = min(confidences) * (1.0 if plausible else 0.0)

        return ClockReading(
            seconds_remaining=float(seconds),
            text=text,
            confidence=round(confidence, 4),
            per_digit_confidence=tuple(round(c, 4) for c in confidences),
        )


# Fallback proportional split, used only when ink segmentation cannot find
# three digits. Kept because a reading with low confidence is more useful
# than an exception on one bad frame.
_FALLBACK_BOUNDS = [(0.00, 0.28), (0.38, 0.66), (0.68, 1.00)]


def _split_mss_cells(patch: np.ndarray) -> list[np.ndarray]:
    """Split an M:SS patch into its three digit cells, by ink.

    Segmented from the digits' own column projection rather than at fixed
    fractions of the ROI. The fixed-fraction version is the obvious approach
    -- the layout IS fixed -- but it was measurably wrong: it clipped strokes
    and let neighbouring digits bleed across cell edges, and scored 50% on a
    recording where every frame's answer was known.

    The problem is that digit glyphs are not equal width. A "1" is half the
    width of a "0", so the gaps between digits move as the clock counts down,
    and no single set of boundaries is right for every value. Ink segmentation
    tracks that automatically.

    The colon is discarded by size: it is two small dots, far narrower than
    any digit, so filtering runs by width removes it without matching it.
    """
    h, w = patch.shape
    _score, ink = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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

    # The colon is separated by HEIGHT, not width. Width alone does not work:
    # a "1" is barely wider than the colon, so any width threshold that keeps
    # the 1 also keeps the colon -- which then makes four runs, fails the
    # "exactly three" test, and silently falls back to fixed boundaries for
    # every frame. That is what happened here, and the visible symptom was a
    # single corrupted template: the "1" cell caught the colon alongside it,
    # so the "1" glyph trained as a blend and was thereafter confused with 3.
    #
    # A digit spans most of the ROI's height; the colon is two dots in the
    # middle third. That separates them cleanly regardless of glyph width.
    digits = []
    for a, b in runs:
        column_ink = np.where((ink[:, a:b] > 0).any(axis=1))[0]
        if column_ink.size and (column_ink[-1] - column_ink[0] + 1) >= h * 0.5:
            digits.append((a, b))

    if len(digits) != 3:
        return [patch[:, int(a * w):int(b * w)] for a, b in _FALLBACK_BOUNDS]
    return [patch[:, a:b] for a, b in digits]
