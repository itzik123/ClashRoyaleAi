"""The match clock, read by digit template matching.

The clock is a fixed-position, fixed-font, four-glyph string over eleven
symbols (0-9 and the colon). Nearest-neighbour over eleven templates is a
handful of array operations, gives a confidence for free (the margin between
best and second-best), and cannot produce a character outside the alphabet; a
general OCR engine pays tens of milliseconds for generality this does not need.

Templates are extracted from the capture itself at its own resolution and
cached in the calibration profile's directory; shipped templates would carry a
rasterisation assumption that fails as quietly low scores. Until they exist
`read` raises: the timeline's resync depends on this reading.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DIGIT_ALPHABET = "0123456789"

# Below this normalised correlation the best match is not trusted. Genuine
# confusions (8 vs 6, 3 vs 8) also score high, so the margin below matters
# more.
MIN_TEMPLATE_SCORE = 0.60

# Required gap between best and second-best: the real discriminator. An
# occluded or blurred digit matches several templates almost equally, which
# should lower confidence rather than be resolved arbitrarily.
MIN_TEMPLATE_MARGIN = 0.05


class ClockCalibrationMissing(NotImplementedError):
    """No clock ROI or no digit templates. See this module's docstring."""


@dataclass(frozen=True)
class ClockReading:
    seconds_remaining: float
    text: str
    confidence: float
    per_digit_confidence: tuple[float, ...]


# Canonical glyph size (height, width) every crop is normalised to. See
# normalise_glyph.
GLYPH_SHAPE = (26, 18)


def normalise_glyph(cell: np.ndarray, shape: tuple[int, int] = GLYPH_SHAPE) -> np.ndarray:
    """Tightly crop a digit to its own ink and rescale to a canonical box.

    The clock's cells differ in width and the glyph sits at a different offset
    in each, so pooling raw crops of one digit averages misaligned copies.
    Cropping to the ink's bounding box removes cell width, position and stroke
    scale at once.

    Aspect ratio is preserved: a "1" is about a third the width of a "0", and
    stretching it to a fixed width makes it match 3, 7 and 9 almost as well as
    itself. The glyph is scaled by height and centre-padded.

    Otsu rather than a fixed threshold: the wooden banner's brightness varies
    with the arena skin, while the digits are always the brightest thing in the
    cell.
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
            # Reported, not resolved: MatchClock rejects implausible
            # corrections on exactly this basis (see its resync()).
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

        # M:SS: three digit cells, the colon's column skipped rather than
        # classified.
        cells = _split_mss_cells(patch)
        digits, confidences = [], []
        for cell in cells:
            glyph, confidence = self.templates.classify(cell)
            digits.append(glyph)
            confidences.append(confidence)

        text = f"{digits[0]}:{digits[1]}{digits[2]}"
        seconds = int(digits[0]) * 60 + int(digits[1]) * 10 + int(digits[2])

        # A seconds field above 59 is proof of a misread, so confidence
        # collapses even if the per-digit margins looked fine.
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
# three digits: a low-confidence reading beats an exception on one bad frame.
_FALLBACK_BOUNDS = [(0.00, 0.28), (0.38, 0.66), (0.68, 1.00)]


def _split_mss_cells(patch: np.ndarray) -> list[np.ndarray]:
    """Split an M:SS patch into its three digit cells, by ink.

    Fixed fractions of the ROI clip strokes and let digits bleed across edges,
    because glyphs are not equal width and the gaps move as the clock counts
    down. Column-projection segmentation tracks that.
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

    # The colon is separated by height, not width: a "1" is barely wider than
    # the colon, so a width threshold keeping the 1 keeps the colon too, making
    # four runs and silently falling back to fixed boundaries. A digit spans
    # most of the ROI's height; the colon is two dots in the middle third.
    digits = []
    for a, b in runs:
        column_ink = np.where((ink[:, a:b] > 0).any(axis=1))[0]
        if column_ink.size and (column_ink[-1] - column_ink[0] + 1) >= h * 0.5:
            digits.append((a, b))

    if len(digits) != 3:
        return [patch[:, int(a * w):int(b * w)] for a, b in _FALLBACK_BOUNDS]
    return [patch[:, a:b] for a, b in digits]
