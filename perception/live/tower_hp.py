"""Princess Tower HP from the printed numeral, as the fraction GameState wants.

`readers/tower_numerals.py` reads the absolute HP printed above each Princess
bar (acceptance: 2030 on both enemy towers of the reference frame). The live
adapter used CRBAB's bar instead, whose `_calculate_hp` returns 0.0 both for an
empty bar and for one it cannot colour-match -- at 549x976 it read a
full-health Princess as 0.0 on every frame, so every Princess reached the
mirror as "not measured" and the teacher scored every candidate against full
towers (`MirrorGaps.towers_assumed_full`).

The numeral is ABSOLUTE and `TowerObservation` is a FRACTION of the tower's own
maximum, which depends on the account's tower level and is not on screen (1750
at level 4, 1890 at level 5, the engine's 2534 at level 9). So this tracks each
tower's maximum through the match:

  * Towers start full and never heal, so the maximum is the highest value
    read -- but only once it has been read at least CONFIRMATIONS times. One
    misread glyph ("2030" read as "8030") would otherwise set a ceiling that
    makes every later reading look like a damaged tower, for the rest of the
    match.
  * Started mid-match, the first readings are already below the true maximum
    and the fraction reads high until nothing better exists. Nothing on screen
    can fix that; `reset()` at match start is the remedy.

When the numeral cannot be read the bar's reading stands (and is "not
measured" when the bar failed too). Destruction is left to a caller with
history, as before: an unreadable numeral is not a destroyed tower.

Templates exist for 549x976 only (`config/templates/tower_549x976`); a frame of
any other size is not read, and the bar reading stands.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from contracts import TowerObservation

TEMPLATES_DIR = (Path(__file__).resolve().parent.parent
                 / "config" / "templates" / "tower_549x976")
TEMPLATE_FRAME_SIZE = (549, 976)            # (width, height) the templates fit

#: Times a value must be read before it can be a tower's maximum.
CONFIRMATIONS = 2


def princess_bar_boxes() -> dict[str, tuple[int, int, int, int]]:
    """GameState field -> CRBAB's bar bbox in its detector frame, from CRBAB's
    own constants, so a tower's position has one definition."""
    from clashroyalebuildabot import constants as C  # noqa: PLC0415

    def box(x, y):
        return (x, y, x + C.HP_WIDTH, y + C.HP_HEIGHT)
    return {
        "own_princess_left": box(C.LEFT_PRINCESS_HP_X, C.ALLY_PRINCESS_HP_Y),
        "own_princess_right": box(C.RIGHT_PRINCESS_HP_X, C.ALLY_PRINCESS_HP_Y),
        "opp_princess_left": box(C.LEFT_PRINCESS_HP_X, C.ENEMY_PRINCESS_HP_Y),
        "opp_princess_right": box(C.RIGHT_PRINCESS_HP_X, C.ENEMY_PRINCESS_HP_Y),
    }


def detector_size() -> tuple[int, int]:
    from clashroyalebuildabot.constants import (  # noqa: PLC0415
        SCREENSHOT_HEIGHT,
        SCREENSHOT_WIDTH,
    )
    return (SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT)


class _TowerMax:
    """One tower's maximum, confirmed by repetition."""

    def __init__(self):
        self.counts: Counter = Counter()

    def observe(self, value: int) -> None:
        self.counts[int(value)] += 1

    @property
    def value(self) -> int | None:
        confirmed = [v for v, n in self.counts.items() if n >= CONFIRMATIONS]
        return max(confirmed) if confirmed else None


class TowerHpTracker:
    """Per-match state: each Princess Tower's confirmed maximum."""

    def __init__(self, reader=None, boxes=None, det_size=None):
        if reader is None:
            from readers.clock import DigitTemplates  # noqa: PLC0415
            from readers.tower_numerals import TowerNumeralReader  # noqa: PLC0415
            reader = TowerNumeralReader(DigitTemplates.load(TEMPLATES_DIR))
        self.reader = reader
        self.boxes = boxes if boxes is not None else princess_bar_boxes()
        self.det_size = det_size if det_size is not None else detector_size()
        self.reset()

    def reset(self) -> None:
        """A new match: every maximum is forgotten."""
        self._max = {name: _TowerMax() for name in self.boxes}
        self.numeral_reads = 0
        self.fallbacks = 0

    def read(self, native_frame, fallback: dict) -> dict:
        """The four Princess TowerObservations, numeral first.

        `fallback` holds the bar's TowerObservation per field; a field this
        cannot improve keeps it.
        """
        out = dict(fallback)
        h, w = native_frame.shape[:2]
        if (w, h) != TEMPLATE_FRAME_SIZE:
            self.fallbacks += len(self.boxes)
            return out
        for name, bar in self.boxes.items():
            reading = self.reader.read(native_frame, bar, self.det_size)
            tower = self._max[name]
            if reading.measured:
                tower.observe(reading.value)
            top = tower.value
            if reading.measured and top:
                self.numeral_reads += 1
                out[name] = TowerObservation(
                    hp_fraction=min(1.0, reading.value / top),
                    hp_measured=True, destroyed=False)
            else:
                self.fallbacks += 1
        return out

    def maxima(self) -> dict:
        return {name: t.value for name, t in self._max.items()}
