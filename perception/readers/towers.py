"""Tower HP, read from the on-screen bars. VALIDATION ONLY.

READ THIS BEFORE USING ANYTHING IN THIS FILE
--------------------------------------------
Nothing here may enter the observation vector, reach the policy, or be fed
into the simulator. Not as a feature, not as a correction, not as a tie-break.

The brief was explicit that HP should not be an input, and it is right: the
battle is deterministic, so once WHAT was played, WHERE and WHEN are known,
the simulator derives HP itself. Observing it as an input would replace a
derived quantity with a noisy measurement of the same thing and hide errors
in the event stream instead of exposing them.

But a state estimator with no observed quantity at all cannot know it has
drifted. Tower HP is the cheapest possible checksum: six numbers, changing
slowly, rendered as bars at fixed positions. Every upstream failure -- a
missed placement, a misclassified card, a mislocalised tile -- eventually
shows up as predicted HP diverging from observed HP.

So it is measured, compared, and reported. Never consumed.

WHY THE KING TOWERS ARE READ BUT REPORTED SEPARATELY
-----------------------------------------------------
The engine's King Tower has no activation condition -- Tower.h gives it none,
so it fires from tick 0, while the real King is dormant until activated. King
HP therefore diverges systematically from the first second of every match no
matter how good perception is. Folding it into one number would add a large
constant error to every measurement and bury the signal.
SimDriver.divergence excludes the Kings for exactly this reason; they are
still read here because their divergence is itself a useful diagnostic -- it
should track the KNOWN discrepancy, and if it does not, something else is
wrong too.

BAR FRACTION, NOT DIGITS
------------------------
The bar is a filled fraction at a fixed position, so the same pixel-counting
approach as the elixir bar applies, and for the same reasons: no OCR, no
model, deterministic. Converting a fraction to absolute HP needs the tower's
max HP, which depends on the Tower Troop variant in play -- supplied by the
caller, never assumed, because a wrong maximum scales every reading by a
constant and the resulting divergence curve looks like a real drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

TOWER_KEYS = (
    "own_king", "own_princess_left", "own_princess_right",
    "opp_king", "opp_princess_left", "opp_princess_right",
)

PRINCESS_KEYS = tuple(k for k in TOWER_KEYS if "princess" in k)

# HSV windows for the two team colours of the HP bar.
BLUE_HUE_RANGE = (95, 125)
RED_HUE_RANGE = (0, 12)
MIN_SATURATION = 100
MIN_VALUE = 70


class TowerCalibrationMissing(NotImplementedError):
    """Tower bar ROIs or max-HP values are absent."""


@dataclass(frozen=True)
class TowerReading:
    fraction: float
    hp: int
    confidence: float


class TowerHpReader:
    """Reads the six tower HP bars.

    `max_hp` must be supplied per tower. The engine's own values are King
    4008 and Princess whatever towerTroopStats() gives for the variant in
    play (GameManager::addTower) -- but the RECORDED game's values depend on
    card levels, which this simulator does not model at all. So they are a
    caller input, and getting them wrong scales the divergence curve by a
    constant that is easy to mistake for real drift.
    """

    def __init__(
        self,
        rois: dict[str, tuple[int, int, int, int]] | None,
        max_hp: dict[str, int] | None,
    ):
        if not rois:
            raise TowerCalibrationMissing(
                "no tower bar ROIs in the calibration profile -- run "
                "perception/tools/calibrate.py."
            )
        missing = set(TOWER_KEYS) - set(rois)
        if missing:
            raise TowerCalibrationMissing(f"tower ROIs missing for {sorted(missing)}")
        if not max_hp or set(TOWER_KEYS) - set(max_hp):
            raise TowerCalibrationMissing(
                "max HP must be supplied for every tower. The recorded game's "
                "values depend on card levels, which this simulator does not "
                "model -- so there is no correct default, and a wrong one "
                "rescales the whole divergence curve."
            )
        self.rois = rois
        self.max_hp = max_hp

    def read(self, frame: np.ndarray) -> dict[str, TowerReading]:
        return {key: self._read_bar(frame, key) for key in TOWER_KEYS}

    def read_hp(self, frame: np.ndarray) -> dict[str, int]:
        """Just the HP values, in the shape SimDriver.divergence expects."""
        return {key: reading.hp for key, reading in self.read(frame).items()}

    def _read_bar(self, frame: np.ndarray, key: str) -> TowerReading:
        x, y, w, h = self.rois[key]
        patch = frame[y:y + h, x:x + w]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

        hue = BLUE_HUE_RANGE if key.startswith("own") else RED_HUE_RANGE
        lo = np.array([hue[0], MIN_SATURATION, MIN_VALUE], np.uint8)
        hi = np.array([hue[1], 255, 255], np.uint8)
        mask = cv2.inRange(hsv, lo, hi) > 0

        band = mask[h // 4: max(h // 4 + 1, 3 * h // 4), :]
        column_filled = band.mean(axis=0) > 0.5

        # The bar drains from one end, so the fill is a prefix. Measuring the
        # longest prefix rather than the total count means a stray matching
        # pixel elsewhere in the ROI -- a bit of team-coloured scenery -- adds
        # nothing instead of inflating the reading.
        prefix = 0
        for filled in column_filled:
            if not filled:
                break
            prefix += 1
        fraction = prefix / max(1, len(column_filled))

        # If the total matching count greatly exceeds the prefix, the ROI is
        # picking up something beyond the bar and the reading is suspect.
        total = int(column_filled.sum())
        confidence = 1.0 if total <= prefix + 2 else 0.4

        return TowerReading(
            fraction=round(fraction, 4),
            hp=int(round(fraction * self.max_hp[key])),
            confidence=confidence,
        )


def divergence_excluding_kings(predicted: dict[str, int], observed: dict[str, int]) -> float:
    """Mean absolute HP error over the Princess towers only.

    Mirrors SimDriver.divergence so the two cannot drift apart. See this
    module's docstring for why the Kings are excluded.
    """
    keys = [k for k in PRINCESS_KEYS if k in predicted and k in observed]
    if not keys:
        return 0.0
    return sum(abs(predicted[k] - observed[k]) for k in keys) / len(keys)


def king_divergence(predicted: dict[str, int], observed: dict[str, int]) -> float:
    """King-tower divergence, reported separately as a diagnostic.

    Expected to be non-zero and to grow -- the engine's King never sleeps.
    Useful because it should track that KNOWN discrepancy; if it behaves
    differently, something beyond the activation gap is wrong.
    """
    keys = [k for k in ("own_king", "opp_king") if k in predicted and k in observed]
    if not keys:
        return 0.0
    return sum(abs(predicted[k] - observed[k]) for k in keys) / len(keys)
