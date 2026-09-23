"""Tower HP, read from the on-screen bars. Validation only.

Nothing here may enter the observation, reach the policy, or be fed to the
simulator. The battle is deterministic, so once what was played, where and when
are known, the simulator derives HP itself; observing it as an input would
replace a derived quantity with a noisy measurement of it and hide errors in
the event stream.

But an estimator with no observed quantity cannot know it has drifted. Tower HP
is the cheapest checksum: six slowly changing numbers at fixed positions, and
every upstream failure (a missed placement, a misclassified card, a
mislocalised tile) eventually shows as predicted HP diverging from observed. So
it is measured, compared and reported, never consumed.

The Kings are read but reported separately, since SimDriver.divergence excludes
them.

Bar fraction, not digits: the same pixel-counting as the elixir bar. Converting
to absolute HP needs the tower's max HP, supplied by the caller, since a wrong
maximum scales every reading and the divergence curve then looks like drift.
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

# HSV windows for the HP bar's two team colours.
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

    `max_hp` is supplied per tower: the recorded game's maxima depend on the
    account's tower level, which the simulator does not model, and getting them
    wrong scales the divergence curve by a constant that looks like real drift.
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

        # The bar drains from one end, so the fill is a prefix; measuring the
        # longest prefix means a stray team-coloured pixel elsewhere adds
        # nothing.
        prefix = 0
        for filled in column_filled:
            if not filled:
                break
            prefix += 1
        fraction = prefix / max(1, len(column_filled))

        # A total matching count far above the prefix means the ROI is catching
        # something beyond the bar.
        total = int(column_filled.sum())
        confidence = 1.0 if total <= prefix + 2 else 0.4

        return TowerReading(
            fraction=round(fraction, 4),
            hp=int(round(fraction * self.max_hp[key])),
            confidence=confidence,
        )


def divergence_excluding_kings(predicted: dict[str, int], observed: dict[str, int]) -> float:
    """Mean absolute HP error over the Princess towers only, mirroring
    SimDriver.divergence.
    """
    keys = [k for k in PRINCESS_KEYS if k in predicted and k in observed]
    if not keys:
        return 0.0
    return sum(abs(predicted[k] - observed[k]) for k in keys) / len(keys)


def king_divergence(predicted: dict[str, int], observed: dict[str, int]) -> float:
    """King-tower divergence, reported separately as a diagnostic."""
    keys = [k for k in ("own_king", "opp_king") if k in predicted and k in observed]
    if not keys:
        return 0.0
    return sum(abs(predicted[k] - observed[k]) for k in keys) / len(keys)
