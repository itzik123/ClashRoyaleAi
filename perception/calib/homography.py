"""Screen pixels <-> board tiles, via a planar homography.

The arena is rendered with a tilted perspective camera, so tiles are not
uniform on screen (far rows are shorter); an affine map is wrong by several
tiles at the far end. The playing surface is a plane, and the perspective image
of a plane is exactly a homography: eight degrees of freedom, four point
correspondences, no iteration.

Anchor on a tower's base, not its centre. Only where a tower meets the ground
lies on the modelled plane; its visual centre floats above it, with an error
that grows away from the camera axis and looks fine mid-board.

Calibration error is measured out-of-sample: solve() fits the four Princess
towers and measure_error() scores the held-out bridges and Kings. An exactly
determined 4-point solve has zero residual by construction. See
geometry.calibration_anchors.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from contracts import BoardGeometry, CalibrationProfile


class CalibrationError(ValueError):
    """Calibration is unusable. Never downgraded to a warning."""


class Homography:
    """A solved screen <-> tile mapping."""

    def __init__(self, matrix: np.ndarray):
        matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
        if not np.isfinite(matrix).all():
            raise CalibrationError("homography contains non-finite values")
        if abs(np.linalg.det(matrix)) < 1e-12:
            raise CalibrationError(
                "homography is singular -- the four anchor points are probably "
                "collinear or duplicated. Re-pick them."
            )
        self.matrix = matrix
        self.inverse = np.linalg.inv(matrix)

    # --- construction ---

    @classmethod
    def solve(
        cls,
        screen_points: dict[str, tuple[float, float]],
        tile_points: dict[str, tuple[float, float]],
    ) -> Homography:
        """Solve screen -> tile from named correspondences. Exactly four shared
        names uses the closed-form solve, more uses least squares. No RANSAC:
        with a handful of hand-picked landmarks it would discard a correct
        point to fit three mis-clicked ones.
        """
        names = sorted(set(screen_points) & set(tile_points))
        if len(names) < 4:
            raise CalibrationError(
                f"need >= 4 shared landmarks, got {len(names)}: {names}"
            )

        src = np.array([screen_points[n] for n in names], dtype=np.float32)
        dst = np.array([tile_points[n] for n in names], dtype=np.float32)

        if len(names) == 4:
            matrix = cv2.getPerspectiveTransform(src, dst)
        else:
            matrix, _ = cv2.findHomography(src, dst, method=0)
            if matrix is None:
                raise CalibrationError("findHomography failed to converge")
        return cls(matrix)

    # --- mapping ---

    def screen_to_tile(self, points: np.ndarray) -> np.ndarray:
        """(N,2) screen pixels -> (N,2) continuous tile coordinates."""
        return _apply(self.matrix, points)

    def tile_to_screen(self, points: np.ndarray) -> np.ndarray:
        """(N,2) tile coordinates -> (N,2) screen pixels."""
        return _apply(self.inverse, points)

    def screen_to_tile_index(
        self, points: np.ndarray, geom: BoardGeometry
    ) -> np.ndarray:
        """(N,2) screen pixels -> (N,2) integer tile indices, clamped.

        Rounds rather than floors. The engine places an entity at (x, y) in
        tile (int(x), int(y)), but a detected centre is a measurement with
        error on both sides, so nearest-tile is the right quantiser; flooring
        would bias every detection down and left by half a tile.
        """
        tiles = np.rint(self.screen_to_tile(points)).astype(int)
        tiles[:, 0] = np.clip(tiles[:, 0], 0, geom.width - 1)
        tiles[:, 1] = np.clip(tiles[:, 1], 0, geom.height - 1)
        return tiles

    # --- quality ---

    def measure_error(
        self,
        screen_points: dict[str, tuple[float, float]],
        tile_points: dict[str, tuple[float, float]],
    ) -> dict[str, float]:
        """Per-landmark reprojection error in tiles, plus 'max' and 'rms': tiles
        are the unit that compares against the detector's 1.5-tile placement
        tolerance.
        """
        names = sorted(set(screen_points) & set(tile_points))
        if not names:
            raise CalibrationError("no shared landmarks to measure against")

        src = np.array([screen_points[n] for n in names], dtype=np.float64)
        expected = np.array([tile_points[n] for n in names], dtype=np.float64)
        got = self.screen_to_tile(src)
        per_point = np.linalg.norm(got - expected, axis=1)

        out = {name: float(err) for name, err in zip(names, per_point)}
        out["max"] = float(per_point.max())
        out["rms"] = float(np.sqrt((per_point**2).mean()))
        return out


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Perspective-transform (N,2) points, dividing through by w."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.hstack([pts, np.ones((len(pts), 1))])
    projected = homogeneous @ matrix.T
    w = projected[:, 2:3]
    # A point on the camera's horizon maps to infinity, which inside the arena
    # means a wrong calibration; producing 1e17 would push a bogus tile index
    # downstream.
    if np.any(np.abs(w) < 1e-9):
        raise CalibrationError(
            "point maps to the horizon (w ~ 0) -- calibration is invalid"
        )
    return projected[:, :2] / w


# --- profile persistence ---


def save_profile(profile: CalibrationProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = asdict(profile)
    body["_comment"] = (
        "Calibration for ONE capture resolution. Every pixel constant here is "
        "invalidated by a resolution change, which is why frame_width/"
        "frame_height are part of the profile and asserted on load."
    )
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")


def load_profile(path: Path) -> CalibrationProfile:
    if not path.exists():
        raise CalibrationError(f"no calibration profile at {path}")
    body = json.loads(path.read_text(encoding="utf-8"))
    body.pop("_comment", None)
    body["homography"] = tuple(body["homography"])
    body["anchors_screen"] = {k: tuple(v) for k, v in body.get("anchors_screen", {}).items()}
    body["rois"] = {k: tuple(v) for k, v in body.get("rois", {}).items()}
    return CalibrationProfile(**body)


def homography_from_profile(
    profile: CalibrationProfile, frame_shape: tuple[int, ...]
) -> Homography:
    """Rebuild the mapping, refusing frames the profile was not made for. The size
    check is why this exists: a profile applied to a differently sized frame is
    wrong by a smooth, plausible scale factor with no error anywhere.
    """
    height, width = frame_shape[0], frame_shape[1]
    if (width, height) != (profile.frame_width, profile.frame_height):
        raise CalibrationError(
            f"profile {profile.name!r} was calibrated for "
            f"{profile.frame_width}x{profile.frame_height} but the frame is "
            f"{width}x{height}. Pixel constants do not survive a resolution "
            "change -- calibrate a new profile."
        )
    return Homography(np.array(profile.homography, dtype=np.float64).reshape(3, 3))
