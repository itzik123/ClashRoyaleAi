"""Calibration maths, verified against a known synthetic camera.

No recording is needed to prove the solver correct: a homography is exactly
invertible, so a known transform can be applied to the tile landmarks to
produce synthetic "screen" points, and the solver must recover the transform
that generated them. Anything it gets wrong here it would also get wrong on a
real frame.

What this cannot check is whether a HUMAN picks the right pixels on a real
frame -- that is what the held-out landmark error in stage 0's acceptance
criterion measures, and it needs a recording.
"""

from __future__ import annotations

import numpy as np
import pytest

from calib.homography import CalibrationError, Homography, _apply
from geometry import calibration_anchors, load_geometry, validation_landmarks


def _synthetic_camera() -> np.ndarray:
    """A tile -> screen transform resembling the game's actual camera.

    Scales tiles to pixels, then applies a mild perspective foreshortening in
    y so far rows are compressed -- the property that makes a plain affine fit
    wrong by several tiles at the far end of the board.
    """
    scale = np.array([
        [42.0, 0.0, 120.0],
        [0.0, 38.0, 90.0],
        [0.0, 0.0, 1.0],
    ])
    perspective = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.00042, 1.0],
    ])
    return perspective @ scale


def _project(tile_points: dict[str, tuple[float, float]], camera: np.ndarray) -> dict:
    return {
        name: tuple(_apply(camera, np.array([point]))[0])
        for name, point in tile_points.items()
    }


def test_solver_recovers_a_known_camera():
    geom = load_geometry()
    camera = _synthetic_camera()
    anchors_tile = calibration_anchors(geom)
    anchors_screen = _project(anchors_tile, camera)

    homography = Homography.solve(anchors_screen, anchors_tile)

    # Held-out landmarks: bridges and Kings were never fitted.
    validation_tile = validation_landmarks(geom)
    validation_screen = _project(validation_tile, camera)
    errors = homography.measure_error(validation_screen, validation_tile)

    assert errors["max"] < 0.01, (
        f"solver failed to invert a known camera; per-landmark errors {errors}"
    )


def test_stage0_acceptance_criterion_on_synthetic_camera():
    """The stage-0 bar: held-out landmarks within 0.5 tiles.

    Run here against a perfect camera to prove the criterion is measurable
    and the measurement is out-of-sample. On a real frame the same call
    produces the real number, and the same threshold applies.
    """
    geom = load_geometry()
    camera = _synthetic_camera()
    anchors_tile = calibration_anchors(geom)
    homography = Homography.solve(_project(anchors_tile, camera), anchors_tile)

    validation_tile = validation_landmarks(geom)
    errors = homography.measure_error(_project(validation_tile, camera), validation_tile)
    assert errors["max"] < 0.5


def test_mis_clicked_anchor_shows_up_in_held_out_error():
    """A calibration mistake must be visible, not absorbed by the fit.

    With exactly four correspondences the fit residual is zero by
    construction no matter how badly the points were picked, which is the
    whole reason the error is measured on held-out landmarks instead.
    """
    geom = load_geometry()
    camera = _synthetic_camera()
    anchors_tile = calibration_anchors(geom)
    anchors_screen = _project(anchors_tile, camera)

    # Shift one anchor by 25 pixels, as a hurried click would.
    bad = dict(anchors_screen)
    key = "opp_princess_left"
    bad[key] = (bad[key][0] + 25.0, bad[key][1] - 25.0)

    homography = Homography.solve(bad, anchors_tile)

    # Zero up to float round-off, which is ~5 orders of magnitude below the
    # 0.5-tile acceptance threshold -- i.e. the fit tells you nothing.
    fit_error = homography.measure_error(bad, anchors_tile)
    assert fit_error["max"] < 1e-4, "4-point fit residual should be ~zero even when wrong"

    validation_tile = validation_landmarks(geom)
    held_out = homography.measure_error(_project(validation_tile, camera), validation_tile)
    assert held_out["max"] > 0.5, "a 25px mis-click must break the acceptance criterion"


def test_tile_indices_round_and_clamp():
    geom = load_geometry()
    camera = _synthetic_camera()
    anchors_tile = calibration_anchors(geom)
    homography = Homography.solve(_project(anchors_tile, camera), anchors_tile)

    screen = _project({"p": (7.4, 12.6)}, camera)["p"]
    tiles = homography.screen_to_tile_index(np.array([screen]), geom)
    assert tuple(tiles[0]) == (7, 13), "should round to nearest, not floor"

    far = _project({"p": (-8.0, 99.0)}, camera)["p"]
    tiles = homography.screen_to_tile_index(np.array([far]), geom)
    assert 0 <= tiles[0][0] <= geom.width - 1
    assert 0 <= tiles[0][1] <= geom.height - 1


def test_roundtrip_screen_tile_screen():
    geom = load_geometry()
    camera = _synthetic_camera()
    anchors_tile = calibration_anchors(geom)
    homography = Homography.solve(_project(anchors_tile, camera), anchors_tile)

    tiles = np.array([[0.0, 0.0], [17.0, 33.0], [8.5, 17.0], [4.0, 17.0]])
    back = homography.screen_to_tile(homography.tile_to_screen(tiles))
    assert np.allclose(back, tiles, atol=1e-6)


def test_degenerate_anchors_are_rejected():
    collinear_screen = {
        "a": (0.0, 0.0), "b": (10.0, 0.0), "c": (20.0, 0.0), "d": (30.0, 0.0),
    }
    collinear_tile = {
        "a": (0.0, 0.0), "b": (1.0, 0.0), "c": (2.0, 0.0), "d": (3.0, 0.0),
    }
    with pytest.raises(CalibrationError):
        Homography.solve(collinear_screen, collinear_tile)


def test_too_few_anchors_rejected():
    with pytest.raises(CalibrationError, match=">= 4"):
        Homography.solve({"a": (0.0, 0.0)}, {"a": (0.0, 0.0)})


def test_anchors_and_validation_landmarks_are_disjoint():
    """The held-out set must actually be held out."""
    geom = load_geometry()
    assert not (set(calibration_anchors(geom)) & set(validation_landmarks(geom)))
