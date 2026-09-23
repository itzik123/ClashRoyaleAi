"""Score the calibration against the arena's own rendered tile seams.

    perception/.venv/Scripts/python.exe perception/tools/validate_grid.py

Every landmark-based number is self-referential: the homography is fitted to
the landmarks, and leaving one out mechanically flatters the residual over the
rest. Dropping the two hardcoded `own_princess` constants improved every
landmark metric (in-sample max 0.312 -> 0.173, leave-one-out 0.783 -> 0.537)
while mean |dy| in our own half went 0.105 -> 0.357 tiles; they are the fit's
only near-side y constraint.

The floor is rendered with visible tile seams the fit never saw, so warping a
frame into tile space, where a correct mapping puts every seam on an integer,
gives an error independent of the landmarks, in tiles. This is the number
stage-0 acceptance is read off.

Reported as the sub-tile phase of the seam comb per half of the board per
recording: the halves sit at different depths, so a camera-model error shows as
a difference between them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from calib.homography import Homography, homography_from_profile, load_profile  # noqa: E402
from capture.video import VideoSource  # noqa: E402
from geometry import load_geometry  # noqa: E402

# Rectified pixels per tile: 24 puts quantisation at 1/24 tile, well below the
# residual measured.
PIXELS_PER_TILE = 24

# Clean floor bands in tiles, one per half, excluding towers, river and walls,
# whose strong edges are not seams.
BANDS = {"own": (8.0, 14.0), "opp": (19.0, 25.0)}
X_BAND = (1.0, 17.0)

# Seconds into the battle to sample: the board is nearly empty, as in
# calibration itself.
SAMPLE_AT_S = 20.0

# Stage-0 acceptance in tiles: 0.5 leaves two thirds of the detector's 1.5-tile
# placement budget for detection error.
ACCEPTANCE_TILES = 0.5


def seam_phase(profile: np.ndarray, period: int = PIXELS_PER_TILE) -> float:
    """Sub-tile offset of a comb of period `period` best matching `profile`,
    wrapped to [-0.5, 0.5] (the sign says which way the seam sits). A
    brute-force scan: with ~16 periods the FFT's frequency resolution is
    comparable to the quantity measured.
    """
    prof = np.asarray(profile, dtype=np.float64)
    prof = prof - prof.mean()
    n = len(prof)
    best, best_score = 0.0, -np.inf
    for phi in np.linspace(0.0, 1.0, 401, endpoint=False):
        idx = np.round((np.arange(n // period) + phi) * period).astype(int)
        idx = idx[(idx >= 0) & (idx < n)]
        if idx.size < 4:
            continue
        score = float(prof[idx].mean())
        if score > best_score:
            best_score, best = score, float(phi)
    return best if best <= 0.5 else best - 1.0


def rectify(image: np.ndarray, homography: Homography, geom) -> np.ndarray:
    matrix = np.diag([PIXELS_PER_TILE, PIXELS_PER_TILE, 1.0]) @ homography.matrix
    return cv2.warpPerspective(
        image, matrix, (geom.width * PIXELS_PER_TILE, geom.height * PIXELS_PER_TILE)
    )


def measure(image: np.ndarray, homography: Homography, geom) -> dict[str, tuple[float, float]]:
    """(dx, dy) seam phase in tiles, per board half."""
    rect = rectify(image, homography, geom)
    grey = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY).astype(np.float32)
    out = {}
    for half, (y_lo, y_hi) in BANDS.items():
        sub = grey[
            int(y_lo * PIXELS_PER_TILE):int(y_hi * PIXELS_PER_TILE),
            int(X_BAND[0] * PIXELS_PER_TILE):int(X_BAND[1] * PIXELS_PER_TILE),
        ]
        # Absolute first derivative: seams are darker than tile faces, so the
        # gradient peaks on them whichever side is lit.
        gx = np.abs(cv2.Sobel(sub, cv2.CV_32F, 1, 0, ksize=3)).mean(axis=0)
        gy = np.abs(cv2.Sobel(sub, cv2.CV_32F, 0, 1, ksize=3)).mean(axis=1)
        out[half] = (seam_phase(gx), seam_phase(gy))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, nargs="*", default=None)
    parser.add_argument("--profile", type=Path,
                        default=_ROOT / "config" / "profile_gpg_1920x1080.json")
    parser.add_argument("--dump-rectified", type=Path, default=None,
                        help="write the rectified first frame here, to eyeball")
    args = parser.parse_args()

    videos = args.video or sorted((_ROOT / "assets" / "recordings").glob("*.mp4"))
    if not videos:
        raise SystemExit("no recordings found in perception/assets/recordings/")

    profile = load_profile(args.profile)
    geom = load_geometry()

    print(f"profile {profile.name}  (stored reprojection_error_tiles="
          f"{profile.reprojection_error_tiles:.3f})")
    print(f"seam phase in TILES; + means the rendered seam sits below/right of "
          f"the projected line\n")
    print(f"{'recording':<14}{'half':<6}{'dx':>8}{'dy':>8}")

    per_half: dict[str, list[tuple[float, float]]] = {k: [] for k in BANDS}
    for video in videos:
        source = VideoSource(video)
        homography = homography_from_profile(profile, (source.size[1], source.size[0], 3))
        image = source.read_frame_at(int(SAMPLE_AT_S * source.fps)).image
        source.close()
        if args.dump_rectified and video is videos[0]:
            cv2.imwrite(str(args.dump_rectified), rectify(image, homography, geom))
        for half, (dx, dy) in measure(image, homography, geom).items():
            per_half[half].append((dx, dy))
            print(f"{video.stem[-8:]:<14}{half:<6}{dx:>+8.3f}{dy:>+8.3f}")

    print()
    worst = 0.0
    for half, rows in per_half.items():
        a = np.abs(np.array(rows))
        print(f"{half} half   mean |dx| {a[:, 0].mean():.3f}   mean |dy| "
              f"{a[:, 1].mean():.3f}   n={len(rows)}")
        worst = max(worst, a[:, 0].mean(), a[:, 1].mean())

    verdict = "PASS" if worst < ACCEPTANCE_TILES else "FAIL"
    print(f"\nworst mean phase error {worst:.3f} tiles vs acceptance "
          f"< {ACCEPTANCE_TILES} -> {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
