"""Build a calibration profile from a recording, and score it honestly.

    perception/.venv/Scripts/python.exe perception/tools/calibrate.py \
        --video "perception/assets/recordings/<file>.mp4"

Detected automatically: the ground-plane landmarks a homography needs.

  * the river, the narrow band of saturated cyan water; its centre row is the
    y anchor;
  * the two bridges, the wide non-water gaps in that band; their centres are
    the x anchors;
  * the six tower platforms, low-saturation stone blobs. Kings are ~4 tiles
    wide, Princesses ~3.

The UI rectangles (elixir bar, clock, hand slots) are constants of the fixed
emulator window, measured once and asserted against the frame size on load.

None of the landmark residuals is the acceptance number, since the homography
is fitted to those points. Three are reported because their disagreement is
informative:

  in-sample      fit 8, score 8. Optimistic.
  documented     fit the 4 Princesses, score the bridges and Kings.
  leave-one-out  the least biased landmark estimate, and pessimistic: each
                 landmark carries a definitional bias the full fit absorbs.

The acceptance test is `tools/validate_grid.py`, which scores the mapping
against the arena's rendered tile seams, which the fit never saw. Do not tune
the landmark set to lower the residuals here: dropping the two hardcoded
`own_princess` constants improves every landmark metric and makes the mapping
3.4x worse in our own half (see validate_grid.py).
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

from calib.homography import Homography, save_profile  # noqa: E402
from geometry import load_geometry  # noqa: E402
from capture.video import VideoSource  # noqa: E402
from contracts import CalibrationProfile  # noqa: E402

# Emulator window content within the 1920x1080 desktop capture; the rest is
# OBS, the desktop and the taskbar.
GAME_RECT = (686, 40, 1236, 1012)

# --- UI rectangles, measured once for this fixed window ---
# (x, y, w, h).
UI_ROIS = {
    # Only rows 1000-1005 are free of the elixir number and the "Max: 10"
    # caption, which are drawn on top of the bar and blank whole columns. See
    # readers/elixir.py.
    "elixir_bar": (835, 1000, 381, 6),
    "clock": (1156, 69, 66, 28),
    "hand_slot_0": (809, 854, 98, 118),
    "hand_slot_1": (913, 854, 98, 118),
    "hand_slot_2": (1017, 854, 98, 118),
    "hand_slot_3": (1117, 854, 98, 118),
    "next_card": (717, 955, 42, 52),
}

# --- what the engine says these landmarks are, in tiles ---
# From geometry.load_geometry(), never copied.
def engine_tiles() -> dict[str, tuple[float, float]]:
    g = load_geometry()
    river_y = (g.river_y_start + g.river_y_end) / 2.0
    return {
        "own_princess_left": g.own_princess_left,
        "own_princess_right": g.own_princess_right,
        "opp_princess_left": g.opp_princess_left,
        "opp_princess_right": g.opp_princess_right,
        "left_bridge": (g.left_bridge[0], river_y),
        "right_bridge": (g.right_bridge[0], river_y),
        "own_king": g.own_king,
        "opp_king": g.opp_king,
    }


def landmark_metrics(screen: dict, tiles: dict) -> dict:
    """The three landmark residuals, which deliberately disagree.

    See the module docstring for why all three are reported and why none of
    them is the acceptance number.
    """
    shared = {k: v for k, v in screen.items() if k in tiles}
    names = sorted(shared)

    full = Homography.solve(shared, tiles)
    in_sample = full.measure_error(shared, tiles)

    princesses = {k: v for k, v in shared.items() if "princess" in k}
    documented = None
    if len(princesses) == 4:
        held = {k: v for k, v in shared.items() if "princess" not in k}
        # Four points is a closed-form solve, so its fit residual is zero; only
        # the held-out four carry information.
        documented = Homography.solve(princesses, tiles).measure_error(held, tiles)

    loo = {}
    for name in names:
        rest = {k: v for k, v in shared.items() if k != name}
        if len(rest) < 4:
            continue
        loo[name] = Homography.solve(rest, tiles).measure_error(
            {name: shared[name]}, tiles)[name]

    return {"full": full, "in_sample": in_sample, "documented": documented, "loo": loo}


def find_river_and_bridges(img: np.ndarray) -> dict:
    """River centre row and the two bridge centre columns."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    x0, y0, x1, y1 = GAME_RECT
    water = cv2.inRange(hsv, np.array([84, 110, 90]), np.array([102, 255, 255]))
    water[:, :x0 + 14] = 0
    water[:, x1 - 11:] = 0
    water[:y0 + 360] = 0
    water[y0 + 480:] = 0

    rows = water.sum(axis=1) / 255
    ys = [y for y in range(img.shape[0]) if rows[y] > 120]
    if not ys:
        raise SystemExit(
            "no river found. Either this frame is not mid-battle, or the "
            "arena skin's water is a different colour than this window was "
            "calibrated for."
        )
    river_y = (ys[0] + ys[-1]) / 2.0

    band = water[ys[0]:ys[-1] + 1, :]
    cols = band.sum(axis=0) / 255
    threshold = band.shape[0] * 0.4
    runs, run = [], None
    for x in range(x0 + 14, x1 - 11):
        if cols[x] < threshold:
            if run is None:
                run = x
        else:
            if run is not None and x - run >= 15:
                runs.append((run, x))
            run = None
    # The outermost runs are the arena walls, where the shore hides the water;
    # the bridges are the two widest interior gaps.
    interior = sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:2]
    if len(interior) < 2:
        raise SystemExit(f"expected 2 bridges in the river band, found {len(runs)}")
    interior.sort()
    return {
        "river_y": river_y,
        "left_bridge": ((interior[0][0] + interior[0][1]) / 2.0, river_y),
        "right_bridge": ((interior[1][0] + interior[1][1]) / 2.0, river_y),
    }


def find_towers(img: np.ndarray, river_y: float | None = None) -> dict:
    """The six tower platforms, by their low-saturation stone.

    Each blob is labelled from its own geometry (side of the river, wide or
    narrow, central or flanking), never from its rank among the blobs this
    frame yielded: a sort promotes a King into a Princess slot whenever a blob
    goes missing, and the median over frames then blends two landmarks. The
    King and Princess tests are disjoint (wide and central versus narrow and
    off-centre), so a partial frame is safe to use, which matters because our
    own Princess platforms sit behind their HP bars in most frames.

    Side comes from the river, detected independently and identical to the
    pixel (465.5) across all eight recordings.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    x0, y0, x1, y1 = GAME_RECT
    stone = cv2.inRange(hsv, np.array([0, 0, 60]), np.array([179, 60, 200]))
    stone = cv2.morphologyEx(stone, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    stone = cv2.morphologyEx(stone, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(stone, 8)
    blobs = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 1500 or w < 40 or h < 30:
            continue
        if x < x0 + 14 or x + w > x1 - 6 or y < y0 + 60 or y + h > y0 + 760:
            continue
        blobs.append({"cx": x + w / 2.0, "cy": y + h / 2.0, "w": w, "h": h, "area": area})

    centre_x = (x0 + x1) / 2.0
    if river_y is None:
        river_y = (y0 + y1) / 2.0

    out: dict[str, tuple[float, float]] = {}
    for blob in blobs:
        side = "opp" if blob["cy"] < river_y else "own"
        central = abs(blob["cx"] - centre_x) < 60
        if blob["w"] >= 78 and central:
            name = f"{side}_king"
        elif blob["w"] < 78 and not central:
            flank = "left" if blob["cx"] < centre_x else "right"
            name = f"{side}_princess_{flank}"
        else:
            continue  # neither shape -- a unit, a building, scenery
        if name in out:
            return {}  # two candidates for one tower: ambiguous, drop the frame
        out[name] = (blob["cx"], blob["cy"])
    return out


# Our own Princess platforms are routinely hidden by their HP bars and the
# tower troop sprite. They are supplied as measured constants only when no
# frame in the whole batch produced them, never mixed in per frame.
OWN_PRINCESS_FALLBACK = {
    "own_princess_left": (819.0, 656.0),
    "own_princess_right": (1101.0, 656.0),
}


# Seconds after the arena appears during which the board is still empty enough
# to calibrate on. A deployment lands in one lane and the detector needs only
# three of six towers per frame, so a few seconds of margin costs nothing.
OPENING_WINDOW_S = 4.0


def _battle_start_seconds(source: VideoSource) -> float | None:
    """First moment the arena is up, found by looking for the river. The
    pre-battle stretch varies between recordings.
    """
    duration = source.frame_count / source.fps
    step = 1.0
    t = 0.0
    while t < min(duration, 90.0):
        try:
            img = source.read_frame_at(int(t * source.fps)).image
            find_river_and_bridges(img)
            return t
        except (SystemExit, IndexError):
            t += step
    return None


def collect_landmarks(videos: list[Path], samples_per_video: int = 12
                      ) -> tuple[dict[str, tuple[float, float]], dict[str, float], int]:
    """Median landmark positions over many frames, and their spread.

    A single frame is not enough: a troop on a bridge shifts its detected
    centre by ~10 px (a third of a tile), and a unit on a tower platform can
    make the stone detector latch onto something else. Occlusions are transient
    and the geometry is not, so the median removes them. This is valid only
    because the emulator window is fixed, which `spread` verifies: if the
    window moves, the spread blows up rather than averaging two layouts.
    """
    collected: dict[str, list[tuple[float, float]]] = {}
    used = 0
    for video in videos:
        source = VideoSource(video)
        opening = _battle_start_seconds(source)
        if opening is None:
            source.close()
            continue
        # Only the opening seconds of each battle. The tower detector keys on
        # grey stone, and a deployed Cannon, a Tombstone and several units are
        # grey stone too; over whole matches they outnumber the towers and drag
        # the median off them. In the opening seconds every stone blob is a
        # tower.
        for offset in np.linspace(0.3, OPENING_WINDOW_S, samples_per_video):
            try:
                img = source.read_frame_at(int((opening + offset) * source.fps)).image
                river = find_river_and_bridges(img)
                towers = find_towers(img, river["river_y"])
            except (SystemExit, IndexError):
                continue
            if len(towers) < 3:
                continue
            used += 1
            found = {"left_bridge": river["left_bridge"],
                     "right_bridge": river["right_bridge"], **towers}
            for name, point in found.items():
                collected.setdefault(name, []).append(point)
        source.close()

    if used < 4:
        raise SystemExit(f"only {used} usable frames across {len(videos)} recordings")

    medians, spread = {}, {}
    for name, points in collected.items():
        array = np.array(points)
        medians[name] = (float(np.median(array[:, 0])), float(np.median(array[:, 1])))
        # Interquartile range, not min-max: a few occluded frames must not read
        # as the window having moved.
        iqr = np.percentile(array, 75, axis=0) - np.percentile(array, 25, axis=0)
        spread[name] = float(np.hypot(*iqr))
    return medians, spread, used


def build(videos: list[Path], out_path: Path) -> CalibrationProfile:
    source = VideoSource(videos[0])
    width, height = source.size
    source.close()

    screen, spread, used = collect_landmarks(videos)
    for name, point in OWN_PRINCESS_FALLBACK.items():
        screen.setdefault(name, point)
        spread.setdefault(name, 0.0)

    print(f"landmarks, median over {used} frames from {len(videos)} recording(s):")
    print(f"   {'landmark':20s} {'x':>9s} {'y':>9s} {'IQR px':>8s}")
    for name in sorted(screen):
        print(f"   {name:20s} {screen[name][0]:9.1f} {screen[name][1]:9.1f} "
              f"{spread[name]:8.2f}")
    worst = max(spread.values())
    if worst > 6.0:
        print(f"   WARNING: landmark spread {worst:.1f}px (>0.2 tile). The emulator "
              "window may have moved between recordings -- one profile cannot "
              "cover both.")

    anchors = {k: screen[k] for k in
               ("own_princess_left", "own_princess_right",
                "opp_princess_left", "opp_princess_right") if k in screen}
    if len(anchors) < 4:
        raise SystemExit(f"only found {len(anchors)} anchors: {sorted(anchors)}")

    tiles = engine_tiles()
    metrics = landmark_metrics(screen, tiles)
    homography = metrics["full"]
    in_sample, loo = metrics["in_sample"], metrics["loo"]

    print("\nlandmark residuals, in TILES (all three; none is the acceptance number):")
    print(f"   {'landmark':20s} {'in-sample':>10s} {'leave-1-out':>12s}")
    for name in sorted(screen):
        if name not in tiles:
            continue
        print(f"   {name:20s} {in_sample.get(name, float('nan')):10.2f} "
              f"{loo.get(name, float('nan')):12.2f}")
    loo_max = max(loo.values()) if loo else float("nan")
    loo_rms = float(np.sqrt(np.mean(np.square(list(loo.values()))))) if loo else float("nan")
    print(f"   {'max':20s} {in_sample['max']:10.2f} {loo_max:12.2f}")
    print(f"   {'rms':20s} {in_sample['rms']:10.2f} {loo_rms:12.2f}")
    if metrics["documented"] is not None:
        print(f"\n   documented split (fit 4 Princesses, score bridges+Kings): "
              f"max {metrics['documented']['max']:.2f} "
              f"rms {metrics['documented']['rms']:.2f}")

    # The leave-one-out max, since contracts.py and config/README.md describe
    # this field as measured on held-out landmarks.
    errors = {"max": loo_max}
    profile = CalibrationProfile(
        name=f"gpg_emulator_{width}x{height}",
        frame_width=width,
        frame_height=height,
        homography=tuple(float(v) for v in homography.matrix.reshape(-1)),
        anchors_screen={k: (float(v[0]), float(v[1])) for k, v in screen.items()},
        rois=dict(UI_ROIS),
        reprojection_error_tiles=float(errors["max"]),
    )
    save_profile(profile, out_path)
    print(f"\nwrote {out_path}")

    print(
        "\nThese are LANDMARK residuals and none of them decides acceptance.\n"
        "Run the independent check, which scores the mapping against the\n"
        "arena's own rendered tile seams:\n"
        "  perception/.venv/Scripts/python.exe perception/tools/validate_grid.py"
    )
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, nargs="*", default=None,
                        help="recordings to calibrate from; default is all of them")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    # All recordings by default: the geometry is shared (fixed window) and the
    # occlusions are not.
    videos = args.video or sorted((_ROOT / "assets" / "recordings").glob("*.mp4"))
    if not videos:
        raise SystemExit("no recordings found in perception/assets/recordings/")

    out = args.out or (_ROOT / "config" / "profile_gpg_1920x1080.json")
    build(list(videos), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
