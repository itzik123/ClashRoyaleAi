"""Build a calibration profile from a recording, and score it honestly.

    perception/.venv/Scripts/python.exe perception/tools/calibrate.py \
        --video "perception/assets/recordings/<file>.mp4"

WHAT IS DETECTED AUTOMATICALLY
------------------------------
Everything on the ground plane, because those are the landmarks a homography
needs and the only ones that can be found reliably frame to frame:

  * the river, as the narrow band of saturated cyan water. Its centre row is
    the y anchor.
  * the two bridges, as the wide non-water gaps within that band. Their
    centres are the x anchors: simulator (4, 17) and (14, 17).
  * the six tower platforms, as low-saturation stone blobs. Kings are ~4
    tiles wide, Princesses ~3, which is what separates them.

The UI rectangles (elixir bar, clock, hand slots) are NOT detected -- they are
constants of the emulator window, which is fixed across every recording in
this batch. They were measured once, are recorded here, and are asserted
against the frame size on load.

THE SCORE USED TO BE COMPUTED TWICE. IT NO LONGER IS.
-----------------------------------------------------
Until 2026-07-30 this scored against the engine's stated geometry AND against a
hypothetically corrected one, because the two disagreed and the difference was
the finding (UPSTREAM_REQUESTS.md items 1-2). Both fixes have since landed in
the engine, `engine_tiles()` and the old `corrected_tiles()` were verified
identical, and the comparison was deleted exactly as its own comment said it
should be once they converged.

WHICH NUMBER IS THE ACCEPTANCE NUMBER
-------------------------------------
Not the one printed largest, and not any of the landmark residuals below. Every
landmark metric is self-referential -- the homography is fitted to those points.
Three are reported because they disagree and the disagreement is informative:

  in-sample      fit 8, score 8. Optimistic; this is what the tool used to
                 report as if it were held out.
  documented     fit the 4 Princesses, score the bridges and Kings.
  leave-one-out  the least biased landmark estimate, and pessimistic, because
                 each landmark carries its own definitional bias which the
                 full fit partially absorbs.

The real acceptance test is `tools/validate_grid.py`, which scores the mapping
against the arena's own rendered tile seams -- independent of the landmark set,
because the fit never saw them. Measured 2026-07-30: 0.105-0.224 tiles, PASS.

Do NOT "improve" the landmark set by whatever lowers the residuals here. That
was tried: dropping the two hardcoded `own_princess` constants improves every
landmark metric and makes the actual mapping 3.4x worse in our own half. See
validate_grid.py's docstring.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
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

# Emulator window content, within the 1920x1080 desktop capture. Everything
# else in the frame is OBS, the desktop and the taskbar.
GAME_RECT = (686, 40, 1236, 1012)

# --- UI rectangles, measured once for this fixed window -------------------
# (x, y, w, h). See the module docstring for why these are constants.
UI_ROIS = {
    # Only rows 1000-1005 of the bar are free of the elixir number and the
    # "Max: 10" caption, both of which are drawn ON TOP of it and blank whole
    # columns. See readers/elixir.py's docstring -- reading the full-height
    # bar pinned 450 of 589 samples near 0.5 elixir.
    "elixir_bar": (835, 1000, 381, 6),
    "clock": (1156, 69, 66, 28),
    "hand_slot_0": (809, 854, 98, 118),
    "hand_slot_1": (913, 854, 98, 118),
    "hand_slot_2": (1017, 854, 98, 118),
    "hand_slot_3": (1117, 854, 98, 118),
    "next_card": (717, 955, 42, 52),
}

# --- what the engine says these landmarks are, in tiles -------------------
# Read from geometry.load_geometry() rather than copied. A second copy of the
# board layout is exactly what went stale when the river was moved from
# [16,18) to [15.5,17.5) on 2026-07-29: this file would have gone on scoring
# the bridges against y=17.0 and reported a regression that did not exist.
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
        # Exactly four points is a closed-form solve, so its fit residual is
        # zero by construction -- only the held-out four carry information.
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
    # The outermost runs are the arena walls, where the shore hides the water.
    # The bridges are the two widest interior gaps.
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

    Each blob is labelled from its OWN geometry -- side of the river, wide or
    narrow, central or flanking -- never from its rank among whatever blobs
    this frame happened to yield.

    That distinction is the whole point. The first version sorted: kings by
    width, then the rest split top/bottom and left/right. Fine for one frame
    checked by eye, and quietly catastrophic in aggregate -- when a unit
    stands on a platform and one blob goes missing, the sort promotes a King
    into a Princess slot, and a median over frames then blends two different
    landmarks. Measured: `own_princess_left` came out at (958, 740), the
    King's position, with 132 px of spread, and the fit went from 0.60 to
    4.01 tiles.

    The King and Princess tests are deliberately disjoint (wide AND central
    versus narrow AND off-centre), so a King cannot land in a Princess slot
    however many towers are occluded. That is what makes a partial frame safe
    to use -- necessary, because our own Princess platforms sit behind their
    HP bars in most frames.

    Side comes from the river, which is detected independently and measured
    identical to the pixel (465.5) across all eight recordings.
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

    # Every blob is labelled from its OWN properties -- never from its rank
    # among the blobs that happened to be found this frame. The two tests are
    # disjoint by construction (a King is wide AND central, a Princess is
    # narrow AND off-centre), so a King can never land in a Princess slot no
    # matter which other towers are occluded. That is what makes it safe to
    # accept a partial frame, which matters because our own Princess
    # platforms are hidden behind their HP bars in most frames.
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


# Our own Princess platforms are routinely swallowed by their HP bars and the
# tower troop sprite. They are supplied as measured constants only when NO
# frame across the whole batch produced them -- never mixed in per-frame,
# which would blend a measured value with a constant and hide the fact.
OWN_PRINCESS_FALLBACK = {
    "own_princess_left": (819.0, 656.0),
    "own_princess_right": (1101.0, 656.0),
}


# Seconds after the arena appears during which the board is still empty
# enough to calibrate on. Both players start at 5 elixir, so the earliest
# possible deployment is immediate -- but a deployment lands in one lane and
# the detector needs only three of six towers per frame, so a few seconds of
# margin costs nothing and buys frames from every recording.
OPENING_WINDOW_S = 4.0


def _battle_start_seconds(source: VideoSource) -> float | None:
    """First moment the arena is up, found by looking for the river.

    Recordings start before "Battle" is pressed, and the menu/loading/intro
    stretch varies between them, so this cannot be a constant.
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

    A SINGLE frame is not enough, and the failure is not subtle. Landmarks get
    occluded constantly during a match: a troop standing on a bridge shifts
    the detected bridge centre by ~10 px (a third of a tile), and a unit
    parked on a tower platform can make the stone blob detector latch onto
    something else entirely -- measured on this batch as one recording
    scoring 6.50 tiles from a single frame while the other seven scored ~0.7.

    Taking the median over many frames from many recordings removes that,
    because occlusions are transient and the geometry is not. It is only valid
    because the emulator window is fixed, which is exactly what `spread`
    verifies: if the window ever moves, the spread blows up and says so
    instead of quietly averaging two different layouts together.
    """
    collected: dict[str, list[tuple[float, float]]] = {}
    used = 0
    for video in videos:
        source = VideoSource(video)
        opening = _battle_start_seconds(source)
        if opening is None:
            source.close()
            continue
        # ONLY the opening seconds of each battle, deliberately.
        #
        # Sampling across the whole match seems obviously better -- more
        # frames, more averaging -- and is much worse. The tower detector
        # keys on grey stone, and a deployed Cannon is grey stone; so is a
        # Tombstone, and so are several units. Over eight full matches those
        # outnumber the real towers and drag the median off them entirely:
        # measured, `own_princess_left` came out at (830, 563) with 61 px of
        # spread, versus its true (819, 656), and the fit went from 0.60 to
        # 2.98 tiles.
        #
        # In the opening seconds the board is empty by definition, so every
        # stone blob IS a tower. This is exactly the calibration frame the
        # recording instructions ask for, used the way they intended.
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
        # Interquartile range, not min-max: a handful of occluded frames
        # should not be mistaken for the window having moved.
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

    # Stored value is the LEAVE-ONE-OUT max, because contracts.py and
    # config/README.md both describe this field as measured on held-out
    # landmarks. It used to store the in-sample max, which is a different and
    # more flattering quantity -- 0.31 rather than 0.78 on this batch.
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

    # All recordings by default. More frames from more matches is strictly
    # better here: the geometry is shared (fixed window) and the occlusions
    # that corrupt any single frame are not.
    videos = args.video or sorted((_ROOT / "assets" / "recordings").glob("*.mp4"))
    if not videos:
        raise SystemExit("no recordings found in perception/assets/recordings/")

    out = args.out or (_ROOT / "config" / "profile_gpg_1920x1080.json")
    build(list(videos), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
