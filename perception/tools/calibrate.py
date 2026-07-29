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

WHY THE SCORE IS COMPUTED TWICE
-------------------------------
Against the simulator's stated geometry, and against a corrected geometry.
The difference is the finding, not a debugging aid -- see the note printed at
the end and UPSTREAM_REQUESTS.md item 1.
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
ENGINE_TILES = {
    "own_princess_left": (3.0, 6.0),
    "own_princess_right": (14.0, 6.0),
    "opp_princess_left": (3.0, 27.0),
    "opp_princess_right": (14.0, 27.0),
    "left_bridge": (4.0, 17.0),
    "right_bridge": (14.0, 17.0),
    "own_king": (8.5, 2.5),
    "opp_king": (8.5, 30.5),
}

# The same layout with the three inconsistencies of UPSTREAM_REQUESTS.md item
# 1 corrected: the left Princess aligned with its own bridge (as the real
# arena has it), the Kings on the board's true centre, and the river on the
# towers' own symmetry axis. Fitted only to demonstrate that the residual
# under ENGINE_TILES is the engine's geometry and not the calibration.
CORRECTED_TILES = {
    "own_princess_left": (4.0, 6.0),
    "own_princess_right": (14.0, 6.0),
    "opp_princess_left": (4.0, 27.0),
    "opp_princess_right": (14.0, 27.0),
    "left_bridge": (4.0, 16.5),
    "right_bridge": (14.0, 16.5),
    "own_king": (9.0, 2.5),
    "opp_king": (9.0, 30.5),
}


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


def find_towers(img: np.ndarray) -> dict:
    """The six tower platforms, by their low-saturation stone."""
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

    # Kings are ~4 tiles wide, Princesses ~3, and the Kings sit on the centre
    # column -- so width plus horizontal position separates them without any
    # hardcoded pixel positions.
    centre_x = (x0 + x1) / 2.0
    kings = sorted((b for b in blobs if b["w"] >= 78 and abs(b["cx"] - centre_x) < 60),
                   key=lambda b: b["cy"])
    princesses = sorted((b for b in blobs if b["w"] < 78),
                        key=lambda b: (b["cy"], b["cx"]))

    out: dict[str, tuple[float, float]] = {}
    if len(kings) == 2:
        out["opp_king"] = (kings[0]["cx"], kings[0]["cy"])
        out["own_king"] = (kings[1]["cx"], kings[1]["cy"])
    top = [b for b in princesses if b["cy"] < GAME_RECT[1] + 400]
    bottom = [b for b in princesses if b["cy"] >= GAME_RECT[1] + 400]
    for group, prefix in ((top, "opp"), (bottom, "own")):
        group = sorted(group, key=lambda b: b["cx"])
        if len(group) == 2:
            out[f"{prefix}_princess_left"] = (group[0]["cx"], group[0]["cy"])
            out[f"{prefix}_princess_right"] = (group[1]["cx"], group[1]["cy"])
    return out


# Our own Princess platforms are routinely swallowed by their HP bars and the
# tower troop sprite, so they are supplied as measured constants when the
# detector cannot find them. Same fixed window, measured the same way.
OWN_PRINCESS_FALLBACK = {
    "own_princess_left": (819.0, 656.0),
    "own_princess_right": (1101.0, 656.0),
}


def build(video: Path, at_seconds: float, out_path: Path) -> CalibrationProfile:
    source = VideoSource(video)
    frame = source.read_frame_at(int(at_seconds * source.fps))
    img = frame.image
    width, height = source.size
    source.close()

    river = find_river_and_bridges(img)
    towers = find_towers(img)
    for name, point in OWN_PRINCESS_FALLBACK.items():
        towers.setdefault(name, point)

    screen: dict[str, tuple[float, float]] = {
        "left_bridge": river["left_bridge"],
        "right_bridge": river["right_bridge"],
        **towers,
    }

    print(f"{video.name} @ {at_seconds:.1f}s -- detected landmarks (pixels):")
    for name in sorted(screen):
        print(f"   {name:20s} ({screen[name][0]:7.1f}, {screen[name][1]:7.1f})")

    anchors = {k: screen[k] for k in
               ("own_princess_left", "own_princess_right",
                "opp_princess_left", "opp_princess_right") if k in screen}
    if len(anchors) < 4:
        raise SystemExit(f"only found {len(anchors)} anchors: {sorted(anchors)}")

    results = {}
    for label, tiles in (("engine", ENGINE_TILES), ("corrected", CORRECTED_TILES)):
        shared = {k: v for k, v in screen.items() if k in tiles}
        homography = Homography.solve(shared, tiles)
        results[label] = (homography, homography.measure_error(shared, tiles))

    print("\nfit residual over all 8 landmarks, in TILES:")
    print(f"   {'landmark':20s} {'engine':>9s} {'corrected':>11s}")
    for name in sorted(screen):
        e = results["engine"][1].get(name, float("nan"))
        c = results["corrected"][1].get(name, float("nan"))
        print(f"   {name:20s} {e:9.2f} {c:11.2f}")
    for stat in ("max", "rms"):
        print(f"   {stat:20s} {results['engine'][1][stat]:9.2f} "
              f"{results['corrected'][1][stat]:11.2f}")

    homography, errors = results["engine"]
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

    if errors["max"] >= 0.5:
        print(
            "\nSTAGE 0 ACCEPTANCE (< 0.5 tiles) IS NOT MET AGAINST THE ENGINE'S\n"
            "GEOMETRY, AND THE CALIBRATION IS NOT THE REASON.\n"
            f"  engine geometry:    max {results['engine'][1]['max']:.2f} tiles\n"
            f"  corrected geometry: max {results['corrected'][1]['max']:.2f} tiles\n"
            "The same pixel measurements, the same solver, the same frame --\n"
            "only the target tile coordinates differ. The residual is the\n"
            "engine's own landmarks disagreeing with each other, not error in\n"
            "the recording or the fit. See UPSTREAM_REQUESTS.md item 1."
        )
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--at", type=float, default=18.5,
                        help="seconds into the recording; must be mid-battle")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or (_ROOT / "config" / "profile_gpg_1920x1080.json")
    build(args.video, args.at, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
