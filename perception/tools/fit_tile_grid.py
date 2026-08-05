"""Fit TILE_INIT_X/Y and TILE_WIDTH/HEIGHT for one emulator, in DISPLAY space.

    perception/.venv-dml/Scripts/python.exe perception/tools/fit_tile_grid.py \
        frame.png --annotate out.png

Give it one 720x1280 `adb exec-out screencap` of an in-progress match. An
opening frame is best: the towers are undamaged and no unit is standing on a
landmark.

WHY THESE CONSTANTS NEEDED FITTING AT ALL
-----------------------------------------
Upstream CRBAB ships one set for every device. On this emulator they are ~10%
small in x and ~4% in y, which is a SCALE error, and a scale error does not
look like a bug -- it looks like flakiness. Taps landed correctly near the
river and drifted to over half a tile wrong by our own King, so how wrong a
placement was depended on where it was. That is also why the obvious suspect,
a constant offset like adapter.TILE_Y_OFFSET, could never have explained it.

MEASURED IN DISPLAY SPACE, NOT THROUGH THE DESKTOP CAPTURE
-----------------------------------------------------------
`adb screencap` returns the android framebuffer directly, so the emulator
window's position, size and 0.6% aspect mismatch never enter the arithmetic.
An earlier attempt routed through the desktop homography and disagreed with
this one by nearly a tile.

LANDMARKS
---------
    x   the two bridges, engine x = 4.0 and 14.0 (geometry.LEFT/RIGHT_BRIDGE),
        found as the interior gaps in the river's water mask.
    y   SCALE from the two princess HP bars. They are 27.0 - 6.0 = 21.0 tiles
        apart, and a bar sits at the same unknown offset above its own tower on
        both sides, so the separation is exact without ever locating a tower
        centre -- which is the part that needs grey-stone blob labelling in
        tools/calibrate.py.
        OFFSET from the river centre, engine y = 16.5.

Two checks the fit does not get to choose, both reported:
  * the bridge midpoint must be the display's horizontal centre, because engine
    x = 9.0 (the Kings) is the board's centre;
  * both HP bars must come out the same distance above their towers.
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

from geometry import (  # noqa: E402
    LEFT_BRIDGE,
    OPP_PRINCESS_LEFT,
    OWN_PRINCESS_LEFT,
    RIGHT_BRIDGE,
)

RIVER_ENGINE_Y = 16.5           # (RIVER_Y_START + RIVER_Y_END) / 2
WATER_HSV_LO = np.array([84, 110, 90])
WATER_HSV_HI = np.array([102, 255, 255])


def find_river_and_bridges(img: np.ndarray) -> dict:
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    water = cv2.inRange(hsv, WATER_HSV_LO, WATER_HSV_HI)
    # The card bar and various UI chrome are also blue; the river is the only
    # blue in the middle third.
    water[:int(h * 0.30)] = 0
    water[int(h * 0.60):] = 0

    rows = water.sum(axis=1) / 255
    ys = [y for y in range(h) if rows[y] > w * 0.25]
    if not ys:
        raise SystemExit("no river found -- is this frame mid-battle?")
    river_y = (ys[0] + ys[-1]) / 2.0

    band = water[ys[0]:ys[-1] + 1, :]
    cols = band.sum(axis=0) / 255
    thresh = band.shape[0] * 0.4
    wet = [x for x in range(w) if cols[x] >= thresh]
    lo, hi = wet[0], wet[-1]

    # Only BETWEEN the outermost water columns. Scanning the whole frame makes
    # the shore -- where there is simply no arena -- the widest "gap", and it
    # wins a bridge slot.
    runs, run = [], None
    for x in range(lo, hi + 1):
        if cols[x] < thresh:
            if run is None:
                run = x
        else:
            if run is not None and x - run >= 12:
                runs.append((run, x))
            run = None
    if run is not None and hi - run >= 12:
        runs.append((run, hi))
    if len(runs) < 2:
        raise SystemExit(f"expected 2 bridges in the river, found {len(runs)}")
    two = sorted(sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:2])
    return {"river_y": river_y,
            "water_span": (lo, hi),
            "left_bridge_x": (two[0][0] + two[0][1]) / 2.0,
            "right_bridge_x": (two[1][0] + two[1][1]) / 2.0}


def find_princess_bars(img: np.ndarray) -> dict:
    h = img.shape[0]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red = (cv2.inRange(hsv, np.array([0, 120, 120]), np.array([10, 255, 255]))
           | cv2.inRange(hsv, np.array([160, 120, 120]),
                         np.array([179, 255, 255])))
    blue = cv2.inRange(hsv, np.array([100, 140, 140]),
                       np.array([120, 255, 255]))

    def row(mask, lo, hi):
        sub = mask[lo:hi].sum(axis=1) / 255
        return lo + int(np.argmax(sub))

    return {"opp_bar_y": row(red, int(h * 0.10), int(h * 0.22)),
            "own_bar_y": row(blue, int(h * 0.58), int(h * 0.68))}


def fit(img: np.ndarray) -> dict:
    h, w = img.shape[:2]
    r = find_river_and_bridges(img)
    b = find_princess_bars(img)

    tile_w = (r["right_bridge_x"] - r["left_bridge_x"]) / (
        RIGHT_BRIDGE[0] - LEFT_BRIDGE[0])
    tile_h = (b["own_bar_y"] - b["opp_bar_y"]) / (
        OPP_PRINCESS_LEFT[1] - OWN_PRINCESS_LEFT[1])

    tile_init_x = r["left_bridge_x"] - LEFT_BRIDGE[0] * tile_w
    # android_y = A - engine_y * tile_h, and the actuator writes that as
    # (H - TILE_INIT_Y) - (engine_y - 1) * TILE_HEIGHT.
    a = r["river_y"] + RIVER_ENGINE_Y * tile_h
    tile_init_y = h - a + tile_h

    bridge_mid = (r["left_bridge_x"] + r["right_bridge_x"]) / 2.0
    own_bar_engine = (a - b["own_bar_y"]) / tile_h
    opp_bar_engine = (a - b["opp_bar_y"]) / tile_h
    return {
        **r, **b,
        "TILE_WIDTH": tile_w, "TILE_HEIGHT": tile_h,
        "TILE_INIT_X": tile_init_x, "TILE_INIT_Y": tile_init_y,
        "A": a,
        "bridge_mid": bridge_mid,
        "display_centre": w / 2.0,
        "own_bar_above_tower": own_bar_engine - OWN_PRINCESS_LEFT[1],
        "opp_bar_above_tower": opp_bar_engine - OPP_PRINCESS_LEFT[1],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frame", type=Path)
    ap.add_argument("--annotate", type=Path, default=None)
    args = ap.parse_args()

    img = cv2.imread(str(args.frame))
    if img is None:
        raise SystemExit(f"cannot read {args.frame}")
    h, w = img.shape[:2]
    print(f"frame {w}x{h}")
    f = fit(img)

    print(f"\nlandmarks")
    print(f"  river centre row      {f['river_y']:8.1f}")
    print(f"  water spans x         {f['water_span']}")
    print(f"  bridges x             {f['left_bridge_x']:.1f}  "
          f"{f['right_bridge_x']:.1f}")
    print(f"  princess bars y       {f['opp_bar_y']} (enemy)  "
          f"{f['own_bar_y']} (own)")

    print(f"\nfitted, DISPLAY space")
    for k, upstream in (("TILE_WIDTH", 34), ("TILE_HEIGHT", 27.6),
                        ("TILE_INIT_X", 52), ("TILE_INIT_Y", 296)):
        print(f"  {k:<12} {f[k]:9.3f}   (upstream {upstream})")

    print(f"\nchecks the fit did not choose")
    dx = f["bridge_mid"] - f["display_centre"]
    print(f"  bridge midpoint {f['bridge_mid']:.1f} vs display centre "
          f"{f['display_centre']:.1f}   off by {dx:+.1f} px "
          f"({dx / f['TILE_WIDTH']:+.3f} tiles)")
    print(f"  HP bar sits above its tower by "
          f"{f['own_bar_above_tower']:.2f} tiles (own) and "
          f"{f['opp_bar_above_tower']:.2f} (enemy)")

    if args.annotate:
        vis = img.copy()
        cv2.line(vis, (0, int(f["river_y"])), (w, int(f["river_y"])),
                 (0, 0, 255), 2)
        cv2.line(vis, (0, f["opp_bar_y"]), (w, f["opp_bar_y"]), (255, 0, 255), 2)
        cv2.line(vis, (0, f["own_bar_y"]), (w, f["own_bar_y"]), (255, 255, 0), 2)
        for ex in range(19):
            x = int(f["TILE_INIT_X"] + ex * f["TILE_WIDTH"])
            cv2.line(vis, (x, 60), (x, h - 240), (255, 255, 255), 1)
        for ey in range(1, 34):
            y = int(f["A"] - ey * f["TILE_HEIGHT"])
            cv2.line(vis, (int(f["TILE_INIT_X"]), y),
                     (int(f["TILE_INIT_X"] + 18 * f["TILE_WIDTH"]), y),
                     (255, 255, 255), 1)
        cv2.imwrite(str(args.annotate), vis)
        print(f"\nannotated -> {args.annotate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
