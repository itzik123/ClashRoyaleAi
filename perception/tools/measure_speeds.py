"""Walking speed per card, measured off the video, against the engine's own.

WHY THIS IS THE STRONGEST MEASUREMENT AVAILABLE HERE
-----------------------------------------------------
Reconstructing a whole match needs the placements, and the placements are the
weak link: the detector acquires a unit only once its deploy animation is over
and it is already walking, so most placement TIMES and POSITIONS are estimates.
Speed needs none of that. A track is a sequence of positions and times, and the
ratio is a measurement whether or not anyone knows where the unit came from.

It is also the quantity most recently changed in the engine (the speed-tier
commit), so it is the one where a discrepancy is most actionable.

MEASURING THE WALK, NOT THE AVERAGE
-----------------------------------
Total-distance-over-total-time is the wrong statistic and would understate
every card: a troop spends much of its life standing still -- deploying,
attacking, blocked behind another body, held at the bridge. Those are real
seconds during which speed is zero and the card's WALKING speed is unchanged.

So: per-segment speeds, then the median of the segments that are actually
moving. The moving threshold is a fraction of the track's own fastest segment
rather than an absolute, because a Giant and a Minion do not share a floor.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arena_anchor import default_map            # noqa: E402
from track_appearances import build_tracks      # noqa: E402

# Detector centre jitter is ~0.1-0.2 tiles per frame; at 2 fps that is up to
# 0.4 tiles/s of pure noise. A segment must beat this fraction of the track's
# own peak to count as walking.
MOVING_FRACTION = 0.35
MIN_SEGMENTS = 4


def track_speed(track):
    segs = []
    for (t0, x0, y0, _), (t1, x1, y1, _) in zip(track.pts, track.pts[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        segs.append(math.hypot(x1 - x0, y1 - y0) / dt)
    if len(segs) < MIN_SEGMENTS:
        return None
    peak = max(segs)
    moving = [s for s in segs if s >= peak * MOVING_FRACTION]
    if len(moving) < 2:
        return None
    # Path length travelled, which is the reliability signal that matters.
    # The share of segments counted as "moving" is NOT: the threshold is a
    # fraction of the track's OWN peak, so a unit that never walked more than
    # two tiles still reports a healthy-looking 62% while the median it yields
    # is a shuffle rather than a walk. Measured Musketeer at 0.82 tiles/s that
    # way, against 1.23 and 1.28 for the Valkyrie and Archers that share its
    # speed tier -- an artefact that a relative threshold cannot see and an
    # absolute distance can.
    path = sum(math.hypot(b[1] - a[1], b[2] - a[2])
               for a, b in zip(track.pts, track.pts[1:]))
    return statistics.median(moving), len(moving), len(segs), path


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("detections")
    ap.add_argument("--engine", default=".vid/final.tsv",
                    help="id/name/tiles-per-tick/tiles-per-second dump")
    ap.add_argument("--out")
    a = ap.parse_args()

    data = json.loads(Path(a.detections).read_text())
    amap = default_map()
    for fr in data["frames"]:
        for d in fr["units"]:
            d["x"], d["y"] = (round(v, 2) for v in amap.to_tile(d["px"], d["py"]))

    engine = {}
    for line in Path(a.engine).read_text().splitlines():
        f = line.split("\t")
        if len(f) >= 4:
            engine[f[1].lower().replace(" ", "").replace(".", "")] = float(f[3])

    by_card: dict[str, list] = {}
    for tr in build_tracks(data["frames"]):
        r = track_speed(tr)
        if r:
            by_card.setdefault(tr.name, []).append(r)

    ENGINE_NAME = {"archer": "archers", "minion": "minions", "minipekka": "minipekka",
                   "musketeer": "musketeer", "valkyrie": "valkyrie", "giant": "giant",
                   "knight": "knight", "goblin": "goblins", "skeleton": "skeletons",
                   "hog": "hogrider", "bat": "bats", "mega_minion": "megaminion",
                   "wizard": "wizard", "bomber": "bomber", "prince": "prince"}

    rows = []
    for name, obs in sorted(by_card.items(), key=lambda kv: -len(kv[1])):
        speeds = [o[0] for o in obs]
        eng = engine.get(ENGINE_NAME.get(name, name.replace("_", "")))
        rows.append(dict(card=name, n=len(speeds),
                         measured=round(statistics.median(speeds), 3),
                         spread=round(statistics.pstdev(speeds), 3) if len(speeds) > 1 else None,
                         engine=eng,
                         ratio=round(eng / statistics.median(speeds), 2) if eng else None))

    print(f"{'card':16s} {'n':>3s} {'video t/s':>10s} {'sd':>6s} {'engine t/s':>11s} {'engine/video':>13s}")
    for r in rows:
        print(f"{r['card']:16s} {r['n']:3d} {r['measured']:10.2f} "
              f"{(r['spread'] if r['spread'] is not None else float('nan')):6.2f} "
              f"{(r['engine'] if r['engine'] else float('nan')):11.2f} "
              f"{(r['ratio'] if r['ratio'] else float('nan')):13.2f}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
