"""Detections -> tracks -> placement events.

The unit detector fires per frame with no identity across frames. A PLACEMENT
is the moment a track BEGINS, so the whole job is association: link each
frame's detections to the tracks already running, and whatever fails to link
is new.

THREE THINGS THIS HAS TO GET RIGHT, AND WHY EACH ONE IS HERE
------------------------------------------------------------
1. TOWERS ARE DETECTED AS UNITS. The model classifies a Princess Tower as
   `princess` and a King Tower as whatever is nearest in its vocabulary, every
   frame, forever. Left in, they are six permanent tracks that never move and
   never die -- harmless for the tracker and fatal for the placement list,
   since each would read as a placement at t=0.

   Rejected as a TRACK PROPERTY -- near a tower AND never moved AND still
   there many seconds later -- not by "is this detection near a tower". The
   position-only rule was tried first and it discards real cards: a measured
   Mini PEKKA placed at (3.1, 6.8), one tile behind our own left Princess
   Tower, is a completely ordinary defensive placement and the 2-tile
   exclusion disc deleted it. What separates a tower from a defender is not
   where it is, it is that the tower is still standing there unmoved a match
   later.

2. ONE CARD IS OFTEN SEVERAL BODIES. Minions spawn 3, Archers 2. Three tracks
   beginning within a second of each other, within a tile of each other, on
   the same side, are ONE placement -- and the engine will spawn the other
   bodies itself from the card id, so emitting three would triple the push.

3. A DETECTION GAP IS NOT A DEATH. The detector misses a unit for a frame or
   two constantly (a unit behind a tower, mid-attack animation, low contrast).
   A track is only retired after MAX_GAP seconds, otherwise every miss
   manufactures a fresh "placement" for a unit that was already walking --
   which is the failure mode that would make the engine look wrong when the
   tracker is.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arena_anchor import default_map  # noqa: E402

# A track links to a detection within this many tiles. A Hog covers ~1 tile/s
# and the sample step is 0.5 s, so ~0.5 tiles of real motion; 2.5 leaves room
# for the detector's own centre jitter without linking across lanes.
MAX_LINK_TILES = 2.5
MAX_GAP_S = 1.5
# THE CONFIDENCE GATE IS ON THE TRACK, NOT ON THE DETECTION, and that is the
# difference between finding a placement and finding a unit already walking.
# A card's first second on screen is its deploy animation -- translucent, half
# drawn, partly under the deploy ring -- and it scores far below a settled
# unit. Gating each detection at 0.45 dropped exactly those frames and moved
# one measured Mini PEKKA placement from (4.1, 6.8) at t=61 to (6.2, 11.0) at
# t=65: four seconds late and five tiles up the board, i.e. reported as a
# placement in the middle of our own half when it was really a defence at the
# back. Associate loosely, then keep the track only if it EVER became
# confident.
LINK_CONF = 0.28         # loose enough to catch the deploy frames
TRACK_CONF = 0.55        # ...but the track must earn this at least once
MIN_TRACK_FRAMES = 2     # a one-frame blip is noise, not a unit
TOWER_RADIUS_TILES = 1.6      # a tower's own footprint, not a keep-out zone
TOWER_STATIONARY_TILES = 0.9  # detector centre jitter, measured
SQUAD_WINDOW_S = 1.2
SQUAD_RADIUS_TILES = 3.0

# A PLACEMENT HAPPENS IN THE PLACER'S OWN HALF. That is a rule of the game, not
# a heuristic, and it is the single most useful thing available for telling a
# real placement from a track the detector simply re-acquired after losing a
# unit mid-board. The river band is [15.5, 17.5), so team 0 deploys at y <= 15
# and team 1 at y >= 18.
#
# It removes two different faults at once:
#   * A unit the detector drops and re-finds starts a fresh track wherever it
#     had walked to. An enemy Valkyrie re-acquired at y = 10.1 would otherwise
#     be injected as a SECOND Valkyrie, deep in our half, five seconds after
#     the real one was played.
#   * A side-detector flip becomes harmless instead of catastrophic. A Cannon
#     at (8.1, 9.3) labelled `enemy` cannot be an enemy placement -- nothing
#     can be placed there by that side -- so it is dropped rather than injected
#     onto the wrong team, which would have handed the opponent a free building
#     in our own half.
#
# The known exception is deliberate: once a Princess Tower falls, its side may
# deploy in the enemy half near it. Those placements are lost here, and the
# count of what each rule removed is reported so the loss is visible.
OWN_HALF_MAX_Y = 15.0
OPP_HALF_MIN_Y = 18.0

# STITCHING. The detector loses a unit and re-finds it constantly, and every
# re-acquisition starts a fresh track wherever the unit had walked to. Measured
# on one match: 72 tracks, of which 41 began mid-board. Left unstitched those
# 41 are not merely noise -- they are the reason the placement list looked
# nearly empty, because the own-half rule correctly refuses to call any of them
# a placement, and the real placement they belong to was never seen.
#
# Two tracks are the same unit if they share a class and a side, the gap is
# short, and the distance is one the unit could actually have covered. The
# speed bound is generous (3 tiles/s beats every card in the game) because the
# cost of failing to stitch is a phantom placement, while the cost of
# over-stitching is one merged track -- and the class+side match already makes
# a wrong merge unlikely.
STITCH_MAX_GAP_S = 4.0
STITCH_SPEED_TILES_S = 3.0
STITCH_SLACK_TILES = 1.5

TOWER_TILES = [(3.0, 6.0), (14.0, 6.0), (8.5, 2.5),
               (3.0, 27.0), (14.0, 27.0), (8.5, 30.5)]


@dataclass
class Track:
    name: str
    side: str
    pts: list = field(default_factory=list)      # (t, x, y, conf)

    @property
    def t0(self): return self.pts[0][0]
    @property
    def t1(self): return self.pts[-1][0]
    @property
    def start(self): return self.pts[0][1], self.pts[0][2]

    def last(self): return self.pts[-1][1], self.pts[-1][2]


def _near_a_tower(x, y):
    return any(math.hypot(x - tx, y - ty) < TOWER_RADIUS_TILES
               for tx, ty in TOWER_TILES)


def _is_a_tower(tr: "Track") -> bool:
    """Near a tower's centre AND never moved.

    The lifetime test this used to also require was WRONG and the measurement
    says so: the six towers do not come back as six long tracks, they come
    back as dozens of short ones, because the detector loses and re-acquires
    them constantly. Over one match that let 9 fragments through -- `princess`
    at (2.9, 27.9) and (14.0, 27.9), and a phantom `minipekka` parked on our
    own left Princess Tower at (3.1, 6.7) that re-appeared six separate times.
    Every one of them would have been injected as a real card.

    Standing still on the tower's own tile is the whole signature; how long
    the detector managed to hold onto it is a property of the detector.

    The cost is stated rather than hidden: a BUILDING placed within 1.6 tiles
    of a tower centre is also stationary and is also dropped. That is a real
    placement lost, and it is the right trade here -- towers are present in
    every frame of every match, a building tucked that close is rare, and the
    alternative admits six permanent phantoms.
    """
    x0, y0 = tr.start
    if not _near_a_tower(x0, y0):
        return False
    return max(math.hypot(p[1] - x0, p[2] - y0) for p in tr.pts) < TOWER_STATIONARY_TILES


# The engine board, plus half a tile of slack for detector jitter at the edge.
# Anything outside is not on the arena at all: the battle UI's two player
# avatars sit above the board and the detector finds a unit in each, every
# frame (perception/live/board_filter.py measured 31% of all detections landing
# off-board for exactly this reason). One of them held a 42-frame `archer`
# track at y = -1.5 in this recording.
# y is tightened to the REAL arena's 32 rows rather than the engine's 34: the
# engine adds one row behind each King that the real board does not have, so
# anything the detector reports outside [0, 33] is off the arena by
# construction. It matters -- the bottom-centre player avatar held a 41-frame
# `musketeer` track at y = -0.4, which is 2.9 tiles from the King and so sailed
# past the tower test.
BOARD_X = (-1.0, 18.0)
BOARD_Y = (0.0, 33.0)


def _on_board(x, y):
    return BOARD_X[0] <= x <= BOARD_X[1] and BOARD_Y[0] <= y <= BOARD_Y[1]


def build_tracks(frames):
    live: list[Track] = []
    done: list[Track] = []
    for fr in frames:
        t = fr["t"]
        dets = [d for d in fr["units"]
                if d["conf"] >= LINK_CONF and _on_board(d["x"], d["y"])]
        for tr in list(live):
            if t - tr.t1 > MAX_GAP_S:
                live.remove(tr); done.append(tr)
        # Greedy nearest-first association, same-name preferred. Greedy rather
        # than Hungarian on purpose: the cost matrix is tiny and a wrong link
        # here costs one track, while an unexplainable global assignment costs
        # the ability to read the output at all.
        pairs = sorted(
            ((math.hypot(d["x"] - tr.last()[0], d["y"] - tr.last()[1])
              + (0.0 if d["name"] == tr.name else 1.0), di, ti)
             for di, d in enumerate(dets) for ti, tr in enumerate(live)))
        used_d, used_t = set(), set()
        for cost, di, ti in pairs:
            if cost > MAX_LINK_TILES or di in used_d or ti in used_t:
                continue
            d = dets[di]
            live[ti].pts.append((t, d["x"], d["y"], d["conf"]))
            used_d.add(di); used_t.add(ti)
        for di, d in enumerate(dets):
            if di in used_d:
                continue
            live.append(Track(d["name"], d["side"], [(t, d["x"], d["y"], d["conf"])]))
    done.extend(live)
    return [tr for tr in done
            if len(tr.pts) >= MIN_TRACK_FRAMES
            and max(p[3] for p in tr.pts) >= TRACK_CONF
            and not _is_a_tower(tr)]


# BACK-PROJECTION. The detector does not see a placement, it sees the unit a
# second or two later, already walking. Measured on this recording, the effect
# is not subtle: every enemy Giant in the match was first acquired at
# y = 17.1, 13.5, 17.0 -- inside the river band or past it -- when a Giant
# played at the bridge is placed at y >= 18. Judged on the raw first sighting,
# not one of them is a placement, and the match reconstructs with almost no
# opponent in it.
#
# The track's own first seconds give the velocity, so the fix is to run it
# backwards to where it entered its owner's half. This is an ESTIMATE and is
# labelled as one; the cap keeps it from turning a unit acquired deep in the
# wrong half into a fictional placement, which is the failure that would matter.
BACKPROJECT_MAX_S = 2.5
BACKPROJECT_FIT_POINTS = 3

# DIRECTION OF TRAVEL IS A FREE, INDEPENDENT SIDE LABEL, and it is a better one
# than the side detector. A unit walks toward the OPPONENT's towers, so the sign
# of its vertical travel names its owner without looking at a single pixel of
# health-bar colour.
#
# Measured on this recording, over unstitched tracks (so no merge can be
# blamed) with at least 1.5 tiles of travel in their first 5 seconds:
#
#     30 usable tracks -- 19 agree (63.3%), 11 CONTRADICT (36.7%)
#
# A third of the tracks carry a side that their own motion refutes, including
# unmissable ones: a `minipekka` labelled ally that walked 8.4 tiles DOWN the
# board in five seconds, and a `musketeer` labelled enemy that walked up.
#
# For a state estimator this is worse than the card-identity error already on
# record (perception/README.md, 33.8%): a unit with the wrong NAME is a unit
# with wrong stats, while a unit on the wrong TEAM is a unit fighting for the
# other player -- it inverts the outcome of the fight it is in.
#
# So travel wins where travel is unambiguous, and the count of corrections is
# reported rather than applied silently. The threshold is deliberately well
# above the ~1 tile a retarget can produce, since a unit CAN legitimately walk
# backwards for a few tiles when it turns on a building behind it.
SIDE_FROM_TRAVEL_TILES = 1.5
SIDE_FROM_TRAVEL_WINDOW_S = 5.0


def _entry_velocity(tr: "Track"):
    pts = tr.pts[:BACKPROJECT_FIT_POINTS]
    if len(pts) < 2:
        return None
    dt = pts[-1][0] - pts[0][0]
    if dt <= 0:
        return None
    return (pts[-1][1] - pts[0][1]) / dt, (pts[-1][2] - pts[0][2]) / dt


def back_project(tr: "Track"):
    """(t, x, y) where this unit plausibly entered its own half, or None."""
    t0, x0, y0, _ = tr.pts[0]
    edge = OWN_HALF_MAX_Y if tr.side == "ally" else OPP_HALF_MIN_Y
    inside = y0 <= edge if tr.side == "ally" else y0 >= edge
    if inside:
        return t0, x0, y0, 0.0
    v = _entry_velocity(tr)
    if v is None:
        return None
    vx, vy = v
    # It has to be moving the right way: an ally walks toward higher y, so
    # running it backwards means going DOWN toward its own half.
    if (tr.side == "ally" and vy <= 0.05) or (tr.side == "enemy" and vy >= -0.05):
        return None
    back = (y0 - edge) / vy
    if not 0 < back <= BACKPROJECT_MAX_S:
        return None
    return t0 - back, x0 - vx * back, edge, back


def is_a_placement(tr: "Track") -> bool:
    return back_project(tr) is not None


def side_from_travel(tr: "Track"):
    """The side this track's own motion implies, or None if it does not say."""
    seg = [p for p in tr.pts if p[0] - tr.t0 <= SIDE_FROM_TRAVEL_WINDOW_S]
    if len(seg) < 2:
        return None
    dy = seg[-1][2] - seg[0][2]
    if abs(dy) < SIDE_FROM_TRAVEL_TILES:
        return None
    return "ally" if dy > 0 else "enemy"


def correct_sides(tracks):
    """Overrule the side detector wherever travel disagrees. Returns the count."""
    fixed = 0
    for tr in tracks:
        implied = side_from_travel(tr)
        if implied is not None and implied != tr.side:
            tr.side = implied
            fixed += 1
    return fixed


def stitch(tracks):
    """Join tracks that are the same unit seen across a detection gap."""
    order = sorted(tracks, key=lambda t: t.t0)
    merged_into: dict[int, Track] = {}
    out: list[Track] = []
    for tr in order:
        best, best_cost = None, None
        for cand in out:
            if cand.name != tr.name or cand.side != tr.side:
                continue
            gap = tr.t0 - cand.t1
            if not 0 < gap <= STITCH_MAX_GAP_S:
                continue
            cx, cy = cand.last()
            d = math.hypot(tr.start[0] - cx, tr.start[1] - cy)
            if d > STITCH_SPEED_TILES_S * gap + STITCH_SLACK_TILES:
                continue
            cost = d + gap
            if best_cost is None or cost < best_cost:
                best, best_cost = cand, cost
        if best is None:
            out.append(tr)
        else:
            best.pts.extend(tr.pts)
            best.pts.sort(key=lambda q: q[0])
            merged_into[id(tr)] = best
    return out


def tracks_to_placements(tracks, min_t=0.0):
    """Collapse co-spawning bodies into one placement per card."""
    out = []
    for tr in sorted(tracks, key=lambda t: t.t0):
        bp = back_project(tr)
        if bp is None or bp[0] < min_t:
            continue
        t_place, x, y, back = bp
        for p in out:
            if (p["side"] == tr.side and p["name"] == tr.name
                    and abs(t_place - p["t"]) <= SQUAD_WINDOW_S
                    and math.hypot(x - p["x"], y - p["y"]) <= SQUAD_RADIUS_TILES):
                p["bodies"] += 1
                break
        else:
            out.append(dict(t=round(t_place, 2), name=tr.name, side=tr.side,
                            x=round(x, 2), y=round(y, 2), bodies=1,
                            conf=round(max(q[3] for q in tr.pts), 3),
                            frames=len(tr.pts), back=round(back, 2),
                            seen_t=round(tr.t0, 2),
                            last_t=round(tr.t1, 2)))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("detections")
    ap.add_argument("--out")
    ap.add_argument("--match-start", type=float, default=0.0,
                    help="video time of tick 0; earlier tracks are the intro")
    a = ap.parse_args()
    data = json.loads(open(a.detections).read())
    # The detector wrote tile coordinates through the calibration profile,
    # whose x frame is stale by ~1 tile (see arena_anchor's docstring). Re-map
    # from the pixels it also recorded, so everything downstream is in the
    # engine's own frame and no comparison is silently shifted.
    amap = default_map()
    for fr in data["frames"]:
        for d in fr["units"]:
            d["x"], d["y"] = (round(v, 2) for v in amap.to_tile(d["px"], d["py"]))
    raw = build_tracks(data["frames"])
    # Correct BEFORE stitching: stitching requires a side match, so a track
    # left on the wrong side can never join the unit it belongs to.
    fixed = correct_sides(raw)
    tracks = stitch(raw)
    places = tracks_to_placements(tracks, min_t=a.match_start)
    pre = sum(1 for t in tracks if t.t0 < a.match_start)
    reacq = sum(1 for t in tracks
                if t.t0 >= a.match_start and not is_a_placement(t))
    print(f"{len(data['frames'])} frames -> {len(raw)} tracks -> {len(tracks)} stitched"
          f" -> {len(places)} placements"
          f"   (dropped: {pre} before the match, {reacq} still starting mid-board)")
    print(f"side corrected from direction of travel on {fixed} of {len(raw)} tracks")
    for p in places:
        print(f"  t={p['t']:6.1f}  {p['side']:5s} {p['name']:14s} "
              f"({p['x']:5.1f},{p['y']:5.1f})  bodies={p['bodies']} "
              f"frames={p['frames']} conf={p['conf']} "
              f"{'back %.1fs' % p['back'] if p['back'] else 'seen directly'}")
    if a.out:
        json.dump(dict(placements=places,
                       tracks=[dict(name=t.name, side=t.side, pts=t.pts) for t in tracks]),
                  open(a.out, "w"))


if __name__ == "__main__":
    main()
