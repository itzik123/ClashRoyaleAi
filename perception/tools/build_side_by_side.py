"""Assemble the real-vs-simulated comparison payload.

    video -> detections -> tracks -> placements -> engine replay -> one JSON

The two panels share ONE geometry: both are drawn in engine tile coordinates
over the same 18x34 rect, so a unit at the same tile lands on the same pixel
in both. That is the whole point -- a difference visible on screen is a
difference in the physics, not in the projection.

WHAT THE COMPARISON CAN AND CANNOT SAY
--------------------------------------
The engine is fed the placements read off the video and nothing else. It then
runs its OWN dynamics: targeting, pathing, damage, deaths. So the two panels
agree only insofar as the engine reproduces the real game, and they are
expected to diverge with time since the run.

They cannot agree on everything even in principle:

  * Tower HP is not injected, so a divergence early in the match compounds.
  * The opponent's placements come from the same detector as ours and inherit
    its identity error (perception/README.md measures our own hand identity at
    33.8% against the elixir ledger, and the opponent's side is harder).
  * A card the detector misses is a unit the engine never gets. Missing
    placements make the simulated board EMPTIER, never fuller, so unit-count
    divergence has a known sign and should be read with that in mind.

The honest unit of comparison is therefore an individual push over a few
seconds, not the match outcome.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arena_anchor import default_map  # noqa: E402
from track_appearances import (  # noqa: E402
    build_tracks, correct_sides, side_from_travel, stitch, tracks_to_placements,
)
from measure_speeds import track_speed  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TICKS_PER_SECOND = 10          # CLAUDE.md: derived from 132 attackCooldown rows

# The recordings' deck (CLAUDE.md, "Recording real matches") -> engine card ids
# (verified against CardRegistry by tools/sbs/cardids). Only these eight can
# legitimately appear; anything else the detector reports is a misclassification
# and is counted rather than injected.
DECK = {
    "valkyrie": 10, "archer": 1, "minion": 41, "cannon": 25,
    "fireball": 7, "giant": 2, "musketeer": 6, "minipekka": 5,
}
# Classes the detector confuses with a deck card often enough to be worth
# folding in rather than discarding. Each one is a visual near-neighbour.
ALIASES = {
    "pekka": "minipekka",        # same silhouette, different scale
    "archer_queen": "archer",
    "minion_horde": "minion",
    "bomber": "archer",          # both small, both ranged
}


def to_engine_placements(places, side_of_team):
    """Placement dicts -> (tick, cardId, x, y, team) rows the driver reads."""
    rows, skipped = [], {}
    for p in places:
        name = ALIASES.get(p["name"], p["name"])
        card = DECK.get(name)
        if card is None:
            skipped[p["name"]] = skipped.get(p["name"], 0) + 1
            continue
        rows.append(dict(tick=int(round(p["t"] * TICKS_PER_SECOND)), card=card,
                         x=p["x"], y=p["y"], team=side_of_team[p["side"]],
                         name=name, t=p["t"], conf=p["conf"], bodies=p["bodies"]))
    return rows, skipped


DEFAULT_SIMDRIVE = REPO / ".vid" / "sbs" / "simdrive"


def run_engine(rows, max_ticks, workdir: Path, exe: Path):
    workdir.mkdir(parents=True, exist_ok=True)
    tsv = workdir / "places.tsv"
    tsv.write_text("".join(
        f"{r['tick']}\t{r['card']}\t{r['x']:.3f}\t{r['y']:.3f}\t{r['team']}\n"
        for r in sorted(rows, key=lambda r: r["tick"])))
    if not exe.exists():
        raise SystemExit(
            f"{exe} is not built. Build it, then re-run "
            f"(perception/tools/side_by_side/README.md has the full command):\n"
            f"  wsl g++ -std=c++20 -O2 -Iinclude/core -Iinclude/entities "
            f"-Iinclude/rendering perception/tools/side_by_side/simdrive.cpp "
            f"-o {exe}")
    # Built with `wsl g++` on the box that has no MSVC, so it is an ELF binary
    # and has to be run through wsl too. See CLAUDE.md's toolchain box.
    out = subprocess.run(["wsl", f"./{exe.relative_to(REPO).as_posix()}",
                          tsv.relative_to(REPO).as_posix(), str(max_ticks)],
                         cwd=REPO, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"simdrive failed: {out.stderr[:2000]}")
    ticks: dict[int, list] = {}
    for line in out.stdout.splitlines():
        f = line.split("\t")
        if f[0] != "T":
            continue
        t, eid, card, team, x, y, hp = (int(f[1]), int(f[2]), int(f[3]),
                                        int(f[4]), float(f[5]), float(f[6]), int(f[7]))
        name = f[8] if len(f) > 8 else ""
        # Projectiles and spell effects are entities too and carry no name.
        # They are not units on the board and counting them would inflate the
        # engine's side of the unit-count comparison.
        if not name:
            continue
        ticks.setdefault(t, []).append(
            dict(id=eid, card=card, team=team, x=round(x, 2), y=round(y, 2),
                 hp=hp, name=name))
    return ticks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", required=True)
    ap.add_argument("--video", required=True, help="the cropped arena mp4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sim-fps", type=float, default=5.0,
                    help="how often to keep an engine frame in the payload")
    ap.add_argument("--match-start", type=float, default=5.5,
                    help="video time of engine tick 0, read off the match clock")
    ap.add_argument("--engine-speeds", default=".vid/final.tsv",
                    help="per-card speed dump; build it with "
                         "side_by_side/dump_speeds.cpp")
    ap.add_argument("--simdrive", default=str(DEFAULT_SIMDRIVE),
                    help="the compiled simdrive binary")
    a = ap.parse_args()

    data = json.loads(Path(a.detections).read_text())
    amap = default_map()
    for fr in data["frames"]:
        for d in fr["units"]:
            d["x"], d["y"] = (round(v, 2) for v in amap.to_tile(d["px"], d["py"]))

    raw = build_tracks(data["frames"])
    # Agreement is measured on the RAW tracks, before any correction, or the
    # number would be reporting the correction back to itself.
    agree = contra = 0
    for tr in raw:
        implied = side_from_travel(tr)
        if implied is None or tr.t0 < a.match_start:
            continue
        if implied == tr.side:
            agree += 1
        else:
            contra += 1
    n_side = agree + contra

    # [walking speed, moving segments, total segments, tiles travelled]. The
    # last one is the reliability signal: a speed measured over a unit that
    # crossed the board is a different quality of estimate from one taken off a
    # unit that shuffled two tiles and stood still.
    speeds = {}
    for tr in raw:
        r = track_speed(tr)
        if r:
            speeds.setdefault(tr.name, []).append(
                [round(r[0], 3), r[1], r[2], round(r[3], 2)])

    corrected = correct_sides(raw)
    tracks = stitch(raw)
    places = tracks_to_placements(tracks, min_t=a.match_start)

    # "ally" is whoever is at the bottom of the screen: the recording's owner.
    # The engine calls that team 0.
    rows, skipped = to_engine_placements(places, {"ally": 0, "enemy": 1})
    # Engine tick 0 is the START OF THE MATCH, not the start of the video: the
    # recording opens on the countdown. Shifting here rather than in the driver
    # keeps the video timeline the master clock for the viewer.
    for r in rows:
        r["tick"] = max(0, int(round((r["t"] - a.match_start) * TICKS_PER_SECOND)))
    duration = max((f["t"] for f in data["frames"]), default=0.0)
    max_ticks = int((duration - a.match_start) * TICKS_PER_SECOND) + 20
    ticks = run_engine(rows, max_ticks, REPO / ".vid" / "sbs" / "run",
                       Path(a.simdrive))

    keep = max(1, int(round(TICKS_PER_SECOND / a.sim_fps)))
    sim = [dict(t=round(t / TICKS_PER_SECOND + a.match_start, 2), units=v)
           for t, v in sorted(ticks.items()) if t % keep == 0]

    payload = dict(
        meta=dict(video=data["video"], duration=duration,
                  detect_fps=data["fps"], sim_fps=a.sim_fps,
                  ticks_per_second=TICKS_PER_SECOND, match_start=a.match_start,
                  n_tracks=len(tracks), n_raw_tracks=len(raw),
                  n_placements=len(places),
                  n_injected=len(rows), skipped=skipped,
                  side_agree=agree, side_contra=contra, side_n=n_side,
                  side_corrected=corrected,
                  sim_end=(sim[-1]["t"] if sim else 0.0)),
        speeds=speeds,
        engine_speeds={f.split(chr(9))[1]: float(f.split(chr(9))[3])
                       for f in Path(a.engine_speeds).read_text().splitlines()
                       if len(f.split(chr(9))) >= 4},
        arena=dict(width=18, height=34, river=[15.5, 17.5],
                   engine=dict(bridges=[2.5, 14.5], princess_x=[3.0, 14.0],
                               king_x=8.5, princess_y=[6.0, 27.0],
                               king_y=[2.5, 30.5], bridge_y=16.5),
                   measured=dict(bridges=[3.12, 13.98], bridge_y=16.09,
                                 king_x=[8.33, 8.48], king_y=[1.97, 30.31])),
        placements=rows,
        real=[dict(t=f["t"], units=[u for u in f["units"] if u["conf"] >= 0.45])
              for f in data["frames"]],
        tracks=[dict(name=t.name, side=t.side, pts=[[round(p[0], 2), round(p[1], 2),
                                                     round(p[2], 2)] for p in t.pts])
                for t in tracks],
        sim=sim,
        video_b64=base64.b64encode(Path(a.video).read_bytes()).decode(),
    )
    Path(a.out).write_text(json.dumps(payload))
    print(f"raw tracks {len(raw)} -> stitched {len(tracks)}  "
          f"placements {len(places)}  injected {len(rows)}")
    if n_side:
        print(f"side detector vs direction of travel: {agree}/{n_side} agree "
              f"({agree/n_side:.1%}), {corrected} tracks corrected")
    if skipped:
        print("not in the deck, skipped:", skipped)
    print(f"engine ticks {len(ticks)} -> {len(sim)} kept   "
          f"payload {Path(a.out).stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
