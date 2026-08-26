# side_by_side — the real match beside the engine replaying it

Reads a recording, extracts the placements, replays them through the engine,
and renders both boards on one clock so a difference on screen is a difference
in the physics.

**Both panels are drawn in engine tile coordinates over the same 18×34 rect**,
so the same tile is the same pixel in each. That is the whole design: the
projection is shared, so it cannot be the explanation for anything you see.

---

## The pipeline

```bash
# 1. Detect units, 2 fps over the whole match. Slow -- ~2 s/frame on a 4-core
#    box, so ~15 min for a 3-minute match. Run it in the background.
python perception/tools/detect_video.py "perception/videos/<match>.mp4" \
       --out det.json --fps 2

# 2. What the tracker made of it. Read this before trusting anything below --
#    it prints how many tracks were dropped and why.
python perception/tools/track_appearances.py det.json --match-start 5.5

# 3. Walking speed per card, against the engine's own. Needs no placement
#    reconstruction, which is why it is the most trustworthy number here.
python perception/tools/measure_speeds.py det.json

# 4. The arena panel: crop to the playfield rect, downscale, embed.
ffmpeg -i "perception/videos/<match>.mp4" \
       -vf "crop=467:642:727:137,scale=280:-2,fps=10" \
       -c:v libx264 -crf 30 -an out/arena.mp4

# 5. Assemble and render.
python perception/tools/build_side_by_side.py --detections det.json \
       --video out/arena.mp4 --out out/payload.json --match-start 5.5
python perception/tools/side_by_side/render.py \
       perception/tools/side_by_side/viewer.html out/payload.json out/bench.html
```

`--match-start` is the video time of engine tick 0, read off the match clock:
the recording opens on the countdown, so it is **not** zero. Sample the clock
ROI at a few times and subtract — 2:45 left at t = 20 s puts tick 0 at t = 5.

## The C++ side

`simdrive.cpp` takes placements as TSV and dumps every entity every tick.
Deliberately TSV in and out: it has to build under bare `g++` with no JSON
dependency, and the Python side owns the schema.

```bash
# On the box with WSL and no MSVC (see CLAUDE.md's toolchain box):
wsl g++ -std=c++20 -O2 -Iinclude/core -Iinclude/entities -Iinclude/rendering \
    perception/tools/side_by_side/simdrive.cpp -o out/simdrive
```

It is an **ELF** binary, so `build_side_by_side.py` invokes it through `wsl`.
Point it elsewhere with `--simdrive` if you built a native one.

`dump_speeds.cpp` produces the per-card speed table `measure_speeds.py`
compares against, one row per registered troop:

```bash
wsl g++ -std=c++20 -O2 -Iinclude/core -Iinclude/entities -Iinclude/rendering \
    perception/tools/side_by_side/dump_speeds.cpp -o out/dump_speeds
wsl ./out/dump_speeds | sort -n > out/engine_speeds.tsv
```

---

## What this can and cannot say

The engine is handed the placements read off the video and **nothing else** —
it then runs its own targeting, pathing, damage and deaths. So the panels agree
only insofar as the engine reproduces the real game, and they are expected to
drift.

Three limits are structural, not bugs to fix here:

- **Tower HP is never injected.** Any early difference compounds for the rest
  of the match.
- **A missed placement is a missing unit, never a spurious one.** The engine's
  board can only be *emptier*, which fixes the sign of the unit-count gap.
- **The detector acquires a unit late.** It sees the unit already walking, not
  the deploy animation, so placement times and positions are estimates.
  `track_appearances.back_project` runs the track backwards along its own
  velocity to recover them, capped at 2.5 s.

**Read a push, not a match.** These errors accumulate, so the honest unit of
comparison is a few seconds around one placement.

## Coordinates: why not the calibration profile

`config/profile_gpg_1920x1080.json`'s homography was fit against the arena
constants as they stood on 2026-07-30 and never re-fit after the 2026-08-21
re-centring, so it still returns x = 3.94 for a Princess Tower the engine puts
at 3.00. `arena_anchor.py` anchors on the two Princess Towers instead and reads
the scale off their separation. See `UPSTREAM_REQUESTS.md` item 28.

## What it found, first run

`UPSTREAM_REQUESTS.md` items 26–28. In short: the engine's bridges sit half a
tile too far out (item 26, an engine proposal); the Slow and Medium speed tiers
are confirmed against the video while Fast is not (item 27); and the side
detector disagrees with direction of travel on a third of tracks (item 28).
