# perception/ -> simulator: open requests

The backlog was **cleared to empty on 2026-08-24** and its 33 items archived in
`DECISIONS.md` with their headings intact, so a citation like "UPSTREAM item 13"
still resolves there. Numbering continues from 25; everything below is live.

---

## 26. PROPOSED 2026-08-24 — the bridges are half a tile too far out; in the real game a bridge sits directly in front of its Princess Tower

`ArenaLayout.h` puts `LEFT_BRIDGE_X = 2.5` / `RIGHT_BRIDGE_X = 14.5` and
`LEFT_LANE_X = 3.0` / `RIGHT_LANE_X = 14.0`, so a lane runs 0.5 tiles inboard of its
own bridge. **Measured off the recordings, the real arena has no such offset.** The
bridge and the Princess Tower behind it share a column.

### The measurement

Two independent derivations, on `perception/videos/2026-07-29 20-45-03.mp4`
(1920x1080 desktop capture), both agreeing:

**(a) Direct pixels.** Classify each column across the river band by `r - b` on an
untinted in-battle frame (t = 30 s), which separates orange planking from grey-blue
water. Four runs come back: two arena side walls and two bridges.

| | left wall | left bridge | right bridge | right wall |
|---|---|---|---|---|
| desktop px | 702-727 | **784-851** | **1070-1135** | 1192-1229 |
| centre | | **817.5** | **1102.5** | |

The playfield's own floor seams, read as maxima of `|d/dx|` across a quiet strip of
our half (y 560-620), land at 857, 881, 907, 933, 959, 984, 1010, 1035 — a spacing of
**25.5 px/tile**, and a left edge at 857 − 5×25.5 = **729.5**, which agrees with the
left wall ending at 727 to within 2 px. Bridge centres in cell-index coordinates are
then **2.95** and **14.13**.

**(b) Tower-anchored.** `perception/tools/arena_anchor.py` fixes the map on the two
Princess Towers alone and reads the scale off their separation. Landmarks NOT used in
that fit come back as:

| landmark | engine | measured | delta |
|---|---|---|---|
| own Princess | 3.00 / 14.00, y 6.00 | 3.04 / 13.92, y 6.00 | <= 0.08 |
| opp Princess | 3.00 / 14.00, y 27.00 | 2.96 / 14.08, y 26.92 | <= 0.08 |
| King x | 8.50 | 8.33 / 8.48 | <= 0.17 |
| King y | 2.50 / 30.50 | 1.97 / 30.31 | 0.53 / 0.19 |
| **left bridge** | **2.50** | **3.12** | **+0.62** |
| **right bridge** | **14.50** | **13.98** | **-0.52** |
| bridge y | 16.50 | 16.09 | -0.41 |

Everything the fit did not see reproduces to within a fifth of a tile **except the
bridges**, which miss by half a tile in opposite directions — i.e. the engine's
bridges are too far apart, symmetrically.

**(c) The qualitative check that needs no arithmetic.** In screen pixels the left
bridge centres on x = 817.5 and the left Princess Tower on x = 819; the right bridge
on 1102.5 and its tower on 1101. **Bridge and tower are the same column, within
1.5 px of a 25.5 px tile.** The engine separates them by 0.5 tiles.

### Why the existing reasoning reached 2.5

CLAUDE.md's Board-geometry section derives 2.5 from the river row string
`WWBBWWWWWWWWWWBBWW` — bridge cells {2,3} and {14,15}, whose centres are the seams at
2.5 and 14.5. **That string is the thing to re-check**, because it and the measurement
cannot both be right: cells {2,3} centre on 2.5, and the measurement says 2.95-3.12.
A bridge on cells **{3,4}** would centre on 3.5 in edge coordinates and **3.0 in cell
indices**, which is what was measured. So the likely error is the string being one cell
too far out on each side, not the centring convention that was fixed on 2026-08-21 —
that convention is independently corroborated here, since the Kings measure 8.33/8.48
against `CENTER_X = 8.5` and not against the 9.0 the old fit used.

### The edit, if accepted

`ArenaLayout.h`: `LEFT_BRIDGE_X` 2.5 -> 3.0, and `RIGHT_BRIDGE_X` follows by
`mirrorX`. That alone makes bridge and lane share a column, which is the measured
arrangement. `Board`'s river-row construction has to move with it or the two disagree
again — and the pair should be pinned by a test that samples the constructed row
rather than restating the expected columns, so the next arena change cannot leave one
behind.

### Blast radius

**Gameplay-affecting, and not marginally.** Every ground unit crossing the river is
routed to `bridgeXFor(x)` and then to `laneXFor(x)`, so today every crossing unit makes
a half-tile lateral jog on landing that the real game does not ask for. That is the
kind of motion the open collision-wedge defect (`test_navigation_wedge.cpp`'s standing
`[!shouldfail]` case) lives on, and moving the bridge onto the lane removes the jog
entirely. **Every win rate in CLAUDE.md predates it**, and `HeuristicOpponent`'s own
bridge constants must be re-derived rather than left as the seventh stale copy.

### What this does NOT settle

The bridge-y result (16.09 measured against `BRIDGE_Y = 16.5`) is **not** proposed as
a change. The Kings miss their measured y by a similar amount in the same direction
(1.97 against 2.50), which points at a systematic offset in how a sprite's visual
centre maps to its footprint centre rather than at the river band. The x result has no
such companion error — the Princess Towers reproduce their x to 0.08 — which is why
only x is proposed.

One recording, one resolution, one calibration profile. A second recording at a
different resolution would make it two-point.

---

## 27. MEASUREMENT 2026-08-24 — UPSTREAM item 25's missing third Medium data point, and a new question about the Fast tier

UPSTREAM item 25 (now in `DECISIONS.md`) closed with: *"a third measurement in the Medium band would make it
three-point. The two Medium units available in these recordings were both in combat
rather than walking cleanly (0.749 and 1.183, a 1.6x spread inside one tier) and were
discarded rather than averaged."*

`perception/tools/measure_speeds.py` supplies it, over 72 tracks from one match. Per
track: the median of the segments that are moving (>= 35% of that track's own peak), so
deploy and attack pauses do not drag the walk down.

| card | tier | tracks | video tiles/s | tiles walked | engine | engine / video |
|---|---|---|---|---|---|---|
| Archers | Medium | 4 | **1.28 +/- 0.17** | 6.4 | 1.3254 | 1.03 |
| Valkyrie | Medium | 5 | **1.23 +/- 0.22** | 7.9 | 1.3254 | 1.08 |
| Giant | Slow | 3 | **0.96 +/- 0.05** | 10.6 | 0.9940 | 1.03 |
| Mini PEKKA | Fast | 7 | **1.52 +/- 0.17** | 8.7 | 1.9881 | **1.31** |
| Musketeer | Medium | 4 | 0.82 +/- 0.25 | 12.1 | 1.3254 | 1.63 |

### What it corroborates

**The Slow and Medium tiers hold.** Giant lands on 0.96 +/- 0.05 against the engine's
0.994 — the tightest row in the table and a direct confirmation of UPSTREAM item 24's
recalibration. Archers and Valkyrie are two Medium units that **agree with each other**
(1.28, 1.23) and with the engine's 1.3254, which is exactly the three-point
corroboration UPSTREAM item 25 asked for and did not have.

### Musketeer is the same contaminated case UPSTREAM item 25 already saw

Musketeer reads 0.82 — the widest gap in the table and the one to ignore. It shares a
tier with Archers and Valkyrie, so all three must measure alike, and it does not. Note
that UPSTREAM item 25 recorded a discarded Medium reading of **0.749**; this is almost certainly
the same unit failing the same way, now seen twice.

Two explanations were tried and both were **refuted by the data**, which is why the row
is left open rather than explained: that it is stopped shooting for most of its life
(62% of its segments are moving, in line with every other card), and that its tracks are
too short (12.1 tiles is the **longest** median in the table). Neither holds.

### The new question: the Fast tier

Mini PEKKA measures **1.52 +/- 0.17** against the engine's **1.988** — a gap of 2.8
standard errors, on the most-tracked card in the sample.

It is a question about the **tier ratio**, not the calibration. Measured Medium is
~1.25 and measured Fast is ~1.52, a ratio of **1.19**; the official stats put
Fast/Medium at 90/60 = **1.50**, which is what the engine implements. So either the tier
constants do not have the ratio the stat numbers imply, or the Mini PEKKA measurement is
low.

**No change is proposed on this.** One card, one match, and UPSTREAM item 25's own warning about
units in combat applies to a Mini PEKKA as much as to a Musketeer. What it justifies is
a targeted re-measure: Mini PEKKA and one other Fast card, on clean lane walks, sampled
faster than the 2 fps used here.

---

## 28. PERCEPTION-SIDE, NOT AN ENGINE REQUEST 2026-08-24 — two faults the side-by-side bench measured in our own vision stack

Recorded here only because they bound how far anything above can be trusted. Neither
needs an engine change.

### The calibration profile is a SEVENTH copy of the arena constants, and it is stale

`config/profile_gpg_1920x1080.json`'s homography was fit against tile coordinates taken
from `geometry.py` as it stood on 2026-07-30, when the left Princess Tower was believed
to be at x = 4.0 and the board centre at 9.0. The engine reversed both on 2026-08-21.
The profile was never re-fit, so it still returns:

```
own_princess_left  -> x = 3.94   (engine: 3.00)
opp_king           -> x = 8.97   (engine: 8.50)
```

CLAUDE.md's list of stale copies has moved from four to six; this is the seventh, and
it is the first **fitted** one — which is why "derive, do not restate" did not catch it.
A fit encodes the constants it was fitted against, silently and with no literal anywhere
to grep for. `perception/tools/arena_anchor.py` exists so the comparison work above is
not shifted by it; **re-fitting the profile is the real fix and has not been done.**

### The side detector disagrees with direction of travel on a third of tracks

A unit walks toward the opponent, so the sign of its vertical travel names its owner
without reading a pixel. Over the unstitched tracks of one match with at least 1.5 tiles
of travel in their first 5 seconds:

| | |
|---|---|
| usable tracks | 30 |
| direction agrees with `side.onnx` | 19 (63.3%) |
| direction **contradicts** it | **11 (36.7%)** |

Including unmissable ones: a `minipekka` labelled *ally* that walked 8.4 tiles **down**
the board in five seconds, and a `musketeer` labelled *enemy* that walked up.

For a state estimator this is worse than the card-identity error already on record
(README's 33.8%): a unit with the wrong **name** has the wrong stats, while a unit on the
wrong **team** fights for the other player and inverts the fight it is in. It also
explains a reconstruction artefact directly — an `enemy cannon` at (8.1, 9.3), a
position no enemy building can occupy.

`track_appearances.correct_sides` now overrules the detector from travel where travel is
unambiguous, and reports the count. That is a workaround at the consumer; the detector
itself is untouched.

### Reproducing all of it

```
python perception/tools/detect_video.py "perception/videos/<match>.mp4" --out det.json --fps 2
python perception/tools/track_appearances.py det.json --match-start 5.5
python perception/tools/measure_speeds.py det.json
python perception/tools/build_side_by_side.py --detections det.json \
       --video arena.mp4 --out payload.json
```

The engine column comes from `.vid/final.tsv`, the per-card speed dump UPSTREAM item 25 produced.
