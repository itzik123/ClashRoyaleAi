# perception/ as a live sensor — design

Date: 2026-07-30
Status: design agreed, not implemented

---

## 1. Role boundary

`perception/` is a **sensor**. Frame in, "here is exactly what is on the field,
here is our elixir, here is our hand" out. Object detection, ROIs, OCR.

Explicitly **not** perception's job:

- estimating the opponent's hidden elixir — the policy's `aux_elixir_head` is
  trained to do that;
- holding match state in the simulator as an estimator;
- choosing actions, or executing them in the game;
- **rendering the observation vector.** Decided 2026-07-30: perception emits
  `GameState` and stops. Encoding it into the policy's 13,606 floats belongs to
  the training side, which owns the layout and changes it.

This is a deliberate change of direction. The module was previously built on
"vision detects events only, the simulator holds the state". That architecture
is retained only as a test oracle (§5), not on the live path.

### Why the sensor architecture is the right call, measured

The estimator approach compounds errors. Running the frozen replay fixture
through `SimDriver` with *inferred* opponent placements — the realistic
condition, since replays label only our own plays — gives:

```
max divergence 529.5 HP/tower
```

At t=32s the estimate had destroyed the opponent's Princess tower while the
real one stood at 1583 HP. A sensor has no such failure mode: every frame is
read fresh, so a missed reading is corrected by the next one. **The requirement
drops from "never miss an event" to "mostly right per frame."**

One exception, recorded in `BOT_REQUESTS.md` item 4: extra scalars 1-2 are
*cumulative* elixir spend, the only accumulators in the observation. A missed
opponent placement is permanent there.

---

## 2. The output contract

The policy consumes 13,606 floats (`observationSize()`), but almost all of it
is derived rather than perceived:

| part | floats | source |
|---|---|---|
| ch 8 — river/bridges | 612 | **static**, computed, never perceived |
| ch 11-20 — flying, antiair, dps, range, speed | 6,120 | pure function of **card identity** |
| ch 0-7 — type class + HP | 4,896 | type class from identity, HP from the bar |
| ch 9/10 — units per cell | 1,224 | counting detections |
| hand costs + identity one-hots | 744 | hand identity |
| 9 appended scalars | 9 | clock, tower HP, cumulative spend |

So perception's real output is small:

```
per visible entity:  card_id, team, tile_x, tile_y, hp_fraction
per hand slot:       card_id
own elixir, clock + phase, 6x tower HP, own + opponent cumulative elixir spent
```

Roughly 20-40 numbers per frame.

**Decision (2026-07-30): perception emits `GameState`. Nothing else.** No
encoder here. The table above is therefore not a build list for perception — it
is the map of what the *consumer* derives, and it is included so both sides can
see that `GameState` is sufficient to reconstruct all 13,606 floats.

Rationale:

1. A `GameState` can be logged, diffed, and eyeballed. A 13,606-float vector
   cannot.
2. The observation layout already changed once (6253 → 13606 on 2026-07-29).
   A compact state survives the next change; an encoder living on the wrong
   side of the boundary does not. The layout is the training side's to change,
   so the encoder belongs with it.
3. `GameState` is verifiable against the engine with no labels (§5).

**`GameState` is the interface, and it is a shared contract.** It lives in
`contracts.py` — already "dataclasses only, no logic, no dependencies", which
is the right home for exactly this. It must not change without the training
side agreeing (§9).

Two fields the contract has to carry that are easy to forget, both because the
consumer cannot recover them afterwards:

- **per-entity confidence**, for the same reason `contracts.py` already argues
  for it: the engine has no representation of uncertainty, so a tile is either
  occupied or zero and zero means "empty", never "I could not see".
- **`hp_fraction` relative to the card's own max**, not to `MAX_TROOP_HP`. The
  bar on screen is a fraction of the unit's own health, and only `cardtable`
  (§3.1) knows the absolute maximum. Reporting the raw fraction keeps
  perception out of the normalisation business entirely.

---

## 3. Components

### 3.1 `cardtable.py` — attributes, generated not stored

`get_card_info` exposes `cost`, `is_spell`, `is_building` and nothing else. The
melee/ranged/tank type class, `isFlying`, `targetsAir`, damage-per-tick, attack
range and speed all exist in C++ and are **unbound**.

They do not need binding. The observation encoder already writes every one of
them, so injecting one card alone onto an empty board, stepping 1 tick past the
deploy delay, and diffing against an untouched env recovers the card's full
attribute row — read out of the exact encoding the policy consumes, which is
strictly more trustworthy than a hand-transcribed table.

Verified on the full roster: **108 of 132 playable cards, 22 spells skipped,
0.96 s wall time.** Two produce nothing — Royal Ghost (47) and Suspicious Bush
(85), both stealth cards that `isTargetable()` correctly hides.

Spot-check against the header's own documented values (Musketeer):

| channel | raw | decoded | expected |
|---|---|---|---|
| ch5 enemy ranged | 0.1694 | 721 HP | ~720 |
| ch10 count | 0.2 | 1 unit | 1 |
| ch14 antiair | 1.0 | targets air | yes |
| ch18 range | 0.5 | 6.0 tiles | 6.0 |
| ch20 speed | 0.3333 | 0.5 | 0.5 |

**Generated at import, never committed as JSON.** 0.96 s is affordable, and a
committed table is exactly the staleness failure mode `CLAUDE.md` names twice.

Two knock-on wins:

- spawn HP is full HP, so the table yields **max HP per card** — which is what
  converts an observed bar *fraction* into the normalised value the
  observation wants;
- the table yields **cells per card** (Skeleton Army 11, Barbarians 4, Goblins
  4, Knight 1), i.e. how many badges a placement should produce. That becomes a
  free consistency check (§5).

`MAX_UNIT_DPS` (60), `MAX_ATTACK_RANGE` (12), `MAX_UNIT_SPEED` (1.5) and
`MAX_CELL_UNITS` (5) are **not bound**; hardcode them with a comment naming
`include/core/ClashEnv.h`, following the pattern `geometry.py` already uses.
Binding them is worth a one-line request but is not blocking.

### 3.2 `capture/window.py` — live frames

Does not exist today; `capture/window.py` was deliberately not built. Needed
for any live operation. Slots into the existing `FrameSource` ABC so every
offline test keeps working unchanged.

Must report achieved rate, latency and jitter, and must refuse to silently
run below the target rate.

### 3.3 Readers — fix and promote

**Hand identity is measurably broken.** Over all 8 recordings, 328 in-match
candidate plays: the icon template agrees with the elixir ledger on card cost
only **33.8%** of the time, and it over-predicts Giant at **35% against a 12.5%
prior**. Direct proof from the trace: Giant is played out of slot 1 at t=21.0
and one second later reads as present in slot 2, holding for 10 seconds. A
played card goes to the back of an 8-card queue; this is physically impossible
and no debouncer can catch a stably-wrong classifier.

Fixes, in the order the evidence supports:

- **elixir as identifier, not validator.** The elixir reader is the strongest
  in the pipeline (mean confidence 0.978-0.991, 96.7-98.9% above gate) and
  drops cluster cleanly at card costs. Observed drops land within 0.6 of a real
  cost 74.7% of the time versus the template's 33.8% agreement. Use the drop to
  constrain cost (3 / 4 / 5), combined with the known deck and the FIFO cycle
  model, and let the template answer only *which slot changed* — which it is
  far better at than identity.
- **enforce the 20-tick slot cooldown** as a hard rule. Purely definitional,
  and it drove same-slot violations to 0-2 per match.
- **rejection must be revocable by persistence.** A misread does not persist; a
  real play does. A rejected hand that holds >3 s must be accepted and flagged.
  This fired 4-17 times per match — a hot path, not a rare fallback.
- **classifier quality:** matching is on contrast-normalised greyscale, which
  discards colour. Add a hue channel as a second independent score (hue
  survives the affordability dimming that motivated greyscale in the first
  place), and crop the elixir cost badge out of the slot ROI — it is
  structurally identical across cards, so it is pure common-mode signal
  inflating every correlation.

**Tower HP is promoted from validation to primary output.** `readers/towers.py`
currently says "VALIDATION ONLY … Nothing here may enter the observation
vector"; under a sensor architecture it is a primary output. Two changes:

- The calibration profile has **no tower ROIs at all** (only `clock`,
  `elixir_bar`, 4 hand slots, `next_card`), so `TowerHpReader` can never be
  constructed. Derive the six ROIs from `calibrate.py`'s existing `find_towers`
  stone-blob detection rather than hand-measuring six rectangles.
- **OCR the numeral, not the bar.** Tower HP is rendered as a large clean
  number (`1746`, `1602`) drawn *over* the bar, so column-counting would be
  corrupted by the white digits — the identical failure already documented for
  the elixir bar. The numeral also removes the `max_hp` caller input entirely.

Bonus: the engine's tower HP scale matches the real game (princess 2534, king
4008), so OCR'd absolute HP ÷ `MAX_BUILDING_HP` feeds the scalars directly.

**Phase is observable, so the phase schedule is unnecessary.** The clock ROI
reads `Overtime` and a large `x2` badge is drawn on screen. This closes
`README.md` open question 1 outright — no boundaries need to be supplied or
guessed, and `PhaseScheduleUnknownError` stops being a live-path concern.

**Match state machine — a prerequisite, and missing.** The first ~18 s of every
recording is the pre-battle screen: the elixir bar is not yet drawn so the ROI
reads a **confident 0.00**, and the hand slots read garbage such as
`[2,2,2,2]`. This poisoned every measurement taken before it was gated. States:
`pre-match / live / overtime / result`. Gate on the hand slots and elixir bar
being *drawn* — not on the clock, which returns confident readings during the
intro.

### 3.4 `detect/badges.py` — the detection primitive

Every entity — troops and buildings, both teams — carries a team-coloured level
badge with an HP bar attached. Confirmed on our Cannon (blue badge `6`, blue
bar at ~35%) and on enemy Goblins (magenta badge `6`, bar ~55%). The bar
measures roughly **42x10 px** at 1920x1080.

One primitive yields presence, team (badge colour), level (a digit), HP (bar
fill; **absent bar = full HP**, so undamaged is exact), and position (fixed
offset above the entity's ground point).

Why badges rather than sprites: fixed size regardless of unit, high contrast,
two colours only, drawn on top — and in the most crowded frame examined, all
three badges stayed legible while the sprites beneath them were an
unrecognisable pile behind a spell effect.

Known risks, to be handled explicitly:

- badges **occlude one another** (a blue badge was clipped by a magenta one);
- spell VFX produce large saturated magenta regions in the **same colour
  family** as the enemy badge, so the detector must key on the badge's actual
  signature — a small rectangle containing a white digit — not on "magenta
  pixels";
- swarm cards produce a dense badge cluster in few cells.

No training data required.

### 3.5 `detect/identity.py` — open-set card identity

**Target: full open-set recognition over all ~120 cards**, both team colours.

Two design consequences:

- a real classifier (small CNN over sprite crops), not a template set;
- **classify at spawn, then track.** At placement the unit is isolated,
  stationary and unoccluded; mid-fight it is neither. Per-frame appearance
  classification would fail hardest in exactly the moments that matter most.

**Labels come from single-card calibration recordings.** The user will produce
~15 decks covering all ~120 cards and record, for each card, a stretch in which
only that card is played — so every unit on screen for the following seconds is
known to be that card. Zero manual annotation.

The ingestion tool reuses the hand work from §3.3: hand-change detection plus
the elixir-drop constraint locates each placement moment, the badge detector
locates the resulting units, and every harvested crop is auto-labelled with the
one card being played. Crops are harvested across animation frames and angles
by tracking.

Until those recordings exist, identity is stubbed and the 8 recordings already
in `assets/recordings/` are used for testing everything else. Note those are
ladder matches against unknown decks, not mirror matches — they exercise the
readers and the badge detector, but they cannot validate identity.

---

## 4. What leaves the live path

`bridge/sim_driver.py`, the candidate pool, `track/opp_elixir.py` and
`track/opp_deck.py` are state modelling, which is the policy's job. They leave
the live path.

They are **not deleted**: the engine is retained as a test oracle (§5), which
is now its most valuable use here.

---

## 5. Verification — every check is label-free

1. **`GameState` ↔ engine entity list.** Drive the engine to a known state and
   build a `GameState` from its own entity list, then assert that a
   `GameState` produced by perception from a *rendering* of that same state
   agrees on entity set, team, tile and HP. This is a state-level comparison,
   not a vector-level one — the vector belongs to the consumer now — but it
   still catches the whole mirroring-and-coordinate-frame class of bug, which
   is otherwise invisible until the policy plays badly for unexplained reasons.
   The `y` convention is the specific trap: `contracts.py` defines tile_y as
   already mirrored for team 1, and `injectEnemy` takes raw coordinates.
2. **Hand stream impossibility counters**, all of which must reach ~0:
   same-slot gap < 2.0 s; flip-flop A→B→A in one slot; readings rejected as
   duplicate or off-deck.
3. **Elixir budget play count.** Integrate elixir income over the match from
   the clock (2.8 s per elixir, doubling at 120 s) and divide by the deck's
   average cost. An upper bound, not ground truth — it assumes near-complete
   spending — but it brackets the true play count with no labels. Measured
   across all 8 recordings, the elixir-constraint prototype lands at **60-76%
   of budget on seven of eight** (94% on the outlier), i.e. still
   under-detecting — which is the evidence that the constraint belongs on
   identity rather than acting as a rejection filter.
4. **Badge count vs expectation:** cards played × `cells` per card from
   `cardtable`. Free, and it directly measures detector recall.
5. **Tower HP: numeral OCR vs bar fraction** as mutual cross-check.

---

## 6. Milestones

**M1 — live sensor, empty board.** Components 3.1, 3.2, 3.3. Reports elixir,
hand, clock + phase, tower HP and match state, live. Complete and measurable on
its own; unblocked today.

**M2 — board occupancy.** Component 3.4. Adds presence, team, HP, position and
per-cell counts. Fills channels 0-10 partially; attribute channels still
default.

**M3 — identity.** Component 3.5, gated on the calibration recordings. Fills
channels 11-20 and the hand one-hots.

---

## 7. Out of scope

- Sending any input to the game. Nothing here actuates.
- Estimating the opponent's current elixir.
- Any change to `include/`, `src/` or `python_ai/`. Engine requests go to
  `UPSTREAM_REQUESTS.md`; policy suggestions go to `BOT_REQUESTS.md`.

### Landed upstream while this was being written (2026-07-30)

`UPSTREAM_REQUESTS.md` items **1 and 2 are both done** — commit `dd99991`,
with the `.pyd` rebuilt after it, so every measurement in this document already
used the corrected geometry. Verified live from the engine:

```
princess L/R  (4.0, 6.0) (14.0, 6.0)      was x = 3.0 on the left
kings         (9.0, 2.5) (9.0, 30.5)      was x = 8.5
river / bridges unchanged: [15.5, 17.5), (4.0, 16.5) and (14.0, 16.5)
```

**This creates a new work item that did not exist before: re-solve the
calibration profile.** `config/profile_gpg_1920x1080.json` carries
`reprojection_error_tiles = 0.626`, fitted against the *old* tower coordinates.
Per `UPSTREAM_REQUESTS`' own measured table, re-solving against both fixes
should give max **0.31** / rms **0.21** — passing the stage-0 target of < 0.5
for the first time. Cheap: `tools/calibrate.py`. Add to M1.

`DEFAULT_DECK` is now `[10, 1, 41, 25, 7, 2, 6, 5]` — Valkyrie, Archers,
Minions, Cannon, Fireball, Giant, Musketeer, Mini P.E.K.K.A, i.e. exactly the
deck in the recordings. `BOT_REQUESTS.md` item 2 is therefore resolved in the
opposite direction to the one proposed: the deck moved to the recordings rather
than the recordings to the deck. Two consequences here — the
out-of-distribution concern for anything derived from these recordings is gone,
and the 8 icon templates already in `config/templates/icons/` are now the
*correct* deck, so no new templates are needed.

---

## 8. Assumptions to confirm

- **Live capture rate.** The policy decides once per second (`skip_frames = 10`),
  so 2-4 Hz should suffice. Current offline cost is roughly 33 ms/frame for
  elixir + hand + clock + decode, dominated by decode. Unmeasured for live
  capture.
- **Actuation/latency offset** is unmeasured. `PlacementEvent.wall_time_ms`
  exists precisely so it can be measured later.
- **HP bar resolution.** ~42x10 px supports on the order of 20-40 levels, so
  roughly 3-5% of max. Not yet measured against known HP.
- **Colour augmentation** as a fallback for the opposing team's tint is
  untested; the calibration recordings are expected to cover both colours
  directly.

---

## 9. Coordination with the training side

Prior lack of coordination between the two sides has cost real time on this
project, so the boundary is written down rather than assumed.

### Ownership

| artifact | owner | the other side may |
|---|---|---|
| `GameState` in `contracts.py` | **shared** — neither side changes it alone | — |
| observation encoding (the 13,606 floats) | training | read the layout, never write an encoder |
| observation layout / `CH_*` / scalars | training | be notified when it changes |
| `cardtable` probe | perception | rely on it being regenerated, never cached |
| calibration profile, ROIs, homography | perception | — |
| card identity model + calibration recordings | perception | — |
| opponent elixir estimation | training | not implement it here |

### Rules for the two request files

`UPSTREAM_REQUESTS.md` (engine) and `BOT_REQUESTS.md` (policy) are the standing
channel. Two rules, both learned the hard way today:

1. **A status claim carries its evidence, in the file.** On 2026-07-30
   `BOT_REQUESTS.md` was marked up with statuses referring to a
   "Training-side response" section at the bottom — and that section does not
   exist in the file. Item 2's claim was independently verifiable in the code
   and turned out true; item 4's ("void as stated") was not verifiable at all
   from the file. An unverifiable status is worse than an open one, because it
   closes the item.
2. **When a claim is resolved, fix the body, not just the status row.**
   `BOT_REQUESTS.md` item 2's body still reads "the fix belongs to whoever
   records next: play `DEFAULT_DECK`", which is now backwards — the deck moved
   to the recordings instead.

### Two items needing a training-side answer now

Both follow from commit `dd99991`, which perception asked for and which has
landed. Raising them because perception requested the change and flagged the
blast radius, and it should confirm the consequence was taken rather than
assume it:

- **The tower x-geometry fix is gameplay-affecting.** `CLAUDE.md`'s own rule is
  that such a change makes `model_weights.pth`'s win-rate history suspect, and
  `UPSTREAM_REQUESTS` item 1 stated that explicitly under "Blast radius",
  including a full `ClashRoyaleTests` run. Was the test run done, and is the
  current win-rate history being treated as a new baseline?
- **`DEFAULT_DECK` changed**, so any checkpoint trained on
  `[15, 25, 6, 1, 0, 41, 7, 10]` is now playing a different deck than the one
  its hand one-hots were learned on. That is a larger discontinuity than the
  geometry change, and perception has no visibility into whether it was
  intended as a fresh run.

Neither is perception's call. Both are recorded so the answer exists somewhere.
