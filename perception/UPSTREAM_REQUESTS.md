# Simulator changes requested from `perception/`

Written from `perception/`, which modifies nothing outside itself. **Nothing
in this document has been applied by me.**

Last updated 2026-07-30, after items 1-2 landed.

| # | Request | Severity | Status |
|---|---|---|---|
| 0 | River band `[16,18)` → `[15.5,17.5)` | was blocking training | **DONE — verified** |
| 1 | Left Princess towers `x = 3.0` → `4.0` | blocks a stage-0 acceptance target | **DONE — verified** |
| 2 | Kings `x = 8.5` → `9.0` | cosmetic accuracy | **DONE — verified (landed with item 1)** |
| 3 | King Tower has no activation condition | fidelity gap | open, **already worked around, no change needed** |
| 4 | `inject(..., team)` + `get_hand(team)` | convenience | **DONE — already landed 2026-07-29, see below** |
| 5 | Team-1 observation mirrors the truncated row, not the position | **corrupts all self-play** | open, proposed 2026-07-31 |
| 6 | River marker row is 17 for team 0 but 16 for team 1 | same class, smaller | open, proposed 2026-07-31 |

Items 1 and 2 were done together since the measured benefit is combined
(max error 0.63 → 0.31 tiles) and neither is a large or risky edit.

---

## 0. DONE — the river band (verified 2026-07-29)

`Board.h` now reads:

```cpp
float riverY_start = 15.5f;
float riverY_end   = 17.5f;
Vector2D leftBridge { 4.0f, 16.5f };
Vector2D rightBridge{ 14.0f, 16.5f };
```

The band is centred on 16.5, which is the axis the tower layout already
mirrored about (King 2.5 ↔ 30.5, Princess 6.0 ↔ 27.0 under `y → 33 - y`).

**Verified two ways, both independent of the change itself.**

The placement asymmetry is gone. Same test that found it — 20 trials per row,
cheap cards only so affordability never limits, each team in **its own
mirrored frame**:

| row (own frame) | team 0 | team 1 | before the fix |
|---|---|---|---|
| 14 | 15/20 | 16/20 | 17/20 vs 15/20 |
| **15** | **14/20** | **17/20** | **15/20 vs 0/20** |
| 16 | 0/20 | 0/20 | 0/20 vs 0/20 |

Both teams now reach row 15 and stop at 16. `get_own_half_max_y()` returns
15.0, and the rebuilt `.pyd` carries it.

And the calibration residual against real footage improved: **rms 0.40 →
0.32**.

This one mattered beyond geometry: `model.py` computes `own_half_rows` once
and applies the same placement mask to both sides, so team 1's policy was
proposing row-15 placements that `playCard` silently rejected — gradient
spent on an action that could never do anything, and only when playing team 1.

---

## 1. DONE — left Princess towers, `x = 3.0` → `4.0` (verified 2026-07-30)

**Landed exactly as requested**, two lines in `GameManager.h`:

```cpp
addTower(4.0f,  6.0f, 0, "Princess Tower", towerTroopStats(aiTowerTroop));
addTower(4.0f, 27.0f, 1, "Princess Tower", towerTroopStats(oppTowerTroop));
```

`ClashRoyaleTests` re-run in full afterwards (504 test cases, 4329
assertions, 0 warnings, `-Wall -Wextra`) and a `main.cpp` smoke match still
completes normally. `perception/geometry.py`'s `OWN_PRINCESS_LEFT`/
`OPP_PRINCESS_LEFT` updated to match, so `tools/calibrate.py`'s
`engine_tiles()` now returns this directly.

### Why

In the real arena a Princess tower sits directly behind its own bridge —
troops crossing walk straight into it. Measured from the recordings: left
tower centre **x = 819 px**, left bridge centre **x = 821 px**. The same lane,
within measurement error.

The engine puts the left bridge at `x = 4.0` and the left Princess at
`x = 3.0`. A full tile apart. The right side is already correct (both at
14.0), so the left lane is the only one that disagrees with itself.

### What it is worth, measured

Screen→tile homography fitted to the eight arena landmarks, aggregated over
the opening frames of all 8 recordings. Identical pixels, identical solver;
only the target tile coordinates differ:

| target geometry | max error | rms |
|---|---|---|
| engine as-is (river already fixed) | **0.63** | 0.33 |
| **+ left Princess `x` 3.0 → 4.0** | **0.40** ✅ | 0.27 |
| + Kings `x` 8.5 → 9.0 (alone) | 0.51 | 0.37 |
| + both | 0.31 | 0.21 |

**Stage 0's acceptance target is < 0.5 tiles, and this change alone reaches
it.** Without it, every placement perception reports starts with 0.63 tiles
of systematic error before any detection error is added — 42% of stage 3's
1.5-tile budget, spent on nothing.

### Blast radius

Small, but not zero. It moves a tower, so pathing and aggro around the left
lane change slightly, and `isValidPlacement`'s building-overlap check moves
with it. Needs a full `ClashRoyaleTests` run. It changes gameplay, so treat
`model_weights.pth`'s win-rate history as suspect afterwards — the same
caveat as the river fix.

---

## 2. DONE — Kings `x = 8.5` → `9.0` (verified 2026-07-30, landed with item 1)

Board `[0, 18)` has centre 9.0. `addTower` put both Kings at 8.5, so a
4-tile-wide King spanned `[6.5, 10.5)` instead of `[7, 11)`. Measured king
centre 955.75 px sits at 8.79 in bridge-calibrated coordinates — between the
two, closer to 9.0.

Landed together with item 1 rather than alone, per this file's own note that
it "reaches max 0.51 (still failing)" in isolation — combined, held-out
calibration error dropped max 0.63 → 0.31 tiles, rms 0.33 → 0.21.
`test_game_manager.cpp`'s King-position assertions and the building-overlap
tests measuring distance from the King (whose test points shared the King's
old x, so shifting both together preserved the same distances) were updated
to match.

---

## 3. OPEN — King Tower has no activation condition. **No change requested.**

`Tower.h` builds the King like any other tower; `GameManager::reset()` gives
it range 7.0 and a 10-tick cooldown. Nothing makes it dormant, so it fires
from tick 0. The real King is inert until activated.

**Already handled on this side and no change is being asked for.**
`SimDriver.divergence` excludes both Kings and `readers/towers.king_divergence`
reports them separately, so the pipeline's quality metric measures perception
rather than this known gap.

Recorded here only because it is a real behavioural difference that also
affects training: an agent learns that chip damage to the King is punished
immediately, which is not true of the real game.

---

## 4. DONE — `inject(cardId, x, y, team)` and `get_hand(team)` (landed 2026-07-29)

**Already in the engine, exactly as requested below** — `ClashEnv::inject`/
`getHandForTeam` and their `bindings.cpp` entries (`inject`/
`get_hand_for_team`) landed alongside the river fix, before this being "not
blocking" ever mattered. `perception/bridge/sim_driver.py`'s reverse-engineered-
shuffle workaround described below still works and hasn't been switched over
to the new primitives — that's a `perception/`-side follow-up, not an engine
one, and is optional given the workaround already passes with zero
divergence.

### What is awkward

`ClashEnv::injectEnemy` hardcodes team 1:

```cpp
void injectEnemy(int cardId, float x, float y) {
    const auto* card = CardRegistry::getInstance().getCard(cardId);
    if (card) card->spawnEntity(x, y, 1, game.getBoard());
}
```

There is no ally equivalent, so our own placements must go through
`playCard`, which requires the card to be in the simulator's own hand — and
that hand is shuffled at reset by an unseeded `std::mt19937`, cannot be set,
and the queue behind it cannot be read (`getHand()` is team 0 only).

### The workaround, and what it costs

`bridge/sim_driver.py` reverse-engineers the shuffle: draw ~20,000 resets
(0.135 ms each, measured), keep the ~200 whose hand-set matches the real
opening hand, carry them all forward, and eliminate the ones that could not
have dealt what reality dealt. After four confirmed deals every survivor has
our exact cycle, and from there the two FIFOs cannot diverge.

It works — 0 refusals, 0 desyncs, divergence 0 on the control. It costs:

- ~2.7 s of startup per match and ~1.4 s of redundant simulation (200
  environments stepping in lockstep until the pool narrows);
- ~150 lines whose only purpose is to undo a shuffle;
- a residual failure mode: `playCard` still checks elixir and placement
  legality against a board the estimate may have slightly wrong.

### The change

Purely additive. Nothing existing changes behaviour:

```cpp
// ClashEnv.h -- injectEnemy stays exactly as it is, so no caller changes.
void inject(int cardId, float x, float y, int team) {
    const auto* card = CardRegistry::getInstance().getCard(cardId);
    if (card) card->spawnEntity(x, y, team, game.getBoard());
}

std::vector<int> getHandForTeam(int team) const { return game.getHand(team); }
```

```cpp
// bindings.cpp
.def("inject", &ClashEnv::inject,
     py::arg("card_id"), py::arg("x"), py::arg("y"), py::arg("team"))
.def("get_hand_for_team", &ClashEnv::getHandForTeam, py::arg("team"));
```

`game.getHand(int)` and `GameManager::playCard(team, ...)` already exist and
are already public. No effect on the observation vector, the action space, or
any checkpoint — a rebuilt `.pyd` stays compatible with current weights.

---

## Explicitly NOT requested

- **Game phases / overtime.** `ELIXIR_REGEN_RATE` is one `const float`
  applied to both players with no phase concept. The phase is carried as a
  field in `ClockState`, marked not-consumed, and left unconnected on
  purpose. Not asking the engine to grow phases.
- **The ~2% elixir gap.** `0.035/tick × 10 ticks/s` = 2.857 s per elixir
  against the real 2.8 — and 2.80 s is now *measured* off the recordings, not
  assumed. Changing it would be a gameplay change mid-training-run.
  `perception/` uses the real rate for the real opponent and the engine's
  rate when reasoning about the engine, and keeps them explicitly separate.
- **Card levels.** The registry has none; recorded matches do. That is why
  `readers/towers.py` takes max-HP as a caller input instead of assuming the
  engine's values.

---

## 5. OPEN — team 1's observation is displaced one row (proposed 2026-07-31)

`ClashEnv.h`, `extractObservationForTeam`:

```cpp
int rawY = static_cast<int>(entity->position.y);
int y = (team == 0) ? rawY : (BOARD_HEIGHT - 1 - rawY);
```

It mirrors the **truncated row** instead of truncating the **mirrored
position**. `33 - int(y)` and `int(33 - y)` agree only when `y` is an integer,
and disagree by exactly 1 otherwise. Every troop in play sits at a fractional
`y`, so this is a permanent, every-tick, every-entity error — and it applies to
team 1 only. Team 0's path is `rawY`, untouched.

This is why the two towers behave differently and why the bug survived the
2026-07-30 geometry audit: Princesses are at `y = 27.0` (integer, mirrors
correctly), Kings at `y = 30.5` (fractional, off by one). The positions
themselves are symmetric — item 0 verified that — so a coordinate audit finds
nothing. Only the encoder is wrong.

### Measured

Mirror-image Valkyrie pairs injected at `y` and `33 - y`, one tick to merge
`pendingEntities`, then both observations read:

| team 0 `y` | team 1 `y` | team 0 sees | team 1 sees |
|---|---|---|---|
| 8.0 | 25.0 | own row 8, enemy 24 | own row **9**, enemy **25** |
| 8.5 | 24.5 | own row 8, enemy 24 | own row **9**, enemy **25** |
| 10.3 | 22.7 | own row 10, enemy 22 | own row **11**, enemy **23** |
| 12.5 | 20.5 | own row 12, enemy 20 | own row **13**, enemy **21** |

At reset, with no entities at all, `get_observation_for_team(0)` and
`(1)` differ in 56 cells — King towers at rows 2 vs 3 and 30 vs 31, plus their
attribute channels.

**Consequence, measured directly.** The frozen main agent played against an
identical copy of itself, same policy both sides, no gradient, 400 episodes:

```
team 0 score 0.598   (238 W / 2 D / 160 L)   95% CI [0.548, 0.647]   z = +3.90
```

A policy beats *itself* 60/40 purely by being assigned team 0.

### Blast radius — smaller than it looks

**Team 0's observation is bit-identical before and after.** The `team == 0`
branch is not touched, so the trainee's own inputs never change and
`model_weights_selfplay.pth` stays valid — this is not a checkpoint-invalidating
change in the usual sense.

What does change: every neural and scripted opponent starts seeing the board
correctly, so **opponents get stronger**. Expect measured win rates to fall.
Specifically:

- Self-play win rates vs the PFSP pool are currently inflated, and PFSP weights
  opponents by `(1 - winrate)^2`, so the sampling distribution is distorted too.
- The 3 *historical neural* Elo anchors get stronger; the 3 *builtin heuristic*
  anchors are unaffected (`HeuristicOpponent` is C++ and never reads the
  observation). Elo is therefore **not comparable across this fix**.
- Pipeline 1 (`train.py` vs `HeuristicOpponent`) is entirely unaffected.
- Both league-exploiter bursts to date measured this, not exploits: burst #1
  scored 0.585 against what is really a 0.598 null, i.e. nothing.

### Proposed edit

```cpp
int y = (team == 0)
    ? static_cast<int>(entity->position.y)
    : static_cast<int>(std::floor((BOARD_HEIGHT - 1) - entity->position.y));
```

`std::floor` rather than a bare cast so an entity behind the back row gives
`-1` and is rejected by the existing bounds check, instead of truncating
toward zero into row 0.

## 6. OPEN — river marker row differs between perspectives

Same function, the marker is drawn on one hardcoded row:

```cpp
int riverRow = (team == 0) ? 17 : (BOARD_HEIGHT - 1 - 17);   // = 16
```

Team 0 sees the river/bridge marker at row 17, team 1 at row 16 — 36 differing
cells in channel 8 at reset. A network trained as team 0 expects the bridges one
row further forward than team 1 shows it.

**Proposed:** `int riverRow = 17;` for both. Each team's own frame is supposed to
be identical, and keeping 17 leaves team 0's observation unchanged.

Whether 17 is the *right* row for a band of `[15.5, 17.5)` is a separate
fidelity question and deliberately not bundled here.
