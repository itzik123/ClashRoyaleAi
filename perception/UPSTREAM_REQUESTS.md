# Simulator changes requested from `perception/`

Written from `perception/`, which modifies nothing outside itself. **Nothing
in this document has been applied by me.**

Last updated 2026-07-29, after the river fix landed.

| # | Request | Severity | Status |
|---|---|---|---|
| 0 | River band `[16,18)` → `[15.5,17.5)` | was blocking training | **DONE — verified** |
| 1 | Left Princess towers `x = 3.0` → `4.0` | blocks a stage-0 acceptance target | **open — the only thing actually being asked for** |
| 2 | Kings `x = 8.5` → `9.0` | cosmetic accuracy | open, low value, see the numbers |
| 3 | King Tower has no activation condition | fidelity gap | open, **already worked around, no change needed** |
| 4 | `inject(..., team)` + `get_hand(team)` | convenience | open, **not blocking, would delete ~150 lines here** |

If only one thing gets done, it is **item 1**, and it is two characters in two
lines.

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

## 1. OPEN — left Princess towers, `x = 3.0` → `4.0`

**This is the request.** Two lines in `GameManager.h`:

```cpp
addTower(3.0f,  6.0f, 0, "Princess Tower", towerTroopStats(aiTowerTroop));   // -> 4.0f
addTower(3.0f, 27.0f, 1, "Princess Tower", towerTroopStats(oppTowerTroop));  // -> 4.0f
```

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

## 2. OPEN, LOW VALUE — Kings `x = 8.5` → `9.0`

Board `[0, 18)` has centre 9.0. `addTower` puts both Kings at 8.5, so a
4-tile-wide King spans `[6.5, 10.5)` instead of `[7, 11)`. Measured king
centre 955.75 px sits at 8.79 in bridge-calibrated coordinates — between the
two, closer to 9.0.

**Worth doing only alongside item 1, not instead of it.** On its own it
reaches max 0.51 (still failing) and makes rms *worse* (0.33 → 0.37),
because it corrects one landmark while leaving the left lane wrong. Combined
with item 1 it takes max 0.63 → 0.31.

Low priority. I would not spend a test cycle on this alone.

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

## 4. OPEN — `inject(cardId, x, y, team)` and `get_hand(team)`. Not blocking.

**A workaround is implemented, tested, and passes with divergence identically
zero.** This is a request to delete complexity, not to unblock anything. If
the answer is no, nothing breaks.

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
