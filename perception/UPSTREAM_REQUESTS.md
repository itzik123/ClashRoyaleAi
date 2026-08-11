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
| 8 | Fireball (689) misses the Musketeer kill (721 HP) by 32 | **fidelity vs learnability — needs a decision, not a fix** | open, proposed 2026-08-06 |
| 7 | No way to seed the engine's RNG | every A/B test costs ~10x more than it needs to | open, proposed 2026-07-31 |
| 9 | **Troop movement is ~4-5x faster than the real game** | **largest measured sim-to-real gap; miscalibrates every timing the agent learns** | **DONE — applied and verified 2026-08-07** |

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

---

## 7. OPEN — the engine's RNG cannot be seeded (proposed 2026-07-31)

**Not a correctness bug. A cost multiplier on every experiment this project
runs**, including the ones `CLAUDE.md` already recommends re-running.

### What is there now

```cpp
// ClashEnv.h:362
rng(std::random_device{}()) { heuristicOpponent.reset(rng); }
```

Seeded once at construction from `std::random_device`, with no setter.
`MicroRoyaleEnv.reset(seed=...)` looks like it should help but only forwards to
`gymnasium.Env.reset`, which seeds the *wrapper's* RNG, not the engine's.

`CLAUDE.md` already records that the engine has exactly two sources of
randomness — the opening-hand shuffle in `PlayerState::initializeDeck` and
`HeuristicOpponent` — and that "identical inputs give identical outcomes". That
determinism is currently unreachable from outside, because the one thing that
varies cannot be pinned.

### What it costs, measured on the experiment that prompted this

Comparing two observation variants (extra scalar 2 correct vs zeroed) against
`heuristic@1.35`. Because the arms cannot share a seed, the comparison is
unpaired and needs

```
n = 2 * (1.96 + 0.84)^2 * 0.25 / 0.05^2  ~=  1568 episodes per arm
```

to resolve a 5-point win-rate difference at 80% power — **3,136 episodes**. With
a shared seed the same episodes become matched pairs, most of the variance is
the shared opening hand and opponent rolls rather than the treatment, and the
same resolution needs roughly an order of magnitude fewer games.

This is not a one-off. The same shape applies to every question already on the
project's own list: the entropy-rate ordering that `CLAUDE.md` flags as "one run
per configuration, so this ordering may not survive replication", the deck
choice re-opened on 2026-07-30, and the corruption ablation in
`BOT_REQUESTS.md` item 1.

### Second benefit: reproducible failures

A self-play regression currently cannot be replayed. The 2026-07-31 team-1
observation bug was found by running a policy against a bit-exact copy of
itself and noticing 0.598 where 0.500 was expected — a test `CLAUDE.md` now
recommends after any change to the observation, the board, or `stepSelfPlay`.
That test is a coin-flip null measured over hundreds of games precisely because
individual games cannot be reproduced.

### The change

```cpp
void seed(unsigned int s) { rng.seed(s); heuristicOpponent.reset(rng); }
```

plus a `.def("seed", &ClashEnv::seed)` binding, and an optional forward from
`MicroRoyaleEnv.reset(seed=...)` — which is where a caller already expects it.

**Blast radius:** additive. Nothing existing calls it, so unseeded behaviour is
byte-identical and no checkpoint is affected. It is not gameplay-affecting, so
`model_weights.pth`'s win-rate history stands.

**Note the one subtlety:** `rng` is seeded in the constructor and
`heuristicOpponent.reset(rng)` is called there too, so a seed applied after
construction must re-reset the opponent or the two fall out of step. Hence the
second line above.

**Confidence:** the cost is measured; the fix is proposed but the exact edit is
the simulator owner's to make. Filed rather than done, per `CLAUDE.md`.

---

## 8. OPEN — Fireball misses the Musketeer kill by 32 HP (proposed 2026-08-06)

**This is filed as a decision to make, not a defect to fix.** It may well be
correct as-is, and closing the gap would trade fidelity for learnability.

### The measurement

`CardRegistry.h` gives Fireball 689 damage at 4 elixir, radius 2.5. Against
`DEFAULT_DECK`:

| target | cost | HP | dies to 689? |
|---|---|---|---|
| Minions | 3 | 230 x3 | yes |
| Archers | 3 | 304 x2 | yes |
| **Musketeer** | **4** | **721** | **no — survives on 32 HP (4.4%)** |
| Cannon | 3 | 824 | no |
| Mini P.E.K.K.A | 4 | 1390 | no |
| Valkyrie | 4 | 1907 | no |
| Giant | 5 | 3968 | no |

So **every clean Fireball kill in this matchup is a 4-elixir spell killing a
3-elixir card** — a -1 elixir trade that also generates no board presence. The
only target that would make it a clean 4-for-4 survives by 32 HP. Breaking even
requires hitting two cards at once.

That fully explains the behaviour recorded in CLAUDE.md: the policy plays
Fireball on ~0% of steps, and forcing it dropped win rate 97% -> 23%. The low
weighting was never a learning failure. It is a correct valuation of a card
that is negative-EV in its common case.

### Why this is not obviously a bug

689 and 721 appear to be the real tournament-standard (level 11) Clash Royale
values, in which Fireball genuinely does not one-shot a Musketeer — it needs any
chip damage on top (a tower hit, a Zap, one arrow volley). **Needs confirming
against current real-game data before anything is changed.** If it holds, the
engine is right and the awkward EV is a real property of the card.

That matters more here than in most engines, because `perception/` exists
specifically to drive this simulator from real matches. A sim where Fireball
one-shots Musketeers is a sim whose spell decisions do not transfer.

### Options

1. **Change nothing.** Correct if the values are faithful. Fireball stays a
   situational two-for-one card, which is what it is in the real game, and the
   agent's low usage is right rather than pathological.
2. **Fireball 689 -> 725.** Makes it a clean 4-for-4. Cheapest edit, but it
   diverges from the real game on the single most-used spell, and every
   Fireball interaction in every future deck inherits the divergence.
3. **Musketeer 721 -> 685.** Same effect, worse blast radius — it changes every
   matchup the Musketeer appears in, not just the Fireball one.

**Recommendation: option 1, and reach the behaviour through reward shaping
instead** (see the lethal-spell PBRS term and the elixir-value term in
`train.py`). Shaping changes what the agent *learns to value* without changing
what the game *is*, which keeps the perception bridge honest.

Blast radius if 2 or 3 is chosen: gameplay-affecting, so it invalidates the
win-rate history, and a `ClashRoyaleTests` run is required. The observation
layout is untouched, so checkpoints still load.

---

## 9. DONE — troop movement ran ~4-5x faster than the real game (applied and verified 2026-08-07)

**Applied with explicit sign-off**, as `MOVEMENT_SPEED_SCALE = 0.2f` in
`CardStats.h`, used at `CardRegistry.h:125` and both `SpiritEmpressForms.h`
sites that assign `speed` directly and so bypass the factory.

### Verified three ways after the change

| check | before | after | target |
|---|---|---|---|
| reconstruction floor (IoU) | 0.519 | **0.685** | higher |
| engine:real speed ratio (p90) | 3.8x | **0.8x** | 1.0 |
| engine:real speed ratio (median) | 6.5x | **1.3x** | 1.0 |
| time-scale optimum @1.0s horizon | 0.2 | **1.0** | 1.0 |

The two speed statistics bracket 1.0 from opposite sides, which is the most
this data can resolve -- the median under-reads real speed and the p90
over-reads it.

**The time-scale peak moving from 0.2 to 1.0 is the decisive end-to-end
check**: it says the engine's clock and the real game's now agree, measured
against footage rather than against a constant.

Two predictions made before the change and confirmed after it: the
reconstruction floor rose on its own (part of it was the single materialising
tick, which at 5x speed displaced a unit by half a second of real movement),
and the Giant -- the Slow tier -- came out at 0.7x against the other classes'
0.8x, which is the documented flat-scale residual showing up exactly where it
was predicted.

### What the C++ test suite did and did NOT tell us

All 504 cases / 4329 assertions pass unchanged. **This is not evidence the
change is correct.** `test_troop.cpp` constructs `MeleeTroop` directly with a
literal speed, and no test anywhere asserts a registry speed constant, so the
suite covers the movement MECHANISM and has zero coverage of the DATA. The
earlier estimate in this document -- that 207 speed/movement references across
16 test files implied the suite would need review -- mistook grep hits for
coverage. A guard now lives on the perception side
(`test_forecast.test_the_engine_moves_at_roughly_real_game_speed`).

### Still open, NOT addressed by this change

  * **Projectile speed** (`Projectile.h:88`) is untouched and unmeasured. The
    harness is blind to spells -- they are filtered out before reaching the
    board -- and the value-Fireball and lethal-spell shaping both depend on
    when a spell arrives relative to the troops it is aimed at.
  * **Deploy time** is still absent; the real game freezes a troop ~1 s after
    it lands. A second timing error in the same direction.
  * The **flat-scale residual** on the Slow tier, ~20%, left deliberately.

### Original evidence, kept for the record

### How it was measured

`perception/tools/sim_fidelity.py`, over all 8 recordings, 159 paired samples.
Both sides of the comparison are measured, neither is read from a constant:

  * **engine** — inject a card into open ground, step, measure the centroid's
    displacement over a long enough baseline that cell quantisation is <10%
    (`forecast.SimForecaster.measure_speed`).
  * **real** — card classes showing exactly one body on one side in both
    frames of a pair, so displacement is unambiguous without a tracker,
    measured over the 3.0 s horizon.

| card | engine tiles/s | real p90 | ratio | n |
|---|---|---|---|---|
| Minions | 8.02 | 1.03 | 7.8x | 6 |
| Spear Goblins | 10.00 | 1.88 | 5.3x | 5 |
| Musketeer | 5.04 | 1.33 | 3.8x | 33 |
| Valkyrie | 5.04 | 1.33 | 3.8x | 23 |
| Giant | 2.88 | 1.00 | 2.9x | 45 |
| Mini P.E.K.K.A | 7.55 | 3.07 | 2.5x | 19 |
| Archers | 5.09 | 3.50 | 1.5x | 17 |

**The honest figure is a bracket, 3.8x-6.5x, not a point.** The two statistics
have opposite biases: the median under-reads real speed because units that
stop to fight contribute zeros, and the p90 over-reads it because a
mis-associated detection looks like a large jump (Archers at 3.50 tiles/s is
not a real Archer). Median ratio is 6.5x, p90 ratio 3.8x. Do not quote a
tighter number than the bracket from this data.

### An independent cross-check that the measurement is real

The engine has an internal speed-tier convention, stated in its own comments
(`SpiritEmpressForms.h:28,43` — "Fast" 0.85, "Medium" 0.5):

| tier | engine/tick | engine tiles/s | real tiles/s |
|---|---|---|---|
| Slow (Giant) | 0.30 | 3.0 | ~0.75 |
| Medium (Musketeer, Valkyrie, Knight) | 0.50 | 5.0 | ~1.0 |
| Fast (Hog, Mini P.E.K.K.A, Minions) | 0.80-0.85 | 8.0 | ~1.5 |
| Very Fast (Skeletons, Goblins) | 1.00 | 10.0 | ~2.0 |

The engine's Slow:Medium ratio is 0.60 where the real game's is 0.75, so the
Slow tier is *relatively* too slow and should need a divisor 0.8x the size of
Medium's. Measured independently from the footage: Giant 2.9x against
Musketeer/Valkyrie 3.8x, a ratio of **0.76**. Two unrelated sources agreeing
to 5% is the reason to believe this is a scale error and not detector noise.

It also means **a single flat divisor is close but not exact.** Correcting per
tier is exact; a flat divisor leaves the Slow tier ~20-25% off.

### What is NOT claimed

  * That this explains "the whole sim-to-real gap". The gap in *performance*
    has never been measured; only this discrepancy in *speed* has.
  * Exact target values. The measurement supports "~4-5x too fast". It does
    not support setting a specific constant to three digits.

### Where the change would go

| site | what it covers |
|---|---|
| `CardStats::troop()` — `CardRegistry.h:125` | every troop built through the factory (the great majority) |
| `SpiritEmpressForms.h:28,43` | **bypasses the factory** and assigns `stats.speed` directly — a scale applied only in `troop()` would silently miss these two |
| `Troop::update` — `Troop.h:36` | the single point of application; scaling here covers everything at once, including any future bypass |
| `Projectile.h:88` | **separate `speed`, separate question — see below** |

`Troop.h:36` is the smallest and most complete edit (one line, covers every
mover, trivially reversible). Its downside is that `CardStats::speed` then
means something other than what it says. Scaling at the two construction
sites keeps the values honest but must touch both, and would be missed again
by the next card that bypasses the factory.

### Blast radius — this is not "the sim gets more accurate"

Attack cooldowns are already correct against the real game (Musketeer 1.0 s,
Valkyrie 1.5 s, Hog 1.6 s — 132 rows agreeing, see CLAUDE.md) and elixir regen
is within 2%. Slowing movement while those stay fixed **rebalances every card
relationship in the engine**:

  * a troop crossing a defender's range takes ~5x longer, so it absorbs ~5x
    more shots — **ranged units get much stronger relative to melee**;
  * a tank takes ~5x more Princess Tower damage covering the same ground;
  * ~5x more elixir accrues while a push develops, which changes the economy
    the whole game is played on;
  * far fewer engagements fit in a match, so **timeouts get much more common**
    and `TimeoutRules`/`DRAW_PENALTY` become far more prominent;
  * `HeuristicOpponent` and the four scripted bots have thresholds that were
    only ever exercised against the fast physics;
  * the curriculum's 0.80 stage gate and the 1.0-1.5 elixir ladder were
    calibrated against the fast physics;
  * 207 references to speed/position/movement across 16 C++ test files
    (`test_troop.cpp` alone has 33) — the Catch2 suite will need review, and
    whether each failure is "asserting the old physics" or a real problem
    needs a human to read.

One genuine upside: `skip_frames = 10` gives one decision per second, which
CLAUDE.md lists as open problem #4 precisely because it caps tactical
precision. At 5x slower movement, one second covers 5x less board, so that
handicap shrinks by the same factor without any change to the action space.

**Every checkpoint is invalidated and the win-rate history means nothing
afterwards.** That is expected and accepted here — the plan is a fresh run.

### Open questions, not answered by this measurement

1. **Projectiles.** `Projectile.h:88` uses its own `speed` and this harness
   cannot see them: spells in flight are filtered out before they reach the
   board (`is_board_presence`). Whether projectile speed carries the same
   error is **unmeasured**. It matters, because the value-Fireball shaping and
   the lethal-spell PBRS term both landed recently and both depend on when a
   spell arrives relative to the troops it is aimed at.
2. **Deploy time.** The engine has none — `spawnEntity` makes an entity live
   immediately, while the real game freezes a troop ~1 s after it lands. This
   is a second, independent timing error in the same direction, and correcting
   speed does not address it.

### The time-scale sweep — a third, independent confirmation

`--time-scale N` steps the engine by `horizon * N`, which is arithmetically
what dividing every speed by `1/N` does to displacement. Sweeping it measures
the divisor with **no engine change at all**. All scales share one pass, so
every column is scored on identical boards and a difference between them
cannot be sampling. n = 158 per cell.

```
 horizon   stale  rebuilt    x0.1   x0.15    x0.2   x0.25   x0.33    x0.5      x1
    0.5s   0.341    0.258   0.258   0.258   0.258   0.258   0.253   0.253   0.172
    1.0s   0.250    0.187   0.187   0.187   0.216   0.216   0.203   0.141   0.114
    2.0s   0.178    0.135   0.138   0.134   0.143   0.152   0.122   0.111   0.091
    3.0s   0.134    0.110   0.094   0.097   0.104   0.105   0.091   0.075   0.060
```

**`x1` is the worst column at every horizon, and the curve has an interior
maximum at 0.2-0.25** on the two rows that can resolve it — i.e. a divisor of
**4x-5x**, agreeing with both the speed table and the tier cross-check.

Two rows cannot resolve it and must not be read as evidence. At `0.5s` every
scale <= 0.25 rounds to a single tick, so those columns ARE the rebuilt board
(hence the identical 0.258); at `3.0s` the differences are inside the noise.

### What this does NOT show: forward prediction is not yet worth deploying

`rebuilt` is the same reconstruction scored WITHOUT stepping. Against it,
stepping at the right scale genuinely adds information — 0.216 vs 0.187 at
1.0 s, 0.152 vs 0.135 at 2.0 s. So the dynamics do carry real signal.

But **no scale beats `stale`**, which pays no reconstruction cost at all
(perception at t vs perception at t+h). The reconstruction tax — floor 0.519 —
is larger than everything stepping buys back. So the decision loop should keep
acting on the freshest real board, and the lookahead idea stays parked.

It is worth re-asking after this item lands: part of that floor is the one
unavoidable materialising tick, which at 5x speed displaces a unit by half a
second of real movement. Correcting speed should raise the floor on its own.

### How to verify a fix, before spending any training compute

`sim_fidelity.py` is the instrument, and it needs no engine change to predict
what the fix will do: `--time-scale N` steps the engine by `horizon * N`,
which is arithmetically what dividing speed by `1/N` does to displacement.

  1. sweep `--time-scale` and find the scale that maximises occupancy
     agreement — that scale IS the empirical divisor;
  2. apply the change;
  3. re-run with `--time-scale 1.0` and confirm the ratio column collapses
     toward 1.0 and the reconstruction floor rises;
  4. only then retrain.

Step 1 costs one 20-minute pass and no engine change at all.

---

## 12. OPEN — bind `isValidPlacement`, so the action-space mask can stop disagreeing with the engine (proposed 2026-08-11)

**The ask is one read-only accessor.** No gameplay change, no checkpoint
invalidation, no behavioural difference to any existing caller. Everything
else in this item is Python-side and is not being asked for here.

### The measurement

`model.py`'s `placement_mask` grants a troop every one of the 16 own-half
rows. `GameManager::isValidPlacement` does not. Measured on the live
pipeline-2 checkpoint at ep ~45,800, over 1,340 decision steps:

| | |
|---|---|
| chose no-op | 72.9% |
| chose a card | 27.1% (363) |
| ...**accepted by the engine** | **41.3%** (150) |
| ...**silently rejected** | **58.7%** (213) |
| of those, affordability leaks | **0** |

`playCard` returns `false` with no exception and no signal, so a rejected
action is indistinguishable from a no-op in its effect and its advantage is
pure noise entering the gradient. This is the same failure class the
affordability mask was introduced to close, still open on the placement axis.
The affordability half is airtight -- zero leaks in 363 draws.

**94.8% of the rejections are on row y=0**, which is `Board::isBackRowDeadZone`
and is deliberate engine design, not a bug. The remaining 5.2% are the tower
footprints.

### It is entirely static, and per card class

Probed cell by cell over the own half:

| | legal cells | row-0 columns |
|---|---|---|
| Cannon (building) | 208/288 | 6, 7, 11 |
| Archers / Giant (troop) | 242/288 | 6-11 |
| Fireball (spell) | 276/288 | 6-11 |

and legality does **not** depend on board state:

| | legal cells |
|---|---|
| empty board | 208/288 |
| six troops deployed | **208/288** |
| cells lost to units | **0** |

So the correct mask is three constant tables, built once at startup. There is
no runtime query and no per-step cost -- which is why this needs an accessor
and not a fast path.

### Why Python cannot just compute it

`isValidPlacement` combines `board.isBackRowDeadZone`, the board bounds, the
per-card `placementRadius`, `isSpell` and `deployAnywhere`, and the tower
footprint clearance in `CardFactories.h`. Reproducing that in Python is a
second copy of engine geometry, which this file already carries two incidents
about (items 1-2, 5-6) and which CLAUDE.md explicitly forbids. `get_card_info`
already exposes the per-card half; the predicate itself is the missing piece.

### The exact edit

`include/core/ClashEnv.h`, beside the existing forwarders at line 393:

```cpp
    // Read-only. Exposed so the Python action space can be built from the
    // ENGINE's legality rule rather than a second copy of it -- see
    // UPSTREAM_REQUESTS item 12. Pure query: no state is touched.
    bool isValidPlacementForCard(int cardId, float x, float y, int team) const {
        const CardDefinition* def = CardRegistry::getInstance().getCard(cardId);
        if (!def) return false;
        return game.isValidPlacement(team, x, y, def->isSpell,
                                     def->placementRadius, def->deployAnywhere);
    }
```

`src/bindings.cpp`, beside the existing accessors at line 73:

```cpp
        .def("is_valid_placement", &ClashEnv::isValidPlacementForCard,
             py::arg("card_id"), py::arg("x"), py::arg("y"), py::arg("team") = 0,
             "Would playCard accept this card at this point? The exact "
             "predicate playCard uses, exposed so the Python placement mask "
             "is derived from it instead of re-deriving board geometry.")
```

`isValidPlacement` is already public and already `const`. Nothing else moves.

### Blast radius

Additive only. No existing symbol changes signature or behaviour, no
gameplay path is touched, and `model_weights.pth`'s win-rate history is
unaffected by the binding itself.

**The Python mask fix that consumes it is a different matter and is
gameplay-affecting**: removing 58.7% of the policy's card choices from the
action space redistributes probability mass onto real plays, so the play rate
and the elixir economy will both shift. Win rates are not comparable across
it. That is expected and is the point, but it should be stated when it lands.

**Not yet established:** that this defect is *why* the agent turtles or why
the Giant is starved. It is a large real defect that was invisible; the causal
claim needs re-measuring after the fix, not before. An earlier diagnosis of
the same behaviour as a reward-hacking elixir dump was measured and refuted --
a rejected placement spends no elixir at all -- and that mistake is the reason
this item leads with the measurement rather than the story.

### Rebuild note

The `.pyd` post-build copy into `python_ai/` fails with MSB3073 while any
Python process has the module loaded, so the live trainer must be stopped for
the copy step. Compilation itself can be verified without stopping it.

---

## 13. OPEN — state snapshot/restore, so decision-time search becomes possible (proposed 2026-08-11)

**What is being asked for:** a way to copy a `GameManager` deeply, so a caller
can try several candidate actions from one position and keep the best. Today a
copy is shallow — `Board` holds `std::vector<std::shared_ptr<Entity>>`, so the
copy shares every entity with the original and stepping one corrupts the other.

### Why this is worth reading despite the blast radius

A fast deterministic simulator is the project's biggest unexploited asset.
Combat has no RNG at all (the only randomness is `PlayerState::initializeDeck`'s
shuffle and `HeuristicOpponent`), so rolling a candidate action forward gives
*exactly* what would happen. That is a strict policy-improvement operator, and
`python_ai/bc_pretrain.py` — already built, schema pinned, verified end to end —
is exactly the consumer needed to distil the result back into the policy.

Two measurements taken 2026-08-11 size it:

| | cost |
|---|---|
| 20-tick (2 s) engine rollout | **0.34 ms** |
| one `MicroRoyaleNet` forward (CPU) | **50.51 ms** |

The engine is **~150x cheaper than the network that evaluates it**. Search here
is not compute-bound on simulation at all: a 10-second rollout of 12 candidates
costs ~20 ms. That asymmetry is what makes lookahead attractive on this
hardware, and it is the opposite of the usual assumption.

Candidates measurably differ. Over 24 constructed mid-game positions, K=12,
10 s horizon:

| | |
|---|---|
| observable damage swing spread (max-min) | mean **508.7** |
| critic value spread (max-min) | mean **0.738** |
| the two scorers pick the same best candidate | **50%** |

A ~500 damage swing is roughly 15% of a Princess Tower. There is real signal to
choose between.

### The correction to the existing plan — `Entity::clone()` ALREADY EXISTS

`CLAUDE.md` records this item as "needs a virtual `Entity::clone()` across the
whole hierarchy — invasive simulation-core surgery". That is out of date.
`include/entities/Entity.h:113` already declares:

```cpp
virtual std::shared_ptr<Entity> clone(int newId) const { (void)newId; return nullptr; }
```

overridden in `MeleeTroop`, `RangedTroop`, `BuildingTargeter` and
`RangedBuildingTargeter`, each as three lines of implicit-copy-constructor:

```cpp
auto copy = std::make_shared<MeleeTroop>(*this);
copy->id = newId;
copy->hp = 1;              // Clone card: full damage, 1 hp
return copy;
```

**So copy-construction of concrete entity types is already relied on in
production code.** The mechanism is proven; only its coverage and semantics are
wrong for snapshotting.

### Why the existing `clone()` must NOT simply be reused

Two reasons, both of which would corrupt a snapshot silently:

1. **`copy->hp = 1`.** It implements the Clone *card*, whose duplicates have 1
   HP by design. A snapshot needs HP preserved exactly.
2. **The default returns `nullptr`, not an error.** Buildings, Towers,
   `AreaSpell` and `Projectile` have no override, because they were never valid
   Clone targets in the real game. A naive "clone every entity" loop would
   therefore produce a board that has **silently dropped every tower, building,
   spell and projectile** — and would look like a working snapshot.

### Proposed shape (the human decides the details)

- a second virtual, e.g. `virtual std::shared_ptr<Entity> snapshot() const`,
  preserving id and hp exactly, implemented for **every concrete type**
- its base implementation should **fail loudly** (assert / throw), never return
  `nullptr`, so a future entity type cannot silently punch a hole in a snapshot
- `Board::deepCopy()` rebuilding `activeEntities`, `pendingEntities` and
  `idCounter`
- `GameManager` snapshot = that board copy plus its scalar members
  (`currentTick`, `gameOver`, `loserTeam`, `oppElixirMultiplier`, `rng`, the
  deck configs, tower-troop types, `stats`, `playerAI`, `playerOpponent`),
  which are already value types

Estimated size: roughly 3 lines per concrete entity type across ~8-12 classes,
plus the board/manager plumbing. That is meaningfully smaller than "a virtual
clone across the whole hierarchy plus effects".

### The risk I could NOT clear from the outside, and it is the crux

`CombatEntity` holds effects as shared pointers — `onHitEffects`, `deathEffect`,
`periodicEffect`, `onDamageTakenEffect`, `onHitSpawnEffect`,
`transformDeathEffect`, `abilityEffect`. An implicit copy constructor copies the
*pointers*, so a snapshot would **share** those effect objects with the
original.

- if effects are stateless strategy objects, sharing is correct and desirable
- if any effect carries mutable per-instance state, stepping the snapshot
  mutates the original, and the corruption is silent

**This needs verifying before anything is built.** I have not read every effect
implementation and will not assert it either way. The same question applies to
whether any entity caches a pointer to its current target across ticks; if so, a
deep copy must remap those or it will alias into the original board.

### The Catch-22, stated plainly rather than papered over

The obvious question is "what does search buy in win rate?" I tried to answer it
and **could not**, and the reason is structural rather than a matter of effort.

Without snapshotting, a search action can only be tested at a decision point
that can be *constructed* (via `inject`), after which both branches are played
out by the policy. That measures the marginal value of **one** improved action
diluted across ~80 subsequent policy actions. Result over 45 disagreement
states:

```
search branch ahead 51%   policy ahead 47%   tie 2%
paired tower-HP delta +0.0145 +/- 0.2162 (95% CI), n=45
```

The confidence interval is **15x wider than the effect**. Observed paired std
0.74 puts the sample size needed at roughly **20,000 constructed states** — and
even then it would only measure a one-decision intervention, not the
every-decision search that expert iteration actually performs.

**This is explicitly NOT evidence that search does not work.** It is an
underpowered null from an experiment that could not have detected the effect it
was looking for. Recording it that way, rather than quoting "51% vs 47%" as
though it meant something, is the point.

So the evidence that would justify the change is unobtainable without the
change. The honest basis for proceeding is architectural priors — the 150:1
compute asymmetry, the measured candidate spread, the determinism guarantee, and
a distillation pipeline that already exists — not a demonstrated win rate. The
human should decide with that stated, not implied.

### Blast radius

Simulation core. Nothing about existing behaviour changes if the new method is
purely additive and nothing calls it — but any bug in it produces *wrong
simulated futures*, which would then be distilled into the policy as if they
were expert labels. That failure mode is silent, which argues for the
loud-failure default above and for a divergence test: snapshot a live game, step
both copies with identical actions for N ticks, and assert the observations stay
bit-identical. `perception/`'s bridge already demonstrates exactly this kind of
zero-divergence control.
