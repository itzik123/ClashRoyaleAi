# Simulator changes requested from `perception/`

Written from `perception/`, which modifies nothing outside itself. Items are
**proposed** here first, with evidence and blast radius, and applied only with
explicit sign-off — see CLAUDE.md's rule on never changing C++ without
confirming the exact diagnosis and the exact edit first.

Last updated 2026-08-24. Items 0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 16,
19, 21, 22, 24 and 25 are applied; 8, 17, 20 and 23 are still open; 18 is ACCEPTED, WILL NOT FIX
FOR NOW. There is no item 11.

> **Second status audit, 2026-08-23.** This line had drifted again, in all four
> possible directions at once: item 3 was listed OPEN while `Tower.h` already
> carries its activation logic, items 19 and 21 were applied and missing from
> the applied list, item 20 was OPEN and missing from the open list entirely,
> and 18 was filed under "open" when its own header says it is accepted and
> deliberately unfixed. A summary that has to be updated by hand alongside the
> header it summarises is a second copy of the same fact -- the exact pattern
> CLAUDE.md warns about for engine constants, and the second time this specific
> line has gone stale. The per-item headers are authoritative; if the two
> disagree, believe the headers and fix this line.

> **Status audit, 2026-08-19.** Items 5, 6 and 12 were carrying `OPEN` headers
> while the exact code they propose was already merged — verified line by line
> against the tree (5: `ClashEnv.h`'s `std::floor((BOARD_HEIGHT - 1) - position.y)`;
> 6: `constexpr int riverRow = 17;`; 12: `ClashEnv::isValidPlacementForCard` plus
> its `.def("is_valid_placement", ...)`). Their statuses are corrected below.
>
> Recording *why* rather than silently flipping them: this document's value is
> that its status field can be trusted, and three items reaching `applied`
> without the propose → sign-off → implement sequence this file exists to
> enforce is worth one line of history. Items 5 and 6 are gameplay-affecting by
> their own text ("opponents get stronger… Elo is therefore not comparable
> across this fix"), and item 12's Python side is flagged the same way — so the
> win-rate history around them should be read with that in mind.

| # | Request | Severity | Status |
|---|---|---|---|
| 0 | River band `[16,18)` → `[15.5,17.5)` | was blocking training | **DONE — verified** |
| 1 | Left Princess towers `x = 3.0` → `4.0` | blocks a stage-0 acceptance target | **DONE — verified** |
| 2 | Kings `x = 8.5` → `9.0` | cosmetic accuracy | **DONE, then REVERSED 2026-08-21 — see item 2** |
| 3 | King Tower has no activation condition | fidelity gap | open, **already worked around, no change needed** |
| 4 | `inject(..., team)` + `get_hand(team)` | convenience | **DONE — already landed 2026-07-29, see below** |
| 5 | Team-1 observation mirrors the truncated row, not the position | **corrupts all self-play** | **DONE — applied (status corrected 2026-08-19)** |
| 6 | River marker row is 17 for team 0 but 16 for team 1 | same class, smaller | **DONE — applied (status corrected 2026-08-19)** |
| 8 | Fireball (689) misses the Musketeer kill (721 HP) by 32 | **fidelity vs learnability — needs a decision, not a fix** | open, proposed 2026-08-06 |
| 7 | No way to seed the engine's RNG (TWO generators, not one) | every A/B test cost ~10x more; invalidated a control 2026-08-20 | **DONE 2026-08-21** |
| 9 | **Troop movement is ~4-5x faster than the real game** | **largest measured sim-to-real gap; miscalibrates every timing the agent learns** | **DONE — applied and verified 2026-08-07** |
| 12 | Bind `isValidPlacement` so the action mask stops disagreeing with the engine | 58.7% of card choices silently rejected | **DONE — `is_valid_placement` is bound in the current `.pyd`** |
| 10 | State-estimator write half: `set_elixir_for_team` / `set_hand_for_team` | search over a reconstructed state scored a fabricated hand/elixir | **DONE — applied 2026-08-17, recorded here 2026-08-19** |
| 13 | **State snapshot/restore, so decision-time search becomes possible** | unblocks the biggest unexploited asset | **DONE — applied and verified 2026-08-11** |
| 14 | Offence structurally under-priced; deploy time added | win condition was unplayable by construction | **DONE — raised and applied 2026-08-19** |
| 18 | Troops deadlock in the concave pocket between two buildings | 7 permanent stalls per 343k unit-ticks; fix means real pathfinding | **ACCEPTED AS-IS 2026-08-20 — will not fix for now** |
| 16 | Bind `TimeoutRules::resolve` so match outcomes have one definition | **8** Python copies, all missing the HP tie-break | **DONE — signed off and applied 2026-08-20** |
| 17 | Const accessors for internal timing state | divergence tests cannot see cooldowns/fuses | open, proposed 2026-08-19, **ergonomics not coverage** |

Items 1 and 2 were done together since the measured benefit is combined
(max error 0.63 → 0.31 tiles) and neither is a large or risky edit.

**There is no item 11.** 10 and 11 were both skipped when the numbering jumped
9 → 12; 10 has since been filled in (retroactively, for the setters), 11 has
not. Not reused, so older references to "item 12"/"item 13" stay valid.

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

> **REVERSED 2026-08-21.** A professional player's audit put the King back at
> **8.5** and the left Princess back at **3.0**, and the reason this fit looked
> good is worth keeping: x is a CELL INDEX clamped to [0, 17], so the board's
> centre -- the fixed point of the mirror `17 - x` -- is 8.5, not 9.0. The
> landmark fit was anchored on that half-tile convention error, which is why its
> residual improved (0.63 -> 0.31 tiles) while the layout became symmetric about
> the WRONG centre. The corrected arena is symmetric about 8.5 and matches the
> real game's river row `WWBBWWWWWWWWWWBBWW`. All of it now lives in
> `include/core/ArenaLayout.h` and is bound to Python, so no consumer keeps a
> copy. See CLAUDE.md's "Board geometry".


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

> **REVERSED 2026-08-21.** A professional player's audit put the King back at
> **8.5** and the left Princess back at **3.0**, and the reason this fit looked
> good is worth keeping: x is a CELL INDEX clamped to [0, 17], so the board's
> centre -- the fixed point of the mirror `17 - x` -- is 8.5, not 9.0. The
> landmark fit was anchored on that half-tile convention error, which is why its
> residual improved (0.63 -> 0.31 tiles) while the layout became symmetric about
> the WRONG centre. The corrected arena is symmetric about 8.5 and matches the
> real game's river row `WWBBWWWWWWWWWWBBWW`. All of it now lives in
> `include/core/ArenaLayout.h` and is bound to Python, so no consumer keeps a
> copy. See CLAUDE.md's "Board geometry".


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

## 3. APPLIED 2026-08-21 — King Tower activation

> **Applied.** `Tower` carries a latching `awake` flag: Princess Towers
> construct awake, the King asleep (only `GameManager::addTower`'s
> `symbol == 'R'` branch builds one). It wakes permanently on any damage or on
> any friendly Princess Tower being destroyed, and while asleep
> `Tower::findTarget` returns `nullptr` so it can neither acquire nor fire. It
> stays targetable and damageable throughout.
>
> Measured with `tools/audit/king_activation_audit.cpp` — lone Hog, identical
> placement, only dormancy varying: **1268 tower damage with the Kings awake,
> 2219 with them dormant**, and the Hog survives to tick 170 instead of 122.
> The 1268 reproduces the figure the 2026-08-20 sight audit recorded.
>
> **This was requested only as a note, explicitly with "no change requested",
> and was implemented anyway** because the 2026-08-21 player audit asked for it
> directly. GAMEPLAY-AFFECTING: every win rate earned before it is historical.
> Checkpoints are unaffected.
>
> The perception-side workarounds below are now unnecessary but harmless, and
> are left in place: excluding the Kings from `divergence` still measures
> perception rather than engine fidelity, which is what that metric is for.

`Tower.h` built the King like any other tower; `GameManager::reset()` gave
it range 7.0 and a 10-tick cooldown. Nothing made it dormant, so it fired
from tick 0. The real King is inert until activated.

`SimDriver.divergence` excludes both Kings and `readers/towers.king_divergence`
reports them separately, so the pipeline's quality metric measures perception
rather than this gap.

Recorded originally only because it is a real behavioural difference that also
affects training: an agent learns that chip damage to the King is punished
immediately, which is not true of the real game.

---

## 14. DONE — offence was structurally under-priced; deploy time added (raised and applied 2026-08-19)

**RESOLVED by adding a 1.0 s deploy time** (`CardStats.h` `DEPLOY_TIME_TICKS = 10`,
set in `CardFactories::applyCardMetadata`, consumed in `CombatEntity::update`).
The human authorised the engine change after reading the investigation below.

**Controlled result.** Same harness, same supported push, ~161 scored states
each, engine the only difference:

| engine | marginal value of a supported push | 95% CI |
|---|---|---|
| deploy time 0 (old) | -73.7 HP | [-349.5, +195.3] |
| deploy time 10 (new) | **+448.5 HP** | [+137.3, +760.1] |

The defence's cost to answer a 4-elixir commitment rose 1.07 -> 2.93 elixir. A
naked win condition correctly got WORSE (1025 -> 343 tower damage in a punish
window, since it now stands inert under tower fire) while an escorted push holds
at 993.8 -- which is how real Clash prices those two plays.

546 C++ cases pass. The original investigation is kept below unchanged, because
the reasoning that selected deploy time out of several candidate deviations is
the part worth re-reading if this is ever revisited.

### Original investigation (2026-08-19), kept for the record

**No change is being requested yet.** This is a measured observation with a
diagnosis I cannot complete from the Python side, written up per CLAUDE.md's
rule that engine changes need the exact diagnosis first. It is deliberately NOT
a proposed edit: every candidate fix here is gameplay-affecting and would
invalidate the win-rate history of every checkpoint.

### What was measured

The 2026-08-19 curriculum pivot replaced the opponent-elixir-multiplier ladder
with a competence ladder at a symmetric 1.0x economy, on the hypothesis
(`DECISIONS.md`, "The 1.5x Curriculum Overfitting Hypothesis") that a permanent
multiplier is what priced the win condition at zero. The falsifier was run with
NO network on either side — both players are the deterministic
`python_ai/opponents/teacher.py` — so the historical confound between "the environment
prices this badly" and "this net cannot execute it" is removed.

**The hypothesis was not confirmed.** At a symmetric 1.0x economy, committing
the win condition is still strongly net-negative:

| measurement | result |
|---|---|
| attack vs cycle, win rate (n=100 paired) | **−0.330**, p=5.7e-08 |
| ...enemy tower damage dealt | 6326 vs 6938 (attack is not even ahead) |
| ...our tower damage taken | **5772.7 vs 3049.3** |
| never playing it at all (`ban` arm) | **0.910** — best of the three |
| marginal value of one commitment (n=220) | **−298.2 tower HP**, CI [−494, −107] |

### Why it is probably not the card, the King, or the timing

* **Not the card.** A lone Hog injected on an empty board deals **2536 tower
  damage** — a full Princess Tower — and dies at tick 190. (This also means
  CLAUDE.md's older "317 tower damage in 40 s" no longer reproduces.)
* **Not item 3 above.** The enemy King firing from tick 0 does reach a Hog
  attacking either Princess (distance 6.1 against its 7.0 range), but the
  unopposed number above already includes that and the Hog still takes the
  tower.
* **Not the answer's availability.** The defender answers with Skeletons (35 of
  40 trials) and Ice Spirit (31 of 40) — a 1-for-4 trade in its favour, which is
  REAL Clash. Forcing both out of its hand moved the Hog only 634.0 → 665.7.
* **Not timing, and this is the surprising one.** Tightening the commit gate
  makes it monotonically worse below ~3 elixir: −298.2 at ≤7.0, −268.6 at ≤3.0,
  **−585.4 at ≤1.5** (p=9.2e-06). Naturally-occurring low opponent elixir means
  they JUST SPENT, so their push is already on the board — the opposite of a
  punish window.

Against that, the punish mechanic itself demonstrably works when constructed
artificially: forcing the defender's bar to 1.0 with nothing on the board is
worth **+391 tower HP (+62%)**, 256 vs 158 hp per elixir committed.

### The question for the engine owner

**Does a defender in this engine recover its tempo faster than the real game
allows?** The pattern — punish pays when constructed, never occurs naturally,
and defence answers a 4-cost commitment for ~1.2 elixir while conceding ~634 HP
— is what you would see if the defending side can re-establish a threat sooner
than a real opponent could. Two known deviations already recorded in CLAUDE.md
point the same direction and neither has been measured for this effect:

1. **No deploy time.** The engine spawns a troop active; the real game freezes
   it ~1 s after it lands. That 1 s is paid by the DEFENDER in the real game
   (their answer arrives late), so removing it is a systematic subsidy to
   defence — and it applies on every defensive placement, i.e. far more often
   than on the occasional attack.
2. **`Projectile.h:88` has its own untouched `speed`**, never recalibrated
   alongside the 2026-08-07 `MOVEMENT_SPEED_SCALE` fix.

### What I was NOT asking for (at the time)

No edit. Specifically not a reward-side or curriculum-side fix: **four Hog
mechanisms have already been built and measured null** (a reward multiplier, an
advisor target, random forcing, gate-timed smart forcing), and this pivot is the
fifth thing that did not move it. Adding a sixth on the policy side would repeat
a pattern this project has already paid for five times.

The cheap next step, if it is wanted, is to measure deploy time in isolation:
add a spawn delay behind a flag, default off, and re-run
`python_ai/eval/prove_environment.py --mode marginal`. That is a gameplay-affecting
change and would invalidate every win rate, so it is the human's call.

Harnesses, all new and all read-only against the engine:
`python_ai/eval/prove_environment.py` (win-rate arms + marginal value),
`python_ai/eval/prove_wincon_trade.py` (elixir trade, supported push, punish window),
`python_ai/eval/prove_teacher.py` (teacher strength bars).

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

## 5. DONE — team 1's observation is displaced one row (proposed 2026-07-31, applied; status corrected 2026-08-19)

> Applied. `ClashEnv::extractObservationForTeam` now reads
> `static_cast<int>(std::floor((BOARD_HEIGHT - 1) - entity->position.y))` — the
> proposed edit below, verbatim. The header said `OPEN` until 2026-08-19; see
> the status audit at the top of this file. Gameplay-affecting: Elo and win
> rates are not comparable across it.

Original proposal, kept for the record:


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

## 6. DONE — river marker row differs between perspectives (applied; status corrected 2026-08-19)

> Applied. `ClashEnv.h` now reads `constexpr int riverRow = 17;` for both teams
> — the proposed edit below, verbatim. The header said `OPEN` until 2026-08-19;
> see the status audit at the top of this file.

Original proposal, kept for the record:


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

## 7. DONE — the engine can be seeded (proposed 2026-07-31, applied and verified 2026-08-21)

**Not a correctness bug. A cost multiplier on every experiment this project
runs**, including the ones `CLAUDE.md` already recommends re-running.

> **⚠ 2026-08-21: THE EDIT ORIGINALLY PROPOSED BELOW WOULD NOT HAVE FIXED THE
> OPENING HAND.** It seeds `ClashEnv::rng`, which feeds only
> `HeuristicOpponent`. The opening-hand shuffle runs on `GameManager::rng` — a
> **second, independent** `std::mt19937`, also seeded from `std::random_device`
> at construction. Anyone applying the one-liner and then testing two envs for
> an identical opening hand would have found it still random, and the natural
> conclusion ("seeding doesn't work") would have been wrong. The corrected edit
> is in "The change" below and covers both generators.

### What is there now

TWO independent unseeded generators, neither reachable from Python:

```cpp
// include/core/ClashEnv.h:132 / :383   -- feeds HeuristicOpponent only
std::mt19937 rng;
... rng(std::random_device{}()) { heuristicOpponent.reset(rng); }

// include/core/GameManager.h:50 / :202 -- feeds the OPENING HAND
std::mt19937 rng;
... : gameOver(false), loserTeam(-1), rng(std::random_device{}()) {
```

`GameManager::reset()` (line 589-590) is where the hand is dealt:

```cpp
playerAI.initializeDeck(aiDeckConfig, rng);
playerOpponent.initializeDeck(oppDeckConfig, rng);
```

and `PlayerState::initializeDeck` shuffles a permutation of the 8 deck indices
with `std::shuffle(order.begin(), order.end(), rng)`, taking the first four as
the hand and **the rest as the starting `deckQueue` order**. So the seed governs
both the opening hand *and* the cycle order — which for 2.6 Hog Cycle is the
more important half.

`MicroRoyaleEnv.reset(seed=...)` looks like it should help but only forwards to
`gymnasium.Env.reset`, which seeds the *wrapper's* RNG, not either engine one.

`grep -c seed src/bindings.cpp` finds two hits, both in comments. There is no
binding.

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

### NEW EVIDENCE, 2026-08-20: it invalidated a control and nearly produced a wrong conclusion

`env.snapshot()` (2026-08-11) removed this as the blocker on *paired* A/B tests
**within a single process**, and `CLAUDE.md` recorded that correctly. What it
does NOT give is comparability **across process invocations**, and that gap has
now cost something concrete.

A combo-family ablation in `python_ai/eval/prove_combos.py` was designed with a
built-in validity check: run 4 shares `--seed 300` with run 3, so its untreated
OFF arm should reproduce run 3's OFF arm exactly. It read **0.475 against
0.537**. Nothing was wrong with either run — `--seed` reaches only the
*teachers'* `numpy` RNG, so the two invocations drew entirely different match
populations, and the arm levels were never comparable in the first place.

Two runs had also happened to report the same OFF arm (0.537 twice, at different
seeds), which made the design look sound until it was tested. **The failure mode
is that a mis-specified control looks like a failed comparison.**

Consequences carried in `CLAUDE.md` and the harness docstring: within a run the
snapshot pairing is sound and the paired delta is valid; across runs only
**deltas** are comparable, never arm levels. Any "re-run at the same seed and
check the baseline matches" design in this repo is invalid until this lands.

### Second benefit: reproducible failures

A self-play regression currently cannot be replayed. The 2026-07-31 team-1
observation bug was found by running a policy against a bit-exact copy of itself
and noticing 0.598 where 0.500 was expected — a test `CLAUDE.md` now recommends
after any change to the observation, the board, or `stepSelfPlay`. That test is a
coin-flip null measured over hundreds of games precisely because individual games
cannot be reproduced.

Also live: `python_ai/tests` has a nondeterministic **skip count** (347-348 pass,
1-2 skip across identical runs) because two cases depend on this shuffle.

### The change — CORRECTED, both generators

```cpp
// include/core/GameManager.h  (public)
void seed(unsigned int s) { rng.seed(s); }

// include/core/ClashEnv.h     (public)
// Seeds BOTH generators and re-deals, so the opening hand and cycle order are
// pinned as well as the heuristic's rolls. reset() is what calls
// initializeDeck, so seeding without it would leave the CURRENT hand untouched
// and only affect the next episode -- the surprising half of this API.
void seed(unsigned int s) {
    rng.seed(s);
    heuristicOpponent.reset(rng);
    game.seed(s ^ 0x9E3779B9u);
    reset();
}
```

The `0x9E3779B9` offset keeps the two streams from being identical, which
matters because both are `std::mt19937` and one of them shuffling first would
otherwise correlate the heuristic's choices with the hand.

Plus the binding:

```cpp
.def("seed", &ClashEnv::seed, py::arg("seed"))
```

and an optional forward from `MicroRoyaleEnv.reset(seed=...)`, which is where a
caller already expects it.

**The ordering subtlety is the part to get right.** `initializeDeck` runs inside
`GameManager::reset()`, which the `GameManager` *constructor* also calls. So a
seed applied after construction only takes effect on the next `reset()` — hence
the explicit `reset()` in the edit above. A `seed()` that did not re-deal would
look like it silently did nothing.

### Acceptance test, already written

`python_ai/tests/test_engine_seeding.py::test_two_envs_with_the_same_seed_deal_the_same_opening`
is committed and **skips** with a clear reason while `seed` is unbound. It
asserts that two envs seeded alike produce identical hands for BOTH teams and
identical cycle order over a full rotation, and that two different seeds
actually differ (so it cannot pass vacuously against a degenerate shuffle).
Rebuild the `.pyd` and it runs.

**Blast radius:** additive. Nothing existing calls it, so unseeded behaviour is
byte-identical and no checkpoint is affected. Not gameplay-affecting, so
`model_weights.pth`'s win-rate history stands.

**APPLIED AND VERIFIED 2026-08-21.** Both generators are seeded, `ClashEnv::seed`
re-deals, and the binding is live. `test_engine_seeding.py` went from 4 skips to
**5 passed / 1 skipped**, the remaining skip being the diagnosis test that
retires itself once `seed()` exists. Full suites after the change: Python
**364 passed / 3 skipped**, C++ **550 cases, 5,341 assertions, all passing**.

The two shuffle-dependent Python tests that used to make the skip count
nondeterministic can now be pinned; that is a follow-up, not part of this item.

*Historical note on why this sat open so long:* the sessions that filed it
believed this machine had no C++ toolchain. It does — VS 2022 Community, just
not on PATH. See CLAUDE.md's environment section for how that error was made.

**Original confidence note, kept:** the cost is measured twice; the exact edit
was filed rather than applied, per `CLAUDE.md` (no `cl`/`cmake`/`msbuild`/`g++`/`clang++`, both Visual
Studio directories empty, WSL not installed), so it could not have been
compiled or tested here even if the rule allowed it.

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

## 10. DONE — state-estimator write half: `set_elixir_for_team` / `set_hand_for_team` (applied 2026-08-17, recorded here 2026-08-19)

**Recorded after the fact.** These landed in `26de409` ("Engine: set_elixir /
set_hand, so search can evaluate the REAL position") without ever being written
up here — the numbering jumped 9 → 12 and items 10 and 11 simply did not exist.
Filling 10 so the file stops implying nothing happened between them. Same class
of gap as the item 5/6/12 status drift noted at the top of this file.

### What landed

```cpp
// ClashEnv.h
void setElixirForTeam(int team, float value) { game.setElixir(team, value); }
bool setHandForTeam(int team, const std::vector<int>& cards) {
    return game.setHand(team, cards);
}
```
```cpp
// bindings.cpp
.def("set_elixir_for_team", ...)
.def("set_hand_for_team",  ...)
```

### Why it was needed

The read half (`inject`, item 4) could rebuild the BOARD, but elixir and the
hand still came from `reset()` — so a reconstructed position carried the right
units and a **fabricated** hand and elixir. `ClashEnv.h`'s own comment names
`perception/forecast.py` as the motivating case. Decision-time search over such
a state scores fiction rather than the real position, which is what made this
the blocking half rather than a convenience.

### `setHandForTeam` REFUSES rather than accepting a misread

It returns `false` and mutates nothing when the size is wrong, when a card is
not in `player.hand + player.deckQueue`, or on a duplicate. **Callers must check
the return.** This matters on real perception output: measured over 2,510 live
frames, ~0.5% off-deck reads and ~1.7% duplicate reads survive match gating, so
the guard fires in practice — it is not theoretical. A silently-ignored refusal
would leave the estimator confidently wrong.

### Current consumer status

`perception/forecast.py` has **not** been migrated to these, deliberately: it
consumes only `game_state.units`, and both of its consumers score tower-excluded
unit occupancy, so the fabricated hand/elixir sit outside every reported metric
today. There is also no team-1 hand source on this side at all (team-1 *elixir*
does have one — `track/opp_elixir.py`, and live, the net's own
`predict_opp_elixir`, which is exactly what ClashEnv's comment anticipates).
Migrate when a consumer starts scoring something hand- or elixir-dependent.

`perception/bridge/sim_driver.py` still reverse-engineers the opening hand by
repeated `reset()` draws — `set_hand_for_team` is the intended replacement for
that ~150-line workaround, not yet applied.

---

## 12. DONE — bind `isValidPlacement`, so the action-space mask can stop disagreeing with the engine (proposed 2026-08-11, applied; status corrected 2026-08-19)

> Applied. `ClashEnv::isValidPlacementForCard` exists and `bindings.cpp`
> registers `.def("is_valid_placement", ...)` — the proposed edit below,
> verbatim. The summary table already said DONE; only this header was stale.
> The Python side that consumes the mask is gameplay-affecting, so win rates
> are not comparable across it.

Original proposal, kept for the record:


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

## 13. DONE — state snapshot/restore, so decision-time search becomes possible (applied and verified 2026-08-11)

**Applied as proposed below**, with one design change found during
implementation. What landed:

| site | what it does |
|---|---|
| `Entity::snapshot()` (`Entity.h`) | new virtual, **throws `std::logic_error`** naming the offending type via `typeid` |
| `Entity::remapSnapshotReferences()` (`Entity.h`) | new virtual, default no-op, second pass over a fresh copy |
| the 8 concrete types | one line each: `make_shared<T>(*this)` |
| `Projectile` | overrides the remap; gains `getTargetId()` for the tests |
| `Board::deepCopy()` (`Board.h`) | the copy itself |

**The design change:** the remap is a **virtual on `Entity`**, not a
`dynamic_pointer_cast<Projectile>` inside `Board::deepCopy`. It has to be —
`Projectile.h` includes `Board.h`, so `Board` **cannot** include `Projectile.h`
to know what a projectile is. Routing it through a virtual also matches the
idiom `Entity` already uses three times (`onDeath`, `clampPosition`,
`onNearbyDeath`) for exactly this "let Board act without knowing the concrete
type" problem, and it keeps `Projectile::target` private — no public setter for
a member that nothing else should ever write.

### A second aliasing case, not in the proposal below, found while implementing

`Board::statsEvents` is a `StatsEventBus` holding `shared_ptr<IStatsObserver>`,
and unlike the effects those collectors are **stateful** — running damage
totals, kill attribution, match outcome. Copying the subscriber list would post
every hypothetical hit in every rollout into the **real match's** statistics,
and those feed the reward shaping, so a search would silently rewrite the
returns it was being scored against. Exactly the `Projectile::target` failure
shape one level up, and it would have been just as invisible.

`deepCopy` starts the copy with an empty bus. A caller that wants stats off a
rollout subscribes its own collectors to the copy. Covered by
`"deepCopy does not carry the stats subscriber list"`.

### Verified

`tests/core/test_board_deepcopy.cpp` — 13 cases, 323 assertions. Full suite
**520 cases / 4756 assertions, 0 warnings** under the existing `-Wall -Wextra`.
The pre-existing 507 cases pass unchanged.

The acceptance test is the zero-divergence control this section asked for:
snapshot a mid-game board (6 towers, both lanes pushing with `DEFAULT_DECK`
cards, projectiles in flight), step original and copy with identical inputs for
120 ticks, and require every entity to match **exactly** — id, hp, team, cardId,
position and projectile target — compared in **vector order**, since
`resolveCollisions` walks `activeEntities` as an ordered `i<j` loop and two
boards holding the same entities in a different order drift apart on their own.

**The negative case reproduces the corruption rather than describing it.** A
snapshot-only copy (no remap) with a shot in flight is stepped, and the
**original** board's troop is the one that loses 250 hp while the copy's is
untouched — the live game damaged by a simulation nobody stepped it in.

**Mutation-tested, so the suite is known to have teeth:** with the remap pass
commented out, **5 of the 13 cases fail**, including the 120-tick divergence
test and "stepping a deep copy leaves the original untouched". Restored and
re-run green.

**One test was flaky at ~1 run in 20 and the flake was mine, not the
engine's.** "Two snapshots given different actions diverge from each other"
stepped both branches 100 times and required the entity lists to differ. They
genuinely RECONVERGE: a lone troop with no support walks into the enemy
Princess Towers, dies without landing a hit, and elixir re-caps at 10, so both
boards end up holding the same six full-health towers and nothing else. The
assertion was false about Clash Royale, not about the snapshot. Fixing it
surfaced a second one immediately -- `playCard` queues into `pendingEntities`
and `getEntities()` exposes only `activeEntities`, so a successful play is
invisible until the next `step()` commits it, and comparing before that shows
two identical boards. Now asserted over a short horizon after one commit step,
and **verified across 200 consecutive full-suite runs**.

Worth recording because a single green run would have shipped both: a suite run
once is not a suite that passes, and "the futures diverge forever" is the kind
of assumption that is obviously wrong once stated and invisible until it flakes.

### Blast radius, as measured rather than estimated

**Zero deletions in every C++ file** — `git diff --numstat` reports `+82/-0`,
`+66/-0`, `+48/-0` and so on across all ten headers. Not one existing line was
modified, so no existing code path can behave differently. **Not
gameplay-affecting: `model_weights.pth`'s win-rate history stands, and no
checkpoint is invalidated.** The rebuilt `.pyd` reports `observation_size()`
13606, unchanged, and plays a full match normally.

### The manager/env layer — also landed, same day

`GameManager::snapshot()` and `ClashEnv::snapshot()`, the latter bound to
Python as **`env.snapshot()`**. `GameManager` copy-constructs (every remaining
member is already a value type — ticks, flags, `rng`, deck configs, both
`PlayerState`s) and then replaces the two members holding `shared_ptr`.

Two implementation notes worth keeping:

**`ClashEnv::snapshot` needs a tagged constructor, not copy-then-fix.**
`GameManager` holds `const float ELIXIR_REGEN_RATE`, so its implicit copy
*assignment* is deleted and `game = other.game.snapshot()` does not compile.
Copy-*initialising* it in the member list works, and in C++17 the prvalue is
elided straight into place.

**The replay logger deliberately does NOT carry across.** `GameLogger`
accumulates a `TickSnapshot` per tick with an `EntitySnapshot` per entity, so by
mid-match it is the largest thing in the object; copying a thousand ticks of
history to simulate twenty is what makes lookahead look infeasible when it is
not. A rollout is also a hypothetical, and its ticks do not belong in a replay
of the real match.

**A third aliasing hazard, and the mirror-image trap next to it.**
`MatchStatistics` holds `shared_ptr` to *stateful* collectors, so the copy must
not share them — same shape as `Projectile::target`, one level up again. But
the obvious fix, calling `attach()` on the copied board, is equally wrong in the
opposite direction: `attach()` builds **fresh zeroed** collectors, and since
train.py's tower term is potential-based over *cumulative* damage, every
candidate would then score as the same enormous instant loss — a search that
looks like it works and never prefers anything. `snapshotFor()` deep-copies each
collector instead. Both failure directions have their own test.

### Measured from Python, on the rebuilt `.pyd`

| | cost |
|---|---|
| `env.snapshot()` | **0.033 ms** |
| one `step()` (10 ticks) | 0.027 ms |
| K=12 candidates at a 2 s horizon | **1.1 ms** |
| one `MicroRoyaleNet` forward | ~50 ms |

So the simulation is free and the **scoring** is the entire budget — the
opposite of the usual assumption, and the reason `python_ai/eval/search_ab_test.py`
batches all K candidate evaluations into a single forward.

### This partly supersedes item 7 (RNG seeding)

Item 7 asks for `seed()` because unpaired A/Bs need ~1,568 episodes per arm.
`snapshot()` delivers the pairing directly: reset once, snapshot, and hand both
arms a bit-identical opening — same shuffled hand, same heuristic lane. Item 7
is still worth having for reproducible *failures*, but it is no longer what
blocks paired experiments.

### The Catch-22 below is now RESOLVED, and the answer was yes

This section originally recorded that the evidence justifying snapshotting was
unobtainable without snapshotting: the one attempt to size search's value gave
`+0.0145 +/- 0.2162` over 45 constructed states, a confidence interval **15x
wider than the effect**, and the honest reading was "underpowered null", not
"search does not work".

With the mechanism built, the same question answered cleanly.
`python_ai/eval/search_ab_test.py`, 160 **paired** trials (both arms handed a
bit-exact copy of one reset), 1.5x opponent elixir, ep-64k checkpoint,
K ~= 3 candidates at a 4 s horizon, scored by the network's own critic:

| | |
|---|---|
| greedy policy win rate | **0.625** |
| + 1-ply search | **0.944** |
| paired delta | **+0.319**, 95% CI [+0.237, +0.401] |
| discordant pairs | 56 search-better / 5 search-worse / 99 tied |
| exact McNemar | **p = 5.6e-12** |
| deviation rate | 13.8% (5,764 of 41,792 decisions) |
| cost | 2.2x wall clock |

Three things about *why* this worked where the earlier attempt could not:

  * **Pairing, which is what snapshot() bought.** The earlier design compared
    one improved action diluted across ~80 subsequent policy actions. This one
    compares whole episodes that share an opening, so search acts at every
    decision and the shared variance is removed rather than averaged over.
  * **An opponent with headroom.** At 1.0x elixir both arms win ~100% and the
    delta is exactly zero -- by ceiling, not by search being useless. Measured,
    4/4 trials at 1.000 vs 1.000, before switching to 1.5x.
  * **Greedy is candidate 0**, so search deviates only when the critic prefers
    something else, and any loss it takes is the critic being wrong.

**The interpretation matters more than the number.** The scorer is the same
network's critic, so search beating the network's own action head by 32 points
says the **value head is much better than the action head is at exploiting it**.
That is exactly the gap expert iteration exists to close, and it locates the
underfit in the policy head rather than the critic.

**Still not established:** one checkpoint, one deck, one opponent, one horizon;
nothing about neural opponents in the PFSP league; and nothing about whether
distillation back into the policy works, which is the only experiment that
would change training.

---

## 13 (original proposal, kept for the record) — state snapshot/restore (proposed 2026-08-11)

**What is being asked for:** a way to copy a `GameManager` deeply, so a caller
can try several candidate actions from one position and keep the best. Today a
copy is shallow — `Board` holds `std::vector<std::shared_ptr<Entity>>`, so the
copy shares every entity with the original and stepping one corrupts the other.

### Why this is worth reading despite the blast radius

A fast deterministic simulator is the project's biggest unexploited asset.
Combat has no RNG at all (the only randomness is `PlayerState::initializeDeck`'s
shuffle and `HeuristicOpponent`), so rolling a candidate action forward gives
*exactly* what would happen. That is a strict policy-improvement operator, and
`python_ai/trainers/bc_pretrain.py` — already built, schema pinned, verified end to end —
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

**VERIFIED 2026-08-11 — effects are safe to share, and one real aliasing case
exists elsewhere.**

*Effects: cleared, and by the type system rather than by inspection.* All five
effect interfaces declare their entry point `const`:

```cpp
IOnHitEffect::apply(std::shared_ptr<CombatEntity>) const = 0;
IDeathEffect::apply(Board&, const Vector2D&, int) const = 0;
IPeriodicEffect::apply(Board&, const Vector2D&, int) const = 0;
IOnDamageTakenEffect::apply(CombatEntity&) const = 0;
IAbilityEffect::apply(Board&, CombatEntity&) const = 0;
```

and a search across every file defining or using those interfaces finds **zero
occurrences of `mutable` and zero of `const_cast`**. Concrete effects
(`FreezeOnHit`, `PoisonOnHit`, `CurseOnHit`, and the ~20 in `include/core/`)
carry only construction-time parameters — `ticks`, `slowFactor`,
`damagePerTick`. They are stateless strategy objects, so sharing them across a
snapshot is correct, and a deep copy of them would be wasted work.

*Target caching: one real case, and it is the dangerous kind.*
`Projectile.h:16` holds

```cpp
std::weak_ptr<Entity> target;      // a MEMBER, persists across ticks
```

Every other `shared_ptr<Entity>` in the hierarchy — `CombatEntity.h:836`,
`CombatEntity.h:1108/1110`, `BuildingTargeter.h:31/33` — is a **local** inside
`findTarget`/`resolveCurrentTarget`, recomputed per tick, and therefore harmless.

`Projectile::target` is not. An implicit copy carries the pointer verbatim, so a
projectile in flight inside a snapshot would home on, and deal damage to, an
entity in the **original live board**. That is silent cross-simulation
corruption of exactly the kind this section was written to catch — the search
would be quietly damaging the real game it is supposed to be reasoning about.

So the deep copy must build an `old entity id -> new entity` map and remap
`Projectile::target` through it. Scope: one member, one class, plus the map the
board copy already has to build. Being a `weak_ptr` it will not keep the
original alive, so the failure would be wrong damage rather than a leak — which
is worse, because it is invisible.

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

---

## 16. DONE — bind `TimeoutRules::resolve`, so match outcomes have one definition (proposed 2026-08-19, signed off and applied 2026-08-20)

> **Applied.** `ClashEnv::resolveTimeoutOutcome()` calls
> `TimeoutRules::resolve(game.getBoard()).loserTeam` and is bound as
> `resolve_timeout_outcome`. Exactly the edit proposed below.
>
> **Why the count rose to eight before this landed.** Between the proposal and
> the sign-off, the same tower-count-only scorer appeared in three MORE new
> files (`eval/prove_combos.py`, `eval/prove_teacher.py`,
> `eval/prove_environment.py`), on top of the five already fixed. Fixing
> instances demonstrably does not hold when the wrong version is four lines
> long and the right one is unreachable from Python.
>
> **`python_ai/eval/match_outcome.py` keeps its hand-written mirror as a
> FALLBACK, on purpose.** The binding lives in a compiled `.pyd`, and this repo
> is worked from at least one machine that cannot rebuild it (see CLAUDE.md's
> Machine A / Machine B box). It feature-detects with `hasattr` so a checkout
> whose `.pyd` predates the binding degrades to the mirror instead of raising
> `AttributeError` mid-evaluation, after the match is already played. Verified:
> both paths return identical verdicts on all six rule cases, including the
> 3-3-towers / 1200-vs-90 case every broken scorer called a draw.
>
> **Retire the mirror** once every environment is known to carry a fresh
> binary: drop `_USE_BINDING` and the fallback branch, and delete
> `python_ai/tests/test_match_outcome_is_the_only_scorer.py`, which exists only
> because the mirror is hand-written.
>
> **Also added: `tests/core/test_timeout_rules.cpp`.** TimeoutRules had NO
> tests at all despite deciding every timed-out match. Now covers all three
> rules, their precedence (count outranks HP), weakest-vs-total, Towers-only
> (a Cannon is not a crown), dead-tower exclusion, and the new accessor.

Original proposal, kept for the record:

**The ask is one read-only accessor.** No gameplay change, no observation or
action-space change, no checkpoint invalidation, no behavioural difference to
any existing caller.

### The problem

`include/core/TimeoutRules.h` is the engine's definition of who won: tower count
first, then the weakest surviving tower's HP, and only an exact tie is a draw.
It has exactly one call site, `ClashEnv::calculateReward`, and it is not
reachable from Python by any other route.

So every Python script that wants an outcome without going through `reward`
re-derives one. **Five did, and all five implemented only the first of the three
rules:**

| file | what it decides |
|---|---|
| `python_ai/eval/net_ab.py` | greedy-episode win rate |
| `python_ai/eval/net_h2h.py` | head-to-head duel score |
| `python_ai/eval/net_h2h_search.py` (×2) | search leaf value, and the ship/no-ship duel number |
| `python_ai/tools/validate_pipeline.py` | the side-asymmetry regression check |

Each read `get_towers_alive(0/1)` and returned a draw whenever the counts
matched. A match ending 3–3 on towers but 1200 HP against 90 HP on the weakest
is a clear win by the engine's own rules, and all five called it a draw — in the
scripts whose entire output is a win rate.

Fixed on the Python side in the same commit as this proposal
(`python_ai/eval/match_outcome.py`), by reading the six tower-HP scalars the
observation already carries and reapplying the rule by hand. That works, and it
is tested — but it is a **hand-written mirror of engine logic**, which is the
exact pattern CLAUDE.md records going stale twice before (`model.py`'s
"18*16=288", `calibrate.py` scoring bridges against `y=17.0` after the river
moved). If `TimeoutRules` ever gains a fourth rule, nothing makes that file
follow.

### The exact edit

```cpp
// ClashEnv.h -- purely additive.
// -1 = draw, 0 = team 0 lost, 1 = team 1 lost. Same convention as
// MatchRules::Outcome::loserTeam, which is what TimeoutRules already returns.
int resolveTimeoutOutcome() const {
    return TimeoutRules::resolve(game.getBoard()).loserTeam;
}
```

```cpp
// bindings.cpp
.def("resolve_timeout_outcome", &ClashEnv::resolveTimeoutOutcome)
```

`TimeoutRules::resolve` is already `static`, already takes `const Board&`, and
already only reads. `ClashEnv` already includes what it needs via `GameManager`.

### What it is worth

`python_ai/eval/match_outcome.py` collapses from a hand-maintained reimplementation
(~50 lines of rule-mirroring plus the float-resolution argument for comparing
normalised HP instead of raw ints) to a pass-through. The five call sites do not
change again. Any future rule change propagates for free instead of silently
not propagating.

### Blast radius

Effectively none. Read-only, additive, no existing symbol changes meaning, and
a rebuilt `.pyd` stays compatible with current weights. The one caveat is the
usual one: it needs a `.pyd` rebuild before the Python side can use it, so
`match_outcome.py` should keep its current implementation as the fallback until
that lands rather than being deleted in the same change.

---

## Lower-priority C++ observations (recorded, not requested)

Surfaced by the 2026-08-19 review of the `dd99991..HEAD` batch. Neither is a
bug today; both are recorded so they are not rediscovered from scratch.

**`Board::getNextWaypoint`'s two branches are mirror images.** The
`isCurrentBelow` and `isCurrentAbove` paths run the same "compute the near bank,
check `distanceTo <= WAYPOINT_ARRIVAL_EPS`, otherwise return it" logic with
`riverY_start`/`riverY_end` swapped. A shared helper would remove the risk of
the epsilon check being fixed on one side and not the other — the same drift
this file's own history is full of. Not requested: it is a refactor of live
pathing code, and the current version is correct and tested.

**`Entity::snapshot()` depends on an unenforced "effects are stateless"
invariant.** `Entity.h` documents that `onHitEffects`/`deathEffect`/
`periodicEffect`/`abilityEffect` are deliberately *shared* rather than deep-copied
between an entity and its snapshot, because every effect interface declares
`apply() const` and none holds mutable state. That is true of the ~30
implementations today, but nothing enforces it — not a `const`-only member type,
not a `static_assert`, not a test. A future effect using a `mutable` counter
("every 3rd hit stuns") would let a *hypothetical* search rollout mutate state
the *live* match reads back, which is precisely the failure class
`test_board_deepcopy.cpp` exists to prevent and the one vector it does not cover.
Worth a comment at minimum; a fixture card with a stateful effect would turn it
into a real test.

---

## 17. OPEN — const accessors for internal timing state, so divergence tests can see it (proposed 2026-08-19)

**Ergonomics, not a coverage unblocker.** Read that sentence before deciding:
everything below is testable *today* through seams that already exist, so this
buys earlier and better-localised failure messages, not new capability. Filed
because the alternative — behavioural proxies — is what the current tests use,
and one of them is genuinely lossy (see "the one real gap").

### The ask

Const getters for fields that have no public reader, on the model of
`Projectile::getTargetId()` (`Projectile.h:113`), which was added verbatim
"so the deepCopy divergence tests can assert the remap actually happened,
instead of inferring it from damage landing in the right place":

| type | fields |
|---|---|
| `CombatEntity` | `currentCooldown`, `currentTargetId`, `ticksOnTarget`, `ticksSinceLastHit`, `currentHitCount` |
| `Building` | `ticksAlive` |
| `AreaSpell` | `delayTicks`, `remainingHits`, `tickInterval` |
| `Projectile` | `returnDelayTicks`, `outboundHitLanded` |

### Why

`tests/core/test_board_deepcopy.cpp`'s `EntityRow` compares only externally
observable state — id, hp, team, cardId, x, y, projectileTargetId. A copy that
diverged **only** in internal timing would pass its 120-tick window until the
difference happened to surface as an hp or position change. Since a search
rollout's whole job is to predict the next second or two accurately, a
timing-only desync is exactly the defect class that matters and exactly the one
the acceptance test cannot currently name.

### The one real gap, and the honest limit of the rest

Most of these ARE reachable behaviourally, which is why this is filed as
ergonomics:

- `currentCooldown` — `CombatEntity::seedCooldown(int)` is already **public**
  (`CombatEntity.h:654`) and its own comment calls it "the one seam". Seed it,
  copy, step both, assert the first hit lands on the same tick.
- `Building::ticksAlive` — decay fires on `ticksAlive % 10 == 0`, so stepping to
  a tick that is NOT a multiple of 10 before copying makes a reset detectable.
  (Worth noting: the existing `WARMUP_TICKS = 60` IS a multiple of 10, so a
  reset would currently stay phase-aligned and be invisible.)
- `AreaSpell::delayTicks` / `remainingHits` — copy mid-fuse or mid-volley and
  compare hp trajectories.
- `Projectile::outboundHitLanded` — snapshot an Executioner axe after the
  outbound hit and assert the return lands on the same tick.

**`ticksOnTarget` is the exception and is the strongest argument here.** Its only
public proxy is `getDamagePerTick()`, which collapses it into at most four ramp
buckets via `getCurrentDamage()` — so a behavioural test detects a *stage*
desync, never a *tick* desync. `getCurrentDamage()` also folds in
`rangeFalloff`/`rangeBand` through `lastAttackDistance`, so on a falloff card the
proxy varies continuously and the ramp stage stops being separable at all.

### Blast radius

None. Additive `const` getters returning by value; no existing symbol changes
meaning, no field becomes writable, no gameplay path is touched. **Not
gameplay-affecting, so `model_weights.pth`'s win-rate history is unaffected** —
same framing item 13 used for its own additive surface.

### What was done instead, pending a decision

`tests/core/test_snapshot_champion_state.cpp` (new) covers the
`championSlots` ↔ `Board` cross-structure invariant, which needed no engine
change at all. `lastAttackDistance` is already public and varies at runtime in
the existing fixture, so it is the one cheap non-vacuous addition to `EntityRow`
available without this request.

---

## 18. ACCEPTED, WILL NOT FIX FOR NOW — troops deadlock in the concave pocket between two buildings (proposed 2026-08-20)

> **DECISION, 2026-08-20.** Signed off as a known, accepted defect: the
> architectural assessment below is agreed (a reactive steering model always has
> local minima in concave pockets), and global path planning is judged too
> expensive for rollout throughput at present. The defect is rare
> (7 / 342,563 unit-ticks), isolated to a player's own back corner, and never
> touches a bridge. It stays pinned as `[!shouldfail]`.
>
> **This is a deferral, not a dismissal.** Reopen it if any of these change: the
> stall rate rises, a stall is ever observed on or near a bridge, or rollout
> throughput stops being the binding constraint on path planning.

**Severity:** low frequency, permanent per occurrence. **Blast radius of the
proposed fix: the movement core, every unit, every match — which is exactly why
it is here rather than applied.**

Found during the bridge-navigation audit. It is NOT a bridge bug and it is not
the bug that audit was opened for (that one — the bridge-EXIT absorbing state in
`Board::getNextWaypoint` — is fixed and verified; see CLAUDE.md).

### The measurement

`tools/audit/soak.cpp`, 60 randomized full matches, 308,464 unit-ticks watched.
A stall is counted only when a unit is alive, past deploy time, and stationary
for 50+ consecutive ticks *during every one of which nothing was inside its own
effective attack reach*.

| | |
|---|---|
| stalls | **7** |
| ...on or beside a bridge | **0** |
| longest observed | **~490 ticks**, i.e. until the match ended |
| rate | 7 per **342,563** unit-ticks (0.002%) |

(Measured at 4 per 308,464 before the sight/attack fix landed in the same audit;
that change alters engagement geometry, so the figure is re-quoted against the
current tree rather than carried over. Every one of the 7 is the same shape --
a unit in its OWN back corner, pinched between a friendly tower and either a
second building or the board edge.)

Every one was a unit pinched between two buildings in its own back corner. The
first, dumped in full:

```
Ice Golem  (11.360, 2.939)  walking north toward (14, 15.5)
King Tower ( 9.000, 2.500)  r=2.0 -> minimum separation 2.4, actual 2.401
Cannon     (12.032, 4.169)  r=1.0 -> minimum separation 1.4, actual 1.401
```

It is an **attracting fixed point**, not a knife edge. The trace shows the
approach converging geometrically — y = 2.9071, 2.9244, 2.9323, 2.9359, 2.9376,
2.9385, 2.9389, 2.9391, 2.9392 — so nearby states are pulled in rather than
passing through.

### Mechanism

`Board::pushAwayFrom` adds a small perpendicular slide so a unit travels *around*
an obstacle instead of sticking to it. That works for ONE obstacle. With two, the
slides can point in opposing tangential directions and cancel: the unit steps
toward its waypoint inside `moveTowards`, and the post-move `resolveCollisions`
pass pushes it straight back. Net displacement converges to exactly zero.

The geometry says no local rule can fix it here: the King's and the Cannon's
minimum separations sum to **3.8** while their centres are **3.46** apart, so
there is no route between them at all. Escaping requires a multi-tile detour
around the outside of one of them.

### Two fixes were implemented and MEASURED, and both are rejected

Reported because the negative results are the useful part — they are what turns
"we should nudge stuck units" into "a nudge is not enough".

| attempt | movement over 120 ticks |
|---|---|
| tangential slide, handedness flipped every 15 ticks | **0.027 tiles** — fifteen ticks of progress undone by the next fifteen |
| wall slide along the nearest blocker, side chosen by tangent · desired-direction | **0.000001 tiles**, settling at a NEW fixed point (11.3102, 2.96839) |

Each merely relocated the equilibrium. Both were reverted; the tree contains
neither.

### What would actually fix it

Global path planning instead of purely reactive steering — a flow field or A*
over the 18×34 grid, which is small enough that the cost is negligible next to
the 0.015 ms an engine step already takes. That is a redesign of how every unit
moves, it is **gameplay-affecting**, and it would invalidate the win-rate
history of every checkpoint. It needs a decision, not a patch.

### Pinned meanwhile

`tests/core/test_navigation_wedge.cpp` reproduces it deterministically and is
tagged `[!shouldfail]`: the suite stays green, the defect stays on record and
executable, and the case turns RED the moment somebody fixes it. It also carries
a companion test asserting that unobstructed movement is still exactly
speed-per-tick, which is the guard any future fix has to clear.

---

## 19. APPLIED 2026-08-21 — the observation's bridge marker was not re-centred with the board

> **Applied, the same day it was proposed.** Fixed with the stronger of the two
> options this item offers: `Board::isOnBridge(float x)` is now the single
> definition of "is this column a bridge", called from BOTH `clampToBoard` and
> `extractObservationForTeam`. A shared FORMULA would not have prevented the
> drift -- the bug IS two correct-looking expressions of one question drifting
> apart -- so the two callers share a FUNCTION.
>
> It takes a float so one function serves both: physics passes continuous
> positions, the encoder passes integer cell centres. Cell `i` covers
> `[i-0.5, i+0.5]`, so asking about centre `i` against a seam-centred 2.5
> selects exactly cells 2 and 3.
>
> Verified end to end: both teams' channel 8 now reads `WWBBWWWWWWWWWWBBWW`,
> matching `clampToBoard` column for column. The regression test in
> `tests/core/test_arena_layout.cpp` compares the channel against
> `clampToBoard` rather than against expected columns, so it cannot go stale the
> way the encoder did; `tools/audit/verify_pyd.py` checks the same row on the
> built `.pyd`, since that is the artifact training loads.
>
> Sensitivity was measured rather than assumed
> (`tools/audit/bridge_mask_probe.cpp`): the old predicate differs from the
> physics at 4 of 18 columns, so the new test genuinely fails against it.
>
> **This item's measurement was right and its severity assessment was right.**
> The "zero overlap on the left bridge" finding is exactly what makes it worse
> than a half-tile cosmetic drift.


**The arena re-centring removed three stale copies of the bridge columns and
left a fourth, inside `extractObservationForTeam` itself.** `ArenaLayout.h` now
owns the geometry and `Board`, `HeuristicOpponent`, `tactics.py` and
`perception/geometry.py` all derive from it. The observation encoder does not.

This is the **second** time channel 8 has been wrong (item 6 was the row; this
is the columns), and the first time it has disagreed with the engine's own
pathing.

### What is there now

`include/core/ClashEnv.h:183`, inside the river/bridge marker loop:

```cpp
constexpr int riverRow = 17;
for (int x = 0; x < BOARD_WIDTH; ++x) {
    if ((x >= 3 && x <= 4) || (x >= 13 && x <= 14)) {
        obs[getIndex(8, riverRow, x)] = 1.0f;
```

Those literals were correct for the old bridge centres 4.0 / 14.0. They are a
hand-truncated copy of `centre ± 1.0`, and nothing ties them to the centre.

### The engine's actual bridges

`ArenaLayout::LEFT_BRIDGE_X = 2.5f`, `RIGHT_BRIDGE_X = 14.5f` (seam-centred, so
`± BRIDGE_HALF_WIDTH = 1.0` spans exactly two tiles), consumed by
`Board::clampToBoard`:

```
walkable   x in [1.5, 3.5]   and  [13.5, 15.5]
i.e. cells      2, 3               14, 15
marker says     3, 4               13, 14
```

### Measured, not inferred

A ground troop (Giant) was injected at each of the 18 columns on the own side
and stepped until it crossed; the columns it was ever observed occupying at
river rows 16/17 were recorded:

```
troops actually cross at columns:  [2, 14]
observation channel 8 marks:       [3, 4] and [13, 14]
```

**The LEFT bridge marker has ZERO overlap with where units actually cross.** It
marks column 4, which is now water, and omits column 2 entirely. The right
bridge is half right — 14 is correct, 13 is water.

(Units funnel to the bridge centre, so they occupy the truncated centre cell
rather than both bridge cells; the point is that 4 and 13 are unreachable and 2
is unmarked.)

### Why this one is worth fixing promptly

Channel 8 is the network's **only** spatial cue for where the bridges are, and
`DEFAULT_DECK` is the 2.6 Hog Cycle, in which bridge placement is the entire win
condition. A net reading this channel is told the left lane crosses at a column
no unit can occupy.

It also silently confounds any measurement taken between the re-centring and the
fix, because `UtilityTeacher` and `tactics.py` read observations by contract
(`perception/tests/test_encoder_matches_engine.py` pins that contract) while the
engine paths on the geometry.

### Proposed edit

Derive the test from `ArenaLayout` rather than adding a fifth literal:

```cpp
constexpr int riverRow = 17;
for (int x = 0; x < BOARD_WIDTH; ++x) {
    const float fx = static_cast<float>(x);
    const bool onBridge =
        (fx >= ArenaLayout::LEFT_BRIDGE_X  - Board::BRIDGE_HALF_WIDTH &&
         fx <= ArenaLayout::LEFT_BRIDGE_X  + Board::BRIDGE_HALF_WIDTH) ||
        (fx >= ArenaLayout::RIGHT_BRIDGE_X - Board::BRIDGE_HALF_WIDTH &&
         fx <= ArenaLayout::RIGHT_BRIDGE_X + Board::BRIDGE_HALF_WIDTH);
    obs[getIndex(8, riverRow, x)] = onBridge ? 1.0f : -1.0f;
}
```

This reproduces `clampToBoard`'s own predicate on cell centres, so the two can
no longer disagree. Cell `i` covers `[i - 0.5, i + 0.5]`, so testing the integer
centre marks cells 2, 3, 14, 15 — the walkable set.

**Alternative worth considering instead:** expose the predicate once on `Board`
(`bool isOnBridge(float x) const`) and call it from both sites. That is the
stronger fix, since `clampToBoard` and the encoder would then share code rather
than share a formula. Slightly larger blast radius.

### Blast radius

**GAMEPLAY-AFFECTING for learning, not for simulation.** No unit moves
differently — `clampToBoard` is untouched. What changes is what the network is
told, on 4 cells of one channel on one row.

**Checkpoints are not architecturally invalidated** (no shape change, `NUM_CHANNELS`
and `observation_size()` unchanged), but any policy that learned bridge
positions from this channel learned them from the wrong columns, so **win rates
earned between the re-centring and this fix are not comparable** to either side.

**No Python change is needed, which was worth checking rather than assuming.**
An earlier draft of this item claimed `perception/geometry.py` and the
round-trip test would have to move with it. Both are already derived:

* `perception/geometry.py` reads `engine.ARENA_LEFT_BRIDGE_X` /
  `ARENA_RIGHT_BRIDGE_X` / `ARENA_BRIDGE_Y` off the binding, with a fallback
  only for when the binding is absent.
* `python_ai/models/perception_encoder.py`'s `base_spatial()` takes the whole
  river/tower plane from `_probe(None)` -- the engine's own fresh-board
  observation -- explicitly so that "a future move propagates instead of
  diverging".

So `perception/tests/test_encoder_matches_engine.py` keeps passing without
edits, and this really is a ONE-SITE fix. The mechanism that was supposed to
prevent this class of drift worked everywhere except in the encoder that
produces the number in the first place.

### Verification

1. Re-run the injection sweep above: the marked columns must equal the columns a
   ground troop can occupy at rows 16/17.
2. `perception/.venv/Scripts/python.exe -m pytest perception/tests -q`.
3. `python_ai/venv/Scripts/python.exe -m pytest python_ai/tests -q`.
4. The Catch2 suite (582 cases, exactly one `[!shouldfail]`).

---

## 20. OPEN — spawned entities carry no `cardId`, so a replay cannot name them

**Found 2026-08-21 while fixing `web/viewer.html`'s entity inspector.** The
viewer-side half is fixed and shipped; this item is the engine-side half, which
is a C++ change and therefore a proposal rather than an edit.

### What is there now

`Entity.h:45` declares `int cardId = -1`. `CardFactories::applyCardMetadata`
assigns a real id for a card played from hand, and nothing assigns one for an
entity spawned by another entity — death-spawns (`SpawnOnDeath`,
`SpawnOnDeathForEnemyTeam`), spawner buildings (`PeriodicSpawnEffect`,
`ProximityGatedPeriodicSpawnEffect`, `CappedSpawnOnHitEffect`) and tower troops
(`TowerTroops.h`). Those keep the `-1`.

### Consequence

`GameLogger` writes `cardId` per entity plus an id-keyed `cardMeta` block, so a
replay is self-describing for every card played from hand **and for nothing
else**. A `-1` entity cannot use the id lookup and falls through to the
viewer's symbol-keyed tables, which cover ~45 of the 173 registry entries.

Where that missed, the last resort was the RAW SYMBOL — and `'?'` is
`CardDefinition`'s default symbol (`CardRegistry.h:100`) as well as the explicit
symbol of Cannon Cart (69) and Guards (76). So a card could be displayed to the
user as a literal `?`. That is the reported bug.

### Measured

Built from a live replay's own `cardMeta` (132 playable + 41 Evolution entries):

| | |
|---|---|
| distinct symbols in the registry | **90** |
| symbols shared by more than one card | **36 (40%)** |
| cards whose symbol is `'?'` | Cannon Cart (69), Guards (76) |

So symbol is not a safe key even as a fallback: for a `-1` entity a collision is
not resolvable, which is why the viewer now flags such a resolution as
"identified by symbol" rather than presenting it as certain.

### Blast radius — NOT physics-affecting, but IS reward-affecting

`cardId` is read by stats attribution (`StatsEvents`, `MatchStatistics`) and by
the logger. Nothing in `include/entities/` branches on it to decide movement,
targeting or damage, so propagating it **cannot change a simulation outcome**.

It WOULD change per-card stats attribution: a Skeleton spawned by a Tombstone
would begin attributing its damage to a card rather than to nothing, and
`get_elixir_value_killed_by` / `get_damage_dealt_by_card` both read those — and
those feed reward shaping. That makes it reward-affecting even though it is not
physics-affecting, which is precisely why it is worth deciding deliberately
rather than patching in passing.

### Options

1. **Propagate a `cardId` to spawned entities.** Simplest. Changes stats
   attribution as described above, so every reward-shaped number earned before
   it would be earned under a different attribution.
2. **Add a separate `spawnedByCardId`** and leave `cardId` alone. The logger
   gains one field and the viewer one fallback; stats attribution is untouched.
   **This is the option that fixes the display without touching the reward
   path**, and is the recommendation.
3. **Do nothing.** What shipped: the viewer resolves these by symbol against the
   replay's own `cardMeta` and never renders a bare `?`. Residual defect is the
   40% symbol collision rate above — a `-1` entity on a shared symbol may be
   shown under the wrong name and HP maximum, flagged as inferred.

### Verification if option 2 is taken

1. A replay containing a Tombstone or Witch must show every spawned body with a
   real name in the viewer's inspector, with no "identified by symbol" note.
2. `get_elixir_value_killed_by` totals must be **unchanged** against a
   pre-change run on the same seed — that is the whole point of option 2.
3. The Catch2 suite (619 cases, exactly one `[!shouldfail]`).

---

## 21. APPLIED 2026-08-23 — a self-play step that does not build observations, because teacher rollouts build 1,000,152 of them per 48 episodes and read 51,566 (proposed 2026-08-23)

### What is there now

`ClashEnv::stepSelfPlay` ends with

```cpp
return { extractObservationForTeam(0), extractObservationForTeam(1), totalReward, isDone };
```

so **every call constructs both teams' 13,606-float observation vectors**, whatever
the caller wants. `SelfPlayStepResult`'s fields are bound with `def_readonly`,
which converts to a Python list on ATTRIBUTE ACCESS — so a caller that ignores
the fields pays the full C++ construction and none of the pybind marshalling.

`UtilityTeacher.execute_steps` (`python_ai/opponents/teacher.py:1382`, `:1384`)
is that caller. It rolls a candidate forward in 10-tick chunks and **discards
the returned object every time**:

```python
if self.team == 0:
    s.step_self_play(slot, x, y, oslot, ox, oy, nxt - t)
else:
    s.step_self_play(oslot, ox, oy, slot, x, y, nxt - t)
```

The one observation a rollout actually reads is the single
`s.get_observation_for_team(me)` at the end of `rollout_stats`, for
`positional_advantage`.

### Measured — on the training box, not inferred

`python_ai/tools/profile_training.py --mode sync --episodes 40 --teacher-stage 5`,
i5-13420H, real `.pyd`, 48 episodes, 268.93 s wall:

| span | calls | self s | us/call |
|---|---|---|---|
| `rollout.step` | **500,076** | 40.064 | 80.11 |
| `rollout.obs` | 51,566 | 11.850 | 229.79 |
| `live.snapshot` | 51,566 | 2.015 | 39.07 |
| `live.step` | 7,952 | 0.839 | 105.56 |
| `rollout.info` | 1,682,847 | 4.837 | 2.87 |

Derived from those counts:

* 51,566 snapshots / 7,952 decisions = **6.49 candidate rollouts per decision**
* 500,076 rollout steps / 51,566 rollouts = **9.70 chunks per rollout** (horizon
  100 in 10-tick chunks, minus early game-over breaks)
* every chunk builds TWO observations, so rollouts construct
  **1,000,152 observation vectors** and read **51,566**

**A 19.4 : 1 build-to-read ratio.** `rollout.step` is 14.9% of wall clock in the
sync profile, and the async profile puts the whole environment at 39.6%.

C++-side cost of the pieces, measured with no interpreter in the process
(`tools/audit/engine_profile.cpp`, WSL g++ -O2 — ratios transfer, absolute ms
do not):

```
extractObservationForTeam   0.0106 ms      GameManager::step()  0.0006 ms/tick
stepSelfPlay(skip=10)       0.0543 ms      of which 2 observations = 39%
skipFrames sweep:  slope 0.0008 ms/TICK,  intercept 0.0436 ms/CALL
                   -> 49% of the per-call intercept is the two vectors
```

A stage-5 candidate rollout measures **0.522 ms** against a **0.090 ms** floor
(snapshot + 100 ticks + the one observation it reads) — **5.8x**.

### Mechanism

Observation construction is `O(entities)` plus a 13,606-float allocate-and-zero.
Physics is 0.0006 ms/tick. So a rollout chunk spends more time describing the
board to nobody than it spends simulating it.

### Proposed edit

Extract the tick loop, so the two entry points cannot diverge in what they
simulate — the only difference is what they RETURN.

```cpp
// NEW, next to SelfPlayStepResult
struct SelfPlayFastResult { float reward0; bool done; };

private:
    struct SelfPlayTickOutcome { float reward; bool done; };
    // The self-play tick loop, with NO observation construction. Shared, so
    // stepSelfPlay and stepSelfPlayFast can never disagree about the physics.
    SelfPlayTickOutcome runSelfPlayTicks(<the existing 11 parameters>);

public:
    SelfPlayStepResult stepSelfPlay(<unchanged signature>) {
        SelfPlayTickOutcome o = runSelfPlayTicks(...);
        return { extractObservationForTeam(0), extractObservationForTeam(1),
                 o.reward, o.done };
    }
    // NEW
    SelfPlayFastResult stepSelfPlayFast(<same signature>) {
        return { runSelfPlayTicks(...).reward, runSelfPlayTicks(...).done };  // one call, see impl
    }
```

plus `src/bindings.cpp`:

```cpp
py::class_<SelfPlayFastResult>(m, "SelfPlayFastResult")
    .def_readonly("reward0", &SelfPlayFastResult::reward0)
    .def_readonly("done", &SelfPlayFastResult::done);
...
.def("step_self_play_fast", &ClashEnv::stepSelfPlayFast, ...)
```

and one call-site change in `python_ai/opponents/teacher.py::execute_steps`.

### Blast radius

* **`stepSelfPlay` is behaviour-identical.** The loop body moves verbatim; the
  return statement is unchanged. Nothing that calls it can observe a difference.
* **NOT gameplay-affecting.** No observation, action space, reward or physics
  change. `model_weights.pth` and `model_weights_selfplay.pth` stay valid, and
  win rates remain comparable across this change.
* **The RL learning signal cannot change**, and that is provable rather than
  argued: the vectors being removed are never read by anything. The teacher's
  chosen action is a function of `rollout_stats`, which reads
  `get_observation_for_team` separately and is untouched.
* **Additive binding.** An older `.pyd` simply lacks `step_self_play_fast`.
  Because a silent fallback would hide a stale `.pyd` — a trap this repo has
  already been bitten by — the Python side probes ONCE at import and raises with
  a message naming the rebuild, rather than degrading quietly.
* Other discard-the-result callers exist and can adopt it later:
  `envs/selfplay_env.py:368,380` (phase-2 warm-up), `envs/scenario_offense.py:124`,
  and several `eval/` harnesses. **This proposal changes only the teacher**, the
  one on the training hot path.

### What it does NOT address

`gym_wrapper.step` reads `observation0` and never touches `observation1`, so it
also builds one vector per decision for nothing — but it needs the other, so it
needs a *different* fix (a team-selective step) and is 7,952 calls against
500,076. Out of scope here; noted so it is not forgotten.

### Verification

1. C++ suite green, with the `[!shouldfail]` navigation-wedge case still the
   only failure and the runner exiting 0.
2. A new Catch2 case asserting `stepSelfPlayFast` and `stepSelfPlay` leave the
   env in the SAME state from the same snapshot — same tick, same reward, same
   done, same subsequent observation. That is the property the refactor must
   preserve, and it is the one a shared tick loop makes true by construction.
3. `tools/audit/engine_profile.cpp` before/after on the rollout block.
4. `profile_training.py --mode sync/--mode async` on the training box.

### APPLIED — measured result

`ClashEnv::stepSelfPlayFast` + `SelfPlayFastResult`, both entry points sharing
`runSelfPlayTicks`. The diff proves the claim that matters: inside the moved
code the ONLY changed line is the return statement.

**Correctness.**

* C++ suite **627 cases / 626 passed / 1 failed as expected**, runner exit 0,
  against a **625 / 624 / 1** baseline — the two new cases, no regression.
  (`tests/core/test_selfplay_fast_step.cpp`; built under wsl g++ on the box with
  no MSVC.)
* **The teacher's decisions are unchanged**: 12/12 configurations
  action-identical over stages 2-5 x 3 seeds, comparing full action sequences
  from identically-seeded runs. That is the property that would actually hurt if
  it broke, since every candidate score now flows through the new path.

**Speed** — paired, both arms alternating inside ONE process, only the engine
method differing (the slow arm routes `step_self_play_fast` back to
`step_self_play` at the boundary, so the teacher's Python is byte-identical):

| | ms per teacher decision |
|---|---|
| observations built (old) | 8.690 |
| **not built (new)** | **7.851** |
| saving | **0.839 (9.7%), 1.11x** |

**23 of 24 trials favour the fast arm, exact sign test p = 3e-06.** 27,298
observation vectors are no longer built per trial.

Cross-checked against the component measurement rather than trusted alone:
13,649 chunk steps x 2 observations x 0.0106 ms = 0.79 ms/decision predicted,
0.839 measured.

**Two measurement traps this hit, both recorded because they nearly produced
false results.**

1. **A background compile made the fix look 5.6x SLOWER.** Run as two separate
   invocations while the C++ suite was building on the same 4 cores, every span
   moved together — including `live.snapshot`, which the change does not touch.
   That uniformity is the signature of machine state, not of a code difference.
   Alternating arms inside one process removes it, and is now how the harness
   works.
2. **`UtilityTeacher` defaults to `seed=None`**, i.e. `np.random.default_rng(None)`,
   which is entropy-seeded. The first equivalence run reported **10 of 12
   configurations "diverged"** when the only difference was the teacher's own
   lane bias. Any A/B over this class must pass `seed=`.

**Scope, restated.** ~9.7% of teacher decision time. The teacher is roughly 59%
of the sync profile's wall clock, and the environment as a whole is 39.6% of the
async one at ~2.6x parallel efficiency, so the expected end-to-end throughput
effect is **low single digits** — this is the safe, zero-risk win, not the
answer to phase-1 throughput. The async profile puts the MODEL at 55.8%.

---

## 22. APPLIED 2026-08-24 — the live mirror cannot be given the real position: no tower HP, no unit HP, no clock, and every re-injected unit is inert for a second (proposed 2026-08-24)

### What this is for

The live-play pipeline, with the teacher as OUR agent in a real match:
perception reads the screen, `forecast.py` rebuilds a mirror `ClashRoyaleEnv`,
`UtilityTeacher` proposes candidates and ranks them by rolling each forward
5-10 s on `env.snapshot()`, and the winner is tapped by `live/actuator.py`.

Every joint of that already exists except one: **the mirror cannot be told what
the real position is.** The teacher then ranks candidates against a fabricated
board, which is the one failure that makes every other component's correctness
irrelevant.

### What is there now

`forecast.py` rebuilds by `reset()` + `inject()`, and its own docstring lists
what that loses. Two of the six were closed on 2026-08-17 (`set_elixir_for_team`
/ `set_hand_for_team`, landed but **still unwired** — that is a Python-side TODO,
not a platform limit). Four remain:

| gap | status |
|---|---|
| unit HP | injected units spawn at FULL health. No setter. |
| tower HP | always full after `reset()`. No setter. |
| match clock | always zero. No setter. |
| entity removal | a unit perception no longer sees cannot be deleted. |

**Removal is dissolved by the architecture rather than by an API.** The live
loop rebuilds the mirror from the latest perception snapshot at every decision
and never carries state forward, so there is nothing stale to delete. No removal
binding is requested. This is worth stating because the obvious incremental
design — stream deltas into a long-lived env — would need one, and would also
be unsafe (see Blast radius).

That leaves three. **Verification of this proposal found a fourth that nobody
had named, and it is probably the largest of the four.**

### The fourth gap: every re-injected unit is inert for a full second

Call chain, verified by reading, not inferred:

```
ClashEnv::inject  ->  card->spawnEntity(x, y, team, board)
                  ->  CardFactories::applyCardMetadata
                  ->  entity->deployTicksRemaining = DEPLOY_TIME_TICKS   // CardFactories.h:36
```

`DEPLOY_TIME_TICKS` is 10, i.e. 1.0 s. So a rebuilt board hands **every** enemy
unit a fresh deploy timer — including a Hog Rider that has been running for six
seconds. In every rollout, on every candidate, on every decision, the teacher
believes it has one extra second before anything on the board can act.

It is a defensive subsidy paid to US, and it is exactly the shape of a defect
this repo has already measured once.

**Why it likely dominates the other three.** CLAUDE.md's 2026-08-19 section
measured the SAME ONE SECOND in the opposite direction — a missing deploy second
subsidising the defender — and found it was the mathematical flaw suppressing
win conditions. Controlled, same harness, only the constant varying:

| engine | marginal value of a supported push |
|---|---|
| `DEPLOY_TIME_TICKS = 0` | **-73.7** HP, CI [-349.5, +195.3] |
| `DEPLOY_TIME_TICKS = 10` | **+448.5** HP, CI [+137.3, +760.1] |

One second of deploy inertness moved a push by ~520 tower HP and flipped the win
condition from negative to positive value. The rebuild currently applies that
second to every enemy unit in the mirror.

### Measured evidence for the three requested gaps

**Tower HP — and the trap that makes the naive setter wrong.**
`perception/README.md` finding 6, read off a clean frame at t=20 s before
anything is damaged, so the on-screen numbers are true maxima:

| | our Princess | opponent's Princess | engine |
|---|---|---|---|
| max HP | 1750 (level 4) | 1890 (level 5) | **2534** (level 9) |

**Absolute HP is not comparable, and it is wrong by a DIFFERENT factor per
player.** Perception already reports a FRACTION for exactly this reason, taking
the maximum from the first undamaged reading rather than a supplied table.
Tower HP itself is read as the absolute printed numeral
(`readers/tower_numerals.py`, finding 9) and validated by read-back over 4 full
matches: 542 steps, 12 upward jumps, **97.8% consistent** with the fact that
tower HP never rises.

**Unit HP.** `live/unit_hp.py` reports `UnitHp.fraction`. Scored as an "is this
unit damaged" detector over 226 detections, reweighted from a stratified sample
to the population (finding 7): precision **0.79 -> 0.98**, recall
**0.34 -> 0.56**.

State this honestly: **roughly half of damaged units will still inject at full
HP.** This improves a biased estimator, it does not fix it. It is still a strict
gain, because today's effective recall is 0.

**Match clock.** `ClashEnv::currentTick` exists and is already read into the
observation (`ClashEnv.h:331`, `currentTick / maxTicks`) and into the done
condition (`:383`, `:542`). It is only ever set to 0 by `reset()` (`:517`) and
incremented in the step loops.

**The honest limit here is larger than the setter.** The engine models **no
double or triple elixir**: `ELIXIR_REGEN_RATE` is a constant (`GameManager.h:39`)
scaled only by `oppElixirMultiplier` (`:661-662`). So even a perfect clock
leaves every rollout during 2x mispriced on both sides. A clock setter buys the
observation's time scalar and timeout proximity — real, but modest. **Filed
separately as item 23** rather than folded in here, because it is a genuine
gameplay change that would affect training, while everything in this item is
additive and inert unless called.

### Proposed edit

Four changes. Every one is additive, and every new parameter defaults to
current behaviour.

**(a) `ClashEnv::setTowerHp(int team, int slot, float hp) -> bool`**

`slot`: 0 = King, 1 = left Princess, 2 = right Princess, in board coordinates
(not team-relative), so the caller is not asked to mirror anything.

- Clamps to `(0, maxHp]`.
- **Returns `false` and changes nothing on `hp <= 0`**, following
  `setHandForTeam`'s precedent: refuse rather than accept a misread. A 0-HP
  tower that still occupies its cell and still fires is worse than no update.
- Takes **engine-absolute HP**. The level conversion stays on the perception
  side, where the per-player max already lives, per CLAUDE.md's
  no-second-copies rule. Python passes `fraction * engine_max`.
- **Side effect to name, because it is gameplay-visible and correct:**
  `Tower::awake` latches on the invariant `hp < maxHp`, so injecting a damaged
  tower wakes the King. That matches the real game and is the desired
  behaviour, but it means `setTowerHp` is not a pure state write.

**(b) `ClashEnv::destroyTower(int team, int slot) -> bool`**

Routes through the engine's existing destruction path so the crown, the King
wake and `LanePath`'s retargeting all fire. Requested because (a) refuses
`hp <= 0` and a destroyed tower would otherwise be inexpressible in the mirror —
which would make every rollout wrong from the moment a tower falls, i.e. exactly
when the position matters most.

**(c) `ClashEnv::inject(int cardId, float x, float y, int team, float hp = -1.0f, int deployTicks = -1)`**

Two optional parameters on the existing method:

- `hp < 0` keeps full health (current behaviour).
- `deployTicks < 0` keeps `DEPLOY_TIME_TICKS` (current behaviour);
  `0` spawns an already-deployed unit, which is what a rebuilt board wants for
  every unit that was already on screen.

Optional-with-preserving-defaults rather than a new method, so **no existing
caller changes at all** — `forecast.py`, `prove_*.py` and the audit tools keep
compiling and keep behaving identically.

**(d) `ClashEnv::setCurrentTick(int tick)`**

Clamps to `[0, maxTicks]`. Trivial, but it is the one field that cannot be
reconstructed by any combination of the others.

Bindings mirror these as `set_tower_hp`, `destroy_tower`, `set_current_tick`,
and two new `py::arg`s with defaults on the existing `inject`.

### Blast radius — additive, and NOT gameplay-affecting

**No checkpoint is invalidated and no win rate is invalidated**, and that is a
design goal rather than a happy accident — the same discipline the `place_hires`
zero-init used:

- `setTowerHp`, `destroyTower` and `setCurrentTick` are new methods. Nothing in
  either training pipeline calls them. An uncalled method cannot change a
  rollout.
- `inject`'s two new parameters default to exactly today's behaviour, so every
  existing call site is bit-identical.
- No observation, action-space, reward or architecture change.

**This must be VERIFIED, not assumed** — CLAUDE.md records that a "purely
additive" `.pyd` change was checked by diffing the commits (119 insertions, 0
deletions, no simulation code) rather than trusted. Same bar here.

**The one thing this proposal deliberately does NOT enable: concurrent mutation
of a live env.** The setters are for a single-owner rebuild at decision time.
`src/bindings.cpp` contains **no `gil_scoped_release` and no `call_guard`** —
verified by grep — so every engine call holds the GIL for its full duration, and
a second thread writing into an env while the teacher rolls candidates forward
would both stall perception and score candidates against different worlds. The
Python side must own the mirror on one thread.

### Verification

| check | bar |
|---|---|
| C++ suite | 622 cases, 1 `[!shouldfail]` wedge, runner exits 0 |
| Python suite | 401 collected, 399 pass / 2 skip |
| perception suite | 353 pass / 1 skip |
| `inject` back-compat | a board built with no new args is bit-identical to today's, over the full observation |
| deploy bypass | a unit injected with `deployTicks=0` moves on tick 1; with the default it does not move until tick 11 |
| tower fraction round trip | `set_tower_hp(t, s, f * max)` then read back through the observation returns `f` within float tolerance, for both teams |
| refusal | `set_tower_hp(t, s, 0.0)` returns `false` AND leaves HP unchanged — both halves, since a refusal that still writes is the worst outcome |
| destruction | `destroy_tower` awards the crown, wakes that team's King, and changes `get_towers_alive` |
| absorbing states | `waypoint_probe` still reports 0, since a re-injected unit at an arbitrary position is a new entry path into `getNextWaypoint` |

That last row is not boilerplate. This engine has shipped **two** absorbing
states at the bridge mouths, the second one found only because a sweep covered
every branch rather than the one the reproduction took. Injecting units at
arbitrary perceived positions with `deployTicks=0` puts entities into
`getNextWaypoint` at positions no normal spawn produces.

### Options considered and rejected

- **Stream deltas into a long-lived env.** Needs a removal API, needs entity
  identity across frames (blocked by item 20 — spawned entities carry no
  `cardId`), and is unsafe under the GIL finding above. Rebuild-per-decision
  costs ~1 ms and needs none of it.
- **`set_entity_hp(entityId, hp)`.** Cleaner in principle, but requires stable
  entity ids across the binding boundary, which item 20 says do not exist. The
  `hp` parameter on `inject` needs no identity at all: the caller sets HP on the
  unit it is creating, in the same call.
- **Clamping tower HP to 1 instead of refusing.** Biases toward over-defending a
  tower that is already gone, and does it silently.

### APPLIED — approved, implemented and measured 2026-08-24

Approved by the human on the proposal above, then built TDD: the 19 new cases
in `tests/core/test_state_setters.cpp` were written against compiling stubs
and **17 were watched to fail** before any implementation existed.

**Two of those 19 passed against a do-nothing stub and had to be
strengthened**, which is the part worth carrying. `setTowerHp REFUSES hp<=0`
passes trivially against a setter that refuses *everything*, and `an injected
hp above the maximum is clamped` passes trivially against one that ignores
`hp` — and ignoring it IS the pre-item-22 behaviour. Both now carry a positive
control that must fire first. Same rule this file already states for the
deploy-zone probe: **when a measurement's failure mode is maximal
permissiveness, it needs an internal control that MUST fire.**

**What shipped**, all additive:

| | |
|---|---|
| `Building::getMaxHp()` | the accessor the clamp needed; `maxHp` was protected with no reader |
| `GameManager::{findTower,setTowerHp,destroyTower,getTowerHp,getTowerMaxHp,setCurrentTick,getCurrentTick}` | the implementations |
| `ClashEnv::{setTowerHp,destroyTower,getTowerHp,getTowerMaxHp,setCurrentTick}` | wrappers |
| `ClashEnv::inject(..., hp = -1.0f, deployTicks = -1)` | two optional params |
| `bindings.cpp` | all of the above, plus `hp` / `deploy_ticks` keywords on `inject` |

`destroyTower` routes through `takeDamage` (the entry point a real killing blow
uses) and then pins the postcondition, because `CombatEntity::takeDamage` can
ABSORB via shield/parry/mid-dash invulnerability. No Tower carries any of those
today; the pin means this stays a destruction if one ever does.

**MEASURED — the deploy gap, and the probe that measured nothing first.**
Enemy Hog injected at (9.0, 20.0), both sides no-oping:

| | ticks until it first damages our tower |
|---|---|
| `inject(default)` | 88 |
| `inject(deploy_ticks=0)` | **78** |

**Exactly 10 ticks — `DEPLOY_TIME_TICKS` to the tick**, which is the
confirmation the mechanism is the one diagnosed. In the window where arrival
decides the outcome it is worth **317 tower HP, one full Hog hit**:

| window | default | deploy_ticks=0 | delta |
|---|---|---|---|
| 100 ticks | 317 | 634 | **+317** |
| 110 ticks | 634 | 951 | **+317** |
| 120 ticks | 951 | 951 | 0 |

**Read the 120-tick row: the first probe used 140 ticks and reported a delta of
ZERO.** Over a long enough window the Hog deals its full damage either way, so
a total-damage probe SATURATES and reports that a working fix does nothing.
Arrival time is the quantity that can see it. Third instance in this repo of a
control saturating — the air-targeting probe in the 2026-08-20 deck QA is the
same failure.

**Verification, against the bars this item set:**

| check | bar | result |
|---|---|---|
| C++ suite | 1 `[!shouldfail]`, exit 0 | **646 cases / 6,409 assertions, 645 pass, 1 failed as expected, exit 0** |
| Python suite | no regression | **399 passed / 2 skipped** — unchanged |
| perception suite | no regression | **366 passed / 1 skipped** (353 + 13 new) |
| `inject` back-compat | bit-identical | full observation equal after 50 ticks, 4-arg vs explicit defaults |
| deploy bypass | moves on tick 1 | pinned, C++ and Python |
| tower fraction round trip | within tolerance | exact for all three slots, both teams |
| refusal | returns false AND leaves hp | both halves pinned |
| destruction | crown + King wake + count | pinned, and team-scoped |
| absorbing states | `waypoint_probe` still 0 | **0 / 8,661,439 positions**, and 0 / 2,584,034 in the lane-composition sweep under all three tower configurations |

**The C++ count was already stale.** 646 − 19 new = **627**, against the 622
this file and CLAUDE.md record. The baseline was 5 ahead before this work
started, so do not read the jump as belonging to item 22 — the same arithmetic
trap the 2026-08-24 row-compaction section records for the Python count.

**The absorbing-state bar was worth keeping.** `deploy_ticks=0` puts entities
into `getNextWaypoint` at arbitrary perceived positions with no deploy delay to
absorb the first tick — a genuinely new entry path, and this engine has shipped
two absorbing states at the bridge mouths already, the second found only
because a sweep covered every branch rather than the one the reproduction took.
Re-run after the change: **0 absorbing states**, both sweeps, all three tower
configurations.

**What is NOT verified, and should be said plainly:** none of this has faced a
real screen. Every number above is engine-side. Whether perception's readings
are good enough to make the mirror worth having is a Stage 2 question, and the
unit-HP recall of 0.34–0.56 is the figure to watch — roughly half of damaged
units will still arrive at full health.

**A binding-surface regression test now exists** at
`perception/tests/test_engine_state_setters.py` (13 cases). The C++ suite
cannot see pybind at all, and this repo has twice had a stale `.pyd` hide a
landed setter for days with the C++ suite green throughout.

---

## 23. OPEN — item 7 seeded the engine and nothing was migrated to it; one RNG path is still unreachable from Python (proposed 2026-08-24)

**Not a correctness bug. The unfinished half of item 7**, and the reason that
item's stated benefit — *reproducible failures* — has still not been collected
three days after it landed.

> **SCOPE NOTE, because this file is for engine changes.** Two of the three
> paths below are **Python**, in `python_ai/`, and by this repo's own division
> of labour they belong in `BOT_REQUESTS.md` (training-side suggestions), not
> here. They are written up here anyway because they are meaningless apart from
> item 7 and splitting one finding across two files is how item 7's own status
> went stale in the first place. **Only §C is an engine request.** §A and §B
> were applied on 2026-08-24 at the human's explicit instruction and are
> recorded, not requested.

### What item 7 actually delivered

Verified by reading, 2026-08-24:

```cpp
// include/core/ClashEnv.h
void seed(unsigned int s) {
    rng.seed(s);                      // HeuristicOpponent
    heuristicOpponent.reset(rng);
    game.seed(s ^ 0x9E3779B9u);       // opening hand + cycle order
    reset();                          // initializeDeck runs INSIDE reset()
}
```
```cpp
// src/bindings.cpp:164
.def("seed", &ClashEnv::seed, py::arg("seed"))
```

`seed()` ends in `reset()`, which makes it a **drop-in replacement for
`reset()`** at any call site that wants determinism — that property is what
makes §A and §B one-line changes rather than restructuring.

### The three paths that bypassed it

| # | path | reachable from Python? | status |
|---|---|---|---|
| A | `MicroRoyaleEnv.reset(seed=...)` | yes | **applied 2026-08-24** |
| B | `prove_combos.py`'s five harnesses | yes | **applied 2026-08-24** |
| C | `sample_random_deck`'s static generator | **no** | **this request** |

---

### A. `MicroRoyaleEnv.reset(seed=...)` accepted a seed and dropped it — APPLIED

`python_ai/envs/gym_wrapper.py:333`, before:

```python
def reset(self, seed=None, options=None):
    super().reset(seed=seed)          # seeds the WRAPPER's np_random only
    ...
    obs_list = self.game.reset()      # engine re-deals, unseeded
```

`gymnasium.Env.reset(seed=...)` seeds `self.np_random`. It cannot reach either
engine generator, and `MicroRoyaleEnv` reads `self.np_random` nowhere. So the
argument was accepted, had no effect on anything the env actually does, and
**looked like it worked** — which is strictly worse than not accepting it,
because the gymnasium contract says a caller may rely on it.

`UPSTREAM_REQUESTS.md` item 7 named this exact trap ("`MicroRoyaleEnv.reset(seed=...)`
looks like it should help but only forwards to `gymnasium.Env.reset`") at a time
when there was no binding to forward to. There has been one since 2026-08-21.

**Applied:** one guarded line forwarding to `self.game.seed(seed)`. Guarded on
`seed is not None` because the gymnasium convention is that a seed is passed
once and subsequent `reset()` calls continue the stream — seeding on every reset
would make every episode of a run identical, which is a far worse failure than
the one being fixed.

### B. `prove_combos.py` never called it — APPLIED

Five harnesses (`run_usage`, `run_reserve_ab`, `run_combo_ab`,
`run_profile_sweep`, `run_vs_net`), each building its opening as
`CE(...)` then `.reset()`, and `--seed` reaching only `make_teacher`'s own RNG.
This is the harness whose mis-specified control "cost a 10-minute run and nearly
produced a wrong conclusion about which combo family was responsible for a
trend" — the single most-cited piece of evidence for item 7.

**Applied:** `.reset()` → `.seed(args.seed + ENGINE_SEED_OFFSET + i)` at all
five sites, plus the module docstring, which still told readers the shuffle was
unseeded and that item 7 was open.

**Why an offset rather than `args.seed + i`:** the teachers already draw from
`args.seed + i`. Reusing it for the engine would move a teacher's lane bias and
the hand it was dealt together across openings — the same correlation
`ClashEnv::seed` avoids internally with its `^ 0x9E3779B9` between the two
engine generators, for the same reason.

---

### C. THE ENGINE REQUEST — `sample_random_deck` cannot be seeded

```cpp
// src/bindings.cpp:292
m.def("sample_random_deck", []() {
    static std::mt19937 rng(std::random_device{}());
    return sampleRandomDeck(rng);
});
```

A **third** `std::mt19937`, function-local `static`, seeded from
`std::random_device`, with no parameter and no seeding entry point. `ClashEnv::seed`
cannot reach it — it is not a member of anything.

**Why it matters, concretely.** `gym_wrapper.reset()` calls it on the
`randomize_opp_deck` path:

```python
random_deck = list(clash_royale_env.sample_random_deck())
self.game.set_opponent_deck(random_deck)
```

So **§A's fix is incomplete exactly where deck randomisation is on.** A run with
`randomize_opp_deck=True` now has a reproducible opening hand, cycle order and
heuristic roll, and a still-random *opponent deck* — which is the largest single
source of episode-to-episode variance of the four. Phase 1's `random_opponent`
and the scripted bots' randomised decks are the configurations this affects, and
they are the ones TODO.md item 6 wants extended, not retired.

The `static` also means the stream is **process-global and order-dependent**:
two envs constructed in the same process interleave draws from one generator, so
even seeding it would only be reproducible for a fixed construction order. Worth
knowing before anyone calls this a one-liner.

### Options

1. **Add a module-level seeding function** — `m.def("seed_deck_sampler", ...)`
   setting the same static. Smallest diff; leaves the process-global stream and
   its order-dependence in place, so it buys reproducibility only for a fixed
   call order. Adequate for a single-env eval harness, not obviously adequate
   for `num_envs = 8`.
2. **Give `sample_random_deck` an optional seed argument** — `sample_random_deck(seed=None)`,
   constructing a local generator when one is passed and falling through to the
   static otherwise. Every existing zero-argument call site keeps its current
   behaviour bit-for-bit, and a caller that wants determinism gets a stream that
   is not shared with anyone. **Recommended.** It is additive, it does not
   change the meaning of any existing call, and it is the only option that
   survives vectorised envs.
3. **Move the generator into `ClashEnv`** and have `ClashEnv::seed` cover it.
   Rejected: `sample_random_deck` is deliberately module-level because it is
   called *before* a deck exists to construct an env with, and `train.py` /
   `train_selfplay.py` call it outside any env at all.
4. **Change nothing, and document it.** Defensible — deck randomisation exists
   to create variety, and a caller who wants a reproducible deck can pass one
   explicitly via `set_opponent_deck`. If this is the choice, §A's docstring
   should say so, because "seeded" will otherwise be read as "reproducible".

### Blast radius

**Options 1 and 2 are additive and NOT gameplay-affecting.** No existing call
site changes behaviour, no observation changes, no checkpoint and no win-rate
history is invalidated. Option 2 touches one lambda in `src/bindings.cpp` and
nothing else; `sampleRandomDeck` itself already takes an `std::mt19937&` and is
unchanged.

### What is NOT verified, and it is the whole verification section

**Nothing below the reading level. No number in this item was measured, and
none could be.** The machine this was written on has MSVC 2022 and WSL but
**no Python 3.11, no `python_ai/venv`, no `perception/.venv` and no built
`clash_royale_env.pyd`** — so nothing that imports the engine runs here at all.

What was actually done: the three paths were read, `sub`-style exact-match edits
were applied to §A and §B, and both files were confirmed to parse under
Python 3.13. That is enough to claim the seed now *reaches* `ClashEnv::seed`,
and it is **not** enough to claim any run is reproducible.

**The acceptance test is the one item 7 already wrote** and it has never been
run against §A or §B:

```python
a = MicroRoyaleEnv(cfg); b = MicroRoyaleEnv(cfg)
oa, _ = a.reset(seed=7); ob, _ = b.reset(seed=7)
assert (oa == ob).all()          # identical opening hand AND cycle order
```

plus, for §B, two `prove_combos.py --seed 300` invocations whose OFF arms
report the **same level**, not merely the same delta — the check that failed in
2026-08-20 and produced the evidence item 7 was argued from. Until someone with
a 3.11 environment runs both, §A and §B are *plausible and unverified*, and this
file should keep saying so.
## 24. APPLIED 2026-08-23 — the Giant walks at 0.600 tiles/s and the real one walks at 0.99, measured on two independent recordings (proposed 2026-08-23)

**GAMEPLAY-AFFECTING. Every win rate earned before this is historical.** Checkpoints
are NOT invalidated: the observation, action space and architecture are untouched, and
`sightRange`/`speed` are not observation channels (`CH_RANGE` carries attackRange, and
`CH_SPEED` carries the value being changed, so the OBSERVATION still describes whatever
the entity actually has -- no tensor changes shape).

### The measurement

`perception/videos/` was read frame by frame: the arena homography in
`config/profile_gpg_1920x1080.json` re-validated against these files (worst anchor error
**0.312 tiles**), units located with `units_M_480x352.onnx`, positions taken at the
sprite's FEET and mapped through the homography. A stationary Cannon read (8.5, 9.2)
across five samples with a spread of +/-0.05 tiles, which is the independent check that
the mapping is stable.

| | speed | window | residual sd |
|---|---|---|---|
| video 1 (`20-45-03`), Giant | **0.999** tiles/s | 26 frames, 6.8 s | 0.174 tiles |
| video 2 (`20-48-33`), Giant | **0.974** tiles/s | 140 frames, 13.7 s | 0.298 tiles |
| engine | **0.600** tiles/s | -- | -- |

Two different matches, opposite lanes, opposite directions of travel, agreeing to 2.5%.
The engine is **1.64x too slow** for this card.

### CORRECTION, same day: the Fast tier is too slow as well

This item first claimed the defect was Giant-specific, on a Mini P.E.K.K.A fit that
matched the engine. **That fit was contaminated and the claim was wrong.** Its window
(118.8-122.4 s) began and ended while the unit was STALLED -- visible in the trace as a
flat y ~= 17.9 for the first 0.8 s -- which pulled the fitted speed down to 1.528 and
made the engine's 1.600 look correct. Refitted on the clean descent alone:

| | window | speed | residual sd |
|---|---|---|---|
| Mini P.E.K.K.A, contaminated | 118.8-122.4 s | 1.528 | 0.373 |
| Mini P.E.K.K.A, **clean descent** | 120.3-124.5 s, 40 frames | **2.003** | 0.195 |

*A fit window is part of the measurement. Choosing one that spans combat measures the
combat.* Same failure shape as the "three rising samples" walk-start estimator and the
p90 speed estimator, both discarded earlier in the same session.

### What the corrected numbers say

| card | tier | video | engine | video/engine |
|---|---|---|---|---|
| Giant | Slow | 0.987 | 0.600 | **1.65x** |
| Mini P.E.K.K.A | Fast | 2.003 | 1.600 | **1.25x** |

**Both are too slow, by different factors** -- so this is not one card, and it is not a
uniform clock error either. The invariant that survives is the RATIO, and it is where
the engine actually breaks:

| | Fast : Slow |
|---|---|
| the recordings | **2.03** |
| real game, published tiers (90 / 45 tiles per minute) | **2.00** |
| **this engine** | **2.67** |

The footage reproduces the published tier ratio to 1.5%, which is strong independent
corroboration that the measurement pipeline is sound -- the two cards were tracked in
different matches, different lanes, opposite directions. The engine is the outlier.

Note the units: the published figure is tiles per MINUTE in REAL-game tiles, while
everything measured here is in ENGINE tiles (the homography is fitted to the engine's
own tower layout). Absolute tiles/second therefore are NOT comparable across the two,
and the ratio is -- which is why the ratio is the claim.

### Why MOVEMENT_SPEED_SCALE cannot fix a ratio

`CardStats.h`'s `MOVEMENT_SPEED_SCALE = 0.2f` is a **global multiplier**, and a global
multiplier preserves ratios by construction. CLAUDE.md records the ratio as wrong -- "the
engine's own Slow:Medium tier ratio (0.60 against the real 0.75)" -- and cites it as one
of the three pieces of evidence justifying the 2026-08-07 movement fix. That fix applied
one scale to every card, so it could move the average and could not touch the per-tier
error. The ratio is still wrong today, in the same direction.

### The edit

`include/core/CardRegistry.h:695`, the sixth argument (speed):

```cpp
-  add(troop(2, "Giant", 5.0f, Archetype::MeleeBuildingTargeter, 3968, 0.3f, 1.2f, 253, 15, 'G')...
+  add(troop(2, "Giant", 5.0f, Archetype::MeleeBuildingTargeter, 3968, 0.5f, 1.2f, 253, 15, 'G')...
```

`0.5f * MOVEMENT_SPEED_SCALE * 10 ticks/s = 1.000 tiles/s`, against 0.987 measured
(1.3% high). The neighbouring literal 0.4f gives 0.800, which is 19% low.

### Blast radius, and what was deliberately NOT changed

Six other cards share the `0.3f` literal because they share a tier -- **Royal Giant,
Sparky, Electro Giant, Elixir Golem, Lava Hound, Hero Giant**. Only the Giant was
measured, so only the Giant was changed. The **17 cards on `0.8f`** (the Fast band) are
~1.25x slow by the corrected Mini P.E.K.K.A measurement and were also left alone.

**This edit therefore does not make the tier structure correct.** It puts one measured
card on its measured value. Applying it moves the engine's Fast:Slow ratio from 2.67 to
1.60, against a real 2.00 -- closer in magnitude, wrong in the other direction. Making
the ratio right needs the Fast band moved too (0.8f -> 1.0f, giving 2.000 tiles/s against
2.003 measured), which is 17 further cards and was not done unilaterally.

That leaves a real, deliberate inconsistency, and the reason for accepting it is the
part worth reading. Setting the Giant to `0.5f` puts it on the SAME literal as the 80
cards in the middle band, i.e. it collapses "Slow" into "Medium". That may be correct --
or it may mean the middle band is itself mislabelled -- and the data cannot currently
tell the two apart: the two Medium units measurable in these recordings gave **0.749**
(Valkyrie) and **1.183** (Musketeer), a 1.6x spread inside one tier, because both were
in combat rather than walking cleanly. Sweeping six unmeasured cards onto a number
derived from one card, while the anchor that would justify it is that noisy, is exactly
the extrapolation this document exists to prevent.

**To close it properly**, measure a clean walking segment for a Medium card and for a
second Slow card (Golem or P.E.K.K.A) on these same recordings, then either sweep the
tier or split it deliberately.

### Reproducing it

```bash
tools/audit/video_replay.cpp        # one card's trajectory, tick by tick
tools/audit/video_scenario.cpp      # a multi-card push vs a defending tower's HP
tools/audit/video_replay_log.cpp    # the same push as a web/viewer.html replay
```

**SUPERSEDED by item 25**, which replaced the whole speed model rather than this one
card. The measurement above stands; the single-literal edit it describes was folded into
the tier rewrite.

---

## 25. APPLIED 2026-08-24 — the whole speed model: the engine had no tiers, and 104 of 131 troops were wrong (proposed 2026-08-24)

**GAMEPLAY-AFFECTING, AND THE LARGEST SUCH CHANGE IN THIS FILE. Every win rate, every
Elo anchor and every reward curve earned before today describes a different game.**
`model_weights_selfplay.pth` now plays an environment it did not train in and will be
weaker until retrained. Checkpoints still LOAD — observation, action space and
architecture are untouched — but their measured strength is void.

### What was wrong

The real game gives every card ONE speed number, in tiles per MINUTE, and it takes only
five values: **30 / 45 / 60 / 90 / 120**. Confirmed against Supercell's own exported
table (`cards_stats_characters.json` in RoyaleAPI/cr-api-data): exactly those five values
across 119 characters, no others.

This engine had **nine** ad-hoc literals and no tier concept at all, so cards sharing one
real tier were scattered across different speeds. Every one of these is Slow (45):

| card | before | after |
|---|---|---|
| Golem | 0.400 tiles/s | 0.994 |
| Giant, Royal Giant, Lava Hound, Electro Giant, Elixir Golem | 0.600 | 0.994 |
| P.E.K.K.A. | 0.800 | 0.994 |

**No value of `MOVEMENT_SPEED_SCALE` could ever have fixed that.** It is a global
multiplier and preserves ratios by construction — which is exactly why the ratio
CLAUDE.md flagged as wrong on 2026-08-07 ("Slow:Medium 0.60 against the real 0.75")
survived that fix untouched, and why `CardStats.h` still carried a comment saying the
tiers were "left uncorrected on purpose".

Measured over the whole registry: **only 5 of 109 resolvable troops were within 5% of
correct. 104 were not.** The dominant error was a uniform **×1.33 on 53 cards** (the
Medium band) with tier-assignment errors layered on top, ranging to ×2.49.

### The calibration, and why it is not 1/60

A REAL tile is not an ENGINE tile — this board's tower layout differs from the real
arena's — so the conversion is measured, not assumed. Two cards tracked frame by frame
through `perception/videos/` with the homography in `config/profile_gpg_1920x1080.json`
(re-validated on these files, worst anchor error 0.312 tiles):

| card | real stat | engine tiles/s | implied factor |
|---|---|---|---|
| Giant | 45 | 0.987 (two recordings, 2.5% apart) | 0.02193 |
| Mini P.E.K.K.A | 90 | 2.003 | 0.02226 |

**Two cards, two different tiers, one constant, agreeing to 1.5%.** The footage also
reproduces the published Fast:Slow ratio — 2.03 measured against 2.00 published — so the
recordings and Supercell's table corroborate each other independently.

`REAL_TILES_PER_MIN_TO_ENGINE = 0.011045f`, and the five tiers derive from it.

### Verification, which is the part that makes this safe

The rewrite was scripted, so it was round-tripped rather than trusted: rebuild, re-dump
every spawned troop's actual speed, and assert it equals its official tier times the
calibration. **109 / 109 match, 0 mismatches.**

A first attempt was DISCARDED and is worth recording. It replaced only the first regex
match per card, and several cards appear more than once — `troop(24, "Skeletons", ...)`
exists both as a death-spawn helper and as the playable card. It silently edited the
helper and left the real card alone. The round-trip check is what caught it; reverting
and redoing with all-occurrences replacement is what fixed it.

### What was deliberately NOT changed

**13 cards have no official row** and keep their previous values: Berserker, Ronin,
Goblin Machine, Goblin Demolisher, Furnace, Heal Spirit, Suspicious Bush, Rune Giant,
Little Prince, Goblinstein, Boss Bandit, Spirit Empress. They are newer than the exported
table. They now sit off-tier (1.000 / 1.400 / 1.700 tiles/s) and should be assigned once
a source covers them.

The **9 Hero variants** WERE changed, by mirroring their base card's tier rather than by
guessing: "Hero Giant" takes Giant's tier, "Hero Knight" takes Knight's. They are
engine-invented and have no real counterpart, but the naming makes the intent explicit,
and leaving them behind would have put Hero Giant at 0.600 against Giant's 0.994.

### What this does not settle

The calibration rests on **two** measured cards. It is strongly corroborated — different
tiers, different matches, and it reproduces the published ratio — but a third measurement
in the Medium band would make it three-point. The two Medium units available in these
recordings were both in combat rather than walking cleanly (0.749 and 1.183, a 1.6×
spread inside one tier) and were discarded rather than averaged.

`perception/tools/sim_fidelity.py` is the harness the old `CardStats.h` comment named for
settling this, and it has NOT been re-run against the new tiers.


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

## 27. MEASUREMENT 2026-08-24 — item 25's missing third Medium data point, and a new question about the Fast tier

Item 25 closed with: *"a third measurement in the Medium band would make it
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
0.994 — the tightest row in the table and a direct confirmation of item 24's
recalibration. Archers and Valkyrie are two Medium units that **agree with each other**
(1.28, 1.23) and with the engine's 1.3254, which is exactly the three-point
corroboration item 25 asked for and did not have.

### Musketeer is the same contaminated case item 25 already saw

Musketeer reads 0.82 — the widest gap in the table and the one to ignore. It shares a
tier with Archers and Valkyrie, so all three must measure alike, and it does not. Note
that item 25 recorded a discarded Medium reading of **0.749**; this is almost certainly
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

**No change is proposed on this.** One card, one match, and item 25's own warning about
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

The engine column comes from `.vid/final.tsv`, the per-card speed dump item 25 produced.
