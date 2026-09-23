# ClashRoyaleEnv

A Clash Royale battle simulator in C++ (MSVC/CMake/pybind11) with a PPO agent
in Python, plus `perception/`, which reads real matches off the screen and
drives the simulator from them as a state estimator.

---

## Rules — read before editing anything

**`include/`, `src/` and `python_ai/` are READ-ONLY unless explicitly asked.**
A training run is usually live against them. Touching `python_ai/` can kill a
multi-hour run; touching `include/`/`src/` invalidates checkpoints.

**Never change C++ without confirming the exact diagnosis and the exact edit
first.** Even when clearly justified. Write the proposal to
`perception/UPSTREAM_REQUESTS.md` (measured evidence, blast radius, options)
and let the human decide.

**Any gameplay-affecting engine change invalidates `model_weights.pth`'s
win-rate history.** Say so when proposing one.

**Don't put a second copy of an engine constant in Python.** Derive it from the
bindings. This has gone stale twice: `model.py`'s "18*16=288" comment, and
`perception/tools/calibrate.py` scoring bridges against `y=17.0` after the
river moved. `perception/geometry.py` shows the pattern — live where derivable,
hardcoded with a comment naming the header where not.

---

## 2026-09-15: the pre-launch audit — read this before the from-scratch run

The next run starts **from scratch** with a **different deck** (chosen later).
Checkpoint invalidation was therefore free, and the audit hunted for anything
that would crash, silently switch off, or quietly mis-tune under that. The
operator's side is `docs/runbooks/FINAL_RUN_RUNBOOK.md`; this is what is now true of the code.
Reports and probes lived in the session scratchpad; every number below was
measured on the rebuilt `.pyd`.

**The recurring bug, four more times: a mechanism keyed to the 2.6 deck that goes
SILENT under another deck.** None raised, warned or logged. Now:

- **The deck is `CLASH_DECK`** (ids or names; `python_ai/deck.py`, re-exported as
  `gym_wrapper.DEFAULT_DECK`). **`envs/deck_contract.validate_deck` prints what the
  deck turns on and off at the top of every run** and refuses a Champion/Hero deck
  (ability training is not implemented) before any env is built. The checkpoint
  records the deck; resuming under a different one warns loudly.
- **The advisor target is derived, not three literal ids.** On 5 of 8 plausible
  replacement decks Cannon 25 / Fireball 7 / Hog 15 were all absent, so the
  coverage term (10% of log 612 on the placement head) trained on zero cards.
  `advisors/card_probes.py` measures what a card does by injection — a spell's
  radius and damage (Fireball reads exactly 2.5 / 689), whether a building
  actually attacks (behaviourally: Elixir Collector's DPS *channel* is non-zero),
  whether a troop walks to buildings. The shipped deck's table is unchanged.
- **ONE win-condition resolver** (`teacher.resolve_win_condition`, used by the
  teacher AND by `gym_wrapper._find_win_condition`, which feeds
  `W_WIN_CONDITION_DAMAGE`). Both copies ranked by cost: the reward went dead for
  siege/spell/Miner decks and credited a 2-elixir Ice Golem on a Champion deck;
  the teacher named Wall Breakers in Miner control and Lava Hound in LavaLoon.
  Eligible cards (building-targeter, deploy-anywhere, siege building, spawning
  spell) are ranked by MEASURED tower damage from a LEGAL attack cell, absolute
  first then per elixir. Two traps found while fixing it: per-elixir first named
  the Miner over the Balloon (the 300-tick probe saturates at one Princess, so
  cheap cards win), and a rolling spell probed on a cell it cannot be cast on beat
  the Graveyard. The floor only excludes no-route cards: a Skeleton Barrel (81 HP
  per elixir, a real win condition) scores level with an Ice Golem (84, a tank),
  so no floor separates "real win condition" from "tank".
- **Scenario injection follows the deck**: 60% of its weight is boards only an
  area spell answers, now drawn only if the deck holds one. It also held a
  **seventh stale arena copy** (`tower_x = 4.0 / 14.0`, the pre-re-centring layout).
- **Placement mask**: deploy-anywhere troops (Miner 242 vs engine 520 cells) had
  the enemy half deleted by the own-half row rule; **every Evolution** had an
  all-False legality row because the table iterated `get_all_card_ids()`, which
  filters Evolutions out. Validated against the engine's behaviour (elixir spent),
  not against the predicate.

**Engine (UPSTREAM item 28): damage to a SPAWNED body was booked as TOWER
damage** — the collector called "not in CardRegistry" a tower, and spawned bodies
are unregistered too (810 for one Goblin Barrel). It fed the agent's tower
potential and the teacher's rollout. `DamageDealtEvent.targetIsTower` now carries
the target's own `isTower()`. C++ suite **714 cases, 713 pass, 1 expected failure,
exit 0**.

**Reward: +0.084 on the first real step of every episode.** On the autoreset
phantom step `infos` is empty and elixir defaults to 0; the next step's solvency
potential was diffed against that. Twice the term's legitimate per-episode
magnitude, and the entire reason it failed to telescope. `engine_stats.reseat_prev_stats`.
Also measured, NOT changed: the tower PBRS term telescopes exactly but Phi(terminal)
is not zeroed (-0.465 per episode at init), so it is not policy-invariant as
documented — it carries an implicit terminal tower-HP bonus aligned with the
timeout tiebreak. `W_ELIXIR_OVERFLOW` never fires at init (P(elixir >= 9) = 0.00%).

**Teacher: the top rung froze on a full bar.** The rung-10 reactive counter
answered every attack, so against an opponent banked on 10 elixir every push
scored negative (30 of 116 passive matches). It switches off after
`COUNTER_PASSIVE_DECISIONS` (8) decisions of opponent inactivity. **Re-measured
after all of today's teacher and stat fixes**, same 29 decks x 4 seeds at rung 10
against a do-nothing opponent: failed to three-crown **30 -> 3**, ran the clock out
**13 -> 0**, mean ticks **1097 -> 615**, 0 crashes. The three are two X-Bow decks
taking two crowns by regulation (slow siege) and the no-win-condition control.

**Cold start** (a random-init net lives at rung 0):
- Rung 0 had no valve at all; 4,000 episodes at 0.00 fired nothing. New
  `floor_alarm` (`>>> [FLOOR]`, `Training/Curriculum_FloorAlarm`) after 1,000.
- PFSP's cold-start branch was unreachable with the floor at 0.0, and the shipped
  deck priors (measured against a trained 2.6 policy) sent a fresh net the four
  hardest decks 59.9% of the time. The no-signal branch is reachable and UNIFORM;
  the priors survive only as `prior_win_rate_vs_26hog_2026_09_03` (not read).
- 30% of `note_progress` calls were dropped behind the scenario return; the
  6->11 remap re-ran on any table-size change; `setup_ab_arm` wrote unstamped
  stages (`--stage 5` became rung 10); the plateau tracker reset on every resume.
  All fixed.

**An unattended run survives now**: a resume error crashes instead of moving the
checkpoint to `.bak` and deleting the TensorBoard log; checkpoint replace retries
Windows `PermissionError` and keeps a `.prev`; Ctrl-C saves; phase 2 follows
`CLASH_WEIGHTS`/`CLASH_LOGDIR` and the handoff is watched for 60 s; **phase 2's
opponent pool is this lineage's snapshots only** (`lineage_started_at`;
`historical_checkpoints/` held 51 from the previous run); `monitor_run` reads win
rate, the floor alarm, reward, rung and modal share; `run_watchdog` follows the
handoff instead of spawning duplicate phase-2 trainers. New
`Placement/ModalShare/<card>` — the collapse detector this file names and nothing
logged.

**The next-card aux loss fights the policy at a fresh start — now ramped in.**
Measured from a random init at the from-scratch config over 12 real updates, with
the aux gradient isolated by a two-pass difference: on the LSTM it grew from 0.62x
to **1.67x** the size of every other term combined (CNN trunk 0.63x -> **2.01x**),
with cosine to them falling -0.37 -> **-0.84** (trunk -0.90). The coefficient was
sized against a CE of ~ln(8); a fresh 185-way head starts at ln(185) = 5.22.
`PPOConfig.aux_warmup_episodes` = 2000 ramps it linearly from 0
(`CLASH_AUX_WARMUP_EPISODES=0` restores the old behaviour). This is a MECHANISM
measurement, not a measured win-rate gain; the refutation experiment is stated at
the constant. Whether the anti-alignment persists after the ramp is open (TODO).

**Champion / Hero abilities are trained (2026-09-16).** Before this a Champion
deck could not be trained at all: nothing sampled an ability and `setup` raised.
`rl/abilities.py` owns the three things that go silently wrong -- ENGINE SLOTS
ARE DECK INDICES 1 AND 2 (a lone Champion at index 2 is head 0 driving slot 2,
not slot 1), the activate arm is MASKED BY READINESS (`is_champion_ability_ready`,
which the env surfaces per slot; an unready activation would otherwise put noise
in the ratio), and the log-prob is computed from the same masked logits at
rollout and update. The ability is part of the JOINT action, so a row counts as a
DECISION when the card head had a choice OR an ability was ready. Phase 2's frozen
opponent samples its own abilities; the phase-1 mirror teacher activates by a
plain heuristic (ready AND enemy force > `DECK_COVERAGE_THREAT_HP` on the board --
readiness alone fires the instant a Champion lands, measured for Golden Knight,
Archer Queen and Monk). With no Champion in the deck there are no heads, no buffer
fields and no extra terms, and that path is unchanged -- pinned by the epoch-0
ratio self-check, now logged every update as
`Loss/Ratio_Dev_First_Minibatch` (must read ~0; a Champion deck measures < 1e-4).

**Preflight gates made honest**: the engine-staleness check compared mtimes and
failed on a verified-current build (content-based now); the side null needed a
deleted checkpoint (a seeded random-init net now, a sharper subject).

**Measured and deliberately NOT changed:**
- **Observation channels 0-7 (assign vs max).** A swarm deck hides **21.3%** of
  troop HP (2.6: 0.8%, control: 1.7%), but max would recover only ~2.3 points —
  the loss is one-value-per-cell, not the write rule. Not worth a late change to
  every HP estimate the teacher and advisor read.
- ~~**The Fireball-keyed spell shaping terms.**~~ **FIXED 2026-09-23** -- see
  the next section.
- ~~**Champion ability training.**~~ **IMPLEMENTED 2026-09-16** -- see below.

**Refuted, do not re-raise:** the recurrent PPO core is sound (ratio at epoch 0
max |r-1| 4.8e-07, values 1e-08, masks recomputed bit-identically, LSTM state
threading correct, placement entropy normalised by the REACHABLE cell count). The
team-1 placement mask is a true mirror (80,784 comparisons, 0 asymmetries). The
zero-gradient `cnn_trunk.6/7.reduce/spread` at init are NOT dead: they sit behind
the zero-initialised `expand`, which does receive gradient.

---

## 2026-09-23: the spell terms follow the deck, and spells hit towers 4x too hard

**`main` could not start a run until this landed.** A session on 2026-09-16 left
TODO 00.3 half-applied and uncommitted: `rewards/shaping.py` read `spell_*` keys
nothing produced, so `compute_shaping` raised `KeyError('spell_in_hand')` on the
first step of every episode. Finished, not reverted:

- **Both spell terms follow the deck's own damage spell**
  (`card_probes.damage_spell`, ranked by MEASURED Crown Tower damage, then cost),
  published by both envs through ONE builder, `gym_wrapper.deck_spell_info`. The
  keys are REQUIRED -- the draft's silent Fireball fallback was the same defect by
  another road. A spell-less deck publishes zeros and both terms are exactly zero,
  never nan. `SPELL_SOLVENCY_RESERVE` is retired: the reserve is the spell's cost.
- **CONTROL: the 2.6 deck is bit-identical.** Same 12 seeded teacher-vs-teacher
  matches through `extract_engine_stats -> compute_shaping`, old code vs new: 0 of
  2,280 steps differ, total and per term. A Rocket-for-Fireball deck read exactly
  zero on all 2,372 steps before; 25 lethal and 12 value steps after.
- **`card_probes.spell_effect` cut damage-over-time spells off halfway** (a fixed
  40-tick read): Poison 368 of its 736, Goblin Curse 129 of 258 -- and its own
  docstring quoted the 368 as proof it reproduced the registry. It reads to
  completion now; across all 22 spells only those two values moved. That doubles
  Poison's damage cap in the advisor target.
- **The teacher aimed every spell with Fireball's disc** (`_top_spell_cells`,
  both spell combos, the rung-0 rules gate) -- the copy the audit's advisor fix
  missed. Engine-scored on 320 mid-match boards, elixir value killed: Rocket
  **+38%** (better on 75 boards, worse on 4), Zap +11%, Poison +7%, Arrows +4%,
  Fireball the same cell on 320/320. `teacher.spell_geometry`.

**SPELLS HIT CROWN TOWERS FOR 100% OF THEIR DAMAGE -- proposed, NOT changed**
(`perception/UPSTREAM_REQUESTS.md` item 29). The real game publishes a separate
Crown Tower damage per spell: Fireball 159 of 688, Rocket 371 of 1484, The Log 41
of 268 ([DeckShop](https://www.deckshop.pro/card/damage), friendly level 11 -- the
level the registry's troop numbers match). `AreaSpell::spellTowerDamageMultiplier`
exists and only Hero Ice Golem's Snowstorm sets it. Measured consequences: a
Princess Tower falls to 4 Fireballs (real: 16); one Rocket on a tower pays the
agent **+0.185** shaping in one step (real: +0.046); in 48 teacher-vs-teacher
matches direct spells are 7.0% of all tower damage (real ratios: 1.3%), 83% of it
The Log, **53% of whose casts roll into a Princess Tower** for 269 (real: 41). The
Python side is already correct under either engine: the lethal window reads
`card_probes.spell_tower_damage`, which MEASURES the tower.

**The general lesson, and it is the audit's own, one level down:** a fix that
derives a quantity is only as good as the instrument it derives from. The advisor
had been moved off Fireball's constants onto `spell_effect` -- whose damage figure
was truncated for exactly the cards (DoT spells) that differ most from Fireball.
And the teacher kept a third copy of the geometry the audit had removed from the
advisor. **When one copy is fixed, grep for the call with its DEFAULTS**
(`spell_catch_map(obs)` with no radius), not just for the constant's name.

## Environment — the things that waste an hour

**`clash_royale_env.pyd` is built for Python 3.11 only.** The default `python`
is 3.13/3.14 depending on the box, and it fails with `ImportError: DLL load
failed`, which reads like a corrupt build and is only a version mismatch. Use
one of:

```bash
python_ai/venv/Scripts/python.exe        # training env — do not modify
perception/.venv/Scripts/python.exe      # perception env
py -3.11
```

Both venvs and the `.pyd` are gitignored build artifacts, so **a fresh clone
has none of them** and neither does every machine — see the toolchain box
below. Where they are absent, Python work is verified by stubbing
`clash_royale_env` (constants only; `python_ai/eval/match_outcome.py`'s tests
are the worked example), plus `py_compile` and static reading. That is enough
for logic, and is not enough for anything that needs real engine dynamics.

**`cmake`, `cl` and `msbuild` are not on PATH.** Rebuild the engine with:

```bash
"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" \
  build_python\clash_royale_env.vcxproj /p:Configuration=Release /p:Platform=x64 /m
```

**Run it from the PowerShell tool, never Bash.** Git Bash's MSYS path
translation rewrites the switches -- `/p:Configuration=Release` arrives as
`p:Configuration=Release` and `/m` becomes `M:/` -- which fails with
`MSBUILD : error MSB1008: Only one project can be specified`. That reads like a
bad project argument and is purely argument mangling.

**The post-build copy into `python_ai/` fails (MSB3073) if any Python process
has the `.pyd` loaded** -- Windows won't overwrite a mapped DLL. That looks
exactly like "the compile is broken" and almost never is. Check
`ps -W | grep -i python` first.

The C++ test suite builds from the same generated solution and runs directly:

```bash
"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" \
  build_python\ClashRoyaleTests.vcxproj /p:Configuration=Release /p:Platform=x64 /m
./build_python/Release/ClashRoyaleTests.exe
```

Measured 2026-09-15 after the pre-launch audit: **714 test cases, 8,169
assertions**, 713 pass and **exactly one fails "as expected"** -- (673 / 6,511 on
2026-08-26)
`test_navigation_wedge.cpp`'s `[!shouldfail]` case, which pins the open
collision-wedge defect. The runner exits 0 in that state; a non-zero exit or a
second failure is a real regression. (It read 650 cases / 6,423 assertions on
2026-08-24 and 619 / 5,907 on 2026-08-21; the counts move as tests are added, so
treat the **shape** -- one expected failure, exit 0 -- as the invariant, not the
number.)

**Adding a test FILE needs the build run TWICE, and it must live in
`tests/core/` or `tests/entities/`.** The glob is
`tests/entities/*.cpp tests/core/*.cpp` (CMakeLists.txt), NOT `tests/**` as this
file claimed until 2026-08-26 -- a new `tests/rendering/` directory is silently
ignored, compiles nothing, and reports a green build. `CONFIGURE_DEPENDS` means
the first MSBuild re-globs and regenerates the vcxproj --
and then links from its pre-reconfigure target list and reports **success** with
the new file absent from the binary. The tag matches zero cases and it looks
like the tests silently failed to register. The second invocation compiles it.
After adding a file, confirm its tag actually matches cases before believing a
green run. (Touching `Board.h` / `CombatEntity.h` / `Tower.h` also forces a full
rebuild of all ~36 test TUs -- budget 8-15 minutes, not 90 seconds.)

> **⚠ THE TOOLCHAIN DIFFERS BETWEEN THE MACHINES THIS REPO IS WORKED ON. Do
> not trust any absolute claim in this section, including this one — run the
> two probes below.** This block has now been rewritten in three directions
> (2026-08-19, 2026-08-21, 2026-08-20 again) because each session described
> *its own* box in a file that is shared, and each read the previous
> description as an error rather than as a different environment.
>
> **Machine A (has MSVC, no WSL)** — verified 2026-08-21 by building the
> `.pyd` and the suite: `C:\Program Files\Microsoft Visual Studio\2022\` holds
> a full `Community` install with `cl.exe` (14.44.35207), `MSBuild.exe` and
> its own `cmake.exe`; `python_ai/venv` exists; `wsl --version` reports WSL is
> not installed.
>
> **Machine B (has WSL, no MSVC)** — verified 2026-08-20, twice, by two
> independent tools:
> ```
> Get-ChildItem 'C:\Program Files\Microsoft Visual Studio\2022'   -> nothing
> Test-Path '...\2022\Community\MSBuild\Current\Bin\MSBuild.exe'  -> False
> ls python_ai/venv                                              -> not found
> wsl --version                                                  -> 2.6.1.0
> wsl g++ --version                                              -> 13.3.0
> ```
> On this box the `.pyd` genuinely cannot be rebuilt and nothing importing
> `clash_royale_env` runs — but the Catch2 suite builds fine under `wsl g++`
> (`-std=c++20 -Wall -Wextra` over `tests/**` plus Catch2's amalgamated
> source, fetched into a scratch dir since it is not vendored). 553 cases were
> built and run that way on 2026-08-20.
>
> **Machine C (has BOTH, and no Python 3.11)** — verified 2026-08-24, and it
> is the reason this box is worth a third entry: the two probes below BOTH
> answer yes, and the machine still cannot run anything that imports the
> engine.
> ```
> Test-Path '...\2022\Community\MSBuild\Current\Bin\MSBuild.exe'  -> True
> wsl --version                                                  -> 2.5.10.0
> py -0p            -> 3.14, 3.13, 3.9   ** NO 3.11 **
> ls python_ai/venv, perception/.venv                            -> not found
> find . -name '*.pyd'  /  ls -d build_python                    -> nothing
> ```
> The `.pyd` is a 3.11-ABI extension, so with no 3.11 there is nothing to
> import it FROM even after a successful build — a different failure from
> Machine B's, reached from the opposite direction. Python work here is
> verified by `py_compile` under 3.13 and static reading, and that is the
> ceiling until the runtime exists.
>
> **Machine D (FULLY EQUIPPED — everything works)** — verified 2026-08-24 by
> building the `.pyd` and the Catch2 suite, running both suites, and running
> `verify_pyd.py`:
> ```
> Test-Path '...\2022\Community\MSBuild\Current\Bin\MSBuild.exe'  -> True
> py -0p                          -> 3.14, 3.13, ** 3.11 present **
> python_ai\venv, perception\.venv                               -> both exist
> python_ai\clash_royale_env.pyd, build_python\                   -> both exist
> ```
> This is Machine C's box after `tools/setup_dev_env.ps1` was run, or one set up
> the same way. C++ suite 650 cases and Python suite 403 passed / 2 skipped both
> execute here. **If your probes look like this, no workaround in this section
> applies to you** — build and run directly.
>
> One trap that only appears on a working box: the MSB3073 post-build copy
> failure is REAL and fires whenever any Python process holds the `.pyd`,
> including a `pytest` run you did not start. The compile SUCCEEDS and only the
> copy into `python_ai/` fails, so `build_python/Release/` has a newer `.pyd`
> than `python_ai/` — which is exactly what "stale .pyd" looks like later.
> Check `Get-CimInstance Win32_Process -Filter "Name like '%python%'"` and read
> the COMMAND LINE before killing anything; it may not be yours.
>
> **`tools/setup_dev_env.ps1` exists for exactly this box** (2026-08-24): it
> installs 3.11, builds both venvs, configures `build_python/` and builds the
> `.pyd` and the suite. Read its Step 3 comment before hand-rolling a build —
> **nothing in this repo generates `build_python/`**, which is the directory
> README.md and this file both tell you to build from. `CMakePresets.json`
> only defines Ninja presets emitting to `out/build/<preset>`, which produce no
> `.vcxproj` at all. The documented MSBuild command assumes a configure step
> that was done by hand once and never written down.
>
> **So: probe, don't inherit.** Two commands settle it in seconds — and on
> Machine C both answer yes, so add a third:
> ```
> powershell -NoProfile -Command "Test-Path 'C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe'"
> wsl g++ --version
> py -0p            # 3.11 present? the .pyd loads on nothing else
> ```
> Whichever answers, use that route. `cl`, `cmake`, `msbuild`, `g++` and
> `clang++` are not on PATH on **either** machine, so `command -v` finds
> nothing regardless and is not a useful test — which is exactly why the
> absolute paths above exist.
>
> **The standing lesson, which both corrections agree on:** an environment
> claim earns its keep only with the command that produced it and that
> command's output. A count is not an inspection (`ls | wc -l` returning 1 for
> a directory whose single entry is `Community` is not evidence of absence),
> and equally, "it is missing on my box" is not evidence it is missing on
> yours. State the probe, not the conclusion.

**The `claude` CLI is not on PATH either** — it's inside the desktop app, at
`%APPDATA%\Claude\claude-code\<version>\claude.exe`. Needed for
non-interactive plugin/marketplace management, since `/plugin` only opens an
interactive dialog.

**No GPU.** `torch==2.13.0+cpu`, i5-13420H. Budget work in gradient steps, not
hours — see "Measured baselines" for the current rate (2,873 episodes and 92
PPO updates per hour at `num_envs = 8`). A full phase 1 to ~60k episodes is
~21 h.

---

## Engine facts worth knowing before you measure anything

**10 ticks = 1 second.** Derived from `CardStats::attackCooldown`, which is
stored in ticks while the real game publishes the same figure in seconds —
Archers 9/0.9s, Musketeer 10/1.0s, Knight 12/1.2s, Valkyrie 15/1.5s, Hog
16/1.6s, King Tower 10/1.0s. 132 independent rows agreeing on one ratio, which
is a far stronger source than any single constant. Single conversion point:
`perception/timebase.py`.

**Elixir runs ~2% slow.** `ELIXIR_REGEN_RATE = 0.035`/tick × 10 ticks/s =
2.857 s per elixir, against the real game's 2.8 — and 2.80 s is *measured* off
the recordings, not assumed. Don't "fix" one to match the other without
revisiting `perception/track/opp_elixir.py`, whose missed-placement alarm
depends on using the real rate.

**The match runs 1x / 2x / 3x elixir, since 2026-09-02.** It was a flat 1x for
this project's whole history — the simulator's economy was the opening two
minutes of a real match stretched across all six.
`GameManager::elixirMultiplierAtTick` is the single definition:
**`DOUBLE_ELIXIR_TICK = 1200` (2:00)** and **`TRIPLE_ELIXIR_TICK = 1800`
(3:00)**, the REAL game's schedule (3:00 regular with double from 2:00, then
overtime at triple), not a rounder one.

**Elixir was STARVED, which is what justified this** — measured over 4,229
decisions of `model_weights_phase4.pth` at stage 3, before the change:

| | mean | median | P(≥ 9.0) | P(≤ 4.0) |
|---|---|---|---|---|
| agent | 2.16 | 1.95 | **0.0%** | **91.0%** |
| teacher | 1.81 | 1.45 | **0.0%** | 91.3% |

Zero overflow in 4,229 decisions — `W_ELIXIR_OVERFLOW` never fired once. Both
sides spent every drop the tick it arrived, so income genuinely was the binding
constraint and added income gets spent rather than discarded. **Had the bars
been pooling near 10.0 this change would have been refuted**, and that is the
measurement to re-run before proposing any further economy change: pooling
means the constraint is decision-making, not income.

**And it is why Fireball was correctly unplayed.** Best-case catch, computed as
an upper bound (perfect information, best of all 612 centres): **median ONE
unit**, P(≥3 units) = 15.5%. A 4-elixir spell whose best possible play usually
catches one target is not +EV, so declining it was right about the environment,
not a training failure.

**Read the phase placement honestly, because one of the two barely matters.**
Match end tick is mean 1758 (2:56): **92% of matches reach 2:00 but only 8%
reach 4:00**, and by share of ticks actually played, **32.1% fall after 2:00
against 3.0% after 4:00**. Double elixir is a third of all gameplay; triple at
the real 3:00 is ~8%, and triple at 4:00 — the placement originally proposed —
would have been 3% and effectively inert.

**The causal link "more elixir → bigger clusters" IS now measured, and it holds
modestly.** Same policy, same stage, same 24 episodes across the change, so the
engine is the only variable:

| | flat 1x | 1x/2x/3x |
|---|---|---|
| enemy units on board, mean | 3.62 | 3.89 |
| best Fireball catch, **median** | **1** | **1** |
| P(catch ≥ 3) | 15.5% | 21.8% |
| P(catch ≥ 4) | 6.1% | 10.4% |
| agent elixir, mean | 2.16 | 2.50 |
| agent P(≥ 9.0) | 0.0% | **0.6%** |

The gain is real but lives in the **tail** — P(catch ≥ 4) rises 70% relative,
while the **median best-case Fireball still catches exactly one unit**. Read
that median before expecting Fireball to become obviously correct. And this is
the pre-change policy throughout: it measures whether the ENVIRONMENT makes
clusters at unchanged behaviour, not what a retrained policy would do.
`DEFAULT_DECK` still bounds it — 2.6 Hog Cycle is nearly all single-body cards,
so no amount of elixir makes a swarm in a mirror. What this buys Fireball is
more 4-cost support alive at once (killing a Musketeer is an even trade plus
chip), which is real but smaller than swarm-clearing.

**Overflow became possible for the first time** (0.0% → 0.6%), so
`W_ELIXIR_OVERFLOW` now fires where it never had — and that broke the premise
of `test_aux_task_is_not_a_memory_probe.py`. That file's 2026-08-27 finding was
that the deleted opponent-elixir head measured nothing, since the target is an
affine function of two present scalars (MAE **0.0000**). Both halves moved:
`rate*t` is no longer one slope, so the original basis now scores **1.4265** —
no better than predict-the-mean — and integrating the schedule restores the
exact fit; and the opponent now overflows unaided, which is genuinely
unrecoverable because the cap discards elixir no scalar records (per episode:
never-capped residual **−0.035**, capped **+0.76 / +1.63 / +5.38**). The
conclusion survives in the overflow-free regime; a rising MAE under phases is
an **overflow detector**, still not a memory diagnostic.

The phase **composes** with the curriculum's `oppElixirMultiplier` rather than
replacing it — a 1.5x stage-5 opponent in double elixir gets 3.0x. That is
intended: the phase is a property of the match clock and applies to both
players; the multiplier is a per-opponent handicap on top.

**GAMEPLAY-AFFECTING, and the economy is the substrate every card's value sits
on** — relative card value, how many cards are worth holding, when a push is
affordable. Every win rate in "Measured baselines" was earned under a different
economy, and the curriculum's win-rate gates are calibrated against a teacher
that now plays a materially different late game.

**`NUM_EXTRA_SCALARS` 9 → 10** for the multiplier (normalised by
`MAX_ELIXIR_MULTIPLIER`), so `observation_size()` **13976 → 13977** and, because
this one was appended to the extra scalars rather than to the end,
`CYCLE_START` **13606 → 13607**. Every checkpoint needs migrating, but cheaply:
`python_ai/tools/migrate_checkpoint_elixir_phase.py` zero-pads
`scalar_mlp.extra` from `Linear(9, 12)` to `Linear(10, 12)` — **120 of
1,900,165 parameters**, and the migrated net is bit-identical on every
observation it could already see. **It pads the Adam moments too, and that is
half the job**: the weight alone produces a file that loads a model fine and
then throws on `optimizer.load_state_dict`.

**`MAX_MATCH_ELIXIR` 140 → 280, and this was a latent defect the change would
have tripped.** 140 was sized against a flat-1x match (`3600 × 0.035 + 5 =
131`). Phased income reaches ~278, so **both** elixir-spend scalars would have
saturated at 1.0 partway through every match and stayed there — the agent going
blind to the economy exactly when it decides the game, with nothing raising.
Same failure class as the constants the 2026-08-07 speed fix invalidated: a
normaliser calibrated against a measurement a later change moved.

**Troop movement was 4-5× too fast until 2026-08-07.** `CardStats::speed` is
tiles per *tick*, so the registry's Giant `0.3f` meant **3.0 tiles/s** against
a real-game Slow of ~0.75 — a Giant crossed bridge-to-tower in ~3.5 s.
`MOVEMENT_SPEED_SCALE = 0.2f` in `CardStats.h` now converts the registry's
tier literals into real-game tiles/tick. It is applied at **three** sites:
`CardRegistry::troop()` plus both `stats.speed` assignments in
`include/core/SpiritEmpressForms.h` (the file is in `core/`, not `entities/`),
which set the speed directly and so bypass `troop()`.

Those two are raw literals rather than `SPEED_*` tiers, and that is **correct,
not an oversight**: Spirit Empress is one of the 13 cards on the 2026-08-24
speed rework's "no official row" list, deliberately left off-tier. `0.85 × 0.2`
and `0.5 × 0.2` give 1.700 and 1.000 tiles/s, both in that rework's documented
off-tier set. Verified 2026-08-24.

Measured three independent ways against the 8 recordings, all agreeing
(`perception/UPSTREAM_REQUESTS.md` item 9 carries the evidence): per-card speed
off real footage, the engine's own Slow:Medium tier ratio, and a time-scale
sweep in `perception/tools/sim_fidelity.py` whose optimum moved from 0.2 to
**1.0** across the fix — the end-to-end confirmation that the engine's clock
and the real game's now agree.

Three things worth carrying forward:

- **The C++ suite did not catch this and cannot.** All 504 cases passed before
  and after. `test_troop.cpp` builds `MeleeTroop` with a literal speed and
  nothing anywhere asserts a registry speed constant — the suite covers the
  movement *mechanism* and is blind to the *data registry*. 207 grep hits for
  "speed" across 16 test files are not coverage. The guard lives on the
  perception side instead.
- **It is a rebalance, not an accuracy fix.** Cooldowns and elixir regen were
  already right, so slowing movement alone means a troop absorbs ~5× more
  shots crossing a defender's range, tanks take ~5× more tower damage for the
  same ground, ~5× more elixir accrues per push, and timeouts get much more
  common. Every win rate in "Measured baselines" predates it.
- **`skip_frames = 10` hurts ~5× less now.** One second covers ~5× less board,
  so open problem #4 shrank by that factor for free.

**The speed fix exposed a latent deadlock at the bridge mouths, fixed
2026-08-09.** Troops occasionally froze mid-crossing for 10+ seconds with no
enemy within 10 tiles. Two pieces of code disagreed about what "arrived" means:

- `Board::getNextWaypoint` classified sides with `currentPos.y <= riverY_start`
  — **inclusive** — so a unit standing exactly on the near bank was still
  "below" and was handed `{bridgeX, riverY_start}`, *the point it already
  occupied*.
- `Troop::moveTowards` refused to move when `distToWaypoint > 0.01f` was false.

Position unchanged → identical waypoint next tick → **absorbing state**. A
0.01-radius trap disc at each of the four bridge mouths, escapable only by
retargeting, death or a collision nudge.

It is a genuine *latent* bug, not something the speed change introduced: the
chance a step ends inside the disc is ≈ `0.01 / step size`, so it went from
~3% at the old 0.3 tiles/tick to ~17% at 0.06 — a 5× rise exactly tracking the
5× slowdown. Measured 2 events in 4 post-fix replays (41,954 unit-ticks), 0 in
3 pre-fix replays (7,512 unit-ticks), at `(4.00, 15.50)` and `(14.01, 15.49)`.

Fixed by making the two sites share one `Board::WAYPOINT_ARRIVAL_EPS` and
having `getNextWaypoint` hand back the *far* bank once a unit has arrived at
the near one. **Gameplay-affecting** — win rates from before it are not
comparable. Regression tests in `tests/core/test_board.cpp` sweep both bridge
mouths at finer-than-epsilon resolution; they fail on the old code with
`0.0f > 0.01f`.

The general lesson: **two independent copies of "close enough" is a deadlock
waiting for the right step size.** Anywhere a mover's stop-condition and a
planner's arrival-condition are separate literals, they can disagree.

> **AND THIS FIX WAS INCOMPLETE. A SECOND absorbing state, in the same function,
> was found on 2026-08-20** -- it guarded the two branches where a unit stands on
> the bank it is LEAVING and left the branch where the unit is arriving at the
> FAR bank unguarded, which stalled ~20% of lone ground crossings. See the
> 2026-08-20 simulator-audit section. When a fix of this shape lands, sweep
> EVERY branch of the function, not the one the reproduction happened to take.

**Still wrong, unmeasured, and in the same direction:** `Projectile`'s own
`speed` was never recalibrated alongside the movement fix.

**SPEED TIERS REACH SPAWNED UNITS TOO, since 2026-08-26 -- the 2026-08-24
rework did not.** That rework round-tripped "109 / 109 match", and 109 is the
count of cards with an OFFICIAL ROW; the child `CardStats` that death effects
spawn have none and were never in the set. Measured with
`tools/audit/spawn_speed_audit.cpp` (spawn every card, fire its death effects,
read `Troop::getSpeed()` off what arrives): **8 of 109 units off-tier, now 6.**

- **Golemite** was a raw `0.2f` = 0.400 tiles/s, BELOW `SPEED_VERY_SLOW`
  (0.663) -- slower than any card in the real game's published table, and 2.5x
  slower than the Golem it splits from. Now `SPEED_SLOW`, mirroring its parent
  (the rule the rework itself used for the 9 Hero variants).
- **Bats** needed no external source: the registry contradicted ITSELF. Playable
  card id 78 is `SPEED_VERY_FAST` (2.651); the child stats a Night Witch
  releases were `0.85f` (1.700). Same name, same 81 hp / 81 damage / 12-tick
  cooldown, 36% apart. Now matches card 78.
- **Still open: Goblin Brawler (-31)** at 1.400, 5.6% off MEDIUM. No playable
  counterpart and no official row, so there is nothing to make the call from --
  deliberately not guessed.

**AND THE SAME PASS MISSED NINETEEN MORE, found 2026-08-26 by a different
question.** "Is this speed near SOME tier" is far weaker than it looks: a unit
that should be MEDIUM but sits on SLOW passes it, because SLOW *is* a tier.
`0.5f` resolves to 1.000 tiles/s, 0.6% off `SPEED_SLOW` -- so nineteen child
helpers carrying pre-rework literals all read as "on tier" while sitting on the
WRONG one. The question that finds them needs no external source at all: **does a
spawned unit agree with the playable card of the same name?**

| unit | its card says | the spawned copy ran at |
|---|---|---|
| Goblins | 2.651 (VERY_FAST) | 2.000 |
| Spear Goblins | 2.651 (VERY_FAST) | 2.000, and one helper at 1.000 |
| Barbarians | 1.325 (MEDIUM) | 1.000 |
| Phoenix | 1.325 (MEDIUM) | 1.000 |
| Skeletons | 1.988 (FAST) | 2.000 |

Every hut-spawned Goblin, every Barbarian a Battle Ram or Barbarian Barrel
dropped, and the Phoenix that rises from its own egg were all a quarter slower
than the card that spawned them. Fixed by giving each helper its card's
`SPEED_*` constant; pinned by *"every spawned unit moves at the speed of its own
playable card"* in `tests/core/test_card_registry.cpp`, which is generic rather
than a list, so a new helper cannot reintroduce it.

**DO NOT read this off the source.** A grep for non-`SPEED_*` literals finds 54
and looks like a huge gap; the measurement finds 8. Most of those literals land
within 1% of a tier by coincidence (`0.5f` -> 1.000 against SLOW's 0.994,
`1.0f` -> 2.000 against FAST's 1.988). The first inference in this audit was
seven times too large. Run the instrument.

**A freeze lasts exactly as many ticks as it says, since 2026-08-26.** It did
not before: `freezeTicks` had two readers straddling its own decrement --
`CombatEntity::update` drained the attack cooldown *before* it, `Troop::
moveTowards` re-read it *after* -- so `applyFreeze(N)` slowed attacks for N ticks
and movement for N-1, and a 1-tick stun did not stop movement at all. Movement
now reads `CombatEntity::frozenThisTick`, published once at the top of update().
GAMEPLAY-AFFECTING; every registered stun (Zap 5, Electro Spirit 3, Ice Spirit
10, Freeze 40) got one tick longer in movement terms.

The general rule this adds to the two bridge-mouth absorbing states: those were
*two copies* of one number, and the rule written down for them ("two independent
copies of close enough is a deadlock waiting for the right step size") does not
cover this, because here there was only ONE copy. What differed was **when each
reader sampled it**. So: *one fact, two readers, and a write in between* is the
same hazard, and grepping for duplicate literals will not find it.

**A deployed building dies exactly at its lifetime, since 2026-08-26.** Decay is
`maxHp / (lifetimeTicks / 10)` per second in INTEGER arithmetic and nothing
consulted the clock, so a building lived until repeated subtraction happened to
finish it: a Cannon (824 hp / 300 ticks) decayed 27/s, sat on 14 hp at 30.0 s and
died at **31.0 s**. `Building::update` now expires on `ticksAlive >=
lifetimeTicks`, with `hp = 0` rather than `takeDamage()` -- expiry is not damage,
so a shield must not absorb it and no OnDamageTakenEffect fires for a clock
running out. It read as correct because the one test covering it used
`hp = 3000, lifetime = 300` and 3000/30 = 100 exactly, while being NAMED "fully
decays to 0 exactly at its configured lifetime". Same failure class as the
`sight >= attack` blind spot: a check that cannot fail for the inputs it is
given.

**Two 2.6-deck cards were the wrong card, until 2026-08-26.**

- **Ice Golem's slow is on its DEATH EXPLOSION, not its attack.** It carried
  `.withOnHit(FreezeOnHit(30, 0.65f))` -- Ice Wizard's effect, copied along with
  a "same story as Ice Wizard" comment, while that same comment said the death
  slow was "not modeled". Both halves wrong in the same direction, and the
  direction is what costs: an Ice Golem targets BUILDINGS, so the phantom slow
  landed on a Crown Tower or a Cannon -- something that cannot walk out of it --
  and refreshed every 2.5 s hit, a standing 35% cut to the fire rate of whatever
  it tanked. `AreaDamageOnDeath` now takes the same optional on-hit slot
  `AreaSpell` already had (nullptr default, every other caller bit-identical).
- **Ice Spirit STUNS (0.0f), it does not half-slow (0.5f).** This engine has one
  convention for each and they are not interchangeable: a stun is
  `FreezeOnHit(ticks, 0.0f)` at every other site (Zap, Electro Spirit, Zappies,
  Freeze), and 0.5-0.7 is the SLOW band (Ice Wizard, Ice Golem's explosion). The
  Evolution's 51-tick window was collapsed to the base card's 10 at the same
  time: 51 ticks at 0.0f would be a 5.1-second unbroken hard stun off a 1-elixir
  card, a bigger invention than the delayed second pulse it stood in for.

**`pullToward`/`pushAway` refuse a non-positive distance, and nothing may move
an entity by writing `position` directly (2026-08-26).** Both rules exist for
the same reason `Entity.h` already gives for the building exemption -- enforced
once, centrally, because a per-call-site check is one somebody forgets.

Callers compute the distance as "how far do I still have to close"
(`dist - meleeRange`), which goes NEGATIVE against an already-close target, and
`moveBy = (distance < dist) ? distance : dist` then took the negative and ran
the pull BACKWARDS. `GoldenKnightDashEffect` did exactly that: measured, a
Golden Knight 0.5 tiles from his target finished the ability at 1.0 -- retreating
from what he dashed at, ten times over.

And `HeroGiantHurlEffect` wrote `victim->position.x = ...` directly, skipping
`exemptFromForcedMovement` entirely, so it threw enemy BUILDINGS across the
arena (measured: a Cannon from x=6.0 to x=11.0). Its selector
`findHpExtremeEnemy` also excluded only `isTower()` while its own comment said
"enemy TROOP"; it now excludes `isBuilding()`. Lane mirroring goes through
`mirrorToOppositeLane()`, third member of the pull/push family and carrying the
same guard.

**Two one-time effects were re-arming on every application (2026-08-26).**
`CursedHogOnHit` wrapped the victim's `deathEffect` in a fresh
`CompositeDeathEffect` per HIT, so N hits nested N composites and spawned N hogs
on death (measured: 3 hits, 3 hogs). `RoyalChefBuffEffect` re-fed whichever ally
was nearest, compounding `hp += hp / 10` every 280 ticks (measured: 1000 -> 1100
-> 1771 over six servings, +77%, unbounded in match length). Both are now
latched -- `CombatEntity::curseDeathSpawnAttached` and `royalChefServed`. The
curse's DURATION still refreshes on every hit; only the spawn is one-time.

**`MatchRules::evaluate` identifies the King by TYPE, not by symbol, since
2026-08-26 -- and `'R'` is NOT unique to the King.** Card id 93 (Mortar) is
registered with symbol `'R'` too, so a living Mortar reported its owner's King as
alive: with that King destroyed, `evaluate()` returned "not over",
`cleanDeadEntities` erased the King the same tick, and from then on the ONLY
thing answering was the Mortar. The match did not end, and the win was converted
into a timeout that `TimeoutRules` then decided on towers instead. Reachable in
ordinary training -- Mortar is in the registry and phase 1's `random_opponent`
samples random decks. The guard is now `isTower() && symbol == 'R'`, which is
what `Tower::update`'s Princess count and `TimeoutRules::resolve` already did.

**The four existing MatchRules tests could not have caught it**, because they
built their stand-in King as a `DummyEntity` wearing `'R'` -- which is precisely
what a Mortar is at runtime. The double reproduced the bug and then asserted it
was correct. They build real `Tower`s now. **A test double that is a
non-instance wearing the discriminator cannot test the discriminator.**

**Deploy time WAS the other half of this and is now FIXED (2026-08-19)** --
`DEPLOY_TIME_TICKS = 10`. It turned out to be the mathematical flaw suppressing
win conditions, because a missing deploy second is a subsidy paid to the
DEFENDER on every placement. See the 2026-08-19 deploy-time section below for
the controlled measurement.

**Observation changed on 2026-07-29.** `NUM_CHANNELS` 9 → 21,
`observation_size()` 6253 → **13606**, `NUM_EXTRA_SCALARS = 9` appended after
the one-hots. Channels 0-8 keep their old meaning; 9-20 are per-team attribute
channels indexed via the bound `CH_*` constants. Checkpoints older than that
date are architecturally dead (`python_ai/archive_pre_obs_v3/`). `NUM_CARD_IDS`
is 185.

**AND AGAIN ON 2026-08-27: `observation_size()` 13606 → 13976.** Two
`NUM_CARD_IDS`-wide blocks appended, carrying what the OPPONENT has played —
`UPSTREAM_REQUESTS.md` item 24, the fix for this file's own open problem #3:

```
CYCLE_START + c                    seen[c]     1.0 once they have played card c
CYCLE_START + NUM_CARD_IDS + c     recency[c]  exp(-(now - lastPlayed)/200 ticks)
```

**Every checkpoint trained before this date is architecturally dead**, the
same way the 2026-07-29 change killed the ones before it: `scalar_size` goes
754 → 1124, so `scalar_mlp`'s first layer changes shape.
`load_state_dict_flexible` will warm-start everything else and report the
discard, which is the correct behaviour, not a workaround — the policy has to
relearn what to do with the new input.

**Why this is not cheating**, which is the design constraint the whole feature
turns on. The encoder already draws the line correctly for elixir: the agent
gets the opponent's cumulative SPEND ("you see every card they play and you
know what it costs") and never their current bar. By that same test a card's
IDENTITY is observable — a human watching the screen knows it was a Hog and
knows roughly when it can be back. The encoder was keeping the cost sum and
discarding the identity, which is the one summary that destroys cycle
information. The opponent's HAND stays hidden.

Three things worth carrying:

- **The hook is `GameManager::playCard`, not `ClashEnv`.** `HeuristicOpponent`
  reaches `playCard` directly rather than through `ClashEnv::step`, so hooking
  the env would have missed every opponent play in phase 1 while looking fine.
  Same shape as the five damage entry points behind `Tower::awake`'s HP latch.
- **`inject` deliberately does NOT record a play.** It places a BODY and
  bypasses `playCard` by design. `ClashEnv::notePlayedCard` (bound as
  `note_played_card`) is the separate recorder `perception/` must call, or the
  channel works in training and is silently empty in deployment.
- **`CARD_ID_COUNT` now lives in `CardRegistry.h`**, with
  `ClashEnv::NUM_CARD_IDS` as an alias, because `GameManager` cannot include
  `ClashEnv.h` and a second literal `185` was the alternative.

**AND IT DETONATED A LATENT DEFECT — `UPSTREAM_REQUESTS.md` item 25.** Five
Python call sites located the extra-scalar tail by subtracting from the end,
`observation_size() - NUM_EXTRA_SCALARS`, which is correct only while those
scalars are LAST. Appending the cycle blocks behind them made every one read
card-recency floats instead — including two reading `enemy_tower_hp`, which
feeds `compute_shaping`'s tower potential, so **the reward itself would have
gone quietly wrong with nothing raising**. Fixed by binding forward offsets
`EXTRA_SCALARS_START` and `CYCLE_START`, and deriving `observationSize()` from
them so the size and the offsets cannot disagree.

The rule this adds: **locate a section of the observation by a FORWARD offset,
never by subtracting from the end.** This layout only ever grows by appending,
so a forward offset is stable by construction and a backward one is a
scheduled defect — it survives until the next append and then fails silently.

**Combat is deterministic.** The only RNG in the engine is the opening-hand
shuffle (`PlayerState::initializeDeck`) and `HeuristicOpponent`. Nothing in
`include/entities/` is random — so identical inputs give identical outcomes,
which is what makes the perception bridge's zero-divergence control possible.

**Board geometry — all of it lives in `include/core/ArenaLayout.h` since
2026-08-21, and is bound to Python.** River `[15.5, 17.5)`. King `x` **8.5**,
Princess Towers **3.0 / 14.0**, bridges **2.5 / 14.5**, `BRIDGE_Y` 16.5.

x is a **cell index** clamped to `[0, WIDTH-1]`, so the board's centre — the
fixed point of the mirror `17 - x` — is **8.5**, not 9.0. That is the x-analogue
of `extractObservationForTeam`'s `y -> 33 - y`, and the convention
`isBackRowDeadZone` already used. Every pair mirrors: 3.0 ↔ 14.0, 2.5 ↔ 14.5,
8.5 onto itself.

**A two-tile bridge's centre sits on the SEAM between its tiles**, which is why
2.5 and not 3.0: `clampToBoard`'s `±BRIDGE_HALF_WIDTH` then spans exactly cells
2 and 3, reproducing the real river row `WWBBWWWWWWWWWWBBWW`. Centred on a tile
it was three columns wide.

This **reverses** the 2026-07-30 move (left Princess 3.0 → 4.0, Kings 8.5 → 9.0)
that `perception/UPSTREAM_REQUESTS.md` items 1-2 recorded. That fit was anchored
on the same half-tile convention error, which is why its residual looked small
(0.63 → 0.31 tiles) while the layout was symmetric about the wrong centre.

**Nothing may keep a second copy.** Four did, and all four were stale:
`HeuristicOpponent`'s bridges (3.5/13.5 — already wrong against Board's own
4.0/14.0 before the correction), `tactics.py`, `perception/geometry.py`, and
`test_sim_driver.py`'s observation reads. `ArenaLayout` is bound as
`clash_royale_env.ARENA_*` plus `arena_king_y(team)` / `arena_princess_y(team)`,
surfaced through `python_ai/engine_constants.py`. Derive; do not restate.

**...and there were SIX, not four. `web/viewer.html` was the sixth, found
2026-08-21.** It hardcoded `[[3, 4], [13, 14]]` — the arena as it stood BEFORE
the re-centring — and painted **four of the eighteen columns wrong in both
directions**: 2 and 15 are real bridge and were drawn as water, 4 and 13 are
water and were drawn as bridge. Same shape as the observation-encoder bug
(§ "The observation encoder was NOT updated with the arena"), found the same
day; this was the copy nobody thought to check after that one was fixed.

Order of discovery, since the count in this file keeps moving: the four above,
then `extractObservationForTeam`'s channel 8, then the viewer, then (2026-08-26)
`python_ai/envs/scenario_offense.py` -- stale and default-OFF -- and
`include/rendering/TerminalRenderer.h`, which painted the bridges at columns
3-5 and 13-15 against a real 2-3 and 14-15: **four of eighteen columns wrong**,
and a three-tile bridge, the shape from before the 2026-08-21 re-centring.

The renderer is the one that should never have been a copy at all. Unlike the
viewer it holds a `const Board&`, and `Board::isOnBridge` is public -- it could
always have derived. It now does, via `TerminalRenderer::riverRow(board)`.

**THE VIEWER IS A DIFFERENT KIND OF COPY, AND THAT IS THE PART WORTH
CARRYING.** The other five were all reachable by the rule this section states.
`ArenaLayout` is bound to Python, so `geometry.py`, `tactics.py` and
`engine_constants.py` *can* derive, and the encoder lives in the engine itself
and could be made to share `Board::isOnBridge`. **The viewer can do neither.**
It is a `file://` HTML page whose only input is a replay JSON, and

    the replay JSON carries no arena geometry at all.

So "derive; do not restate" is not advice the viewer is able to follow. It is
*structurally required* to restate, and therefore structurally guaranteed to go
stale — it was simply a question of which arena change would do it. Discipline
was never going to prevent this one.

Two consequences:

- **The honest fix is to put the geometry in the replay** — `GameLogger::save`
  already writes a live `cardMeta` block from `CardRegistry` for exactly this
  reason, and an `arena` block from `ArenaLayout` would close the hole the same
  way. That is a C++ change and so a proposal, not a change made in passing.
- **Until then the viewer's copy is annotated with the header that owns each
  value** (`ArenaLayout.h`'s `LEFT_BRIDGE_X`, `Board.h`'s `BRIDGE_HALF_WIDTH`
  and river band), which is the fallback `perception/geometry.py` already uses
  and the best available when derivation is impossible.

**And the detector is cheap: compare RENDERED PIXELS against the engine's own
river row.** Sampling the drawn canvas across the river band and classifying
each column warm/cool must reproduce `WWBBWWWWWWWWWWBBWW` — the same string the
encoder fix was verified against. A test that pins the expected *columns*
instead would need hand-editing on the next arena change and would go stale
exactly the way the thing it checks did.

**The general rule this adds:** a second copy is unavoidable wherever a
consumer cannot reach the source of truth. Ask of every such copy not "is it
correct" but *"could this consumer derive it if it wanted to?"* — where the
answer is no, the copy is a scheduled defect, and the fix belongs in the
FORMAT, not in the consumer.

**The team-1 observation was displaced one row until 2026-07-31.**
`extractObservationForTeam` mirrored the *truncated row* rather than the
*mirrored position* — `33 - int(y)` instead of `int(33 - y)`, which agree only
when `y` is an integer. Every troop sits at a fractional `y`, so team 1's whole
observation was off by one row, every tick, for its own and enemy units alike,
while team 0's was correct. The river marker was separately on row 17 for team 0
and row 16 for team 1. Both fixed; `UPSTREAM_REQUESTS.md` items 5-6 carry the
evidence.

Two things make this worth remembering. **It hid from a coordinate audit**: the
positions were symmetric the whole time (King 2.5 ↔ 30.5, Princess 6.0 ↔ 27.0),
and the Princesses mirrored *correctly* because 27.0 is an integer — only the
fractional-`y` King looked wrong, which reads like a tower bug rather than an
encoder bug. **And it was invisible in every training metric**, because the
trainee is always team 0: everything looked healthy while every opponent played
blind. The test that found it is worth keeping — run a policy against a
bit-exact copy of itself and check the score is 0.50. It measured **0.598**
before the fix and **0.520** after (n=400, 95% CI [0.471, 0.569]).

**The King Tower sleeps, since 2026-08-21.** It was dormant-less for this
project's whole history — `Tower.h` gave it no activation condition and it fired
from tick 0, which is why every earlier passage says to exclude the Kings when
comparing against real footage. **That caveat no longer applies.**

`Tower::awake` is a latching flag: Princess Towers construct awake, the King
asleep (only `GameManager::addTower`'s `symbol == 'R'` branch ever builds one).
It wakes permanently on either real-game trigger — any damage, or any Princess
Tower on its own team being destroyed — and while asleep `Tower::findTarget`
returns `nullptr`, so it can neither acquire nor fire. It stays targetable and
damageable throughout, which is what lets the damage trigger fire at all.

Two implementation choices worth not undoing. The damage trigger latches on the
HP invariant `hp < maxHp` rather than overriding a damage entry point, because
damage reaches a tower through five of them (`Projectile::applyHit`,
`AreaSpell::update`, direct `performAttack`, splash, poison DoT) and hooking one
would silently miss the rest; being a latch, a heal cannot re-sleep it. And the
Princess trigger records the team's living Princess count on its FIRST update
rather than testing `< 2`, because a King on a bare board — which every unit
test builds — has zero Princesses and the naive rule wakes it on tick one.

**Measured** (`tools/audit/king_activation_audit.cpp`), lone Hog, identical
placement, only the Kings' dormancy varying:

| | tower damage | Hog dies at |
|---|---|---|
| Kings awake (old) | 1268 | tick 122 |
| **Kings dormant (new)** | **2219** | tick 170 |

**+951 damage, +75%.** The 1268 reproduces the figure the 2026-08-20 audit
recorded, which is the cross-check that makes the pair trustworthy. Defence is
materially weaker than every win rate in this file was earned against.

**Sight range is per-card, complete, and 5.5 is a SOURCED value — not a
fallback.** 46 of the 148 non-spell cards carry an explicit `withSightRange`
across 9 values (4.0–11.5); the other 102 read `CardStats::sightRange`'s 5.5,
which commit `1b17844` established IS the catalogue's value for them ("the main
standard for most cards") after cross-referencing the whole registry. Do not
read a default as an omission here — that hypothesis was raised and refuted.

**For a building-targeter, `sightRange` IS the Cannon-pull radius.**
`BuildingTargeter::findTarget` diverts to the closest non-tower building within
`effectiveSightTo`, so this one number decides how far off-lane a defensive
building can sit and still drag a win condition off the tower — the interaction
`tactics.py`'s entire Cannon rule is built on. Measured by trajectory deviation
against a no-Cannon control (`tools/audit/pull_range_probe.cpp`): Hog Rider
(9.5) pulls at **10.5** tiles, Balloon (7.7) at 8.75, Giant (7.5) at 7.25, and
every card on the 5.5 default at **~5.25** — half the Hog's. Read the pull
BEHAVIOURALLY, never as "the Cannon lost hp": `Building::update` decays it every
10 ticks on its own, so a damage test reports a pull at every offset and the
sweep saturates.

**SIGHT IS STRICT CENTRE-TO-CENTRE, since 2026-08-28.** `effectiveSightWith`
returned `sightRange + myRadius + effectiveRadiusOf(target)`, inflating every
unit's aggro radius by both hitboxes -- a Hog Rider's catalogued 9.5 became
**10.9** against a Cannon, a 15% free extension applied game-wide. Radii answer
a HITBOX question ("can these two touch"); the published sight ranges are
centre-to-centre, and 9.5 must mean 9.5. The registry's values were already
correct and match the published table exactly (9 distinct values, 4.0-11.5) --
only the COMPARISON was wrong, so fixing the one function fixes every card
systemically.

**One floor remains, and it is not a softening.** Attack range in this engine
IS surface-to-surface (`effectiveRangeTo`), so a unit whose attack REACH
exceeds its sight can hit what it cannot acquire -- it never targets and stands
idle. That is exactly the 2026-08-20 free-siege defect below. Sight is
therefore floored at the unit's own attack reach -- but **only for a card whose
raw `sightRange` already covers its raw `attackRange`**, which the catalogue
guarantees for all 148. Where it does not hold the unit stays blind past its
sight, and that guard is load-bearing: `test_combat_entity.cpp` builds dummies
with `attackRange = 20` against the 5.5 default precisely to pin that a huge
attack range does NOT buy vision, and an unguarded floor makes them see
everything. The floor binds only where `attackRange` is within ~2 tiles of
`sightRange` (towers, Musketeer-likes); for every long-sight card it is
irrelevant -- a Hog's floor is `0.8+0.4+1.0 = 2.2` against a sight of 9.5.

**Measured on the reported case** (Hog spawned at (14.0, 17.5), Cannon at
(5.0, 11.0)):

| | acquires at |
|---|---|
| old, radius-inflated | 10.783 tiles |
| **new, strict** | walks past 9.789 untouched, acquires at **9.160** |

**AND THE CROSS-LANE PULL SURVIVES IT -- for a geometric reason, not a defect.**
The lanes sit at `x = 3.0` and `x = 14.0`, so a unit walking one lane passes
**9.0 tiles** from a building in the other, and 9.0 < 9.5. No euclidean sight
value at or above the Hog's catalogue figure can exclude it. Making a win
condition immune to an off-lane Cannon requires **PATH distance** (down, across
a bridge, back up -- well over 14 tiles), which is a different mechanism and is
NOT what this change implements. Do not re-derive this: the euclidean rule is
now exactly the catalogue, and the remaining pull is the arena's geometry.

GAMEPLAY-AFFECTING: every unit's aggro radius shrank by roughly its own plus its
target's radius, so defensive-building pulls, kiting distances and every win
rate earned before this date describe a different engine. Pinned by
`test_sight_range.cpp`'s `[centre_to_centre]` cases, including a parameterised
sweep over four catalogue values (9.5 / 7.7 / 7.5 / 7.0) that bounds the rule
from BOTH sides, so a uniformly-blind engine cannot pass it either.

**The `sight >= attack` invariant is BLIND to a missing value for most cards.**
It only fires when `sightRange < attackRange`, and **130 of 148** non-spell cards
have `attackRange <= 5.5` — every one of the 102 default cards is inside that
blind spot by definition. That is how Bomb Tower and Three Musketeers survived
until 2026-08-21: both were caught only because their `attackRange` was 6.0. So
a decision and an omission were indistinguishable in the code. Closed on
2026-08-23 by a catalogue pinning all 148 (`tests/core/test_sight_range.cpp`,
generated from the engine by `tools/audit/sight_audit.cpp`) plus a completeness
check — a new card must state a sight range, and choosing 5.5 is a fine answer.
Base↔Evolution agree 38/38, and the evolved spawn path 0/41.

**Card registry**: 132 playable ids (max 175) plus 41 Evolutions at ids
123-163, so 173 entries total. `getAllCardIds()` filters Evolutions out.
**Evolutions reuse the base card's name verbatim** — id 1 and id 128 are both
"Archers" — so a name can never identify a card on its own. There are no card
levels at all. Gaps at ids 16, 37, 38.

**Driving the simulator from outside is awkward, deliberately worked around.**
`injectEnemy` hardcodes team 1 (no ally equivalent); `step()` also runs the
heuristic opponent; `step_self_play()` avoids that but needs the card to be in
the simulator's own hand, which is shuffled by an unseeded `mt19937`, cannot be
set, and whose queue cannot be read (`getHand()` is team 0 only). The
workaround — reverse-engineering the shuffle by search — is in
`perception/bridge/sim_driver.py`. Useful measured facts: `reset()` costs
0.135 ms and its shuffle is uniform over all 70 hand-sets.

**`python_ai/replays/*.json` carry a labelled placement stream.**
`rl/replay.py`'s annotator adds `actionCardId` / `actionX` / `actionY` per tick, which
`GameLogger` itself does not write. Two catches: it labels **only the learner's
own plays** (the heuristic opponent's are logged nowhere), and **the training
run rewrites that directory continuously** — observed dropping from 8 files to
1 within minutes. Frozen fixtures live in `perception/tests/assets/`.

> **Since 2026-09-15 the deck is `CLASH_DECK`** (`python_ai/deck.py`); the list
> below is `SHIPPED_DECK`, what runs when it is unset. The next run uses a
> different deck — read "2026-09-15: the pre-launch audit" first.

`DEFAULT_DECK = [15, 6, 25, 40, 24, 72, 33, 7]` (`python_ai/deck.py`, re-exported by
`python_ai/envs/gym_wrapper.py`) — **the classic 2.6 Hog Cycle**, since 2026-08-16: Hog Rider, Musketeer,
Cannon, Ice Golem, Skeletons, Ice Spirit, The Log, Fireball. Costs 1-4, avg
2.625, spread 3. Win condition Hog Rider (4); 3 of 8 hit air (Musketeer, Ice
Spirit, and Fireball as a spell); **two** spells (Fireball, The Log).

All eight were verified FUNCTIONAL in this engine, not merely present in the
registry: Hog crosses and deals 317 tower damage in 40 s; a Cannon prevents
3,823 HP against a lone enemy Hog; The Log takes a Skeleton clump 12 → 5
bodies and Fireball 12 → 3; Skeletons spawn 3 bodies, Ice Golem tanks ~17 s.

**This replaces the Giant deck and gives up the recordings tie** that the
2026-07-30 revert was made for — behaviour cloning from
`perception/assets/recordings/` is not available against this deck. Paid on
purpose: BC was blocked on extraction anyway, and the deck was changed to test
the learning mechanism. It also removes the starved-win-condition risk that
revert re-accepted, structurally rather than probabilistically: **nothing here
costs more than 4.**

`test_default_deck_is_the_26_hog_cycle_and_every_card_is_cheap_enough` pins
both the identity and the `max(cost) <= 4` property.

**Fireball injected out of range does nothing, and that is not a bug.** A first
probe scored `inject(FIREBALL)` at 0.000 value-killed and briefly looked like
broken kill attribution; injected ON the clump it kills 12 → 3 bodies normally.
`get_troop_damage_dealt` includes **our own towers shooting**, so it is not
evidence a spell landed. Nothing in `prove_placement.py` is invalidated.

**`isValidPlacement` checks a body's FOOTPRINT against the board edge, since
2026-08-28.** It checked the CENTRE only, while the building-overlap loop three
lines below it already reasoned in footprints (`placedRadius + r`) -- two
conventions for one geometric question, the same shape as the sight-vs-attack
band. Measured through the binding: a Cannon (`placementRadius` =
`Building::COLLISION_RADIUS` = 1.0) was accepted at `x = 0.0` spanning
[-1.0, +1.0] and at `x = 17.0` spanning [16.0, 18.0].

The edge that matters is the PHYSICAL one, and this is the part that is easy to
get wrong: **cell i covers `[i - 0.5, i + 0.5]`** (`Board::CELL_HALF_EXTENT`,
the convention `isOnBridge` already documents and `clampToBoard` depends on), so
an 18-wide board runs `x` in **[-0.5, 17.5]**, not [0, 17]. Checking footprints
against the INDEX range instead would reject a 0.4-radius TROOP at both edge
columns and silently delete two of eighteen columns from the action space while
looking like a bounds fix.

Net effect, verified through the rebuilt `.pyd`: **buildings lose exactly
columns 0 and 17; troops lose nothing.** Spells are exempt -- a spell's radius
is an area of effect, not a body, and clipping the arena edge is normal for one
(same reason the overlap loop is inside the `!isSpell` branch).

GAMEPLAY-AFFECTING: the legal building area shrinks, so the placement head's
learned distribution and every win rate earned before this date describe a
slightly different action space. Pinned by `test_game_manager.cpp`'s
"...FOOTPRINT leaves the arena, and only its footprint", whose troop SECTION is
the load-bearing half.

**Three things reported as placement bugs on 2026-08-28 were already enforced**
and are worth not re-investigating: a building on the RIVER (rejected by the
own-half rule plus `OWN_HALF_RIVER_BUFFER`), a building fully OUT OF BOUNDS
(rejected by the centre test), and a building OVERLAPPING A PRINCESS TOWER
(rejected by the overlap loop -- towers are Buildings with a 1.5 radius, and
even 0.5 tiles away is refused). Only the footprint gap was real.

**Rolling spells are cast on the own half OR the river, and no further, since
2026-08-29.** They were exempt from the own-half rule like every other spell.
Measured on the ep-32,484 policy over 60 sampled episodes, **53.4% of its Log
placements were on the ENEMY half**, spread to y = 33 -- and a roller travels
FORWARD, so a Log at y = 31 rolls away from everything and off the board. Every
other card in the deck placed 100% on its own half.

**The bound is getRiverEnd() (17.5), not the own-half line (15.0), and that is
load-bearing.** The Log reaches the enemy Princess Tower only from the bridge:

| cast at | leading edge reaches | tower damage |
|---|---|---|
| y = 15.0 (own-half max) | 25.1 | **0** |
| y = 16.0 | 26.1 | 269 |

The tower's near edge is 25.5. A STRICT own-half rule would have made the tower
physically unreachable by a Log from any legal cell -- silently deleting the
interaction the 10.1 range exists for. Including the river band keeps it by one
row. Pinned in `tests/core/test_game_manager.cpp`'s `[roll][bridge]` case,
which asserts the arithmetic from `Board`'s own river accessors rather than
restating it.

Keyed on `CardDefinition::rollRange > 0`, so it catches exactly ids 33, 101 and
174 and nothing else; Fireball and Giant Snowball still go anywhere. Mirrored
for team 1 about the same band.

GAMEPLAY-AFFECTING: The Log is in `DEFAULT_DECK` and roughly half its placement
mass was on cells that are now masked off, so the placement head's learned
distribution for that card is invalidated.

**NOT a real-game rule, as far as I know.** Clash Royale lets every spell,
rollers included, deploy anywhere. This is a deliberate ACTION-SPACE decision --
it removes a large region of provably useless placements -- not a fidelity fix,
and it is recorded that way so nobody later "corrects" the engine toward it.

**The Log and Barbarian Barrel ROLL, since 2026-08-28.** They were static
circular `AreaSpell`s detonating once at the tap point. They are now dynamic
bodies that sweep a **rectangular corridor** forward from where they land,
advancing every tick and damaging each target **at most once**.

| | width | range | speed | knockback |
|---|---|---|---|---|
| The Log (33) | 3.9 | 10.1 | 1.0 tiles/tick | 0.8 |
| Barbarian Barrel (101, 174) | 2.6 | 4.5 | 0.5 tiles/tick | none |

**3.9 is a WIDTH and was being read as a RADIUS**, so The Log's old footprint
was a circle of diameter 7.8 — twice as wide laterally as the real card, and
it never moved. Same for the Barrel at 2.5.

**The 10.1 is load-bearing.** An enemy Princess Tower's centre sits **10.50**
tiles from `BRIDGE_Y` and its near edge **9.00** (radius 1.5), so a Log dropped
at the bridge reaches the tower with 1.1 tiles to spare and never reaches its
centre. Hit detection is therefore **surface-to-surface**, the same convention
`effectiveRangeTo` already uses — a centre-to-centre test would put the tower
out of reach and silently delete the card's main use. Verified through the
rebuilt `.pyd`: a Log injected at `(LEFT_LANE_X, BRIDGE_Y)` deals exactly 269
to the enemy Princess Tower. `test_area_spell.cpp` pins it against
`ArenaLayout`, never a hardcoded 9.0.

**Knockback is DIRECTIONAL, and that is the whole tactical point.** Where a
unit is caught ACROSS the corridor decides which way it is thrown: dead centre
is shoved forward along the roll, the lateral edges are flung sideways in
opposite directions, and everything between blends. That is what splits a
grouped push. `pushAway(entity, logCentre, d)` **cannot** express it — a radial
push from a point falls off with longitudinal distance, not lateral offset, so
a unit level with the log and one at its nose get thrown the same way. Hence
`pushAlong(entity, dirX, dirY, distance)` in `Entity.h`, fourth member of the
`pullToward`/`pushAway`/`mirrorToOppositeLane` family and carrying the same two
guards — so a Log still cannot move a building.

**GAMEPLAY-AFFECTING, and The Log is in `DEFAULT_DECK`.** Its area went from a
7.8-wide circle to a 3.9-wide moving corridor, so every win rate earned before
this date describes a different card. Checkpoints still load — nothing about
the observation or the action space moved.

Three implementation notes worth not re-deriving:

- **`rollSpeed` is tiles per TICK, stated directly and deliberately NOT routed
  through `MOVEMENT_SPEED_SCALE`.** That scale converts the registry's troop
  `SPEED_*` tier literals; a spell has no tier, and borrowing the conversion
  would make these two numbers mean something different from every other speed
  in the registry. Speed and knockback are **not sourced** — the published data
  gave width and range only — so they are documented approximations, unlike the
  width, the range and the lateral-throw mechanic.
- **The Barrel's Barbarian now spawns where the barrel STOPS**, not where it
  was thrown, because `spawnOnDetonate` fires at the end of the roll. That
  broke one Hero test that drove the spell exactly 9 ticks ("delay, then
  detonate"); the real floor is now 8 delay + 9 roll = 17.
- **`CombatEntity::effectiveRadiusOf` became a public static** so the sweep's
  corridor test can use it. `AreaSpell` is not a `CombatEntity`, and the
  alternative was a second copy of its two lines — the duplication that
  function's own comment exists to prevent.

**And the replay format carries the corridor, because the viewer cannot derive
it.** `cardMeta` now emits `rollWidth`/`rollRange` per card (0 for everything
else), and `web/viewer.html` draws the swept rectangle from those rather than
from a table of its own. This is the fix the "THE VIEWER IS A DIFFERENT KIND OF
COPY" section above argues for: a `file://` page whose only input is the replay
JSON is *structurally forced* to restate whatever it is not told, so the answer
belongs in the FORMAT. Verified by sampling rendered pixels — the body paints
3.91 tiles wide, centred on the lane, advancing exactly one tile per tick.

### The state setters an estimator writes with (item 22, 2026-08-24)

Additive, behaviour-preserving defaults, **no checkpoint or win rate
invalidated**. The narrative and the measurements are in `DECISIONS.md`; the
contract is here because callers need it and getting any of the four wrong is
silent.

```
set_tower_hp(team, slot, hp) -> bool      destroy_tower(team, slot) -> bool
get_tower_hp / get_tower_max_hp(team, slot)
set_current_tick(tick)
inject(card_id, x, y, team, hp=-1.0, deploy_ticks=-1)
set_elixir_for_team(team, value)          set_hand_for_team(team, cards) -> bool
seed(s)
```

- **`slot` is 0=King, 1=LEFT Princess, 2=RIGHT, in BOARD coordinates for BOTH
  teams** — never team-relative. The caller is a sensor reading a screen;
  asking it to mirror its own coordinates is the convention error that put the
  arena half a tile off-centre.
- **Tower HP is ABSOLUTE in, FRACTION out of perception.** Engine towers are
  level 9 (Princess 2534, King 4008); a real account's are often level 4-5
  (1750/1890) — wrong by a **different factor per player**. Pass
  `fraction * get_tower_max_hp(...)`. **Never inject an absolute reading.**
- **`hp <= 0` is REFUSED, not clamped.** A 0-hp tower that still occupies its
  cell and still fires is a position the real game cannot be in. Killing one has
  side effects (the crown, the King's wake trigger, `LanePath` retargeting) that
  belong to `destroy_tower`.
- **`deploy_ticks=0` for any unit perception can already SEE.** `inject` →
  `spawnEntity` → `applyCardMetadata` sets `deployTicksRemaining =
  DEPLOY_TIME_TICKS` unconditionally, so a mirror otherwise hands **every** unit
  a fresh deploy second — including one that has been walking for six. That is a
  standing defensive subsidy on every candidate. Measured at **317 tower HP**
  (one Hog hit) in the 100-110 tick window.
- **`set_hand_for_team` returns a bool and can REFUSE.** A silently-ignored
  refusal is worse than no update: the teacher's `slot` indexes the mirror's
  hand and the actuator taps the real one, and they are the same number only if
  this returned True.
- **`seed(s)` seeds both engine generators and re-deals**, so it is a drop-in
  for `reset()` wherever determinism is wanted. `sample_random_deck` is **not**
  covered — see `UPSTREAM_REQUESTS.md` item 23C.

---

## The training mechanism

**Every checkpoint and log destination is ANCHORED, not cwd-relative (fixed
2026-08-25).** Until then `train.py` read a bare `"model_weights.pth"`, and the
checkpoints live in `python_ai/`, so **which directory you launched from decided
whether a run resumed or started fresh** — silently, in both directions:

```
launched from the repo root  -> os.path.exists() False -> starts FRESH
launched from python_ai/     -> os.path.exists() True  -> RESUMES
```

`rl/checkpointing.py` now owns the single resolution point: `weights_path()`
anchors `.pth` files on `PACKAGE_DIR`, `run_path()` anchors `runs/`,
`historical_checkpoints/` and `stage_checkpoints/` on `REPO_ROOT` — which is
where each already physically sat, so nothing moved. An **absolute**
`CLASH_WEIGHTS` / `CLASH_LOGDIR` is still honoured verbatim, because that
override is what keeps an experiment arm off the live checkpoint.

Two consequences worth knowing before you edit anything here:

- **`chdir` is no longer test isolation.** The first suite run after the fix
  wrote a random-init 20-step checkpoint straight into
  `python_ai/model_weights.pth` and eight untrained snapshots into the real PFSP
  pool. Tests now redirect through `CLASH_WEIGHTS` / explicit `directory=`
  arguments, and `test_rl_base_trainer.py` carries an autouse tripwire that
  fails loudly if anything writes the live checkpoint.
- **`discover_historical_checkpoints(directory=...)`** exists for the same
  reason, mirroring `save_historical_snapshot`'s long-standing `directory=`.

Pinned by `python_ai/tests/test_checkpoint_paths.py`.

Two sequential pipelines. `train.py` hands off by `subprocess.Popen`-ing
`train_selfplay.py` (with `-u`, or the live diagnostics buffer and the log stays
0 bytes) and exiting.

```
train.py            phase 1  "mirror"           vs the UtilityTeacher, which
       |                                        since 2026-09-03 plays a POOL
       |                                        of real meta decks, not ours
       |  win rate >= PHASE2_ENTRY_WIN_RATE (0.60)   <-- gates THIS step only
       |  AND curriculum stage >= PHASE2_MIN_CURRICULUM_STAGE (rung 8)
       v
                    phase 1  "random_opponent"  vs rotating pool decks
       |  RANDOM_OPPONENT_EPISODE_BUDGET (5,000) episodes IN THIS PHASE
       v
train_selfplay.py   phase 2  PFSP league        vs frozen snapshots +
                                               4 scripted bots + exploiters
```

### THE OPPONENT PLAYS REAL META DECKS, since 2026-09-03

The phase-1 opponent played **our own deck** for every episode of every run this
project has ever done (`opp_deck` defaults to `ai_deck`). It no longer does:
`opponents/deck_pool.py` loads `opponents/decks/meta_decks.json` — 16 RoyaleAPI
ladder archetypes, named cards, hand-editable — and each worker samples one per
episode by PFSP weight. **The 2.6 mirror is still in the pool**, as one matchup
of sixteen. `CLASH_PHASE1_DECK_POOL=0` restores the old behaviour.

**THIS IS THE FIX FOR THE THREE DEAD CARDS, and the diagnosis is the 2026-08-29
autopsy's own — read the other way round.** That autopsy established Cannon /
The Log / Fireball sat at P(play | in hand) ≤ 0.009 for 30,000 episodes and that
the card head was **right** to price them there: forced through `env.step` a
Cannon at the policy's own cell was worth +185 tower HP, better in 6 of 14
states. Three separate attempts to overrule it (coverage floor at two
coefficients, a threat gate, forced sampling at five doses) all cost win rate.

Every one of those measurements held the OPPONENT'S DECK fixed at the mirror.
Measured with the deck as the independent variable
(`eval/measure_deck_matchups.py`, 16 decks x 30 episodes, ep-32,484 policy,
~82,000 decisions), **the 2.6 mirror ranks 16th of 16 on all three opportunity
metrics at once**:

| per decision, best case | mirror | pool median | ratio |
|---|---|---|---|
| Fireball catch (enemy HP) | **323** | 494 | 1.53x |
| The Log catch (3.9x10.1 corridor) | **145** | 273 | 1.88x |
| threat HP on our half (the Cannon's driver) | **270** | 503 | 1.86x |
| enemy units on board | **1.0** | ~2.0 | ~2x |

The extremes are wider: `royal_hogs_furnace` offers **884** of Fireball value
(2.7x the mirror), `miner_poison_control` **427** of Log value (2.9x), and
`mega_knight_ram` **858** of threat (3.2x). So the card was never underpriced —
it was correctly priced FOR A DISTRIBUTION WE CHOSE. Cannon answers a tank
walking at your tower, Fireball answers a medium-HP cluster, The Log answers a
ground swarm, and a 2.6 mirror produces one Hog, one Musketeer and 1-elixir
Skeletons respectively.

**Read the take-up column honestly**: it stays at 0.00-0.06 for all three cards
across ALL sixteen decks. This is a mirror-trained policy meeting these decks
for the first time, so the pool changes what the cards are WORTH and does not by
itself change what the policy does. **The dead cards are a retraining
prediction, not a demonstrated fix.**

**THE HUMAN META'S "COUNTERS TO 2.6" ARE NOT THIS ENGINE'S COUNTERS**, which is
the finding that decides how the pool is filtered. Decks tagged `anti_26` from
the real game's matchup lore (Tornado king-activation, Inferno Tower, Mega
Knight, LavaLoon) average a **0.589** win rate against the current policy, while
untagged decks average **0.433** — the tags are, if anything, backwards. What
actually beats a cycle deck here is raw statline: P.E.K.K.A. bridge spam scores
**0.067** and Mega Knight Ram **0.200**, while Graveyard control and Mortar
cycle — both textbook 2.6 counters — sit at **0.867** and **0.800**. This is
`gym_wrapper`'s own long-standing note ("giant units crush cycle decks in this
engine") arriving from a new direction.

So the pool is filtered by **measurement, never by the tags**. A deck under
`POOL_WINRATE_FLOOR` (0.20) drops to `POOL_UNWINNABLE_WEIGHT` — about 9% of
episodes across all such decks, rather than the 20% a shared floor gave — and
climbs back out on its own as the agent improves, which a static exclusion list
cannot do. The tags stay in the JSON as commentary and gate nothing.

> **AND THAT FILTER WAS MEASURING THE WRONG THING. `POOL_WINRATE_FLOOR` is 0.0
> — OFF — since 2026-09-06.** Its input is a win rate, and a win rate against
> this pool confounds *how hard the deck is* with *whether the teacher can
> pilot it*. It could not pilot five of the sixteen (next section), so the
> floor read "the agent beats this deck" as "winnable" when the cause was a
> broken opponent, and "the agent loses to this deck" as "structurally lost"
> when the cause was a teacher that pilots heavy decks correctly by accident —
> "send the most expensive building-targeter to the bridge" **is** the Royal
> Giant plan.
>
> The episode shares it produced, against the 2026-09-05 per-deck measurement:
>
> | deck | win | share of episodes | |
> |---|---|---|---|
> | dart_bait_cycle | 0.20 | **26.7%** | teacher never played the Barrel |
> | mortar_cycle | 0.20 | **26.7%** | teacher never played the Mortar |
> | rg_fisherman / royal_hogs / mega_knight_ram | 0.00 | **0.84% each** | |
>
> **Over half the run went to two decks the opponent could not play, and the
> three that beat it 40-0 got 0.84% each.** On the shipped priors it is worse:
> 2% of episodes total to every deck below the floor. A 2.6 cycle deck is BUILT
> to answer a big push with minimal elixir — two Cannons, a whole cycle inside
> one defence — so those matchups are the only place that skill can be learned,
> and they were the ones being skipped.
>
> **The zero-gradient worry is real and belongs to the RUNG, not the deck.**
> Difficulty here is the teacher's lookahead. Measured 2026-09-06 on the
> ep-83,128 policy against the FIXED teacher, 20 greedy episodes per deck:
> `rg_fisherman_cycle` scores **0.800 at rung 2** against **0.000 at rung 10**.
> So a deck is not unwinnable, it is unwinnable *at a rung* — and the fix for a
> hopeless matchup is to lower the rung, which the ladder then climbs back.
> `pfsp_weights(floor=...)` restores the old behaviour in one argument.

**THE COLD START IS REAL AND IS NOT SOLVED.** A random-init policy won **0 of
its first 100 episodes** against the pool, against ~0.07 for the historical
mirror run over the same span, and that was true both before and after the
count-weighted estimator below -- so do not read the estimator as a fix for it.
A fresh net against Mega Knight and Giant beatdown is simply harder than a fresh
net against its own cycle deck. Watch the first ~1,000 episodes of any fresh run;
if the win rate is still pinned at zero, trim the pool to the winnable band with
`"enabled": false` (no code change, no checkpoint invalidated).

**IT ONLY APPLIES TO A FRESH RUN, and that is the practical answer.** Measured
on the same sweep, `model_weights_phase5.pth` (ep 32,484) averages **0.527**
unweighted across the 16 decks, clears 0.40 against **11 of 16**, and scores a
PFSP-weighted **0.422** -- just above `PLATEAU_MIN_WIN_RATE`, so the ladder's
valves are live from the first episode. Resuming skips the cold start entirely,
and nothing in the observation or action space moved, so the checkpoint loads
unchanged. **Resume; do not restart.** *(Superseded for the 2026-09 final run,
which restarts on a new deck by choice: see the pre-launch audit section — the
cold-start fixes there exist for exactly this.)*

Verified end to end on a COPY of that checkpoint: the loader announced
`checkpoint stage 3 was written against a 6-rung teacher table; remapped to rung
6 (same lookahead horizon)` and then trained at **0.51** over its first 26
episodes at rung 6 -- the rung that consumed 23,040 episodes -- which is above
`PLATEAU_MIN_WIN_RATE` and so is the condition for the ladder to move at all.

**What IS established is that the SAMPLING self-corrects**, simulated against a
policy 0.15x the strength of the one that produced the priors: the share of
episodes spent on decks that policy cannot win falls from **42% at episode 0** to
roughly 7-13% within a few hundred episodes, and the mix converges onto the decks
it has the best chance against. That happens because the live estimate crosses
`POOL_WINRATE_FLOOR` and PFSP parks the deck -- the mechanism working as intended.

**The estimate is count-weighted early** -- `alpha = max(0.05, 1/(n+1))`, the
running mean decaying into the EWMA -- so a deck's estimate reflects THIS policy
within ~10 matches instead of ~60. That is an estimator-quality argument, not a
measured win-rate gain: over 8 seeds the effect on episode share is inside the
noise (6.8-17.8% either way). It is kept because a prior taken from a DIFFERENT
policy must not outlive contact with the current one, which is exactly what the
priors' own docstring promises.

**Sampling is PFSP, NOT deck-at-a-time mastery**, and the spread is why: the
measured win rates run 0.067 to 1.000, so a "beat each deck to a threshold"
schedule stalls permanently on the first hard one — the exact failure being
fixed. It is also the lesson phase 2 already learned one level up, where a
ladder was replaced by PFSP because a ladder never revisits and so cannot
prevent forgetting.

### THE TEACHER COULD NOT PILOT 5 OF THE 16 DECKS (fixed 2026-09-06)

`card_roles` promotes **the most expensive BUILDING-TARGETER** to win condition.
That is right for a Hog, a Giant, a Royal Giant or a Battle Ram and finds
**nothing** in a deck whose route to a tower is a siege building or a spell —
so `mortar_cycle`, `xbow_30_cycle`, both bait decks and `graveyard_control` got
`wincon_id = None`, and **every combo family bails on that**. The teacher was
left purely reactive with a third of the pool.

**Measured against an opponent that does NOTHING** (`step_self_play`, both sides
no-op — plain `step` runs the C++ heuristic, which defends). A competent pilot
of any of these decks should three-crown a defenceless opponent inside three
minutes:

| deck | twr dmg | ticks | elixir spent | the named card's own spend |
|---|---|---|---|---|
| hog_26_mirror | 8515 | 718 | 23.4 | Hog Rider 6.0 |
| xbow_30_cycle | 2486 | **3600 (timeout)** | **6.0** | X-Bow **0.0** |
| graveyard_control | 2407 | **3600 (timeout)** | **6.1** | Graveyard **0.0** |
| mortar_cycle | 4683 | 1769 | 15.0 | Mortar **0.0** |
| classic_log_bait | 5405 | 1734 | 7.9 | Goblin Barrel **0.0** |

**It never spent one elixir on the card the deck is named for**, and with X-Bow
and Graveyard it could not finish a defenceless match in the full 3600 ticks,
spending 6 of ~278 available income — **2.2%**.

**AND THE CELL IS AS LOAD-BEARING AS THE CARD.** Sweeping every legal cell,
scored as enemy tower HP actually lost:

| card | cells that damage the tower | best | where |
|---|---|---|---|
| Mortar | **34 of 170** | 1596 | rows 13-15 only |
| X-Bow | **34 of 170** | 3824 | row 15, centre columns |
| Goblin Barrel | 588 of 588 | 1320 | on the enemy tower (600 in our own half) |
| Graveyard | **0 of 588** | **0** | nowhere — see below |

A siege building one row too far back is worth *exactly* as much as not playing
it — 136 of its 170 legal cells score precisely zero. So `best_building_cell`'s
defensive answer was not merely suboptimal, it was total.

**The fix is derived by injection, never a card-name list** — the technique
`_card_table_uncached` already uses. `siege_reach(card_id)` injects a building at
the furthest-forward legal row and reads tower HP lost; `spell_spawns_bodies`
injects a spell and counts friendly bodies. **The spawn, not the damage, is the
discriminator for a spell**: every direct spell hurts a tower it is cast on, so
"does it damage the tower" would promote Rocket in log bait and Fireball in
mortar cycle. The fallback fires **only when no building-targeter exists**, so
the eleven decks that already resolved a win condition are bit-identical.

Re-measured after the fix, same probe:

| deck | before | after |
|---|---|---|
| mortar_cycle | 4683 / 1769 t | **8930 / 882 t** — now the pool's strongest |
| xbow_30_cycle | 2486 / **3600 t** | **8424 / 1529 t**, 75% three-crowns |
| classic_log_bait | 5405 / 1734 t | **7994 / 512 t** |
| graveyard_control | 2407 / **3600 t** | 4323 / 2520 t — still last, see item 27 |

**GAMEPLAY-AFFECTING**: five of sixteen opponents got materially stronger, so no
win rate is comparable across this date. The eleven unchanged decks moved ±10%
in this probe, which is the teacher's own unseeded profile draw, not an effect.

**GRAVEYARD'S CADENCE EXACTLY CANCELLED A TOWER'S FIRE RATE (fixed 2026-09-06).**
It dealt **zero** tower damage from all 588 legal cells -- and the first
explanation given here, "it spawns only one skeleton", was WRONG. Counted where
nothing can kill them, the bodies climb 1,2,...,**9** exactly as
`withRepeats(9, 10)` asks. What the zero measured was a KILL RATE EQUAL TO THE
SPAWN RATE: one 81-hp Skeleton per 10 ticks against a Princess Tower firing once
per 10 ticks leaves a standing population of 1 forever and lets none of them
live long enough to swing.

**An instantaneous count cannot tell "nothing spawned" from "everything spawned
and died on schedule."** That is the saturating-measurement trap in the mirror:
the "maximal permissiveness" rule under Measurement discipline has a twin where
the failure mode is maximal SUPPRESSION, and it needs the same thing -- a
control that must fire, here a board on which death is impossible.

Real card: one Skeleton every **0.5 s**, 12 total since the 2026-01-06 balance
change. Now `withRepeats(12, 5)`, so arrivals outrun a tower 2:1, which is the
mechanic the card is built on. GAMEPLAY-AFFECTING: `graveyard_control` stops
being a seven-card deck, so the agent's 0.925 against it is void.
`UPSTREAM_REQUESTS.md` item 27.

**The general shape, which this project keeps meeting:** a rule that is correct
for the case it was written against, silently returning "nothing" outside it.
`wincon_id = None` did not raise, did not warn, and reads downstream as a
teacher that is merely passive.

### RESUMING INTO THE POOL: DROP THE RUNG FIRST (measured 2026-09-03)

**A mirror-trained checkpoint resumed into the pool AT THE RUNG IT EARNED
COLLAPSES.** Measured twice, from `model_weights_phase5.pth` (ep 32,484, rung 6
after the legacy remap), two independent arms:

| episodes in | 0 | 30 | 60 | 100 | 130 |
|---|---|---|---|---|---|
| arm A win rate | 0.58 | 0.47 | 0.34 | 0.16 | **0.00** |
| arm B win rate | 0.58 | 0.47 | 0.35 | 0.24 | **0.00** |

Both fired `[STALL] Curriculum DEMOTED to stage 5`, and arm A recovered to
0.30 / 0.21 immediately after the demotion cleared the window.

**THE CAUSE IS TWO DIFFICULTY AXES MOVING AT ONCE**, which is the trap the
eleven-rung table above exists to remove -- applied to the rung axis and then
NOT applied to the resume path. The checkpoint earned rung 6 against the MIRROR;
resuming it into the pool asks it to absorb a 5-second teacher AND sixteen
unfamiliar decks in the same episode.

**And the number that made this look safe was measured on the wrong axis.** The
"PFSP-weighted 0.422" above comes from `measure_deck_matchups.py` at **teacher
rung 1**. It says nothing about rung 6, and it was used to argue a resume would
start competitive. Read every pool win rate in this file with its RUNG attached.

**At rung 2 the same checkpoint, same pool, same aux fix is healthy** -- win rate
holds ~0.56 and mean episode reward goes **-2.19 -> +1.46** over the first 30
episodes, against -1.7 to -2.8 for the rung-6 arms. So the recipe is: introduce
the pool at a LOW rung and let the ladder climb it, changing one axis at a time.

**Set the rung explicitly and STAMP `teacher_table_size` when you do.** Writing
`curriculum_stage = 2` into a checkpoint that lacks the stamp gets it remapped as
a legacy index to rung 4 -- the migration doing exactly its job on a number that
did not need migrating.

### 490 EPISODES ON THE POOL: USAGE MOVES, SELECTIVITY DOES NOT

The first real training run under the pool (rung 2, aux heads reset, ep 32,484
-> 32,970), measured on a frozen 2,446-state bank with
`eval/probe_card_discrimination.py`. `signal` is |delta| / step-to-step scatter
across four checkpoints; below ~2 a trend is not separable from PPO jitter.

| card | marginal | signal | ratio (hi/lo) | signal | place_q | signal |
|---|---|---|---|---|---|---|
| Fireball | 0.0080 -> **0.0159** | 3.6 | 5.32 -> 6.54 | **0.8** | 0.685 -> 0.685 | 0.0 |
| The Log | 0.0186 -> **0.0284** | 2.3 | 5.28 -> **4.01** | 2.4 | 0.217 -> 0.231 | 1.6 |
| Cannon | 0.0230 -> 0.0245 | 0.6 | 3.62 -> **2.79** | 4.3 | n/a | |

**THE USAGE RISE IS REAL AND IT IS THE WRONG KIND.** Fireball's probability
nearly doubled and The Log's rose by half, both above the jitter. But `p_lo`
rose for all three cards with real signal (5.2 / 3.7 / 1.7) -- the policy is
putting more mass on them ON BOARDS THAT OFFER THEM NOTHING -- and two of the
three discrimination ratios FELL with signal above 2. Fireball's ratio gain is
not established (0.8; it ran 5.32 -> 4.87 -> 7.85 -> 6.54).

This is the marginal drift that made every previous attempt at these cards cost
win rate, arriving from the environment side instead of from a coverage floor.
Win rate over the same window oscillated 0.36-0.49 with no trend, and mean
episode reward stayed positive throughout (0.8-2.3), so nothing broke -- it
simply did not buy the conditional.

**AND PLACEMENT DID NOT MOVE AT ALL** -- 0.685 -> 0.685 for Fireball, signal
0.0. That is the number the 2026-08-29 autopsy identified as the binding
constraint, and 490 episodes did not touch it. The card head is cheap to move
and the placement head is not, so a short run necessarily produces exactly this
shape: more usage, same aiming.

**READ THE PER-CARD CONSTRAINT BEFORE EXPECTING ANY OF THIS TO CONVERT.**
Fireball's placement head already collects **68.5%** of the achievable catch, so
its usage rise plausibly converts. The Log's collects **21.7%**, so playing it
more is playing a badly-aimed 2-elixir spell more often. One number for "the
three dead cards" hides that they have different problems.

**What this does NOT establish**: that the pool fails. 490 episodes is ~1.5% of
the 32,000 that produced the policy being measured, the run was healthy
throughout, and the opportunity the pool creates is measured and large. It
establishes that the deck pool alone, on this timescale, moves the level and not
the condition -- and that the next lever is the one the autopsy already named,
placement, where this repo's measured-positive tools are decision-time search
(+0.319) and expert iteration (+0.045).

### THE STALE AUX HEAD, found the same day and NOT the cause

`aux_card_head` predicts which of 185 cards the opponent plays next and reads
`hx` with **no detach**, so its error backpropagates through the LSTM and the
whole trunk. A mirror-trained checkpoint has seen 8 of those cards; the pool
shows it ~60. Measured on resume:

| | mirror run | pool resume | uniform = ln(185) |
|---|---|---|---|
| `NextCardCE` | 1.45-1.58 | **13.76** | 5.22 |

**Worse than uniform** is the signature of a stale classifier: not ignorant,
confidently wrong. `aux_card_scale = 0.02` was calibrated when the CE was ~1.5,
putting the weighted term at ~0.015 -- comparable to the actor loss. At 13.76 it
is 0.138 against an actor loss of ~0.020, i.e. the shared trunk pulled **~7x
harder toward "learn 60 unfamiliar card identities" than toward "win"**.

Fixed two ways: `ppo.py` rescales the aux term by a DETACHED factor so its
magnitude cannot exceed the uniform CE (not `clamp(max=)`, which zeroes the
gradient above the ceiling and would freeze the head exactly when it must
relearn; at or below the ceiling the factor is 1.0 and every earlier run is
bit-identical), and `tools/reset_aux_heads.py` re-initialises both aux heads --
52,170 params, 2.75% of the net, none on the acting path, Adam moments zeroed.
Verified: `NextCardCE` 13.76 -> 4.30 on the first update, and
`probe_card_discrimination` returns bit-identical numbers afterwards, so the
policy provably did not move.

**IT IS NOT THE COLLAPSE CAUSE.** Arms A and B above differ by exactly this fix
and their trajectories match within noise. A real defect, worth fixing, wrongly
blamed at first -- recorded here so the next reader does not re-derive the
attribution and stop there.

**GAMEPLAY-AFFECTING**: the opponent distribution changed, so every curriculum
gate is calibrated against a different opponent and **no win rate is comparable
across this date**. Checkpoints still load — the observation and action space
are untouched.

**`PHASE2_ENTRY_WIN_RATE` does not gate the pipeline handoff**, despite its
name. It gates `mirror` → `random_opponent`, together with a stage floor the
diagram used to omit: **both** the win rate and `PHASE2_MIN_CURRICULUM_STAGE`
must be satisfied. An earlier version put the 0.60 gate on the handoff arrow
and cost a live run two wrong predictions about when it would transition.

**`PHASE2_TOTAL_EPISODE_CAP` no longer exists** (corrected 2026-08-25; the
constant was replaced on 2026-08-09 and this diagram kept its name and its
40,000 for sixteen days). A cap on TOTAL episodes was the wrong quantity: how
long the agent spent against random decks depended entirely on how fast it
cleared the mirror curriculum — clear it in 3k episodes and you got 37k of
random decks, take 35k and you got 5k. It is now
`RANDOM_OPPONENT_EPISODE_BUDGET = 5000`, a budget measured **inside** the
phase, so the handoff is 5,000 episodes after entering `random_opponent`
whenever that happens.

In `random_opponent` the console prints **two** stage numbers, `4/2`. The
first is the frozen mirror stage; the second is the CURRENT random deck's own
progress through the same six stages. It resets to 0 every time a new deck is
sampled (`CurriculumManager` in `rl/curriculum.py`), so `4/5 -> 4/0` is a new deck, not a regression.

### Network (`models/net.py`, 1.89 M params)

`MicroRoyaleNet` — the LSTM is 96% of the parameters:

| | params | |
|---|---|---|
| `cnn_trunk` | 14,464 | 21→16→32 conv, 2× MaxPool(ceil) → `32×9×5`, then 2× `DilatedContextBlock` |
| `scalar_mlp` | 4,850 | `ScalarEncoder`: 1124 scalars → 64, in four branches |
| `lstm` | **1,804,288** | `LSTMCell(1504, 256)`, stepped manually |
| `card_head` | 1,285 | 256 → 5 (4 hand slots + no-op) |
| `place_ctx` + `place_up` | 14,593 | ctx→32ch, broadcast-add, 2× (upsample+conv) → 612 |
| `value_head` | 257 | critic |
| `aux_card_head` | 47,360 | opponent NEXT-CARD classifier (256 -> 185) |

**PHASE 4, 2026-08-27 — three architectural bottlenecks, all measured.** The
net went 1,907,329 → **1,846,963** parameters (it got SMALLER) for **+14.8%**
update wall-clock, measured interleaved against a ±0.8% noise floor.

**The trunk's receptive field was 10x10 and is now the whole board (34x18).**
A bridge sits at y=16.5 and the enemy King at y=30.5 — **14 rows** — so no
single convolutional feature could relate "their win condition just crossed" to
"this is the tower it is walking at", and the placement head reads the spatial
map DIRECTLY, so the limit reached the ACTION and not merely the
representation. Two `DilatedContextBlock`s (`CONTEXT_DILATIONS = (2, 2)`) are
**appended** to `cnn_trunk`; at the pooled 9x5 map one cell is worth 4 input
cells, so a 3x3 at dilation d buys 8d input rows. 10 + 16 + 16 = 42 ≥ 34.

Four things worth carrying:

- **Appending is load-bearing, not style.** Two callers slice the trunk by
  INDEX — `extract_features_hires` and `hires_features` take `cnn_trunk[:2]`
  for the full-resolution placement branch. Inserting anywhere but the end
  hands that branch a different tensor with no error anywhere.
- **The final 1x1 is ZERO-INITIALISED**, so the block is an exact identity at
  init and a warm-started checkpoint's trunk is bit-identical — the same device
  `place_hires[-1]` already used. The change is strictly additive.
- **...and that made the existing receptive-field guard silently obsolete
  rather than loudly failing.** A zero gate emits zero gradient, so
  `test_net_trunk_receptive_field.py` kept measuring 10x10 and kept PASSING
  across the change that invalidated its own docstring. It now builds the base
  trunk explicitly with `context_dilations=()` — measuring by construction
  instead of by accident. **A guard that stops guarding without failing is
  worse than one that never existed.**
- **DILATION IS ONLY CHEAP WHILE IT IS SMALL RELATIVE TO ITS MAP.** The first
  attempt used dilations (2, 4) and cost **1.533x** — a hard budget fail for
  6,784 parameters. Isolated at the real shape (500, 32, 9, 5): d=2 costs
  8.26 ms and **d=4 costs 43.72 ms for the SAME parameter count**, because
  `padding=4` on a 9x5 map pads to 17x13 — 221 cells against 45 — so ~80% of
  the work is padding. Two d=2 blocks reach the same 32 rows for 14.6 ms.
  Width is also NOT the axis: the earlier measurement in that same test file
  has doubling the channels at 2.28x for exactly zero extra reach.

**`scalar_mlp` is four semantic branches, not `Linear(1124, 64)`.** 72,000 →
**4,850** parameters, at **0.998x** — free. econ (elixir + 4 costs) → 8, hand
(4 one-hots through one shared `Linear(185, 5)` per slot) → 20, extra (clock,
both spends, 6 tower HPs) → **12, an EXPANSION**, and cycle (`seen`/`recency`
through one shared card projection) → 24. **The sum must stay 64**: it is
`lstm_input_dim - cnn_out_dim`, and moving it reshapes `LSTMCell(1504, 256)` —
1,804,288 parameters, 96% of the net.

**And the premise needed correcting first.** "17.6:1 compression destroys the
cycle" does not survive the mathematics — a random projection of 370 dims into
64 largely preserves it (Johnson–Lindenstrauss). The defect is about
**learning, not information**: in one `Linear(1124, 64)` every output row spans
all 1124 inputs, so a gradient step improving HAND encoding rewrites the very
rows the cycle is read through. It was not compressed away, it was
continuously perturbed by another objective's learning, which is worse because
it never converges. The test that pins this therefore takes a real optimizer
step on a hand-only objective and asks whether the encoder's RESPONSE to the
cycle moved — and a paired contrast test **requires the monolithic layer to
fail the same procedure**, so the instrument cannot be passing for a trivial
reason.

**`bptt_chunk` 25 → 50, because the credit horizon was shorter than a card
rotation.** Derived from the engine, not assumed: cycling a card back means
playing the other four, and at `DEFAULT_DECK`'s average cost with the elixir
rate MEASURED off a live env (28.571 ticks/elixir, i.e. exactly 1/0.035) that
is `4 x 2.625 x 28.571 = 300 ticks = 30 DECISIONS` at `skip_frames = 10`. So
gradients were truncated before one rotation completed — **the same shape of
defect as the gamma correction made the same day**, and pointed at the same
capability: item 24 had just put `seen[]`/`recency[]` in the observation to
make card counting possible, and no gradient path reached far enough to learn
from it. *The information arrived and the credit path did not.*

Nearly free (**1.052x**) because `update_timestep` and `num_minibatches` are
fixed, so the segment count halves as the length doubles and a minibatch still
pushes 500 flat rows through the trunk — 84% of the update. Only the LSTM loop
changes shape (25 calls at batch 20 → 50 at batch 10) and it is 16%.
**The real cost is not wall clock**: segments per minibatch fall 20 → 10, and
those segments are the batch dimension of every gradient estimate. That is
bounded at ≥ 8 by `test_bptt_credit_horizon.py`, which is what refuses L=100
(1.227x, and 5 segments) — on batch width, not on time. Overridable with
`CLASH_BPTT_CHUNK`.

**Measuring any of this needs care on this box.** The update-chunk benchmark
(`eval/profile_architecture.py`) drifted **3x in absolute terms** across one
session from thermal throttling, so only INTERLEAVED within-run ratios mean
anything — and a fixed arm order is not enough either: with two IDENTICAL arms
the second read **6.8% slower**, every round, because it inherits the cache the
first just evicted. The order is alternated and every comparison carries an
identical-arms control. With 21 repeats that control sits at 0.99–1.02.

`ceil_mode=True` on both pools is load-bearing: 34 rows floor-divide to 8 and
would silently delete the back row behind the King.

Ability heads exist **only if the deck has a Champion/Hero**
(`num_ability_slots > 0`). With a Champion-less deck they sampled pure noise
every tick, widened the PPO ratio's variance, and spent up to `2·log2 = 1.386`
of the entropy budget keeping two irrelevant coin flips random.

### Action space

`(card_index, cell_index)`, sampled autoregressively — placement is conditioned
on the card already chosen, via that card's identity embedding. Two masks, both
recomputed from the stored observation rather than buffered, so the rollout and
the PPO update cannot drift apart:

- **affordability** (`affordability_mask`) — which hand slots are playable at
  the current elixir. `masked_fill(-inf)`, never a large finite value: a finite
  value still leaves nonzero probability *and* an entropy contribution for an
  illegal action. The no-op column is always legal, so no row is all `-inf`.
- **placement legality** (`placement_mask`) — troops get the own half
  (16 rows), spells get all 34. `isValidPlacement` exempts spells from the
  own-half rule but the action space used to cap `target_y` for every card, so
  a Fireball could physically never cross the river.

`cell_to_xy` uses **truncation**, matching the engine's own
`static_cast<int>(position.y)`. Never `round()` — see the discretization note
under "Open problems".

`skip_frames = 10`, so **one decision per second** and 10 ticks pass with no
observation. That is a hard ceiling on timing precision (spell-on-moving-troop,
drop-as-they-cross-the-bridge, kiting all want 0.1–0.3 s).

### Reward

> **THE DISCOUNT HAS TO OUTLAST THE MATCH, and it did not until 2026-08-27.
> `gamma` 0.99 -> 0.999.** `1/(1-gamma)` is the horizon in DECISIONS and a full
> match is `max_ticks/skip_frames` = 3600/10 = **360**, so a horizon of 100
> covered barely a quarter of the game and every TERMINAL quantity decayed
> together to `0.99^360 = 0.0268` before it could compete with any immediate
> shaping term:
>
> | | discounted to episode start |
> |---|---|
> | win (+1) | 0.0268 |
> | `DRAW_PENALTY` (1.0), raised 0.2 -> 1.0 *to stop stalling* | 0.0268 |
> | `W_TOWER_DESTROYED` (0.6), undiscounted, fires mid-episode | **0.6000** |
>
> One crown was worth **22.4x winning the game it serves**. Measured over 10
> greedy episodes of `model_weights_phase1_v5.pth`: the terminal reward
> contributed **0.0195** of the discounted return, so **~98% of the objective
> was shaping**. That is a hard ceiling on strategic depth and not a tuning
> nicety — a sacrifice play (concede tower HP now, win later) costs 0.5 at once
> and repays 0.027 at the end, so gradient ascent on that objective cannot find
> one at any network capacity.
>
> **It DRIFTED; it was not wrong when written.** `weights.py` justified 0.6 as
> "above the discounted value of a win (~0.28 at these episode lengths)", and
> at the ~112-decision episodes of that era `0.99^112 = 0.32` made 0.6 a
> deliberate ~2x bias. The **2026-08-07 movement-speed fix tripled match
> length** — this file's own warning is "no win rate below survives it" — and
> the reward constants were never re-derived against it. Same failure class as
> the four stale arena copies: a constant calibrated against a measurement a
> later change invalidated.
>
> **The variance objection was measured and does not apply here.** GAE's
> advantage estimator has its own lookahead `1/(1-gamma*lambda)`, and
> `gae_lambda = 0.9` already pins that near 10 steps — 9.17 at gamma 0.99,
> 9.91 at 0.999. Over 6 real episodes the critic's target barely moves:
> mean |return| 0.385 -> 0.416 (+8%), std 0.362 -> 0.384 (+6%), max **unchanged**
> at 1.719. So this buys a 26x increase in outcome weight for ~6% more spread.
> **No annealing**, deliberately: OpenAI Five ramped gamma to control early
> variance, and the table shows there is none here to control.
>
> **GAMEPLAY-AFFECTING in the sense that matters** — it changes the OBJECTIVE,
> so every win rate in "Measured baselines" was earned against a different one.
> Checkpoints still LOAD, but their critic was fitted to the old discount and
> must re-converge; early movement in `Loss/Critic` and explained variance on a
> resumed run is expected, not a fault. `CLASH_GAMMA` restores the old value.
>
> Pinned by `tests/test_reward_horizon_invariant.py`, which asserts the
> RELATIONSHIP (horizon >= match length; crown <= 3x a win; `DRAW_PENALTY`
> above one step of the overflow penalty) rather than the number — so retuning
> either side stays free and letting them drift apart again does not.
>
> **`compute_shaping`/`solvency_shaping` now REQUIRE `gamma`** — no default.
> `rewards/` is an enforced leaf that may import neither `rl/` nor `trainers/`
> (`test_package_layout.py`), so the default could not be *derived*; requiring
> it satisfies the no-second-copies rule by ABSENCE, which is stronger. Unit
> tests pass their own fixed fixture value, which is correct rather than a copy
> — they pin the FORMULA, and a test reading live config would move its own
> expected answers. `EXPLOITER_GAMMA` now derives from `PPOConfig.gamma` too.

Sparse `±1` on win/loss plus `compute_shaping()`:

| term | weight | form |
|---|---|---|
| **tower** HP | `W_BLDG = 0.5` | **potential-based**: `γΦ(s′) − Φ(s)` |
| troop HP | `W_TROOPS = 0.1` | delta |
| elixir trade | `W_ELIXIR_TRADE = 0.03` | delta |
| tower destroyed | `W_TOWER_DESTROYED = 0.6` | **deliberately NOT** PBRS |
| elixir overflow | `W_ELIXIR_OVERFLOW = 0.1` above 9.0 | per-step penalty |
| draw | `DRAW_PENALTY = 1.0` | terminal |

The PBRS form keeps the `γ` — Ng et al.'s policy-invariance result requires it,
and dropping it is a different (biased) shaping that looks almost identical.

**And the γ alone is not enough: Φ(terminal) is never zeroed, so the tower term
is NOT policy-invariant** (TODO 00.4, measured 2026-09-23). It telescopes
exactly to γ^T·Φ(s_T), a disguised terminal reward: over 16 seeded rung-3 mirror
matches, **+0.107 on wins and −0.169 on losses** (range ±0.41) — a tower-margin
bonus worth 10-17% of the ±1 outcome, sign-aligned with it and with the timeout
tiebreak. The lethal-spell and solvency potentials leave smaller residues
(nonzero in 2 of 16 matches, up to +0.125 and −0.019). Kept, as a benign bias;
making all three strictly invariant is one line (`γΦ(s′)` times `1 − done` in
`compute_shaping`) and is an objective change for the maintainer to choose.

**Deployed buildings are priced with the troops, not with the towers**
(2026-08-06). The potential used to read the engine's `buildingDamageDealt`,
which is towers *plus* deployed buildings, so damage to the agent's own Cannon
was charged at the Princess-Tower rate. That is the wrong price for a
sacrificial card: losing the Cannon's 824 HP cost `0.5·824/4008 = 0.1028`, while
killing with it paid only `0.1·hp/4256` — it had to kill **5.3× its own HP to
break even**. Parking it in a back corner cost exactly **zero**, because decay
emits no `DamageDealtEvent` at all (`Building::update`), so the engine never
charged for it dying of old age. Guaranteed-zero beat probably-negative, and the
measured policy did exactly what that asked: **27.9% of Cannons went to
(11,2)/(11,3), behind its own King**, mean placement `y = 6.3` — behind its own
Princess Towers. `DamageByTargetTypeCollector` now splits towers out, and the
break-even is 1:1. Same failure shape as the old symmetric elixir term: *doing
nothing was the safe, guaranteed-zero outcome.*

A caution recorded with it: forcing the Cannon to a fixed "better" cell did
**not** improve win rate (centre 0.605 vs baseline 0.610 over 200 episodes
each). Forced-corner was the worst arm at 0.515, so the corner really is bad,
but the learned state-dependent mix beats every fixed cell. A forced-placement
A/B cannot tell you what a policy would learn under a corrected reward — those
are different questions, and only the second one justified this change.

`W_TOWER_DESTROYED` breaking policy-invariance **is the point**. Policy-invariant
tower shaping measurably left *pure defence* as the true optimum against this
engine's opponent: over 8,300 episodes win-condition usage rose to 7.3% early
then decayed back to **0.7%** — the agent stopped playing its win condition at
all. `DRAW_PENALTY` exists for the same class of reason: a timeout used to be a
guaranteed-zero outcome, i.e. safe.

### Curriculum (both phases run the same ladder)

**ELEVEN rungs since 2026-09-03**, gated on **raw** win rate ≥ **0.65** over
100 episodes — plus a plateau valve and a backstop, because a level gate alone
demonstrably cannot end a stall. It was six rungs at 0.80.

**The rungs are the TEACHER'S LOOKAHEAD, not an elixir handicap** (corrected
2026-08-25; this section still described the retired multiplier ladder — the
pivot itself is in `DECISIONS.md`, "2026-08-19: the curriculum pivot"). Read
`TEACHER_STAGES` in `opponents/teacher.py` for the live values:

| rung | horizon | epsilon | k_cells | max_combos | reactive | was |
|---|---|---|---|---|---|---|
| 0 | 0 t (rules only) | 0.30 | 1 | 0 | no | stage 0 |
| 1 | 10 t (1 s) | 0.20 | 1 | 0 | no | stage 1 |
| 2 | 20 t (2 s) | 0.15 | 1 | 0 | no | |
| 3 | 20 t (2 s) | 0.12 | 2 | 2 | no | |
| 4 | 30 t (3 s) | 0.10 | 2 | 2 | no | stage 2 |
| 5 | 40 t (4 s) | 0.08 | 2 | 3 | no | |
| 6 | 50 t (5 s) | 0.05 | 2 | 3 | no | stage 3 |
| 7 | 60 t (6 s) | 0.04 | 3 | 3 | no | |
| 8 | 70 t (7 s) | 0.02 | 3 | 4 | no | stage 4 |
| 9 | 85 t (8.5 s) | 0.01 | 3 | 4 | no | |
| 10 | 100 t (10 s) | 0.00 | 3 | 4 | **yes** | stage 5 |

**WHY: the old table moved three and four knobs per rung**, so "advance one
stage" was never a small step — 2 → 3 was horizon 30→50 *and* epsilon
0.10→0.05 *and* max_combos 2→3. The measured cost is in
`runs/phase4/train-20260828-rolling-spells.log`: **episodes 9,640 → 32,680,
23,040 consecutive episodes (~24 h at 943 ep/h) at stage 3, win rate
oscillating 0.15–0.60 against a 0.80 gate, ZERO advances.** Each rung now moves
one axis (rung 3 moves two only because `max_combos` is inert below
`COMBO_MIN_HORIZON_TICKS`).

**A SAVED `curriculum_stage` IS AN INDEX INTO WHATEVER TABLE WAS LIVE**, so the
6→11 change silently reinterprets every older checkpoint — old stage 3 is 5 s
and new rung 3 is 2 s. `teacher.remap_legacy_stage` converts by HORIZON and
`state_dict` now stamps `teacher_table_size` so the remap runs exactly once.
The same trap caught two constants that named a rung by literal:
`PHASE2_MIN_CURRICULUM_STAGE` (now `remap_legacy_stage(4)` = rung 8, still
70 ticks) and `make_replays.play_and_log_vs_teacher(stage=...)` (now the derived
top rung). **Grep for any new literal that indexes this table.**

**Three ways off a rung, and the level gate is the weakest of them.**

| exit | condition | meaning |
|---|---|---|
| gate | 100-ep win rate ≥ 0.65 | mastery |
| **plateau** | 500-ep mean stopped improving for 1,500 ep, and ≥ 0.40 | converged |
| backstop | 500-ep mean < 0.40 and no improvement for 4,000 ep | over-promoted → **demote** |
| stall | 100-ep win rate ≤ 0.10 sustained | catastrophe → demote |

The old ladder had only the first and last, i.e. a gate for MASTERY and a valve
for CATASTROPHE and **nothing for the band in between** — which is where a
curriculum run actually spends its time, and is exactly the state the 23,040
episodes were spent in. Replaying that oscillation through the new manager fires
before episode 4,000 (`test_curriculum_plateau.py`) -- ~19,000 episodes, about
20 hours, returned to the run.

**The plateau is the WORKHORSE and the gate is the fast path, and that is forced
by PFSP rather than chosen.** With the deck pool on, sampling weights a deck by
`(1 - win_rate)^2`, so the run is deliberately spent on the worst matchups and
the readable win rate is REGULATED toward the hard end of the pool. Measured on
the 2026-09-03 sweep's own numbers: the agent beats three decks at 0.70–1.00 and
the episode-weighted pool win rate still reads **0.581**. So a level gate can be
structurally unreachable on a heterogeneous pool no matter how strong the agent
gets — the same unreachability the 0.80 mirror gate had, reached by a different
road. `PLATEAU_MIN_WIN_RATE = 0.40` is a floor on that PFSP-weighted mixture and
is **not comparable to a mirror win rate**.

**The teacher's play bar tapers from 6.0 elixir, not 9.0, since 2026-08-28.**
`play_margin = 3.0` is the utility a candidate must beat to be worth playing,
and `effective_play_margin` tapers it to zero as the bar approaches overflow.
That taper was anchored on `ELIXIR_OVERFLOW_AT = 9.0`, which left the bar at its
FULL height for every elixir value from 0 to 9 -- i.e. across almost the whole
operating range -- so the freeze the taper exists to prevent happened anyway.

Measured on `replays/replay_ep2018.json` (stage 1): the teacher lost a Princess
Tower at **tick 159** having spent 2 elixir (one Ice Golem, tick 111) while its
bar ran 5.0 -> 8.6, and its next play landed at **tick 561** -- the exact tick
elixir first reached 9.60, i.e. the first moment the taper did anything.
`MARGIN_TAPER_START = 6.0` is now a separate constant; `score`'s own
`overflow_relief` still uses 9.0, because that one is about refunding the COST
charge when income is genuinely being discarded, which really is a 9.0 question.

GAMEPLAY-AFFECTING for phase 1: the stage 0/1 teacher defends materially more,
so every curriculum win-rate gate is calibrated against a different opponent and
win rates earned before this date are not comparable across it.

Stage 5 has `win_rate_threshold = None` — there is no further auto-advance, so
the ladder has no natural end and a stopping rule has to be chosen by hand.

The paragraph below is the history of the ladder this REPLACED, kept because it
is why difficulty is lookahead rather than economy.

Gradual steps replaced an old `1.0 → 1.75 → 3.0` jump where the agent went
0-for-2000+ episodes at 1.75x with zero improvement. The gate was 0.90 and that
was a measured dead end: the final stage's opponent gets 1.5x elixir, and the
agent plateaued around 0.86 after ~76k episodes of dedicated stage-5 training.

Phase transition is evaluated **before** stage advancement — ordering is
load-bearing, because the stage gate calls `outcome_history.clear()`.

### Phase 2 league (`train_selfplay.py`)

**PFSP**, not a ladder. Every env, every reset, independently samples an
opponent weighted `max(floor, (1 − winrate)^2)`. This replaced a linear ladder
that had two structural failures: once the genuinely-weak snapshots were
exhausted every "next" opponent was a near-mirror of the trainee (so the gate
became unreachable — a mirror matchup sits at 50% by construction), and a
ladder never revisits, so nothing prevented catastrophic forgetting.
`PFSP_MIN_WEIGHT = 0.05` guarantees every opponent stays in rotation forever.

Pool members:

- **Historical snapshots**, saved every **2,000** episodes
  (`HISTORICAL_CHECKPOINT_INTERVAL_EPISODES`). Phase-2 snapshots younger than
  `MIN_OPPONENT_AGE_EPISODES = 6000` are excluded — otherwise the pool fills
  with coin-flip mirrors of the current trainee. **Both numbers were stale here
  (5,000 and 15,000); corrected 2026-08-25.** They are COUPLED — the age gate
  is kept at exactly 3x the interval, and `checkpointing.py` says so — so
  changing one alone silently changes which snapshots are eligible. The
  interval was re-denominated 5,000 → 2,000 on 2026-08-09 because the
  2026-08-07 speed fix made one episode carry ~2.2x more transitions.
- **4 scripted bots** — Rusher / Defender / Cycler / Counter, permanent members.
  Defender+Counter get `DEFENSIVE_SCRIPTED_MIN_WEIGHT = 0.8`, overriding PFSP,
  because PFSP's own criterion works *against* seeing them: mastering them
  drives their weight to the floor. First tried at 0.20 and confirmed too low —
  in a ~98-member pool that is ~7.7% combined share, statistically invisible.
- **Exploiters** (`exploiter.py`) — **currently disabled** (`EXPLOITER_ENABLED
  = False` since 2026-08-11), so the pool has had no new exploiter members since
  then. See below.
- **`BUILTIN_ANCHORS`** for evaluation only: the C++ heuristic at 1.00/1.35/1.50
  elixir, Elo 1200/1500/1700. These must use `gym_wrapper.MicroRoyaleEnv`, not
  `MicroRoyaleSelfPlayEnv`, because `stepSelfPlay` deliberately never calls
  `opponentTurn()` — the heuristic would silently never run.

**Scenario injection** (`SCENARIO_INJECTION_PROB = 0.30`): start 30% of episodes
already facing a win-condition on the bridge. A "defend or lose the tower in
~40 ticks" moment is rare and its causal link is buried in a long GAE trace, so
the reflex never accumulated gradient. The opponent is *not* frozen during a
scenario, and truncation uses a value bootstrap, never a terminal.

**Live strategy diagnostics** in the console line: `Cards/Game`, `Plays`,
`Elixir@Play`, `AvgTicks`, `ScenDef`, `ScenOff`, plus the return-side block
added 2026-08-11 — `ROI`, `Worst`, `Fwd`, `TwrDmg/1k`. Every PPO update also
prints `H(place|card)` per deck card and the no-op arm.

`ROI` is `get_elixir_value_killed_by / get_elixir_spent_on_card`, summed over
the deck. It exists because **spread is not evidence of improvement**: raising
placement entropy spreads placements whether or not the policy got better, so
`Fwd` and the entropy series can all rise from noise alone. ROI is the half
that noise pushes the *other* way. Read them as a pair — spread up with ROI up
or flat is learning; spread up with ROI down is randomness.

Definitions that are load-bearing: ROI is a ratio of **sums** over the window,
not a mean of per-episode ratios (an unplayed card contributes 0/0, and
averaging those moves the number for reasons unrelated to the card). `TwrDmg`
is per 1000 **ticks**, because `AvgTicks` moved 24% in one hour after the
placement-mask fix and unnormalized damage would have read as more pressure
when it was only longer matches. `Fwd` is confounded with entropy by
construction and is only interpretable next to ROI.

### Exploiter (`exploiter.py`)

**TURNED OFF since 2026-08-11 — `EXPLOITER_ENABLED = False`, so
`should_run_burst()` returns `False` unconditionally and no burst has run since.**
Everything below describes the mechanism as built and as it behaves when
re-enabled; none of it is running today. It was disabled because the main agent
was found to be reward-hacking (parking buildings at y=0), and an exploiter
cannot punish a strategy whose payoff comes from the *reward function* rather
than from the opponent — so the burst spends ~17% of throughput learning to beat
something that is not the actual problem. The re-enable condition is stated in
`exploiter.py`: fix the engine/shaping first, so a parked building is no longer
free. **Do not read a stagnation plateau as "the exploiter found nothing" —
check `EXPLOITER_ENABLED` before assuming it ran at all.**

AlphaStar's league exploiter — the piece the pool was missing. Every neural
opponent in it is a *past self*, so self-play was free to cycle rather than
improve. A separate net trains **only** against a frozen copy of the current
main agent, is free to find one specific hole, snapshots into the shared pool,
and is re-seeded from the main agent every 2 bursts so it looks for a *new* hole.

1000-episode bursts every **20,000** episodes of **pipeline 2's own** counter
(which starts at 0 at handoff, not continuing phase 1's), first at its ep
10,000.

**The cost was badly underestimated at first.** The original ~14% tax divided a
57-minute burst by an assumed 6.5-7 hours per 10,000 main episodes. Both halves
were wrong: a real burst takes **4,512 s (75 min, 4.5 s/episode**, not the
3.4 s/episode extrapolated from a 66-episode sample), and the main loop covers
10,000 episodes in **~2.4 h** (4,185 ep/hour, measured between two pipeline-2
snapshot timestamps with the intervening burst subtracted). That is a **~34%**
tax at a 10,000 cycle, which is why the cycle is now 20,000 — about 17%, i.e.
what the original figure was believed to cost.

**Entropy coefficients here are fractions of each head's maximum**, divided by
`log(N)` exactly as both trainers do. They were raw nats until 2026-07-31, which
was harmless for the card head (`log 5 = 1.609`) and catastrophic for placement
(`log 612 = 6.417`): the placement coefficient was effectively **73×** the main
agent's, the entropy bonus reached ~0.96 against an actor loss of ~0.05, and the
policy dissolved into uniform placement — 51.9% → **86.7%** of max entropy, 28 →
260 effective cells, top-1 probability 0.260 → **0.018**. It went 166-839-2
against the agent it was a bit-exact copy of. The card head, whose coefficient
happened to be right, was untouched at 39.9% → 36.9%. Two heads, one network,
one optimizer, one batch, and only the mis-scaled one collapsed. The burst now
prints its placement-entropy drift and warns above 0.70 so this cannot recur
silently.

**Both bursts run so far found nothing**, and the reason is instructive: they
predate the team-1 observation fix, so the true null was not 0.50 but **0.598**
(see Board geometry). Burst #0 scored 0.536 — *below* null — and burst #1 scored
0.585, i.e. on it. Roughly 2.5 hours of compute measured a board artifact. Burst
#1's five-slice trend was flat (0.59 → 0.54 → 0.56 → 0.66 → 0.55, χ² ≈ 5.9 on
4 df), which is the diagnostic the slices exist for: rising means the
1000-episode budget binds, flat means there is no gradient to follow. **No burst
has yet been read against a valid null.**

Deliberately a **self-contained module with its own compact PPO loop**, not a
mode switch inside the main loop. A mode switch would need guards scattered
through ~700 lines to keep exploiter episodes out of the main episode counter,
reward histories, entropy schedule, eval cadence and checkpoint writes — one
missed guard silently corrupts the run. Contained, the worst case is a weak
snapshot. Verified: after a full burst the main agent's weights are
bit-identical and carry no gradients.

Its snapshots are eligible **immediately** (the age gate only matches
`_pipeline2_ep<N>` filenames) — an exploiter goes stale as the hole is patched,
not as it ages.

### Behaviour cloning (`bc_pretrain.py`) — built, deliberately not wired in

Pinned `.npz` schema (`DATASET_SCHEMA`), a recurrent BC trainer, and
`action_match_rate`. `load_dataset` **refuses** a dataset recorded against a
different `observation_size()` rather than reinterpreting it.

Not called from either trainer. Cloning the *scripted* teachers is of unclear
and possibly negative value — they are weak and narrow, and seeding the policy
inside their behaviour is the opposite of what the league buys. Measured on 60
episodes / 1,799 placements: train loss 51.96 → 44.21 monotone, held-out card
match 0.300 → 0.319, held-out **cell match 0.073 against a 0.273
always-guess-the-modal-cell baseline** — i.e. it has not learned the marginal,
let alone state-dependent placement. Underfitting, not a defect.

The value of the module today is that the pipeline is verified end to end, so
when `perception/` can emit real human demonstrations the only new variable is
the data. **The recordings now exist** (8 matches in
`perception/assets/recordings/`), so the extraction step is the live blocker.

---

## How the learning mechanism got here — moved to `docs/DECISIONS.md`

The chronological narrative — every measured result, every reversal, and the
reasoning behind the constants above — was extracted to **`docs/DECISIONS.md`** on
2026-08-24. 12 sections, unchanged and with their headings intact, so any
passage cited by title still resolves.

Go there for *why*; stay here for *what is true now*. In particular the
following live in `DECISIONS.md`, and are the ones most often cited:

| section | what it settles |
|---|---|
| 2026-08-19: the curriculum pivot | why the 1.5x multiplier was replaced, and the hypothesis it did NOT confirm |
| 2026-08-19 (later): DEPLOY TIME | the offence/defence balance fix, and the ~520 tower HP that second is worth |
| 2026-08-20 (later): the MULTI-CARD teacher | `TODO.md` item 1 — combos, and the win-rate NULL reported as one |
| 2026-08-21: the teacher's ECONOMY | `play_margin` 0.05 -> 3.0, and why `w_pos` is REFUTED |
| 2026-08-21: REACTIVE ROLLOUTS | the one-directional bias, and the horizon gate |
| 2026-08-20: the simulator audit | two absorbing states, one fixed; sight vs attack range |
| 2026-08-21: four fidelity fixes | the arena's coordinates, King activation, blind lane pathing |
| 2026-08-24: the placement head | row compaction, 1.52x, and why bit-exactness is unavailable |
| 2026-08-24: the live-mirror state setters | item 22 — the setters Stage 2 calls, and the saturating probe that measured zero |
| 2026-08-26: the C++ simulator audit | seven defects: Ice Golem/Ice Spirit, the freeze off-by-one, building expiry, the Mortar/King symbol clash |
| 2026-08-26 (perf): the hot paths | 2.4-4.4x on collisions, 25x on the observation encoder, and the bit-equivalence sweep |
| 2026-08-26 (part 2): abilities and spawned units | the backwards pull, Hero Giant hurling buildings, and the speed tiers that never reached a Golemite |

---

## NEXT UP: Stage 2 — see `TODO.md` item 0

The full brief — goal, the two modules to write, the three invariants that can
go silently wrong, and the limits already measured — is **`docs/TODO.md` item 0**,
because pending work belongs in the pending-work list and this file had a second
copy of it. Design: `docs/design/specs/2026-08-24-live-teacher-play-design.md`.
The measurements behind it: `DECISIONS.md`, "2026-08-24: the live-mirror state
setters".

One line of it belongs here rather than there, because it is an engine fact and
not a task: **do not add threads.** `src/bindings.cpp` releases the GIL
**nowhere**, so a rollout thread BLOCKS perception rather than running beside
it, and a concurrent writer would score candidates against different worlds.
The budget does not require one — teacher 14.1 ms + rebuild ~1 ms against a
1000 ms period.

---

## Measured baselines — use these, don't re-derive them

> **Checkpoint names in the passages below are PROVENANCE, not files.** The
> 2026-08-19 cleanup kept only `model_weights.pth` (ep 7,063, phase 1) and
> `model_weights_selfplay.pth` (ep 31,312, the 2.6 Hog Cycle baseline).
> `model_weights_cured.pth`, `_hires`, `_dist_e3` and the `archive_pre_*`
> lineages were deleted along with the Giant deck they were trained on. A number
> attributed to one of them still says where it came from; the weights are gone.

**EVERYTHING IN THIS SECTION PREDATES THE 2026-08-07 MOVEMENT-SPEED FIX AND
NO WIN RATE BELOW SURVIVES IT.** Troops now move at ~1/5 the speed every one
of these numbers was earned at, which changes the relative value of every card
in the deck (see "Engine facts"). **The episodes/hour figures do NOT hold
either** — measured 1,301 ep/hour on the first post-fix run against the 2,873
recorded below, a 2.2× drop. This was written here as "throughput still holds,
it is wall-clock not gameplay", and that was wrong: a match whose troops move
5× slower needs far more TICKS to reach a decision, so each episode now
contains proportionally more transitions. It is not a timeout effect — the
draw rate is 0.00, matches still finish inside `maxTicks`, they just use more
of the clock. What *does* still hold is transitions and gradient steps per
hour, which is the quantity this file already says to budget in. A phase 1 to
~60k episodes is now ~46 h, not ~21 h. The *methodological* baselines hold
(opponent-elixir MAE ≈ 1.35 for predict-the-mean, 0.273 for
always-guess-the-modal-cell, the Elo formula's behaviour near 1.0). Treat
every win rate, reward curve and stage number as historical.

**Every phase-2 win rate and Elo below predates the 2026-07-31 observation fix
and is not comparable across pipeline-2 episode 31,753.** Before that fix the
trainee was always on the favoured side (0.598 against a bit-exact copy of
itself), so self-play win rates were inflated and PFSP's `(1 - winrate)^2`
weighting was distorted with them. The 3 *historical neural* Elo anchors got
stronger across the fix; the 3 *builtin heuristic* anchors did not, since
`HeuristicOpponent` is C++ and never reads the observation. **Phase 1 is
unaffected** for the same reason, and checkpoints are *not* invalidated — the
`team == 0` branch was untouched, so the trainee's own inputs are bit-identical
and `model_weights_selfplay.pth` resumed normally.

**Throughput** (i5-13420H, CPU-only, `num_envs = 8`):

| | ep/hour | updates/hour |
|---|---|---|
| before the batched update | 1,167 | 36 |
| historical, phase 1 (pre 2026-08-07) | 2,873 | 92 |
| historical, phase 2 (pre 2026-08-07) | 4,185 | — |
| historical, phase 1 (post speed fix) | 1,301 | — |
| **MEASURED 2026-08-25, phase 1 stage 0** | **943** | **77** |

**Use the 943 / 77 row to size a run.** Measured directly after the 2026-08-24
speed-tier rework on a fresh random-init policy: 420 s, `num_envs = 8`, stage 0,
110 episodes and 9 updates — **3.82 s/episode, 46.7 s/update**. Every row above
it describes an engine that no longer exists.

Two things to read correctly. **Episodes/hour fell 3x against the 2,873 and only
0.84x against the updates figure**, which is the same decoupling this file
already records for the 2026-08-07 fix: a longer match carries proportionally
more transitions, so gradient steps per hour is the quantity that is roughly
conserved and the one to budget in. And this row is **stage 0** — the teacher
is rules-only there (0.62 ms/decision) against 12.4 ms at stage 5, and stronger
opposition also lengthens matches, so expect ep/hour to FALL as the curriculum
advances.

At this rate: **10k episodes ≈ 10.6 h, 40k ≈ 42 h, 60k ≈ 64 h.**

Phase 2 is the faster of the two and the ~2.4 h it takes to cover 10,000
episodes is the number to divide by when sizing anything against it — assuming
6.5-7 h there is what made the exploiter's cost estimate 2.5× too low.

Where update time goes (87% of wall clock is the PPO update, 13% the rollout;
within one BPTT chunk): **CNN trunk 43%, placement head 41%, LSTM loop 16%.**
The conv placement head costs **33×** the dense head it replaced — 11× fewer
parameters but ~33× the FLOPs, a deliberate trade for translation equivariance.

`forward_sequence` batches everything downstream of the LSTM into one call over
`L·B` instead of `L` calls at `B`. Measured **1.82× on the forward, 2.54× on
end-to-end updates/hour** (the backward benefits too). Verified equal to the
looped path: values/aux/hidden bit-identical, `-inf` masks identical, PPO ratio
exactly 1.0, max logit delta **1.1e-08 vs a float32 eps of 1.19e-07**.

**Current run (v4), 10,030 episodes / 3.5 h, stage 1:**

| | |
|---|---|
| win rate | 0.313 → 0.733 |
| avg reward | −0.73 → +1.00 |
| critic explained variance | 0.000 → 0.718 (max 0.832) |
| `Aux/OppElixir_MAE` | 1.27 → 0.83 (min 0.77) |
| total entropy | 6.72 → 4.98 |
| ClipFrac | peaked 0.275, now falling |

Baselines that make those numbers mean something:

- **Opponent-elixir MAE**: predict-the-mean ≈ **1.35**; "read your own elixir"
  is *no better* than the mean (ratio 0.96–1.05 across seeds, corr ≈ 0.27).
  **That control was the wrong one, and the conclusion drawn from it —
  "0.77 is a real capability, not a trivial correlation" — is WRONG. Corrected
  2026-08-27.** The opponent's elixir is an *affine function of two scalars the
  observation already carries*: `elixir(t) = start + rate*t − spent(t)`, where
  `t` is extra-scalar 0 and `spent(t)` is extra-scalar 2. Ordinary least
  squares on those two, four parameters and no recurrence at all, scores
  **MAE 0.0000** (2,606 samples, 8 episodes; each scalar ALONE scores ~1.42,
  i.e. no better than the mean, so the fit genuinely uses both).

  Two consequences. The aux head's 0.77 is a **shortfall against an achievable
  zero**, not a capability. And the auxiliary loss exerts almost no
  representational pressure — it is arithmetic on two present inputs, not
  opponent modelling, so it cannot be doing the job it was added for.
  `python_ai/tests/test_aux_task_is_not_a_memory_probe.py` pins the
  measurement. A task that would actually require memory — predict which CARD
  the opponent plays next — is `perception/UPSTREAM_REQUESTS.md` item 24.
- **Placement cell match**: random is 1/612 = 0.0016, but the scripted teachers
  use only ~75 distinct cells and always-guess-the-modal-cell scores **0.273**.
  Compare against the marginal, never against random.
- **Elo from the anchor roster** is `400·log10(1/score − 1)` and explodes near
  score 1.0 — a single lost game at score 0.9 moves it ~141 points. A saturated
  anchor is simultaneously uninformative and extremely noisy.

**Comparing runs honestly:** at matched episodes 0–10,100, the current
architecture is **roughly level with the old 60k baseline on win rate** (it was
behind between ep 2,500–7,500 and overtook at the end), both reaching stage 1.
It plays the *corrected symmetric river*, so it earns similar numbers on a
harder board — but that is not quantified. The clear wins so far are throughput,
critic quality, and new capabilities; **not** demonstrated end strength.

---

## Measurement discipline — the traps this project has actually fallen into

Migrated here from `handoff.md` on 2026-08-19 when the session handoffs were
retired. Each one cost real time; none of them is hypothetical.

- **Never validate a mask against the predicate that generated it.** A live run
  reported "0 illegal placements out of 32" by checking placements against
  `is_valid_placement` — the same predicate the mask was built from. The check
  was circular and could not fail. The placements were in fact landing off the
  board entirely, for an unrelated reason (the tile grid), and this test was
  structurally incapable of noticing.

- **A cross-check anchored where the error is zero proves nothing.** The bad
  2026-08-05 tile-grid refit passed its own check — "engine x 9.0 lands on
  display centre 360" — because both fits were centre-anchored and the scale
  error grows outward. It was wrong by 28 px at the board edge. **Check the
  edges of the range you care about, not its middle.**

- **When a measurement's failure mode is maximal permissiveness, it needs an
  internal control that MUST fire.** Every failure mode of the deploy-zone probe
  returns "everything is legal": an unaffordable card selects nothing, a stale
  baseline already holds the tint, a tap outside the board is untinted because
  there is no board there. The fix is a band deep in the enemy half that has to
  be tinted whenever a card is really selected.

- **The game is a better oracle than the simulator, and it is free.** Clash
  Royale tints the region you may not deploy into, so differencing a selected
  frame against an unselected one reads the real rule for 288 cells at once, at
  zero elixir. That answered in one screenshot what a season of
  `is_valid_placement` comparisons could not.

- **Screen on a different metric than you confirm on, and never extend a run
  after seeing a marginal result.** An exploratory n=200 arm gave +0.105 at
  p=0.044 and it was NOISE — a confirmatory run at 4x the power collapsed it to
  +0.016. Continuing the first run instead of launching a fresh one with n fixed
  in advance is optional stopping, and it manufactures results.

---

## Open problems — the EVIDENCE behind them

> **`docs/TODO.md` is the authoritative task list.** It was consolidated on
> 2026-08-19 from this section and from three session handoffs, and every item
> in it was verified against the source tree. This section is not a second task
> list: it is the measured history that justifies those items, kept here because
> the numbers are the part that is expensive to re-derive. **If the two ever
> disagree, `TODO.md` is what someone is acting on — fix this section.**

**Ranked by expected value, from the architectural review.**

**NEXT UP (2026-08-19) — upgrade the Utility Teacher to evaluate MULTI-CARD
COMBO placements (e.g. Ice Golem + Hog), so it can exploit the engine mechanics
deploy time just created.** This is `TODO.md` item 1. Numbered separately from
the list below only because those items cross-reference each other by number; by
expected value this is now the top item.

*Why it is top.* Deploy time made the escorted push the correct play and the
naked push the punished one, measured on the same engine: a lone commitment
scores **−556.3** tower HP marginally while a supported one scores **+448.5**
(CI [+137.3, +760.1]), and escorting is worth **+650 HP** in a punish window.
The teacher cannot make that play. `UtilityTeacher._cells_for` proposes cells
for ONE card per decision and `score` ranks single candidates, so its whole
attack repertoire is "send the win condition to a bridge, alone" — the exact
play the new physics correctly punishes.

That is why the strategy-level win-rate arm still reads attack 0.490 vs cycle
0.715. **That number is now a property of the teacher's repertoire, not of the
engine**, and it is the one place the two can still be confused.

*What it needs.* Candidate generation over short SEQUENCES rather than single
cells — at minimum (tank now, win condition next decision, same lane) — and a
score that can attribute value to the pair. The rollout machinery already
supports it: `rollout_stats` takes a candidate and rolls forward, so a two-step
candidate is a two-step rollout on the same snapshot. The cost is the thing to
watch, since width is what search is expensive in (an engine step is 0.015 ms;
each extra candidate is a whole rollout) — so enumerate a handful of curated
combos, not the cross product.

*Do NOT confuse this with a fifth Hog mechanism.* The four that returned null
all tried to move a POLICY toward a play the environment priced negatively. This
is the opposite situation: the environment now prices the play POSITIVELY and
the teacher simply cannot express it.

1. **Human-replay imitation is now unblocked.** The recordings exist; the
   extraction step (`perception/` → the `bc_pretrain` `.npz` schema) is the
   blocker. Self-play discovers strategies but not *the distribution humans
   play*; AlphaStar's supervised stage was load-bearing, not optional.

2. **Decision-time lookahead / search — UNBLOCKED 2026-08-11, mechanism built
   and exposed.** A fast deterministic simulator is owned and was unused. Roll
   the top-K candidate actions forward and pick by value; that is a strict
   improvement operator, and distilling it back is expert iteration. The engine
   is **~150× cheaper than the network that scores it** — a 20-tick rollout
   costs 0.34 ms against one `MicroRoyaleNet` forward at 50.51 ms — so search
   here is not compute-bound on simulation at all.

   **What now exists:** `Board::deepCopy()`, `GameManager::snapshot()` and
   `ClashEnv::snapshot()`, bound to Python as `env.snapshot()`. Measured from
   Python: **0.033 ms per snapshot**, 0.027 ms per 10-tick step, so a **K=12
   sweep at a 2 s horizon costs 1.1 ms** — against ~50 ms for the single network
   forward that scores it. Simulation is free; *scoring* is the entire budget,
   which is why `python_ai/eval/search_ab_test.py` batches all K candidate
   evaluations into one forward.

   **An unexpected second payoff: this is also a seeding substitute.**
   `UPSTREAM_REQUESTS.md` item 7 asks for `ClashEnv::seed()` because unpaired
   A/B tests need ~1,568 episodes per arm to resolve 5 win-rate points. Snapshot
   gives the same pairing without it — reset once, snapshot, hand both arms a
   bit-identical opening (same shuffled hand, same heuristic lane). Item 7 is
   still worth doing for reproducible *failures*, but it is no longer the
   blocker on paired experiments.

   **Ceiling effect, measured, and it will bite any future A/B here:** against
   `heuristic@1.00` the ep-64k policy wins ~100%, so both arms of a search-vs-
   policy comparison saturate and the delta is pinned at 0 by the *opponent*,
   not by search being useless (4/4 trials, 1.000 vs 1.000). Run comparisons at
   **1.5× opponent elixir**, where there is headroom on both sides.

   **First measurement, 2026-08-11 — search wins, and by a lot.**
   `python_ai/eval/search_ab_test.py`, 160 paired trials at 1.5× opponent elixir,
   ep-64k checkpoint, `DEFAULT_DECK`, K≈3 candidates, 4 s horizon, critic-scored:

   | | |
   |---|---|
   | greedy policy win rate | **0.625** |
   | + 1-ply search | **0.944** |
   | paired delta | **+0.319**, 95% CI [+0.237, +0.401] |
   | discordant pairs | 56 search-better / 5 search-worse / 99 tied |
   | exact McNemar | **p = 5.6e-12** |
   | deviation rate | 13.8% (5,764 of 41,792 decisions) |
   | cost | 2.2× wall clock (11.9 → 26.0 s/episode) |

   **Read the deviation rate first.** Search overrode the greedy action on only
   ~1 decision in 7, and greedy is always candidate 0, so search can only
   deviate when the critic prefers something else. A +32-point swing off 13.8%
   of decisions is the headline, but the *interpretation* is the useful part:
   the scorer is this same network's own critic, so **the value head is
   substantially better than the action head is at exploiting it.** That is
   precisely the condition under which expert iteration pays — there is a gap
   to distil, and it is the policy head, not the critic, that is underfit.

   **What this does NOT establish**, stated plainly because item 13's history is
   an underpowered null that was nearly over-read in the other direction:
   one checkpoint, one deck, one opponent (the C++ heuristic), one horizon. It
   says nothing about neural opponents in the PFSP league. The effect is also
   regime-specific: at 1.0× elixir it is exactly zero, by ceiling.

   **2026-08-12 — distilling it back does NOT work yet. Measured, negative.**
   `python_ai/trainers/expert_iteration.py`. 80 episodes of search-labelled play (20,333
   decisions), distilled into the policy with the trunk/LSTM/critic frozen and
   only the action heads trainable (15,878 of 1.88 M params):

   | | null | after distillation |
   |---|---|---|
   | `card_match_decisions` | 0.6330 | **0.6960** (+0.063) |
   | `cell_match` | 0.5765 | 0.5730 (**−0.004**) |
   | critic drift `|dV|` | — | **0.000000** (freeze verified) |

   The behaviour transferred. **The win rate did not.** Paired greedy-vs-greedy,
   no search in either arm:

   | run | n | original | distilled | delta | p |
   |---|---|---|---|---|---|
   | exploratory | 200 | 0.570 | 0.675 | +0.105 [+0.008, +0.202] | 0.044 |
   | **confirmatory** | **800** | **0.634** | **0.650** | **+0.016 [−0.030, +0.061]** | **0.553** |

   **The exploratory p = 0.044 was noise and the confirmatory run at 4× the power
   killed it** — 178 better / 166 worse is a coin flip. Do not resurrect the
   +0.105; it is the single most over-readable number this project has produced.
   The confirmatory run was launched as a *fresh* experiment with n fixed in
   advance rather than by extending the first, because continuing after seeing a
   marginal result is optional stopping and would have manufactured a result.

   Three things worth carrying:

   - **The control arm's own variance is the trap.** Original greedy measured
     0.625, 0.570, 0.700 and 0.634 across four runs at 1.5×. Any comparison here
     under a few hundred paired trials is measuring that, not the treatment.
   - **What search does is WAIT.** Of 2,126 card-level deviations, **1,847 were
     search declining to play where greedy plays**, against 137 the reverse —
     13:1. But search waits *conditionally*, when the critic dislikes this
     specific play; BC on 89.5%-already-agreed labels mostly teaches the
     *marginal* "no-op more often". Same blind spot this file already records
     three times, inverted: imitating an aggregate that cannot express the
     conditional.
   - **The placement head is starved by construction.** The expert no-ops 89.7%
     of the time, so placement trains on 10.3% of rows (2,099) — near the 1,799
     that already underfit in `bc_pretrain`. Half of what search does (781
     cell-level deviations) transfers not at all.

   **Those two levers are now measured DEAD, and a third works. 2026-08-12.**

   The obvious fixes — upweight the disagreement rows, unfreeze the trunk —
   were ablated as a 2×2 grid (`--ablate`), 64 train / 16 held-out episodes.
   Held-out `disagreement_match` produced a clean-looking ladder, 0.235 → 0.246
   → 0.372 → 0.655, **and it is an artifact.** 87% of the expert's overrides are
   "wait where greedy plays", so a policy that simply no-ops more scores on that
   metric for free, and the four configs came out perfectly monotonic in their
   no-op rate (0.872 / 0.874 / 0.899 / 0.967) with each score predicted by that
   rate alone. Config D, the "best" by the old metric, waits on 80% of rows
   *regardless* of the expert — upweighting rows that are 87% "wait" by 8× just
   teaches always-wait.

   **`conditional_lift` is the metric that survives**, and it is the fourth time
   this file records the same lesson: an aggregate cannot see a conditional.
   Condition on rows where the ORIGINAL policy plays, then compare
   `p1 = P(policy waits | expert waited)` against
   `p0 = P(policy waits | expert played)`. Indiscriminate drift moves both
   together; only a state-conditional rule separates them.

   | config | lift | 95% CI | no-op | critic dV | p1/p0 |
   |---|---|---|---|---|---|
   | null | +0.0000 | — | 0.826 | 0.000000 | — |
   | hard-label, frozen | +0.0462 | ±0.0675 | 0.872 | 0.000000 | 1.19 |
   | hard-label, frozen, 8× | +0.0454 | ±0.0685 | 0.874 | 0.000000 | 1.18 |
   | hard-label, full | +0.0658 | ±0.0753 | 0.899 | 0.048086 | 1.17 |
   | hard-label, full, 8× | −0.0101 | ±0.0601 | 0.967 | 0.045452 | 0.99 |
   | **distribution, frozen** | **+0.1027** | **±0.0408** | **0.822** | **0.000000** | **3.07** |
   | **distribution, full** | **+0.1415** | **±0.0508** | 0.835 | 0.036895 | 2.42 |

   **What works is distilling the search's VALUE DISTRIBUTION, not its argmax**
   (`--train-dist`, AlphaZero's recipe). A single hard label strips the margin —
   it cannot distinguish "waiting is marginally better" from "playing here is a
   blunder", which is the distinction a conditional rule is made of. Recording
   the full ranked candidate set and fitting `softmax(values/T)` makes both arms
   significant where no hard-label arm was. The frozen arm's no-op rate is
   **0.822 against a null of 0.826**: it is not waiting *more*, it is waiting
   *differently*, 3:1 on the rows the expert judged bad, at a constant restraint
   budget. Prefer frozen — nearly the same lift as full, zero critic drift
   (the critic *is* the expert, so drifting it degrades future labels), and a
   sharper ratio.

   Three practical notes:

   - **Temperature is the knob, and it must be picked from the target's own
     entropy, not tuned on the outcome.** Candidate value spread is mean 0.221 /
     median 0.193, at which T=0.25 puts the target at 94% of maximum entropy —
     near-uniform, no signal, while looking like it is training. T=0.05 puts it
     at ~55%. `--target-entropy` prints the table.
   - **A no-op duplication silently destroyed the first attempt.**
     `(NOOP, gx, gy)` and `(NOOP, 0, 0)` are the same action — `step()` ignores
     placement for the no-op — but differ as tuples, so the no-op was emitted
     twice and scored twice, identically. Harmless for the win-rate A/B (the
     duplicate ties, argmax returns greedy) and fatal here: 99.1% of rows had a
     target whose median value spread was **exactly 0.0**. After the fix, 32.6%
     of rows carry a real distribution (median spread 0.193) and search is ~2×
     faster, since most rows now have one candidate and skip rollout entirely.
   - **Distribution labels also fix the placement starvation.** Hard labels
     trained placement on 2,099 rows (10.3%, near the 1,799 that already
     underfit in `bc_pretrain`); every row with ≥2 candidates now contributes
     placement gradient, 6,900 rows (32.6%).

   **It DOES convert to win rate — +0.045, and that is the whole of it.**
   Measured over three paired greedy-vs-greedy evals (no search in either arm),
   reported together because reporting only the last one would be dishonest:

   | net | n | original | distilled | delta | 95% CI | p |
   |---|---|---|---|---|---|---|
   | hard-label | 800 | 0.634 | 0.650 | +0.016 | [−0.030, +0.061] | 0.553 |
   | distribution, 4 ep | 800 | 0.629 | 0.666 | +0.038 | [−0.005, +0.080] | 0.095 |
   | **distribution + DAgger** | **1600** | **0.649** | **0.693** | **+0.045** | **[+0.013, +0.077]** | **0.0074** |

   The final result is significant and survives Bonferroni for the three tests
   (α = 0.0167). 377 better / 306 worse / 917 tied.

   **Read the effect size honestly: ~4.5 win-rate points, which is ~14% of the
   +0.319 that search itself buys.** And the last two nets are statistically
   indistinguishable from each other (+0.038 vs +0.045) — most of the jump in
   significance came from doubling n, not from DAgger making a better policy.
   Do not claim DAgger raised the win rate; claim it raised the conditional lift
   (+0.1535 → +0.1733 at fixed data budget) and that the win rate was then
   resolved by power.

   The coverage-linearity model — expected gain ≈ p1 × 0.319 — was stated before
   each eval and **over-predicted by 30–40% both times** (+0.050 predicted vs
   +0.0375; +0.072 vs +0.0447). It is useful for sizing experiments and should
   be discounted accordingly, not trusted as a point estimate.

   **The screening ladder, and the one lever that is actively harmful:**

   | config | lift | p1 | p0 | ratio | no-op |
   |---|---|---|---|---|---|
   | null | +0.0000 | 0.0000 | 0.0000 | — | 0.802 |
   | 4 ep / 80 eps | +0.1027 | 0.1523 | 0.0496 | 3.07 | 0.822 |
   | 16 ep / 80 eps | +0.1268 | 0.2236 | 0.0968 | 2.31 | 0.834 |
   | 16 ep / 180 eps | +0.1535 | 0.2850 | 0.1315 | 2.17 | 0.843 |
   | **DAgger / 180 eps** | **+0.1733** | 0.2924 | 0.1191 | **2.45** | 0.843 |

   More epochs fixed a genuine underfit (loss still falling, argmax-agreement
   still rising at epoch 3) and bought 47% coverage for one training run.
   **More DATA is where it goes wrong**: coverage kept climbing while the lift
   stalled, because p0 rose faster than p1 — selectivity fell 3.07 → 2.31 → 2.17
   and the no-op rate walked toward the expert's 0.896. That is the hard-label
   marginal-drift failure returning by a slower road. DAgger, at a *fixed* 180-
   episode budget (80 original + 100 from the distilled policy), is the only
   thing that reversed it, improving lift and selectivity together.

   Corroboration from collection rather than from held-out metrics: search
   overrides the distilled policy on **12.1%** of decisions against **14.8%**
   for the original. But search ON TOP of the distilled policy scores 0.940 vs
   0.944 — better candidate proposals did not make the expert better, which is
   the clearest single sign that the remaining gap is not a proposal problem.

   **Where this leaves the idea.** A reactive policy head amortises roughly a
   seventh of what 1-ply search does. The residual is plausibly structural: the
   critic gets to RUN the simulator four seconds forward, and the policy only
   ever sees `s`. Distillation is worth keeping — +0.045 for zero inference cost
   is real — but **search at inference remains ~7× more valuable than distilling
   it**, and that is the honest ranking of the two options.

   **This entry used to say it needed "a virtual `Entity::clone()` across the
   whole hierarchy plus effects — invasive simulation-core surgery". That was
   wrong on all three counts** and deterred the work for months. The reality,
   established 2026-08-11:

   - **`clone()` already exists** (`Entity.h`), overridden in four types as
     three lines of implicit-copy-constructor each. Copy-construction of
     concrete entities is already relied on in production.
   - **Effects need no deep copy.** All five effect interfaces declare `apply`
     `const`, and a search across every file defining or using them finds zero
     `mutable` and zero `const_cast`. They are stateless strategy objects, so
     sharing them across a snapshot is *correct* and deep-copying them would be
     wasted work.
   - **The hierarchy is 8 concrete types, not "the whole hierarchy"** —
     `MeleeTroop`, `RangedTroop`, `BuildingTargeter`, `RangedBuildingTargeter`,
     `AreaSpell`, `Projectile`, `Tower`, `Building`. Everything else
     (`CardEntity`, `CombatEntity`, `Troop`) is abstract or never instantiated.

   What it *does* need, and what the old framing missed entirely: `clone()`
   cannot be reused, because it sets `hp = 1` (it implements the Clone *card*)
   and its default returns `nullptr` rather than failing — so a naive
   "clone every entity" loop yields a board that has **silently dropped every
   tower, building, spell and projectile** and still looks like it worked.
   Hence a separate `snapshot()` with a throwing default.

   The one genuine hazard is **`Projectile::target`** (`Projectile.h`), the
   only entity-pointer *member* in the hierarchy — every other
   `shared_ptr<Entity>` is a per-tick local inside `findTarget`/
   `resolveCurrentTarget`. An implicit copy carries it verbatim, so a
   projectile in flight inside a snapshot would deal damage to an entity on the
   **original live board**. Being a `weak_ptr`, the symptom is wrong damage
   rather than a leak — invisible, not loud. `Board::deepCopy()` therefore
   builds an old-id → new-entity map and remaps that one member through it.

   **A second aliasing case of the same shape, found while implementing and
   not in the original proposal: `Board::statsEvents`.** The bus holds
   `shared_ptr<IStatsObserver>`, and unlike the effects those collectors are
   *stateful* — running damage totals, kill attribution. Copying the subscriber
   list would post every hypothetical hit in every rollout into the **live
   match's** statistics, and those feed the reward shaping, so a search would
   silently corrupt the returns it was being scored against. `deepCopy` starts
   the copy with an empty bus.

   **And the mirror-image trap one level up:** `GameManager::snapshot()` must
   *not* fix that by calling `MatchStatistics::attach()` on the copied board.
   `attach()` builds **fresh zeroed** collectors, so a snapshot would report a
   match where nobody had dealt any damage — and since the tower term is
   potential-based over *cumulative* damage, every candidate would score as the
   same enormous instant loss. That looks exactly like a working search that
   simply never prefers anything. `snapshotFor()` deep-copies each collector so
   totals carry over. Both directions are pinned by tests.

   Evidence and the full write-up: `perception/UPSTREAM_REQUESTS.md` item 13.

3. **Remaining observation gaps.** Cells still *overwrite* rather than
   accumulate in channels 0–7 (`obs[idx] = normalizedHp`), so a Skeleton Army
   collapses — `CH_COUNT` mitigates but does not fix it.

   ~~No card-cycle tracking (which of the opponent's 8 cards are available) — a
   core human skill.~~ **CLOSED 2026-08-27**: the opponent's `seen[]` and
   decaying `recency[]` are in the observation (see "AND AGAIN ON 2026-08-27"
   under Engine facts, and `UPSTREAM_REQUESTS.md` item 24). Note what that does
   and does not settle — the INFORMATION is now present and pinned by 17 tests
   across both suites; whether the policy exploits it is unmeasured and needs a
   phase 1. The cheap read is the auxiliary task item 24 proposes (predict the
   opponent's NEXT card, which unlike the elixir head is genuinely unsolvable
   from present scalars), not a win rate.

4. **One decision per second** caps tactical precision (see Action space).

5. **One deck, mirror matchups.** "A great player" implies arbitrary matchups.
   Phase 1's `random_opponent` and the scripted bots' randomised decks are
   partial; phase 2's neural opponents all play `DEFAULT_DECK`.

6. **Shaping is a hand-designed proxy and caps the ceiling.** Non-PBRS terms
   bias the optimum by construction. Troop *damage* is also the wrong metric —
   Clash is decided by kills and elixir advantage, not HP chipped. Long term
   these should anneal toward zero.

**Two concerns I raised and then measured away — do not re-raise without new
data.** The placement entropy coefficient pinning at its 0.01 floor for ~54% of
a run is **not** the floor binding: measured slope −0.0417/1000 ep, crossing
below target at ep 5,572 (I had linearly extrapolated ~16,400 from an
early-training slope, which was wrong), reaching 17.6 effective cells at best.
And low early ClipFrac is **not** a property of the conv head — it rises with
training in every run; at matched episodes the old and new architectures are
0.061 vs 0.071.

**...and THREE more, measured away on 2026-08-27 during the Phase 3 RL audit.**
Each of these is a standard, literature-backed thing to suspect, each looked
right from a grep, and each was refuted by the instrument. They are recorded
because refuting one costs an hour and re-suspecting it is free.

- **Missing orthogonal initialization is worth ~nothing here.** The net uses
  PyTorch defaults everywhere (only `place_hires[-1]` is explicitly zeroed), and
  "orthogonal init, gain 0.01 on the policy head" is one of the most-cited PPO
  implementation details (Engstrom et al. 2020; Huang et al. 2022). Its purpose
  is a near-uniform initial policy. Measured over 10 seeds on a fresh board,
  the DEFAULT init already delivers exactly that: card head **H/Hmax = 0.9996**
  (max prob 0.2089 against a uniform 0.2000), placement head **H/Hmax = 1.0000**
  (max prob 0.0017 against a uniform 0.0017), value head V(s0) = −0.009 ± 0.043.
  The reason is structural and will hold for any init: `hx` starts at exactly
  zero, so every head sees a zero input and emits ~zero logits. There is
  nothing for gain 0.01 to improve.

- **The scalars are NOT drowned out by the spatial features.** The LSTM input is
  1440 spatial + 64 scalar = 1504, so scalars are 4.3% of its width, which looks
  alarming for a game where elixir and hand contents decide everything. By
  gradient it is the opposite: `d(card logits)/d(input)` per DIMENSION is
  **2.26x higher for scalars** than for spatial cells. Width is not weight.

- **There is no measurable actor/critic destructive interference on the shared
  trunk.** Cosine similarity between the actor's and the critic's gradients,
  real returns and real normalized GAE advantages, 6 independent rollouts, at
  the checkpoint's own gamma: `cnn_trunk` −0.058 ± 0.191, `scalar_mlp`
  +0.103 ± 0.264, `lstm` +0.016 ± 0.053 — all indistinguishable from zero. The
  decisive detail is that `cnn_trunk`'s cosine **flipped sign between two runs
  of the same configuration** (−0.170 then +0.166), i.e. the statistic is
  noise-dominated and n=6 cannot resolve it at all. So splitting the actor and
  critic into separate networks — which would roughly double trunk cost on a
  machine where the trunk is already 43% of the update — is NOT justified by
  anything measured. `|0.5*g_critic|/|g_actor|` is 1.66 / 0.40 / 0.49 on the
  three shared modules, i.e. no runaway either. It reads ~2.55 on `cnn_trunk`
  when returns are computed at gamma 0.999 against a critic fitted at 0.99;
  that is the un-refitted critic, not an architectural property, and it is the
  transitional effect the gamma change predicts.

**Watch these three during any run:** ~~`Aux/OppElixir_MAE`~~ — **the head it
measured was DELETED on 2026-08-28.** The metric could not see what it was read
for (a stateless least-squares fit on two scalars already in the observation
scores 0.0000, so any reading below the ~1.3 threshold is consistent with zero
memory), and the task therefore shaped nothing. Replaced by
**`Aux/NextCard_CE` / `Aux/NextCard_Acc`** — cross-entropy and top-1 accuracy
on "which card does the opponent play next", which is NOT solvable from present
scalars and so is a real ask on the recurrent state. Read CE against
`ln(8) = 2.08` and accuracy against `0.125`: those are what a policy with no
cycle knowledge scores, and beating them is the entire point of the head. See
`test_aux_task_is_not_a_memory_probe.py` (why the old one went) and
`test_aux_next_card_task.py` (what the new one guarantees). The other two
stand:
`Entropy/Placement_Target` vs
`_Measured` (tracking, not fighting), and `Cards/Game` in phase 2 — it sat at
**5.6/8 flat across 50,000 episodes** in the run before the exploiter existed,
which is the plateau signature the exploiter is meant to break. It has still
never moved off ~5.3 — and note that the exploiter has been **off since
2026-08-11**, so the plateau persisting is not evidence the exploiter failed to
break it.

**`Entropy/Placement_Measured` changed meaning on 2026-08-11** and is not
comparable across that date — it now averages over steps that actually placed
a card, and reads roughly 0.37 lower than the same series before it. Watch it
against `Entropy/Placement_Measured_NoOp`: the no-op arm sits near 0.97 of max
and is what the old definition was mostly measuring.

**And watch `Entropy/Placement_ByCard_Min`** — the conditional collapse
detector. The aggregate provably cannot see a per-card collapse, because a
mixture of eight sharp, well-separated modes has high entropy even when every
component is a delta. That is not hypothetical: on 2026-08-11 the aggregate
read a healthy 0.462 while Cannon sat at **0.017 of max with 96.4% of its mass
on one cell**, and Fireball and Giant were pinned to that same cell. Any card
near 0 here is placing at a fixed point regardless of the board.

**...and `ByCard_Min` is itself the wrong statistic — use MODAL SHARE.**
Measured 2026-08-14 on `model_weights_dist_e3.pth` over 40 greedy episodes:

| card | plays | modal cell | modal share | H(place\|card) |
|---|---|---|---|---|
| Mini PEKKA | 230 | (14,15) | 19.0% | **0.086** |
| Cannon | 24 | (11,0) | **91.0%** | 0.098 |
| Fireball | 2 | (11,0) | 58.4% | 0.141 |
| Giant | 2 | (11,0) | 54.1% | 0.147 |

Mini PEKKA has the **lowest** entropy in the deck and is the **most-played**
card, so `ByCard_Min` flags the healthiest card and clears the three dead ones.
A good head is sharp but moves its mode with the board; a broken one returns one
cell regardless of it. **Count how often each card's argmax cell repeats across
states**, not how peaked the distribution is. Sixth instance of this project's
recurring failure — an aggregate that shares an assumption with what it checks.

**And add a fourth: the placement PHASE histogram.** Count `int(actionX) % 4`
over `replays/*.json` and compare against the 27.8/27.8/22.2/22.2 null that 18
columns imply. It is the only cheap detector for the checkerboard failure above,
which `Entropy/Placement_Measured` provably cannot see. Collapse runs of
identical decisions first — `annotate_replay_with_agent_info` stamps each
decision onto 10 consecutive ticks, so a naive count is 10× too large and any
χ² computed on it is meaningless.

**And check the side null before trusting any self-play number.** Run a policy
against a bit-exact copy of itself and confirm team 0 scores ~0.50 over a few
hundred episodes. It is cheap, it needs no training, and it is the only one of
these that catches a fault the ordinary metrics cannot see at all — the 2026-07-31
observation bug sat at 0.598 while every other diagnostic read healthy. Worth
re-running after any change to the observation, the board, or `stepSelfPlay`.

---

## Layout

**`python_ai/` became a PACKAGE on 2026-08-20**, one subpackage per
responsibility. Two mechanisms replace ~30 copies of one fact: importing
`python_ai` appends its own directory to `sys.path` (which is what makes
`import clash_royale_env` work — the `.pyd` is unpackaged and lives there), and
`python_ai.PACKAGE_DIR` / `REPO_ROOT` replace the `__file__`-relative
directories the harnesses used to build checkpoint and header paths from.

Entry-point scripts carry a four-line bootstrap putting the repo root on
`sys.path`, so **both invocations keep working**:

```bash
python_ai/venv/Scripts/python.exe python_ai/eval/prove_hog.py
python_ai/venv/Scripts/python.exe -m python_ai.eval.prove_hog
```

```
include/, src/       C++ engine. READ-ONLY by default.
tests/               C++ Catch2 tests (ClashRoyaleTests).
python_ai/           READ-ONLY by default — training runs here.
  __init__.py          PACKAGE_DIR / REPO_ROOT, and the one sys.path append
                       that makes the compiled engine importable everywhere.
  engine_constants.py  Board/observation/HP constants + card_name(), derived
                       ONCE from the bindings. The enforcement point for
                       CLAUDE.md's no-second-copies rule.
  shipping.py          The deployable configuration, named in one place.
  deck.py              THE trainee deck: CLASH_DECK (ids or names), validated
                       against the engine. A leaf every layer may import.

  models/
    net.py             MicroRoyaleNet (was model.py). ALL observation-layout
                       knowledge lives here; everything derived from the
                       bindings. Also owns LSTM_HIDDEN, as a class attribute.
    policy_io.py       Checkpoint -> ready net. Exists so a probe does not
                       import an experiment script (and through it both
                       trainers) just to read one .pth.
    perception_encoder.py  GameState -> the net's observation layout.

  envs/
    gym_wrapper.py     MicroRoyaleEnv + DEFAULT_DECK. Phase 1's env.
    selfplay_env.py    MicroRoyaleSelfPlayEnv + the PFSP constants. Phase 2's.
    scripted_opponents.py  Rusher/Defender/Cycler/Counter. A POLICY, lifted
                       out of an ENVIRONMENT's method.
    scenarios.py       Scenario injection: reshapes the START-STATE
                       distribution, never the reward.
    scenario_offense.py  Proposal A, default-OFF.

  opponents/
    deck_pool.py       The phase-1 opponent deck pool: loads meta_decks.json,
                       validates every card against the registry, and holds the
                       PFSP weighting. The enforcement point for "the opponent
                       is a REAL deck, and which one is decided by measurement".
    decks/meta_decks.json
                       The pool itself -- 16 RoyaleAPI ladder archetypes, named
                       cards, hand-editable, with the measured `prior_win_rate`
                       that seeds PFSP. EDIT THIS, not the code; then run
                       `python -m python_ai.opponents.deck_pool` to validate.
    teacher.py         UtilityTeacher: phase 1's opponent. Rules propose,
                       simulation ranks. Difficulty is lookahead, not elixir.
                       Since 2026-08-20 a candidate is a SEQUENCE of
                       placements, not one cell, so it can plan the escorted
                       push the deploy-time change made correct. The follow-up
                       GAP is searched (1/3/5 s) because the pair's price is
                       paid across it.

  advisors/
    tactics.py         The deterministic, engine-validated placement advisor.
    advisor_target.py  That advisor's score surface as a TRAINING TARGET for
                       the placement-coverage term. The only place that knows
                       which cards have rules.
    hybrid_policy.py   Inference-time composition of net + advisor + gate.

  rewards/
    weights.py         Every W_* and threshold, and which terms are
                       potential-based and which DELIBERATELY biasing. The
                       measurements behind them are in docs/DECISIONS.md.
    shaping.py         compute_shaping and the terms it composes. No torch.
    elixir_shaping.py  The potential-based solvency term.

  rl/                  THE PPO ALGORITHM, free of any Clash-specific policy
                       decision. Direction is trainers -> rl, never back.
    config.py          PPOConfig + EntropyConfig (frozen). PHASE1_ENTROPY and
                       PHASE2_ENTROPY keep the two pipelines' deliberately
                       DIFFERENT entropy settings.
    base_trainer.py    The loop, as a template method. Subclasses supply the
                       opponent, the bookkeeping and the periodic work — and
                       CANNOT change the rollout's arithmetic.
    ppo.py             PPOUpdater + UpdateStats: the ~250-line minibatch
                       update that was duplicated verbatim.
    gae.py             One GAE. The truncation-bootstrap form is a strict
                       generalization of the plain one, and it is tested.
    buffer.py          RolloutBuffer; add() refuses a partial row.
    entropy.py         EntropyController. The history of the normaliser bugs
                       it prevents is in docs/DECISIONS.md.
    curriculum.py      CURRICULUM_STAGES (module scope, so a test can import
                       it) + CurriculumManager.
    engine_stats.py    infos -> the stats dict, and why every default is the
                       value that contributes exactly zero.
    episode_metrics.py The rolling windows both pipelines report from.
    coverage.py        Placement-coverage sampling.
    checkpointing.py   The three checkpoint destinations, kept separate.
    replay.py          Demo-replay recording and annotation.

  trainers/
    train.py           Pipeline 1: the teacher opponent + the phase machine.
    train_selfplay.py  Pipeline 2: the league + scenario-aware bookkeeping +
                       the live strategy read-out.
    league.py          PFSP pool discovery, the fixed Elo roster, evaluation.
    strategy_metrics.py  ROI / Fwd / Cards-per-game. Read them in PAIRS.
    exploiter.py       League exploiter, self-contained PPO loop. OFF.
    bc_pretrain.py     Behaviour cloning + the demonstration .npz schema.
    distill_tactics.py Distilling the advisor into the placement head.
    expert_collect.py / expert_distill.py / expert_metrics.py /
    expert_iteration.py  Search -> labels -> student, and the CLI over them.

  search/
    config.py          SearchCfg. Imports nothing but `dataclasses`, so
                       shipping.py can read it cheaply.
    search.py          The lookahead itself: +0.4025 win rate at horizon 12.
    realtime_search.py Live-loop wrapper.

  eval/                Measurement harnesses. None is imported by the
                       training path.
    stats.py           Paired bootstrap CI + exact sign test, ONCE. Four
                       harnesses each had their own copy.
    match_outcome.py   TimeoutRules' verdict, read from the engine.
    prove_*.py         Engine-scored: the engine is the oracle.
    measure_deck_matchups.py
                       Per-opponent-deck win rate AND the per-decision
                       OPPORTUNITY each deck offers the three historically dead
                       cards (best Fireball catch, best Log corridor catch,
                       threat HP). Read the two together: opportunity is what
                       the BOARD offers, take-up is what the policy does with
                       it, and the 2026-09-03 finding is that the mirror is
                       16th of 16 on opportunity while being the only deck the
                       policy wins 100% of.
    prove_combos.py    Multi-card usage, the combos-on/off A/B, and the
                       lookahead sweep against a checkpoint. Reports USAGE and
                       WIN RATE separately, because a win-rate arm alone cannot
                       tell "the combo did not help" from "the combo never
                       happened".
    probe_*.py         Behavioural read-outs of a policy.
    *_ab.py            Paired A/B comparisons.

  tools/
    validate_pipeline.py  Pre-flight: PFSP routing, scenario contracts,
                       advisor targeting at scale, search cost ratio, spell
                       anneal, side null, C++ suite. The 20/20 gate.
    monitor_run.py     Health daemon for an unattended run. Read-only.
    setup_ab_arm.py    Builds a FULL training checkpoint for an experiment arm.
    make_replays.py    Replay generation for the viewer.

  tests/               The Python suite. conftest.py holds the two shared
                       fixtures, helpers.py the two shared builders.
tools/audit/         Standalone measurement instruments for the C++ engine,
                     built by tools/audit/build.ps1 (cl.exe directly against the
                     header-only engine -- deliberately NOT CMake targets, so
                     they cannot force a reconfigure of the solution the .pyd
                     and the Catch2 suite build from).
  bridge_audit.cpp     Per-tick crossing trajectories: stalls, jitter, latency.
  waypoint_probe.cpp   Analytic sweep for absorbing states in getNextWaypoint.
  soak.cpp             Randomized full matches; flags any unit stationary with
                       nothing in its own reach. Has a `trace` mode that
                       re-runs one seeded match and follows one entity.
  deck_audit.cpp       Per-card behaviour: identity, offence, defence, air,
                       deploy, and the sight/attack dead band.
  stall_repro.cpp      Minimal deterministic reproductions.
  spawn_speed_audit.cpp  Spawns every registered card, fires its death effects,
                       and reads Troop::getSpeed() off whatever arrives --
                       answering "is this unit on a real speed tier" by
                       MEASUREMENT. Reading the source instead gives an answer
                       7x too large; see the 2026-08-26 part 2 section.
  collision_bench.cpp  A/B for the 2026-08-26 hot-path rework. Holds BOTH the
                       old and new resolveCollisions/resolvePositionAgainstBuildings
                       in one binary, so the comparison cannot be confounded by
                       the gameplay fixes that shipped alongside; also sweeps
                       8,572 cells proving the two answer identically.
  verify_pyd.py        Post-rebuild gate: proves python_ai/clash_royale_env.pyd
                       actually carries the current engine, which the C++ suite
                       cannot tell you. Run it before any training run.
                       Uses step_self_play, never step -- step runs the
                       HeuristicOpponent, which defends, and a Giant stopped by
                       a defender says nothing about navigation.
.claude/CLAUDE.md    This file: the knowledge base. Lives in .claude/ (Claude
                     Code loads it from there too) so the public root stays
                     clean -- moved 2026-09-23 along with the three below.
docs/TODO.md         The single, verified list of pending work.
docs/DECISIONS.md    The narrative: every measured result and reversal.
docs/runbooks/       FINAL_RUN_RUNBOOK.md, ELIXIR_PHASE_RUNBOOK.md.
docs/design/         Design specs and implementation plans. Write NEW ones
                     here (docs/design/specs, docs/design/plans), not in the
                     superpowers plugin's default docs/superpowers/.
LICENSE              MIT. The vendored CRBAB keeps its own MIT notice in
                     perception/clashroyalebuildabot/LICENSE.md.
.github/workflows/cpp-tests.yml
                     CI: builds and runs the Catch2 suite with MSVC on
                     windows-latest, on pushes/PRs touching include/, src/,
                     tests/ or CMakeLists.txt. Drives the README's live badge.
                     It does NOT build the .pyd or run the Python suites.
docs/assets/         README GIF and the 1280x640 social preview image.
python_ai/archive_*/ Local checkpoint archives, gitignored since 2026-09-23.
perception/          Screen -> placement events -> simulator as estimator.
                     Self-contained: own venv, own requirements.txt.
web/viewer.html      Replay viewer.
```

`perception/` is the one place to edit freely. It has its own docs:

- **`perception/README.md`** — per-stage status with measured numbers, the
  recording spec, open questions.
- **`perception/UPSTREAM_REQUESTS.md`** — every simulator change currently
  being asked for, with evidence and blast radius.

Run its tests with:

```bash
perception/.venv/Scripts/python.exe -m pytest perception/tests -q
```

354 tests (353 pass, 1 skipped), none requiring an emulator — they run against
frozen replay fixtures, a synthetic camera, or video generated at test time.

The Python suite is **369 collected (367 pass, 2 skipped)** as of 2026-08-21,
re-run against the rebuilt `.pyd` after the simulator audit's engine changes. It
was **349 collected (347-348 pass, 1-2 skipped)** since the
multi-card teacher landed on 2026-08-20, and was **312** after the 2026-08-20
restructuring, up from 90; the skip count varies run to run because two cases
depend on the unseeded opening-hand shuffle. Run it with:

```bash
python_ai/venv/Scripts/python.exe -m pytest python_ai/tests -q
```

---

## Recording real matches (for `perception/`)

Full spec in `perception/README.md`. The parts that silently ruin a batch:

- **Constant frame rate.** `capture/video.py` measures it and refuses a
  variable-rate file rather than producing timestamps that drift during
  fights, which is exactly when placements happen.
- **Same resolution for every file.** A change invalidates every pixel
  constant in the calibration profile.
- **Start recording before pressing Battle**, and keep the opening seconds:
  calibration needs frames where the board is still empty, because the tower
  detector keys on grey stone and a deployed Cannon is grey stone.

Current batch: 8 matches, 1920×1080 desktop capture (the emulator window
occupies x 686-1236, y 40-1012), CFR at 30.0001 fps, jitter 0.0000.

The deck used in them: Valkyrie(10), Archers(1), Minions(41), Cannon(25),
Fireball(7), Giant(2), Musketeer(6), Mini P.E.K.K.A(5).

---

## Plugins

`superpowers@claude-plugins-official` v6.2.0 is enabled at project scope
(`.claude/settings.json`). The marketplace it comes from is registered in
**user** settings, so a fresh clone of this repo needs:

```bash
claude plugin marketplace add anthropics/claude-plugins-official
```

Note `.claude/settings.local.json` holds machine-specific permission entries
and should not be committed.
