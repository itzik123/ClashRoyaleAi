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

Measured 2026-08-24 after the backlog sweep: **650 test cases, 6,423
assertions**. 649 pass and **exactly one fails "as expected"** --
`test_navigation_wedge.cpp`'s `[!shouldfail]` case, which pins the open
collision-wedge defect. The runner exits 0 in that state; a non-zero exit or a
second failure is a real regression. (It read 619 cases / 5,907 assertions on
2026-08-21; the counts move as tests are added, so treat the **shape** -- one
expected failure, exit 0 -- as the invariant, not the number.)

**Adding a test FILE needs the build run TWICE.** CMake globs `tests/**` with
`CONFIGURE_DEPENDS`, so the first MSBuild re-globs and regenerates the vcxproj --
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

**Troop movement was 4-5× too fast until 2026-08-07.** `CardStats::speed` is
tiles per *tick*, so the registry's Giant `0.3f` meant **3.0 tiles/s** against
a real-game Slow of ~0.75 — a Giant crossed bridge-to-tower in ~3.5 s.
`MOVEMENT_SPEED_SCALE = 0.2f` in `CardStats.h` now converts the registry's
tier literals into real-game tiles/tick. It is applied at **three** sites:
`CardRegistry.h:127` plus both `include/core/SpiritEmpressForms.h` assignments
(lines 32 and 50 — the file is in `core/`, not `entities/`), which set
`stats.speed` directly and so bypass `CardStats::troop()`.

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

**Still wrong, unmeasured, and in the same direction:** `Projectile.h:88` has
its own untouched `speed`, never recalibrated alongside the movement fix.

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

Order of discovery, since the count in this file has now moved twice: the four
above, then `extractObservationForTeam`'s channel 8, then the viewer.

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
`train.py:252` adds `actionCardId` / `actionX` / `actionY` per tick, which
`GameLogger` itself does not write. Two catches: it labels **only the learner's
own plays** (the heuristic opponent's are logged nowhere), and **the training
run rewrites that directory continuously** — observed dropping from 8 files to
1 within minutes. Frozen fixtures live in `perception/tests/assets/`.

`DEFAULT_DECK = [15, 6, 25, 40, 24, 72, 33, 7]` (`python_ai/envs/gym_wrapper.py`)
— **the classic 2.6 Hog Cycle**, since 2026-08-16: Hog Rider, Musketeer,
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
train.py            phase 1  "mirror"           vs the UtilityTeacher
       |  win rate >= PHASE2_ENTRY_WIN_RATE (0.60)   <-- gates THIS step only
       |  AND curriculum stage >= PHASE2_MIN_CURRICULUM_STAGE (4)
       v
                    phase 1  "random_opponent"  vs randomised decks
       |  RANDOM_OPPONENT_EPISODE_BUDGET (5,000) episodes IN THIS PHASE
       v
train_selfplay.py   phase 2  PFSP league        vs frozen snapshots +
                                               4 scripted bots + exploiters
```

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
sampled (`train.py:1376`), so `4/5 -> 4/0` is a new deck, not a regression.

### Network (`model.py`, 1.88 M params)

`MicroRoyaleNet` — the LSTM is 96% of the parameters:

| | params | |
|---|---|---|
| `cnn_trunk` | 7,680 | 21→16→32 conv, 2× MaxPool(ceil) → `32×9×5` |
| `scalar_mlp` | 48,320 | 754 scalars → 64 |
| `lstm` | **1,804,288** | `LSTMCell(1504, 256)`, stepped manually |
| `card_head` | 1,285 | 256 → 5 (4 hand slots + no-op) |
| `place_ctx` + `place_up` | 14,593 | ctx→32ch, broadcast-add, 2× (upsample+conv) → 612 |
| `value_head` | 257 | critic |
| `aux_elixir_head` | 257 | opponent-elixir estimator |

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

Six stages, gated on **raw** win rate ≥ 0.80 over 100 episodes.

**The rungs are the TEACHER'S LOOKAHEAD, not an elixir handicap** (corrected
2026-08-25; this section still described the retired multiplier ladder — the
pivot itself is in `DECISIONS.md`, "2026-08-19: the curriculum pivot"). Read
`TEACHER_STAGES` in `opponents/teacher.py` for the live values:

| stage | horizon | epsilon | k_cells | max_combos | reactive |
|---|---|---|---|---|---|
| 0 | 0 t (rules only) | 0.30 | 1 | 0 | no |
| 1 | 10 t (1 s) | 0.15 | 1 | 0 | no |
| 2 | 30 t (3 s) | 0.10 | 2 | 2 | no |
| 3 | 50 t (5 s) | 0.05 | 2 | 3 | no |
| 4 | 70 t (7 s) | 0.02 | 3 | 4 | no |
| 5 | 100 t (10 s) | 0.00 | 3 | 4 | **yes** |

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

## How the learning mechanism got here — moved to `DECISIONS.md`

The chronological narrative — every measured result, every reversal, and the
reasoning behind the constants above — was extracted to **`DECISIONS.md`** on
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

---

## NEXT UP: Stage 2 — see `TODO.md` item 0

The full brief — goal, the two modules to write, the three invariants that can
go silently wrong, and the limits already measured — is **`TODO.md` item 0**,
because pending work belongs in the pending-work list and this file had a second
copy of it. Design: `docs/superpowers/specs/2026-08-24-live-teacher-play-design.md`.
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
  is *no better* than the mean (ratio 0.96–1.05 across seeds, corr ≈ 0.27). So
  0.77 is a real capability, not a trivial correlation.
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

> **`TODO.md` is the authoritative task list.** It was consolidated on
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

   - **`clone()` already exists** (`Entity.h:113`), overridden in four types as
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

   The one genuine hazard is **`Projectile::target`** (`Projectile.h:16`), the
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
   collapses — `CH_COUNT` mitigates but does not fix it. No card-cycle tracking
   (which of the opponent's 8 cards are available) — a core human skill.

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

**Watch these three during any run:** `Aux/OppElixir_MAE` (below ~1.3 means the
recurrent state genuinely counts), `Entropy/Placement_Target` vs
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
    weights.py         Every W_* and threshold, with the measurement that
                       justifies it, and an explicit list of which terms are
                       policy-invariant and which are DELIBERATELY biasing.
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
    entropy.py         EntropyController, with the seven-instance history of
                       normalizer bugs it exists to prevent.
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
  verify_pyd.py        Post-rebuild gate: proves python_ai/clash_royale_env.pyd
                       actually carries the current engine, which the C++ suite
                       cannot tell you. Run it before any training run.
                       Uses step_self_play, never step -- step runs the
                       HeuristicOpponent, which defends, and a Giant stopped by
                       a defender says nothing about navigation.
CLAUDE.md            This file: the knowledge base.
TODO.md              The single, verified list of pending work.
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
