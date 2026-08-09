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
on this machine is 3.14, and it fails with `ImportError: DLL load failed`,
which reads like a corrupt build and is only a version mismatch. Use one of:

```bash
python_ai/venv/Scripts/python.exe        # training env — do not modify
perception/.venv/Scripts/python.exe      # perception env
py -3.11
```

**`cmake`, `cl` and `msbuild` are not on PATH.** Rebuild the engine with:

```bash
"C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe" build_python\clash_royale_env.vcxproj /p:Configuration=Release /p:Platform=x64 /m
```

The MSBuild binary lives under VS "18" while the actual toolset comes from the
VS2022 install. **The post-build copy into `python_ai/` fails (MSB3073) if any
Python process has the `.pyd` loaded** — Windows won't overwrite a mapped DLL.
That looks exactly like "the compile is broken" and almost never is. Check
`ps -W | grep -i python` first.

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
`CardRegistry.h:125` plus both `SpiritEmpressForms.h` assignments, which set
`stats.speed` directly and so bypass `CardStats::troop()`.

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

**Still wrong, unmeasured, and in the same direction:** `Projectile.h:88` has
its own untouched `speed`, and the engine has **no deploy time** at all while
the real game freezes a troop ~1 s after it lands.

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

**Board geometry.** River `[15.5, 17.5)`, bridges at `(4, 16.5)` and
`(14, 16.5)` — re-centred on 2026-07-29 to fix an asymmetry where team 1 had
one row less placeable ground than team 0. Towers corrected 2026-07-30 per
`perception/UPSTREAM_REQUESTS.md` items 1-2: left Princess `x` 3.0 → 4.0 (now
flush with its own bridge, like the right side already was), Kings `x` 8.5 →
9.0 (the board's measured true centre). Combined held-out calibration error
dropped max 0.63 → 0.31 tiles.

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

**The King Tower never sleeps.** `Tower.h` gives it no activation condition, so
it fires from tick 0 while the real King is dormant until activated. Any
comparison against real footage must exclude the Kings.

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

`DEFAULT_DECK = [10, 1, 41, 25, 7, 2, 6, 5]` (`python_ai/gym_wrapper.py`) —
Valkyrie, Archers, Minions, Cannon, Fireball, Giant, Musketeer, Mini PEKKA.
Costs 3-5, avg 3.75, spread 2. Win condition Giant (5); only 3 of 8 cards hit
air (Archers, Minions, Musketeer); one spell (Fireball).

**This deck is chosen to match the recordings**, deliberately overriding the
cost-curve argument — see the block comment above the literal, and "Open
problems" for the risk it re-introduces.

---

## The training mechanism

Two sequential pipelines. `train.py` hands off by `subprocess.Popen`-ing
`train_selfplay.py` (with `-u`, or the live diagnostics buffer and the log stays
0 bytes) and exiting.

```
train.py            phase 1  "mirror"           vs the C++ HeuristicOpponent
       |  win rate >= PHASE2_ENTRY_WIN_RATE (0.60)   <-- gates THIS step only
       v
                    phase 1  "random_opponent"  vs randomised decks
       |  episodes_completed >= PHASE2_TOTAL_EPISODE_CAP (40,000)
       v
train_selfplay.py   phase 2  PFSP league        vs frozen snapshots +
                                               4 scripted bots + exploiters
```

**`PHASE2_ENTRY_WIN_RATE` does not gate the pipeline handoff**, despite its
name. It gates `mirror` → `random_opponent` (`train.py:1299`). The handoff to
pipeline 2 is a plain episode count (`train.py:1035`) with no win-rate
condition at all — phase 2 has no natural stopping point, so the cap is what
ends pipeline 1. An earlier version of this diagram put the 0.60 gate on the
handoff arrow and cost a live run two wrong predictions about when it would
transition.

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

Six stages, opponent elixir multiplier `1.0 → 1.5` in 0.1 steps, gated on
**raw** win rate ≥ 0.80 over 100 episodes.

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

- **Historical snapshots**, saved every 5,000 episodes. Phase-2 snapshots
  younger than `MIN_OPPONENT_AGE_EPISODES = 15000` are excluded — otherwise the
  pool fills with coin-flip mirrors of the current trainee.
- **4 scripted bots** — Rusher / Defender / Cycler / Counter, permanent members.
  Defender+Counter get `DEFENSIVE_SCRIPTED_MIN_WEIGHT = 0.8`, overriding PFSP,
  because PFSP's own criterion works *against* seeing them: mastering them
  drives their weight to the floor. First tried at 0.20 and confirmed too low —
  in a ~98-member pool that is ~7.7% combined share, statistically invisible.
- **Exploiters** (`exploiter.py`) — see below.
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
`Elixir@Play`, `AvgTicks`, `ScenDef`.

### Exploiter (`exploiter.py`)

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

## How the learning mechanism got here

Roughly chronological. Every entry is here because it was *measured*, and
several reversed an earlier conclusion.

**Action-space correctness (the largest single win).** Before affordability
masking, `GameManager::playCard` returned `false` silently on insufficient
elixir — no exception, no penalty, no signal. Measured over 243 real decision
steps: **74.9% of steps attempted an unaffordable card and only 11.5% actually
placed one**, so ~87% of every rollout's stored actions did not affect the world
and their advantages were pure noise entering the gradient. Structural, not
transient: 0.035 elixir/tick × 10 ticks = 0.35 elixir per decision against
3–5-cost cards, so only 27.2% of steps had *any* legal play. Rejected-action
rate went 86.8% → **0.0%**.

**Gaussian → categorical placement.** The old head sampled from a `Normal` with
a learned `placement_log_std`. A Gaussian's entropy is `log σ + const`, so the
entropy bonus's gradient w.r.t. `log_std` is a **constant 1**, data-independent —
a fixed upward push that the noisy policy gradient loses to. Measured on the
checkpoints: `log_std` went −2.0 → −1.86 over 68,515 episodes, i.e. σ *grew*,
reaching **2.66 tiles** of noise in x. The two bridges are 10 tiles apart, so
the agent's own noise was half the distance it needed to resolve. Categorical
over 612 cells fixed it at the root: entropy bounded by `log(612)`, falls
naturally as the policy sharpens, exact tile, and maskable.

**Dead ability heads removed** (see Network above).

**Spell y-range.** `target_y` was capped at `getOwnHalfMaxY()` for every card,
so Fireball could not cross the river — 1/8 of the deck unusable for its purpose
at any amount of training. Fixed by the `get_card_info` binding plus a
card-conditional mask, not by widening the space unconditionally.

**Deck choice is a learnability constraint, not flavour.** With the
5-cost-heavy Giant deck the agent **never played Giant once** across four full
runs; only 5–7 of 8 cards were ever used. Cause: at 0.35 elixir/decision a
5-cost card is legal only after ~14 consecutive non-spending steps, so its slot
is masked out nearly every time it is checked and never accumulates gradient.
It was replaced by a 3–4 cost / avg 3.50 / spread 1 Hog cycle where no card is
systematically starved — the fix is the cost curve, **not** a bigger entropy
bonus (that was tried and measured to fail).

**...and then deliberately reverted, 2026-07-30.** `DEFAULT_DECK` is the Giant
deck again, because it is the deck in the 8 real recordings and matching them is
worth more than the cost curve — human demonstrations are the highest-value
unblocked item and unusable against a different deck. The risk above is
knowingly re-accepted. Two things differ from when it was measured: the
affordability mask makes "cannot afford Giant" observable rather than a silent
`playCard` failure, and card entropy is now adaptively held near 0.35 of max.
**Open question, not a settled one** — watch whether Giant is ever played.

**Adaptive per-head entropy** (`coef *= exp(rate·(target − measured))`),
replacing hand-tuned coefficients. Motivated by measurement at episode 20,245 of
the run it replaced: placement entropy at 26% of max, **card entropy at 0.4% of
max — effectively a constant**. A shared scalar coefficient could not fix both;
scaling it to rescue the card head crushed placement (top-5 cell share 36% →
70%, left-lane usage 33% → 13%).

`ENTROPY_ADAPT_RATE = 0.5` was lowered to 0.15 and then **reverted**: the lower
gain did smooth the controller (mean |entropy − target| fell, placement stopped
free-falling) but the resulting policy measured *worse* — 81.7% [74–88] vs
96.7% [92–99] against scripted opponents at stage 3, CIs not overlapping. The
current reading is that the large swings act as periodic exploration re-boosts.
Caveat recorded in the code: **one run per configuration**, so this ordering may
not survive replication.

`ENTROPY_COEF_FLOOR` was raised 0.002 → 0.01 because at 0.002 the floor was an
off switch rather than a floor and the head collapsed until the controller
noticed.

**Explained variance, not raw critic MSE.** I called the critic broken three
times off a flat MSE around 0.11. Raw MSE is bounded below by irreducible noise
in the returns and is uninterpretable alone. Measured explained variance at the
time: **+0.64 — healthy.** `Loss/Critic_Explained_Variance` exists because of
this. Value clipping is scaled to the batch's return std (`VF_CLIP_STD_FRAC`)
after measuring that at episode 14,666, 46.1% of samples needed the critic to
move more than the old fixed 0.2.

**Truncated BPTT with stored hidden states**, `bptt_chunk = 25`,
`ppo_epochs = 4`, `num_minibatches = 8`. Replaced treating the whole 500-step
rollout as one sequence with `num_minibatches=1, ppo_epochs=2` — exactly 2
optimizer steps per 4,000 transitions, 79% of the time spent unrolling one
500-long chain. Throughput went **228 → 4,224 gradient steps/hour at lower
wall-clock per update** (27.3 s vs 31.6 s).

**Random opponent → `HeuristicOpponent`** (C++). The built-in random
`opponentTurn()` never punished a bad trade, read a push, or played around a
threat. This was the change with the clearest effect on training quality — and
the uncomfortable lesson was that ~100 lines of heuristics beat a
47,000-episode policy **71%**.

**Timeouts stopped being free.** `TimeoutRules` (new header, MatchRules/
GameManager/Board/Tower untouched) decides a timed-out match on surviving tower
count, then on lowest weakest-tower HP, and only an exact tie is a draw. Before
this every timeout scored 0.0, teaching the agent that running the clock out was
neutral.

**Stage checkpointing.** Added after losing the ability to answer a question:
a narrow policy that had just cleared a stage's 80% gate could not be re-probed
because the checkpoints were already gone.

**A hypothesis I disproved.** I believed the policy under-weighted Fireball
because of a learning failure. Forcing Fireball usage dropped win rate
**97% → 23%** (CIs [83–99] vs [12–41]). The low weighting was a *correct
valuation*. This is the reason the current default is to measure before
"fixing" an apparent behavioural gap.

**2026-07-29, the big representational change.** Observation 6253 → 13606
(21 channels + 9 appended scalars: elapsed time, both sides' cumulative elixir
spend, 6 tower HPs); dense placement head → fully-convolutional; auxiliary
opponent-elixir head; annealed placement entropy target; exploiter league; BC
pipeline. Rationale for the observation work: it had **no temporal component at
all**, so tick 100 and tick 3500 with the same board were the same input — which
made the state non-Markovian for two decisions that now exist (`TimeoutRules`
made clock management real) and left part of the critic's residual variance
*structurally* unlearnable rather than undertrained. Air/ground had existed in
the engine since flying cards were added and was visible nowhere.

Same day: a `CardRegistry` audit found **Musketeer, Wizard, Spear Goblins and
Ice Wizard could not shoot air** (plus their Hero variants, plus Evolution Dart
Goblin regressing from its own base card). Musketeer is in `DEFAULT_DECK`. The
new `CH_ANTIAIR` channel is what surfaced it. Also that day: the river was
re-centred, which had been giving team 0 an extra placement row.

**2026-08-09, the placement head had a fixed favourite cell and it was not a
reward problem.** The "Cannon pathology" — 27.9% of Cannons parked behind the
agent's own King — was attributed to the building-vs-tower reward price and
partly fixed there on 2026-08-06. That diagnosis was wrong, or at least badly
incomplete. Re-measured on the ep~129k checkpoint over 803 sampled placements:
the concentration is **card-independent**. Giant 44.4%, Valkyrie 38.9%,
Musketeer 35.1%, Cannon 27.1% — the Cannon was not even the worst card, and no
asymmetry in how *deployed buildings* are priced can make a **Giant** walk
behind its own King half the time.

The cause was `place_up`. `placement_given_card` adds the card/state context as
a **spatially uniform** vector, and a `ConvTranspose2d(k=2, s=2)` applies a
different weight to each of the 4 positions in its output block — so a uniform
input does *not* give a uniform output. Two stacked layers imprint a fixed
period-4 `(x mod 4, y mod 4)` bias on the logit map that is identical for every
card and every state; the card can only shift the whole map by a constant, never
change which cell inside the period wins. Measured three ways, all agreeing:

- **Weight space:** on a spatially constant input the old head's logit surface
  is **100% explained by phase alone**, at every input magnitude tested.
- **Behaviour:** **73.0%** of that agent's placements landed on `x ≡ 3 (mod 4)`
  against a 22.2% null — χ² = 94.8 on 3 df — and it used only **91 of 288**
  legal cells.
- **A/B against the current agent** under identical conditions: 28.9% of all
  placements on (11,2)/(11,3) versus **0.9%**, 91 cells versus **208**.

Fixed by nearest-neighbour upsample + stride-1 conv (Odena, Dumoulin & Olah,
*Deconvolution and Checkerboard Artifacts*). Verified: a spatially constant
input now produces an **exactly** flat interior (range 0.000000), and the head
is **1.8× faster** (136 ms vs 246 ms per fwd+bwd at batch 256) with slightly
fewer parameters — `ConvTranspose2d` is poorly optimized on CPU and the 32→16→8
channel taper cut real work. A FLOP count predicted a 2.6× *slowdown* and was
simply wrong about wall clock. **This invalidates every checkpoint**: `place_up`
no longer shape-matches, so `load_state_dict_flexible` will warm-start
everything else and leave the placement head fresh.

Two lessons worth carrying:

- **The concentration was visible in the replays the whole time and nobody
  looked at the marginal.** `Entropy/Placement_Measured` was tracking its target
  throughout — a policy can put 30% on one cell and still hit a moderate entropy
  target by spreading the rest, so the entropy metric **cannot** detect this.
  The cheap detector is a histogram of `x mod 4` over `replays/*.json`.
- **The thing that didn't break localized it again.** Same shape as the
  team-1 observation bug: it was the cards that had *no* reason to be affected —
  Giant, Valkyrie — that ruled out the reward explanation and pointed at the
  head.

**2026-07-31, the self-play opponent had been playing blind.** Chased from an
odd number rather than a hypothesis, which is why it is worth recording as a
method. Exploiter burst #1 came back 0.585 with a flat five-slice trend — but
its *first* slice was already 0.59, and a re-seeded exploiter's first slice is
played by a bit-exact copy of its opponent, so it should sit at the null.
Either it learned a hole in ~490 gradient steps and then stopped, or the null
was never 0.50. Running the frozen agent against a copy of itself gave
**0.598** as team 0 (n=400, z = +3.90) — a policy beating *itself* 60/40 purely
by side assignment. Root cause was a one-row displacement in team 1's
observation (see Board geometry); after the fix the same test gives 0.520.

Three lessons, all of which nearly hid it:

- **The bug was invisible in every training metric.** The trainee is always
  team 0, so win rate, reward, entropy, aux MAE and explained variance all
  looked healthy while every opponent saw the board wrong.
- **The dissociation was the evidence, not the aggregate.** In the earlier
  entropy-collapse bug the card head was fine and only the mis-scaled placement
  head blew up; here the integer-`y` Princesses mirrored correctly and only the
  fractional-`y` King did not. In both cases the thing that *didn't* break
  localized the cause faster than the thing that did.
- **I over-read the Elo trend twice before this landed** — first predicting
  saturation noise, then calling a single 2.1σ excursion a real gain that
  "held". Six evals of `heuristic@1.50` test as flat (χ² = 7.4 on 5 df,
  p ≈ 0.19). At n=50 per anchor that metric cannot resolve much; treat any
  single-eval move as noise until it repeats.

---

## Measured baselines — use these, don't re-derive them

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
| **current, phase 1** | **2,873** | **92** |
| **current, phase 2** | **4,185** | — |

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

## Open problems and what would actually move the needle

**Ranked by expected value, from the architectural review.**

1. **Human-replay imitation is now unblocked.** The recordings exist; the
   extraction step (`perception/` → the `bc_pretrain` `.npz` schema) is the
   blocker. Self-play discovers strategies but not *the distribution humans
   play*; AlphaStar's supervised stage was load-bearing, not optional.

2. **Decision-time lookahead / search — the biggest unexploited asset.** A fast
   deterministic simulator is owned and unused. Roll the top-K candidate actions
   ~2 s forward and pick by value; that is a strict improvement operator, and
   distilling it back is expert iteration. **Blocked:** `Board` holds
   `shared_ptr<Entity>`, so a `GameManager` copy is *shallow*. Needs a virtual
   `Entity::clone()` across the whole hierarchy plus effects — invasive
   simulation-core surgery, needs explicit sign-off.

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
never moved off ~5.3.

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

```
include/, src/       C++ engine. READ-ONLY by default.
python_ai/           READ-ONLY by default — training runs here.
  model.py             MicroRoyaleNet. ALL observation-layout knowledge lives
                       here; everything is derived from the bindings.
  gym_wrapper.py       MicroRoyaleEnv + DEFAULT_DECK.
  train.py             Pipeline 1 + the shaping/curriculum constants that
                       train_selfplay.py imports.
  train_selfplay.py    Pipeline 2: PFSP league, scripted bots, scenarios, eval.
  exploiter.py         League exploiter, self-contained PPO loop.
  bc_pretrain.py       Behaviour cloning + the demonstration .npz schema.
  curriculum.py        LEGACY, imported by nothing. Random-action probe from
                       before the real stage system existed. Ignore it.
  archive_*/           Checkpoints invalidated by engine/architecture changes.
tests/               C++ Catch2 tests (ClashRoyaleTests).
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

58 tests, none requiring an emulator — they run against frozen replay
fixtures, a synthetic camera, or video generated at test time.

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
