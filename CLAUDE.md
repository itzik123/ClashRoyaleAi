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

`DEFAULT_DECK = [15, 6, 25, 40, 24, 72, 33, 7]` (`python_ai/gym_wrapper.py`)
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

**2026-08-11, the entropy controller was regulating the noise of non-actions.**
`train_selfplay.py` averaged placement entropy over `mb_decision`, which is
`card_mask.sum() > 1` — **"a card was AFFORDABLE", not "a card was PLAYED"**.
The placement head is sampled on every such step, including the ones where the
policy chose the no-op and the sampled cell never reaches the board. Measured
over 549 decision steps at ep~62,200:

| | of max |
|---|---|
| reported `Entropy/Placement_Measured` | 0.462 |
| ...on no-op steps (cell never used) | **0.850** |
| ...on real placements | **0.090** |

Target was 0.25. So the controller saw a surplus, concluded there was too much
exploration, and pinned the coefficient to its 0.01 floor — **57% of updates in
the hour before the fix** — while the policy that actually places cards sat ~3×
*below* target. Conditioned on a card, Cannon was at 0.017 of max, Fireball
0.048 and Giant 0.074, all three pinned to the same cell `(11,0)`. Fixed by
masking the entropy term to real placements; verified the bonus's gradient is
now exactly zero on no-op steps. The coefficient left the floor within one
update and quadrupled in 30 minutes. **Gameplay-affecting** — it changes the
loss.

Three things worth carrying:

- **A pre-fix control checkpoint measured Cannon at 0.012 / 94% on `(11,0)`**,
  identical. That ruled out the placement-mask fix from two days earlier as the
  cause and localized it to the metric. Running the *older* checkpoint under
  the *current* code is what made that a one-command test.
- **The cost was real and the obvious fix was inert.** Over 40 episodes the
  Cannon took 369 elixir and returned 20 (ROI 0.05) against Musketeer's 1.27;
  100% of 123 Cannons were placed at y=0, 82.9% expired by decay rather than
  dying, and **93.5% never had an enemy inside their 5.5-tile range for a
  single tick**. The proposal on the table was to penalise back-row structures
  and structures that "fire ≤3 times" — but P(shots ≤ 3) was **100.0%**, so
  both predicates have zero variance, carry no spatial gradient, and reduce to
  a constant added to the card's cost. The only thing they could move is
  P(play Cannon), i.e. the guaranteed-zero trap this file already documents
  twice. **A penalty cannot move a distribution with no mass to move.**
- **Same blind spot as the checkerboard bias and the team-1 observation bug**,
  now three for three: an aggregate that cannot see a conditional collapse.
  `Entropy/Placement_ByCard_Min` is the detector, and it is cheap — it is
  computed from tensors the update already has.

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

**2026-08-14, three of eight cards were dead and it was a gradient COVERAGE hole,
not the reward.** Full write-up and every number: `PLACEMENT_COLLAPSE.md`.

Cannon, Fireball and Giant were played on ~2% of plays and their placement head
returned the *same* cell — (11,0), our own back row — in 54–91% of states. The
reward was the obvious suspect and it is not the cause. The cause is that both
the actor loss and the placement entropy bonus flow through
`placement_given_card` for the **chosen** card only, so a card the policy has
stopped playing receives **exactly zero** placement gradient from either term,
forever. `card_id_embed` is per-card, so this is mechanical, not statistical —
`test_python_ai.py::test_unchosen_card_gets_no_gradient` asserts it is `== 0.0`.

That is a self-sustaining deadlock: frozen map → the card really is worthless →
card head suppresses it → no gradient → still frozen. **More training cannot
escape it**, which is why it survived every previous fix.

It is not a valuation, and one measurement settles that. Injecting a Cannon at
each candidate cell and running the engine a full 300-tick lifetime, over 449
threatened states, scored by tower HP preserved: the policy's own cell **121 HP**,
a **random legal cell 396 HP**, oracle 1389. Paired, random − policy = **+274 HP,
95% CI [+221, +328]**. A policy cannot be correctly valuing a card it places
*significantly worse than chance*. Same for Fireball against
`get_elixir_value_killed_by` over 1,059 states: policy **0.022** elixir and it
caught anything at all **0.1–0.5%** of the time, against **1.103 for a random
legal cell** — 50× worse than random.

Dating it: the 08-09 phase-1 net placed Cannon at (3,15), modal share 9.5%,
H=0.368 — healthy and state-dependent. The collapse appears in the 08-11 net,
bracketing `e16cdd7` ("Measure placement entropy on real placements, not on
no-ops"). **That commit was right about its own defect and had an unmeasured
side effect**: the no-op steps it stopped rewarding were the only thing holding
open the maps of cards that are never played. The signature is specific — in the
pre-fix net Cannon and Giant had the two *highest* per-card placement entropies
(0.368, 0.409); after it they have the lowest. The rank order inverted for
exactly the unplayed cards. The same-day placement-legality mask (`24a2c78`) was
checked and ruled out: (3,15) is still legal, so the head abandoned an available
cell rather than being masked off one.

Fixed by a **coverage term** — one uniformly-sampled *affordable* slot per step
contributes placement entropy at a fixed `PLACEMENT_COVERAGE_COEF = 0.02`. It is
a regularizer, not part of the PPO objective (it never touches `new_logprobs`,
so the ratio is untouched; pinned by test). The coefficient is deliberately
**not** tied to the adaptive `ent_coef_place`, which falls exactly when real
placements are sharp — the condition under which an unplayed card is freezing.
`Entropy/Placement_Coverage` is the new freeze detector. No new parameters, so
**no checkpoint is invalidated**.

Two further results worth carrying:

- **Leading a spell target is HARMFUL here.** Fireball has `spellDelayTicks=10`,
  so aiming where the target will be looks obviously right. Measured paired over
  1,059 states against the engine's own value-killed: lead=0 captures **75.5%**
  of achievable value, lead=10 **52.4%**. The blast radius is 2.5 tiles while 1 s
  of movement is 0.6–1.6 tiles, so a moving target stays inside the blast anyway
  — while a target standing still (engaged, the common case) is led straight off
  the edge of it. `tactics.py` defaults `lead_ticks=0`; do not "fix" it.
- **"Defensive apathy" is bankruptcy, not apathy or hoarding.** P(play) looks
  flat against threat (9.4% → 13.7%), but **60.8% of decisions during a big push
  are below 3 elixir** — the cheapest card in the deck — so P(play) there is
  0.0% by arithmetic. Conditioned on affordability the response is real:
  27.8% → 34.9%. The agent spends ~105 elixir per episode against ~98 of income
  and sits under 3 elixir **65.3%** of the time.

**RE-RUN, and the coverage term does NOT break the lock — it moves it.** Three
arms in parallel, byte-identical code, full-checkpoint seeds so the entropy
target anneals correctly, ~80 updates, compared at matched episode ~64,800 and
scored by the engine on 835/1,573 paired states. Cannon: control 116.4 HP vs
treatment 182.0 HP (bootstrap CI excludes 0 but **sign test p = 0.158**, so no);
Fireball: treatment kills **0.000** elixir, identical to control. Treatment's
top-1 probability is **0.051** (Cannon) and **0.006** (Fireball) against a
uniform 1/612 = 0.0016 — the distribution went nearly FLAT, and **the argmax of
a flat map is an arbitrary constant**, so a greedy policy still plays one fixed
cell. The frozen cell relocated (11,0) → (6,0).

**Entropy is a MARGINAL objective — it says "be spread out", not "depend on the
board".** A card with no other gradient has nothing telling it which cell is
right in which state. Closing a coverage hole needs a TARGET, not noise. (The
control also partially unfroze on its own — 88.7% → 55.4% modal share — so part
of the original collapse was the mis-set entropy target below, not the coverage
hole alone.)

**What does work is the deterministic advisor** (`tactics.py`), same protocol,
same states: Cannon **564.1 HP** preserved vs 353.5 for a random legal cell and
**12.1 for the trained policy**; Fireball **2.405 elixir** killed vs 0.276
random and **0.000** for the policy (396 better / **0 worse** of 950 states,
p = 1.2e-119). Distilling it into the head (`distill_tactics.py`, frozen trunk)
moved Cannon significantly (+161.9 HP, p = 2.0e-06) and Fireball not at all,
both still below random — the head's spatial signal is a 9×5 pooled map
upsampled 4× on a frozen trunk, so an exact-cell target is close to
inexpressible (CE fell 180.9 → 21.4 with argmax match stuck at 0.0%).

**Bankruptcy: fixed as a statistic, no outcome gain.** The potential-based
solvency term (`elixir_shaping.py`) measured **-1.0 points, p = 0.21** over 80
updates — policy-invariance is what makes it safe and also what limits it. An
inference-time reserve gate (`tactics.SolvencyGate`, refuse spends below 4
elixir while nothing attacks) fixes it outright over 130 paired openings:
bankruptcy **72.7% → 39.2%, 130/130 episodes, p = 1.5e-39**, with **total elixir
spent statistically unchanged** (102 → 98, p = 0.25) — so it moved *when* the
bot spends, not how much. But tower HP lost (p = 0.25) and win rate
(−0.042, p = 0.63) are both flat. **Having elixir does not help while the
placements are worth less than random**; placement quality is the binding
constraint.

**2026-08-14, the hybrid: route around the broken head rather than repair it.
+11.8 win-rate points, p = 1.9e-05.** `python_ai/hybrid_policy.py` — the network
keeps WHAT to play and WHEN; `tactics.py` decides WHERE for Cannon, Fireball and
Giant, and `SolvencyGate` vetoes spends that would bankrupt it. Two independent
pre-registered paired runs, opponent 1.5x:

| | n | neural | hybrid | delta | p |
|---|---|---|---|---|---|
| exploratory | 250 | 0.646 | 0.720 | +0.074 [−0.004, +0.150] | 0.070 |
| **confirmatory** | **600** | **0.584** | **0.703** | **+0.118 [+0.067, +0.171]** | **1.9e-05** |

Damage DEALT rose (+544/ep, p=0.010) while damage taken fell (−1011/ep,
p=7.4e-09) on unchanged spending — it attacks better *and* defends better. The
Giant rule was validated before shipping: bridge on the weaker-defended lane
scores **535.6** enemy tower damage against **3.3** for the policy's own cell
(n=913, p=3.0e-87). Wired into the live loop behind `--no-tactical`.

**The component that had to be REMOVED is the instructive part.** Letting the
tactical officer *initiate* Cannon/Fireball (rather than only place them) looked
necessary, since the commander's take-up of those cards is 0.06/0.00/0.00. A
5-arm ablation killed it: initiating ~4 Cannons a match cut tower damage DEALT
by 3,410/ep and win rate by 0.367 (p=0.013). **Defending better is worthless if
it is paid for with the attack.** The placement override works precisely because
it is PASSIVE — it changes where a card lands, never how often one is played, so
it cannot spend elixir the commander did not already commit. Same reason a FLAT
solvency reserve had to become `min(reserve, opponent_elixir)`: a flat reserve
blocks exactly the spends that build a push.

**The FIRST attempt at the coverage A/B was INCONCLUSIVE and the reason is a trap
worth knowing: warm-starting a converged policy into a FRESH training state
re-arms the initial entropy target and the controller dissolves the policy.** Seeding a
bare `state_dict` takes `train.py`'s legacy-checkpoint path, which resets
`episodes_completed` to 0, so `placement_entropy_target(0)` returns
`ENTROPY_TARGET_PLACEMENT_START = 0.65` against a policy measuring 0.11. Both
arms spent the run being inflated toward that target and ended at ~0.83 of max
with top-1 probability 0.002–0.008 — near-uniform placement, in the control as
well as the treatment. Same shape as the exploiter's 2026-07-31 entropy-scaling
failure. **Always resume through the full-checkpoint path when the entropy
schedule matters.**

Corollary: **modal share degenerates on a near-uniform distribution** — the
argmax of a flat map is arbitrary but deterministic, and both arms reported a
90–100% modal share while being flat. Read modal share next to top-1
probability; the collapse in the table above has top-1 at 0.62–0.83, which is
what makes it real.

**2026-08-14, the placement head got its resolution back: a zero-initialized
high-resolution residual branch (`place_hires`), and it works.** `cnn_trunk`
pools twice, so the head reads a **9×5** map of a 34×18 board and `place_up`
blows it back up; the card context enters as a spatially uniform vector.
`place_hires` adds a parallel path from the trunk's own **pre-pool 16×34×18**
activation at one-tile resolution, conditioned on the same `(hx, card)` context,
added to the coarse logits as a residual. 3,993 new parameters (+0.2%).

**No checkpoint is invalidated, and that was a design goal rather than luck.**
The handoff proposed concatenating into `place_up`, which changes its shape and
therefore discards the trained placement head from every checkpoint — the price
the 2026-08-09 checkerboard fix had to pay. Zero-initializing the branch's final
conv makes it an **exact** no-op at init (weight *and* bias zeroed, so the
residual add is exact, not approximate), so an old checkpoint loads and behaves
bit-identically and the cards that already work keep working. The loader
confirms it: `warm-started 27/27 tensor(s), re-initialized: []`. Gradient still
flows — a zero conv has a nonzero gradient of its own, so it leaves zero on step
one and the layer beneath it starts learning on step two.

Measured as a controlled A/B (`prove_hires.py`): one collection of 2,376 states
from the seed policy's own trajectory, contiguous-tail held-out split, both arms
from the same checkpoint and seed, identical epochs/lr/anchor, the **only**
difference being whether `place_hires` trains. Held out (n=582/653):

| | control (coarse) | +hires |
|---|---|---|
| Fireball exact cell | 0.0% | **63.9%** |
| Fireball mean distance | 14.25 tiles | **3.31** |
| Fireball top-1 p | **0.0017** | 0.1002 |
| Fireball modal share | 98.0% | 13.0% |
| Cannon within 2 tiles | 0.2% | **10.0%** |
| Cannon mean distance | 9.69 tiles | **6.89** |
| Cannon modal share | 88.8% | 31.8% |

**Read the control's top-1 probability first: 0.0017 against a uniform
1/612 = 0.00163.** Fit to the advisor's exact cell, the coarse head does not
merely fail — it *dissolves to uniform*, and then reports a 98.0% modal share,
which is the degenerate reading this file already warns about. Its training loss
plateaus at ~21 (matching `distill_tactics.py`'s 21.4) while the branch's keeps
falling to 15.1. Replicated on two independent collections.

**ENGINE-SCORED, which is the only verdict that counts** (`prove_placement.py`,
paired on states drawn by the seed policy, 12 episodes):

| | Fireball, elixir killed (n=1937) | Cannon, tower HP preserved (n=894) |
|---|---|---|
| seed (the shipping net) | 0.062 | 102.2 |
| control (coarse-distilled) | 0.000 | 161.3 |
| **+hires** | **1.863** | **232.6** |
| random legal cell | 0.346 | 357.0 |
| advisor (the ceiling) | 2.647 | 539.1 |

**Fireball is fixed.** It beats a random legal cell by **5.4×**
(+1.516, 95% CI [+1.368, +1.669], **561 better / 71 worse**, p = 1.8e-95) and
reaches **70% of the advisor's** value. This is the first time the placement
head has ever beaten random for that card — every prior measurement had it at
0.000–0.022 against random's 0.276–1.103, i.e. **worse than chance**. Against
the control it is 600 better / **0 worse**.

**The Cannon is improved but NOT fixed, and the honest statement is that it is
still worse than random.** +71.2 HP over the control (p = 0.029) and +130.4 over
the seed (p = 0.0034), but −124.4 against a random legal cell (p = 2.5e-09) and
−306.6 against the advisor. Consistent with everything else measured about it:
with a plateau target the head learns *roughly where* (mean distance 9.69 → 6.89
tiles, modal share 88.8% → 31.8%) and still commits to a cell worth less than
chance. **The tactical override in `hybrid_policy.py` therefore stays on for the
Cannon** — it is not yet redundant, and turning it off would give back the
+11.8 win-rate points.

Three things worth carrying:

- **A claim in the handoff was too strong, and testing it directly is what
  found the real story.** The premise was that the coarse head *cannot express*
  an exact cell (from "CE fell 180.9 → 21.4 with argmax match stuck at 0.0%").
  Tested on the task reduced to its essential — 14 boards differing only in
  which column holds one enemy — **the coarse head fits 14/14 exactly**.
  Nearest-upsample followed by 3×3 convs lets a fine cell mix neighbouring
  pooled cells, so sub-block position *is* recoverable. The limit is real but
  it is capacity at scale, not impossibility. `test_python_ai.py` keeps
  both results.
- **The Cannon's exact cell is a BAD SUPERVISION TARGET and that is a property
  of the teacher, not the student.** `building_score_map` scatters flat discs,
  so the top of the surface is a large exact-tie plateau and `np.argmax`
  returns its top-left cell by row-major accident. Quantified: lowering the
  softmax temperature cannot push the Cannon target below **~66% of maximum
  entropy** (Fireball reaches 41%), because the ties never break. So Cannon
  exact-match near 0 is expected and mostly uninformative — judge a building by
  distance and by engine score, never by exact cell.
- **The soft/neighbourhood target the handoff recommended first (A1) is
  measured NEUTRAL-to-WORSE here — do not re-try it without new reason.** As a
  third arm on the same collection (KL to `softmax(standardized advisor score /
  T)`, T=0.25 chosen off the printed entropy table): Fireball within-2 71.2% vs
  the argmax arm's 71.4% and mean distance 4.49 vs 3.31; Cannon within-2 8.4%
  vs 10.0%. It spreads mass over the neighbourhood exactly as designed and buys
  nothing. The plateau argument predicted it would rescue the Cannon; it did
  not. **Resolution was the binding constraint, not target softness.**

**2026-08-14 (later), two GAMEPLAY-AFFECTING changes: the spell anneal is wired
in, and the coverage term was given a target.**

**`spell_value_weight` was dead code and now runs.** Both trainers called
`compute_shaping(stats, prev_stats, gamma=gamma)` with no `w_spell`, so the
Fireball-value weight sat at `W_SPELL_VALUE_START = 0.08` for the whole of
training and the anneal its own comment block describes never happened. Nothing
detected it because **no test ever varied the argument** —
`test_compute_shaping_actually_responds_to_w_spell` is the regression that now
would. The term is deliberately NOT potential-based, so it biases the optimum by
construction; a schedule that never reaches zero is a permanent bias nobody
chose. Every win rate before this was earned under a constant 0.08.

`SPELL_VALUE_ANNEAL_START` (env `CLASH_SPELL_ANNEAL_START`, default 0) exists
because a warm start resumes PAST the horizon: `model_weights_selfplay.pth` is
at episode 64,309 against a 40,000-episode anneal, so a faithful wiring pins
`w_spell` at FINAL from the first step and the anneal cannot be observed at all.
At the default the behaviour is exactly the original intent.

**The coverage term now carries a TARGET where the advisor has one**
(`advisor_target.py`). `PLACEMENT_COVERAGE_COEF` adds an ENTROPY bonus on one
sampled affordable slot, and for a card the policy never plays that bonus is the
*only* placement gradient in the objective — so it pushes the map toward
uniform, which is precisely what the hi-res distillation is trying to undo. The
two are in direct opposition and coverage runs for all of training.

The resolution is a **row mask**: a row either has an advisor target and gets
KL to the advisor's masked score map, or it does not and keeps the entropy
bonus. Never both — entropy says "be spread out", KL says "be here", and a row
carrying both asks the head for two incompatible things. This is the fourth time
this project has landed on the same conclusion: *closing a coverage hole needs a
target, not noise.*

Three things worth carrying:

- **The gate is the load-bearing part.** `tactics` always returns a cell, and on
  a quiet board that cell is a default — the defensive pocket for a building, an
  arbitrarily tie-broken lane for the Giant. Training on defaults teaches a
  CONSTANT, which is the exact pathology being repaired. `target_logits_for`
  returns `None` there and the row falls back to entropy. This mirrors
  `distill_tactics.collect`'s `cover > 0 or catch > 0` filter.
- **The limiter was AFFORDABILITY, not the gate, and it was worth measuring
  rather than guessing.** Over 258 decision steps the sampled coverage slot held
  an advisor card 32.6% of the time and the advisor then spoke on 90% of those.
  Cannon(3)/Fireball(4)/Giant(5) are exactly the cards a near-bankrupt agent
  cannot afford — it sits under 3 elixir on 65.3% of decisions — so a uniform
  draw over *affordable* slots is biased toward the cheap cards, which are also
  the ones already getting actor gradient because they are the ones being
  played. `placement_coverage_slots` grew a `slot_weights` argument
  (`CLASH_ADVISOR_SLOT_WEIGHT`, default 5.0); measured `Advisor/Rows` 13-19 →
  23-25 per minibatch. Weights re-rank, never remove: a row whose affordable
  slots all weigh zero falls back to the unweighted mask rather than through to
  the no-op.
- **Read `Advisor/KL` and `Advisor/Rows` as a PAIR, never KL alone.** KL falls
  both when the head learns the surface and when the advisor simply stops
  speaking, and those are opposite situations. Same shape as every other
  aggregate this file warns about.

The coverage slot is now sampled **once at rollout time and buffered**, not
resampled inside every PPO epoch: an advisor target has to be computed against
the observation the slot was drawn on, and a fresh draw in the update would pair
one card's logits with another card's target.

**MEASURED, and it works. Pre-registered A/B, engine-scored, paired on states
drawn by one reference policy.** Two arms from `model_weights_hires.pth` on a
full training state at episode 64,309, byte-identical code, 80 updates each
(fixed in advance), the only difference being `CLASH_ADVISOR_COVERAGE_COEF`
(0 vs 0.10). Scored by `prove_placement.py --episodes 16`:

| arm | Fireball, elixir killed (n=3106) | Cannon, tower HP preserved (n=1793) |
|---|---|---|
| random legal cell | 0.515 | 456.9 |
| seed (hires-distilled) | 2.066 | 269.2 |
| control (coverage entropy only) | 2.387 | 402.9 |
| **treatment (advisor target)** | **2.969** | **480.3** |
| advisor (the ceiling) | 3.013 | 702.7 |

* **Fireball, treatment vs control: +0.583, 95% CI [+0.505, +0.661],
  p = 1.2e-47.** And treatment vs the ADVISOR is +0.043, CI [−0.018, +0.102],
  **p = 0.163 — statistically indistinguishable from the ceiling.** This is the
  first time the head has matched the advisor on any card.
* **Cannon, treatment vs control: +77.4, CI [+37.0, +118.6], p = 0.0012.**
  Against a random legal cell it is +23.4, CI [−35.3, +82.1], p = 0.118 — i.e.
  **no longer significantly worse than chance**, which every prior measurement
  was (−124.4, p = 2.5e-09). It is still far below the advisor (−222,
  p = 6.8e-22), so **the Cannon override stays on.**

**The frozen cell is gone, and that is the qualitative result:**

| net | Cannon modal | share | Fireball modal | share | cells used |
|---|---|---|---|---|---|
| seed | (11,0) | 46.2% | (11,0) | 32.3% | 90 / 210 |
| control | (11,0) | 38.4% | (11,0) | 25.8% | 116 / 241 |
| **treatment** | **(3,15)** | **19.1%** | **(13,14)** | **6.6%** | **138 / 278** |

`(11,0)` is the pathological own-back-row cell this file has tracked since
2026-08-11. The control is still parked on it for both cards; the treatment is
not, at top-1 probability 0.076 / 0.056 — far above uniform 1/612 = 0.0016, so
this is a head that is SHARP and MOVES ITS MODE, not a dissolved one. That is
exactly the signature the modal-share detector was defined to look for.

**A prediction of the handoff that did NOT survive contact, stated because it
was the stated reason for doing this work.** §3 Step 1 predicted the entropy
coverage term would ERODE the distilled placement head under PPO. It does not:
the control IMPROVED over its seed on both cards (Fireball 2.066 → 2.387,
Cannon 269.2 → 402.9) across 80 updates. So the conflict is real in mechanism
but the erosion is not observable at this horizon. The advisor target is worth
having because it is much BETTER, not because the alternative decays.

**AND IT HOLDS UNDER A LONG RUN. 2026-08-15, ep 78,270 (13,961 episodes / 524
PPO updates of pipeline 2 with the term on, zero alarms throughout).** Same
harness, same reference policy, same paired protocol:

| | seed (hires) | control (80 upd, coef 0) | **main (524 upd, coef 0.10)** | advisor | random |
|---|---|---|---|---|---|
| Cannon, tower HP preserved (n=1582) | 276.0 | 403.0 | **553.1** | 687.9 | 411.3 |
| Fireball, elixir killed (n=2979) | 1.506 | 2.190 | **2.565** | 2.464 | 0.469 |

* **The Cannon beats a random legal cell for the first time in this project's
  history: +141.8, 95% CI [+74.5, +210.2], p = 0.0071.** Every prior
  measurement had it BELOW chance (−124.4, p = 2.5e-09 at v1.2.0; +23.4,
  p = 0.118 and not significant at 80 updates). The advisor still beats it
  (−134.9, p = 5.0e-10), so **the Cannon override stays on** — but the gap has
  closed from −306.6 (v1.2.0) to −222 (80 updates) to −134.9.
* **Fireball has overtaken the advisor**, +0.100 elixir in the net's favour.
  The two tests disagree on significance — the bootstrap CI [−0.175, −0.025]
  excludes zero, the exact sign test does not (223 better / 252 worse,
  p = 0.199) — because the advisor wins more pairs while the net wins bigger
  ones. **The defensible claim is "no longer distinguishable from the advisor,
  and certainly not worse", which is enough to retire that override.**

**The dynamism table is the qualitative proof, and Fireball's is the cleanest
result this metric has ever produced:**

| net | Cannon modal | share | top-1 | Fireball modal | share | top-1 | cells |
|---|---|---|---|---|---|---|---|
| seed | (11,0) | 44.9% | 0.140 | (11,0) | 32.8% | 0.087 | 88 / 188 |
| control | (11,0) | 27.1% | 0.050 | (11,0) | 16.9% | 0.030 | 126 / 228 |
| **main** | **(16,15)** | **17.0%** | 0.062 | **(4,17)** | **12.7%** | **0.112** | **135 / 256** |

Fireball's modal share fell 32.8% → 12.7% while its top-1 probability ROSE
0.087 → 0.112. That is the exact signature this file defines as healthy and
which no previous net has shown: **more confident within a state, less
repetitive across states.** Its modal cell moved from the own-back-row (11,0)
to (4,17) — just across the river at the left bridge. The Cannon's moved to
(16,15), the right bridge mouth on our own side.

**Watch this in any future run:** modal share and top-1 falling TOGETHER is
dissolution toward uniform, not a cure. They diverged here, which is what makes
it real.

**THE GIANT IS CURED TOO, and it was the worst of the three** (`prove_giant.py`,
new — `prove_placement.py` scores only Cannon and Fireball, so this card had
never been engine-scored at all). Enemy tower damage over 600 ticks, paired on
states drawn by the cured net, n=2784:

| arm | tower damage | modal cell | share | cells used |
|---|---|---|---|---|
| v1.2.0 | **13.7** | (11,0) | 49.7% | 38 |
| **cured** | **231.9** | (1,15) | 24.7% | **140** |
| advisor | 380.5 | (14,15) | 57.7% | 2 |
| random legal cell | 86.7 | — | — | 242 |

**+145.2 vs a random legal cell (p = 6.5e-19)** and **+218.2 vs v1.2.0
(p = 1.0e-98)**. The shipping net's Giant was 6x WORSE than chance at 13.7 —
matching the 3.3 recorded when `best_giant_cell` was written — and used 38 of
612 cells with half its mass on (11,0). The advisor still wins (−148.6,
p = 1.1e-69), and note WHY: its rule is essentially two cells (57.7% on one of
the two bridges), which is a very strong prior this board rewards.

**So all three starved cards moved from worse-than-chance to better-than-chance.**
Fireball reached the advisor; Cannon and Giant beat random but remain below it.

**THE TACTICAL OVERRIDE IS NOW REDUNDANT — a measured NULL, which is the point.**
`hybrid_ab.py --per-card`, 200 paired openings on the cured net, solvency gate
held ON in every arm so only placement varies:

| arm | win rate | vs neural | p |
|---|---|---|---|
| neural (no placement override) | 0.507 | — | — |
| cannon_only | 0.510 | +0.003 | 1.0 |
| cannon_giant | 0.480 | −0.028 | 0.54 |
| all_three | 0.480 | −0.028 | 0.56 |

At v1.2.0 the same override was worth **+11.8 points, p = 1.9e-05**. This n had
the power to see an effect that size and it is gone. The officer is not helping
because there is no longer a hole for it to fill. **`hybrid_policy.py` can drop
to gate-only.**

**AND THE UNCOMFORTABLE HALF, which is the more useful result.** Two paired
win-rate comparisons of the cured net against v1.2.0's
`model_weights_selfplay.pth`, both side-controlled:

| opponent | v1.2.0 | cured | delta |
|---|---|---|---|
| C++ `HeuristicOpponent` @1.5x (`net_ab.py`, 200 paired openings) | 0.6225 | **0.5100** | −0.1125, CI [−0.2025, −0.0200] |
| **v1.2.0 itself**, head-to-head, sides swapped (`net_h2h.py`, 120 pairings) | 0.3875 | **0.6125** | **+0.1125, CI [+0.054, +0.171]** |

**This is SPECIALIZATION, not degradation, and only a neural opponent could tell
the two apart.** 13,961 episodes in a PFSP league whose members are all neural
made the net significantly better against that opponent class — it beats its own
predecessor — while losing ground against the C++ heuristic, which pipeline 2
never shows it. Read either number alone and you get the wrong answer.

Two controls that make the attribution stick:

- **The seed is not the cause.** `model_weights_hires.pth` was never win-rate
  tested (the v1.2.0 handoff says so). Measured: 0.6125 vs v1.2.0's 0.6350,
  delta −0.0225, CI [−0.100, +0.055], p = 0.76 — indistinguishable. The whole
  −0.1125 came from the training run, not from the distilled starting point.
- **Sides were swapped** in the head-to-head. A policy beat a bit-exact copy of
  itself 0.598 once purely by side assignment; it reads 0.530 today, small but
  not zero, and a one-sided duel would fold that straight into the result.

**So `model_weights_selfplay.pth` was NOT replaced.** Two checkpoints now exist
with different strengths, and which is "better" depends on the opponent you
care about. The real game is not the C++ heuristic, which argues for the cured
net; but nothing here measures the real game.

**A resume trap that is specific to `place_hires` and bit pipeline 2 only.**
The branch added 6 parameters, so a pre-2026-08-14 checkpoint's optimizer
describes 26 and the net has 32. `train.py` degrades gracefully (it gates
optimizer restore on a clean model load); **`train_selfplay.py` loads it
unconditionally and dies at startup** with "loaded state dict contains a
parameter group that doesn't match the size of optimizer's group" — and
`load_state_dict_flexible` reports CLEAN, because all 33 tensors are supplied.
`setup_ab_arm.py` remaps the moments **by name**: the new parameters land at
indices 22-27, in the MIDDLE of `named_parameters()` order rather than appended,
so the obvious "keep 0..25 and append the rest" hands the trunk's moments to the
placement head — a run that trains, looks healthy, and is quietly wrong.

**2026-08-15, the heuristic regression is fixed by WIRING IN SEARCH, and the
weights never changed.** The cured net's loss against the C++ heuristic
(0.51 vs v1.2.0's 0.6225) was going to be answered with a from-scratch
search-distilled run plus a mixed curriculum. Both premises were measured
first, and both are false:

- **The mixed curriculum already exists and did not work.**
  `BUILTIN_TRAINING_OPPONENTS` (2026-07-31) puts the heuristic in the PFSP pool
  at `BUILTIN_MIN_WEIGHT = 0.5`. Reproducing the env's own pool construction and
  sampler at the cured net's state (48-member pool, measured win rates) gives a
  realized share of **20.8%** — ~2,897 of that run's 13,905 episodes were
  ALREADY played against the heuristic, and it regressed anyway. Routing is
  correct (`game.step()`, not `step_self_play`), so this is not the silent-
  no-opponent trap. Adding heuristic exposure is a no-op.
- **Search cannot bootstrap a from-scratch run.** Search scores candidates with
  the net's OWN critic, so at random init the expert is not an expert. Paired,
  n=60: random init @1.0x scores **0.450 policy / 0.417 search** (ns) while
  overriding **21.5%** of decisions; the trained net @1.5x scores 0.483 / 0.667
  at **10.2%**. At init it deviates twice as often and gains nothing, so early
  distillation targets are noise.

**`horizon` is the lever, and depth is nearly free.** One engine step costs
0.015 ms; one scored candidate costs a network row at 0.13 ms — so lookahead is
~9x cheaper than width. Sweep (cured net vs heuristic@1.5x, n=80 paired):
horizon 4 → 0.667 (1.6x cost), 4 with K≤13 → 0.788 (1.7x), 8 → 0.925 (1.5x),
**12 → 0.963 (1.5x)**, 20 → 0.875 (1.8x). It degrades past ~12 because a
candidate rollout assumes both sides no-op and 20 s of that stops resembling
the game. Chosen on the sweep, then CONFIRMED on a fresh independent run.

**Confirmatory, `model_weights_cured.pth` + search horizon 12:**

| criterion | bar | result |
|---|---|---|
| vs C++ heuristic@1.5x (n=400 paired) | >0.65 | **0.9225** (policy 0.5200, delta +0.4025 CI [+0.3486, +0.4564], 174 better / 13 worse) |
| h2h vs v1.2.0 as shipped (n=150 swapped) | >0.60 | **0.7200** CI [0.6750, 0.7650] |
| Cannon placement, shipping config | dynamic | modal 12.9% over **36** cells (greedy: 17.1% over 20) |

**THE UNCOMFORTABLE CONTROL, and it is the more useful result. Give BOTH sides
search and the cured net's advantage disappears:** 0.4425, CI [0.3825, 0.5025]
(n=100 swapped) — no difference resolved, point estimate favouring v1.2.0. So
the +0.72 is **search, not the cured weights**. Search and the placement cure
repair the same weakness, and they do not stack. Anything claiming the cure
made a stronger network has to answer this number.

Two smaller things worth carrying:

- **The x mod 4 detector fires on a healthy net, because its null is wrong for
  a trained policy.** The shipping config measured 43.6% on x ≡ 1 (mod 4),
  χ² = 96.3 — which reads exactly like the 2026-08-09 checkerboard bug. It is
  not. The discriminating test is the one the fix was verified with: on a
  SPATIALLY CONSTANT input, phase explains **6.8–14.0%** of the head's surface,
  against the broken head's **100%**. A trained policy concentrates on columns
  for tactical reasons and the uniform-column null cannot tell that from a head
  artifact. Run the constant-input test before believing the histogram.
- **v1.2.0 is a fair baseline under the new architecture.** It loads with
  `place_hires` freshly initialized, and `place_hires.2` weight AND bias are
  both exactly zero, so the residual add contributes exactly nothing and the
  checkpoint plays bit-identically to how it shipped. Verified, not assumed.

The shipping configuration is named in one place, `python_ai/shipping.py`, so an
evaluation and a deployment cannot drift onto different settings. Harnesses
added: `net_h2h_search.py` (either side may search) and
`prove_placement_shipping.py` (dynamism over cells ACTUALLY played).

**2026-08-16, distilling the h=12 expert into the CURED net does NOT work, and
the reason is that the cure already took the absorbable part.** The obvious next
step after the above was to bake search's +0.4025 into the weights, since the
h=4 expert had distilled for +0.045 (p = 0.0074). Run at the best known recipe —
value-DISTRIBUTION labels, frozen trunk, 16 epochs, 180 episodes, seeded from
`model_weights_cured.pth`. Collection: 48,035 rows, expert win rate 0.9528,
deviation 12.76%. T=0.05 chosen from `--target-entropy` BEFORE any outcome
(spread mean 0.2801 / median 0.1795, putting the target at 0.573 of max
entropy), 21.8% of rows carrying ≥2 candidates.

**Three independent measurements, all null:**

| | h=4 into v1.2.0 (worked) | h=12 into cured |
|---|---|---|
| conditional lift | **+0.1535** | **−0.0032 ± 0.0230** |
| p1/p0 selectivity | 2.17 | **0.92** |
| paired greedy A/B | +0.045, p = 0.0074 | **+0.0250, CI [−0.0509, +0.1009], p = 0.608** (n=300, 72 better / 65 worse) |
| search delta ON the student | 0.319 → 0.12-ish (deviation 14.8% → 12.1%) | **+0.3583 [+0.258, +0.459]**, deviation 13.17% → **13.06%** |

**Read the last row first — it is the one that settles it.** Search is worth
+0.4025 on the cured net and +0.3583 on its distilled student, with heavily
overlapping CIs, and it overrides the student just as often as the teacher
(13.06% vs 13.17%). A student that had absorbed the expert would be deviated
from LESS. Essentially nothing transferred.

**The seed is the variable, not the expert.** The distillation that worked was
seeded from `model_weights_selfplay.pth`, whose placement head was *broken* —
Cannon, Fireball and Giant all placing WORSE THAN RANDOM. That is an enormous
gap for an expert to teach into. Seeded from the cured net there is no such
gap: the condition this file names for expert iteration to pay ("the value head
is substantially better than the action head is at exploiting it") is largely
gone once the action head works. This is the same fact the compute-matched h2h
control reported from the other side — **search and the placement cure repair
the same weakness, so they neither stack nor substitute for each other's
absence.**

What is left for search to add is the part that requires ACTUALLY RUNNING THE
SIMULATOR twelve seconds forward, and a reactive head that only ever sees `s`
has no way to represent it. That was already the standing hypothesis ("the
residual is plausibly structural"); this is the first measurement that isolates
it from the placement confound.

Two caveats recorded honestly:

- **The fit is UNDERCONVERGED and that is not the explanation.** Loss was still
  falling at epoch 15 (2.8605 → 2.4645) and argmax agreement still rising
  (0.433 → 0.446), so more epochs is the one untried lever. But underfitting
  does not predict the search-on-student result: a partially-fit student would
  still be deviated from less often, and it was not.
- **`--train` and `--train-dist` are INDEPENDENT flags, not a mode selector.**
  Passing both runs hard-label BC to completion first and only then the
  distribution phase — ~2.7 h of wasted compute here. It did not contaminate
  the result (the distribution phase re-loads `student` from `--weights`), but
  the hard-label pass also overwrites `--out` on its way past.

**`model_weights_h12dist.pth` is NOT shipped.** `shipping.py` stays on
`model_weights_cured.pth` + search horizon 12.

**2026-08-16, "the bot plays like a disconnected zombie": one of the three
reported symptoms is real, and its cause is the ENTROPY CONTROLLER's
normalizer, not the observation, the architecture or the reward.** Prompted by
a live emulator session reporting suicide placements, back-corner spam and
defensive apathy. All three were tested in the simulator against
`model_weights_cured.pth` before anything was changed. Two do not reproduce.

**REFUTED — "it drops a Musketeer on top of a Mini PEKKA".** Musketeer had
never been engine-scored here (`prove_placement.py` covers Cannon/Fireball,
`prove_giant.py` the Giant). Scored on the same protocol — snapshot, inject at
the proposed cell, run a 300-tick lifetime, read the engine — over 337 paired
states against a random legal cell:

| | policy | random legal cell | delta |
|---|---|---|---|
| elixir value killed | **5.745** | 4.481 | **+1.264**, CI [+0.650, +1.872] |
| tower damage taken | **1411.3** | 1708.3 | **297 HP less** |

144 better / 106 worse / 87 tied. The head places the Musketeer **better than
chance**, which is the opposite of the Cannon/Fireball/Giant pathology this
file spent 2026-08-11 to 08-15 curing.

**REFUTED — "it is disconnected from the board".** Counterfactual ablation:
delete EVERY enemy troop from the observation (their cells zeroed across all
enemy channels; towers, own units, elixir, hand and the recurrent state left
untouched) and re-query the same net from the same hidden state. Pooled over
1,532 states, the placement distribution moves **TV = 0.444** and the argmax
cell changes in **60.4%** of states, against **TV = 0.625** for switching to a
different card — a change the head is known to condition on. Per card it is
stronger still: Mini PEKKA 0.844 / 83.3%, Valkyrie 0.762 / 93.1%. The head
reads the board.

**CONFIRMED, and it is a REGULARIZER BUG. `LOG_N_CARD = log(hand_size + 1)` is
a maximum the masked distribution can never reach.** Both trainers divided the
card head's entropy by `log(5)` and drove the result to
`ENTROPY_TARGET_CARD = 0.35`. But `mb_decision` means "at least one card was
AFFORDABLE", and the affordability mask leaves only `(#affordable + 1)` legal
arms. Measured over 706 decision steps:

| n_legal on a decision step | share | reachable max |
|---|---|---|
| **2** (one affordable card + no-op) | **54.1%** | log 2 = 0.693 |
| 3 | 15.6% | 1.099 |
| 4 | 25.8% | 1.386 |
| 5 | 4.5% | 1.609 |

The target `0.35 × log(5) = 0.5633` nats is **81.3% of the reachable maximum**
on the majority case. So the controller demanded the play/wait choice be near a
coin flip — and then **read 0.3125 against its own 0.35 target and kept RAISING
the coefficient**, while the policy's real randomness was **0.4906 of
reachable**.

The downstream chain is the reported symptom:

| | |
|---|---|
| mean elixir | **2.25 / 10** |
| P(nothing affordable) | **73.9%** |
| ...during the largest threat bucket | **78.5%** — worst when defending matters |
| P(play), no threat → HUGE threat | 0.1008 → **0.1016** (flat) |
| P(play \| something affordable) | 0.3655 → 0.4719 (the response IS there) |

**The agent is not apathetic and not blind. It is bankrupt, and the regularizer
is what bankrupts it** — it spends on sight because it is being paid to flip a
coin. This is the same shape as the 2026-07-31 exploiter collapse (raw nats vs
`log 612`) and the 2026-08-11 no-op placement average: **a normalizer that does
not hold in the regime being measured.** Seventh instance.

Fixed by normalizing each head per-step by `log(n_legal)` — recovered from the
card mask, and from `torch.isfinite(pl_seq)` for placement, so no plumbing is
needed. The placement head had the same defect more mildly (troops see ~242 of
612 cells, so its annealed target was inflated ~17%) and is fixed identically.
`test_card_entropy_must_be_normalized_by_the_REACHABLE_maximum` pins the
invariant: **uniform over the legal arms must read exactly 1.0, at any number
of arms.** Under the old divisor a uniform 2-arm row read 0.431.

**The tell was already in the codebase.** `train_selfplay.py`'s per-card
diagnostic has always divided by `log(n_legal)` and its comment gives the
reason — "a spell sees 588 cells, a plain troop 242, the Cannon 208 ... a raw
nat count is not comparable across cards". The DIAGNOSTIC was right and the
OBJECTIVE was never given the same treatment.

**GAMEPLAY-AFFECTING**: it changes the loss in both pipelines.

**Also 2026-08-16: "perfect defense" is now expressible, and the obvious way to
do it was the wrong way.** The tower term is linear and SYMMETRIC — 100 HP
chipped off the enemy pays exactly what 100 HP taken costs — so the reward is
indifferent between "trade 500 for 500" and "take 0, deal 0". The tempting fix,
weighting damage TAKEN above damage DEALT, is the one thing that must not be
done: this file already records that policy-invariant tower shaping left PURE
DEFENCE as the optimum and win-condition usage decayed to **0.7%**.

`flawless_defense_bonus()` instead pays `W_FLAWLESS_DEFENSE = 0.5` scaled by
the fraction of our own tower HP still standing, **only on a WIN**. Gating on
the win is the whole safety argument: a turtle that stalls into a timeout
collects nothing and still pays `DRAW_PENALTY`, and a loss collects nothing, so
the term cannot reorder win/loss/draw at all — it can only rank WINS against
each other. Not potential-based, so biasing by construction, the same
eyes-open trade as `W_TOWER_DESTROYED`. Three tests pin it, including that a
DRAW collects nothing however clean, and that the post-autoreset counter is
read through a running max (otherwise a scraped win is paid as flawless).

**And a live/sim drift worth knowing: there are THREE disagreeing `max_ticks`
defaults.** `ClashEnv.h`'s C++ constructor says 3600, **`bindings.cpp` says
1800**, and `gym_wrapper` — which trains the policy — passes 3600. A bare
`ClashRoyaleEnv(deck, deck)` from Python silently gets a HALF-LENGTH match.

`perception_encoder.py` had been written against the binding default, dividing
the time scalar by 1800 while the policy it feeds was trained at 3600 — **the
deployed agent's clock ran at twice the rate it had learned**, on the one input
clock management depends on. Invisible in every simulator metric, because
nothing in the simulator path uses that file. The round-trip bit-exactness
tests did not catch it; they *pinned* it, by building their env with the bare
constructor. Both are fixed, and the tests now pass `TRAINING_MAX_TICKS`
explicitly.

**PRE-REGISTERED PREDICTIONS for the from-scratch 2.6 run, written before it
had trained.** The entropy fix is justified by a mechanism, not yet by an
outcome. These are the numbers that falsify it, all measurable with
`probe_defense.py` / `probe_entropy_norm.py` on the new net:

1. **Mean elixir rises above 2.25 / 10** and **P(nothing affordable) falls
   below 73.9%**. This is the direct claim. If bankruptcy persists at the same
   level, the normalizer was not the binding cause and the diagnosis is wrong.
2. **P(play) stops being flat against threat.** It was 0.1008 → 0.1016 from no
   threat to the largest. Any real defensive reflex has to show up here.
3. **`Policy/Entropy_Card_Frac` settles near 0.35 rather than below it.** It is
   now measured against a reachable ceiling, so the controller should be able
   to hold its target instead of chasing one it cannot reach.
4. **The card entropy COEFFICIENT should fall early, not rise.** A fresh net is
   near-uniform over its legal arms (~1.0 of reachable), i.e. far ABOVE target,
   so the controller must push down. Observed in the first updates: 0.0500 →
   0.0363. Under the old normalizer it rose instead.

Prediction 4 is already confirmed; 1-3 need a trained net.

**FIRST READ AT ep 600 (still very immature). Prediction 2 confirmed hard,
prediction 1 half-failed, and MY PREDICTION WAS PARTLY THE WRONG METRIC.**

| | old net (Giant deck) | ep-600 net (2.6) |
|---|---|---|
| P(nothing affordable) | 73.9% | **52.9%** |
| mean elixir | 2.25 | **1.84** (went DOWN) |
| P(play) overall | 0.105 | **0.237** |
| P(play \| affordable), no threat → HUGE | 0.3655 → 0.4719 | **0.4606 → 0.8601** |

**Prediction 2 is the real result.** The defensive reflex went from a 29%
relative rise across the threat range to an 87% rise ending at **0.86** — the
agent now plays a card on 86% of decisions where it can afford one during a big
push. That was the symptom being chased and it is gone.

**Prediction 1's "mean elixir rises" was a BAD PREDICTION and it failed.** Mean
elixir is not deck-invariant: 2.6 Hog Cycle has two 1-cost cards, so
"affordable" is satisfied at 1 elixir and the agent can correctly hold less
while having MORE options. The deck-invariant version of the claim —
P(nothing affordable) — improved 73.9% → 52.9%. Use that one; do not compare
mean elixir across decks.

**THE HONEST LIMIT, stated because the numbers above are otherwise
over-readable: the entropy fix and the deck change landed TOGETHER, so this
comparison cannot attribute the improvement to either one.** A cheaper deck
alone would raise P(play) and lower P(nothing affordable). The clean
attribution needs a from-scratch control arm on 2.6 with
`new_ent_card / log(n_legal)` reverted — ~28 h — and nothing here substitutes
for it. What IS unconfounded is the mechanism: the old divisor scores a
literal coin flip at 0.413 against a 0.35 target, which is arithmetic, not a
measurement.

One more early read, same caveat: Musketeer at ep 600 already preserves
**+458 HP** vs a random legal cell (CI [+329, +583], 276 better / 161 worse)
against the shipped net's +297 — but on a different deck, so not comparable.

**And the fresh run makes the defect legible in one number.** At ep ~180 the
untrained policy measures `Policy/Entropy_Card_Frac` = **0.958** — 95.8% of its
REACHABLE maximum, i.e. very nearly a coin flip, correctly far above the 0.35
target, so the controller drives the coefficient down to its 0.01 floor.

Under the OLD divisor that same uniform policy would have read
`0.958 × log(2)/log(5) = 0.413`. **A policy that is flipping a literal coin on
"play or wait" scored 0.413 against a 0.35 target — i.e. the old controller
considered near-maximum randomness to be roughly correct, and pushed UP from
there.** That single comparison is the whole bug.

Both heads now read on a scale where **1.0 means uniform over the legal arms**,
which is what makes the target interpretable at all. Placement reads 0.996
against its 0.649 annealed start, also correctly falling.

**RUN 1 PROGRESS, and an alarm I raised and then measured away.** The
from-scratch 2.6 run cleared the mirror phase fast — stage gates at ep 1,615 /
2,778 / 3,588 / 6,216 (opp elixir 1.0 → 1.4), then **phase advanced to
`random_opponent` at ep 6,373** on a 0.60 mirror win rate. Handoff to
`train_selfplay.py` scheduled at ep 11,373 (a 5,000-episode random-deck budget,
NOT the 40,000 cap this file used to name — the budget is what fires first).

Perfect defense, same checkpoint (ep 6,053), two opponent strengths:

| | @1.0x | @1.4x (live difficulty) |
|---|---|---|
| win rate | 1.000 | 0.700 |
| tower HP left \| WIN | 0.939 | 0.661 |
| flawless wins | **47.5%** | **0.0%** |
| crowns conceded \| win | 0.025 | 0.571 |

`W_FLAWLESS_DEFENSE` is measurably shaping behaviour (0.820 → 0.939 HP on wins
between ep 2,523 and 6,053 at matched 1.0x), **but zero flawless wins at 1.4x**
— the "zero tower damage" standard is currently a property of facing a weak
opponent, not a learned skill. Do not quote the 1.0x number alone; it is
saturated at win rate 1.000.

**THE ALARM: three of eight cards fell to near-zero usage — Hog 0.8%, Fireball
0.6%, Cannon 1.6% — the same three ROLES (win condition, spell, building) that
collapsed in the Giant deck.** That shape is why it looked structural.

**IT IS NOT THE PLACEMENT COLLAPSE. Measured, `prove_hog.py` (new), n=2,552
paired states, enemy tower damage over 600 ticks:**

| | |
|---|---|
| policy cell | **470.3** |
| random legal cell | 441.8 |
| delta | **+28.5**, 95% CI [+6.3, +51.7], 826 better / 734 worse |
| placement | modal (0,12) at **4.7%**, **155 distinct cells** |

Better than chance, and the dynamism is the **healthiest a win condition has
ever measured here** — against v1.2.0's Giant at 38 cells / 49.7% modal and the
cured net's 140 / 24.7%. A frozen head returns one cell; this one uses 155. So
the low usage is a **card-head VALUATION, not a broken placement function**, and
the documented response applies: do not force it. Forcing Fireball once dropped
win rate 97% → 23% because the low weighting was correct.

What stays open is whether the valuation is OPTIMAL. +28.5 on a base of 442 is
only +6.5% over random, so the Hog is placed better than chance but not
strongly. That question needs a forced-usage A/B, which is exactly the
experiment whose last outcome was "the policy was right".

**Two things this audit did NOT fix, stated so they are not mistaken for
solved.** `skip_frames = 10` (one decision per second) is a harder ceiling for
2.6 Hog Cycle than it was for Giant beatdown — pulling a Hog with a Cannon and
timing an Ice Spirit are sub-second decisions — and it was left alone because
changing it is an unmeasured throughput/precision trade, not because it is
fine. And the observation still carries **no card-cycle information**, which is
the single most deck-specific gap: 2.6 is *defined* by cycling back to Hog
faster than the opponent cycles their answer, and the net can see only the 4
cards in hand.

---

## The 1.5x Curriculum Overfitting Hypothesis

**Status: HYPOTHESIS, stated with its falsifier. Not established.** It is
recorded here because it is the best available explanation for a result that
survived four independent attempts to fix it, and because the test that would
kill it is cheap.

**The claim.** Every win-rate measurement that matters in this project is taken
against the C++ `HeuristicOpponent` at a **permanent 1.5x elixir multiplier**,
and phase 1's curriculum tops out at 1.5x as well. But the Hog Rider is a
*punish* card: its value comes from exploiting a temporary elixir deficit in the
opponent's economy. **An opponent with a permanent 1.5x multiplier never has a
meaningful deficit to punish.** Against a permanently over-resourced opponent,
spending 4 elixir on a unit with zero defensive utility is close to strictly
dominated -- whatever it deals, the answer is always affordable, and the
counter-push arrives against 4 fewer elixir of defence.

If that is right, the agent has not failed to learn its win condition. It has
*correctly solved the environment we built*, and the environment is closer to a
tower-defence survival mode than to Clash Royale. The 0% win-condition usage and
the cheap-cycle defensive turtle are then the mathematical optimum, not a
pathology -- which is exactly what every intervention has independently found.

**What it explains.** Four separate fixes, each targeting a different mechanism,
all produced nothing:

| intervention | result |
|---|---|
| `W_WIN_CONDITION_DAMAGE` reward multiplier | fired thousands of times, moved nothing |
| Hog advisor (bridge placement, engine-validated +176.0 tower damage) | usage 0.4% -> 0.2% |
| forced usage at epsilon = 0.15 | -0.125, p = 0.044 |
| SMART forcing (advisor timing gate + advisor cell, gate correctly calibrated to 1.5x) | **-0.300, p = 3.2e-06** |

A card that is genuinely viable should have responded to at least one of those.
A card that is *correctly valued at zero by the environment* responds to none of
them, which is what was observed.

**THE FALSIFIER, and it is one command.** Run the same paired forced-usage A/B
at `--opp-elixir 1.0`:

```bash
python_ai/venv/Scripts/python.exe python_ai/force_hog_ab.py     --weights <net> --n 120 --opp-elixir 1.0 --force-prob 0.0 --smart-force
```

* If the penalty **shrinks or reverses** at 1.0x, the hypothesis is supported
  and the correct response is a CURRICULUM change -- train and evaluate across
  a range of multipliers including 1.0, rather than pinning everything at 1.5x.
* If the penalty **persists at ~-0.30**, the hypothesis is dead and the Hog is
  unviable in this engine's physics regardless of the opponent's economy. The
  honest response then is to stop rehabilitating it.

**A CEILING CAVEAT that must not be ignored when reading that test.** This file
already records that at 1.0x the ep-64k policy wins ~100%, so both arms can
saturate and pin the delta at 0 *by the opponent*, not by the treatment -- which
would look like support for the hypothesis while measuring nothing. The result
is only interpretable if the BASELINE arm is below ceiling. Check the baseline
win rate before reading the delta; if it is >= 0.95, the test is void and needs
an intermediate multiplier (1.2-1.3) instead.

**TESTED, 2026-08-18, AND THE HYPOTHESIS IS SUPPORTED -- BUT NOT PROVEN.** The
same SMART-forced A/B (advisor timing gate + advisor bridge cell, gate
calibrated to each multiplier), n=120 paired, ep-25202 net:

| opponent | baseline win | SMART win | delta | p |
|---|---|---|---|---|
| **1.00x** | **1.000** | 1.000 | +0.0000 | -- VOID (ceiling) |
| **1.25x** | 0.950 | 0.825 | **-0.1250** | 0.0059 |
| **1.50x** | 0.617 | 0.317 | **-0.3000** | 3.2e-06 |

**The cost of playing the win condition falls monotonically as the opponent's
economy falls.** That is the hypothesis's central prediction and it holds.

The 1.00x row is VOID exactly as the caveat above warned -- the baseline
saturates at 1.000 and the delta is pinned by the opponent, not the treatment.
It is reported rather than dropped because deleting a void arm after seeing it
is how a ceiling gets mistaken for a cure.

**And the honest alternative explanation, which is NOT fully excluded.** A
baseline near 1.0 has less room to lose, so some of the shrinkage is
compression. Normalising by available headroom: at 1.25x the penalty is
0.125/0.950 = **13%** of what could be lost; at 1.5x it is 0.300/0.617 =
**49%**. Still smaller at the lower multiplier, so the effect survives that
correction -- but 1.25x is close enough to the ceiling that the point deserves
replication at a multiplier where the baseline sits nearer 0.7-0.8.

**WHAT THIS DOES NOT SAY: the Hog is not rehabilitated at any multiplier
tested.** The penalty shrinks; it never reverses. Even at 1.25x, forcing the
win condition with perfect timing and an engine-validated placement costs a
significant 12.5 win-rate points. So the correct response is a CURRICULUM
change -- train and evaluate across a range of opponent economies instead of
pinning everything at 1.4-1.5x -- and NOT more Hog-specific machinery. Four
such mechanisms have now been built and measured, and all four returned null.

**A second, cheaper prediction worth checking.** If the hypothesis holds, the
win condition should be *more* used, not less, by any policy trained with 1.0x
exposure. Nothing in the current run provides that -- phase 1's curriculum
starts at 1.0x but the agent passes through it in ~1,600 episodes and spends the
remaining ~24,000 at 1.4-1.5x.

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
   which is why `python_ai/search_ab_test.py` batches all K candidate
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
   `python_ai/search_ab_test.py`, 160 paired trials at 1.5× opponent elixir,
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
   `python_ai/expert_iteration.py`. 80 episodes of search-labelled play (20,333
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
never moved off ~5.3.

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
  advisor_target.py    The advisor's score surface as a TRAINING TARGET for the
                       placement-coverage term, plus the loss both trainers
                       call. The only place that knows which cards have rules.
  setup_ab_arm.py      Builds a FULL training checkpoint for an experiment arm
                       (and remaps optimizer moments by NAME across a
                       checkpoint that gained parameters).
  validate_pipeline.py Pre-flight: PFSP routing, scenario contracts, advisor
                       targeting at scale, search cost ratio, spell anneal,
                       side null, C++ suite.
  monitor_run.py       Health daemon for an unattended run. Read-only.
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

344 tests (343 pass, 1 skipped), none requiring an emulator — they run against frozen replay
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
