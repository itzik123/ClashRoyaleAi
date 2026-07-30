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
                    phase 1  "random_opponent"  vs randomised decks
       |  win rate >= PHASE2_ENTRY_WIN_RATE (0.60) and stage >= 4
       v
train_selfplay.py   phase 2  PFSP league        vs frozen snapshots +
                                               4 scripted bots + exploiters
```

### Network (`model.py`, 1.88 M params)

`MicroRoyaleNet` — the LSTM is 96% of the parameters:

| | params | |
|---|---|---|
| `cnn_trunk` | 7,680 | 21→16→32 conv, 2× MaxPool(ceil) → `32×9×5` |
| `scalar_mlp` | 48,320 | 754 scalars → 64 |
| `lstm` | **1,804,288** | `LSTMCell(1504, 256)`, stepped manually |
| `card_head` | 1,285 | 256 → 5 (4 hand slots + no-op) |
| `place_ctx` + `place_up` | 15,073 | ctx→32ch, broadcast-add, 2× deconv → 612 |
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
| tower/building HP | `W_BLDG = 0.5` | **potential-based**: `γΦ(s′) − Φ(s)` |
| troop HP | `W_TROOPS = 0.1` | delta |
| elixir trade | `W_ELIXIR_TRADE = 0.03` | delta |
| tower destroyed | `W_TOWER_DESTROYED = 0.6` | **deliberately NOT** PBRS |
| elixir overflow | `W_ELIXIR_OVERFLOW = 0.1` above 9.0 | per-step penalty |
| draw | `DRAW_PENALTY = 1.0` | terminal |

The PBRS form keeps the `γ` — Ng et al.'s policy-invariance result requires it,
and dropping it is a different (biased) shaping that looks almost identical.

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

1000-episode bursts every 10,000 episodes of **pipeline 2's own** counter (which
starts at 0 at handoff, not continuing phase 1's), first at its ep 10,000.
Measured 3.4 s/episode → ~57 min per burst, a ~14% throughput tax.

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

---

## Measured baselines — use these, don't re-derive them

**Throughput** (i5-13420H, CPU-only, `num_envs = 8`):

| | ep/hour | updates/hour |
|---|---|---|
| before the batched update | 1,167 | 36 |
| **current** | **2,873** | **92** |

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
which is the plateau signature the exploiter is meant to break.

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
