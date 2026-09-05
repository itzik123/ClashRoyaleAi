# Upstream requests — simulator changes being asked for

> **This file was deliberately emptied by the maintainer** once every
> historical item in it had been fixed upstream — it is not lost data, and it
> does not need recovering. `CLAUDE.md` still cites items 1-2, 9, 13, 22 and
> 23C by number as the evidence behind landed decisions (the arena re-centring,
> the movement-speed fix, snapshot/deepCopy, the live-mirror state setters);
> those citations now resolve to nothing here, which is expected, because the
> decisions they justified are already in the engine. Numbering resumes at 24.

Each item states the measured evidence, the exact proposed change, and the
blast radius. Per `CLAUDE.md`, nothing here is applied to `include/` or `src/`
without the human confirming the diagnosis and the edit.

---

## Item 24 — the opponent's played cards are observable and are being discarded

**Status: IMPLEMENTED 2026-08-27**, on the maintainer's explicit sign-off
lifting the C++ read-only rule for this item. **Class:** observation content.
**Raised:** 2026-08-27, Phase 3 RL audit.

Option B shipped. `observation_size()` 13606 -> **13976**; `CYCLE_START` is
13606 exactly, which is the check that the blocks are a pure append and every
pre-existing index is unmoved. See "What actually shipped" at the end of this
item for the parts that differ from the proposal.

### The gap

`ClashEnv::extractObservationForTeam` encodes the agent's own hand — costs plus
a 185-wide one-hot of card identity per slot — and, for the opponent, exactly
one number: `elixirSpent(1 - team) / MAX_MATCH_ELIXIR`.

The encoder's own comment is what makes this worth raising, because it gets the
principle exactly right and then keeps the wrong half of it:

> ELIXIR SPENT, both sides, cumulative this match. Deliberately the SPEND and
> not the opponent's current elixir: spend is what a human can actually observe
> (**you see every card they play and you know what it costs**), current elixir
> is hidden information.

"You see every card they play" is the correct observability test, and by it the
**card identities are observable** — a human watching the same match knows the
opponent has shown Hog, Musketeer and Fireball, and that their Fireball cannot
be back for another ~20 seconds. The encoder reduces that to a scalar sum of
costs, which is the one summary that destroys exactly the information cycle
tracking needs.

`CLAUDE.md`'s open problem #3 already lists "no card-cycle tracking (which of
the opponent's 8 cards are available) — a core human skill". This item is the
concrete proposal for it, plus the measurement showing the current substitute
does not stand in for it.

### Evidence

**1. The auxiliary head that is supposed to carry opponent modelling requires
no opponent modelling.** `python_ai/tests/test_aux_task_is_not_a_memory_probe.py`
measures it. The aux target (opponent's current elixir) is an affine function
of two scalars already in the observation — `time` and `opponent elixir spent`
— because `elixir(t) = start + rate*t - spent(t)`:

| predictor | MAE |
|---|---|
| predict-the-mean | 1.4271 |
| time only | 1.4223 |
| opponent elixir spent only | 1.4268 |
| **time + opp_spent, ordinary least squares** | **0.0000** |

2,606 samples, 8 episodes. Four parameters and no recurrence solve it exactly.
So `Aux/OppElixir_MAE` cannot be read as "the recurrent state genuinely counts"
the way `CLAUDE.md` currently instructs — a stateless linear map clears the
documented ~1.3 threshold by more than an order of magnitude, and the trained
head's 0.77-0.83 is a *shortfall* against an achievable zero rather than a
capability. The one auxiliary task in the network that is nominally about
modelling the opponent is arithmetic on two present scalars.

**2. The agent cannot learn the cycle from the spawn sequence instead.** In
principle the LSTM could infer the cycle by remembering which enemy units have
appeared. Two measured obstacles:

- `bptt_chunk = 25`, so the gradient that would teach the hidden state to carry
  something only spans **25 decisions**. A full 8-card cycle at typical play is
  ~20-30 decisions, i.e. the learning horizon sits *exactly at* one cycle
  length — the regime where cross-cycle structure is hardest to acquire. (The
  hidden state is carried forward at inference; it is the credit assignment
  that is truncated, which is the half that matters for whether the behaviour
  is ever learned.)
- Spawns are not a clean signal of card identity anyway: Evolutions reuse the
  base card's name and id space is shared, and several cards spawn units whose
  identity does not name the card played.

**3. The information is free at the source.** `GameManager` already knows every
card each side plays — `PlayerState` cycles a real deck. Nothing has to be
inferred; this is a question of what gets copied into the observation vector.

### Proposed change

Two options. **B is the recommendation**, because A's blast radius is severe
and mostly unnecessary.

**Option A — full per-slot one-hot of the opponent's known hand.**
Mirror the agent's own encoding: `HAND_SIZE * NUM_CARD_IDS` extra floats.
Rejected: +740 floats, and it hands the agent the opponent's *current hand*,
which is genuinely hidden information — the same mistake the encoder's comment
correctly refuses for elixir. It would train a policy that cannot be deployed
through `perception/`.

**Option B — a "seen" vector plus a recency channel, both observable.**
Append `2 * NUM_CARD_IDS` floats to the extra-scalar tail:

```
seen[c]    1.0 if the opponent has played card c at least once this match
recency[c] exp(-(currentTick - lastPlayedTick[c]) / RECENCY_TAU), else 0.0
```

Both are computable by a human watching the screen and by `perception/`, which
is the deployability test this project already applies. `seen` is deck
discovery ("they have shown 6 of 8"); `recency` is cycle position ("their
Fireball went 4 seconds ago"), decaying rather than a raw timestamp so it needs
no normalization against match length and degrades gracefully.

`RECENCY_TAU` should be one cycle, ~200 ticks (20 s), and belongs in
`ArenaLayout.h` or alongside `ELIXIR_REGEN_RATE` — **not** duplicated in
Python; expose it through the bindings the way `ARENA_*` already is.

### Blast radius

- **`observation_size()` changes**, 13606 -> 13606 + 2*185 = **13976**. That is
  architecturally fatal to every existing checkpoint, exactly as the
  2026-07-29 9 -> 21 channel change was (`python_ai/archive_pre_obs_v3/`).
  `bc_pretrain.load_dataset` already refuses a mismatched dataset, which is the
  correct behaviour and will fire.
- `MicroRoyaleNet.scalar_size` is derived from the bindings, so the Python side
  needs no edit — but `NUM_EXTRA_SCALARS` is currently `9` and the two new
  blocks are per-card, not scalars, so the layout constant and its consumers
  (`models/net.py`'s `extra_start`, `perception_encoder.py`) must be checked
  rather than assumed.
- **Not gameplay-affecting**: it adds observation content and touches no
  simulation rule, so win rates remain comparable *in principle* — but only
  against a policy retrained from scratch, which is a full phase 1.
- Cost per step is two array writes on a card play and a 370-float append per
  observation; the encoder is already 25x faster since the 2026-08-26 perf work,
  so this is not a throughput concern.

### What would make this worth the retrain

State the prediction before running it, per this project's own rule on optional
stopping. If the cycle information is load-bearing, the cheapest confirmation
is not a win rate but a **new auxiliary task**: ask the head to predict *which
card the opponent plays next*. Unlike the elixir head this is not solvable from
present scalars, so its accuracy against a marginal baseline (always guess their
most-played card) is a direct read on whether the representation carries cycle
structure at all. Run that before spending a phase 1 on the win-rate question.

**This remains UNMEASURED.** The feature is implemented and tested; whether it
raises the learning ceiling is a training-run question and nothing here claims
it does.

### What actually shipped, where it differs from the proposal

- **`CARD_ID_COUNT` moved to `CardRegistry.h`.** `NUM_CARD_IDS` was declared
  inside `ClashEnv`, which `GameManager` cannot include -- so sizing the
  tracking table there would have meant a second literal `185`. The constant
  now lives next to the ids it bounds and `ClashEnv::NUM_CARD_IDS` is an alias;
  the Python binding is unchanged.

- **The hook is `GameManager::playCard`, not `ClashEnv`.** `HeuristicOpponent`
  reaches `playCard` directly rather than through `ClashEnv::step`, so hooking
  the env would have missed every opponent play in phase 1. Same shape as the
  five damage entry points that made `Tower::awake` latch on an HP invariant.

- **`ClashEnv::notePlayedCard` was added, and `inject` deliberately does NOT
  call it.** `inject` places a BODY and bypasses `playCard` by design (a
  mirrored unit must not cost the mirror elixir), so without a separate
  recorder the feature would work in training and silently do nothing in
  deployment through `perception/`. Keeping them separate also stops scenario
  setup from asserting "they just played this" when it only placed a unit.

- **Item 25 was implemented at the same time, because item 24 detonated it.**
  Five Python call sites located the extra-scalar tail as
  `observation_size() - NUM_EXTRA_SCALARS`. That is correct only while the
  extra scalars are last, and appending the cycle blocks behind them made every
  one read card-recency floats instead -- including two reading
  `enemy_tower_hp`, which feeds `compute_shaping`'s tower potential, so the
  REWARD would have gone quietly wrong with nothing raising. `EXTRA_SCALARS_START`
  and `CYCLE_START` are now bound forward offsets and `observationSize()` is
  derived from them, so the size and the offsets cannot disagree.

- **Mirror records its own id (164), not the duplicated card's.** What left
  the hand is what has to cycle back, even though the unit an observer SEES is
  the mirrored one.

### Tests

`tests/core/test_card_cycle_observation.cpp` (9 cases) pins the offset, the
opponent-not-self direction, the decay constant, the seen/recency distinction,
a rewound clock not pushing recency above 1, snapshot inheritance, and reset
clearing. `python_ai/tests/test_opponent_cycle_observation.py` (8 cases) pins
the Python side, including the regression that would have been silent: the
tower-HP scalars must still read as tower HP, asserted on a property the wrong
region cannot have (all six are strictly positive on a fresh board, while every
cycle float there is exactly zero).

---

## Item 25 — `NUM_EXTRA_SCALARS` order is positional and undocumented downstream

**Status: IMPLEMENTED 2026-08-27**, alongside item 24 -- which is what turned
this from a tidy-up into a live defect. Raised as "minor"; it was not.

The nine appended scalars are `[time, spent_self, spent_opp, towerHp[0][0..2],
towerHp[1][0..2]]`, and consumers index them by offset from the end of the
observation. Nothing binds that order to a name, so inserting a scalar anywhere
but the end silently re-points every consumer — including
`python_ai/tests/test_aux_task_is_not_a_memory_probe.py`, which reads
`-N_EXTRA + 0` and `-N_EXTRA + 2` and would keep passing while measuring the
wrong columns.

Proposal: bind named offsets (`EXTRA_TIME`, `EXTRA_SPENT_SELF`,
`EXTRA_SPENT_OPP`, `EXTRA_TOWER_HP_BASE`) through `src/bindings.cpp` the way
`CH_*` already is, and have Python derive from them. Cheap, and it removes a
whole class of silent-drift bug before item 24 adds two more blocks to the tail.

---

## Item 26 — the match never leaves single elixir, and it caps two cards' EV

**Status: IMPLEMENTED 2026-09-02.** GAMEPLAY-AFFECTING; see Blast radius.

### The gap

`GameManager::step` regenerated elixir at a flat `ELIXIR_REGEN_RATE = 0.035`
per tick for the whole match. Real Clash Royale runs three phases -- single,
double from 2:00, triple from 3:00 (overtime) -- so the simulator's economy was
the opening two minutes of a real match, stretched over all six.

The hypothesis this was raised under: with no double/triple phase the board
never carries a large simultaneous push, so Fireball and Cannon never reach
positive EV and a cheap-cycle policy dominates by construction.

### Evidence

Measured 2026-09-02 on `model_weights_phase4.pth` (ep 32,484) at teacher stage
3, 24 episodes / 4,229 decisions, sampled (on-policy), before any change.

**Elixir is not husbanded, it is STARVED.** This is the finding that carries
the item:

| | mean | median | P(>= 9.0) | P(<= 4.0) |
|---|---|---|---|---|
| agent | 2.16 | 1.95 | **0.0%** | **91.0%** |
| teacher | 1.81 | 1.45 | **0.0%** | 91.3% |

Zero overflow in 4,229 decisions -- `W_ELIXIR_OVERFLOW` never fires. Both sides
spend every drop the tick it arrives, so income really is the binding
constraint on how much can be on the board at once, and added income will be
SPENT rather than discarded. (Had the bars been pooling near 10.0 the whole
proposal would be refuted: more income would then be waste, and the constraint
would have been decision-making instead.)

**And the board shows it.** Best-case Fireball catch, computed as an upper
bound -- perfect information, the best of all 612 placement centres, 2.5-tile
radius from `CardRegistry.h:837`:

```
enemy units on board   mean 3.62   median 3   max 9
BEST fireball catch    mean 1.51   median 1   max 5
  P(catch >= 3 units) = 15.5%
  P(catch >= 4 units) =  6.1%
```

A 4-elixir spell whose best possible play usually catches ONE unit cannot be
+EV, and the policy declining to play it is correct behaviour against the
environment as it stood, not a training failure.

### What the measurement REFUTED, and why the schedule is not the one proposed

The proposal as raised put double at 2:00-4:00 and triple at 4:00-5:00. Match
lengths say the triple phase would have been nearly inert there:

```
match end tick   mean 1758 (2:56)   median 1710   max 3556
reached 2:00 (tick 1200):  92% of matches
reached 4:00 (tick 2400):   8% of matches
share of all ticks played that fall after 2:00:  32.1%
share of all ticks played that fall after 4:00:   3.0%
```

Double elixir touches a THIRD of all gameplay. Triple at 4:00 touches 3% -- a
whole code path, an observation value and a full retrain for something the
agent would see in the last seconds of one match in twelve. Shipped with the
REAL game's schedule instead (double 2:00, triple 3:00), which puts triple at
~8% of ticks: still small, but 2.7x the proposed placement and correct as
fidelity rather than a compromise.

### The link that WAS unmeasured -- now measured, and it holds modestly

"More elixir" -> "bigger clusters" was assumed when this was raised. Measured
after the change, same policy, same stage 3, same 24 episodes (4,162 decisions
against 4,229 before) so that the ONLY variable is the engine:

| | flat 1x | 1x/2x/3x |
|---|---|---|
| enemy units on board, mean | 3.62 | **3.89** |
| best Fireball catch, mean | 1.51 | **1.71** |
| best Fireball catch, **median** | **1** | **1** |
| P(catch >= 2) | 28.5% | **36.7%** |
| P(catch >= 3) | 15.5% | **21.8%** |
| P(catch >= 4) | 6.1% | **10.4%** |
| best catch, max | 5 | **7** |
| agent elixir, mean | 2.16 | **2.50** |
| agent P(>= 9.0) | 0.0% | **0.6%** |

**The direction is right and the size is modest.** P(catch >= 3) rises 41% in
relative terms and P(catch >= 4) by 70%, so the improvement is real but lives
in the TAIL -- the median best-case Fireball still catches exactly ONE unit.
Anyone expecting Fireball to become obviously correct should read that median
first.

Two things this does NOT say. It is the SAME policy throughout, trained under
flat elixir and never taught to exploit double, so this measures whether the
ENVIRONMENT produces more clusters at unchanged behaviour -- the right
controlled question, but not what a retrained policy would do. And
`DEFAULT_DECK` still bounds it: 2.6 Hog Cycle is almost entirely single-body
cards (only Skeletons gives 3), so no amount of elixir makes a swarm in a
mirror. What double elixir actually buys Fireball is MORE 4-COST SUPPORT ALIVE
AT ONCE -- killing a Musketeer is an even trade plus tower chip. Read a
post-retrain Fireball number against that claim, not against "swarms now
exist".

**Overflow is now possible for the first time.** `P(>= 9.0)` moved 0.0% ->
0.6% for the agent, so `W_ELIXIR_OVERFLOW` starts firing where it never had.
That also broke the premise of
`python_ai/tests/test_aux_task_is_not_a_memory_probe.py` -- see below.

### An unplanned consequence: the old aux task is no longer trivial

That test file documented (2026-08-27) that the deleted `Aux/OppElixir_MAE`
head measured nothing, because the opponent's hidden elixir is an affine
function of two scalars already in the observation: `start + rate*t - spent`,
recoverable at **MAE 0.0000**. Its closing caveat said the relation holds only
while nothing clamps at the 10 cap, and that a policy baiting the opponent into
overflow "would make the task genuinely non-trivial".

Nothing baited anyone; tripling the income was enough. Both halves moved:

- `rate * t` is no longer a single slope, so the ORIGINAL basis now scores
  **1.4265** -- no better than predict-the-mean. Integrating the schedule
  (`[income(t), spent, 1]`) restores the exact fit.
- The opponent now **overflows unaided**, and the cap discards elixir no scalar
  records. Per episode against the exact analytic model: the three that never
  capped residual at **-0.035** (one tick of regen, i.e. noise); the three that
  did carry **+0.76, +1.63 and +5.38**.

The file's conclusion survives in the overflow-free regime and its basis was
updated; two tests were added pinning both new facts. A rising MAE under phases
is an OVERFLOW detector, and still not a memory diagnostic.

### What shipped

- `GameManager::DOUBLE_ELIXIR_TICK` (1200) / `TRIPLE_ELIXIR_TICK` (1800) /
  `MAX_ELIXIR_MULTIPLIER` (3.0), and `elixirMultiplierAtTick(tick)` as a public
  static -- one definition, since ClashEnv, the bindings, GameLogger, the
  viewer and `perception_encoder.py` all need it.
- `step()` reads the phase ONCE per tick and hands the same value to both
  players. The phase COMPOSES with `oppElixirMultiplier` rather than replacing
  it, so a 1.5x curriculum opponent in double elixir gets 3.0x.
- `NUM_EXTRA_SCALARS` 9 -> 10; scalar 9 is the multiplier / 3.0. Observable by
  the same test the cycle blocks are argued from -- a human sees the "2x
  ELIXIR" banner. Symmetric across teams, unlike the tower block.
- **`MAX_MATCH_ELIXIR` 140 -> 280.** Latent defect the change would have
  tripped: 140 was sized against a flat-1x match (3600 x 0.035 + 5 = 131).
  Phased income reaches ~278, so BOTH spend scalars would have saturated at 1.0
  part way through every match and stayed there -- going blind exactly when the
  economy decides the game, with nothing raising.
- `elixirPhases` block in the replay JSON, and the viewer's `x1/x2/x3` badge
  reading it. The viewer is the structurally-forced-to-restate case, so the fix
  belongs in the FORMAT (same argument as `cardMeta` and `rollWidth`).
- `python_ai/tools/migrate_checkpoint_elixir_phase.py` -- zero-pads
  `scalar_mlp.extra` from `Linear(9, 12)` to `Linear(10, 12)` AND the matching
  Adam moments, which is half the job and is what would otherwise throw on
  resume.

### Blast radius

**GAMEPLAY-AFFECTING, and about as wide as it gets.** The economy is the
substrate every card's value sits on: relative card value, the correct number
of cards to hold, when a push is affordable, and therefore every win rate in
CLAUDE.md's "Measured baselines". The curriculum's win-rate gates are
calibrated against a teacher that now plays a materially different late game.

**Every checkpoint needs migrating**, though cheaply -- 120 of 1,900,165
parameters. `observation_size()` 13976 -> **13977** and `CYCLE_START`
13606 -> **13607**; both are derived everywhere that matters, and the two
deliberate tripwires (`tests/core/test_clash_env.cpp`,
`tools/audit/verify_pyd.py`) were updated as the acknowledgement.

### Tests

- `tests/core/test_game_manager.cpp` `[elixir_phase]` -- three cases: the
  schedule on BOTH sides of each boundary, income MEASURED off the bar (not
  re-asserting the schedule function), and composition with
  `oppElixirMultiplier`. The mid-window crossing case pins per-TICK evaluation;
  a schedule sampled once per `step()` passes every other assertion.
- `tools/audit/elixir_phase_audit.cpp` -- the same measurements plus the
  observation layout, runnable without the .pyd. Written because a training run
  held the .pyd mapped while this was authored, so the ordinary build was
  unavailable; it is the reason the C++ was verified at all before landing.
- `tools/audit/verify_pyd.py` now checks the phase scalar is present, at the
  index the layout claims, and symmetric across teams.

---

## Item 27 — Graveyard spawns ONE skeleton and deals zero tower damage from any cell

**Status: OPEN, proposal only. Class:** card behaviour (`CardRegistry.h`).
**Found** 2026-09-06 while auditing whether the UtilityTeacher can pilot every
deck in the phase-1 pool.

### The measurement

Graveyard (card id 110, cost 5, `is_spell`) was swept over **all 588 of its
legal cells** on an empty board, both sides no-op, 1200 ticks per cell, scored
as enemy Princess/King Tower HP **actually lost**:

```
Mortar          34 / 170 legal cells damage the enemy tower   best 1596
X-Bow           34 / 170                                      best 3824
Goblin Barrel  588 / 588                                      best 1320
Graveyard        0 / 588                                      best    0
```

Body count on an empty board, sampled every 10 ticks after the cast:

```
Goblin Barrel   3, 3, 2, 1, 1, 0, 0 ...      (three goblins, ~50 ticks)
Skeletons       3, 0, 0 ...                  (the 1-cost card, for scale)
Graveyard       1, 1, 0, 0, 0, 0, 0 ...      ONE body, dead inside 20 ticks
```

Reproduced through the real play path as well as through `inject` --
`set_hand_for_team` + `step_self_play(slot, 3.0, 27.0, ...)` on the enemy
Princess Tower gives the same one body and the same **0** tower damage at
t = 20, 40, 70, 120, 220, 420, 820 and 1210 ticks.

### Why it matters

`graveyard_control` is one of the sixteen phase-1 opponent decks, so a 5-elixir
card that does nothing makes it a **seven-card deck**. That is not a small
handicap and it contaminates a measurement the project has already leaned on:
the agent's 0.925 greedy win rate against `graveyard_control` (2026-09-05,
40 episodes) reads as strength and is substantially the opponent being a card
down. It is the softest deck in the pool on the passive-opponent probe both
before and after the teacher was fixed (4,323 tower damage against a pool median
of ~7,500, and the only deck still needing 2,520 ticks to close against an
opponent doing nothing).

In the real game Graveyard spawns roughly 15-20 skeletons over ~10 seconds
inside a radius; one body that dies immediately is not a weaker version of that,
it is a different card.

### What is NOT being claimed

The exact spawn count, interval and radius are **not** measured here and no
number is proposed for them — the probe establishes only that the current
behaviour (one body, once) cannot be right for a 5-elixir win condition, not
what the right figures are. Whoever fixes it should take the count/interval from
the same catalogue the sight-range and speed-tier reworks used.

### Blast radius

**GAMEPLAY-AFFECTING.** `graveyard_control` becomes materially stronger, so
every win rate measured against that deck is invalidated — including the 0.925
above and whatever the current run records before a fix lands. Nothing else in
the pool plays Graveyard, and `DEFAULT_DECK` does not, so the agent's own
behaviour and every checkpoint are untouched. No observation or action-space
change; no checkpoint migration.

### Interim handling

None. The deck is **left enabled**: with the win-rate floor now off
(`deck_pool.POOL_WINRATE_FLOOR`, 2026-09-06) PFSP already parks a deck the agent
beats at the 0.05 minimum weight, so `graveyard_control` costs ~0.5% of episodes
and disabling it would buy nothing while hiding the defect.
