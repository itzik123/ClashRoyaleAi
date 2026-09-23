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
radius from Fireball's `CardRegistry.h` entry:

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

## Item 27 — Graveyard's spawn CADENCE exactly cancelled a tower's fire rate

**Status: FIXED 2026-09-06** on the maintainer's explicit instruction.
**Class:** card balance data (`CardRegistry.h`).

### The first diagnosis here was WRONG, and how it was wrong is the useful part

This item originally read "Graveyard spawns ONE skeleton and deals zero tower
damage from any cell", and concluded the spawn machinery was broken. The zero
was real and reproducible; the explanation was not.

The probe counted **instantaneous** bodies on the board next to an enemy
Princess Tower, and read 1. Re-run where nothing can kill the skeletons -- our
own back corner -- the count climbs 1,2,3,...,**9** and stops, which is exactly
what `withRepeats(9, 10)` asks for. The machinery was always fine.

What the original probe actually measured was a **kill rate equal to the spawn
rate**. One 81-hp Skeleton per 10 ticks meets a Princess Tower firing once per
10 ticks, so each one dies before its successor lands and the standing
population is permanently 1. Zero of them ever survive long enough to attack,
which is why all 588 legal cells scored 0.

**A saturating measurement again** -- CLAUDE.md's own rule, "when a
measurement's failure mode is maximal permissiveness it needs an internal
control that MUST fire". Here the failure mode was maximal *suppression*: an
instantaneous count cannot distinguish "nothing spawned" from "everything
spawned and died on schedule". The control that settles it is a board where
death is impossible, and it costs one line.

### The real defect, and the fix

The cadence, not the mechanism. Published card: one Skeleton every **0.5 s**,
totalling **12** since the 2026-01-06 balance change. The registry had one per
**1.0 s**, totalling 9 -- half the arrival rate, which is precisely the rate at
which a tower deletes them one for one.

    .withRepeats(9, 10)   ->   .withRepeats(12, 5)

Arrivals now outrun a tower's fire rate 2:1, which is the mechanic that makes
the card work in the real game.

### Blast radius

**GAMEPLAY-AFFECTING.** `graveyard_control` goes from effectively a seven-card
deck to a real one, so every win rate measured against it is invalid -- in
particular the agent's 0.925 (2026-09-05, 40 greedy episodes), which was largely
the opponent being a card down. Nothing else in the pool plays Graveyard and
`DEFAULT_DECK` does not, so the agent's own action space and every checkpoint
are untouched.

### Not changed, and deliberately

The deploy delay is 8 ticks against the real card's 2.2 s (22 ticks), so the
engine's Graveyard starts sooner than it should. Left alone: it is a separate
question from the cadence, it moves the card in the opposite direction, and one
gameplay change at a time is how this repo keeps win rates attributable.

---

## Item 28 — damage to a SPAWNED body was booked as TOWER damage

**Status: IMPLEMENTED 2026-09-15**, in the pre-launch audit for the from-scratch
run, under the maintainer's standing instruction for that audit to use judgement
rather than stop for sign-off. **Class:** statistics / reward input.
**NOT gameplay-affecting** (no entity behaves differently). **Reward-affecting**:
the agent's tower potential and the teacher's rollout score both read this stat.

### The gap

`DamageByTargetTypeCollector::onDamageDealt` (`include/core/stats/StatsCollectors.h`)
classified a target as a Tower when `CardRegistry::getCard(e.targetCardId) ==
nullptr`. Towers carry unregistered sentinel ids (`TOWER_KING_ID = -2`,
`TOWER_PRINCESS_ID = -3`) — and so do spawned helper bodies (`-1`, `-10 .. -48`:
Goblin Barrel's goblins, a Graveyard's skeletons, Goblin Gang, a Battle Ram's
Barbarians). A tower shooting those bodies was booked as its owner dealing TOWER
damage. The same shape as the Mortar/King symbol clash: a discriminator that
something else can also wear.

### Evidence

Audit 05's `t5_towerstat_probe.py`: a team-0 card injected in front of the team-1
left Princess, both sides idle 300 ticks, team 0's towers provably untouched.

| team-0 card | `get_tower_damage_dealt(1)` BEFORE | AFTER | team-0 tower HP lost |
|---|---|---|---|
| Hog Rider / Knight / Giant / Musketeer / Skeletons / Barbarians / Minions / Lava Hound / Golem / Miner / Balloon (controls) | 0 | 0 | 0 |
| **Goblin Gang** | **540** | 0 | 0 |
| **Goblin Barrel** | **810** | 0 | 0 |
| **Graveyard** | **1080** | 0 | 0 |
| **Battle Ram** (death-spawned Barbarians) | **1440** | 0 | 0 |

After the fix that damage appears under `get_troop_damage_dealt(1)` instead
(Goblin Barrel 810, Graveyard 1080, Battle Ram 2160 including the Ram itself).

Consequences, both measured by audit 05:

* **the agent's reward** — `rewards/shaping.py`'s tower potential reads
  `team0_tower_damage - team1_tower_damage`, so the agent was PAID `W_BLDG` for
  its towers killing an opponent's Goblin Barrel and CHARGED when its own spawns
  were shot. Five of the sixteen opponent pool decks field such cards.
* **the teacher** — `rollout_stats` reads `tower_taken =
  get_tower_damage_dealt(opp)`, so every attack whose bodies are spawned was
  scored as the teacher's own tower being hit (`def -12.96` for a Goblin Barrel
  with no counter in the rollout). This alone froze `graveyard_control` at the
  top rung (1/0/1/3 crowns against a do-nothing opponent -> 3/3/3/3 with
  tower HP read directly).

### What shipped

`DamageDealtEvent` gains `bool targetIsTower = false`, last and defaulted so an
8-value aggregate init still compiles and still means "not a tower". Every one of
the 12 production emit sites stamps it from the target's own `isTower()`. The
collector classifies on the flag; an unregistered NON-tower falls through to troop
damage, with `def` null-checked (the first build dereferenced it and segfaulted on
the first spawned body — caught by the suite before anything else ran).

### Blast radius

* `test_match_statistics.cpp`'s King-tower case built its event with a bare `-2`
  and relied on registry absence; it now passes `targetIsTower = true`. It was a
  test double wearing the discriminator.
* Buildings-vs-troops classification is unchanged (still the registry's
  `isBuilding`), so `buildingDamageDealt` and every deployed-building number keep
  their meaning. Only tower damage for spawned targets moved.
* Every win rate earned against spawner decks, and every teacher score involving a
  spawned attacker, was earned on the wrong stat. Checkpoints still load.

### Not fixed, recorded

* **Last-hit overkill** is still booked in full (a Giant deals 3,795 "tower
  damage" to remove 3,546 HP). Small, and it only inflates the final hit on a
  dying tower.
* **`ElixirValueKilledCollector` credits zero elixir for a spawned body** (it
  skips unregistered victims), so a Fireball clearing a Goblin Gang earns no
  value. A valuation approximation, not a sign error.

### Tests

`tests/core/test_clash_env.cpp` `[stats][tower_damage][spawned]`: four spawners,
each asserting zero booked tower damage WITH the precondition that team 0's towers
were genuinely untouched, plus a CONTROL that a real tower hit is still booked.
`tests/core/test_match_statistics.cpp`: an unregistered target is a troop unless
the event says tower. Suite: 714 cases, 713 passed, 1 failed as expected, exit 0.

---

## Item 29 — spells hit a Crown Tower for 100% of their damage; the real game charges 15-30%

**Status: PROPOSED 2026-09-23 — NOT applied.** **Class:** gameplay fidelity.
**GAMEPLAY-AFFECTING.** The next run starts from scratch, so checkpoint
invalidation is free *now* and never again at this price.

### The gap

The real game gives every damaging spell a separate **Crown Tower damage**
statistic, a published fraction of its troop damage. This engine has the
mechanism — `AreaSpell::spellTowerDamageMultiplier`, applied in both the disc
branch and the rolling-sweep branch of `AreaSpell.h` — and exactly **one**
caller sets it: Hero Ice Golem's Snowstorm, at 0.05. `CardFactories::spawnSpell`
never passes it, so every played spell card hits a tower at the default 1.0.

The registry's TROOP numbers match the real game's current values almost
exactly (Fireball 689, Rocket 1485, Zap 192), so the registry was built from
the right source. The Crown Tower column is the part that never came across.
Nothing in the repo mentioned it (a grep of every `.md` for "crown tower
damage" returned nothing before this item).

### Evidence

**Engine**, measured by injection on 2026-09-16 and reproduced by
`card_probes.spell_tower_damage`: each spell cast by team 0 on the centre of
team 1's left Princess Tower, the tower's own HP loss read after the effect
settles, against a stationary P.E.K.K.A. for the troop number. Every
direct-damage spell reads tower / troop = **1.00**.

**Real game**, [DeckShop's spell damage chart](https://www.deckshop.pro/card/damage),
friendly level 11 (the level whose troop numbers the registry carries):

| spell | engine troop | engine vs tower | **real vs tower** | engine / real |
|---|---|---|---|---|
| Fireball | 689 | 689 | **159** | 4.3x |
| Rocket | 1485 | 1485 | **371** | 4.0x |
| The Log | 269 | 269 | **41** | 6.6x |
| Poison | 736 (92 x 8) | 736 | **184** | 4.0x |
| Arrows | 369 (123 x 3) | 369 | **93** | 4.0x |
| Zap | 192 | 192 | **58** | 3.3x |
| Lightning | 1057 | 1057 | **286** | 3.7x |
| Giant Snowball | 179 | 179 | **54** | 3.3x |
| Earthquake | 246 (82 x 3) | 246 | **159** | 1.5x |

Four spells' TROOP damage already disagrees with the same source and should be
resolved before a tower value is attached: Tornado (engine 154 / source 84),
Goblin Curse (258 / 210), Vines (405 / 306), Void (1020 / 2088, tiered).

### What it does downstream — measured

* **A Princess Tower (2534) falls to 4 Fireballs or 2 Rockets.** At real Crown
  Tower damage it is 16 and 7.
* **The agent's reward pays ~4x for a spell on a tower.** Through the real
  `compute_shaping` (tower PBRS term, gamma 0.999): one Rocket on a tower is
  **+0.185** in a single step — 18.5% of a win's terminal reward, guaranteed and
  undefendable — against +0.046 at the real value. Fireball +0.086 vs +0.020;
  The Log +0.034 vs +0.005.
* **The lethal-spell window is ~4x too wide.** "Tower HP <= my spell's damage"
  opens at 689 for Fireball, 27% of a Princess Tower; the real game's is 159,
  6%. (The Python side no longer depends on this being fixed: since 2026-09-16
  the window is keyed to the MEASURED tower damage,
  `card_probes.spell_tower_damage`, so it follows the engine either way.)
* **In actual play the effect is modest today, and concentrated in The Log.**
  48 teacher-vs-teacher matches at rung 4, pool decks on both sides, every
  spell actually cast attributed exactly (the same spell re-cast at the same
  board point on an empty board carrying the enemy towers' current HP; spawning
  spells excluded, since their tower damage is the body's):

  | spell | casts | hit a tower | tower damage | at the real ratio |
  |---|---|---|---|---|
  | The Log | 211 | **112** | 29,629 | 4,533 |
  | Poison | 44 | 4 | 2,784 | 696 |
  | Lightning | 11 | 1 | 1,057 | 286 |
  | Tornado | 42 | 5 | 924 | 297 |
  | Arrows | 20 | 2 | 738 | 188 |
  | Fireball | 166 | 1 | 689 | 159 |
  | Zap / Rocket | 59 / 13 | 0 / 0 | 0 | 0 |

  **7.0% of all tower damage** (35,821 of 514,759) is direct spell damage; at
  real ratios it would be **1.3%**. The teacher does not aim big spells at
  towers (1 Fireball in 166). But **53% of its Logs roll into a Princess
  Tower** — cast defensively at the river, a 10.1-tile roll reaches the tower
  on the way — for 269 each where the real card does 41.

* **What is NOT measured, and is the real risk: a learning agent.** The
  teacher is hand-written and does not look for chip; PPO will. An undefendable
  +0.185 per Rocket is exactly the kind of reliable payoff it finds first, and
  the habit it builds does not transfer to the real game `perception/` exists
  to play.

### Proposed change (exact)

Option **A (recommended): an absolute per-hit Crown Tower value, sourced per
card.**

1. `include/core/CardStats.h`, beside `spellKnockback`:

   ```cpp
   // Damage ONE hit of this spell deals to a Crown Tower (King or Princess).
   // -1 (the default) keeps the troop damage, i.e. today's behaviour. The real
   // game publishes this per spell (15-30% of troop damage); see
   // perception/UPSTREAM_REQUESTS.md item 29 for the source.
   int spellCrownTowerDamage = -1;
   CardStats& withCrownTowerDamage(int perHit) { spellCrownTowerDamage = perHit; return *this; }
   ```

2. `include/entities/AreaSpell.h`: a public member `int crownTowerDamage = -1;`,
   and in BOTH tower branches (the disc's `dealt = entity->isTower() ? ...` and
   the roll's twin):

   ```cpp
   int dealt = entity->isTower()
       ? (crownTowerDamage >= 0 ? crownTowerDamage
                                : static_cast<int>(effectiveDamage * spellTowerDamageMultiplier))
       : effectiveDamage;
   ```

   Set after construction in `CardFactories::spawnSpell`, beside
   `configureRoll`, for the reason that function already gives (the constructor
   is 23 parameters wide and shared with five non-card call sites):
   `spell->crownTowerDamage = stats.spellCrownTowerDamage;`.
   `AreaSpell::snapshot()` is the implicit copy, so the member rides along.

3. `include/core/CardRegistry.h`: per-hit values, the source's total divided by
   the registry's own hit count.

   | card | `.withCrownTowerDamage(...)` |
   |---|---|
   | Fireball (7) | 159 |
   | Rocket (30) | 371 |
   | The Log (33) | 41 |
   | Zap (29) | 58 |
   | Lightning (31) | 286 (per strike) |
   | Giant Snowball (100) | 54 |
   | Arrows (3) | 31 (x 3 waves = 93) |
   | Poison (32) | 23 (x 8 pulses = 184) |
   | Earthquake (103) | 53 (x 3 pulses = 159) |

   Evolutions carrying these spells need the same values.

Why an absolute value rather than a multiplier: the existing branch truncates
`static_cast<int>(damage * multiplier)`, so a multiplier derived as
`159.0f / 689.0f` depends on float rounding to land on 159 rather than 158. All
nine values above were checked in float32 and DO land exactly, so option B is
safe for them today; the absolute field is the one that cannot drift.

Option **B**: `withCrownTowerMultiplier(float)` feeding the existing
`spellTowerDamageMultiplier`. Smaller diff, no new AreaSpell member.

Option **C**: do nothing, and record the engine as a deliberately
spell-strong variant. Defensible only if sim-to-real transfer is not a goal.

### Blast radius

* **Gameplay-affecting for every deck with a damage spell**, the 2.6 deck's
  Fireball and Log among them. Every win rate in `CLAUDE.md` was earned with
  ~4x spell chip. No observation or action-space change, so checkpoints LOAD;
  it is the win-rate history that stops being comparable.
* **Python needs NO change.** `card_probes.spell_tower_damage` measures the
  tower, so the lethal window, `validate_deck`'s report and
  `test_damage_spell_is_deck_derived` follow the engine automatically. The
  teacher's rollouts score tower HP read from the engine, so they re-price on
  their own.
* **The Log** changes most (6.6x) and is in `SHIPPED_DECK`.
* `tests/entities/test_area_spell.cpp` pins The Log reaching the tower from the
  bridge for 269 against `ArenaLayout`; it would read 41. The REACH assertion
  is the load-bearing half and is unaffected.
* `verify_pyd.py` and any C++ test asserting a spell's tower damage will move.

### Tests (to add with the change)

* Parameterised over the nine cards: cast on a Princess Tower centre, assert HP
  lost == the table's total. Plus the CONTROL that must not move: the same cast
  on a stationary P.E.K.K.A. still loses the full troop damage.
* The King Tower takes the same Crown Tower value (the real game does not
  distinguish King from Princess here).
* A spell with no value set is bit-identical to today (default -1).

---

## Item 30 — three spawned/secondary Spear-Goblin-type units cannot hit air

**Status: PROPOSED 2026-09-23 — NOT applied.** **Class:** registry data.
**GAMEPLAY-AFFECTING** for decks holding Goblin Gang, Rascals or Goblin Hut —
two pool decks (`dart_bait_cycle`, `classic_log_bait_inferno`) field Goblin Gang.

### The gap

The registry contradicts itself about one unit, the shape the 2026-08-26 speed
pass found for Bats and Goblins. The playable Spear Goblins (id 23) carry
`.withTargetsAir()`, and so do the ones riding a Goblin Giant (helper -26). Three
other copies do not:

| helper | where | `.withTargetsAir()` |
|---|---|---|
| -17 "Spear Goblins" | `goblinHutSpearGoblinStats()` (Goblin Hut's spawns) | **missing** |
| -27 "Spear Goblins" | Goblin Gang's inline `withSecondaryUnit(...)` | **missing** |
| -28 "Rascals" (the Girls) | Rascals' inline `withSecondaryUnit(...)` | **missing** |

In the real game all three are ranged units that hit air. The Rascal Girls
helper also carries a raw `0.5f` speed rather than a `SPEED_*` tier -- the exact
pattern the 2026-08-26 pass fixed for death-spawned children. Its pinning test
("every spawned unit moves at the speed of its own playable card") evidently does
not reach SECONDARY units; worth confirming when this is fixed.

### Evidence

Measured through the rebuilt `.pyd`, one enemy Balloon held stationary, the card
placed beside it, 50 s, against the same board without the card:

| card | damage to the Balloon | anti-air cells / occupied (obs channel 13) |
|---|---|---|
| Spear Goblins (control) | **567** | 2 / 2 |
| Goblin Giant (control: helper -26 has the flag) | **324** | 2 / 3 |
| **Goblin Gang** | **0** | 0 / 4 |
| **Rascals** | **0** | 0 / 2 |
| **Goblin Hut** | **0** | 0 / 1 |

Engine targeting itself is correct (`CombatEntity.h`'s target filter, `!entity->isFlying ||
targetsAir`): Knight, Skeletons, Ice Golem, Hog Rider and Cannon all deal 0 to a
held Balloon and Musketeer deals 1085. Only these three helpers' DATA is wrong.

### Proposed change (exact)

`include/core/CardRegistry.h`:

1. `goblinHutSpearGoblinStats()`: append `.withTargetsAir()`.
2. Goblin Gang's secondary `troop(-27, "Spear Goblins", ...)`: append
   `.withTargetsAir()` before `.withOffsets(...)` (order does not matter).
3. Rascals' secondary `troop(-28, "Rascals", ...)`: append `.withTargetsAir()`;
   and replace the raw `0.5f` speed with the Rascal Girls' tier from the same
   source the 2026-08-24 speed rework used (not guessed here).

Plus a GENERIC test beside the speed one in `tests/core/test_card_registry.cpp`:
*every spawned or secondary unit that shares a name with a playable card agrees
with that card on `targetsAir` (and speed)* -- generic, so the next helper cannot
reintroduce this. It must iterate secondary units as well as death / periodic /
spell spawns, since that is the family the speed test missed.

### Blast radius

* Goblin Gang and Rascals become real anti-air answers (the whole reason a bait
  deck carries Goblin Gang is its Spear Goblins); Goblin Hut starts defending air.
* The teacher needs no change: `card_probes.damages_air` MEASURES the card, so it
  re-classifies all three automatically once the engine is fixed.
* Win rates against `dart_bait_cycle` / `classic_log_bait_inferno` move.

### Night Witch: NOT a defect -- the probe was wrong (resolved 2026-09-23)

First recorded here as "her bats dealt 0 to a held Balloon, cause unknown". The
bats are fine: attributed by card id, her periodic Bats (helper -14) dealt **1215**
to a held Balloon and 1539 to a held Knight. The zero came from two flaws in the
PROBE, both worth knowing:

* **A saturating control.** With the Balloon held on our own half, our towers
  destroy it in BOTH arms, so "with minus without" read 0 while the bats were
  hitting it -- the maximal-suppression trap `CLAUDE.md` describes for Graveyard.
* **The attacker walked away.** Placed on the enemy half, Night Witch (who cannot
  target air) walked at the enemy tower and her bats spawned beside it, hitting
  the tower instead of the Balloon.

The three helpers above are unaffected: their evidence is the source (no
`.withTargetsAir()`) and the per-entity observation flag, not a with/without
difference.
