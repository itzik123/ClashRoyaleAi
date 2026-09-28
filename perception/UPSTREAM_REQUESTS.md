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

**Status: PARTIALLY IMPLEMENTED 2026-09-25.** Option A's mechanism landed with
item 32 (`CardStats::spellCrownTowerDamage` -> `AreaSpell::crownTowerDamage`,
used by both tower branches), and **only Poison** sets it: 21 per pulse, 168
in all, the current value from the addendum below. Every other spell in the
tables still deals 100% to towers and awaits sign-off. Item 32l (disc spells
now hit what their hitbox overlaps) makes that gap wider: a Fireball centred
up to 2.5 + 1.5 = 4.0 tiles from a Princess Tower now hits it. So the rest of
this item matters more now, not less. The half-state is already visible in
training: `pekka_bridge_spam`'s spell reward terms, ranked by measured tower
damage, now follow its Zap (192 to a tower; real 48) instead of its Poison
(168, real). **Class:** gameplay fidelity.
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

### Addendum 2026-09-25: the source table above is stale for 8 of its 9 spells

DeckShop's chart lagged the **official June 1, 2026 balance notes**, which cut
Crown Tower damage for almost every spell ("Crown Tower Damage Reduction for
Spells", [June notes](https://supercell.com/en/games/clashroyale/blog/release-notes/june-balance-changes-2026/)).
Fireball was then cut again on September 8
([September notes](https://supercell.com/en/games/clashroyale/blog/release-notes/september-balance-changes-2027/)),
and that is the ONE value DeckShop had current. That is how a partly-updated
chart passed for a current one. Use these values, level 11:

| spell | table above | **current** | source |
|---|---|---|---|
| Fireball | 159 | **159** | Sep 8 (was 207 -> 172 in June) |
| Rocket | 371 | **342** | June |
| The Log | 41 | **35** | June |
| Zap | 58 | **48** | June |
| Lightning | 286 | **265** per strike | June |
| Giant Snowball | 54 | **45** | June |
| Arrows | 93 | **75** total (25 per wave) | June |
| Poison | 184 | **168** total (**21** per pulse) | June; plus the Aug 26 fix "Poison now deals the correct tower damage" |
| Earthquake | 159 | **147** total (49 per pulse) | June |
| Freeze / Rage / Vines | -- | 37 / 45 / 70 | June |

Re-measured through `tools/audit/gy_deck_audit.cpp` (item 32): Poison cast on a
Princess Tower still deals **92 per pulse, 736 in total**, 4.4x the real 168.
That was before item 32 shipped; the same instrument now reads **21 a pulse,
168 in all**.

---

## Item 30 — three spawned/secondary Spear-Goblin-type units cannot hit air

**Status: PARTIALLY IMPLEMENTED 2026-09-25**, with item 32. Goblin Hut's helper
(-17) now has `.withTargetsAir()`, its 1.6 s hit speed and a 0.5 s deploy.
Tested: its goblins dealt 648 to a flying enemy that took 0 before. Every
Spear Goblin copy (23, -17, -26, -27) has the 1.6 s hit speed. **Still open:**
Goblin Gang's Spear Goblins (-27) and the Rascal Girls (-28) still cannot hit
air. Those were not in the sign-off for item 32. **Class:** registry data.
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

### Re-confirmed 2026-09-25, by a different instrument

`tools/audit/gy_deck_audit.cpp` (item 32), section 11: a stationary FLYING
enemy 4 tiles from a Goblin Hut for 10 s. The Hut spawned 4 Spear Goblins and
they dealt **0** to it, with all towers asleep so nothing else could act. This
item's fix is still the whole answer. Fold in two data changes to the same
helpers while they are open: Spear Goblins' hit speed is **1.6 s** since the
official August 4, 2026 notes (`attackCooldown` 17 -> 16, all four copies:
23, -17, -26, -27), and the Hut itself is at **1180** hp, not 1228 (item 32n).
Both shipped with item 32, together with the Hut's air fix.

---

## Item 31 — a unit on a bank line vibrates at the bridge mouth forever

**Status: IMPLEMENTED 2026-09-24**, on the maintainer's sign-off, exactly as
proposed below. `test_board.cpp` `[orbit]` stuck on 1,276 of 3,208 crossings
before the change and on 0 after. The C++ suite ran 715 cases, 1 expected
failure, exit 0. The rebuilt `.pyd` sticks on 0 of 21 bank-line starts (was
14 of 21). **Class:** movement / pathing. **GAMEPLAY-AFFECTING**, which voids
`model_weights.pth`'s win-rate history: every troop's last step onto a
waypoint changes. Found while rendering
promo footage (`tools/promo/mosaic.py`), where a Giant visibly shook in place
at a bridge for 22 seconds.

### The gap

`Troop::moveTowards` always moves a full `speed` step toward its waypoint and
never shortens the last one. The step lands past the waypoint unless it happens
to finish inside `WAYPOINT_ARRIVAL_EPS` (0.01).

That is harmless almost everywhere, because an overshoot normally carries the
unit into a new region and `getNextWaypoint` hands it the next point. It is
fatal on a **bank line**:

1. Something pushes a ground unit off a bridge deck while it is in the river
   band: knockback (Fireball, The Log) or a collision.
2. `Board::clampToBoard` snaps it to the nearer bank, so y is exactly
   `riverY_start` (15.5) or `riverY_end` (17.5).
3. The bank tests are inclusive, so the unit is still "below", and
   `getNextWaypoint` returns the bridge mouth `{bridgeX, riverY_start}`. The
   unit walks to it **sideways, along y = 15.5**.
4. It overshoots in x. y is still 15.5, so the region is unchanged and it gets
   the same waypoint back. It overshoots the other way, and repeats. This is a
   **period-2 orbit**: the unit alternates between d and (step - d) either
   side of the mouth, and neither is within 0.01.

The two earlier bridge-mouth fixes (2026-08-09, 2026-08-20) removed **fixed
points**, where the waypoint handed back was the point the unit already stood
on. This one is an orbit, so an "is the waypoint where I stand" guard cannot
see it. The escape is luck: the final step must land within 0.01 of the mouth,
which happens with probability about `2 * EPS / step`, roughly 20% for a
Giant.

### Evidence

**The reported case**, `match_060` from the promo cache (teacher vs teacher,
rung 10, giant_double_dragon vs rg_fisherman_cycle):

| tick | Giant (team 0) | |
|---|---|---|
| 280 | (14.56, 15.76) | on the right bridge deck, in the river band |
| 281 | (15.55, 15.50), hp 3968 -> 3279 | red Fireball at (13, 16): 689 damage plus knockback off the deck (deck is x in [13.5, 15.5]); clamp snaps y to 15.5 |
| 282-292 | x 15.45 -> 14.46 along y = 15.50 | walks sideways to the mouth (14.5, 15.5) |
| 293-516 | x flips 14.46 <-> 14.55 every tick | **224 ticks (22.4 s) stuck**, with nothing near it for the first 100 |
| 517 | (14.57, 15.50) -> (14.60, 15.66) | three friendly Skeletons walk into it and the collision nudge frees it |

**Prevalence**: across 128 teacher-vs-teacher matches (16 meta decks,
rung 10, 897,708 unit-ticks), there were **24 oscillation runs of 8+ ticks, in
23 matches (18%)**. The longest was a Lumberjack stuck for **524 ticks (52 s)**;
the next longest ran 224, 143, 123, 103 and 87 ticks. **Every one** was on a
bank line (14 at y = 15.5, 10 at y = 17.5), at both bridges and for both
teams. The detector counts consecutive ticks where
`pos(t) == pos(t-2) != pos(t-1)`. Runs of 1-2 are ordinary collision reversals
(241 of them); every run of 6+ ticks (26 of 26) was on a bank line, so the
split is clean.

**Empty-board reproduction** (fresh env, `inject(card, x, y, team, -1, 0)`,
both sides holding, no other units):

| start | result after 60-80 ticks |
|---|---|
| Giant (2), team 0, (15.55, 15.5) | stuck: x flips 14.46 <-> 14.56 at y = 15.5 |
| Giant, team 0, x0 = 15.50 .. 16.50 step 0.05 on y = 15.5 | **14 of 21 stuck** |
| Giant, team 1, (15.55, 17.5) | stuck at (14.56, 17.5): the mirror bank |
| Hog Rider (15), team 0, (15.55, 15.5) | escapes: its last step happens to land within 0.01 |

### Proposed change (exact)

`include/entities/Troop.h`, `Troop::moveTowards`: cap the step at the remaining
distance, so the unit lands **on** the waypoint (distance 0 <= EPS) and
`getNextWaypoint` hands it the far bank on the next tick:

```cpp
float currentSpeed = frozenThisTick ? speed * freezeSlow : speed;
// Land on the waypoint, never past it: an overshoot along a bank line keeps
// the unit "below" the river and hands it the same mouth back (item 31).
float step = std::min(currentSpeed, distToWaypoint);
Vector2D newPos;
newPos.x = position.x + (dx / distToWaypoint) * step;
newPos.y = position.y + (dy / distToWaypoint) * step;
```

This is the only troop mover. `CombatEntity::moveTowards` is the stationary
no-op, and flying units go through the same function with
`riverIgnores = true`.

**Options considered and not recommended:**

- Treat "on the bank line and within one step of the mouth" as arrived in
  `getNextWaypoint`. That puts the mover's step size into the planner: a third
  copy of "close enough", the pattern that caused the first absorbing state.
- Snap a knocked-off unit back onto the deck instead of onto the bank. That
  changes what knockback does, which is a gameplay rule, to avoid a pathing
  defect.

### Blast radius

- The cap binds only when the waypoint is closer than one step (<= 0.27 tiles
  even for the fastest tier): at bridge mouths, at `LanePath`'s lane waypoint,
  and when a chased target is within one step (rare, since attack range stops
  movement first).
- Each binding forfeits the rest of that tick's step, so arriving at a waypoint
  costs at most one tick (0.1 s). Deterministic trajectories change, so replays
  and win rates are not comparable across this change.
- Removes up to 52 s stalls from about 18% of matches: a stuck win condition,
  or a tank frozen in front of its own support.

### Tests (to add with the change)

In `tests/core/test_board.cpp` (the file that pins the earlier bridge-mouth
states; it is in the build glob): a ground troop placed on each bank line
(y = `getRiverStart()` and `getRiverEnd()`), swept in x at finer than one
step (0.01) across 2 tiles either side of each mouth (`ArenaLayout` bridge x,
never literals), for both teams and at least two speed tiers (Giant SLOW, Hog
VERY_FAST). Each must cross into the far half within `2 tiles / step + 10`
ticks. On the current code the Giant sweep fails (14 of 21 at 0.05 spacing).

---

## Item 32 — one real deck audited card by card: Evo Furnace, Hero Barbarian Barrel, Evo Bats, Poison, Giant Skeleton, Graveyard, Berserker, Goblin Hut

**Status: IMPLEMENTED 2026-09-25**, on the maintainer's sign-off covering every
finding, with three baselines fixed by the maintainer: the Furnace spawns every
**7 s**, the Giant Skeleton has **3126** hp (3361 x 0.93) and the Fire Spirit
hits for **215**. Seven parts differ from the proposal below; see "What
shipped". The C++ suite ran 730 cases with exactly 1 expected failure, exit 0.
**Class:** one outright bug (32a), gameplay fidelity (the rest).
**GAMEPLAY-AFFECTING.** The observation and action space are unchanged
(`observation_size()` 13977), so checkpoints load. Their win-rate history
does not carry over.

A user's ladder deck (screenshot, level 16), audited against the real game as
of 2026-09-25. Engine ids in slot order: `[138, 174, 126, 32, 39, 110, 51, 95]`,
which `validateDeckSlots` accepts as-is: slot 0 is the Evolution slot, slot 1
the Hero slot, slot 2 the Wild slot. Berserker shows a Hero marker in the
screenshot but sits in a normal slot, so it plays as the base card.

**Instrument:** `tools/audit/gy_deck_audit.cpp`. Build it with
`powershell -File tools/audit/build.ps1 gy_deck_audit` and run
`tools/audit/bin/gy_deck_audit.exe`. It is deterministic: two builds, one by
hand and one by the script, printed identical output. It spawns each unit on a
real `GameManager` board with the towers asleep. It records every
`DamageDealtEvent`, spawn and position, and prints each measurement next to the
real value. 32a was also reproduced through the `.pyd` (`ClashRoyaleEnv`,
`step_self_play`), the path training uses.

**Sources.** Supercell's official balance posts for
[March](https://supercell.com/en/games/clashroyale/blog/release-notes/march-balance-changes-2026/),
[May](https://supercell.com/en/games/clashroyale/blog/release-notes/may-balance-changes-2026/),
[June](https://supercell.com/en/games/clashroyale/blog/release-notes/june-balance-changes-2026/),
[August](https://supercell.com/en/games/clashroyale/blog/news/final-august-balance-changes-826/)
and [September](https://supercell.com/en/games/clashroyale/blog/release-notes/september-balance-changes-2027/)
2026 (the URL says 2027; the post is dated 23 Sept 2026), plus
[August 2025](https://supercell.com/en/games/clashroyale/blog/release-notes/august-balance-changes-2/).
Mechanics come from the Fandom wiki page of each card and its Evolution and
Hero subpages. Stats come from [DeckShop](https://www.deckshop.pro/card/hitpoints)
at friendly level 11, the level the registry carries. **Where two sources
disagree, the official post wins.** The wiki's stat tables often lag its own
change history (Giant Skeleton, Goblin Hut and Barbarian Barrel all do), and
DeckShop is partly stale too; see item 29's addendum.

### Findings

| | card | engine (measured) | real | severity |
|---|---|---|---|---|
| **32a** | Hero Barbarian Barrel | Rowdy Reroll can **never** be activated in a match | 1 elixir, single use, once the Barbarian lands | **bug** |
| **32b** | Furnace | its Fire Spirits **never die and never splash**: 7 hits in 8 s, 2 of a 3-clump damaged | one kamikaze hit, 215 area damage, radius 2.3 | high |
| **32c** | Giant Skeleton | bomb is **instant**, 300 damage, radius 2.0 centre-to-centre, no knockback; dying at a tower it deals **0** to it | 3.0 s fuse, radius 3, ~886, knockback; full damage to the tower | high |
| **32d** | Berserker | "enrage": hit speed 0.6 -> 0.5 -> 0.4 s as hp falls; speed 1.40; range 1.0 | flat 0.6 s; **Fast** (1.988); range 0.8 | high |
| **32e** | Furnace, Hero Barbarian | speed **1.000** (a raw `0.5f`) | **Medium**, 1.325 | medium |
| **32f** | Evo Furnace | hot spawn every 2.4 s **always** | 2.4 s **only while attacking**, alternating sides | medium |
| **32g** | Evo Bats | +24 hp per hit, cap 242, spawns at 121 | +76 per attack (2 x 38, 0.5 s apart), cap 244, spawns at 122 | medium |
| **32h** | Hero Barbarian Barrel | reroll corridor 1.4 wide, **hits air**, **heals 0** | width 2.6, ground only, heals 50% of damage dealt | medium |
| **32i** | Poison, Graveyard (all multi-hit spells) | repeats every **interval + 1** ticks: Poison 1.1 s, Graveyard 0.6 s | 1.0 s and 0.5 s | medium |
| **32j** | Graveyard | every Skeleton spawns **at the cast point**, first at 0.9 s | fixed ring about 3.3 tiles out, first at 2.2 s | medium |
| **32k** | Poison | **no slow** (Knight 1.325 inside and out) | enemy troops 15% slower | medium, blocked by 32m |
| **32l** | Poison (all disc spells) | centre-to-centre test: a unit overlapping the edge is missed | hitbox overlap (the engine's rolling spells already use it) | medium, confidence moderate |
| **32m** | engine-wide | a slow after any earlier stun **is a full stun**; a stunned idle unit **still attacks once** | neither | high |
| **32n** | several | data drift, see the table below | | low |

Item 29 (spell Crown Tower damage) and item 30 (Goblin Hut's Spear Goblins
cannot hit air) are re-confirmed by this audit; see their addenda.

### What shipped (2026-09-25), measured

The same instrument, `tools/audit/gy_deck_audit.cpp`, before and after:

| | before | after | real |
|---|---|---|---|
| Hero Barbarian Barrel ability via `playCard` | never ready | ready; costs 1.0; single use | 1 elixir, single use |
| reroll: a unit 1.0 off the line / a flyer / heal | 0 / 233 / +0 | 232 / 0 / +half the damage | hit / ground only / 50% |
| Furnace spirit vs a 3-clump | 7 hits, 2 hit, spirit lives | 1 shot, all 3 take 215, spirit dies | same |
| Evo Furnace spawns, idle / attacking | every 2.4 s / 2.4 s | 7 s / 2.4 s, alternating -1, +1 tiles | 7 s / 2.4 s, alternating |
| Furnace, Hero Barbarian speed | 1.000 | 1.325 | Medium |
| Berserker hit interval at 100 / 50 / 10% hp | 6 / 5 / 4 ticks | 6 / 6 / 6 | 0.6 s |
| Berserker speed / range | 1.400 / 1.0 | 1.988 / 0.8 | Fast / 0.8 |
| Giant Skeleton bomb | instant, 300, radius 2.0 | 3.0 s fuse, 886, radius 3 + hitbox | 3 s, ~886, 3 |
| ... dying at a Princess Tower | 0 to the tower | 886 | the full bomb |
| Evo Bats heal per attack / cap / spawn hp | +24 / 242 / 121 | +38 now, +38 at 0.5 s / 244 / 122 | 2 x 38 / 244 / 122 |
| Poison pulse spacing | 11 ticks | 10 | 1 s |
| Poison to a Princess Tower | 92 a pulse | 21 a pulse | 21 |
| Poison: troop centred 3.6 out | 0 | 736 | hit (hitbox overlaps) |
| enemy Knight inside Poison | 1.325 tiles/s | 1.127 (x0.85) | 15% slower |
| Graveyard first spawn / spacing / points | 0.9 s / 0.6 s / all at the cast point | 2.2 s / 0.5 s / 7 fixed points | 2.2 s / 0.5 s / 7 fixed points |
| Goblin Hut vs a flyer (10 s) | 0 | 648 | shoots air |
| Barbarian Barrel cast in the river (y 15.4-17.4) | allowed | refused | own side only |
| a stunned idle Knight, target walks in | 1 hit | 0 | 0 |
| 0.65 slow, 3 s after a 0.5 s stun | 0.000 tiles/s | 0.862 | 0.862 |

**Where it differs from the proposal:**

* **32a:** `step()` adopts only bodies that *declare* a late slot
  (`abilitySlotCardId != 0`). Clone copies drop that claim
  (`CombatEntity::becomeCloneCopy`, now shared by the four leaf `clone()`s).
  The first cut adopted any Hero/Champion body and so made a Clone copy
  activatable. `test_game_manager.cpp`'s "A cloned Champion can never
  activate the ability" caught it. A new case pins the same for a cloned
  Hero Barbarian.
* **32g:** two real pulses, 38 on the hit and 38 five ticks later
  (`CardStats::healOnHitSecondPulseDelayTicks`), rather than one heal of 76.
* **32j:** *not* the golden-angle ring proposed above. With every point on
  the 3.3 ring, a Graveyard centred on a defended Princess Tower landed 0 hits
  in 15 s. Every Skeleton needs its 1 s deploy plus ~0.9 tiles of walking,
  and the tower shoots it before that ends. Both Princess Towers reach part of
  the ring. The real layout is seven fixed points, cycled, oriented by side,
  near the edge (RoyaleAPI on the 12 Jan 2026 rework), with one Skeleton
  rising "right on the Crown Tower" (June notes). It now uses the cast point
  plus six ring points 60 degrees apart, starting on the caster's forward
  axis. The coordinates are unpublished, so this is an approximation. Tower
  damage in 15 s against a defended Princess Tower now depends on placement,
  as it should:

  | cast | tower damage | Skeleton hits |
  |---|---|---|
  | centred on the tower | 81 | 1 |
  | 1.5 or 2.5 behind it | 243 | 3 |
  | 2.0 in front of it | 405 | 5 |
  | 1.5 to its outer side | 810 | 10 |

  Before the fix, every Skeleton rose on the cast point: 729 centred, all
  stacked where one splash spell catches them. The deploy time of a Skeleton
  after it rises is the engine's 1 s default. That value is not sourced and
  is the other lever on these numbers.
* **32k:** a **movement-only** slow on troops (`MoveSlowOnHit`,
  `CombatEntity::applyMoveSlow`), refreshed for 10 ticks by each pulse. It is
  not the proposed `FreezeOnHit(11, 0.85)`, which would also have stretched
  attack cooldowns. The real Poison does not affect attack speed.
* **32m:** freezes are held in three concurrent slots, each on its own clock;
  the strongest active one applies. The proposal only reset the factor on
  expiry. That fixes a slow after an *ended* stun, but it would still have
  turned a 0.5 s Zap during an Ice Wizard's refreshing slow into a stun for as
  long as the slow kept refreshing. Now it is the stun for 0.5 s, then the
  slow. `freezeTicks`/`freezeSlow` are kept as the view (longest remaining,
  strongest factor), so every existing reader and test still works. The stun
  gate is as proposed.
* **32n:** the Goblin Hut's Spear Goblins also **deploy in 0.5 s**
  (`CardStats::deployTicks`, default `DEPLOY_TIME_TICKS`). This correction was
  missed at first: the audit above marked the Hut's spawn timing correct, but
  the official August 2025 notes read "Spear Goblin Deploy Delay: 1sec ->
  0.5sec". Also every Spirit is now 215 hp (the same official line), and the
  Evolved Barbarians' hp bonus is 0% (official 4 Aug 2026: both forms 716).
  That note's Blade Rage 3 s -> 5 s is **not** applied: it was not a finding
  here.
* **32d:** the enrage machinery had no other caller, so it was removed with
  its four generic tests rather than left as a mechanic no card has.

**Also changed:** `test_hero_abilities.cpp` (Barbarian 716; the reroll's
new constructor), `test_card_registry.cpp` (the bomb is fused; Battle Ram's
Barbarians 716), `test_area_spell.cpp` (hitbox overlap; exact repeat
spacing). New: `tests/core/test_deck_audit_item32.cpp`, 19 cases tagged
`[item32]`, each written to fail on the pre-fix engine.

**The Python suites, on the rebuilt `.pyd`** (`verify_pyd.py` OK, observation
size 13977). `perception/`: 384 passed. Its one failure and two collection
errors come from a separate, older problem: `mvp_loop` still looks for
`DEFAULT_DECK` in `gym_wrapper.py`, from before the 2026-09-15 move to
`deck.py`. `python_ai/`: **998 passed, 8 failed, 2 skipped**, and all 8 were
Python pinning the behaviour this item changed. They were left red at first,
because `python_ai/` is read-only without an explicit ask.

**FIXED 2026-09-27**, on the maintainer's go-ahead. `python_ai/` now reads
**1007 passed, 0 failed, 2 skipped**; one rewritten test is now two cases.
`perception/` is unchanged at 384 passed, with the same three `DEFAULT_DECK`
failures. The three fixes:

* **4 x Graveyard win condition** (`test_teacher_wincon_resolver`,
  `test_teacher_deck_generalisation`). `teacher.wincon_damage_per_elixir`
  cast a body-spawning spell dead-centre on the enemy Princess Tower. That is
  now the Graveyard's worst cell (81 in 30 s, where the floor is 250), so
  `graveyard_control` resolved **no win condition**. The teacher's own cast
  (`_cells_for` -> `_tower_cells`) aimed at the same cell.
  **Fix:** `teacher.spell_attack_offset` tries 25 whole-cell offsets around
  the tower and keeps the best. The resolver probes there, and the teacher
  casts there (`_spell_tower_cells`, mirrored between the towers).
  * Graveyard: best 2 cells outward and 1 beyond, 972 in 30 s, 194 per
    elixir (WEAK, above the floor).
  * Goblin Barrel: its best is 2 cells in front of the tower, 1320 against
    720 on it, which is the 2026-09-06 all-cell sweep's best. It now reads
    440 per elixir, up from 240.
* **`roller_damage(Barbarian Barrel)` read 0** (`test_log_damage_is_derived`).
  It cast from `BRIDGE_Y`, where the Barrel is now refused. **Fix:** it steps
  back to the furthest castable row. It reads 232; The Log still reads 269.
* **3 x spell geometry** (`test_teacher_spell_geometry` x2,
  `test_advisor_target`). `card_probes.spell_effect` measures Fireball's
  catch radius as 2.75 (2.5 plus the hitbox, on its 0.25 grid), while
  `tactics.FIREBALL_RADIUS = 2.5` restated the old one. So the advisor target
  and the teacher aimed with 2.75, and the hybrid policy, the distillation
  and the live loop with 2.5. **Fix:** `tactics.FIREBALL_RADIUS` is measured
  at import through `card_probes.spell_radius`, the radius half of
  `spell_effect`. It was split out so it imports nothing else, and costs 5 ms.

One pin was rewritten, not just re-run. `test_the_proposed_barrel_cell_lands_on_an_enemy_princess_tower`
required the tower's own cell. Now
`test_a_spawning_spell_lands_beside_an_enemy_tower_where_it_measures_best`
requires the teacher's cell, for a Goblin Barrel and for a Graveyard, to be
the best of the 25 around that tower, measured independently. The old code
fails it for both.

**Measured end to end** with the passive-opponent probe: the teacher pilots
the deck as team 1 and team 0 does nothing, 4 seeds at rungs 0 and 10, same
engine, only the Python differs.

| deck | before | after |
|---|---|---|
| `graveyard_control` | no win condition, 0 Graveyards cast; **0 of 8** three-crowns (4 ran the clock out, 4 took one tower) | **8 of 8** three-crowns, in 385-1085 ticks |
| the audited deck | no win condition; 5 of 8 | Graveyard; **8 of 8**, in 415-1203 ticks |
| `dart_bait_cycle` | 8 of 8, Goblin Barrel on the tower | 8 of 8, 2 cells in front; rung 10 mean 385 -> 259 ticks |

**Still open:** the teacher's combo families place the win condition on
`tactics.best_hog_cell`, the bridge, whatever the card is. That is wrong for
a Graveyard or a Goblin Barrel (`TODO.md` item 000).

**Not done, and why.** No first-hit wind-up anywhere (engine-wide, needs a
sourced value for every card). Troop collision radii are unchanged (all 0.4).
The Furnace spawn interval stays 7 s by the maintainer's call, although the
wiki says 5. Item 29's other spells and item 30's Goblin Gang / Rascals are
still open. **Found in passing and not fixed:** the Cannon Cart grounds itself
with a stun (`transformBecomesStationary` -> `applyFreeze(ticks, 0.0f)`),
which also holds its cooldown, so it stops firing for its last 15 s. That was
true before this item too. It is filed as a separate task. And a 0-damage
spell (Graveyard, Freeze) still calls `takeDamage(0)` and emits a 0-amount
`DamageDealtEvent` on every enemy in its disc, every pulse
(`AreaSpell::update`). A centred Graveyard "hits" the tower 12 times that way.
So an event count is not a hit count. The instrument counts amount > 0 since
2026-09-26. Nothing in gameplay changes; whether any statistics collector
counts events rather than summing amounts was not checked.

### 32a — Hero Barbarian Barrel's ability is unreachable

**Evidence.** The Hero is played through `GameManager::playCard` with 10
elixir, and the Barbarian (id -47, `isHero`, `abilityUsesRemaining = 1`) is on
the board after 17 ticks. Then:

```
isChampionAbilityReady(0, 1) = false      (checked at +10, +20, +30, +40 ticks)
activateChampionAbility(0, 1) = false      elixir unchanged
```

The same result comes through the `.pyd`: `step_self_play(...,
activate_ability0_slot1=True)` spends nothing and does nothing. Calling
`CombatEntity::activateAbility` directly does fire the effect, so the defect is
in how the Hero is found, not in the effect.

**Cause.** A slot's ability belongs to the entity recorded in
`ChampionSlotState::trackedEntityId`. That is set in exactly one place: the
loop at the end of `playCard`. The loop scans the entities pending at play
time for `(isChampion || isHero) && cardId == deckConfig[slot]`. At play time
the only pending entity is the rolling spell, which is not a `CombatEntity`.
The Barbarian spawns about 17 ticks later, when the barrel stops, and carries
cardId -47, not 174. So nothing ever matches, and `findChampionInSlot` returns
nullptr for the whole match. The existing tests could not see this: one checks
the registry flags, the other calls the effect on a hand-built entity. Neither
goes through `playCard`.

**It is SILENT, the class the pre-launch audit hunted.** Nothing raises.
`rl/abilities.py` masks the activate arm by readiness, so a Hero Barbarian
Barrel deck trains an ability head that is masked on every step.

**Proposed change (exact).**

1. `CardStats.h`, beside `isHero`:
   ```cpp
   // The deck card whose Heroic/Wild slot this body belongs to, when that is
   // not its own cardId: Hero Barbarian Barrel's Barbarian (-47) is spawned by
   // the rolling spell (174) ~17 ticks after the play. 0 = my own cardId.
   int abilitySlotCardId = 0;
   CardStats& withAbilitySlotCard(int cardId) { abilitySlotCardId = cardId; return *this; }
   ```
   Add the same field to `CombatEntity`, and copy it in
   `CardFactories::applyCardMetadata`.
2. `CardRegistry.h`, `heroBarbarianBarrelBarbarianStats()`: append `.withAbilitySlotCard(174)`.
3. `GameManager.h`: move the tracking loop out of `playCard` into
   `void trackSlotEntities(int team, size_t fromPending)`. Match on
   `(ce->abilitySlotCardId != 0 ? ce->abilitySlotCardId : ce->cardId) == deckConfig[slot]`,
   and add `ce->team == team`. `playCard` keeps its call with `pendingBefore`.
   `step()` gains `trackSlotEntities(0, 0); trackSlotEntities(1, 0);` right
   before each of its two `board.commitPendingEntities(currentTick)` calls.
   That catches bodies spawned during entity updates and during the previous
   tick's death effects.

Not recommended: giving the Barbarian the card's id 174. That id is registered
as a spell (`isSpell`), and consumers key on `cardId`, including
`get_card_info`, the replay's `cardMeta` and the card probes.

**Test.** Add to `test_hero_abilities.cpp`, going **through `playCard`**:
deck `{138, 174, 126, 32, 39, 110, 51, 95}`, `setHand` including 174, 10
elixir. Play the card and step 20 ticks. `isChampionAbilityReady(0, 1)` must
be true. `activateChampionAbility(0, 1)` must be true and cost exactly 1.0. A
second activation must be false (single use). Today this fails at the first
assertion.

### 32b — the Furnace's Fire Spirits are immortal single-target shooters

`furnaceFireSpiritStats()` (-15) is `troop(... 230, SPEED_VERY_FAST, 2.5f, 207,
10 ...)` with only `.withTargetsAir()`. There is no `.withDieAfterFirstHit()`
(the playable Fire Spirit, id 73, has it) and no splash (73 says "splash not
modelled"). **Measured:** one Furnace spirit against a tight clump of three
ground units hit **7 times in 8 s, killed one and was still alive**. The real
spirit launches itself once for area damage on all three and dies. Spawned
every 7 s, each one is a permanent 207-damage-per-second turret with range 2.5
walking beside the Furnace. That makes the engine's Furnace far stronger than
the real card, and it has nothing in common with a Fire Spirit's actual job,
which is one burst of splash.

**Proposed change (exact).** In `CardRegistry.h`:

```cpp
static CardStats furnaceFireSpiritStats() {
    return troop(-15, "Fire Spirit", 0.0f, Archetype::RangedSquad, 215, SPEED_VERY_FAST, 2.5f, 215, 10, '<')
        .withTargetsAir()
        .withSplash(2.3f)          // Fire Spirit area radius (wiki, since 2021-09-06)
        .withDieAfterFirstHit();
}
```

Make the same change to card 73: hp 215 and damage 215 (official 4 Aug and
16 Sep 2026), and `.withSplash(2.3f)` in place of the "not modelled" comment.
The splash path already exists: `RangedTroop` hands `splashRadius` to its
`Projectile`, which splashes around the target on arrival.

### 32c — Giant Skeleton's bomb

`.withDeathEffect(std::make_shared<AreaDamageOnDeath>(2.0f, 300))`, whose
registry comment says "not sourced". **Measured:** the bomb fires on the tick
of death. It deals 300 within 2.0 tiles, centre to centre (1.9 hit, 2.4
missed), with no knockback. A Giant Skeleton that died while hitting a
Princess Tower was **2.62 tiles** from its centre (0.4 + 0.8 + 1.5 at maximum
reach), and the bomb dealt the tower **0**. The chip play the card is known
for does not exist in the engine. `AreaDamageOnDeath` also emits no
`DamageDealtEvent`, so the event-based statistics (damage by card, kill
attribution) never credit the bomb. The reward's tower term reads tower HP, so
it would see bomb damage if any landed.

**Real, since 4 May 2026.** The bomb drops on death and explodes **3.0 s**
later. Radius **3** (wiki). Damage **+29%** on the wiki's 688, so about 886.
Knockback. The same damage to a Crown Tower, where it was 2x before May.
Hitpoints **-7%**, so 3361 becomes about 3125. Collision radius 0.75 (the
engine gives every troop 0.4; out of scope here). The official post quotes
absolute values (hp 1413 -> 1313, bomb 209 -> 269) that fit no level-11 table,
including DeckShop's, which still shows 3361. Only its percentages are used.

**Proposed change (exact).** Add a new death effect, `include/core/DelayedAreaDamageOnDeath.h`.
It drops an `AreaSpell` with a fuse. That reuses `AreaSpell`'s timing,
knockback, tower rule, snapshot and `DamageDealtEvent` attribution, and an
`AreaSpell` is untargetable, as the real bomb is.

```cpp
class DelayedAreaDamageOnDeath : public IDeathEffect {
    float radius; int damage; int fuseTicks; float knockback; int sourceCardId;
public:
    DelayedAreaDamageOnDeath(float r, int d, int fuse, float kb, int card)
        : radius(r), damage(d), fuseTicks(fuse), knockback(kb), sourceCardId(card) {}
    void apply(Board& board, const Vector2D& p, int team) const override {
        auto bomb = std::make_shared<AreaSpell>(board.allocateId(), p.x, p.y, team, radius, damage,
            fuseTicks, 'J', nullptr, false, 1, 0, false, 1.0f, 0, knockback);
        bomb->name = "Giant Skeleton Bomb";
        bomb->cardId = sourceCardId;
        board.addEntity(bomb);
    }
};
```

Registry: `troop(39, ..., 3125, ...)` and
`.withDeathEffect(std::make_shared<DelayedAreaDamageOnDeath>(3.0f, 886, 29, 1.0f, 39))`.
The fuse is 29, not 30, because `AreaSpell` applies on the update after
`delayTicks` reaches 0, so 29 lands at 3.0 s. The knockback of 1.0 is
Fireball's engine value and is not sourced.

### 32d — Berserker

`troop(51, ..., 896, 0.7f, 1.0f, 102, 6, 'v').withEnrage(0)`. **Measured:** hit
intervals of 6 / 5 / 4 ticks at 100% / 50% / 10% hp, speed 1.400 tiles/s,
range 1.0. The real card hits at a flat 0.6 s. None of its balance history
mentions a rage. The self-heal that the registry comment calls an "optional
modifier" is one of the wiki's **event Modifiers**, not part of the card. It moves Fast: the wiki and DeckShop's speed list agree, and it is 1.988
in engine units. Its range is Melee: Short, 0.8.

**Proposed change (exact):**
`add(troop(51, "Berserker", 2.0f, Archetype::MeleeSquad, 896, SPEED_FAST, 0.8f, 102, 6, 'v'));`.
This is the only `withEnrage` caller, so the enrage machinery in `CardStats`
and `CombatEntity` becomes dead code. Removing it is optional.

### 32e — raw speed literals on this deck

| unit | registry | measured | real |
|---|---|---|---|
| Furnace (70, and both forms of 138) | `0.5f` | 1.000 | Medium since 2025-10-06 (wiki, DeckShop) |
| Hero Barbarian (-47) | `0.5f` | 1.000 | Medium (the Barbarian) |

Replace both with `SPEED_MEDIUM`. **Why the existing guards missed them.**
`0.5f` resolves to within 0.6% of `SPEED_SLOW`, so a "near some tier" test
passes it; that is the trap `CLAUDE.md` records. The generic "every spawned
unit moves at the speed of its own playable card" test matches on name, and
this helper is named "Hero Barbarian Barrel", not "Barbarians". Extend that
test with Hero bodies against their base card.

### 32f — Evolved Furnace's hot spawn ignores whether it is attacking

**Measured.** Held in place with nothing in sight, the evolved Furnace still
spawned at 3.4, 5.8, 8.2 s and onward, every 2.4 s. Every spirit spawns on the
Furnace's own point. The card text and wiki say the 2.4 s rate applies "when
it's hot and attacking", and that the spirits spawn alternately to the left
and right.

**Proposed change.** Add `int periodicHotIntervalTicks` to `CardStats` and
`CombatEntity`, set to 24 on the evolved form, leaving the base 70. In
`CombatEntity::update`'s periodic block, treat "hot" as having landed an
attack within one cooldown (`ticksSinceLastHit <= attackCooldown`):

```cpp
const bool hot = periodicHotIntervalTicks > 0 && ticksSinceLastHit <= attackCooldown;
if (hot && periodicTicksUntilNext > periodicHotIntervalTicks) periodicTicksUntilNext = periodicHotIntervalTicks;
// ...existing countdown; on fire:
periodicTicksUntilNext = hot ? periodicHotIntervalTicks : periodicIntervalTicks;
```

The alternating spawn sides have no sourced offset, so leave them unmodelled.

### 32g — Evolved Bats heal a third of what they should

**Measured.** Each hit adds +24, up to 242. Real: 122 hp at spawn, and each
attack heals two 38-hp pulses 0.5 s apart (76 in total), overhealing up to 244
(wiki table, level 11). **Change:** `troop(78, "Bats", ..., 122, ...)` and
`.withHealOnHit(76, 244)` in the evolved form. The two pulses land as one
here, 0.5 s early, which is a documented approximation.

### 32h — the Rowdy Reroll effect itself (behind 32a)

`HeroBarbarianBarrelRerollEffect(3.0f, 0.7f, 233)` is correct on the 3-tile
roll (official, May 2026), on single use, and on half damage to towers (116,
matching the wiki's Crown Tower column). **Measured**, calling it directly:

* A ground unit 1.0 tile off the line took **0**. The 0.7 argument is a
  half-width, so the corridor is 1.4 wide. The real width is 2.6, and the
  main roll already uses 2.6 with a surface test.
* A **flying** unit on the line took **233**. The real reroll targets Ground
  only.
* The Barbarian healed **0** (300 hp before and after). The real reroll heals
  it for 50% of the damage dealt.

**Change:** construct it as `(3.0f, 1.3f, 232, 716)`. In the loop, skip
`entity->isFlying`, and test `dist > halfWidth + CombatEntity::effectiveRadiusOf(*entity)`,
the main roll's surface convention. Sum `dealt`, then
`self.hp = std::min(self.hp + totalDealt / 2, maxHp)` with the new `maxHp`
argument.

### 32i — every multi-hit spell repeats one tick late

`AreaSpell::update` sets `delayTicks = tickInterval` after each application.
The countdown then consumes `tickInterval` updates and applies on the next
one, so the period is **interval + 1**. Measured: Poison pulses at ticks
1, 12, 23, … 78, which is 1.1 s instead of the registry comment's "every
second". Graveyard spawns every 6 ticks instead of "every 0.5 s". The same code
path gives Arrows (volleys 3 ticks apart, not 2), Evolved Zap, Earthquake,
Goblin Curse, Void and Vines. Total damage is unchanged; only the cadence
stretches. **Change:**
`delayTicks = (tickInterval > 0) ? tickInterval - 1 : 0;`. Tests pinning pulse
ticks will move.

Also, the Graveyard comment "arrivals outrun a Princess Tower's fire rate 2:1"
is out of date. The tower fires every 8 ticks, so the ratio is 1.6:1 at the
intended 0.5 s and 1.33:1 at the measured 0.6 s.

### 32j — Graveyard's spawn point and first spawn

**Measured.** 12 Skeletons, first at tick 9, and every one spawns at the cast
point (the distances of 0.0-0.3 are collision nudges). Real: 12 Skeletons
since June 2026 ("removing one of the Skeletons that rises right on the
Crown Tower"), the first at **2.2 s** (wiki, since 2020-12-09), then every
0.5 s. Since 12 Jan 2026 they rise in a **defined pattern**, not a random one,
near the edge, with a spawn radius of **3.3** since 2 Feb 2026. A centre
spawn stacks every Skeleton on one point, so a single small splash (The Log,
Arrows, Zap) catches all of them as they rise. The real ring spreads them out.

**Change.** Raise `spellDelayTicks` from 8 to 21, so the first spawn lands at
2.2 s. Add a spawn ring to `AreaSpell` (new `CardStats::spellSpawnRingRadius`,
set to 3.3 for Graveyard). Application `k` spawns at
`clampToBoard(position + 3.3 * (cos(k * 2.39996), sin(k * 2.39996)))`, where
k comes from the spell's own hit counter. The effect object must stay
stateless, per the snapshot invariant in `CLAUDE.md`. The real pattern is
fixed but unpublished, so the golden-angle sequence is a documented
approximation. Whether each real Skeleton then has its own deploy delay after
rising is not sourced; today every one gets the standard 1 s.

### 32k — Poison's 15% slow is missing (apply only after 32m)

**Measured.** An enemy Knight walked 1.325 tiles/s outside the Poison and
1.325 inside it. **Change:**
`.withSpellOnHit(std::make_shared<FreezeOnHit>(11, 0.85f))`, so each pulse
refreshes an 11-tick 15% slow. **Do not apply it before 32m(i)**: today any
unit stunned earlier in the match would take the whole Poison as an 8-second
hard stun. Two caveats. The engine's slow also stretches attack cooldowns;
that is the game's generic slow, but the wiki names only movement. And
`FreezeOnHit` would also reach buildings, where the real slow is "enemy
troops" only.

### 32l — disc spells test the centre, not the hitbox (confidence: moderate)

`AreaSpell`'s disc branch uses `distanceTo(entity->position) <= radius`.
**Measured:** a troop centred 3.6 tiles from a Poison of radius 3.5 took 0,
although 3.2 tiles of it lie inside the cloud. The engine's own rolling spells
test against the target's surface (`effectiveRadiusOf`), as does the "reaches
the tower from the bridge" rule `CLAUDE.md` pins for The Log. That is the real
game's convention as far as this audit can tell. **Change:**
`<= radius + CombatEntity::effectiveRadiusOf(*entity)`. The blast radius is
every disc spell, towers included: Poison or Fireball dropped beside a Princess
Tower would start hitting it. Verify against footage before applying.

### 32m — two stun defects, engine-wide

1. **`freezeSlow` never resets.** `applyFreeze` does
   `freezeSlow = std::min(freezeSlow, slowFactor)`, and nothing restores it
   to 1.0 when the freeze ends. **Measured:** a Knight under a 0.65 slow moved
   0.862 tiles/s, but **0.000** if it had taken a 0.5 s stun three seconds
   earlier. After the first Zap, Electro Spirit or Ice Spirit of a match,
   every later slow on that unit is a full stun: Ice Wizard, Ice Golem's
   death slow, Giant Snowball, Earthquake, and 32k.
2. **A stunned unit whose cooldown is already 0 still attacks once.** The
   targeting path re-acquires with `findTarget` whether or not the unit is
   frozen, and the attack is gated only on `currentCooldown == 0`. A stun
   holds the cooldown still but does not block the swing. **Measured:** an
   idle Knight stunned for 5 s hit a unit that walked into reach, for 202.
   The same applies to an idle tower under Freeze: it fires its first shot.
   This audit's first harness run froze the towers with `applyFreeze` and got
   stray 90-damage tower shots anyway.

**Change**, in `CombatEntity::update`:

```cpp
const float slowThisTick = freezeSlow;           // beside frozenThisTick
freezeSlowThisTick = slowThisTick;               // new member, read by Troop::moveTowards
const bool stunned = wasFrozen && slowThisTick <= 0.0f;
...
if (freezeTicks > 0) {
    freezeTicks--;
    if (currentCooldown > 0.0f) currentCooldown -= slowThisTick;
    if (freezeTicks == 0) freezeSlow = 1.0f;     // a slow ends with its freeze
}
...
// the attack, jump and hook branches each gain `!stunned &&`
```

In `Troop::moveTowards`, change `speed * freezeSlow` to
`speed * freezeSlowThisTick`, so the last tick of a slow still moves slowly.
Gate the branches on `stunned`, not `wasFrozen`: a slowed unit is "frozen" in
this engine and must keep attacking.

### 32n — data drift (level 11)

| unit | registry | current | source |
|---|---|---|---|
| Giant Skeleton hp | 3361 | ~3125 | official May 2026 (-7%) |
| Barbarian hp: BB's (-23), Hero's (-47), and every other copy (8, -10, -16, -35) | 691 | **716** | official Aug 4, 2026 |
| Goblin Hut hp | 1228 | **1180** | wiki history (-4%, 2025-10-06); DeckShop 1180 |
| Fire Spirit hp / damage (-15 and 73) | 230 / 207 | **215 / 215** | official Aug 4 / Sep 16, 2026 |
| Spear Goblins hit speed (23, -17, -26, -27) | 1.7 s | **1.6 s** | official Aug 4, 2026 |
| Barbarian Barrel damage (101, 174) | 233 | 232 | DeckShop; wiki Hero table |
| Evolved Bats spawn hp | 121 | 122 | wiki (+50%) |
| Barbarian Barrel placement | own half **and the river** (y <= 17.5, measured) | **own side only** | wiki. The engine's roller rule was written for The Log; Barbarian Barrel has its own, stricter rule |

The Spirit hitpoint cut applies to all four Spirits (Ice, Heal and Electro
too), which is outside this deck.

**Unresolved, not proposed.**
* **Furnace spawn interval.** 7 s per the official August 2025 notes, which
  is the engine's value. The wiki and one third-party site say 5 s since
  4 August 2026, but Supercell's posted notes for that date never mention
  Furnace. The engine keeps 7 s until someone confirms in-game.
* **Furnace damage and hp.** 179 / 727 per the wiki and the official 2025
  nerf (the engine's values). DeckShop says 135 / 896, where 896 is the
  pre-nerf hp.

**Engine-wide, noted and not proposed.**
* No unit has a first-hit (wind-up) time. Every unit strikes one tick after
  reaching range, against real wind-ups of 0.2 s (Berserker), 0.3 s (Giant
  Skeleton), 0.4 s (Barbarian), 0.5 s (Skeleton, Spear Goblin) and 0.6 s
  (Bats).
* Every troop's collision radius is 0.4.

### Verified correct

* **Bats:** 5 bodies, 81 hp, 81 damage, 1.2 s (the March 2026 buff), Very Fast,
  flying, hits air and ground. They die to one Poison pulse; Evolved Bats
  survive one. The registry's Evolution cycle count is 2 for both Evolved
  Bats and Evolved Furnace, as on the wiki. The probe does not exercise it.
* **Poison:** 8 x 92 = 736, radius 3.5 (measured centre to centre), and no damage to your
  own units (your Skeletons inside your own Poison took 0).
* **Giant Skeleton:** 276 damage, 1.3 s (the January 2026 buff), Medium,
  range 0.8, sight 5.0, ground-only. Enemy Bats are untouchable by it (0
  damage in 6 s).
* **Graveyard:** 12 Skeletons (the June 2026 value), radius 4; each Skeleton
  has 81 hp, 81 damage, 1.1 s, Fast, range 0.5.
* **Barbarian Barrel:** 233 per unit hit. The corridor is 2.6 wide against the
  target's surface (1.6 tiles off-axis hit, 1.9 missed), 4.5 long against the
  surface (4.8 hit, 5.2 missed), ground only, no knockback. The Barbarian
  spawns where the barrel stops (17 ticks), with 192 damage, 1.4 s and
  Medium speed.
* **Goblin Hut:** nothing spawns while idle. Spawns are gated on an enemy
  within 6 tiles, 2.2 s apart (the April 2026 value). It expires at exactly
  30 s and releases one Spear Goblin on death.
* **Furnace:** 727 hp, 179 damage every 1.7 s (January 2026), range 5.5 (June
  2026), hits air. One spirit every 7 s, the first at 8 s (deploy plus one
  interval).
* **Hero Barbarian Barrel:** 1 elixir, single use (August 2026: every Hero
  ability is single-use).
* **Deck-slot legality** for this deck.

### Blast radius

* **Pool decks carrying an affected card.** Furnace: `royal_hogs_furnace`.
  Graveyard: `graveyard_control`. Poison: `graveyard_control`,
  `miner_poison_control`, `pekka_bridge_spam`. Bats: `wall_breakers_cycle`,
  `mega_knight_ram`. Barbarian Barrel: six decks.
* **32m reaches every deck with a stun or a slow**, including `SHIPPED_DECK`
  (Ice Spirit's stun, Ice Golem's death slow).
* **32i and 32l change every multi-hit or disc spell.** 32l also changes
  Fireball in `SHIPPED_DECK`.
* **No observation or action-space change.** Checkpoints load; win-rate
  history stops being comparable.
* **Python needs a change after all.** This bullet said "no change" when it
  was written: `card_probes` measures spell damage, tower damage and air
  targeting by injection, and with 32a fixed `rl/abilities.py` unmasks the
  reroll on its own. Both still hold. But three probes cast on a fixed cell or
  restated a measured constant, and the shipped engine broke them: eight
  `python_ai` tests went red. Fixed 2026-09-27; see "The Python suites" above.
* **Tests that will move:** `test_hero_abilities.cpp` (the Barbarian's 691 hp
  and the reroll's 0.7 half-width) and any test pinning Poison, Graveyard or
  Arrows pulse ticks. `test_default_deck_qa.cpp` is also affected if 32m
  changes an Ice Spirit timing it pins.

---

## Item 33 — a Tombstone on the river corner traps its own Skeletons forever

**Status: PROPOSED 2026-09-28, NOT changed.** Needs the maintainer's choice of
fix (options below). **Class:** movement / pathing. **GAMEPLAY-AFFECTING** if
fixed. Found by the Reflex Lab's curation suite (`tools/lab/curate.mjs`, the
`check` step), which drops any card whose rollouts contain a stalled unit;
the Tombstone was dropped from the lab's roster for this.

### The gap

A ground troop whose straight-line path to its waypoint runs through a
building is pushed back out **radially** by `Board::resolvePositionAgainstBuildings`
-> `pushAwayFrom`, with only a fixed 0.05 tangential nudge. Against open
ground that slides it around the building. Wedged between the building, the
board edge and the river bank, the radial push points into the edge and the
bank, and `clampToBoard` returns it to **exactly** where it started: a fixed
point, the same class as the three bridge-mouth absorbing states (items 31 and
the 2026-08-09/08-20 fixes).

The worked case, team 0 Tombstone at (1,15), Skeleton at (0.00, 15.50), waypoint
the left bridge mouth (2.5, 15.5):

1. `moveTowards` steps 0.2 toward +x: (0.20, 15.50).
2. That is 0.94 from the Tombstone's centre, inside its 1.0 + 0.4 clearance;
   `pushAwayFrom` moves it 0.46 along (-0.85, 0.53), plus the nudge:
   about (-0.16, 15.78).
3. `clampToBoard`: x -> 0 (board edge); y = 15.78 is in the river band off
   the bridge, -> 15.5 (bank). Back to (0.00, 15.50). Every tick.

### Evidence

`tools/audit/building_trap_audit.cpp` places a spawner at EVERY legal cell on
an empty board, per team, and flags a spawn stationary 30+ ticks with nothing
in its own reach (the soak's criterion):

| card | team 0 | team 1 |
|---|---|---|
| Tombstone (96) | **2 of 170** cells: (1,15), (2,15); Skeleton stuck at (0.00, 15.50) | 2 of 170: (16,18), (15,18); stuck at (17.00, 17.50) |
| Goblin Hut (95) | 0 of 170 | 0 of 170 |
| Bomb Tower (27) | 0 of 170 | 0 of 170 |

Team 1's cells are the exact 180-degree rotations of team 0's, so the rule is
**not** team-biased; the nudge's fixed handedness makes the trap one-sided per
team (the left corner for team 0 only), which is the rotation symmetry the
arena has anyway. The run deck's Goblin Hut is **not** affected.

Per-tick trace (`lab_cli frames`, Hog Rider dropped at (14,18), Tombstone at
(1,15) at tick 10): each spawn wave puts one Skeleton at (2.40, 14.95), which
crosses normally, and one clamped to x = 0, which reaches (0.00, 15.50) at
tick ~65 and is still there at tick 164. Three waves, three stuck Skeletons.

### Options (not decided)

- **A. Slide in the mover.** In `Troop::moveTowards`, when the resolved and
  clamped position is within a small fraction of the step of where the unit
  started while its waypoint is farther, retry the step along the building's
  tangent (the side that does not immediately re-clamp), resolving and
  clamping again. Local to the one troop mover, binds only when stuck. A
  memoryless tangent choice can dither on a concave pocket, so it needs the
  same sweep test as item 31's.
- **B. Choose the nudge's handedness from the travel direction** in
  `pushAwayFrom` (needs the direction passed in). Smallest edit; fixes this
  pocket; leaves any pocket where both tangents clamp.
- **C. Keep spawns off the far side.** Spawn the Tombstone's bodies on the
  side facing its lane. Fixes the spawner case only; a player-placed troop
  behind a corner building is untested.

Recommendation: A, measured with `building_trap_audit` (must read 0 for every
spawner, both teams) plus the soak. Not prototyped.

### Blast radius

Two cells per team for the Tombstone, in the lab's measurement; any building
flush with an edge and the river bank is a candidate pocket. A fix changes
trajectories wherever a troop brushes a building, so replays and win rates are
not comparable across it.

## Item 34 — the Battle Ram keeps swinging at a building; the real one breaks on its first hit

**Status: PROPOSED 2026-09-28, NOT changed.** **Class:** card behaviour.
**GAMEPLAY-AFFECTING** if fixed. Found while adding the Battle Ram to the
Reflex Lab (`web/lab/`), which ships it as the engine has it.

### The gap

The real Battle Ram hits the first building it reaches once (double damage
if it is charging), then **breaks**, releasing its two Barbarians. Two
independent sources, fetched 2026-09-28: the Clash Royale Wiki's card page
("Once it reaches a building or it is destroyed, it will break and reveal the
two Barbarians underneath", via search; the page itself answers automated
fetches with HTTP 402) and RoyaleRep's card text ("Two Barbarians charge
behind a ram, then leap out swinging after it breaks").

The engine's Battle Ram (`CardRegistry.h`, id 81) is an ordinary
building-targeter with a charge: it keeps attacking, 192 a hit every 1.4 s
after the 384 charge hit, until something kills it, and only then releases the
Barbarians (`SpawnOnDeath`).

### Evidence

`tools/lab/out/lab_cli.exe trace 81 14 20` (Ram dropped for team 1 at the
right bridge, no defender, tick by tick):

```
t= 97  tower hit: -384          the charge hit
t=106  Battle Ram gone          killed by the tower 0.9 s later, not broken
t=107  Barbarians x2 appear
```

Against a Princess Tower the two nearly coincide, because the tower kills the
Ram's 691 hp almost at once. Against a defending **building** they do not: a
Cannon pulled Ram keeps swinging at the Cannon until the Cannon or the tower
kills it, instead of hitting once and turning into two Barbarians on the spot.
That changes exactly the interaction the lab lets a visitor explore (pulling
the Ram with a building).

### The fix (proposed)

One builder call: `.withDieAfterFirstHit()` on the base Battle Ram, the
mechanism Wall Breakers and the Spirits already use
(`CombatEntity::dieAfterFirstHit` sets `hp = 0` on the hit, so the ordinary
death path fires `SpawnOnDeath`). To confirm after the change,
`lab_cli trace 81 14 20` should show the Ram gone on the tick of its first
hit and the Barbarians the tick after.

Not in scope, and deliberately left alone:

- **The Evolution (161)**: its "Head-First Ram" keeps double damage on every
  later hit, which implies it does not break; not verified.
- **Stats.** The engine's 691 hp and 192 damage look like Barbarian-scale
  numbers, and the sources above suggest a level-11 Ram closer to ~920 hp and
  ~575 charge damage (the Wiki's level-1 row, 430 / 135 / 270, scaled by the
  usual 10% a level; RoyaleRep lists a 573 charge). Neither source was
  readable in full, so no number is proposed here; the maintainer can read
  the in-game card at level 11.

### Blast radius

Every Battle Ram push: the Barbarians arrive at the first building about one
Ram lifetime earlier, and a building that pulls the Ram takes one hit instead
of several. Win rates for decks with the Ram are not comparable across it.
The lab's Battle Ram tables would need rebuilding (`tools/lab/curate.mjs`
rebuilds whatever is missing: delete `tools/lab/out/tables/81_*`).
