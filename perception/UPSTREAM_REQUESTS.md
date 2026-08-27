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
