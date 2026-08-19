# The Utility-Teacher Curriculum

**Date:** 2026-08-19
**Status:** design approved, implementation pending
**Replaces:** the opponent-elixir-multiplier curriculum in `train.py`

---

## 1. The problem this solves

Every win-rate number that matters in this project was earned against the C++
`HeuristicOpponent` at a **permanent 1.0x-1.5x elixir multiplier**, and phase 1's
curriculum tops out at 1.5x. `CLAUDE.md`'s "1.5x Curriculum Overfitting
Hypothesis" records the consequence and the measurement that supports it: the
cost of playing the win condition falls monotonically as the opponent's economy
falls.

| opponent | baseline win | SMART-forced win | delta | p |
|---|---|---|---|---|
| 1.00x | 1.000 | 1.000 | +0.0000 | VOID (ceiling) |
| 1.25x | 0.950 | 0.825 | -0.1250 | 0.0059 |
| 1.50x | 0.617 | 0.317 | -0.3000 | 3.2e-06 |

Four separate interventions built to make the agent play its win condition
(`W_WIN_CONDITION_DAMAGE`, the Hog advisor, forced usage at eps=0.15, SMART
forcing with a calibrated gate) all returned null or negative. Each one tries to
move a **policy**; none of them changes the **payoff**.

### 1.1 The trap in the obvious fix

Deleting the multiplier is necessary and **not sufficient**. At 1.0x the C++
heuristic is beaten ~100% -- recorded for the ep-64k Giant-deck net and again for
the ep-25202 2.6 net (win rate 1.000, `CLAUDE.md` "RUN 1 PROGRESS"). Removing the
handicap without replacing the opponent converts a *mispriced* environment into a
*zero-gradient* one. The 1.00x row above is VOID for exactly this reason.

So the multiplier must be **replaced by competence**, not merely removed.

### 1.2 Why the teacher must be hand-written, not neural

The project already owns a strong decision-time expert: `model_weights_cured.pth`
plus search at horizon 12, which scores 0.9225 against `heuristic@1.5x`. It
cannot be used here. Search scores candidates with the **net's own critic**, so at
random init the expert is not an expert -- measured, n=60 paired: random init
@1.0x scores 0.450 policy / 0.417 search while overriding 21.5% of decisions,
against the trained net's 0.483 / 0.667 at 10.2%. At init it deviates twice as
often and gains nothing.

A hand-written utility function has no cold start. **That is the decisive
argument for Proposal B and it is not about the bot being "nicer" to train
against.**

---

## 2. Decision: Proposal A vs Proposal B

### Proposal A -- scenario injection

**Verdict: keep, secondary, default-off until the payoff is verified.**

Scenario injection changes the *state distribution*, not the *payoff*. If the
environment prices the Hog negatively, injecting states where it "should" be good
simply pays the negative price more often -- which is precisely the observed
dose-response of the forced-usage experiments (usage up, win rate down,
monotone).

Its real value is credit assignment: a punish window is rare and its causal link
is buried in a long GAE trace. That is the same argument that justified the
existing defensive `SCENARIO_INJECTION_PROB = 0.30`. It is a **sampling
efficiency** tool and it is worth having -- after the payoff is right, not before.

### Proposal B -- the utility-search heuristic teacher

**Verdict: this is the primary mechanism**, with three corrections.

**Correction 1: "no if-else trees" is half wrong.** The hand-written rules in
`tactics.py` are the strongest placement signal this project has measured:

| card | advisor | trained net | random legal cell |
|---|---|---|---|
| Cannon (tower HP preserved) | **564.1** | 12.1 | 353.5 |
| Fireball (elixir killed) | **2.405** | 0.000 | 0.276 |
| Giant (enemy tower damage) | **535.6** | 3.3 | 86.7 |

Rules become the **candidate generator**; forward simulation becomes the
**ranker**. Enumerating 4 cards x 612 tiles and simulating each costs ~100x more
for no gain, and `CLAUDE.md` already records that the advisor's essentially
two-cell Giant rule is "a very strong prior this board rewards."

**Correction 2: the utility terms are readable, not estimated.** After a rollout
the engine hands back `get_tower_damage_dealt`, `get_troop_damage_dealt`,
`get_elixir_value_killed_by`, `get_towers_alive`, `get_elixir_spent`. The utility
function is a linear read of the rollout's own statistics. Only *cycle value* and
*opportunity cost* need hand-derivation, and opportunity cost falls out of the
no-op baseline (section 4.3).

**Correction 3: determinism is a liability in a sparring partner.** A fully
deterministic opponent is memorizable -- the same echo-chamber failure the pivot
is meant to avoid, with one opponent instead of a league. The decision rule stays
deterministic; the **profile** is drawn per episode.

### The synthesis

- B is the mechanism: fix the payoff with a symmetric economy and a competent
  opponent.
- A is a conditional accelerator: offensive scenario injection, default-off,
  enabled only after the environment-property test (section 6.1) passes.
- The curriculum ladder becomes **opponent competence** at a fixed 1.0x economy.

### Decisions taken by the human, 2026-08-19

1. **C++ `HeuristicOpponent`: evaluation anchor only.** It never trains the agent
   again. It stays as a fixed anchor so results remain comparable with every
   historical number in `CLAUDE.md`. The specialization risk (a net trained
   without it measured 0.51 against it, vs v1.2.0's 0.6225) is knowingly
   re-accepted in exchange for clean attribution.
2. **Multiplier: stripped from the curriculum, API retained.** Training is pinned
   at 1.0x with a test that fails if anything raises it. `set_opponent_elixir_
   multiplier` stays bound and callable so the A/B harnesses can still sweep it --
   including the falsifier that justified this pivot. The C++ hook is **not**
   touched: removing it is a gameplay-affecting engine change that destroys
   measurement capability for zero benefit.
3. **Full validation gate before Episode 0**: environment-property test, teacher
   strength test, then a short from-scratch smoke run. No matched control arm on
   the old curriculum -- `CLAUDE.md`'s own recorded guidance is not to burn a
   training run on retroactive attribution.

---

## 3. Why a symmetric economy restores the win condition

Recorded here because it is the mechanism the whole design rests on, and because
it predicts the shape of the environment-property test's result.

A win condition pays when it arrives while the opponent cannot answer -- inside
the window after they have spent, before they have regenerated the cost of an
answer. With regeneration rate `r` and multiplier `m` that window lasts about
`answer_cost / (m * r)`, so at `m = 1.5` every punish window is two thirds its
natural length. Meanwhile our 4 elixir is spent regardless, and what the opponent
does with their surplus scales with `m`. Expected value moves as `1/m` while
expected counter-cost moves as `m`: the ratio degrades on the order of `1/m^2`.

Defence moves the opposite way. A defensive card's value scales with the volume
of threats it answers, and a multiplier increases exactly that volume. So the
multiplier does not merely shift the optimum -- it **inverts the ranking of
strategy classes**. That is why four policy-side interventions could not work.

**Falsifiable consequence, and this is the test in section 6.1:** at `m = 1`,
banning the win condition should *cost* win rate. At `m = 1.5` it should cost
nothing, or help.

---

## 4. Component design

### 4.1 `python_ai/teacher.py` -- `UtilityTeacher`

**Contract.**

```python
class UtilityTeacher:
    def __init__(self, deck, team, profile=None, horizon_ticks=30,
                 k_cells=2, epsilon=0.0, seed=None): ...
    def reset(self, rng=None): ...
    def act(self, env, obs_own) -> tuple[int, float, float]: ...
```

`act` returns `(slot, x, y)` in the actor's **own mirrored frame**, where
`slot == HAND_SIZE` is the no-op. `env` is the live `ClashRoyaleEnv` (needed only
for `snapshot()`); `obs_own` is `get_observation_for_team(team)`.

**One class plays either side.** `stepSelfPlay` mirrors team 1's y as
`realY1 = (BOARD_HEIGHT - 1) - targetY1` and leaves x alone, and
`get_observation_for_team(1)` is already mirrored to match. So a bot written
purely against "own observation -> own action frame" is side-agnostic by
construction. This is what makes the Teacher-vs-Teacher test in section 6.1
possible at all.

**All spatial reasoning reads the observation, never `getEntities()`.** Same
contract `tactics.py` holds, and for the same reason:
`perception/tests/test_encoder_matches_engine.py` pins the live encoder bit-equal
to `getObservationForTeam(0)`, so anything computed from the observation behaves
identically in simulation and on a real screen. The teacher therefore transfers
to `perception/` unchanged.

**Engine constants are read from the bindings, never re-typed.** `CLAUDE.md`'s
rule, which this project has paid for twice.

### 4.2 Candidate generation (rules)

For each affordable hand slot, propose a small set of cells by card role. Roles
are resolved from `get_card_info` plus the observation's archetype channels --
never from a hardcoded id list, so the teacher survives a deck change.

| role | candidates |
|---|---|
| win condition (building-targeter) | both bridge cells at `BRIDGE_ROW` (`tactics.best_hog_cell` picks the weaker lane; the other bridge is the second candidate) |
| spell | `tactics.best_spell_cell` + top-2 cells of the catch map |
| building | `tactics.best_building_cell` + the centre-pull cell |
| other troop | defensive meet-forward cell on the threat lane; support cell behind our own advancing push if one exists |

Plus the always-present no-op. Typical `K ~ 11`.

**Measured cost** (i5-13420H, this box, 2026-08-19):

```
snapshot          0.0352 ms
10-tick step      0.0598 ms
candidate @3s     0.2500 ms   ->  K=16 costs 4.00 ms/decision
candidate @6s     0.4403 ms   ->  K=16 costs 7.04 ms/decision
```

Gated to fire only when the bot intends to act (~15% of decisions in 2.6), that
is ~0.2 s/episode against a ~2.8 s episode. Inside budget.

### 4.3 Ranking (forward simulation)

Each candidate is rolled `horizon_ticks` forward on an `env.snapshot()`, with both
sides no-op after the play. `CLAUDE.md` records that beyond ~12 s a
both-sides-no-op rollout "stops resembling the game", so 30-60 ticks (3-6 s) sits
safely inside the validated regime.

Score is **differenced against the no-op rollout**:

```
U(c) = w_twr   * d(enemy tower damage dealt by us)
     + w_def   * d(-our tower damage taken)
     + w_trade * (d(elixir value we killed) - d(elixir value they killed))
     + w_crown * d(tower-count differential)
     + w_cycle * cycle_value(card)
```

Two properties make this work and both are deliberate:

- **No-op scores exactly 0**, so every candidate carries a *marginal* value and
  the bot holds elixir whenever nothing is worth doing. This is how the
  blueprint's `- w5 * OpportunityCost` is implemented: as a baseline, not as a
  fifth hand-tuned weight fighting the other four. A separate opportunity-cost
  term would need calibrating against terms measured in different units; a
  baseline needs nothing.
- **Every `d(...)` is an engine statistic**, not a guess. The weights only trade
  off quantities the engine already computes.

`cycle_value` is the one hand-derived term (section 4.4).

### 4.4 Cycle tracking

Our own cycle is **exact and free**: the hand is a deterministic rotation, so
observing hand transitions reconstructs the queue with no engine change and no
new binding. The opponent's is estimated from cards we have seen them play.

It enters as a **scored term, never a hard rule** -- a wrong estimate then
degrades the ranking slightly instead of freezing the bot into a wrong line.
Concretely: a bonus for playing a cheap card when the win condition is >= 2 cards
away, and a bonus on the win condition itself when the opponent's known counters
are estimated to be out of cycle.

`tactics.hog_should_commit` already gates on an opponent-elixir estimate and is
engine-validated; the cycle term supplements it rather than replacing it.

### 4.5 Difficulty ladder -- competence, not economy

Both sides always at 1.0x.

| stage | horizon (ticks) | epsilon | k_cells |
|---|---|---|---|
| 0 | 0 (rules only) | 0.30 | 1 |
| 1 | 0 (rules only) | 0.15 | 1 |
| 2 | 30 | 0.10 | 2 |
| 3 | 30 | 0.05 | 2 |
| 4 | 60 | 0.02 | 3 |
| 5 | 60 | 0.00 | 3 |

`epsilon` is the probability of substituting a uniformly random *legal* action for
the argmax. Stage 0-1 run with no search at all, which is both the weakest and the
cheapest setting -- the early curriculum costs nothing.

The gate is unchanged: raw win rate >= 0.80 over 100 episodes, evaluated **before**
stage advancement (that ordering is load-bearing -- the stage gate calls
`outcome_history.clear()`).

### 4.6 Profile randomization

Per episode, draw a weight profile from `{aggressive, balanced, defensive}` and a
lane bias. The decision rule stays deterministic given the profile. This is the
anti-memorization measure; without it a deterministic opponent invites exactly the
single-counter-line collapse the pivot exists to avoid.

### 4.7 `gym_wrapper.MicroRoyaleEnv` integration

`MicroRoyaleEnv.step` has exactly one engine call site. With a teacher configured
it routes through `step_self_play(card0, x0, y0, slot1, x1, y1, skip_frames)`
instead of `step(...)`; otherwise the env is byte-identical to today.

`stepSelfPlay` deliberately never calls `opponentTurn()`, so the C++
`HeuristicOpponent` is silently absent on that path -- which is exactly what is
wanted here, and is the same fact that forces `BUILTIN_ANCHORS` to use
`MicroRoyaleEnv` rather than the self-play env. Both `step` and `stepSelfPlay`
accumulate `calculateReward()` identically, so the reward stream is unchanged.

### 4.8 `train.py` curriculum

`CURRICULUM_STAGES` loses its `opp_elixir_multiplier` column and gains
`teacher_stage`. The entropy-reset, stage-snapshot and phase-transition machinery
around it is untouched.

`PHASE2_ENTRY_WIN_RATE` keeps its current meaning (it gates `mirror` ->
`random_opponent`, **not** the pipeline handoff, despite the name).

### 4.9 Offensive scenario injection (Proposal A), default-off

New phase-1 scenario families alongside the existing defensive ones:

- `punish_window` -- we hold ~8 elixir with the win condition in hand; the
  opponent has just committed and sits near 1.
- `counter_push` -- our defence has just survived and units remain alive to
  support a counter-attack.

Controlled by its own probability constant, default **0.0**. Enabled only if
section 6.1 passes.

---

## 5. What is explicitly NOT being built

- **No C++ change.** Everything above is expressible through existing bindings:
  `snapshot`, `step_self_play`, `get_hand_for_team`, `get_elixir_for_team`,
  `get_observation_for_team`, and the stats accessors. `CLAUDE.md`'s rule stands.
- **No removal of `set_opponent_elixir_multiplier`.** Decision 2 above.
- **No neural distillation of the teacher in this phase.** `CLAUDE.md` records
  that distilling an expert into a net whose weakness the expert does not share
  returns null (h=12 into the cured net: conditional lift -0.0032 +/- 0.0230,
  paired A/B +0.0250 p=0.608). Distillation is a later question and it is gated on
  the teacher first being demonstrably better than the policy.
- **No change to the reward function.** The pivot is about the environment's
  payoff structure, not the shaping terms. Adding a reward term at the same time
  would make the smoke run uninterpretable.

---

## 6. Validation, in order

### 6.1 Environment-property test -- `python_ai/prove_environment.py`

**The zero-training falsifier, and the cleanest version of the test `CLAUDE.md`
asks for.** No network is involved, so no training, no checkpoint, and no policy
confound.

Teacher vs Teacher. Arm A: full deck. Arm B: identical, except the win condition
is removed from the candidate set (never played, still cycled). Paired via
`snapshot()` so both arms play a bit-identical opening. Swept over opponent
multiplier in `{1.0, 1.25, 1.5}`.

| result | reading |
|---|---|
| banning the wincon **costs** win rate at 1.0x and costs nothing (or helps) at 1.5x | hypothesis supported; the new environment prices the win condition correctly; **proceed** |
| banning it costs nothing at every multiplier | the Hog is unviable in this engine's physics regardless of economy; the pivot does not fix it and the honest response is to stop rehabilitating the card |
| banning it costs win rate at every multiplier | the teacher is too weak for its own economy to matter; fix the teacher before reading anything else |

**Ceiling caveat, carried forward from `CLAUDE.md` and binding here too.** The
result is only interpretable if the baseline arm is below ceiling. If arm A's win
rate is >= 0.95 the arm is void and needs an intermediate setting -- exactly how
the 1.00x row of the original test came out. Check the baseline before reading
the delta.

### 6.2 Teacher strength -- `python_ai/prove_teacher.py`

Two bars, and **both** must be cleared:

1. **Strong enough**: beats the C++ `HeuristicOpponent` decisively at 1.0x. If it
   does not, it is not an upgrade over what phase 1 already had.
2. **Not a wall**: still beatable by `shipping.py`'s config (cured weights +
   search h=12). An opponent that nothing can beat produces a flat reward signal,
   which is the zero-gradient failure in a new costume.

Also reported: teacher vs C++ heuristic at 1.5x, to place it on the existing
ladder alongside every historical number.

### 6.3 From-scratch smoke run

~2-4k episodes from Episode 0 on the new curriculum. Pre-registered metrics,
compared against the values `CLAUDE.md` records for the ep-600 2.6 run:

| metric | ep-600 reference | direction required |
|---|---|---|
| win-condition usage | 0.8% | must rise materially |
| `P(nothing affordable)` | 52.9% | must not regress |
| `P(play \| affordable)`, no threat -> HUGE | 0.4606 -> 0.8601 | reflex must persist |
| `Policy/Entropy_Card_Frac` | ~0.35 target | settles near target |

The win-condition usage number is the one that decides whether the pivot worked.
Everything else is a guard against fixing offence by breaking defence.

### 6.4 Regression

- `python_ai/test_python_ai.py` passes, plus new tests for: teacher side-agnosticism,
  the 1.0x pin, no-op scoring exactly 0, and candidate legality.
- `perception/.venv/Scripts/python.exe -m pytest perception/tests -q` still passes
  (344 tests) -- the teacher shares `tactics.py`, which perception depends on.
- The C++ suite is untouched and must stay at 504 passing.

---

## 7. Consequences to record in `CLAUDE.md`

- **Gameplay-affecting.** The phase-1 opponent changes identity and the economy
  changes. Every win rate earned against `heuristic@{1.0..1.5}x` becomes
  historical and is not comparable to anything produced after this.
- Checkpoints are **not** invalidated -- no observation, action-space or
  architecture change. Weights are being wiped by choice, not by necessity.
- `CLAUDE.md` names `probe_defense.py` and `probe_entropy_norm.py`, neither of
  which exists. The real files are `probe_perfect_defense.py` and
  `probe_card_usage.py`. Correct this in the same doc pass.
