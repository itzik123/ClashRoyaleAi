# Live teacher-as-agent play — design

**Date:** 2026-08-24
**Goal:** `UtilityTeacher` plays as US (team 0) in a real Clash Royale match —
perception reads the screen, a mirror `ClashRoyaleEnv` is rebuilt from it, the
teacher proposes candidates and ranks them by 5-10 s forward rollouts, and the
winning placement is tapped into the live game.

**Engine dependency:** `perception/UPSTREAM_REQUESTS.md` item 22 (proposed
2026-08-24, awaiting decision). Stage 1 below does not need it; Stage 2 does.

---

## 1. What already exists, and must not be rebuilt

Three findings reshaped this design before it was written. Each is checkable.

**The pipeline is already three threads, and two of them are done.**

| thread | file | status |
|---|---|---|
| producer — capture, detect, GameState, elixir ledger | `perception/live/pipeline.py` | exists |
| decision — encode, policy, act, at `DECISION_HZ = 1.0` | `perception/live/mvp_loop.py` | exists; gains a new policy |
| actuator — depth-1 queue, drops rather than blocks | `perception/live/actuator.py` | exists |

**Steps 2 and 3 of the naive pipeline are one step.** `UtilityTeacher` *is*
"rules propose, simulation ranks" — `rollout_stats` already calls
`env.snapshot()` and rolls each candidate forward internally. There is no
separate "evaluate the teacher's options" stage to build, and building one would
simulate everything twice.

**There is no rollout latency problem to solve.** Measured figures already in
CLAUDE.md and `realtime_search.py`:

| | cost |
|---|---|
| `UtilityTeacher` decision, stage 5, `play_margin=3.0` | **14.1 ms** |
| `reset()` | 0.135 ms |
| `snapshot()` | 0.021 ms p50 |
| `step(10 ticks)` | 0.020 ms p50 |
| **decision period** | **1000 ms** |

The mirror rebuild plus the teacher is ~15 ms, i.e. **~1.5% of the budget.**

---

## 2. Architecture

```
  producer thread                decision thread (1 Hz)           actuator thread
  ---------------                ----------------------           ---------------
  capture                        latest Snapshot
    -> detect                      -> MirrorBuilder.build()  ~1 ms
    -> GameState                   -> teacher.act(env, obs)  ~14 ms
    -> publish Snapshot  --------->-> Decision                ----> enqueue (depth 1)
                                                                    -> select slot tap
                                                                    -> tile tap
```

### Why not "continuously stream into C++"

Rejected, for three reasons in descending order of hardness.

1. **The GIL is never released.** `src/bindings.cpp` contains no
   `gil_scoped_release` and no `call_guard` — verified by grep. Every engine
   call holds the GIL for its whole duration, so a background rollout thread
   does not run *beside* perception; it **blocks** it. Threading the rollout
   converts a 14 ms cost into a 14 ms stall of the detector.
2. **It breaks the comparison the teacher depends on.** The teacher snapshots
   once and rolls every candidate from that snapshot. A second thread mutating
   the env mid-decision would score candidates against *different worlds*, so
   the ranking — the entire output — would be meaningless.
3. **There is no budget pressure to justify the risk.** ~1.5% of the period.

**So "streaming" is: rebuild the mirror from the latest published Snapshot at
each decision.** Discrete, single-threaded, ~1 ms.

### What that choice buys for free

`forecast.py` lists four unclosed reconstruction gaps. Rebuild-per-decision
**deletes one of them outright**: "an entity perception no longer sees cannot be
deleted" cannot bite when nothing is carried forward. No removal API is needed,
and item 22 does not request one.

### The escape hatch, deliberately not built

If the budget ever tightens, `python_ai/search/realtime_search.py` already
implements the deadline-bounded anytime pattern (compute a safe action first,
then improve until a wall-clock deadline). Wiring it is a later, separate
decision. Building it now would be speculative.

---

## 3. New components

Both live in `perception/`, the freely-editable area. **No file under
`python_ai/`, `include/` or `src/` is modified by this design.**

### 3.1 `perception/live/mirror.py` — `MirrorBuilder`

One job: turn a perception `Snapshot` into a `ClashRoyaleEnv` positioned as
closely as the bindings allow, and **report what it could not set**.

```
build(snapshot) -> (env, MirrorGaps)
```

- `reset()`, then `inject()` every perceived unit at its engine-convention tile.
- `set_elixir_for_team(0, my_elixir)` and `set_elixir_for_team(1, opp_elixir)`.
- `set_hand_for_team(0, hand)` — **the bool return is checked**. It refuses
  rather than accepting a misread, and a silently-ignored refusal is worse than
  no update at all.
- `MirrorGaps` records, per build, exactly what was fabricated: tower HP,
  per-unit HP, clock, and (Stage 1 only) the deploy-time subsidy.

`MirrorGaps` is not decoration. It is what lets us measure whether item 22 is
worth what it costs, instead of asserting it.

**Why a new module rather than extending `forecast.py`:** `forecast.py` is the
fidelity-measurement path and two harnesses score against it
(`tools/sim_fidelity.py`, `tools/measure_decoupling.py`). It consumes exactly
one field, `game_state.units`, and its metrics are defined tower-excluded
precisely because the rest is fabricated. Changing what it reconstructs would
move numbers those harnesses report.

**What is reused, concretely, rather than reimplemented.** `MirrorBuilder`
imports two things from `forecast.py`:

- **`bodies_per_card`**, and the group-by-`(card_sim_id, team)` step in front of
  it. A card that spawns three bodies must be injected **once per three detected
  bodies**, not once per body — reimplementing that wrong gives a mirror with
  3x the Skeletons, which is a threat over-estimate that would look like a
  teacher that panics.
- **The ordering constraint that comes with it:** `bodies_per_card` **resets the
  env itself**, so every body count must be resolved *before* the `reset()` that
  begins the build. Calling it mid-injection wipes the board being built. This
  is a landmine in a helper's semantics, not a visible parameter, and is the
  single strongest argument against a second copy of the loop.

What `MirrorBuilder` owns and `forecast.py` does not: the elixir and hand
setters, `MirrorGaps`, and (Stage 2) tower HP, clock and deploy-time.

### 3.2 `perception/live/teacher_policy.py` — `TeacherPolicy`

Implements the existing policy interface, so it drops into the loop beside
`ScriptedPolicy` and `NeuralPolicy` with no change to the loop's shape:

```
decide(gs, ready, now) -> Decision
```

Holds one persistent `UtilityTeacher(deck, team=0)` — persistent because the
teacher carries `self.pending`, the second half of a committed combo, across
decisions. The **env** is rebuilt per decision; the **teacher** is not.

### 3.3 Wiring

`--policy teacher` in `mvp_loop.py`, alongside the two existing policies.
`--act` remains opt-in, so a forgotten flag watches rather than plays.

---

## 4. Correctness invariants

These are the joints where this integration can be silently wrong.

**Hand-slot alignment.** The teacher returns a `slot` index into the mirror's
hand; the actuator taps a slot index in the real hand. These are the same number
**only if `set_hand_for_team` succeeded.** On refusal the decision must be
dropped, not played — a placement against a misaligned hand plays a card we did
not choose.

**Decision cadence is coupled to the teacher's plan clock.** `pending_ticks -=
COMBO_FOLLOWUP_DELAY_TICKS` (10 ticks = 1.0 s) runs **once per `act()` call**,
so the teacher assumes exactly one decision per second. `DECISION_HZ = 1.0`
matches today. But a *dropped* decision (a perception stall) advances the plan
by 1 s of teacher-time while 2 s of wall time passed, and a 3 s planned escort
gap silently becomes 4 s. The loop must log decision-interval drift, and
`FRESHNESS_WAIT_CAP_S = 0.15` puts a known 15% jitter on top. If drift proves
material, the fix is a proposal against `teacher.py`, not a local patch —
`python_ai/` is read-only.

**Tower HP is a fraction, never an absolute.** Engine towers are level 9 (2534);
the recordings are levels 4-5 (1750 ours, 1890 theirs) — wrong by a *different*
factor per player. Perception reports a fraction; `MirrorBuilder` multiplies by
the engine's own max. The level knowledge stays on the perception side, where it
already lives.

**The env has exactly one owner thread.** Enforced by construction: only the
decision thread ever holds a reference.

**Dry run is the default.** Acting requires `--act`.

---

## 5. Staging

### Stage 1 — no engine change (buildable today)

Everything in §3, using only bindings that already exist. Elixir and hand are
real; tower HP, unit HP and clock are fabricated and *recorded as fabricated* in
`MirrorGaps`.

Stage 1 is not a throwaway. It is the harness that measures what Stage 2 buys —
the same discipline the deploy-time change used, where the constant was zeroed
and restored to get a controlled before/after rather than a claim.

### Stage 2 — after item 22 lands

`MirrorBuilder` gains four lines: `set_tower_hp`, `destroy_tower`,
`set_current_tick`, and `deploy_ticks=0` on every inject of an
already-on-screen unit. `MirrorGaps` shrinks to what remains (units perception
scored as undamaged but which are not — recall 0.34-0.56).

**Expected order of impact, stated in advance so it can be wrong:** deploy time
first (a 1 s inertness subsidy on every enemy unit; the same second was measured
at ~520 tower HP in the 2026-08-19 controlled A/B), then tower HP (it decides
whether a finishing rollout registers a crown), then unit HP, then the clock
(smallest — the engine models no double elixir, so a correct clock still
misprices late-match rollouts).

---

## 6. Error handling

| condition | response |
|---|---|
| `set_hand_for_team` returns `False` | drop the decision, count it, do not play |
| snapshot staler than `MAX_STALENESS_MS` | existing `action_gate` refuses; unchanged |
| teacher raises | catch, count, fall through to no-op; never kill the loop |
| perception thread dies | existing detection in `mvp_loop`; unchanged |
| actuator queue full | existing depth-1 drop; unchanged |
| a perceived unit maps to no sim id | skip that unit, count it in `MirrorGaps` |

The bias throughout is toward **no-op over a wrong action**: in a real match a
mistimed card is worse than a skipped decision.

---

## 7. Testing

Testable with no emulator, against frozen fixtures — matching the existing
perception suite (353 pass / 1 skip):

- `MirrorBuilder` on a frozen `GameState` fixture produces the expected
  occupancy in the observation.
- `set_hand_for_team` refusal path drops the decision (fault injected).
- `MirrorGaps` reports every fabricated field; a test fails if a new gap is
  added without being recorded.
- `TeacherPolicy.decide` returns a legal `Decision` for a fixture board.
- Decision-interval drift logging fires when intervals are skewed.
- Stage 2 only: a unit injected with `deploy_ticks=0` moves on tick 1.

The end-to-end path against a live emulator is **not** unit-testable and is
validated by a dry-run session with `MirrorGaps` logging, which is what Stage 1
is for.

---

## 8. Open questions

1. **Opponent hand is unobservable.** `set_hand_for_team(1, ...)` has no source —
   perception cannot see the opponent's hand. The mirror's team-1 hand stays
   fabricated. This matters less than it sounds (the teacher rolls candidates
   forward with the opponent no-oping anyway), but it caps what a reactive
   rollout could ever do here.
2. **Double elixir is unmodelled.** Filed separately; noted in item 22.
3. **Is 1 Hz enough for the teacher's combos?** A planned escort gap is
   expressed in decisions, so decision rate and tactical timing are the same
   knob. Not resolvable before Stage 1 data.
