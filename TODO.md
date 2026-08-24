# TODO — the single list of genuinely pending work

Consolidated 2026-08-19 from `CLAUDE.md`, three session handoffs,
`PLACEMENT_COLLAPSE.md`, two executed plans, `perception/README.md`,
`perception/UPSTREAM_REQUESTS.md` and `perception/BOT_REQUESTS.md`.

**Everything here was verified against the source tree, not copied from a
document's own status field.** Items whose work turned out to be already merged
were deleted rather than carried forward — three of them had been sitting in
`UPSTREAM_REQUESTS.md` marked OPEN while the exact code they proposed was in the
tree.

Rules that apply to every item below:

- `include/`, `src/` and `python_ai/` are READ-ONLY unless explicitly asked.
  C++ changes need the exact diagnosis and the exact edit confirmed **first**,
  and go to `perception/UPSTREAM_REQUESTS.md` as a proposal.
- Any gameplay-affecting change invalidates the win-rate history. Say so.
- Python is 3.11 only: `python_ai/venv/Scripts/python.exe`.

---

## 0. NEXT UP — Stage 2: the live teacher-as-agent loop

**The engine half landed 2026-08-24** (`UPSTREAM_REQUESTS.md` item 22,
APPLIED). **Only Python remains, and all of it is in `perception/`, which is
freely editable.**

Design: `docs/superpowers/specs/2026-08-24-live-teacher-play-design.md`.
The measured deploy-time result behind it: `DECISIONS.md`, "2026-08-24: the
live-mirror state setters". Stage 1 (a passive gap-logger) was **deliberately
skipped** by the human once item 22 was approved.

**Goal:** `UtilityTeacher` plays as US (team 0) in a real match. Perception
reads the screen → mirror env → teacher proposes AND ranks by 5-10 s rollouts
→ the best placement is tapped.

**Two new modules, neither of which exists yet:**

1. `perception/live/mirror.py` — `MirrorBuilder.build(snapshot) -> (env, MirrorGaps)`.
   `reset()` → `inject(..., hp=frac*full, deploy_ticks=0)` per unit →
   `set_elixir_for_team` both sides → `set_hand_for_team(0, hand)` **and CHECK
   ITS BOOL** → `set_tower_hp` / `destroy_tower` per tower → `set_current_tick`.
2. `perception/live/teacher_policy.py` — `TeacherPolicy.decide(gs, ready, now)
   -> Decision`, holding ONE persistent `UtilityTeacher(deck, team=0)` (it
   carries `self.pending` across decisions) while the ENV is rebuilt per
   decision. Wired as `--policy teacher` beside `ScriptedPolicy`/`NeuralPolicy`.
   **`--act` stays opt-in; dry run is the default.**

**Do not add threads.** The pipeline is already three (producer in
`live/pipeline.py`, decision loop at `DECISION_HZ = 1.0`, actuator with a
depth-1 drop queue). `src/bindings.cpp` releases the GIL **nowhere**, so a
rollout thread would BLOCK perception rather than run beside it, and a
concurrent writer would score candidates against different worlds. Budget is
not the issue: teacher **14.1 ms** + rebuild ~1 ms against a **1000 ms**
period. "Streaming" = rebuild the mirror from the latest Snapshot at each
decision — which also deletes `forecast.py`'s entity-removal gap for free.

**The three joints where this goes silently wrong:**

- **Ordering.** REUSE `forecast.py`'s `bodies_per_card` and its
  group-by-`(card_sim_id, team)`. It **resets the env itself**, so every body
  count must be resolved BEFORE the build's `reset()`. Getting it wrong gives
  3x the Skeletons — which reads as "the teacher panics", not as a bug.
- **Hand-slot alignment.** The teacher's `slot` indexes the mirror's hand; the
  actuator taps the real one. Same number ONLY if `set_hand_for_team` returned
  True. On refusal **drop the decision** — a placement against a misaligned
  hand plays a card nobody chose.
- **Cadence.** `pending_ticks -= COMBO_FOLLOWUP_DELAY_TICKS` (10) runs **once
  per `act()` call**, so the teacher assumes exactly 1 decision/second. A
  DROPPED decision advances the plan 1 s in teacher-time while 2 s of wall time
  passed, skewing every combo gap. Log decision-interval drift. Any fix belongs
  in a proposal — `python_ai/` is read-only.

Bias throughout: **no-op over a wrong action.** A mistimed card is worse than a
skipped decision.

**Known limits, already measured — do not rediscover.** Unit-HP **recall
0.34-0.56** at precision 0.98, so ~half of damaged units still arrive at full
health. The opponent's hand is unobservable, so team 1's stays fabricated. The
engine models **no double elixir** (`ELIXIR_REGEN_RATE` is constant), so late
rollouts stay mispriced even with a correct clock — filed as its own item, NOT
folded into 22.

---

## 1. ✅ DONE (2026-08-20/21) — Utility Teacher evaluates MULTI-CARD COMBO placements, and can now afford them

**Built, tested and measured.** `UtilityTeacher` candidates are now SEQUENCES of
placements rather than single cells, and the teacher plans, commits to and
executes two-card combos. Full write-up with every number is in
`DECISIONS.md`, "2026-08-20 (later): the MULTI-CARD teacher".

The short version:

- `Candidate` carries a tuple of `PlacementStep`s; `.slot/.x/.y` still mean the
  first step, so nothing downstream changed. Six curated families
  (`supported_push`, `counter_push`, `defensive_stack`, `cheap_defence`,
  `spell_then_push`, `push_then_spell`), round-robin under a `max_combos`
  budget that is a new competence axis in `TEACHER_STAGES` (0/0/2/3/4/4).
- **No C++ change was needed.** A 0-tick `step_self_play` places nothing and
  `gym_wrapper` gives the teacher one action per decision, so a combo is a
  sequence across consecutive decisions -- which is the shape the +448.5 tower
  HP "supported push" was measured at anyway. Per-entity deploy time verified
  from Python.
- **The generator alone was not enough**, and that is the transferable result: a
  pair priced at 5-6 elixir was affordable in 2 of ~2,400 decisions. What fixed
  it was making the follow-up GAP a searched axis (1/3/5 s), because the pair's
  cost is paid across the gap. A flat savings charge was tried first and is
  measured DEAD -- do not re-propose it.
- **Win rate: SUPERSEDED. Pooled over five paired runs (200 openings) the combo
  machinery costs about 4 points against a mirror** -- deltas +0.000/-0.031/
  -0.081/-0.031/-0.056, pooled -0.0398, CI [-0.076, -0.004]. It was called a
  null at n=40 and that was a power limit, not a result. Kept on as a
  REPERTOIRE choice (the engine rewards escorted pushes; the teacher still beats
  the C++ heuristic 1.000 and the old teacher 95-5), with `max_combos = 0` and
  `combo_families` as one-line off switches. **The lever is completion: 60% of
  chosen combos leave a first card down for a plan that never finishes.**
- **A control failed and the reason is reusable:** `--seed` did not make
  `prove_combos.py` reproducible, because `ClashEnv::reset()`'s opening shuffle
  was unseeded. Within a run the snapshot pairing is sound; ACROSS runs only the
  deltas were comparable, never the arm levels. **The engine half was fixed on
  2026-08-21** — `ClashEnv::seed()` seeds both generators and re-deals — **but
  `prove_combos.py` still never calls it**, so the caveat holds for the harness
  exactly as it stands today. That migration is item 8 below.

### The economy follow-up: DONE 2026-08-21

`play_margin` 0.05 -> 3.0, plus an overflow taper and a follow-up exemption.
Combo share of plays 2.2% -> 14.4%, the old teacher loses ~94% head to head, and
stage 5 still scores 1.000 against the C++ heuristic. Full write-up in
`DECISIONS.md`, "2026-08-21: the teacher's ECONOMY".

**`w_pos` was the obvious lever and is MEASURED WRONG -- do not re-propose it.**
Swept 20 -> 8 it raises elixir (1.83 -> 2.67) and drives combo share to
1.1% -> 0.0/0.0/0.2/0.0/0.2%. It prunes plays by HP-per-elixir, and an escorted
push (388 HP/elixir) sits below a naked Hog (424), so it kills the combo before
the cheap cards it was meant to replace.

**What is still open, and it is narrow.** The combos that actually complete are
DEFENSIVE (`cheap_defence`, `spell_then_push`, `defensive_stack`); the escorted
win-condition push is 3 of 47. That is a hand-co-occurrence and price problem,
not a scoring one -- the tank and the win condition are both in hand on ~4-9% of
decisions and the pair is the deck's most expensive. Worth knowing before
anyone reads "14.4% of plays are combos" as "the Ice Golem + Hog push is now
standard".

---

## 2. Wire decision-time search into the live loop

**The largest measured gap between the benchmark and real play**, and plausibly
small.

Search is **not** in `perception/live/mvp_loop.py` — verified 2026-08-19: the
neural path builds the observation through `perception_encoder` and calls the
policy head directly. There is no candidate rollout and no engine snapshot in
that path.

So the configuration measured at **0.9225** cannot be run live. Live play gets
the net **greedy**, which measured **0.5200** on the same opponent.

The loop already owns a simulator mirror — `perception/forecast.py` holds a real
`ClashRoyaleEnv` and calls `get_observation_for_team(0)` — which is exactly the
object `_search_action` needs to snapshot.

**Caveat:** both those numbers were measured on the retired Giant-deck net. The
*mechanism* is what transfers; re-measure the magnitude before quoting it.

**Files:** `perception/live/mvp_loop.py`, `perception/forecast.py`,
`python_ai/search/realtime_search.py`, `python_ai/shipping.py`.

---

## 3. Human-replay imitation — extraction is the blocker

The recordings exist (8 matches in `perception/assets/recordings/`). The
`bc_pretrain` `.npz` pipeline is built and verified end to end. The missing step
is `perception/` → that schema.

Self-play discovers strategies but not *the distribution humans play*;
AlphaStar's supervised stage was load-bearing, not optional.

**Blocked by:** the recordings use the Giant deck (Valkyrie, Archers, Minions,
Cannon, Fireball, Giant, Musketeer, Mini P.E.K.K.A) and `DEFAULT_DECK` is now
2.6 Hog Cycle. This tie was knowingly given up on 2026-08-16. Either record new
matches on 2.6, or accept cross-deck transfer as a separate question.

**Files:** `python_ai/trainers/bc_pretrain.py` (`DATASET_SCHEMA`), `perception/`.

---

## 4. Remaining observation gaps

Both are real, both are cheap to state and not cheap to fix.

- **Cells overwrite rather than accumulate** in channels 0–7
  (`obs[idx] = normalizedHp`), so a Skeleton Army collapses to one body.
  `CH_COUNT` mitigates but does not fix it.
- **No card-cycle tracking** — verified 2026-08-19: `model.py` has no cycle
  channel at all, while `teacher.py` carries a working `CycleTracker`. This is
  the single most deck-specific gap, because **2.6 is *defined* by cycling back
  to the Hog faster than the opponent cycles their answer**, and the net can see
  only the 4 cards currently in hand.

Both are engine-side (`extractObservationForTeam`) → propose in
`UPSTREAM_REQUESTS.md` first. Changing `NUM_CHANNELS` **invalidates every
checkpoint**.

---

## 5. `skip_frames = 10` — one decision per second

A hard ceiling on tactical precision. Pulling a Hog with a Cannon and timing an
Ice Spirit are sub-second decisions, so this binds harder for 2.6 Hog Cycle than
it did for Giant beatdown.

Left alone so far because changing it is an **unmeasured throughput/precision
trade**, not because it is fine. The 2026-08-07 movement-speed fix shrank the
damage by ~5× for free (one second now covers ~5× less board).

---

## 6. One deck, mirror matchups

"A great player" implies arbitrary matchups. Phase 1's `random_opponent` and the
scripted bots' randomised decks are partial; phase 2's neural opponents all play
`DEFAULT_DECK`.

---

## 7. Shaping is a hand-designed proxy and caps the ceiling

Non-PBRS terms (`W_TOWER_DESTROYED`, `W_FLAWLESS_DEFENSE`, the spell-value term)
bias the optimum by construction — deliberately, and each is documented with
why. Troop *damage* is also the wrong metric: Clash is decided by kills and
elixir advantage, not HP chipped. Long term these should anneal toward zero.

---

## 8. ⚠ The engine seeding fix is APPLIED but UNVERIFIED, and one path is still unseedable

Engine request 7 landed 2026-08-21 (`ClashEnv::seed`). **No Python consumer was
migrated to it for three days**, so the benefit the item was argued from —
reproducible runs and reproducible failures — had still not been collected.
Two consumers were migrated on 2026-08-24; a third path cannot be, and needs an
engine change. Full write-up: `perception/UPSTREAM_REQUESTS.md` item 23.

**Applied 2026-08-24, in the read-only tree, at explicit instruction:**

- `python_ai/envs/gym_wrapper.py` — `reset(seed=...)` accepted a seed and
  dropped it. `super().reset(seed=seed)` seeds the *wrapper's* `np_random`,
  which this env reads nowhere. Now forwards to `self.game.seed(seed)`, guarded
  on `is not None` so a run does not collapse to one repeated episode.
- `python_ai/eval/prove_combos.py` — five harnesses built each opening with
  `CE(...).reset()` and never called `seed()`. All five now use
  `.seed(args.seed + ENGINE_SEED_OFFSET + i)`; `seed()` ends in `reset()`, so
  it is a drop-in. The stale module docstring was corrected and the old warning
  kept as the acceptance criterion.

**THE VERIFICATION IS THE PENDING PART, and it is the whole item.** Nothing was
run. The machine this was done on has no Python 3.11, no venv and no built
`.pyd`, so nothing importing the engine executes there at all — only
`py_compile` under 3.13. Two checks settle it, and neither has been done:

```python
a, b = MicroRoyaleEnv(cfg), MicroRoyaleEnv(cfg)
assert (a.reset(seed=7)[0] == b.reset(seed=7)[0]).all()
```
```bash
# the 2026-08-20 control that failed, re-run: the OFF arms must now agree on
# LEVEL, not merely on delta
python_ai/venv/Scripts/python.exe -m python_ai.eval.prove_combos --seed 300 ...
```

Until both pass, **the old rule stands: across runs compare deltas only, never
arm levels.** Item 1's caveat above is written that way on purpose.

**Still unseedable, and it needs C++:** `sample_random_deck` draws from a
function-local `static std::mt19937` seeded from `std::random_device`
(`src/bindings.cpp:292`). `ClashEnv::seed` cannot reach it. So a run with
`randomize_opp_deck=True` now has a reproducible hand, cycle and heuristic roll
and a **still-random opponent deck** — the largest of the four variance
sources. `UPSTREAM_REQUESTS.md` item 23C proposes an optional seed argument;
additive, not gameplay-affecting, no checkpoint invalidated.

---

## Engine requests still open (`perception/UPSTREAM_REQUESTS.md`)

| # | Request | Status |
|---|---|---|
| 8 | Fireball (689) misses the Musketeer kill (721 HP) by 32 | **open — a decision, not a defect** |

**One row, and its own recommendation is to change nothing — so the effective
count of open engine requests is zero.** Items 3 and 7 sat here as "open" until
2026-08-24 and both had in fact landed. Recorded so neither is re-proposed:

- **Item 3 — King Tower activation: APPLIED 2026-08-21.** `Tower` carries a
  latching `awake` flag; the King constructs asleep, `findTarget` returns
  `nullptr` while asleep, and it wakes permanently on any damage or on a
  friendly Princess falling. Measured at 1268 tower damage awake vs 2219
  dormant. `tests/core/test_king_activation.cpp`.
- **Item 7 — engine seeding: DONE 2026-08-21.** `ClashEnv::seed(s)` seeds
  `ClashEnv::rng` *and* `GameManager::rng` (an XOR offset keeps the two streams
  from correlating) and then re-deals — `initializeDeck` runs inside
  `GameManager::reset()`, so a seed applied after construction would otherwise
  be a silent no-op. Bound as `seed`; eight test modules call it, and the
  nondeterministic `pytest.skip("Cannon not in the opening hand this shuffle")`
  this item was argued from no longer exists in the tree.
  **The Python consumers were never migrated — that is item 8 above, and it is
  a Python task, not an engine request.**

**Item 8's recommendation is option 1 — change nothing.** 689 and 721 appear to
be faithful tournament-standard values, and `perception/` exists specifically to
drive this simulator from real matches. Reach the behaviour through shaping.

---

## Perception-side open work

From `perception/README.md` and `perception/BOT_REQUESTS.md`.

1. **Stage 3 — opponent placement detection is blocked on data.**
   `detect/placements.py` raises. Needs the next recording batch.
2. **Card identity is measurably wrong.** The icon template agrees with the
   elixir ledger on card cost only **33.8%** of the time (328 in-match plays)
   and over-predicts Giant at 35% against a 12.5% prior.
3. **Phase boundaries unknown.** `clock/match_clock.py` raises
   `PhaseScheduleUnknownError` until real single/double/triple/overtime
   boundaries are supplied.
4. **Card map review — 52 rows.** 41 Evolution pairings and 11 ids ≥ 165 need
   confirming against the build being recorded.
5. **The ledger/`unconfirmed` mismatch.** 5 of 28 post-fix placements still go
   unconfirmed. Remaining suspects: the ~1.6 s decide→land round trip, and the
   integer elixir reading. Measure with `tools/placement_truth.py`'s oracle
   rather than the ledger, since the ledger is the thing under suspicion.
   **Do not re-open "the game refuses our placements"** without new live
   evidence — measured at 0/28 across two matches.
6. **Train against perception-shaped observation noise** (`BOT_REQUESTS` item 1)
   — reasoned, not measured. Deferred: harness yes, training change not yet.

---

## Deliberately NOT on this list

Recorded so they are not re-proposed. Each was measured, not assumed.

- **A fifth Hog-specific policy mechanism.** Four returned null. The open
  question is the engine's offence/defence cost balance, and deploy time was the
  answer to it — item 1 is the follow-through, not a fifth mechanism.
- **Re-distilling search into a cured seed.** Refuted three ways on 2026-08-16
  (conditional lift −0.0032, selectivity 0.92, paired A/B p = 0.608). Search and
  the placement cure repair the same weakness and do not stack.
- **Scaling the distillation dataset.** Actively harmful — selectivity falls
  3.07 → 2.31 → 2.17 as coverage rises.
- **Re-adding heuristic exposure to the PFSP pool.** Already 20.8% realized, and
  it regressed anyway.
- **A from-scratch AlphaZero loop with the current search module.** Inert at
  init — search scores candidates with the net's own critic, so at random init
  the expert is not an expert (21.5% deviation, zero gain).
- **The soft/neighbourhood advisor target (A1).** Measured neutral-to-worse.
  Resolution was the binding constraint, not target softness.
- **Re-enabling the exploiter.** Off since 2026-08-11. Its documented re-enable
  precondition is an engine/shaping change, not a Python one.
- **Penalising back-row or low-shot structures.** `P(shots ≤ 3)` was **100.0%**,
  so the predicate has zero variance and no spatial gradient. *A penalty cannot
  move a distribution with no mass to move.*
- **Leading a spell target.** `lead=0` captures 75.5% of achievable value,
  `lead=10` only 52.4%. `tactics.py` defaults to 0; do not "fix" it.
