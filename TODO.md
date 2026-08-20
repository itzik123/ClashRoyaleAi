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

## 1. ⭐ Upgrade the Utility Teacher to evaluate MULTI-CARD COMBO placements

**This is the top item by expected value. It is the one thing the engine now
rewards and no component can express.**

### Why it is first

The 2026-08-19 deploy-time change (`DEPLOY_TIME_TICKS = 10`) made the *escorted*
push the correct play and the *naked* push the punished one. Measured on the
same engine, same harness, same protocol:

| commitment | marginal value | 95% CI |
|---|---|---|
| lone win condition | **−556.3** tower HP | — |
| supported push (tank one decision ahead) | **+448.5** tower HP | [+137.3, +760.1] |
| escorting, in a punish window | **+650 HP** | [+429, +878] |

`UtilityTeacher` cannot make that play. `_cells_for` proposes cells for **one
card per decision** and `score` ranks single candidates, so its entire attack
repertoire is "send the win condition to a bridge, alone" — precisely the play
the new physics correctly punishes.

That is why the strategy-level arm still reads attack 0.490 vs cycle 0.715.
**That number is now a property of the teacher's repertoire, not of the
engine**, and it is the one place the two can still be confused.

### What it needs

- Candidate generation over short **sequences** rather than single cells — at
  minimum *(tank now, win condition next decision, same lane)*. Today
  `candidates()` emits a flat list of single-card `Candidate(slot, cid, x, y,
  role)` objects, and `_cells_for`'s own docstring says "1-3 tactically sensible
  cells for **one card**".
- A score that can attribute value to the pair. `score()` currently ranks one
  candidate against the no-op baseline.
- **The rollout machinery already supports it, and the change is small.**
  `rollout_stats` takes `env.snapshot()`, plays the candidate with one
  `step_self_play(slot, x, y, ..., 10)`, then no-ops the remaining horizon in
  10-tick chunks. A two-card sequence is the same loop with the second card
  played into one of those chunks instead of a no-op — no new engine capability
  is needed.

### The cost to watch

Width is what search is expensive in — one engine step is 0.015 ms, but each
extra candidate is a whole rollout plus a network row at ~0.13 ms. **Enumerate a
handful of curated combos, not the cross product.** Depth is nearly free; that
is why `TEACHER_STAGES` progresses on horizon (0 → 100 ticks) and not on width.

### Do NOT confuse this with a fifth Hog mechanism

Four policy-side mechanisms have been built and measured null (a reward
multiplier, an advisor target, random forcing, gate-timed smart forcing). All
four tried to move a **policy** toward a play the environment priced negatively.
This is the opposite situation: the environment now prices the play
**positively** and the teacher simply cannot express it.

**Files:** `python_ai/opponents/teacher.py` (`_cells_for`, `candidates`, `score`,
`rollout_stats`). **Harness:** `python_ai/eval/prove_environment.py --mode marginal`
— and read the *supported* arm, never the lone-Hog arm, which is now a strawman.

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

## Engine requests still open (`perception/UPSTREAM_REQUESTS.md`)

| # | Request | Status |
|---|---|---|
| 3 | King Tower has no activation condition | open, **already worked around — no change requested** |
| 7 | The engine's RNG cannot be seeded | **open** — additive, not gameplay-affecting |
| 8 | Fireball (689) misses the Musketeer kill (721 HP) by 32 | **open — a decision, not a defect** |

**Item 7 is worth doing and is cheap.** `env.snapshot()` (2026-08-11) removed it
as the blocker on *paired* A/B tests, but not on **reproducible failures**. Live
evidence from this session: `test_python_ai.py` skipped a different number of
tests across two identical runs, because one of them
(`pytest.skip("Cannon not in the opening hand this shuffle")`) depends on the
unseeded `mt19937`. A test suite whose skip count is nondeterministic is exactly
the cost this item describes.

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
