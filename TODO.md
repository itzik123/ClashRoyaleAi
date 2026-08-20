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

## 1. ✅ DONE (2026-08-20) — Utility Teacher evaluates MULTI-CARD COMBO placements

**Built, tested and measured.** `UtilityTeacher` candidates are now SEQUENCES of
placements rather than single cells, and the teacher plans, commits to and
executes two-card combos. Full write-up with every number is in CLAUDE.md,
"2026-08-20 (later): the MULTI-CARD teacher".

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
- **Win rate is a measured NULL across four paired runs** (+0.000, -0.031,
  -0.081, -0.031; every CI spans zero; pooled -0.036 at n=160 against a ~0.057
  half-width). Kept on because it makes a play the engine rewards expressible at
  ~4% throughput, and `max_combos = 0` is action-identical to the old teacher
  (0 mismatches / 919 decisions). **The pooled -0.036 is the thing to
  re-measure at higher n before this teacher fronts a long training run.**
- **A control failed and the reason is reusable:** `--seed` does not make
  `prove_combos.py` reproducible, because `ClashEnv::reset()`'s opening shuffle
  is unseeded (engine request 7, still open). Within a run the snapshot pairing
  is sound; ACROSS runs only the deltas are comparable, never the arm levels.

### What this leaves open, and it is now an ECONOMY question

Combos are chosen on well under 1% of decisions, and the binding constraint is
no longer the candidate generator -- it is that the teacher's bar sits at a p90
of **3.30**, so it can rarely buy a two-card play at all. It spends continuously
because `w_pos` (20.0) credits any cheap troop merely for standing forward: a
1-cost body scores about 1.2 utility for 1 elixir, so almost every cheap card
beats holding.

That is a SCORING question about `PROFILES`, and it should be answered the way
the profiles were meant to be -- `prove_teacher.py --sweep` selects on win rate
against the C++ heuristic and the winner is CONFIRMED on a fresh independent
run. It is deliberately NOT a sixth combo family; the generator is not what is
binding any more.

**Do not read this as "the teacher should hoard".** A flat reserve was measured
dead here, and CLAUDE.md already records that the shipped solvency gate fixed
bankruptcy as a statistic with no outcome gain. The open question is whether
`w_pos` is simply too high, which is a different and cheaper experiment.

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

**Item 7 got more expensive to live without on 2026-08-20.** A combo A/B
control was designed around "same `--seed`, so the OFF arm should reproduce";
it cannot, because the opening shuffle is unseeded, and the mis-specified
control cost a 10-minute run and nearly produced a wrong conclusion about which
combo family was responsible for a trend. See CLAUDE.md's multi-card section.

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
