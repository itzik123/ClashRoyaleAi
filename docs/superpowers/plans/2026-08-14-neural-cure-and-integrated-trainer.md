# Neural Placement Cure + Integrated Training Pipeline — Implementation Plan

> **For agentic workers:** executed inline in this session (no subagent dispatch —
> the operator's directive forbids delegation prompts and the AgentTool was not
> requested). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the tactical override with a network that places Fireball,
Cannon and Giant correctly on its own, by giving the placement-coverage term a
TARGET instead of noise, then train it in a single loop that carries PFSP,
scenario injection, the advisor target and search-distilled labels together.

**Architecture:** The coverage slot moves from "resampled inside every PPO
epoch" to "sampled once at rollout time and buffered with an advisor target
computed at that moment". Rows that carry a target get KL to the advisor's
masked score map; rows that do not keep today's entropy bonus. Nothing touches
`new_logprobs`, so the PPO ratio is unchanged and the term stays a regularizer.

**Tech Stack:** Python 3.11 (`python_ai/venv`), torch 2.13.0+cpu, pybind11
engine `clash_royale_env.pyd`, pytest.

## Global Constraints

- **`include/` and `src/` are READ-ONLY this session.** CLAUDE.md forbids C++
  changes without the human confirming diagnosis and exact edit first, and the
  human is offline for 25 h. Any C++ finding goes to
  `perception/UPSTREAM_REQUESTS.md` as a proposal, not a commit.
- **No emulator / BlueStacks.** Simulator and offline fixtures only.
- Python 3.11 only: `python_ai/venv/Scripts/python.exe`. The default `python`
  is 3.14 and fails with `ImportError: DLL load failed`.
- Any change to the reward is **gameplay-affecting**: prior win rates stop
  being comparable and CLAUDE.md must say so.
- Resume ONLY through the full-checkpoint path. Seeding a bare `state_dict`
  resets `episodes_completed` to 0, which re-arms the placement entropy target
  at 0.65 and dissolves the policy (measured, twice).
- The coverage/advisor term must never enter `new_logprobs`. Pinned by test.
- `F.kl_div` over `-inf` cells is `nan`; use `distill_tactics.masked_kl`.
- `place_ctx_hi` matches the `place_ctx` prefix. Any `startswith("place_ctx")`
  parameter selection silently trains half the hi-res branch.

---

## Two premises in the mission brief that the repo contradicts

Recorded here because the plan deviates from the brief on both, deliberately.

**1. "From scratch / zeroed weights for the new architecture."** The new
architecture does **not** invalidate checkpoints. `place_hires` is
zero-initialized in weight *and* bias, so it is an exact no-op at init and
v1.2.0's loader reports `warm-started 27/27 tensor(s), re-initialized: []`.
That was a design goal, not luck (handoff §2.1). Meanwhile phase 1 to ~60k
episodes costs ~46 h post-speed-fix and this session has ~25 h, so a
from-scratch run cannot reach the maturity at which placement quality is even
measurable — which would make the mission's own final deliverable ("prove the
placement is cured") unreachable.

Resolution: the **shipping run warm-starts from `model_weights_selfplay.pth`
through the full-checkpoint path**. A genuinely-fresh-weights arm runs
alongside at a smaller budget as Task 9, because it answers a real question the
warm-start cannot — whether the advisor target *prevents* the collapse from
ever forming, rather than repairing one. Both are reported.

**2. "Integrate decision-time search into the training flow."** Search inside
the rollout is measured at 2.2× wall clock and is off-policy w.r.t. the PPO
ratio. The measured-positive use is offline: value-**distribution** distillation,
+0.045 win rate (p = 0.0074), where hard labels gave +0.016 (p = 0.55) and more
data was actively harmful. So search is wired as a periodic offline label
burst (Task 8), not as a per-step override, and it is gated behind a flag that
Phase 3 must clear before the long run uses it.

---

## File structure

| file | responsibility | change |
|---|---|---|
| `python_ai/advisor_target.py` | **new.** Advisor score map → masked target logits, per card, with the "advisor has nothing to say" gate. The only place that knows which cards have rules. | create |
| `python_ai/train.py` | shaping constants; wire `spell_value_weight`; rollout-time coverage slot + target buffer; KL branch in the update | modify |
| `python_ai/train_selfplay.py` | same two edits, pipeline 2 | modify |
| `python_ai/exploiter.py` | `w_spell` at its `compute_shaping` call | modify |
| `python_ai/test_python_ai.py` | tests for every claim below | modify |
| `python_ai/monitor_run.py` | **new.** Health daemon: NaN, weight norm, entropy, spell weight, per-card modal share | create |
| `python_ai/validate_pipeline.py` | **new.** Phase 3 heavy validation harness | create |

---

## Task 1: `spell_value_weight` is wired in and anneals

**Files:**
- Modify: `python_ai/train.py` (the schedule, both `compute_shaping` call sites)
- Modify: `python_ai/train_selfplay.py`, `python_ai/exploiter.py` (call sites)
- Test: `python_ai/test_python_ai.py`

**Interfaces:**
- Produces: `train.spell_value_weight(eps_done) -> float`, now honest.
  `SPELL_VALUE_ANNEAL_START` (env `CLASH_SPELL_ANNEAL_START`, default 0) and
  `SPELL_VALUE_ANNEAL_EPISODES` (env `CLASH_SPELL_ANNEAL_EPISODES`, default
  40000). Fraction is `clip((eps - START) / LENGTH, 0, 1)`. With the defaults
  the behaviour is bit-identical to the docstring's original intent.

- [ ] **Step 1: failing test** — weight is `W_SPELL_VALUE_START` at
      `eps=START`, `W_SPELL_VALUE_FINAL` at `eps >= START+LENGTH`, monotone
      between; and `compute_shaping` actually responds to `w_spell` (two calls
      differing only in `w_spell` must differ when a Fireball killed value).
- [ ] **Step 2: run, expect FAIL** on the offset not existing.
- [ ] **Step 3:** add the offset + env overrides; delete the "NOT WIRED IN"
      docstring and replace it with what it now does; pass
      `w_spell=spell_value_weight(episodes_completed)` at all three call sites.
- [ ] **Step 4:** run, expect PASS.
- [ ] **Step 5:** log `Shaping/SpellValueWeight` to TensorBoard each update, so
      Phase 4's "prove it anneals" is answerable from the run's own record.
- [ ] **Step 6:** commit.

**Why the offset:** a warm-start at episode 64,309 is already past the 40,000
horizon, so a faithful wiring pins the weight at 0.0 for the entire session and
there is nothing to observe annealing. The offset makes the schedule span the
run that is actually happening, without changing the from-scratch semantics.

---

## Task 2: `advisor_target.py` — score map → target logits

**Files:**
- Create: `python_ai/advisor_target.py`
- Test: `python_ai/test_python_ai.py`

**Interfaces:**
- Produces:
  - `ADVISOR_CARDS: dict[int, str]` — `{25: "building", 7: "spell", 2: "cell"}`
  - `target_logits_for(obs_np, card_id, legal_np, T) -> np.ndarray | None`
    — `(612,)` float32 target logits, `-inf` on illegal cells, or `None` when
    the advisor has nothing to say for this state. Reuses
    `tactics.building_score_map`, `tactics.spell_catch_map`,
    `tactics.best_giant_cell` and `prove_hires.soft_target_logits` so the
    surface that is distilled is the same object the advisor plays.
  - `TARGET_TEMPERATURE` (env `CLASH_ADVISOR_TARGET_T`, default 0.25)

- [ ] **Step 1: failing tests.**
      (a) an empty board returns `None` for every card — the "nothing to say"
          gate, which is what stops the term teaching a constant;
      (b) a board with one enemy clump returns a Fireball target whose argmax
          is within 2 tiles of `tactics.best_spell_cell`;
      (c) every returned vector is `-inf` exactly on `~legal`;
      (d) Giant's target is a delta at `best_giant_cell`, and `masked_kl`
          against it equals cross-entropy to that cell to 1e-5.
- [ ] **Step 2:** run, expect FAIL (module missing).
- [ ] **Step 3:** implement.
- [ ] **Step 4:** run, expect PASS.
- [ ] **Step 5:** commit.

---

## Task 3: coverage slot sampled at ROLLOUT time and buffered

**Files:**
- Modify: `python_ai/train.py`, `python_ai/train_selfplay.py`
- Test: `python_ai/test_python_ai.py`

**Interfaces:**
- Produces: `coverage_slot_seq (L,B) long` and `coverage_target_seq (L,B,612)
  float32` buffers, plus `coverage_has_target_seq (L,B) float32`.

The slot is currently drawn inside the epoch loop, so it is a different card on
every epoch of every minibatch. A buffered advisor target requires one fixed
slot per timestep. This also removes a real (if small) inconsistency: the
coverage entropy the controller logs is currently averaged over a resampled
card set.

- [ ] **Step 1: failing test** — `placement_coverage_slots` called once and
      buffered yields the SAME slot across two update epochs, and every sampled
      slot is affordable in its own row's mask.
- [ ] **Step 2:** run, expect FAIL.
- [ ] **Step 3:** hoist the sample to the rollout loop, store, and read the
      stored value in the update.
- [ ] **Step 4:** run, expect PASS.
- [ ] **Step 5:** commit.

---

## Task 4: the KL branch — the cure itself

**Files:**
- Modify: `python_ai/train.py`, `python_ai/train_selfplay.py`
- Test: `python_ai/test_python_ai.py`

**Interfaces:**
- Consumes: Task 2's `target_logits_for`, Task 3's buffers.
- Produces: `ADVISOR_COVERAGE_COEF` (env `CLASH_ADVISOR_COVERAGE_COEF`, default
  0.05; `0` restores v1.2.0 behaviour exactly, which is the control arm).

Loss becomes, on the coverage pass only:

```
rows WITH an advisor target : + ADVISOR_COVERAGE_COEF * masked_kl(cf_logits, target)
rows WITHOUT one            : - PLACEMENT_COVERAGE_COEF * H(cf_logits)/log N   (unchanged)
```

The two are mutually exclusive per row. That is the whole resolution of the
mathematical conflict: entropy says "be spread out", KL says "be here", and
applying both to one row is asking the head for both at once.

- [ ] **Step 1: failing tests.**
      (a) with `ADVISOR_COVERAGE_COEF=0` the total loss is bit-identical to
          v1.2.0's — the control arm must be provably the old code;
      (b) the advisor term contributes **exactly zero** gradient to
          `card_head` and `value_head` and to `new_logprobs` (the PPO ratio is
          untouched — same property the entropy coverage term has and the same
          test shape as `test_unchosen_card_gets_no_gradient`);
      (c) a row with a target gets no entropy bonus and vice versa;
      (d) with a target present, one optimizer step moves the coverage card's
          argmax cell TOWARD the advisor cell (distance strictly decreases,
          averaged over a fixed fixture).
- [ ] **Step 2:** run, expect FAIL.
- [ ] **Step 3:** implement in both trainers.
- [ ] **Step 4:** run, expect PASS.
- [ ] **Step 5:** commit.

---

## Task 5: the erosion measurement the handoff asks for first

**Files:** none (uses existing harnesses)

Handoff §3 Step 1: "This predicts §2.1's gain will erode under PPO. That
prediction is not yet measured — measuring it is cheap and is the first thing
to do."

- [ ] **Step 1:** short PPO run from `model_weights_hires.pth`, full-checkpoint
      path, `CLASH_ADVISOR_COVERAGE_COEF=0` (v1.2.0 objective), ~60 updates.
- [ ] **Step 2:** `prove_placement.py` before and after, same seed states.
- [ ] **Step 3:** record the delta. If Fireball's engine score falls back
      toward random, the conflict is real and Task 4 is justified by
      measurement rather than by argument.

---

## Task 6: the A/B that decides whether the cure ships

**Files:** none (uses Task 4's env flag)

Two arms, byte-identical code, same seed checkpoint, matched episodes, the ONLY
difference being `CLASH_ADVISOR_COVERAGE_COEF`:

| arm | flag |
|---|---|
| control | `CLASH_ADVISOR_COVERAGE_COEF=0` (v1.2.0 objective) |
| treatment | `CLASH_ADVISOR_COVERAGE_COEF=0.05` |

- [ ] **Step 1:** run both to matched episode count.
- [ ] **Step 2:** score with `prove_placement.py` — engine-scored, paired, the
      only verdict that counts. Fireball by elixir killed, Cannon by tower HP
      preserved. Read modal share next to top-1 probability, never alone.
- [ ] **Step 3:** decide. Ship the treatment only if it beats the control on
      the engine score; report both either way.

---

## Task 7: unfreeze the CNN trunk — ONLY if Task 6 clears, with the guards

Handoff §3 Step 2. Ordered second because the trunk feeds the critic and the
critic is the scorer search depends on.

- [ ] Keep the critic frozen; assert drift `0.000000`.
- [ ] KL-anchor the card head and the alive cards' placement maps.
- [ ] Re-run the side null (policy vs bit-exact copy of itself ≈ 0.50).
- [ ] Run soft-target-with-unfrozen-trunk as an arm, not as the default.

---

## Task 8: search-distilled labels as a periodic burst

**Files:** Modify `python_ai/train_selfplay.py` (scheduling only)

Reuses `expert_iteration.py --train-dist` unchanged, as a self-contained burst
on the exploiter's precedent (a mode switch inside the main loop needs guards
scattered through ~700 lines and one missed guard silently corrupts the run).

- [ ] Gate behind `CLASH_SEARCH_BURST_EVERY` (default 0 = off).
- [ ] Verify in Phase 3 that a burst leaves the main agent's weights
      bit-identical and carrying no gradients, exactly as the exploiter's
      verification does.
- [ ] Enable for the long run only if that verification is clean.

---

## Task 9: Phase 3 validation harness + Phase 4 run and monitor

**Files:** Create `python_ai/validate_pipeline.py`, `python_ai/monitor_run.py`

- [ ] `validate_pipeline.py`: PFSP sampling distribution vs its
      `(1-winrate)^2` spec; scenario injection actually produces a threatened
      board; advisor targeting on thousands of engine states; search throughput;
      spell weight schedule; side null; full C++ Catch2 suite.
- [ ] Must be heavy enough to be the compute burn Phase 3 asks for.
- [ ] `monitor_run.py`: NaN/inf in losses, parameter norm drift, RSS growth
      (memory leak), entropy vs target, `Shaping/SpellValueWeight` annealing,
      and per-card modal share + top-1 probability for Fireball and Cannon.
- [ ] Launch the long run warm-started through the full-checkpoint path.
- [ ] Launch the fresh-weights arm (the brief's literal from-scratch request)
      at a smaller budget.
- [ ] Final: `prove_placement.py` and `hybrid_ab.py --n 600`, then decide
      per-card whether the override can come off.

---

## Self-review

**Spec coverage.** Phase 1 → Tasks 1–4. Phase 1's "test the resolution before
unfreezing the trunk" → Tasks 5–6 gate Task 7. Phase 2 advisor → Task 4;
PFSP + scenario injection → already present, validated in Task 9; search →
Task 8. Phase 3 → Task 9's validator. Phase 4 → Task 9's run + monitor.

**Gap I am accepting knowingly:** Task 7 (unfreeze trunk) is gated on Task 6
clearing AND on there being enough wall clock left. If it does not run, the
final report says so rather than quietly dropping it.

**Type consistency.** `target_logits_for` returns `np.ndarray | None`
everywhere; the trainers convert to torch once, at buffer-write time.
`masked_kl(new, old)` argument order is `(new_logits, old_logits)` and is used
that way in Task 4 with the advisor as `old`.
</content>
</invoke>
