# Handoff — the neural placement cure, measured and shipped

Written 2026-08-15, after an autonomous session. Read this before touching
`advisor_target.py`, the coverage term, `hybrid_policy.py`, or the placement
head.

Everything here is measured. Where a claim is weaker than it looks, it says so,
and **three predictions that did not survive contact are recorded** — including
one that was the stated motivation for the whole workstream.

---

## 0. What changed, in one paragraph

The placement-coverage term stopped being pure entropy and was given the
advisor's own score surface as a **target**. Over 524 PPO updates that took the
network's Fireball placement from *worse than a random cell* to **level with the
deterministic advisor**, and the Cannon from *significantly worse than chance*
to **significantly better than chance** for the first time in this project's
history. Separately, `spell_value_weight` — dead code since the term was
written — is wired in and annealing.

| | |
|---|---|
| shipping checkpoint | `python_ai/model_weights_cured.pth` (episode 78,214) |
| training run | pipeline 2, 13,961 episodes / 524 updates, **0 alarms** |
| branch | `feat/neural-cure` |
| tests | 57 in `python_ai/test_python_ai.py` (was 41) |
| validator | `validate_pipeline.py`, 21/22 |

---

## 1. The mechanism

`PLACEMENT_COVERAGE_COEF` adds an **entropy bonus** on one uniformly-sampled
affordable slot per step. For a card the policy never plays, that bonus is the
*only* placement gradient in the objective — and it pushes the map toward
uniform, which is exactly what a distilled placement map is not. The two are in
direct opposition, and coverage runs for all of training.

The fix is a **row mask**. A row either carries an advisor target and gets KL to
its masked score map, or it does not and keeps the entropy bonus. Never both:
entropy says "be spread out", KL says "be here". This is the fourth time this
project has landed on *closing a coverage hole needs a target, not noise*.

Three details that are load-bearing:

- **The gate.** `tactics` always returns a cell; on a quiet board that cell is a
  default (the defensive pocket, an arbitrarily tie-broken lane). Training on
  defaults teaches a CONSTANT — the exact pathology being repaired.
  `target_logits_for` returns `None` there.
- **The slot weighting.** The limiter was affordability, not the gate: measured
  over 258 decision steps, the sampled slot held an advisor card 32.6% of the
  time and the advisor then spoke on 90% of those. Cannon(3)/Fireball(4)/
  Giant(5) are exactly what a near-bankrupt agent cannot afford.
  `CLASH_ADVISOR_SLOT_WEIGHT` (default 5.0) spends the scarce budget on them.
- **Read `Advisor/KL` and `Advisor/Rows` as a pair.** KL falls both when the
  head learns the surface and when the advisor goes quiet.

---

## 2. The numbers

Engine-scored, paired on states drawn by one fixed reference policy
(`prove_placement.py`). Cannon by tower HP preserved over a full 300-tick
lifetime; Fireball by `get_elixir_value_killed_by`.

| | seed (hires) | control (coef 0) | **cured (coef 0.10)** | advisor | random |
|---|---|---|---|---|---|
| Cannon (n=1582) | 276.0 | 403.0 | **553.1** | 687.9 | 411.3 |
| Fireball (n=2979) | 1.506 | 2.190 | **2.565** | 2.464 | 0.469 |

- Cannon vs random: **+141.8, CI [+74.5, +210.2], p = 0.0071.** First time above
  chance. Was −124.4 (p = 2.5e-09) at v1.2.0.
- Cannon vs advisor: **−134.9, p = 5.0e-10.** Still behind, gap closing
  (−306.6 → −222 → −134.9 across three measurements).
- Fireball vs advisor: **+0.100 in the net's favour.** Bootstrap CI excludes
  zero, exact sign test does not (223/252, p = 0.199). The honest claim is
  **"not distinguishable, certainly not worse."**

**Dynamism — the qualitative proof:**

| net | Cannon modal / share / top-1 | Fireball modal / share / top-1 | cells |
|---|---|---|---|
| seed | (11,0) 44.9% 0.140 | (11,0) 32.8% 0.087 | 88 / 188 |
| cured | **(16,15) 17.0% 0.062** | **(4,17) 12.7% 0.112** | 135 / 256 |

Fireball's modal share fell while its top-1 probability **rose**. That is the
signature of a head that became more confident *within* a state and less
repetitive *across* states, and no previous net has shown it. The pathological
own-back-row cell `(11,0)` is gone from both cards.

---

## 2b. Win rate: the override is redundant, and the net SPECIALIZED

**The override is a measured null now** (`hybrid_ab.py --per-card`, 200 paired
openings, solvency gate ON in every arm so only placement varies):

| arm | win rate | vs neural | p |
|---|---|---|---|
| neural (no placement override) | 0.507 | — | — |
| cannon_only | 0.510 | +0.003 | 1.0 |
| cannon_giant | 0.480 | −0.028 | 0.54 |
| all_three | 0.480 | −0.028 | 0.56 |

At v1.2.0 the same override was worth +11.8 points (p = 1.9e-05). This n had
power to see that, and it is gone. **`hybrid_policy.py` can drop to gate-only.**

**Then the part that needs reading carefully.**

| opponent | v1.2.0 | cured | delta |
|---|---|---|---|
| C++ `HeuristicOpponent` @1.5x (`net_ab.py`, n=200 paired) | 0.6225 | **0.5100** | −0.1125, CI [−0.2025, −0.0200] |
| **v1.2.0 itself**, head-to-head, sides swapped (`net_h2h.py`, n=120) | 0.3875 | **0.6125** | **+0.1125, CI [+0.054, +0.171]** |

**Specialization, not degradation.** 13,961 episodes in an all-neural PFSP
league made the net better against that opponent class — it beats its own
predecessor — and worse against the C++ heuristic, which pipeline 2 never shows
it. **Either number alone gives the wrong answer.**

Two controls make that attribution stick:

- **The seed is not the cause.** `model_weights_hires.pth` had never been
  win-rate tested; it measures 0.6125 vs v1.2.0's 0.6350 (delta −0.0225,
  p = 0.76). The regression came from the run, not the starting point.
- **Sides were swapped.** A policy once beat a bit-exact copy of itself 0.598
  purely by side assignment; it reads 0.530 today, and a one-sided duel would
  fold that into the result.

**`model_weights_selfplay.pth` was deliberately NOT replaced.** Two checkpoints
with different strengths now exist. Which is "better" depends on the opponent
you care about, and **nothing here measures the real game.**

---

## 3. Predictions that did NOT survive — read these first

**The stated motivation for this work did not reproduce.** The v1.2.0 handoff
predicted the entropy coverage term would ERODE the distilled placement head
under PPO, and called measuring it "the first thing to do". It does not erode:
the control arm IMPROVED over its seed on both cards (Fireball 1.506 → 2.190,
Cannon 276.0 → 403.0) over 80 updates. The conflict is real in mechanism, but
the erosion is not observable at this horizon. **The advisor target is worth
having because it is much better, not because the alternative decays.**

**My own validator failed a check it had no power to answer.** The
advisor-vs-random value check FAILED on the Cannon at n=28 on states drawn from
a no-op-only rollout, where nobody ever defends and the board is already lost.
It now prints rather than judges, and the verdict is left to
`prove_placement.py` (n=1582 on the right distribution). *A check that cannot
answer its question should not have a PASS/FAIL.*

**Three monitor notifications reported episode numbers that exist in no
artifact** (79,582 / 80,912 / 82,163) while `train.log`, the TensorBoard event
file and the checkpoint all agree on 78,270 / 78,248 / 78,214. Cause not
identified. Trust the on-disk artifacts; `monitor_run.py`'s episode readout is
suspect and worth fixing before it is relied on again.

---

## 4. Traps added this session

- **`train_selfplay.py` dies on any pre-`place_hires` checkpoint.** The branch
  added 6 parameters, so the optimizer state describes 26 and the net has 32.
  `train.py` degrades gracefully; pipeline 2 does not, and
  `load_state_dict_flexible` reports CLEAN because all 33 tensors are supplied.
  `setup_ab_arm.py --base` remaps the moments **by name** — the new parameters
  land at indices 22-27, in the MIDDLE of `named_parameters()` order, so a
  positional remap hands the trunk's moments to the placement head.
- **`spell_value_weight` needs `CLASH_SPELL_ANNEAL_START` on any warm start.**
  A run resuming past the 40,000-episode horizon is pinned at FINAL from step
  one and the anneal is unobservable.
- **numpy's `std` is population, torch's is sample.** Two standardizations of
  the same score map disagreed by 0.1% of a logit until a test pinned them.

---

## 5. What to do next, in order

1. **Retire the Fireball override, keep the Cannon's.** The per-card ablation
   is wired: `hybrid_ab.py --per-card` runs neural / cannon_only /
   cannon_giant / all_three with the solvency gate held ON in every arm.
2. **Re-measure the Cannon after more training.** Its gap to the advisor has
   closed monotonically across three measurements. It is the one card still
   needing the officer.
3. **The Giant is now scored too** (`prove_giant.py`, written to close this
   gap). n=2784: v1.2.0 **13.7** tower damage, cured **231.9**, advisor 380.5,
   random 86.7. **+145.2 vs random (p = 6.5e-19), +218.2 vs v1.2.0
   (p = 1.0e-98)** — it was 6x WORSE than chance and used 38 of 612 cells with
   half its mass on (11,0); it now uses 140. Still below the advisor (−148.6),
   whose rule is effectively two cells (57.7% on one bridge), a strong prior
   this board rewards. **All three starved cards now beat chance.**
4. **Decision-time search remains unwired.** Validated this session — snapshot
   0.020 ms, 10-tick step 0.026 ms, a K=12 2s sweep at 0.87 ms against 5.65 ms
   for one network forward — so the "simulation is free, scoring is the budget"
   argument holds. The measured-positive use is offline value-DISTRIBUTION
   distillation (+0.045), not per-step override.
5. **The exploiter stays off.** Disabled 2026-08-11 after two bursts found
   nothing at a ~20% compute tax. Its documented re-enable precondition — that
   a parked building no longer be free — is an engine reward change, not a
   Python one.

---

## 6. Repo state

New: `advisor_target.py`, `setup_ab_arm.py`, `validate_pipeline.py`,
`monitor_run.py`, `model_weights_cured.pth`.
Modified: `train.py`, `train_selfplay.py`, `exploiter.py`, `hybrid_ab.py`,
`test_python_ai.py`, `CLAUDE.md`.

Working directories under `_runs/` (gitignored) hold the A/B arms, the main run,
the from-scratch arm and every log quoted above. Checkpoints were backed up to
`_session_backup_20260814/` before anything wrote.

**No C++ was changed.** CLAUDE.md forbids it without the human confirming
diagnosis and edit, and the human was offline. **No emulator was used.**
