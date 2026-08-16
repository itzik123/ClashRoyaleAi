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

---
---

# Session 2 — 2026-08-16: search shipped, distillation refuted, and the
# architectural problem that is actually left

Appended after a second autonomous session. Sections 7-10 supersede nothing
above; they answer the question section 5 left open ("the measured-positive use
is offline value-DISTRIBUTION distillation"), and the answer is **no longer
true on this seed**.

---

## 7. What shipped: search at inference, weights unchanged

The heuristic regression (0.51 vs v1.2.0's 0.6225) was going to be fixed with a
from-scratch search-distilled run plus a mixed curriculum. **Both premises were
measured first and both were false** — see CLAUDE.md's 2026-08-15 entry for the
numbers. Briefly: the mixed curriculum already existed at a realized **20.8%**
heuristic share and had not prevented the regression; and search at random init
is inert (overrides 21.5% of decisions, gains nothing), so it cannot bootstrap
a from-scratch run.

The lever nobody had swept was `horizon`. Depth costs engine steps (0.015 ms),
width costs network rows (0.13 ms), so lookahead is ~9x cheaper than candidates.
Going h=4 → h=12 took search from 0.667 to 0.963 **and got cheaper**.

| | bar | result |
|---|---|---|
| vs C++ heuristic@1.5x (n=400 paired) | >0.65 | **0.9225** (greedy 0.5200) |
| h2h vs v1.2.0 as shipped (n=150 swapped) | >0.60 | **0.7200** CI [0.675, 0.765] |

**And the control that keeps this honest:** give BOTH sides search and the
cured net scores **0.4425** CI [0.3825, 0.5025] against v1.2.0 — no difference
resolved. The gain is search, not the cured weights.

Shipping config is named in one place: `python_ai/shipping.py`.

---

## 8. Distillation is REFUTED on a cured seed — do not re-run it

The obvious follow-up was to bake the h=12 expert's +0.4025 into the weights,
since the h=4 expert had distilled for +0.045. Run at the best known recipe
(value-DISTRIBUTION labels, frozen trunk, 16 epochs, 180 episodes, T=0.05 chosen
from `--target-entropy` before any outcome). **Three independent measurements,
all null:**

| | h=4 → v1.2.0 (worked) | h=12 → cured (this) |
|---|---|---|
| conditional lift | +0.1535 | **−0.0032 ± 0.0230** |
| p1/p0 selectivity | 2.17 | **0.92** |
| paired greedy A/B | +0.045, p=0.0074 | **+0.0250, p=0.608** (n=300) |
| search delta ON the student | fell, deviation 14.8%→12.1% | **+0.3583**, deviation 13.17%→**13.06%** |

**The last row is the one that settles it.** Search is worth +0.4025 on the
teacher and +0.3583 on its student, with overlapping CIs, and overrides the
student just as often. A student that absorbed the expert would be deviated
from *less*.

**THE SEED IS THE VARIABLE, NOT THE EXPERT.** The distillation that worked was
seeded from a net whose placement head was *broken* — three cards placing worse
than random. That gap is what it taught into. The cure closes it, so there is
nothing left to copy. Said the other way, and this is the same fact the
compute-matched control reports from the other side: **search and the placement
cure repair the same weakness.** They do not stack, and neither substitutes for
the other's absence.

What remains is the part that requires **actually running the simulator** twelve
seconds forward. A reactive head that only ever sees `s` has no way to represent
it. Section 9 is about that.

---

## 9. THE FUTURE BLUEPRINT — giving the policy lookahead it can own

This is the design problem, not a training run. **Do not spend compute on any of
these before the design is argued through**; three sessions of this project have
now been spent on things that measurement killed in under an hour.

The framing: the critic is allowed to *roll the world forward* and the actor is
not. Every option below is a way of closing that specific asymmetry. They are
ordered by expected value per unit of risk, which is not the same as by
ambition.

### 9.1 Recurrent rollout head — cheapest, most likely to pay

Let the policy do internally what search does externally: propose a candidate,
imagine `k` steps, and score. Concretely a small head that takes `(h_t, card,
cell)` and predicts the value the critic *would* assign after a `k`-step
rollout — trained on labels the search already produces for free.

Why this is different from what just failed: distillation copied the search's
*output* (an action distribution). This copies the search's *intermediate
quantity* (the per-candidate value), which is a far denser and better-posed
target — one scalar per candidate per state rather than one categorical over
612 cells. The 48k-row collection already contains `cand_value`; **no new data
is needed to test it.**

Risk: it is still a reactive approximation, so it may hit the same ceiling. The
cheap falsification is whether predicted candidate values correlate with the
search's actual ones on held-out states — measurable in minutes, before any
policy training.

### 9.2 Learned forward model (latent dynamics) — the real fix, the real cost

MuZero's shape: learn `(z_t, a) -> z_{t+1}` in a latent space plus reward/value
heads, then plan in latent space at inference. The policy stops needing the
simulator because it carries its own.

Why it is genuinely attractive here and not just fashionable: **we own a fast
deterministic simulator**, so the forward model can be trained on unlimited
perfectly-labelled transitions rather than scraped from play. That removes the
single hardest part of MuZero.

Why it is expensive and risky: it is a new network, a new loss, a new failure
surface, and this project's own history says every such addition took several
sessions to debug (the checkerboard head, the team-1 observation, the entropy
scaling). Budget it as a workstream, not an experiment. **Do not start it to
chase +0.04.**

### 9.3 Give the actor the critic's rollout as an INPUT

The cheapest structural change: run the search, and feed its per-candidate
values into the policy head as features rather than distilling them away. This
does not remove the inference cost — it is not a substitute for 9.1/9.2 — but it
converts search from an *override* into *information the policy can learn to
use*, which is a strictly better use of the same compute and would let the
policy learn *when to trust it*.

### 9.4 Deeper/wider search, and why it is nearly exhausted

h=20 was measured WORSE than h=12 (0.875 vs 0.963) because a candidate rollout
assumes both sides no-op, and 20 s of that stops resembling the game. **The next
gain here is not depth, it is a better opponent model inside the rollout** —
even the opponent's greedy action instead of a no-op. Cheap to try, and it is
the one search improvement not yet measured.

### 9.5 What NOT to do

- **Do not distil this expert again on a cured seed.** Refuted above, three ways.
- **Do not scale the distillation dataset.** Already known harmful (selectivity
  falls, no-op rate drifts toward the expert's).
- **Do not re-add heuristic exposure to the PFSP pool.** Already 20.8%.
- **Do not run a from-scratch AlphaZero loop with this search module.** Inert at
  init; the expert only exists after the critic is trained.

---

## 10. Live play — the gap between the benchmark and the emulator

**Search is NOT wired into `perception/live/mvp_loop.py`.** The loop's neural
path builds the 13,606-float observation through `perception_encoder` and calls
the policy head directly; there is no candidate rollout and no engine snapshot
in that path. So the configuration that measures **0.9225** cannot currently be
run live — live play gets the cured net **greedy**, which measures **0.5200**
against the heuristic.

That is the single highest-value integration task outstanding, and it is
plausibly small: the live loop already owns a simulator mirror
(`perception/forecast.py` holds a real `ClashRoyaleEnv` and calls
`get_observation_for_team(0)`), which is exactly the object `_search_action`
needs to snapshot. **Wiring search into the live loop is worth more than any
weight change currently on the table** — it is the difference between a 0.52
agent and a 0.92 one, already measured, already paid for.

Also note `mvp_loop.py`'s module docstring is **stale**: it says the trained
policy cannot be reached because no encoder exists. The encoder exists and
`--policy neural` works. Fix the docstring before it misleads someone again.
