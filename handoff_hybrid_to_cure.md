# Handoff — v1.2.0, the live-ready hybrid, and what the neural cure needs next

Written 2026-08-14, at tag **v1.2.0**. Read this before touching the placement
head, `tactics.py`, `hybrid_policy.py` or the live loop.

Everything here is measured. Where a number is not measured it says so, and
where a previous session's claim was **weakened by measurement** it says that
too — three such corrections are recorded below, and they are the most useful
part of this document.

---

## 0. What v1.2.0 is, in one paragraph

The bot's placement decision for three cards was taken **away** from the network
and given to a deterministic advisor (`tactics.py`), which is worth **+11.8
win-rate points** (p = 1.9e-05). That is the shipping configuration and it now
runs live against the real game. Separately, the network's placement head was
given the resolution it was missing (`place_hires`), which fixes **Fireball**
outright by the engine's own scoring — but those trained weights are an
experiment, not the shipping net. **v1.2.0's runtime behaviour is exactly the
hybrid baseline**, because the new branch is zero-initialized and the shipping
checkpoint predates it.

| | |
|---|---|
| shipping checkpoint | `python_ai/model_weights_selfplay.pth` (episode 64,309) |
| shipping policy | `python_ai/hybrid_policy.py` — network chooses WHAT/WHEN, advisor chooses WHERE for Cannon/Fireball/Giant, `SolvencyGate` vetoes bankrupting spends |
| live entry point | `perception/live/mvp_loop.py --policy neural --act` |
| tests | 42 in `python_ai/test_python_ai.py`, 337 + 1 skipped in `perception/tests` |

```bash
python_ai/venv/Scripts/python.exe -m pytest python_ai/test_python_ai.py -q
```
```bash
perception/.venv/Scripts/python.exe -m pytest perception/tests -q
```

---

## 1. v1.2.0 baseline metrics — these are the numbers to beat

### 1.1 Win rate (simulator, paired, opponent 1.5× elixir)

Two independent **pre-registered** paired runs, `env.snapshot()` giving both
arms a bit-identical opening, net `model_weights_selfplay.pth`:

| | n | neural | hybrid | delta | p |
|---|---|---|---|---|---|
| exploratory | 250 | 0.646 | 0.720 | +0.074 [−0.004, +0.150] | 0.070 |
| **confirmatory** | **600** | **0.584** | **0.703** | **+0.118 [+0.067, +0.171]** | **1.9e-05** |

Confirmatory run, full metric set (n = 600):

| metric | neural | hybrid | paired delta | p |
|---|---|---|---|---|
| win rate | 0.584 | **0.703** | +0.118 | 1.9e-05 |
| bankrupt (<3 elixir) | 72.7% | **41.7%** | −31.0 pts, 600/600 | 4.8e-181 |
| mean elixir | 2.22 | 3.45 | +1.23, 600/600 | 4.8e-181 |
| elixir spent / ep | 103 | 105 | +1.6 | 0.042 |
| plays / ep | 29.3 | 30.4 | +1.1 | 0.003 |
| tower HP dealt / ep | 6545 | **7089** | +544 | 0.010 |
| tower HP lost / ep | 5575 | **4564** | −1011 | 7.4e-09 |

It attacks better *and* defends better on essentially unchanged spending.
`init_cannon/fireball/giant` are all 0.00 — the entire gain is the **passive**
placement override plus the solvency gate.

**The control arm's own variance is the trap.** Plain greedy has measured
0.570–0.775 across runs of this same net. Any comparison here under a few
hundred *paired* trials is measuring that, not the treatment.

```bash
python_ai/venv/Scripts/python.exe python_ai/hybrid_ab.py --n 600
```

### 1.2 Live, on the real game (new in v1.2.0)

One Training Camp match, `--policy neural --act --ensure-match`, DirectML:

| | |
|---|---|
| decisions | 250 in 260 s, **0.96 Hz** |
| decisions over the 1000 ms budget | **0 / 250** |
| board age | mean 511 ms, median 479, p95 852, max 1149 |
| over the staleness cap | **0 / 250** |
| perception thread | 663 boards @ 2.55 Hz, **0 errors** |
| perceive breakdown (median) | decode 48.9 ms / detector 224.5 ms / adapter 75.1 ms |
| placements issued | 18; **17/18** confirmed by at least one oracle |
| actuator | raw-evdev, 0 dropped, 0 errors |

**Two honest caveats.** The advisor fired **once** in 250 decisions — the
commander's take-up of Cannon/Fireball/Giant is ~0, so the live sample of the
override path is thin, and the live evidence is that the loop *runs*, not that
the advisor *helps*. And the bot **lost that match 0–3** to Trainer Red; live
play is gated by perception fidelity, and **no live win rate has ever been
measured** (the operator reports it plays visibly better with the hybrid on;
that is an observation, not a measurement).

The reproducible half of this check needs no emulator and is deterministic:

```bash
perception/.venv/Scripts/python.exe -m perception.live.mvp_loop --policy neural --frames perception/assets/live/match_practice_01 --seconds 45
```

### 1.3 Placement quality, scored by the engine

The definitive metric: snapshot the match, `inject()` the card at the proposed
cell (injection costs no elixir, so the rest of the match is untouched), run
forward, read the engine's own accounting. Paired over states drawn by one fixed
reference policy (`prove_placement.py`, 12 episodes, opponent 1.5×):

| arm | Fireball, elixir killed (n=1937) | Cannon, tower HP preserved (n=894) |
|---|---|---|
| shipping net alone | 0.062 | 102.2 |
| coarse-distilled control | 0.000 | 161.3 |
| **net + `place_hires`** | **1.863** | **232.6** |
| random legal cell | 0.346 | 357.0 |
| **advisor (what ships)** | **2.647** | **539.1** |

Read this as the ranking it is: **the advisor is still the best placer for both
cards**, which is why the override stays on. The head went from *worse than
chance* to **5.4× better than chance** for Fireball, and is still below chance
for the Cannon.

---

## 2. Everything measured today, and what each one taught

### 2.1 The hi-res branch works, and it is non-destructive (`prove_hires.py`)

`cnn_trunk` pools twice, so the head read a **9×5** map of a 34×18 board.
`place_hires` adds a parallel path from the trunk's own **pre-pool 16×34×18**
activation at one-tile resolution, conditioned on the same `(hx, card)` context,
added to the coarse logits as a residual. **3,993 parameters (+0.2%).**

Controlled A/B: one collection of 2,376 states from the seed policy's own
trajectory, contiguous-tail held-out split, both arms from the same checkpoint
and seed, identical epochs/lr/anchor. The **only** difference is whether
`place_hires` trains. Held out (n = 582 / 653):

| | control (coarse) | +hires |
|---|---|---|
| Fireball exact cell | 0.0% | **63.9%** |
| Fireball mean distance | 14.25 tiles | **3.31** |
| Fireball top-1 p | **0.0017** | 0.1002 |
| Fireball modal share | 98.0% | 13.0% |
| Cannon within 2 tiles | 0.2% | **10.0%** |
| Cannon mean distance | 9.69 tiles | **6.89** |
| Cannon modal share | 88.8% | 31.8% |

**Read the control's top-1 probability first: 0.0017 against a uniform
1/612 = 0.00163.** Fit to the advisor's exact cell, the coarse head does not
sharpen on a wrong cell — it *dissolves to uniform*, and then reports a 98.0%
modal share, which is the degenerate reading CLAUDE.md warns about. Its loss
plateaus at ~21 (matching the earlier 21.4) while the branch's falls to 15.1.
Replicated on two independent collections.

**Why it costs no checkpoint.** The branch's final conv is zero-initialized in
weight *and* bias, so it contributes an **exact** zero at init and an existing
checkpoint behaves bit-identically. The loader says so:
`warm-started 27/27 tensor(s), re-initialized: []`. The handoff's original plan
(concatenate into `place_up`) would have changed that layer's shape and
discarded the trained placement head from every checkpoint. Gradient still
flows: a zero conv has a nonzero gradient of its own, so it leaves zero on the
first step and the layer under it starts learning on the second.

**Cost:** placement head fwd+bwd at B=256 went 85.0 → 105.7 ms (+24% of the
head, which is ~41% of update time, so roughly **+10% per PPO update**).

### 2.2 CORRECTION — "the head cannot express an exact cell" was too strong

The premise for this whole workstream was `distill_tactics.py`'s signature:
cross-entropy fell 180.9 → 21.4 while exact-cell argmax match never left
**0.0%**, read as inexpressibility. Tested directly, on the task reduced to its
essential — 14 boards differing only in which column holds one enemy, answer in
that column — **the coarse head fits 14/14 exactly.** Nearest-upsample followed
by 3×3 convs lets a fine cell mix *neighbouring* pooled cells, so sub-block
position is recoverable in principle.

The limit is real but it is **capacity at scale**, not impossibility. Both
results are kept in `test_python_ai.py` so the correction cannot be lost.

### 2.3 CORRECTION — the Cannon's exact cell is the teacher's fault

`building_score_map` accumulates coverage by scattering **flat discs**, so every
cell reaching the same enemies scores *exactly* the same and the only
tie-breakers are two step functions. The top of the surface is therefore a large
exact-tie plateau, and `np.argmax` returns its top-left cell by row-major
accident — a target that jumps discontinuously while the advisor is genuinely
**indifferent** across all of it.

Quantified: lowering the softmax temperature **cannot** push the Cannon target
below **~66% of maximum entropy** (Fireball reaches 41%), because the ties never
break.

| T | Cannon | Fireball |
|---|---|---|
| 0.10 | 65.8% | 41.5% |
| 0.25 | 68.3% | 41.8% |
| 1.00 | 86.6% | 52.7% |
| 2.00 | 96.7% | 89.6% |

So Cannon exact-match near zero is **expected and uninformative**. Judge a
building by mean distance and by engine score, never by exact cell.
`building_score_map` was split out of `best_building_cell` so the surface that
is distilled and the cell that is played cannot drift apart (pinned by test).

### 2.4 NEGATIVE — the soft neighbourhood target does not help (with a frozen trunk)

Run as a third arm on the same collection: KL to
`softmax(standardized advisor score / T)`, T = 0.25 chosen off the entropy table
above rather than tuned on the outcome.

| | control | argmax target | **soft target** |
|---|---|---|---|
| Fireball within 2 | 0.0% | **71.4%** | 71.2% |
| Fireball mean distance | 14.25 | **3.31** | 4.49 |
| Fireball exact cell | 0.0% | **63.9%** | 0.8% |
| Cannon within 2 | 0.2% | **10.0%** | 8.4% |
| Cannon mean distance | 9.69 | **6.89** | 7.92 |

It spreads mass over the neighbourhood exactly as designed and buys nothing.
The plateau argument in §2.3 predicted it would rescue the Cannon. It did not.
**Resolution was the binding constraint, not target softness.**

**Scope this result precisely before discarding the idea:** it was measured with
the **trunk frozen**. Soft target + *unfrozen* trunk is a different experiment
and is not refuted by the above — see §3.

### 2.5 The live loader was strict and would have crashed every live run

`NeuralPolicy.__init__` called `load_state_dict` with the default
`strict=True`, so every pre-v1.2.0 checkpoint raises on the new `place_hires`
keys. It now loads with `strict=False` **plus** an explicit check that the only
missing keys are that branch — `strict=False` alone would also silently swallow
a genuinely mismatched checkpoint. **Found by running the frame-replay path, not
by any test.** Two diagnostic probes had the same defect and were fixed with
`load_state_dict_flexible`.

The loader's own message was also misleading: it reported "re-initialized" for
both "a trained tensor was discarded" and "the net has parameters this
checkpoint predates". Those are opposite situations and only one is bad; they
now print differently.

### 2.6 A latent bug found in the sweep and deliberately NOT fixed

`spell_value_weight` is **never called**. Both `compute_shaping` call sites omit
`w_spell`, so the Fireball-value shaping weight is pinned at
`W_SPELL_VALUE_START = 0.08` for the whole of training and the anneal to zero
that its own comment block describes **never happens**.

Left in place with a warning docstring rather than wired in or deleted: wiring
it changes the reward on every step of every future run (gameplay-affecting,
needs its own measurement and its own decision), and deleting it would erase the
evidence that the intended schedule exists. **Decide this deliberately before
the next long run.**

---

## 3. THE BLUEPRINT FOR THE NEURAL CURE — do these in order

The goal is a network that does not need the tactical officer. Two things stand
between here and there, and they are independent.

### Step 1 — give the coverage term a TARGET (highest value, not more distillation)

**The problem, stated mechanically.** `PLACEMENT_COVERAGE_COEF` (0.02, on by
default) adds an **entropy bonus** on one uniformly-sampled affordable slot per
step. For a card the policy does not play, that bonus is the **only** placement
gradient in the entire objective — and it pushes the map toward uniform. A
distilled Fireball map is exactly such a card's map. So distillation and
coverage are in **direct opposition**: distillation puts mass on the right cell,
coverage flattens it, and coverage runs for all of training.

This predicts that §2.1's gain will erode under PPO. **That prediction is not
yet measured** — measuring it is cheap and is the first thing to do (train from
the hires-distilled net for a few hundred updates and re-run `prove_placement`).

**The fix follows the conclusion this project already reached twice** — *closing
a coverage hole needs a TARGET, not noise.* For the sampled coverage slot: if
the advisor has a rule for that card, add KL to its masked score map; otherwise
fall back to today's entropy bonus. The pieces already exist:

- `tactics.building_score_map` / `tactics.spell_catch_map` — the surfaces
- `distill_tactics.collect(..., want_maps=True)` — collection with maps
- `prove_hires.soft_target_logits` — score map → target logits at temperature T

Compute the target **once per rollout step**, not per PPO epoch, and buffer it
(~0.2 ms per advisor call; ~9.8 MB per card per rollout at 500 steps × 8 envs).
Gate it behind an env var so it can be ablated, like the existing coverage term.

**The honest counter-argument, stated so you can weigh it:** §2.4's soft target
was *also* a target rather than noise, and it bought nothing. The difference is
that §2.4 replaced a good hard target with a soft one, whereas this replaces
pure noise with a target. That is an argument, not evidence.

### Step 2: unfreeze the CNN trunk — but only with these guards

Ordered second because it is the most invasive change available: the trunk feeds
the **critic**, and the critic is the scorer that decision-time search depends
on. Every distillation result above was obtained with the trunk frozen, so the
trunk's features are the one thing never yet varied.

The specific reason to expect something: **Fireball moved and the Cannon did
not.** Fireball needs "where is the enemy clump", which conv1 clearly carries at
full resolution. The Cannon needs "where will this push be in 2 s, and which
lane is threatened" — a *derived* quantity nothing in the observation encodes
directly. If the frozen trunk does not carry it, no head on top can read it, and
unfreezing is the only lever that changes that.

Guards, all of which have bitten this project before:

1. **Keep the critic frozen** and assert drift is `0.000000`, exactly as
   `distill_tactics.py` and `expert_iteration.py` already do. A drifting critic
   silently degrades every future search label.
2. **KL-anchor the card head and the alive cards' placement maps.** Five of
   eight cards currently work; unfreezing the shared trunk can drag them.
3. **Re-run the side null** — a policy against a bit-exact copy of itself must
   score ~0.50. It is the only diagnostic that catches an observation-shaped
   fault, and it read 0.598 once while every other metric looked healthy.
4. **This is where the soft target is worth retrying** (§2.4 scoped it to a
   frozen trunk). Run it as an arm, not as the default.

### Step 3: only then, re-measure the override

The override is justified for exactly as long as the advisor beats the head
(§1.3). Re-run `prove_placement.py` after Steps 1–2 and turn the override off
**per card**, not wholesale, and only where the head has overtaken the advisor.
Then re-run `hybrid_ab.py` at n ≥ 600 — the control's own variance is 0.570–0.775
and anything smaller measures noise.

**Do not let the officer INITIATE plays.** A 5-arm ablation measured that
letting it start Cannon/Fireball plays (rather than only place them) cut tower
damage dealt by 3,410/ep and win rate by 0.367 (p = 0.013). The override works
*because* it is passive: it changes where a card lands, never how often one is
played, so it cannot spend elixir the commander did not already commit.

### Explicitly NOT the next step

- **More distillation data.** Coverage kept climbing while lift stalled in the
  expert-iteration work; more data is where that went wrong, and DAgger at a
  fixed budget was the only thing that reversed it.
- **Re-trying the entropy coverage term as a fix on its own.** Measured: it
  moves the frozen cell, it does not break the lock.
- **Decision-time search (Path B).** Still worth ~7× what distilling it buys
  (+0.319 vs +0.045), but it costs 2.2× wall clock and the live loop already
  spends ~350 ms of its 1000 ms budget on perception. It is a training-label
  generator here, not a live option.

---

## 4. Traps — do not re-pay for these

Carried forward, all still live:

- **Warm-starting a converged policy into a FRESH training state re-arms the
  initial entropy target and the controller dissolves the policy.** Seeding a
  bare `state_dict` takes `train.py`'s legacy path and resets
  `episodes_completed` to 0, so `placement_entropy_target(0)` returns 0.65
  against a policy at 0.11. **Always resume through the full-checkpoint path.**
- **A short resumed run writes NO checkpoint** (`last_save_ep` is set on
  resume). Use `CLASH_SAVE_EVERY`.
- **A stage-5 resume flips to `random_opponent` after 100 episodes**, swapping
  the opponent's deck mid-experiment. Pin with `CLASH_PHASE2_ENTRY_WIN_RATE=2.0`.
- **`F.kl_div` over `-inf` cells gives `nan`.** Use `distill_tactics.masked_kl`.
- **Modal share degenerates on a near-uniform distribution** — always read it
  next to top-1 probability. §2.1's control is the worked example: 98.0% modal
  share at a top-1 of 0.0017.
- **Stepping the LSTM once per query** instead of once per timestep silently
  runs the reference policy at double clock. `prove_placement.step_net` is right.
- **`get_elixir_value_killed_by(CANNON)` is the wrong metric for a building** —
  it scores distraction, most of its job, at zero. Use tower HP preserved.

New today:

- **Two boards that differ only in the elixir scalar are ONE board to the
  placement head.** The first version of the expressivity test sampled 8 rollout
  states that collapsed to 7 distinct spatial maps, making the task unfittable
  by any head at any resolution. Build such fixtures with `env.inject`.
- **`place_ctx_hi` matches the `place_ctx` prefix.** Any code selecting
  trainable parameters by `name.startswith("place_ctx")` will silently train
  half the hi-res branch. `prove_hires.fit` handles this explicitly; copy that
  pattern.
- **The tests are one file now** (`python_ai/test_python_ai.py`). If you split
  them again, keep the section docstrings — they carry the *why*.

---

## 5. Repo state at v1.2.0

| | |
|---|---|
| tag | `v1.2.0` on `main` |
| deleted | `python_ai/curriculum.py` (legacy gym registration, imported by nothing) |
| merged | 4 test files → `python_ai/test_python_ai.py` (42 tests, node-id set verified identical) |
| deduplicated | `_to_scalar`, was two byte-identical copies → `gym_wrapper._to_scalar` |
| new | `python_ai/prove_hires.py`, `MicroRoyaleNet.place_hires` / `place_ctx_hi` / `extract_features_hires` / `hires_features` |

**Kept deliberately, do not "clean" them up:** the checkpoints, `archive_*/`,
`historical_checkpoints/` and `stage_checkpoints/` are untracked, so deleting
them is unrecoverable, and they are the only record of ~46 h of measured
training. The A/B and probe harnesses stay for the same reason — each one backs
a number quoted in the docs, and deleting the harness makes the number
unreproducible.

**Experimental checkpoints from today** (none of them shipping, none win-rate
tested): `model_weights_hires.pth` (the +hires distilled net of §2.1),
`model_weights_hires_control.pth`, `model_weights_hires_soft.pth`.

**Reproduce today's headline:**
```bash
python_ai/venv/Scripts/python.exe python_ai/prove_hires.py --episodes 12 --epochs 20 --soft-T 0.25
```
```bash
python_ai/venv/Scripts/python.exe python_ai/prove_placement.py --seed model_weights_selfplay.pth --control model_weights_hires_control.pth --treatment model_weights_hires.pth --episodes 12
```
