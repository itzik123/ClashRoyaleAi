# Handoff — 2026-08-14: the hybrid works, the neural cure is next

Read this before touching the placement head, `tactics.py`, or the live loop.
Everything here is measured. Where a number is not measured it says so.

The one-line state: **the bot is +11.8 win-rate points better because the
placement decision was taken AWAY from the network for three cards.** The
network's placement head is still broken. Fixing it properly is the next job and
this document is the blueprint.

---

## 0. What to read, in order

| file | what |
|---|---|
| `PLACEMENT_COLLAPSE.md` | the full investigation, every number, every negative result |
| `python_ai/hybrid_policy.py` | the shipping policy — commander + tactical officer |
| `python_ai/tactics.py` | the deterministic advisor, and the two measured surprises |
| this file §5 | **the blueprint for the cure** |

CLAUDE.md has been updated with the durable lessons; this file is the working
handoff and can be deleted once §5 is done.

---

## 1. The defect, restated precisely

The placement head is a **constant function** for the three cards the policy
stopped playing, and its cells are worth **less than a random legal cell**:

| | Cannon (tower HP preserved) | Fireball (elixir killed) | Giant (tower dmg dealt) |
|---|---|---|---|
| trained policy | 12.1 | 0.000 | 3.3 |
| random legal cell | 353.5 | 0.276 | 94.3 |
| **advisor** | **564.1** | **2.405** | **535.6** |

All engine-scored: snapshot the live match, `inject()` the card at a candidate
cell (injection costs no elixir, so the rest of the match is untouched), run
forward, read the engine's own accounting. n = 478 / 950 / 913, p up to 1.2e-119.

**Root cause (mechanical, not statistical):** both the actor loss and the
placement entropy bonus flow through `placement_given_card` for the **chosen**
card only, so a card the policy stops playing receives **exactly zero** placement
gradient forever. `card_id_embed` is `nn.Linear(num_card_ids, 16, bias=False)`,
so column *c* belongs to card *c* alone and nothing downstream of the LSTM
consumes an unchosen card's embedding.
`python_ai/test_placement_coverage.py::test_unchosen_card_gets_no_gradient`
asserts the gradient is `== 0.0` in exact arithmetic. That is a self-sustaining
deadlock — frozen cell → card really is worthless → card head suppresses it →
no gradient — which is why more training never fixed it.

---

## 2. Experiments run this session

### 2.1 Failed: entropy coverage term (`PLACEMENT_COVERAGE_COEF`)
Restores a gradient to unplayed cards via placement entropy on one sampled
affordable slot per step. 3 arms, parallel, byte-identical code, ~80 PPO updates,
matched at episode ~64,800, engine-scored on 835/1,573 paired states.

| Cannon | modal cell | top-1 p | tower HP preserved |
|---|---|---|---|
| seed | (11,0) | 0.920 | 135.9 |
| control (coef 0) | (11,0) | 0.550 | 116.4 |
| treatment (coef 0.02) | (6,0) | **0.051** | 182.0 |

Sign test p = 0.158; Fireball unchanged at 0.000. **The lock moved, it did not
break.** Top-1 of 0.051 is near-uniform (uniform = 1/612 = 0.0016), and *the
argmax of a flat map is an arbitrary constant*. **Entropy is a MARGINAL
objective** — "be spread out", not "depend on the board".

The code is still in `train.py`/`train_selfplay.py` and is ON by default at 0.02.
It is harmless (a regularizer that never touches `new_logprobs`; pinned by
`test_coverage_does_not_change_the_ppo_ratio`) and it does close a real gradient
hole. **It is not a fix on its own.** Ablate with
`CLASH_PLACEMENT_COVERAGE_COEF=0`.

### 2.2 Failed: advisor distillation into the head (`distill_tactics.py`)
Advisor cells as a supervised target for the dead cards, trunk/LSTM/critic frozen
(critic drift verified 0.000000), alive cards held by a KL anchor.
Cannon **+161.9 HP, p = 2.0e-06** — real but still below random. Fireball
**0.000, no movement at all**. Cross-entropy fell 180.9 → 21.4 while exact-cell
argmax match stayed **0.0%**. That signature — loss falling, argmax never
matching — is a target the head **cannot represent**, not one it has not learned.
See §5 Path A.

### 2.3 Failed: potential-based solvency shaping (`elixir_shaping.py`)
Policy-invariant by construction, so safe and also limited. Over ~80 updates:
bankruptcy **−1.0 points, 95% CI [−2.7, +0.6], p = 0.21**. Every other metric's
CI contains 0. Built, 9 unit tests (telescoping, "a pure hoarder earns nothing"),
wired in, **on by default** via `CLASH_SOLVENCY`. Keep or disable freely.

### 2.4 Worked: the hybrid
See §3.

### 2.5 Removed after measurement: tactical INITIATION
Letting the officer *start* Cannon/Fireball plays (not just place them) looked
necessary — the commander's take-up is 0.06/0.00/0.00. A 5-arm ablation, n=30:

| arm | win rate | tower HP DEALT/ep | cannons initiated |
|---|---|---|---|
| neural | 0.600 | 7627 | — |
| gate only | 0.633 | 6659 | 0.00 |
| **placement only** | **0.733** | 7655 | 0.00 |
| initiate | 0.467 | 5390 | 4.03 |
| full stack | 0.233 (p=0.013) | 4216 (p=0.0014) | 4.73 |

**Defending better is worthless if it is paid for with the attack.** The
placement override survives precisely because it is PASSIVE: it changes where a
card lands, never how often one is played. `initiate=False` is the default;
`initiate=True` is kept only so the ablation reproduces.

---

## 3. The hybrid's baseline — these are the numbers to beat

`python_ai/hybrid_policy.py`. Two independent **pre-registered** paired runs
(`env.snapshot()` gives both arms a bit-identical opening), opponent 1.5x elixir,
net `model_weights_selfplay.pth`:

| | n | neural | hybrid | delta | p |
|---|---|---|---|---|---|
| exploratory | 250 | 0.646 | 0.720 | +0.074 [−0.004, +0.150] | 0.070 |
| **confirmatory** | **600** | **0.584** | **0.703** | **+0.118 [+0.067, +0.171]** | **1.9e-05** |

Confirmatory run, full metric set (n=600):

| metric | neural | hybrid | paired delta | p |
|---|---|---|---|---|
| win rate | 0.584 | **0.703** | +0.118 | 1.9e-05 |
| bankrupt <3 elixir | 72.7% | **41.7%** | −31.0 pts, 600/600 | 4.8e-181 |
| mean elixir | 2.22 | 3.45 | +1.23, 600/600 | 4.8e-181 |
| elixir spent / ep | 103 | 105 | +1.6 | 0.042 |
| plays / ep | 29.3 | 30.4 | +1.1 | 0.003 |
| tower HP DEALT / ep | 6545 | **7089** | +544 | 0.010 |
| tower HP lost / ep | 5575 | **4564** | −1011 | 7.4e-09 |

It attacks better AND defends better on essentially unchanged spending.
`init_cannon/fireball/giant` are all 0.00, so the entire gain is the passive
placement override plus the solvency gate.

**Two cautions.** The neural baseline measured 0.646 and 0.584 across the two
runs — that is the control-arm variance CLAUDE.md records (0.570–0.775), and it
is why both arms must share an opening. And this is a **workaround**: the
placement head is still worse than random, and the ~7x value decision-time search
demonstrated is still unclaimed.

Reproduce:
```bash
python_ai/venv/Scripts/python.exe python_ai/hybrid_ab.py --n 600
```

---

## 4. Architectural changes made

| file | change | risk |
|---|---|---|
| `python_ai/tactics.py` | **NEW.** Deterministic advisor: `best_spell_cell`, `best_building_cell`, `best_giant_cell`, `SolvencyGate`, `TacticalOverride`. Reads the OBSERVATION, so it behaves identically in sim and live. | none (additive) |
| `python_ai/hybrid_policy.py` | **NEW.** The shipping policy. | none (additive) |
| `python_ai/elixir_shaping.py` | **NEW.** PBRS solvency potential. | inert, measured |
| `python_ai/model.py` | `forward_sequence(..., extra_card_idx_seq=None)` returns a 6th value. **No new parameters → no checkpoint invalidated.** Both trainers updated (the only callers). | low |
| `python_ai/train.py` | coverage term, solvency term, and four env-var overrides: `CLASH_PLACEMENT_COVERAGE_COEF`, `CLASH_SOLVENCY`, `CLASH_NUM_ENVS`, `CLASH_PHASE2_ENTRY_WIN_RATE`, `CLASH_SAVE_EVERY`. Defaults preserve prior behaviour except the two new loss terms. | **gameplay-affecting** |
| `python_ai/train_selfplay.py` | same coverage term | **gameplay-affecting** |
| `perception/live/mvp_loop.py` | `NeuralPolicy(tactical=True, reserve=4.0)`; CLI `--no-tactical`, `--reserve`. | live path |

**The live override is handed the INTERSECTION of engine legality and actuator
reachability.** Handing it only the engine mask would let it propose engine
row 0, whose tap lands below the arena — exactly the bug the tile-grid fix
closed. An override is precisely the kind of code that reintroduces it.

Harnesses added: `prove_placement.py` (engine-scored, paired, with an advisor
arm), `prove_solvency.py`, `hybrid_ab.py` (`--ablate`), `gate_ab.py`,
`tactical_ab.py`, `distill_tactics.py`.

Tests: 31 pass in `python_ai/` (`test_tactics.py`, `test_placement_coverage.py`,
`test_elixir_shaping.py`); perception unchanged at 337 passed / 1 skipped.

```bash
python_ai/venv/Scripts/python.exe -m pytest python_ai/test_tactics.py python_ai/test_placement_coverage.py python_ai/test_elixir_shaping.py -q
perception/.venv/Scripts/python.exe -m pytest perception/tests -q
```

---

## 5. THE BLUEPRINT FOR THE CURE

Two paths. **They are not alternatives — Path A is a prerequisite for getting
full value from Path B**, because search proposes candidates and a head that
cannot express a good cell cannot propose one.

### Path A — give the placement head the resolution it lacks

**The diagnosis, precisely.** `cnn_trunk` is
`Conv(21→16) → MaxPool(2, ceil) → Conv(16→32) → MaxPool(2, ceil)`, so a
34×18 board becomes **9×5**. `placement_given_card` then feeds that 32×9×5 map
through `place_up` (`Upsample×2 → Conv → Upsample×2 → Conv → Conv`) back to
36×20 and crops to 34×18. **One pooled cell covers roughly 4×4 board tiles.**
The card context enters as a spatially uniform vector added to that map.

So the head's spatial vocabulary is ~4-tile blocks. An exact-cell target is close
to inexpressible, which is exactly what §2.2 measured: CE fell 4× while argmax
match never left 0.0%. Fireball is worse than Cannon because it must localise an
enemy clump anywhere on 34 rows, and the frozen trunk evidently does not carry
that feature at all.

> **STATUS 2026-08-14 — A2 is DONE and Fireball is FIXED (engine-scored 1.863
> elixir killed vs 0.346 for a random cell and 2.647 for the advisor, p=1.8e-95,
> 561 better / 71 worse). The Cannon improved but is still below random, so the
> tactical override stays on for it.** A1 is done and does NOT help.
> The diagnosis above is right that resolution is the binding constraint and
> wrong that the target is inexpressible — the coarse head fits an exact-cell
> task at small scale (14/14), it just cannot at real scale, where it dissolves
> to *uniform* (top-1 p 0.0017 vs a uniform 0.00163) rather than sharpening on
> the wrong cell. A2 was built as a **zero-initialized residual branch** instead
> of a concat into `place_up`, which keeps every checkpoint valid. See
> CLAUDE.md's 2026-08-14 `place_hires` entry, `prove_hires.py`, and
> `test_placement_hires.py`. A1's soft target measured neutral-to-worse as a
> third arm on the same data. A3 remains untried and is still ordered last.

**Do these in order. Each is independently measurable.**

**A1. Soft neighbourhood target (cheap, do first, no architecture change).**
Replace the exact-cell cross-entropy in `distill_tactics.py` with a KL to a
target distribution spread over a disc/Gaussian centred on the advisor's cell,
σ ≈ 1.5 tiles — matched to the 4× upsample block, i.e. to what the head can
actually represent. Reuse `masked_kl()` (already in that file); it zeroes the
non-finite terms that `placement_mask`'s `-inf` cells otherwise turn into `nan`.
Expected: Cannon should clear the 353.5 HP random baseline. If Fireball still
does not move, the trunk is the binding constraint → A2.

**A2. High-resolution skip connection (the recommended structural fix).**
`cnn_trunk` is a `nn.Sequential`; split it so the **pre-pool** 16×34×18 feature
map is available, and concatenate it into `place_up`'s final stage (which is
already at 36×20 — crop/pad to align). This gives the placement head a full-
resolution path *without* touching the features the card head, critic, LSTM and
aux head consume, so nothing else in the network is disturbed.
- Cost: one 3×3 conv at full resolution. The placement head is already ~41% of
  update time; measure before and after, and note `place_up` got **1.8× faster**
  when `ConvTranspose2d` was replaced, so there is headroom.
- **This changes `place_up`'s shape → the placement head will not load from
  existing checkpoints.** `load_state_dict_flexible` warm-starts everything else
  and reinitialises it, which is the same trade the 2026-08-09 checkerboard fix
  made. Say so when proposing it.

**A2b. THE NEXT STEP, and it is not more distillation: the entropy coverage
term will erode what A2 just bought.** This is a mechanical argument, not a
measurement, and it should be measured before it is trusted —
`PLACEMENT_COVERAGE_COEF` adds an *entropy bonus* on one uniformly-sampled
affordable slot per step. For a card the policy does not play, that bonus is the
**only** placement gradient in the whole objective, and it pushes the map toward
uniform. A distilled Fireball map is exactly such a card's map. So the two
mechanisms are in direct opposition: distillation puts mass on the right cell,
coverage pushes it flat, and coverage runs for the whole of training.

The fix follows the conclusion this project already reached twice — *closing a
coverage hole needs a TARGET, not noise* — so the coverage term should carry the
advisor's map rather than entropy: for the sampled slot, if the advisor has a
rule for that card, add KL to its (masked) score map; otherwise fall back to the
entropy bonus as today. The pieces now exist: `tactics.building_score_map`,
`tactics.spell_catch_map`, `distill_tactics.collect(..., want_maps=True)` and
`prove_hires.soft_target_logits`. Compute the target once per rollout step (not
per PPO epoch) and buffer it; the advisor costs ~0.2 ms per call.

Note the one result that argues against assuming this will work: the soft target
in A1 was *also* a target rather than noise, and it bought nothing. The
difference is that A1 replaced a good hard target with a soft one, while this
replaces pure noise with a target — but that is an argument, not evidence.

**A3. Unfreeze the trunk — only if A1+A2 are insufficient.** Ordered last
because it is the most invasive: the trunk feeds the critic, and the critic is
the scorer decision-time search depends on. If you do it, keep the critic frozen,
add a KL anchor on the card head, and verify critic drift is 0.000000 the way
`distill_tactics.py` already does.

**How to know it worked.** `prove_placement.py` is ready and includes an
**advisor arm as the ceiling**:
```bash
python_ai/venv/Scripts/python.exe python_ai/prove_placement.py \
    --seed model_weights_selfplay.pth --control <before>.pth --treatment <after>.pth --episodes 8
```
Targets, in order: beat **random** (Cannon 353.5 HP, Fireball 0.276 elixir), then
approach the **advisor** (564.1 / 2.405). Report modal share **next to top-1
probability** — modal share alone degenerates on a near-uniform distribution
(§2.1) and will lie to you.

### Path B — decision-time search, now with good candidates

CLAUDE.md: 1-ply search buys **+0.319** win rate at 2.2× wall clock, and 87% of
its overrides are "wait where greedy plays". The engine is ~150× cheaper than
one network forward, so **scoring is the entire budget**, not simulation.

Previously, "better candidate proposals did not make the expert better"
(0.940 vs 0.944) — but that was measured with the **collapsed** placement head
proposing candidates. It now has a source of genuinely good cells.

**B1.** In `search_ab_test.py::_build_candidates`, add the advisor's cell for
each affordable card as an extra candidate alongside the policy's top-K. Beware
the no-op duplication bug already documented there: `(NOOP, gx, gy)` and
`(NOOP, 0, 0)` are the same action and scoring both silently destroyed an earlier
run.
**B2.** Measure search **on top of the hybrid**, not on top of the raw neural
policy — the hybrid is the new baseline and the comparison must be against it.
**B3.** If search + hybrid beats hybrid, expert-iterate it back with the
**value-distribution** recipe (`expert_iteration.py --train-dist`, T chosen from
`--target-entropy`, trunk frozen), which is the only distillation variant that
has ever converted here (+0.045, p=0.0074). Hard-label argmax does not work.

**Live constraint:** the loop acts at 1 Hz and search costs 2.2× wall clock. It
is affordable in simulation for training labels; whether it fits the live budget
is unmeasured.

---

## 6. Traps discovered this session — do not re-pay for these

- **Warm-starting a converged policy into a FRESH training state re-arms the
  initial entropy target and the controller dissolves the policy.** Seeding a
  bare `state_dict` takes `train.py`'s legacy path, resets `episodes_completed`
  to 0, so `placement_entropy_target(0)` returns 0.65 against a policy at 0.11.
  Both arms of the first A/B went to near-uniform placement. **Always seed a full
  checkpoint** — `scratchpad/make_seed.py` shows the required keys.
- **A short resumed run writes NO checkpoint.** `last_save_ep = episodes_completed`
  on resume, so the first save is 500 episodes later and `timeout` kills the
  process before the end-of-loop save. Use `CLASH_SAVE_EVERY`.
- **A stage-5 resume flips to `random_opponent` after 100 episodes** (win rate
  ≥ `PHASE2_ENTRY_WIN_RATE`, and `PHASE2_MIN_CURRICULUM_STAGE` is 4), swapping
  the opponent's deck mid-experiment. Pin it with `CLASH_PHASE2_ENTRY_WIN_RATE=2.0`.
- **`F.kl_div` over `placement_mask`'s `-inf` cells gives `nan`**
  (`0 * (-inf − -inf)`). Use `masked_kl()`.
- **Modal share degenerates on a near-uniform distribution** — the argmax of a
  flat map is arbitrary but deterministic, so a dissolved head reports 99% modal
  share. Always read it next to top-1 probability.
- **Stepping the LSTM once per query** instead of once per timestep silently runs
  the reference policy at double clock. `prove_placement.step_net` does it right.
- **`get_elixir_value_killed_by(CANNON)` is the wrong metric for a building** —
  it credits only its killfeed and scores distraction, most of its job, at zero.
  Use tower HP preserved against a no-building counterfactual.

---

## 7. Open, and explicitly not done

- The placement head is **still worse than random**. §5 is the fix.
- The coverage term and the solvency term are both **on by default** and both are
  measured to do nothing useful on their own. Neither is load-bearing for the
  +11.8; disable freely while experimenting.
- ~~**Nothing here has been run against the live emulator.**~~ **DONE
  2026-08-14 — the hybrid runs live and holds its budget.** One Training Camp
  match, `--policy neural --act --ensure-match`, DirectML:

  | | |
  |---|---|
  | decisions | 250 in 260 s, **0.96 Hz** |
  | decisions over the 1000 ms budget | **0/250** |
  | board age | mean 511 ms, p95 852, over the staleness cap on **0/250** |
  | perception thread | 663 boards at 2.55 Hz, **0 errors** |
  | placements issued | 18, with **17/18** confirmed by at least one oracle |
  | advisor / gate | both active (`advisor ON for card ids [2, 7, 25]`) |

  No crash, no actuator drops, cadence held. Two honest caveats: the **advisor
  fired only once** in the match (the commander's take-up of Cannon/Fireball/
  Giant is ~0, which is the known upstream problem, so the live sample of the
  override path is thin), and **the bot lost the match 0–3** to Trainer Red —
  live play is gated by perception fidelity, not by this change, and no live
  win rate has ever been measured. The reproducible part of the check is the
  frame-replay path, which exercises the identical chain deterministically:
  ```bash
  perception/.venv/Scripts/python.exe -m perception.live.mvp_loop \
      --policy neural --frames perception/assets/live/match_practice_01 --seconds 45
  ```

- **The live loader was strict and the architecture change would have crashed
  it.** `NeuralPolicy.__init__` called `load_state_dict` with the default
  `strict=True`, so every pre-2026-08-14 checkpoint would raise on the new
  `place_hires` keys. Now loads with `strict=False` plus an explicit check that
  the *only* missing keys are that branch — `strict=False` alone would also
  swallow a genuinely wrong checkpoint. Caught by running the replay path, not
  by any test.
- Giant initiation is off. The Giant *placement* rule is validated (535.6 vs 3.3)
  but the commander almost never plays Giant, so that rule is currently latent.
  Reviving it means initiation, which measured harmful — revisit only with a
  cost-aware trigger, not the flat one that failed.
