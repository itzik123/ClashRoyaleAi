# The placement-head coverage hole — 2026-08-14

Three reported behaviours (cowardly Cannon, blind Fireball, defensive apathy)
were investigated. **Two of them are the same bug**, and it is neither a reward
bug nor a training-budget problem. The third is a different mechanism than the
one hypothesised, and the measurement inverts it.

Everything below is measured. Nets probed: `model_weights_dist_e3.pth` (the live
net), `model_weights_selfplay.pth` (2026-08-11), `model_weights.pth`
(2026-08-09). Opponent is the C++ `HeuristicOpponent` at 1.5x elixir — the
regime CLAUDE.md specifies, because at 1.0x the policy wins ~100% and every
comparison saturates.

---

## 1. What is actually wrong: the placement head is a CONSTANT function

40 greedy episodes, ground truth taken from the engine's own per-tick entity
list rather than the lossy observation:

| card | plays | modal cell | modal share | H(place\|card) |
|---|---|---|---|---|
| Archers | 238 | — | — | 0.191 |
| Mini PEKKA | 230 | (14,15) | 19.0% | **0.086** |
| Minions | 229 | (2,15) | 9.6% | 0.117 |
| Valkyrie | 220 | (1,1) | 9.7% | 0.193 |
| Musketeer | 218 | (1,1) | 9.8% | 0.205 |
| **Cannon** | **24** | **(11,0)** | **91.0%** | 0.098 |
| **Fireball** | **2** | **(11,0)** | **58.4%** | 0.141 |
| **Giant** | **2** | **(11,0)** | **54.1%** | 0.147 |

Three of eight cards are dead and all three return the *same* cell — (11,0),
our own back row behind the King.

**Entropy is not the discriminator, and this matters.** Mini PEKKA has the
LOWEST entropy in the deck and is the most-played card. What separates a healthy
head from a broken one is **modal-cell stability across states**: a good head is
sharp but moves its mode with the board (19%), a broken one returns the same cell
regardless of what is happening (91%). CLAUDE.md already prescribes
`Entropy/Placement_ByCard_Min` as the conditional-collapse detector; on this net
that detector reads Mini PEKKA as the worst card in the deck and Cannon as
healthier than it. **It is the wrong statistic.** Modal share is the right one.

That is the fifth instance of this project's recurring failure mode — an
aggregate that cannot see the thing it is checking.

## 2. It is not a valuation. The policy places worse than chance.

The obvious competing explanation is that these cards are genuinely bad and the
policy correctly declines them. That is testable, and it is false.

Using `env.snapshot()` + `inject()` (injection costs no elixir, so the rest of
the match is untouched), a Cannon was placed at each candidate cell and the
engine run forward one full 300-tick Cannon lifetime. Scored two ways over 449
threatened states:

| Cannon cell | tower HP preserved | elixir value killed |
|---|---|---|
| **policy's own cell** | **121.3** | **0.074** |
| random legal cell | 395.7 | 1.153 |
| advisor (`tactics.py`) | 665.1 | 1.160 |
| oracle (best evaluated) | 1388.8 | 4.437 |

Paired, same states: **random − policy = +274.4 HP, 95% CI [+221.0, +327.9]**.
The policy's Cannon killed nothing at all in 98.4% of placements.

**A policy cannot be correctly valuing a card it places significantly worse than
random.** Declining a bad card is a valuation; placing it worse than chance
requires an actively wrong function. Hypothesis (a) is dead.

Same story for Fireball, scored against `get_elixir_value_killed_by` over 1,059
states — the exact quantity `spell_value_shaping` is paid on:

| Fireball aim | elixir value killed | caught anything |
|---|---|---|
| **trained policy** | **0.022** | **0.1–0.5%** |
| random legal cell | 1.103 | — |
| advisor | 2.524 | 51–61% |
| oracle | 2.944 | 100% |

The policy's Fireball aim is **50x worse than aiming at random**.

## 3. Root cause: both loss terms score only the CHOSEN card

`train.py`'s update computes placement through
`placement_given_card(..., card_idx=mb_card_actions)` and nothing else. So:

* the **actor loss** flows only through the chosen (card, cell) pair;
* the **entropy bonus** is gated by `mb_placed`, i.e. steps that actually placed
  a card.

A card the policy has stopped playing therefore receives **exactly zero**
placement gradient from either term, forever. This is mechanical, not
statistical: `card_id_embed` is `nn.Linear(num_card_ids, 16, bias=False)`, so
column *c* belongs to card *c* alone, and nothing downstream of the LSTM
consumes an unchosen card's embedding. `test_python_ai.py` asserts the
gradient is `== 0.0` in exact arithmetic.

That closes a **deadlock**:

```
card not played  ->  no placement gradient  ->  map frozen at a fixed cell
      ^                                                     |
      |                                                     v
card head suppresses it  <-  playing it is genuinely bad  <-'
```

It is self-sustaining, which is why more training never fixed it.

### Dating the onset

| net | date | Cannon modal cell | modal share | H |
|---|---|---|---|---|
| `model_weights.pth` | 08-09 | (3,15) | **9.5%** | **0.368** |
| `model_weights_selfplay.pth` | 08-11 | (11,0) | 86.0% | 0.031 |
| `model_weights_dist_e3.pth` | 08-13 | (11,0) | 91.0% | 0.098 |

The 08-09 net placed its Cannon *forward, at the river, state-dependently*. The
collapse appears in the 08-11 net, bracketing commit `e16cdd7` of that day
("Measure placement entropy on real placements, not on no-ops", 08-11 15:48).

**That commit was correct for the defect it targeted** — the entropy controller
was regulating the noise of non-actions and pinning its coefficient to the
floor. It had an unmeasured side effect: the no-op steps it stopped rewarding
were the only thing holding open the placement maps of cards that are never
played.

The signature is specific enough to be worth stating. In the pre-fix net,
**Cannon and Giant had the two HIGHEST per-card placement entropies** (0.368,
0.409) precisely because they were rarely played and so were shaped mostly by
the unconditional entropy bonus. After the fix they have the lowest. The rank
order inverted for exactly the unplayed cards. Little else predicts that.

A confounder was checked and ruled out: the placement-legality mask landed the
same day (`24a2c78`, 13:52). But (3,15) — the old net's modal Cannon cell — is
still legal under it, as is (11,0) for every card. The head abandoned an
available cell; it was not masked off one.

## 4. Issue 3 is a different bug, and the hypothesis was backwards

Reported as apathy driven by elixir hoarding. P(play) does look nearly flat
against incoming threat:

| threat on our half | decisions | P(play) | mean elixir |
|---|---|---|---|
| none | 7865 | 9.4% | 2.53 |
| ≤4 elixir | 1506 | 11.9% | 2.50 |
| 5–8 | 963 | 11.9% | 2.48 |
| >8 | 922 | 13.7% | 2.71 |

But conditioning on whether *anything was affordable* inverts the reading:

| threat | P(play) all | P(play \| ≥3 elixir) |
|---|---|---|
| none | 9.4% | 27.8% |
| ≤4 | 11.9% | 33.3% |
| 5–8 | 11.9% | 34.3% |
| >8 | 13.7% | **34.9%** |

And during a big push:

```
0-3 elixir:  561 decisions (60.8%)   P(play)  0.0%
3-5 elixir:  291 decisions (31.6%)   P(play) 33.7%
5-7 elixir:   49 decisions ( 5.3%)   P(play) 38.8%
7-11 elixir:  21 decisions ( 2.3%)   P(play) 42.9%
```

**The agent is not hoarding — it is bankrupt.** 60.8% of decisions during a
devastating push are below the cost of the cheapest card in the deck, so P(play)
is 0.0% by arithmetic, not by choice. Across all decisions it sits under 3
elixir **65.3%** of the time, spending ~105 elixir per episode against ~98 of
income. Its response to threat is real but weak, and it is usually broke when
the answer is needed.

Two contributing causes, both already documented elsewhere in this repo:

* Its one dedicated defensive building is the Cannon, which section 1 shows is
  dead. Fixing the deadlock is therefore a direct part of fixing defence.
* The project's own strongest result already says what the remedy is:
  decision-time search buys **+0.319** win rate and **87% of its overrides are
  "wait where greedy plays"** (1,847 vs 137). The critic knows when to hold;
  the action head does not use it. Expert iteration recovers ~14% of that.

`W_ELIXIR_OVERFLOW` penalises elixir above 9 and nothing rewards holding a
defensive reserve, so the shaping pushes in the spend direction only. That is a
plausible contributing factor but **it is not established as the cause and no
reward change is claimed here.**

---

## 5. What was changed

### `model.py` — `forward_sequence(..., extra_card_idx_seq=None)`
Optional second placement pass over the same `flat_hx`/`spatial`. **No new
parameters, so no checkpoint is invalidated.** Returns `None` when not
requested; both trainers updated for the new return arity (they are the only
callers).

### `train.py` / `train_selfplay.py` — the coverage term
Each timestep samples one *affordable* hand slot uniformly
(`placement_coverage_slots`) and adds its normalized placement entropy to the
loss at a fixed `PLACEMENT_COVERAGE_COEF = 0.02`.

* It is a **regularizer, not part of the PPO objective** — it never touches
  `new_logprobs`, so the ratio is unaffected and the update remains a valid PPO
  step. Pinned by `test_coverage_does_not_change_the_ppo_ratio`.
* The coefficient is **fixed, not tied to the adaptive `ent_coef_place`**. The
  controller lowers that coefficient when real placements are sharp enough,
  which is exactly the condition under which an unplayed card is freezing;
  coupling them would switch coverage off precisely when it is needed.
* The **reported** `Entropy/Placement_Measured` still averages over real
  placements only, so the 2026-08-11 fix is preserved. A new
  `Entropy/Placement_Coverage` series is the freeze detector.
* Cost: the placement head is ~41% of update time and this runs it once more.
  Sampling one slot rather than all four keeps that at ~1x extra instead of ~4x,
  and still reaches every card ~1000 times per rollout.

### `tactics.py` — deterministic advisor
Spell aiming and defensive building placement, computed **from the observation**
so it behaves identically in simulation and on the live screen (the perception
encoder is pinned bit-equal to `getObservationForTeam(0)`).

Two things measured here came out against the obvious guess, both on 1,059
states paired against the engine oracle:

| | lead=0 | lead=10 (1.0 s) |
|---|---|---|
| damage-capped | **75.5%** | 52.4% |
| kill-weighted | 75.0% | 51.2% |

* **Leading the target is harmful and `lead_ticks` defaults to 0.** Fireball has
  `spellDelayTicks=10`, so leading looks obviously right and costs 23 points of
  achievable value. The blast radius is 2.5 tiles while 1 s of movement is only
  0.6–1.6 tiles at post-2026-08-07 speeds, so a moving target stays inside the
  blast anyway — while a target standing still (engaged in combat, the common
  case in states worth spelling) gets led straight off the edge of it. Leads of
  0–6 ticks were indistinguishable; only the full 10 hurt.
* **Weighting by what the spell kills rather than damages changes nothing**
  (75.0 vs 75.5). That variant was built, measured paired, and deleted.

### Tests
`python_ai/test_python_ai.py` (the coverage section), 5 cases:

```bash
python_ai/venv/Scripts/python.exe -m pytest python_ai/test_python_ai.py -q -k coverage
```

`test_unchosen_card_gets_no_gradient` asserts the *defect* (gradient exactly
zero) and is the test that fails on the old code path.

---

## 5b. THE RE-RUN: the entropy coverage term does NOT break the lock

The A/B in §6 was botched by a warm-start bug. It was fixed (full-checkpoint
seed, so `episodes_completed = 64309` and the entropy target anneals to its
FINAL 0.25; `curriculum_stage = 5`, which cannot auto-advance; the
mirror→random_opponent phase flip pinned shut) and re-run as three arms in
parallel, byte-identical code, ~80 PPO updates each, compared at matched
episodes ~64800:

| arm | coverage coef | solvency |
|---|---|---|
| A control | 0 | off |
| B treatment | 0.02 | off |
| D | 0.02 | on |

Scored by the engine, paired on 835 identical states (Cannon) / 1,573
(Fireball), states drawn by the shared seed policy:

| Cannon | modal cell | modal share | top-1 p | tower HP preserved |
|---|---|---|---|---|
| seed | (11,0) | 88.7% | 0.920 | 135.9 |
| control | (11,0) | 55.4% | 0.550 | 116.4 |
| **treatment** | **(6,0)** | **79.4%** | **0.051** | **182.0** |
| random legal cell | — | — | — | **394.7** |

control → treatment = +65.6 HP, bootstrap CI [+23.7, +108.8] but **sign test
p = 0.158** (91 better / 72 worse). When those two disagree, believe the sign
test. Fireball is unambiguous: treatment kills **0.000** elixir, identical to
control and seed, against 0.562 for a random cell.

**The lock did not break — it moved.** Top-1 probability of 0.051 (Cannon) and
0.006 (Fireball) is a near-uniform distribution (uniform = 1/612 = 0.0016). The
term did exactly what it was designed to do, raise entropy, and that turns out
not to be the same thing as fixing the defect: **the argmax of a flat map is an
arbitrary constant**, so a greedy policy still plays one fixed cell. The frozen
cell simply relocated from (11,0) to (6,0).

The general lesson, and it is the one worth keeping: **entropy is a MARGINAL
objective.** It says "be spread out", not "depend on the board". A card with no
other gradient has nothing telling it which cell is right in which state.
Closing a coverage hole needs a TARGET, not noise.

Second finding from the same run: **the control also partially unfroze**
(88.7% → 55.4% modal share, 5 → 22 cells used) with the coverage term switched
off, purely from the correctly-annealed entropy controller and shared-weight
updates. So part of the original collapse was the mis-set entropy target
documented in §6, not the coverage hole alone.

## 5c. What DOES work: the deterministic advisor

Same protocol, same states, the advisor scored as an extra arm:

| | Cannon (tower HP preserved) | Fireball (elixir killed) |
|---|---|---|
| seed policy | 12.1 | 0.000 |
| control | 55.5 | 0.000 |
| distilled (§5d) | 217.4 | 0.000 |
| random legal cell | 353.5 | 0.276 |
| **advisor** | **564.1** | **2.405** |

advisor vs the trained head: Cannon **+346.7 HP**, sign test p = 1.5e-25
(259 better / 73 worse); Fireball **+2.405 elixir**, p = 1.2e-119
(396 better / **0 worse**, n=950). The advisor beats a random cell by 1.6x on
Cannon and 8.7x on Fireball.

This is the answer to the original brief's option (B) for the Fireball, and the
measurement says it applies to the Cannon equally.

## 5d. Distilling the advisor into the head: significant, still insufficient

`distill_tactics.py` — advisor cells as a supervised target for the dead cards
only, trunk/LSTM/critic frozen (critic drift verified 0.000000), alive cards
held in place by a KL anchor so the shared `place_ctx`/`place_up` cannot drag
them.

Cannon improved significantly: **55.5 → 217.4 HP, +161.9, sign test p = 2.0e-06**
(43 better / 9 worse). Fireball did not move **at all** (0.000). Both remain
below a random cell.

The likely reason is architectural and worth recording: the placement head's
spatial signal comes from a **9×5 pooled feature map upsampled 4×**, on top of a
FROZEN CNN trunk. An exact-cell target is close to inexpressible at that
resolution, and for Fireball — which must localise an enemy clump anywhere on a
34-row board — the frozen trunk evidently does not carry the needed feature at
all. Cross-entropy fell 180.9 → 21.4 while exact-cell argmax match stayed at
0.0%, which is the signature of a target the head cannot represent rather than
one it has not yet learned.

Two implications: unfreezing the trunk is the next thing to try, and a soft
(neighbourhood) target is more honest than exact-cell at this resolution.

## 5e. Bankruptcy: fixed as a statistic, with no outcome gain

Two interventions were built and measured against §4's diagnosis.

**Potential-based shaping (`elixir_shaping.py`) — no measured effect.**
Phi(s) = -w * max(0, 4 - elixir)/4, discounted form, so policy-invariant by
Ng et al. Chosen because the failure looked like credit assignment rather than a
mis-specified objective: search optimises the SAME reward and gains +0.319 by
waiting, so waiting is already better under the current objective. 9 unit tests
including a telescoping check (the term must add ~0 to an episode's return) and
a "pure hoarder earns nothing" check. Measured over ~80 PPO updates (arm D vs
arm B, matched episodes): bankruptcy **-1.0 points, 95% CI [-2.7, +0.6],
p = 0.21**, and every other metric's CI contains 0. Policy-invariance is
precisely what makes it safe and also what limits it -- it can only change how
fast an optimum is found, and 80 updates is not that.

**An inference-time reserve gate (`tactics.SolvencyGate`) — works completely,
on the statistic.** Refuses spends that would drop below 4 elixir *while nothing
is attacking* (enemy HP on our half < 400); under threat it opens fully, so
spending to zero on a real answer stays available. 130 paired openings, same net,
bit-identical starts via `env.snapshot()`:

| | gate off | gate on | paired delta | p |
|---|---|---|---|---|
| bankrupt <3 elixir | 72.7% | **39.2%** | **-33.5 pts, 130/130** | **1.5e-39** |
| mean elixir | 2.22 | **3.86** | +1.63, 130/130 | 1.5e-39 |
| elixir spent / ep | 102 | 98 | -3.6 | 0.25 |
| tower HP lost / ep | 5509 | 5489 | -19 | 0.25 |
| win rate | 0.596 | 0.554 | -0.042 | 0.63 |

**Read the third row before the first.** Total elixir spent is statistically
unchanged, so the gate did not buy solvency by making the bot passive -- it
moved *when* the same spending happens. That is the objection this result has to
survive, and it survives it.

**And read the last two rows honestly: it does not convert.** Tower HP lost and
win rate are flat. Having elixir available does not help while what the bot does
with it is still worth less than a random cell (§5c). The three reported
problems are coupled, and fixing elixir timing alone yields no outcome gain --
which is itself the useful finding, because it says placement quality is the
binding constraint, not elixir.

## 5f. The hybrid: neural commander, deterministic tactical officer

Since three attempts to repair the placement head inside the network failed
(§5b, §5d) and the advisor is decisively better (§5c), the WHERE decision is
taken out of the network for Cannon, Fireball and Giant. `hybrid_policy.py`:

    commander (MicroRoyaleNet)  WHAT to play and WHEN -- untouched
    tactical officer            veto spends that would bankrupt us;
                                place those three cards where the advisor says

**A Giant rule had to be validated before it could be shipped**, since the brief
asked for one and none existed. Injecting a Giant at a candidate cell and
running 600 ticks, scoring enemy tower damage over 913 states:

| Giant placement | enemy tower damage |
|---|---|
| the policy's own cell | 3.3 |
| explicit back row | 3.0 |
| random legal cell | 94.3 |
| **bridge, weaker-defended lane** | **535.6** (+532.3, p = 3.0e-87) |

**INITIATION WAS BUILT, MEASURED HARMFUL, AND REMOVED.** The commander's take-up
of the dead cards is 0.06/0.00/0.00, so it seemed obvious that overriding only
"where" would fire too rarely to matter and the officer should also START those
plays. A 5-arm ablation at n=30 says otherwise:

| arm | win rate | tower HP DEALT/ep | cannons initiated |
|---|---|---|---|
| neural | 0.600 | 7627 | — |
| gate only | 0.633 | 6659 | 0.00 |
| **placement only** | **0.733** | 7655 | 0.00 |
| initiate | 0.467 | 5390 | 4.03 |
| full stack | 0.233 | 4216 | 4.73 |

Initiating a Cannon four times a match crowds out the commander's own offence --
tower damage DEALT falls 3,410/episode on the full stack (6 better / 24 worse,
p = 0.0014) and win rate with it (-0.367, p = 0.013). **Defending better is
worthless if it is paid for with the attack.** The placement override survives
precisely because it is PASSIVE: it changes where a card lands and never how
often one is played, so it cannot spend elixir the commander did not already
commit.

A second mechanism fault, found the same way: a FLAT solvency reserve is
anti-offense, because building a push means spending exactly when nothing is
attacking, which is the only case a flat gate blocks. The reserve now shrinks to
`min(reserve, opponent_elixir)` using the network's own auxiliary head.

### Result — exploratory, n=250 paired

| metric | neural | hybrid | paired delta | p |
|---|---|---|---|---|
| **win rate** | 0.646 | **0.720** | **+0.074** [-0.004, +0.150] | **0.070** |
| bankrupt <3 elixir | 72.8% | 41.6% | -31.1 pts, 250/250 | 1.1e-75 |
| mean elixir | 2.22 | 3.45 | +1.23, 250/250 | 1.1e-75 |
| elixir spent / ep | 102 | 103 | +0.5 | 0.30 |
| tower HP DEALT / ep | 7295 | 7029 | -267 | 0.49 |
| tower HP lost / ep | 5102 | 4467 | -634 | 0.066 |

Offence is preserved (damage dealt flat, p=0.49) now that initiation is gone,
spending is unchanged, and defence improves directionally. **But +0.074 at
p = 0.070 is not significance**, and n was pre-registered at 250, so this run
cannot be extended -- that is optional stopping, and this project has already
had a +0.105 at p=0.044 evaporate at 4x the power.

### Result — CONFIRMATORY, independent run, n=600 paired

| metric | neural | hybrid | paired delta | p |
|---|---|---|---|---|
| **win rate** | 0.584 | **0.703** | **+0.118** [+0.067, +0.171] | **1.9e-05** |
| bankrupt <3 elixir | 72.7% | 41.7% | -31.0 pts, 600/600 | 4.8e-181 |
| mean elixir | 2.22 | 3.45 | +1.23, 600/600 | 4.8e-181 |
| elixir spent / ep | 103 | 105 | +1.6 | 0.042 |
| plays / ep | 29.3 | 30.4 | +1.1 | 0.003 |
| **tower HP DEALT / ep** | 6545 | **7089** | **+544** | **0.010** |
| **tower HP lost / ep** | 5575 | **4564** | **-1011** | **7.4e-09** |

**Both runs agree in direction and the confirmatory one is unambiguous:
+11.8 win-rate points, 171 better / 100 worse, p = 1.9e-05.** Offence did not
merely survive, it improved (+544 damage dealt, p = 0.010) while damage taken
fell by 1,011 (p = 7.4e-09) -- the hybrid attacks more effectively AND defends
better, on essentially unchanged spending (+1.6 elixir/episode).

Two cautions to read alongside it. The neural baseline measured 0.646 in the
first run and 0.584 in the second, which is the control-arm variance CLAUDE.md
records (0.570-0.775) and exactly why both arms must share an opening. And
`init_cannon/fireball/giant` are all 0.00, confirming the entire gain comes from
the passive placement override plus the solvency gate -- nothing here is
initiating plays.

**This is the first change in this investigation to move win rate.** It works by
routing around the defective component rather than repairing it: the network
keeps the decision it is good at (what and when) and loses the one it is
measurably worse-than-random at (where).

## 6. The FIRST training A/B was INCONCLUSIVE, and why

Two arms were run from the same seed (`model_weights_selfplay.pth`), byte-identical
code, `CLASH_PLACEMENT_COVERAGE_COEF` 0.02 vs 0, compared at matched episodes
(507 vs 506, both curriculum stage 3). **It does not answer the question**, and
the reason is worth more than the result would have been.

| | seed | control ep506 | treatment ep507 |
|---|---|---|---|
| Cannon H(place\|card) | 0.036 | 0.995 | 0.996 |
| Fireball | 0.124 | 0.999 | 0.999 |
| Giant | 0.108 | 0.998 | 0.998 |
| `Entropy/Placement_Measured` | — | 0.110 → 0.824 | 0.116 → 0.829 |
| `Entropy/Placement_Target` | — | **0.6499** | **0.6499** |

Both arms dissolved into near-uniform placement (top-1 probability 0.002–0.008
against a uniform 1/612 = 0.0016). The coverage term is not what did that, and
neither is its absence — **the adaptive entropy controller did, in both arms.**

The cause is my harness, not the fix. Seeding a bare `state_dict` takes
`train.py`'s legacy-checkpoint path, which warm-starts the weights and resets the
training state, so `episodes_completed` becomes 0 and
`placement_entropy_target(0)` returns `ENTROPY_TARGET_PLACEMENT_START = 0.65`.
The seeded policy measures 0.11. The controller saw a 6x shortfall against a
target meant for episode 0, and spent the whole run inflating placement entropy
to close it. That force is roughly an order of magnitude larger than a 0.02
coefficient, so it swamped the treatment.

**This is a real trap, not just an experimental slip: warm-starting a converged
policy into a fresh training state re-arms the initial entropy target and the
controller will dissolve the policy.** It is the same shape as the exploiter's
2026-07-31 entropy-scaling failure (51.9% → 86.7% of max, 28 → 260 effective
cells) — a mis-set entropy target destroying a working placement head — and it
would hit anyone resuming a phase-2 net through phase 1's legacy path.

A second, smaller lesson: **modal share degenerates when a distribution is near
uniform.** Both arms report a 90–100% modal share at (6,0) while being flat —
the argmax of an almost-flat map is arbitrary but deterministic. Modal share is
only meaningful read next to top-1 probability; the section-1 table has top-1 at
0.62–0.83, which is what makes 91% there a genuine collapse rather than an
artifact.

**What still stands without this run**, and what the case for the fix actually
rests on:

* the defect is *mechanical*, and the test asserts an exactly-zero gradient
  rather than a small one;
* the behavioural consequence is measured against the engine's own accounting,
  not against a model (§2);
* the fix is a strictly additive regularizer that provably cannot corrupt the
  PPO ratio (§5), so its downside risk is bounded to the cost of one extra
  placement pass.

**What is still owed**: a run seeded with a *full* checkpoint (so
`episodes_completed` and the annealed target are correct), long enough for the
policy to re-sharpen, measuring per-card modal share and top-1 probability
together. Until then the coverage term is justified by mechanism and by the
behavioural measurements, **not** by a demonstrated training outcome.

## 6b. Injecting the advisor does NOT (yet) raise win rate

`python_ai/tactical_ab.py`, 40 paired openings, 1.5x opponent elixir, three arms
from a bit-identical opening:

| arm | win rate | forced cannon | forced fireball |
|---|---|---|---|
| control (greedy) | 0.775 | 0 | 0 |
| advisor cells | 0.725 | 23 | 6 |
| own-cell control | 0.700 | 22 | 3 |

advisor − control = **−0.050**, 95% CI [−0.171, +0.071], exact McNemar
**p = 0.69** (2 better / 4 worse / 34 tied). **No effect is demonstrated, and
the point estimate is negative.** Do not read the advisor > own-cell ordering
(0.725 vs 0.700) as support for anything — it rests on 1 vs 4 discordant pairs.

Two things this does establish:

* **The solvency gate works.** The ungated first version forced 5.5 Cannons and
  2.75 Fireballs per episode; gated, it forces 0.58 and 0.15 — a ~10x
  reduction — and both forced arms are now within noise of the control instead
  of losing outright.
* **The control arm measured 0.775 here** against 0.625 / 0.570 / 0.700 / 0.634
  in CLAUDE.md's four previous runs of the same net. That spread *is* the reason
  n=40 cannot resolve anything, and it is why no claim is made.

The honest conclusion is the useful one: **putting the Cannon in the right place
does not by itself fix the bot.** That is consistent with §4 — the policy is
broke 65% of the time and its timing is state-insensitive, so a better cell for
a card it rarely affords is a small lever. The placement collapse is a real,
proven defect worth fixing on its own terms; it is not the whole of what is
wrong.

## 7. Honest limits

* The **coverage fix restores exploration, it does not supply skill.** It stops
  a card's map from freezing and lets the policy gradient act again. Uniform
  placement is only worth having because it measured 3.3x better than the frozen
  cell — that is a low bar, not a good policy.
* Win rate is **not** established for any change here. At stage 0 (1.0x
  opponent elixir) the net wins ~100%, so the validation run measures the
  collapse statistic, not strength. CLAUDE.md's own figure — the control arm
  measured 0.625 / 0.570 / 0.700 / 0.634 across four runs at 1.5x — means a few
  hundred paired trials would measure that variance, not the treatment.
* The advisor captures ~75% of achievable spell value and ~48% of achievable
  Cannon value. It is better than the policy by a wide margin and well short of
  optimal.
* No reward-function change was made, and that is a deliberate conclusion
  rather than an omission. The Cannon reward pricing was already corrected on
  2026-08-06; this investigation found no evidence it is wrong, and the
  measurements point upstream of the reward entirely. Patching shaping on this
  evidence would be acting against the measurement — the failure is that a card
  the policy stopped playing stopped receiving gradient, which no reward term
  can reach.
* **Issue 3 has a proven diagnosis and no implemented fix.** Bankruptcy is
  established (§4); the remedy is not. The candidate is a solvency term
  mirroring `W_ELIXIR_OVERFLOW` — penalise being below a reserve *while a threat
  is present*, so it is state-conditional and cannot be farmed by hoarding
  (overflow still punishes that). It is deliberately NOT implemented, because
  this file's own §6 shows how easily an unvalidated change to the loss is
  swamped or misread, and CLAUDE.md records several shaping terms that
  backfired. What is already proven to help is decision-time search (+0.319),
  whose entire content is "wait, conditionally" — the exact behaviour the
  bankruptcy measurement says is missing.
