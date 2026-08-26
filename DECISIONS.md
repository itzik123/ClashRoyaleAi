# DECISIONS — how the learning mechanism got here

Extracted from `CLAUDE.md` on 2026-08-24, unchanged. This is the **narrative**
half of the knowledge base: roughly chronological, every entry here because it
was *measured*, and several reversing an earlier conclusion.

**`CLAUDE.md` remains the operational reference** — rules, engine facts, the
training mechanism, measured baselines, measurement discipline, open problems
and layout. Read that first. Come here for *why* a number is what it is, what
was tried and refuted, and which traps have already been paid for.

**Nothing in this file was rewritten in the move.** Section titles, figures and
dates are exactly as they stood in `CLAUDE.md`, so any passage cited elsewhere
by its heading still resolves — the citation just points at this file now.

**Refuted ideas are load-bearing here.** A measured null is recorded as
prominently as a win, because the cost this project keeps paying is
re-proposing something that was already tried. `TODO.md`'s "Deliberately NOT on
this list" is the index to those; the evidence is below.

---

## How the learning mechanism got here

Roughly chronological. Every entry is here because it was *measured*, and
several reversed an earlier conclusion.

**Action-space correctness (the largest single win).** Before affordability
masking, `GameManager::playCard` returned `false` silently on insufficient
elixir — no exception, no penalty, no signal. Measured over 243 real decision
steps: **74.9% of steps attempted an unaffordable card and only 11.5% actually
placed one**, so ~87% of every rollout's stored actions did not affect the world
and their advantages were pure noise entering the gradient. Structural, not
transient: 0.035 elixir/tick × 10 ticks = 0.35 elixir per decision against
3–5-cost cards, so only 27.2% of steps had *any* legal play. Rejected-action
rate went 86.8% → **0.0%**.

**Gaussian → categorical placement.** The old head sampled from a `Normal` with
a learned `placement_log_std`. A Gaussian's entropy is `log σ + const`, so the
entropy bonus's gradient w.r.t. `log_std` is a **constant 1**, data-independent —
a fixed upward push that the noisy policy gradient loses to. Measured on the
checkpoints: `log_std` went −2.0 → −1.86 over 68,515 episodes, i.e. σ *grew*,
reaching **2.66 tiles** of noise in x. The two bridges are 10 tiles apart, so
the agent's own noise was half the distance it needed to resolve. Categorical
over 612 cells fixed it at the root: entropy bounded by `log(612)`, falls
naturally as the policy sharpens, exact tile, and maskable.

**Dead ability heads removed** (see Network above).

**Spell y-range.** `target_y` was capped at `getOwnHalfMaxY()` for every card,
so Fireball could not cross the river — 1/8 of the deck unusable for its purpose
at any amount of training. Fixed by the `get_card_info` binding plus a
card-conditional mask, not by widening the space unconditionally.

**Deck choice is a learnability constraint, not flavour.** With the
5-cost-heavy Giant deck the agent **never played Giant once** across four full
runs; only 5–7 of 8 cards were ever used. Cause: at 0.35 elixir/decision a
5-cost card is legal only after ~14 consecutive non-spending steps, so its slot
is masked out nearly every time it is checked and never accumulates gradient.
It was replaced by a 3–4 cost / avg 3.50 / spread 1 Hog cycle where no card is
systematically starved — the fix is the cost curve, **not** a bigger entropy
bonus (that was tried and measured to fail).

**...and then deliberately reverted, 2026-07-30.** `DEFAULT_DECK` is the Giant
deck again, because it is the deck in the 8 real recordings and matching them is
worth more than the cost curve — human demonstrations are the highest-value
unblocked item and unusable against a different deck. The risk above is
knowingly re-accepted. Two things differ from when it was measured: the
affordability mask makes "cannot afford Giant" observable rather than a silent
`playCard` failure, and card entropy is now adaptively held near 0.35 of max.
**Open question, not a settled one** — watch whether Giant is ever played.

**Adaptive per-head entropy** (`coef *= exp(rate·(target − measured))`),
replacing hand-tuned coefficients. Motivated by measurement at episode 20,245 of
the run it replaced: placement entropy at 26% of max, **card entropy at 0.4% of
max — effectively a constant**. A shared scalar coefficient could not fix both;
scaling it to rescue the card head crushed placement (top-5 cell share 36% →
70%, left-lane usage 33% → 13%).

`ENTROPY_ADAPT_RATE = 0.5` was lowered to 0.15 and then **reverted**: the lower
gain did smooth the controller (mean |entropy − target| fell, placement stopped
free-falling) but the resulting policy measured *worse* — 81.7% [74–88] vs
96.7% [92–99] against scripted opponents at stage 3, CIs not overlapping. The
current reading is that the large swings act as periodic exploration re-boosts.
Caveat recorded in the code: **one run per configuration**, so this ordering may
not survive replication.

`ENTROPY_COEF_FLOOR` was raised 0.002 → 0.01 because at 0.002 the floor was an
off switch rather than a floor and the head collapsed until the controller
noticed.

**Explained variance, not raw critic MSE.** I called the critic broken three
times off a flat MSE around 0.11. Raw MSE is bounded below by irreducible noise
in the returns and is uninterpretable alone. Measured explained variance at the
time: **+0.64 — healthy.** `Loss/Critic_Explained_Variance` exists because of
this. Value clipping is scaled to the batch's return std (`VF_CLIP_STD_FRAC`)
after measuring that at episode 14,666, 46.1% of samples needed the critic to
move more than the old fixed 0.2.

**Truncated BPTT with stored hidden states**, `bptt_chunk = 25`,
`ppo_epochs = 4`, `num_minibatches = 8`. Replaced treating the whole 500-step
rollout as one sequence with `num_minibatches=1, ppo_epochs=2` — exactly 2
optimizer steps per 4,000 transitions, 79% of the time spent unrolling one
500-long chain. Throughput went **228 → 4,224 gradient steps/hour at lower
wall-clock per update** (27.3 s vs 31.6 s).

**Random opponent → `HeuristicOpponent`** (C++). The built-in random
`opponentTurn()` never punished a bad trade, read a push, or played around a
threat. This was the change with the clearest effect on training quality — and
the uncomfortable lesson was that ~100 lines of heuristics beat a
47,000-episode policy **71%**.

**Timeouts stopped being free.** `TimeoutRules` (new header, MatchRules/
GameManager/Board/Tower untouched) decides a timed-out match on surviving tower
count, then on lowest weakest-tower HP, and only an exact tie is a draw. Before
this every timeout scored 0.0, teaching the agent that running the clock out was
neutral.

**Stage checkpointing.** Added after losing the ability to answer a question:
a narrow policy that had just cleared a stage's 80% gate could not be re-probed
because the checkpoints were already gone.

**A hypothesis I disproved.** I believed the policy under-weighted Fireball
because of a learning failure. Forcing Fireball usage dropped win rate
**97% → 23%** (CIs [83–99] vs [12–41]). The low weighting was a *correct
valuation*. This is the reason the current default is to measure before
"fixing" an apparent behavioural gap.

**2026-07-29, the big representational change.** Observation 6253 → 13606
(21 channels + 9 appended scalars: elapsed time, both sides' cumulative elixir
spend, 6 tower HPs); dense placement head → fully-convolutional; auxiliary
opponent-elixir head; annealed placement entropy target; exploiter league; BC
pipeline. Rationale for the observation work: it had **no temporal component at
all**, so tick 100 and tick 3500 with the same board were the same input — which
made the state non-Markovian for two decisions that now exist (`TimeoutRules`
made clock management real) and left part of the critic's residual variance
*structurally* unlearnable rather than undertrained. Air/ground had existed in
the engine since flying cards were added and was visible nowhere.

Same day: a `CardRegistry` audit found **Musketeer, Wizard, Spear Goblins and
Ice Wizard could not shoot air** (plus their Hero variants, plus Evolution Dart
Goblin regressing from its own base card). Musketeer is in `DEFAULT_DECK`. The
new `CH_ANTIAIR` channel is what surfaced it. Also that day: the river was
re-centred, which had been giving team 0 an extra placement row.

**2026-08-09, the placement head had a fixed favourite cell and it was not a
reward problem.** The "Cannon pathology" — 27.9% of Cannons parked behind the
agent's own King — was attributed to the building-vs-tower reward price and
partly fixed there on 2026-08-06. That diagnosis was wrong, or at least badly
incomplete. Re-measured on the ep~129k checkpoint over 803 sampled placements:
the concentration is **card-independent**. Giant 44.4%, Valkyrie 38.9%,
Musketeer 35.1%, Cannon 27.1% — the Cannon was not even the worst card, and no
asymmetry in how *deployed buildings* are priced can make a **Giant** walk
behind its own King half the time.

The cause was `place_up`. `placement_given_card` adds the card/state context as
a **spatially uniform** vector, and a `ConvTranspose2d(k=2, s=2)` applies a
different weight to each of the 4 positions in its output block — so a uniform
input does *not* give a uniform output. Two stacked layers imprint a fixed
period-4 `(x mod 4, y mod 4)` bias on the logit map that is identical for every
card and every state; the card can only shift the whole map by a constant, never
change which cell inside the period wins. Measured three ways, all agreeing:

- **Weight space:** on a spatially constant input the old head's logit surface
  is **100% explained by phase alone**, at every input magnitude tested.
- **Behaviour:** **73.0%** of that agent's placements landed on `x ≡ 3 (mod 4)`
  against a 22.2% null — χ² = 94.8 on 3 df — and it used only **91 of 288**
  legal cells.
- **A/B against the current agent** under identical conditions: 28.9% of all
  placements on (11,2)/(11,3) versus **0.9%**, 91 cells versus **208**.

Fixed by nearest-neighbour upsample + stride-1 conv (Odena, Dumoulin & Olah,
*Deconvolution and Checkerboard Artifacts*). Verified: a spatially constant
input now produces an **exactly** flat interior (range 0.000000), and the head
is **1.8× faster** (136 ms vs 246 ms per fwd+bwd at batch 256) with slightly
fewer parameters — `ConvTranspose2d` is poorly optimized on CPU and the 32→16→8
channel taper cut real work. A FLOP count predicted a 2.6× *slowdown* and was
simply wrong about wall clock. **This invalidates every checkpoint**: `place_up`
no longer shape-matches, so `load_state_dict_flexible` will warm-start
everything else and leave the placement head fresh.

Two lessons worth carrying:

- **The concentration was visible in the replays the whole time and nobody
  looked at the marginal.** `Entropy/Placement_Measured` was tracking its target
  throughout — a policy can put 30% on one cell and still hit a moderate entropy
  target by spreading the rest, so the entropy metric **cannot** detect this.
  The cheap detector is a histogram of `x mod 4` over `replays/*.json`.
- **The thing that didn't break localized it again.** Same shape as the
  team-1 observation bug: it was the cards that had *no* reason to be affected —
  Giant, Valkyrie — that ruled out the reward explanation and pointed at the
  head.

**2026-08-11, the entropy controller was regulating the noise of non-actions.**
`train_selfplay.py` averaged placement entropy over `mb_decision`, which is
`card_mask.sum() > 1` — **"a card was AFFORDABLE", not "a card was PLAYED"**.
The placement head is sampled on every such step, including the ones where the
policy chose the no-op and the sampled cell never reaches the board. Measured
over 549 decision steps at ep~62,200:

| | of max |
|---|---|
| reported `Entropy/Placement_Measured` | 0.462 |
| ...on no-op steps (cell never used) | **0.850** |
| ...on real placements | **0.090** |

Target was 0.25. So the controller saw a surplus, concluded there was too much
exploration, and pinned the coefficient to its 0.01 floor — **57% of updates in
the hour before the fix** — while the policy that actually places cards sat ~3×
*below* target. Conditioned on a card, Cannon was at 0.017 of max, Fireball
0.048 and Giant 0.074, all three pinned to the same cell `(11,0)`. Fixed by
masking the entropy term to real placements; verified the bonus's gradient is
now exactly zero on no-op steps. The coefficient left the floor within one
update and quadrupled in 30 minutes. **Gameplay-affecting** — it changes the
loss.

Three things worth carrying:

- **A pre-fix control checkpoint measured Cannon at 0.012 / 94% on `(11,0)`**,
  identical. That ruled out the placement-mask fix from two days earlier as the
  cause and localized it to the metric. Running the *older* checkpoint under
  the *current* code is what made that a one-command test.
- **The cost was real and the obvious fix was inert.** Over 40 episodes the
  Cannon took 369 elixir and returned 20 (ROI 0.05) against Musketeer's 1.27;
  100% of 123 Cannons were placed at y=0, 82.9% expired by decay rather than
  dying, and **93.5% never had an enemy inside their 5.5-tile range for a
  single tick**. The proposal on the table was to penalise back-row structures
  and structures that "fire ≤3 times" — but P(shots ≤ 3) was **100.0%**, so
  both predicates have zero variance, carry no spatial gradient, and reduce to
  a constant added to the card's cost. The only thing they could move is
  P(play Cannon), i.e. the guaranteed-zero trap this file already documents
  twice. **A penalty cannot move a distribution with no mass to move.**
- **Same blind spot as the checkerboard bias and the team-1 observation bug**,
  now three for three: an aggregate that cannot see a conditional collapse.
  `Entropy/Placement_ByCard_Min` is the detector, and it is cheap — it is
  computed from tensors the update already has.

**2026-07-31, the self-play opponent had been playing blind.** Chased from an
odd number rather than a hypothesis, which is why it is worth recording as a
method. Exploiter burst #1 came back 0.585 with a flat five-slice trend — but
its *first* slice was already 0.59, and a re-seeded exploiter's first slice is
played by a bit-exact copy of its opponent, so it should sit at the null.
Either it learned a hole in ~490 gradient steps and then stopped, or the null
was never 0.50. Running the frozen agent against a copy of itself gave
**0.598** as team 0 (n=400, z = +3.90) — a policy beating *itself* 60/40 purely
by side assignment. Root cause was a one-row displacement in team 1's
observation (see Board geometry); after the fix the same test gives 0.520.

Three lessons, all of which nearly hid it:

- **The bug was invisible in every training metric.** The trainee is always
  team 0, so win rate, reward, entropy, aux MAE and explained variance all
  looked healthy while every opponent saw the board wrong.
- **The dissociation was the evidence, not the aggregate.** In the earlier
  entropy-collapse bug the card head was fine and only the mis-scaled placement
  head blew up; here the integer-`y` Princesses mirrored correctly and only the
  fractional-`y` King did not. In both cases the thing that *didn't* break
  localized the cause faster than the thing that did.
- **I over-read the Elo trend twice before this landed** — first predicting
  saturation noise, then calling a single 2.1σ excursion a real gain that
  "held". Six evals of `heuristic@1.50` test as flat (χ² = 7.4 on 5 df,
  p ≈ 0.19). At n=50 per anchor that metric cannot resolve much; treat any
  single-eval move as noise until it repeats.

**2026-08-14, three of eight cards were dead and it was a gradient COVERAGE hole,
not the reward.** (`PLACEMENT_COLLAPSE.md` carried the long-form write-up and was
retired on 2026-08-19 once the bug was cured and confirmed cured; every number it
reported that still matters is in this section and the ones below it.)

Cannon, Fireball and Giant were played on ~2% of plays and their placement head
returned the *same* cell — (11,0), our own back row — in 54–91% of states. The
reward was the obvious suspect and it is not the cause. The cause is that both
the actor loss and the placement entropy bonus flow through
`placement_given_card` for the **chosen** card only, so a card the policy has
stopped playing receives **exactly zero** placement gradient from either term,
forever. `card_id_embed` is per-card, so this is mechanical, not statistical —
`test_python_ai.py::test_unchosen_card_gets_no_gradient` asserts it is `== 0.0`.

That is a self-sustaining deadlock: frozen map → the card really is worthless →
card head suppresses it → no gradient → still frozen. **More training cannot
escape it**, which is why it survived every previous fix.

It is not a valuation, and one measurement settles that. Injecting a Cannon at
each candidate cell and running the engine a full 300-tick lifetime, over 449
threatened states, scored by tower HP preserved: the policy's own cell **121 HP**,
a **random legal cell 396 HP**, oracle 1389. Paired, random − policy = **+274 HP,
95% CI [+221, +328]**. A policy cannot be correctly valuing a card it places
*significantly worse than chance*. Same for Fireball against
`get_elixir_value_killed_by` over 1,059 states: policy **0.022** elixir and it
caught anything at all **0.1–0.5%** of the time, against **1.103 for a random
legal cell** — 50× worse than random.

Dating it: the 08-09 phase-1 net placed Cannon at (3,15), modal share 9.5%,
H=0.368 — healthy and state-dependent. The collapse appears in the 08-11 net,
bracketing `e16cdd7` ("Measure placement entropy on real placements, not on
no-ops"). **That commit was right about its own defect and had an unmeasured
side effect**: the no-op steps it stopped rewarding were the only thing holding
open the maps of cards that are never played. The signature is specific — in the
pre-fix net Cannon and Giant had the two *highest* per-card placement entropies
(0.368, 0.409); after it they have the lowest. The rank order inverted for
exactly the unplayed cards. The same-day placement-legality mask (`24a2c78`) was
checked and ruled out: (3,15) is still legal, so the head abandoned an available
cell rather than being masked off one.

Fixed by a **coverage term** — one uniformly-sampled *affordable* slot per step
contributes placement entropy at a fixed `PLACEMENT_COVERAGE_COEF = 0.02`. It is
a regularizer, not part of the PPO objective (it never touches `new_logprobs`,
so the ratio is untouched; pinned by test). The coefficient is deliberately
**not** tied to the adaptive `ent_coef_place`, which falls exactly when real
placements are sharp — the condition under which an unplayed card is freezing.
`Entropy/Placement_Coverage` is the new freeze detector. No new parameters, so
**no checkpoint is invalidated**.

Two further results worth carrying:

- **Leading a spell target is HARMFUL here.** Fireball has `spellDelayTicks=10`,
  so aiming where the target will be looks obviously right. Measured paired over
  1,059 states against the engine's own value-killed: lead=0 captures **75.5%**
  of achievable value, lead=10 **52.4%**. The blast radius is 2.5 tiles while 1 s
  of movement is 0.6–1.6 tiles, so a moving target stays inside the blast anyway
  — while a target standing still (engaged, the common case) is led straight off
  the edge of it. `tactics.py` defaults `lead_ticks=0`; do not "fix" it.
- **"Defensive apathy" is bankruptcy, not apathy or hoarding.** P(play) looks
  flat against threat (9.4% → 13.7%), but **60.8% of decisions during a big push
  are below 3 elixir** — the cheapest card in the deck — so P(play) there is
  0.0% by arithmetic. Conditioned on affordability the response is real:
  27.8% → 34.9%. The agent spends ~105 elixir per episode against ~98 of income
  and sits under 3 elixir **65.3%** of the time.

**RE-RUN, and the coverage term does NOT break the lock — it moves it.** Three
arms in parallel, byte-identical code, full-checkpoint seeds so the entropy
target anneals correctly, ~80 updates, compared at matched episode ~64,800 and
scored by the engine on 835/1,573 paired states. Cannon: control 116.4 HP vs
treatment 182.0 HP (bootstrap CI excludes 0 but **sign test p = 0.158**, so no);
Fireball: treatment kills **0.000** elixir, identical to control. Treatment's
top-1 probability is **0.051** (Cannon) and **0.006** (Fireball) against a
uniform 1/612 = 0.0016 — the distribution went nearly FLAT, and **the argmax of
a flat map is an arbitrary constant**, so a greedy policy still plays one fixed
cell. The frozen cell relocated (11,0) → (6,0).

**Entropy is a MARGINAL objective — it says "be spread out", not "depend on the
board".** A card with no other gradient has nothing telling it which cell is
right in which state. Closing a coverage hole needs a TARGET, not noise. (The
control also partially unfroze on its own — 88.7% → 55.4% modal share — so part
of the original collapse was the mis-set entropy target below, not the coverage
hole alone.)

**What does work is the deterministic advisor** (`tactics.py`), same protocol,
same states: Cannon **564.1 HP** preserved vs 353.5 for a random legal cell and
**12.1 for the trained policy**; Fireball **2.405 elixir** killed vs 0.276
random and **0.000** for the policy (396 better / **0 worse** of 950 states,
p = 1.2e-119). Distilling it into the head (`distill_tactics.py`, frozen trunk)
moved Cannon significantly (+161.9 HP, p = 2.0e-06) and Fireball not at all,
both still below random — the head's spatial signal is a 9×5 pooled map
upsampled 4× on a frozen trunk, so an exact-cell target is close to
inexpressible (CE fell 180.9 → 21.4 with argmax match stuck at 0.0%).

**Bankruptcy: fixed as a statistic, no outcome gain.** The potential-based
solvency term (`elixir_shaping.py`) measured **-1.0 points, p = 0.21** over 80
updates — policy-invariance is what makes it safe and also what limits it. An
inference-time reserve gate (`tactics.SolvencyGate`, refuse spends below 4
elixir while nothing attacks) fixes it outright over 130 paired openings:
bankruptcy **72.7% → 39.2%, 130/130 episodes, p = 1.5e-39**, with **total elixir
spent statistically unchanged** (102 → 98, p = 0.25) — so it moved *when* the
bot spends, not how much. But tower HP lost (p = 0.25) and win rate
(−0.042, p = 0.63) are both flat. **Having elixir does not help while the
placements are worth less than random**; placement quality is the binding
constraint.

**2026-08-14, the hybrid: route around the broken head rather than repair it.
+11.8 win-rate points, p = 1.9e-05.** `python_ai/advisors/hybrid_policy.py` — the network
keeps WHAT to play and WHEN; `tactics.py` decides WHERE for Cannon, Fireball and
Giant, and `SolvencyGate` vetoes spends that would bankrupt it. Two independent
pre-registered paired runs, opponent 1.5x:

| | n | neural | hybrid | delta | p |
|---|---|---|---|---|---|
| exploratory | 250 | 0.646 | 0.720 | +0.074 [−0.004, +0.150] | 0.070 |
| **confirmatory** | **600** | **0.584** | **0.703** | **+0.118 [+0.067, +0.171]** | **1.9e-05** |

Damage DEALT rose (+544/ep, p=0.010) while damage taken fell (−1011/ep,
p=7.4e-09) on unchanged spending — it attacks better *and* defends better. The
Giant rule was validated before shipping: bridge on the weaker-defended lane
scores **535.6** enemy tower damage against **3.3** for the policy's own cell
(n=913, p=3.0e-87). Wired into the live loop behind `--no-tactical`.

**The component that had to be REMOVED is the instructive part.** Letting the
tactical officer *initiate* Cannon/Fireball (rather than only place them) looked
necessary, since the commander's take-up of those cards is 0.06/0.00/0.00. A
5-arm ablation killed it: initiating ~4 Cannons a match cut tower damage DEALT
by 3,410/ep and win rate by 0.367 (p=0.013). **Defending better is worthless if
it is paid for with the attack.** The placement override works precisely because
it is PASSIVE — it changes where a card lands, never how often one is played, so
it cannot spend elixir the commander did not already commit. Same reason a FLAT
solvency reserve had to become `min(reserve, opponent_elixir)`: a flat reserve
blocks exactly the spends that build a push.

**The FIRST attempt at the coverage A/B was INCONCLUSIVE and the reason is a trap
worth knowing: warm-starting a converged policy into a FRESH training state
re-arms the initial entropy target and the controller dissolves the policy.** Seeding a
bare `state_dict` takes `train.py`'s legacy-checkpoint path, which resets
`episodes_completed` to 0, so `placement_entropy_target(0)` returns
`ENTROPY_TARGET_PLACEMENT_START = 0.65` against a policy measuring 0.11. Both
arms spent the run being inflated toward that target and ended at ~0.83 of max
with top-1 probability 0.002–0.008 — near-uniform placement, in the control as
well as the treatment. Same shape as the exploiter's 2026-07-31 entropy-scaling
failure. **Always resume through the full-checkpoint path when the entropy
schedule matters.**

Corollary: **modal share degenerates on a near-uniform distribution** — the
argmax of a flat map is arbitrary but deterministic, and both arms reported a
90–100% modal share while being flat. Read modal share next to top-1
probability; the collapse in the table above has top-1 at 0.62–0.83, which is
what makes it real.

**2026-08-14, the placement head got its resolution back: a zero-initialized
high-resolution residual branch (`place_hires`), and it works.** `cnn_trunk`
pools twice, so the head reads a **9×5** map of a 34×18 board and `place_up`
blows it back up; the card context enters as a spatially uniform vector.
`place_hires` adds a parallel path from the trunk's own **pre-pool 16×34×18**
activation at one-tile resolution, conditioned on the same `(hx, card)` context,
added to the coarse logits as a residual. 3,993 new parameters (+0.2%).

**No checkpoint is invalidated, and that was a design goal rather than luck.**
The handoff proposed concatenating into `place_up`, which changes its shape and
therefore discards the trained placement head from every checkpoint — the price
the 2026-08-09 checkerboard fix had to pay. Zero-initializing the branch's final
conv makes it an **exact** no-op at init (weight *and* bias zeroed, so the
residual add is exact, not approximate), so an old checkpoint loads and behaves
bit-identically and the cards that already work keep working. The loader
confirms it: `warm-started 27/27 tensor(s), re-initialized: []`. Gradient still
flows — a zero conv has a nonzero gradient of its own, so it leaves zero on step
one and the layer beneath it starts learning on step two.

Measured as a controlled A/B (`prove_hires.py`): one collection of 2,376 states
from the seed policy's own trajectory, contiguous-tail held-out split, both arms
from the same checkpoint and seed, identical epochs/lr/anchor, the **only**
difference being whether `place_hires` trains. Held out (n=582/653):

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
merely fail — it *dissolves to uniform*, and then reports a 98.0% modal share,
which is the degenerate reading this file already warns about. Its training loss
plateaus at ~21 (matching `distill_tactics.py`'s 21.4) while the branch's keeps
falling to 15.1. Replicated on two independent collections.

**ENGINE-SCORED, which is the only verdict that counts** (`prove_placement.py`,
paired on states drawn by the seed policy, 12 episodes):

| | Fireball, elixir killed (n=1937) | Cannon, tower HP preserved (n=894) |
|---|---|---|
| seed (the shipping net) | 0.062 | 102.2 |
| control (coarse-distilled) | 0.000 | 161.3 |
| **+hires** | **1.863** | **232.6** |
| random legal cell | 0.346 | 357.0 |
| advisor (the ceiling) | 2.647 | 539.1 |

**Fireball is fixed.** It beats a random legal cell by **5.4×**
(+1.516, 95% CI [+1.368, +1.669], **561 better / 71 worse**, p = 1.8e-95) and
reaches **70% of the advisor's** value. This is the first time the placement
head has ever beaten random for that card — every prior measurement had it at
0.000–0.022 against random's 0.276–1.103, i.e. **worse than chance**. Against
the control it is 600 better / **0 worse**.

**The Cannon is improved but NOT fixed, and the honest statement is that it is
still worse than random.** +71.2 HP over the control (p = 0.029) and +130.4 over
the seed (p = 0.0034), but −124.4 against a random legal cell (p = 2.5e-09) and
−306.6 against the advisor. Consistent with everything else measured about it:
with a plateau target the head learns *roughly where* (mean distance 9.69 → 6.89
tiles, modal share 88.8% → 31.8%) and still commits to a cell worth less than
chance. **The tactical override in `hybrid_policy.py` therefore stays on for the
Cannon** — it is not yet redundant, and turning it off would give back the
+11.8 win-rate points.

Three things worth carrying:

- **A claim in the handoff was too strong, and testing it directly is what
  found the real story.** The premise was that the coarse head *cannot express*
  an exact cell (from "CE fell 180.9 → 21.4 with argmax match stuck at 0.0%").
  Tested on the task reduced to its essential — 14 boards differing only in
  which column holds one enemy — **the coarse head fits 14/14 exactly**.
  Nearest-upsample followed by 3×3 convs lets a fine cell mix neighbouring
  pooled cells, so sub-block position *is* recoverable. The limit is real but
  it is capacity at scale, not impossibility. `test_python_ai.py` keeps
  both results.
- **The Cannon's exact cell is a BAD SUPERVISION TARGET and that is a property
  of the teacher, not the student.** `building_score_map` scatters flat discs,
  so the top of the surface is a large exact-tie plateau and `np.argmax`
  returns its top-left cell by row-major accident. Quantified: lowering the
  softmax temperature cannot push the Cannon target below **~66% of maximum
  entropy** (Fireball reaches 41%), because the ties never break. So Cannon
  exact-match near 0 is expected and mostly uninformative — judge a building by
  distance and by engine score, never by exact cell.
- **The soft/neighbourhood target the handoff recommended first (A1) is
  measured NEUTRAL-to-WORSE here — do not re-try it without new reason.** As a
  third arm on the same collection (KL to `softmax(standardized advisor score /
  T)`, T=0.25 chosen off the printed entropy table): Fireball within-2 71.2% vs
  the argmax arm's 71.4% and mean distance 4.49 vs 3.31; Cannon within-2 8.4%
  vs 10.0%. It spreads mass over the neighbourhood exactly as designed and buys
  nothing. The plateau argument predicted it would rescue the Cannon; it did
  not. **Resolution was the binding constraint, not target softness.**

**2026-08-14 (later), two GAMEPLAY-AFFECTING changes: the spell anneal is wired
in, and the coverage term was given a target.**

**`spell_value_weight` was dead code and now runs.** Both trainers called
`compute_shaping(stats, prev_stats, gamma=gamma)` with no `w_spell`, so the
Fireball-value weight sat at `W_SPELL_VALUE_START = 0.08` for the whole of
training and the anneal its own comment block describes never happened. Nothing
detected it because **no test ever varied the argument** —
`test_compute_shaping_actually_responds_to_w_spell` is the regression that now
would. The term is deliberately NOT potential-based, so it biases the optimum by
construction; a schedule that never reaches zero is a permanent bias nobody
chose. Every win rate before this was earned under a constant 0.08.

`SPELL_VALUE_ANNEAL_START` (env `CLASH_SPELL_ANNEAL_START`, default 0) exists
because a warm start resumes PAST the horizon: `model_weights_selfplay.pth` is
at episode 64,309 against a 40,000-episode anneal, so a faithful wiring pins
`w_spell` at FINAL from the first step and the anneal cannot be observed at all.
At the default the behaviour is exactly the original intent.

**The coverage term now carries a TARGET where the advisor has one**
(`advisor_target.py`). `PLACEMENT_COVERAGE_COEF` adds an ENTROPY bonus on one
sampled affordable slot, and for a card the policy never plays that bonus is the
*only* placement gradient in the objective — so it pushes the map toward
uniform, which is precisely what the hi-res distillation is trying to undo. The
two are in direct opposition and coverage runs for all of training.

The resolution is a **row mask**: a row either has an advisor target and gets
KL to the advisor's masked score map, or it does not and keeps the entropy
bonus. Never both — entropy says "be spread out", KL says "be here", and a row
carrying both asks the head for two incompatible things. This is the fourth time
this project has landed on the same conclusion: *closing a coverage hole needs a
target, not noise.*

Three things worth carrying:

- **The gate is the load-bearing part.** `tactics` always returns a cell, and on
  a quiet board that cell is a default — the defensive pocket for a building, an
  arbitrarily tie-broken lane for the Giant. Training on defaults teaches a
  CONSTANT, which is the exact pathology being repaired. `target_logits_for`
  returns `None` there and the row falls back to entropy. This mirrors
  `distill_tactics.collect`'s `cover > 0 or catch > 0` filter.
- **The limiter was AFFORDABILITY, not the gate, and it was worth measuring
  rather than guessing.** Over 258 decision steps the sampled coverage slot held
  an advisor card 32.6% of the time and the advisor then spoke on 90% of those.
  Cannon(3)/Fireball(4)/Giant(5) are exactly the cards a near-bankrupt agent
  cannot afford — it sits under 3 elixir on 65.3% of decisions — so a uniform
  draw over *affordable* slots is biased toward the cheap cards, which are also
  the ones already getting actor gradient because they are the ones being
  played. `placement_coverage_slots` grew a `slot_weights` argument
  (`CLASH_ADVISOR_SLOT_WEIGHT`, default 5.0); measured `Advisor/Rows` 13-19 →
  23-25 per minibatch. Weights re-rank, never remove: a row whose affordable
  slots all weigh zero falls back to the unweighted mask rather than through to
  the no-op.
- **Read `Advisor/KL` and `Advisor/Rows` as a PAIR, never KL alone.** KL falls
  both when the head learns the surface and when the advisor simply stops
  speaking, and those are opposite situations. Same shape as every other
  aggregate this file warns about.

The coverage slot is now sampled **once at rollout time and buffered**, not
resampled inside every PPO epoch: an advisor target has to be computed against
the observation the slot was drawn on, and a fresh draw in the update would pair
one card's logits with another card's target.

**MEASURED, and it works. Pre-registered A/B, engine-scored, paired on states
drawn by one reference policy.** Two arms from `model_weights_hires.pth` on a
full training state at episode 64,309, byte-identical code, 80 updates each
(fixed in advance), the only difference being `CLASH_ADVISOR_COVERAGE_COEF`
(0 vs 0.10). Scored by `prove_placement.py --episodes 16`:

| arm | Fireball, elixir killed (n=3106) | Cannon, tower HP preserved (n=1793) |
|---|---|---|
| random legal cell | 0.515 | 456.9 |
| seed (hires-distilled) | 2.066 | 269.2 |
| control (coverage entropy only) | 2.387 | 402.9 |
| **treatment (advisor target)** | **2.969** | **480.3** |
| advisor (the ceiling) | 3.013 | 702.7 |

* **Fireball, treatment vs control: +0.583, 95% CI [+0.505, +0.661],
  p = 1.2e-47.** And treatment vs the ADVISOR is +0.043, CI [−0.018, +0.102],
  **p = 0.163 — statistically indistinguishable from the ceiling.** This is the
  first time the head has matched the advisor on any card.
* **Cannon, treatment vs control: +77.4, CI [+37.0, +118.6], p = 0.0012.**
  Against a random legal cell it is +23.4, CI [−35.3, +82.1], p = 0.118 — i.e.
  **no longer significantly worse than chance**, which every prior measurement
  was (−124.4, p = 2.5e-09). It is still far below the advisor (−222,
  p = 6.8e-22), so **the Cannon override stays on.**

**The frozen cell is gone, and that is the qualitative result:**

| net | Cannon modal | share | Fireball modal | share | cells used |
|---|---|---|---|---|---|
| seed | (11,0) | 46.2% | (11,0) | 32.3% | 90 / 210 |
| control | (11,0) | 38.4% | (11,0) | 25.8% | 116 / 241 |
| **treatment** | **(3,15)** | **19.1%** | **(13,14)** | **6.6%** | **138 / 278** |

`(11,0)` is the pathological own-back-row cell this file has tracked since
2026-08-11. The control is still parked on it for both cards; the treatment is
not, at top-1 probability 0.076 / 0.056 — far above uniform 1/612 = 0.0016, so
this is a head that is SHARP and MOVES ITS MODE, not a dissolved one. That is
exactly the signature the modal-share detector was defined to look for.

**A prediction of the handoff that did NOT survive contact, stated because it
was the stated reason for doing this work.** §3 Step 1 predicted the entropy
coverage term would ERODE the distilled placement head under PPO. It does not:
the control IMPROVED over its seed on both cards (Fireball 2.066 → 2.387,
Cannon 269.2 → 402.9) across 80 updates. So the conflict is real in mechanism
but the erosion is not observable at this horizon. The advisor target is worth
having because it is much BETTER, not because the alternative decays.

**AND IT HOLDS UNDER A LONG RUN. 2026-08-15, ep 78,270 (13,961 episodes / 524
PPO updates of pipeline 2 with the term on, zero alarms throughout).** Same
harness, same reference policy, same paired protocol:

| | seed (hires) | control (80 upd, coef 0) | **main (524 upd, coef 0.10)** | advisor | random |
|---|---|---|---|---|---|
| Cannon, tower HP preserved (n=1582) | 276.0 | 403.0 | **553.1** | 687.9 | 411.3 |
| Fireball, elixir killed (n=2979) | 1.506 | 2.190 | **2.565** | 2.464 | 0.469 |

* **The Cannon beats a random legal cell for the first time in this project's
  history: +141.8, 95% CI [+74.5, +210.2], p = 0.0071.** Every prior
  measurement had it BELOW chance (−124.4, p = 2.5e-09 at v1.2.0; +23.4,
  p = 0.118 and not significant at 80 updates). The advisor still beats it
  (−134.9, p = 5.0e-10), so **the Cannon override stays on** — but the gap has
  closed from −306.6 (v1.2.0) to −222 (80 updates) to −134.9.
* **Fireball has overtaken the advisor**, +0.100 elixir in the net's favour.
  The two tests disagree on significance — the bootstrap CI [−0.175, −0.025]
  excludes zero, the exact sign test does not (223 better / 252 worse,
  p = 0.199) — because the advisor wins more pairs while the net wins bigger
  ones. **The defensible claim is "no longer distinguishable from the advisor,
  and certainly not worse", which is enough to retire that override.**

**The dynamism table is the qualitative proof, and Fireball's is the cleanest
result this metric has ever produced:**

| net | Cannon modal | share | top-1 | Fireball modal | share | top-1 | cells |
|---|---|---|---|---|---|---|---|
| seed | (11,0) | 44.9% | 0.140 | (11,0) | 32.8% | 0.087 | 88 / 188 |
| control | (11,0) | 27.1% | 0.050 | (11,0) | 16.9% | 0.030 | 126 / 228 |
| **main** | **(16,15)** | **17.0%** | 0.062 | **(4,17)** | **12.7%** | **0.112** | **135 / 256** |

Fireball's modal share fell 32.8% → 12.7% while its top-1 probability ROSE
0.087 → 0.112. That is the exact signature this file defines as healthy and
which no previous net has shown: **more confident within a state, less
repetitive across states.** Its modal cell moved from the own-back-row (11,0)
to (4,17) — just across the river at the left bridge. The Cannon's moved to
(16,15), the right bridge mouth on our own side.

**Watch this in any future run:** modal share and top-1 falling TOGETHER is
dissolution toward uniform, not a cure. They diverged here, which is what makes
it real.

**THE GIANT IS CURED TOO, and it was the worst of the three** (`prove_giant.py`,
new — `prove_placement.py` scores only Cannon and Fireball, so this card had
never been engine-scored at all). Enemy tower damage over 600 ticks, paired on
states drawn by the cured net, n=2784:

| arm | tower damage | modal cell | share | cells used |
|---|---|---|---|---|
| v1.2.0 | **13.7** | (11,0) | 49.7% | 38 |
| **cured** | **231.9** | (1,15) | 24.7% | **140** |
| advisor | 380.5 | (14,15) | 57.7% | 2 |
| random legal cell | 86.7 | — | — | 242 |

**+145.2 vs a random legal cell (p = 6.5e-19)** and **+218.2 vs v1.2.0
(p = 1.0e-98)**. The shipping net's Giant was 6x WORSE than chance at 13.7 —
matching the 3.3 recorded when `best_giant_cell` was written — and used 38 of
612 cells with half its mass on (11,0). The advisor still wins (−148.6,
p = 1.1e-69), and note WHY: its rule is essentially two cells (57.7% on one of
the two bridges), which is a very strong prior this board rewards.

**So all three starved cards moved from worse-than-chance to better-than-chance.**
Fireball reached the advisor; Cannon and Giant beat random but remain below it.

**THE TACTICAL OVERRIDE IS NOW REDUNDANT — a measured NULL, which is the point.**
`hybrid_ab.py --per-card`, 200 paired openings on the cured net, solvency gate
held ON in every arm so only placement varies:

| arm | win rate | vs neural | p |
|---|---|---|---|
| neural (no placement override) | 0.507 | — | — |
| cannon_only | 0.510 | +0.003 | 1.0 |
| cannon_giant | 0.480 | −0.028 | 0.54 |
| all_three | 0.480 | −0.028 | 0.56 |

At v1.2.0 the same override was worth **+11.8 points, p = 1.9e-05**. This n had
the power to see an effect that size and it is gone. The officer is not helping
because there is no longer a hole for it to fill. **`hybrid_policy.py` can drop
to gate-only.**

**AND THE UNCOMFORTABLE HALF, which is the more useful result.** Two paired
win-rate comparisons of the cured net against v1.2.0's
`model_weights_selfplay.pth`, both side-controlled:

| opponent | v1.2.0 | cured | delta |
|---|---|---|---|
| C++ `HeuristicOpponent` @1.5x (`net_ab.py`, 200 paired openings) | 0.6225 | **0.5100** | −0.1125, CI [−0.2025, −0.0200] |
| **v1.2.0 itself**, head-to-head, sides swapped (`net_h2h.py`, 120 pairings) | 0.3875 | **0.6125** | **+0.1125, CI [+0.054, +0.171]** |

**This is SPECIALIZATION, not degradation, and only a neural opponent could tell
the two apart.** 13,961 episodes in a PFSP league whose members are all neural
made the net significantly better against that opponent class — it beats its own
predecessor — while losing ground against the C++ heuristic, which pipeline 2
never shows it. Read either number alone and you get the wrong answer.

Two controls that make the attribution stick:

- **The seed is not the cause.** `model_weights_hires.pth` was never win-rate
  tested (the v1.2.0 handoff says so). Measured: 0.6125 vs v1.2.0's 0.6350,
  delta −0.0225, CI [−0.100, +0.055], p = 0.76 — indistinguishable. The whole
  −0.1125 came from the training run, not from the distilled starting point.
- **Sides were swapped** in the head-to-head. A policy beat a bit-exact copy of
  itself 0.598 once purely by side assignment; it reads 0.530 today, small but
  not zero, and a one-sided duel would fold that straight into the result.

**So `model_weights_selfplay.pth` was NOT replaced.** Two checkpoints now exist
with different strengths, and which is "better" depends on the opponent you
care about. The real game is not the C++ heuristic, which argues for the cured
net; but nothing here measures the real game.

**A resume trap that is specific to `place_hires` and bit pipeline 2 only.**
The branch added 6 parameters, so a pre-2026-08-14 checkpoint's optimizer
describes 26 and the net has 32. `train.py` degrades gracefully (it gates
optimizer restore on a clean model load); **`train_selfplay.py` loads it
unconditionally and dies at startup** with "loaded state dict contains a
parameter group that doesn't match the size of optimizer's group" — and
`load_state_dict_flexible` reports CLEAN, because all 33 tensors are supplied.
`setup_ab_arm.py` remaps the moments **by name**: the new parameters land at
indices 22-27, in the MIDDLE of `named_parameters()` order rather than appended,
so the obvious "keep 0..25 and append the rest" hands the trunk's moments to the
placement head — a run that trains, looks healthy, and is quietly wrong.

**2026-08-15, the heuristic regression is fixed by WIRING IN SEARCH, and the
weights never changed.** The cured net's loss against the C++ heuristic
(0.51 vs v1.2.0's 0.6225) was going to be answered with a from-scratch
search-distilled run plus a mixed curriculum. Both premises were measured
first, and both are false:

- **The mixed curriculum already exists and did not work.**
  `BUILTIN_TRAINING_OPPONENTS` (2026-07-31) puts the heuristic in the PFSP pool
  at `BUILTIN_MIN_WEIGHT = 0.5`. Reproducing the env's own pool construction and
  sampler at the cured net's state (48-member pool, measured win rates) gives a
  realized share of **20.8%** — ~2,897 of that run's 13,905 episodes were
  ALREADY played against the heuristic, and it regressed anyway. Routing is
  correct (`game.step()`, not `step_self_play`), so this is not the silent-
  no-opponent trap. Adding heuristic exposure is a no-op.
- **Search cannot bootstrap a from-scratch run.** Search scores candidates with
  the net's OWN critic, so at random init the expert is not an expert. Paired,
  n=60: random init @1.0x scores **0.450 policy / 0.417 search** (ns) while
  overriding **21.5%** of decisions; the trained net @1.5x scores 0.483 / 0.667
  at **10.2%**. At init it deviates twice as often and gains nothing, so early
  distillation targets are noise.

**`horizon` is the lever, and depth is nearly free.** One engine step costs
0.015 ms; one scored candidate costs a network row at 0.13 ms — so lookahead is
~9x cheaper than width. Sweep (cured net vs heuristic@1.5x, n=80 paired):
horizon 4 → 0.667 (1.6x cost), 4 with K≤13 → 0.788 (1.7x), 8 → 0.925 (1.5x),
**12 → 0.963 (1.5x)**, 20 → 0.875 (1.8x). It degrades past ~12 because a
candidate rollout assumes both sides no-op and 20 s of that stops resembling
the game. Chosen on the sweep, then CONFIRMED on a fresh independent run.

**Confirmatory, `model_weights_cured.pth` + search horizon 12:**

| criterion | bar | result |
|---|---|---|
| vs C++ heuristic@1.5x (n=400 paired) | >0.65 | **0.9225** (policy 0.5200, delta +0.4025 CI [+0.3486, +0.4564], 174 better / 13 worse) |
| h2h vs v1.2.0 as shipped (n=150 swapped) | >0.60 | **0.7200** CI [0.6750, 0.7650] |
| Cannon placement, shipping config | dynamic | modal 12.9% over **36** cells (greedy: 17.1% over 20) |

**THE UNCOMFORTABLE CONTROL, and it is the more useful result. Give BOTH sides
search and the cured net's advantage disappears:** 0.4425, CI [0.3825, 0.5025]
(n=100 swapped) — no difference resolved, point estimate favouring v1.2.0. So
the +0.72 is **search, not the cured weights**. Search and the placement cure
repair the same weakness, and they do not stack. Anything claiming the cure
made a stronger network has to answer this number.

Two smaller things worth carrying:

- **The x mod 4 detector fires on a healthy net, because its null is wrong for
  a trained policy.** The shipping config measured 43.6% on x ≡ 1 (mod 4),
  χ² = 96.3 — which reads exactly like the 2026-08-09 checkerboard bug. It is
  not. The discriminating test is the one the fix was verified with: on a
  SPATIALLY CONSTANT input, phase explains **6.8–14.0%** of the head's surface,
  against the broken head's **100%**. A trained policy concentrates on columns
  for tactical reasons and the uniform-column null cannot tell that from a head
  artifact. Run the constant-input test before believing the histogram.
- **v1.2.0 is a fair baseline under the new architecture.** It loads with
  `place_hires` freshly initialized, and `place_hires.2` weight AND bias are
  both exactly zero, so the residual add contributes exactly nothing and the
  checkpoint plays bit-identically to how it shipped. Verified, not assumed.

The shipping configuration is named in one place, `python_ai/shipping.py`, so an
evaluation and a deployment cannot drift onto different settings. Harnesses
added: `net_h2h_search.py` (either side may search) and
`prove_placement_shipping.py` (dynamism over cells ACTUALLY played).

**2026-08-16, distilling the h=12 expert into the CURED net does NOT work, and
the reason is that the cure already took the absorbable part.** The obvious next
step after the above was to bake search's +0.4025 into the weights, since the
h=4 expert had distilled for +0.045 (p = 0.0074). Run at the best known recipe —
value-DISTRIBUTION labels, frozen trunk, 16 epochs, 180 episodes, seeded from
`model_weights_cured.pth`. Collection: 48,035 rows, expert win rate 0.9528,
deviation 12.76%. T=0.05 chosen from `--target-entropy` BEFORE any outcome
(spread mean 0.2801 / median 0.1795, putting the target at 0.573 of max
entropy), 21.8% of rows carrying ≥2 candidates.

**Three independent measurements, all null:**

| | h=4 into v1.2.0 (worked) | h=12 into cured |
|---|---|---|
| conditional lift | **+0.1535** | **−0.0032 ± 0.0230** |
| p1/p0 selectivity | 2.17 | **0.92** |
| paired greedy A/B | +0.045, p = 0.0074 | **+0.0250, CI [−0.0509, +0.1009], p = 0.608** (n=300, 72 better / 65 worse) |
| search delta ON the student | 0.319 → 0.12-ish (deviation 14.8% → 12.1%) | **+0.3583 [+0.258, +0.459]**, deviation 13.17% → **13.06%** |

**Read the last row first — it is the one that settles it.** Search is worth
+0.4025 on the cured net and +0.3583 on its distilled student, with heavily
overlapping CIs, and it overrides the student just as often as the teacher
(13.06% vs 13.17%). A student that had absorbed the expert would be deviated
from LESS. Essentially nothing transferred.

**The seed is the variable, not the expert.** The distillation that worked was
seeded from `model_weights_selfplay.pth`, whose placement head was *broken* —
Cannon, Fireball and Giant all placing WORSE THAN RANDOM. That is an enormous
gap for an expert to teach into. Seeded from the cured net there is no such
gap: the condition this file names for expert iteration to pay ("the value head
is substantially better than the action head is at exploiting it") is largely
gone once the action head works. This is the same fact the compute-matched h2h
control reported from the other side — **search and the placement cure repair
the same weakness, so they neither stack nor substitute for each other's
absence.**

What is left for search to add is the part that requires ACTUALLY RUNNING THE
SIMULATOR twelve seconds forward, and a reactive head that only ever sees `s`
has no way to represent it. That was already the standing hypothesis ("the
residual is plausibly structural"); this is the first measurement that isolates
it from the placement confound.

Two caveats recorded honestly:

- **The fit is UNDERCONVERGED and that is not the explanation.** Loss was still
  falling at epoch 15 (2.8605 → 2.4645) and argmax agreement still rising
  (0.433 → 0.446), so more epochs is the one untried lever. But underfitting
  does not predict the search-on-student result: a partially-fit student would
  still be deviated from less often, and it was not.
- **`--train` and `--train-dist` are INDEPENDENT flags, not a mode selector.**
  Passing both runs hard-label BC to completion first and only then the
  distribution phase — ~2.7 h of wasted compute here. It did not contaminate
  the result (the distribution phase re-loads `student` from `--weights`), but
  the hard-label pass also overwrites `--out` on its way past.

**`model_weights_h12dist.pth` is NOT shipped.** `shipping.py` stayed on
`model_weights_cured.pth` + search horizon 12 — until 2026-08-19, when that net
was retired with the Giant deck and `SHIPPING_WEIGHTS` moved to
`model_weights_selfplay.pth`. The horizon-12 evidence was measured on the RETIRED
net; see `shipping.py`'s docstring, which says so plainly.

**2026-08-16, "the bot plays like a disconnected zombie": one of the three
reported symptoms is real, and its cause is the ENTROPY CONTROLLER's
normalizer, not the observation, the architecture or the reward.** Prompted by
a live emulator session reporting suicide placements, back-corner spam and
defensive apathy. All three were tested in the simulator against
`model_weights_cured.pth` before anything was changed. Two do not reproduce.

**REFUTED — "it drops a Musketeer on top of a Mini PEKKA".** Musketeer had
never been engine-scored here (`prove_placement.py` covers Cannon/Fireball,
`prove_giant.py` the Giant). Scored on the same protocol — snapshot, inject at
the proposed cell, run a 300-tick lifetime, read the engine — over 337 paired
states against a random legal cell:

| | policy | random legal cell | delta |
|---|---|---|---|
| elixir value killed | **5.745** | 4.481 | **+1.264**, CI [+0.650, +1.872] |
| tower damage taken | **1411.3** | 1708.3 | **297 HP less** |

144 better / 106 worse / 87 tied. The head places the Musketeer **better than
chance**, which is the opposite of the Cannon/Fireball/Giant pathology this
file spent 2026-08-11 to 08-15 curing.

**REFUTED — "it is disconnected from the board".** Counterfactual ablation:
delete EVERY enemy troop from the observation (their cells zeroed across all
enemy channels; towers, own units, elixir, hand and the recurrent state left
untouched) and re-query the same net from the same hidden state. Pooled over
1,532 states, the placement distribution moves **TV = 0.444** and the argmax
cell changes in **60.4%** of states, against **TV = 0.625** for switching to a
different card — a change the head is known to condition on. Per card it is
stronger still: Mini PEKKA 0.844 / 83.3%, Valkyrie 0.762 / 93.1%. The head
reads the board.

**CONFIRMED, and it is a REGULARIZER BUG. `LOG_N_CARD = log(hand_size + 1)` is
a maximum the masked distribution can never reach.** Both trainers divided the
card head's entropy by `log(5)` and drove the result to
`ENTROPY_TARGET_CARD = 0.35`. But `mb_decision` means "at least one card was
AFFORDABLE", and the affordability mask leaves only `(#affordable + 1)` legal
arms. Measured over 706 decision steps:

| n_legal on a decision step | share | reachable max |
|---|---|---|
| **2** (one affordable card + no-op) | **54.1%** | log 2 = 0.693 |
| 3 | 15.6% | 1.099 |
| 4 | 25.8% | 1.386 |
| 5 | 4.5% | 1.609 |

The target `0.35 × log(5) = 0.5633` nats is **81.3% of the reachable maximum**
on the majority case. So the controller demanded the play/wait choice be near a
coin flip — and then **read 0.3125 against its own 0.35 target and kept RAISING
the coefficient**, while the policy's real randomness was **0.4906 of
reachable**.

The downstream chain is the reported symptom:

| | |
|---|---|
| mean elixir | **2.25 / 10** |
| P(nothing affordable) | **73.9%** |
| ...during the largest threat bucket | **78.5%** — worst when defending matters |
| P(play), no threat → HUGE threat | 0.1008 → **0.1016** (flat) |
| P(play \| something affordable) | 0.3655 → 0.4719 (the response IS there) |

**The agent is not apathetic and not blind. It is bankrupt, and the regularizer
is what bankrupts it** — it spends on sight because it is being paid to flip a
coin. This is the same shape as the 2026-07-31 exploiter collapse (raw nats vs
`log 612`) and the 2026-08-11 no-op placement average: **a normalizer that does
not hold in the regime being measured.** Seventh instance.

Fixed by normalizing each head per-step by `log(n_legal)` — recovered from the
card mask, and from `torch.isfinite(pl_seq)` for placement, so no plumbing is
needed. The placement head had the same defect more mildly (troops see ~242 of
612 cells, so its annealed target was inflated ~17%) and is fixed identically.
`test_card_entropy_must_be_normalized_by_the_REACHABLE_maximum` pins the
invariant: **uniform over the legal arms must read exactly 1.0, at any number
of arms.** Under the old divisor a uniform 2-arm row read 0.431.

**The tell was already in the codebase.** `train_selfplay.py`'s per-card
diagnostic has always divided by `log(n_legal)` and its comment gives the
reason — "a spell sees 588 cells, a plain troop 242, the Cannon 208 ... a raw
nat count is not comparable across cards". The DIAGNOSTIC was right and the
OBJECTIVE was never given the same treatment.

**GAMEPLAY-AFFECTING**: it changes the loss in both pipelines.

**Also 2026-08-16: "perfect defense" is now expressible, and the obvious way to
do it was the wrong way.** The tower term is linear and SYMMETRIC — 100 HP
chipped off the enemy pays exactly what 100 HP taken costs — so the reward is
indifferent between "trade 500 for 500" and "take 0, deal 0". The tempting fix,
weighting damage TAKEN above damage DEALT, is the one thing that must not be
done: this file already records that policy-invariant tower shaping left PURE
DEFENCE as the optimum and win-condition usage decayed to **0.7%**.

`flawless_defense_bonus()` instead pays `W_FLAWLESS_DEFENSE = 0.5` scaled by
the fraction of our own tower HP still standing, **only on a WIN**. Gating on
the win is the whole safety argument: a turtle that stalls into a timeout
collects nothing and still pays `DRAW_PENALTY`, and a loss collects nothing, so
the term cannot reorder win/loss/draw at all — it can only rank WINS against
each other. Not potential-based, so biasing by construction, the same
eyes-open trade as `W_TOWER_DESTROYED`. Three tests pin it, including that a
DRAW collects nothing however clean, and that the post-autoreset counter is
read through a running max (otherwise a scraped win is paid as flawless).

**And a live/sim drift worth knowing: there are THREE disagreeing `max_ticks`
defaults.** `ClashEnv.h`'s C++ constructor says 3600, **`bindings.cpp` says
1800**, and `gym_wrapper` — which trains the policy — passes 3600. A bare
`ClashRoyaleEnv(deck, deck)` from Python silently gets a HALF-LENGTH match.

`perception_encoder.py` had been written against the binding default, dividing
the time scalar by 1800 while the policy it feeds was trained at 3600 — **the
deployed agent's clock ran at twice the rate it had learned**, on the one input
clock management depends on. Invisible in every simulator metric, because
nothing in the simulator path uses that file. The round-trip bit-exactness
tests did not catch it; they *pinned* it, by building their env with the bare
constructor. Both are fixed, and the tests now pass `TRAINING_MAX_TICKS`
explicitly.

**PRE-REGISTERED PREDICTIONS for the from-scratch 2.6 run, written before it
had trained.** The entropy fix is justified by a mechanism, not yet by an
outcome. These are the numbers that falsify it, all measurable with
`probe_perfect_defense.py` / `probe_card_usage.py` on the new net:

1. **Mean elixir rises above 2.25 / 10** and **P(nothing affordable) falls
   below 73.9%**. This is the direct claim. If bankruptcy persists at the same
   level, the normalizer was not the binding cause and the diagnosis is wrong.
2. **P(play) stops being flat against threat.** It was 0.1008 → 0.1016 from no
   threat to the largest. Any real defensive reflex has to show up here.
3. **`Policy/Entropy_Card_Frac` settles near 0.35 rather than below it.** It is
   now measured against a reachable ceiling, so the controller should be able
   to hold its target instead of chasing one it cannot reach.
4. **The card entropy COEFFICIENT should fall early, not rise.** A fresh net is
   near-uniform over its legal arms (~1.0 of reachable), i.e. far ABOVE target,
   so the controller must push down. Observed in the first updates: 0.0500 →
   0.0363. Under the old normalizer it rose instead.

Prediction 4 is already confirmed; 1-3 need a trained net.

**FIRST READ AT ep 600 (still very immature). Prediction 2 confirmed hard,
prediction 1 half-failed, and MY PREDICTION WAS PARTLY THE WRONG METRIC.**

| | old net (Giant deck) | ep-600 net (2.6) |
|---|---|---|
| P(nothing affordable) | 73.9% | **52.9%** |
| mean elixir | 2.25 | **1.84** (went DOWN) |
| P(play) overall | 0.105 | **0.237** |
| P(play \| affordable), no threat → HUGE | 0.3655 → 0.4719 | **0.4606 → 0.8601** |

**Prediction 2 is the real result.** The defensive reflex went from a 29%
relative rise across the threat range to an 87% rise ending at **0.86** — the
agent now plays a card on 86% of decisions where it can afford one during a big
push. That was the symptom being chased and it is gone.

**Prediction 1's "mean elixir rises" was a BAD PREDICTION and it failed.** Mean
elixir is not deck-invariant: 2.6 Hog Cycle has two 1-cost cards, so
"affordable" is satisfied at 1 elixir and the agent can correctly hold less
while having MORE options. The deck-invariant version of the claim —
P(nothing affordable) — improved 73.9% → 52.9%. Use that one; do not compare
mean elixir across decks.

**THE HONEST LIMIT, stated because the numbers above are otherwise
over-readable: the entropy fix and the deck change landed TOGETHER, so this
comparison cannot attribute the improvement to either one.** A cheaper deck
alone would raise P(play) and lower P(nothing affordable). The clean
attribution needs a from-scratch control arm on 2.6 with
`new_ent_card / log(n_legal)` reverted — ~28 h — and nothing here substitutes
for it. What IS unconfounded is the mechanism: the old divisor scores a
literal coin flip at 0.413 against a 0.35 target, which is arithmetic, not a
measurement.

One more early read, same caveat: Musketeer at ep 600 already preserves
**+458 HP** vs a random legal cell (CI [+329, +583], 276 better / 161 worse)
against the shipped net's +297 — but on a different deck, so not comparable.

**And the fresh run makes the defect legible in one number.** At ep ~180 the
untrained policy measures `Policy/Entropy_Card_Frac` = **0.958** — 95.8% of its
REACHABLE maximum, i.e. very nearly a coin flip, correctly far above the 0.35
target, so the controller drives the coefficient down to its 0.01 floor.

Under the OLD divisor that same uniform policy would have read
`0.958 × log(2)/log(5) = 0.413`. **A policy that is flipping a literal coin on
"play or wait" scored 0.413 against a 0.35 target — i.e. the old controller
considered near-maximum randomness to be roughly correct, and pushed UP from
there.** That single comparison is the whole bug.

Both heads now read on a scale where **1.0 means uniform over the legal arms**,
which is what makes the target interpretable at all. Placement reads 0.996
against its 0.649 annealed start, also correctly falling.

**RUN 1 PROGRESS, and an alarm I raised and then measured away.** The
from-scratch 2.6 run cleared the mirror phase fast — stage gates at ep 1,615 /
2,778 / 3,588 / 6,216 (opp elixir 1.0 → 1.4), then **phase advanced to
`random_opponent` at ep 6,373** on a 0.60 mirror win rate. Handoff to
`train_selfplay.py` scheduled at ep 11,373 (a 5,000-episode random-deck budget,
NOT the 40,000 cap this file used to name — the budget is what fires first).

Perfect defense, same checkpoint (ep 6,053), two opponent strengths:

| | @1.0x | @1.4x (live difficulty) |
|---|---|---|
| win rate | 1.000 | 0.700 |
| tower HP left \| WIN | 0.939 | 0.661 |
| flawless wins | **47.5%** | **0.0%** |
| crowns conceded \| win | 0.025 | 0.571 |

`W_FLAWLESS_DEFENSE` is measurably shaping behaviour (0.820 → 0.939 HP on wins
between ep 2,523 and 6,053 at matched 1.0x), **but zero flawless wins at 1.4x**
— the "zero tower damage" standard is currently a property of facing a weak
opponent, not a learned skill. Do not quote the 1.0x number alone; it is
saturated at win rate 1.000.

**THE ALARM: three of eight cards fell to near-zero usage — Hog 0.8%, Fireball
0.6%, Cannon 1.6% — the same three ROLES (win condition, spell, building) that
collapsed in the Giant deck.** That shape is why it looked structural.

**IT IS NOT THE PLACEMENT COLLAPSE. Measured, `prove_hog.py` (new), n=2,552
paired states, enemy tower damage over 600 ticks:**

| | |
|---|---|
| policy cell | **470.3** |
| random legal cell | 441.8 |
| delta | **+28.5**, 95% CI [+6.3, +51.7], 826 better / 734 worse |
| placement | modal (0,12) at **4.7%**, **155 distinct cells** |

Better than chance, and the dynamism is the **healthiest a win condition has
ever measured here** — against v1.2.0's Giant at 38 cells / 49.7% modal and the
cured net's 140 / 24.7%. A frozen head returns one cell; this one uses 155. So
the low usage is a **card-head VALUATION, not a broken placement function**, and
the documented response applies: do not force it. Forcing Fireball once dropped
win rate 97% → 23% because the low weighting was correct.

What stays open is whether the valuation is OPTIMAL. +28.5 on a base of 442 is
only +6.5% over random, so the Hog is placed better than chance but not
strongly. That question needs a forced-usage A/B, which is exactly the
experiment whose last outcome was "the policy was right".

**Two things this audit did NOT fix, stated so they are not mistaken for
solved.** `skip_frames = 10` (one decision per second) is a harder ceiling for
2.6 Hog Cycle than it was for Giant beatdown — pulling a Hog with a Cannon and
timing an Ice Spirit are sub-second decisions — and it was left alone because
changing it is an unmeasured throughput/precision trade, not because it is
fine. And the observation still carries **no card-cycle information**, which is
the single most deck-specific gap: 2.6 is *defined* by cycling back to Hog
faster than the opponent cycles their answer, and the net can see only the 4
cards in hand.

---

## The 1.5x Curriculum Overfitting Hypothesis

**Status: HYPOTHESIS, stated with its falsifier. Not established.** It is
recorded here because it is the best available explanation for a result that
survived four independent attempts to fix it, and because the test that would
kill it is cheap.

**The claim.** Every win-rate measurement that matters in this project is taken
against the C++ `HeuristicOpponent` at a **permanent 1.5x elixir multiplier**,
and phase 1's curriculum tops out at 1.5x as well. But the Hog Rider is a
*punish* card: its value comes from exploiting a temporary elixir deficit in the
opponent's economy. **An opponent with a permanent 1.5x multiplier never has a
meaningful deficit to punish.** Against a permanently over-resourced opponent,
spending 4 elixir on a unit with zero defensive utility is close to strictly
dominated -- whatever it deals, the answer is always affordable, and the
counter-push arrives against 4 fewer elixir of defence.

If that is right, the agent has not failed to learn its win condition. It has
*correctly solved the environment we built*, and the environment is closer to a
tower-defence survival mode than to Clash Royale. The 0% win-condition usage and
the cheap-cycle defensive turtle are then the mathematical optimum, not a
pathology -- which is exactly what every intervention has independently found.

**What it explains.** Four separate fixes, each targeting a different mechanism,
all produced nothing:

| intervention | result |
|---|---|
| `W_WIN_CONDITION_DAMAGE` reward multiplier | fired thousands of times, moved nothing |
| Hog advisor (bridge placement, engine-validated +176.0 tower damage) | usage 0.4% -> 0.2% |
| forced usage at epsilon = 0.15 | -0.125, p = 0.044 |
| SMART forcing (advisor timing gate + advisor cell, gate correctly calibrated to 1.5x) | **-0.300, p = 3.2e-06** |

A card that is genuinely viable should have responded to at least one of those.
A card that is *correctly valued at zero by the environment* responds to none of
them, which is what was observed.

**THE FALSIFIER, and it is one command.** Run the same paired forced-usage A/B
at `--opp-elixir 1.0`:

```bash
python_ai/venv/Scripts/python.exe python_ai/eval/force_hog_ab.py     --weights <net> --n 120 --opp-elixir 1.0 --force-prob 0.0 --smart-force
```

* If the penalty **shrinks or reverses** at 1.0x, the hypothesis is supported
  and the correct response is a CURRICULUM change -- train and evaluate across
  a range of multipliers including 1.0, rather than pinning everything at 1.5x.
* If the penalty **persists at ~-0.30**, the hypothesis is dead and the Hog is
  unviable in this engine's physics regardless of the opponent's economy. The
  honest response then is to stop rehabilitating it.

**A CEILING CAVEAT that must not be ignored when reading that test.** This file
already records that at 1.0x the ep-64k policy wins ~100%, so both arms can
saturate and pin the delta at 0 *by the opponent*, not by the treatment -- which
would look like support for the hypothesis while measuring nothing. The result
is only interpretable if the BASELINE arm is below ceiling. Check the baseline
win rate before reading the delta; if it is >= 0.95, the test is void and needs
an intermediate multiplier (1.2-1.3) instead.

### Why a permanent multiplier suppresses punish cards, mechanically

A multiplier is not the same kind of difficulty as a better opponent, and the
difference is the whole point. A stronger POLICY makes every one of our options
harder to execute roughly uniformly. A resource multiplier does something else:
it changes which STRATEGY CLASS is optimal, and it does so asymmetrically.

**The value of a punish card is priced against a window that the multiplier
shortens.** A win condition pays when it arrives while the opponent cannot
answer -- i.e. inside the window after they have just spent, before they have
regenerated the cost of an answer. With regeneration rate `r` and multiplier
`m`, that window lasts about `answer_cost / (m * r)`. At `m = 1.5` every such
window is **two thirds** its natural length, so P(the Hog arrives unanswered)
falls by roughly the same factor.

**And the cost of the same play RISES with `m`.** Our 4 elixir is spent
regardless; what the opponent then does with their surplus scales with `m`. So
across the multiplier the expected value moves roughly as `1/m` while the
expected counter-cost moves as `m`, and the ratio degrades on the order of
`1/m^2`. That is a very steep suppression for a parameter we treated as a mere
difficulty dial.

**Defence, meanwhile, gets MORE valuable under the same change.** A defensive
card's value scales with the volume of threats it answers, and a multiplier
increases exactly that volume. So the multiplier pushes offence down and defence
up simultaneously. The optimum does not merely shift; the ranking of strategy
classes inverts.

**This is why the interventions could not work.** Four mechanisms were built to
make the agent play its win condition -- a reward multiplier, an advisor target,
random exploration, gate-timed forcing -- and every one of them tries to move a
POLICY. None of them changes the PAYOFF. If the environment prices the Hog at
negative value, a correctly-functioning learner will keep finding its way back
to 0%, and a forcing mechanism will simply pay the negative price more often.
That is precisely the observed dose-response: usage up, win rate down, monotone.

**It predicts the spell too, and that prediction was already sitting in the
data.** Fireball is also a tempo/punish play, and it sits at 0.1% usage -- the
same pit, never separately investigated. Under this hypothesis the two are one
phenomenon, not two coincidences.

**What it does NOT license.** It does not say the agent is strong, and it does
not say the defence is "bulletproof". The 1.00x arm's 1.000 win rate says one
specific non-neural opponent, at its weakest setting, cannot beat this policy --
a fact this file already recorded for the ep-64k net and the reason the ceiling
caveat exists. It is not a statement about Clash Royale, about human opponents,
or about the policy's robustness to a strategy class it has never faced.

**TESTED, 2026-08-18, AND THE HYPOTHESIS IS SUPPORTED -- BUT NOT PROVEN.** The
same SMART-forced A/B (advisor timing gate + advisor bridge cell, gate
calibrated to each multiplier), n=120 paired, ep-25202 net:

| opponent | baseline win | SMART win | delta | p |
|---|---|---|---|---|
| **1.00x** | **1.000** | 1.000 | +0.0000 | -- VOID (ceiling) |
| **1.25x** | 0.950 | 0.825 | **-0.1250** | 0.0059 |
| **1.50x** | 0.617 | 0.317 | **-0.3000** | 3.2e-06 |

**The cost of playing the win condition falls monotonically as the opponent's
economy falls.** That is the hypothesis's central prediction and it holds.

The 1.00x row is VOID exactly as the caveat above warned -- the baseline
saturates at 1.000 and the delta is pinned by the opponent, not the treatment.
It is reported rather than dropped because deleting a void arm after seeing it
is how a ceiling gets mistaken for a cure.

**And the honest alternative explanation, which is NOT fully excluded.** A
baseline near 1.0 has less room to lose, so some of the shrinkage is
compression. Normalising by available headroom: at 1.25x the penalty is
0.125/0.950 = **13%** of what could be lost; at 1.5x it is 0.300/0.617 =
**49%**. Still smaller at the lower multiplier, so the effect survives that
correction -- but 1.25x is close enough to the ceiling that the point deserves
replication at a multiplier where the baseline sits nearer 0.7-0.8.

**WHAT THIS DOES NOT SAY: the Hog is not rehabilitated at any multiplier
tested.** The penalty shrinks; it never reverses. Even at 1.25x, forcing the
win condition with perfect timing and an engine-validated placement costs a
significant 12.5 win-rate points. So the correct response is a CURRICULUM
change -- train and evaluate across a range of opponent economies instead of
pinning everything at 1.4-1.5x -- and NOT more Hog-specific machinery. Four
such mechanisms have now been built and measured, and all four returned null.

**A second, cheaper prediction worth checking.** If the hypothesis holds, the
win condition should be *more* used, not less, by any policy trained with 1.0x
exposure. Nothing in the current run provides that -- phase 1's curriculum
starts at 1.0x but the agent passes through it in ~1,600 episodes and spends the
remaining ~24,000 at 1.4-1.5x.

---

## 2026-08-19: the curriculum pivot, and the hypothesis it did NOT confirm

**The elixir-multiplier curriculum is GONE from training.** `CURRICULUM_STAGES`
no longer carries `opp_elixir_multiplier`; both sides run at a symmetric 1.0x
forever and the six rungs index `teacher.TEACHER_STAGES` — lookahead
0/0/30/30/60/60 ticks with epsilon 0.30 → 0.00. **Difficulty is COMPETENCE, not
economy.** `set_opponent_elixir_multiplier` stays bound and callable for the ~15
measurement harnesses that sweep it;
`test_training_never_raises_the_opponent_elixir_multiplier` fails if training
touches it again. **GAMEPLAY-AFFECTING**: every win rate earned against
`heuristic@{1.0..1.5}x` is historical. Checkpoints are NOT invalidated — the
observation, action space and architecture are untouched.

**Phase 1's opponent is now `teacher.UtilityTeacher`** (`CLASH_PHASE1_OPPONENT`,
default `teacher`). The C++ `HeuristicOpponent` is an EVAL ANCHOR only.

### Why the multiplier had to be replaced rather than merely deleted

At 1.0x the C++ heuristic is beaten ~100% — recorded for the ep-64k Giant net
and the ep-25202 2.6 net. **Deleting the handicap alone converts a MISPRICED
environment into a ZERO-GRADIENT one**, which is exactly why the original
hypothesis test's 1.00x row was VOID. The replacement has to be a real opponent.

**It cannot be the neural search we already own.** Search scores candidates with
the net's OWN critic, so at random init the expert is not an expert (measured:
random init deviates on 21.5% of decisions and gains nothing). A hand-written
utility function has no cold start. That is the whole argument for `teacher.py`.

### The teacher: rules propose, simulation ranks

`tactics.py`'s engine-validated cells are the candidate generator; each
candidate is rolled forward on `env.snapshot()` and scored by DIFFERENCE against
the NO-OP rollout, so no-op scores exactly 0 and every score is marginal.
Side-agnostic by construction, reads observations only, ~3.5 ms/decision.

Two things the design brief got wrong, both corrected with the reason:

- **The no-op baseline does NOT absorb opportunity cost.** It removes
  counterfactual value; the elixir actually spent still has to be charged.
- **Scored on realized damage alone the teacher would never attack and would
  turtle** — a Hog needs ~13 s to cross and the horizon is 3–6 s. That is the
  pathology the pivot exists to remove, so `positional_advantage` is the
  hand-written stand-in for the value bootstrap search gets from the critic.

**Both strength bars pass.** vs the C++ heuristic @1.0x: stage 5 scores 1.000
(stages 0–4: 0.54/0.29/0.71/0.96/0.92). The ep-31312 2.6 net beats it 0.775 (CI
[0.663, 0.875], n=20 side-swapped) — materially harder than the heuristic that
same net beats ~100%, and not a wall. **And it is not a turtle**: its own
win-condition usage RISES with competence, 7.1% → 13.4% → **16.5%** of plays,
against the RL agent's 0.8%.

### THE HYPOTHESIS IS NOT CONFIRMED. Five measurements, all pointing the same way.

The pivot was built on the 1.5x hypothesis: that a symmetric economy would make
the win condition pay. **It does not.** `prove_environment.py` runs the
falsifier with NO network — both sides are the deterministic teacher, so the
historical confound between "the environment prices this badly" and "this net
cannot execute it" is gone.

**Symmetric, teacher vs teacher, n=100 paired.** Arms are `attack` (bridge) vs
`cycle` (same card, same spend, same hand rotation, placed in our own back half
at ~3 tower damage against 535.6). The `ban` arm is reported but CONFOUNDED — a
card never played never leaves the hand, so banning also clogs a slot:

| opp elixir | attack | cycle | delta | p |
|---|---|---|---|---|
| 1.00 | 0.510 | 0.840 | **−0.330** | 5.7e-08 |
| 1.10 | 0.145 | 0.455 | −0.310 | 4.5e-07 |
| 1.20 | 0.015 | 0.130 | −0.115 | 7.6e-05 |

At 1.0x with the continuous readout: attack deals **less** enemy tower damage
(6326 vs 6938, p=0.13) and takes **nearly double** (5772.7 vs 3049.3). The `ban`
arm scores **0.910** — never playing the win condition at all beats attacking
with it by 0.37.

**The historical result REPLICATES with a deterministic bot**, so it was never a
policy failure: vs the C++ heuristic, n=80, 1.00x VOID at ceiling, 1.25x
**−0.1875** (p=0.0096), 1.50x **−0.1062** (p=0.00049).

### The mechanism, isolated — and it is NOT the card, the King, or the timing

- **The card is fine.** A lone Hog injected on an empty board deals **2536 tower
  damage** (a full Princess Tower) and dies at tick 190. So neither the Hog nor
  the enemy King firing from tick 0 (`Tower.h` has no activation condition,
  unlike the real game) is what suppresses it. **CLAUDE.md's old "317 tower
  damage in 40 s" figure no longer reproduces** — do not quote it.
- **The punish mechanic is real.** Forcing the defender's bar to 1.0:
  **1025.0 tower damage vs 634.0** at match elixir, i.e. **+391 HP (+62%)**,
  256 vs 158 hp per elixir committed.
- **Escorting helps, conditionally.** Ice Golem first is +344 HP (CI [+186,
  +524]) against a solvent defender and actively wasteful against a broke one
  (dmg/elixir 256 → 194). A correct and fairly subtle result, and the reason a
  lone-Hog-only measurement would have been a strawman.
- **Cheap answers are not the blocker.** The defender answers with Skeletons (35
  of 40 trials) and Ice Spirit (31 of 40) — a 1-for-4 trade in its favour, which
  is REAL Clash, not an artifact. Forcing both out of its hand moved the Hog only
  634.0 → 665.7, because it cycles them back inside the horizon.

**THE GATE SWEEP IS WHAT SETTLES IT, and it inverts the premise.** Marginal value
of ONE commitment, paired, both sides playing on afterwards:

| gate: commit while opp elixir ≤ | states | tower-HP delta | 95% CI | sign test |
|---|---|---|---|---|
| 7.0 (shipping) | 220 | −298.2 | [−494, −107] | 87/105, p=0.22 |
| 3.0 | 160 | −268.6 | [−455, −90] | 50/81, p=0.0085 |
| **1.5** | 162 | **−585.4** | [−816, −360] | 46/100, **p=9.2e-06** |

**Tighter timing is WORSE.** The pivot assumed a punish window is a state where
the opponent cannot afford the answer. Artificially constructed, that state pays
(+391 HP). **Naturally occurring, it is the opposite: they are low BECAUSE THEY
JUST SPENT, so their push is already on the board and the correct move is to
defend.** Committing 4 elixir there is the worst available moment, which is what
the −585.4 row measures.

So the punish window exists in principle and **essentially never occurs in this
environment's dynamics**. That is a property of the engine's PACE — not the
elixir multiplier, not the policy, not the placement.

### What this licenses and what it does not

Every number above is against ONE opponent, whose attack repertoire is a bridge
push (lone or escorted) and whose scorer carries a known bias:
**`UtilityTeacher.rollout_stats` rolls candidates forward with BOTH SIDES
NO-OPING**, so it structurally cannot see the counter-push — the entire cost of
attacking. Left as a documented limitation rather than silently fixed, because
the fix costs ~20x per decision and the gate sweep shows the cheap alternative
(a solvency threshold) would not have helped at any threshold.

So: "the win condition is net-negative" means *against a search-based defender
at a symmetric economy with this repertoire*. It is **not** a proof that no
attacking strategy pays here. What five independent measurements DO rule out is
the specific claim the pivot was built on — that removing the multiplier
suffices.

**Do not start a from-scratch run expecting Hog usage to rise.** The curriculum
is still worth shipping: it removes a measured mispricing and a zero-gradient
ceiling, and the teacher is a much better sparring partner than the heuristic.
It is not a win-condition cure. Offensive scenario injection
(`scenario_offense.py`) stays DEFAULT-OFF; its gating condition was not met.

The open question is now an ENGINE question — the cost balance between offence
and defence — and belongs in `perception/UPSTREAM_REQUESTS.md`, not in more
policy-side machinery. **That would be the fifth Hog mechanism; four have
already returned null.**

### Also fixed on the way

**`python_ai/clash_royale_env.pyd` was STALE** — it predated commit `26de409`,
so `set_elixir_for_team` / `set_hand_for_team` were missing even though
`bindings.cpp` exports them. That is the MSB3073 post-build copy failure this
file already documents, and it silently blocked two measurements. Replaced from
`build_python/Release/`. Verified gameplay-identical rather than assumed: both
commits since that build are purely additive accessors (119 insertions, 0
deletions, no simulation code), and the perception suite reads 353 passed / 1
skipped either side of the swap.

**`inject` QUEUES a spawn.** The unit does not reach the board, or any
observation, until one tick is stepped — measured 0.0 enemy mass immediately
after `inject`, 0.399 after one tick. Any harness that injects and then reads an
observation without stepping is looking at an empty board.

---

## 2026-08-19 (later): DEPLOY TIME landed, and it fixes the offence/defence balance

**ENGINE CHANGE, GAMEPLAY-AFFECTING.** A freshly placed troop or building is now
inert for `DEPLOY_TIME_TICKS = 10` (1.0 s): on the board, **targetable and
damageable**, but unable to move, acquire a target or attack. Spells keep their
own `spellDelayTicks`, towers are never inert, projectiles are unaffected.
**Every win rate earned before this is historical.** Checkpoints are not
invalidated (no observation/action/architecture change).

`CardStats.h` `DEPLOY_TIME_TICKS` → set in `CardFactories::applyCardMetadata` →
consumed in `CombatEntity::update`.

### Why it was the right root cause

The omission was **not symmetric**. A defender places INTO an existing threat and
needs its answer to act NOW; an attacker places before contact and would have
spent that second walking anyway. So the missing second was a standing subsidy to
DEFENCE, charged on every defensive placement — which is exactly the shape of
what was measured: the defence answered a 4-elixir commitment for **1.07 elixir**
while the card itself was fine (2536 tower damage unopposed).

### THE CONTROLLED RESULT

Same harness, same supported push (tank one decision ahead, win condition
behind), same protocol, ~161 scored states each. **The only difference is the
engine:**

| engine | marginal value of a supported push | 95% CI | sign test |
|---|---|---|---|
| `DEPLOY_TIME_TICKS = 0` (old) | **−73.7** HP | [−349.5, +195.3] | 70/78, p=0.57 |
| `DEPLOY_TIME_TICKS = 10` (new) | **+448.5** HP | [+137.3, +760.1] | 88/64, p=0.062 |

**The supported arm alone does not do it — on the old engine it is a null.
Deploy time is what flips the win condition to positive value.** The control was
run by rebuilding with the constant zeroed and then restoring it, because the
supported arm was itself a new treatment and without the control this would have
been "we changed two things and something improved".

The elixir trade moved the same way, n=60 paired:

| | before | after |
|---|---|---|
| defence elixir to answer a lone Hog | 1.07 | **2.93** |
| resulting trade on a 4-cost commitment | −2.93 | **−1.07** |

### AND IT MADE THE GAME MORE LIKE CLASH, which is the better headline

The naked Hog got **worse** and the escorted push got **better**. In a punish
window (defender forced to 1.0 elixir), lone Hog fell 1025 → **343** tower damage
— it now stands inert under tower fire for a second — while the escorted push
holds at **993.8**, and escorting is worth **+650 HP, CI [+429, +878]**.

That is correct Clash: you do not send a naked win condition, and the engine now
prices that. It also means **the lone-Hog measurements are now a strawman** —
`prove_environment.py --mode marginal` scores a lone commitment at −556.3 and a
supported one at +448.5 **on the same engine**. Always read the supported arm.

### What is still negative, and why it is not alarming

The strategy-level win-rate arms (`--opponent teacher`, n=100 paired) still
favour not attacking: attack 0.490 vs cycle 0.715 (−0.225, improved from −0.330;
damage-taken gap 2724 → 1576). **That arm sends a LONE Hog**, because
`teacher._cells_for` proposes one card per decision and the scorer cannot plan a
two-card push. So the teacher's own repertoire, not the engine, is now the
binding constraint on that number.

For TRAINING this is the environment you want: **supported pushes pay, naked
pushes are punished.** A policy is not restricted to the teacher's repertoire and
can learn the difference — which is the first time that has been true here.

### The teacher's ladder is now lookahead-driven

`TEACHER_STAGES` horizons: **0 / 10 / 30 / 50 / 70 / 100 ticks** (0 → 10 s), with
epsilon 0.30 → 0.00. Stage 0 is deliberately short-sighted so an unpolished agent
can beat it; the top rung simulates a full 10 s, long enough to watch a push
arrive, be answered and be counter-pushed.

Measured cost at the 10 s rung: **3.26 ms/decision, 0.71 s/episode** against a
~22 s per-env episode at `num_envs = 8`. Depth is cheap, width is not.

**It stops at 10 s deliberately.** `rollout_stats` rolls forward with both sides
no-oping, and past ~12 s of that a rollout stops resembling the game — the neural
search measured horizon 20 WORSE than 12 (0.875 vs 0.963) for this exact reason.

### Two traps worth carrying

- **Inert is not time-stopped.** The first implementation returned from the top
  of `CombatEntity::update`, which also froze freeze/shield/buff/curse/ability
  cooldown and the poison damage-over-time mark — a unit deployed into a Poison
  would have been briefly immune. The bail-out belongs AFTER the status-timer
  block and BEFORE the action block. `test_game_manager`'s champion-cooldown case
  is what caught it, which is why that failure was worth diagnosing instead of
  editing away.
- **15 existing C++ tests spawned a card and asserted on the very next tick.**
  They now advance past deploy explicitly via `advancePastDeploy()` rather than
  absorbing the delay into a changed magic number. Tests that build entities
  directly (`StationaryCombatant`, a bare `MeleeTroop`) were untouched, because
  `applyCardMetadata` is what assigns deploy time — that split is why most of the
  suite needed no change.

---

## 2026-08-20: the structural refactor. NOT gameplay-affecting.

**Read this first: no win rate is invalidated and no checkpoint is dead.** The
observation, the action space, the architecture and the reward are untouched;
`model_weights.pth` and `model_weights_selfplay.pth` load and behave exactly as
before. Everything below is about where code LIVES, not what it computes.

`python_ai/` was a flat directory of 45 modules. `train.py` was 2,404 lines with
a 1,620-line `train_ppo()`; `train_selfplay.py` was 2,884 with a 1,565-line
`train_selfplay_ppo()`. It is now a package of eleven subpackages, and those two
files are 506 and 488 lines.

### What was actually duplicated, and what replaced it

The overlap between the two trainers was not incidental — it was the whole
algorithm. Same hyperparameters, same 17-key stats extraction, same rollout with
the same three masks, same GAE, same ~250-line minibatch update, same entropy
controller, same replay recorder, same checkpoint cadence. The differences fit
on one screen:

| aspect | pipeline 1 | pipeline 2 |
|---|---|---|
| opponent | UtilityTeacher | PFSP league / scripted |
| episode bookkeeping | curriculum + phase | scenarios + strategy ROI |
| GAE | plain | truncation bootstrap |
| draw penalty keyed on | `dones` | `terminateds` only |
| entropy config | `PHASE1_ENTROPY` | `PHASE2_ENTROPY` |
| periodic work | replay, handoff | replay, eval, exploiter |

So `rl/base_trainer.py` owns the loop as a TEMPLATE METHOD and the two trainers
own only their differences. The template shape (rather than
composition-by-callback) is deliberate: the loop's ORDER is itself load-bearing
in several places — the phase gate must be evaluated before the stage gate, the
hidden state must be captured before the LSTM advances, the coverage slot must
be drawn on the observation the action is taken from — and a template puts that
order in exactly one place. **No subclass may override `collect_rollout` or
`run_update`**; `tests/test_rl_base_trainer.py` asserts it.

### Three couplings that were removed, each with a cost that had been paid

- **`shipping.py` imported a 1,152-line experiment harness to reach a
  dataclass.** `SearchCfg` lived in `expert_iteration.py`, so the one file that
  names the deployable configuration pulled in `bc_pretrain`, torch and a card
  registry probe. It is now `search/config.py` and imports nothing but
  `dataclasses`.
- **Five modules imported `_build_candidates` / `_search_action` /
  `_policy_head` / `_greedy_from_logits`** — underscore-private names, across
  module boundaries, out of an A/B harness whose `main()` runs a whole
  experiment. The search is `search/search.py` under public names now.
- **FOUR copies of paired-bootstrap CI + exact sign test.** For measurement code
  that is worse than ordinary duplication: four copies is four chances for one
  to quietly use a different tail or a one-sided test. One `eval/stats.py`, with
  the win-rate variant (normal-approximation CI + McNemar + the power line)
  preserved exactly, because every expert-iteration delta recorded above was
  produced by that arithmetic.

### Two behaviour changes, both stated rather than slipped in

- **Pipeline 1 now also logs `Entropy/Placement_ByCard_Min`**, the conditional
  freeze detector this file names as the cheap detector for a per-card collapse.
  Diagnostic only — computed from tensors the update already had, and phase 1
  previously had no per-card diagnostic at all.
- **`exploiter.py` used a THIRD copy of the stats dict, and it was missing
  `team0_wincon_damage`** — so the exploiter alone trained without the
  win-condition term, optimizing a different reward from the agent it hunts. It
  now uses the shared extractor. GAMEPLAY-AFFECTING for the exploiter, which has
  been disabled since 2026-08-11, so no run is invalidated.

### One LATENT BUG the extraction removed, for free

`train.py` had two hand-written `torch.save({...})` blocks: the periodic one
persisted `ent_coef_card` / `ent_coef_place`, and the FINAL one at the stop
point did not. So the very last checkpoint pipeline 1 wrote — the one pipeline 2
bootstraps from, and the one any resume picks up — silently dropped the
converged entropy controller and sent it back to its 0.05 / 0.06 seed values.

That is exactly the failure already on record from the other pipeline (observed
2026-07-30: placement reset from a converged 0.0132 to 0.06 and took ~5,600
episodes to walk back, with nothing warning). One `save_checkpoint()` makes the
two saves the same object by construction, and
`test_the_final_save_carries_the_SAME_keys_as_the_periodic_one` pins it.

Nothing else changed about either save. Pipeline 2's periodic cadence now also
honours `CLASH_SAVE_EVERY` (it was a hardcoded 500); the default is identical.

### What the tests now pin that they could not before

`CURRICULUM_STAGES` was a LOCAL of `train_ppo()`, so the suite parsed `train.py`
with `ast` to recover the literal — a constant nothing can import is a constant
nothing can check, and a parser that finds nothing is only distinguishable from
one that finds the wrong thing by the assertion it raises. It is at module scope
in `rl/curriculum.py` now, and the state machine's ORDERING is tested by running
the two gates the wrong way round and watching the phase transition starve.

`test_training_never_raises_the_opponent_elixir_multiplier` also grew: it is
AST-based rather than line-based (the line scan skipped anything starting with
`#`, which silently exempted every mention inside a DOCSTRING — exactly where a
future author would explain the ban before reintroducing it), and it scans
`rl/` as well as `trainers/train.py`, because the loop moved. It is still
deliberately scoped to PIPELINE 1: `BUILTIN_TRAINING_OPPONENTS` really does put
`heuristic@1.35` and `@1.50` in phase 2's pool, where a multiplier is the
anchor's identity rather than a curriculum handicap.

The suite went **90 -> 312 tests**. The 2,170-line `test_python_ai.py` was
itself a monolith and had grown a cross-section dependency invisible from inside
it (`_shaping_stats` defined in one section, used in another); it is eight files
plus `conftest.py` and `helpers.py`.

### The one trap this created, for anyone adding a script

`python_ai` is a package rooted at the REPO ROOT, so a module run as a file
(`python python_ai/eval/prove_hog.py`) does not have `python_ai.*` on its path.
Every runnable script therefore carries a four-line bootstrap, and
`test_package_layout.py` fails if one is added without it —
`trainers/bc_pretrain.py` was missing it and could not be run as a script at all.

The other half of the same trap: `python_ai.PACKAGE_DIR` is where the `.pth`
files live, and it is NOT the directory a moved script sits in. Every
`__file__`-relative checkpoint path became silently wrong the moment the file
moved one level down; they all read `PACKAGE_DIR` now.

---

## 2026-08-20 (later): the MULTI-CARD teacher. TODO.md item 1.

**Not gameplay-affecting for the agent, and no checkpoint is invalidated** — the
observation, action space, architecture and reward are untouched. What changed
is `UtilityTeacher`'s repertoire, so **phase 1's opponent is stronger in kind,
and win rates measured against `teacher@stage N` before this date describe a bot
that could only play one card per decision.**

The 2026-08-19 deploy-time change made the escorted push the correct play
(+448.5 tower HP marginally) and the naked push the punished one (−556.3), and
`UtilityTeacher` could not express the escorted one: `_cells_for` proposed cells
for a single card and `score` ranked single candidates. That, not the engine,
was the remaining reason `prove_environment.py`'s strategy arm read attack 0.490
vs cycle 0.715.

### The engine fact that decided the design

**`step_self_play(slot, x, y, ..., 0)` places nothing.** Placement is processed
inside the tick loop, so a 0-tick call spends no elixir and puts no unit on the
board (measured; pinned by
`test_a_zero_tick_step_places_nothing_which_is_why_a_combo_is_a_SEQUENCE`). And
`gym_wrapper.step` hands the teacher exactly one `(slot, x, y)` per decision.

So there is no such thing as a simultaneous two-card placement, and a combo is a
**sequence across consecutive decisions** — which is also the shape the +448.5
measurement was taken at. **No C++ change was needed or made.** Deploy time is
per-entity (`CardFactories::applyCardMetadata`), verified from Python: a card
placed 10 ticks after another still sits out its own full second before moving.

### What was built

`Candidate` now carries a **tuple of `PlacementStep`s** rather than one cell;
`.slot/.x/.y` survive and mean the first step, so every caller that unpacks
three values is untouched. `.placements` is the `[(slot, x, y), ...]` form.
Six curated families, taken round-robin under a `max_combos` budget:

| family | shape |
|---|---|
| `supported_push` | tank, then the win condition behind it, same lane |
| `counter_push` | the CHEAPEST body as escort instead of the tank |
| `defensive_stack` | building (or the centre pull), then a body on the threat |
| `cheap_defence` | the two cheapest bodies onto the same threat |
| `spell_then_push` | clear the lane, then walk into it |
| `push_then_spell` | the win condition, then the spell answering its answer |

`max_combos` is a **third competence axis** in `TEACHER_STAGES` (0/0/2/3/4/4),
zero on the two shortest rungs because their horizons cannot reach a follow-up.

### The generator was necessary and NOT sufficient, and that is the useful result

A first version shipped only the families, at a fixed one-decision gap. Measured
teacher-vs-teacher at stage 5 over ~2,400 decisions:

| | |
|---|---|
| elixir mean / p90 | **1.79 / 3.30** |
| states where ANY combo was affordable | **2** |
| ...and both were the 5.00 opening bar | |

At a one-second gap both cards must be affordable at once — Ice Golem + Hog is
6, Skeletons + Hog is 4.65 — against a bar whose p90 is 3.30. **The teacher
could propose a play it could never afford.** Same shape as three other
already-recorded traps: a mechanism whose precondition never occurs.

**A FLAT SAVINGS CHARGE WAS TRIED AND IS MEASURED DEAD.** Charging every
marginal spend while a push was within six decisions of affordable moved the bar
**1.79 → 1.73** across reserves of 0.0 / 1.5 / 3.0 / 5.0 — not at all, and if
anything the wrong way. A per-decision charge cannot manufacture multi-second
saving when the bot has many attractive cheap plays and the defence exemption
keeps firing. Removed rather than shipped, on the principle this file already
records for back-row structure penalties: *a penalty cannot move a distribution
with no mass to move.* Do not re-propose it.

### What DID work: the gap is a searched axis

The pair's cost is paid **across** the gap, not at the plan. `COMBO_FOLLOWUP_DELAYS
= (10, 30, 50)` — 1 s, 3 s, 5 s — so 3 s of regeneration is worth 1.05 elixir and
5 s 1.75, moving Skeletons + Hog from 4.65 to 3.95 and **3.25**, i.e. from never
to sometimes. A gap is offered only when the rollout runs past it, so a short
rung simply sees fewer gaps. Proposals went **2 → 94-109** per ~2,500 decisions.

**And the ranking is TACTICAL, not a terminal-evaluation artifact** — worth
checking, because `positional_advantage` is read at the END of the horizon, so a
card placed at t=50 is fresher there than one placed at t=10 and could win for no
tactical reason. Measured with the bar pinned at 8.0 so all three gaps are
affordable and ONLY the gap varies, n=21 states, same pair, same cells:

| gap | mean score | median |
|---|---|---|
| **10 (1 s)** | **+5.281** | +4.524 |
| 30 (3 s) | +2.529 | +2.039 |
| 50 (5 s) | +1.761 | +1.555 |

Monotone toward the TIGHT escort, which is what "tank one decision ahead" says it
should be. The 5 s gap dominates real play purely because it is the only one that
can be paid for.

### The plan, and why the follow-up is OFFERED rather than executed

`act()` returns one placement, so the second half is carried as `pending` with a
tick countdown and offered as an ordinary candidate on the decision it comes due
— **re-scored, not executed blindly.** That is not a weaker plan: by then the
tank is physically on the board, so a solo rollout of the win condition already
sees the escort, and adding a synergy bonus would double-count it. A narrow
`plan_reserve_penalty` charges other spends that would leave the committed
follow-up unaffordable, exempted above `tactics.HOG_MAX_THREAT` so defence is
never blocked.

**Measured abandonment, 19 chosen plans:** 3 completed, 10 lost the argmax when
due, 5 became unaffordable, 1 lost its hand slot. The dominant reason is the
design working — the board moved and the simulator preferred something else.

### PHASE 3: the lookahead sweep against the ep-31,312 net

`prove_combos.py --vs-net`, 24 paired openings per horizon, sides swapped,
teacher at stage 5 with only `horizon_ticks` varying. **The reported score is
the TEACHER's.** (The brief asked for "ep 25202"; that checkpoint was deleted in
the 2026-08-19 cleanup, and `model_weights_selfplay.pth` at ep 31,312 is the
surviving descendant of the same 2.6 run.)

| horizon | teacher score | 95% CI | combos chosen | completed |
|---|---|---|---|---|
| 30 (3 s) | 0.031 | [0.000, 0.083] | 3 | 0 |
| 50 (5 s) | 0.094 | [0.021, 0.177] | 3 | 1 |
| 70 (7 s) | 0.083 | [0.021, 0.167] | 18 | 4 |
| **100 (10 s)** | **0.115** | [0.042, 0.198] | 23 | 6 |
| 120 (12 s) | 0.083 | [0.021, 0.167] | 46 | 3 |

**No depth in 3-12 s makes the teacher beat this net.** Depth helps from 3 s to
10 s and then stops: 12 s is worse than 10 s, which independently reproduces the
reason `TEACHER_STAGES` tops out at 100 ticks and the reason the neural search
measured horizon 20 worse than 12 -- `rollout_stats` rolls forward with both
sides no-oping, and past ~12 s of that a rollout stops resembling the game. The
CIs overlap heavily at n=24, so read the 3 s -> 10 s rise as directional and the
10 s -> 12 s fall as consistent-with-known-physics, not as resolved.

**CROSS-CHECKED against the established harness**, which is the only reason
these numbers are quotable: `prove_teacher.py --vs shipping --n 20 --stage 5`
gives net 0.8625, CI [0.7500, 0.9500], i.e. teacher 0.1375 -- agreeing with this
harness's 0.115 at the same horizon.

**The teacher itself is not weakened. Both of `prove_teacher.py`'s bars still
pass**, re-run after the change (n=20 per stage, vs the C++ HeuristicOpponent at
1.0x): 0.500 / 0.600 / 0.750 / 1.000 / 0.975 / **1.000** for stages 0-5. Monotone
in competence and saturated at the top, as before.

CLAUDE.md previously recorded the net beating the teacher **0.775**; it is
0.8625 now. That measurement predates `DEPLOY_TIME_TICKS`, which specifically
punishes the naked bridge push that was the teacher's entire attack repertoire --
so the widening gap is the expected consequence of the engine fix, and is the
gap this item exists to close.

### The win-rate verdict is a NULL, and it is reported as one

`prove_combos.py --combo-ab`, stage 5, 40 paired openings per run, sides
swapped, only team 0's combo width varying. **Four runs, and every one of them
is individually a null:**

| run | families on the ON arm | delta | 95% CI | sign test |
|---|---|---|---|---|
| 1 | 5 (no `cheap_defence`) | **+0.000** | [−0.081, +0.081] | 9/9, p = 1 |
| 2 | 6 | −0.031 | [−0.113, +0.050] | 8/10, p = 0.815 |
| 3 | 6 | −0.081 | [−0.175, +0.006] | 7/14, p = 0.189 |
| 4 | 5 (`cheap_defence` ablated) | −0.031 | [−0.113, +0.050] | 8/12, p = 0.503 |

Pooled: **6 families −0.056, 5 families −0.016, all four −0.036 (n=160)**, against
a per-run 95% half-width of ~0.081 and ~0.057 for two pooled runs. So the
difference between the two configurations is deep inside the noise and **the
ablation neither convicts nor exonerates `cheap_defence`.**

**SUPERSEDED 2026-08-21 -- POOLED OVER FIVE RUNS IT IS NOT A NULL.** A fifth run
at the new `play_margin` came back **-0.056**, giving five independent paired
deltas:

    +0.000   -0.031   -0.081   -0.031   -0.056       4 negative / 0 positive

Levels are not comparable across runs (unseeded openings), but each DELTA is an
unbiased estimate of the same quantity, so the deltas pool. Over **200 paired
openings**:

    pooled mean  -0.0398
    95% CI       [-0.0760, -0.0036]   from the per-run half-width
    95% CI       [-0.0665, -0.0131]   from the observed between-run spread

Both exclude zero. **The combo machinery costs the teacher about 4 win-rate
points against a mirror.** "A measured null" was correct at n=40 and wrong at
n=200 -- the honest statement is a small, real negative that four separate runs
individually lacked the power to resolve. It is recorded this way round because
the mistake is the instructive part: four non-significant results with a
consistent sign are not four nulls.

**The likely mechanism, and it is actionable:** completion is 40%, so 60% of
chosen combos leave a first card on the board for a plan that never completes --
paying half a push and getting none of it. Raising completion is the lever, not
deleting families.

**Kept ON, and that is a VALUES call rather than a measurement.** A sparring
partner's worth is its REPERTOIRE, not its mirror win rate: the engine has
rewarded escorted pushes since deploy time landed, and a teacher that cannot
express them teaches the agent nothing about facing them. The price is ~4 points
against itself while it still beats the C++ heuristic 1.000 and beats the
pre-2026-08-21 teacher 95-5. `max_combos = 0` (action-identical to the old
teacher) and `combo_families` are the one-line off switches.

**THE METHODOLOGICAL FINDING IS THE MORE VALUABLE HALF, and it invalidated my
own control.** Run 4 was designed with a built-in validity check: it shares
seed 300 with run 3, so its OFF arm should have reproduced run 3's OFF arm
exactly. It did not — **0.475 against 0.537** — and the reason is that
`ClashEnv::reset()`'s opening-hand shuffle was **UNSEEDED** (`UPSTREAM_REQUESTS.md`
item 7 — **since FIXED, applied and verified 2026-08-21**; see "`env.seed()`
works and this file was stale about it" below, which supersedes the bullet at
the end of this list). `--seed` reaches only the teachers' own RNG, which at
stage 5 is just the lane bias, so two invocations of this harness draw entirely
different match populations no matter what seed is passed.

The consequence is specific and worth carrying:

- **Within a run the pairing is sound** — both arms play the same
  `base.snapshot()`, so the paired delta is valid and that is what is reported.
- **Across runs only the DELTAS are comparable, never the arm levels.** Runs 2
  and 3 both happening to report an OFF arm of 0.537 was coincidence, and
  reading that as reproducibility is what made run 4's control look like a
  failed comparison rather than a mis-specified one.
- Any future "run it again at a different seed and check the baseline matches"
  design in this repo is invalid for the same reason until item 7 lands.
  **ITEM 7 HAS SINCE LANDED (2026-08-21), so this bullet no longer applies to a
  harness that actually calls `seed()`** — arm levels reproduce to four decimals
  across independent invocations. It still applies to every harness that does
  not, which is most of `prove_*.py`. Check the harness before assuming either
  way; the failed run-4 control above remains a correct account of what happened
  at the time, not a live constraint.

**Kept ON by default, with the trend stated rather than buried.** The reasons:
every individual run is a null, stage 5 still beats the C++ heuristic 1.000, and
the families give the teacher a repertoire the post-deploy-time engine actually
rewards. Against that, the pooled −0.036 is the single thing most worth
re-measuring at higher n before this teacher fronts a long training run. Two
one-line off switches exist and both are tested: `max_combos = 0` (which is
action-identical to the pre-2026-08-20 teacher) and `combo_families` (which
drops a single family).

Latency: **3.0-3.3 -> 4.2-5.5 ms per decision**, i.e. under a second added to a
~22 s episode.

**With `max_combos = 0` the new teacher is ACTION-IDENTICAL to the old one** --
0 mismatches over 919 decisions across 16 matches at stages 2 and 5, compared
against `main`'s teacher on paired snapshots. So the change is purely additive,
and stages 0-1 are bit-identical to what the curriculum had before.

### What this does and does not license

It licenses: the teacher can now *express and use* the play the engine rewards,
which is what item 1 asked for, and it costs ~1.7 ms per decision to do it.

It does **not** license "the teacher is stronger". The win-rate arm is a measured
null, and combos are chosen on well under 1% of decisions because the binding
constraint is no longer the generator but the **economy** — a bot whose bar sits
at p90 3.30 cannot often buy a two-card play. That is now the honest open
question, and it is a scoring question (`w_pos` credits any cheap troop for
standing forward, so every cheap card looks profitable), not a candidate-
generation one.

---

## 2026-08-21: the teacher's ECONOMY. `play_margin` 0.05 -> 3.0, and two mechanisms it needed.

**GAMEPLAY-AFFECTING for phase 1.** Every win rate earned against
`teacher@stage N` before this date describes a bot that dumped its elixir
continuously. Checkpoints are NOT invalidated -- observation, action space,
architecture and reward are untouched.

The multi-card generator (2026-08-20) worked and was never used: combos were
chosen on well under 1% of decisions because the teacher's bar sat at a p90 of
3.30 and an escorted push costs 5-6. This is the follow-through on that.

### `w_pos` is the obvious lever and it is REFUTED

The standing hypothesis was that `w_pos = 20.0` over-rewards cheap units on the
board, so lowering it would make the bot save for a combo. Swept 20 -> 8, n=14
shared openings, stage 5 vs the C++ heuristic:

| `w_pos` | elixir | combos, % of plays |
|---|---|---|
| 20 | 1.83 | 1.1% |
| 16 | 1.83 | **0.0%** |
| 14 | 1.88 | **0.0%** |
| 12 | 2.02 | 0.2% |
| 10 | 2.13 | **0.0%** |
| 8 | 2.67 | 0.2% |

**Half right and half wrong.** The economy half holds -- elixir does rise. The
combo half is backwards: usage never rises and mostly goes to zero.

The mechanism, and it is arithmetic rather than a guess. A play's score is
roughly `w_pos * (HP / MAX_TROOP_HP) * row_weight - w_cost * cost`, so lowering
`w_pos` raises the HP-PER-ELIXIR a play must clear. Ranked:

    Ice Golem  658 HP/elixir     Hog (naked)  424
    escorted push (Skel+Hog) 388     Musketeer 240     Skeletons 243

The escorted push sits **below** the naked Hog, so `w_pos` kills the combo
BEFORE the cheap cards it was supposed to replace. The naked-unit reward and the
combo reward are the same term; that weight cannot separate them.

### The actual defect: `play_margin` was an off switch

`play_margin` is the bot's one economy control -- "a play must beat holding by
more than this" -- and its own docstring says it exists "so rollout noise on a
dead board cannot talk the bot into dumping". It was **0.05**, against a
**measured median of 1.37** for exactly the plays it exists to stop (the plays
actually chosen while the win condition sat in hand and nothing threatened,
measured 2026-08-20). 27x too low to ever bind.

Same shape as `HOG_DEFENSIVE_RESERVE`'s first value of 3.0 opening its gate on
0 of 542 states: a constant chosen on plausibility that turns out to be a no-op.
Third instance in this file.

### THE MEASUREMENT THAT NEARLY SHIPPED A ZERO-GRADIENT ENVIRONMENT

Against the C++ heuristic, a fixed `play_margin = 3.0` looked excellent -- it
beat the shipped profile **0.969 [0.917, 1.000]** head to head. Against a
**passive** opponent it is a disaster, and an episode-0 agent is passive:

| margin (fixed) | plays | tower dmg/match | wins | elixir |
|---|---|---|---|---|
| 0.05 | 39 (14.9%) | 9143 | 6/6 | 2.53 |
| 1.0 | 46 (15.2%) | 9329 | 6/6 | 2.90 |
| 2.0 | 39 (6.0%) | 8324 | 6/6 | 7.29 |
| **3.0** | **14 (0.7%)** | **4063** | **5/6** | **9.56** |

At 3.0 the bot froze at max elixir, threw away almost all income, halved its
tower damage and dropped a match it should win trivially. Matches ran 4x longer
(1880 decisions across 6 vs 261) because it was timing out rather than winning.

**Nothing scores above a fixed high bar when the opponent does nothing** -- the
candidate rollout and the no-op baseline look nearly identical on a quiet board.
So the bar that makes the teacher strong against an active opponent makes it
inert against a weak one, which is precisely the zero-gradient environment the
2026-08-19 curriculum pivot exists to remove. **It would have passed every
benchmark in this repo**, because every one of them uses an active opponent.

**Fixed by tapering the margin to zero across the overflow line**
(`effective_play_margin`), reusing `score`'s own relief -- same
`ELIXIR_OVERFLOW_AT = 9.0`, same shape. Above it the bar is discarding income,
so holding is not free and a marginal play stops having to justify itself. Same
probe, with the taper:

| margin (tapered) | plays | tower dmg/match | wins | elixir |
|---|---|---|---|---|
| 0.05 | 42 (14.8%) | 9319 | 6/6 | 2.63 |
| 1.0 | 51 (14.0%) | 9378 | 6/6 | 4.07 |
| 2.0 | 38 (13.1%) | 9257 | 6/6 | 4.33 |
| **3.0** | **43 (11.9%)** | **9243** | **6/6** | **5.34** |
| 4.0 | 42 (11.8%) | 8937 | 6/6 | 6.18 |

### The second mechanism: a DUE follow-up is not charged the margin

At 3.0 the second half of a committed combo stopped clearing the bar, so plans
were paid for and then abandoned -- the naked-first-card outcome the whole
multi-card change exists to avoid.

`play_margin` stops DUMPING: spending on a marginal play when holding was free.
For a follow-up holding is **not** free -- the first card is already on the board
and already paid for, so declining does not bank the elixir, it wastes the
commitment. `margin_for` returns 0 for `kind == "followup"`.

Narrow by construction, and this is what keeps it from becoming blind
commitment: the follow-up still has to be the ARGMAX over every other
candidate. It skips only the floor whose premise is false. Parallel to
`plan_reserve_penalty`'s exemption for the same step.

### Why 3.0 and not 4.0

Selected by sweep, confirmed on a fresh independent run, both paired on shared
openings within one process:

SELECTION, on the pre-taper code (n=24, so these rank the margin, they are not
the shipped numbers):

| margin | vs the OLD teacher | elixir p90 | overflow | combos, % of plays |
|---|---|---|---|---|
| 0.05 | 0.500 | 3.55 | 0.1% | 2.2% |
| 2.0 | 0.812 | 5.60 | 1.3% | 6.9% |
| **3.0** | **0.969 [0.917, 1.000]** | 7.95 | 4.3% | **14.4%** |
| 4.0 | 0.969 | 9.46 | **12.9%** | 18.2% |

4.0 is not better: identical strength, three times the wasted income. 3.0 is the
knee.

**SHIPPED NUMBERS, re-measured on the final code** (taper + follow-up exemption
in place, n=20 shared openings) -- these are the ones to quote:

| margin | vs heuristic | vs the other | elixir p90 | overflow | plays | combos, % of plays |
|---|---|---|---|---|---|---|
| 0.05 (old) | 1.000 | **0.050** [0.000, 0.113] | 3.50 | 0.0% | 831 | 2.0% |
| **3.0 (new)** | 1.000 | 0.500 (itself) | 7.47 | 2.7% | 443 | **11.3%** |

The old teacher wins **5%** of head-to-head matches against the new one.

**And the cost, stated because a win rate hides it: the teacher plays roughly
HALF as many cards** (831 -> 443 over the same openings). It is much stronger
and much quieter. For a training opponent that is a real trade -- fewer plays
per match is less variety for the student to face -- and it is the thing to
watch if phase 1 ever looks like it has stopped learning. Against a PASSIVE
opponent the taper holds the play rate at 11.9% of decisions against the old
14.9%, so the collapse is bounded; against an active one it is a genuine halving.

Latency also rose **5 -> 14.1 ms/decision**, and not because of the margin: a
solvent teacher can AFFORD more cards, so the generator proposes ~663 combos
per 2,000 decisions where it used to propose ~100, and each one is a rollout.
About 13% of a 22 s episode, up from 4%. `max_combos` bounds it.

### The strength ladder, re-measured with the final code

`prove_teacher.py --vs heuristic --n 20`, every rung, against the C++
HeuristicOpponent at 1.0x:

| stage | before | after |
|---|---|---|
| 0 (rules-only) | 0.500 | 0.325 |
| 1 | 0.600 | **0.750** |
| 2 | 0.750 | **0.900** |
| 3 | 1.000 | 1.000 |
| 4 | 0.975 | **1.000** |
| **5** | 1.000 | **1.000** |

Every rung that uses the ROLLOUT path improved. Stage 0 is the one that fell,
and it is the one rung `play_margin` provably cannot reach -- it takes
`_rules_only`, which ranks by role priority and never reads the margin
(pinned by `test_the_margin_does_not_reach_the_rules_only_rungs`). Its epsilon
is 0.30, its CI is [0.150, 0.500] against the previous [0.325, 0.700], and the
openings differ between runs because the shuffle is unseeded. Noise.

### What the combos actually ARE, which the 14% aggregate hides

Per family, teacher vs teacher, 10 matches -- chosen / completed:

| family | chosen | completed |
|---|---|---|
| `cheap_defence` | 25 | 9 |
| `spell_then_push` | 11 | 6 |
| `defensive_stack` | 8 | 3 |
| `supported_push` | 2 | 0 |
| `counter_push` | 1 | 1 |

**The combos that fire are mostly DEFENSIVE.** The escorted win-condition push
-- the play this whole line of work was aimed at -- is 3 of 47. That is not a
bug and it is not the margin: it needs the tank AND the win condition in hand at
the same time (~4-9% of decisions) and it is the most expensive pair in the
deck. `counter_push` is the same tactic at a lower price and is what actually
lands.

Stated explicitly because this file records the same failure four times
already: **an aggregate that cannot see the conditional.** "14.4% of plays are
combos" is true and would be badly misread as "the Ice Golem + Hog push is now
standard". It is not.

Completion rose 7% -> 40% (19 of 47) once a due follow-up stopped being charged
the dumping margin. Before that fix plans were paid for and abandoned, which is
the naked-first-card outcome the multi-card work exists to avoid.

### The methodological note that made all of this measurable

Every arm above is paired on **shared openings within ONE process**, because the
engine's opening shuffle is unseeded and cross-invocation arm levels are not
comparable (`UPSTREAM_REQUESTS.md` item 7, and the failed control recorded under
the multi-card section). `prove_combos.py --profile-sweep` builds its root envs
once and hands every arm a `snapshot()`, and carries a self-check: the shipped
value must score exactly 0.500 against itself. It does.

**That self-check also caught a stale constant.** The harness had its own copy
of the default margin, which silently became wrong the moment the default moved
0.05 -> 3.0 -- so the sweep labelled its rows against a baseline that was no
longer the baseline. It now reads the value off `UtilityTeacher`. Same rule
CLAUDE.md already states for engine constants, and the third time a second copy
has gone stale here.

---

## 2026-08-21: REACTIVE ROLLOUTS. The rollout opponent stopped standing still.

**GAMEPLAY-AFFECTING for phase 1.** Every win rate earned against
`teacher@stage N` before this date describes a bot whose scorer was blind to
being answered. Checkpoints are NOT invalidated -- observation, action space,
architecture and reward are untouched.

`rollout_stats` rolled every candidate forward with BOTH SIDES NO-OPING, so a
naked bridge push was scored against an opponent who never dropped a Cannon.
`UtilityTeacher.counter_schedule` / `counter_action` now place a defensive
answer inside the rollout; `reactive` is a fourth axis in `TEACHER_STAGES`
(**stage 5 only** -- see the horizon gate below) and `reactive_rollout=False`
is the one-line off switch.

### The bias was real, large and ONE-DIRECTIONAL

Ground truth = a full stage-5 `UtilityTeacher` playing the other side, which is
a DIFFERENT policy from any candidate responder, so the comparison is not
circular. Over 150 naked bridge pushes, paired on the same snapshots:

| | no-op rollout predicts | truth | bias |
|---|---|---|---|
| tower damage dealt | 587.5 | 139.5 | **+448.0** |
| elixir lost | 0.09 | 2.84 | -2.75 |
| **says PLAY** | **90.7%** | **7.3%** | **125 false GO / 0 false HOLD** |

Reacting cuts the tower-damage bias to +19.0 (MAE 448 -> 108; paired -340.3,
CI [-395.2, -285.3], 94 closer / 9 further, p = 5.5e-19).

### WHAT IT FIXES IS NARROWER THAN IT SOUNDS, and the headline metric was CONFOUNDED

Overall argmax agreement with truth goes 36.4% -> 52.7%, and **that number is
not the result.** Truth holds on 45% of decisions and the responder holds on
78%, so declining more scores on it for free. Decomposed over 110 decisions
(truth plays 60, holds 50):

| estimator | hold+hold | SAME PLAY | missed | false GO | **P(same \| truth plays)** |
|---|---|---|---|---|---|
| no-op | 23 | 17 | 26 | **27** | **28.3%** |
| reflex | 46 | 12 | 40 | **4** | 20.0% |
| reflex, bar 0 | 17 | 17 | 17 | 33 | 28.3% |
| rules (attacking) | 45 | 11 | 38 | 5 | 18.3% |
| rules, bar 0 | 28 | 17 | 23 | 22 | 28.3% |

**Every estimator tops out at exactly 28.3% -- none ever beats the BLIND one at
picking WHICH play to make**, at any responder fidelity or any margin. What is
bought is `false GO 27 -> 4` and win-condition plays 8 -> 3. Two repairs of the
resulting passivity were tried and both traded quality for play-rate 1:1: an
answer budget (rate 21.8% -> 49.1%, rho +0.292 -> +0.007) and re-sweeping
`play_margin` (rate restored, agreement falls with it).

What DOES improve is the RANKING: rho vs truth -0.003 (no-op) -> +0.292
(reflex, p = 0.0022) -> +0.434 (rules, p = 0.00026). **The shipped scorer's
ordering over candidates was uncorrelated with what a real opponent would make
of them.**

### The responder is OPEN-LOOP because the cost knee said so

Four responders, paired win rate, 150 seeded openings each, sides swapped,
control `noop vs noop == 0.5000` exactly:

| arm | cost/decision | delta vs no-op | 95% CI | p |
|---|---|---|---|---|
| **scripted (open-loop)** | **0.91x** | **+0.1583** | [+0.1033, +0.2117] | 1.9e-07 |
| reflex, stride 50 | 1.95x | +0.2050 | [+0.1500, +0.2567] | 3.3e-11 |
| reflex, stride 30 | 2.63x | +0.2450 | [+0.1933, +0.2933] | 1.4e-15 |
| reflex, stride 10 | 5.50x | +0.2217 | [+0.1667, +0.2733] | 4.4e-12 |

**All four beat the control. Paired ARM vs ARM on the same openings, all SIX
comparisons are NULL** (p from 0.0857 to 0.832) -- ~60 of 150 openings tie, so
only ~88 pairs carry signal. The decision therefore collapsed to cost and the
cheapest arm won. **Four overlapping marginal CIs are not a ranking**; the
point estimates were read as one three separate times during this work and the
pairwise test refuted each reading.

**Note the NON-MONOTONICITY**, since it is why "more is better" was rejected:
stride 10 costs 2.1x stride 30 and scores BELOW it. Deciding every chunk makes
the rollout opponent superhuman -- observed dumping Skeletons, Ice Spirit,
Cannon AND Ice Golem onto one Hog. Same shape as the neural search measuring
horizon 20 worse than 12, and the teacher's ladder topping out at 10 s.

**The numpy-observation binding was proposed and then NOT filed.** 81% of a
CLOSED-LOOP responder's cost is marshalling 13,606 floats across pybind
(0.611 ms of a 0.752 ms decision, against a 0.0023 ms memcpy floor). The
open-loop responder reads no observation at all, so the binding buys nothing
here. Do not re-propose it for this reason.

### Verified on the SHIPPED code, not on the prototype

TDD had the feature implemented fresh from tests, so the production class is
not the measured prototype and was re-measured rather than assumed:

    reactive ON vs OFF   0.6500   +0.1500  [+0.0896, +0.2104]  p = 1.41e-05
                                  52 better / 16 worse / 52 level

Cost, production class: **1.10x per decision**, measured interleaved,
min-of-repeats, over 60 shared states. **Do not quote a per-MATCH figure from
that harness** -- it times a 20-opening control against a 100-opening treatment,
so its ratio moved 1.71x / 0.90x / 1.34x across three runs of the same code and
is measuring the sample, not the arm. The per-decision number is the one taken
under a controlled protocol.

### THE HORIZON GATE, found in pre-merge verification and not before

The first wiring turned `reactive` on at stages 2-5 on a cold-start argument.
That is measured WRONG, and the mechanism is an asymmetry between when the two
halves of an exchange land: **the counter is charged at +10 ticks, while the
attack's payoff needs ~130** (a Hog crossing ~12 tiles at Fast speed). A rollout
shorter than the crossing therefore charges the answer in full and credits none
of the push. Against a PASSIVE opponent -- which is exactly what an episode-0
agent is -- 20 seeded openings, share of decisions that landed a card, and how
many openings froze to under 5 plays in 120 decisions:

| horizon | reactive OFF | ON | froze |
|---|---|---|---|
| 30 | 12.2% | 9.8% | **3/20** |
| 50 | 11.6% | 10.1% | **3/20** |
| 70 | 12.5% | 11.3% | 1/20 |
| **100** | 11.8% | **12.3%** | **0/20** |

So enabling it below 100 ships the zero-gradient failure the 2026-08-19
curriculum pivot exists to remove. `COUNTER_MIN_HORIZON_TICKS = 100` gates it,
and only stage 5 clears the gate -- which is also the only rung the +0.1500 was
ever measured at. After the gate, horizon 30 returns to min 6 / median 9 plays
and 0/25 frozen, identical to reactive OFF.

**How it surfaced is the transferable part.** It did not appear in any
aggregate: the stage-5 passive probe reads 12.3% both ways, and the win-rate
A/B is run at stage 5 where the bias is gone. It surfaced as a FLAKY TEST --
`test_teacher_is_side_agnostic`, which happens to drive a horizon-30 teacher
against a do-nothing opponent, i.e. precisely the unmeasured corner. Ninth
instance of this file's recurring lesson: an aggregate cannot see a conditional,
and here the conditional was the horizon.

**Combos are REINFORCED, not suppressed**, which was worth checking because an
escorted push draws two counters where a naked one draws one. Measured over 12
seeded matches: combo share of plays **10.0% -> 14.3%**. The naked push is
penalised harder than the escorted one, so the escort becomes relatively more
attractive -- the direction the deploy-time physics say it should go.

### Two things found on the way

**`test_the_teacher_actually_executes_a_planned_pair_end_to_end` was FLAKY and
had been all along.** `test_teacher_combos._env()` called a bare `reset()`, so
team 1's hand and the whole 40-tick warm-up came from `std::random_device` and
every invocation staged a different position. Invisible while the scorer was
lenient; a 1-in-3 flake once reactive rollouts tightened the margins. Now
seeded -- the staged state picks a combo in 33 of the first 40 seeds, so seed 0
is not cherry-picked.

**`env.seed()` works and this file was stale about it.** It seeds BOTH
`std::mt19937`s and re-deals; `UPSTREAM_REQUESTS.md` item 7 is DONE. Arm levels
reproduced to four decimals across two independent invocations of the win-rate
harness. The old rule -- "across runs only DELTAS are comparable, never arm
levels" -- no longer holds for a harness that calls `seed()`. Most `prove_*.py`
harnesses still do not.

---


## 2026-08-20: the simulator audit. TWO absorbing states, one of them fixed.

Opened on a report that "tanks and win conditions lag or get stuck on the
bridges" before starting an AlphaZero run. The report was **correct and
understated**, and chasing it turned up a second, unrelated defect that was
quietly worth more.

Instruments live in `tools/audit/` (standalone, compiled against the header-only
engine by `tools/audit/build.ps1`, deliberately NOT CMake targets so they cannot
perturb the generated solution the `.pyd` and the Catch2 suite build from).

### 1. The bridge EXIT trap. FIXED. GAMEPLAY-AFFECTING.

`Board::getNextWaypoint` had a second absorbing state, one branch away from the
one fixed on 2026-08-09 and with exactly the same shape.

The 2026-08-09 fix guarded the two branches where a unit is standing on the bank
it is LEAVING (`isCurrentBelow` / `isCurrentAbove`). It did not guard the branch
where the unit is INSIDE the river band and within epsilon of the bank it is
ARRIVING at. That branch returned `{bridgeX, riverY_end}` with no arrival check,
so a step landing at y = 17.4995 was handed (bridgeX, 17.5), refused to move
because 0.0005 <= `WAYPOINT_ARRIVAL_EPS`, and never moved again. **Four exit
traps, the mirror image of the four entry traps.**

Measured with `tools/audit/bridge_audit.cpp`, a per-tick trajectory sweep of 306
lone crossings — the Catch2 suite asserts on END STATES and the symptom is a
property of the TRAJECTORY, which is why nothing caught it:

| card | crossed before | after | longest stall before | after |
|---|---|---|---|---|
| Giant | 27/34 | **34/34** | 756 | 10 |
| Musketeer | 26/34 | **34/34** | 813 | 10 |
| Valkyrie | 26/34 | **34/34** | 813 | 10 |
| Mini PEKKA | 28/34 | **34/34** | 843 | 10 |
| Ice Golem | 28/34 | **34/34** | 796 | 10 |
| Skeletons | 101/102 | **102/102** | 844 | 9 |
| Hog Rider | 34/34 | 34/34 | 10 | 10 |
| Ice Spirit / Minions | all | all | 10 / 9 | 10 / 9 |

**~20% of lone ground crossings never completed at all.** The residual stall of
10 ticks is `DEPLOY_TIME_TICKS` and is correct. After the fix, an analytic sweep
of **8,661,439 board positions** against 8 destinations finds **zero** absorbing
states anywhere on the board (`tools/audit/waypoint_probe.cpp`).

Three things worth carrying:

- **The existing regression test could not see it, and the reason is precise.**
  `test_board.cpp`'s sweep pairs each bank with the one direction in which that
  bank is the ENTRY — near bank against a northern target, far bank against a
  southern one. The trap lives in the other two combinations. It tested exactly
  the complement of where the bug was. The new sweep is the full cross product.
- **Speed determined who it hit, which made it look card-specific.** The chance
  a step lands in a 0.01 disc is about `0.01 / step size`, so slow tanks were
  worst (Giant 0.06/tick) and fast cards escaped (Hog 0.16). That is why the
  report named tanks and win conditions.
- **A CROWD hides it.** 45 Skeletons, 10 Hogs and 6 Giants all crossed fine in
  the crowd sweep, because collision jostling knocks units out of the trap. It
  is a LONE-unit bug — which is precisely the "send the Hog to the bridge" case.

### 2. Sight and attack range were measured differently. FIXED. GAMEPLAY-AFFECTING, and the bigger of the two.

`findTarget` gated candidates on `dist <= sightRange`, a RAW centre-to-centre
distance. Attacking gated on `effectiveRangeTo() = attackRange + own radius +
target radius`. **Two conventions for the same geometric question**, so between
them lay a band in which an attacker could hit something it could not SEE — and
therefore never acquired, and stood idle.

A Princess Tower has `attackRange == sightRange == 7.5` and radius 1.5, so it
reached a troop at 9.4 but saw one only within 7.5. A Musketeer stops at her own
effective range of `6.0 + 0.4 + 1.5 = 7.9` — **inside that band every time.**

| Musketeer placed at | tower damage dealt | damage she took |
|---|---|---|
| 6.5 – 7.5 tiles | 1519 | 721 (dies) |
| **8.0 tiles** | **5355** | **0** |
| 10.5 tiles | 4704 | **0** |

A Princess Tower has 3204 hp. So one 4-elixir card removed a tower and started on
the next **without taking a scratch**, from any placement at 8+ tiles. Fixed by
adding `effectiveSightTo()` alongside `effectiveRangeTo()` and using it at the
two comparison sites (`CombatEntity::findTarget`, `BuildingTargeter::findTarget`).

The invariant, now stated in the code: **as long as `sightRange >= attackRange`,
effective sight >= effective attack range, so nothing can ever attack what it
cannot see.** `sightRange`'s own comment already gave buildings
`sightRange == attackRange` on the reasoning that "sight beyond attack range
would never actually matter" — that reasoning is right, and this is what makes
it true. Equal NUMBERS are not equal RANGES when one is measured
surface-to-surface and the other centre-to-centre.

**Measured impact, controlled A/B** (the fix stashed and restored, everything
else identical; lone attacker at the bridge, 600 ticks):

| | tower damage before | after |
|---|---|---|
| Hog Rider | 2534 | **1268** |
| Musketeer | 6542 (survived) | **1302** (dies) |
| Ice Golem | 336 | **84** |

**Defence got materially stronger, and part of that is a known divergence being
amplified — say so when quoting these.** Damage attribution against a lone Hog,
read off `MatchStatistics` by the reserved tower cardIds:

| | King | Princess | King's share |
|---|---|---|---|
| before | 90 | 1620 | 5.3% |
| after | 630 | 1080 | **36.8%** |

The King Tower in this engine **never sleeps** (`Tower.h`, no activation
condition — long-standing, item 3 in `UPSTREAM_REQUESTS.md`). Widening its
effective sight from 7.0 to 9.4 lets it join fights it previously sat out, so
this fix makes an existing fidelity gap bite harder. The fix is right on its own
terms; the interaction is real and is the thing to watch.

**Every win rate earned before this is historical.** Checkpoints are NOT
invalidated — no observation, action-space or architecture change.

One existing test moved and it is worth knowing why:
`test_combat_entity.cpp`'s "never picks an enemy beyond sightRange" used a
distance of 6.0 against the default 5.5, chosen when sight was centre-to-centre.
Surface-to-surface, 5.5 covers 6.3 between two troops, so the constant moved to
7.0. **The invariant it protects is unchanged**; only the convention it was
written against was corrected, and a positive companion case was added.

### 3. The two-obstacle collision wedge. NOT FIXED, deliberately, and pinned.

A unit pinched in the concave pocket between two buildings stops permanently.
`Board::pushAwayFrom`'s perpendicular slide exists to stop a unit sticking on ONE
obstacle; with two, the slides can oppose and cancel, and the post-move
`resolveCollisions` pass returns the unit to where it started. It is an
**attracting** fixed point — the approach converges geometrically.

Measured over 60 randomized full matches, 342,563 unit-ticks
(`tools/audit/soak.cpp`): **7 stalls, none on or near a bridge** (0.002% of
unit-ticks), every one in a player's own back corner pinched between a friendly
tower and either a second building or the board edge; longest ~490 ticks, i.e.
until the match ended. It read 4 per 308,464 before the sight/attack fix in the
same audit changed engagement geometry -- re-measured rather than carried over.

**Two local fixes were implemented and measured, and both only MOVED the
equilibrium** — 0.027 tiles per 120 ticks for a timer-flipped tangential slide,
0.000001 tiles for a geometry-chosen wall slide, which settled at a new fixed
point. Both reverted. The geometry says why no local rule suffices: in the
measured case the two obstacles' minimum separations sum to 3.8 while their
centres are 3.46 apart, so **there is no route between them at all** and escape
needs a multi-tile detour, i.e. global planning.

**ACCEPTED AS A KNOWN DEFECT, 2026-08-20** — signed off rather than fixed,
because global path planning is too expensive for rollout throughput at present
and the defect is rare, isolated and never on a bridge. Reopen it if the rate
rises, if a stall is ever seen near a bridge, or if throughput stops being the
binding constraint. Written up as `UPSTREAM_REQUESTS.md` item 18, because the
real fix is a flow field or A* over the 18x34 grid — a redesign of the movement
core, gameplay-affecting for every unit. `tests/core/test_navigation_wedge.cpp`
reproduces it deterministically and is tagged **`[!shouldfail]`**: the suite
stays green, the defect stays executable, and the case turns RED the moment
somebody fixes it.

### 4. DEFAULT_DECK behavioural QA

`tests/core/test_default_deck_qa.cpp` (new, 13 cases) pins what the eight cards
DO, not just which they are — `test_card_registry.cpp` already covers identity.
Every assertion was measured with `tools/audit/deck_audit.cpp` first.

Card stats read off spawned entities, all matching the real game closely:
Hog 1697hp/317dmg/1.6s/Fast, Musketeer 721/217/1.0s/range 6, Cannon
824/202/range 5.5, Ice Golem 1315/84/2.5s/Slow, Skeletons 3 bodies at 81/81,
Ice Spirit 230/110/range 2.5. Air targeting is correct for all eight (Musketeer,
Ice Spirit and Fireball hit air; Hog, Cannon, Ice Golem, Skeletons and The Log
do not). Defence against a lone Hog: **a Cannon prevents all 2534 tower hp**,
Skeletons and Musketeer prevent 2217, and a Hog "answering" a Hog prevents 315 —
which is our own tower shooting, not the Hog defending.

**Three of these tests were wrong before they were right, and all three failed
the same way: attributing to the card something the towers did.**

- The air probe reported that The Log, the Hog and the Cannon all "hit air".
  The Minions were flying into our own towers. Same trap CLAUDE.md already
  records for `get_troop_damage_dealt`. Fixed by differencing against a
  no-card control — and then the control SATURATED (minions dead in both arms,
  every card differencing to 0), which needed a placement 11 tiles from every
  tower and a short window. The final Catch2 form abandons the scenario
  entirely for a bare board and a stationary flying dummy.
- The Hog-ignores-troops test first required the Hog to SURVIVE 120 ticks.
  Three Skeletons are ~220 dps and a Princess Tower another 382; a 1697 hp Hog
  is dead in under three seconds. Survival was never the property worth
  pinning — "deals zero damage to them, by card" is.
- A bare-`Board` harness that never called `commitPendingEntities` reported the
  Ice Spirit as unable to hit air. `addEntity` queues, so a RangedTroop's
  projectile was created and never entered the world.

### 5. Latency: no problem found

`tools/audit/bridge_audit.cpp latency`: **0.0023 ms/tick at 1 unit, 0.0171 at
30.** Pathfinding is not a cost centre and never was; the "lag" in the report
was the deadlock, not slowness.

### Suite counts after this work

**C++ 582 cases / 5,737 assertions** (was 550 / 5,341), of which 1 is the
`[!shouldfail]` wedge. **Python 366 passed / 2 skipped. Perception 353 passed /
1 skipped.** (Superseded by the 2026-08-21 section below: 619 / 5,907, Python
367 / 2.)

---

## 2026-08-21: four fidelity fixes from a professional player's audit

**GAMEPLAY-AFFECTING (items 1-3). Every win rate, Elo figure and placement score
in this file is now historical.** Checkpoints still LOAD — no observation,
action-space, reward or architecture change — but their measured strength no
longer means anything. The `phase1_v5` run in progress when this started was
stopped for exactly that reason.

**Two of the four requested items turned out to be already implemented**, and
the investigation is recorded because assuming otherwise would have wasted a
day. Rule A (strict sight) already shipped: 58 per-card `withSightRange` values
matching the player's catalog, filtered through `effectiveSightTo`. And
`TimeoutRules` already implemented both tiebreakers and was wired into the
reward path — measured with `tools/audit/timeout_audit.cpp`, both sides passive
reaches tick 3600 and resolves, and against the heuristic 0 of 6 matches even
reach the limit.

### 1. The arena's coordinates

See "Board geometry" above for the layout and the convention error behind it.
The short version: x is a cell index, so the centre is 8.5, not 9.0; and a
two-tile bridge is centred on the seam between its tiles, not on a tile.
Confirmed against the player's own map of the river row before anything else was
touched, using `tools/audit/board_map.cpp`, which renders the arena FROM THE
LIVE ENGINE rather than from a hardcoded copy.

### 2. King Tower activation

See "The King Tower sleeps" above. Largest gameplay effect of the four.

### 3. Blind lane pathing — Rule B, the only genuinely missing behaviour

`findTarget`'s fallback with nothing in sight was *closest enemy Tower by raw
distance*. With one enemy Princess destroyed that sends a unit diagonally across
the arena to the OTHER lane's Princess. `include/core/LanePath.h` replaces it:
the objective is **my own lane's** enemy Princess if alive, else the enemy King,
and a King objective is APPROACHED up the lane (via the empty Princess slot)
rather than cut diagonally from the bridge.

Lane is nearest-bridge re-evaluated per call — the same rule `getNextWaypoint`
uses to pick a crossing, so objective and crossing agree by construction and
nothing new is carried through `deepCopy`/`snapshot`. Shared by `CombatEntity`
and `BuildingTargeter` so a building-targeter cannot disagree with a troop about
where its lane leads.

**THE REPRODUCTION IS POSITION-DEPENDENT, and that is the part to remember.**
With team 1's left Princess dead, a unit in the left lane measures:

| unit at | → right Princess | → King | closest-tower picks |
|---|---|---|---|
| (2.5, 12.0) | 18.90 | 19.45 | **right Princess — wrong** |
| (2.5, 14.0) | 17.36 | 17.56 | **right Princess — wrong** |
| (2.5, 15.0) | 16.62 | 16.62 | tie |
| (2.5, 16.5) — the bridge | 15.57 | 15.23 | King — *right, by accident* |

Crossover at y ≈ 15.0. **At the bridge mouth the broken rule already answers
correctly**, so a test written there passes against unfixed code and proves
nothing — the same "cross-check anchored where the error is zero" trap as the
2026-08-05 tile-grid refit. `test_lane_pathing.cpp` anchors at y = 13.0 and
pins BOTH halves of that table so the anchor cannot silently drift back.

Blast radius is smaller than it sounds: with both Princesses alive, "my lane's
Princess" and "closest tower" agree. Behaviour only diverges once one is down.

### 4. The timeout verdict — a CONSUMER defect, not a rule defect

`web/viewer.html` re-derived the outcome from "are both King Towers alive?" and
called everything else a draw, so a timed-out match with a badly damaged
Princess displayed **"Draw. Timeout — both King Towers still standing"**. Its
comment said it "mirrors `MatchRules::evaluate` exactly" — which was TRUE and was
the bug: `MatchRules` answers *"has a King died yet?"* and correctly says no
right up to the limit. Who WON at the limit is `TimeoutRules`.

`TimeoutRules::decide()` now splits the RULE from the data gathering, so
`GameLogger` (const, holds no Board) can reach the same verdict from its own tick
snapshots instead of reimplementing it. The replay JSON carries
`"result": {loserTeam, timedOut, reason}` and the viewer reads it, with a
fallback for older replays that is a PORT of TimeoutRules rather than the
king-alive shortcut.

**The Python guard against this exact pattern walks `.py` files only, which is
precisely why it never saw a `.html` file.** It now checks the viewer directly.

Absolute HP, not percentage — the real game breaks this tie on fraction, and
King 4008 vs Princess 2534 makes the two disagree often. A deliberate, known
divergence, specified by the audit.

### 5. The observation encoder was NOT updated with the arena, and that is the
### most dangerous bug of the batch

Found by another session's RL tests after everything above had shipped.
`ClashEnv::extractObservationForTeam` painted observation **channel 8** -- the
river/bridge mask, the only thing telling the network where it can cross --
from a hardcoded `(x >= 3 && x <= 4) || (x >= 13 && x <= 14)`, while the physics
used `Board`'s bridges. The arena correction moved the bridges and only the
physics followed:

```
column        012345678901234567
physics       WWBBWWWWWWWWWWBBWW
old encoder   WWWBBWWWWWWWWBBWWW
mismatch        ^ ^        ^ ^
```

**Four of the eighteen columns were wrong, in BOTH directions**: columns 2 and
15 are real bridge and were shown as water; 4 and 13 are water and were shown as
bridge. So the agent's map of where it could cross was half wrong, on every
observation of every tick of every episode -- while the entire C++ suite stayed
green, because nothing compared that channel against the movement rule it is
supposed to describe.

Fixed by `Board::isOnBridge(float x)`, called from BOTH `clampToBoard` and the
encoder. **A shared formula would not have prevented this; a shared function
does** -- the original bug is precisely two correct-looking expressions of the
same question drifting apart. It takes a float so one function serves the
continuous positions physics uses and the integer cell centres the encoder uses;
cell `i` covers `[i-0.5, i+0.5]`, so asking about cell centre `i` against a
seam-centred 2.5 selects exactly cells 2 and 3.

The regression test compares channel 8 against **`clampToBoard`**, for both
teams, rather than against expected columns -- a test pinning the columns would
need hand-editing on the next arena change and would go stale exactly the way
the encoder did. `verify_pyd.py` checks the same row, because the `.pyd` is what
TRAINING loads and a stale one is how a fixed engine still trains wrong.

**The general lesson, and it is the fourth instance in this file:** when the
same fact is encoded for the SIMULATION and for the OBSERVATION, they are two
copies, and the observation copy is the one no test looks at. The team-1 row
displacement (2026-07-31), the river marker row, and this are the same failure.

### What the audit turned up that nobody asked for

**Two more cards had the 2026-08-20 sight/attack dead band.** Pinning the
catalog was meant to be a tests-only task; sweeping the invariant "no card can
attack further than it can see" across all 132 playable cards found **Bomb
Tower** and **Three Musketeers**, both with `attackRange` 6.0 against the 5.5
default sight. Both could hit what they could not see, so they never acquired it
and stood idle. Neither is in `DEFAULT_DECK` — which is why 582 passing tests
never noticed — but both appear in random-deck opponents, so phase 1's
`random_opponent` stage has been training against two broken cards.

Two things about that test worth copying: it reads `sightRange` off the SPAWNED
ENTITY rather than the registry literal, so it also proves `applyCardMetadata`
copies the value across; and it collects EVERY offender before asserting. The
first run reported only the Bomb Tower and looked like one isolated bug — Three
Musketeers appeared only after that was fixed.

**The `[!shouldfail]` collision-wedge case stopped failing**, which reads like
the defect being fixed and is not: one wall of the measured pocket was the King
Tower, and the King moved. `pushAwayFrom`'s opposing-slide cancellation is
untouched. Re-anchored by translating the pocket by the same −0.5;
`tools/audit/soak.cpp` re-measured **3 stalls per 363,249 unit-ticks, none on or
near a bridge** against 7 per 342,563 before. Same defect, same kind.
UPSTREAM item 18 stands.

### Absorbing states: the bar this work had to clear

Lane pathing adds a new intermediate waypoint, which is exactly how both of this
engine's shipped deadlocks were born. Four guards, all measured after the change:

| instrument | result |
|---|---|
| `waypoint_probe` original sweep | **0** absorbing / 8,661,439 positions |
| `waypoint_probe` lane composition | **0** absorbing / 2,584,034 positions, under 3 tower configurations |
| `bridge_audit` | **34/34** crossings every card (squads 102/102), both directions, longest stall == `DEPLOY_TIME_TICKS` |
| `soak` | 3 / 363,249 unit-ticks, none near a bridge |

The composition sweep runs with all towers alive AND with each enemy Princess
dead in turn, because with both alive `approachPoint` is the identity and a
sweep of only that case measures nothing while looking exhaustive.

### Suite counts after this work

**C++ 622 cases / 5,962 assertions**, of which 1 is the `[!shouldfail]` wedge;
runner exits 0. **Python 369 collected (366-367 pass, 2-3 skipped -- the skip
count varies with the unseeded opening-hand shuffle). Perception 353 passed /
1 skipped.**

---

## 2026-08-24: the placement head stopped computing rows nobody reads. 1.52x on the update.

**NOT gameplay-affecting in the usual sense.** The loss function is unchanged,
and its VALUE is bit-identical from identical weights -- every field of
`UpdateStats` compares exactly. Checkpoints load and behave identically; no
observation, action-space, reward or architecture change. **But the weight
TRAJECTORY does not reproduce across this commit**, and half this section is
about why that is unavoidable rather than a defect.

### The premise that was wrong, and it would have caused a silent regression

The change was proposed as: the coverage forward computes a 500-row placement
map while `Advisor/Rows` sits at 23-25, so ~95% is waste. **That reading is
wrong.** `advisor_target.coverage_terms` splits the rows TWO ways:

    has  = has_target * decision          ~23-25 rows -> KL to the advisor
    no_t = (1 - has_target) * decision    every OTHER decision row -> ENTROPY

at a live `PLACEMENT_COVERAGE_COEF = 0.02`. The advisor row count bounds the KL
half ONLY. Slicing the forward to those rows would delete the entropy half and
re-open the 2026-08-14 placement collapse (Cannon 91.0% modal share on (11,0),
121 tower HP preserved against 396 for a RANDOM legal cell -- worse than
chance). A win-rate arm would have caught that eventually; nothing in the update
itself would have complained at all.

### What IS dead: 63.2% of BOTH placement forwards

Every consumer of both placement maps is masked by `decision` -- `actor_loss` by
`mb_decision`, placement entropy by `mb_placed` (a subset), `clip_frac` by
`mb_decision`, and both halves of `coverage_terms` by `decision`. And `decision`
is itself `(card_mask.sum(1) > 1) * valid`, so it is a subset of `valid` and
nothing else can reach those rows either.

Measured on `model_weights_selfplay.pth` over 1500 steps: **decision fires on
0.368 of rows.** So 63.2% of both forwards, and both backwards, was being
multiplied by exactly zero.

`net.forward_sequence` grew an optional `active_rows`; `PPOUpdater` passes the
decision rows. `active_rows=None` reproduces the old path exactly, so no other
caller is affected. Two details are load-bearing:

- **The filler on skipped rows is ZERO, not `-inf`.** An all-`-inf` row makes
  `Categorical.entropy()` return `nan`, and `nan * 0.0` is `nan`, which would
  poison every masked sum in the update. Any FINITE filler gives
  `finite * 0.0 == 0.0` exactly -- which is what the old path produced there --
  so every masked reduction keeps its shape, order and values, and the loss
  comes out bit-identical.
- **The two `place_ctx` Linears are computed on the FULL batch and sliced.**
  GEMM is not batch-size invariant on this backend (`Linear(280->32)`: 4.768e-07
  forward, 2.289e-05 on `grad_W`, batch 500 vs 167). They are ~0.5% of the
  head's cost, so computing them full is nearly free, and it is what makes the
  LOGITS on the kept rows bit-identical.

**Measured, interleaved arms, min-of-repeats, one shared batch, production
config (500x8, bptt 25, 8 minibatches, 4 epochs), decision rate 0.363:**

| | s/update |
|---|---|
| baseline (full rows) | 27.38 |
| **compacted** | **17.99** |

**1.522x, 9.39 s/update, 34.3% of the update.** The 27.38 s baseline reproduces
the 27.3 s this file already records, which is the cross-check that makes the
ratio quotable rather than just plausible.

### Bit-exactness at the WEIGHT level is unavailable, and that is a backend fact

It was the goal, and no row-compaction scheme can meet it. **`Conv2d`'s weight
gradient is a reduction over the batch dimension, and MKL-DNN re-blocks that
reduction when the batch size changes -- for SOME shapes and not others.**
Measured 500 -> 184, dropped rows carrying exactly-zero upstream gradient:

| layer | shape | grad_W |
|---|---|---|
| `place_up.1` Conv2d(32,16) | 18x10 | **bit-identical** |
| `place_up.4` Conv2d(16,8) | 36x20 | differs 5.814e-03 |
| `place_up.6` Conv2d(8,1) | 36x20 | differs 2.808e-03 |
| `place_hires.0` Conv2d(24,8) | 34x18 | differs 4.883e-03 |

**The shape-dependence is the trap.** The first probe tested only 18x10, got
"invariant", and that claim was written into a code comment before a sweep over
the other three shapes refuted it. An invariance that holds for one layer is not
a property of `Conv2d`. Same standing lesson this file already records for
environment claims: **state the probe, not the conclusion** -- and check the
edges of the range, not one point in it.

### The drift COMPOUNDS -- and a semantic no-op does exactly the same thing

Asked whether the round-off stays bounded, it does not. Two arms, same batch,
same seed, same minibatch permutation, `drift = max|w_A - w_B|` against
`signal = max|w_A - w_0|`:

| update | drift | signal | drift/signal |
|---|---|---|---|
| 1 | 1.038e-03 | 2.371e-03 | 0.438 |
| 10 | 6.340e-03 | 2.500e-02 | 0.254 |
| 50 | 1.308e-01 | 1.348e-01 | **0.970** |
| 100 | 2.573e-01 | 2.567e-01 | **1.002** |
| 200 | 5.180e-01 | 5.456e-01 | 0.949 |

By update ~50 the arms differ by as much as either has moved from init: in
weight space they are DIFFERENT RUNS. The mechanism is Adam, not the
convolution -- Adam's step is `lr * m_hat / sqrt(v_hat)`, which at the first
step is `lr * sign(g)` for any `|g| >> eps`, so a round-off difference in a
near-zero gradient becomes a full +/- `lr` step difference immediately. With
`lr = 3e-4` and 8 optimizer steps per update that predicts ~1e-3 after one
update; observed 1.038e-03.

**THE CONTROL IS WHAT MAKES THAT NUMBER READABLE, and without it the result is
alarming for no reason.** Four arms, same protocol: A uncompacted; B uncompacted
with ONE weight nudged by a single ULP; C compacted; **D uncompacted with the
placement head run in TWO HALF-BATCHES and concatenated** -- semantically a
no-op, every row seeing identical weights and inputs, with the GEMM half
controlled for so that only the conv's blocking varies.

| update | A-B (1 ULP) | A-C (compaction) | A-D (semantic no-op) |
|---|---|---|---|
| 1 | 5.821e-11 | 1.038e-03 | 5.316e-04 |
| 2 | 5.821e-11 | 2.058e-03 | 9.936e-04 |
| 5 | 5.821e-11 | 1.904e-03 | 1.213e-03 |
| 10 | 5.821e-11 | 6.340e-03 | 2.394e-03 |
| 35 | 5.821e-11 | 7.245e-02 | 5.176e-02 |
| **50** | **5.821e-11** | **1.308e-01** (r 0.970) | **6.826e-02** (r 0.506) |

**Arm D tracks arm C within a factor of 2 at every milestone, and is ~9 orders
of magnitude above arm B.** Splitting a batch in half and concatenating cannot
change what is computed, so the divergence is attributable to floating-point
accumulation order ALONE, and compaction is not doing anything wrong. Note also
that arm B does NOT amplify: a 1-ULP WEIGHT perturbation never flips a
gradient's sign, so this trainer is not chaotic under just any nudge -- it is
specifically sensitive to GRADIENT perturbations, which is exactly what Adam's
sign-like first step implies. That asymmetry is why arm B alone would have been
a misleading control and arm D was necessary.

A free cross-check fell out of running the two experiments independently: arm C
measured 7.245e-02 at update 35 and 1.308e-01 at update 50 in BOTH invocations,
to every printed digit. The compacted path is deterministic run-to-run, which is
the property `test_the_compacted_path_is_REPRODUCIBLE_run_to_run` pins at small
scale and this confirms at 50 updates.

Provenance, since the two tables stop at different points: the 2-arm drift run
completed all 200 updates (2630 s); the 4-arm control was killed by its own
`timeout` after update 50 while the full pytest suite was competing for the same
8 threads. Update 50 is past the saturation knee in the 2-arm table, so nothing
in the conclusion depends on the missing rows -- but the control has NOT been
run to 200 and should not be quoted as though it had.

**The consequence for how to read this repo's history: trajectory identity was
never available to this trainer under ANY reordering, including the
`forward_sequence` batching already shipped** (recorded above as "max logit
delta 1.1e-08 vs a float32 eps of 1.19e-07" -- a single-forward comparison that
was never run out to 200 updates, and would have shown the same thing). The
achievable bar, and the one `tests/test_rl_ppo_compaction.py` now enforces:

| claim | status |
|---|---|
| placement logits on kept rows | **`torch.equal`** |
| loss + every `UpdateStats` field, from identical weights | **exact** |
| the compacted path re-run on the same input | **bit-identical** |
| weight trajectory across the commit | **does not reproduce** |

That last row is a ONE-TIME discontinuity, not ongoing noise: `active_rows` is
derived from the stored `decision` column, so batch sizes are fixed for a given
batch and permutation and every kernel takes the same path. Paired harnesses
keep working; only comparisons that STRADDLE this commit are affected.

### Two further optimizations, scoped and deliberately NOT taken

Both relax the equivalence bar further, and neither is worth doing until the
placement head is again the binding constraint. Recorded so they need not be
re-derived:

- **Share `place_hires`' first conv between the two forwards.** `h_hi` is
  `cat(hires_map, ctx_hi_expanded)`, and `hires_map` is IDENTICAL across the
  chosen-card and coverage calls, so `conv2d(cat(A,c), W)` factors exactly as
  `conv2d(A, W_a) + conv2d(c, W_c)` and the expensive 16-channel half
  (~1.06M MACs/row of the head's ~2.8M) could be computed once instead of
  twice. Worth roughly another 10-15%. It is NOT bit-exact even at the LOSS
  level -- splitting a 24-channel accumulation into 16 + 8 reorders the sum --
  so it gives up the one guarantee compaction keeps.
- **Fuse the two placement calls into one 2N-batch call.** They differ only in
  `card_idx`, so concatenating them halves the per-call overhead. Possibly
  bit-exact, since convolution is per-sample along the batch dimension -- but
  the measurements above show batch size changing the blocking, so
  **bit-exactness here must be proven empirically before it is claimed**, which
  is precisely the mistake the 18x10 probe made.

Not recommended: slicing the coverage forward to the advisor rows. See the top
of this section -- that is a loss change wearing an optimization's clothes.

### Suite count after this work

**Python 401 collected, 399 passed / 2 skipped**, of which 10 are
`tests/test_rl_ppo_compaction.py`. C++ untouched -- this is a Python-only
change, so the 622-case Catch2 figure above still stands.

Note the arithmetic: 401 - 10 = **391 before this change, against the 369 this
file recorded**. That count was already stale by 22 when this work started, so
do not read the jump as belonging to this section. Counting the new file's own
cases with `--collect-only` rather than differencing the totals is what made
that visible.

---

## 2026-08-24: the live-mirror state setters. NOT gameplay-affecting.

`UPSTREAM_REQUESTS.md` item 22, APPLIED. A `ClashRoyaleEnv` rebuilt from
perception was a fresh board wearing the real one's unit layout. Four gaps
closed, all additive with behaviour-preserving defaults, so **no checkpoint and
no win rate is invalidated** — verified, not assumed: the only deleted lines in
`include/`+`src/` are `inject`'s own signature and its binding.

```
set_tower_hp(team, slot, hp) -> bool      destroy_tower(team, slot) -> bool
get_tower_hp / get_tower_max_hp(team, slot)
set_current_tick(tick)
inject(card_id, x, y, team, hp=-1.0, deploy_ticks=-1)
```

**`slot` is 0=King, 1=LEFT Princess, 2=RIGHT, in BOARD coordinates for BOTH
teams** — never team-relative. The caller is a sensor reading a screen, and
asking it to mirror its own coordinates is the convention error that put the
arena half a tile off-centre.

**Tower HP is ABSOLUTE in, FRACTION out of perception, and that split is
load-bearing.** This engine's towers are level 9 (Princess 2534, King 4008); a
real account's are often level 4-5 (1750 ours, 1890 theirs) — wrong by a
**different factor per player**. So the setter takes engine-absolute HP and
Python passes `fraction * get_tower_max_hp(...)`. The level knowledge stays on
the perception side, where it already lives. Never inject an absolute reading.

**`hp <= 0` is REFUSED, not clamped** (same contract as `set_hand_for_team`). A
0-hp tower that still occupies its cell and still fires is a position the real
game cannot be in, and killing one has side effects — the crown, the King's
princess-count wake trigger, `LanePath`'s retargeting — that belong to
`destroy_tower`. `destroy_tower` routes through `takeDamage` and then pins the
postcondition, because `CombatEntity::takeDamage` can absorb (shield, parry,
mid-dash). No Tower carries those today; the pin means it stays a destruction
if one ever does.

**`deploy_ticks` is the one nobody asked for and it is the largest of the
four.** `inject -> spawnEntity -> applyCardMetadata` sets
`deployTicksRemaining = DEPLOY_TIME_TICKS` unconditionally, so a mirror handed
**every** unit a fresh deploy second — including one that had been walking for
six. Every rollout believed it had an extra second before anything could act: a
standing defensive subsidy on every candidate. Pass **0** for any unit
perception can already SEE.

Measured, enemy Hog, both sides no-oping — ticks until it first damages our
tower: **88 default, 78 at `deploy_ticks=0`. Exactly `DEPLOY_TIME_TICKS`**, and
worth **317 tower HP (one Hog hit)** in the 100-110 tick window.

> **THE FIRST PROBE MEASURED ZERO, and this is the transferable part.** A
> 140-tick window reports a delta of **0**, because over a window that long the
> Hog deals its full damage either way — the measurement SATURATES and a
> working fix looks inert. Arrival TIME is the quantity that can see it. Third
> instance in this file of a saturating control; the deck-QA air probe is the
> same failure.

Two tests **passed against do-nothing stubs** and had to be strengthened:
"refuses `hp<=0`" passes trivially against a setter that refuses everything,
and "clamps `hp` to full" passes trivially against one that IGNORES `hp` —
which is the pre-item-22 behaviour. Both now carry a positive control that
fires first. *When a measurement's failure mode is maximal permissiveness, it
needs an internal control that MUST fire.*

Suites after: **C++ 646 cases / 6,409 assertions** (645 pass, 1 `[!shouldfail]`,
exit 0), **Python 399/2 skipped** unchanged, **perception 366/1 skipped**
(353 + 13 new binding tests in `perception/tests/test_engine_state_setters.py`,
which exist because the C++ suite cannot see pybind at all and a stale `.pyd`
has twice hidden a landed setter for days). `waypoint_probe` re-run — **0
absorbing states / 8,661,439 positions** — because `deploy_ticks=0` is a new
entry path into `getNextWaypoint` at arbitrary positions.

**The C++ count was already stale by 5**: 646 − 19 new = 627, against the 622
recorded above. Do not read the jump as belonging to this work.


---

## 2026-08-26: the C++ simulator audit. SEVEN defects, six of them gameplay-affecting.

A file-by-file pass over `include/`, `src/` and `tests/`, run on a branch of its
own (`audit-cpp-simulator`) while the Python side was audited separately. The
brief was card mechanics first — specifically `DEFAULT_DECK`, the 2.6 Hog Cycle —
then micro-optimisation, with tests for everything.

**GAMEPLAY-AFFECTING. Every win rate in CLAUDE.md predates all of this.** Six of
the seven change what the simulator does; the seventh (`MatchRules`) changes when
a match ends, which is worse. Suite before: 650 cases / 6,423 assertions. After:
**662 cases / 6,481 assertions**, still exactly one expected failure
(`test_navigation_wedge.cpp`'s open `[!shouldfail]` wedge defect, untouched).

### 1. Ice Golem slowed everything it attacked, and its death explosion slowed nothing

`CardRegistry.h`, card id 40 (and id 175, Hero Ice Golem, which copies its stats).
Registered as:

```cpp
.withOnHit(std::make_shared<FreezeOnHit>(30, 0.65f))
.withDeathEffect(std::make_shared<AreaDamageOnDeath>(2.0f, 84))
```

That is Ice Wizard's effect, and the comment above it opened *"same story as Ice
Wizard"*. The real Ice Golem has **no on-attack slow at all**; the slow belongs
to its death explosion. The same comment then said the death slow was *"not
modeled"* — so the card had a mechanic it should not have had, and lacked the one
it should.

The direction is what makes it expensive rather than cosmetic. An Ice Golem is a
**building-targeter**, so the phantom slow never landed on a troop that could walk
out of it: it landed on a Crown Tower or a Cannon, and refreshed on every 2.5 s
hit. Measured against a Princess Tower over 60 ticks before the fix,
`freezeTicks` was pinned above zero for the whole engagement — a standing **35%
cut to the fire rate of whatever the Ice Golem was tanking**. In a Hog Cycle deck
the Ice Golem's job is precisely to tank the tower while the Hog connects.

**Fix.** `AreaDamageOnDeath` gained the same optional `onHit` slot `AreaSpell`
already carries (`nullptr` default, so every other caller — Golem, Balloon, Giant
Skeleton, the Barbarian Barrel remnant — is bit-identical), and the effect moved
to the trigger the real card uses. The 30 ticks / 0.65 are the registry's own
numbers **moved, not re-sourced**: neither was ever part of the sourced stats
data, same caveat as `splashRadius`.

**The test that was holding it in place.** `test_card_registry.cpp` had a case
named *"Ice Golem applies freeze on hit via the on-hit decorator"* asserting
exactly the wrong behaviour. It was rewritten, with a control that the attack
really lands (so "no freeze" is a statement about the hit, not about nothing
happening) and a second half proving the death explosion does slow.

### 2. Ice Spirit was a 50% slow, not a stun

`CardRegistry.h`, card id 72: `FreezeOnHit(10, 0.5f)`.

This engine already has two conventions and they are not interchangeable. A
**stun** is `FreezeOnHit(ticks, 0.0f)` — Zap, Electro Spirit, Zappies and the
Freeze spell all use exactly `0.0f`. The **0.5–0.7 band is a slow** — Ice Wizard
`0.65f`, Ice Golem's death explosion `0.65f`. Ice Spirit sat in the slow band
while being the card whose entire function is stopping a Prince mid-charge or
pinning a Hog at the bridge. The Evolution's own comment, 700 lines further down,
calls the mechanic a *"stun"*.

**Fix.** `0.0f`.

**The Evolution needed a judgement call, and it went the conservative way.** The
evolved form was `FreezeOnHit(51, 0.5f)` — 51 ticks chosen to stand in for the
real card's *delayed second pulse* (1 s stun, 3 s gap, 1.1 s stun) as one longer
window at half strength. That approximation does not survive the correction:
`51` ticks at `0.0f` is a **5.1-second unbroken hard stun off a 1-elixir card**,
an invention several times larger than the gap it was patching. Both halves now
carry the base card's 10-tick stun, the second pulse joins this file's documented
unmodeled list (nothing here can re-apply an on-hit effect after a delay), and
the evolution differs from its base by exactly the splash boost the source
states.

### 3. A freeze slowed attacks for N ticks and movement for N−1

Found *by* fixing Ice Spirit: with a true stun applied, the victim still moved
`0.1325` tiles — **exactly one tick's worth**.

`freezeTicks` had two readers straddling its own decrement, inside one
`update()`:

```
CombatEntity::update()   reads freezeTicks, decrements it, drains cooldown by freezeSlow
Troop::moveTowards()     reads freezeTicks AGAIN, after that decrement
```

So the last tick of every freeze read as thawed and the unit moved at full speed.
`applyFreeze(1, 0.0f)` — and the registry holds 3-tick and 5-tick stuns (Electro
Spirit, Zap) — did not stop movement **at all**.

This is the same shape as the two bridge-mouth absorbing states already recorded
in CLAUDE.md: one fact, two readers, disagreeing. It is worth noticing that the
*general rule* those bugs produced ("two independent copies of close enough is a
deadlock waiting for the right step size") did not catch this one, because here
there was only ONE copy of the number — what differed was *when* each reader
sampled it. **Add the mutation-ordering case to that rule: one fact, two readers,
and a write in between.**

**Fix.** `CombatEntity::frozenThisTick`, published once at the top of `update()`
before the decrement and read by `moveTowards`. It is transient (recomputed every
tick), so `snapshot()` needs no change — the implicit copy carries it.

### 4. Buildings outlived their own lifetime

`Building::update`. Decay is `maxHp / (lifetimeTicks / 10)` per second in
**integer** arithmetic, and **nothing ever consulted the clock** — a building
died whenever repeated subtraction happened to reach zero.

For a Cannon (824 hp, 300 ticks): `824 / 30 = 27`, so 30 decays remove 810 and it
sits on **14 hp at 30.0 s**, dying on the next decay at **31.0 s**. A free extra
second of the 2.6 deck's only defensive building, every cycle — roughly two extra
Cannon shots.

It read as correct because the single test covering it used `hp = 3000,
lifetime = 300`, and `3000 / 30 = 100` exactly. **That test is named "Building
fully decays to 0 exactly at its configured lifetime."** The contract was already
written down; only a divisible hp made it look true. This is the same failure
class as the `sight >= attack` blind spot — a check that cannot fail for the
inputs it is given.

**Fix.** An explicit expiry branch before the decay schedule, using `hp = 0`
rather than `takeDamage()`. Expiry is not damage: a shield (Cannon Cart) must not
absorb it, a parry (Ronin) must not negate it, and no `OnDamageTakenEffect`
should fire for a clock running out.

### 5. A Mortar answered to "is your King Tower alive?"

`MatchRules::evaluate` identified the King as `entity->symbol == 'R'`. So does
card id 93 — **Mortar**, registered `building(93, "Mortar", 4.0f, 1369, 'R', ...)`,
along with its Evolution.

While a Mortar is alive, its owner's King reports as alive. With the King
actually destroyed, `evaluate()` returns `{over: false}`, `cleanDeadEntities`
erases the dead King on the same tick, and from the next tick the **only** thing
answering that question is the Mortar. The match does not end. The win is not
recorded until the Mortar expires or the episode times out — at which point
`TimeoutRules` decides it on towers instead, which is a different (and possibly
opposite) verdict.

Mortar is in the registry and phase 1's `random_opponent` samples random decks,
so this is reachable in ordinary training, silently, as wins converted into
timeouts.

**Every other site already guarded by type.** `Tower::update`'s Princess count
uses `isTower() && symbol != 'R'`. `TimeoutRules::resolve` uses a `dynamic_cast`
and says in its own comment that a symbol check would be fragile *"if tower
symbols are ever changed for the renderer"* — the hazard was written down, in the
file next door, and this one site did not take it.

**Fix.** `entity->isTower() && entity->symbol == 'R'`.

**The suite could not have caught this, and the reason is instructive.** All four
existing `MatchRules` cases built a stand-in King as
`DummyEntity(..., 'R')` — a non-Tower entity wearing the symbol, which is
*precisely* what a Mortar is at runtime. The test double reproduced the bug and
then asserted the buggy behaviour was correct. They now build real `Tower`s.

### 6. `pushAwayFrom` under-pushed from dead centre by exactly 1.0 tile

Found by a characterization test written to protect the optimisation in §7 —
i.e. by writing down what the function was *supposed* to do before changing how
it did it.

```cpp
if (dist < 0.001f) { dx = 1.0f; dy = 0.0f; dist = 1.0f; }
float push = minDist - dist;
```

The `dist = 1.0f` exists to make `dx / dist` a unit vector. It then flowed into
`push`, so a point sitting exactly on an obstacle's centre was pushed out by
`minDist - 1.0` instead of by `minDist`. Measured: a troop landing dead-centre on
a Building (`minDist` 1.4) moved **0.403 tiles and was still inside the
footprint**; on a King Tower (`minDist` 2.4) it moved 1.4 of the 2.4 it needed.
It escaped over two ticks instead of one, and the shortfall was 1.0 tile every
time — the fake distance.

**Fix.** Separate the direction from the distance. The ordinary path is
bit-identical (`ux == dx / dist`, same operands in the same order); only the
coincident branch changes.

### 7. The built-in opponent defended against spells

`HeuristicOpponent::act` scanned every living enemy entity for the deepest
incursion and excluded only Towers. An `AreaSpell` sits on the board at its
impact point with `hp = 1` for the length of its fuse, and a `Projectile` likewise
while in flight — so a Fireball thrown at this bot's own tower registered as the
deepest threat on the board and bought a full defensive placement of **the
strongest card it could afford**, against something that was never a unit.

This is phase 1's `ClashEnv::step()` opponent and all three `BUILTIN_ANCHORS` in
the phase-2 Elo roster, so it changes what those anchors measure.

**Fix.** Skip `!entity->isTargetable()` — the discriminator the rest of the engine
already uses for this exact question (`ClashEnv::extractObservationForTeam`:
*"Projectiles and pending spells are not board presence"*). It also, correctly,
hides a cloaked unit from a bot that could not have seen it.

The file had **no test coverage at all** before this; `tests/core/test_heuristic_opponent.cpp`
is new, and carries the control (it must still answer a real Hog) that keeps the
negative case from passing vacuously.

### Latent, fixed while passing: projectile line-splash

`RangedBuildingTargeter::performAttack` and `Tower::performAttack` both construct
a `Projectile` without forwarding `lineSplash` / `lineSplashRange`, which
`RangedTroop::performAttack` does forward. A card of either archetype configured
for a piercing line would have silently fired an ordinary circular-splash shot —
no compile error, no test failure. Latent today (Royal Giant is the only
`RangedBuildingTargeter` and sets neither; no Tower Troop does), which is exactly
why it was worth closing rather than leaving.

---

## 2026-08-26 (perf): the hot paths. 2.4-4.4x on collisions, 25x on the encoder.

Measured with `tools/audit/collision_bench.cpp`, built standalone against the
header-only engine, reporting the **minimum** of 40 repeats with the median
alongside (same methodology as `engine_profile.cpp`: the cost is deterministic
and the noise strictly additive, so the minimum is the least contaminated
estimate and a wide min/median gap flags a noisy box instead of hiding inside an
average).

**Both implementations live in that one file, deliberately.** The alternative —
time the new code, stash the diff, rebuild, time the old — also swaps in the
gameplay fixes above, so the two runs would be simulating different matches and
the comparison would be measuring the workload rather than the code. Holding the
board fixed and swapping only the FUNCTION removes that confound.

| board | function | old | new | |
|---|---|---|---|---|
| 20 entities | `resolveCollisions` | 3.040 us | 1.250 us | **2.43x** |
| | `resolvePositionAgainstBuildings` x12 troops | 1.150 us | 0.388 us | **2.97x** |
| 32 entities | `resolveCollisions` | 7.860 us | 3.325 us | **2.36x** |
| | `resolvePositionAgainstBuildings` x24 troops | 2.797 us | 0.760 us | **3.68x** |
| 48 entities | `resolveCollisions` | 14.010 us | 5.190 us | **2.70x** |
| | `resolvePositionAgainstBuildings` x40 troops | 5.686 us | 1.292 us | **4.40x** |

At 32 entities the two together fell from **10.66 us to 4.09 us per tick**. The
ratio grows with population, which is the point: both were quadratic in board
size to answer questions about a handful of entities.

**What changed.**

- `resolveCollisions` is O(n²) and was making four virtual calls per PAIR
  (`getCollisionRadius` and `isTargetable`, both sides) for facts constant across
  the whole call — a radius never changes, targetability cannot change within a
  tick, and nothing in this function kills anything. Hoisted to one pass building
  a POD scratch array: ~1,700 virtual dispatches become 60 at n = 30. Positions
  are **not** hoisted; they are the one thing the function mutates.
- `resolvePositionAgainstBuildings` runs once per MOVING TROOP per tick and
  walked the entire entity list with a virtual call per candidate, to consult the
  six Crown Towers and one or two deployed buildings. It now reads a cached index
  of colliders, invalidated by the only two things that change membership
  (`commitPendingEntities`, `cleanDeadEntities`). Liveness is deliberately NOT
  cached — hp changes constantly, so the `isAlive()` check stayed in the loop.
- `cleanDeadEntities` made three full passes every tick. The overwhelming
  majority of ticks have no deaths at all, and all three passes are provably
  no-ops in that case, so it now short-circuits on a single scan.
- `CombatEntity::findTarget` and `BuildingTargeter::findTarget` re-derived the
  attacker's OWN collision radius — a loop invariant, and a virtual call — once
  per candidate. Hoisted. Identical arithmetic in identical order, so the result
  is bit-for-bit unchanged.

**Bit-equivalence was verified, not assumed.** A faster function that answers
differently is not an optimisation. The benchmark sweeps
`resolvePositionAgainstBuildings` over 8,572 board cells comparing old against
new: **IDENTICAL on every one**. The 8 cells sitting exactly on a collider's
centre are excluded and that exclusion is the point rather than a fudge — those
are the cells §6 above deliberately changed.

### The observation encoder: 0.0694 ms -> 0.0027 ms

`extractObservationForTeam` is the hottest function in the C++ layer by a wide
margin — `engine_profile.cpp` measured it at **63x the physics tick it
describes** (0.0694 ms against 0.0011 ms), and at 111.8% of the whole
`stepSelfPlay` call it is returned from.

It was allocating twice per call. `std::vector<float> obs(spatialSize, 0.0f)`
gives capacity exactly 12,852; the 754-float scalar tail is then appended with
`push_back`, so the first one reallocates and copies all 12,852 floats it just
finished zeroing — and the grown block crosses the allocator's large-block
threshold. Reserving `observationSize()` up front and then `resize`-ing to the
spatial block leaves one allocation.

The per-hand-slot card one-hot was 185 branchy `push_back`s (740 per
observation); it is now a zero-fill plus one store.

| | min | median |
|---|---|---|
| `extractObservationForTeam(0)` before | 0.0694 ms | 0.0756 ms |
| after | **0.0027 ms** | 0.0031 ms |
| `stepSelfPlay(skip=10)` before | 0.1242 ms | 0.1509 ms |
| after | **0.0410 ms** | 0.1118 ms |

**This number was checked against its own instrument's blind spot.**
`engine_profile.cpp` defeats dead-store removal with `if (obs.empty())`, which
forces the VECTOR to exist but not its CONTENTS — and everything here is
inlinable, so in principle a compiler could drop writes nobody reads and make the
encoder look arbitrarily cheap. `collision_bench.cpp` therefore also times a
variant that **checksums every element**: **9.80 us**, and that figure includes a
13,606-element summation the real caller never pays. So even the pessimistic
bound is 7x better than the 69.4 us it replaced. The speedup is real, not an
elision artifact.

Two smaller encoder changes rode along: `dynamic_cast<Building*>` became
`entity->isBuilding()` (exactly equivalent — only `Building` overrides it, and
`Tower` inherits — without the RTTI walk), and the six tower-HP scalars are built
in ONE pass instead of six full board scans each with a `dynamic_cast` per
entity, roughly 200 RTTI queries per observation to produce six floats.

That last one also removed a **seventh copy of the board centre**: it decided
left-vs-right with `BOARD_WIDTH / 2.0f` (9.0) where the rest of the engine uses
`ArenaLayout::CENTER_X` (8.5), and `GameManager::findTower` answers the identical
question with the latter. No tower currently sits in the half-tile between them,
so it was right by luck.

---

## 2026-08-26 (part 2): the ability effects and the spawned units

The first pass covered core combat, the board, and match rules. This one covers
the ~19 bespoke Champion/Hero ability effects and the registry's child
CardStats. Seven more defects, five of them in code that had no test coverage
of any kind.

### 8. `pullToward` ran BACKWARDS on a negative distance

Every caller computes the argument as "how far do I still have to close" --
`dist - meleeRange`, `dist - effectiveAttackRange + 0.1f`. When the target is
already inside that range the subtraction goes negative, and

```cpp
float moveBy = (distance < dist) ? distance : dist;
```

takes the negative value happily, so the pull becomes a push.

`GoldenKnightDashEffect` was the live case: `pullToward(self, target->position,
dist - 1.0f)` against an enemy closer than 1.0 tiles. Measured: a Golden Knight
0.5 tiles from his target ends the ability at 1.0 tiles -- he **retreats** from
what he dashed at, and he is registered with `maxDashes = 10`.

Every other caller is safe only by accident of ordering. `CombatEntity::update`'s
jump and hook branches are both `else if`s reached only when `dist >
effectiveAttackRange`, so their subtraction cannot go negative -- today.

**Fixed in `pullToward`/`pushAway` themselves**, not at the call site, for the
reason `exemptFromForcedMovement` gives three lines above them: a per-call-site
clamp is a check somebody will forget to add. Something that genuinely wants to
move away calls `pushAway`.

### 9. Hero Giant hurled BUILDINGS across the arena

`Entity.h` states the rule plainly, and states why it lives where it lives:

> Both are a no-op on a Building regardless of which mechanic is calling --
> buildings are stationary, full stop... Enforced here, once, rather than at
> every individual call site, so nothing can reintroduce this bug by forgetting
> a per-site check.

`HeroGiantHurlEffect` reintroduced it, by not calling either function:

```cpp
victim->position.x = static_cast<float>(board.getWidth() - 1) - victim->position.x;
```

A raw position write skips the guard entirely. And its victim selector,
`findHpExtremeEnemy`, excluded only `isTower()` -- not deployed buildings --
while its own comment said it selects an "enemy TROOP". A Cannon is the
highest-HP thing inside a 3-tile radius far more often than a troop is, so Hurl
spent most of its uses throwing a stationary building into the other lane.
Measured: a Cannon at x = 6.0 ended at x = 11.0.

**Two fixes, because there were two holes.** `findHpExtremeEnemy` now excludes
`isBuilding()` (strictly wider than `isTower()`, since Tower derives from
Building, so nothing previously excluded is now admitted). And the raw write is
replaced by a new `mirrorToOppositeLane(entity, boardWidth)` in `Entity.h`,
third member of the pull/push family and carrying the same guard -- which also
removes the last two restatements of the mirror formula.

### 10. Mother Witch's curse stacked a death effect per HIT

```cpp
target->deathEffect = std::make_shared<CompositeDeathEffect>({ target->deathEffect, spawnEffect });
```

on **every hit**. Three hits produced three nested composites and three hogs on
death; a Musketeer taking twenty hits died into twenty of them, and the chain's
depth grew with the hit count. Measured: three applications, three hogs.

The curse's *duration* is meant to refresh on every hit -- that half was right.
Only the *spawn* is a one-time arming, now latched by
`CombatEntity::curseDeathSpawnAttached`.

### 11. The Royal Chef fed the same troop forever

`RoyalChefBuffEffect` picks the nearest ally and applies `hp += hp / 10` plus a
damage buff lasting 999999 ticks. Nothing excluded an ally it had already fed,
so a tank parked beside the tower was fed every 280 ticks and compounded
geometrically. Measured: **1000 -> 1100 after one serving, -> 1771 after six**
(+77%, exactly 1.1^6), unbounded in match length.

The real card grants a troop "+1 Level", once. Latched by
`CombatEntity::royalChefServed`, so the chef moves on to a troop that has not
eaten.

### 12. The speed-tier rework never reached the SPAWNED units

The 2026-08-24 rework put every playable card on one of five real tiers and
round-tripped **"109 / 109 match, 0 mismatches"**. 109 is the count of cards
with an OFFICIAL ROW. The child `CardStats` that death effects spawn have no row
of their own and were never in that set.

**This was measured, not inferred, and the first inference was WRONG.** A grep
of the registry found 54 speed literals that are not `SPEED_*` constants, which
looked like a large uncovered population. It is not: most of those literals land
within 1% of a tier by coincidence (`0.5f` gives 1.000 tiles/s against SLOW's
0.994; `1.0f` gives 2.000 against FAST's 1.988), and many are helpers nothing
reaches. `tools/audit/spawn_speed_audit.cpp` spawns every registered card, fires
its death effects, and reads `Troop::getSpeed()` off what actually arrives:

**8 units off-tier out of 109, of which 5 are documented.** Reading the source
would have produced a number seven times too large. State the probe, not the
conclusion.

| unit | tiles/s | nearest tier | off by | |
|---|---|---|---|---|
| Golemite (-1) | 0.400 | VERY_SLOW 0.663 | 39.6% | **below the real game's floor** |
| Bats (-12, -14) | 1.700 | FAST 1.988 | 14.5% | contradicts playable card id 78 |
| Goblin Brawler (-31) | 1.400 | MEDIUM 1.325 | 5.6% | still open, see below |
| Berserker, Boss Bandit, Heal Spirit, Ronin, Spirit Empress | | | | documented "no official row" |

**Golemite** was on a raw `0.2f` -- 0.400 tiles/s against `SPEED_VERY_SLOW`'s
0.663. The real game's published table bottoms out at 30 tiles/min, so this unit
moved slower than **any card in Clash Royale**, and 2.5x slower than the Golem
it splits out of. Fixed to `SPEED_SLOW` by mirroring its parent -- the same rule
the rework itself used for the 9 Hero variants, "by mirroring their base card's
tier rather than by guessing".

**Bats** needed no external source at all, because *the registry contradicts
itself*: playable card id 78 is `SPEED_VERY_FAST` (2.651 tiles/s) and the child
stats a Night Witch releases on death are a raw `0.85f` (1.700). Same name, same
81 hp, same 81 damage, same 12-tick cooldown, 36% different speed. One of the
two is on a real tier. Fixed to match card 78.

After: **6 off-tier, 0 below the floor.** Five of the six are the documented
list.

**Still open, deliberately not guessed: Goblin Brawler (-31)** at 1.400 tiles/s,
5.6% off MEDIUM. Goblin Cage's spawn has no playable counterpart in this
registry to cross-check against and no official row, so there is nothing here to
make the call from -- unlike Golemite (parent) and Bats (its own card). It
belongs with the 5 documented off-tier cards until a source covers it.

**A note on that documented list.** `DECISIONS.md` says "13 cards have no
official row" and then names twelve: Berserker, Ronin, Goblin Machine, Goblin
Demolisher, Furnace, Heal Spirit, Suspicious Bush, Rune Giant, Little Prince,
Goblinstein, Boss Bandit, Spirit Empress. The measurement finds only five of
them still off-tier by more than 1% -- the rest sit within 1% of a tier by
coincidence, which is worth knowing before anyone "fixes" them.

### The negative result, which is worth as much as the defects

Every `FreezeOnHit` in the registry was checked against its card's real
mechanic. **The stun-vs-slow convention holds everywhere except the two cards
already fixed in part 1** (Ice Spirit, Ice Golem). Electro Wizard, Electro
Dragon, Electro Spirit, Zappies, Goblinstein and the Freeze spell are all
correctly `0.0f` stuns; Ice Wizard, Giant Snowball, Earthquake and the Princess
Evolution are all correctly in the 0.5-0.7 slow band. There is no third case.

Likewise, only one registration matched the "Ice Golem shape" -- a
building-targeter carrying an on-hit effect that can therefore only ever land on
a building -- and it was a false positive: Goblinstein's `withOnHit` sits on its
secondary ranged unit, not on the building-targeter.

---

# ARCHIVE — `perception/UPSTREAM_REQUESTS.md` and `perception/BOT_REQUESTS.md` (retired 2026-08-24)

Both backlog files were worked to empty on 2026-08-24: every item was either
verified already applied, resolved by a decision, or implemented. Their contents
are preserved here **verbatim**, because 112 references across 47 files cite
these items by number — including engine headers that cannot be edited in
passing (`Entity.h`, `Tower.h`, `CardStats.h`, `ClashEnv.h`, `bindings.cpp`).

This is the same move CLAUDE.md made on 2026-08-24 when its own narrative was
extracted here: headings are kept intact so any passage cited by number still
resolves. A citation reading "UPSTREAM_REQUESTS.md item 13" resolves to
"UPSTREAM item 13" below.

**Read these as history, not as pending work.** The live residue that came out
of the sweep was moved to `TODO.md`, which is the authoritative task list. Three
things in particular are NOT settled by anything below:

- **BOT item 3** (channels 0-7 assign rather than accumulate) is agreed, has a
  one-line fix, and was deliberately left queued pending a clean restart.
- **BOT item 7's** live-adapter wiring: `readers/tower_numerals.py` exists and is
  tested, but `live/adapter.py` still reads the bar.
- **BOT item 1** (perception-shaped observation noise) stays deferred behind the
  sensor's own ~15% unit-misnaming and 33.8% card-identity error.

Statuses inside the archived text are as they stood when each item was written
and were NOT rewritten; several are stale by their own later notes (item 1 and
item 2's arena move was reversed on 2026-08-21; BOT item 2's "RESOLVED" predates
the 2026-08-16 deck change that gave the recordings tie up on purpose).

### UPSTREAM source document: Simulator changes requested from `perception/`

Written from `perception/`, which modifies nothing outside itself. Items are
**proposed** here first, with evidence and blast radius, and applied only with
explicit sign-off — see CLAUDE.md's rule on never changing C++ without
confirming the exact diagnosis and the exact edit first.

Last updated 2026-08-24. Items 0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 16,
19, 21, 22, 24 and 25 are applied; 8, 17, 20 and 23 are still open; 18 is ACCEPTED, WILL NOT FIX
FOR NOW. There is no item 11.

> **Second status audit, 2026-08-23.** This line had drifted again, in all four
> possible directions at once: item 3 was listed OPEN while `Tower.h` already
> carries its activation logic, items 19 and 21 were applied and missing from
> the applied list, item 20 was OPEN and missing from the open list entirely,
> and 18 was filed under "open" when its own header says it is accepted and
> deliberately unfixed. A summary that has to be updated by hand alongside the
> header it summarises is a second copy of the same fact -- the exact pattern
> CLAUDE.md warns about for engine constants, and the second time this specific
> line has gone stale. The per-item headers are authoritative; if the two
> disagree, believe the headers and fix this line.

> **Status audit, 2026-08-19.** Items 5, 6 and 12 were carrying `OPEN` headers
> while the exact code they propose was already merged — verified line by line
> against the tree (5: `ClashEnv.h`'s `std::floor((BOARD_HEIGHT - 1) - position.y)`;
> 6: `constexpr int riverRow = 17;`; 12: `ClashEnv::isValidPlacementForCard` plus
> its `.def("is_valid_placement", ...)`). Their statuses are corrected below.
>
> Recording *why* rather than silently flipping them: this document's value is
> that its status field can be trusted, and three items reaching `applied`
> without the propose → sign-off → implement sequence this file exists to
> enforce is worth one line of history. Items 5 and 6 are gameplay-affecting by
> their own text ("opponents get stronger… Elo is therefore not comparable
> across this fix"), and item 12's Python side is flagged the same way — so the
> win-rate history around them should be read with that in mind.

| # | Request | Severity | Status |
|---|---|---|---|
| 0 | River band `[16,18)` → `[15.5,17.5)` | was blocking training | **DONE — verified** |
| 1 | Left Princess towers `x = 3.0` → `4.0` | blocks a stage-0 acceptance target | **DONE — verified** |
| 2 | Kings `x = 8.5` → `9.0` | cosmetic accuracy | **DONE, then REVERSED 2026-08-21 — see item 2** |
| 3 | King Tower has no activation condition | fidelity gap | open, **already worked around, no change needed** |
| 4 | `inject(..., team)` + `get_hand(team)` | convenience | **DONE — already landed 2026-07-29, see below** |
| 5 | Team-1 observation mirrors the truncated row, not the position | **corrupts all self-play** | **DONE — applied (status corrected 2026-08-19)** |
| 6 | River marker row is 17 for team 0 but 16 for team 1 | same class, smaller | **DONE — applied (status corrected 2026-08-19)** |
| 8 | Fireball (689) misses the Musketeer kill (721 HP) by 32 | **fidelity vs learnability — needs a decision, not a fix** | open, proposed 2026-08-06 |
| 7 | No way to seed the engine's RNG (TWO generators, not one) | every A/B test cost ~10x more; invalidated a control 2026-08-20 | **DONE 2026-08-21** |
| 9 | **Troop movement is ~4-5x faster than the real game** | **largest measured sim-to-real gap; miscalibrates every timing the agent learns** | **DONE — applied and verified 2026-08-07** |
| 12 | Bind `isValidPlacement` so the action mask stops disagreeing with the engine | 58.7% of card choices silently rejected | **DONE — `is_valid_placement` is bound in the current `.pyd`** |
| 10 | State-estimator write half: `set_elixir_for_team` / `set_hand_for_team` | search over a reconstructed state scored a fabricated hand/elixir | **DONE — applied 2026-08-17, recorded here 2026-08-19** |
| 13 | **State snapshot/restore, so decision-time search becomes possible** | unblocks the biggest unexploited asset | **DONE — applied and verified 2026-08-11** |
| 14 | Offence structurally under-priced; deploy time added | win condition was unplayable by construction | **DONE — raised and applied 2026-08-19** |
| 18 | Troops deadlock in the concave pocket between two buildings | 7 permanent stalls per 343k unit-ticks; fix means real pathfinding | **ACCEPTED AS-IS 2026-08-20 — will not fix for now** |
| 16 | Bind `TimeoutRules::resolve` so match outcomes have one definition | **8** Python copies, all missing the HP tie-break | **DONE — signed off and applied 2026-08-20** |
| 17 | Const accessors for internal timing state | divergence tests cannot see cooldowns/fuses | open, proposed 2026-08-19, **ergonomics not coverage** |

Items 1 and 2 were done together since the measured benefit is combined
(max error 0.63 → 0.31 tiles) and neither is a large or risky edit.

**There is no item 11.** 10 and 11 were both skipped when the numbering jumped
9 → 12; 10 has since been filled in (retroactively, for the setters), 11 has
not. Not reused, so older references to "item 12"/"item 13" stay valid.

---

### UPSTREAM item 0. DONE — the river band (verified 2026-07-29)

`Board.h` now reads:

```cpp
float riverY_start = 15.5f;
float riverY_end   = 17.5f;
Vector2D leftBridge { 4.0f, 16.5f };
Vector2D rightBridge{ 14.0f, 16.5f };
```

The band is centred on 16.5, which is the axis the tower layout already
mirrored about (King 2.5 ↔ 30.5, Princess 6.0 ↔ 27.0 under `y → 33 - y`).

**Verified two ways, both independent of the change itself.**

The placement asymmetry is gone. Same test that found it — 20 trials per row,
cheap cards only so affordability never limits, each team in **its own
mirrored frame**:

| row (own frame) | team 0 | team 1 | before the fix |
|---|---|---|---|
| 14 | 15/20 | 16/20 | 17/20 vs 15/20 |
| **15** | **14/20** | **17/20** | **15/20 vs 0/20** |
| 16 | 0/20 | 0/20 | 0/20 vs 0/20 |

Both teams now reach row 15 and stop at 16. `get_own_half_max_y()` returns
15.0, and the rebuilt `.pyd` carries it.

And the calibration residual against real footage improved: **rms 0.40 →
0.32**.

This one mattered beyond geometry: `model.py` computes `own_half_rows` once
and applies the same placement mask to both sides, so team 1's policy was
proposing row-15 placements that `playCard` silently rejected — gradient
spent on an action that could never do anything, and only when playing team 1.

---

### UPSTREAM item 1. DONE — left Princess towers, `x = 3.0` → `4.0` (verified 2026-07-30)

> **REVERSED 2026-08-21.** A professional player's audit put the King back at
> **8.5** and the left Princess back at **3.0**, and the reason this fit looked
> good is worth keeping: x is a CELL INDEX clamped to [0, 17], so the board's
> centre -- the fixed point of the mirror `17 - x` -- is 8.5, not 9.0. The
> landmark fit was anchored on that half-tile convention error, which is why its
> residual improved (0.63 -> 0.31 tiles) while the layout became symmetric about
> the WRONG centre. The corrected arena is symmetric about 8.5 and matches the
> real game's river row `WWBBWWWWWWWWWWBBWW`. All of it now lives in
> `include/core/ArenaLayout.h` and is bound to Python, so no consumer keeps a
> copy. See CLAUDE.md's "Board geometry".


**Landed exactly as requested**, two lines in `GameManager.h`:

```cpp
addTower(4.0f,  6.0f, 0, "Princess Tower", towerTroopStats(aiTowerTroop));
addTower(4.0f, 27.0f, 1, "Princess Tower", towerTroopStats(oppTowerTroop));
```

`ClashRoyaleTests` re-run in full afterwards (504 test cases, 4329
assertions, 0 warnings, `-Wall -Wextra`) and a `main.cpp` smoke match still
completes normally. `perception/geometry.py`'s `OWN_PRINCESS_LEFT`/
`OPP_PRINCESS_LEFT` updated to match, so `tools/calibrate.py`'s
`engine_tiles()` now returns this directly.

#### Why

In the real arena a Princess tower sits directly behind its own bridge —
troops crossing walk straight into it. Measured from the recordings: left
tower centre **x = 819 px**, left bridge centre **x = 821 px**. The same lane,
within measurement error.

The engine puts the left bridge at `x = 4.0` and the left Princess at
`x = 3.0`. A full tile apart. The right side is already correct (both at
14.0), so the left lane is the only one that disagrees with itself.

#### What it is worth, measured

Screen→tile homography fitted to the eight arena landmarks, aggregated over
the opening frames of all 8 recordings. Identical pixels, identical solver;
only the target tile coordinates differ:

| target geometry | max error | rms |
|---|---|---|
| engine as-is (river already fixed) | **0.63** | 0.33 |
| **+ left Princess `x` 3.0 → 4.0** | **0.40** ✅ | 0.27 |
| + Kings `x` 8.5 → 9.0 (alone) | 0.51 | 0.37 |
| + both | 0.31 | 0.21 |

**Stage 0's acceptance target is < 0.5 tiles, and this change alone reaches
it.** Without it, every placement perception reports starts with 0.63 tiles
of systematic error before any detection error is added — 42% of stage 3's
1.5-tile budget, spent on nothing.

#### Blast radius

Small, but not zero. It moves a tower, so pathing and aggro around the left
lane change slightly, and `isValidPlacement`'s building-overlap check moves
with it. Needs a full `ClashRoyaleTests` run. It changes gameplay, so treat
`model_weights.pth`'s win-rate history as suspect afterwards — the same
caveat as the river fix.

---

### UPSTREAM item 2. DONE — Kings `x = 8.5` → `9.0` (verified 2026-07-30, landed with item 1)

> **REVERSED 2026-08-21.** A professional player's audit put the King back at
> **8.5** and the left Princess back at **3.0**, and the reason this fit looked
> good is worth keeping: x is a CELL INDEX clamped to [0, 17], so the board's
> centre -- the fixed point of the mirror `17 - x` -- is 8.5, not 9.0. The
> landmark fit was anchored on that half-tile convention error, which is why its
> residual improved (0.63 -> 0.31 tiles) while the layout became symmetric about
> the WRONG centre. The corrected arena is symmetric about 8.5 and matches the
> real game's river row `WWBBWWWWWWWWWWBBWW`. All of it now lives in
> `include/core/ArenaLayout.h` and is bound to Python, so no consumer keeps a
> copy. See CLAUDE.md's "Board geometry".


Board `[0, 18)` has centre 9.0. `addTower` put both Kings at 8.5, so a
4-tile-wide King spanned `[6.5, 10.5)` instead of `[7, 11)`. Measured king
centre 955.75 px sits at 8.79 in bridge-calibrated coordinates — between the
two, closer to 9.0.

Landed together with item 1 rather than alone, per this file's own note that
it "reaches max 0.51 (still failing)" in isolation — combined, held-out
calibration error dropped max 0.63 → 0.31 tiles, rms 0.33 → 0.21.
`test_game_manager.cpp`'s King-position assertions and the building-overlap
tests measuring distance from the King (whose test points shared the King's
old x, so shifting both together preserved the same distances) were updated
to match.

---

### UPSTREAM item 3. APPLIED 2026-08-21 — King Tower activation

> **Applied.** `Tower` carries a latching `awake` flag: Princess Towers
> construct awake, the King asleep (only `GameManager::addTower`'s
> `symbol == 'R'` branch builds one). It wakes permanently on any damage or on
> any friendly Princess Tower being destroyed, and while asleep
> `Tower::findTarget` returns `nullptr` so it can neither acquire nor fire. It
> stays targetable and damageable throughout.
>
> Measured with `tools/audit/king_activation_audit.cpp` — lone Hog, identical
> placement, only dormancy varying: **1268 tower damage with the Kings awake,
> 2219 with them dormant**, and the Hog survives to tick 170 instead of 122.
> The 1268 reproduces the figure the 2026-08-20 sight audit recorded.
>
> **This was requested only as a note, explicitly with "no change requested",
> and was implemented anyway** because the 2026-08-21 player audit asked for it
> directly. GAMEPLAY-AFFECTING: every win rate earned before it is historical.
> Checkpoints are unaffected.
>
> The perception-side workarounds below are now unnecessary but harmless, and
> are left in place: excluding the Kings from `divergence` still measures
> perception rather than engine fidelity, which is what that metric is for.

`Tower.h` built the King like any other tower; `GameManager::reset()` gave
it range 7.0 and a 10-tick cooldown. Nothing made it dormant, so it fired
from tick 0. The real King is inert until activated.

`SimDriver.divergence` excludes both Kings and `readers/towers.king_divergence`
reports them separately, so the pipeline's quality metric measures perception
rather than this gap.

Recorded originally only because it is a real behavioural difference that also
affects training: an agent learns that chip damage to the King is punished
immediately, which is not true of the real game.

---

### UPSTREAM item 14. DONE — offence was structurally under-priced; deploy time added (raised and applied 2026-08-19)

**RESOLVED by adding a 1.0 s deploy time** (`CardStats.h` `DEPLOY_TIME_TICKS = 10`,
set in `CardFactories::applyCardMetadata`, consumed in `CombatEntity::update`).
The human authorised the engine change after reading the investigation below.

**Controlled result.** Same harness, same supported push, ~161 scored states
each, engine the only difference:

| engine | marginal value of a supported push | 95% CI |
|---|---|---|
| deploy time 0 (old) | -73.7 HP | [-349.5, +195.3] |
| deploy time 10 (new) | **+448.5 HP** | [+137.3, +760.1] |

The defence's cost to answer a 4-elixir commitment rose 1.07 -> 2.93 elixir. A
naked win condition correctly got WORSE (1025 -> 343 tower damage in a punish
window, since it now stands inert under tower fire) while an escorted push holds
at 993.8 -- which is how real Clash prices those two plays.

546 C++ cases pass. The original investigation is kept below unchanged, because
the reasoning that selected deploy time out of several candidate deviations is
the part worth re-reading if this is ever revisited.

#### Original investigation (2026-08-19), kept for the record

**No change is being requested yet.** This is a measured observation with a
diagnosis I cannot complete from the Python side, written up per CLAUDE.md's
rule that engine changes need the exact diagnosis first. It is deliberately NOT
a proposed edit: every candidate fix here is gameplay-affecting and would
invalidate the win-rate history of every checkpoint.

#### What was measured

The 2026-08-19 curriculum pivot replaced the opponent-elixir-multiplier ladder
with a competence ladder at a symmetric 1.0x economy, on the hypothesis
(`DECISIONS.md`, "The 1.5x Curriculum Overfitting Hypothesis") that a permanent
multiplier is what priced the win condition at zero. The falsifier was run with
NO network on either side — both players are the deterministic
`python_ai/opponents/teacher.py` — so the historical confound between "the environment
prices this badly" and "this net cannot execute it" is removed.

**The hypothesis was not confirmed.** At a symmetric 1.0x economy, committing
the win condition is still strongly net-negative:

| measurement | result |
|---|---|
| attack vs cycle, win rate (n=100 paired) | **−0.330**, p=5.7e-08 |
| ...enemy tower damage dealt | 6326 vs 6938 (attack is not even ahead) |
| ...our tower damage taken | **5772.7 vs 3049.3** |
| never playing it at all (`ban` arm) | **0.910** — best of the three |
| marginal value of one commitment (n=220) | **−298.2 tower HP**, CI [−494, −107] |

#### Why it is probably not the card, the King, or the timing

* **Not the card.** A lone Hog injected on an empty board deals **2536 tower
  damage** — a full Princess Tower — and dies at tick 190. (This also means
  CLAUDE.md's older "317 tower damage in 40 s" no longer reproduces.)
* **Not item 3 above.** The enemy King firing from tick 0 does reach a Hog
  attacking either Princess (distance 6.1 against its 7.0 range), but the
  unopposed number above already includes that and the Hog still takes the
  tower.
* **Not the answer's availability.** The defender answers with Skeletons (35 of
  40 trials) and Ice Spirit (31 of 40) — a 1-for-4 trade in its favour, which is
  REAL Clash. Forcing both out of its hand moved the Hog only 634.0 → 665.7.
* **Not timing, and this is the surprising one.** Tightening the commit gate
  makes it monotonically worse below ~3 elixir: −298.2 at ≤7.0, −268.6 at ≤3.0,
  **−585.4 at ≤1.5** (p=9.2e-06). Naturally-occurring low opponent elixir means
  they JUST SPENT, so their push is already on the board — the opposite of a
  punish window.

Against that, the punish mechanic itself demonstrably works when constructed
artificially: forcing the defender's bar to 1.0 with nothing on the board is
worth **+391 tower HP (+62%)**, 256 vs 158 hp per elixir committed.

#### The question for the engine owner

**Does a defender in this engine recover its tempo faster than the real game
allows?** The pattern — punish pays when constructed, never occurs naturally,
and defence answers a 4-cost commitment for ~1.2 elixir while conceding ~634 HP
— is what you would see if the defending side can re-establish a threat sooner
than a real opponent could. Two known deviations already recorded in CLAUDE.md
point the same direction and neither has been measured for this effect:

1. **No deploy time.** The engine spawns a troop active; the real game freezes
   it ~1 s after it lands. That 1 s is paid by the DEFENDER in the real game
   (their answer arrives late), so removing it is a systematic subsidy to
   defence — and it applies on every defensive placement, i.e. far more often
   than on the occasional attack.
2. **`Projectile.h:88` has its own untouched `speed`**, never recalibrated
   alongside the 2026-08-07 `MOVEMENT_SPEED_SCALE` fix.

#### What I was NOT asking for (at the time)

No edit. Specifically not a reward-side or curriculum-side fix: **four Hog
mechanisms have already been built and measured null** (a reward multiplier, an
advisor target, random forcing, gate-timed smart forcing), and this pivot is the
fifth thing that did not move it. Adding a sixth on the policy side would repeat
a pattern this project has already paid for five times.

The cheap next step, if it is wanted, is to measure deploy time in isolation:
add a spawn delay behind a flag, default off, and re-run
`python_ai/eval/prove_environment.py --mode marginal`. That is a gameplay-affecting
change and would invalidate every win rate, so it is the human's call.

Harnesses, all new and all read-only against the engine:
`python_ai/eval/prove_environment.py` (win-rate arms + marginal value),
`python_ai/eval/prove_wincon_trade.py` (elixir trade, supported push, punish window),
`python_ai/eval/prove_teacher.py` (teacher strength bars).

---

### UPSTREAM item 4. DONE — `inject(cardId, x, y, team)` and `get_hand(team)` (landed 2026-07-29)

**Already in the engine, exactly as requested below** — `ClashEnv::inject`/
`getHandForTeam` and their `bindings.cpp` entries (`inject`/
`get_hand_for_team`) landed alongside the river fix, before this being "not
blocking" ever mattered. `perception/bridge/sim_driver.py`'s reverse-engineered-
shuffle workaround described below still works and hasn't been switched over
to the new primitives — that's a `perception/`-side follow-up, not an engine
one, and is optional given the workaround already passes with zero
divergence.

#### What is awkward

`ClashEnv::injectEnemy` hardcodes team 1:

```cpp
void injectEnemy(int cardId, float x, float y) {
    const auto* card = CardRegistry::getInstance().getCard(cardId);
    if (card) card->spawnEntity(x, y, 1, game.getBoard());
}
```

There is no ally equivalent, so our own placements must go through
`playCard`, which requires the card to be in the simulator's own hand — and
that hand is shuffled at reset by an unseeded `std::mt19937`, cannot be set,
and the queue behind it cannot be read (`getHand()` is team 0 only).

#### The workaround, and what it costs

`bridge/sim_driver.py` reverse-engineers the shuffle: draw ~20,000 resets
(0.135 ms each, measured), keep the ~200 whose hand-set matches the real
opening hand, carry them all forward, and eliminate the ones that could not
have dealt what reality dealt. After four confirmed deals every survivor has
our exact cycle, and from there the two FIFOs cannot diverge.

It works — 0 refusals, 0 desyncs, divergence 0 on the control. It costs:

- ~2.7 s of startup per match and ~1.4 s of redundant simulation (200
  environments stepping in lockstep until the pool narrows);
- ~150 lines whose only purpose is to undo a shuffle;
- a residual failure mode: `playCard` still checks elixir and placement
  legality against a board the estimate may have slightly wrong.

#### The change

Purely additive. Nothing existing changes behaviour:

```cpp
// ClashEnv.h -- injectEnemy stays exactly as it is, so no caller changes.
void inject(int cardId, float x, float y, int team) {
    const auto* card = CardRegistry::getInstance().getCard(cardId);
    if (card) card->spawnEntity(x, y, team, game.getBoard());
}

std::vector<int> getHandForTeam(int team) const { return game.getHand(team); }
```

```cpp
// bindings.cpp
.def("inject", &ClashEnv::inject,
     py::arg("card_id"), py::arg("x"), py::arg("y"), py::arg("team"))
.def("get_hand_for_team", &ClashEnv::getHandForTeam, py::arg("team"));
```

`game.getHand(int)` and `GameManager::playCard(team, ...)` already exist and
are already public. No effect on the observation vector, the action space, or
any checkpoint — a rebuilt `.pyd` stays compatible with current weights.

---

### UPSTREAM — Explicitly NOT requested

- **Game phases / overtime.** `ELIXIR_REGEN_RATE` is one `const float`
  applied to both players with no phase concept. The phase is carried as a
  field in `ClockState`, marked not-consumed, and left unconnected on
  purpose. Not asking the engine to grow phases.
- **The ~2% elixir gap.** `0.035/tick × 10 ticks/s` = 2.857 s per elixir
  against the real 2.8 — and 2.80 s is now *measured* off the recordings, not
  assumed. Changing it would be a gameplay change mid-training-run.
  `perception/` uses the real rate for the real opponent and the engine's
  rate when reasoning about the engine, and keeps them explicitly separate.
- **Card levels.** The registry has none; recorded matches do. That is why
  `readers/towers.py` takes max-HP as a caller input instead of assuming the
  engine's values.

---

### UPSTREAM item 5. DONE — team 1's observation is displaced one row (proposed 2026-07-31, applied; status corrected 2026-08-19)

> Applied. `ClashEnv::extractObservationForTeam` now reads
> `static_cast<int>(std::floor((BOARD_HEIGHT - 1) - entity->position.y))` — the
> proposed edit below, verbatim. The header said `OPEN` until 2026-08-19; see
> the status audit at the top of this file. Gameplay-affecting: Elo and win
> rates are not comparable across it.

Original proposal, kept for the record:


`ClashEnv.h`, `extractObservationForTeam`:

```cpp
int rawY = static_cast<int>(entity->position.y);
int y = (team == 0) ? rawY : (BOARD_HEIGHT - 1 - rawY);
```

It mirrors the **truncated row** instead of truncating the **mirrored
position**. `33 - int(y)` and `int(33 - y)` agree only when `y` is an integer,
and disagree by exactly 1 otherwise. Every troop in play sits at a fractional
`y`, so this is a permanent, every-tick, every-entity error — and it applies to
team 1 only. Team 0's path is `rawY`, untouched.

This is why the two towers behave differently and why the bug survived the
2026-07-30 geometry audit: Princesses are at `y = 27.0` (integer, mirrors
correctly), Kings at `y = 30.5` (fractional, off by one). The positions
themselves are symmetric — item 0 verified that — so a coordinate audit finds
nothing. Only the encoder is wrong.

#### Measured

Mirror-image Valkyrie pairs injected at `y` and `33 - y`, one tick to merge
`pendingEntities`, then both observations read:

| team 0 `y` | team 1 `y` | team 0 sees | team 1 sees |
|---|---|---|---|
| 8.0 | 25.0 | own row 8, enemy 24 | own row **9**, enemy **25** |
| 8.5 | 24.5 | own row 8, enemy 24 | own row **9**, enemy **25** |
| 10.3 | 22.7 | own row 10, enemy 22 | own row **11**, enemy **23** |
| 12.5 | 20.5 | own row 12, enemy 20 | own row **13**, enemy **21** |

At reset, with no entities at all, `get_observation_for_team(0)` and
`(1)` differ in 56 cells — King towers at rows 2 vs 3 and 30 vs 31, plus their
attribute channels.

**Consequence, measured directly.** The frozen main agent played against an
identical copy of itself, same policy both sides, no gradient, 400 episodes:

```
team 0 score 0.598   (238 W / 2 D / 160 L)   95% CI [0.548, 0.647]   z = +3.90
```

A policy beats *itself* 60/40 purely by being assigned team 0.

#### Blast radius — smaller than it looks

**Team 0's observation is bit-identical before and after.** The `team == 0`
branch is not touched, so the trainee's own inputs never change and
`model_weights_selfplay.pth` stays valid — this is not a checkpoint-invalidating
change in the usual sense.

What does change: every neural and scripted opponent starts seeing the board
correctly, so **opponents get stronger**. Expect measured win rates to fall.
Specifically:

- Self-play win rates vs the PFSP pool are currently inflated, and PFSP weights
  opponents by `(1 - winrate)^2`, so the sampling distribution is distorted too.
- The 3 *historical neural* Elo anchors get stronger; the 3 *builtin heuristic*
  anchors are unaffected (`HeuristicOpponent` is C++ and never reads the
  observation). Elo is therefore **not comparable across this fix**.
- Pipeline 1 (`train.py` vs `HeuristicOpponent`) is entirely unaffected.
- Both league-exploiter bursts to date measured this, not exploits: burst #1
  scored 0.585 against what is really a 0.598 null, i.e. nothing.

#### Proposed edit

```cpp
int y = (team == 0)
    ? static_cast<int>(entity->position.y)
    : static_cast<int>(std::floor((BOARD_HEIGHT - 1) - entity->position.y));
```

`std::floor` rather than a bare cast so an entity behind the back row gives
`-1` and is rejected by the existing bounds check, instead of truncating
toward zero into row 0.

### UPSTREAM item 6. DONE — river marker row differs between perspectives (applied; status corrected 2026-08-19)

> Applied. `ClashEnv.h` now reads `constexpr int riverRow = 17;` for both teams
> — the proposed edit below, verbatim. The header said `OPEN` until 2026-08-19;
> see the status audit at the top of this file.

Original proposal, kept for the record:


Same function, the marker is drawn on one hardcoded row:

```cpp
int riverRow = (team == 0) ? 17 : (BOARD_HEIGHT - 1 - 17);   // = 16
```

Team 0 sees the river/bridge marker at row 17, team 1 at row 16 — 36 differing
cells in channel 8 at reset. A network trained as team 0 expects the bridges one
row further forward than team 1 shows it.

**Proposed:** `int riverRow = 17;` for both. Each team's own frame is supposed to
be identical, and keeping 17 leaves team 0's observation unchanged.

Whether 17 is the *right* row for a band of `[15.5, 17.5)` is a separate
fidelity question and deliberately not bundled here.

---

### UPSTREAM item 7. DONE — the engine can be seeded (proposed 2026-07-31, applied and verified 2026-08-21)

**Not a correctness bug. A cost multiplier on every experiment this project
runs**, including the ones `CLAUDE.md` already recommends re-running.

> **⚠ 2026-08-21: THE EDIT ORIGINALLY PROPOSED BELOW WOULD NOT HAVE FIXED THE
> OPENING HAND.** It seeds `ClashEnv::rng`, which feeds only
> `HeuristicOpponent`. The opening-hand shuffle runs on `GameManager::rng` — a
> **second, independent** `std::mt19937`, also seeded from `std::random_device`
> at construction. Anyone applying the one-liner and then testing two envs for
> an identical opening hand would have found it still random, and the natural
> conclusion ("seeding doesn't work") would have been wrong. The corrected edit
> is in "The change" below and covers both generators.

#### What is there now

TWO independent unseeded generators, neither reachable from Python:

```cpp
// include/core/ClashEnv.h:132 / :383   -- feeds HeuristicOpponent only
std::mt19937 rng;
... rng(std::random_device{}()) { heuristicOpponent.reset(rng); }

// include/core/GameManager.h:50 / :202 -- feeds the OPENING HAND
std::mt19937 rng;
... : gameOver(false), loserTeam(-1), rng(std::random_device{}()) {
```

`GameManager::reset()` (line 589-590) is where the hand is dealt:

```cpp
playerAI.initializeDeck(aiDeckConfig, rng);
playerOpponent.initializeDeck(oppDeckConfig, rng);
```

and `PlayerState::initializeDeck` shuffles a permutation of the 8 deck indices
with `std::shuffle(order.begin(), order.end(), rng)`, taking the first four as
the hand and **the rest as the starting `deckQueue` order**. So the seed governs
both the opening hand *and* the cycle order — which for 2.6 Hog Cycle is the
more important half.

`MicroRoyaleEnv.reset(seed=...)` looks like it should help but only forwards to
`gymnasium.Env.reset`, which seeds the *wrapper's* RNG, not either engine one.

`grep -c seed src/bindings.cpp` finds two hits, both in comments. There is no
binding.

#### What it costs, measured on the experiment that prompted this

Comparing two observation variants (extra scalar 2 correct vs zeroed) against
`heuristic@1.35`. Because the arms cannot share a seed, the comparison is
unpaired and needs

```
n = 2 * (1.96 + 0.84)^2 * 0.25 / 0.05^2  ~=  1568 episodes per arm
```

to resolve a 5-point win-rate difference at 80% power — **3,136 episodes**. With
a shared seed the same episodes become matched pairs, most of the variance is
the shared opening hand and opponent rolls rather than the treatment, and the
same resolution needs roughly an order of magnitude fewer games.

#### NEW EVIDENCE, 2026-08-20: it invalidated a control and nearly produced a wrong conclusion

`env.snapshot()` (2026-08-11) removed this as the blocker on *paired* A/B tests
**within a single process**, and `CLAUDE.md` recorded that correctly. What it
does NOT give is comparability **across process invocations**, and that gap has
now cost something concrete.

A combo-family ablation in `python_ai/eval/prove_combos.py` was designed with a
built-in validity check: run 4 shares `--seed 300` with run 3, so its untreated
OFF arm should reproduce run 3's OFF arm exactly. It read **0.475 against
0.537**. Nothing was wrong with either run — `--seed` reaches only the
*teachers'* `numpy` RNG, so the two invocations drew entirely different match
populations, and the arm levels were never comparable in the first place.

Two runs had also happened to report the same OFF arm (0.537 twice, at different
seeds), which made the design look sound until it was tested. **The failure mode
is that a mis-specified control looks like a failed comparison.**

Consequences carried in `CLAUDE.md` and the harness docstring: within a run the
snapshot pairing is sound and the paired delta is valid; across runs only
**deltas** are comparable, never arm levels. Any "re-run at the same seed and
check the baseline matches" design in this repo is invalid until this lands.

#### Second benefit: reproducible failures

A self-play regression currently cannot be replayed. The 2026-07-31 team-1
observation bug was found by running a policy against a bit-exact copy of itself
and noticing 0.598 where 0.500 was expected — a test `CLAUDE.md` now recommends
after any change to the observation, the board, or `stepSelfPlay`. That test is a
coin-flip null measured over hundreds of games precisely because individual games
cannot be reproduced.

Also live: `python_ai/tests` has a nondeterministic **skip count** (347-348 pass,
1-2 skip across identical runs) because two cases depend on this shuffle.

#### The change — CORRECTED, both generators

```cpp
// include/core/GameManager.h  (public)
void seed(unsigned int s) { rng.seed(s); }

// include/core/ClashEnv.h     (public)
// Seeds BOTH generators and re-deals, so the opening hand and cycle order are
// pinned as well as the heuristic's rolls. reset() is what calls
// initializeDeck, so seeding without it would leave the CURRENT hand untouched
// and only affect the next episode -- the surprising half of this API.
void seed(unsigned int s) {
    rng.seed(s);
    heuristicOpponent.reset(rng);
    game.seed(s ^ 0x9E3779B9u);
    reset();
}
```

The `0x9E3779B9` offset keeps the two streams from being identical, which
matters because both are `std::mt19937` and one of them shuffling first would
otherwise correlate the heuristic's choices with the hand.

Plus the binding:

```cpp
.def("seed", &ClashEnv::seed, py::arg("seed"))
```

and an optional forward from `MicroRoyaleEnv.reset(seed=...)`, which is where a
caller already expects it.

**The ordering subtlety is the part to get right.** `initializeDeck` runs inside
`GameManager::reset()`, which the `GameManager` *constructor* also calls. So a
seed applied after construction only takes effect on the next `reset()` — hence
the explicit `reset()` in the edit above. A `seed()` that did not re-deal would
look like it silently did nothing.

#### Acceptance test, already written

`python_ai/tests/test_engine_seeding.py::test_two_envs_with_the_same_seed_deal_the_same_opening`
is committed and **skips** with a clear reason while `seed` is unbound. It
asserts that two envs seeded alike produce identical hands for BOTH teams and
identical cycle order over a full rotation, and that two different seeds
actually differ (so it cannot pass vacuously against a degenerate shuffle).
Rebuild the `.pyd` and it runs.

**Blast radius:** additive. Nothing existing calls it, so unseeded behaviour is
byte-identical and no checkpoint is affected. Not gameplay-affecting, so
`model_weights.pth`'s win-rate history stands.

**APPLIED AND VERIFIED 2026-08-21.** Both generators are seeded, `ClashEnv::seed`
re-deals, and the binding is live. `test_engine_seeding.py` went from 4 skips to
**5 passed / 1 skipped**, the remaining skip being the diagnosis test that
retires itself once `seed()` exists. Full suites after the change: Python
**364 passed / 3 skipped**, C++ **550 cases, 5,341 assertions, all passing**.

The two shuffle-dependent Python tests that used to make the skip count
nondeterministic can now be pinned; that is a follow-up, not part of this item.

*Historical note on why this sat open so long:* the sessions that filed it
believed this machine had no C++ toolchain. It does — VS 2022 Community, just
not on PATH. See CLAUDE.md's environment section for how that error was made.

**Original confidence note, kept:** the cost is measured twice; the exact edit
was filed rather than applied, per `CLAUDE.md` (no `cl`/`cmake`/`msbuild`/`g++`/`clang++`, both Visual
Studio directories empty, WSL not installed), so it could not have been
compiled or tested here even if the rule allowed it.

---

### UPSTREAM item 8. OPEN — Fireball misses the Musketeer kill by 32 HP (proposed 2026-08-06)

**This is filed as a decision to make, not a defect to fix.** It may well be
correct as-is, and closing the gap would trade fidelity for learnability.

#### The measurement

`CardRegistry.h` gives Fireball 689 damage at 4 elixir, radius 2.5. Against
`DEFAULT_DECK`:

| target | cost | HP | dies to 689? |
|---|---|---|---|
| Minions | 3 | 230 x3 | yes |
| Archers | 3 | 304 x2 | yes |
| **Musketeer** | **4** | **721** | **no — survives on 32 HP (4.4%)** |
| Cannon | 3 | 824 | no |
| Mini P.E.K.K.A | 4 | 1390 | no |
| Valkyrie | 4 | 1907 | no |
| Giant | 5 | 3968 | no |

So **every clean Fireball kill in this matchup is a 4-elixir spell killing a
3-elixir card** — a -1 elixir trade that also generates no board presence. The
only target that would make it a clean 4-for-4 survives by 32 HP. Breaking even
requires hitting two cards at once.

That fully explains the behaviour recorded in CLAUDE.md: the policy plays
Fireball on ~0% of steps, and forcing it dropped win rate 97% -> 23%. The low
weighting was never a learning failure. It is a correct valuation of a card
that is negative-EV in its common case.

#### Why this is not obviously a bug

689 and 721 appear to be the real tournament-standard (level 11) Clash Royale
values, in which Fireball genuinely does not one-shot a Musketeer — it needs any
chip damage on top (a tower hit, a Zap, one arrow volley). **Needs confirming
against current real-game data before anything is changed.** If it holds, the
engine is right and the awkward EV is a real property of the card.

That matters more here than in most engines, because `perception/` exists
specifically to drive this simulator from real matches. A sim where Fireball
one-shots Musketeers is a sim whose spell decisions do not transfer.

#### Options

1. **Change nothing.** Correct if the values are faithful. Fireball stays a
   situational two-for-one card, which is what it is in the real game, and the
   agent's low usage is right rather than pathological.
2. **Fireball 689 -> 725.** Makes it a clean 4-for-4. Cheapest edit, but it
   diverges from the real game on the single most-used spell, and every
   Fireball interaction in every future deck inherits the divergence.
3. **Musketeer 721 -> 685.** Same effect, worse blast radius — it changes every
   matchup the Musketeer appears in, not just the Fireball one.

**Recommendation: option 1, and reach the behaviour through reward shaping
instead** (see the lethal-spell PBRS term and the elixir-value term in
`train.py`). Shaping changes what the agent *learns to value* without changing
what the game *is*, which keeps the perception bridge honest.

Blast radius if 2 or 3 is chosen: gameplay-affecting, so it invalidates the
win-rate history, and a `ClashRoyaleTests` run is required. The observation
layout is untouched, so checkpoints still load.

---

### UPSTREAM item 9. DONE — troop movement ran ~4-5x faster than the real game (applied and verified 2026-08-07)

**Applied with explicit sign-off**, as `MOVEMENT_SPEED_SCALE = 0.2f` in
`CardStats.h`, used at `CardRegistry.h:125` and both `SpiritEmpressForms.h`
sites that assign `speed` directly and so bypass the factory.

#### Verified three ways after the change

| check | before | after | target |
|---|---|---|---|
| reconstruction floor (IoU) | 0.519 | **0.685** | higher |
| engine:real speed ratio (p90) | 3.8x | **0.8x** | 1.0 |
| engine:real speed ratio (median) | 6.5x | **1.3x** | 1.0 |
| time-scale optimum @1.0s horizon | 0.2 | **1.0** | 1.0 |

The two speed statistics bracket 1.0 from opposite sides, which is the most
this data can resolve -- the median under-reads real speed and the p90
over-reads it.

**The time-scale peak moving from 0.2 to 1.0 is the decisive end-to-end
check**: it says the engine's clock and the real game's now agree, measured
against footage rather than against a constant.

Two predictions made before the change and confirmed after it: the
reconstruction floor rose on its own (part of it was the single materialising
tick, which at 5x speed displaced a unit by half a second of real movement),
and the Giant -- the Slow tier -- came out at 0.7x against the other classes'
0.8x, which is the documented flat-scale residual showing up exactly where it
was predicted.

#### What the C++ test suite did and did NOT tell us

All 504 cases / 4329 assertions pass unchanged. **This is not evidence the
change is correct.** `test_troop.cpp` constructs `MeleeTroop` directly with a
literal speed, and no test anywhere asserts a registry speed constant, so the
suite covers the movement MECHANISM and has zero coverage of the DATA. The
earlier estimate in this document -- that 207 speed/movement references across
16 test files implied the suite would need review -- mistook grep hits for
coverage. A guard now lives on the perception side
(`test_forecast.test_the_engine_moves_at_roughly_real_game_speed`).

#### Still open, NOT addressed by this change

  * **Projectile speed** (`Projectile.h:88`) is untouched and unmeasured. The
    harness is blind to spells -- they are filtered out before reaching the
    board -- and the value-Fireball and lethal-spell shaping both depend on
    when a spell arrives relative to the troops it is aimed at.
  * **Deploy time** is still absent; the real game freezes a troop ~1 s after
    it lands. A second timing error in the same direction.
  * The **flat-scale residual** on the Slow tier, ~20%, left deliberately.

#### Original evidence, kept for the record

#### How it was measured

`perception/tools/sim_fidelity.py`, over all 8 recordings, 159 paired samples.
Both sides of the comparison are measured, neither is read from a constant:

  * **engine** — inject a card into open ground, step, measure the centroid's
    displacement over a long enough baseline that cell quantisation is <10%
    (`forecast.SimForecaster.measure_speed`).
  * **real** — card classes showing exactly one body on one side in both
    frames of a pair, so displacement is unambiguous without a tracker,
    measured over the 3.0 s horizon.

| card | engine tiles/s | real p90 | ratio | n |
|---|---|---|---|---|
| Minions | 8.02 | 1.03 | 7.8x | 6 |
| Spear Goblins | 10.00 | 1.88 | 5.3x | 5 |
| Musketeer | 5.04 | 1.33 | 3.8x | 33 |
| Valkyrie | 5.04 | 1.33 | 3.8x | 23 |
| Giant | 2.88 | 1.00 | 2.9x | 45 |
| Mini P.E.K.K.A | 7.55 | 3.07 | 2.5x | 19 |
| Archers | 5.09 | 3.50 | 1.5x | 17 |

**The honest figure is a bracket, 3.8x-6.5x, not a point.** The two statistics
have opposite biases: the median under-reads real speed because units that
stop to fight contribute zeros, and the p90 over-reads it because a
mis-associated detection looks like a large jump (Archers at 3.50 tiles/s is
not a real Archer). Median ratio is 6.5x, p90 ratio 3.8x. Do not quote a
tighter number than the bracket from this data.

#### An independent cross-check that the measurement is real

The engine has an internal speed-tier convention, stated in its own comments
(`SpiritEmpressForms.h:28,43` — "Fast" 0.85, "Medium" 0.5):

| tier | engine/tick | engine tiles/s | real tiles/s |
|---|---|---|---|
| Slow (Giant) | 0.30 | 3.0 | ~0.75 |
| Medium (Musketeer, Valkyrie, Knight) | 0.50 | 5.0 | ~1.0 |
| Fast (Hog, Mini P.E.K.K.A, Minions) | 0.80-0.85 | 8.0 | ~1.5 |
| Very Fast (Skeletons, Goblins) | 1.00 | 10.0 | ~2.0 |

The engine's Slow:Medium ratio is 0.60 where the real game's is 0.75, so the
Slow tier is *relatively* too slow and should need a divisor 0.8x the size of
Medium's. Measured independently from the footage: Giant 2.9x against
Musketeer/Valkyrie 3.8x, a ratio of **0.76**. Two unrelated sources agreeing
to 5% is the reason to believe this is a scale error and not detector noise.

It also means **a single flat divisor is close but not exact.** Correcting per
tier is exact; a flat divisor leaves the Slow tier ~20-25% off.

#### What is NOT claimed

  * That this explains "the whole sim-to-real gap". The gap in *performance*
    has never been measured; only this discrepancy in *speed* has.
  * Exact target values. The measurement supports "~4-5x too fast". It does
    not support setting a specific constant to three digits.

#### Where the change would go

| site | what it covers |
|---|---|
| `CardStats::troop()` — `CardRegistry.h:125` | every troop built through the factory (the great majority) |
| `SpiritEmpressForms.h:28,43` | **bypasses the factory** and assigns `stats.speed` directly — a scale applied only in `troop()` would silently miss these two |
| `Troop::update` — `Troop.h:36` | the single point of application; scaling here covers everything at once, including any future bypass |
| `Projectile.h:88` | **separate `speed`, separate question — see below** |

`Troop.h:36` is the smallest and most complete edit (one line, covers every
mover, trivially reversible). Its downside is that `CardStats::speed` then
means something other than what it says. Scaling at the two construction
sites keeps the values honest but must touch both, and would be missed again
by the next card that bypasses the factory.

#### Blast radius — this is not "the sim gets more accurate"

Attack cooldowns are already correct against the real game (Musketeer 1.0 s,
Valkyrie 1.5 s, Hog 1.6 s — 132 rows agreeing, see CLAUDE.md) and elixir regen
is within 2%. Slowing movement while those stay fixed **rebalances every card
relationship in the engine**:

  * a troop crossing a defender's range takes ~5x longer, so it absorbs ~5x
    more shots — **ranged units get much stronger relative to melee**;
  * a tank takes ~5x more Princess Tower damage covering the same ground;
  * ~5x more elixir accrues while a push develops, which changes the economy
    the whole game is played on;
  * far fewer engagements fit in a match, so **timeouts get much more common**
    and `TimeoutRules`/`DRAW_PENALTY` become far more prominent;
  * `HeuristicOpponent` and the four scripted bots have thresholds that were
    only ever exercised against the fast physics;
  * the curriculum's 0.80 stage gate and the 1.0-1.5 elixir ladder were
    calibrated against the fast physics;
  * 207 references to speed/position/movement across 16 C++ test files
    (`test_troop.cpp` alone has 33) — the Catch2 suite will need review, and
    whether each failure is "asserting the old physics" or a real problem
    needs a human to read.

One genuine upside: `skip_frames = 10` gives one decision per second, which
CLAUDE.md lists as open problem #4 precisely because it caps tactical
precision. At 5x slower movement, one second covers 5x less board, so that
handicap shrinks by the same factor without any change to the action space.

**Every checkpoint is invalidated and the win-rate history means nothing
afterwards.** That is expected and accepted here — the plan is a fresh run.

#### Open questions, not answered by this measurement

1. **Projectiles.** `Projectile.h:88` uses its own `speed` and this harness
   cannot see them: spells in flight are filtered out before they reach the
   board (`is_board_presence`). Whether projectile speed carries the same
   error is **unmeasured**. It matters, because the value-Fireball shaping and
   the lethal-spell PBRS term both landed recently and both depend on when a
   spell arrives relative to the troops it is aimed at.
2. **Deploy time.** The engine has none — `spawnEntity` makes an entity live
   immediately, while the real game freezes a troop ~1 s after it lands. This
   is a second, independent timing error in the same direction, and correcting
   speed does not address it.

#### The time-scale sweep — a third, independent confirmation

`--time-scale N` steps the engine by `horizon * N`, which is arithmetically
what dividing every speed by `1/N` does to displacement. Sweeping it measures
the divisor with **no engine change at all**. All scales share one pass, so
every column is scored on identical boards and a difference between them
cannot be sampling. n = 158 per cell.

```
 horizon   stale  rebuilt    x0.1   x0.15    x0.2   x0.25   x0.33    x0.5      x1
    0.5s   0.341    0.258   0.258   0.258   0.258   0.258   0.253   0.253   0.172
    1.0s   0.250    0.187   0.187   0.187   0.216   0.216   0.203   0.141   0.114
    2.0s   0.178    0.135   0.138   0.134   0.143   0.152   0.122   0.111   0.091
    3.0s   0.134    0.110   0.094   0.097   0.104   0.105   0.091   0.075   0.060
```

**`x1` is the worst column at every horizon, and the curve has an interior
maximum at 0.2-0.25** on the two rows that can resolve it — i.e. a divisor of
**4x-5x**, agreeing with both the speed table and the tier cross-check.

Two rows cannot resolve it and must not be read as evidence. At `0.5s` every
scale <= 0.25 rounds to a single tick, so those columns ARE the rebuilt board
(hence the identical 0.258); at `3.0s` the differences are inside the noise.

#### What this does NOT show: forward prediction is not yet worth deploying

`rebuilt` is the same reconstruction scored WITHOUT stepping. Against it,
stepping at the right scale genuinely adds information — 0.216 vs 0.187 at
1.0 s, 0.152 vs 0.135 at 2.0 s. So the dynamics do carry real signal.

But **no scale beats `stale`**, which pays no reconstruction cost at all
(perception at t vs perception at t+h). The reconstruction tax — floor 0.519 —
is larger than everything stepping buys back. So the decision loop should keep
acting on the freshest real board, and the lookahead idea stays parked.

It is worth re-asking after this item lands: part of that floor is the one
unavoidable materialising tick, which at 5x speed displaces a unit by half a
second of real movement. Correcting speed should raise the floor on its own.

#### How to verify a fix, before spending any training compute

`sim_fidelity.py` is the instrument, and it needs no engine change to predict
what the fix will do: `--time-scale N` steps the engine by `horizon * N`,
which is arithmetically what dividing speed by `1/N` does to displacement.

  1. sweep `--time-scale` and find the scale that maximises occupancy
     agreement — that scale IS the empirical divisor;
  2. apply the change;
  3. re-run with `--time-scale 1.0` and confirm the ratio column collapses
     toward 1.0 and the reconstruction floor rises;
  4. only then retrain.

Step 1 costs one 20-minute pass and no engine change at all.

---

### UPSTREAM item 10. DONE — state-estimator write half: `set_elixir_for_team` / `set_hand_for_team` (applied 2026-08-17, recorded here 2026-08-19)

**Recorded after the fact.** These landed in `26de409` ("Engine: set_elixir /
set_hand, so search can evaluate the REAL position") without ever being written
up here — the numbering jumped 9 → 12 and items 10 and 11 simply did not exist.
Filling 10 so the file stops implying nothing happened between them. Same class
of gap as the item 5/6/12 status drift noted at the top of this file.

#### What landed

```cpp
// ClashEnv.h
void setElixirForTeam(int team, float value) { game.setElixir(team, value); }
bool setHandForTeam(int team, const std::vector<int>& cards) {
    return game.setHand(team, cards);
}
```
```cpp
// bindings.cpp
.def("set_elixir_for_team", ...)
.def("set_hand_for_team",  ...)
```

#### Why it was needed

The read half (`inject`, item 4) could rebuild the BOARD, but elixir and the
hand still came from `reset()` — so a reconstructed position carried the right
units and a **fabricated** hand and elixir. `ClashEnv.h`'s own comment names
`perception/forecast.py` as the motivating case. Decision-time search over such
a state scores fiction rather than the real position, which is what made this
the blocking half rather than a convenience.

#### `setHandForTeam` REFUSES rather than accepting a misread

It returns `false` and mutates nothing when the size is wrong, when a card is
not in `player.hand + player.deckQueue`, or on a duplicate. **Callers must check
the return.** This matters on real perception output: measured over 2,510 live
frames, ~0.5% off-deck reads and ~1.7% duplicate reads survive match gating, so
the guard fires in practice — it is not theoretical. A silently-ignored refusal
would leave the estimator confidently wrong.

#### Current consumer status

`perception/forecast.py` has **not** been migrated to these, deliberately: it
consumes only `game_state.units`, and both of its consumers score tower-excluded
unit occupancy, so the fabricated hand/elixir sit outside every reported metric
today. There is also no team-1 hand source on this side at all (team-1 *elixir*
does have one — `track/opp_elixir.py`, and live, the net's own
`predict_opp_elixir`, which is exactly what ClashEnv's comment anticipates).
Migrate when a consumer starts scoring something hand- or elixir-dependent.

`perception/bridge/sim_driver.py` still reverse-engineers the opening hand by
repeated `reset()` draws — `set_hand_for_team` is the intended replacement for
that ~150-line workaround, not yet applied.

---

### UPSTREAM item 12. DONE — bind `isValidPlacement`, so the action-space mask can stop disagreeing with the engine (proposed 2026-08-11, applied; status corrected 2026-08-19)

> Applied. `ClashEnv::isValidPlacementForCard` exists and `bindings.cpp`
> registers `.def("is_valid_placement", ...)` — the proposed edit below,
> verbatim. The summary table already said DONE; only this header was stale.
> The Python side that consumes the mask is gameplay-affecting, so win rates
> are not comparable across it.

Original proposal, kept for the record:


**The ask is one read-only accessor.** No gameplay change, no checkpoint
invalidation, no behavioural difference to any existing caller. Everything
else in this item is Python-side and is not being asked for here.

#### The measurement

`model.py`'s `placement_mask` grants a troop every one of the 16 own-half
rows. `GameManager::isValidPlacement` does not. Measured on the live
pipeline-2 checkpoint at ep ~45,800, over 1,340 decision steps:

| | |
|---|---|
| chose no-op | 72.9% |
| chose a card | 27.1% (363) |
| ...**accepted by the engine** | **41.3%** (150) |
| ...**silently rejected** | **58.7%** (213) |
| of those, affordability leaks | **0** |

`playCard` returns `false` with no exception and no signal, so a rejected
action is indistinguishable from a no-op in its effect and its advantage is
pure noise entering the gradient. This is the same failure class the
affordability mask was introduced to close, still open on the placement axis.
The affordability half is airtight -- zero leaks in 363 draws.

**94.8% of the rejections are on row y=0**, which is `Board::isBackRowDeadZone`
and is deliberate engine design, not a bug. The remaining 5.2% are the tower
footprints.

#### It is entirely static, and per card class

Probed cell by cell over the own half:

| | legal cells | row-0 columns |
|---|---|---|
| Cannon (building) | 208/288 | 6, 7, 11 |
| Archers / Giant (troop) | 242/288 | 6-11 |
| Fireball (spell) | 276/288 | 6-11 |

and legality does **not** depend on board state:

| | legal cells |
|---|---|
| empty board | 208/288 |
| six troops deployed | **208/288** |
| cells lost to units | **0** |

So the correct mask is three constant tables, built once at startup. There is
no runtime query and no per-step cost -- which is why this needs an accessor
and not a fast path.

#### Why Python cannot just compute it

`isValidPlacement` combines `board.isBackRowDeadZone`, the board bounds, the
per-card `placementRadius`, `isSpell` and `deployAnywhere`, and the tower
footprint clearance in `CardFactories.h`. Reproducing that in Python is a
second copy of engine geometry, which this file already carries two incidents
about (items 1-2, 5-6) and which CLAUDE.md explicitly forbids. `get_card_info`
already exposes the per-card half; the predicate itself is the missing piece.

#### The exact edit

`include/core/ClashEnv.h`, beside the existing forwarders at line 393:

```cpp
    // Read-only. Exposed so the Python action space can be built from the
    // ENGINE's legality rule rather than a second copy of it -- see
    // UPSTREAM_REQUESTS item 12. Pure query: no state is touched.
    bool isValidPlacementForCard(int cardId, float x, float y, int team) const {
        const CardDefinition* def = CardRegistry::getInstance().getCard(cardId);
        if (!def) return false;
        return game.isValidPlacement(team, x, y, def->isSpell,
                                     def->placementRadius, def->deployAnywhere);
    }
```

`src/bindings.cpp`, beside the existing accessors at line 73:

```cpp
        .def("is_valid_placement", &ClashEnv::isValidPlacementForCard,
             py::arg("card_id"), py::arg("x"), py::arg("y"), py::arg("team") = 0,
             "Would playCard accept this card at this point? The exact "
             "predicate playCard uses, exposed so the Python placement mask "
             "is derived from it instead of re-deriving board geometry.")
```

`isValidPlacement` is already public and already `const`. Nothing else moves.

#### Blast radius

Additive only. No existing symbol changes signature or behaviour, no
gameplay path is touched, and `model_weights.pth`'s win-rate history is
unaffected by the binding itself.

**The Python mask fix that consumes it is a different matter and is
gameplay-affecting**: removing 58.7% of the policy's card choices from the
action space redistributes probability mass onto real plays, so the play rate
and the elixir economy will both shift. Win rates are not comparable across
it. That is expected and is the point, but it should be stated when it lands.

**Not yet established:** that this defect is *why* the agent turtles or why
the Giant is starved. It is a large real defect that was invisible; the causal
claim needs re-measuring after the fix, not before. An earlier diagnosis of
the same behaviour as a reward-hacking elixir dump was measured and refuted --
a rejected placement spends no elixir at all -- and that mistake is the reason
this item leads with the measurement rather than the story.

#### Rebuild note

The `.pyd` post-build copy into `python_ai/` fails with MSB3073 while any
Python process has the module loaded, so the live trainer must be stopped for
the copy step. Compilation itself can be verified without stopping it.

---

### UPSTREAM item 13. DONE — state snapshot/restore, so decision-time search becomes possible (applied and verified 2026-08-11)

**Applied as proposed below**, with one design change found during
implementation. What landed:

| site | what it does |
|---|---|
| `Entity::snapshot()` (`Entity.h`) | new virtual, **throws `std::logic_error`** naming the offending type via `typeid` |
| `Entity::remapSnapshotReferences()` (`Entity.h`) | new virtual, default no-op, second pass over a fresh copy |
| the 8 concrete types | one line each: `make_shared<T>(*this)` |
| `Projectile` | overrides the remap; gains `getTargetId()` for the tests |
| `Board::deepCopy()` (`Board.h`) | the copy itself |

**The design change:** the remap is a **virtual on `Entity`**, not a
`dynamic_pointer_cast<Projectile>` inside `Board::deepCopy`. It has to be —
`Projectile.h` includes `Board.h`, so `Board` **cannot** include `Projectile.h`
to know what a projectile is. Routing it through a virtual also matches the
idiom `Entity` already uses three times (`onDeath`, `clampPosition`,
`onNearbyDeath`) for exactly this "let Board act without knowing the concrete
type" problem, and it keeps `Projectile::target` private — no public setter for
a member that nothing else should ever write.

#### A second aliasing case, not in the proposal below, found while implementing

`Board::statsEvents` is a `StatsEventBus` holding `shared_ptr<IStatsObserver>`,
and unlike the effects those collectors are **stateful** — running damage
totals, kill attribution, match outcome. Copying the subscriber list would post
every hypothetical hit in every rollout into the **real match's** statistics,
and those feed the reward shaping, so a search would silently rewrite the
returns it was being scored against. Exactly the `Projectile::target` failure
shape one level up, and it would have been just as invisible.

`deepCopy` starts the copy with an empty bus. A caller that wants stats off a
rollout subscribes its own collectors to the copy. Covered by
`"deepCopy does not carry the stats subscriber list"`.

#### Verified

`tests/core/test_board_deepcopy.cpp` — 13 cases, 323 assertions. Full suite
**520 cases / 4756 assertions, 0 warnings** under the existing `-Wall -Wextra`.
The pre-existing 507 cases pass unchanged.

The acceptance test is the zero-divergence control this section asked for:
snapshot a mid-game board (6 towers, both lanes pushing with `DEFAULT_DECK`
cards, projectiles in flight), step original and copy with identical inputs for
120 ticks, and require every entity to match **exactly** — id, hp, team, cardId,
position and projectile target — compared in **vector order**, since
`resolveCollisions` walks `activeEntities` as an ordered `i<j` loop and two
boards holding the same entities in a different order drift apart on their own.

**The negative case reproduces the corruption rather than describing it.** A
snapshot-only copy (no remap) with a shot in flight is stepped, and the
**original** board's troop is the one that loses 250 hp while the copy's is
untouched — the live game damaged by a simulation nobody stepped it in.

**Mutation-tested, so the suite is known to have teeth:** with the remap pass
commented out, **5 of the 13 cases fail**, including the 120-tick divergence
test and "stepping a deep copy leaves the original untouched". Restored and
re-run green.

**One test was flaky at ~1 run in 20 and the flake was mine, not the
engine's.** "Two snapshots given different actions diverge from each other"
stepped both branches 100 times and required the entity lists to differ. They
genuinely RECONVERGE: a lone troop with no support walks into the enemy
Princess Towers, dies without landing a hit, and elixir re-caps at 10, so both
boards end up holding the same six full-health towers and nothing else. The
assertion was false about Clash Royale, not about the snapshot. Fixing it
surfaced a second one immediately -- `playCard` queues into `pendingEntities`
and `getEntities()` exposes only `activeEntities`, so a successful play is
invisible until the next `step()` commits it, and comparing before that shows
two identical boards. Now asserted over a short horizon after one commit step,
and **verified across 200 consecutive full-suite runs**.

Worth recording because a single green run would have shipped both: a suite run
once is not a suite that passes, and "the futures diverge forever" is the kind
of assumption that is obviously wrong once stated and invisible until it flakes.

#### Blast radius, as measured rather than estimated

**Zero deletions in every C++ file** — `git diff --numstat` reports `+82/-0`,
`+66/-0`, `+48/-0` and so on across all ten headers. Not one existing line was
modified, so no existing code path can behave differently. **Not
gameplay-affecting: `model_weights.pth`'s win-rate history stands, and no
checkpoint is invalidated.** The rebuilt `.pyd` reports `observation_size()`
13606, unchanged, and plays a full match normally.

#### The manager/env layer — also landed, same day

`GameManager::snapshot()` and `ClashEnv::snapshot()`, the latter bound to
Python as **`env.snapshot()`**. `GameManager` copy-constructs (every remaining
member is already a value type — ticks, flags, `rng`, deck configs, both
`PlayerState`s) and then replaces the two members holding `shared_ptr`.

Two implementation notes worth keeping:

**`ClashEnv::snapshot` needs a tagged constructor, not copy-then-fix.**
`GameManager` holds `const float ELIXIR_REGEN_RATE`, so its implicit copy
*assignment* is deleted and `game = other.game.snapshot()` does not compile.
Copy-*initialising* it in the member list works, and in C++17 the prvalue is
elided straight into place.

**The replay logger deliberately does NOT carry across.** `GameLogger`
accumulates a `TickSnapshot` per tick with an `EntitySnapshot` per entity, so by
mid-match it is the largest thing in the object; copying a thousand ticks of
history to simulate twenty is what makes lookahead look infeasible when it is
not. A rollout is also a hypothetical, and its ticks do not belong in a replay
of the real match.

**A third aliasing hazard, and the mirror-image trap next to it.**
`MatchStatistics` holds `shared_ptr` to *stateful* collectors, so the copy must
not share them — same shape as `Projectile::target`, one level up again. But
the obvious fix, calling `attach()` on the copied board, is equally wrong in the
opposite direction: `attach()` builds **fresh zeroed** collectors, and since
train.py's tower term is potential-based over *cumulative* damage, every
candidate would then score as the same enormous instant loss — a search that
looks like it works and never prefers anything. `snapshotFor()` deep-copies each
collector instead. Both failure directions have their own test.

#### Measured from Python, on the rebuilt `.pyd`

| | cost |
|---|---|
| `env.snapshot()` | **0.033 ms** |
| one `step()` (10 ticks) | 0.027 ms |
| K=12 candidates at a 2 s horizon | **1.1 ms** |
| one `MicroRoyaleNet` forward | ~50 ms |

So the simulation is free and the **scoring** is the entire budget — the
opposite of the usual assumption, and the reason `python_ai/eval/search_ab_test.py`
batches all K candidate evaluations into a single forward.

#### This partly supersedes item 7 (RNG seeding)

Item 7 asks for `seed()` because unpaired A/Bs need ~1,568 episodes per arm.
`snapshot()` delivers the pairing directly: reset once, snapshot, and hand both
arms a bit-identical opening — same shuffled hand, same heuristic lane. Item 7
is still worth having for reproducible *failures*, but it is no longer what
blocks paired experiments.

#### The Catch-22 below is now RESOLVED, and the answer was yes

This section originally recorded that the evidence justifying snapshotting was
unobtainable without snapshotting: the one attempt to size search's value gave
`+0.0145 +/- 0.2162` over 45 constructed states, a confidence interval **15x
wider than the effect**, and the honest reading was "underpowered null", not
"search does not work".

With the mechanism built, the same question answered cleanly.
`python_ai/eval/search_ab_test.py`, 160 **paired** trials (both arms handed a
bit-exact copy of one reset), 1.5x opponent elixir, ep-64k checkpoint,
K ~= 3 candidates at a 4 s horizon, scored by the network's own critic:

| | |
|---|---|
| greedy policy win rate | **0.625** |
| + 1-ply search | **0.944** |
| paired delta | **+0.319**, 95% CI [+0.237, +0.401] |
| discordant pairs | 56 search-better / 5 search-worse / 99 tied |
| exact McNemar | **p = 5.6e-12** |
| deviation rate | 13.8% (5,764 of 41,792 decisions) |
| cost | 2.2x wall clock |

Three things about *why* this worked where the earlier attempt could not:

  * **Pairing, which is what snapshot() bought.** The earlier design compared
    one improved action diluted across ~80 subsequent policy actions. This one
    compares whole episodes that share an opening, so search acts at every
    decision and the shared variance is removed rather than averaged over.
  * **An opponent with headroom.** At 1.0x elixir both arms win ~100% and the
    delta is exactly zero -- by ceiling, not by search being useless. Measured,
    4/4 trials at 1.000 vs 1.000, before switching to 1.5x.
  * **Greedy is candidate 0**, so search deviates only when the critic prefers
    something else, and any loss it takes is the critic being wrong.

**The interpretation matters more than the number.** The scorer is the same
network's critic, so search beating the network's own action head by 32 points
says the **value head is much better than the action head is at exploiting it**.
That is exactly the gap expert iteration exists to close, and it locates the
underfit in the policy head rather than the critic.

**Still not established:** one checkpoint, one deck, one opponent, one horizon;
nothing about neural opponents in the PFSP league; and nothing about whether
distillation back into the policy works, which is the only experiment that
would change training.

---

### UPSTREAM item 13 (original proposal, kept for the record) — state snapshot/restore (proposed 2026-08-11)

**What is being asked for:** a way to copy a `GameManager` deeply, so a caller
can try several candidate actions from one position and keep the best. Today a
copy is shallow — `Board` holds `std::vector<std::shared_ptr<Entity>>`, so the
copy shares every entity with the original and stepping one corrupts the other.

#### Why this is worth reading despite the blast radius

A fast deterministic simulator is the project's biggest unexploited asset.
Combat has no RNG at all (the only randomness is `PlayerState::initializeDeck`'s
shuffle and `HeuristicOpponent`), so rolling a candidate action forward gives
*exactly* what would happen. That is a strict policy-improvement operator, and
`python_ai/trainers/bc_pretrain.py` — already built, schema pinned, verified end to end —
is exactly the consumer needed to distil the result back into the policy.

Two measurements taken 2026-08-11 size it:

| | cost |
|---|---|
| 20-tick (2 s) engine rollout | **0.34 ms** |
| one `MicroRoyaleNet` forward (CPU) | **50.51 ms** |

The engine is **~150x cheaper than the network that evaluates it**. Search here
is not compute-bound on simulation at all: a 10-second rollout of 12 candidates
costs ~20 ms. That asymmetry is what makes lookahead attractive on this
hardware, and it is the opposite of the usual assumption.

Candidates measurably differ. Over 24 constructed mid-game positions, K=12,
10 s horizon:

| | |
|---|---|
| observable damage swing spread (max-min) | mean **508.7** |
| critic value spread (max-min) | mean **0.738** |
| the two scorers pick the same best candidate | **50%** |

A ~500 damage swing is roughly 15% of a Princess Tower. There is real signal to
choose between.

#### The correction to the existing plan — `Entity::clone()` ALREADY EXISTS

`CLAUDE.md` records this item as "needs a virtual `Entity::clone()` across the
whole hierarchy — invasive simulation-core surgery". That is out of date.
`include/entities/Entity.h:113` already declares:

```cpp
virtual std::shared_ptr<Entity> clone(int newId) const { (void)newId; return nullptr; }
```

overridden in `MeleeTroop`, `RangedTroop`, `BuildingTargeter` and
`RangedBuildingTargeter`, each as three lines of implicit-copy-constructor:

```cpp
auto copy = std::make_shared<MeleeTroop>(*this);
copy->id = newId;
copy->hp = 1;              // Clone card: full damage, 1 hp
return copy;
```

**So copy-construction of concrete entity types is already relied on in
production code.** The mechanism is proven; only its coverage and semantics are
wrong for snapshotting.

#### Why the existing `clone()` must NOT simply be reused

Two reasons, both of which would corrupt a snapshot silently:

1. **`copy->hp = 1`.** It implements the Clone *card*, whose duplicates have 1
   HP by design. A snapshot needs HP preserved exactly.
2. **The default returns `nullptr`, not an error.** Buildings, Towers,
   `AreaSpell` and `Projectile` have no override, because they were never valid
   Clone targets in the real game. A naive "clone every entity" loop would
   therefore produce a board that has **silently dropped every tower, building,
   spell and projectile** — and would look like a working snapshot.

#### Proposed shape (the human decides the details)

- a second virtual, e.g. `virtual std::shared_ptr<Entity> snapshot() const`,
  preserving id and hp exactly, implemented for **every concrete type**
- its base implementation should **fail loudly** (assert / throw), never return
  `nullptr`, so a future entity type cannot silently punch a hole in a snapshot
- `Board::deepCopy()` rebuilding `activeEntities`, `pendingEntities` and
  `idCounter`
- `GameManager` snapshot = that board copy plus its scalar members
  (`currentTick`, `gameOver`, `loserTeam`, `oppElixirMultiplier`, `rng`, the
  deck configs, tower-troop types, `stats`, `playerAI`, `playerOpponent`),
  which are already value types

Estimated size: roughly 3 lines per concrete entity type across ~8-12 classes,
plus the board/manager plumbing. That is meaningfully smaller than "a virtual
clone across the whole hierarchy plus effects".

#### The risk I could NOT clear from the outside, and it is the crux

`CombatEntity` holds effects as shared pointers — `onHitEffects`, `deathEffect`,
`periodicEffect`, `onDamageTakenEffect`, `onHitSpawnEffect`,
`transformDeathEffect`, `abilityEffect`. An implicit copy constructor copies the
*pointers*, so a snapshot would **share** those effect objects with the
original.

- if effects are stateless strategy objects, sharing is correct and desirable
- if any effect carries mutable per-instance state, stepping the snapshot
  mutates the original, and the corruption is silent

**VERIFIED 2026-08-11 — effects are safe to share, and one real aliasing case
exists elsewhere.**

*Effects: cleared, and by the type system rather than by inspection.* All five
effect interfaces declare their entry point `const`:

```cpp
IOnHitEffect::apply(std::shared_ptr<CombatEntity>) const = 0;
IDeathEffect::apply(Board&, const Vector2D&, int) const = 0;
IPeriodicEffect::apply(Board&, const Vector2D&, int) const = 0;
IOnDamageTakenEffect::apply(CombatEntity&) const = 0;
IAbilityEffect::apply(Board&, CombatEntity&) const = 0;
```

and a search across every file defining or using those interfaces finds **zero
occurrences of `mutable` and zero of `const_cast`**. Concrete effects
(`FreezeOnHit`, `PoisonOnHit`, `CurseOnHit`, and the ~20 in `include/core/`)
carry only construction-time parameters — `ticks`, `slowFactor`,
`damagePerTick`. They are stateless strategy objects, so sharing them across a
snapshot is correct, and a deep copy of them would be wasted work.

*Target caching: one real case, and it is the dangerous kind.*
`Projectile.h:16` holds

```cpp
std::weak_ptr<Entity> target;      // a MEMBER, persists across ticks
```

Every other `shared_ptr<Entity>` in the hierarchy — `CombatEntity.h:836`,
`CombatEntity.h:1108/1110`, `BuildingTargeter.h:31/33` — is a **local** inside
`findTarget`/`resolveCurrentTarget`, recomputed per tick, and therefore harmless.

`Projectile::target` is not. An implicit copy carries the pointer verbatim, so a
projectile in flight inside a snapshot would home on, and deal damage to, an
entity in the **original live board**. That is silent cross-simulation
corruption of exactly the kind this section was written to catch — the search
would be quietly damaging the real game it is supposed to be reasoning about.

So the deep copy must build an `old entity id -> new entity` map and remap
`Projectile::target` through it. Scope: one member, one class, plus the map the
board copy already has to build. Being a `weak_ptr` it will not keep the
original alive, so the failure would be wrong damage rather than a leak — which
is worse, because it is invisible.

#### The Catch-22, stated plainly rather than papered over

The obvious question is "what does search buy in win rate?" I tried to answer it
and **could not**, and the reason is structural rather than a matter of effort.

Without snapshotting, a search action can only be tested at a decision point
that can be *constructed* (via `inject`), after which both branches are played
out by the policy. That measures the marginal value of **one** improved action
diluted across ~80 subsequent policy actions. Result over 45 disagreement
states:

```
search branch ahead 51%   policy ahead 47%   tie 2%
paired tower-HP delta +0.0145 +/- 0.2162 (95% CI), n=45
```

The confidence interval is **15x wider than the effect**. Observed paired std
0.74 puts the sample size needed at roughly **20,000 constructed states** — and
even then it would only measure a one-decision intervention, not the
every-decision search that expert iteration actually performs.

**This is explicitly NOT evidence that search does not work.** It is an
underpowered null from an experiment that could not have detected the effect it
was looking for. Recording it that way, rather than quoting "51% vs 47%" as
though it meant something, is the point.

So the evidence that would justify the change is unobtainable without the
change. The honest basis for proceeding is architectural priors — the 150:1
compute asymmetry, the measured candidate spread, the determinism guarantee, and
a distillation pipeline that already exists — not a demonstrated win rate. The
human should decide with that stated, not implied.

#### Blast radius

Simulation core. Nothing about existing behaviour changes if the new method is
purely additive and nothing calls it — but any bug in it produces *wrong
simulated futures*, which would then be distilled into the policy as if they
were expert labels. That failure mode is silent, which argues for the
loud-failure default above and for a divergence test: snapshot a live game, step
both copies with identical actions for N ticks, and assert the observations stay
bit-identical. `perception/`'s bridge already demonstrates exactly this kind of
zero-divergence control.

---

### UPSTREAM item 16. DONE — bind `TimeoutRules::resolve`, so match outcomes have one definition (proposed 2026-08-19, signed off and applied 2026-08-20)

> **Applied.** `ClashEnv::resolveTimeoutOutcome()` calls
> `TimeoutRules::resolve(game.getBoard()).loserTeam` and is bound as
> `resolve_timeout_outcome`. Exactly the edit proposed below.
>
> **Why the count rose to eight before this landed.** Between the proposal and
> the sign-off, the same tower-count-only scorer appeared in three MORE new
> files (`eval/prove_combos.py`, `eval/prove_teacher.py`,
> `eval/prove_environment.py`), on top of the five already fixed. Fixing
> instances demonstrably does not hold when the wrong version is four lines
> long and the right one is unreachable from Python.
>
> **`python_ai/eval/match_outcome.py` keeps its hand-written mirror as a
> FALLBACK, on purpose.** The binding lives in a compiled `.pyd`, and this repo
> is worked from at least one machine that cannot rebuild it (see CLAUDE.md's
> Machine A / Machine B box). It feature-detects with `hasattr` so a checkout
> whose `.pyd` predates the binding degrades to the mirror instead of raising
> `AttributeError` mid-evaluation, after the match is already played. Verified:
> both paths return identical verdicts on all six rule cases, including the
> 3-3-towers / 1200-vs-90 case every broken scorer called a draw.
>
> **Retire the mirror** once every environment is known to carry a fresh
> binary: drop `_USE_BINDING` and the fallback branch, and delete
> `python_ai/tests/test_match_outcome_is_the_only_scorer.py`, which exists only
> because the mirror is hand-written.
>
> **Also added: `tests/core/test_timeout_rules.cpp`.** TimeoutRules had NO
> tests at all despite deciding every timed-out match. Now covers all three
> rules, their precedence (count outranks HP), weakest-vs-total, Towers-only
> (a Cannon is not a crown), dead-tower exclusion, and the new accessor.

Original proposal, kept for the record:

**The ask is one read-only accessor.** No gameplay change, no observation or
action-space change, no checkpoint invalidation, no behavioural difference to
any existing caller.

#### The problem

`include/core/TimeoutRules.h` is the engine's definition of who won: tower count
first, then the weakest surviving tower's HP, and only an exact tie is a draw.
It has exactly one call site, `ClashEnv::calculateReward`, and it is not
reachable from Python by any other route.

So every Python script that wants an outcome without going through `reward`
re-derives one. **Five did, and all five implemented only the first of the three
rules:**

| file | what it decides |
|---|---|
| `python_ai/eval/net_ab.py` | greedy-episode win rate |
| `python_ai/eval/net_h2h.py` | head-to-head duel score |
| `python_ai/eval/net_h2h_search.py` (×2) | search leaf value, and the ship/no-ship duel number |
| `python_ai/tools/validate_pipeline.py` | the side-asymmetry regression check |

Each read `get_towers_alive(0/1)` and returned a draw whenever the counts
matched. A match ending 3–3 on towers but 1200 HP against 90 HP on the weakest
is a clear win by the engine's own rules, and all five called it a draw — in the
scripts whose entire output is a win rate.

Fixed on the Python side in the same commit as this proposal
(`python_ai/eval/match_outcome.py`), by reading the six tower-HP scalars the
observation already carries and reapplying the rule by hand. That works, and it
is tested — but it is a **hand-written mirror of engine logic**, which is the
exact pattern CLAUDE.md records going stale twice before (`model.py`'s
"18*16=288", `calibrate.py` scoring bridges against `y=17.0` after the river
moved). If `TimeoutRules` ever gains a fourth rule, nothing makes that file
follow.

#### The exact edit

```cpp
// ClashEnv.h -- purely additive.
// -1 = draw, 0 = team 0 lost, 1 = team 1 lost. Same convention as
// MatchRules::Outcome::loserTeam, which is what TimeoutRules already returns.
int resolveTimeoutOutcome() const {
    return TimeoutRules::resolve(game.getBoard()).loserTeam;
}
```

```cpp
// bindings.cpp
.def("resolve_timeout_outcome", &ClashEnv::resolveTimeoutOutcome)
```

`TimeoutRules::resolve` is already `static`, already takes `const Board&`, and
already only reads. `ClashEnv` already includes what it needs via `GameManager`.

#### What it is worth

`python_ai/eval/match_outcome.py` collapses from a hand-maintained reimplementation
(~50 lines of rule-mirroring plus the float-resolution argument for comparing
normalised HP instead of raw ints) to a pass-through. The five call sites do not
change again. Any future rule change propagates for free instead of silently
not propagating.

#### Blast radius

Effectively none. Read-only, additive, no existing symbol changes meaning, and
a rebuilt `.pyd` stays compatible with current weights. The one caveat is the
usual one: it needs a `.pyd` rebuild before the Python side can use it, so
`match_outcome.py` should keep its current implementation as the fallback until
that lands rather than being deleted in the same change.

---

### UPSTREAM — Lower-priority C++ observations (recorded, not requested)

Surfaced by the 2026-08-19 review of the `dd99991..HEAD` batch. Neither is a
bug today; both are recorded so they are not rediscovered from scratch.

**`Board::getNextWaypoint`'s two branches are mirror images.** The
`isCurrentBelow` and `isCurrentAbove` paths run the same "compute the near bank,
check `distanceTo <= WAYPOINT_ARRIVAL_EPS`, otherwise return it" logic with
`riverY_start`/`riverY_end` swapped. A shared helper would remove the risk of
the epsilon check being fixed on one side and not the other — the same drift
this file's own history is full of. Not requested: it is a refactor of live
pathing code, and the current version is correct and tested.

**`Entity::snapshot()` depends on an unenforced "effects are stateless"
invariant.** `Entity.h` documents that `onHitEffects`/`deathEffect`/
`periodicEffect`/`abilityEffect` are deliberately *shared* rather than deep-copied
between an entity and its snapshot, because every effect interface declares
`apply() const` and none holds mutable state. That is true of the ~30
implementations today, but nothing enforces it — not a `const`-only member type,
not a `static_assert`, not a test. A future effect using a `mutable` counter
("every 3rd hit stuns") would let a *hypothetical* search rollout mutate state
the *live* match reads back, which is precisely the failure class
`test_board_deepcopy.cpp` exists to prevent and the one vector it does not cover.
Worth a comment at minimum; a fixture card with a stateful effect would turn it
into a real test.

---

### UPSTREAM item 17. OPEN — const accessors for internal timing state, so divergence tests can see it (proposed 2026-08-19)

**Ergonomics, not a coverage unblocker.** Read that sentence before deciding:
everything below is testable *today* through seams that already exist, so this
buys earlier and better-localised failure messages, not new capability. Filed
because the alternative — behavioural proxies — is what the current tests use,
and one of them is genuinely lossy (see "the one real gap").

#### The ask

Const getters for fields that have no public reader, on the model of
`Projectile::getTargetId()` (`Projectile.h:113`), which was added verbatim
"so the deepCopy divergence tests can assert the remap actually happened,
instead of inferring it from damage landing in the right place":

| type | fields |
|---|---|
| `CombatEntity` | `currentCooldown`, `currentTargetId`, `ticksOnTarget`, `ticksSinceLastHit`, `currentHitCount` |
| `Building` | `ticksAlive` |
| `AreaSpell` | `delayTicks`, `remainingHits`, `tickInterval` |
| `Projectile` | `returnDelayTicks`, `outboundHitLanded` |

#### Why

`tests/core/test_board_deepcopy.cpp`'s `EntityRow` compares only externally
observable state — id, hp, team, cardId, x, y, projectileTargetId. A copy that
diverged **only** in internal timing would pass its 120-tick window until the
difference happened to surface as an hp or position change. Since a search
rollout's whole job is to predict the next second or two accurately, a
timing-only desync is exactly the defect class that matters and exactly the one
the acceptance test cannot currently name.

#### The one real gap, and the honest limit of the rest

Most of these ARE reachable behaviourally, which is why this is filed as
ergonomics:

- `currentCooldown` — `CombatEntity::seedCooldown(int)` is already **public**
  (`CombatEntity.h:654`) and its own comment calls it "the one seam". Seed it,
  copy, step both, assert the first hit lands on the same tick.
- `Building::ticksAlive` — decay fires on `ticksAlive % 10 == 0`, so stepping to
  a tick that is NOT a multiple of 10 before copying makes a reset detectable.
  (Worth noting: the existing `WARMUP_TICKS = 60` IS a multiple of 10, so a
  reset would currently stay phase-aligned and be invisible.)
- `AreaSpell::delayTicks` / `remainingHits` — copy mid-fuse or mid-volley and
  compare hp trajectories.
- `Projectile::outboundHitLanded` — snapshot an Executioner axe after the
  outbound hit and assert the return lands on the same tick.

**`ticksOnTarget` is the exception and is the strongest argument here.** Its only
public proxy is `getDamagePerTick()`, which collapses it into at most four ramp
buckets via `getCurrentDamage()` — so a behavioural test detects a *stage*
desync, never a *tick* desync. `getCurrentDamage()` also folds in
`rangeFalloff`/`rangeBand` through `lastAttackDistance`, so on a falloff card the
proxy varies continuously and the ramp stage stops being separable at all.

#### Blast radius

None. Additive `const` getters returning by value; no existing symbol changes
meaning, no field becomes writable, no gameplay path is touched. **Not
gameplay-affecting, so `model_weights.pth`'s win-rate history is unaffected** —
same framing item 13 used for its own additive surface.

#### What was done instead, pending a decision

`tests/core/test_snapshot_champion_state.cpp` (new) covers the
`championSlots` ↔ `Board` cross-structure invariant, which needed no engine
change at all. `lastAttackDistance` is already public and varies at runtime in
the existing fixture, so it is the one cheap non-vacuous addition to `EntityRow`
available without this request.

---

### UPSTREAM item 18. ACCEPTED, WILL NOT FIX FOR NOW — troops deadlock in the concave pocket between two buildings (proposed 2026-08-20)

> **DECISION, 2026-08-20.** Signed off as a known, accepted defect: the
> architectural assessment below is agreed (a reactive steering model always has
> local minima in concave pockets), and global path planning is judged too
> expensive for rollout throughput at present. The defect is rare
> (7 / 342,563 unit-ticks), isolated to a player's own back corner, and never
> touches a bridge. It stays pinned as `[!shouldfail]`.
>
> **This is a deferral, not a dismissal.** Reopen it if any of these change: the
> stall rate rises, a stall is ever observed on or near a bridge, or rollout
> throughput stops being the binding constraint on path planning.

**Severity:** low frequency, permanent per occurrence. **Blast radius of the
proposed fix: the movement core, every unit, every match — which is exactly why
it is here rather than applied.**

Found during the bridge-navigation audit. It is NOT a bridge bug and it is not
the bug that audit was opened for (that one — the bridge-EXIT absorbing state in
`Board::getNextWaypoint` — is fixed and verified; see CLAUDE.md).

#### The measurement

`tools/audit/soak.cpp`, 60 randomized full matches, 308,464 unit-ticks watched.
A stall is counted only when a unit is alive, past deploy time, and stationary
for 50+ consecutive ticks *during every one of which nothing was inside its own
effective attack reach*.

| | |
|---|---|
| stalls | **7** |
| ...on or beside a bridge | **0** |
| longest observed | **~490 ticks**, i.e. until the match ended |
| rate | 7 per **342,563** unit-ticks (0.002%) |

(Measured at 4 per 308,464 before the sight/attack fix landed in the same audit;
that change alters engagement geometry, so the figure is re-quoted against the
current tree rather than carried over. Every one of the 7 is the same shape --
a unit in its OWN back corner, pinched between a friendly tower and either a
second building or the board edge.)

Every one was a unit pinched between two buildings in its own back corner. The
first, dumped in full:

```
Ice Golem  (11.360, 2.939)  walking north toward (14, 15.5)
King Tower ( 9.000, 2.500)  r=2.0 -> minimum separation 2.4, actual 2.401
Cannon     (12.032, 4.169)  r=1.0 -> minimum separation 1.4, actual 1.401
```

It is an **attracting fixed point**, not a knife edge. The trace shows the
approach converging geometrically — y = 2.9071, 2.9244, 2.9323, 2.9359, 2.9376,
2.9385, 2.9389, 2.9391, 2.9392 — so nearby states are pulled in rather than
passing through.

#### Mechanism

`Board::pushAwayFrom` adds a small perpendicular slide so a unit travels *around*
an obstacle instead of sticking to it. That works for ONE obstacle. With two, the
slides can point in opposing tangential directions and cancel: the unit steps
toward its waypoint inside `moveTowards`, and the post-move `resolveCollisions`
pass pushes it straight back. Net displacement converges to exactly zero.

The geometry says no local rule can fix it here: the King's and the Cannon's
minimum separations sum to **3.8** while their centres are **3.46** apart, so
there is no route between them at all. Escaping requires a multi-tile detour
around the outside of one of them.

#### Two fixes were implemented and MEASURED, and both are rejected

Reported because the negative results are the useful part — they are what turns
"we should nudge stuck units" into "a nudge is not enough".

| attempt | movement over 120 ticks |
|---|---|
| tangential slide, handedness flipped every 15 ticks | **0.027 tiles** — fifteen ticks of progress undone by the next fifteen |
| wall slide along the nearest blocker, side chosen by tangent · desired-direction | **0.000001 tiles**, settling at a NEW fixed point (11.3102, 2.96839) |

Each merely relocated the equilibrium. Both were reverted; the tree contains
neither.

#### What would actually fix it

Global path planning instead of purely reactive steering — a flow field or A*
over the 18×34 grid, which is small enough that the cost is negligible next to
the 0.015 ms an engine step already takes. That is a redesign of how every unit
moves, it is **gameplay-affecting**, and it would invalidate the win-rate
history of every checkpoint. It needs a decision, not a patch.

#### Pinned meanwhile

`tests/core/test_navigation_wedge.cpp` reproduces it deterministically and is
tagged `[!shouldfail]`: the suite stays green, the defect stays on record and
executable, and the case turns RED the moment somebody fixes it. It also carries
a companion test asserting that unobstructed movement is still exactly
speed-per-tick, which is the guard any future fix has to clear.

---

### UPSTREAM item 19. APPLIED 2026-08-21 — the observation's bridge marker was not re-centred with the board

> **Applied, the same day it was proposed.** Fixed with the stronger of the two
> options this item offers: `Board::isOnBridge(float x)` is now the single
> definition of "is this column a bridge", called from BOTH `clampToBoard` and
> `extractObservationForTeam`. A shared FORMULA would not have prevented the
> drift -- the bug IS two correct-looking expressions of one question drifting
> apart -- so the two callers share a FUNCTION.
>
> It takes a float so one function serves both: physics passes continuous
> positions, the encoder passes integer cell centres. Cell `i` covers
> `[i-0.5, i+0.5]`, so asking about centre `i` against a seam-centred 2.5
> selects exactly cells 2 and 3.
>
> Verified end to end: both teams' channel 8 now reads `WWBBWWWWWWWWWWBBWW`,
> matching `clampToBoard` column for column. The regression test in
> `tests/core/test_arena_layout.cpp` compares the channel against
> `clampToBoard` rather than against expected columns, so it cannot go stale the
> way the encoder did; `tools/audit/verify_pyd.py` checks the same row on the
> built `.pyd`, since that is the artifact training loads.
>
> Sensitivity was measured rather than assumed
> (`tools/audit/bridge_mask_probe.cpp`): the old predicate differs from the
> physics at 4 of 18 columns, so the new test genuinely fails against it.
>
> **This item's measurement was right and its severity assessment was right.**
> The "zero overlap on the left bridge" finding is exactly what makes it worse
> than a half-tile cosmetic drift.


**The arena re-centring removed three stale copies of the bridge columns and
left a fourth, inside `extractObservationForTeam` itself.** `ArenaLayout.h` now
owns the geometry and `Board`, `HeuristicOpponent`, `tactics.py` and
`perception/geometry.py` all derive from it. The observation encoder does not.

This is the **second** time channel 8 has been wrong (item 6 was the row; this
is the columns), and the first time it has disagreed with the engine's own
pathing.

#### What is there now

`include/core/ClashEnv.h:183`, inside the river/bridge marker loop:

```cpp
constexpr int riverRow = 17;
for (int x = 0; x < BOARD_WIDTH; ++x) {
    if ((x >= 3 && x <= 4) || (x >= 13 && x <= 14)) {
        obs[getIndex(8, riverRow, x)] = 1.0f;
```

Those literals were correct for the old bridge centres 4.0 / 14.0. They are a
hand-truncated copy of `centre ± 1.0`, and nothing ties them to the centre.

#### The engine's actual bridges

`ArenaLayout::LEFT_BRIDGE_X = 2.5f`, `RIGHT_BRIDGE_X = 14.5f` (seam-centred, so
`± BRIDGE_HALF_WIDTH = 1.0` spans exactly two tiles), consumed by
`Board::clampToBoard`:

```
walkable   x in [1.5, 3.5]   and  [13.5, 15.5]
i.e. cells      2, 3               14, 15
marker says     3, 4               13, 14
```

#### Measured, not inferred

A ground troop (Giant) was injected at each of the 18 columns on the own side
and stepped until it crossed; the columns it was ever observed occupying at
river rows 16/17 were recorded:

```
troops actually cross at columns:  [2, 14]
observation channel 8 marks:       [3, 4] and [13, 14]
```

**The LEFT bridge marker has ZERO overlap with where units actually cross.** It
marks column 4, which is now water, and omits column 2 entirely. The right
bridge is half right — 14 is correct, 13 is water.

(Units funnel to the bridge centre, so they occupy the truncated centre cell
rather than both bridge cells; the point is that 4 and 13 are unreachable and 2
is unmarked.)

#### Why this one is worth fixing promptly

Channel 8 is the network's **only** spatial cue for where the bridges are, and
`DEFAULT_DECK` is the 2.6 Hog Cycle, in which bridge placement is the entire win
condition. A net reading this channel is told the left lane crosses at a column
no unit can occupy.

It also silently confounds any measurement taken between the re-centring and the
fix, because `UtilityTeacher` and `tactics.py` read observations by contract
(`perception/tests/test_encoder_matches_engine.py` pins that contract) while the
engine paths on the geometry.

#### Proposed edit

Derive the test from `ArenaLayout` rather than adding a fifth literal:

```cpp
constexpr int riverRow = 17;
for (int x = 0; x < BOARD_WIDTH; ++x) {
    const float fx = static_cast<float>(x);
    const bool onBridge =
        (fx >= ArenaLayout::LEFT_BRIDGE_X  - Board::BRIDGE_HALF_WIDTH &&
         fx <= ArenaLayout::LEFT_BRIDGE_X  + Board::BRIDGE_HALF_WIDTH) ||
        (fx >= ArenaLayout::RIGHT_BRIDGE_X - Board::BRIDGE_HALF_WIDTH &&
         fx <= ArenaLayout::RIGHT_BRIDGE_X + Board::BRIDGE_HALF_WIDTH);
    obs[getIndex(8, riverRow, x)] = onBridge ? 1.0f : -1.0f;
}
```

This reproduces `clampToBoard`'s own predicate on cell centres, so the two can
no longer disagree. Cell `i` covers `[i - 0.5, i + 0.5]`, so testing the integer
centre marks cells 2, 3, 14, 15 — the walkable set.

**Alternative worth considering instead:** expose the predicate once on `Board`
(`bool isOnBridge(float x) const`) and call it from both sites. That is the
stronger fix, since `clampToBoard` and the encoder would then share code rather
than share a formula. Slightly larger blast radius.

#### Blast radius

**GAMEPLAY-AFFECTING for learning, not for simulation.** No unit moves
differently — `clampToBoard` is untouched. What changes is what the network is
told, on 4 cells of one channel on one row.

**Checkpoints are not architecturally invalidated** (no shape change, `NUM_CHANNELS`
and `observation_size()` unchanged), but any policy that learned bridge
positions from this channel learned them from the wrong columns, so **win rates
earned between the re-centring and this fix are not comparable** to either side.

**No Python change is needed, which was worth checking rather than assuming.**
An earlier draft of this item claimed `perception/geometry.py` and the
round-trip test would have to move with it. Both are already derived:

* `perception/geometry.py` reads `engine.ARENA_LEFT_BRIDGE_X` /
  `ARENA_RIGHT_BRIDGE_X` / `ARENA_BRIDGE_Y` off the binding, with a fallback
  only for when the binding is absent.
* `python_ai/models/perception_encoder.py`'s `base_spatial()` takes the whole
  river/tower plane from `_probe(None)` -- the engine's own fresh-board
  observation -- explicitly so that "a future move propagates instead of
  diverging".

So `perception/tests/test_encoder_matches_engine.py` keeps passing without
edits, and this really is a ONE-SITE fix. The mechanism that was supposed to
prevent this class of drift worked everywhere except in the encoder that
produces the number in the first place.

#### Verification

1. Re-run the injection sweep above: the marked columns must equal the columns a
   ground troop can occupy at rows 16/17.
2. `perception/.venv/Scripts/python.exe -m pytest perception/tests -q`.
3. `python_ai/venv/Scripts/python.exe -m pytest python_ai/tests -q`.
4. The Catch2 suite (582 cases, exactly one `[!shouldfail]`).

---

### UPSTREAM item 20. OPEN — spawned entities carry no `cardId`, so a replay cannot name them

**Found 2026-08-21 while fixing `web/viewer.html`'s entity inspector.** The
viewer-side half is fixed and shipped; this item is the engine-side half, which
is a C++ change and therefore a proposal rather than an edit.

#### What is there now

`Entity.h:45` declares `int cardId = -1`. `CardFactories::applyCardMetadata`
assigns a real id for a card played from hand, and nothing assigns one for an
entity spawned by another entity — death-spawns (`SpawnOnDeath`,
`SpawnOnDeathForEnemyTeam`), spawner buildings (`PeriodicSpawnEffect`,
`ProximityGatedPeriodicSpawnEffect`, `CappedSpawnOnHitEffect`) and tower troops
(`TowerTroops.h`). Those keep the `-1`.

#### Consequence

`GameLogger` writes `cardId` per entity plus an id-keyed `cardMeta` block, so a
replay is self-describing for every card played from hand **and for nothing
else**. A `-1` entity cannot use the id lookup and falls through to the
viewer's symbol-keyed tables, which cover ~45 of the 173 registry entries.

Where that missed, the last resort was the RAW SYMBOL — and `'?'` is
`CardDefinition`'s default symbol (`CardRegistry.h:100`) as well as the explicit
symbol of Cannon Cart (69) and Guards (76). So a card could be displayed to the
user as a literal `?`. That is the reported bug.

#### Measured

Built from a live replay's own `cardMeta` (132 playable + 41 Evolution entries):

| | |
|---|---|
| distinct symbols in the registry | **90** |
| symbols shared by more than one card | **36 (40%)** |
| cards whose symbol is `'?'` | Cannon Cart (69), Guards (76) |

So symbol is not a safe key even as a fallback: for a `-1` entity a collision is
not resolvable, which is why the viewer now flags such a resolution as
"identified by symbol" rather than presenting it as certain.

#### Blast radius — NOT physics-affecting, but IS reward-affecting

`cardId` is read by stats attribution (`StatsEvents`, `MatchStatistics`) and by
the logger. Nothing in `include/entities/` branches on it to decide movement,
targeting or damage, so propagating it **cannot change a simulation outcome**.

It WOULD change per-card stats attribution: a Skeleton spawned by a Tombstone
would begin attributing its damage to a card rather than to nothing, and
`get_elixir_value_killed_by` / `get_damage_dealt_by_card` both read those — and
those feed reward shaping. That makes it reward-affecting even though it is not
physics-affecting, which is precisely why it is worth deciding deliberately
rather than patching in passing.

#### Options

1. **Propagate a `cardId` to spawned entities.** Simplest. Changes stats
   attribution as described above, so every reward-shaped number earned before
   it would be earned under a different attribution.
2. **Add a separate `spawnedByCardId`** and leave `cardId` alone. The logger
   gains one field and the viewer one fallback; stats attribution is untouched.
   **This is the option that fixes the display without touching the reward
   path**, and is the recommendation.
3. **Do nothing.** What shipped: the viewer resolves these by symbol against the
   replay's own `cardMeta` and never renders a bare `?`. Residual defect is the
   40% symbol collision rate above — a `-1` entity on a shared symbol may be
   shown under the wrong name and HP maximum, flagged as inferred.

#### Verification if option 2 is taken

1. A replay containing a Tombstone or Witch must show every spawned body with a
   real name in the viewer's inspector, with no "identified by symbol" note.
2. `get_elixir_value_killed_by` totals must be **unchanged** against a
   pre-change run on the same seed — that is the whole point of option 2.
3. The Catch2 suite (619 cases, exactly one `[!shouldfail]`).

---

### UPSTREAM item 21. APPLIED 2026-08-23 — a self-play step that does not build observations, because teacher rollouts build 1,000,152 of them per 48 episodes and read 51,566 (proposed 2026-08-23)

#### What is there now

`ClashEnv::stepSelfPlay` ends with

```cpp
return { extractObservationForTeam(0), extractObservationForTeam(1), totalReward, isDone };
```

so **every call constructs both teams' 13,606-float observation vectors**, whatever
the caller wants. `SelfPlayStepResult`'s fields are bound with `def_readonly`,
which converts to a Python list on ATTRIBUTE ACCESS — so a caller that ignores
the fields pays the full C++ construction and none of the pybind marshalling.

`UtilityTeacher.execute_steps` (`python_ai/opponents/teacher.py:1382`, `:1384`)
is that caller. It rolls a candidate forward in 10-tick chunks and **discards
the returned object every time**:

```python
if self.team == 0:
    s.step_self_play(slot, x, y, oslot, ox, oy, nxt - t)
else:
    s.step_self_play(oslot, ox, oy, slot, x, y, nxt - t)
```

The one observation a rollout actually reads is the single
`s.get_observation_for_team(me)` at the end of `rollout_stats`, for
`positional_advantage`.

#### Measured — on the training box, not inferred

`python_ai/tools/profile_training.py --mode sync --episodes 40 --teacher-stage 5`,
i5-13420H, real `.pyd`, 48 episodes, 268.93 s wall:

| span | calls | self s | us/call |
|---|---|---|---|
| `rollout.step` | **500,076** | 40.064 | 80.11 |
| `rollout.obs` | 51,566 | 11.850 | 229.79 |
| `live.snapshot` | 51,566 | 2.015 | 39.07 |
| `live.step` | 7,952 | 0.839 | 105.56 |
| `rollout.info` | 1,682,847 | 4.837 | 2.87 |

Derived from those counts:

* 51,566 snapshots / 7,952 decisions = **6.49 candidate rollouts per decision**
* 500,076 rollout steps / 51,566 rollouts = **9.70 chunks per rollout** (horizon
  100 in 10-tick chunks, minus early game-over breaks)
* every chunk builds TWO observations, so rollouts construct
  **1,000,152 observation vectors** and read **51,566**

**A 19.4 : 1 build-to-read ratio.** `rollout.step` is 14.9% of wall clock in the
sync profile, and the async profile puts the whole environment at 39.6%.

C++-side cost of the pieces, measured with no interpreter in the process
(`tools/audit/engine_profile.cpp`, WSL g++ -O2 — ratios transfer, absolute ms
do not):

```
extractObservationForTeam   0.0106 ms      GameManager::step()  0.0006 ms/tick
stepSelfPlay(skip=10)       0.0543 ms      of which 2 observations = 39%
skipFrames sweep:  slope 0.0008 ms/TICK,  intercept 0.0436 ms/CALL
                   -> 49% of the per-call intercept is the two vectors
```

A stage-5 candidate rollout measures **0.522 ms** against a **0.090 ms** floor
(snapshot + 100 ticks + the one observation it reads) — **5.8x**.

#### Mechanism

Observation construction is `O(entities)` plus a 13,606-float allocate-and-zero.
Physics is 0.0006 ms/tick. So a rollout chunk spends more time describing the
board to nobody than it spends simulating it.

#### Proposed edit

Extract the tick loop, so the two entry points cannot diverge in what they
simulate — the only difference is what they RETURN.

```cpp
// NEW, next to SelfPlayStepResult
struct SelfPlayFastResult { float reward0; bool done; };

private:
    struct SelfPlayTickOutcome { float reward; bool done; };
    // The self-play tick loop, with NO observation construction. Shared, so
    // stepSelfPlay and stepSelfPlayFast can never disagree about the physics.
    SelfPlayTickOutcome runSelfPlayTicks(<the existing 11 parameters>);

public:
    SelfPlayStepResult stepSelfPlay(<unchanged signature>) {
        SelfPlayTickOutcome o = runSelfPlayTicks(...);
        return { extractObservationForTeam(0), extractObservationForTeam(1),
                 o.reward, o.done };
    }
    // NEW
    SelfPlayFastResult stepSelfPlayFast(<same signature>) {
        return { runSelfPlayTicks(...).reward, runSelfPlayTicks(...).done };  // one call, see impl
    }
```

plus `src/bindings.cpp`:

```cpp
py::class_<SelfPlayFastResult>(m, "SelfPlayFastResult")
    .def_readonly("reward0", &SelfPlayFastResult::reward0)
    .def_readonly("done", &SelfPlayFastResult::done);
...
.def("step_self_play_fast", &ClashEnv::stepSelfPlayFast, ...)
```

and one call-site change in `python_ai/opponents/teacher.py::execute_steps`.

#### Blast radius

* **`stepSelfPlay` is behaviour-identical.** The loop body moves verbatim; the
  return statement is unchanged. Nothing that calls it can observe a difference.
* **NOT gameplay-affecting.** No observation, action space, reward or physics
  change. `model_weights.pth` and `model_weights_selfplay.pth` stay valid, and
  win rates remain comparable across this change.
* **The RL learning signal cannot change**, and that is provable rather than
  argued: the vectors being removed are never read by anything. The teacher's
  chosen action is a function of `rollout_stats`, which reads
  `get_observation_for_team` separately and is untouched.
* **Additive binding.** An older `.pyd` simply lacks `step_self_play_fast`.
  Because a silent fallback would hide a stale `.pyd` — a trap this repo has
  already been bitten by — the Python side probes ONCE at import and raises with
  a message naming the rebuild, rather than degrading quietly.
* Other discard-the-result callers exist and can adopt it later:
  `envs/selfplay_env.py:368,380` (phase-2 warm-up), `envs/scenario_offense.py:124`,
  and several `eval/` harnesses. **This proposal changes only the teacher**, the
  one on the training hot path.

#### What it does NOT address

`gym_wrapper.step` reads `observation0` and never touches `observation1`, so it
also builds one vector per decision for nothing — but it needs the other, so it
needs a *different* fix (a team-selective step) and is 7,952 calls against
500,076. Out of scope here; noted so it is not forgotten.

#### Verification

1. C++ suite green, with the `[!shouldfail]` navigation-wedge case still the
   only failure and the runner exiting 0.
2. A new Catch2 case asserting `stepSelfPlayFast` and `stepSelfPlay` leave the
   env in the SAME state from the same snapshot — same tick, same reward, same
   done, same subsequent observation. That is the property the refactor must
   preserve, and it is the one a shared tick loop makes true by construction.
3. `tools/audit/engine_profile.cpp` before/after on the rollout block.
4. `profile_training.py --mode sync/--mode async` on the training box.

#### APPLIED — measured result

`ClashEnv::stepSelfPlayFast` + `SelfPlayFastResult`, both entry points sharing
`runSelfPlayTicks`. The diff proves the claim that matters: inside the moved
code the ONLY changed line is the return statement.

**Correctness.**

* C++ suite **627 cases / 626 passed / 1 failed as expected**, runner exit 0,
  against a **625 / 624 / 1** baseline — the two new cases, no regression.
  (`tests/core/test_selfplay_fast_step.cpp`; built under wsl g++ on the box with
  no MSVC.)
* **The teacher's decisions are unchanged**: 12/12 configurations
  action-identical over stages 2-5 x 3 seeds, comparing full action sequences
  from identically-seeded runs. That is the property that would actually hurt if
  it broke, since every candidate score now flows through the new path.

**Speed** — paired, both arms alternating inside ONE process, only the engine
method differing (the slow arm routes `step_self_play_fast` back to
`step_self_play` at the boundary, so the teacher's Python is byte-identical):

| | ms per teacher decision |
|---|---|
| observations built (old) | 8.690 |
| **not built (new)** | **7.851** |
| saving | **0.839 (9.7%), 1.11x** |

**23 of 24 trials favour the fast arm, exact sign test p = 3e-06.** 27,298
observation vectors are no longer built per trial.

Cross-checked against the component measurement rather than trusted alone:
13,649 chunk steps x 2 observations x 0.0106 ms = 0.79 ms/decision predicted,
0.839 measured.

**Two measurement traps this hit, both recorded because they nearly produced
false results.**

1. **A background compile made the fix look 5.6x SLOWER.** Run as two separate
   invocations while the C++ suite was building on the same 4 cores, every span
   moved together — including `live.snapshot`, which the change does not touch.
   That uniformity is the signature of machine state, not of a code difference.
   Alternating arms inside one process removes it, and is now how the harness
   works.
2. **`UtilityTeacher` defaults to `seed=None`**, i.e. `np.random.default_rng(None)`,
   which is entropy-seeded. The first equivalence run reported **10 of 12
   configurations "diverged"** when the only difference was the teacher's own
   lane bias. Any A/B over this class must pass `seed=`.

**Scope, restated.** ~9.7% of teacher decision time. The teacher is roughly 59%
of the sync profile's wall clock, and the environment as a whole is 39.6% of the
async one at ~2.6x parallel efficiency, so the expected end-to-end throughput
effect is **low single digits** — this is the safe, zero-risk win, not the
answer to phase-1 throughput. The async profile puts the MODEL at 55.8%.

---

### UPSTREAM item 22. APPLIED 2026-08-24 — the live mirror cannot be given the real position: no tower HP, no unit HP, no clock, and every re-injected unit is inert for a second (proposed 2026-08-24)

#### What this is for

The live-play pipeline, with the teacher as OUR agent in a real match:
perception reads the screen, `forecast.py` rebuilds a mirror `ClashRoyaleEnv`,
`UtilityTeacher` proposes candidates and ranks them by rolling each forward
5-10 s on `env.snapshot()`, and the winner is tapped by `live/actuator.py`.

Every joint of that already exists except one: **the mirror cannot be told what
the real position is.** The teacher then ranks candidates against a fabricated
board, which is the one failure that makes every other component's correctness
irrelevant.

#### What is there now

`forecast.py` rebuilds by `reset()` + `inject()`, and its own docstring lists
what that loses. Two of the six were closed on 2026-08-17 (`set_elixir_for_team`
/ `set_hand_for_team`, landed but **still unwired** — that is a Python-side TODO,
not a platform limit). Four remain:

| gap | status |
|---|---|
| unit HP | injected units spawn at FULL health. No setter. |
| tower HP | always full after `reset()`. No setter. |
| match clock | always zero. No setter. |
| entity removal | a unit perception no longer sees cannot be deleted. |

**Removal is dissolved by the architecture rather than by an API.** The live
loop rebuilds the mirror from the latest perception snapshot at every decision
and never carries state forward, so there is nothing stale to delete. No removal
binding is requested. This is worth stating because the obvious incremental
design — stream deltas into a long-lived env — would need one, and would also
be unsafe (see Blast radius).

That leaves three. **Verification of this proposal found a fourth that nobody
had named, and it is probably the largest of the four.**

#### The fourth gap: every re-injected unit is inert for a full second

Call chain, verified by reading, not inferred:

```
ClashEnv::inject  ->  card->spawnEntity(x, y, team, board)
                  ->  CardFactories::applyCardMetadata
                  ->  entity->deployTicksRemaining = DEPLOY_TIME_TICKS   // CardFactories.h:36
```

`DEPLOY_TIME_TICKS` is 10, i.e. 1.0 s. So a rebuilt board hands **every** enemy
unit a fresh deploy timer — including a Hog Rider that has been running for six
seconds. In every rollout, on every candidate, on every decision, the teacher
believes it has one extra second before anything on the board can act.

It is a defensive subsidy paid to US, and it is exactly the shape of a defect
this repo has already measured once.

**Why it likely dominates the other three.** CLAUDE.md's 2026-08-19 section
measured the SAME ONE SECOND in the opposite direction — a missing deploy second
subsidising the defender — and found it was the mathematical flaw suppressing
win conditions. Controlled, same harness, only the constant varying:

| engine | marginal value of a supported push |
|---|---|
| `DEPLOY_TIME_TICKS = 0` | **-73.7** HP, CI [-349.5, +195.3] |
| `DEPLOY_TIME_TICKS = 10` | **+448.5** HP, CI [+137.3, +760.1] |

One second of deploy inertness moved a push by ~520 tower HP and flipped the win
condition from negative to positive value. The rebuild currently applies that
second to every enemy unit in the mirror.

#### Measured evidence for the three requested gaps

**Tower HP — and the trap that makes the naive setter wrong.**
`perception/README.md` finding 6, read off a clean frame at t=20 s before
anything is damaged, so the on-screen numbers are true maxima:

| | our Princess | opponent's Princess | engine |
|---|---|---|---|
| max HP | 1750 (level 4) | 1890 (level 5) | **2534** (level 9) |

**Absolute HP is not comparable, and it is wrong by a DIFFERENT factor per
player.** Perception already reports a FRACTION for exactly this reason, taking
the maximum from the first undamaged reading rather than a supplied table.
Tower HP itself is read as the absolute printed numeral
(`readers/tower_numerals.py`, finding 9) and validated by read-back over 4 full
matches: 542 steps, 12 upward jumps, **97.8% consistent** with the fact that
tower HP never rises.

**Unit HP.** `live/unit_hp.py` reports `UnitHp.fraction`. Scored as an "is this
unit damaged" detector over 226 detections, reweighted from a stratified sample
to the population (finding 7): precision **0.79 -> 0.98**, recall
**0.34 -> 0.56**.

State this honestly: **roughly half of damaged units will still inject at full
HP.** This improves a biased estimator, it does not fix it. It is still a strict
gain, because today's effective recall is 0.

**Match clock.** `ClashEnv::currentTick` exists and is already read into the
observation (`ClashEnv.h:331`, `currentTick / maxTicks`) and into the done
condition (`:383`, `:542`). It is only ever set to 0 by `reset()` (`:517`) and
incremented in the step loops.

**The honest limit here is larger than the setter.** The engine models **no
double or triple elixir**: `ELIXIR_REGEN_RATE` is a constant (`GameManager.h:39`)
scaled only by `oppElixirMultiplier` (`:661-662`). So even a perfect clock
leaves every rollout during 2x mispriced on both sides. A clock setter buys the
observation's time scalar and timeout proximity — real, but modest. **Filed
separately as item 23** rather than folded in here, because it is a genuine
gameplay change that would affect training, while everything in this item is
additive and inert unless called.

#### Proposed edit

Four changes. Every one is additive, and every new parameter defaults to
current behaviour.

**(a) `ClashEnv::setTowerHp(int team, int slot, float hp) -> bool`**

`slot`: 0 = King, 1 = left Princess, 2 = right Princess, in board coordinates
(not team-relative), so the caller is not asked to mirror anything.

- Clamps to `(0, maxHp]`.
- **Returns `false` and changes nothing on `hp <= 0`**, following
  `setHandForTeam`'s precedent: refuse rather than accept a misread. A 0-HP
  tower that still occupies its cell and still fires is worse than no update.
- Takes **engine-absolute HP**. The level conversion stays on the perception
  side, where the per-player max already lives, per CLAUDE.md's
  no-second-copies rule. Python passes `fraction * engine_max`.
- **Side effect to name, because it is gameplay-visible and correct:**
  `Tower::awake` latches on the invariant `hp < maxHp`, so injecting a damaged
  tower wakes the King. That matches the real game and is the desired
  behaviour, but it means `setTowerHp` is not a pure state write.

**(b) `ClashEnv::destroyTower(int team, int slot) -> bool`**

Routes through the engine's existing destruction path so the crown, the King
wake and `LanePath`'s retargeting all fire. Requested because (a) refuses
`hp <= 0` and a destroyed tower would otherwise be inexpressible in the mirror —
which would make every rollout wrong from the moment a tower falls, i.e. exactly
when the position matters most.

**(c) `ClashEnv::inject(int cardId, float x, float y, int team, float hp = -1.0f, int deployTicks = -1)`**

Two optional parameters on the existing method:

- `hp < 0` keeps full health (current behaviour).
- `deployTicks < 0` keeps `DEPLOY_TIME_TICKS` (current behaviour);
  `0` spawns an already-deployed unit, which is what a rebuilt board wants for
  every unit that was already on screen.

Optional-with-preserving-defaults rather than a new method, so **no existing
caller changes at all** — `forecast.py`, `prove_*.py` and the audit tools keep
compiling and keep behaving identically.

**(d) `ClashEnv::setCurrentTick(int tick)`**

Clamps to `[0, maxTicks]`. Trivial, but it is the one field that cannot be
reconstructed by any combination of the others.

Bindings mirror these as `set_tower_hp`, `destroy_tower`, `set_current_tick`,
and two new `py::arg`s with defaults on the existing `inject`.

#### Blast radius — additive, and NOT gameplay-affecting

**No checkpoint is invalidated and no win rate is invalidated**, and that is a
design goal rather than a happy accident — the same discipline the `place_hires`
zero-init used:

- `setTowerHp`, `destroyTower` and `setCurrentTick` are new methods. Nothing in
  either training pipeline calls them. An uncalled method cannot change a
  rollout.
- `inject`'s two new parameters default to exactly today's behaviour, so every
  existing call site is bit-identical.
- No observation, action-space, reward or architecture change.

**This must be VERIFIED, not assumed** — CLAUDE.md records that a "purely
additive" `.pyd` change was checked by diffing the commits (119 insertions, 0
deletions, no simulation code) rather than trusted. Same bar here.

**The one thing this proposal deliberately does NOT enable: concurrent mutation
of a live env.** The setters are for a single-owner rebuild at decision time.
`src/bindings.cpp` contains **no `gil_scoped_release` and no `call_guard`** —
verified by grep — so every engine call holds the GIL for its full duration, and
a second thread writing into an env while the teacher rolls candidates forward
would both stall perception and score candidates against different worlds. The
Python side must own the mirror on one thread.

#### Verification

| check | bar |
|---|---|
| C++ suite | 622 cases, 1 `[!shouldfail]` wedge, runner exits 0 |
| Python suite | 401 collected, 399 pass / 2 skip |
| perception suite | 353 pass / 1 skip |
| `inject` back-compat | a board built with no new args is bit-identical to today's, over the full observation |
| deploy bypass | a unit injected with `deployTicks=0` moves on tick 1; with the default it does not move until tick 11 |
| tower fraction round trip | `set_tower_hp(t, s, f * max)` then read back through the observation returns `f` within float tolerance, for both teams |
| refusal | `set_tower_hp(t, s, 0.0)` returns `false` AND leaves HP unchanged — both halves, since a refusal that still writes is the worst outcome |
| destruction | `destroy_tower` awards the crown, wakes that team's King, and changes `get_towers_alive` |
| absorbing states | `waypoint_probe` still reports 0, since a re-injected unit at an arbitrary position is a new entry path into `getNextWaypoint` |

That last row is not boilerplate. This engine has shipped **two** absorbing
states at the bridge mouths, the second one found only because a sweep covered
every branch rather than the one the reproduction took. Injecting units at
arbitrary perceived positions with `deployTicks=0` puts entities into
`getNextWaypoint` at positions no normal spawn produces.

#### Options considered and rejected

- **Stream deltas into a long-lived env.** Needs a removal API, needs entity
  identity across frames (blocked by item 20 — spawned entities carry no
  `cardId`), and is unsafe under the GIL finding above. Rebuild-per-decision
  costs ~1 ms and needs none of it.
- **`set_entity_hp(entityId, hp)`.** Cleaner in principle, but requires stable
  entity ids across the binding boundary, which item 20 says do not exist. The
  `hp` parameter on `inject` needs no identity at all: the caller sets HP on the
  unit it is creating, in the same call.
- **Clamping tower HP to 1 instead of refusing.** Biases toward over-defending a
  tower that is already gone, and does it silently.

#### APPLIED — approved, implemented and measured 2026-08-24

Approved by the human on the proposal above, then built TDD: the 19 new cases
in `tests/core/test_state_setters.cpp` were written against compiling stubs
and **17 were watched to fail** before any implementation existed.

**Two of those 19 passed against a do-nothing stub and had to be
strengthened**, which is the part worth carrying. `setTowerHp REFUSES hp<=0`
passes trivially against a setter that refuses *everything*, and `an injected
hp above the maximum is clamped` passes trivially against one that ignores
`hp` — and ignoring it IS the pre-item-22 behaviour. Both now carry a positive
control that must fire first. Same rule this file already states for the
deploy-zone probe: **when a measurement's failure mode is maximal
permissiveness, it needs an internal control that MUST fire.**

**What shipped**, all additive:

| | |
|---|---|
| `Building::getMaxHp()` | the accessor the clamp needed; `maxHp` was protected with no reader |
| `GameManager::{findTower,setTowerHp,destroyTower,getTowerHp,getTowerMaxHp,setCurrentTick,getCurrentTick}` | the implementations |
| `ClashEnv::{setTowerHp,destroyTower,getTowerHp,getTowerMaxHp,setCurrentTick}` | wrappers |
| `ClashEnv::inject(..., hp = -1.0f, deployTicks = -1)` | two optional params |
| `bindings.cpp` | all of the above, plus `hp` / `deploy_ticks` keywords on `inject` |

`destroyTower` routes through `takeDamage` (the entry point a real killing blow
uses) and then pins the postcondition, because `CombatEntity::takeDamage` can
ABSORB via shield/parry/mid-dash invulnerability. No Tower carries any of those
today; the pin means this stays a destruction if one ever does.

**MEASURED — the deploy gap, and the probe that measured nothing first.**
Enemy Hog injected at (9.0, 20.0), both sides no-oping:

| | ticks until it first damages our tower |
|---|---|
| `inject(default)` | 88 |
| `inject(deploy_ticks=0)` | **78** |

**Exactly 10 ticks — `DEPLOY_TIME_TICKS` to the tick**, which is the
confirmation the mechanism is the one diagnosed. In the window where arrival
decides the outcome it is worth **317 tower HP, one full Hog hit**:

| window | default | deploy_ticks=0 | delta |
|---|---|---|---|
| 100 ticks | 317 | 634 | **+317** |
| 110 ticks | 634 | 951 | **+317** |
| 120 ticks | 951 | 951 | 0 |

**Read the 120-tick row: the first probe used 140 ticks and reported a delta of
ZERO.** Over a long enough window the Hog deals its full damage either way, so
a total-damage probe SATURATES and reports that a working fix does nothing.
Arrival time is the quantity that can see it. Third instance in this repo of a
control saturating — the air-targeting probe in the 2026-08-20 deck QA is the
same failure.

**Verification, against the bars this item set:**

| check | bar | result |
|---|---|---|
| C++ suite | 1 `[!shouldfail]`, exit 0 | **646 cases / 6,409 assertions, 645 pass, 1 failed as expected, exit 0** |
| Python suite | no regression | **399 passed / 2 skipped** — unchanged |
| perception suite | no regression | **366 passed / 1 skipped** (353 + 13 new) |
| `inject` back-compat | bit-identical | full observation equal after 50 ticks, 4-arg vs explicit defaults |
| deploy bypass | moves on tick 1 | pinned, C++ and Python |
| tower fraction round trip | within tolerance | exact for all three slots, both teams |
| refusal | returns false AND leaves hp | both halves pinned |
| destruction | crown + King wake + count | pinned, and team-scoped |
| absorbing states | `waypoint_probe` still 0 | **0 / 8,661,439 positions**, and 0 / 2,584,034 in the lane-composition sweep under all three tower configurations |

**The C++ count was already stale.** 646 − 19 new = **627**, against the 622
this file and CLAUDE.md record. The baseline was 5 ahead before this work
started, so do not read the jump as belonging to item 22 — the same arithmetic
trap the 2026-08-24 row-compaction section records for the Python count.

**The absorbing-state bar was worth keeping.** `deploy_ticks=0` puts entities
into `getNextWaypoint` at arbitrary perceived positions with no deploy delay to
absorb the first tick — a genuinely new entry path, and this engine has shipped
two absorbing states at the bridge mouths already, the second found only
because a sweep covered every branch rather than the one the reproduction took.
Re-run after the change: **0 absorbing states**, both sweeps, all three tower
configurations.

**What is NOT verified, and should be said plainly:** none of this has faced a
real screen. Every number above is engine-side. Whether perception's readings
are good enough to make the mirror worth having is a Stage 2 question, and the
unit-HP recall of 0.34–0.56 is the figure to watch — roughly half of damaged
units will still arrive at full health.

**A binding-surface regression test now exists** at
`perception/tests/test_engine_state_setters.py` (13 cases). The C++ suite
cannot see pybind at all, and this repo has twice had a stale `.pyd` hide a
landed setter for days with the C++ suite green throughout.

---

### UPSTREAM item 23. OPEN — item 7 seeded the engine and nothing was migrated to it; one RNG path is still unreachable from Python (proposed 2026-08-24)

**Not a correctness bug. The unfinished half of item 7**, and the reason that
item's stated benefit — *reproducible failures* — has still not been collected
three days after it landed.

> **SCOPE NOTE, because this file is for engine changes.** Two of the three
> paths below are **Python**, in `python_ai/`, and by this repo's own division
> of labour they belong in `BOT_REQUESTS.md` (training-side suggestions), not
> here. They are written up here anyway because they are meaningless apart from
> item 7 and splitting one finding across two files is how item 7's own status
> went stale in the first place. **Only §C is an engine request.** §A and §B
> were applied on 2026-08-24 at the human's explicit instruction and are
> recorded, not requested.

#### What item 7 actually delivered

Verified by reading, 2026-08-24:

```cpp
// include/core/ClashEnv.h
void seed(unsigned int s) {
    rng.seed(s);                      // HeuristicOpponent
    heuristicOpponent.reset(rng);
    game.seed(s ^ 0x9E3779B9u);       // opening hand + cycle order
    reset();                          // initializeDeck runs INSIDE reset()
}
```
```cpp
// src/bindings.cpp:164
.def("seed", &ClashEnv::seed, py::arg("seed"))
```

`seed()` ends in `reset()`, which makes it a **drop-in replacement for
`reset()`** at any call site that wants determinism — that property is what
makes §A and §B one-line changes rather than restructuring.

#### The three paths that bypassed it

| # | path | reachable from Python? | status |
|---|---|---|---|
| A | `MicroRoyaleEnv.reset(seed=...)` | yes | **applied 2026-08-24** |
| B | `prove_combos.py`'s five harnesses | yes | **applied 2026-08-24** |
| C | `sample_random_deck`'s static generator | **no** | **this request** |

---

#### A. `MicroRoyaleEnv.reset(seed=...)` accepted a seed and dropped it — APPLIED

`python_ai/envs/gym_wrapper.py:333`, before:

```python
def reset(self, seed=None, options=None):
    super().reset(seed=seed)          # seeds the WRAPPER's np_random only
    ...
    obs_list = self.game.reset()      # engine re-deals, unseeded
```

`gymnasium.Env.reset(seed=...)` seeds `self.np_random`. It cannot reach either
engine generator, and `MicroRoyaleEnv` reads `self.np_random` nowhere. So the
argument was accepted, had no effect on anything the env actually does, and
**looked like it worked** — which is strictly worse than not accepting it,
because the gymnasium contract says a caller may rely on it.

`UPSTREAM_REQUESTS.md` item 7 named this exact trap ("`MicroRoyaleEnv.reset(seed=...)`
looks like it should help but only forwards to `gymnasium.Env.reset`") at a time
when there was no binding to forward to. There has been one since 2026-08-21.

**Applied:** one guarded line forwarding to `self.game.seed(seed)`. Guarded on
`seed is not None` because the gymnasium convention is that a seed is passed
once and subsequent `reset()` calls continue the stream — seeding on every reset
would make every episode of a run identical, which is a far worse failure than
the one being fixed.

#### B. `prove_combos.py` never called it — APPLIED

Five harnesses (`run_usage`, `run_reserve_ab`, `run_combo_ab`,
`run_profile_sweep`, `run_vs_net`), each building its opening as
`CE(...)` then `.reset()`, and `--seed` reaching only `make_teacher`'s own RNG.
This is the harness whose mis-specified control "cost a 10-minute run and nearly
produced a wrong conclusion about which combo family was responsible for a
trend" — the single most-cited piece of evidence for item 7.

**Applied:** `.reset()` → `.seed(args.seed + ENGINE_SEED_OFFSET + i)` at all
five sites, plus the module docstring, which still told readers the shuffle was
unseeded and that item 7 was open.

**Why an offset rather than `args.seed + i`:** the teachers already draw from
`args.seed + i`. Reusing it for the engine would move a teacher's lane bias and
the hand it was dealt together across openings — the same correlation
`ClashEnv::seed` avoids internally with its `^ 0x9E3779B9` between the two
engine generators, for the same reason.

---

#### C. THE ENGINE REQUEST — `sample_random_deck` cannot be seeded

```cpp
// src/bindings.cpp:292
m.def("sample_random_deck", []() {
    static std::mt19937 rng(std::random_device{}());
    return sampleRandomDeck(rng);
});
```

A **third** `std::mt19937`, function-local `static`, seeded from
`std::random_device`, with no parameter and no seeding entry point. `ClashEnv::seed`
cannot reach it — it is not a member of anything.

**Why it matters, concretely.** `gym_wrapper.reset()` calls it on the
`randomize_opp_deck` path:

```python
random_deck = list(clash_royale_env.sample_random_deck())
self.game.set_opponent_deck(random_deck)
```

So **§A's fix is incomplete exactly where deck randomisation is on.** A run with
`randomize_opp_deck=True` now has a reproducible opening hand, cycle order and
heuristic roll, and a still-random *opponent deck* — which is the largest single
source of episode-to-episode variance of the four. Phase 1's `random_opponent`
and the scripted bots' randomised decks are the configurations this affects, and
they are the ones TODO.md item 6 wants extended, not retired.

The `static` also means the stream is **process-global and order-dependent**:
two envs constructed in the same process interleave draws from one generator, so
even seeding it would only be reproducible for a fixed construction order. Worth
knowing before anyone calls this a one-liner.

#### Options

1. **Add a module-level seeding function** — `m.def("seed_deck_sampler", ...)`
   setting the same static. Smallest diff; leaves the process-global stream and
   its order-dependence in place, so it buys reproducibility only for a fixed
   call order. Adequate for a single-env eval harness, not obviously adequate
   for `num_envs = 8`.
2. **Give `sample_random_deck` an optional seed argument** — `sample_random_deck(seed=None)`,
   constructing a local generator when one is passed and falling through to the
   static otherwise. Every existing zero-argument call site keeps its current
   behaviour bit-for-bit, and a caller that wants determinism gets a stream that
   is not shared with anyone. **Recommended.** It is additive, it does not
   change the meaning of any existing call, and it is the only option that
   survives vectorised envs.
3. **Move the generator into `ClashEnv`** and have `ClashEnv::seed` cover it.
   Rejected: `sample_random_deck` is deliberately module-level because it is
   called *before* a deck exists to construct an env with, and `train.py` /
   `train_selfplay.py` call it outside any env at all.
4. **Change nothing, and document it.** Defensible — deck randomisation exists
   to create variety, and a caller who wants a reproducible deck can pass one
   explicitly via `set_opponent_deck`. If this is the choice, §A's docstring
   should say so, because "seeded" will otherwise be read as "reproducible".

#### Blast radius

**Options 1 and 2 are additive and NOT gameplay-affecting.** No existing call
site changes behaviour, no observation changes, no checkpoint and no win-rate
history is invalidated. Option 2 touches one lambda in `src/bindings.cpp` and
nothing else; `sampleRandomDeck` itself already takes an `std::mt19937&` and is
unchanged.

#### What is NOT verified, and it is the whole verification section

**Nothing below the reading level. No number in this item was measured, and
none could be.** The machine this was written on has MSVC 2022 and WSL but
**no Python 3.11, no `python_ai/venv`, no `perception/.venv` and no built
`clash_royale_env.pyd`** — so nothing that imports the engine runs here at all.

What was actually done: the three paths were read, `sub`-style exact-match edits
were applied to §A and §B, and both files were confirmed to parse under
Python 3.13. That is enough to claim the seed now *reaches* `ClashEnv::seed`,
and it is **not** enough to claim any run is reproducible.

**The acceptance test is the one item 7 already wrote** and it has never been
run against §A or §B:

```python
a = MicroRoyaleEnv(cfg); b = MicroRoyaleEnv(cfg)
oa, _ = a.reset(seed=7); ob, _ = b.reset(seed=7)
assert (oa == ob).all()          # identical opening hand AND cycle order
```

plus, for §B, two `prove_combos.py --seed 300` invocations whose OFF arms
report the **same level**, not merely the same delta — the check that failed in
2026-08-20 and produced the evidence item 7 was argued from. Until someone with
a 3.11 environment runs both, §A and §B are *plausible and unverified*, and this
file should keep saying so.
### UPSTREAM item 24. APPLIED 2026-08-23 — the Giant walks at 0.600 tiles/s and the real one walks at 0.99, measured on two independent recordings (proposed 2026-08-23)

**GAMEPLAY-AFFECTING. Every win rate earned before this is historical.** Checkpoints
are NOT invalidated: the observation, action space and architecture are untouched, and
`sightRange`/`speed` are not observation channels (`CH_RANGE` carries attackRange, and
`CH_SPEED` carries the value being changed, so the OBSERVATION still describes whatever
the entity actually has -- no tensor changes shape).

#### The measurement

`perception/videos/` was read frame by frame: the arena homography in
`config/profile_gpg_1920x1080.json` re-validated against these files (worst anchor error
**0.312 tiles**), units located with `units_M_480x352.onnx`, positions taken at the
sprite's FEET and mapped through the homography. A stationary Cannon read (8.5, 9.2)
across five samples with a spread of +/-0.05 tiles, which is the independent check that
the mapping is stable.

| | speed | window | residual sd |
|---|---|---|---|
| video 1 (`20-45-03`), Giant | **0.999** tiles/s | 26 frames, 6.8 s | 0.174 tiles |
| video 2 (`20-48-33`), Giant | **0.974** tiles/s | 140 frames, 13.7 s | 0.298 tiles |
| engine | **0.600** tiles/s | -- | -- |

Two different matches, opposite lanes, opposite directions of travel, agreeing to 2.5%.
The engine is **1.64x too slow** for this card.

#### CORRECTION, same day: the Fast tier is too slow as well

This item first claimed the defect was Giant-specific, on a Mini P.E.K.K.A fit that
matched the engine. **That fit was contaminated and the claim was wrong.** Its window
(118.8-122.4 s) began and ended while the unit was STALLED -- visible in the trace as a
flat y ~= 17.9 for the first 0.8 s -- which pulled the fitted speed down to 1.528 and
made the engine's 1.600 look correct. Refitted on the clean descent alone:

| | window | speed | residual sd |
|---|---|---|---|
| Mini P.E.K.K.A, contaminated | 118.8-122.4 s | 1.528 | 0.373 |
| Mini P.E.K.K.A, **clean descent** | 120.3-124.5 s, 40 frames | **2.003** | 0.195 |

*A fit window is part of the measurement. Choosing one that spans combat measures the
combat.* Same failure shape as the "three rising samples" walk-start estimator and the
p90 speed estimator, both discarded earlier in the same session.

#### What the corrected numbers say

| card | tier | video | engine | video/engine |
|---|---|---|---|---|
| Giant | Slow | 0.987 | 0.600 | **1.65x** |
| Mini P.E.K.K.A | Fast | 2.003 | 1.600 | **1.25x** |

**Both are too slow, by different factors** -- so this is not one card, and it is not a
uniform clock error either. The invariant that survives is the RATIO, and it is where
the engine actually breaks:

| | Fast : Slow |
|---|---|
| the recordings | **2.03** |
| real game, published tiers (90 / 45 tiles per minute) | **2.00** |
| **this engine** | **2.67** |

The footage reproduces the published tier ratio to 1.5%, which is strong independent
corroboration that the measurement pipeline is sound -- the two cards were tracked in
different matches, different lanes, opposite directions. The engine is the outlier.

Note the units: the published figure is tiles per MINUTE in REAL-game tiles, while
everything measured here is in ENGINE tiles (the homography is fitted to the engine's
own tower layout). Absolute tiles/second therefore are NOT comparable across the two,
and the ratio is -- which is why the ratio is the claim.

#### Why MOVEMENT_SPEED_SCALE cannot fix a ratio

`CardStats.h`'s `MOVEMENT_SPEED_SCALE = 0.2f` is a **global multiplier**, and a global
multiplier preserves ratios by construction. CLAUDE.md records the ratio as wrong -- "the
engine's own Slow:Medium tier ratio (0.60 against the real 0.75)" -- and cites it as one
of the three pieces of evidence justifying the 2026-08-07 movement fix. That fix applied
one scale to every card, so it could move the average and could not touch the per-tier
error. The ratio is still wrong today, in the same direction.

#### The edit

`include/core/CardRegistry.h:695`, the sixth argument (speed):

```cpp
-  add(troop(2, "Giant", 5.0f, Archetype::MeleeBuildingTargeter, 3968, 0.3f, 1.2f, 253, 15, 'G')...
+  add(troop(2, "Giant", 5.0f, Archetype::MeleeBuildingTargeter, 3968, 0.5f, 1.2f, 253, 15, 'G')...
```

`0.5f * MOVEMENT_SPEED_SCALE * 10 ticks/s = 1.000 tiles/s`, against 0.987 measured
(1.3% high). The neighbouring literal 0.4f gives 0.800, which is 19% low.

#### Blast radius, and what was deliberately NOT changed

Six other cards share the `0.3f` literal because they share a tier -- **Royal Giant,
Sparky, Electro Giant, Elixir Golem, Lava Hound, Hero Giant**. Only the Giant was
measured, so only the Giant was changed. The **17 cards on `0.8f`** (the Fast band) are
~1.25x slow by the corrected Mini P.E.K.K.A measurement and were also left alone.

**This edit therefore does not make the tier structure correct.** It puts one measured
card on its measured value. Applying it moves the engine's Fast:Slow ratio from 2.67 to
1.60, against a real 2.00 -- closer in magnitude, wrong in the other direction. Making
the ratio right needs the Fast band moved too (0.8f -> 1.0f, giving 2.000 tiles/s against
2.003 measured), which is 17 further cards and was not done unilaterally.

That leaves a real, deliberate inconsistency, and the reason for accepting it is the
part worth reading. Setting the Giant to `0.5f` puts it on the SAME literal as the 80
cards in the middle band, i.e. it collapses "Slow" into "Medium". That may be correct --
or it may mean the middle band is itself mislabelled -- and the data cannot currently
tell the two apart: the two Medium units measurable in these recordings gave **0.749**
(Valkyrie) and **1.183** (Musketeer), a 1.6x spread inside one tier, because both were
in combat rather than walking cleanly. Sweeping six unmeasured cards onto a number
derived from one card, while the anchor that would justify it is that noisy, is exactly
the extrapolation this document exists to prevent.

**To close it properly**, measure a clean walking segment for a Medium card and for a
second Slow card (Golem or P.E.K.K.A) on these same recordings, then either sweep the
tier or split it deliberately.

#### Reproducing it

```bash
tools/audit/video_replay.cpp        # one card's trajectory, tick by tick
tools/audit/video_scenario.cpp      # a multi-card push vs a defending tower's HP
tools/audit/video_replay_log.cpp    # the same push as a web/viewer.html replay
```

**SUPERSEDED by item 25**, which replaced the whole speed model rather than this one
card. The measurement above stands; the single-literal edit it describes was folded into
the tier rewrite.

---

### UPSTREAM item 25. APPLIED 2026-08-24 — the whole speed model: the engine had no tiers, and 104 of 131 troops were wrong (proposed 2026-08-24)

**GAMEPLAY-AFFECTING, AND THE LARGEST SUCH CHANGE IN THIS FILE. Every win rate, every
Elo anchor and every reward curve earned before today describes a different game.**
`model_weights_selfplay.pth` now plays an environment it did not train in and will be
weaker until retrained. Checkpoints still LOAD — observation, action space and
architecture are untouched — but their measured strength is void.

#### What was wrong

The real game gives every card ONE speed number, in tiles per MINUTE, and it takes only
five values: **30 / 45 / 60 / 90 / 120**. Confirmed against Supercell's own exported
table (`cards_stats_characters.json` in RoyaleAPI/cr-api-data): exactly those five values
across 119 characters, no others.

This engine had **nine** ad-hoc literals and no tier concept at all, so cards sharing one
real tier were scattered across different speeds. Every one of these is Slow (45):

| card | before | after |
|---|---|---|
| Golem | 0.400 tiles/s | 0.994 |
| Giant, Royal Giant, Lava Hound, Electro Giant, Elixir Golem | 0.600 | 0.994 |
| P.E.K.K.A. | 0.800 | 0.994 |

**No value of `MOVEMENT_SPEED_SCALE` could ever have fixed that.** It is a global
multiplier and preserves ratios by construction — which is exactly why the ratio
CLAUDE.md flagged as wrong on 2026-08-07 ("Slow:Medium 0.60 against the real 0.75")
survived that fix untouched, and why `CardStats.h` still carried a comment saying the
tiers were "left uncorrected on purpose".

Measured over the whole registry: **only 5 of 109 resolvable troops were within 5% of
correct. 104 were not.** The dominant error was a uniform **×1.33 on 53 cards** (the
Medium band) with tier-assignment errors layered on top, ranging to ×2.49.

#### The calibration, and why it is not 1/60

A REAL tile is not an ENGINE tile — this board's tower layout differs from the real
arena's — so the conversion is measured, not assumed. Two cards tracked frame by frame
through `perception/videos/` with the homography in `config/profile_gpg_1920x1080.json`
(re-validated on these files, worst anchor error 0.312 tiles):

| card | real stat | engine tiles/s | implied factor |
|---|---|---|---|
| Giant | 45 | 0.987 (two recordings, 2.5% apart) | 0.02193 |
| Mini P.E.K.K.A | 90 | 2.003 | 0.02226 |

**Two cards, two different tiers, one constant, agreeing to 1.5%.** The footage also
reproduces the published Fast:Slow ratio — 2.03 measured against 2.00 published — so the
recordings and Supercell's table corroborate each other independently.

`REAL_TILES_PER_MIN_TO_ENGINE = 0.011045f`, and the five tiers derive from it.

#### Verification, which is the part that makes this safe

The rewrite was scripted, so it was round-tripped rather than trusted: rebuild, re-dump
every spawned troop's actual speed, and assert it equals its official tier times the
calibration. **109 / 109 match, 0 mismatches.**

A first attempt was DISCARDED and is worth recording. It replaced only the first regex
match per card, and several cards appear more than once — `troop(24, "Skeletons", ...)`
exists both as a death-spawn helper and as the playable card. It silently edited the
helper and left the real card alone. The round-trip check is what caught it; reverting
and redoing with all-occurrences replacement is what fixed it.

#### What was deliberately NOT changed

**13 cards have no official row** and keep their previous values: Berserker, Ronin,
Goblin Machine, Goblin Demolisher, Furnace, Heal Spirit, Suspicious Bush, Rune Giant,
Little Prince, Goblinstein, Boss Bandit, Spirit Empress. They are newer than the exported
table. They now sit off-tier (1.000 / 1.400 / 1.700 tiles/s) and should be assigned once
a source covers them.

The **9 Hero variants** WERE changed, by mirroring their base card's tier rather than by
guessing: "Hero Giant" takes Giant's tier, "Hero Knight" takes Knight's. They are
engine-invented and have no real counterpart, but the naming makes the intent explicit,
and leaving them behind would have put Hero Giant at 0.600 against Giant's 0.994.

#### What this does not settle

The calibration rests on **two** measured cards. It is strongly corroborated — different
tiers, different matches, and it reproduces the published ratio — but a third measurement
in the Medium band would make it three-point. The two Medium units available in these
recordings were both in combat rather than walking cleanly (0.749 and 1.183, a 1.6×
spread inside one tier) and were discarded rather than averaged.

`perception/tools/sim_fidelity.py` is the harness the old `CardStats.h` comment named for
settling this, and it has NOT been re-run against the new tiers.



---

### BOT source document: Suggestions for the agent, from `perception/`

Written from `perception/`, which does not modify `python_ai/` or the C++ core.
These are things noticed while building the live sensor that only the training
side can act on.

Companion to `UPSTREAM_REQUESTS.md` (engine changes). Same rules: measured
evidence where it exists, guesses labelled as guesses, blast radius stated.

Last updated 2026-07-30. **Reviewed from the training side 2026-07-30 — see
"Training-side response" at the bottom. Item 2 is resolved and item 4 is
answered by measurement; the header claim that nothing has been applied is no
longer true.**

| # | Suggestion | Value | Confidence | Status |
|---|---|---|---|---|
| 1 | Train against perception-shaped observation noise | **high** | reasoned, not measured | deferred — harness yes, training change not yet |
| 2 | `DEFAULT_DECK` is not the deck being recorded | high, and free | measured | **RESOLVED 2026-07-30**, opposite direction |
| 3 | Channels 0-7 assign rather than accumulate | medium | measured upstream, worse under real perception | agreed; cheaper than stated; queued |
| 4 | `elixirSpent` is the one unrecoverable field | medium | structural | **MEASURED — void as stated, real risk is elsewhere** |
| 5 | Tower levels are asymmetric in reality | low | measured | agreed, no action |
| 7 | Princess HP must be read from the numeral, not the bar | **high** | measured, 101/101 | perception-side; adapter no longer reports 0.0 as destroyed |
| 8 | Opponent cumulative spend over-counts 2.1x — do not emit it | **high** | measured on a live match + 900-episode A/B | **RESOLVED 2026-07-31** — option 1 adopted, measured free |

---

### BOT item 1. Train against perception-shaped observation noise

The policy is trained exclusively on observations generated by the engine,
where every field is exact: unit HP to the float, identity certain, position
the entity's own `position.x/y`. A live sensor cannot deliver that, and the
error profile is not Gaussian — it is structured:

- **HP is quantised, though less coarsely than first assumed.** Every entity —
  troop and building alike — carries a team-coloured level badge with an HP bar
  attached, measured at roughly **42x10 px** at 1920x1080 (`frame_210s`, our
  Cannon at ~35%). That supports on the order of 20-40 distinguishable levels,
  not the 4-6 an earlier eyeball estimate suggested. Full HP is signalled by
  the bar being *absent*, so "undamaged" is exact and everything else is
  quantised to roughly 3-5% of max.
- **Identity is occasionally wrong, and wrong in a biased way.** Measured on
  our own hand over 8 recordings: the icon template agrees with the elixir
  ledger on card cost only **33.8%** of the time, and it over-predicts one
  specific card (Giant) at 35% where its prior is 12.5%. Confusions are
  between visually similar cards, so a mistake lands on a *neighbouring* set
  of attributes rather than a random one.
- **Units are occasionally missed entirely** in a crowd, where badges and
  sprites occlude each other.
- **Position is quantised to a tile** and carries the homography's residual
  (0.63 tiles against current engine geometry, 0.40 with `UPSTREAM_REQUESTS`
  item 1).

A policy that has only ever seen exact observations may be relying on
precision that will not exist at deployment. Domain randomisation over these
specific corruptions during training — HP bucketing, occasional identity
swaps between similar cards, random unit dropout, sub-tile position jitter —
would cost nothing at inference and is the standard fix for exactly this
sim-to-real shape.

**This one is reasoned, not measured.** Nobody has yet run the policy on a
perception-derived observation, so the size of the gap is unknown. It is first
on the list because it is cheap and because the alternative is discovering the
gap after the sensor is finished.

**Blast radius:** training only. No engine change, no observation-layout
change, no checkpoint invalidation. Worth an ablation rather than a blanket
change.

---

### BOT item 2. `DEFAULT_DECK` is not the deck in the recordings

`gym_wrapper.DEFAULT_DECK = [15, 25, 6, 1, 0, 41, 7, 10]` — Hog Rider, Cannon,
Musketeer, Archers, Knight, Minions, Fireball, Valkyrie.

The 8 recorded matches are played with Valkyrie(10), Archers(1), Minions(41),
Cannon(25), Fireball(7), **Giant(2)**, Musketeer(6), **Mini P.E.K.K.A(5)** —
confirmed from the hand icons on screen and from
`config/templates/icons/icons.json`.

Six of eight cards are shared. The differences matter more than the count:
the recordings have no **Hog Rider**, which is the win condition the policy's
whole strategy was built around, and they contain **Giant** and **Mini
P.E.K.K.A**, which the policy has never seen.

Consequence: any observation derived from these recordings puts the policy far
out of distribution, and the one-hot identity block (4 x 185) will light up
slots it has no learned response for.

**The fix is free and belongs to whoever records next: play `DEFAULT_DECK`.**
It costs one deck change in the game client and needs new icon templates
(`tools/build_icon_templates.py`, cheap). The alternative — retraining on the
recorded deck — costs a full run.

Flagged here rather than acted on because it is a decision about what the
agent should be good at, not a perception detail.

---

### BOT item 3. Channels 0-7 assign rather than accumulate

Already known upstream (`CLAUDE.md`, open problem 3): `obs[idx] = normalizedHp`
means the last entity written to a cell wins, so a Skeleton Army collapses.
`CH_COUNT` was added to mitigate it.

Worth re-raising only because **real matches make this common rather than
occasional**. In the recorded frames, units routinely stand in the same tile
mid-fight — two Goblins overlapping so closely that their level badges collide
(`frame_95s`). Under engine self-play with `DEFAULT_DECK` the effect is mild;
against real ladder decks with swarm cards it is not.

No proposal attached. The mitigation already exists and the fix (accumulate,
or store max-HP-of-cell alongside count) is an observation-layout change,
which invalidates checkpoints. Recording it so the decision is informed.

---

### BOT item 4. `elixirSpent` is the one field a sensor cannot recover from

Extra scalars 1 and 2 are *cumulative* per-match elixir spend. Every other
field in the observation is instantaneous, so a sensor that misreads one frame
is corrected by the next. These two are accumulators: a missed opponent
placement is baked in permanently and every later frame inherits the error.

This matters because the auxiliary opponent-elixir head is trained to estimate
hidden elixir *from* these — so an undercount does not merely add noise, it
biases the input the aux head depends on, in one direction (always low).

Not asking for a change. Two things would help if the training side ever wants
them:

- the aux head's robustness to a systematically-low `opp elixir spent` is
  worth an ablation, since that is the deployment condition;
- if the observation ever grows a field, a *count* of detected opponent plays
  alongside the spend would let a consumer notice the undercount. Perception
  can report its own confidence here; today there is nowhere to put it.

---

### BOT item 5. Tower levels are asymmetric in reality

`addTower` gives both teams identical Princess HP (2534) and King HP (4008).
Measured from the recordings, the two players' towers carry **different level
badges** — ours level 4, the opponent's level 5 in
`2026-07-29 20-58-14` — and therefore different max HP.

**Corrected 2026-07-30 — an earlier version of this item claimed the engine's
absolute values match the real game closely enough to feed the scalars
directly. That was wrong.** Measured off a clean frame at t=20s, before anything
is damaged, so the on-screen numbers are the true maxima:

| | recordings | engine |
|---|---|---|
| our Princess (badge 4) | **1750** | 2534 |
| opponent Princess (badge 5) | **1890** | 2534 |

1750 and 1890 are exactly the real game's level-4 and level-5 Princess values;
2534 is its **level 9**. The gap is ~30%, and it differs per player because the
two sides are at different levels.

Handled entirely on the perception side — `GameState` reports tower
`hp_fraction` rather than absolute HP, and the maximum is measured from the
first undamaged reading rather than supplied. **No engine or training change is
requested.**

Still worth knowing on the training side: a policy trained on perfectly
symmetric towers has never seen "my towers are weaker than theirs" as a
starting condition, which is the normal case on ladder.

### BOT item 6. The detector misnames ~15% of the units it finds

Not a request for an engine or training change — a measurement the training
side needs when reading any perception-derived number, because it bounds what
the spatial channels can be worth today.

60 detections were hand-labelled to fit the HP reader (2026-07-31). The labels
came back with a second finding nobody asked for:

| fault | count | detector classes |
|---|---|---|
| wrong card name | 9 / 60 (15%) | `knight` x6, `valkyrie` x2, `archer` |
| not a unit at all | 3 / 60 (5%) | `minipekka` x3 |

`knight` is the worst offender and this is the second time it has been: it also
produced all 102 of the off-board phantoms that `board_filter.py` exists to
reject (the two player avatar icons). On-board it is being applied to
Barbarians and to a Mini P.E.K.K.A. The three "not a unit" cases are all
`minipekka`, and one was confirmed by eye to be the enemy **Princess tower**.

Two consequences that matter downstream:

- A wrong card id is not a small error. `card_sim_id` drives channels 11-20
  entirely — flying, anti-air, DPS, range, speed — so a misnamed unit writes a
  confidently wrong attribute row, which is indistinguishable from a correct
  one. It is worse than a dropped detection.
- A tower detected as a unit is worse still, because towers are already
  represented separately, so it double-counts.

Neither is fixable in `perception/`: the model is upstream's, vendored
unchanged so it stays updatable. Filing it here so the number is on record when
item 1's corruption ablation is finally run — **unit identity error is measured
at ~15%, not hypothetical**, and that is the corruption worth measuring first.

---

### BOT item 7. Princess HP should be read from the NUMERAL, not the bar

**Perception-side work, not a training request.** Filed here so the tower-HP
scalars (extra 3-8) are not trusted more than they deserve until it is done.

CRBAB's `_calculate_hp` matches the bar's two colours and returns 0.0 when it
can match neither. Measured on the first live 549×976 capture, over 101 frames
of one match:

| tower | 0.0 readings | truth |
|---|---|---|
| `right_ally_princess` | **101 / 101** | standing, **full** — its ROI reads 1890 |
| `left_ally_princess` | 4 / 101 | standing, 1759 |
| `left_enemy_princess` | 4 / 101 | standing |
| `right_enemy_princess` | 4 / 101 | standing |

The 4 shared zeros are pre-match frames. One tower in four therefore fails
*completely and silently*, and the failure is the same value as "destroyed".

The adapter previously resolved 0.0 → `destroyed=True`, justified on 71 ladder
frames where a real decay to zero looked monotone and physical. That
justification is now falsified — a full tower reads 0.0 — and the adapter
reports `hp_measured=False` instead. A live tower reported dead tells the
policy a lane is already lost.

**The numeral is the better signal and it is right there.** 1759 and 1890 are
legible at this resolution, give ABSOLUTE HP, need no colour match, survive
occlusion of the bar, and remove the tower-level problem in item 5 entirely —
no max-HP table, no per-account level. `readers/clock.py` already has a
digit-template classifier built for glyphs of this kind.

Same convention already proven on the King (`live/king_hp.py`), which reads its
numeral region and validated 8/8.

---

### BOT item 8. Opponent cumulative spend over-counts 2.1× — do not emit it yet

Measured on the first live match (204 s in-game, 943 detector frames).

**Our own spend works.** It is read from the elixir bar, which is the strongest
reader in the pipeline, with the hand used only to name the card afterwards:

| | |
|---|---|
| cards implied | **27** |
| affordable ceiling at the measured regen | **~28** |
| conservation residual | **+14%**, one-sided |

The residual is the right sign and size: `gained − spent − (final − initial)`
is positive because regen while the bar sits at the 10 cap is invisible, and
the bar was capped in 13% of samples.

**The elixir reader also recovers the real game's schedule from pixels alone**,
which is a strong independent validation:

Binned against **match** time, whose origin comes from the on-screen clock
rather than from the recording (see below):

| match elapsed | median | phase |
|---|---|---|
| 20–100 s | 2.73, 2.77, 2.85, 2.81 s/elixir | **1× = 2.8** |
| 100–120 s | *(n=2, too few to read)* | boundary |
| 120–200 s | 1.42, 1.42, 1.50, 1.43 s | **2× = 1.4** |

Within ~1.5% of both rates, and the 1×→2× transition localises to **[100, 120] s**
against a real boundary at 120 s.

**Correction to an earlier version of this entry**, which said the boundary
"landed exactly at t=120 s". That was measured in 60-second bins against the
first in-game FRAME, not against match start — it coincided numerically with
the right answer from the wrong origin, and 60-second bins cannot localise a
boundary to better than ±30 s anyway. The table above is binned at 20 s from
the clock-derived origin.

The clock origin is itself measured, not assumed. Reading the panel on five
frames and adding the recording timestamp gives a constant:

| wall t | clock | sum |
|---|---|---|
| 39.9 s | 2:40 | 199.9 |
| 73.4 | 2:07 | 200.4 |
| 107.2 | 1:33 | 200.2 |
| 142.0 | 0:58 | 200.0 |
| 173.5 | 0:27 | 200.5 |

Constant to ±0.25 s, so `remaining = 199.9 − t` and match start is t = 19.9 s.
The panel turns red with "Overtime" at t ≈ 200.2, i.e. **180.3 s elapsed**,
which is 3:00 of regular time to within a second. Two useful consequences: the
clock reader's digit templates can be auto-labelled from timestamps with no
hand annotation, and the first 20 s of overtime measure **2×, not 3×** — stated
as measured, since this recording is one match against one bot.

One reader caveat: it emits **spurious single-frame drops to 0** (4.2% of
samples), because `_calculate_elixir` takes the first window whose rolling std
falls under a threshold. A double 3-median removes them; the physics is what
makes that safe.

**The opponent's spend does not work, and the failure is structural.** There is
no opponent elixir bar, so spend can only come from units appearing:

| | |
|---|---|
| placements detected | **64** (208 elixir) |
| physically affordable | ~108 elixir → **~31 cards** |
| implied opponent elixir | **−103**, below zero for **98%** of the match |

The sign is the diagnostic: negative means placements were *invented*. Two
rounds of fixes took it 141 → 64 (longer re-association, grouping multi-unit
cards, tolerating the detector's renames) and it is still 2.1× over.

**It cannot be tuned away.** Reaching ~34 placements needs grouping at 4 s and
12 tiles — and 12 tiles is two-thirds of the board's 18, so a Giant at one
bridge and Archers at the other three seconds later merge into one placement.
The parameter that fixes the count destroys the distinction the count is for.

Two by-products worth keeping:

- **The opponent's deck falls out of the frequency histogram.** The 8 most
  common cards are exactly 8 plausible ones; all 7 remaining are singletons and
  all 7 are misclassifications. That is what `track/opp_deck.py` is for, and it
  is a cheap, reliable signal even though the count is not.
- 46 of 64 placements are single-unit, though the opponent's deck is full of
  3-unit cards. The multi-unit groups are being split, which is the same
  fragmentation seen from the other side.

**Decision needed.** Options, in the order I would take them:

1. **Emit our own spend, mark the opponent's unmeasured.** Costs one of nine
   extra scalars and fabricates nothing. Needs the training side to say what an
   unmeasured scalar should carry.
2. Build a real tracker — Hungarian assignment with a motion model, constrained
   by the recovered deck. This is the actual fix and is not a small job.
3. Clamp the ledger to physics (spend cannot exceed elixir earned). Bounds the
   error but biases it systematically high, and hides the failure.

Recorded because item 4 already flagged `elixirSpent` as the one unrecoverable
field. That was reasoning; this is the measurement, and it is worse than the
reasoning assumed.

#### Update 2026-07-31 — the network needs this far less than I argued

I claimed the aux elixir head depends on extra scalar 2, reasoning that current
elixir is near-arithmetic on two supplied inputs
(`now ~= start + regen(t) - spent`, with `t` scalar 0 and `spent` scalar 2).
**Measured on the live phase-2 checkpoint at episode 75,045, that is wrong.**

24 episodes collected once and replayed under each corruption, so trajectories
are identical and the comparison is paired. The transform was verified to
mutate exactly one column, index 13599:

| condition | MAE | vs baseline | % of the way to predict-the-mean |
|---|---|---|---|
| baseline | 0.966 | — | — |
| **scalar 2 zeroed** | **0.962** | **−0.005** | −1% |
| scalar 2 × 2.1 (our measured error) | 0.972 | +0.005 | 2% |
| scalar 2 saturated at 1.0 | 1.006 | +0.039 | 12% |
| predict-the-mean | 1.299 | | 100% |

Deleting the field costs nothing. Even pinning it at 1.0 costs 12% of the gap
to the trivial baseline. The response is monotone in perturbation size, so the
dependence is real but weak.

Two consequences:

- **Zeroing beats passing our inflated estimate** (−0.005 against +0.005), so
  the option that fabricates nothing is also the empirically better one. Those
  usually trade off.
- The absolute level here (0.966) is worse than the 0.770 on record because
  this is a phase-2 self-play checkpoint measured against the phase-1 heuristic
  opponent, over episodes averaging 81 steps. The paired comparison is
  unaffected.

**This measures the aux head, not the policy.** Scalar 2 also feeds the card,
placement and value heads through the same trunk, and any of those can degrade
while a 257-parameter linear probe sits still. A win-rate test against
`heuristic@1.35` is the deciding measurement; an n=20 pilot put the delta at
−0.100 ± 0.183, i.e. pointing the wrong way but not yet distinguishable from
zero.

Note also that any such test measures the cost of **removing an input from a
network that learned with it**. A policy trained with the field zeroed from the
start could adapt; that is a different and more expensive question.

#### RESOLVED 2026-07-31 — zeroing costs nothing, measured

900 episodes against `heuristic@1.35`, arms interleaved so machine load (the
phase-2 run was training throughout) falls on both equally, actions sampled
rather than greedy to match deployment:

| arm | W / L | score |
|---|---|---|
| scalar 2 correct | 368 / 82 | 0.8178 |
| scalar 2 zeroed | 382 / 68 | 0.8489 |

**delta +0.031, 95% CI [−0.018, +0.080], p = 0.21.** The interval excludes any
degradation worse than **1.8 points** against a 5-point threshold. The nominal
+3.1 favouring zeroing is not significant and should not be read as a benefit;
there is no mechanism by which deleting an input helps.

**Option 1 adopted.** `GameState.opp_elixir_spent` is `None` — never `0.0`,
since a zero cannot be told apart from "they have spent nothing". What the
encoder writes into observation slot 13599 for a `None` is the training side's
call, not perception's.

Our own spend is emitted and is tracked in `live/elixir_ledger.py`, which is a
separate module from the adapter on purpose: the adapter is stateless per
frame, and a cumulative total is the opposite. Replayed over the same match it
gives **29 cards and a +12% residual**, against the offline analysis's 27 and
+14% — the small gap is the live version's causal 3-median, which cannot see
the future and therefore lags one sample.

Three limits on this result, none of which change the decision:

- It is one anchor. `heuristic@1.35` is out of distribution for a self-play
  trained net, which is the leading explanation for why the zeroed arm led
  nominally. A self-play version of the test (null exactly 0.500, so the
  control arm self-validates) is written and unrun — it costs ~1568 per arm
  because variance peaks at p=0.5.
- It measures **removing an input from a network trained with it**. A retrain
  without the field could do better or worse.
- Arms could not be paired: the engine's RNG has no setter, which is
  `UPSTREAM_REQUESTS.md` item 7 and is why this needed 900 episodes instead of
  a tenth of that.

---

### BOT source document: Training-side response, 2026-07-30

Reviewed by the training side. Verdict per item, with the measurements that
were missing.

### BOT — Item 2 — RESOLVED, in the opposite direction to the recommendation

This item recommended re-recording with `DEFAULT_DECK` and called the
alternative "costs a full run". The alternative was chosen deliberately: on
2026-07-30 `DEFAULT_DECK` became `[10, 1, 41, 25, 7, 2, 6, 5]` — exactly the
recorded deck — and a fresh run started on it. The full-run cost was knowingly
paid, because human demonstrations are the highest-value unblocked item and are
worthless against a deck the policy does not play.

The trade this item did not price: the recorded deck is costs 3-5, avg 3.75,
spread 2, which is the deck `gym_wrapper.py` had already documented as the one
where **Giant was never played once across four full runs**. That risk is
re-accepted with eyes open. First measurement after 13,520 episodes on the new
deck, via `python_ai/eval/probe_card_usage.py`:

| card | cost | % of plays | % steps legal | take-up |
|---|---|---|---|---|
| Archers | 3 | 20.9% | 6.1% | 3.42 |
| Cannon | 3 | 19.4% | 9.6% | 2.01 |
| Minions | 3 | 19.1% | 4.3% | 4.39 |
| Mini PEKKA | 4 | 19.1% | 3.8% | 5.06 |
| Valkyrie | 4 | 18.5% | 4.3% | 4.25 |
| Musketeer | 4 | 2.8% | 6.4% | **0.44** |
| Fireball | 4 | 0.3% | 6.6% | **0.05** |
| Giant | 5 | **0.0%** | 1.6% | **0.00** |

Take-up = play share / legality share; 1.0 means picked in proportion to
opportunity. Two distinct failures, needing opposite fixes: **Giant is starved**
(legal on only 1.6% of steps — the cost curve, as predicted), while
**Musketeer and Fireball are declined** despite being legal about as often as
Cannon. Note Fireball's low usage may be correct — forcing it was previously
measured to drop win rate 97% -> 23%.

### BOT — Item 4 — measured, and the concern is void as stated

The structural argument is right (cumulative fields cannot self-correct, and the
error is one-directional). The conclusion does not follow, because **the aux head
does not use those fields at all.**

Controlled ablation (`python_ai/eval/probe_aux_robustness.py` plus a follow-up over
every input group): clean rollouts recorded once, then the network replayed over
the *same* observations with inputs zeroed, so trajectory and targets are fixed
and MAEs are comparable. At episode 13,520, opponent elixir mean 2.37 / std 1.49,
predict-the-mean baseline MAE 1.187:

| ablation | aux MAE | delta |
|---|---|---|
| clean | 0.956 | — |
| time fraction -> 0 | 0.956 | +0.000 |
| own elixir spent -> 0 | 0.956 | +0.000 |
| **opp elixir spent -> 0** | 0.955 | **-0.001** |
| all three -> 0 | 0.955 | -0.001 |
| whole appended 9-scalar tail -> 0 | 0.929 | -0.027 |
| **entire spatial block -> 0** | **2.298** | **+1.342** |

Zeroing the opponent-spend scalar *entirely* — a 100% undercount, far past any
realistic sensor error — moves MAE by 0.001. The head is estimating opponent
elixir from **the board**: units present imply elixir recently spent. Removing
the board pushes it to 2.298, well past the 1.187 baseline.

**So a perception undercount of opponent spend cannot degrade this head.** But
the risk does not disappear, it relocates: the head depends entirely on the
spatial channels, which is precisely what perception is *worst* at — missed
units in crowds, occluded badges, identity confusion. That is items 1 and 3,
not item 4. This item pointed at the wrong field.

**Caveat:** measured at 13,520 episodes with the head only ~20% better than
guessing the mean (0.956 vs 1.187); it reached 0.76 in an earlier run. The
result could change as the head improves, and re-running the ablation is cheap.

### BOT — Item 3 — agreed, and the fix is much cheaper than this item states

This item says the fix is "an observation-layout change, which invalidates
checkpoints". That is true of *accumulating*, but not of the fix actually worth
making:

```cpp
obs[getIndex(channel, y, x)] = normalizedHp;                      // now: last writer wins
obs[getIndex(channel, y, x)] = std::max(obs[...], normalizedHp);  // proposed
```

One line. No layout change, no size change, no structural invalidation. It makes
channels 0-7 mean "the strongest unit in this cell" instead of "whichever entity
the iteration happened to reach last", and makes them consistent with
`CH_DPS`/`CH_RANGE`/`CH_SPEED`, which already take the max.

Summing would be wrong for a reason this item does not mention: three Skeletons
would then read like a PEKKA, which is exactly the confusion the type channels
exist to prevent. Multiplicity already has a home in `CH_COUNT`.

Queued rather than applied: it changes observation semantics, so it wants a
clean restart, and three runs have already been lost this week to mid-run engine
changes. Batch it with the next intentional observation change.

### BOT — Item 1 — agreed in principle, deferred on timing, and one number looks like a sensor bug

Three objections, in order of weight:

1. **"the icon template agrees with the elixir ledger on card cost only 33.8% of
   the time" is not a training requirement, it is a broken sensor.** For 8 cards
   spanning 5 distinct costs, 33.8% is barely above chance. The fix is the
   sensor. Training a policy to tolerate ~66% identity error would cost real
   capability and buy nothing once perception improves.
2. **Premature.** Domain randomisation trades peak performance for robustness.
   The policy is at ~0.58 win rate against a heuristic bot at curriculum stage 3
   of 6 — nowhere near a ceiling worth trading away. Standard sequencing is
   competence first, randomisation as fine-tuning.
3. **Unfalsifiable right now.** No one has run the policy on a perception-derived
   observation, so adding corruption would produce no measurable answer about
   whether it helped.

What the training side would accept today, and what this item itself suggests
("worth an ablation rather than a blanket change"): build the corruption set as
an **evaluation** wrapper and measure how much the current policy degrades under
each corruption separately. Cheap, touches no training, and produces the number
that would justify the training change. `probe_aux_robustness.py` is the pattern.

Given item 4's result, the highest-value corruption to measure first is **unit
dropout / identity error in the spatial channels**, not scalar noise.

### BOT — Item 5 — agreed, low value, no action

This item requests no change and the training side concurs. Its one real point
(the policy has never seen "my towers are weaker than theirs" as a starting
condition) is a domain-randomisation item and belongs with item 1.

