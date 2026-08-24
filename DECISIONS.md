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
