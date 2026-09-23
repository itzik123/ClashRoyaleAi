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

The last section, "the history that used to live in the source comments", holds
what was moved out of code comments on 2026-09-24, grouped by module.

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

## 2026-08-26 (part 3): the registry sweep, and a question that beat a better instrument

Covering the ~165 non-deck cards, `GameLogger.h`, `TerminalRenderer.h` and
`src/main.cpp`. Suite 669 -> **673 cases / 6,511 assertions**, exit 0, still one
expected failure.

### Nineteen spawned units were a quarter slower than their own card

The headline is not the defect, it is how it was found -- because I got it wrong
twice first.

A source scan for non-`SPEED_*` literals found 54 and I read that as "the
2026-08-24 tier pass missed ~40 units". Then I built
`tools/audit/spawn_speed_audit.cpp` to *measure* rather than infer, and it
reported only 8 off-tier, of which 5 were the documented "no official row"
cards. That looked like the source scan had been alarmist and the registry was
nearly clean.

**Both readings were wrong, and the instrument was the reason.** It classified
each unit against its NEAREST tier, so it was asking *"is this on a tier"* --
and `0.5f` resolves to 1.000 tiles/s, which is 0.6% off `SPEED_SLOW`. A unit
that should be MEDIUM but sits on SLOW passes that check perfectly. The tool was
built to catch exactly this class and could not see it.

The question that works needs no external source: **does a spawned unit agree
with the playable card of the same name?** Same hp, same range, same damage,
same cooldown, different speed -- the registry contradicting itself, and the
card is the copy the tier pass actually reached.

| unit | card | spawned copy | |
|---|---|---|---|
| Goblins | 2.651 | 2.000 | -24.6% |
| Spear Goblins | 2.651 | 2.000 (one helper 1.000) | -24.6% / -62% |
| Barbarians | 1.325 | 1.000 | -24.6% |
| Phoenix | 1.325 | 1.000 | -24.6% |
| Skeletons | 1.988 | 2.000 | +0.6% |

Nineteen child registrations retiered to their card's constant. The test is
GENERIC -- it walks every card, spawns it, runs 120 ticks so periodic spawners
fire, kills everything so death spawns land, and compares every resulting troop
against the card of its own name. Four compound cards are excluded by name
(Goblin Machine, Goblinstein, Ram Rider, Rascals), whose secondary unit is
deliberately a different creature registered under the parent's name.

**The lesson is the one CLAUDE.md already states in another form:** a check that
shares an assumption with the thing it checks cannot see past it. "Near a tier"
and "on the right tier" are different questions, and only the second one is the
invariant.

### The renderer was an eighth copy of the arena

`TerminalRenderer::render` painted the river as
`(x >= 3 && x <= 5) || (x >= 13 && x <= 15)` -- bridges at columns 3-5 and
13-15 against a real 2-3 and 14-15. **Four of eighteen columns wrong**: column 2
is bridge and was drawn as water; 4, 5 and 13 are water and were drawn as
bridge. A three-tile bridge, i.e. the pre-2026-08-21 shape.

It is the copy that least deserved to exist. CLAUDE.md argues at length that
`web/viewer.html` is *structurally forced* to restate the geometry, its only
input being a replay JSON that carries none. The renderer has no such excuse --
it holds a `const Board&`, and `Board::isOnBridge` is public. It now derives.

**A trap worth recording about the test.** The obvious check -- compare the
rendered row against `Board::isOnBridge` -- is circular, because `riverRow()` is
built from `isOnBridge`. So is the physics probe: `clampToBoard` calls
`isOnBridge` too. The genuinely independent anchor is the literal
`WWBBWWWWWWWWWWBBWW`, the same string the observation-encoder fix was verified
against. The physics comparison is kept anyway, for the failures that are NOT
the predicate -- wrong row index, wrong width, an off-by-one in the paint loop,
which is most of what actually went wrong.

### `src/main.cpp` totalled an X-Bow as a Princess Tower

Same symbol-alias class as the Mortar/King defect in part 1: it summed King HP
by `symbol == 'R'` and Princess HP by `symbol == 'P'`. Card id 92 (X-Bow) is
registered with `'P'` and card id 93 (Mortar) with `'R'`, so a deployed X-Bow
inflated the Princess total and a Mortar the King's. Display-only in the demo
binary, and fixed with `isTower()` for consistency.

### What the sweep did NOT find, which is worth stating

Reported as negatives so nobody re-derives them:

- **No duplicate playable card ids** across 130 `add()` calls.
- **No evolution stat inversions.** A first scan flagged Evolved Skeletons at
  cooldown 11 -> 12; that was my regex running past the block terminator and
  picking up Bats from the next registration. Both halves are 11.
- **The stun/slow convention holds everywhere else.** Every `FreezeOnHit` in the
  registry was checked against its card: `0.0f` for stuns (Electro Wizard,
  Electro Dragon, Electro Spirit, Zappies, Freeze, Goblinstein), the 0.5-0.7
  band for slows (Ice Wizard, Giant Snowball, Earthquake, Princess Evolution).
  Ice Golem and Ice Spirit, fixed in part 1, were the only two violations.
- **`GameLogger::resultJson` was already right**, and pointedly so: it keys on
  the reserved tower `cardId`s rather than the symbol, which is exactly the
  guard `MatchRules::evaluate` was missing. Its JSON escaping is also complete,
  including the `\u` path for control characters.
- **"Melee with long range" is a modeling convention, not a bug.** Electro
  Wizard, Minions, Inferno Dragon and Electro Dragon are `MeleeSquad` with
  ranges of 2.5-5.0 because `MeleeTroop::performAttack` is direct damage and
  spawns no projectile. The archetype name means "direct damage" here.

### And a documentation defect that would have cost someone a build

CLAUDE.md said "CMake globs `tests/**`". It does not -- `CMakeLists.txt` globs
`tests/entities/*.cpp tests/core/*.cpp`. A test file added under any other
subdirectory is silently ignored: it compiles nothing, registers nothing, and
the build reports success. Corrected, since the existing note about needing two
builds would otherwise send someone hunting the wrong cause.

---

## 2026-08-26 (part 4): `eval/` and `tools/`, and a gate check that could not fail

Same defect classes as the C++ audit, hunted in the measurement harnesses.
Python suite 500 -> **505 passed, 2 skipped**.

### The pre-flight gate's legality check was structurally incapable of failing

`tools/validate_pipeline.py` -- the 20/20 gate -- carried:

```python
t = AT.target_logits_for(obs, cid, legal[cid])
if np.isfinite(t[~legal[cid]]).any():
    illegal += 1
```

`advisor_target._standardize` builds its output as `np.full(N_CELLS, -inf)` and
only ever writes cells that are IN `legal`. So `isfinite(t[~legal])` is False for
every possible input, on every board, forever. The gate printed "0 violations
over N targets" and would have printed exactly that with the legality table
completely wrong.

**Measured rather than reasoned**, because inference had already misled me twice
in part 3: 106 real targets over a real match, and not one had finite mass
outside `legal`. The condition never fired because it cannot.

CLAUDE.md lists this trap by name from a previous occurrence -- *"Never validate
a mask against the predicate that generated it... The check was circular and
could not fail"* -- and this is the same shape, in the file whose whole job is
catching things before a run starts.

The fix consults the ENGINE, which was bound and available the whole time
(`bindings.cpp:181`, `is_valid_placement` -> `ClashEnv::isValidPlacementForCard`).
That is strictly stronger: `legal` comes from the NET's `_placement_legal`
table, and whether THAT agrees with what the engine accepts is the question
worth asking -- it is exactly the disagreement that once put placements off the
board entirely. It is sub-sampled on the same cadence as the engine-scored value
probe, because it costs one pybind call per finite cell per card.

A second `check()` asserts the probe actually ran, since a zero denominator
would make the first one pass for the worst possible reason.

### The Cannon pull-pocket was measured around the wrong centre

`eval/probe_perfect_defense.py`:

```python
# x=9 is the board centre; ...
central = np.mean((np.abs(xs - 9.0) <= 3.0) & (ys >= 5) & (ys <= 12))
```

The centre is **8.5**, not 9.0 -- x is a cell index in [0, 17], so the centre and
the fixed point of the mirror `17 - x` is `(18-1)/2`. 9.0 is the precise
half-tile error that put the whole arena off-centre until the 2026-08-21
re-centring, and the comment asserted it as fact.

The window was therefore `[6.0, 12.0]` instead of `[5.5, 11.5]` -- shifted half a
tile toward the right lane, counting x=12 as central while excluding x=5.5. This
probe reports the number the Cannon reward-hacking investigation reads. Now
derived from `engine_constants.BOARD_CENTER_X` (8.5, confirmed against the live
bindings), with the half-width named rather than left as a second bare literal.

### Thirty flat-cell decode sites restated the board width

`cell % 18` / `cell // 18` across eight files in `eval/` and `tools/`.
`MicroRoyaleNet.cell_to_xy` -- the canonical decoder the placement head itself
uses -- derives this from the engine (`self.board_width`). Every harness that
retyped it as 18 is another copy of a board constant, and if the grid ever
changes the net decodes correctly while these scripts silently feed the engine
transposed coordinates. All thirty now read `engine_constants.BOARD_W`.

### What was checked and found clean

- **The tower-count scoring pattern has NOT regrown.** `6790e91` guards it with a
  grep-based test after eight instances in three waves; the tree scans clean.
- **`profile_training.py`'s `n = 13606`** is a dead initializer, overwritten
  immediately by the live engine's `observation_size()` or derived from the
  header. Not a stale copy.
- **`np.bincount(..., minlength=612)`** is benign: `minlength` only pads, and
  `bincount` sizes itself to the data regardless.
- **`3600` as maxTicks** appears ~20 times, but it is the engine constructor's
  own default being passed explicitly, not a value that can go stale silently.
  Left alone; noted.

---

## 2026-09-23: after the pre-launch audit -- main could not start a run

One unattended session, commits `6fabb53 .. 7188328`. Python only; the two
engine findings are proposals (`perception/UPSTREAM_REQUESTS.md` items 29, 30).

**The first finding was that `main` was broken.** A session on 2026-09-16 had
left TODO 00.3 half-applied and uncommitted: `rewards/shaping.py` read `spell_*`
keys nothing produced, so `compute_shaping` raised `KeyError` on the first step
of every episode. The runbook's launch would have died at once. The work was
FINISHED rather than reverted (its design was right), and the original is kept
as a tagged stash, `superseded-wip-00.3-2026-09-16`.

What landed, each with its control:

| item | what was wrong | the control that held |
|---|---|---|
| 00.3 spell terms | keyed to card 7; a Rocket deck read 0 on 2,372 of 2,372 steps | 2.6 deck bit-identical, 0 of 2,280 steps differ |
| spell probe | read DoT spells for 40 ticks: Poison 368 of 736 | only 2 of 22 spells' values moved; every radius identical |
| teacher spell aim | every spell aimed with Fireball's disc; Rocket +38% value with its own | Fireball's cell identical on 320/320 boards |
| 0c overflow test | failed 3 in 20 on untouched main (flag = "touched the cap") | constructed: identical scalars, elixir 7.00 vs 0.45 |
| 00.6 spawners | Splashyard named TOMBSTONE its win condition | all 16 pool decks resolve as before |
| 00.4 PBRS docs | "EXACTLY ZERO" / "policy-invariant" were false | measured residues; code left, one-line option given |
| 00.9 settings stamp | a resume under different CLASH_* was silent | a same-settings resume prints nothing |
| 00.5 air defence | rung 0 answered a Balloon with Skeletons | Hog control bit-identical; Balloon 1481 -> 1131 |
| 00.8 phase-2 PFSP | per-opponent estimates lost on every resume | a real match is counted exactly once |

Every behaviour-change test was checked to FAIL with the old behaviour patched
back in, and every control to pass under both.

**Three first readings of mine were wrong, and each was caught by measuring
before acting.** They are the part of this entry most worth keeping:

- **"The overflow test failure is mine."** The teacher commit landed just before
  it appeared. Twenty runs of the test's own sampler on untouched main: 3
  failures on each side. Pre-existing; fixed on its own merits.
- **"The observation's anti-air channel says every card hits air."** A probe read
  1.0 for Knight, Hog and Skeletons. The channel is right per entity; the probe
  took the max over a whole plane that also holds the caster's own TOWERS, which
  all target air. Read at the card's own cell, all 13 cards were correct. The
  same mistake is now warned against in `card_probes.damages_air`.
- **"Spells beat win conditions per elixir."** Rocket's 247 tower HP per elixir
  looked like it out-scored real win conditions -- but the resolver scores those
  on an EMPTY board, where it saturates at one Princess (an undefended Hog reads
  634). The comparison was dropped before it reached a document.

**Measured and left for a decision:**

- **Spells hit Crown Towers for 100% of their damage** (UPSTREAM 29). Real game:
  15-30%. One Rocket on a tower pays the agent +0.185 shaping (real +0.046); in
  48 teacher-vs-teacher matches direct spells are 7.0% of all tower damage (1.3%
  at real ratios), 83% of it The Log, 53% of whose casts roll into a tower.
- **Spawned Spear Goblins and Rascal Girls cannot hit air** (UPSTREAM 30):
  Goblin Gang, Rascals and Goblin Hut deal 0 to a held Balloon over 50 s; their
  helpers lack `.withTargetsAir()`. (Night Witch looked the same and was
  CLEARED: her bats deal 1215 to a held Balloon. The zero was the probe -- a
  control our own towers saturated, and an attacker that walked off.)
- **The ladder's bottom is flat.** A rung-r teacher against rung 0, 2.6 mirror,
  24 seat-swapped matches each: rungs 1 and 2 score 0.375 and 0.583 -- not
  measurably harder than rung 0 -- while rung 3-4 reach 0.79 and rungs 5-10 win
  every match. A lone push is defended WORSE at rung 2 than at rung 0 (Hog 1294
  vs 687 tower HP lost, 12 seeds): a 2 s rollout cannot see a push arrive. Not
  harmful to learning (an agent that clears rung 0 clears these quickly), but
  three rungs of little signal.

Python suite 931 -> 1000 passed (2 skipped); the runbook's pre-flight 22/22 on
the new main; 28 plausible trainee decks, three of them Champion decks, pass a
construct-and-play sweep and are tabled in `docs/runbooks/FINAL_RUN_RUNBOOK.md`.

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

---

## 2026-09-24: the history that used to live in the source comments

The source comments were cut back to what the code needs. The measurement history they
carried (what a constant was before, which bug a guard exists for, the number that
settled a design) is kept here, grouped by module. Each entry names the file it came
from. The code and `.claude/CLAUDE.md` describe the current state; this section records
how it got there. Figures are as recorded at the time, so read an old measurement with
its date in mind.

### python_ai/

#### Rewards

- **W_ELIXIR_TRADE, 0.15 -> 0.03.** The original term was symmetric (it also charged the agent's own spend). Charging at the instant of spending, while the payoff arrived gradually and discounted, made doing nothing a guaranteed-zero outcome; self-play converged to both sides holding elixir into draws. Made one-sided. A reward decomposition over 12 games then put it at +0.3585 of the discounted return, more than the +0.2781 for winning; efficient defence maximises it without ever threatening a tower. The greedy policy never played its win condition across 47,000 self-play episodes and beat an attacking scripted opponent 13-0-2 without one. Cut to 0.03.
- **W_TOWER_DESTROYED = 0.6.** Exists because pure-PBRS tower shaping left pure defence optimal (win-condition usage rose to 7.3%, then decayed to 0.7% over 8,300 episodes). It was set "above the discounted value of a win (~0.28)" at ~112-decision episodes and gamma 0.99. The 2026-08-07 speed fix tripled match length, so by 2026-08-27 it paid 22.4x a win. With gamma 0.999 a win is worth 0.999^360 = 0.698, so a crown now pays ~0.86x a win. Whether the bias is still needed is an open training-run question.
- **DRAW_PENALTY 0.2 -> 1.0.** 0.2, discounted over a long episode, could not outweigh the per-step "safe to do nothing" incentive once self-play drifted toward mutual passivity.
- **W_FLAWLESS_DEFENSE / FLAWLESS_REQUIRES_CROWN (2026-08-17).** The tower term is linear and symmetric, so "take no damage" is not expressible in it; weighting damage taken above damage dealt would reintroduce the pure-defence optimum. Gating the bonus on a win makes it rank wins only. Against a weak opponent, though, winning was nearly free, and conceding a Princess cost 2534/9076 of the bonus = 0.140 while a successful Hog paid 0.5 * 470/4008 = 0.059 through the tower potential. At ep 6,053 the Hog fell to 0.8% of plays, Fireball 0.6%, Cannon 1.6%, with 78.6% of plays on four cheap defensive cards (prove_hog.py showed the Hog's placement was fine: +28.5 over a random cell). Requiring a crown aligned the two terms.
- **W_WIN_CONDITION_DAMAGE = 1.0.** Chosen over forcing the card head toward the Hog, which was measured harmful (forcing Fireball dropped win rate 97% -> 23%). A connecting Hog deals ~470-630; at 1.0 that adds ~0.117, making a successful Hog worth ~0.176 against the 0.140 a lost Princess costs: just past break-even.
- **Spell terms follow the deck (2026-09-16).** Before, the lethal term was keyed to Fireball (card 7) and the value term priced every cast at 4 elixir, which would pay +0.5 for an even Rocket trade. `SPELL_SOLVENCY_RESERVE = 4.0` was retired; the reserve is the spell's own cost.
- **spell_value_weight was dead code until 2026-08-14.** Both trainers called compute_shaping without `w_spell`, so the weight sat at 0.08 for all of training and the anneal never ran; `test_compute_shaping_actually_responds_to_w_spell` now guards it. Gameplay-affecting. `SPELL_VALUE_ANNEAL_START` exists because the checkpoint at the time was at episode 64,309 against a 40,000-episode horizon, which would have pinned a resumed run at FINAL and made the anneal unobservable.
- **auto_reset_mask.** On an auto-reset the potential terms evaluate gamma*Phi(new episode) - Phi(finished episode): measured -0.3038 for a match 3000 tower damage ahead. The guard used to be two copies of `* (1 - prev_dones)` in base_trainer and exploiter; the exploiter shipped without its copy through burst #0.
- **tower_potential, towers only (2026-08-06).** It used to read building damage (towers plus deployed buildings). Losing the Cannon's 824 HP cost 0.5 * 824/4008 = 0.1028 while killing with it paid 0.1 * hp/4256, so it had to kill 5.3x its own HP to break even; decay emits no damage event, so parking it was free. On the ep-130,306 checkpoint 27.9% of Cannons went behind the King at (11,2)/(11,3), mean y = 6.3.
- **compute_shaping's gamma.** It used to default to a literal 0.99, correct only while PPOConfig.gamma was 0.99. The earlier tower term also dropped the gamma entirely (w * (Phi(s') - Phi(s))), which is not policy-invariant and let the agent trade winning for accumulated shaping.
- **Elixir solvency (elixir_shaping.py).** `model_weights_dist_e3.pth`, 40 greedy episodes at 1.5x opponent elixir: it spent ~105 elixir per episode against ~98 income, sat below 3 elixir on 65.3% of decisions and on 60.8% during a big enemy push. Conditioned on affordability its threat response was real (27.8% -> 34.9%): insolvency, not apathy. The card-entropy controller was tested as the cause and refuted over 601 decisions: target 0.35 of log 5 (0.563 nats), measured 0.491 of log 5 (0.791 nats), P(play) forced 18.1% vs carried 45.1%. Search optimises the same reward and gains +0.319 with 87% of overrides being "wait" (1,847 vs 137), so waiting is already better under the objective: a credit-assignment failure, hence PBRS. SOLVENCY_RESERVE = 4.0 was set from the Giant deck (every defensive answer cost 4).
- **bankruptcy_rate's 3.0** was justified as "the cheapest card, so below it P(play) is 0 by arithmetic" for the Giant deck; false for the 2.6 deck (cheapest is 1). Kept because the 65.3% / 60.8% baselines were measured against it.

#### Package root, shipping and constants

- **One sys.path append (2026-08-20).** Before python_ai became a package, every module carried its own `sys.path.insert(0, dirname(__file__))`, ~30 copies.
- **OpenBLAS thread cap.** `import numpy` (scipy-openblas) commits 32.1 MB of private memory per BLAS thread and defaults to one thread per core: on 12 logical processors 397.4 MB unset against 44.1 MB at 1 thread (fits 44.1 + 32.1*(threads-1)). The working set stays ~16 MB, so it shows only in private commit. Setting the variable after numpy loads is a no-op (403.3 MB). OMP_NUM_THREADS is not capped: torch costs ~166 MB at import regardless and ~4.4 MB per OMP thread, and the main process spends ~87% of its time in the conv-bound update.
- **Shipping weights.** Until 2026-08-19 shipping.py named `model_weights_cured.pth` (ep 78,214), trained on the Giant deck and retired with it; then `model_weights_selfplay.pth` (ep 31,312), which the cleanup deleted, leaving every deployment path raising FileNotFoundError. It now names `model_weights_live.pth`, a frozen copy of phase 9 at ep 115,173.
- **Search horizon sweep** (retired cured net vs heuristic@1.5x, n=80 each, paired): horizon 4: 0.483 -> 0.667 (1.6x cost); horizon 4, K<=13: 0.450 -> 0.788 (1.7x); horizon 8: 0.525 -> 0.925 (1.5x); horizon 12: 0.563 -> 0.963 (1.5x, chosen, then confirmed on a fresh run); horizon 20: 0.488 -> 0.875 (1.8x). One engine step costs 0.015 ms and one scored candidate a 0.13 ms network row, so depth costs ~1/9 of width. Giving both sides search erased the advantage: cured scored 0.4425 [0.3825, 0.5025] against v1.2.0 (n=100), so the +0.72 headline was search, not the weights.
- **Search turned off (2026-09-06).** Paired, seeded, UtilityTeacher rung 3 on the 16-deck pool, ep-111k policy (`eval/search_vs_greedy_pool_ab.py`), greedy 0.844 in every arm: horizon 4 0.531 (-0.313 [-0.531, -0.125]), horizon 8 0.312 (-0.531), horizon 12 0.375 (-0.469); the shipping configuration measured -0.433 [-0.633, -0.233], p = 0.00098. Every positive search result in the repo (the +0.319, the sweep) was measured against the C++ heuristic that also steps the rollouts.
- **Not shipped.** The tactical override measured +0.003 / -0.028 / -0.028 on the cured net (null) against +11.8 points at v1.2.0. SolvencyGate cut bankruptcy 72.7% -> 39.2% without moving win rate.
- **Arena copies.** Before ArenaLayout.h was bound, tactics.py had OWN_KING (9.0, 2.5), Princess x = 4.0 and bridges (4, 14); HeuristicOpponent had bridges at 3.5/13.5, already off Board's own 4.0/14.0.
- **Forward offsets (2026-08-27).** Appending the card-cycle blocks behind the extra scalars broke five call sites that used `observation_size() - NUM_EXTRA_SCALARS`, two of them reading `enemy_tower_hp` for the tower potential.
- **deck.py.** The deck used to be a literal in gym_wrapper.py; an engine-refused deck (a Champion outside slots 1-2) surfaced as a ValueError deep inside module import.

#### The PPO machinery (rl/)

- **gamma 0.99 -> 0.999 (2026-08-27).** At 0.99 the horizon was 100 decisions against a 360-decision match: a win decayed to 0.0268 while W_TOWER_DESTROYED (0.6, undiscounted, mid-episode) was worth 22.4x it. Over 10 greedy episodes of `model_weights_phase1_v5.pth` the terminal reward was 0.0195 of the discounted return. A sacrifice play (concede HP now, win later) cost 0.5 at once and repaid 0.027. The potential-based terms were never at fault: a PBRS term telescopes to gamma^T*Phi(s_T) - Phi(s_0) and keeps its ~2:1 proportion to a win at any gamma. The non-invariant terms (W_TOWER_DESTROYED, the one-sided elixir trade, the overflow penalty) dwarfed it. The variance objection was measured away: GAE's lookahead 1/(1-gamma*lambda) is 9.17 at 0.99 and 9.91 at 0.999; over 6 episodes (mean 322 decisions) mean |return| 0.385 -> 0.416, std 0.362 -> 0.384, max unchanged at 1.719. No annealing (OpenAI Five ramped gamma to control early variance; there is none to control here). Changes the objective, so win rates are not comparable across it; `CLASH_GAMMA` restores the old value.
- **gae_lambda 0.95 -> 0.9.** num_envs, value clipping, entropy decay and reward shaping all left Loss/Critic on the same noisy plateau across ~2,650 episodes.
- **Truncated BPTT.** The update used to replay each env's whole 500-step rollout as one sequence with 1 minibatch and 2 epochs: 2 optimizer steps per 4,000 transitions. Loss/Clip_Fraction read 0.0000 across all 228 updates of an 18,740-episode run (epoch 0's ratio is 1 by construction, leaving one Adam step), and ~79% of the epoch was the backward pass through one 500-long chain. With chunks of 25: 160 segments per rollout, 32 optimizer steps (16x), and sequential LSTM calls per epoch fell 500 -> 200.
- **bptt_chunk 25 -> 50 (2026-08-27).** A card rotation is 4 x 2.625 elixir x 28.571 ticks = 300 ticks = 30 decisions, so 25 truncated credit before one rotation, right after the opponent's seen[]/recency[] were added to make card counting possible. Measured 1.090x per update (only the LSTM loop, 16% of the update, changes shape). The real cost is segments per minibatch 20 -> 10. L=100 (1.227x, 5 segments) is refused on batch width.
- **vf_clip_std_frac.** Measured on a live rollout at ep 14,666 (stage 3): GAE return std 0.530, |return - V| median 0.181, p90 0.539; 46.1% of samples needed the critic to move more than 0.2.
- **Aux next-card head replaced the opponent-elixir head (2026-08-28).** OLS on extra-scalars 0 and 2 predicts opponent elixir at MAE 0.0000 over 2,606 samples, while the trained head sat at 0.77. eval/probe_card_counting.py found the policy discarding the cycle: trained hx decoded the next card +0.013 over a random projection at ep 1,522 and -0.025 at ep 2,054. 0.5 was sized for CE ~ln(8) = 2.08 against an actor loss ~0.1.
- **aux_warmup_episodes = 2000 (2026-09-15).** Fresh init, from-scratch phase-1 config, 12 updates: the aux gradient on the LSTM grew from 0.62x to 1.67x all other terms combined (CNN trunk 0.63x -> 2.01x), cosine to them -0.37 -> -0.84 (trunk -0.90). A mechanism measurement, not a win-rate gain. 2,000 episodes is ~160 updates (~2 h). Refuted if a paired from-scratch A/B at 0 vs 2000 shows the warm-up arm no better on reward at episode 2,000 and worse on Aux/NextCard_CE afterwards.
- **cycle_id_coef (2026-08-28).** On the ep-7,200 checkpoint the aux term supplied 0.27-0.40% of the gradient on the cycle-branch parameters (18x outgunned by the actor term, ~250x by actor+critic+entropy); PPO turned the branch into a 1-D opponent-tempo readout that decoded the next card worse than at init (+0.169 vs +0.195). Matching needed a coefficient near 2.5, which would dominate the loss; hence the detach.
- **save_every_episodes.** A resume sets last_save_ep to the resumed episode, so at the 500 default a shorter experiment wrote nothing, and `timeout` kills the process before the end-of-loop save.
- **Entropy controller.** Fixed coefficients: card scale 2.0 collapsed the card head to 10% of max entropy; 4.0 recovered it to ~35% but placement fell from H=4.38 to 2.63 at the same stage (40 -> 20 cells, top-5 share 36% -> 70%, left lane 33% -> 13%). A fixed 0.65 placement target made the coefficient rise monotonically (0.1286 -> 0.1476 over 50,000 self-play episodes), hence the anneal.
- **adapt_rate_card 0.15 -> 0.5.** The lower gain smoothed the controller, but the policy measured worse at stage 3: 81.7% [74-88] vs 96.7% [92-99] (one run each). Suspected mechanism: large swings act as periodic exploration boosts.
- **Placement gain windup (2026-08-11).** The 0.5 gain had been tuned against a placement-entropy signal that was ~85% no-op steps (near-constant 0.85-0.97). After the signal was corrected the coefficient ran 0.0100 -> 0.433 over ~50 updates, six of eight cards reached 0.93-1.00 of maximum placement entropy, ROI 0.96 -> 0.87, win rate 0.67 -> 0.51. Lesson: fixing a sensor invalidates any gain tuned against the broken one. Phase 2 then got gain 0.10, a 0.10 per-update step cap and a 0.20 ceiling.
- **coef_floor 0.002 -> 0.01.** At 0.002 the entropy term stopped opposing the policy gradient at all. The card head's controller was left uncapped because it was fixing a real collapse (0.084 of max against a 0.35 target, ~5 of 8 cards).
- **Elixir-multiplier curriculum replaced by teacher competence (2026-08-19).** SMART-forced Hog A/B (advisor timing gate + bridge cell), n=120 paired: 1.00x baseline 1.000 vs forced 1.000 (ceiling); 1.25x 0.950 vs 0.825 (-0.125, p=0.0059); 1.50x 0.617 vs 0.317 (-0.300, p=3.2e-06). A punish window lasts about answer_cost / (m * r), so at m=1.5 it is two-thirds its natural length. Four interventions to rehabilitate the Hog had returned null because each moved the policy, not the payoff. Deleting the multiplier alone would have made phase 1 zero-gradient (the C++ heuristic is beaten ~100% at 1.0x). CURRICULUM_STAGES used to be a local of train_ppo(); the tests recovered it by parsing train.py with `ast`.
- **Gate 0.80 -> 0.65 (2026-09-03), with the 11-rung table.** Simulated 2026-08-26: an agent of true skill 0.70 clears one 100-episode window 1.6% of the time but at least one within 3,000 episodes 91.6% of the time, so 0.80 was an optional-stopping test with an effective threshold near 0.70.
- **Plateau valve and backstop (2026-09-03).** The 2026-08-28 run spent episodes 9,640 -> 32,680 (23,040 episodes, ~24 h) at stage 3, win rate 0.15-0.60, zero advances; the stall valve caught only the two deepest troughs and the run climbed back both times. Replayed through the new manager the plateau fires within ~2,000 episodes (test_curriculum_plateau.py bounds it under 4,000). On the 2026-09-03 sweep (8 decks, 0.07-1.00) the agent beat three decks at 0.70-1.00 while the episode-weighted pool win rate read 0.581. The first backstop measured time since entering the rung and would have cut off a still-climbing agent.
- **Progress signal (2026-09-06).** Live phase-9 run, ep 83,128 -> 87,540: the unweighted per-deck mean rose 0.301 -> 0.534 with all sixteen decks improving while the readable rate sat at ~0.50; the plateau valve fired twice, and the promoted rung then made the agent worse (0.535 -> 0.522, readable rate 0.21, within 600 episodes). With the backstop's floor on the readable rate, an unweighted per-deck mean of 0.62 (beating fourteen of sixteen decks) read as long_mean 0.29 and demoted rung 3 -> 2 -> 1.
- **Regression trigger.** Measured twice at rung 3 -> 4 (the first horizon increase, 20 -> 30 ticks): 0.535 -> 0.522 over 600 episodes, then 0.637 -> 0.583 over 1,800 with the worst deck 0.12 -> 0.07; both caught by hand. `best_rung_mean` read 0.608 for a rung entered at 0.637, hiding a fifth of the fall. A backstop demotion once logged "sustained win rate at or below 10%" at an actual 0.39, hence `last_demotion_reason`.
- **Floor alarm (2026-09-15, audit 04 C1).** Driven through the manager, 4,000 episodes at rung 0 with a 0.00 win rate fired no event, and monitor_run.py read no win rate at all.
- **Checkpoint remap (audit 04 C5).** The remap used to re-run on any table-size mismatch: one added rung sent a rung-3 checkpoint to rung 7 while printing "(same lookahead horizon)". The plateau tracker was not persisted until 2026-09-15 (audit 08, gap 5), costing up to ~2,000 episodes of plateau delay per crash.
- **The `placed` mask (2026-08-11).** Averaging placement entropy over no-op steps let the head earn the bonus for free: 0.462 reported = 0.850 on no-op steps vs 0.090 on real placements, against a 0.25 target.
- **Batched forward_sequence.** One pass over L*B instead of L calls at B: 1.82x on the forward, max abs logit delta 1.1e-08 vs float32 eps 1.19e-07.
- **Deck-coverage threat gate.** The ungated floor cost 0.42 win-rate points; a Cannon is worth +841 tower HP under attack and ~nothing on a quiet board.
- **Aux-loss cap (2026-09-03).** aux_card_scale = 0.02 was calibrated at CE ~1.5 (weighted term ~0.015, comparable to the actor loss). The deck pool took the opponent from 8 mirror cards to ~60 and the head's CE jumped to 13.76 (uniform is ln(185) = 5.22): the weighted term reached 0.138 against an actor loss of ~0.020, and win rate fell 0.58 -> 0.00 in 126 episodes. (CLAUDE.md later records that this was not the collapse cause: arms with and without the fix matched.)
- **Cycle-branch identity loss.** PPO was measured destroying card identity in the cycle branch at 248:1; the branch has 3,752 parameters.
- **Non-finite guard (2026-08-26).** One bad step turned 32 of 32 parameter tensors NaN, permanently, and the run kept training and overwrote the last good checkpoint.
- **Undefined decision stats.** P(nothing affordable) is 73.9% per step; a recorded 0.0 entropy drove the card coefficient 0.05 -> 0.0596 (+19%) in one update on a batch with no information.
- **One loop (2026-08-20).** `train.train_ppo()` and `train_selfplay.train_selfplay_ppo()` were 1,620 and 1,565 lines, overlapping in the whole algorithm (hyperparameters, the 17-key stats extraction, the rollout and its masks, GAE, the 250-line update, the entropy controller, replays, checkpoint cadence).
- **Replays anchored.** Left cwd-relative, the launch directory decided where replays landed (a stale replays/replay_ep1000.json sat at the repo root while python_ai/replays/ never existed).
- **Seeded reset.** A bare `reset()` left the C++ opening-hand shuffle on OS entropy, so a seeded run reproduced its weights and minibatch order while dealing different hands, and still announced itself deterministic.
- **Ctrl-C save (audit 08, gap 6).** An interrupt used to lose up to CLASH_SAVE_EVERY episodes.
- **Rollout hi-res map.** The rollout recomputed `cnn_trunk[:2]` inside placement_given_card, running Conv2d(21->16) at full 34x18 twice per step. Threading the map: 500 steps x 8 envs, network portion, 6.505 s -> 2.103 s (67.7% saved).
- **First-step shaping (audit 03, R1).** Phi_solvency(elixir=0) is -0.1, so every episode's first real step was paid +0.084 against the phantom step's defaults.
- **Truncation read from `truncateds`.** It used to be inferred from `dones & (abs(raw_rewards) > 0.5)`. An exact-tie timeout pays ~0, so it was treated as a truncation: charged DRAW_PENALTY and also credited gamma*V(final_obs), partly refunding the penalty.
- **Advantage normalization over `decision`.** Normalizing over `valid` (a superset including forced steps) left the actor's rows off-centre: over three rollouts at a 308-episode checkpoint (decision 73.7-75.1% of rows) centring error -0.0312 / -0.0820 / -0.0073 of a unit std and scale error 0.9757x / 0.9492x / 1.0041x.
- **Modal share** was not logged anywhere until 2026-09-15 (audit 08). Deck/MinCardProb sat near 0.001 for 30,000 episodes in the 2026-08-28 run while Policy/Entropy_Card_Frac held its 0.35 target exactly.
- **Anchored checkpoint paths (2026-08-25).** Every destination used to be a bare relative string, so the launch directory decided whether a run resumed or started fresh, and whether the PFSP pool was found. atomic_save exists because `save_checkpoint` used to overwrite in place, and an interrupted write (Ctrl-C, OOM, full disk) left the only copy truncated; the payload is ~3x the parameter count (model plus Adam's two moments). A reader holding the checkpoint made `os.replace` raise and killed the trainer (audit 08). The `.prev` backup came from audit 08, gap 6. CLAUDE.md's CLASH_* stamp is TODO 00.9: a resume under a different CLASH_GAMMA continued silently under a different objective.
- **Snapshot interval 5000 -> 2000 (2026-08-09).** A re-denomination: the speed fix left gradient steps per hour unchanged but cut episodes per hour 2,873 -> 1,301, so an episode carried ~2.2x more transitions (5000 / 2.2 ~= 2,270). Not 1,000, since 5,000 was itself a judgement call, and every extra pool member dilutes PFSP shares (which already forced DEFENSIVE_SCRIPTED_MIN_WEIGHT to 0.8).
- **Stage snapshots** exist because the narrow stage-4 policy (8 of 288 cells, 5 of 8 cards, one lane) was measured only against its own 1.4x-elixir opponent, and the per-stage weights needed to tell collapse from a rational response to being out-elixired had been deleted.
- **Deck coverage floor (2026-08-28 run).** The agent used five of eight cards for 30,000 episodes while EntropyController held target_card 0.35 almost exactly (0.339 -> 0.350): entropy over four hand slots plus no-op is blind to the deck marginal (a policy that no-ops ~81% and spreads the rest over five cards sits at target). Placement_ByCard only logs played cards; the advisor KL still sat at 0.73 after 24,000 episodes. At ep 32,484, P(play | in hand): Skeletons 0.3429, Ice Spirit 0.3247, Ice Golem 0.1868, Hog Rider 0.0704, Musketeer 0.0662, The Log 0.0091, Cannon 0.0053, Fireball 0.0011. The floor 0.02 is 3.3x below the weakest live card and 2.2x above the strongest dead one.
- **Linear hinge refuted.** d(linear)/d(logit) at p = 0.0011 is 0.000275 vs 0.2497 for the log hinge (17x weaker where help was needed). Three paired 18-minute arms from ep 32,484: MinCardProb change -0.0007 at coef 0, +0.0018 at 2, +0.0004 at 8, a null.
- **Log hinge measured harmful.** Paired arms from ep 32,484 at stage 3, scenario injection on in all: coef 0.00: win 0.600 -> 0.520, MinCardProb 0.00126 -> 0.00058, H_card 0.34 -> 0.32; coef 0.05: 0.600 -> 0.340, 0.00093 -> 0.00189, 0.38 -> 0.43; coef 0.20: 0.620 -> 0.120, 0.00177 -> 0.00892, 0.38 -> 0.57. A full launch reproduced the 0.20 arm (0.60 -> 0.11 over 180 episodes, reward +1.96 -> -1.46). Scenario injection alone cost 0.08; the floor the other 0.42. Threat-gated: coef 0.20 0.600 -> 0.140 (no better), 0.60 0.610 -> 0.350 (non-monotone); the gate was open on 29.4% of 977 decision rows (94.4% inside scenarios, 26.9% outside, median threat 0 HP). The "+841 HP" that motivated the gate used `inject`, which skips the elixir cost; through `env.step` with the Cannon in hand and affordable in threatened states: policy's action 5241 HP conceded vs forced Cannon 5056, +185 HP net, better in 6 of 14 states. Cannon placement captured 27.5% of achievable value; the card head was correctly pricing a weak placement head. The measured-positive levers for placement are decision-time search (+0.319) and expert iteration (+0.045).
- **Phantom-step reseat (2026-09-15).** +0.084 on every episode's first real step, twice the solvency term's legitimate per-episode magnitude.
- **Entropy normalization defects.** 2026-07-31: the exploiter used raw nats where the main loop used fractions, making its placement coefficient effectively 73x too large; the policy dissolved to 86.7% of maximum placement entropy and went 166-839-2 against the agent it was a bit-exact copy of. 2026-08-11: placement entropy was averaged over steps where a card was affordable, not played (0.462 reported = 0.850 no-op vs 0.090 real); the controller pinned the coefficient to its floor for 57% of updates while real placement collapsed. 2026-08-16: both heads divided by log(total arms); ~2 card arms were legal on 54.1% of decisions, so a 0.35*log(5) target was 81.3% of the reachable log(2) and held play/wait near a coin flip (elixir spent on sight, mean 2.25/10, nothing affordable on 73.9% of steps, P(play) flat against threat).
- **Controller state checkpointed.** On 2026-07-30 a phase-2 resume reset the coefficients to 0.05/0.06, taking placement from a converged 0.0132 and ~5,600 episodes to walk back to 0.0205.
- **Re-boost was dead code until 2026-08-09.** Pipeline 2 passed the raw episode counter to the anneal, so the stall re-boost printed a message and changed nothing (it fired twice at a pool win rate of 0.49).
- **normalize's docstring** said `valid` until 2026-08-27 while base_trainer had long passed `decision`. The unbiased-std NaN on a < 2-row batch was measured 2026-08-26 to turn 32 of 32 parameter tensors NaN. The vf_clip_range line (`clamp(vf_clip_std_frac * r.std(), min=eps_clip)`) looked floored while passing NaN into the critic loss, and the guard then dropped every minibatch: the update learned nothing.
- **Explained variance** exists because the critic was called broken three times off a flat MSE near 0.11 while explained variance measured +0.64.
- **drain()**: obs 13606 x 8 envs x 4 bytes = 0.435 MB/step, x 500 = 217.7 MB, 435.4 MB with both copies, against a main process at 1194 MB private commit.
- **Placement coverage (2026-08-14, model_weights_dist_e3.pth).** 40 greedy episodes at 1.5x opponent elixir: Cannon modal cell (11,0) 91.0%, 24 plays, H 0.098; Fireball (11,0) 58.4%, 2 plays, H 0.141; Giant (11,0) 54.1%, 2 plays, H 0.147; Mini PEKKA (14,15) 19.0%, 230 plays, H 0.086. Scored by tower HP preserved over a Cannon's 300-tick life across 449 threatened states, the policy's cell saved 121 HP vs 396 for a random legal cell (paired -274 HP, 95% CI [-328, -221]). The 2026-08-09 phase-1 net placed Cannon at (3,15), modal share 9.5%, H 0.368. The collapse appeared with e16cdd7 ("Measure placement entropy on real placements, not on no-ops"), correct for its own defect, whose side effect was removing the only gradient that held unplayed cards' maps open: Cannon and Giant went from the highest per-card placement entropies to the lowest. The term costs a second placement-head pass (~41% of update time); sampling one slot reaches every card ~1,000 times per rollout. Slot weights exist because the starved cards (Cannon 3, Fireball 4, Giant 5) are the ones affordability hides: the agent sat below 3 elixir on 65.3% of decisions.
- **optim_step.** Measured 2026-08-26 on a 32-tensor net: 32 of 32 NaN after one bad step, still 32 of 32 after a clean one. Five optimizer loops (rl/ppo.py, trainers/exploiter.py, bc_pretrain.py, distill_tactics.py, expert_distill.py) plus two harnesses had each written the same unguarded pair.
- **Seeding.** Before rl/seeding.py there was no seed anywhere in the training path. BaseTrainer.setup then called `envs.reset()` unseeded, so at seed 4242 two runs had identical network init but different opening hands ([[7, 24, 6, 33], [24, 25, 6, 7]] vs [[24, 33, 25, 72], [40, 15, 25, 24]]) while printing "Deterministic run".
- **Teacher debug.** Over 1,080 decisions of contested teacher-vs-teacher play: 3.60 candidates per decision (median 3, p90 9, max 13), 22% with none, 33% with more than four. A tail-read optimisation of save_log did not help (the cost is the disk sync). Stamping records per tick made a fixture 11 MB against 2.97 MB.
- **Modal share** (placement_stats.py): measured 2026-08-14, Placement_ByCard_Min flagged the healthiest card (most played, modal share 19%) and cleared one with 91% of its mass on a single cell. Not logged anywhere until 2026-09-15 (audit 08).

#### The network (models/)

- **Evolution legality (audit 06, E-2).** Deriving the legality table and spell flags from `get_all_card_ids()` left every Evolution row all-False, so an Evolution in the deck could never be placed (Evolution Archers has 242 legal cells). **Miner (audit 06, E-1):** reading spell_flags for the row rule confined deploy-anywhere troops to their own half.
- **Placement legality mask.** On the ep~45,800 checkpoint over 1,340 decision steps, 58.7% of card choices were rejected by playCard, 94.8% of them on row y=0 (the back-row dead zone). Legality measured 208/288 legal cells on an empty board and 208/288 with six troops down. A destroyed own Princess frees its footprint: 242 -> 251 cells each; an enemy Princess 242 -> 242. The base table costs ~113k predicate calls and 2.35 s; the first per-tower mask version was 53% more expensive than the base mask, hence the pre-OR-ed state table (4 x 186 x 612 bools, ~455 KB).
- **Placement space covers the whole board.** The action space used to cap target_y at 15.5 for every card, so Fireball could not cross the river. The "18*16=288" comment in placement_given_card was corrected on 2026-08-27 (the head is 34 x 18 = 612).
- **Dilated context blocks (2026-08-27).** Isolated fwd+bwd at (500, 32, 9, 5): identity 0.97 ms, d=1 7.30 ms, d=2 8.26 ms, d=4 43.72 ms (padding 9x5 to 17x13, 221 cells vs 45). Two d=2 blocks 14.6 ms vs 50.1 ms for (2,4). Doubling trunk width cost 2.28x for zero reach. Bridge y=16.5 to enemy King y=30.5 is 14 rows; to the Princess Towers (y=27.0) 10.5.
- **Scalar encoder branches (2026-08-27).** 72,000 -> 4,850 parameters at 0.998x update time. "17.6:1 compression destroys the cycle" does not hold (a random 370 -> 64 projection preserves structure, Johnson-Lindenstrauss); the defect was optimisation interference, which the test pins with a real optimizer step.
- **Cycle branch collapse (2026-08-28, model_weights_phase4.pth, ep ~7,200, 24 episodes, 5,366 decisions, linear probes, lift over marginal for the next card).** obs cycle_raw (ceiling) +0.412; branch PPO-trained +0.169 (41%); branch at init +0.195; hx trained +0.046; hx at init +0.135; everything except the cycle +0.031. The shared projection over a sum-pooled bag lets the objective extract only aggregate activity: mean pairwise |cos| of the eight deck columns 0.191 +- 0.017 -> 0.461, effective rank 7.48 -> 6.01, PC1 32% -> 54% of variance, column norms grew 2.1x. Linear(370, 8)+ReLU trained for the task keeps 92% of the ceiling, 24 dims 96%. Given an identity gradient the collapsed weights recover +0.390 and |cos| returns to 0.202. The skip: through the LSTM the branch's card signal is retained 27% trained (+0.169 -> +0.046) vs 69% untrained.
- **Placement head history.** The Gaussian head's log_std climbed from -2.0 to -1.86 over 68,515 episodes (sigma 0.156 normalised = 2.66 tiles in x, 2.46 in y, with bridges 10 tiles apart). The ConvTranspose2d head (fixed 2026-08-09): on the ep~129k checkpoint 75.0% of logit-map variance was explained by (x mod 4, y mod 4); 73.0% of placements landed on x = 3 (mod 4) vs a 22.2% null (chi^2 = 94.8, 3 df); 28.9% on (11,2)/(11,3); 91 of 288 legal cells used; card-independent (Giant 44.4%, Valkyrie 38.9%, Cannon 27.1%), and wrongly blamed at first on the building reward.
- **Full-resolution branch (2026-08-14).** distill_tactics.py: cross-entropy to the advisor's exact cell fell 180.9 -> 21.4 while exact argmax match stayed 0.0%, the signature of a target the coarse head cannot express. A concatenation into place_up would have discarded every checkpoint's placement head; the zero-init residual did not.
- **Row compaction (2026-08-24).** Decision fired on 0.368 of rows (model_weights_selfplay.pth, 1,500 steps), so 63.2% of placement convolutions were multiplied by an exact zero. Linear(280->32) differs by 4.768e-07 forward and 2.289e-05 in grad_W between batch 500 and 167. Conv2d grad_W, 500 -> 184 rows: place_up.1 (32->16, 18x10) bit-identical; place_up.4 (16->8, 36x20) 5.814e-03; place_up.6 (8->1, 36x20) 2.808e-03; place_hires.0 (24->8, 34x18) 4.883e-03. A first check at 18x10 alone read "independent" and was wrong.
- **Affordability mask.** Measured on the trained policy of the time (243 decision steps): 74.9% of steps tried to play an unaffordable card and only 11.5% placed one; at 0.35 elixir per decision against 3-5 cost cards, 0.55 of 4 slots were playable on average and 27.2% of steps had any legal option.
- **forward_sequence.** 25 calls of the card/value/aux heads took 35.3 ms vs 0.7 ms for one call on 200 rows (48x); the placement head gained 1.3x; ~5% of update time overall.
- **hand_costs_from_obs** had three literal copies (hybrid_policy.py, eval/gate_ab.py, perception/live/mvp_loop.py).
- **policy_io.py** exists because eighteen scripts imported `load_net` from the 1,143-line `expert_iteration.py`, pulling in both PPO trainers, behaviour cloning and the search harness to read one `.pth`. `load_state_dict_flexible` lived in `train.py` (moved to avoid a circular import) and made four non-training scripts import a 2,400-line trainer. `LSTM_HIDDEN = 256` was typed in five files and imported from `search_ab_test` by four more. The mismatch-warning dedupe was added after ~4,000 repeats of one line from a single mismatched PFSP checkpoint. Widened heads used to be counted as discards, which made losing the card head, both placement contexts and the critic at once read as an ordinary warm start (tests/test_flexible_load_grows_widened_heads.py).
- **perception_encoder.py.** The observation grew 6253 -> 13606 on 2026-07-29 (NUM_CHANNELS 9 -> 21). Writing 0.0 for the opponent's spend, measured on a frozen checkpoint over 900 episodes: delta +0.031, 95% CI [-0.018, +0.080] (inferring spend from units over-counts 2.1x, perception/BOT_REQUESTS.md item 8). The time scalar was normalised by 1800 until it was found running 2x fast (0.50 handed to the net at 90 s where training showed 0.25). Towers occupy 34 floats a units-only encoder left at zero: the entire round-trip difference. The Princess tower-scalar scaling was hidden because Kings are exactly 4008. The elixir-phase scalar (2026-09-02) must not be rescaled by max ticks, or a live match enters triple elixir at 1:30.

#### Environments (envs/)

- **Deck history.** Before the 2.6 Hog Cycle (2026-08-16) the default deck went: Balloon Freeze (Baby Dragon, Knight, Inferno Dragon, Balloon, Bowler, Tornado, Freeze, Barbarian Barrel), replacing an earlier Giant Beatdown; then Giant beatdown (Valkyrie, Archers, Minions, Cannon, Fireball, Giant, Musketeer, Mini PEKKA), which dropped the two ramping building-shredders; then a Hog cycle (Hog Rider, Cannon, Musketeer, Archers, Knight, Minions, Fireball, Valkyrie) because Giant (5 elixir) was never played in any probe across four runs: at 0.35 elixir per decision a 5-cost card is legal only after ~14 non-spending steps, so its slot was nearly always masked and never got gradient (costs 3-5, avg 3.75 vs 3-4, avg 3.50); then on 2026-07-30 back to the Giant deck because it is the deck in the 8 recorded real matches (behaviour cloning); then the 2.6 Hog Cycle, giving up the recordings tie. Known simplifications noted for the Balloon deck: Bowler has no knockback, and Balloon's death bomb always fires instantly.
- **_find_win_condition (audit 07 F2, audit 05, 2026-09-15).** Its own "most expensive building-targeter" copy returned None for siege/spell/deploy-anywhere decks (Mortar, X-Bow, Graveyard, Goblin Barrel, Miner), silently switching off W_WIN_CONDITION_DAMAGE, and on a deck without a real win condition promoted a 2-elixir Ice Golem.
- **deck_spell_info (TODO 00.3).** It replaced `train_FIREBALL_ID = 7`; with a Rocket in Fireball's place both spell terms read exactly zero on all 2,372 measured steps.
- **Phase-1 defensive scenarios.** In the 2026-08-28 run's 32,680 episodes the agent never met a manufactured defence moment, and Cannon / Log / Fireball play probabilities settled at 0.0053 / 0.0091 / 0.0011.
- **Deck-pool stats on resume.** Measured on the 2026-09-04 run after 12,589 episodes: xbow_30_cycle 0.181 and mortar_cycle 0.176 learned vs priors 0.867 and 0.800, so their PFSP weights were off ~37x after every restart; the 100-episode win rate fell 0.42 -> 0.143 and took ~500 episodes to recover. Count-weighted alpha: over 8 simulated seeds, the effect on episode share is inside the noise (6.8-17.8% of episodes on unwinnable decks either way).
- **PFSP over a ladder.** The earlier linear ladder (training_selfplay_run1.log) stalled once the weak snapshots were exhausted: the next opponent was a near-mirror of the trainee and the gate became unreachable at ~50%; and it never revisited, so nothing prevented forgetting.
- **Builtin heuristic as a phase-2 training opponent (2026-07-31).** 13 evals over 65,000 phase-2 episodes showed no movement against it: 0.84 against the pool, 0.92-1.00 against the neural anchors, flat against the heuristic (chi2 13.1 on 12 df at @1.50, 17.4 at @1.35, critical 21.03). The measured gap was 0.83 at @1.35 and 0.60 at @1.50 (1.00 at @1.00). BUILTIN_MIN_WEIGHT 0.5 gives ~29% of episodes at a pool of ~20 (2*0.5 / (16*0.05 + 2*0.8 + 2*0.5)), ~23% at 40; 0.8 would have made it ~64%.
- **PFSP stats on resume (TODO 00.8).** pfsp_stats lived only in worker memory, so every resume reset ~100 opponents to 0.5 until each worker re-met each ~12 times.
- **Unloadable pool entries.** The pool was written non-atomically until atomic_save landed on 2026-08-26, so a truncated file could already be sitting there.
- **Scenario card requirement.** giant_commit asked for a card DEFAULT_DECK cannot contain, hit the cap on every draw, and was inert for three days before its removal on 2026-08-19. reset() costs 0.135 ms and its shuffle is uniform over all 70 hand-sets.
- **Opponent hi-res threading.** The trainee's equivalent change measured 67.7% of the rollout's network time.
- **Deck contract (2026-09-15 audit).** Mechanisms keyed to the 2.6 deck that went silent under another: the advisor-target term (10% of log 612 on the placement head) trained on zero cards for 5 of 8 plausible replacement decks; the win-condition reward went dead for siege/spell/Miner decks; the spell terms were keyed to Fireball (id 7); 60% of scenario injection built boards answered by a Fireball.
- **Scenario arena copies.** `_BRIDGE_LANES` read [3.5, 13.5] from the pre-2026-08-21 arena; x = 13.5 is the edge of cell 13, water, so every right-lane bridge push spawned off the bridge and nothing noticed. `_scenario_fireball_tower_value` had `4.0 if ... else 14.0` (mirrored about 9.0 instead of 8.5): the left cluster sat one tile off its tower, a lane asymmetry in 30% of injected scenarios. scenario_offense had `BRIDGE_XS = (4.0, 14.0)` (x = 4 is water), unnoticed because the module is default-off.
- **Fireball scenarios (2026-08-09).** Over 120 trials per constructed situation with Fireball affordable, the policy put 0.03-0.11 on it vs a 0.20 uniform baseline; cast mean distance to the target centroid was 2.8 tiles on its own half and 7.7-11.5 at an enemy tower, against a 2.5 radius. Forcing Fireball use had collapsed win rate 97% -> 23%. The two Fireball scenarios were a reallocation from the bridge pushes, whose ScenDef had reached 0.99.
- **The Giant scenario.** On the ep-17k checkpoint over 1,839 decision steps the Giant was in hand 80.6% of the time but legal on 4.7% (5.9% of in-hand), picked at P = 0.157 when legal vs a uniform 0.200, sampled on 0.54% of steps, 1 of 900 logged placements. giant_commit (an unmasked-Giant start) was removed 2026-08-19: after the deck change it spent 24 futile resets hunting for the card and ran a quiet board, diluting 17% of the injection budget (weight 1.0 of 6.0). Gameplay-affecting; ScenDef/ScenOff are not comparable across it.
- **Offensive scenarios ("Proposal A", 2026-08-19).** Four interventions to make the agent play its win condition (a reward multiplier, an advisor target, random forcing at eps=0.15, gate-timed SMART forcing) all returned null or negative because each moved the policy while the environment priced the card negatively. The punish window is the one situation the agent had essentially never been in, sitting below 3 elixir on 65.3% of decisions. `inject` then one tick: enemy mass read 0.0 immediately and 0.399 after a tick. The setter bindings came in commit 26de409 (2026-08-17), whose post-build copy failed with MSB3073.
- **Defensive scripted floor.** At 0.20 each, Defender+Counter got ~7.7% of samples in a ~98-member pool (1-4 games per 50-episode window), invisible in Episode_Length_Ticks_50 after 3,000+ episodes. An isolated eval against only them gave AvgTicks 240 and 339 vs the live ~250-290 (Counter games up to 1,000 ticks). 0.8 targets ~25% (2*0.8 / (96*0.05 + 2*0.8)). Scanning only our half for threats left 55-70% of head-to-head games ending in early blowouts (110-230 ticks): reaction latency, not sampling weight, was the bottleneck.

#### Opponents and advisors

- **Why the teacher exists (2026-08-19).** See the curriculum note: forcing the win condition cost -0.125 at 1.25x and -0.300 at 1.50x. At 1.0x the C++ heuristic is beaten ~100% (ep-64k Giant net; ep-25,202 2.6 net), so removing the multiplier without a better opponent would be zero-gradient. Neural search as teacher was rejected: at random init, paired n=60 @1.0x, policy 0.450 / search 0.417 while overriding 21.5% of decisions, vs the trained net's 0.483 / 0.667 at 10.2%.
- **Rules as candidate generator.** tactics.py's rules vs trained net vs random legal cell: Cannon (tower HP preserved) 564.1 / 12.1 / 353.5; Fireball (elixir killed) 2.405 / 0.000 / 0.276; Giant (enemy tower damage) 535.6 / 3.3 / 86.7. Cost on the i5-13420H (2026-08-19): snapshot 0.0352 ms, 10-tick step 0.0598 ms, candidate @3 s 0.2500 ms (K=16: 4.00 ms), @6 s 0.4403 ms (K=16: 7.04 ms).
- **Frame trap.** Verified 2026-08-19: own-frame y=15.0 maps to absolute 18.0 and is legal for team 1, while the unmirrored 15.0 is not.
- **MARGIN_TAPER_START = 6.0 (2026-08-28).** replay_ep2018.json (stage 1): the teacher lost a Princess at tick 159 having spent 2 elixir (one Ice Golem, tick 111) while its bar ran 5.0 -> 8.6; its next play landed at tick 561, when elixir first reached 9.60. Gameplay-affecting.
- **play_margin 0.05 -> 3.0 (2026-08-21).** 0.05 against a measured median of 1.37 for the marginal cheap plays it exists to stop, so it never bound; the bot spent to ~1.8 elixir continuously and an escorted push was affordable on 2 of ~2,400 decisions. Sweep vs the old profile, paired, sides swapped: margin 0.05: 0.500, elixir p90 3.55, overflow 0.1%, combos 2.2% of plays; 2.0: 0.812, 5.60, 1.3%, 6.9%; 3.0: 0.969, 7.95, 4.3%, 14.4%; 4.0: 0.969, 9.46, 12.9%, 18.2%. Lowering w_pos 20 -> 8 raised elixir 1.83 -> 2.67 but combos went 1.1% -> ~0%: w_pos prunes by HP per elixir, and an escorted push (388 HP/elixir) sits below a naked Hog (424). A fixed 3.0 froze the bot against a passive opponent (14 plays across 6 matches, elixir pinned at 9.56, tower damage 9143 -> 4063), hence the taper.
- **siege_reach / win-condition probes (2026-09-06).** Mortar: 34 of 170 legal cells damage the tower (rows 13-15), 136 do zero; Cannon, Tesla, Inferno Tower, Bomb Tower, Tombstone score 0 everywhere; Mortar 1596, X-Bow 3824 (centre columns; 2534 at the edge). Spell bodies on an empty board: Goblin Barrel 3, Graveyard 1, Fireball / The Log / Rocket / Zap / Arrows / Poison / Tornado 0. Goblin Barrel: 1320 on the tower vs 600 in our half; the old path aimed it at enemy clusters and on a quiet board proposed cell (0, 0).
- **Win-condition resolver (2026-09-15, audit 05 / audit 07 F2).** Ranking by cost promoted a 2-elixir Ice Golem on a Miner deck, returned None for Miner control, and inverted LavaLoon (Lava Hound over Balloon). Per-elixir-first ranking named the Miner (1746 absolute) over the Balloon (2534). A rolling spell injected on the tower scored 617 and was promoted over the Graveyard in graveyard_control. `wincon_damage_per_elixir` at 300 ticks: Mighty Miner 1636, Royal Hogs 773, Goblin Drill 664, Ram Rider 652, X-Bow 637, Hog Rider 634, Battle Ram 634, Miner 582, Royal Giant 525, Giant 507, Balloon 507, Goblin Giant 422, Mortar 399, Wall Breakers 350, Golem 312, Electro Giant 256, Goblin Barrel 240, Graveyard 146, Lava Hound 129, Ice Golem 84, Skeleton Barrel 81, Fireball 0, Cannon 0. A floor at 150 would have returned None for graveyard_control.
- **siege_building (2026-09-23, TODO 00.6).** `siege_reach > 0` admitted every spawner: Tombstone (1134 in a 1200-tick window) was named over a Graveyard (730 in 300 ticks) in a Splashyard deck, and a Barbarian Hut outranked the Giant beside it at 6182. Bodies at 400 ticks: X-Bow 0, Mortar 0; Barbarian Hut 5, Tombstone 6, Goblin Cage 1, Goblin Hut 1, Goblin Drill 4. Goblin Drill from the siege row: 0 tower damage in 300 ticks vs 2654 beside the tower. Over 28 decks only the two named changed; all 16 pool decks resolved as before.
- **spell_geometry (2026-09-16).** Every spell had been aimed with Fireball's disc (radius 2.5, 689 cap). On 320 mid-match boards against eight pool decks, engine-scored elixir value killed: Rocket 1.85 -> 2.54 (+38%; better on 75 boards, worse on 4), Zap 0.62 -> 0.69 (5/0), Poison 1.42 -> 1.52 (18/8), Arrows 1.50 -> 1.56 (7/2), Lightning 3.50 -> 3.50 (8/8), Fireball the same cell on 320/320. Capped at 689, a Rocket on a tank scored no better than on a Skeleton pile. In _rules_only, Fireball's 689 had held a Zap back from a 243-HP Skeleton clump and thrown a Rocket at a lone Musketeer.
- **Eleven-rung table (2026-09-03).** The six-rung table moved three or four axes per stage (2 -> 3 was horizon 30 -> 50, epsilon 0.10 -> 0.05, max_combos 2 -> 3); the 2026-08-28 run spent 23,040 episodes at stage 3. Rung 10 is bit-identical to old stage 5. Cost: ~0.7 ms per 10 s candidate, ~8 ms per decision at K~11.
- **Combo timing (2026-08-20).** Teacher vs teacher at stage 5 over ~2,400 decisions: bar mean 1.79, p90 3.30; a pair was affordable in exactly two states (the 5.00 opening). 3 s of regeneration is 1.05 elixir, 5 s 1.75, moving Skeletons + Hog (4.65) to 3.95 and 3.25. With the bar pinned at 8.0 (n=21 states), gap 10: +5.281 mean score, 30: +2.529, 50: +1.761; real play chose the 5 s gap 16 of 19 times only because it was the one affordable. Combos were chosen on 0.28% of decisions; Ice Golem + Hog (6) was affordable on 0.3% of decisions. A flat savings charge moved the bar 1.79 -> 1.73 across reserves 0.0 / 1.5 / 3.0 / 5.0. Deploy-time measurements behind supported_push: lone commitment -556.3 tower HP, supported +448.5 [+137.3, +760.1], escorting in a punish window +650 [+429, +878].
- **Reactive rollouts (2026-08-21).** Four responders over 150 seeded openings (sides swapped, control 0.500): open-loop +0.1583 [+0.1033, +0.2117] at 0.91x the no-op's cost; every comparison against the three closed-loop variants null (p 0.0857-0.832). A responder deciding every chunk scored +0.2217 vs +0.2450 for every third, at 2.1x the price. Against a passive opponent, 20 seeded openings, share of decisions landing a card OFF/ON and openings frozen (< 5 plays in 120): horizon 30: 12.2%/9.8%, 3/20; 50: 11.6%/10.1%, 3/20; 70: 12.5%/11.3%, 1/20; 100: 11.8%/12.3%, 0/20. COUNTER_PASSIVE_DECISIONS (2026-09-15, audit 05 BUG 2): unconditional, the counter froze the top rung against a passive opponent in 30 of 116 matches.
- **step_self_play_fast (UPSTREAM item 21).** At stage 5: ~9.7 chunks per rollout x ~6.5 rollouts per decision = 1,000,152 observation vectors built per 48 episodes against 51,566 read.
- **Rung-0 air rule (TODO 00.5).** The rung-0 teacher answered a lone Balloon with Skeletons and Ice Golem more often than with Musketeer or Ice Spirit.
- **plan_reserve_penalty.** HOG_MAX_THREAT was calibrated against a median threat of 721 on a contested board.
- **Champion abilities (2026-09-16).** Until then the teacher never activated anything; readiness is true the moment a Champion lands with spare elixir (measured for Golden Knight, Archer Queen and Monk).
- **The mirror's cost.** Three of the eight 2.6 cards sat at P(play | in hand) <= 0.009 for 30,000 episodes. The 2026-08-29 autopsy found the card head right to price them there: forced through env.step, a Cannon at the policy's own cell was worth +185 tower HP net in the best state it gets. measure_deck_matchups.py then made the opponent's deck the variable (see CLAUDE.md, "THE OPPONENT PLAYS REAL META DECKS").
- **Deck-pool priors.** Uniform 0.5 priors take ~20 matches per deck to move: a few thousand episodes of a fresh net fed P.E.K.K.A. bridge spam (measured 0.067) as often as the mirror (1.000). So the 2026-09-03 sweep's numbers shipped as priors. Removed 2026-09-15 (audit 04, C3): simulated across 8 workers they gave no episode-share benefit over uniform (every estimate collapses within ~10 matches for a fresh net) and a 3.2x larger estimate error at episode 3,000; xbow_30_cycle shipped at 0.867 and read 0.36 against a true 0.02.
- **POOL_WINRATE_FLOOR off (2026-09-06).** The teacher could not pilot five of sixteen decks (with mortar, xbow, both bait decks and graveyard it never spent elixir on the named card). Episode shares under the floor vs the 2026-09-05 per-deck measurement: dart_bait_cycle 0.20 win -> 26.7% of episodes; mortar_cycle 0.20 -> 26.7%; rg_fisherman, royal_hogs, mega_knight_ram 0.00 -> 0.84% each. The unwinnable-weight sizing: with a shared 0.05 the unwinnable decks took 20% of episodes.
- **POOL_SIGNAL_FLOOR (audit 04, C2, 2026-09-15).** With the gate at 0.0 the cold-start fallback was unreachable (rates clamp to [0, 1], so `wr >= 0.0` always held) and every test passed an explicit floor=0.20. A random-init net got the four hardest decks 59.9% of the time. The earlier easiest-first fallback (bare `wr**2`) sent the hardest deck to 0.1% of episodes.
- **card_probes (audit 07 F1, 2026-09-15).** The advisor answered "what does this card do" with CANNON_ID = 25, FIREBALL_ID = 7, HOG_ID = 15; on 5 of 8 plausible replacement decks none was present, so the advisor-target term (10% of the placement head's signal) trained on zero cards. The spell probe first let the target walk during the cast delay and read Fireball's radius as 1.0. Until 2026-09-16 it idled a fixed 40 ticks: Poison (`withRepeats(8, 10)`, 736 over 80 ticks) read 368 and Goblin Curse (43 x 6 = 258) read 129, and the docstring quoted the 368 as proof it matched the registry. A harness had restated The Log's damage as 240 against the registry's 269.
- **Crown Tower damage.** Published (DeckShop, level 11): Fireball 159 of 688, Rocket 371 of 1484, The Log 41 of 268; this engine charged towers 100% for every registered spell (measured 2026-09-16).
- **damage_spell (TODO 00.3).** Over 12 seeded matches a Hog deck with Rocket instead of Fireball had both spell terms at exactly zero on all 2,372 steps.
- **building_defends (2026-09-23, TODO 00.6).** It read `siege_reach > 0` until then, excluding every spawner before the behavioural test ran.
- **damages_air (2026-09-23).** Musketeer, Ice Spirit, Archers, Minions, Tesla and Fireball hit air; Knight, Skeletons, Ice Golem, Hog Rider, Cannon, Bomb Tower and The Log do not. A whole-plane max of the anti-air channel read 1.0 for every card.
- **Why the advisor exists.** On `model_weights_dist_e3.pth` (40 greedy episodes, 1.5x opponent elixir) three of eight cards were dead with a constant placement head: Cannon modal cell (11,0) 91.0%, Fireball (11,0) 58.4%, Giant (11,0) 54.1%; (11,0) is behind our King, and 95.8% of the Cannons placed there never had an enemy in range. See the placement-coverage note.
- **Stale copies.** `RIVER_Y = 15.5` was a literal until 2026-08-20 (the river moved on 2026-07-29). OWN_KING / OWN_PRINCESS held x = 9.0 / 4.0 after the arena moved to 8.5 / 3.0. `FIREBALL_DAMAGE = 689.0` was repeated here under a comment claiming it was read from the registry, which exposed no damage. The extra-scalar offsets were negative indices until 2026-08-29 and wrong since the cycle blocks landed on 2026-08-27: obs[-9] was index 13967, inside the card-recency block, so `elapsed_ticks` read 0.0 at engine tick 300. That was a sixth call site the 2026-08-27 sweep could not find (it searched for `observation_size() - NUM_EXTRA_SCALARS`); with the clock pinned at zero, `opp_elixir_estimate` returned the starting elixir forever and the Hog gate was gated on a constant.
- **Spell catch map.** On 1,059 states paired against the engine's get_elixir_value_killed_by (share of achievable value): damage-cap 75.5% at lead 0 vs 52.4% at lead 10; kill-weighted 75.0% vs 51.2%. Leads of 0-6 ticks were indistinguishable; only the full 10 hurt. A module-level `_FIREBALL_DISC` was computed at import and never read.
- **Win-condition cell.** Injecting a Giant and running 600 ticks over 913 states, enemy tower damage: policy's own cell 3.3, back row (y=1) 3.0, lane mid (y=11) 172.8, random legal cell 94.3, bridge on the weaker lane 535.6 (+532.3 vs policy, sign p = 3.0e-87).
- **Hog gate calibration.** A 3.0 reserve (set on Clash theory) opened the gate on 0 of 542 states: the policy's elixir median was 2.25, p90 4.35. At 0 it fires whenever the Hog is affordable (~10% of states). HOG_MAX_THREAT was first 0.35 (a normalised fraction against an absolute-HP map), ~3 orders of magnitude too strict; threat_level's median on a contested board is 721.
- **Air threats (2026-09-23).** Rung-0 teacher holding the 2.6 deck, lone pushes, 40 s: a Balloon took 1322 tower HP (over 1000 in 5 of 12 seeds), Minions 71, Baby Dragon 126.
- **SolvencyGate.** The solvency PBRS term moved the bankruptcy rate by -1.0 points over ~80 PPO updates (95% CI [-2.7, +0.6], p = 0.21). 739 of the bot's plays happened at zero threat vs 126 during a big push. A flat reserve of 4 cut tower damage dealt 8,739 -> 4,094 per episode over 6 paired openings (0 of 6 better, sign p = 0.03). The opponent-elixir estimate then came from the aux head (MAE ~0.83-1.0 vs a ~1.35 mean baseline), since deleted.
- **TacticalOverride.** Ungated, forcing the dead cards played 5.5 Cannons and 2.75 Fireballs per episode (~27 extra elixir) for an agent spending ~105 of ~98 income; both forced arms lost to the control.
- **building_score_map (2026-08-14).** On 594 held-out states Fireball (smooth target) reached 54.3% exact-cell match after the hi-res branch; the Cannon stayed at 0.2% while its top-1 probability rose 10x and its modal share fell 89.8% -> 54.9%: the head became sharp and state-dependent against an argmax that is noise. Scattering coverage took a call from ~10 ms to ~0.2 ms.
- **Advisor target.** Entropy-only coverage conflicted with the distilled hi-res maps (prove_hires.py measured Fireball 5.4x better than a random legal cell). The 3-arm coverage A/B moved the frozen cell (11,0) -> (6,0) without breaking the lock. prove_placement.py: Cannon 539.1 HP preserved vs the net's 102.2; Fireball 2.647 elixir killed vs 0.062. The rule table was three literal ids (Cannon 25, Fireball 7, Hog 15; a Giant entry removed 2026-08-17) until audit 07 F1 (2026-09-15). Over 258 decision steps the sampled slot held an advisor card 32.6% of the time and the advisor spoke on 90% of those. The masked_kl NaN: with a = exp(lo), b = lo - ln, grad_a = grad_term * b = 0 * nan = NaN on masked cells; harmless only while the target never required grad and no row was all -inf.
- **Human prior.** Held-out log-likelihood per placement over 5 episode-level splits: marginal -4.6368 vs 18-way context key -4.7782; a 3-way lane key beat the marginal by only +0.0075 against a ~0.03 split spread (~74 placements per bucket against 612 cells). The marginal: +0.930 nats over uniform (2.53x likelihood), consistent on all five splits, top-1 cell 0.081 vs 0.0038 uniform-over-legal. Coverage-slot target rate over 495 threatened states: prior off 23.4% (Cannon 0.94, Fireball 0.95, Hog 0.02, other five 0.00); on 100%. Reporting prior cards while disabled made the draw [5, 5, 5, 5] on a hand where the control is [1, 5, 1, 1]. The first blur (per-row convolution) was slow enough to time out an offline sweep.
- **Hybrid policy.** Measured placement value (Cannon tower HP preserved / Fireball elixir killed): trained policy 12.1 / 0.000, random legal cell 353.5 / 0.276, advisor 564.1 / 2.405. Three in-network repairs failed (entropy coverage; advisor distillation, frozen and with an anchor). Commander take-up of the dead cards: Cannon 0.06, Fireball / Giant 0.00. 30 paired openings: neural 0.600 win, 7,627 tower HP dealt/ep; gate 0.633, 6,659; place 0.733, 7,655; initiate 0.467, 5,390, 4.03 Cannons initiated/ep; full 0.233, 4,216, 4.73. Tower damage fell 2,236/ep with initiation alone and 3,410 with the full stack (6 better / 24 worse, p = 0.0014); win rate -0.367 (p = 0.013). The Giant path was removed on 2026-08-19 (Giant left the deck 2026-08-16); it had collapsed two of hybrid_ab.py's --per-card arms into duplicates. The old surface dispatch fell through to the building surface for any unrecognised card.

#### Search

- **Search results.** 160 paired trials at 1.5x opponent elixir, ep-64k checkpoint: greedy 0.625, 1-ply search 0.944, delta +0.319 [95% CI +0.237, +0.401], exact McNemar p = 5.6e-12, 2.2x wall clock; search deviated on 13.8% of decisions. Distillation: +0.045. search.py moved out of eval/search_ab_test.py on 2026-08-20 (six modules imported its underscore-private helpers); config.py moved out of the 1,162-line expert_iteration.py the same day.
- **SearchCfg's docstring (corrected 2026-09-06)** had claimed rollouts assume both sides no-op. A 400-tick rollout with our side idle put 2 enemy bodies out and took 1,302 of our tower HP: the heuristic plays. Against the UtilityTeacher every horizon measured negative (-0.313 at 4, -0.531 at 8, -0.469 at 12) while the heuristic numbers still reproduce.
- **Widened proposals (2026-09-03).** Mean top-1 per card on the ep-32,484 policy: Hog 0.7654, Musketeer 0.3939, Skeletons 0.3527 | Ice Golem 0.1259, Fireball 0.1074, Ice Spirit 0.0987, Cannon 0.0970, The Log 0.0288; the largest gap in the diffuse region is 0.227, and 0.25 sits in it. Searching a diffuse head's top-k gained +58 tower HP over its argmax, a wide sweep +994 (81% of the engine oracle). `max_candidates` returned 7 while the search emitted up to 150 (three widened cards could emit 460 before the cap). Appending the argmax after the cap put emission at 147 against a bound of 145; the first cap test missed it because a uniform distribution's argmax lands on the stride grid.
- **No-op duplication.** With the policy no-oping ~80-90% of the time, most rows had two candidates that were the same action, so distillation targets had a median value spread of exactly 0.0.
- **terminal_score** was `reward * weight`, which scored a draw 0.0 against a loss's -10.0.
- **Realtime latency.** Measured on this box during pipeline-2 training: h=12, K=4 had a faster p99 (67.7 ms) than h=2, K=4 (393.9 ms), and greedy alone spiked to 260 ms. Component costs, 3,000 reps: snapshot 0.021 ms p50 / 0.078 p99; 10-tick step 0.020 / 0.145; extract_features batch=1 4.33 / 27.5 ms; batch=7 7.04 / 18.5 ms. A horizon-12 rollout of 7 candidates is 7 x 12 x 0.020 = 1.7 ms of engine time.

#### Trainers

- **Phase-1 opponent (2026-08-19).** The default moved from the C++ heuristic to the UtilityTeacher; gameplay-affecting, checkpoints not invalidated.
- **Phase-1 defensive scenarios (2026-08-29).** The 2026-08-28 run's 32,680 episodes never met a defend-or-lose moment; Cannon / Log / Fireball ended at P(play | in hand) 0.0053 / 0.0091 / 0.0011, while a well-placed Cannon prevents a full Princess Tower (2,536 HP) in a real threat state. Scenario episodes were then excluded from the gate window (they would cap the achievable win rate near 0.70 against a 0.80 gate); returning before the deck read-out dropped ~30% of note_progress calls (audit 04, C6). Counters logged on cadence: audit 04 C8.
- **Deck pool (2026-09-03).** See the deck-pool notes; the autopsy's +185 HP Cannon was better in 6 of 14 states, and a coverage floor at two coefficients, a threat gate and forced sampling at five doses all cost win rate.
- **Phase-2 gates.** PHASE2_WIN_RATE_GATE was lowered 0.90 -> 0.80: under the old 1.5x-elixir final stage the strongest policy topped out at ~0.79 after ~76k stage-5 episodes. The mirror-exit stage moved from the final stage to 4 (the 1.5x opponent taught "survive being out-resourced"; Run I sat at stage 4 for 13k episodes with Win_Rate_100 0.555 -> 0.624, max 0.73). PHASE2_ENTRY_WIN_RATE was split out at 0.60 (the policy measured 0.62-0.69 at stage 4). The literal 4 was renumbered on 2026-09-03 via remap_legacy_stage (it would have meant 30 ticks instead of 70).
- **random_opponent budget.** The per-deck ladder replaced a flat episode count per deck (generalization.log showed the flat count gave no real chance to adapt). MAX_EPISODES_PER_RANDOM_DECK went 5000 -> 1250 on 2026-08-09, when the phase budget became 5,000: at 5,000 one pathological draw could consume the whole phase. The phase used to be a total-episode cap (150,000 -> 40,000) and became a within-phase budget on 2026-08-09. It was kept, small, because the "95% against random decks on first contact" evidence predated the speed and placement-head fixes, and it is the only overfitting check before the 30+ hour league.
- **Resume crash (audit 08, gap 2).** A RuntimeError during restore used to move the checkpoint to .bak, then setup() deleted the TensorBoard log while the process continued with the checkpoint's weights and announced a fresh start.
- **Plateau signal.** Measured ep 83,128 -> 87,540: the unweighted deck mean went 0.301 -> 0.534 with all sixteen decks rising while the readable rate stayed flat; the plateau valve advanced the rung twice on that flatness.
- **Pipeline-2 launch (audit 08, gap 3).** The Popen handle was discarded and the parent exited, so a child dying at startup ended a multi-day run silently; its logs were cwd-relative and truncated on relaunch.
- **selfplay_paths (audit 08, gap 3).** The paths were constants, so a phase 1 run under CLASH_WEIGHTS / CLASH_LOGDIR handed off to a child that bootstrapped from python_ai/model_weights.pth: missing (crash) or another run's (wrong network).
- **Re-boost while winning** measurably hurt in the earlier ladder-based run. The anneal clock read the raw episode counter until 2026-08-09, making the re-boost dead code.
- **PFSP stats on resume** (TODO 00.8): see the selfplay_env note.
- **Exploiter motivation.** Across 50,000 self-play episodes of an earlier run every behavioural diagnostic was flat (cards per game 5.66 -> 5.59, plays 12.4 -> 10.7, game length 1085 -> 946 ticks) while the win rate against the sampled pool looked fine.
- **Exploiter cost.** First estimated at ~14% of throughput (57-minute burst over an assumed 6.5-7 h per 10,000 main episodes, from a 66-episode sample at 3.4 s/episode). Measured: a burst took 4,512 s (75 min, 4.5 s/episode) and the main loop covers 10,000 episodes in ~2.4 h (4,185 ep/h, between the ep5,049 and ep10,075 snapshots with the burst subtracted), a ~34% tax, so the cycle went 10,000 -> 20,000 on 2026-07-30 (~17%). After the speed fix bursts took 10,114 s and 9,961 s (~20% at 20,000).
- **Exploiter entropy units.** Burst #0 applied the coefficients to raw nats: 0.15 raw on placement (log 612 = 6.417) was 73x the main agent's live 0.0132; the entropy bonus reached ~0.96 against an actor loss of ~0.05. Measured on its snapshot vs its seed: placement entropy 51.9% -> 86.7% of max, effective cells 28 -> 260, top-1 0.260 -> 0.018; it went 166-839-2 against the agent it was an exact copy of. The card head (0.05 raw ~ parity with the live 0.0767) was unchanged, 39.9% -> 36.9%. Burst #0 also lacked the phantom-step shaping mask, and the exploiter's hand-built stats dict lacked team0_wincon_damage.
- **Exploiter history.** Disabled 2026-07-31 after three bursts: null / overall / tail / slices: #0 0.598 / 0.536 / 0.52 / (none); #1 0.598 / 0.585 / 0.55 / .59 .54 .56 .66 .55; #2 0.520 / 0.577 / 0.54 / .65 .59 .50 .60 .54 (z ~ 1.95, decaying; placement entropy sharpened 0.30 -> 0.20, so it was optimizing). Every phase-2 opponent was a mirror. Re-enabled 2026-08-09 once the pool held scripted and builtin members, prompted by drift between ep 4,127 and 17,050: placements at y <= 5 9.8% -> 27.5%, Cannon behind its King 4.7% -> 49.0%, mean placement y 11.52 -> 9.37, Fireball mean y 30.50 -> 12.57. Disabled again 2026-08-11: burst #0 @ ep 18,540 282-719-1 (0.28; slices .39 .22 .21 .29 .30), #1 @ ep 38,542 183-816-2 (0.18; .20 .21 .20 .14 .17). The main agent was dumping 54% of placements on row y=0 to shed elixir before W_ELIXIR_OVERFLOW charged for holding it. The Giant, separately, was affordable on only 4.7% of steps (P(Giant | legal) = 0.157 vs uniform 0.200).
- **League.** MIN_OPPONENT_AGE_EPISODES went 15,000 -> 6,000 on 2026-08-09 with the snapshot interval 5,000 -> 2,000 (15,000 / 2.2 ~= 6,800 after the speed fix; 3 x 2,000 = 6,000). EVAL_GAMES_PER_OPPONENT 10 -> 50: at 10 games one flipped game on an anchor at 0.9 moved its Elo by 141 and the average by ~28; across 8 evaluations spanning 35,000 episodes: mean 1755, std 53, range 164. Builtin anchors: 4 of 5 historical anchors sat at 0.90-1.00 for 8 evaluations, and the ~100-line heuristic beat a 47,000-episode policy 71% of the time; the policy scored 100% vs 1.0x and 97% vs 1.2x. Evenly spaced anchors: near score 1.0 one lost game is a ~222-point swing. The lineage filter (audit 08, 2026-09-15): the shared directory held 51 snapshots from the previous run.
- **Strategy metrics.** Measured in phase 1: the greedy policy had abandoned 2 of 8 cards (the win condition and the spell) and won by cheap defence, at a mean 3.6/10 elixir when it acted. AvgTicks moved 24% in one hour after the placement-mask fix, which is why TwrDmg is per 1,000 ticks.
- **Behaviour cloning (2026-07-29).** 60 episodes / 4,082 steps / 1,799 placements over 75 cells; 0.0% mask drops after the truncation fix; train loss 51.96 -> 44.21 over 10 epochs; held-out card match 0.300 -> 0.319; held-out cell match 0.000 -> 0.073, against 0.0016 random and 0.273 for always guessing the single most common cell. Underfitting (four teachers in one dataset; Defender/Counter place in response to the opponent). The rounding bug: a Rusher placed at getOwnHalfMaxY() = 15.5 (at the time) rounded to row 16, outside rows 0..15, so every Rusher demonstration was dropped and placement imitation sat at ~0.9%. The teacher no-ops on ~60% of steps, putting the trivial baseline for plain card_match near 0.75.
- **Tactical distillation.** The 3-arm coverage run (80 PPO updates, matched episodes, byte-identical code), Cannon at ep~64,800: seed modal cell (11,0) 88.7%, top-1 0.920, engine-scored 135.9 HP; control (coef 0) (11,0) 55.4%, 0.550, 116.4 HP; treatment (coef 0.02) (6,0) 79.4%, 0.051, 182.0 HP; random legal cell 394.7 HP. Fireball top-1 fell to 0.006 (uniform is 0.0016): entropy rose as designed and the head still returned one fixed cell. The frozen-trunk recipe came from expert iteration (+0.1027 vs +0.1415 lift, zero critic drift). masked_kl's -inf handling: the first run read `loss nan` from the first batch with the freeze intact.
- **Expert collection.** Hard labels gave +0.0462 lift, the value distribution +0.1027. The first attempt emitted the no-op twice ((NOOP, gx, gy) and (NOOP, 0, 0)), so 99.1% of rows had a target with zero value spread. With the default search (k_cards=3, k_cells=2) at most 7 candidates were emitted (4/2 -> 9, 3/3 -> 10, 5/3 -> 13); K_MAX grew to 160 when widened search arrived. Keeping the top-K on overflow collapsed the value spread 0.6371 -> 0.1036 (84%), moving the calibrated temperature from 0.592 to 0.858 of maximum entropy. Truncation by generation order could drop the search winner itself. The mirror ranks 16th of 16 on opportunity for the spell/building cards (Fireball catch 323 HP vs pool median 494; Log 145 vs 273). A gate measurement once ran against the heuristic while reporting the teacher because the config used `opponent_kind`.
- The expert-iteration record (hard labels +0.016 [-0.030, +0.061], distribution distillation, DAgger +0.045, p = 0.0074) is in CLAUDE.md's open problem #2. From the code: the ablation's four configs came out monotone in no-op rate (0.872 / 0.874 / 0.899 / 0.967); ~90% of labels (86.2% at a 13.8% deviation rate) match the policy; the expert no-ops 89.7% of the time; the balancing upweight is 89.54 / 10.46 ~= 8.6; an ~800-trial run showed ~6,700 paired trials are needed to resolve per-config effects; `--full-finetune` measured critic drift 0.048086; T=0.25 put the target at 94% of maximum entropy against a value spread of mean 0.221 / median 0.187-0.193 (0.05 puts it at ~50-55%). The exact `!= 0.0` freeze check fired on every frozen run (44 tensors verified bit-identical; only card_head and place_ctx/place_up changed). Loading all observations at once cost ~1.1 GB per 80 episodes.

#### Evaluation harnesses (eval/)

- **Side null.** Before the 2026-07-31 observation fix (`33 - int(y)` instead of `int(33 - y)`) it measured 0.598, after 0.520 (n=400). A first attempt used the env's teacher against a separately constructed one and read 0.300: the unpinned profile made them different opponents. Measured 2026-08-26, stage 0, pinned "balanced", n=100: team-0 share of decided games 0.510, 95% CI [0.412, 0.608], 0 draws.
- **net_h2h / net_ab.** Written when the cured net measured 11 points below v1.2.0 against the C++ heuristic after 13,961 episodes in an all-neural PFSP league; the per-card ablation's raw win rate (0.507) sat below the 0.584 recorded for v1.2.0 in a different harness on different openings. The control's own variance across runs of one net was 0.570-0.775. The side bias read 0.530 at the time. Every harness that retyped the board width as 18 was a second copy of a board constant.
- **match_outcome.** net_ab.py, net_h2h.py and net_h2h_search.py (twice) each scored step 1 alone and called everything else a draw, so a 3-3 finish at 1,200 HP vs 90 HP on the weakest tower was reported as a draw. The mirror originally located the tail as `observation_size() - NUM_EXTRA_SCALARS`, which broke on 2026-08-27. The binding came from UPSTREAM_REQUESTS.md item 16 (applied 2026-08-20).
- **gate_ab.** The control arm alone measured 0.570-0.775 across runs of the same net.
- **entropy_pull_geometry.** Written to test whether the placement entropy coefficient, pinned at its 0.01 floor, explains The Log's placement drift after rolling spells were capped at own half + river (2026-08-29). Prediction: large pull and high progress for The Log, small pull for Fireball; otherwise the entropy floor is not the explanation. Six of 334 states (1.8%) had no catch value on any legal cell and turned every mean into nan before nanmean.
- **probe_card_usage.** Written when the open question about the Giant deck was whether the 5-cost Giant was ever played (never, across four runs). Measured affordability sat at 3-9% for every card.
- **bench_search_latency.** Horizon 12 was chosen offline on wall-clock-per-episode grounds (1.5x cost); the live constraint is the per-decision tail.
- **force_card_ab.** The ep-59,000 gate re-run found forcing The Log at the policy's own argmax worth +749 tower HP on the 32 most favourable states, against +46 HP for improving where it lands. Forcing Fireball once dropped win rate 97% -> 23% (force_hog_ab.py), and four earlier attempts to raise these cards by pushing the marginal all cost win rate.
- **tactical_ab.** Cannon / Fireball / Giant were played on ~2% of plays with the head returning (11,0) in 54-91% of states. Fireball fired at 690 HP caught (3 x 230 Minions), Cannon at 721 threat (one Musketeer). The ungated first run forced ~27 extra elixir per episode into an agent spending ~105 against ~98 of income, and both forced arms lost: it measured bankruptcy.
- **hybrid_ab.** The gate A/B at n=130 had a 95% CI of width 0.243. As of 2026-08-14 the head matched the advisor on Fireball (p = 0.163) while losing badly on Cannon (p = 6.8e-22); the hybrid had measured +11.8 win-rate points. The full hybrid lost 57% of its tower damage dealt in smoke tests, which the opponent-aware reserve did not recover. The solvency gate moved bankruptcy 72.7% -> 41.7%. The per-card arms were "cannon_only" / "cannon_giant" / "all_three" until 2026-08-19; Giant left DEFAULT_DECK on 2026-08-16, so two arms were bit-identical duplicates and fireball_only was missing.
- **prove_hog.** The 2.6 run at ep 6,053 played Hog Rider on 0.8% of decisions. PLACEMENT_COLLAPSE.md had found the policy's Cannon cell preserved 121 HP against a random legal cell's 396, worse than chance. A lone Hog from the bridge measured 317 damage in 40 s.
- **probe_placement_prior_ab.** On `model_weights_dist_e3.pth` Mini PEKKA had the lowest per-card entropy and was the most-played card while Cannon had 91.0% of its mass on one cell; a play-conditioned readout let the 2026-08-14 collapse run for thousands of episodes.
- **stats.** Four near-identical "paired bootstrap + exact sign test" implementations existed (`prove_environment.paired`, `prove_placement.paired_report`, `prove_solvency.boot_ci`/`boot_diff`, `expert_iteration.report_paired`). The Fireball result at ep 78,270 had the bootstrap CI exclude zero while the sign test did not (223 better / 252 worse, p = 0.199). The power line in `report_paired_winrate` exists because an exploratory n=200 arm gave +0.105 at p=0.044 and a confirmatory run at 4x the power collapsed it to +0.016; every expert-iteration delta in CLAUDE.md came from that arithmetic.
- **search_ab_test.** Split from `search/search.py` because five modules imported this harness's private functions. Written when the engine RNG could not be seeded (UPSTREAM item 7: ~1,568 episodes per arm unpaired to resolve 5 points at 80% power); UPSTREAM item 13 records an earlier attempt whose CI came out 15x wider than the effect. At 1.0 opponent elixir the ep-64k policy won 4/4 trials 1.000 vs 1.000; the policy family plateaued around 0.86 at 1.5x.
- **probe_perfect_defense.** The 2026-08-06 Cannon pathology put 27.9% of Cannons at (11,2)/(11,3) behind its own King. The pull pocket's centre was hardcoded as 9.0 with a comment calling x=9 the board centre; it is 8.5 ((18-1)/2), and the half-tile error counted x=12 as central while excluding x=5.5.
- **prove_solvency.** Baseline on `model_weights_dist_e3.pth` (40 greedy episodes, 1.5x): below 3 elixir 65.3% of all decisions and 60.8% during a big push (>8), mean elixir 2.53, 29.1 plays/episode, ~105 elixir spent against ~98 income. The ">8 elixir" bucket originally summed the elixir cost of enemy troops past the river.
- **search_vs_greedy_pool_ab.** The +0.319 behind expert iteration was measured 2026-08-11 against the C++ heuristic at 1.5x, ep-64k Giant-deck checkpoint, horizon 4; deck, opponent, policy and engine have all changed since. The 2026-09-05 gate re-run found search's per-card aiming advantage non-significant (The Log +46 HP p=0.146, Cannon +262 p=0.092, Fireball +200 p=0.227). More data from a weak expert degraded selectivity 3.07 -> 2.31 -> 2.17. Before the teacher was seeded, the greedy arm scored 0.750 and 0.875 on identical seeds (epsilon 0.12 at rung 3). Horizon 12 was the 2026-08 sweep's optimum (0.963). terminal_weight 10.0 predates the 2026-09-06 rule ending a match at 3:00 on a crown lead, after which 12-step rollouts reach terminals far more often. A full teacher as rollout opponent would cost 5.0 ms vs 0.6 ms per decision rules-only.
- **net_h2h_search.** net_h2h.py measured the cured net at 0.6125 against v1.2.0 greedy-vs-greedy; search was worth +0.183 against the C++ heuristic (2026-08-15, n=60 paired, CI [+0.032, +0.334]). The leaf value and duel verdict used tower count alone until match_outcome existed, calling every equal-count finish a draw; search A/B figures from before that fix are not comparable. Side bias read 0.598 once and 0.530 at the time.
- **profile_architecture.** Built for phase 4's 15% throughput budget (receptive field, scalar encoder, BPTT horizon), at 943 ep/hour with the trunk 43% and placement head 41% of update wall clock; `forward_sequence` was worth 1.82x. The first version rebuilt the batch (500 x 13,976 floats, 28 MB) and an Adam state per round and reported a baseline 35% slower than the same net alone. With two identical arms a fixed order read the second 6.8% slower, larger than the effects being measured. The interleaving follows tools/audit/collision_bench.cpp.
- **probe_card_counting.** Item 24 put the opponent's seen[]/recency[] in the observation and phase 4 gave it a 24-dim scalar branch. The random-init floor is the analogue of the OLS control that showed the opponent-elixir head's MAE 0.77 was arithmetic on two present scalars (OLS scored 0.0000).
- **prove_wincon_trade.** Written after `prove_environment.py --mode winrate` found, at a symmetric 1.0x economy, a teacher committing its win condition at the bridge scoring 0.510 against one dumping it in its own back half at 0.840 (n=100 paired, delta -0.330, p=5.7e-08). Two explanations: the trade is bad (card unviable), or the teacher's timing is bad. Unopposed control, 2026-08-19: a lone Hog at either bridge on an empty board dealt 2536 (a full Princess Tower) and died at tick 190, when the King still fired from tick 0. The defender answered with Skeletons in 35 of 40 trials and Ice Spirit in 31. At the time the observation carried no opponent-cycle information, so neither teacher nor policy could aim for the no-answer window; inject needed one tick before the unit was on the board (0.0 mass before, 0.399 after).
- **force_hog_ab.** The 2.6 net played its win condition on 0.0-0.4% of decisions. The one earlier forced-usage experiment (Fireball) dropped win rate 97% -> 23%. prove_hog measured the net's Hog cell at ep 18,013 as indistinguishable from chance, and at another point -48.3 against a random legal cell. Leaving gate_mult at 1.0 in a 1.5x test under-estimated the opponent's bar by ~50%, opening the gate precisely when they were banked.
- **prove_placement.** The first, botched A/B reported a "99% modal share" for a policy at 0.999 normalised entropy. An earlier version called the LSTM once per query and again for the reference action, advancing the reference's recurrent state twice per environment step. Unpaired, ~1,568 episodes per arm resolve 5 win-rate points.
- **prove_hires.** `distill_tactics.py` fitted the advisor's exact cell with the trunk frozen: cross-entropy fell 180.9 -> 21.4 while exact-cell argmax match stayed at 0.0%, read as the head being unable to represent an exact cell. `test_python_ai.py` found that premise too strong: on 14 boards differing only in the column of one enemy, the coarse head fits 14/14. A near-uniform soft target (~95% of max entropy) once cost an expert-iteration run.
- **prove_teacher.** At 1.0x the C++ heuristic failed bar 1 (the ep-64k and ep-25202 nets both beat it ~100%). The side bias once measured 0.598 on a bit-exact copy. `_score` originally used surviving tower count alone and ignored TimeoutRules' HP tie-break.
- **measure_deck_matchups.** The 2026-08-29 deck autopsy found three cards at P(play | in hand) <= 0.009 and the card head right to price them there: forced through `env.step`, a Cannon at the policy's own cell was worth +185 tower HP, better in 6 of 14 states. All of those measurements held the opponent's deck fixed at the mirror. The Log's corridor replaced a 7.8-wide circle on 2026-08-28. LOG_DAMAGE was a literal 240.0 labelled "CardRegistry.h" while the registry says 269, capping each body 11% low; CLAUDE.md's 145 / 273 / 427 Log-opportunity figures (recorded before 2026-09-23) used the old cap. The reference loop originally used `int(y - LOG_RANGE)`, making the Log 11.1 tiles long.
- **probe_card_discrimination.** Built after the 2026-09-03 deck pool predicted Cannon / The Log / Fireball would become worth their elixir. Fireball sat at P = 0.0011, and the run-to-run noise floor for this class of probe is +-0.005. The 2026-08-29 autopsy put the real constraint on placement: "the card head was correctly pricing a broken placement head".
- **probe_placement_oracle.** Commit 1b77f27 measured widened search on the Cannon capturing 81% of the oracle ceiling (+994 of +1223 tower HP) against +236 for un-widened search, the justification for distilling search into the placement head. The Log and Fireball had quality_hi 0.217 and 0.685, and The Log had the flattest head in the deck (top-1 0.0288). A hand-rolled build_net once passed `len(DEFAULT_DECK)` as MicroRoyaleNet's first positional argument, so the loader discarded `cnn_trunk.0.weight`. The first version passed `opponent_kind`, which gym_wrapper does not read, and measured against the C++ heuristic on the mirror (16th of 16 on opportunity). Without de-duplication, n=16 was really n=2.
- **prove_environment.** Built to test the 1.5x hypothesis: a permanent opponent-elixir multiplier suppresses punish cards, since a punish window lasts about `answer_cost / (m * r)` while the counter-cost of our spend scales with m; every earlier test ran a policy through the environment. The back-half placement measured ~3 enemy tower damage against 535.6 at the bridge. At n=12 a mirror-strength opponent at 1.25x or 1.5x won every game. The ceiling rule is what voided the original test's 1.00x row. HOG_MAX_OPP_ELIXIR shipped at 7.0; the trade probe measured a lone Hog at 158.5 hp/elixir against a defender at match elixir vs 256.2 against one forced to 1.0. After deploy time landed, a lone Hog in a punish window fell 1025 -> 343 tower damage while the escorted push held at 993.8.
- **prove_combos.** The first version of the combo change proposed plays the bot could never afford (P(bar >= 6) = 0.3%). Seeding was corrected 2026-08-24: before, `--seed` reached only the teachers' RNG (at stage 5, just the lane bias) while the opening-hand shuffle was unseeded (UPSTREAM item 7, applied 2026-08-21 but not wired in here); an ablation meant to reproduce an earlier run's OFF arm at seed 300 (0.537) reported 0.475. That pair is the acceptance test: if two `--seed 300` runs still disagree on the OFF arm level, compare deltas only. The session brief's "ep 25202" checkpoint was deleted in the 2026-08-19 cleanup; `model_weights_selfplay.pth` (ep 31,312) is its descendant. DEFAULT_PLAY_MARGIN was once a restated copy that went stale when play_margin moved 0.05 -> 3.0. ENGINE_SEED_OFFSET mirrors the `^ 0x9E3779B9` ClashEnv::seed uses between its two generators. Stage 5 scored 1.000 against the C++ heuristic.

#### Tools

- **make_replays.** The control arm alone measured 0.625 / 0.570 / 0.700 / 0.634 across four runs; the distilled net's measured effect was +0.045 (~1 game in 22) and search deviated on 13.8% of decisions. The four search imports were lost in the 2026-08-20 package restructuring, so the script died with `NameError: LSTM_HIDDEN` until 2026-08-21. `play_and_log_vs_teacher`'s stage default was a literal 5: the top rung of the six-rung table, the middle of the eleven-rung one. Until 2026-09-04 the tool recorded the 2.6 mirror unconditionally, so every viewer replay showed the matchup ranked 16th of 16 on opportunity while the trainer had played a 16-deck pool since 2026-09-03.
- **migrate_checkpoint_elixir_phase.** Written for the 2026-09-02 elixir-phase scalar (NUM_EXTRA_SCALARS 9 -> 10): 12 numbers out of 1,900,165, preserving inputs the policy had read for 32,484 episodes. `test_checkpoint_paths.py` caught the first draft writing with torch.save. The four-branch ScalarEncoder dates from 2026-08-27.
- **reset_aux_heads.** Measured 2026-09-03 resuming `model_weights_phase5.pth` into the 16-deck pool: NextCard CE 13.76 against ln(185) = 5.22; the weighted term reached ~7x the actor loss while win rate fell 0.58 -> 0.00 in 126 episodes (later shown not to be the collapse cause; see CLAUDE.md "THE STALE AUX HEAD").
- **monitor_run.** The win-rate / floor-alarm checks were added 2026-09-15 (audit 08). The advisor target buffer was ~9.8 MB per rollout. The spell anneal was wired in 2026-08-14 after being dead code for the whole life of the term. The collapse detector once watched Cannon, Fireball and Giant, and Giant was no longer in the deck. The copy-before-load fix is audit 08 gap 4.
- **run_watchdog.** PFSP deck estimates ride in the checkpoint since 2026-09-04; before that an automatic restart discarded them each time. Before 2026-09-15 (audit 08) it matched only `*trainers.train*`: after the handoff it saw "process gone", relaunched phase 1, which launched another phase 2, up to five phase-2 trainers on one checkpoint. `--weights` once defaulted to model_weights_phase7.pth, which existed.
- **setup_ab_arm.** The bare-state_dict trap cost one inconclusive experiment: target 0.65 against a converged policy at ~0.11. `place_hires` added 6 parameters (26 -> 32), landing at positions 22-27; the failure was caught smoke-testing pipeline 2. For the hires experiment, of 27 shared tensors 18 were bit-identical and 9 changed (card_id_embed, place_ctx, place_up). Without the table stamp, `--stage 5` became rung 10 (audit 04 C4).
- **track_placement.** At ep 32,875 quality_hi was Fireball 0.675, The Log 0.203; 490 episodes of deck pool did not move it (signal 0.0). The stale aux head read 13.76 and pulled the trunk ~7x harder toward card identities than toward winning. Reading only the newest event file printed "worst deck None" after each of three restarts in one session. The project was once fooled by a three-point read whose fourth point reversed it.
- **profile_training.** The conv placement head costs ~33x the dense head it replaced; `forward_sequence` was worth 1.82x. The pybind half of the boundary was measured on the real module at 0.611 ms to marshal 13,606 floats against a 0.0023 ms memcpy floor. A profiler span of 2 us on a call made 400,000 times an hour moves what it measures.
- **validate_pipeline.** The legality check it replaced tested `isfinite(t[~legal])` against the table the target was built from and reported "0 violations" over thousands of targets. The defensive-bot floor exists because at 0.20 the two bots got ~7.7% combined in a ~98-member pool. The scenario-contract check once collapsed three contracts into one and failed 15/32 against correct code; its label once said "all five" while printing 4/4 after giant_commit was removed. The advisor value check once FAILED on the Cannon (advisor -16.8 HP vs random 117.2, n=28); prove_placement.py settled it at n = 894 (Cannon) and 1937 (Fireball). Search costs as documented: snapshot 0.033 ms, 10-tick step 0.027 ms, 20-tick rollout 0.34 ms vs a 50.51 ms forward. The side null read 0.598 (z = +3.90) on 2026-07-31; its checkpoint was deleted 2026-08-19, leaving it unrunnable for weeks until the random-init fallback. On 2026-08-20 `build_test` held a binary 18 h older than the engine: the gate reported "546 test cases / 5,316 assertions" and passed while the current binary had 582 / 5,737. On 2026-09-15 tooling rewrote six headers byte for byte and the mtime gate failed a verified-current build.

#### Python tests

- **test_monitor_sees_a_dead_run.** Audit 08 (2026-09-15): the monitor read no win rate, reward or curriculum state, and its placement probe watched a Giant not in the deck; a from-scratch run losing 4,000 straight games reported "0 alarm(s)".
- **test_exploiter_refuses_champion_decks.** TODO 00.1(b).
- **test_watchdog_follows_the_handoff.** Audit 08 gap 1: `*trainers.train*` missed the phase-2 child launched by path (a backslash where the pattern wants a dot), so after the handoff the watchdog relaunched phase 1, which launched another phase 2: up to five concurrent phase-2 trainers on one checkpoint. `--weights` defaulted to an old run's checkpoint.
- **conftest.** The two fixtures moved verbatim from the single `test_python_ai.py` when it was split on 2026-08-20; "improving" `fresh_obs`'s return shape broke the first attempt at the split.
- **test_teacher_uses_champion_abilities.** Until 2026-09-16 the teacher issued no ability (gym_wrapper passed False for team 1), so the phase-1 mirror of a Champion deck played it as a plain troop. Golden Knight, Archer Queen and Monk were all ready from their first second.
- **test_placement_modal_share.** Measured 2026-08-14: the lowest-entropy card was the most-played (modal share 19%) while a card at 91% modal share had higher entropy. Audit 08 found no modal-share scalar logged anywhere.
- **test_log_damage_is_derived.** `LOG_DAMAGE = 240.0` was labelled "CardRegistry.h" while the registry says 269, capping every Log-opportunity figure 11% low per body; the reference loop its docstring called the test oracle had never been run.
- **test_scenarios_follow_the_deck.** The two spell scenarios are 60% of the injection weight, 18% of all phase-1 episodes at SCENARIO_INJECTION_PROB = 0.30. `giant_commit` was removed 2026-08-19 for the same reason at 17% of the budget.
- **test_clash_settings_stamp.** TODO 00.9: only the deck was recorded, so a resume under a different (or missing) CLASH_GAMMA continued silently.
- **test_deck_definition.** Until 2026-09-15 changing the deck meant editing the `DEFAULT_DECK` literal in read-only `envs/gym_wrapper.py`, unvalidated and unlogged.
- **test_advisor_target_is_deck_agnostic.** Until 2026-09-15 `ADVISOR_CARDS` was three literal ids (Cannon 25, Fireball 7, Hog 15): on 5 of 8 plausible replacement decks the coverage term (10% of log 612 on the placement head) trained on zero cards (audit 07, F1).
- **test_validate_pipeline_gates_are_honest.** 2026-09-15: tooling rewrote six headers byte for byte and the mtime gate failed a behaviourally verified build. `model_weights_selfplay.pth` was deleted 2026-08-19, leaving the side null permanently unrunnable.
- **test_phase2_pfsp_persistence.** TODO 00.8: estimates lived only in worker memory; with 8 workers and ~100 opponents each needed ~1/PFSP_EMA_ALPHA = 12 re-meetings to recover.
- **test_champion_abilities.** Before 2026-09-16 a Champion deck could not be trained: no ability was sampled, `base_trainer.setup` raised, and `validate_deck` refused it.
- **test_aux_warmup.** Measured 2026-09-15 at the from-scratch config over 12 real updates: the aux gradient grew 0.62x -> 1.67x the size of every other term combined on the LSTM (0.63x -> 2.01x on the CNN trunk), cosine -0.37 -> -0.84 (-0.90 on the trunk).
- **test_deck_contract.** The 2026-09-15 audit found four deck-keyed mechanisms silent under another deck: the advisor target, the win-condition reward, the Fireball-keyed spell terms, the Fireball-only scenarios. Champion decks were refused until 2026-09-16; a no-Fireball deck warned "the spell terms contribute nothing" until then.
- **test_teacher_counter_is_not_unconditional.** Audit 05 BUG 2, 2026-09-15: 30 of 116 top-rung matches against a do-nothing opponent froze, bar full on 65-96% of decisions, six at zero tower damage; `classic_log_bait_inferno` went 1/3/0/3 crowns over seeds 1-4 vs 3/3/3/3 with the counter off. The counter was adopted on +0.158 paired win rate against an active opponent.
- **test_unattended_run_survives.** Audit 08 (2026-09-15) found the core resume sound and these gaps around it: monitor_run.py held the checkpoint open on every check; the plateau tracker reset cost up to ~2,000 episodes per crash; historical_checkpoints/ held 51 snapshots from the previous lineage.
- **helpers.** Both builders were defined in one section of the old monolithic `test_python_ai.py` and used from another; splitting it turned that into an ImportError.
- **test_teacher_spell_geometry.** `_top_spell_cells`, both spell combos and the rung-0 gate called `tactics.spell_catch_map(obs)` with Fireball defaults. Measured on 320 mid-match boards against eight pool decks, engine-scored by elixir value killed: Rocket +38% (better on 75, worse on 4), Zap +11%, Poison +7%, Arrows +4%, Lightning level, Fireball's cell identical on all 320.
- **test_rl_buffer.** The buffer replaced per-field append/stack/clear triples repeated in three places across two files.
- **test_teacher_air_defence.** TODO 00.5: the rung-0 gate picked Skeletons and Ice Golem against a lone Balloon more often than Musketeer or Ice Spirit. Measured, rung-0 teacher with the 2.6 deck, 36 seeds paired, 40 s of a lone push: Balloon 1481 -> 1131 tower HP lost (9 better / 2 worse / 25 tied), Lava Hound 877 -> 807 (17 / 7 / 12), Hog Rider 810 -> 810 (the ground control, bit-identical); pooled 26 better / 9 worse, sign test p ~ 0.006.
- **test_tactics_scalar_offsets.** tactics.py read `obs[-9]` / `obs[-7]` until 2026-08-29, pointing into the card-recency block after the 2026-08-27 append: `elapsed_ticks` read 0.0 at engine tick 300. The sweep that fixed five other sites searched for `observation_size() - NUM_EXTRA_SCALARS` and could not match a negative index.
- **test_selfplay_pool_robustness.** The pool was written non-atomically until `atomic_save` landed 2026-08-26, so truncated entries (OOM kill, full disk) can already exist in `historical_checkpoints/`.
- **test_curriculum_from_scratch.** Audit 04, 2026-09-15. C1: driven through the real manager, 4,000 episodes at rung 0 / 0.00 fired no event, and monitor_run.py read no win rate. C5: adding one rung sent a rung-3 checkpoint to rung 7 while printing "(same lookahead horizon)". C6: `on_episode_end` skipped the deck read-out for scenario episodes, ~30% of calls. C4: `--stage 5` (40 ticks) resumed at rung 10 (100 ticks).
- **test_deck_pool_cold_start.** Audit 04 C2/C3, 2026-09-15. The fallback tested `wr >= floor` with the floor at 0.0 since 2026-09-06, so it could never fire; its tests all passed an explicit `floor=0.20`. The shipped priors (measured against a trained 2.6 policy) gave a random-init net the four hardest decks 59.9% of episodes and the mirror 1.0%; simulated over 8 workers they bought no episode-share benefit and 3.2x larger estimate error at episode 3,000. The old fallback's bare wr**2 gave the hardest deck 0.1%.
- **test_replay_path_anchoring.** Checkpoints and runs were anchored 2026-08-25; `replays/` was missed although `run_path`'s docstring claimed to cover it and it was already imported into base_trainer.py. A stale `replays/replay_ep1000.json` sat at the repo root and a smoke run from a scratch dir wrote `replays/` there.
- **test_first_step_after_autoreset_is_not_rewarded.** Audit 03 R1, 2026-09-15 (`probe_reset_hygiene.py`): phantom at t=141 with elixir 0.000, first real step t=142 at 3.350, Phi_solv(prev) = -0.10000, Phi_solv(cur) = -0.01625, spurious F = +0.08377, unmasked. The solvency term's legitimate per-episode magnitude was 0.039; the error varied with scenarios' banked starting elixir (~30% of episodes) and was the entire reason the term failed to telescope.
- **test_deck_pool_persistence.** Measured on the live 2026-09-04 run after 12,589 episodes, learned vs JSON prior: xbow_30_cycle 0.181 vs 0.867, mortar_cycle 0.176 vs 0.800, dart_bait_cycle 0.111 vs 0.533, hog_26_mirror 0.406 vs 1.000. xbow's weight was wrong by ~37x on resume; the 100-episode win rate fell 0.42 -> 0.143 and took ~500 episodes to recover, every restart.
- **test_rl_buffer_drain.** `run()` cleared the buffer after `run_update()`, so both copies stayed resident through the update (~87% of the cycle): obs 13,606 x 8 envs x 4 bytes = 0.435 MB/step, x 500 steps = 217.7 MB, 435.4 MB for both, against a main process measured at 1,194 MB private commit.
- **test_trainer_seed_reaches_the_engine.** `seed_everything` covered torch / numpy / random, `worker_seeds` seeded only each env's scenario Generator, and `BaseTrainer.setup` called `envs.reset()` unseeded. Two trainers with CLASH_SEED=4242 had identical network init but opening hands [[7, 24, 6, 33], [24, 25, 6, 7]] vs [[24, 33, 25, 72], [40, 15, 25, 24]], while printing "Deterministic run: seed=4242".
- **test_rl_episode_metrics.** Nine deques declared identically in two files became EpisodeMetrics. The all-draw decisive rate was previously pinned at 0.0, conflicting with the module's NaN rule; summary() once used NaN, 0.0 and None for "undefined" in one dict.
- **test_tactics_disc_offsets.** `spell_catch_map` rebuilt the offsets once per occupied enemy cell while `_reach_cover` in the same module hoisted them; `_FIREBALL_DISC` was computed at import and never read.
- **test_teacher_wincon_resolver.** Until 2026-09-15 two copies ranked by cost ("the most expensive building-targeter"); audits 05 and 07 measured W_WIN_CONDITION_DAMAGE dead for 4 of 8 plausible replacement decks (siege, spawning spell, deploy-anywhere), a Miner deck's teacher naming a 2-elixir Ice Golem, Miner control with no win condition, and LavaLoon inverted (Lava Hound over Balloon). While fixing it: per-elixir ranking named the Miner over the Balloon (the probe saturates at one Princess, 2534 HP), and Barbarian Barrel beat the Graveyard in graveyard_control when probed on an illegal cell. TODO 00.6: `siege_reach > 0` admitted spawner buildings; over 28 decks Splashyard named Tombstone over its Graveyard and a Barbarian Hut outranked the Giant (6182 in a 1200-tick siege window against a troop's 300-tick one). A Goblin Drill measured 0 from its own siege row and 2654 beside the tower in 300 ticks.
- **test_search_opponent_model.** Measured 2026-09-06 on the ep-111k policy, paired and seeded, greedy constant at 0.844: search 0.531 / 0.312 / 0.375 at horizons 4 / 8 / 12 (-0.313 / -0.531 / -0.469). Every positive search result (+0.319, the +0.4025 sweep) was against the heuristic. SearchCfg's docstring said "a candidate rollout assumes BOTH SIDES NO-OP", which misled the first diagnosis until a rollout was inspected: 2 enemy bodies and 1302 tower HP inside a supposedly empty rollout.
- **test_phase1_scenarios_do_not_break_the_gate.** With the old 0.80 gate, 30% injection capped the achievable win rate at ~0.70; the 2026-08-28 run already looked like an agent plateaued just short of the bar for other reasons.
- **test_rl_optim_step.** Measured 2026-08-26: 32 of 32 parameter tensors NaN after one poisoned step, still NaN after a clean one. Five loops had the same unguarded pattern: rl/ppo.py, trainers/exploiter.py, trainers/bc_pretrain.py, trainers/distill_tactics.py, trainers/expert_distill.py.
- **test_placement_legality_is_static.** The cache's original premise was full board-state independence ("208/288 legal cells on an empty board and with six troops down"). Measured 2026-08-27: troops and enemy Princess deaths change nothing; our own Princess dying frees +9 cells (its 3x3 footprint).
- **test_opponent_cycle_observation.** Five call sites computed the tail as `observation_size() - NUM_EXTRA_SCALARS`, correct only while the extra scalars were last; UPSTREAM_REQUESTS.md item 25 predicted the silent reward corruption, fixed with one bound forward offset.
- **test_placement_mask_covers_every_card_class.** Until 2026-09-15 (audit 06, E-1) `placement_mask` ANDed an own-half row mask with the legality table, deleting the enemy half for deploy-anywhere troops: Miner 242 cells vs the engine's 520 (rows 16..33 lost), Goblin Drill 170 vs 372. Every Evolution had an all-False legality row and `_spell_flags` missed Evolution spells; the engine plays Evolution Archers on 242 cells.
- **test_envs_scripted_opponents.** The opponents were 120 lines in `MicroRoyaleSelfPlayEnv._scripted_opponent_action`, testable only by building a whole self-play env. The defensive floor at 0.20 gave ~7.7% combined share in a ~98-member pool. `find_incursion` once scanned only up to int(MAX_Y); an isolated eval found 55-70% of head-to-head games ending in an early blowout regardless of sampling exposure (reaction latency, not weight, was the bottleneck).
- **test_rl_config.** The hyperparameters were ~200 lines of locals in `train_ppo()`, byte-identical to `train_selfplay_ppo()`'s copy apart from three entropy numbers. gamma 0.99 -> 0.999 on 2026-08-27: at 0.99 a terminal reward decayed to 0.99^360 = 0.0268, one crown was worth 22.4x a win. bptt_chunk 25 -> 50 the same day (160 segments / 20 per minibatch before).
- **test_rl_advantage_masking.** Measured before the fix over three rollouts off a 308-episode checkpoint (decision covered 73.7-75.1% of rows): centering error -0.0312 / -0.0820 / -0.0073 of a unit std, scale 0.9757 / 0.9492 / 1.0041.
- **test_validate_pipeline_legality.** The retired check printed "0 violations" over thousands of targets.
- **test_strategy_metrics.** `AvgTicks` moved 24% in one hour after the placement-mask fix. ~6.0 cards/game was the phase-1 failure mode (win condition and spell abandoned). `fireball_tower_value` once pushed ScenDef toward 1.0.
- **test_rl_ppo_degenerate_batches.** `forward_sequence`'s comment records P(nothing affordable) = 73.9% per step. `ent_card_mean` / `ent_place_mean` were appended unconditionally; rl/ppo.py's fallback ("keep the old denominator rather than feed the controller a 0") had the same hole one level up.
- **test_threat_gated_deck_coverage.** Paired arms on the ep-32,484 checkpoint, scenarios on: coef 0.00 win 0.600 -> 0.520, 0.05 -> 0.340, 0.20: 0.620 -> 0.120 (a full launch reproduced it). Over 20 defensive scenario states, tower HP conceded: no Cannon 4938, Cannon at the policy's cell 4097 (+841, better in 14/20), at the best cell 2848 (+2090). Training/Win_Rate_100 excludes scenario episodes, so the crash was measured exactly on the quiet boards.
- **test_phase1_defensive_scenarios.** Scenario injection was pipeline-2 only; across the 2026-08-28 phase-1 run (32,680 episodes) P(play | in hand) was 0.0053 Cannon, 0.0091 The Log, 0.0011 Fireball, while a well-placed Cannon prevents a full Princess Tower (2,536 HP) in a threat state. `_BRIDGE_LANES` was [3.5, 13.5] from the pre-2026-08-21 arena, so half of every bridge push landed on water (the eighth stale arena copy). The truncation test failed intermittently under -q until the engine was seeded.
- **test_placement_mask_after_tower_loss.** Measured: own Princess destroyed 242 -> 251 legal cells (+9), enemy 242 -> 242. The mask forbade legal cells rather than permitting illegal ones. GAMEPLAY-AFFECTING: widened the action space in a state most matches reach. A first draft used `hand.index(card)`, which raised for undealt cards.
- **test_scenario_offense.** Split from the monolithic `test_python_ai.py` 2026-08-20. The state setters came in commit 26de409.
- **test_aux_next_card_task.** The task was swapped from opponent elixir 2026-08-28: OLS on two observed scalars scored MAE 0.0000 against the head's 0.77. `eval/probe_card_counting.py` found the trained hx decoded next-card +0.013 over a random projection at ep 1522 and -0.025 at ep 2054.
- **test_rl_engine_stats.** The stats dict was written inline three times (two trainers and the exploiter); the exploiter's lacked `team0_wincon_damage` and trained without the win-condition term. A towers-alive default of 0 would be a phantom +1.8 reward. The solvency-default residual was +0.001 at gamma 0.99. The test imported SPELL_SOLVENCY_RESERVE until 2026-09-16 (also 4.0, so it passed by coincidence).
- **test_flexible_load_grows_widened_heads.** 2026-08-28, phase-1 run, episode 7,609: the cycle skip connection gave card_head, value_head, place_ctx and place_ctx_hi 24 new input columns; the loader discarded all four (the whole card policy, both placement contexts and the critic) while reporting a warm start of the other 48 tensors. Win rate fell 0.80 -> 0.15 within 80 episodes, the curriculum demoted, ~1,500 episodes lost. Zero-padding follows `place_hires[-1]` and `DilatedContextBlock`'s zero-initialised final 1x1.
- **test_phase1_deck_pool_wiring.** The trainer's `random_opponent` phase once relied on `set_opponent_deck` surviving reset. A flat-rate EWMA kept a fresh net on the ep-32,484 policy's hard-but-winnable decks: 0 wins in 50 episodes. The estimator test measured against POOL_WINRATE_FLOOR until that became 0.0 on 2026-09-06, after which it could not fail.
- **test_rollout_no_redundant_conv.** Measured 500 steps x 8 envs, best of 3, network only: conv1 calls 2 -> 1 per step, 6.505 s -> 2.103 s (67.7% saved), placement logits bit-identical. `forward_sequence` already threaded `hires_seq` in the update path.
- **test_eval_stats.** Four harnesses had their own paired-bootstrap + sign-test copies; `prove_hog.py` was a fifth, resampling `np.random.choice` 5000x on the unseeded global RNG, so its "better / worse than chance" verdict could flip between runs. The Fireball result at ep 78,270: CI excluded zero, sign test did not (223 / 252, p = 0.199). The exploratory +0.105 at p = 0.044 collapsed to +0.016 at 4x power. Unpaired, ~1,568 episodes per arm resolve 5 points.
- **test_search_widens_diffuse_heads.** 14 threatened scenario states, Cannon in hand and paid through env.step, tower HP conceded: policy 5110; head argmax 4932 (+178, better 6/14); search top-k 4874 (+236, 8/14); search wide 4116 (+994, 10/14); engine oracle 3886 (+1223, 14/14, sign p ~ 6e-5). Widening was worth 4.2x and captured 81% of the ceiling; the Cannon was not marginal, its placement was. Mean top-1 per card, ep-32,484: Hog 0.7654, Musketeer 0.3939, Skeletons 0.3527 | gap 0.227 | Ice Golem 0.1259, Fireball 0.1074, Ice Spirit 0.0987, Cannon 0.0970, The Log 0.0288; 0.25 sits 1.4x below Skeletons and 2.0x above Ice Golem. Ice Golem and Ice Spirit were among the most played (P(play|in hand) 0.187 and 0.325).
- **test_rewards_weights.** `spell_value_weight` was dead code for a training era (no test varied its argument), so every win rate then was earned at a constant 0.08. SPELL_SOLVENCY_RESERVE (4.0) was retired 2026-09-16. `tactics.py` carried a bare `FIREBALL_DAMAGE = 689.0` under a comment claiming it was read from the registry.
- **test_blas_thread_caps.** Measured on a 12-logical-processor box, private commit added by `import numpy`: unset 397.4 MB, 1 thread 44.1, 2 76.1, 4 140.4, 8 268.8, 12 397.5 (private ~= 44.1 + 32.1 * (threads - 1)); working set flat at ~16 MB. Set before numpy 43.9 MB; gymnasium first then set 403.3; never set 403.3. Both trainers imported gymnasium before python_ai. At num_envs=8 that was nine processes paying ~353 MB each. torch costs ~166 MB at import and ~4.4 MB per OMP thread; the main process spends ~87% of wall clock in the update.
- **test_human_prior.** Reporting the deck from `prior_cards()` while disabled made the coverage draw [5, 5, 5, 5] instead of the old [1, 5, 1, 1], confounding the control arm. The quiet-board rule is backed by the measured 2026-08-14 collapse.
- **test_rl_gae.** The "strict, provable generalization" claim was a comment in `train_selfplay.py` that nothing checked. `explained_variance` guarded its degenerate case while `normalize` did not; `run_update` filtered by `valid` for explained variance and the clip range but not for the advantage normaliser.
- **test_advisor_kl_gradient_safety.** With the old `where` form, grad_a = 0 * nan = NaN and grad_b = 0; it stayed harmless only because the target was a constant and no row was all `-inf`, and a fully masked row put NaN into the trainable `new_logits.grad` while the forward read 0.0.
- **test_shaping.** The reset guard lived in the callers (`base_trainer` and `exploiter` multiplied by `(1 - prev_dones)`); the exploiter shipped without it through burst #0. Moving it into `compute_shaping` changed no win rate, since both live callers already zeroed those steps.
- **test_deck_pool.** 2026-09-06: the floor parked six of sixteen decks at 0.84% of episodes each while the two decks the teacher could not pilot took 53%. Until 2026-09-15 the no-signal branch was easiest-first and, with the production gate at 0.0, unreachable.
- **test_shaping_gamma_is_not_a_second_copy.** `compute_shaping` and `solvency_shaping` both defaulted gamma to 0.99; `tools/validate_pipeline.py` and ~10 test call sites relied on it. Deriving it from PPOConfig was tried first and is forbidden by the layering rule. The sixth instance of the no-second-copies rule, and the first where the copy was a function default (so greps for restated constants missed it).
- **test_match_outcome_is_the_only_scorer.** Seven instances in three waves: 2026-08-19 net_ab, net_h2h, net_h2h_search x2; the same day tools/validate_pipeline.py (where equal-count finishes fell into `draws` and absorbed the signal the side null exists to detect); 2026-08-20 eval/prove_combos.py, prove_teacher.py, prove_environment.py. web/viewer.html scored timed-out matches from "both King Towers alive?" until 2026-08-21 ("Draw. Timeout - both King Towers still standing" on a 3-3 with a Princess at 90 HP). UPSTREAM_REQUESTS.md item 16 proposed binding TimeoutRules directly (since done: `resolve_timeout_outcome`).
- **test_teacher_deck_generalisation.** Measured 2026-09-06 against a do-nothing opponent (tower damage / ticks / elixir spent / the named card's spend): hog_26_mirror 8515 / 718 / 23.4 / Hog 6.0; xbow_30_cycle 2486 / 3600 / 6.0 / X-Bow 0.0; graveyard_control 2407 / 3600 / 6.1 / Graveyard 0.0; mortar_cycle 4683 / 1769 / 15.0 / Mortar 0.0; classic_log_bait 5405 / 1734 / 7.9 / Goblin Barrel 0.0. 34 of a siege building's 170 legal cells damage the tower. A Goblin Barrel is worth 1320 on the tower vs 600 in our own half; on a quiet board the spell rule proposed cell (0, 0). 2026-09-15 resolver, 300-tick damage: Miner 1746 vs Wall Breakers 700, Balloon 2534 vs Lava Hound 901. get_tower_damage_dealt read 1620 on an empty board.
- **test_reward_horizon_invariant.** weights.py set W_TOWER_DESTROYED = 0.6 "above the discounted value of a win (~0.28 at these episode lengths)", measured at ~112 decisions per episode; the 2026-08-07 speed fix tripled match length. Measured 2026-08-27, `model_weights_phase1_v5.pth`, 10 greedy episodes: mean 297 decisions (median 360), terminal contribution to the discounted return 0.0195, a win discounted to 0.99^360 = 0.0268, a crown worth 22.4x a win, ~98% of the objective shaping. A sacrifice play cost 0.5 at once and repaid 0.027.
- **test_phase2_truncation_end_to_end.** Before it, Phase2Trainer was covered only by declared-attribute checks; the truncation fields had never travelled the whole path, and phase 1 returns early from `_truncation_bootstrap`, so the live smoke run could not reach them either. A first draft used update_timestep=4 and silently skipped the negative control; passing "scenario_prob" in env_config was silently ignored.
- **test_expert_capacity_and_temperature.** Measured over 100 decisions with >= 2 candidates: median 77, p90 139, max 150 candidates; 87.0% of rows over K_MAX=8. Value spread full set mean 0.5786 / median 0.6186, top-8 mean 0.2111 / median 0.1055; overflow rows 0.6371 -> 0.1036 (84% compressed). Target entropy fraction, full vs top-8: T=0.02 0.288 vs 0.538; T=0.05 0.592 vs 0.858; T=0.25 0.959 vs 0.994. Before the cap the worst case was 1 + k_cards * 153 = 460; `max_candidates` returned 7 while search emitted up to 150. Appending the argmax after the cap gave 147 against an analytic bound of 145.
- **test_rl_seeding.** Before seeding, pipeline 2 drew from `np.random.default_rng()` (unseeded), `random.choice` and `np.random.choice`, plus torch init and the minibatch permutation; there was no seed field or CLASH_SEED. bc_pretrain seeded its data collection and then shuffled unseeded.
- **test_elixir_shaping.** Split from the monolithic `test_python_ai.py` 2026-08-20. `bankruptcy_rate`'s docstring justified floor=3.0 as the cheapest card of the Giant deck; corrected 2026-08-27. The 65.3% baseline is quoted against it.
- **test_rl_entropy.** Seven recorded instances of an entropy normaliser that did not hold in the regime it was measured in, three in this controller. Card entropy 0.10 once collapsed the policy to 5 of 8 cards. Under the old log(total-arms) divisor a fresh policy read 0.413 and the controller pushed up (prediction 4 of the 2026-08-16 fix). 0.002 was measurably an off switch. 2026-08-11: gain 0.5 against an error of 0.16 multiplied the coefficient by 1.083 per update, ~55x over 50 updates, reaching 0.433 and dissolving the policy. Until 2026-08-09 pipeline 2 passed the raw episode counter, making the stall re-boost dead code (it fired twice at a pool win rate of 0.49 and changed nothing). 2026-07-30: a phase-2 resume reset placement from a converged 0.0132 to 0.06, ~5,600 episodes to walk back.
- **test_placement_coverage.** Split from the monolithic `test_python_ai.py` 2026-08-20. The coverage hole was the root cause of the (11,0) placement collapse; PLACEMENT_COLLAPSE.md has the behavioural measurements.
- **test_deck_coverage_penalty.** Across the 2026-08-28 phase-1 run the card entropy target held at 0.339-0.350 over 24,000 episodes while the policy used five of eight cards and no-oped ~81% of steps. P(play | in hand) at ep 32,484: Skeletons 0.3429, Ice Spirit 0.3247, Ice Golem 0.1868, Hog Rider 0.0704, Musketeer 0.0662, The Log 0.0091, Cannon 0.0053, Fireball 0.0011; the floor 0.02 is 3.3x below the weakest live card and 2.2x above the strongest dead one. The first (linear) hinge's gradient at Fireball's p = 0.0011 was 0.00027, 17x weaker than at p = 0.019; three paired 18-minute arms on the ep-32,484 checkpoint moved MinCardProb -0.0007 / +0.0018 / +0.0004 at coef 0 / 2 / 8.
- **test_bptt_credit_horizon.** bptt_chunk was 25 before 2026-08-27, shorter than the 30-decision rotation, just as the observation gained `seen[]`/`recency[]` (item 24). Doubling it measured 1.090x (the flat 500 rows through the trunk are 84% of the update; the LSTM loop, 25 calls at batch 20 -> 50 at batch 10, is 16%); segments per minibatch 20 -> 10, optimizer steps per rollout unchanged at 32. The previous calibrated-once constant, W_TOWER_DESTROYED, drifted 22x wrong when match length moved.
- **test_package_layout.** shipping.py once imported a 1,152-line experiment harness (and through it bc_pretrain, torch and a card-registry probe) to reach one dataclass. Five modules imported `_build_candidates` / `_search_action` from search_ab_test. bc_pretrain.py lacked the script bootstrap and could not be run. The arena rule's count of stale copies moved from four to six (the observation encoder's channel 8, web/viewer.html); the 2026-08-21 re-centring moved the bridges 3.0/14.0 -> 2.5/14.5. deck.py joined the root on 2026-09-15. scenario_offense's `BRIDGE_XS` read (4.0, 14.0), water under the new arena; it was default-off, so no run was harmed.
- **test_damage_spell_is_deck_derived.** TODO 00.3 (2026-09-15 audit): W_LETHAL_SPELL and W_SPELL_VALUE_START were keyed to Fireball. Measured 2026-09-16 over 12 seeded matches through the trainer's reward path: a Rocket-for-Fireball Hog deck had both terms at exactly zero on all 2,372 steps. A fixed 40-tick probe read Poison 368 of 736 (and its docstring quoted the 368 as proof it matched the registry) and Goblin Curse 129 of 258. A hardcoded 4.0 denominator paid a Rocket +0.5 for an even trade. The first draft read `stats.get("spell_damage", FIREBALL_DAMAGE)`, and had Rocket as 32 (Poison), Zap as 12 (Skeleton Army), Arrows as 8 (Barbarians). The end-to-end 2.6 control (old vs new code, bit for bit) is recorded in the commit.
- **test_entropy_and_defense.** The 2026-08-16 audit, the two defects the from-scratch 2.6 run was gated on. Measured on model_weights_cured.pth over 706 decision steps: 54.1% left exactly two legal arms, where a 0.35 target against log(5) is 0.5633 nats, 81.3% of what log(2) = 0.693 can carry; mean elixir 2.25/10, nothing affordable on 73.9% of steps (78.5% under a big push), P(play) flat against threat (0.1008 none vs 0.1016 largest). A uniform 2-arm row read ~0.43 under the old divisor. Policy-invariant tower shaping let win-condition usage decay to 0.7%. The Giant deck's Giant was never played across four full runs. A test once injected a Musketeer under a comment saying Archers. At ep 6,053 Hog usage was 0.8% while the agent won ~100% at 1.0x by defending.
- **test_rl_curriculum.** The curriculum was untestable inside `train_ppo()`; the suite parsed train.py with ast to recover CURRICULUM_STAGES. The ladder was strictly one-way; 0-for-2000+ episodes at the old 1.75x step. Measured 2026-08-26: with the 0.80 gate re-tested every episode, an agent of true skill 0.70 had 1.6% chance per window and 91.6% within 3000 episodes, an effective gate of ~0.70. The random-deck ladder already had its `max_episodes_per_deck` valve.
- **test_search_and_checkpointing.** SearchCfg lived in a 1,152-line experiment harness. `max_candidates == 7` was asserted until 2026-09-03 while search emitted up to 150. The config's allowed stdlib widened from {"dataclasses"} on 2026-09-03 for `os`. Search vs greedy, paired and seeded, rung 3 on the 16-deck pool, ep-111k (eval/search_vs_greedy_pool_ab.py), greedy 0.844: search 0.531 / 0.312 / 0.375 at horizons 4 / 8 / 12; on shipping's own config -0.433 [-0.633, -0.233], p = 0.00098.
- **test_net_dilated_context.** Phase 4 bottleneck 1 (2026-08-27). The first attempt used dilations (2, 4) and cost 1.533x; the shipped (2, 2) reaches 10 + 16 + 16 = 42 >= 34 rows (see CLAUDE.md). The LSTM is 1,804,288 of ~1.9M parameters.
- **test_engine_seeding.** The original item-7 proposal seeded only `ClashEnv::rng`, which would have left the hand random. 2026-08-20: a combo-family ablation expected the untreated arm to reproduce at the same --seed and got 0.475 vs 0.537 (--seed reached only the teachers' RNG). The test was written 2026-08-20 against an engine without `seed`, gated on hasattr, and went green 2026-08-21 when the rebuilt .pyd landed; three sessions believed the machine had no toolchain because it is not on PATH. Item 23C (sample_random_deck's static) chose option 2 (a private generator per seeded call) over option 1 (seeding the static), which could only reproduce for a fixed construction order.
- **test_net_trunk_receptive_field.** Measured 2026-08-27. With a zeros input the base field read 7x7 instead of 9x10. fwd+bwd over one 200-sample BPTT minibatch: current 21->16->32 (7,680 params, LSTM in 1440, RF 10) 85.4 ms 1.00x; wider 21->32->64 (24,576, 2880, 10) 194.9 ms 2.28x; deeper 21->32->32->64 (33,824, 2880, 14) 218.7 ms 2.56x; dilated (7,680, 1440, 14) 91.3 ms 1.07x. After the context blocks landed this file's assertions kept passing across the change that invalidated its docstring, until `context_dilations=()` was made explicit.
- **test_scalar_encoder_branches.** Phase 4 bottleneck 2 (2026-08-27). The monolithic layer had 72,000 parameters; the branched one 4,850. The shared card projection costs 2,960 parameters against a dense layer's 8,880. The cycle branch was detached 2026-08-28 after measuring `model_weights_phase4.pth` at ep ~7,200: PPO supplied 99.6% of the gradient on these parameters and spent it on a 1-D opponent-tempo readout, leaving the branch decoding the opponent's next card worse than at init (+0.169 lift vs +0.195; the observation itself carries +0.412). The eight deck columns of `cycle_card.weight` had collapsed toward one direction: mean pairwise |cos| 0.191 +- 0.017 -> 0.461, effective rank 7.48 -> 6.01.
- **test_rl_ppo.** PPOUpdater replaced ~250 lines duplicated verbatim between the trainers. 2026-08-11 placement-entropy defect: reported 0.462 was 0.850 on no-op steps against 0.090 on real placements, against a 0.25 target. The forced-step inflation was ~3.7x (all steps / decision steps). Measured on model_weights_cured.pth: 54.1% of decisions left exactly two legal card arms; downstream, elixir spent on sight, mean 2.25/10, nothing affordable on 73.9% of steps, P(play) flat against threat (0.1008 -> 0.1016). Seven incidents trace to the normaliser. 2026-08-26: 100% of parameters NaN after one bad step and still after a clean one.
- **test_curriculum_plateau.** runs/phase4/train-20260828-rolling-spells.log: episodes 9,640 -> 32,680 (23,040 episodes, ~24 h at 943 ep/h) at stage 3, win rate 0.15-0.60, zero advances; the ladder then had only a 0.80 mastery gate and a 0.10 catastrophe valve. The first backstop used elapsed time at the rung and would have cut off a climbing agent. 2026-09-06, live phase-9 run ep 83,128 -> 87,540: the unweighted per-deck mean rose 0.301 -> 0.534 on all sixteen decks while the readable rate sat at ~0.50, and the valve fired twice. The same day the agent averaged 0.62 per deck (beating 14 of 16) while the PFSP-weighted rate read 0.29, and the backstop demoted rung 3 -> 2 -> 1. Moving both floors onto the progress signal then hid real over-promotion, measured twice at rung 3 -> 4 (horizon 20 -> 30 ticks): 0.535 -> 0.522 over 600 episodes, and 0.637 -> 0.583 over 1,800 (worst deck 0.12 -> 0.07), both caught by hand.
- **test_checkpoint_paths.** `Phase1Trainer.__init__` read `os.environ.get("CLASH_WEIGHTS", "model_weights.pth")`; launched from the repo root a run started fresh, from python_ai/ it resumed. Measured 2026-08-25: model_weights.pth carried episode 7,063, phase random_opponent, stage 4 and a decayed entropy coefficient from an engine two gameplay changes old. `log_dir` was also cwd-relative and `shutil.rmtree`d on a non-resume start. The package-wide scan found seven more bare `torch.save` calls in trainers/, including exploiter.py's burst snapshot into the shared pool. The checkpoint payload is ~3x the parameter count (model + Adam's two moments). `eval/probe_card_usage.py` had described the trainer's writes as "atomically-enough", true only of the uniquely named snapshots.
- **test_tactics.** Split from the monolithic `test_python_ai.py` 2026-08-20. The river moved 2026-07-29 ([16,18) -> [15.5,17.5)) and the towers 2026-07-30; both times a stale Python copy survived (models/net.py's '18*16=288' comment, calibrate.py scoring bridges at y=17.0). The bridge/tower block regex-scraped GameManager.h / Board.h until 2026-08-21, when ArenaLayout.h was bound and the initialisers stopped containing literals. The ungated A/B lost because the agent already sat under 3 elixir 65% of the time. The Cannon rate-limit test skipped on a fraction of runs when the Cannon was not dealt (skip count drifting 2-3) until the hand was set explicitly. The Giant bridge rule scored 535.6 enemy tower damage against 3.3 for the policy's own cell over 913 states. A flat solvency reserve cost 4,645 tower damage dealt per episode.
- **test_teacher.** The utility teacher replaced the elixir-multiplier curriculum on 2026-08-19 (the 1.5x handicap priced the win condition negatively, monotone across 1.0/1.25/1.5x, and four Hog interventions returned null). The side-agnostic test tripped its own >= 5 floor occasionally until seeded (2026-08-21). Two engine-driven CycleTracker tests passed vacuously and one stayed flaky. TEACHER_STAGES went 6 -> 11 rungs 2026-09-03; `max_combos` was added 2026-08-20. CURRICULUM_STAGES was a local of `train_ppo()` and tests parsed train.py with ast. The old multiplier scan was line-based and skipped anything starting with `#`, exempting docstrings.
- **test_rl_base_trainer.** Written with the BaseTrainer extraction, whose stated risk was finding a wiring bug three days into a 72-hour run. The `workdir` fixture was a bare chdir until checkpoints were anchored on 2026-08-25; the first suite run afterwards wrote a random-init 20-step checkpoint into the live `python_ai/model_weights.pth`. `replays/` was anchored later via `run_path`. The final-save test pins a latent bug: `train.py` had two hand-written `torch.save` blocks and the final one dropped `ent_coef_card`/`ent_coef_place`, the same failure observed 2026-07-30 in the other pipeline (placement reset 0.0132 -> 0.06, ~5,600 episodes to walk back). `_truncation_bootstrap` used to classify terminals by `abs(raw_reward) > 0.5`, which bootstrapped exact-tie timeouts as truncations and made classification depend on the reward scale. Phase 1 gained scenario injection 2026-08-29 because 32,680 episodes without a "defend or lose the tower" moment left Cannon/Log/Fireball at P(play|in hand) 0.0053/0.0091/0.0011. Audit 08 gap 2: `except RuntimeError` on resume moved the checkpoint to .bak and rmtree'd the TensorBoard log, then continued half-restored. Gap 6: Ctrl-C lost up to CLASH_SAVE_EVERY episodes. Champion ability training landed 2026-09-16; before it a Champion deck raised at setup.
- **test_advisor_target.** Split from `test_python_ai.py` 2026-08-20. Both changes are from 2026-08-14 and gameplay-affecting: `spell_value_weight` was dead code (neither trainer passed `w_spell`, so the weight sat at START for all of training), and the coverage term was pure entropy, which flattens exactly what distillation builds. `model_weights_selfplay.pth` was at episode 64,309 against a 40,000-episode anneal horizon, hence the start offset. The Cannon's tie plateau (handoff 2.3): temperature could not push its target below ~66% of max entropy while Fireball reached ~41%. `_hog_commit_obs` writes the elixir scalar directly partly because `set_elixir_for_team` was missing from stale .pyd copies. The Fireball-hand fixture list passed alone and failed in the full suite because the global RNG position differed.
- **test_teacher_reactive.** `rollout_stats` rolled every candidate with both sides no-oping. Against a full stage-5 teacher as ground truth over 150 naked bridge pushes: tower damage predicted 587.5 vs truth 139.5 (bias +448), elixir lost 0.09 vs 2.84, "play" 90.7% vs 7.3%, i.e. 125 false GO and 0 false HOLD. P(same | truth plays) stayed 28.3% for every responder; false GO fell 27 -> 4. Four responders, paired teacher-vs-teacher, 150 openings, control noop-vs-noop = 0.5000: scripted open-loop 0.91x cost +0.1583 [+0.1033, +0.2117]; reflex stride 50 1.95x +0.2050; stride 30 2.63x +0.2450; stride 10 5.50x +0.2217 (it dumped four cards on a single Hog). Arm-vs-arm all six comparisons null (p 0.086-0.832, ~60 of 150 openings tie), so the cheapest won. The ladder was first [F,F,T,T,T,T] on the cold-start argument alone; the binding constraint was the horizon. Passive-opponent sweep, 20 openings, share of decisions landing a card OFF/ON (froze): h30 12.2/9.8% (3/20), h50 11.6/10.1% (3/20), h70 12.5/11.3% (1/20), h100 11.8/12.3% (0/20). The +0.1500 win rate was measured at h100. The table went 6 -> 11 rungs 2026-09-03.
- **test_rl_ppo_compaction.** Placement head ~41% of update time. On model_weights_selfplay.pth (ep 31,312) over 1,500 steps decision fired on 36.8% of rows, so 63.2% of both placement forwards and backwards was multiplied by zero. `Advisor/Rows` sat at 23-25 per 500-row minibatch; slicing coverage to those would re-open the 2026-08-14 collapse (Cannon 91.0% modal share at (11,0), preserving 121 tower HP vs 396 for a random legal cell). Batch 500 -> 184 with zero upstream gradient on dropped rows: Linear(280->32) fwd 4.8e-07 grad_W 2.3e-05 (fixed by computing place_ctx on the full batch and slicing, ~0.5% of head cost); Conv2d place_up.1 (18x10) bit-identical; place_up.4 (36x20) 5.8e-03; place_up.6 2.8e-03; place_hires.0 (34x18) 4.9e-03. The first probe tested only 18x10 and wrongly concluded invariance. Measured weight delta after one step 1.2e-10 relative. P(nothing affordable) was once measured at 73.9%. Landing compaction was a one-time trajectory discontinuity.
- **test_placement_hires.** Split from `test_python_ai.py` 2026-08-20. The handoff diagnosed the coarse head as unable to express an exact cell (distill_tactics CE fell 180.9 -> 21.4 while exact-cell argmax stayed 0.0%); tested directly, both arms fit 14/14 on the single-column task, so the claim was weakened to "a resolution increase to be measured at scale" (prove_hires.py). The handoff proposed concatenating into `place_up`, which would have discarded every checkpoint's placement head as the 2026-08-09 checkerboard fix did. The cycle skip entered the placement context 2026-08-28. The column fixture read `DEFAULT_DECK[1]` as "Archers" until the 2026-08-16 switch to 2.6 made it a Musketeer (one body vs two), and the head fit 12/14, reported as a false architecture regression. The random-state version had 8 states collapse to 7 distinct spatial maps. A test lighting row 33 passed with ceil_mode off (shape (32,8,4), no dead row) and was replaced.
- **test_aux_task_is_not_a_memory_probe.** CLAUDE.md listed `Aux/OppElixir_MAE` "below ~1.3 means the recurrent state genuinely counts" and defended the head's 0.77-0.83 after ruling out "read your own elixir" (ratio 0.96-1.05, corr 0.27). Measured 2026-08-27 on 2,606 samples / 8 random episodes: predict-the-mean 1.4271, time only 1.4223, opp spent only 1.4268, time + opp spent OLS 0.0000. The offsets read from the END (`-N_EXTRA + k`) until 2026-08-27; the item-24 cycle blocks appended after the extra scalars would have re-pointed them (UPSTREAM item 25). The 2026-09-02 elixir phases (item 26) broke the single-slope basis (1.4265) and made the opponent overflow unaided: per episode, never-capped residual -0.035, capped +0.76 / +1.63 / +5.38; 0.3% of samples sat at the cap while 3 of 6 episodes were affected. The head was deleted 2026-08-28 for `Aux/NextCard_CE`. The overflow test fished random matches until TODO 0c closed 2026-09-23: it skipped ~half the time and failed ~15% (3 of 20) because touching the cap counted as contamination (contaminated MAE 0.007 vs clean 0.0078). The slot cooldown (20 ticks) was found by the play verification.
- **test_teacher_combos.** TODO item 1. Deploy time (2026-08-19, DEPLOY_TIME_TICKS = 10): lone win condition -556.3 tower HP, supported push +448.5 [+137.3, +760.1], escort in a punish window +650 [+429, +878]; prove_environment's strategy arm read attack 0.490 vs cycle 0.715. The 0-tick fact was measured 2026-08-20. `_env` used a bare `reset()` and became a ~1-in-3 flake once reactive rollouts tightened scores; seed 0 chooses a combo in 33 of the first 40 seeds. Stage-5 teacher-vs-teacher, ~2,400 decisions: elixir mean 1.79, p90 3.30, only 2 states with any combo affordable (both at the 5.00 opening bar). A flat savings charge (reserves 0/1.5/3/5) moved the mean bar 1.79 -> 1.73. Combos were chosen ~2 per 950 decisions (bar reaching the price on ~1.4%), later 0.28% of decisions. `combo_families` exists because two of three win-rate A/B runs pointed negative. play_margin: marginal cheap plays scored a median 1.37 (2026-08-20); 3.0 vs shipped 0.969 [0.917, 1.000], 4.0 the same 0.969 with overflow 4.3% -> 12.9%. Against a passive opponent a fixed 3.0 froze: 14 plays in 6 matches, elixir 9.56, tower damage 9143 -> 4063. The taper was anchored at ELIXIR_OVERFLOW_AT until replay_ep2018.json (2026-08-28, stage 1): Princess Tower lost at tick 159 with 2 elixir spent while the bar ran 5.0 -> 8.6, next play at tick 561 when elixir first reached 9.60.

### The engine and its instruments

#### Engine headers (include/)

- **TargetingHelpers.h.** `findHpExtremeEnemy` excluded only `isTower()` until the 2026-08-26 audit; a Cannon was usually the highest-HP thing inside Hero Giant's 3-tile grab radius, so Hurl mostly threw a stationary building into the other lane. `isBuilding()` is strictly wider.
- **RangedBuildingTargeter.h.** lineSplash/lineSplashRange were once dropped from the projectile constructor; latent because Royal Giant, the only card of this archetype, sets neither.
- **CursedHogOnHit.h.** Before the 2026-08-26 latch, three curse applications produced three hogs on death.
- **Troop.h.** The movement dead zone and getNextWaypoint's arrival test were two independent 0.01f literals and deadlocked troops at the bridge mouths for 10+ s after the 2026-08-07 speed fix. Movement read freezeTicks after update() decremented it, so the final tick of every freeze moved at full speed (fixed 2026-08-26).
- **StatsEvents.h.** Until UPSTREAM item 28 (2026-09-15) the tower-damage collector classified a target as a tower by "not in CardRegistry", which also matched spawned helper bodies (ids -1, -10 .. -48); a tower shooting one Goblin Barrel's goblins booked 810 as TOWER damage, feeding the agent's tower potential and the teacher's rollout.
- **TerminalRenderer.h.** `riverRow` replaced a hardcoded `(x >= 3 && x <= 5) || (x >= 13 && x <= 15)`, the eighth stale arena copy found: bridges at {3,4,5,13,14,15} vs the real {2,3,14,15}, four of eighteen columns wrong, and three tiles wide (the pre-2026-08-21 shape). Unlike web/viewer.html it could always derive, since it holds a `const Board&`.
- **Building.h.** Before 2026-08-26 nothing consulted the clock: a Cannon (824 hp / 300 ticks) decayed 27/s, sat on 14 hp at 30.0 s and died at 31.0 s. The only test used hp 3000 / lifetime 300 (divisible) while being named "fully decays to 0 exactly at its configured lifetime".
- **BuildingTargeter.h.** The in-sight test was an `else if` after the tower branch, so every non-tower building beat every tower at any distance. replays/hog_test_1.json (2026-08-28): the Hog turned toward a Cannon 9.160 away while 7.714 from a Princess Tower and stayed wrong for seven ticks. The old comment described the two tiers as a deliberate model of the real game; it was not.
- **ArenaLayout.h.** Before this header the geometry lived in Board's bridge members, GameManager::setupTowers' six literals and HeuristicOpponent's LEFT/RIGHT_BRIDGE_X (already stale at 3.5/13.5). Corrected 2026-08-21 from a professional player's audit: King at 9.0 and left Princess at 4.0 were symmetric about 9.0, the same half-tile error the river had; the 2026-07-30 recording fit was anchored on it. Centring bridges on a tile made them three columns wide.
- **TowerTroops.h.** A flying troop parked in range once took zero damage from a Princess Tower, because applyCardMetadata reset the constructor's `targetsAir = true` to CardStats' false. Tower Princess is ~26% more hp than the default and is opt-in, so no existing run is affected.
- **TimeoutRules.h.** Before this class every timed-out match scored as a draw (calculateReward returned 0.0 while isGameOver was false). `decide` was split out because consumers that could not call `resolve` kept re-deriving the verdict wrongly: eight times across three waves in Python, and web/viewer.html still reported "Draw" on 2026-08-21 for a match with a Princess Tower at 90 hp. Absolute-hp tiebreak was specified by the 2026-08-21 audit. The census moved to MatchRules on 2026-09-06 for the overtime rule; `isTower()` replaced a dynamic_cast.
- **Tower.h.** The King fired from tick 0 until 2026-08-21 (UPSTREAM item 3); the 2026-08-20 sight fix had widened its reach from 7.0 to 9.4 tiles and raised its share of damage against a lone Hog from 5.3% to 36.8%. The targeting radius 3.3 was fitted to observed Hog/Cannon play; centre-to-centre diverted at 6, 7 and 8 tiles.
- **MatchRules.h.** `evaluate` once returned team 0 first and never noticed a simultaneous double KO. It checked only `symbol == 'R'` until 2026-08-26, so a living Mortar (card 93, also 'R') kept a fallen King's match running into a timeout decided by TimeoutRules. Rules 2 and 3 (crown lead at 3:00, sudden death) arrived 2026-09-06; mean match end had been tick 1758, so many matches previously ran to 3600 and are now decided at 1800. No win rate is comparable across that date.
- **Projectile.h.** `target` is the only entity-pointer member in the hierarchy; an implicit copy would home on and damage the original board's entity. Found while implementing Board::deepCopy (2026-08-11).
- **HeuristicOpponent.h.** The original opponentTurn() ran every tick and, once it held 4 elixir, played a random affordable card at a random x in [2,15] with y fixed at 25. After ~47,000 episodes against it the policy won 100% at 1.0x using six of eight cards, never its win condition or spell (13-0-2 against an attacking scripted bot in a probe): an opponent that walks units into a defended lane on a timer makes pure defence optimal. Pending spells (hp 1 at the impact point) and projectiles used to count as incursions, so a Fireball aimed at the bot's tower bought a full defensive placement. The bridge constants were 3.5/13.5 against Board's 4.0/14.0, half a tile off for as long as they existed.
- **LanePath.h.** Rule B of the 2026-08-21 player audit. The old blind fallback was the nearest enemy tower by raw distance; with team 1's left Princess dead, a left-lane unit at (2.5, 12.0) chose the right Princess (18.90 vs King 19.45), at (2.5, 14.0) likewise (17.36 vs 17.56), tied at (2.5, 15.0), and was right by accident at the bridge mouth (15.57 vs 15.23), which is why the regression tests anchor further back.
- **StatsCollectors.h.** The tower/deployed-building split landed 2026-08-06. With W_BLDG = 0.5 and MAX_BUILDING_HP = 4008, losing a Cannon's 824 hp cost 0.1028 of shaping while killing a troop paid 0.1 * hp / 4256, so a Cannon had to kill 5.3x its own hp to break even; 27.9% of Cannons went to (11,2)/(11,3) behind the King, mean y = 6.3. Until 2026-09-15 a target not in CardRegistry was classed as a tower, so a tower shooting spawned bodies booked TOWER damage for its owner: 810 for one Goblin Barrel, 1080 for a Graveyard, 1440 for a Battle Ram's Barbarians. The elixir-value collector prices in cost so a Fireball dealing 689 to a 230-hp Minion does not look three times better than one dealing exactly 230.
- **GameLogger.h.** `escapeChar` once wrapped the char unescaped; Ram Rider's '"' symbol corrupted every replay containing one, and then every replay once cardMeta emitted per-card symbols. The replay `name` field answers UPSTREAM item 20, whose original diagnosis ("spawned entities carry no cardId") was wrong; 36 of the registry's 90 symbols are ambiguous. web/viewer.html derived the result from "are both Kings alive?" and called everything else a draw; its comment said it "mirrors MatchRules::evaluate exactly", which was true and was the bug. A 3-3 match with one Princess at 90 hp showed as "Draw. Timeout - both King Towers still standing". The viewer painted four of eighteen columns as the wrong terrain after the 2026-08-21 arena re-centring, which is why geometry goes in the replay format.
- **AreaSpell.h.** The rolling sweep landed 2026-08-28; before it The Log and Barbarian Barrel were static circles detonating at the tap point, with the 3.9 / 2.5 widths read as radii (a 7.8-wide Log). Snowstorm's tower multiplier was the first "reduced damage vs buildings" spell mechanism; Earthquake's real 3.5x vs buildings is not modelled. UPSTREAM item 29 proposes setting this multiplier per spell from the real game's Crown Tower damage.
- **Entity.h.** The 2026-08-26 audit found `pullToward` running backwards on a negative distance: a Golden Knight 0.5 tiles from his target finished his dash at 1.0, ten dashes over. `mirrorToOppositeLane` exists because Mighty Miner and Hero Giant wrote `position.x = width - 1 - x` directly, skipping the building guard; Hero Giant threw a Cannon from x = 6.0 to 11.0. `snapshot()` throws because a nullptr default would silently drop every tower, building, spell and projectile from a copied board (UPSTREAM item 13, 2026-08-11). `pushAlong` was added 2026-08-28 for The Log's lateral throw.
- **Board.h.** The river was [16, 18), centred on 17.0, until 2026-07-29: half a tile off-centre, so team 0 could place on row 15 and team 1 could not in its mirrored frame (15/20 vs 0/20 placements). Bridges were 4.0/14.0 until 2026-08-20/21, symmetric about 9.0 not 8.5, and three tiles wide. Until 2026-08-21 the observation encoder painted bridges from a hardcoded `(x >= 3 && x <= 4) || (x >= 13 && x <= 14)` while the physics used the Board members, so after the correction the network saw columns 2 and 15 as water and 4 and 13 as bridge; a shared function (isOnBridge) replaced the shared formula. `pushAwayFrom`'s coincident branch set dist = 1.0 to normalise and then pushed by minDist - 1.0: a troop dead-centre on a Building (minDist 1.4) moved 0.403 and stayed inside, on a King (2.4) it moved 1.4. The colliders cache and the resolveCollisions hoist are from the 2026-08-26 perf pass (~1,700 virtual dispatches per call at 30 entities down to 60). getNextWaypoint's two absorbing states: the near-bank one after the 2026-08-07 speed fix (2 events in 41,954 unit-ticks at (4.00, 15.50) and (14.01, 15.49)), and the exit-bank one found 2026-08-20, which left 6-8 of every 34 lone ground units uncrossed (Giant 27/34, Musketeer 26/34), frozen for 750-850 ticks.
- **CardStats.h.** Troop movement was 4-5x too fast until 2026-08-07 (per-card footage 3.8-6.5x, a time-scale sweep peaking at 0.2-0.25, and the engine's Slow:Medium ratio 0.60 vs the real 0.75); MOVEMENT_SPEED_SCALE = 0.2 fixed it, first leaving Slow ~20% slow. The speed tiers (2026-08-24) came from Supercell's exported table (five values across 119 characters, UPSTREAM item 25); Giant/Golem/P.E.K.K.A/Royal Giant/Lava Hound were all real Slow but spread across 0.4/0.6/0.8/1.0 tiles/s. The two calibration cards gave 0.02193 and 0.02226 tiles/s per stat unit, and the footage reproduced the published Fast:Slow ratio (2.03 vs 2.00). Deploy time (2026-08-19) was measured before it was added: committing the win condition scored -0.330 win rate (n=100 paired, p=5.7e-08) and -298.2 tower HP marginally (n=220), the defence answering a 4-elixir commit for ~1.2 elixir; unopposed the Hog dealt 2536 tower damage, and tightening the commit timing made things worse, pointing at defender tempo.
- **GameManager.h.** The elixir phases landed 2026-09-02 (UPSTREAM item 26), after 4,229 decisions of model_weights_phase4.pth at stage 3 showed both sides starved (mean 2.16 and 1.81 elixir, P(>= 9.0) = 0.0%) and the best possible Fireball catching a median of one unit (>= 3 only 15.5%). The state-estimator setters (setElixir/setHand, 2026-08-17; towers and clock, item 22, 2026-08-24) exist because forecast.py rebuilt boards via reset() with a fabricated hand and elixir, and sim_driver.py reverse-engineered the shuffle to recover only the opening hand; perception reads elixir at mean confidence 0.990 over 587 samples. Card-cycle tracking arrived 2026-08-27 (item 24). Rolling-spell placement bound (2026-08-29): cast at y = 15.0 reaches 25.1 for 0 tower damage, at y = 16.0 reaches 26.1 for 269; the ep-32,484 policy put 53.4% of its Logs on the enemy half over 60 episodes. The footprint edge check (2026-08-28): a Cannon was accepted at x = 0.0 spanning [-1, 1] and at 17.0 spanning [16, 18]. activateChampionAbility's uses check: Boss Bandit's nonzero cooldown had masked a uses-limited ability charging elixir for no-op activations (Hero Mini P.E.K.K.A has cooldown 0). The tower x-coordinates moved 2026-07-30 (left Princess 3.0 -> 4.0, King 8.5 -> 9.0, from a recording homography: max error 0.63 -> 0.31 tiles) and back on 2026-08-21, when that fit was found anchored on the half-tile convention error.
- **ClashEnv.h.** The attribute channels 9-20 arrived 2026-07-29: isFlying/targetsAir existed but were visible nowhere in the observation, so no air/ground counterplay was learnable. The extra scalars came later (time: tick 100 and tick 3500 with the same board were the same input). MAX_MATCH_ELIXIR went 140 -> 280 on 2026-09-02: 140 was sized for flat 1x (3600 * 0.035 + 5 = 131), and phased income (1200 @ 1x = 42, 600 @ 2x = 42, 1800 @ 3x = 189, +5 = 278) would have saturated both spend scalars mid-match. The forward offsets were bound 2026-08-27 after the cycle blocks made five Python sites that subtracted from the end read recency floats as tower HP, one feeding the tower potential (UPSTREAM item 25). The river marker was row 17 for team 0 and 16 for team 1 (36 differing cells on an empty board, item 6). Team 1's observation was one row off every tick (`33 - int(y)`) until 2026-07-31; a policy against a bit-exact copy scored 0.598 as team 0 over 400 episodes (item 5); the Princesses (y = 27.0) mirrored correctly and the Kings (30.5) did not, so a coordinate audit found nothing. Bridge columns were hardcoded `(x >= 3 && x <= 4) || (x >= 13 && x <= 14)` until 2026-08-21. extractObservationForTeam measured 0.069 ms vs a 0.0011 ms physics tick; the reserve() fix and the one-pass tower scalars (formerly ~200 RTTI queries per observation) are from the 2026-08-26 perf pass. The fast self-play path (2026-08-23): 1,000,152 observations built inside rollouts against 51,566 read (19.4 : 1) over 48 stage-5 episodes. isValidPlacementForCard: 58.7% of the ep~45,800 policy's card choices over 1,340 steps were silently refused (item 12). resolveTimeoutOutcome: eight re-derivations across three waves dropped the HP tie-break (item 16). getAllCardIds replaced a hardcoded range(46) pool that went stale at 114 cards. getDamageDealtByCard was added 2026-08-17 for the win-condition damage term. The ClashEnv comments on getElixirForTeam and the estimator setters once pointed at the network's auxiliary opponent-elixir head, deleted 2026-08-28.
- **CombatEntity.h.** frozenThisTick (2026-08-26): update() drained the cooldown before decrementing freezeTicks while moveTowards re-read it after, so applyFreeze(N) slowed attacks N ticks and movement N-1, and a 1-tick stun did not stop movement ("one fact, two readers, a write in between"). Deploy time: the first implementation returned at the very top of update(), freezing poison, shields and Champion cooldowns too; caught by test_game_manager.cpp's champion-cooldown case. Sight: on 2026-08-20 findTarget compared sightRange with centre distance while attacks used effectiveRangeTo; a Princess Tower (reach 9.4, sight 7.5) could not see a Musketeer at 8.0, which destroyed it from 8 tiles taking zero damage (5355 dealt over 300 ticks against 3204 hp). That fix made sight surface-to-surface; on 2026-08-28 it became strict centre-to-centre with the reach floor, after the radius-inflated form gave a Hog 10.9 tiles of aggro against a Cannon. `effectiveRadiusOf` became public static on 2026-08-28 for AreaSpell. The Royal Chef latch: re-feeding compounded `hp += hp / 10`, 1000 -> 1100 -> 1771 over six servings (+77%). The curse latch: N hits nested N composites and N hogs. getTicksOnTarget (item 17) exists because test_snapshot_timing_state.cpp covers the other timing fields through behavioural proxies. The old findTarget comment described a two-tier scan in which a closer tower never stole aggro; towers now compete on distance (see BuildingTargeter).
- **CardRegistry.h.** CARD_ID_COUNT moved from ClashEnv on 2026-08-27 so GameManager could size cycle tracking without a second literal; ids 120-122 once went blind before a bump. Golemite was a raw 0.2f (0.400 tiles/s, below SPEED_VERY_SLOW's 0.663 and 2.5x slower than the Golem) until 2026-08-26. The 2026-08-24 tier pass checked "109 / 109 match" over cards with an official row and missed nineteen child helpers on pre-rework literals that landed on the wrong tier (0.5f = 1.000 tiles/s, 0.6% off SLOW): hut Goblins ran 2.000 vs the card's 2.651, Spear Goblins likewise, Barbarians and Phoenix 1.000 vs 1.325; Night Witch's Bats were 0.85f (1.700) vs card 78's 2.651. Goblin Machine's rocket and Ram Rider's crossbow were an unsourced 50/50 damage split (106 @ 12 ticks, 125 @ 17) before the real splits. The Runner's damage was an unsourced 182 before 175. Ice Wizard once inherited RangedTroop but hit instantly, bypassing the projectile.
- **CardRegistry.h.** Ice Golem carried `.withOnHit(FreezeOnHit(30, 0.65f))`, Ice Wizard's effect copied with a "same story as Ice Wizard" comment that also called the death slow "not modelled"; since it targets buildings, the phantom slow pinned a Crown Tower's or Cannon's freezeTicks for the whole engagement (a standing 35% fire-rate cut, refreshed every 2.5 s hit), fixed 2026-08-26. Ice Spirit was 0.5f (a half slow) and its Evolution's 51-tick window was collapsed to 10, the same day. Bomb Tower (attackRange 6.0) and Three Musketeers sat on the 5.5 sight default, leaving a half-tile band of attack-without-sight, until test_sight_range.cpp's completeness check (2026-08-21/23). Mega Knight's stats were checked against the published table on 2026-09-06 while being blamed for a matchup, which turned out to be a mechanics question. Unsourced guesses later corrected: Dark Prince shield 200 -> 240, Battle Healer heal 60 -> 102, Heal Spirit 110 -> 401, Guards / Royal Recruits shields 65/52 -> 256/240, Cannon Cart "shield" 500 (not a real stat), Berserker's guessed self-heal removed.
- **CardRegistry.h.** Graveyard was `withRepeats(9, 10)` until 2026-09-06: one 81-hp Skeleton per 10 ticks against a Princess Tower firing once per 10 ticks dealt zero tower damage from all 588 legal cells (UPSTREAM item 27). The Ice Spirit Evolution stretched its stun to 51 ticks at 0.5f to stand in for the delayed second pulse; at 0.0f that would be a 5.1 s hard stun off a 1-elixir card, so it was dropped (2026-08-26). Dart Goblin Evolution once lacked withTargetsAir, so evolving it downgraded the card. Mighty Miner's ramp timing was a 15/30 placeholder from Inferno Dragon before the sourced 20/40 (2 s per stage, 2025-01-08 patch; 2.25 s before, 2 s originally in 2023-08-08). Other corrected guesses: Goblin Hut hp 1180 -> 1228 and its spawn mechanism, Goblin Drill death spawn 3 -> 2, Barbarian Hut interval 14 -> 15 s, Archers Evolution range 6.5 -> 6.0, Firecracker Evolution duration 4 s -> 3 s, Skeleton Army Evolution 15 -> 16, Furnace Evolution 3.5 s -> 2.4 s (and a false +180% damage claim), Musketeer Evolution +50% -> +80%, Wizard Evolution shield 300 -> 189, Hunter Evolution interval 5 s -> 8 s, Valkyrie Evolution's invented splash buff removed, P.E.K.K.A. Evolution cap +50% -> +66%, Minion Horde Evolution absorb-shield -> invisibility, Royal Recruits Evolution charge (2.0, 1.5) -> (2.5, 2.0), Tesla Evolution radius 3 -> 6, Barbarians Evolution +30% -> +35%. Mighty Miner was the first Champion; Wall Breakers the pilot Evolution.
- **CardRegistry.h.** The registry ended with a status block recording that all 8 Champions, 4 Tower Troops, Mirror, Spirit Empress and all 41 Evolutions were implemented, Inferno Dragon Evolution last (it needed the ramp grace period and fourth stage). Hero Ice Golem copied Ice Golem's on-attack slow defect and was fixed with it. The Hero pilot pair (Hero Mini P.E.K.K.A., Hero Musketeer) used no new primitives, proving the isHero path before taunt, flight and the rest. The Hero Barbarian Barrel spell lived one tick until the 2026-08-28 rolling sweep (~9 ticks now). Corrected earlier guesses: Executioner Evolution "doubles" -> +75% (was a flat +50%), Giant Snowball Evolution's pull sign, and invented buffs reverted on Mega Knight (+20% damage) and Royal Hogs (hp) Evolutions. countChampions's comment cited GameManager::findChampion, since removed.

#### Bindings and main (src/)

- **bindings.cpp.** Until 2026-08-21 no binding exposed arena geometry, and three copies went stale when the arena was corrected: advisors/tactics.py (King 9.0, Princess 4.0, bridges 4/14), perception/geometry.py and HeuristicOpponent. get_card_info's is_spell fixed an action space that capped target_y at the own half for every card, so Fireball could never cross the river; is_champion replaced a hand-maintained DEFAULT_DECK_ABILITY_SLOTS. sample_random_deck replaced Python "sample, retry until validate_deck_slots passes" loops; the seed argument (item 23C) closed the last source of opponent-deck irreproducibility after item 7 seeded the other two generators.
- **main.cpp.** The tower-HP status line keyed on 'P'/'R' symbols, so a deployed X-Bow counted as a Princess Tower and a Mortar as a King.

#### C++ tests (tests/)

- **test_terminal_renderer.cpp.** Written when TerminalRenderer.h's hardcoded bridge columns were found (the eighth stale arena copy); it had no test, which is how it survived the 2026-08-21 re-centring.
- **test_tower_troops.cpp.** Royal Chef compounding: the same tank was fed every 280 ticks, 1000 -> 1100 -> 1771 over six servings.
- **test_heuristic_opponent.cpp.** The opponent had no coverage before this file.
- **test_troop.cpp.** The river-routing case pinned a hand-computed 10.588172f until the bridge move made it wrong. The freeze case: applyFreeze(N) slowed attacks N ticks and movement N-1, and the registered 3- and 5-tick stuns lost their movement half entirely at N = 1.
- **test_game_logger_entity_names.cpp.** UPSTREAM item 20 reported "spawned entities carry no cardId"; wrong, they carry deliberate negative ids and the right name. 36 of 90 symbols (40%) are shared.
- **test_match_rules.cpp.** The four original cases used a DummyEntity with 'R' as the King and so reproduced the Mortar bug while asserting it correct (fixed 2026-08-26). Regulation/overtime cases arrived 2026-09-06.
- **test_selfplay_fast_step.cpp.** profile_training.py --mode sync, 48 stage-5 episodes: 500,076 chunk calls built 1,000,152 observations against 51,566 read (19.4 : 1, 14.9% of wall clock; UPSTREAM item 21). The first version of the placement case failed with `1.10 < 0.75` (elixir had gone up), revealing the section was a no-op.
- **test_mirror.cpp.** Mirror + Champion was once blocked outright, corrected per game-design feedback.
- **test_arena_layout.cpp.** Before 2026-08-21 channel 8 was painted from a hardcoded `(x >= 3 && x <= 4) || (x >= 13 && x <= 14)`; after the arena correction only the physics moved, so the network saw columns 2 and 15 as water and 4 and 13 as bridge on every observation while every C++ test passed. HeuristicOpponent's 3.5/13.5 bridges were already stale against Board's 4.0/14.0.
- **test_king_activation.cpp.** The dormancy test's first version put a Hog on the far bridge (~15 tiles away, beyond the King's 9.4 reach) and both arms read 1247: anchored where the effect is zero.
- **test_snapshot_champion_state.cpp.** Before this file the snapshot suites all used plain decks (empty championSlots) and the Champion suites never copied a GameManager.
- **test_timeout_rules.cpp.** TimeoutRules had no tests before this file; resolveTimeoutOutcome was bound after eight re-derivations across three waves (UPSTREAM item 16) that reproduced only the tower-count rule; all five original Python scorers reported the 3v3, 1200-vs-90 shape as a draw. The viewer called a timed-out match with a badly damaged Princess a draw.
- **test_snapshot_timing_state.cpp.** UPSTREAM item 17 tabled eleven accessors and was resolved minimally on 2026-08-24 with getTicksOnTarget() only.
- **test_navigation_wedge.cpp.** Measured 2026-08-20 by tools/audit/soak.cpp over 60 matches (308,464 unit-ticks): 4 units stopped 50+ ticks with nothing in reach, all in a two-building pocket; the first dump was Ice Golem (11.360, 2.939) between the King (9.0, 2.5, minDist 2.4, dist 2.401) and a Cannon (12.032, 4.169, minDist 1.4, dist 1.401), converging y = 2.9071 ... 2.9392. Re-anchored 2026-08-21 when the King moved 9.0 -> 8.5 (the configuration stopped trapping); re-measured 4 stalls in 347,501 unit-ticks. Two local repairs tried: a timed handedness flip (0.027 tiles / 120 ticks) and a wall slide (a new fixed point at (11.3102, 2.96839)).
- **test_clash_env.cpp.** Observation growth history: NUM_CARD_IDS 120 -> 175 -> 185; NUM_CHANNELS 9 -> 21 (attribute channels); NUM_EXTRA_SCALARS 0 -> 9 -> 10 (the elixir phase, 2026-09-02, which moved CYCLE_START 13606 -> 13607); CYCLE_BLOCK_SIZE 0 -> 370 (2026-08-27). Spawned-body tower damage (item 28): a team-0 Goblin Barrel on the team-1 left Princess, both idle 300 ticks, read getTowerDamageDealt(1) == 810 with every team-0 tower untouched.
- **test_game_manager_snapshot.cpp.** The divergence test once stepped 100 times and was flaky ~1 in 20 because the futures reconverge.
- **test_lane_pathing.cpp.** The end-to-end stall test first ran the full 900 ticks and failed at 68 stationary ticks: the Golem at (5.85, 28.85) was 3.12 from the King against a reach of 3.15, i.e. attacking.
- **test_card_cycle_observation.cpp.** The first run passed board y=25 for team 1, which mapped to team 0's half and was refused. A literal snapshot recency would read 0.99501, not 1.0, since stepSelfPlay advances a tick past the play.
- **test_board.cpp.** Bridge-mouth trap, measured 2026-08-09 off replay_ep1007 / replay_ep4029: a ground troop stopped at (4.00, 15.50) and (14.01, 15.49) for 100 and 104 ticks with the nearest enemy 9-11 tiles away; zero in three pre-speed-fix replays; the speed fix made a step landing in the disc ~5x likelier (~0.01 / step size; 0.3 -> 0.06). Bridge-exit trap, measured 2026-08-20 by tools/audit/bridge_audit.cpp: 6-8 of every 34 lone ground units never crossed, frozen 750-850 ticks (Giant 27/34, Ice Golem 28/34, Musketeer 26/34, Valkyrie 26/34; Hog and Ice Spirit 34/34; Minions fly), so it looked card-specific. The sweeps once used literal bridge coordinates, which the 2026-08-21 correction turned into sweeps of open water, silently retiring both regression tests.
- **test_hero_abilities.cpp.** The Hero Barbarian Barrel case drove 9 ticks ("delay, then detonate") until the 2026-08-28 rolling sweep. Hero Giant threw a Cannon (x 6.0 -> 11.0) because findHpExtremeEnemy excluded only towers and the effect wrote position directly (2026-08-26).
- **test_state_setters.cpp.** The setters (2026-08-17) exist because forecast.py rebuilt positions via reset() (elixir 5.0, unseeded hand). Perception's icon templates agreed with the elixir ledger only 33.8% of the time. Item 22 (2026-08-24) closed the tower HP, clock and injected-health gaps; the fourth gap (inject -> applyCardMetadata re-arming deploy time, a Hog running for six seconds made inert again) was found while verifying it, worth ~520 tower HP on a supported push. A live agent's clock once ran at twice its training rate because two maxTicks defaults disagreed.
- **test_combat_entity.cpp.** The sight case asserted the opposite until 2026-08-28 (an enemy at 6.0 visible to 5.5 sight "because the body radii are not empty space"), pinning the inflation that gave a Hog 10.9 against a catalogued 9.5. The tower-vs-in-sight case was inverted the same day: any non-tower in sight used to beat a tower at any distance.
- **test_card_registry.cpp.** The Ice Golem case asserted the on-attack freeze until 2026-08-26, pinning the defect. Graveyard at the old 10-tick cadence dealt zero tower damage from all 588 legal cells although all nine Skeletons spawned (2026-09-06). Before the curse latch, a Musketeer taking twenty Mother Witch hits died into twenty hogs. The 2026-08-24 tier rework matched "109 / 109" over cards with an official row; Golemite (raw 0.2f, 0.400 tiles/s vs VERY_SLOW 0.663) and the Night Witch Bats (0.85f, 36% off card 78) were outside it; hut Goblins ran 2.000 vs the card's 2.651.
- **test_default_deck_qa.cpp.** Written 2026-08-20 from tools/audit/deck_audit.cpp; the suite until then covered mechanisms, not registry cards (the blind spot behind the movement-speed bug). deck_audit's air probe used real Minions, which flew off and contaminated readings. damageDealtToAir's first version omitted the commits and reported Ice Spirit unable to hit air. A 2026-07-29 registry audit found Musketeer among four cards that could not shoot air. The Hog-vs-Skeletons case first required survival over 120 ticks and failed (three Skeletons ~220 dps plus a tower's 382 kill a 1697-hp Hog in under three seconds); it then checked the Hog's total damage, which passed only because the pre-tier Hog was too slow to reach the tower in 60 ticks, and failed correctly after UPSTREAM item 25 made it Very Fast. The Cannon case asserted `withCannon == 0` until the same change. Defence figures: unanswered 1268 before King dormancy (2026-08-21) and 2534 after; the Hog "prevented" 315, our own tower shooting. The Cannon expired at 31.0 s instead of 30.0 (824 hp, 27/s decay, 14 hp left), ~3.3% of a free Cannon.
- **test_game_manager.cpp.** Tower coordinates were literals (King 9.0, left Princess 4.0) until the 2026-08-21 correction; the overlap tests still used King x 9.0 in their comments until now. The footprint check (2026-08-28): a Cannon was accepted at x = 0.0 spanning [-1, 1] and at 17.0 spanning [16, 18]. Rolling spells: the ep-32,484 policy put 53.4% of Logs on the enemy half, spread to y = 33.
- **test_sight_range.cpp.** Measured 2026-08-20 by tools/audit/deck_audit.cpp, a lone Musketeer south of an enemy Princess Tower: at 6.5 and 7.5 tiles the tower took 1519 and she died (721); at 8.0 the tower took 5355 and she took 0; at 10.5, 4704 and 0. The first fix made sight surface-to-surface like attack; on 2026-08-28 sight became strict centre-to-centre with the reach floor, after the inflated form gave a Hog 10.9 against a Cannon. Rule A of the player audit was already implemented (58 withSightRange calls) but nothing pinned the numbers. Commit 1b17844 established 5.5 as the sourced default for 102 cards; Bomb Tower and Three Musketeers survived on the default until 2026-08-21, caught only because their attackRange was 6.0. The Evolution filter once returned -1 for all 41 Evolution ids. BuildingTargeter's `else if` bug: replays/hog_test_1.json, the Hog turned toward a Cannon 9.160 away while 7.714 from a Princess Tower and stayed wrong seven ticks; fixing sight alone only moved where the wrong preference fired.

#### Audit instruments (tools/audit/)

- **deck_audit.cpp.** The first air probe reported The Log, Hog Rider, Cannon and Skeletons as hitting air (the towers were killing the Minions); two later versions were saturated, the control Minions ending at 0 hp, so every card read "cannot touch air". The sight/attack section measured the 1.9-tile band that let a Musketeer siege a tower for free before the sight fix.
- **bridge_audit.cpp.** The snap detector first reported false snaps for Giant, Musketeer and Valkyrie, whose speeds (0.06, 0.10, 0.10) divide the 7.5 tiles from the spawn row evenly (17.44 -> 17.50 -> 17.56); reading lastDySign after the update made the reversal test always false. The old Catch2 sweep missed the exit trap by pairing each bank with one direction.
- **spawn_speed_audit.cpp.** Written after the 2026-08-24 tier rework ("109 / 109 match" over cards with an official row); its first version only asked "near some tier" and missed hut Goblins at 2.000 vs the card's 2.651.
- **waypoint_probe.cpp.** Its sweep used literal bridge coordinates until the 2026-08-21 correction turned it into a sweep of open water.
- **pull_range_probe.cpp.** A damage-based pull test saturated because Building::update decays hp on its own.
- **soak.cpp.** The stall detector went wrong three times: its first-sighting guard (`w.last == (0,0) && moved == 0`) could never hold, and the reach test was once checked only when the counter tripped, flagging a Musketeer that had shot an Ice Golem for 53 ticks until the Golem died on that tick.
- **elixir_phase_audit.cpp.** Written 2026-09-02 while a training run held the .pyd mapped, so the ordinary build was unavailable and this was the only way to compile the edited headers.
- **engine_profile.cpp.** gym_wrapper.step read `.observation0` only and teacher.execute_steps discarded the result entirely, which motivated stepSelfPlayFast.
- **verify_pyd.py.** Written after the .pyd went stale on 2026-08-19 (it predated commit 26de409) and quietly blocked two measurements. Its first version used `step` and "failed" 2/10 on a current .pyd because the heuristic opponent defended. Until 2026-08-24 it checked `step_self_play_fast` on the module rather than the class, so the gate reported FAILED on every good .pyd.
- **The river mask.** The observation's channel 8 was painted from a hardcoded `(x >= 3 && x <= 4) || (x >= 13 && x <= 14)` after the 2026-08-21 arena correction moved the physics: columns 2 and 15 (bridge) read as water and 4 and 13 (water) as bridge, with the C++ suite green.

### perception/

#### live/mvp_loop.py

- The docstring once claimed the perception encoder "does not exist yet"; it had existed since 2026-08-16, and the stale text sent a session re-deriving it.
- The live deck was a hand-written CRBAB list and stayed the Giant deck long after training moved to 2.6 Hog Cycle (three of eight cards in common). Nothing detected it.
- The ledger's cost table defaulted to (3, 4, 5), which happened to match the old deck; a 2- or 6-cost card would have produced unexplainable drops.
- Engine legality vs the net's own-half mask, measured on DEFAULT_DECK: 46 cells for every troop, 80 for the Cannon, 12 for Fireball. A 180 s match issued three Cannons at (2,5)/(2,6)/(2,7), inside the left Princess footprint, and none confirmed. The predicate is board-independent (UPSTREAM item 12: 208/288 for Cannon empty vs six troops deployed).
- Tactical advisor evidence: Cannon 564 vs 12 tower HP preserved, Fireball 2.405 vs 0.000 elixir killed, Giant 536 vs 3 tower damage dealt, against the learned cell.
- `net.predict_opp_elixir` was deleted on 2026-08-28 and this call was not updated, so `--policy neural` raised AttributeError on the first in-game frame.
- DeckHandDetector on match_practice_01 (tools/bench_hand.py, 120 blind-labelled crops): identity 90.0% -> 96.7%, impossible cycle transitions 60.0% -> 5.3%. Neither arm names a card on the 87 empty slots; the stock reader reads Musketeer correctly only 46.2% of the time, mostly as Mini P.E.K.K.A.
- The per-frame hand reading changes ~2.5x more often than cards are played, and 88% of its slot changes have no elixir drop; the ledger recovered 27 cards against an affordable ceiling of 28.
- The match gate took off-deck card reads from 7.8% to 0.5% and duplicates from 8.7% to 1.7% over the 2,510-frame live capture; a quarter of every capture is not a battle.
- Offline the perception step medians 190 ms; the first live run implied ~2700 ms. Guessing the cause cost a DirectML venv build addressing 4% of the budget, which is why stages are timed.
- Optimistic debit: before it, four placements were issued against a single unchanged reading of 10.
- 2026-09-06 live run, 35 placements: the tracked hand and the stock read disagreed on 18 (51%); the tracked hand had 0 duplicates, 0 off-deck cards and 6 unreadable slots in 140. The run's 27% unit-appeared rate was mostly this mislabelling, not missed taps.

#### bridge/sim_driver.py, forecast.py

- **sim_driver's pool size.** At ~31 hand-matching candidates about two of the 24 queue orders were typically missing, so roughly one match in four eliminated every candidate and could not recover, which looked like a nondeterministic bridge failure. 200 fixed it.
- **sim_driver's outcome split.** Eliminating every candidate on the incoming-card check once reported "the card was in no hand" when it had been in every hand; hence "refused" vs "desync".
- **sim_driver on the zero-error control:** 0 refusals, 0 desyncs, divergence identically 0. On a training replay refusals appear, as a consequence of estimate drift rather than the bridge.
- **forecast.py was written when the engine could not be snapshotted**, and its docstring kept saying branching search was "blocked on Entity::clone()" for weeks after `snapshot()` landed (2026-08-11), making the repo's top-ranked asset look blocked.
- **forecast.py's hand.** Forecasting one board twice differed in 11 floats, all hand one-hots or costs, and in 0 of 12,852 spatial floats.
- **The stale-.pyd trap.** The state-estimator setters landed 2026-08-17 and were invisible here because the post-build copy had failed; hence `_engine()` preferring build_python/Release/.
- **Swarm injection.** Injecting once per detected body turned four spear goblins into twelve: 12 occupied cells against perception's 7 before a tick was stepped.
- **Speed measurement.** A flat 10-tick baseline reported the Giant at 3.61 tiles/s against 3.0 (quantisation). Starting at (5.5, 4.5) worked only while troops were fast; after the 2026-08-07 speed fix the unit merged into the left Princess cell and the Giant read 0.17 against ~0.58. The single-cell rule returned nan for every swarm card.

#### live/unit_hp.py, live/actuator.py

- **unit_hp, before vs after** (damaged-unit detector, 226 detections, 71 frames): precision 0.79 -> 0.98, recall 0.34 -> 0.56, F1 0.48 -> 0.72. Three faults, each visible only after the previous was fixed: (1) the association window was inverted: it rejected badges more than 12 px below the box top (throwing away matches at dy -13 to -15) and accepted badges up to 104 px above, which belong to other units; every false positive came that way; (2) the fill test was `mean channel >= 185`, which passes the ally bar (mean 190.3) and fails the enemy bar (117.3), so every enemy measured 0%; (3) two units could claim the same badge (two stacked Giants did), and stacked widgets merged into one blob that the size filter discarded (8 of 12 remaining misses).
- **unit_hp's remaining gap.** 8 of the 10 surviving misses have no badge found near the unit; crop #64 has a plainly visible bar that `_bar_beside` reports as not-found. 12 of 20 sampled "no badge" crops were labelled undamaged.
- **The side oracle.** An early reading put badge vs `side.onnx` disagreement at 31% with the badge always right; that was measured through the broken matcher. Re-measured over 67 associated detections: 10% (7 cases). Of five checked by eye, the badge was right twice (#64, #225, our own Valkyrie reported as enemy), wrong twice (#108, an undamaged Mini P.E.K.K.A matched to a neighbour's badge; #194, a false badge on a Giant's foot), and #42 was not a unit.
- **Badge search.** Searching for the bar directly found a bar for 44% of units, with widths averaging 32 against a true 38-56. Harvesting badges frame-wide found 2.5 per frame while per-unit windows associated 0.67. The first hue calibration used one ally badge with invented enemy values and scored 11% recall. Sharing the HSV conversion saved 32 ms of a 93 ms find_badges.
- **Actuator timings** on this emulator: `adb shell echo` 107 ms one-shot / 3.2 ms persistent; `adb shell input` 413 / 412 ms (a persistent channel does nothing for `input`, which spawns a JVM per call); six `sendevent` calls 362 ms vs `input tap` 307 ms. Inline taps dropped the decision loop from 1.0 Hz to ~0.6 Hz.
- **Actuator's first raw-touch bugs**, found by tapping real buttons: assuming the app's frame for the touch axes put a tap for the top-right hamburger on the top-left profile banner; a zero-duration contact dismissed a menu instead of pressing the button.
- **Row 1 unreachable.** The 2026-08-05 tile-grid refit put the arena's bottom edge at y=1003.8 while the game stops accepting taps at y=981, so engine row 1 was tapped at y=989 and measured 0/6 live. The fix belonged in the grid, not in widening the mask.

#### live/elixir_ledger.py, tools/measure_timing.py

- **Why the ledger reads the bar.** Hand-slot changes once scored 1 of 76 against elixir drops over a live match. That measurement predates the 2026-08-05 fix to `adapter._hand_ids` (it read `cards[:4]`, where `cards[0]` is the Next preview), so it is not trustworthy; live runs since show the hand cycling correctly.
- **Ledger results.** One live match, 204 s in-game at ~10 fps off a recording: 27 cards implied against an affordable ceiling of 28, conservation residual +14%. The glitch-to-zero readings are 4.2% of samples. Rejecting drops above the single-card maximum left 39% of a match's elixir unaccounted for (double plays). COST_TOLERANCE 0.55 was fitted at 10 fps; at the live rate a real 3-cost play (3.4 -> 0.4) reads 3 -> 1 and presents as 2.24, rejected at 0.55, accepted at 1.0. An earlier synthetic measurement of the rate-blind version: -5% at 10 Hz, -32% at 2 Hz, -48% at 1.5 Hz, -82% at 1 Hz.
- **measure_timing.py (2026-08-10).** Written when the engine had no deploy time (added 2026-08-19) and after the 2026-08-07 speed fix made a spell's window cover 5x less ground. CRBAB's tower bar fraction took ten distinct values (0.0, 0.46, 0.49, 0.62 ... 1.0) over one 77-frame trial on a tower whose HP never changed, since `_calculate_hp` returns 0.0 for both an empty and an unmatched bar; the numeral reader fixed that, but the cast UI occluded the numeral from 950 ms to 2150 ms after the cast (35 frames). The motivating Fireball was genuinely played (10 -> 6, card cycled) with the indicator on the target tower, and neither enemy tower lost HP. A warm-pixel count reported ~12,000 hits on every frame; the first decay test read the end of the capture as decay and reported a confident 1799 ms for an explosion that was not there (the melee peaked 650 ms from the end; a 3.5 s window left 350 ms). Two whole matches were lost watching a frozen hand (`['minipekka','musketeer','valkyrie','minions']`, 150 s) before CYCLE_TILE.

#### live/deck_hand.py, tools/deploy_zone.py, tools/calibrate.py

- **deck_hand.** The first version used the cost badge alone as the presence gate; since a dimmed card's badge is greyscale, 15.1% of in-match slots (one in seven, exactly when elixir was low) were reported empty. Its four residual errors were all lifted (selected) cards, each recovering to 0.90-0.99 at dy = -11. The hungarian-vs-argmax difference appeared only once dimmed cards were read at all; against the badge-gated sample the two were indistinguishable. On match_practice_01 frame 372 all four slots are desaturated at once; at frame 453 only the Giant (cost 5) is.
- **deploy_zone.** Its first run reported a confident, entirely wrong "288/288 cells legal" because the two 4-cost slots were unaffordable and selected nothing. The select/measure/deselect loop failed the same way (a second tap does not deselect, so every later baseline held the previous tint). Omitting `in_arena` read engine row 0's below-board taps as legal. Taking the extreme tinted row as the river edge put it at y=1265 (the tray highlight) and gave a TILE_HEIGHT of 68.
- **The 2026-08-05 tile-grid refit** scaled y off "the two princess HP bars are 21.0 tiles apart" and x off "the two river gaps are 10.0 tiles apart"; both counts were short by one, inflating TILE_HEIGHT by 4% and TILE_WIDTH by 10%. deploy_zone's tint-edge fit replaced it.
- **calibrate.** Until 2026-07-30 it scored against both the engine's geometry and a hypothetically corrected one (UPSTREAM items 1-2); both fixes landed and the comparison was deleted. validate_grid measured 0.105-0.224 tiles, PASS, that day. The first tower labeller sorted by rank, putting `own_princess_left` at (958, 740), the King's position, with 132 px spread (fit 0.60 -> 4.01 tiles). Sampling whole matches put it at (830, 563) with 61 px spread against a true (819, 656) (fit 0.60 -> 2.98). One recording scored 6.50 tiles from a single frame while the others scored ~0.7. The stored max was once in-sample (0.31) rather than leave-one-out (0.78). The river moved from [16,18) to [15.5,17.5) on 2026-07-29; a copied layout here would have scored bridges against y=17.0.

#### live/adapter.py, capture/window.py, geometry.py

- **adapter.** 31% of raw detections sat outside the arena, all the avatar icons read as `knight`. 97 detector classes against 132 engine cards left 27 names unresolved, including `archer` and `minion`, two of our own eight. Reading `state.cards[:4]` put the Next preview in observation slot 0, shifted every real card right and dropped slot 3, so every placement tapped the neighbour of the card asked for; it cost a whole match, and `ready` and `cards` disagreed by one about "slot 1". The 0.0-means-destroyed rule came from 71 ladder frames where the failure happened not to occur; live at 549x976 `right_ally_princess_hp` read 0.0 in 101 of 101 frames while the tower's numeral showed 1890 (a full level-5 Princess). The badge-vs-side.onnx 31% figure came from the broken matcher.
- **capture/window.** ADB timings: `screencap -p` 2243-4886 ms (10 samples), raw 1414-1885 ms (6). WGC against a fully covered BlueStacks delivered 1920x1020 frames every ~85 ms. An earlier docstring said the window "does NOT need to be visible", conflating covered with minimized. HP reader at scale 1.000/0.850/0.750/0.637: precision 0.98/0.98/0.98/0.97, recall 0.56/0.65/0.53/0.50, F1 0.72/0.78/0.69/0.66, on ~15 weighted positives. The game-rect derivation was 74.7 of the 85 ms `read()` cost and 29% of a recorder frame before it was cached.
- **geometry.** The river once sat at [16.0, 18.0) (centre 17.0) against a tower layout centred on 16.5, giving team 1 one fewer placeable row (README "Findings reported upstream", item 1); re-centred to [15.5, 17.5). A 2026-07-30 homography fit (UPSTREAM items 1-2) moved the left Princess to x=4.0 and the Kings to 9.0; the 2026-08-21 re-centring reversed that to 3.0 and 8.5, and this file's hardcoded copies (King 9.0, left Princess 4.0) went stale until ArenaLayout.h was bound. Observation channel 8 painted {3,4} and {13,14} until 2026-08-21.

#### contracts.py, tests/test_engine_state_setters.py, tools/sim_fidelity.py

- **contracts.Phase** was written when the engine had no phase concept, "so the perception layer is already correct on the day the engine grows phases"; the engine gained the 1x/2x/3x schedule on 2026-09-02 (UPSTREAM item 26). The observation layout changed 6253 -> 13606 on 2026-07-29, the reason GameState is not the vector.
- **The setter tests' controls.** Both the refusal and the clamp tests were first written without controls and passed against a do-nothing stub. The inject-hp test used a single 400 hp point until the 2026-08-24 speed rework (Hog 1.6 -> 2.0 tiles/s) made both arms reach the tower. The deploy test first probed total damage and saturated. Its window was a hardcoded 110 ticks, which after the rework read `1268 > 1268`. The .pyd going stale twice hid `set_elixir_for_team` / `set_hand_for_team` for days with the C++ suite green.
- **sim_fidelity.** Its grid check once demanded max(y) >= 29 and raised a false alarm on a good grid measuring 1..28. It was written when the engine had no deploy time (added 2026-08-19) and before the setters existed.

#### tools/build_*_templates.py, capture/video.py, track/cycle.py

- **Icon templates.** Ungated, the out-of-match "7" badge on a wooden panel formed a 112-member cluster of its own on match_practice_01 and evicted Mini P.E.K.K.A (the match's rarest card); seven cards plus one junk group looked like success. On the 2026-09-02 recording k-means at k=8 spent a cluster on the selected render (pulling Musketeer and Skeletons together) and merged Fireball with Ice Golem, leaving Ice Golem with no template; group sizes ran from 137 (The Log) to 26 (Skeletons).
- **Tower digit templates.** 5,389 crops reduced to 30 groups. Every live capture showed 2030 (full health). NUMBER_CONFIG read grass on the recordings; bar colour-matching put the left bar 30 px off on the pink skin. The white mask put the numeral band at rows 139-147 in all 8 recordings and 130-139 live (one tower destroyed, correctly empty). Otsu merged 588 of 2,711 cells into pairs, including every "90", leaving no clean 9; the white mask gives four runs on "1890" where Otsu gives two. A post-normalisation width filter removed every 0, 4, 6 and 8. Naive k-means++ seeding allocated 243 MB at k=30. readers/clock.py once scored 24.7% from cropping one way and matching another.
- **video.py.** On this batch the median delta gave 30.303 fps and the span 29.999. `sample_every` via grab() took 8 recordings (~59,000 frames at 4 fps sampling) from minutes to seconds.
- **cycle.py.** The FIFO rule had three copies (here, live/hand_tracker.py, track/deduce_identity.py) until 2026-08-24.

#### readers/hand.py, live/placement_confirm.py, tests/test_forecast.py, readers/tower_numerals.py

- **hand.py.** Raw-BGR matching flickered: 0.790 mean confidence, 14% of hands with a duplicate or off-deck card, and phantom plays 0.8 s apart in one slot (impossible under the engine's 20-tick slot cooldown). Without debouncing, a 326 s recording at 6 fps showed 69 "plays", 32 closer together than the cooldown allows (three changes in one slot inside 1.5 s once). Filtering crops on `crop.std() < 18` let the empty between-match tray through, and 1,525 such crops formed their own cluster, displacing a real card. `normalise_icon` was duplicated between this reader and the template builder.
- **placement_confirm.** A run reported "17 issued plays never confirmed" without saying which failure happened. `tools/probe_placement.py` matched by position and named a Musketeer that had merely walked.
- **test_forecast.** Its dynamics test used a 2.0 s horizon that passed only because the old closest-tower rule tie-broke left, off the cell boundary; after lane pathing (2026-08-21) the Giant is at (9.36, 8.48) at 2.0 s. The determinism test first compared whole vectors and failed on 11 hand floats (0 of 12,852 spatial). Giant's speed constant was 0.6 before the 2026-08-24 speed-tier rework (commit 4b31a42). A flat 10-tick baseline once read the Giant at 3.61 tiles/s against 3.0.
- **tower_numerals.** On one Training Camp trial (77 frames) CRBAB read both of our live Princess towers 0.00 throughout, and the tower a Fireball hit went 1.00 -> 0.00 -> 0.62 -> 0.67 -> 1.00, none of it the Fireball; this blocked measuring spell delay. The first ROI ended at bar_top + 2 and segmented "2030" as one cell; luminance-based Otsu did the same on grass.

#### readers/clock.py, live/pipeline.py, tests/test_deck_hand.py

- **clock.py.** The first reader pooled un-normalised crops and scored 24.7% on a recording with every frame's answer known; fixed-fraction segmentation scored 50%. Stretching glyphs to a fixed width caused a persistent 1->3 confusion that survived two other fixes. A width-based colon filter made four runs, fell back to fixed boundaries on every frame, and trained a "1" template blended with the colon.
- **pipeline.py.** Measured live, serial: 29 iterations in 60 s (0.48 Hz), detector at 2.2-2.5 s. Offline the chain is 190 ms a frame (115 ms the ONNX forward pass); the first live run showed a ~2700 ms period, and per-frame totals ranged 1.9-9.3 s.
- **test_deck_hand.** An earlier version required every empty crop to clear MIN_SCORE, the giant recording's numbers written as a universal; hog26 has an empty slot at 0.431, which failed a correct classifier. The `tools.audit_hand_id` import worked from perception/ but raised ModuleNotFoundError under pytest.

#### tests/test_live_elixir_ledger.py, tools/bench_hand.py

- **Ledger fit.** The end-to-end figure the ledger was first fitted against: 29 cards, residual +12%, over 935 frames of a real match (BOT_REQUESTS.md item 8). Before `gained` was modelled, it was read off the bar and the residual was positive by design.
- **bench_hand.** The first label set sampled only badged slots, excluding every unaffordable card and empty slot (20% of in-match slots), and scored 100.0% while the classifier reported `blank` for one slot in seven. Without the warm-up the identical-arms control read 1.426. With a fixed arm order the second identical arm read 6.8% slower every round. Constructing two CardDetectors from one list gave the second thirteen cards and a third eighteen.

#### live/hand_tracker.py, tests/test_sim_driver.py, live/unit_to_card.py

- **hand_tracker.** Over one real match (935 in-game frames, 22 plays) the per-frame hand read: 64 of 73 slot changes (88%) had no elixir drop; 9 of 22 plays (41%) produced a change; 47 of 74 returns (64%) came before an 8-card cycle allows; median unchanged run 0.8 s against ~9.3 s real, i.e. ~11x churn. The detector reported 0 off-deck cards in 3,740 and 0 duplicates in 935 frames; its mode gave 70 distinct hands raw and 27 at a 25-frame window against a true ~23. The unsuppressed desync check fired 44 times against 20 plays and pinned confidence at zero.
- **test_sim_driver.** Feeding reconstructed replay events produced up to 530 HP/tower of divergence and once killed a King ~200 ticks before the real match ended. Its tower cells were hardcoded at the 2026-07-30 positions (King 9.0, left Princess 4.0) and became reads of empty cells after the 2026-08-21 re-centring; every tower read 0 and it looked like a bridge defect. The King was excluded when the engine's King had no dormancy.
- **unit_to_card.** Measured at the time: 97 detector classes against 132 engine card names, 70 matching by name, 27 unmapped. A Fireball in hand once resolved through the unit table to UNKNOWN_CARD_SIM_ID while the three cards spawning same-named units looked fine.
- **match_clock.** Written when the engine had no phase concept; the phase schedule was supplied by the project owner on 2026-07-29.

#### tools/match_nav.py, track/opp_elixir.py, tests/test_live_adapter.py, readers/elixir.py, live/king_hp.py

- **match_nav.** Replaced `enter_training_camp.py` on 2026-08-24, which assumed a lobby start, never answered the confirmation, and kept its own copy of the Training Camp coordinate. adb banner, measured 2026-08-16: warm stdout 1,262,367 bytes starting at the magic; cold 1,263,969 bytes with the magic at offset 85. The capture timeout was 30 s until a just-booted emulator exceeded it.
- **opp_elixir.** Written when the engine had no phase concept, so the multipliers were confined here; the engine gained the schedule on 2026-09-02 (UPSTREAM item 26), which made this table a second copy.
- **test_live_adapter.** The destroyed-tower test asserted 0.0 = destroyed until 2026-07-31, justified on 71 ladder frames with a clean decay (1.000, 0.744, 0.231, 0.077, 0.000). The `cards[:4]` bug, verified live: Next=Giant with hand [Archers, Valkyrie, MiniPEKKA, Cannon] was reported as [giant, archers, valkyrie, minipekka].
- **elixir reader.** A segment-counting reader pinned 450 of 589 samples near 0.5 elixir while the real value ranged 1-10.
- **king_hp.** Written when the emulator was 900x1600, changed to 720x1280 an hour later. On a real ladder frame our King sat at 241 HP while the match was being lost; its bar showed no blue fill at all (the "241" numeral starts at x=164), and presence-on-fill reported it as full.

#### tools/audit_hand_id.py, detect/placements.py, tools/build_digit_templates.py

- **audit_hand_id.** Hand identity once had one published number, "templates agreed with the elixir ledger 33.8% of the time", measured on readers/hand.py, which had no caller; the live loop then read its hand from CRBAB's `CardDetector`, so the figure described dead code and this tool was written to measure the classifier that actually ran. An earlier version counted `X -> blank -> Y` as two transitions and attributed 62.9% of transitions to misclassification on a stable, correct reading. Before `deck` was a parameter, auditing the Hog deck against the Giant default read 100.0% off-deck for every arm; scoring hog26 against the Giant hold time reported "churning 2.3x too fast".
- **detect/placements.** Written before any recording existed; `assets/recordings/` now holds 8 matches, and README.md's stage 3 still asks for further footage covering the opponent side.
- **build_digit_templates.** The first read-back check reported off-by-one misreads at every sample landing on a .8 s offset, before the anchor phase was fitted.

#### readers/towers, test_tower_numerals, test_tile_grid, deduce_identity

- **readers/towers.** Written when the engine's King fired from tick 0; the Kings' divergence was read as a diagnostic that "should track the known discrepancy".
- **test_tower_numerals.** The clock's digit templates did not transfer to tower numerals: confidence 0.08-0.18 against a 0.35 threshold, "3" read as "1".
- **test_tile_grid.** The tile grid was refitted twice; the 2026-08-05 fit was wrong in both axes (10% in scale), passed the centre check, put the arena bottom at y=1003.8 against the game's 981, and tapped engine row 1 at y=989 (0/6 live acceptance). The ledger's "issued but never confirmed" had three other candidate explanations.
- **deduce_identity.** The icon templates read our hand correctly 33.8% of the time (README stage 2, 328 in-match plays over 8 recordings) and over-predicted Giant at 35% against a 12.5% prior. HAND_SIZE/QUEUE_SIZE were redefined in this file until 2026-08-24, three lines below a docstring forbidding second copies.

#### encoder test, measure_decoupling, test_live_deck, compare_card_features, record_match, action_gate

- **test_perception_encoder.** A module-level importorskip once skipped the whole file silently ("1 skipped" instead of the expected failures). The round-trip tests used the bare constructor and so pinned perception_encoder to the 1800-tick pybind default; the encoder divided by 1800 while the policy trained on 3600, so the live agent's clock ran at twice the rate it learned, invisible to every simulator metric. The first construction asserted "X at the placement point" and failed on Minions. A units-only encoder left 34 floats at zero (river and towers).
- **measure_decoupling.** Measured: detector ~290 ms, perceive() ~502 ms, live board age mean 980 ms, p95 3254 ms. A round-number horizon grid reported "too few pairs" for five of eight rows. An own-units-only metric read 0.0000 at every horizon.
- **test_live_deck.** The ledger's (3, 4, 5) default matched the Giant deck exactly, so deriving costs was behaviour-preserving until the switch to 2.6 Hog Cycle (2026-08-17), when four of eight cards became unrepresentable. The live loop played the Giant deck for a day after training moved to 2.6 Hog Cycle.
- **compare_card_features.** The incumbent's frequency spread measured ~6x against ~1.0 for a good reader; a real hand holds ~9.3 s on the Giant deck.
- **record_match.** The live loop spent ~640 ms of its ~917 ms on the YOLO pass; pure capture ran at ~13 fps.
- **action_gate.** In dry run with the producer at 0.36 Hz: 44 taps for ~22 placements, the policy repeating `net slot 0 -> (3,8)` three decisions running. DirectML then made the producer 3.99 Hz.

#### test_live_pipeline, placement_truth, match_state, katacr_format, fit_tile_grid

- **pipeline tests.** The worker exists because inline perception made the live loop 0.48 Hz. An earlier age test put the delay in the source and measured nothing. The first live run's 2700 ms period was attributed to the detector with no evidence; offline it medians 130 ms. Per-frame totals ranged 1.9-9.3 s.
- **placement_truth.** One 160 s live run issued 34 placements, one every 4.7 s against a sustainable one per ~9.5 s, sitting at 0-3 elixir all match. An oracle that outvoted the others is how the project once got "0 illegal placements out of 32".
- **match_state.** Over the 2,510-frame capture in match_practice_01, gating on in_game: off-deck card reads 7.8% -> 0.5%, duplicates 8.7% -> 1.7%, median unchanged hand 0.60 -> 1.20 s, frames kept 74.5%. Before it, `mvp_loop.perceive()` gated only the elixir ledger on the screen.
- **katacr_format.** Deriving the frame rate from the endpoints once reported 641 fps and 1.2e9, collapsing every timestamp to zero.
- **fit_tile_grid** produced the 2026-08-05 fit. It assumed the bridges 10.0 tiles apart (x = 4.0 / 14.0) and the Princess bars 21.0 tiles apart; both counts were short by one, inflating TILE_WIDTH 10% and TILE_HEIGHT 4%, and its centre check (Kings at x = 9.0) passed regardless. An earlier attempt through the desktop homography disagreed with it by nearly a tile. Upstream's defaults were ~10% small in x and ~4% in y on this emulator.

#### probes, prior test, test_live_unit_hp

- **probe_towerhp_ranking.** Written when tower HP could not be written into the engine at all; `set_tower_hp` / `destroy_tower` landed 2026-08-24 (item 22) and sharpened the question. One sampled frame measured hp_fraction 0.87 on our right Princess and 0.81 on the enemy King.
- **probe_placement.** It is how the hand-slot off-by-one was found (the wrong card was played). An earlier version matched by (name, tile) and named a Musketeer that had merely walked.
- **test_placement_prior.** Before the int() guard, truncating int(-0.87) made (8, 0) the Musketeer's modal cell with 7.6% of its mass.
- **unit_hp tests** measured precision 0.79 -> 0.98 and recall 0.34 -> 0.56 on 226 detections from 71 ladder frames.

#### encoder/engine test, engine.py, board_filter, context, test_tappable_rows

- **test_encoder_matches_engine** was written when a live match placed 60% of its cards on the back two rows against 19-21% in simulation. The two encoders agreed exactly on the towers-only baseline (0 of 12,852 spatial and 0 of 754 scalar floats), localising the discrepancy to unit detection.
- **engine.py.** Eleven perception modules imported the binding directly, so which build won depended on import order; fixing forecast.py alone fixed nothing unless it ran first.
- **board_filter.** Over 71 ladder frames the detector produced 328 detections, 102 (31%) off the board, all `knight` (tile (19, 5) x68, (-2, 13) x33); `knight` fell from 111 of 328 (the most common class) to 9 of 226 once they were dropped.
- **context.** Written when the engine had no double-elixir phase (added 2026-09-02).
- **test_tappable_rows.** Before the fix a 180 s live match issued 25 placements, 5 on engine row 0, and reported "18 issued plays never confirmed". An earlier version quoted y=1018 against an arena bottom of 1003.81, both from the later-refuted tile grid, and passed because both numbers came from the same wrong constants.

#### divergence_report, divergence, live/__init__, conftest, board_filter

- **divergence_report** described a video mode as one of "two input modes" until 2026-08-24; no flag existed.
- **replay_mining/divergence.** Tower-presence outcomes came back "draw" on every episode before the terminal-reward reader.
- **live/__init__** once listed `bot.py` as the surviving coordinate oracle (it was deleted in the 2026-08-24 cleanup, its two formulas transcribed into tests/test_live_actuator.py) and called capture/window.py, the match clock and the elixir ledger "not yet built". unit_to_card: 70 of 97 detector classes matched engine card names. unit_hp: precision 0.79 -> 0.98, recall 0.34 -> 0.56.
- **conftest.** A labels file sorting ahead of the replay became paths[0] and failed three tests inside Replay.__init__ with "list indices must be integers", which read like a corrupt replay. python_ai/replays/ was observed dropping from eight files to one mid-session.
- **board_filter tests.** Over 71 ladder frames: 102 of 328 detections, all `knight`, at (19, 5) and (-2, 13).

### web/

#### Replay viewer (web/viewer.html)

- **Legacy tables.** They drifted in practice: Inferno Dragon's symbol '4' was missing from all four, so `MAX_HP[ent.symbol] || ent.hp` gave hpRatio = 1.0 forever and it never visibly took damage; cardMeta in the replay format replaced them.
- **Unknown-card rendering.** `meta.maxHp || '?'` rendered '?' for every spell; `|| ent.symbol` rendered a literal '?' as the NAME of Cannon Cart (69) and Guards (76), whose symbol is '?'. `|| ent.hp` made unrecognised cards' HP bars never move. The name path exists because a symbol is shared by 36 of 90 cards.
- **Tower ids.** The tower panel once looked for ids 100/101/102/110/111/112, which never existed, so tower HP never showed.
- **Bridge columns.** The viewer hardcoded `[[3,4],[13,14]]`, the arena before the 2026-08-21 re-centring, painting four of eighteen columns wrong both ways; the same stale copy hit the observation's channel 8 that day.
- **Winner detection.** The viewer checked only whether both Kings were alive and called anything else a draw, with a comment claiming it "mirrors MatchRules::evaluate exactly" (true, and the bug); a timed-out match with a Princess at 90 hp showed "Draw. Timeout — both King Towers still standing". The Python guard walked .py files only, so the defect survived in JavaScript.
- **Click mapping.** Measured at 420x772 backing inside a 420x580 box: content 315x580 with ~52px letterbox bars, a per-axis scale of 1.00 across and 1.33 down, and 21.2% of dead-centre clicks selecting nothing. The 0.75-cell floor is 16.5px: a tile's half-diagonal is 15.6px and an 8px diagonal slip is 15.1px at the measured 0.75 display scale; a unit one tile away is 22px off.
- **Teacher debug data.** Stamping decisions onto every tick made the fixture 11 MB against 2.97 MB. Rendering the teacher's raw own-frame y put every red candidate on the blue half; over 8 real placements mean |dx| 0.01, |dy| 8.25 as recorded, 0.00 mirrored. A measured decision had ranks 1, 2 and 4 all on (2,15).
- **Rendering.** Painting the static background per frame (612 fillRects plus 52 strokes) measured 127 ms/frame. A territory tint on each half was the biggest reason the viewer did not read as Clash Royale. The flex panel with a transition stayed at its 0 start value indefinitely; `transition: none` opened it to 400px immediately.
- **Hover tooltip.** Removed: it drifted from the cursor under CSS scaling, and its hit-test took the first array match with radii not matching the drawn sizes, so some entities never hit.
