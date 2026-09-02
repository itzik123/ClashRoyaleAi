# The human placement prior

**Status:** Phases A–D complete. **The A/B is NEGATIVE — the prior makes the
policy measurably worse: −0.0725 win rate, 95% CI [−0.1125, −0.0325],
p = 0.00066.** It stays OFF (`CLASH_HUMAN_PRIOR_COEF=0`, the default it already
shipped with), so no rollback was needed. Read Phase D before enabling it.

## Why this exists

The predecessor idea — behaviour cloning from human replays — was killed by
measurement on 2026-08-29. The divergence probe
(`perception/replay_mining/run_probe.py`) replayed 20 expert matches through our
engine open-loop and found:

| | |
|---|---|
| usable horizon | **< 30 s** |
| reconstructions ending early | **20/20** (143.6 s vs a real 245.4 s) |
| outcome agreement | **40% against an 80% base rate** — worse than trivial |
| scrambled-time control | true arm better, ratio ~1.15 — the pipeline is not noise |

The last row is what makes this design possible. **Mining placements works;
carrying a match forward does not.** The prior uses only the half that passed.

## The architectural property that matters

Every training example is an independent triple

```
(context, card, cell)   read from the recorded frame at the moment of the placement
```

No simulation, no error accumulation, no opponent inference. The context is read
from the *recorded* frame, never a simulated one, so none of the divergence
result applies.

## Phase A — prerequisites (done)

**A1. A live bug in `python_ai/advisors/tactics.py`, found while speccing.**
`elapsed_ticks` read `obs[-9]` and the opponent-spend read `obs[-7]` — backward
indices into what became the card-cycle block when it was appended on
2026-08-27. Measured: at engine tick 300 the forward offset reads 300.0 and
`obs[-9]` reads **0.0**.

Consequences while it was live: `elapsed_ticks` was constant zero, so
`opp_elixir_estimate` computed income over zero elapsed time, so `hog_advice`'s
gate — the only win-condition advisor target — was gated on constants.

CLAUDE.md records this defect class and says five call sites were fixed. This was
a sixth, and the sweep could not have found it: that sweep searched for
`observation_size() - NUM_EXTRA_SCALARS`, and these sites spell the same bug as a
negative index. **Same defect, different spelling.**

Fixed to forward offsets derived from `engine_constants`; pinned by
`python_ai/tests/test_tactics_scalar_offsets.py`, which asserts both the
behaviour (the clock tracks the engine) and the structure (every index lands
inside the extra-scalar block, never in the cycle block).

**A2.** All 221 `fast_hog_2.6` episodes cached (1.33 GB, gitignored).

## Phase B — extraction (done)

`perception/replay_mining/`:

| module | role |
|---|---|
| `katacr_format.py` | episode reader; card index; robust timebase |
| `geometry.py` | per-episode affine fit from the four towers |
| `events.py` | labelled ego placements |
| `context.py` | the joint-observable context key, both implementations |
| `extract_prior.py` | episodes to a placement table plus census |
| `prior.py` | counts to a served surface (backoff, blur, legality) |
| `evaluate_prior.py` | the held-out kill criterion |

**Extracted: 10,718 placements from 221/221 episodes.** Transform residual mean
0.246 tiles, p95 0.316. Every deck card has at least 744 samples.

The board transform is **fitted per episode from the four towers** rather than
hardcoded — the engine stays the only source of truth for arena geometry, and the
residual is a first-class output so a bad episode can be dropped rather than
silently reconstructed.

A placement mapping outside our board is **dropped and counted**, never
truncated. Their arena is 32 rows to our 34 and the fit is on interior landmarks,
so extrapolation puts the deepest placements slightly below y=0 where
`int(-0.87) == 0` relocates them to the back row. Before the guard, that artifact
alone made (8,0) the Musketeer's modal cell with 7.6% of its mass.

### The kill criterion fired

Mean held-out log-likelihood per placement, 5 episode-level splits (splitting by
placement would leak, since placements within a match are heavily correlated):

| key | K | sigma=1.0 | vs marginal |
|---|---|---|---|
| uniform | — | −5.5667 | −0.9299 |
| **marginal** | **1** | **−4.6368** | **+0.0000** |
| threat | 2 | −4.6455 | −0.0087 |
| phase | 2 | −4.6629 | −0.0261 |
| lane | 3 | −4.6293 | +0.0075 |
| half | 3 | −4.6576 | −0.0208 |
| full | 18 | −4.7782 | −0.1414 |

**The prior is a large win; the conditioning is not.**

- +0.930 nats over uniform is a factor of **2.53** on the likelihood of a real
  human placement, consistent across all five splits. Top-1 cell 0.081 against a
  uniform-over-legal 0.0038 — **21x**.
- Splitting 10.7k placements 18 ways leaves ~74 per bucket against a 612-cell
  grid; variance costs more than bias buys. Only `lane` beats the marginal at
  all, by +0.0075 against a split-to-split spread of ~0.03 — about one sigma.

**Decision: serve `P(cell | card)`, sigma = 1.0.** `SERVE_KEY` is a module
constant and changing it requires a new measurement, not an opinion. The context
machinery is kept because it produced this table, and the placement table records
a context per row, so it can be revisited with more data.

### Controls (all pass)

These fail loudly if the coordinate frame is wrong — the failure this corpus
invites, since their ego is our team 0 and their y runs the opposite way.

| control | measured |
|---|---|
| Fireball lands on the ENEMY half | **0.672** (only a spell may cross) |
| Hog concentrated in rows 12-15, just below the river | **0.894** |
| Cannon stays on our half, modal row mid-half | 1.000, modal (8,8) |
| no probability mass on illegal cells | 0.000 |

Artifact: `python_ai/advisors/data/hog26_placement_prior.npz`, 9 KB,
stamped with the arena constants it was built against. The loader refuses a
mismatch — the contract `bc_pretrain.load_dataset` already uses for
`observation_size()`.

## Phase C — integration (done)

One seam: `advisor_target.target_logits_for(obs, card_id, legal, T) -> (612,) or None`.
A new `python_ai/advisors/human_prior.py` supplies the same signature, and
`target_logits_for` gains a fallback.

**Precedence: the hand-written rule wins where it speaks; the prior fills the
gaps.** The hand-written rules are engine-validated (Cannon 539.1 HP preserved vs
the net's 102.2); the prior comes from a game whose physics our engine does not
reproduce — which is precisely what the divergence probe established.

| card | today | with the prior |
|---|---|---|
| Cannon, Fireball, Hog | rule, else **entropy** | rule, else **prior** |
| Musketeer, Ice Golem, Skeletons, Ice Spirit, The Log | **entropy only** | **prior** |

The second row is the prize: **5 of 8 deck cards currently have no placement
target at all**, and `advisor_target`'s own docstring says why uniform fails —
"the argmax of a flat map is an arbitrary constant." The prior upgrades that
fallback from uninformed to informed.

Stated honestly: at the marginal level the prior teaches *where* humans put a
card, not *when* or *in response to what*. That is strictly more than uniform and
strictly less than a state-conditional rule.

Also needs revisiting: `ADVISOR_SLOT_WEIGHT = 5.0` steers the coverage draw
toward the 3 ruled cards; with 8 covered that allocation is wrong.

Gated by `CLASH_HUMAN_PRIOR_COEF`, default 0, so both A/B arms run byte-identical
code — the pattern `PLACEMENT_COVERAGE_COEF`'s own comment mandates.

### What it measured out to

Coverage-slot target rate over 495 decision states carrying a live threat:

| | overall | Cannon | Fireball | Hog | the other five |
|---|---|---|---|---|---|
| prior OFF (today) | **23.4%** | 0.94 | 0.95 | 0.02 | **0.00** |
| prior ON | **100%** | 1.00 | 1.00 | 1.00 | **1.00** |

**The prior speaks unconditionally, and that needs defending** — `advisor_target`
is emphatic that a source which always answers teaches a constant. The defence is
that the newly-covered rows were not previously getting a state-*dependent*
target; they were getting the **entropy bonus**, which is equally
state-independent (it pulls every board toward the same uniform map). This swaps
one state-independent pull for a strictly better one and introduces
state-independence nowhere it did not already exist.

What does change is strength: a human marginal is far sharper than uniform, so it
pulls harder. That is what `HUMAN_PRIOR_COEF` is for, and why the A/B watches
per-card **modal share**.

### Implementation notes

- The weight rides through `coverage_has`, which was **already** float32, so no
  new rollout buffer field was needed. 1.0 for a rule, `HUMAN_PRIOR_COEF` for the
  prior.
- `coverage_terms` derives its entropy mask from `has_target > 0` rather than
  `1 - has_target`. With a non-1.0 weight the old form would hand a row 90% of an
  entropy bonus *while* giving it KL — the exact opposition that function exists
  to prevent.
- The KL is divided by the **row count**, not the summed weight. A weighted mean
  would normalise a uniform weight straight back out and the coefficient would
  silently do nothing.
- `blur()` and `geometry_stamp()` live in `human_prior.py` and `perception`
  imports them, so there is one implementation of each. perception may import
  python_ai; the reverse is forbidden.
- `slot_weights_for` now weights every card with a target source. Self-adjusting
  rather than a new knob: with the prior on all eight deck cards carry a target,
  so the draw is uniform among them again — correct, since the starvation
  argument no longer distinguishes them.

## Phase D — measured, and NEGATIVE

Two arms resumed from the same checkpoint (ep 32,484, stage 3, phase mirror —
the one the deck-coverage work measured its dead cards on), bit-identical by
md5, isolated via absolute `CLASH_WEIGHTS`, 75 minutes each, differing only in
`CLASH_HUMAN_PRIOR_COEF`.

### D0 — pre-flight, 22/22 PASS

Side null 0.510 over 300 episodes (z = +0.35). C++ suite 701 cases, one
failed-as-expected, exit 0.

It caught a real defect in this work. `the gate declines on an empty board`
iterates ADVISOR_CARDS and calls `target_logits_for`, which now falls through to
the prior when a rule declines — so all 40 quiet states spoke and the check
failed. The invariant won: `logits_for` takes `obs` and declines on a quiet
board. Re-measured with the prior ON, all four advisor checks pass and coverage
on visited states rises 45.6% → 54.6%.

### D2 — the mechanism works, and is correctly aimed

| card | ΔH(place\|card) | ΔKL(policy‖prior) |
|---|---|---|
| Musketeer | +0.117 | **−0.800** |
| Ice Golem | +0.184 | **−0.548** |
| Ice Spirit | +0.192 | **−0.396** |
| Skeletons | +0.194 | **−0.381** |
| The Log | +0.143 | +0.152 |
| Cannon *(ruled)* | +0.054 | +0.030 |
| Hog *(ruled)* | +0.031 | −0.159 |
| Fireball *(ruled)* | −0.023 | +0.332 |

Four of five targeted cards moved decisively toward the prior; the three ruled
cards did not. No collapse — every modal share under 20% except Hog, already at
62% in BOTH arms. So the term is not inert and not misaimed.

### D3 — paired A/B, n = 800

Paired on bit-identical openings via `env.snapshot()`, greedy both sides,
opponent the C++ heuristic at 1.5x elixir.

| | |
|---|---|
| control | **0.3187** |
| treatment | **0.2462** |
| paired delta | **−0.0725**, 95% CI **[−0.1125, −0.0325]** |
| discordant | 112 better / 170 worse / 518 tied |
| exact sign test | **p = 0.00066** |

The in-training rolling win rate agreed independently: 0.36–0.39 treatment
against 0.42–0.44 control.

### Why, and what the Phase C defence got wrong

Phase C defended serving a state-independent target on the grounds that the rows
it covers were already getting the entropy bonus, which is equally
state-independent. **That defence does not survive this measurement.** The
entropy bonus is a WEAK, DIFFUSE push toward uniform that a trained policy
largely resists. A KL to a specific distribution is a STRONG, DIRECTED pull. The
swap was not neutral: it traded a state-DEPENDENT map the actor loss had learned
for a state-INDEPENDENT one, and "2.53x better than uniform" does not make a
state-independent target better than what the policy already had.

The D1 entropy table predicted the shape of this: the prior sits at 0.92–0.96 of
maximum entropy for Skeletons, Ice Spirit, Ice Golem and Musketeer — close
enough to uniform that pulling toward it is mostly just spreading, and spreading
a trained placement map costs win rate.

### What the p-value does and does not license

n = 800 is over EPISODES, which controls opening and opponent variance and makes
the delta for THESE TWO CHECKPOINTS precise. There is still only n = 1 TRAINING
RUN per arm, and this project measures the control arm's own across-run variance
at 0.570–0.775. So "the prior harms training" is supported by two independent
signals pointing the same way, not established — three or more arm pairs would
be needed to separate the treatment from run variance.

One confound points the safe way: the treatment ran MORE episodes (1,517 vs
1,437) under equal compute and still lost, so the mismatch works against the
treatment being secretly better.

### Standing recommendation

Leave `CLASH_HUMAN_PRIOR_COEF` at 0 — where it already shipped, so no rollback
was needed. The mined prior remains valuable as a measured artifact (2.53x
better than uniform at predicting real human placement), but serving it as a
placement TARGET is refuted at this coefficient. Anything further — a much
smaller coefficient, or gating it to the cards with real signal (The Log,
Musketeer) rather than all five — is a new experiment with its own
pre-registered criterion, not a rescue of this one.

## Known limitations

- **Musketeer loses 40%** of its placements (24.7% out-of-board, 15.7% illegal).
  It is the card 2.6 places deepest, so it is most exposed to the edge
  extrapolation. Its prior is biased shallow. 744 samples survive.
- One player, one deck, 2021 recordings, no Evolutions.
- The source dataset declares **no licence**. The cache is gitignored; only the
  derived 9 KB prior lives in the repo.
- The prior describes the real game's action space, not ours. 5.2% of human
  placements are illegal here — a floor on how well it can ever match.
