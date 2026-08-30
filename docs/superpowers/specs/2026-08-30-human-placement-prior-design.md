# The human placement prior

**Status:** Phases A and B complete and measured. Phase C (integration into the
training path) specified but NOT built — deliberately gated on the result below.

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

## Phase C — integration (specified, NOT built)

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

## Phase D — measurement (specified)

1. `tools/validate_pipeline.py` 20/20; side null at 0.50.
2. Per-card **modal share**, not `ByCard_Min` — CLAUDE.md is explicit that
   entropy flags the healthiest card and clears the dead ones.
3. Paired win-rate A/B via `eval/stats.py`, **n >= 800**; the control arm's own
   variance is +/-0.06.

## Known limitations

- **Musketeer loses 40%** of its placements (24.7% out-of-board, 15.7% illegal).
  It is the card 2.6 places deepest, so it is most exposed to the edge
  extrapolation. Its prior is biased shallow. 744 samples survive.
- One player, one deck, 2021 recordings, no Evolutions.
- The source dataset declares **no licence**. The cache is gitignored; only the
  derived 9 KB prior lives in the repo.
- The prior describes the real game's action space, not ours. 5.2% of human
  placements are illegal here — a floor on how well it can ever match.
