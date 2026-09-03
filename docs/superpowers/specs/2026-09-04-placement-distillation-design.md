# Teaching the placement head to aim: search distillation on the deck pool

**Date:** 2026-09-04
**Status:** approved, pending implementation
**Resumes from:** `model_weights_phase6.pth` — ep 32,875, rung 2, `teacher_table_size = 11`

---

## The problem this addresses

The 2026-09-03 deck pool made Cannon / The Log / Fireball strategically valuable
without making the policy any better at using them. Measured over 490 episodes
(`46bd95`), usage marginals rose with real signal while the state-conditional
ratios did not, and **placement quality did not move at all** — Fireball
0.685 → 0.685, signal 0.0. The 2026-08-29 autopsy had already put the binding
constraint on placement rather than on card selection; the pool did not touch it.

The card head is cheap to move and the placement head is not, so PPO on a short
timescale necessarily produces that shape: more usage, same aiming. The lever
this project has actually measured for placement is decision-time search and
expert iteration, and neither has ever been pointed at these two cards.

## The gate, and what it settled

Distilling an expert that cannot aim is not neutral — CLAUDE.md records that
more data from a WEAK expert actively degrades selectivity (ratio 3.07 → 2.31 →
2.17 as p0 outruns p1). So this was gated on a measurement before any training
was committed.

`python_ai/eval/probe_placement_oracle.py` (new, this session). On the 32
highest-opportunity naturally-occurring states per card, every arm PAID through
the engine and scored by the engine as net tower HP conceded over 20 decisions:

**Both distributions are reported, because they disagree and the disagreement is
the finding.** The first run passed `opponent_kind`, a key `gym_wrapper` does
not read, so it silently fell back to `"builtin"` and measured the 2.6 mirror vs
the C++ heuristic — the very distribution this spec criticises `expert_collect`
for using. Re-run on the teacher + 16-deck pool:

| card | on the MIRROR *(wrong dist)* | on the POOL *(correct dist)* | capture |
|---|---|---|---|
| Cannon *(control)* | +124 HP, p = 1.0 | **+454 HP, p = 0.043** | 30% → **69%** |
| **The Log** | +110 HP, p = 0.035 | **+768 HP, p = 0.0026** | 64% → **85%** |
| Fireball | +145 HP, p = 0.0039 | +173 HP, **p = 0.51** | 72% → 56% |

The identical-arms control read exactly 0.0 in every run, so `snapshot()`
carries the opponent's RNG and the pairing is real rather than assumed.

**THE LOG IS THE TARGET.** +768 HP over its own argmax, 17 better / 3 worse,
capturing 85% of the oracle — and it is the only card significant on BOTH
distributions, which is the one result here worth betting on.

**Fireball is NOT, and that is consistent rather than surprising.** It flips to
p = 0.51 on the pool with 23 of 32 states TIED — search mostly agrees with the
head. Its `place_q` is **0.675**: the head already collects two thirds of the
available value, so there is little headroom to teach. The Log's is **0.203**,
and there is a great deal. One number for "the dead cards" hid that they have
different problems, exactly as the 2026-09-03 autopsy warned.

**Cannon becomes significant on the pool** (+454, 69% of oracle), which is much
closer to `1b77f27`'s story and partially vindicates that measurement — its
apparent failure on the mirror was the distribution, not the harness.

Two caveats that belong with these numbers:

- **This harness is still not a reproduction of `1b77f27`** — that used 14
  injected scenario states, one-sided scoring and the ep-32,484 checkpoint. The
  agreement in DIRECTION is what is being claimed, not the 81% itself.
- **n = 32 with many ties is underpowered**, and the two distributions disagree
  on Fireball in opposite directions. Do not read either Fireball number as
  settled; read The Log's, which survives both.

**A second finding, deliberately not acted on yet:** the critic captures only
64–72% of headroom the engine finds, and the engine is ~150× cheaper per
evaluation than the network. An engine-scored expert is strictly better and is
the obvious next lever — but changing the label distribution and the scorer in
the same run is the two-axes error that collapsed the 2026-09-03 resume.

---

## Design

### 1. Point expert collection at the training distribution

`trainers/expert_collect.py::make_env` builds a raw `ClashRoyaleEnv` with
`DEFAULT_DECK` on **both** sides against the C++ `HeuristicOpponent` at 1.5×
elixir. That is the 2.6 mirror — the matchup that `measure_deck_matchups.py`
ranks **16th of 16** on opportunity for all three of these cards, offering 323 HP
of Fireball catch against a pool median of 494 and 145 of Log catch against 273.

Collecting there would gather labels on precisely the distribution the deck pool
was built to escape, and the states where these two cards matter would barely
appear. Collection must run through `envs/gym_wrapper.MicroRoyaleEnv` with
`opponent_kind="teacher"`, the live rung, and the deck pool on.

This is the single highest-risk defect in the plan: it is silent. A run collected
on the mirror would train, report plausible losses, and teach aiming for boards
the agent no longer sees.

### 2. One DAgger round, distribution labels, frozen trunk

Reuse the measured recipe with no changes to its arithmetic: `--train-dist`,
`freeze_trunk`, temperature solved per round rather than fixed. That is the arm
carrying +0.1733 conditional lift and +0.045 win rate (p = 0.0074, n = 1600).

Distribution labels are load-bearing *for placement specifically*: hard labels
train placement on ~10.3% of rows, while every row with ≥ 2 candidates
contributes placement gradient under distribution labels — 32.6% of rows.

Widening is ON, since it is what the gate measured. Note the consequence
`1b77f27` already recorded: candidates per decision go 2.6 → 38.1 mean, and
48.3% of decisions exceed `expert_collect`'s `K_MAX = 8`. Overflow is kept BY
SCORE rather than truncated, so collection stays correct, but a wide run records
only its best 8 candidates per row.

The output is a **seeded checkpoint, not a finished policy.**

### 3. Build and commit the frozen state bank

`probe_card_discrimination.py` scores checkpoints against a frozen bank of
observation SEQUENCES, replayed through the LSTM so each checkpoint builds its
own hidden state. No bank was ever committed, so the previous session's
`quality_hi` 0.685 / 0.217 and the hi/lo ratios cannot be extended — only
re-based.

Build one from phase6 at rung 2, commit it at
`python_ai/eval/banks/bank_phase6_rung2.npz`, and treat today's reading as the
new baseline. Committing is the point: it is what stops the next session facing
this same discontinuity.

### 4. A read-only trend daemon

Score each new checkpoint against the bank and append to a CSV. Deliberately NOT
wired into `train.py`: replaying the bank through the LSTM inside the training
loop would stall the run, and `train.py` is the live script. Follows
`tools/monitor_run.py`'s existing pattern — read the checkpoint and the event
file, never the trainer's memory, so it cannot perturb what it watches.

### 5. Resume PPO from the distilled checkpoint

Rung 2, deck pool on, ep 32,875 forward. **Rung 2 is not a fresh choice** — it is
what the 2026-09-03 measurement established, where the same checkpoint holds
~0.56 and mean episode reward goes −2.19 → +1.46, against two independent arms
that collapsed to 0.00 win rate within 130 episodes when resumed at rung 6.

The question the run answers is not "does distillation work" — the bank answers
that before a single PPO step. It is **does the distilled placement survive
contact with PPO**, which is unmeasured and is what hours of wall clock can
actually settle.

---

## Instrumentation

Four quantities, each with the baseline that makes it readable.

| quantity | where | read against |
|---|---|---|
| `p_hi` / `p_lo` / ratio, per card | bank probe → CSV | its own value at ep 32,875; a marginal that rises with a FLAT ratio is systemic drift, not learning |
| `quality_hi` (placement quality) | bank probe → CSV | the same; this is the number 490 episodes did not move |
| `Aux/NextCard_CE` | `ppo.py` → TensorBoard | **`ln(185) = 5.22`** is uniform; the stale-head failure read 13.76, i.e. confidently wrong. Must stay below uniform |
| `Decks/WinRate_{Min,Spread}`, per deck | `train.py` → TensorBoard | `POOL_WINRATE_FLOOR = 0.20`; a deck below it should lose episode share and climb back on its own |

Two reading rules carried from CLAUDE.md, because both have already caused a
wrong call in this project:

- **A marginal is not a conditional.** Usage rising with a flat hi/lo ratio is
  the failure mode, not the win. This has now been recorded six times.
- **Signal before trend.** `|delta|` over step-to-step scatter across
  checkpoints; below ~2 a movement is not separable from PPO jitter. Fireball's
  ratio "gain" last session was 0.8 and a three-point read of it looked clean.

---

## What would falsify this

Stated in advance so the run cannot be read charitably after the fact.

- `quality_hi` does not move on the bank **immediately after distillation**, before
  any PPO. The gate says search finds better cells; if distillation cannot
  transfer that, the mechanism is broken and nothing downstream matters.
- `quality_hi` moves after distillation and then **decays back** over PPO
  episodes. That is the finding — PPO washes out distilled placement — and it
  argues for in-loop distillation rather than a one-shot round.
- `p_lo` rises alongside `p_hi` with the ratio flat. Marginal drift again, now
  bought with search instead of with the pool.
- Win rate falls below `PLATEAU_MIN_WIN_RATE = 0.40` and stays there. The
  curriculum's valves are calibrated on the pool being winnable at this rung.

## Explicitly out of scope

- **The engine-scored expert.** Measured as better (the critic captures 64–72%),
  deferred so that only one axis moves. Next lever if this plateaus.
- **In-loop search during rollout.** Costs 2.2×+ wall clock before widening and is
  the unvalidated path; the +0.045 came from offline distribution distillation.
- **Any C++ change.** Nothing here needs one, and it would invalidate the
  checkpoint lineage mid-experiment.
