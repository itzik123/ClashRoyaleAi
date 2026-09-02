# Elixir phases — what landed, what was verified, what to watch

Real-game elixir schedule in the simulator: **1x, double from 2:00
(`DOUBLE_ELIXIR_TICK = 1200`), triple from 3:00 (`TRIPLE_ELIXIR_TICK = 1800`)**.
Built, verified and migrated on 2026-09-03. `UPSTREAM_REQUESTS.md` item 26
carries the full evidence and blast radius.

---

## Verified — every claim with the instrument that produced it

| claim | how | result |
|---|---|---|
| engine builds | MSBuild `clash_royale_env.vcxproj` | 0 warnings, 0 errors, post-build copy OK |
| the `.pyd` carries this engine | `tools/audit/verify_pyd.py` | `observation_size = 13977`, all checks OK |
| schedule correct at both boundaries | `tools/audit/elixir_phase_audit.cpp` | 1x/2x/3x, both sides of 1200 and 1800 |
| income really multiplied | same, measured **off the bar** | 0.7 / 1.4 / 2.1 per 20 ticks |
| evaluated per TICK, not per step | mid-window crossing | 1.085 = 9 @ 1x + 11 @ 2x |
| composes with `oppElixirMultiplier` | same | 1.5x opponent in double → 3.0x, team 0 unaffected |
| phase scalar symmetric across teams | same + `verify_pyd.py` | team 0 == team 1 at all three phases |
| **C++ suite** | `ClashRoyaleTests.exe` | **704 cases, 703 passed, 1 failed as expected, exit 0** |
| **Python suite** | `pytest python_ai/tests` | **729 passed, 2 skipped** |
| **perception suite** | `pytest perception/tests` | **389 passed, 1 skipped** |
| checkpoint migration | `migrate_checkpoint_elixir_phase` | 1 tensor + 2 Adam moments changed; encoder delta **exactly 0.0** |
| migrated net actually runs | 24 episodes vs the teacher at stage 3 | ran clean, 4,162 decisions |
| replay format | generated a 2001-tick replay | `elixirPhases` block written; 1200/600/201 frames per phase |
| viewer badge | driven in-browser on **engine output** | x1→x2 at 1200, x2→x3 at 1800 |
| viewer back-compat | replay with no `elixirPhases` | badge hidden, not defaulted to x1 |

The C++ suite's single failure is `test_navigation_wedge.cpp`'s `[!shouldfail]`
case pinning the open collision-wedge defect — that is the documented
invariant, not a regression.

## Checkpoint

`python_ai/model_weights_phase5.pth` — migrated from `model_weights_phase4.pth`
(ep 32,484). **Resume from this, not from phase4.** `scalar_mlp.extra`
`Linear(9,12) → Linear(10,12)`, zero-padded: 120 of 1,900,165 parameters, and
the migrated net is bit-identical to the original on every observation it could
already see. Adam's `exp_avg`/`exp_avg_sq` for that tensor were padded too —
without that, `optimizer.load_state_dict` throws on resume.

`model_weights_phase4.pth` and `.ep32484.prelaunch.bak.pth` are untouched.

## Two things found on the way that were not part of the plan

**`MAX_MATCH_ELIXIR` 140 → 280.** Sized against a flat-1x match
(`3600 × 0.035 + 5 = 131`); phased income reaches ~278. At 140 both elixir-spend
scalars would have saturated at 1.0 partway through every match and stayed
there — the agent going blind to the economy exactly when it decides the game,
with nothing raising. Same failure class as the constants the 2026-08-07 speed
fix invalidated.

**The old aux-task finding is no longer true as written.**
`test_aux_task_is_not_a_memory_probe.py` documented that the deleted
opponent-elixir head measured nothing, because the target is an affine function
of two present scalars (MAE 0.0000). Phases broke both halves: `rate*t` is no
longer one slope (original basis now scores 1.4265, no better than
predict-the-mean), and the opponent now **overflows unaided**, which is
genuinely unrecoverable because the cap discards elixir no scalar records. The
file's own closing caveat predicted exactly this. Its basis was updated, the
conclusion survives in the overflow-free regime, and two tests were added
pinning the new facts.

## What is invalidated

- **Every win rate in CLAUDE.md's "Measured baselines."** The economy is the
  substrate every card's value sits on.
- **The curriculum's win-rate gates**, calibrated against a teacher that now
  plays a materially different late game. Expect stage transitions at different
  episodes — that is the change working, not a regression.
- The critic, transitionally. Early movement in `Loss/Critic` on a resumed run
  is expected.

Not invalidated: the trained `extra` columns, the LSTM, the trunk, both heads,
the action space, and the placement head's learned distribution.

## What to watch, and against what

Measured before and after with the **same policy**, so the engine is the only
variable — the honest prediction is narrower than the hypothesis that motivated
the change:

| | flat 1x | 1x/2x/3x |
|---|---|---|
| enemy units on board, mean | 3.62 | 3.89 |
| best Fireball catch, **median** | **1** | **1** |
| P(catch ≥ 3) | 15.5% | 21.8% |
| P(catch ≥ 4) | 6.1% | 10.4% |
| agent elixir, mean | 2.16 | 2.50 |
| agent P(≥ 9.0) | 0.0% | 0.6% |

- **Fireball take-up** (`python_ai/eval/probe_card_usage.py`) should rise —
  because a Musketeer trade is available more often, not because clumps
  appeared. The median best-case catch is still one unit.
- **`W_ELIXIR_OVERFLOW` now fires.** It never had. If the bars start pooling at
  10.0, income has stopped being the binding constraint and the premise this
  whole change rests on no longer holds — that is the measurement to re-run
  before proposing any further economy change.
- **`AvgTicks`.** Matches shortened slightly (1758 → 1729 mean). The 32.1% /
  3.0% share-of-play split that justified the schedule is itself a function of
  match length and will drift as the policy retrains.
- **Triple elixir is thin.** Median match end is ~1791 ticks, i.e. just under
  3:00, so barely half of matches reach triple at all and it covers a small
  share of played ticks. It is correct as fidelity; do not expect it to carry
  the effect.

## Unrelated, but you should see it

The **15:44 run collapsed** (`runs/phase4/train-20260902-deck-revival.log`):
win rate 0.60 → 0.11 across 180 episodes, entropy *rising* 0.84 → 1.19 while
the coefficient *decayed* 0.098 → 0.037, `NextCardCE` degrading 1.65 → 1.84,
and a second curriculum demotion. Everything moved together, which reads as
divergence rather than exploration. The 19:27 relaunch ended after ~47 minutes
without writing a log or a checkpoint.

**Diagnose that before attributing anything to the elixir change** — otherwise
the next collapse gets blamed on phases. `model_weights_phase4.pth` (8/29)
predates it, which is why it was the migration source.
