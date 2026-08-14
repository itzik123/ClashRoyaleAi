# Pre-registration — advisor-targeted coverage A/B

Written **before** looking at any engine-scored result, at 2026-08-14 ~21:05,
with both arms at ~62 updates. The point of writing it down is that this
project has twice nearly over-read a marginal number, and once manufactured a
result by continuing a run after seeing it.

## Arms

Byte-identical code, same seed checkpoint (`model_weights_hires.pth` weights on
a full training state at episode 64,309), same stage 5 / 1.5x opponent, same
4 envs, isolated working directories. The ONLY difference:

| arm | `CLASH_ADVISOR_COVERAGE_COEF` |
|---|---|
| control | 0 — v1.2.0's objective, coverage is a pure entropy bonus |
| treatment | 0.10 — coverage rows with an advisor rule get KL to its score map |

Stop at **80 updates each**, fixed in advance. No extension after seeing the
result: continuing a run because the number is marginal is optional stopping
and would manufacture significance.

## Primary measurement

`prove_placement.py --seed model_weights_hires.pth --control ... --treatment ...
--episodes 12`, which draws states from ONE reference policy so both arms face
identical boards, and scores by the engine's own accounting:

* **Fireball** — elixir value killed (`get_elixir_value_killed_by`)
* **Cannon** — tower HP preserved over a full 300-tick lifetime.
  NOT elixir-killed: that scores distraction, most of a building's job, at zero.

Paired bootstrap CI plus an exact sign test on discordant pairs.

## The two questions, and they are separate

1. **Does the entropy coverage term ERODE the distilled placement head?** This
   is the handoff's §3 Step 1 prediction, which it states has never been
   measured. Read the CONTROL against the SEED. If Fireball falls back toward
   the 0.346 random baseline, the conflict is real.
2. **Does the advisor target prevent that?** Read TREATMENT against CONTROL.

## Decision rule, fixed now

Ship the advisor term in the long run (`coef = 0.10`) if **both**:

* Fireball: treatment is not significantly WORSE than control, and
* Cannon: treatment is not significantly WORSE than control.

i.e. the bar is "does no harm", not "wins". Justification: the term's purpose
is to stop coverage flattening a distilled map, so its benefit accrues over a
long run, and 80 updates is chosen to be enough to detect HARM, not enough to
prove a long-horizon benefit. Claiming a win at this n would repeat exactly the
+0.105 exploratory result that a 4x-power confirmatory run later killed.

If either is significantly worse, the long run uses **coef = 0** and the result
is reported as negative.

## What will NOT be claimed either way

* Win rate. Neither arm is long enough, and the control's own win-rate variance
  across runs of this net is 0.570-0.775 — anything under a few hundred paired
  trials measures that, not the treatment.
* That the advisor term fixes the Cannon. Every prior measurement says the
  Cannon's exact cell is a bad supervision target because
  `building_score_map`'s top is an exact-tie plateau; the target here is the
  whole surface, which is the honest encoding of that indifference, but it is
  not a reason to expect exact-cell gains.
* Anything about the live game. No emulator ran this session.
