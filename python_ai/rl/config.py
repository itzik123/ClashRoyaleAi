"""Every hyperparameter of the PPO loop, in one place, with its justification.

Both pipelines used to declare these as ~200 lines of locals inside their own
`train_*_ppo()` function, byte-identical apart from three entropy numbers. That
had two costs beyond the duplication: nothing could read a hyperparameter
without running a training loop (so no test could pin one), and the two copies
were free to drift -- which is exactly how `spell_value_weight` stayed dead code
for a whole training era.

`PPOConfig` is deliberately frozen. A trainer that mutated its own config
mid-run would make a checkpoint's `episodes_completed` meaningless as a
description of what produced it.
"""
import math
import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PPOConfig:
    """The algorithm's own settings. Identical in both pipelines by design --
    same architecture, same algorithm, only the opponent differs."""

    #: 8 workers + 1 CPU-bound main process (no CUDA here, so the update step
    #: itself needs real cores too) comfortably fits 12 logical processors.
    #: Doubling from 4 halved the variance of the per-rollout advantage
    #: normalization and the critic's return targets.
    #:
    #: Overridable so several arms of a controlled experiment fit on one machine
    #: at once. EVERY ARM MUST USE THE SAME VALUE -- it sets the rollout batch
    #: width B, which changes advantage-normalization variance and the number of
    #: BPTT segments per minibatch.
    num_envs: int = int(os.environ.get("CLASH_NUM_ENVS", 8))

    #: THE DISCOUNT MUST OUTLAST THE MATCH. Raised 0.99 -> 0.999 on 2026-08-27.
    #:
    #: `1/(1-gamma)` is the effective horizon in DECISIONS, and a decision here
    #: is one second (`skip_frames=10` at 10 ticks/s). A full match is
    #: `max_ticks/skip_frames` = 3600/10 = **360 decisions**. At 0.99 the
    #: horizon was 100 -- barely a quarter of the game -- so every TERMINAL
    #: quantity decayed to `0.99^360 = 0.0268` of its face value by the time it
    #: had to compete with an immediate shaping term:
    #:
    #:     win  (+1)              -> 0.0268 at episode start
    #:     DRAW_PENALTY (1.0)     -> 0.0268   (raised 0.2->1.0 to stop stalling)
    #:     W_TOWER_DESTROYED      -> 0.6      UNDISCOUNTED, fires mid-episode
    #:
    #: i.e. one crown was worth **22.4x** winning the game it was supposed to
    #: serve. Measured 2026-08-27 over 10 greedy episodes of
    #: `model_weights_phase1_v5.pth`: the terminal reward contributed **0.0195**
    #: of the discounted return, so ~98% of the objective was shaping. A
    #: sacrifice play -- concede tower HP now, win later -- costs 0.5 at once
    #: and repays 0.027 at the end, so gradient ascent on this objective can
    #: never find one. That is the strategic ceiling, in one constant.
    #:
    #: WHICH TERMS WERE ACTUALLY AT FAULT, because "shaping drowned the win" is
    #: too coarse and points at the wrong half. The POTENTIAL-BASED terms were
    #: never the problem: a PBRS term telescopes over an episode to
    #: `gamma^T*Phi(s_T) - Phi(s_0)`, so it carries the SAME `gamma^T` factor
    #: the terminal reward does and stays in fixed proportion to it at any
    #: discount. The tower potential was ~2:1 against a win at gamma 0.99 and
    #: is ~2:1 against a win at 0.999.
    #:
    #: What dwarfed the outcome was the terms that are deliberately NOT
    #: policy-invariant, because those do not carry the factor:
    #: `W_TOWER_DESTROYED` (undiscounted, fires mid-episode), the one-sided
    #: elixir-trade term, and the per-step overflow penalty. So the fix is the
    #: horizon, and the PBRS terms need no compensating retune -- only the
    #: explicit biases should be re-examined against the restored win value.
    #:
    #: WHY IT DRIFTED, rather than having been wrong when written. `weights.py`
    #: still says W_TOWER_DESTROYED is "set above the discounted value of a win
    #: (~0.28 at these episode lengths)", and at the 112-decision episodes of
    #: that era 0.99^112 = 0.32 made 0.6 a deliberate ~2x bias. The 2026-08-07
    #: movement-speed fix tripled match length -- CLAUDE.md's own warning is
    #: "no win rate below survives it" -- and the reward constants were never
    #: re-derived against it. Same failure class as the four stale arena copies:
    #: a constant calibrated against a measurement a later change invalidated.
    #:
    #: WHY 0.999 IS CHEAP HERE, which is not obvious and was measured rather
    #: than assumed. The usual objection is variance: a longer horizon means
    #: noisier returns and a harder critic. It does not apply at this lambda.
    #: GAE's advantage estimator has its own effective lookahead of
    #: `1/(1-gamma*lambda)`, and `gae_lambda=0.9` already pins that near 10
    #: steps -- 9.17 at gamma 0.99, 9.91 at 0.999. Measured over 6 real
    #: episodes (mean 322 decisions), the critic's target barely moves:
    #:
    #:     gamma    mean|return|   std(return)   max|return|
    #:     0.990       0.385          0.362         1.719
    #:     0.999       0.416 (+8%)    0.384 (+6%)   1.719 (unchanged)
    #:
    #: So this buys a 26x increase in the weight of the actual outcome
    #: (0.0268 -> 0.6976) for ~6% more return spread.
    #:
    #: NO ANNEALING, DELIBERATELY. OpenAI Five ramped gamma upward precisely to
    #: control early variance; the table above shows there is no early variance
    #: problem to control here, because lambda already does that job. Adding a
    #: schedule would be machinery bought against a cost that was measured and
    #: found absent.
    #:
    #: GAMEPLAY-AFFECTING IN THE SENSE THAT MATTERS: this changes the OBJECTIVE,
    #: so every win rate in CLAUDE.md's "Measured baselines" was earned against
    #: a different one and is not comparable across this change. Checkpoints
    #: still LOAD (no architecture moved) but their critic was fitted to the old
    #: discount and will need to re-converge; expect Loss/Critic and explained
    #: variance to move early in a resumed run and do not read that as a fault.
    #: `CLASH_GAMMA` overrides it, so the previous behaviour is one env var
    #: away for anyone who needs the old objective to reproduce an old number.
    #:
    #: Pinned by `tests/test_reward_horizon_invariant.py`, which asserts the
    #: RELATIONSHIP (horizon >= match length; crown not >3x a win) rather than
    #: this particular value -- so retuning either side is free, and letting
    #: them drift apart again is not.
    gamma: float = float(os.environ.get("CLASH_GAMMA", 0.999))

    #: Lowered from 0.95: every prior fix (num_envs, value clipping, entropy
    #: decay, reward shaping) left Loss/Critic on the same noisy, non-decreasing
    #: plateau across ~2650 episodes. High lambda leans GAE on multi-step
    #: Monte-Carlo-style returns instead of the value bootstrap; with a critic
    #: that is not converging that keeps the ADVANTAGE TARGET itself noisy no
    #: matter how the critic's own update is regularized -- which is why
    #: clipping the critic's movement alone did not help.
    gae_lambda: float = 0.9

    eps_clip: float = 0.2
    lr: float = 3e-4

    #: Steps collected PER ENVIRONMENT before an update.
    update_timestep: int = 500

    #: --- Truncated BPTT ----------------------------------------------------
    #: The update used to replay each env's WHOLE 500-step rollout through the
    #: LSTM as one sequence, with num_minibatches=1 and ppo_epochs=2 -- exactly
    #: 2 optimizer steps per 4,000 collected transitions. Two measured
    #: consequences:
    #:   * Loss/Clip_Fraction was 0.0000 across ALL 228 updates of an 18,740-
    #:     episode run. Structural, not a plateau: epoch 0 evaluates the policy
    #:     that generated the data, so its ratio is exactly 1 by construction,
    #:     leaving a single Adam step before epoch 1.
    #:   * Profiling put ~5.0s of the ~7s epoch in the backward pass alone,
    #:     i.e. 79%, all of it unrolling one 500-long chain.
    #:
    #: Each (chunk, env) pair becomes an independent training segment starting
    #: from the hidden state actually stored at that timestep during the rollout
    #: -- the standard stored-state approach for recurrent PPO. 500/25 = 20
    #: chunks x 8 envs = 160 segments per rollout, so a minibatch is a real
    #: batch instead of 8 sequences. FASTER as well as more thorough:
    #: sequential LSTMCell calls per epoch drop from 500 to 8 x 25 = 200.
    #: --- THE HORIZON MUST OUTLAST A CARD ROTATION (2026-08-27) ------------
    #: 25 -> 50. `bptt_chunk` is the CREDIT-ASSIGNMENT horizon: gradients stop
    #: at the chunk boundary, so an action whose payoff arrives later than
    #: `bptt_chunk` decisions cannot be reinforced at all.
    #:
    #: A card rotation -- playing the other four cards to cycle a given one back
    #: -- is, derived from the engine rather than assumed (DEFAULT_DECK's costs
    #: via get_card_info, and the elixir rate MEASURED off a live env at 28.571
    #: ticks/elixir):
    #:
    #:     4 cards x 2.625 elixir x 28.571 ticks = 300 ticks = 30 DECISIONS
    #:
    #: at skip_frames=10. So the old 25 truncated the gradient BEFORE one
    #: rotation completed. That is the same shape of defect as the gamma
    #: correction made the same day: the observation had just been extended
    #: with the opponent's `seen[]`/`recency[]` (item 24) to make card counting
    #: possible, and no gradient path reached far enough to learn from it. The
    #: information arrived; the credit path did not.
    #:
    #: WHY IT IS NEARLY FREE. `update_timestep` and `num_minibatches` are
    #: unchanged, so the segment count halves as the segment length doubles and
    #: a minibatch still pushes exactly 500 flat rows through the trunk -- 84%
    #: of the update. Only the LSTM's sequential loop changes shape, 25 calls
    #: at batch 20 becoming 50 at batch 10, and that loop is 16% of the update.
    #: Measured end-to-end: 1.090x per PPO update. L=100 costs 1.227x and is
    #: refused on the batch-width guard below, not on wall clock.
    #:
    #: WHAT IT COSTS THAT IS NOT WALL CLOCK. Segments per minibatch fall 20 ->
    #: 10, and those segments are the batch dimension of every gradient
    #: estimate, so it gets noisier while the number of optimizer steps per
    #: rollout (32) stays the same. That is the real trade. It is bounded by
    #: test_bptt_credit_horizon.py, which refuses fewer than 8 segments per
    #: minibatch -- the thing that stops the horizon being pushed to 250
    #: "because gradients are good" and landing back in the one-long-sequence
    #: regime truncated BPTT was introduced to escape.
    #:
    #: Must divide `update_timestep` (validated in __post_init__), so the
    #: available rungs are 50, 100, 125, 250, 500 -- not arbitrary.
    bptt_chunk: int = int(os.environ.get("CLASH_BPTT_CHUNK", 50))
    ppo_epochs: int = 4

    #: 160 segments / 8 = 20 segments per minibatch, 8 optimizer steps per
    #: epoch, 32 per rollout -- 16x the previous 2.
    num_minibatches: int = 8

    max_grad_norm: float = 0.5

    #: --- Value-clip range, scaled to the RETURN distribution ---------------
    #: This used to reuse eps_clip (0.2) directly. That number bounds the
    #: POLICY's probability RATIO -- a dimensionless quantity -- and reusing it
    #: as an ABSOLUTE bound on how far the critic may move is a unit mismatch.
    #:
    #: Measured on a live 500-step rollout at episode 14,666 (stage 3):
    #:   GAE return std 0.530 | |return - V| median 0.181, p90 0.539
    #:   46.1% of samples needed the critic to move MORE than 0.2
    #:
    #: i.e. on nearly half the batch the critic was forbidden from correcting
    #: its own error in one update. Expressed as a fraction of the batch's own
    #: return spread instead, so it stays correctly scaled if the reward shaping
    #: is ever retuned. Floored at eps_clip so it can never become TIGHTER than
    #: the old behaviour.
    vf_clip_std_frac: float = 1.0

    #: --- Auxiliary task: opponent elixir estimation ------------------------
    #: Weight on the loss that trains MicroRoyaleNet.predict_opp_elixir. The
    #: head predicts the opponent's CURRENT elixir -- hidden information,
    #: deliberately absent from the observation -- from elapsed time and both
    #: sides' cumulative spend, which ARE in it.
    #:
    #: Its gradient flows back into the shared LSTM/CNN trunk, and that is the
    #: entire point: this is representation shaping, not an extra output. The
    #: sparse win/loss signal gives the recurrent state almost no reason to
    #: integrate opponent spending over a whole match, and "how much elixir do
    #: they have right now" is the single most load-bearing latent variable in
    #: the game.
    #:
    #: 0.5 makes the term a real but minority contributor: the target is in
    #: elixir units (0-10) and a well-fit head sits around 1.0-1.5 MAE, i.e. an
    #: MSE of ~1-2, against an actor loss of order 0.1.
    aux_elixir_coef: float = 0.5

    #: Converts the elixir-unit MSE into the same numeric range as the other
    #: loss terms. Kept explicit (rather than folded into aux_elixir_coef) so
    #: the LOGGED diagnostic stays in interpretable elixir units.
    aux_elixir_scale: float = 0.02

    #: Periodic-checkpoint interval, in episodes. Overridable ONLY so a short
    #: controlled run produces matched artifacts: a resume sets last_save_ep to
    #: the resumed episode, so at the 500 default an experiment shorter than 500
    #: episodes finishes having written nothing at all -- and `timeout` kills the
    #: process before the end-of-loop save, so the whole run is unmeasurable.
    #: Leave unset for real runs.
    save_every_episodes: int = int(os.environ.get("CLASH_SAVE_EVERY", 500))

    #: Episodes between demo replays.
    replay_every_episodes: int = 1000

    #: Run-level RNG seed, or None for fresh OS entropy. `CLASH_SEED=<int>`.
    #:
    #: DEFAULT None ON PURPOSE. Seeding by default would silently change what
    #: every existing configuration does, and every win rate recorded in
    #: CLAUDE.md was earned unseeded. Set it for an A/B arm, a reproduction, or
    #: any run whose result someone will have to defend -- see rl/seeding.py
    #: for what one seed does and does not pin.
    seed: Optional[int] = (int(os.environ["CLASH_SEED"])
                           if os.environ.get("CLASH_SEED") else None)

    def __post_init__(self):
        if self.update_timestep % self.bptt_chunk:
            raise ValueError(
                "update_timestep must be divisible by bptt_chunk so every "
                f"chunk is full-length (got {self.update_timestep} / "
                f"{self.bptt_chunk})")

    @property
    def segments_per_rollout(self):
        return (self.update_timestep // self.bptt_chunk) * self.num_envs


@dataclass(frozen=True)
class EntropyConfig:
    """Targets and gains for the adaptive per-head entropy controller.

    Replaces hand-tuned fixed coefficients. Two runs showed why fixed values do
    not work here -- the heads are COUPLED, so correcting one breaks the other:

      card scale 2.0 -> card head collapsed to 10% of its max entropy
      card scale 4.0 -> card recovered to ~35%, but placement fell from H=4.38
                        to H=2.63 at the same stage (40 -> 20 cells, top-5 share
                        36% -> 70%, left lane 33% -> 13%)

    So instead of picking coefficients, pick the ENTROPY LEVEL each head should
    hold and let a controller find the coefficient -- the same idea as SAC's
    automatic temperature tuning.

    THE TWO PIPELINES DELIBERATELY DIFFER, and the differences are all measured
    (see `PHASE1` and `PHASE2` below). Nothing here unifies them: doing so would
    be gameplay-affecting in whichever pipeline moved.
    """

    #: Deliberately FIXED, never annealed. The measured failure mode for this
    #: head is the opposite one: card entropy 0.10 collapsed the policy to 5 of
    #: 8 cards. Narrowing card choice is the known danger.
    target_card: float = 0.35

    #: ANNEALED. With the target pinned at 0.65 for a whole run the placement
    #: coefficient rose monotonically (0.1286 -> 0.1476 across 50,000 self-play
    #: episodes) -- the policy was trying to sharpen its placement the entire
    #: time and the controller kept forcing it back open. Wide while the policy
    #: is still discovering where things go, tight once it is refining.
    #: 0.25 * log(612) = 1.60 nats ~= 5 effective cells: committed, but not a
    #: collapsed point mass. Deliberately NOT annealed to 0 -- some placement
    #: noise is genuinely correct in a game with a live opponent.
    target_placement_start: float = 0.65
    target_placement_final: float = 0.25

    #: Sized to one long run on this machine (~60k episodes in phase 1, ~50k in
    #: phase 2). Past the horizon the target simply stays at FINAL.
    anneal_episodes: int = 60000

    #: Multiplicative control on the normalized entropy fraction:
    #:     coef *= exp(rate * (target - measured))
    #: Reverted 0.15 -> 0.5 after a matched-depth measurement contradicted the
    #: earlier reasoning. The lower gain DID smooth the controller, but the
    #: resulting policy measured WORSE at stage 3: 81.7% [74-88] against
    #: 96.7% [92-99] for the high-gain run, CIs not overlapping. The likely
    #: mechanism, unproven: the large swings act as periodic exploration
    #: re-boosts. CAVEAT: one run per configuration.
    adapt_rate_card: float = 0.5

    #: The placement head gets its OWN gain where the pipeline sets one, because
    #: on 2026-08-11 a rate of 0.5 drove it into windup and dissolved the
    #: policy. The reversion argument above is VOID for this head: it was
    #: measured against a placement-entropy signal that was ~85% no-op steps,
    #: nearly constant at 0.85-0.97, which the controller barely had to act on.
    #: Correcting the signal made it far more responsive and a gain tuned on the
    #: numbed version overreacted -- 0.0100 -> 0.433 over ~50 updates, six of
    #: eight cards at 0.93-1.00 of MAXIMUM placement entropy, ROI 0.96 -> 0.87,
    #: win rate 0.67 -> 0.51.
    #:
    #: The general lesson, worth more than the number: FIXING A SENSOR
    #: INVALIDATES ANY GAIN TUNED AGAINST THE BROKEN ONE.
    adapt_rate_placement: float = 0.5

    #: Hard cap on how far ONE update may move a coefficient, independent of
    #: gain or error size. The gain addresses the cause; this addresses the
    #: failure MODE, so no future retune of a target or a measurement can
    #: compound into a 40x excursion again. None disables it.
    coef_step_max: Optional[float] = None

    #: Raised 0.002 -> 0.01. Healthy measured coefficients are 0.05-0.22, so
    #: 0.002 was not a floor but an off switch: once there the entropy term
    #: stopped opposing the policy gradient at all and the head was free to
    #: collapse until the controller noticed.
    coef_floor: float = 0.01
    coef_ceil_card: float = 0.5

    #: Separate, much lower ceiling for the placement head where a pipeline sets
    #: one: 0.433 already dissolved the policy, so 0.5 was never a safety net.
    #: Deliberately NOT applied to the card head -- that controller is doing the
    #: right thing (card entropy measured 0.084 of max against a 0.35 target,
    #: a real collapse to ~5 of 8 cards) and a global cut would throttle the one
    #: controller that is working.
    coef_ceil_placement: float = 0.5

    #: Seeded at the last hand-tuned effective values so the controller starts
    #: from a known-reasonable point rather than hunting from zero.
    initial_coef_card: float = 0.05
    initial_coef_placement: float = 0.06


#: Pipeline 1 (`trainers/train.py`): the historical configuration, unchanged.
#: One gain, one ceiling, no per-update step cap.
PHASE1_ENTROPY = EntropyConfig()

#: Pipeline 2 (`trainers/train_selfplay.py`): starts the placement anneal lower
#: (0.50, since a phase-2 policy is already past the discovery stage) and adds
#: the three guards the 2026-08-11 dissolution forced.
PHASE2_ENTROPY = EntropyConfig(
    target_placement_start=0.50,
    adapt_rate_placement=0.10,
    coef_step_max=0.10,
    coef_ceil_placement=0.20,
)


def log_reachable(n):
    """log(n) with n floored at 2.

    The floor only guards log(1) = 0: a row with a single legal arm carries zero
    entropy and is excluded from every average by the decision mask anyway.
    """
    return math.log(max(2, n))
