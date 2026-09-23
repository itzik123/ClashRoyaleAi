"""Every hyperparameter of the PPO loop, in one place.

`PPOConfig` is frozen: a trainer that mutated its config mid-run would make a
checkpoint's `episodes_completed` meaningless as a description of what produced
it.
"""
import math
import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PPOConfig:
    """The algorithm's settings, identical in both pipelines; only the opponent
    differs.
    """

    #: 8 workers plus the CPU-bound main process fit 12 logical processors.
    #: Every arm of an experiment must use the same value: it sets the rollout
    #: batch width, and with it advantage-normalization variance and the
    #: segment count per minibatch.
    num_envs: int = int(os.environ.get("CLASH_NUM_ENVS", 8))

    #: The horizon 1/(1-gamma), in decisions, must outlast a match: 3600 ticks
    #: / skip_frames 10 = 360 decisions. At 0.99 a win was worth 0.99^360 =
    #: 0.027 at episode start, and ~98% of the objective was shaping. Cheap
    #: here because gae_lambda already keeps the advantage lookahead near 10
    #: steps. tests/test_reward_horizon_invariant.py pins the relationship, not
    #: the value.
    gamma: float = float(os.environ.get("CLASH_GAMMA", 0.999))

    #: Lower lambda leans GAE on the value bootstrap; at 0.95 the advantage
    #: target stayed noisy however the critic was regularized.
    gae_lambda: float = 0.9

    eps_clip: float = 0.2
    lr: float = 3e-4

    #: Steps collected per environment before an update.
    update_timestep: int = 500

    #: Truncated BPTT: each (chunk, env) pair is a training segment starting
    #: from the hidden state stored during the rollout. The chunk is also the
    #: credit horizon, and must outlast one card rotation (4 cards x 2.625
    #: elixir x 28.6 ticks = 30 decisions). test_bptt_credit_horizon.py keeps
    #: at least 8 segments per minibatch. Must divide update_timestep.
    bptt_chunk: int = int(os.environ.get("CLASH_BPTT_CHUNK", 50))
    ppo_epochs: int = 4

    #: 160 segments / 8 minibatches: 8 optimizer steps per epoch, 32 per
    #: rollout.
    num_minibatches: int = 8

    max_grad_norm: float = 0.5

    #: Value-clip range as a fraction of the batch's return std, floored at
    #: eps_clip. A fixed 0.2 (the ratio clip, a dimensionless number) stopped
    #: the critic correcting itself on ~46% of samples.
    vf_clip_std_frac: float = 1.0

    #: Weight on the opponent next-card cross-entropy
    #: (MicroRoyaleNet.predict_opp_next_card). Unlike opponent elixir, which is
    #: an affine function of two observed scalars, the next card needs play
    #: history, so this asks the LSTM to remember the cycle.
    aux_card_coef: float = 0.5

    #: Brings the cross-entropy into the other terms' range; kept separate so
    #: the logged value stays in nats.
    aux_card_scale: float = 0.02

    #: Episodes over which the aux term ramps from 0 to full weight. A fresh
    #: 185-way head starts at CE ln(185) = 5.22, and at full weight its
    #: gradient out-pulled and opposed the policy on the shared LSTM during the
    #: first updates. 0 disables the ramp. A mechanism measurement, not a
    #: win-rate gain: refuted if a paired from-scratch A/B at 0 vs 2000 shows the
    #: warm-up arm no better on reward at episode 2,000 and worse on
    #: Aux/NextCard_CE afterwards.
    aux_warmup_episodes: int = int(os.environ.get("CLASH_AUX_WARMUP_EPISODES", 2000))

    #: Weight on the cycle-branch next-card loss
    #: (MicroRoyaleNet.predict_cycle_card). Undivided is safe only because
    #: ScalarEncoder detaches the branch, so this gradient reaches nothing
    #: downstream. Read its accuracy against the ~0.22 marginal; a 24-dim
    #: branch tops out near 0.55.
    cycle_id_coef: float = 1.0

    #: Periodic-checkpoint interval. Overridable only so a short experiment
    #: writes matched artifacts; leave unset for real runs.
    save_every_episodes: int = int(os.environ.get("CLASH_SAVE_EVERY", 500))

    #: Episodes between demo replays.
    replay_every_episodes: int = 1000

    #: Run-level RNG seed, or None for OS entropy. Unseeded by default so
    #: existing configurations behave as before; see rl/seeding.py for what a
    #: seed pins.
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

    The heads are coupled, so fixed coefficients do not work: fixing one head's
    entropy broke the other's. Instead each head holds a target entropy
    fraction and the controller finds the coefficient, as in SAC's temperature
    tuning. The two pipelines deliberately differ; unifying them would change
    gameplay in whichever moved.
    """

    #: Fixed, never annealed: narrowing card choice is the known failure (0.10
    #: collapsed the policy to 5 of 8 cards).
    target_card: float = 0.35

    #: Annealed: wide while the policy discovers where things go, tight once it
    #: refines. 0.25 of log(612) is ~5 effective cells. Not annealed to 0,
    #: since some placement noise is correct against a live opponent.
    target_placement_start: float = 0.65
    target_placement_final: float = 0.25

    #: Sized to one long run. Past it the target stays at FINAL.
    anneal_episodes: int = 60000

    #: Multiplicative control on the normalized entropy fraction:
    #:   coef *= exp(rate * (target - measured))
    adapt_rate_card: float = 0.5

    #: The placement head's gain. Phase 2 sets it lower: at 0.5 on the
    #: corrected entropy signal the controller wound up and dissolved the
    #: policy.
    adapt_rate_placement: float = 0.5

    #: Hard cap on how far one update may move a coefficient. None disables it.
    coef_step_max: Optional[float] = None

    #: Healthy coefficients are 0.05-0.22; a lower floor switched the term off.
    coef_floor: float = 0.01
    coef_ceil_card: float = 0.5

    #: Placement's own ceiling, lowered where a pipeline sets it (0.433 already
    #: dissolved the policy). Not applied to the card head, whose controller
    #: works.
    coef_ceil_placement: float = 0.5

    #: Start from the last hand-tuned values.
    initial_coef_card: float = 0.05
    initial_coef_placement: float = 0.06


#: Pipeline 1 (`trainers/train.py`): one gain, one ceiling, no step cap.
PHASE1_ENTROPY = EntropyConfig()

#: Pipeline 2 (`trainers/train_selfplay.py`): starts the placement anneal
#: lower, since the policy is past discovery, and adds the three placement
#: guards.
PHASE2_ENTROPY = EntropyConfig(
    target_placement_start=0.50,
    adapt_rate_placement=0.10,
    coef_step_max=0.10,
    coef_ceil_placement=0.20,
)


def log_reachable(n):
    """log(n) with n floored at 2.

    Only guards log(1) = 0: a row with one legal arm has zero entropy and is
    masked out anyway.
    """
    return math.log(max(2, n))


def aux_warmup_scale(episodes_completed, warmup_episodes):
    """Multiplier on the aux loss: linear 0 -> 1 over `warmup_episodes`."""
    if warmup_episodes <= 0:
        return 1.0
    return float(min(1.0, max(0.0, episodes_completed / float(warmup_episodes))))
