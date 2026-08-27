"""PPOConfig: the hyperparameters, and the invariant that used to be an assert.

These were ~200 lines of locals inside `train_ppo()`, byte-identical to
`train_selfplay_ppo()`'s copy apart from three entropy numbers. Nothing could
read one without running a training loop, so nothing pinned one.
"""
import dataclasses

import pytest

from python_ai.rl.config import PHASE1_ENTROPY, PHASE2_ENTROPY, PPOConfig


def test_the_documented_defaults_are_the_ones_every_measurement_used():
    cfg = PPOConfig()
    # Raised 0.99 -> 0.999 on 2026-08-27. `1/(1-gamma)` is the horizon in
    # DECISIONS and a full match is max_ticks/skip_frames = 360, so at 0.99
    # the horizon covered barely a quarter of the game and every TERMINAL
    # reward decayed to 0.99^360 = 0.0268 -- making one crown worth 22.4x
    # winning. The relationship, rather than this literal, is pinned by
    # tests/test_reward_horizon_invariant.py. See rl/config.py for the
    # measured variance cost (+6% return spread at lambda=0.9).
    assert cfg.gamma == 0.999
    assert cfg.gae_lambda == 0.9
    assert cfg.eps_clip == 0.2
    assert cfg.lr == 3e-4
    assert cfg.update_timestep == 500
    assert cfg.bptt_chunk == 25
    assert cfg.ppo_epochs == 4
    assert cfg.num_minibatches == 8
    assert cfg.max_grad_norm == 0.5
    assert cfg.vf_clip_std_frac == 1.0
    assert cfg.aux_elixir_coef == 0.5
    assert cfg.aux_elixir_scale == 0.02


def test_update_timestep_must_divide_evenly_into_bptt_chunks():
    """A partial trailing chunk would train on a segment shorter than every
    other, silently weighting the end of each rollout differently."""
    with pytest.raises(ValueError, match="divisible"):
        PPOConfig(update_timestep=500, bptt_chunk=32)
    PPOConfig(update_timestep=500, bptt_chunk=25)      # must not raise


def test_segments_per_rollout_is_chunks_times_envs():
    """160 segments at the shipping settings: 20 chunks x 8 envs. A minibatch is
    then 20 segments, i.e. a real batch rather than 8 sequences."""
    cfg = PPOConfig(num_envs=8)
    assert cfg.segments_per_rollout == 160
    assert cfg.segments_per_rollout // cfg.num_minibatches == 20


def test_the_config_is_frozen():
    """A trainer that mutated its own config mid-run would make a checkpoint's
    episodes_completed meaningless as a description of what produced it."""
    cfg = PPOConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.gamma = 0.5


def test_a_smoke_sized_config_is_still_valid():
    """The smoke tests build tiny configs; the invariant must hold there too."""
    tiny = PPOConfig(num_envs=2, update_timestep=50, bptt_chunk=25,
                     num_minibatches=2, ppo_epochs=2)
    assert tiny.segments_per_rollout == 4


def test_the_gamma_used_for_shaping_is_the_same_one_GAE_uses():
    """Potential-based shaping's policy-invariance guarantee holds ONLY if the
    gamma in gamma*Phi(s') - Phi(s) is the gamma the returns discount with.
    One config object is how that is made structural instead of a convention.
    """
    cfg = PPOConfig()
    import inspect

    from python_ai.rl import base_trainer
    src = inspect.getsource(base_trainer.BaseTrainer.collect_rollout)
    assert "gamma=cfg.gamma" in src, (
        "compute_shaping must be passed the config's gamma, not a literal")

    # ...and `compute_shaping` must have NO default to fall back to, which is
    # the half this test could not previously see. While the shaping default
    # was itself 0.99, a caller that forgot the argument got the right answer
    # by luck; the guarantee only became structural when the parameter was
    # made required. tests/test_shaping_gamma_is_not_a_second_copy.py pins it
    # directly -- this line keeps the two facts adjacent, since a future
    # "convenience" default would restore the silent-drift hazard without
    # breaking the assertion above.
    import python_ai.rewards.shaping as _shaping
    assert inspect.signature(_shaping.compute_shaping).parameters[
        "gamma"].default is inspect.Parameter.empty, (
        "compute_shaping's gamma has acquired a default again; that reopens "
        "the silent policy-invariance break this test exists to prevent")
    assert cfg.gamma == PPOConfig.gamma


def test_entropy_configs_are_frozen_too():
    with pytest.raises(dataclasses.FrozenInstanceError):
        PHASE1_ENTROPY.target_card = 0.9


def test_the_two_entropy_configs_differ_only_where_they_are_meant_to():
    """Anything else diverging would be an accidental retune of a live pipeline.

    The four documented differences are the placement anneal start, the
    placement gain, the per-update step cap, and the placement ceiling.
    """
    a = dataclasses.asdict(PHASE1_ENTROPY)
    b = dataclasses.asdict(PHASE2_ENTROPY)
    differing = {k for k in a if a[k] != b[k]}
    assert differing == {"target_placement_start", "adapt_rate_placement",
                         "coef_step_max", "coef_ceil_placement"}
