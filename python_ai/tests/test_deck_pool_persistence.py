"""A restart keeps what the run learned about the deck pool.

PFSP weights by (1 - rate)^2, so resetting learned rates to the JSON priors
mis-weights decks badly on every resume. The damage is invisible because the
count-weighted estimate re-converges within ~10 matches, which reads as
ordinary post-restart noise.
"""
import numpy as np
import pytest

from python_ai.envs import gym_wrapper


def _pool_env():
    env = gym_wrapper.MicroRoyaleEnv({
        "opponent": "teacher",
        "teacher_stage": 0,
        "deck_pool": True,
    })
    if not env.deck_pool_stats:
        pytest.skip("deck pool unavailable in this environment")
    return env


def test_a_fresh_env_starts_from_the_json_priors():
    """The baseline a restore has to improve on."""
    env = _pool_env()
    from python_ai.opponents import deck_pool
    priors = {d.name: d.prior_win_rate for d in deck_pool.load_pool()}
    assert env.get_deck_pool_stats() == pytest.approx(priors)


def test_restored_rates_replace_the_priors():
    env = _pool_env()
    name = next(iter(env.deck_pool_stats))
    env.set_deck_pool_stats({name: 0.123}, {name: 400})
    assert env.get_deck_pool_stats()[name] == pytest.approx(0.123)
    assert env.get_deck_pool_counts()[name] == 400


def test_restoring_rates_WITHOUT_counts_erases_itself_on_the_next_match():
    """Counts must travel with rates: `alpha = max(0.05, 1/(n+1))`, so a restored
    rate with a zero count is overwritten by the next single game.
    """
    env = _pool_env()
    name = next(iter(env.deck_pool_stats))
    env.current_deck_name = name

    # Counts not restored: one loss wipes the estimate.
    env.set_deck_pool_stats({name: 0.90})
    env._record_deck_outcome(False)
    naive = env.get_deck_pool_stats()[name]

    # Counts restored: the same loss barely moves it.
    env2 = _pool_env()
    env2.current_deck_name = name
    env2.set_deck_pool_stats({name: 0.90}, {name: 400})
    env2._record_deck_outcome(False)
    faithful = env2.get_deck_pool_stats()[name]

    assert naive < 0.5, (
        f"expected a zero count to let one game dominate, got {naive:.4f}")
    assert faithful > 0.85, (
        f"a restored count should make one game a small update, got {faithful:.4f}")
    assert faithful - naive > 0.3


def test_unknown_deck_names_are_ignored_rather_than_added():
    """A checkpoint from a different meta_decks.json must not inject decks this
    pool cannot sample.
    """
    env = _pool_env()
    before = set(env.get_deck_pool_stats())
    env.set_deck_pool_stats({"a_deck_that_does_not_exist": 0.5}, {})
    assert set(env.get_deck_pool_stats()) == before


def test_a_disabled_pool_accepts_the_call_and_does_nothing():
    """Phase 1 can run with CLASH_PHASE1_DECK_POOL=0 and the trainer restores
    unconditionally: a no-op, not an exception.
    """
    env = gym_wrapper.MicroRoyaleEnv({"opponent": "teacher", "teacher_stage": 0})
    env.set_deck_pool_stats({"anything": 0.5}, {"anything": 10})
    assert env.get_deck_pool_stats() == {}
