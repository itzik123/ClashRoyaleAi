"""A restart must not throw away what the run learned about the deck pool.

THE DEFECT THIS PINS, measured on the live 2026-09-04 run. `deck_pool_stats`
was initialised from `meta_decks.json`'s `prior_win_rate` on every env
construction and lived only in worker memory, so every restart silently reset
it. After 12,589 episodes the agent had learned:

    deck                learned   JSON prior
    xbow_30_cycle         0.181        0.867
    mortar_cycle          0.176        0.800
    dart_bait_cycle       0.111        0.533
    hog_26_mirror         0.406        1.000

PFSP weights by `(1 - rate)^2`, so xbow's sampling weight was wrong by a factor
of ~37 on resume. The 100-episode win rate fell 0.42 -> 0.143 and took ~500
episodes to climb back, on every restart, with nothing reporting it.

The failure was invisible because the run RECOVERS -- the count-weighted alpha
re-converges each deck within ~10 matches -- so it reads as ordinary post-restart
noise rather than as discarded state.
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
    """The baseline the restore has to improve on."""
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
    """The trap `set_deck_pool_stats`'s docstring exists for.

    `alpha = max(0.05, 1/(n+1))`, so a restored rate carried alongside a ZERO
    count gets alpha = 1.0 on the very next episode and is overwritten by that
    single game's outcome. A restore that erases itself on contact looks exactly
    like the bug it was meant to fix, so this pins that counts travel with rates.
    """
    env = _pool_env()
    name = next(iter(env.deck_pool_stats))
    env.current_deck_name = name

    # counts NOT restored -> one loss wipes the estimate
    env.set_deck_pool_stats({name: 0.90})
    env._record_deck_outcome(False)
    naive = env.get_deck_pool_stats()[name]

    # counts restored -> the same loss barely moves it
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
    """A checkpoint written against a different meta_decks.json must not inject
    decks this pool does not have -- pfsp_weights would then rank a deck that
    can never be sampled."""
    env = _pool_env()
    before = set(env.get_deck_pool_stats())
    env.set_deck_pool_stats({"a_deck_that_does_not_exist": 0.5}, {})
    assert set(env.get_deck_pool_stats()) == before


def test_a_disabled_pool_accepts_the_call_and_does_nothing():
    """Phase 1 can run with CLASH_PHASE1_DECK_POOL=0, and the trainer restores
    unconditionally -- so this must be a no-op, not an exception."""
    env = gym_wrapper.MicroRoyaleEnv({"opponent": "teacher", "teacher_stage": 0})
    env.set_deck_pool_stats({"anything": 0.5}, {"anything": 10})
    assert env.get_deck_pool_stats() == {}
