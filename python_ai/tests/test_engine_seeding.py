"""A seedable engine (`ClashRoyaleEnv.seed`, UPSTREAM_REQUESTS.md item 7).

The engine has two sources of randomness: `GameManager::rng` deals the opening
hand and the starting deckQueue order, and `ClashEnv::rng` drives
HeuristicOpponent. seed() must reach both. The cycle order is checked, not just
the four cards in hand, since `CycleTracker` and `cycle_value` read the queue.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env as E  # noqa: E402
from python_ai.envs import gym_wrapper  # noqa: E402

CE = E.ClashRoyaleEnv
DECK = list(gym_wrapper.DEFAULT_DECK)

#: The binding; the cases are gated on it rather than on a version number.
HAS_SEED = hasattr(CE, "seed")
requires_seed = pytest.mark.skipif(
    not HAS_SEED,
    reason="ClashRoyaleEnv.seed is unbound -- UPSTREAM_REQUESTS.md item 7. "
           "Rebuild the .pyd after applying the edit proposed there.")


def _seeded_env(seed):
    env = CE(DECK, DECK, 3600)
    env.seed(seed)
    return env


def _cycle_order(env, team, plays=8):
    """The order cards leave the hand, which is the queue order. Read
    behaviourally (the queue is not observable): playing hand slot 0 repeatedly
    rotates that slot.
    """
    seen = []
    for i in range(plays):
        hand = list(env.get_hand_for_team(team))
        seen.append(tuple(hand))
        slot = i % CE.HAND_SIZE
        env.set_elixir_for_team(team, 10.0)
        if team == 0:
            env.step_self_play(slot, 9.0, 5.0, -1, 0.0, 0.0, 10)
        else:
            env.step_self_play(-1, 0.0, 0.0, slot, 9.0, 5.0, 10)
    return seen


# --- seed() ---
@requires_seed
def test_two_envs_with_the_same_seed_deal_the_same_opening():
    """Both teams: `GameManager::reset()` deals both from one generator back to
    back.
    """
    a, b = _seeded_env(1234), _seeded_env(1234)
    assert list(a.get_hand_for_team(0)) == list(b.get_hand_for_team(0))
    assert list(a.get_hand_for_team(1)) == list(b.get_hand_for_team(1))


@requires_seed
def test_the_same_seed_also_pins_the_CYCLE_ORDER_not_just_the_hand():
    """`initializeDeck` shuffles all eight and queues the last four, so the seed
    governs the cycle as well as the hand.
    """
    a, b = _seeded_env(99), _seeded_env(99)
    assert _cycle_order(a, 0) == _cycle_order(b, 0)


@requires_seed
def test_different_seeds_actually_differ():
    """Control against an engine that stopped shuffling. Several seed pairs, since
    any two can collide on a 70-way hand draw.
    """
    openings = {s: tuple(_seeded_env(s).get_hand_for_team(0)) for s in range(12)}
    assert len(set(openings.values())) > 1, (
        f"every seed produced the same opening hand {openings} -- the shuffle "
        "is not shuffling")


@requires_seed
def test_seeding_takes_effect_on_the_CURRENT_episode_not_the_next_one():
    """`initializeDeck` runs inside reset(), so seed() must re-deal; otherwise it
    would apply only from the next episode, quietly.
    """
    a = _seeded_env(7)
    b = CE(DECK, DECK, 3600)
    b.reset()                      # burn one unseeded opening first
    b.seed(7)
    assert list(a.get_hand_for_team(0)) == list(b.get_hand_for_team(0))


# --- the unseeded engine and the snapshot workaround ---
def test_the_engine_is_currently_unseedable_and_that_is_why_openings_vary():
    """Skips once seed() exists; kept as the record of the unseeded behaviour.
    """
    if HAS_SEED:
        pytest.skip("seed() has landed -- this test has served its purpose")
    assert not hasattr(CE, "seed"), "seed exists; remove the skip markers above"
    hands = {tuple(CE(DECK, DECK, 3600).get_hand_for_team(0)) for _ in range(40)}
    assert len(hands) > 1, (
        "40 fresh envs all dealt the same opening hand -- that would mean the "
        "shuffle is already deterministic and item 7 is mis-stated")


def test_a_snapshot_DOES_reproduce_an_opening_which_is_the_workaround():
    """`env.snapshot()` pairs openings within one process; it cannot make two
    invocations see the same match population.
    """
    root = CE(DECK, DECK, 3600)
    root.reset()
    a, b = root.snapshot(), root.snapshot()
    assert list(a.get_hand_for_team(0)) == list(b.get_hand_for_team(0))
    assert list(a.get_hand_for_team(1)) == list(b.get_hand_for_team(1))


# --- sample_random_deck ---
# It draws from a function-local static generator in src/bindings.cpp that
# `ClashEnv::seed` cannot reach, shared process-wide. A seeded call therefore
# gets a private generator, so its result does not depend on construction
# order.
def test_sample_random_deck_is_reproducible_when_seeded():
    a = list(E.sample_random_deck(seed=4242))
    b = list(E.sample_random_deck(seed=4242))
    assert a == b, "same seed must yield the same deck"
    assert len(a) == 8


def test_sample_random_deck_different_seeds_differ():
    """Control: a stub returning one constant deck would satisfy reproducibility.
    """
    decks = {tuple(E.sample_random_deck(seed=s)) for s in range(12)}
    assert len(decks) > 1, "distinct seeds must not all collapse to one deck"


def test_sample_random_deck_seeded_stream_is_private_to_the_caller():
    """With every env drawing from the process-global static, a seeded draw must
    not depend on how many unseeded draws came first.
    """
    first = list(E.sample_random_deck(seed=777))
    for _ in range(5):
        E.sample_random_deck()          # interleave unseeded draws
    second = list(E.sample_random_deck(seed=777))
    assert first == second


def test_sample_random_deck_without_a_seed_still_works():
    """Existing zero-argument call sites keep working."""
    deck = list(E.sample_random_deck())
    assert len(deck) == 8
