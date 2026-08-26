"""The acceptance test for `UPSTREAM_REQUESTS.md` item 7 -- a seedable engine.

WHY THIS FILE EXISTS BEFORE THE FEATURE DOES
--------------------------------------------
The engine has exactly two sources of randomness (CLAUDE.md: "Combat is
deterministic"), and NEITHER is reachable from Python:

    GameManager::rng   -> PlayerState::initializeDeck -> the OPENING HAND
                          and the starting deckQueue ORDER
    ClashEnv::rng      -> HeuristicOpponent

That second one is what the original item-7 proposal offered to seed. It is not
the one that deals the hand. So the obvious one-line fix would have left the
opening hand exactly as random as before, and the natural reading of that
result -- "seeding doesn't work" -- would have been wrong.

This test is written to fail for the RIGHT reason, and it is deliberately
strict about the part that matters most for 2.6 Hog Cycle: **the cycle order**,
not just the four cards in hand. A shuffle that pinned the hand and left the
queue random would pass a naive version of this test and still leave every
cross-run experiment unreproducible, because `CycleTracker` and `cycle_value`
read the queue.

WHAT IT COSTS TO NOT HAVE THIS, measured rather than argued: on 2026-08-20 a
combo-family ablation was built around "same --seed, so the untreated arm should
reproduce". It did not (0.475 against 0.537), because `--seed` reaches only the
teachers' numpy RNG. The mis-specified control looked like a failed comparison.
`prove_combos.py`'s docstring and CLAUDE.md now carry the rule that replaced it:
within a run the snapshot pairing is sound, across runs only DELTAS compare.

IT SKIPPED RATHER THAN FAILING, AND THEN IT WENT GREEN. Written 2026-08-20
against an engine with no `seed` binding, gated on `hasattr` so it would light
up by itself the moment a rebuilt `.pyd` landed. That happened 2026-08-21: the
edit is in, and all four gated cases pass. `test_the_engine_is_currently_-
unseedable...` now skips itself with "seed() has landed", which is the whole
point of writing it that way -- the file records both the before and the after
without either being a lie at the time.

The skips existed because three sessions believed this machine had no C++
toolchain. It does; it is simply not on PATH. See CLAUDE.md's environment
section.
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

#: The binding proposed in UPSTREAM_REQUESTS item 7. Everything here is gated on
#: it rather than on a version number, so the tests light up by themselves the
#: moment a rebuilt .pyd lands.
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
    """The order cards leave the hand -- which IS the queue order.

    Read behaviourally rather than from a getter because the queue is not
    observable (`get_hand_for_team` shows only the four in hand). Playing hand
    slot 0 repeatedly rotates exactly that slot, so this walks every slot in
    turn and records what surfaces.
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


# --------------------------------------------------------------------------
# the thing item 7 asks for
# --------------------------------------------------------------------------
@requires_seed
def test_two_envs_with_the_same_seed_deal_the_same_opening():
    """BOTH teams. `GameManager::reset()` deals team 0 and team 1 from the same
    generator back to back, so a seed that fixed only one would mean the fix
    reached `playerAI` and not `playerOpponent` -- and self-play would still be
    irreproducible."""
    a, b = _seeded_env(1234), _seeded_env(1234)
    assert list(a.get_hand_for_team(0)) == list(b.get_hand_for_team(0))
    assert list(a.get_hand_for_team(1)) == list(b.get_hand_for_team(1))


@requires_seed
def test_the_same_seed_also_pins_the_CYCLE_ORDER_not_just_the_hand():
    """The half that actually matters for 2.6.

    `initializeDeck` shuffles all eight indices and uses the last four as the
    starting `deckQueue`, so the seed governs the cycle as well as the hand. A
    fix that pinned only the opening four would leave `CycleTracker` --
    and therefore `teacher.cycle_value` -- reading a different deck every run.
    """
    a, b = _seeded_env(99), _seeded_env(99)
    assert _cycle_order(a, 0) == _cycle_order(b, 0)


@requires_seed
def test_different_seeds_actually_differ():
    """The control that stops this suite passing vacuously.

    Every assertion above is satisfied by an engine that simply stopped
    shuffling. This one fails in that case. It sweeps several seed pairs
    because any two given seeds can collide on a 70-way hand draw by chance --
    `reset()`'s shuffle is uniform over all 70 hand-sets (CLAUDE.md).
    """
    openings = {s: tuple(_seeded_env(s).get_hand_for_team(0)) for s in range(12)}
    assert len(set(openings.values())) > 1, (
        f"every seed produced the same opening hand {openings} -- the shuffle "
        "is not shuffling")


@requires_seed
def test_seeding_takes_effect_on_the_CURRENT_episode_not_the_next_one():
    """The ordering subtlety named in the proposal.

    `initializeDeck` runs inside `GameManager::reset()`, so a `seed()` that did
    not re-deal would leave the hand already in play untouched and silently
    apply only from the following episode. That is the API most likely to be
    got wrong, and the failure is quiet: a caller seeds, reads the hand, and
    gets a different one than the same seed gives elsewhere.
    """
    a = _seeded_env(7)
    b = CE(DECK, DECK, 3600)
    b.reset()                      # burn one unseeded opening first
    b.seed(7)
    assert list(a.get_hand_for_team(0)) == list(b.get_hand_for_team(0))


# --------------------------------------------------------------------------
# what is true TODAY -- these run either way and pin the diagnosis
# --------------------------------------------------------------------------
def test_the_engine_is_currently_unseedable_and_that_is_why_openings_vary():
    """Runs unconditionally, and is the reason the skips above are honest.

    If this ever fails, `seed` has landed and the skip markers should come off.
    """
    if HAS_SEED:
        pytest.skip("seed() has landed -- this test has served its purpose")
    assert not hasattr(CE, "seed"), "seed exists; remove the skip markers above"
    hands = {tuple(CE(DECK, DECK, 3600).get_hand_for_team(0)) for _ in range(40)}
    assert len(hands) > 1, (
        "40 fresh envs all dealt the same opening hand -- that would mean the "
        "shuffle is already deterministic and item 7 is mis-stated")


def test_a_snapshot_DOES_reproduce_an_opening_which_is_the_workaround():
    """Why item 7 is a cost multiplier and not a blocker.

    `env.snapshot()` gives bit-exact pairing WITHIN one process, which is what
    every A/B harness here uses. What it cannot do is make two separate
    invocations see the same match population -- hence "compare deltas, never
    arm levels".
    """
    root = CE(DECK, DECK, 3600)
    root.reset()
    a, b = root.snapshot(), root.snapshot()
    assert list(a.get_hand_for_team(0)) == list(b.get_hand_for_team(0))
    assert list(a.get_hand_for_team(1)) == list(b.get_hand_for_team(1))


# --------------------------------------------------------------------------
# item 23C -- the THIRD generator, which item 7's seed() cannot reach
# --------------------------------------------------------------------------
# `sample_random_deck` draws from a function-local `static std::mt19937` in
# src/bindings.cpp. It is not a member of ClashEnv or GameManager, so
# `ClashEnv::seed` cannot touch it -- which means a run with
# `randomize_opp_deck=True` still gets a random OPPONENT DECK even though its
# opening hand, cycle order and heuristic rolls are all now pinned. That is the
# largest single remaining source of episode-to-episode variance.
#
# The static is also process-global, so two envs built in one process interleave
# draws from one stream. A seeding entry point that set the static would
# therefore only reproduce for a fixed construction order -- which is why the
# accepted fix gives the SEEDED caller a private generator instead.
def test_sample_random_deck_is_reproducible_when_seeded():
    a = list(E.sample_random_deck(seed=4242))
    b = list(E.sample_random_deck(seed=4242))
    assert a == b, "same seed must yield the same deck"
    assert len(a) == 8


def test_sample_random_deck_different_seeds_differ():
    """The control that stops the test above passing vacuously -- a stub that
    returned one constant deck would satisfy reproducibility perfectly."""
    decks = {tuple(E.sample_random_deck(seed=s)) for s in range(12)}
    assert len(decks) > 1, "distinct seeds must not all collapse to one deck"


def test_sample_random_deck_seeded_stream_is_private_to_the_caller():
    """The property option 1 could NOT deliver, and the reason option 2 was
    chosen: at num_envs=8 every env draws from the same process-global static,
    so a seeded draw must not depend on how many unseeded draws happened first.
    """
    first = list(E.sample_random_deck(seed=777))
    for _ in range(5):
        E.sample_random_deck()          # interleave unseeded draws
    second = list(E.sample_random_deck(seed=777))
    assert first == second


def test_sample_random_deck_without_a_seed_still_works():
    """Backwards compatibility: every existing zero-argument call site must keep
    working unchanged."""
    deck = list(E.sample_random_deck())
    assert len(deck) == 8
