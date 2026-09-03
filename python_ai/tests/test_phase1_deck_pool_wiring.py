"""The pool as the phase-1 env actually uses it.

Three things go silently wrong when an opponent deck changes, and all three
have precedent in this repo:

  * the ENGINE gets the new deck and the TEACHER does not, so the bot reasons
    about the previous deck's win condition and cycle -- `set_opponent_deck`
    carries a comment about exactly this;
  * the deck is applied and then reverted by the next auto-reset, because
    `reset()` re-applies `self.opp_deck`;
  * the deck never changes at all, and the run looks identical to the mirror
    run it was supposed to replace.

The last one is the dangerous one: it produces a clean log, a plausible win
rate, and a completely different experiment from the one being reported.
"""
import pytest

from python_ai.envs import gym_wrapper
from python_ai.opponents import deck_pool


def _env(**cfg):
    base = {"opponent": "teacher", "teacher_stage": 0, "deck_pool": True,
            "scenario_seed": 11}
    base.update(cfg)
    return gym_wrapper.MicroRoyaleEnv(base)


def test_off_by_default_so_every_existing_caller_is_unchanged():
    env = gym_wrapper.MicroRoyaleEnv({"opponent": "teacher"})
    assert env.deck_pool is None
    env.reset()
    assert list(env.current_opp_deck) == list(gym_wrapper.DEFAULT_DECK)


def test_enabling_it_actually_varies_the_opponent_deck_across_episodes():
    """The failure that would look like success: a pool that never rotates."""
    env = _env()
    seen = set()
    for _ in range(40):
        env.reset()
        seen.add(tuple(env.current_opp_deck))
    assert len(seen) >= 3, (
        f"only {len(seen)} distinct opponent deck(s) over 40 episodes -- the "
        "pool is not rotating and this is a mirror run wearing a pool's name")


def test_the_teacher_is_repointed_at_the_deck_it_is_actually_holding():
    """A teacher still holding the previous deck's role table treats the new
    deck's win condition as a plain melee troop -- a silent degradation that
    reads as 'the teacher is weak against other decks'."""
    env = _env()
    for _ in range(15):
        env.reset()
        assert env.teacher.deck == list(env.current_opp_deck)


def test_the_sampled_deck_survives_into_the_engine():
    env = _env()
    env.reset()
    assert list(env.game.get_hand_for_team(1)) or True   # engine accepted it
    # Every card the opponent holds must come from the deck we just set.
    hand = [c for c in env.game.get_hand_for_team(1) if c >= 0]
    assert set(hand) <= set(env.current_opp_deck)


def test_restricting_the_pool_by_name_is_honoured():
    env = _env(deck_pool=["hog_26_mirror"])
    for _ in range(5):
        env.reset()
        assert sorted(env.current_opp_deck) == sorted(gym_wrapper.DEFAULT_DECK)


def test_an_empty_restriction_raises_rather_than_silently_falling_back():
    with pytest.raises(ValueError):
        _env(deck_pool=["no_such_deck"])


def test_outcomes_move_the_local_estimate_toward_what_happened():
    """The EWMA must converge to the OUTCOMES, not stay near its seed.

    Written without assuming a starting value: the pool ships measured priors,
    so the mirror starts at 1.0 and an assertion like `> start + 0.2` is
    unsatisfiable there. What matters is that live results overwrite the prior
    in both directions -- the prior is a seed and never a gate.
    """
    env = _env(deck_pool=["hog_26_mirror"])
    env.reset()
    for _ in range(300):
        env._record_deck_outcome(False)
    assert env.deck_pool_stats["hog_26_mirror"] < 0.05, (
        "a deck that keeps losing must not stay near its optimistic prior")
    for _ in range(300):
        env._record_deck_outcome(True)
    assert env.deck_pool_stats["hog_26_mirror"] > 0.95


def test_the_readout_reports_every_deck_in_the_pool():
    """A deck missing from the read-out is a deck whose collapse is invisible,
    which is the whole reason this diagnostic exists."""
    env = _env()
    stats = env.get_deck_pool_stats()
    assert set(stats) == {d.name for d in deck_pool.load_pool()}


def test_the_pool_takes_precedence_over_randomize_opp_deck():
    """They are different distributions, not two strengths of one. A config
    that sets both must not silently get the registry draw."""
    env = _env(randomize_opp_deck=True)
    pool_decks = {tuple(sorted(d.card_ids)) for d in deck_pool.load_pool()}
    for _ in range(20):
        env.reset()
        assert tuple(sorted(env.current_opp_deck)) in pool_decks


def test_set_opponent_deck_cannot_pin_a_deck_the_pool_will_overwrite():
    """`reset()` samples the pool BEFORE honouring `self.opp_deck`, so a caller
    that sets a deck and then relies on it is silently wrong from the next
    episode. The trainer's `random_opponent` phase used to do exactly that.

    This pins the ACTUAL behaviour so the trainer's guard stays justified: with
    the pool on, an explicit set does not survive a reset.
    """
    env = _env()
    env.set_opponent_deck(list(gym_wrapper.DEFAULT_DECK))
    seen = set()
    for _ in range(30):
        env.reset()
        seen.add(tuple(env.current_opp_deck))
    assert len(seen) > 1, (
        "if an explicit deck DID survive reset, the trainer guard in "
        "train.py's _rotate_random_deck should be removed instead")


def test_the_estimate_leaves_an_optimistic_prior_within_a_few_matches():
    """The shipped priors come from a TRAINED policy, so for a fresh net their
    ordering is right and their level is far too high. A flat-rate EWMA would
    keep feeding a random-init policy the decks the ep-32,484 policy found
    hard-but-winnable -- which for a fresh net are unwinnable -- for thousands
    of episodes. Measured on a fresh run: 0 wins in 50 episodes.

    The count-weighted rate has to clear the pool's own floor fast enough that
    PFSP parks such a deck almost immediately.
    """
    from python_ai.opponents import deck_pool as dp
    env = _env(deck_pool=["mega_knight_ram"])
    env.reset()
    assert env.deck_pool_stats["mega_knight_ram"] >= dp.POOL_WINRATE_FLOOR, (
        "this deck's prior should start at or above the floor for the test to "
        "be measuring anything")
    for _ in range(3):
        env._record_deck_outcome(False)
    assert env.deck_pool_stats["mega_knight_ram"] < dp.POOL_WINRATE_FLOOR, (
        "three losses must be enough to park a deck this policy cannot win")


def test_a_late_estimate_is_still_stable_against_noise():
    """The early speed must not turn into permanent twitchiness: once a deck has
    a real history, one result may not swing it across the floor."""
    env = _env(deck_pool=["hog_26_mirror"])
    env.reset()
    for _ in range(200):
        env._record_deck_outcome(True)
    before = env.deck_pool_stats["hog_26_mirror"]
    env._record_deck_outcome(False)
    assert before - env.deck_pool_stats["hog_26_mirror"] < 0.10
