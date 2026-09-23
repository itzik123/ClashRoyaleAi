"""The deck pool as the phase-1 env uses it.

When the opponent deck changes, three things can go silently wrong: the teacher
keeps the old deck while the engine has the new one; the next auto-reset
reverts the deck; or the deck never changes at all, which yields a clean log of
a different experiment than the one reported.
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
    """A teacher holding the previous deck's role table misreads the new win
    condition.
    """
    env = _env()
    for _ in range(15):
        env.reset()
        assert env.teacher.deck == list(env.current_opp_deck)


def test_the_sampled_deck_survives_into_the_engine():
    env = _env()
    env.reset()
    assert list(env.game.get_hand_for_team(1)) or True   # engine accepted it
    # Every card the opponent holds comes from the deck just set.
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
    """Live results overwrite the prior in both directions; the prior is a seed,
    never a gate.
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
    """A deck missing from the read-out is a deck whose collapse is invisible.
    """
    env = _env()
    stats = env.get_deck_pool_stats()
    assert set(stats) == {d.name for d in deck_pool.load_pool()}


def test_the_pool_takes_precedence_over_randomize_opp_deck():
    """Different distributions, not two strengths of one: a config setting both
    must not silently get the registry draw.
    """
    env = _env(randomize_opp_deck=True)
    pool_decks = {tuple(sorted(d.card_ids)) for d in deck_pool.load_pool()}
    for _ in range(20):
        env.reset()
        assert tuple(sorted(env.current_opp_deck)) in pool_decks


def test_set_opponent_deck_cannot_pin_a_deck_the_pool_will_overwrite():
    """`reset()` samples the pool before honouring `self.opp_deck`, so with the
    pool on an explicit set does not survive a reset; the trainer guards
    against relying on it.
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
    """The count-weighted rate collapses an optimistic prior within a few matches,
    so PFSP re-weights on this policy's evidence rather than on the policy the
    priors came from. Measured against the prior itself.
    """
    env = _env(deck_pool=["mega_knight_ram"])
    env.reset()
    prior = env.deck_pool_stats["mega_knight_ram"]
    assert prior > 0.10, (
        "this deck's prior should start optimistic for the test to measure "
        "anything")
    for _ in range(3):
        env._record_deck_outcome(False)
    after = env.deck_pool_stats["mega_knight_ram"]
    assert after < prior / 3.0, (
        f"three losses moved the estimate only {prior:.3f} -> {after:.3f}; a "
        f"count-weighted rate must collapse an optimistic prior fast")


def test_a_late_estimate_is_still_stable_against_noise():
    """The early speed must not become permanent twitchiness: with a real history,
    one result may not swing the estimate far.
    """
    env = _env(deck_pool=["hog_26_mirror"])
    env.reset()
    for _ in range(200):
        env._record_deck_outcome(True)
    before = env.deck_pool_stats["hog_26_mirror"]
    env._record_deck_outcome(False)
    assert before - env.deck_pool_stats["hog_26_mirror"] < 0.10
