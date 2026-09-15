"""The phase-1 meta-deck pool: loading, validation, and PFSP weighting.

The pool decides WHICH OPPONENT the agent trains against, so every failure mode
here is silent by nature -- a dropped deck still trains, a misspelt card still
trains, a champion the teacher cannot use still trains. Each of those is a run
measuring a different opponent distribution than the file says it is measuring.
Hence: every one of them raises, and these tests pin that it raises.
"""
import json

import pytest

import clash_royale_env as E
from python_ai.envs import gym_wrapper
from python_ai.opponents import deck_pool


def _write(tmp_path, decks, **extra):
    blob = {"schema_version": 1, "decks": decks}
    blob.update(extra)
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(blob), encoding="utf-8")
    return str(path)


def _deck(name="d1", cards=None, **extra):
    entry = {"name": name,
             "cards": cards or ["Knight", "Archers", "Giant", "Arrows",
                                "Goblins", "Musketeer", "Fireball", "Skeletons"]}
    entry.update(extra)
    return entry


# ------------------------------------------------------------- the shipped --
def test_the_shipped_pool_loads_and_every_deck_is_legal():
    decks = deck_pool.load_pool()
    assert len(decks) >= 8, "a pool this small cannot teach generalization"
    for d in decks:
        assert len(d.card_ids) == deck_pool.DECK_SIZE
        assert len(set(d.card_ids)) == deck_pool.DECK_SIZE
        # The engine is the authority on deck legality, not this loader.
        assert E.validate_deck_slots(d.card_ids) == "", (
            f"{d.name} is not a legal deck: {E.validate_deck_slots(d.card_ids)}")


def test_the_mirror_is_still_in_the_pool():
    """The 2.6 mirror is DEMOTED to one matchup of many, never deleted.

    It is the deck the agent plays, so it stays the single most informative
    single matchup -- and dropping it entirely would trade one overfit
    distribution for another.
    """
    decks = {d.name: d for d in deck_pool.load_pool()}
    assert "hog_26_mirror" in decks
    assert sorted(decks["hog_26_mirror"].card_ids) == sorted(
        gym_wrapper.DEFAULT_DECK)


def test_the_pool_is_not_mostly_the_agents_own_archetype():
    """A pool of sixteen cycle decks is a mirror with extra steps.

    The three dead cards answer TANKS, CLUSTERS and SWARMS, so the pool has to
    contain decks that produce those. Checked on the tag vocabulary rather than
    on specific decks, so swapping a deck for another of the same archetype
    keeps this passing and gutting the variety does not.
    """
    tags = {t for d in deck_pool.load_pool() for t in d.tags}
    for needed in ("beatdown", "swarm", "cluster", "siege"):
        assert needed in tags, f"the pool offers nothing tagged {needed!r}"


# --------------------------------------------------------------- validation --
def test_an_unknown_card_name_raises_and_names_the_deck_and_the_card(tmp_path):
    path = _write(tmp_path, [_deck(cards=[
        "Knight", "Archers", "Giant", "Arrows",
        "Goblins", "Musketeer", "Firebal", "Skeletons"])])
    with pytest.raises(deck_pool.DeckPoolError) as e:
        deck_pool.load_pool(path)
    assert "d1" in str(e.value) and "Firebal" in str(e.value)
    assert "Fireball" in str(e.value), "the error should suggest the near miss"


@pytest.mark.parametrize("cards,fragment", [
    (["Knight"] * 8, "repeats"),
    (["Knight", "Archers", "Giant"], "expected 8"),
])
def test_a_malformed_deck_raises(tmp_path, cards, fragment):
    path = _write(tmp_path, [_deck(cards=cards)])
    with pytest.raises(deck_pool.DeckPoolError) as e:
        deck_pool.load_pool(path)
    assert fragment in str(e.value)


def test_a_champion_is_refused_because_the_teacher_cannot_use_its_ability(tmp_path):
    """A Champion the teacher holds is a strictly worse card than the real one,
    so the deck silently stops being the deck it is named after."""
    path = _write(tmp_path, [_deck(cards=[
        "Archer Queen", "Archers", "Giant", "Arrows",
        "Goblins", "Musketeer", "Fireball", "Skeletons"])])
    with pytest.raises(deck_pool.DeckPoolError) as e:
        deck_pool.load_pool(path)
    assert "Champion" in str(e.value)
    # ...and the escape hatch works, for when abilities are wired.
    assert len(deck_pool.load_pool(path, allow_champions=True)) == 1


def test_duplicate_deck_names_raise(tmp_path):
    path = _write(tmp_path, [_deck("same"), _deck("same")])
    with pytest.raises(deck_pool.DeckPoolError):
        deck_pool.load_pool(path)


def test_a_disabled_deck_is_kept_in_the_file_and_out_of_the_pool(tmp_path):
    path = _write(tmp_path, [_deck("on"), _deck("off", enabled=False)])
    assert [d.name for d in deck_pool.load_pool(path)] == ["on"]
    assert len(deck_pool.load_pool(path, include_disabled=True)) == 2


def test_an_all_disabled_pool_raises_rather_than_training_on_nothing(tmp_path):
    path = _write(tmp_path, [_deck("off", enabled=False)])
    with pytest.raises(deck_pool.DeckPoolError):
        deck_pool.load_pool(path)


# ------------------------------------------------------------------- PFSP --
def test_pfsp_concentrates_on_hard_but_winnable_decks():
    w = deck_pool.pfsp_weights({"easy": 0.95, "even": 0.50, "hard": 0.25})
    assert w["hard"] > w["even"] > w["easy"], (
        "PFSP must prefer the deck the agent is doing WORST against, among "
        "those it can still win")
    assert sum(w.values()) == pytest.approx(1.0)


def test_a_deck_below_the_win_rate_floor_is_parked_at_the_minimum():
    """A structurally lost matchup is not a curriculum, it is a zero-gradient
    state -- the exact failure the 2026-08-19 curriculum pivot exists to avoid.
    Uncapped `(1-wr)^2` would give it the LARGEST weight of all.
    """
    w = deck_pool.pfsp_weights({"hopeless": 0.02, "hard": 0.25, "even": 0.50},
                               floor=0.20)
    assert w["hopeless"] < w["hard"], (
        "an unwinnable deck must not attract the most training time")


def test_nothing_is_ever_weighted_to_zero():
    """A deck that stops being sampled is a deck the policy is free to forget.
    Same argument as phase 2's PFSP_MIN_WEIGHT, one level down."""
    w = deck_pool.pfsp_weights({f"d{i}": r for i, r in
                                enumerate([0.0, 0.5, 1.0, 1.0, 1.0])})
    assert all(v > 0 for v in w.values())


def test_sampling_respects_the_weights():
    import random
    decks = deck_pool.load_pool()[:3]
    heavy = decks[1].name
    weights = {d.name: (10.0 if d.name == heavy else 0.01) for d in decks}
    rng = random.Random(0)
    picks = [deck_pool.sample_deck(decks, weights, rng).name for _ in range(400)]
    assert picks.count(heavy) > 350
    assert len(set(picks)) > 1, "a floor must still let the others through"


def test_a_pool_nothing_can_beat_yet_falls_back_to_EASIEST_first():
    """The state every fresh run starts in, and PFSP inverts in it.

    `(1 - wr)^2` ranks a 0.02 deck above a 0.19 one, so a policy losing
    everything would be handed the matchup it loses hardest; a flat unwinnable
    weight makes the draw uniform, which is no better. When nothing clears the
    floor the deck axis has no signal and should hand back the most winnable
    opponent instead.
    """
    w = deck_pool.pfsp_weights({"awful": 0.02, "bad": 0.10, "least_bad": 0.19},
                               floor=0.20)
    assert w["least_bad"] > w["bad"] > w["awful"]
    assert sum(w.values()) == pytest.approx(1.0)


def test_one_winnable_deck_is_enough_to_restore_normal_pfsp():
    """The fallback is for 'nothing works', not 'most things do not'."""
    w = deck_pool.pfsp_weights({"awful": 0.02, "bad": 0.10, "ok": 0.55},
                               floor=0.20)
    assert w["ok"] > w["awful"], "normal PFSP must resume once anything clears"


def test_the_shipped_pool_carries_measured_priors():
    """A uniform 0.5 start feeds a fresh net the unwinnable decks as often as
    the mirror for the thousands of episodes the EWMA needs to separate them."""
    decks = deck_pool.load_pool()
    priors = {d.name: d.prior_win_rate for d in decks}
    assert any(p != deck_pool.NEUTRAL_PRIOR for p in priors.values()), (
        "the pool file has lost its measured priors")
    # The episode-share half of this test moved to
    # test_the_shipped_pool_now_spends_the_run_on_the_decks_it_loses_to, which
    # asserts the OPPOSITE now that the floor is off. What stays here is that
    # the priors exist at all.


# --------------------------------------------------------------------------
# 2026-09-06: the floor is OFF by default. See POOL_WINRATE_FLOOR's own comment
# for the measurement; the mechanism is kept and tested, only the default moved.
# --------------------------------------------------------------------------

def test_the_win_rate_floor_is_off_by_default():
    """A deck the agent cannot beat YET must attract the most training time.

    The floor was parking six of sixteen decks at 0.84% of episodes each while
    the two decks the teacher could not even pilot took 53% between them -- so
    it was selecting for TEACHER INCOMPETENCE, not for deck difficulty. The
    heavy decks (Royal Giant, Royal Hogs, Mega Knight, P.E.K.K.A.) are exactly
    the ones a 2.6 cycle deck exists to defend against, and are the only place
    the agent can learn to hold a big push.
    """
    w = deck_pool.pfsp_weights({"crushing": 0.00, "hard": 0.25, "easy": 0.85})
    assert w["crushing"] > w["hard"] > w["easy"], (
        "with the floor off, plain (1-wr)^2 must rank the hardest deck first")


def test_the_floor_mechanism_still_works_when_asked_for():
    """Kept, not deleted: `floor=` restores the old behaviour in one argument,
    which is what makes turning it off a reversible decision rather than a
    rewrite."""
    w = deck_pool.pfsp_weights({"hopeless": 0.02, "hard": 0.25, "even": 0.50},
                               floor=0.20)
    assert w["hopeless"] < w["hard"]


def test_the_shipped_pool_now_spends_the_run_on_the_decks_it_loses_to():
    """The inverse of the assertion this file carried until 2026-09-06.

    That one required <10% of episode 0 to go to decks measured unwinnable. It
    was the right guard for a FRESH net -- which is not the case being run: a
    checkpoint at 83k episodes resumed at a LOW rung meets those decks against a
    teacher with a 2-second lookahead and 15% random actions, which is winnable.
    Rung and deck are two difficulty axes and only one of them is being raised.
    """
    decks = deck_pool.load_pool()
    priors = {d.name: d.prior_win_rate for d in decks}
    w = deck_pool.pfsp_weights(priors)
    hard = sum(v for n, v in w.items() if priors[n] < 0.20)
    assert hard > 0.30, (
        f"only {hard:.0%} of episodes go to the decks the agent loses to")
