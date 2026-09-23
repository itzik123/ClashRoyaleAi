"""The phase-1 meta-deck pool: loading, validation and PFSP weighting.

Every failure here is silent by nature (a dropped deck, a misspelt card, an
unusable Champion all still train, on a different opponent distribution), so
each one raises and these tests pin that.
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


# --- the shipped pool ---
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
    """The mirror is one matchup of many, never deleted: it is still the single
    most informative matchup.
    """
    decks = {d.name: d for d in deck_pool.load_pool()}
    assert "hog_26_mirror" in decks
    assert sorted(decks["hog_26_mirror"].card_ids) == sorted(
        gym_wrapper.DEFAULT_DECK)


def test_the_pool_is_not_mostly_the_agents_own_archetype():
    """The pool must produce the tanks, clusters, swarms and sieges the defensive
    cards answer. Checked on tags, so swapping a deck for another of its
    archetype keeps passing while gutting the variety does not.
    """
    tags = {t for d in deck_pool.load_pool() for t in d.tags}
    for needed in ("beatdown", "swarm", "cluster", "siege"):
        assert needed in tags, f"the pool offers nothing tagged {needed!r}"


# --- validation ---
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
    """A Champion held by a teacher that cannot use its ability is a worse card,
    so the deck would silently stop being the deck it is named after.
    """
    path = _write(tmp_path, [_deck(cards=[
        "Archer Queen", "Archers", "Giant", "Arrows",
        "Goblins", "Musketeer", "Fireball", "Skeletons"])])
    with pytest.raises(deck_pool.DeckPoolError) as e:
        deck_pool.load_pool(path)
    assert "Champion" in str(e.value)
    # ...and the escape hatch works.
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


# --- PFSP ---
def test_pfsp_concentrates_on_hard_but_winnable_decks():
    w = deck_pool.pfsp_weights({"easy": 0.95, "even": 0.50, "hard": 0.25})
    assert w["hard"] > w["even"] > w["easy"], (
        "PFSP must prefer the deck the agent is doing WORST against, among "
        "those it can still win")
    assert sum(w.values()) == pytest.approx(1.0)


def test_a_deck_below_the_win_rate_floor_is_parked_at_the_minimum():
    """With the floor requested, an unwinnable matchup must not get the largest
    weight, as uncapped (1-wr)^2 would give it.
    """
    w = deck_pool.pfsp_weights({"hopeless": 0.02, "hard": 0.25, "even": 0.50},
                               floor=0.20)
    assert w["hopeless"] < w["hard"], (
        "an unwinnable deck must not attract the most training time")


def test_nothing_is_ever_weighted_to_zero():
    """A deck that stops being sampled is one the policy is free to forget (as
    PFSP_MIN_WEIGHT in phase 2).
    """
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


def test_a_pool_nothing_can_beat_yet_is_sampled_uniformly():
    """The state a fresh run starts in: when nothing clears the requested gate
    there is no usable ranking, so shares are equal (see
    test_deck_pool_cold_start.py).
    """
    w = deck_pool.pfsp_weights({"awful": 0.02, "bad": 0.10, "least_bad": 0.19},
                               floor=0.20)
    assert max(w.values()) == pytest.approx(min(w.values()))
    assert sum(w.values()) == pytest.approx(1.0)


def test_one_winnable_deck_is_enough_to_restore_normal_pfsp():
    """The fallback is for "nothing works", not "most things do not"."""
    w = deck_pool.pfsp_weights({"awful": 0.02, "bad": 0.10, "ok": 0.55},
                               floor=0.20)
    assert w["ok"] > w["awful"], "normal PFSP must resume once anything clears"


def test_the_shipped_pool_keeps_its_old_priors_only_as_provenance():
    """The old priors stay in the file under a key the loader does not read; the
    run starts from NEUTRAL_PRIOR (see test_deck_pool_cold_start.py).
    """
    import json
    raw = json.loads(open(deck_pool.DEFAULT_POOL_PATH, encoding="utf-8").read())         if hasattr(deck_pool, "DEFAULT_POOL_PATH") else None
    decks = deck_pool.load_pool()
    assert all(d.prior_win_rate == deck_pool.NEUTRAL_PRIOR for d in decks)
    if raw is not None:
        entries = raw["decks"] if isinstance(raw, dict) and "decks" in raw else raw
        assert any("prior_win_rate_vs_26hog_2026_09_03" in e for e in entries
                   if isinstance(e, dict))


# --- the win-rate floor is off by default; the mechanism is kept ---

def test_the_win_rate_floor_is_off_by_default():
    """A deck the agent cannot beat yet gets the most training time: the heavy
    decks are exactly what a cycle deck exists to defend against. The floor had
    selected for teacher incompetence rather than deck difficulty.
    """
    w = deck_pool.pfsp_weights({"crushing": 0.00, "hard": 0.25, "easy": 0.85})
    assert w["crushing"] > w["hard"] > w["easy"], (
        "with the floor off, plain (1-wr)^2 must rank the hardest deck first")


def test_the_floor_mechanism_still_works_when_asked_for():
    """Kept: `floor=` restores the old behaviour in one argument, so turning it
    off is reversible.
    """
    w = deck_pool.pfsp_weights({"hopeless": 0.02, "hard": 0.25, "even": 0.50},
                               floor=0.20)
    assert w["hopeless"] < w["hard"]


def test_the_shipped_pool_starts_uniform_for_a_new_policy():
    """For a from-scratch run nothing is known yet: equal shares, ranked by the
    live estimator within ~10 matches per deck.
    """
    decks = deck_pool.load_pool()
    w = deck_pool.pfsp_weights({d.name: d.prior_win_rate for d in decks})
    assert max(w.values()) == pytest.approx(min(w.values()))
