"""The card map's contract, and the defined behaviour for unmapped cards."""

from __future__ import annotations

import pytest

import mapping
from mapping import DeckNotRepresentableError, UnknownCardError


def test_every_registered_card_is_present(engine):
    """The map must cover the whole registry, or the generator is stale."""
    for card_id in engine.get_all_card_ids():
        name = engine.get_card_info(card_id)["name"]
        entry = mapping.resolve(name)
        assert entry.sim_id == card_id or entry.sim_evo_id == card_id, (
            f"{name!r} (id {card_id}) resolves to sim_id={entry.sim_id}"
        )


def test_evolutions_are_paired_not_conflated(engine):
    """Evolutions share the base card's name (ids 1 and 128 are both "Archers"),
    so the map must split them or an evolved and a base play become
    indistinguishable.
    """
    archers = mapping.resolve("Archers")
    assert archers.sim_id == 1
    assert archers.sim_evo_id == 128
    assert archers.sim_id_for(evolution=False) == 1
    assert archers.sim_id_for(evolution=True) == 128


def test_evolution_falls_back_to_base_when_unregistered():
    entry = mapping.resolve("Hog Rider")
    assert entry.sim_evo_id is None
    # An un-evolved Hog Rider beats no Hog Rider: unlike an unmapped card,
    # there is no ambiguity about which card was played.
    assert entry.sim_id_for(evolution=True) == entry.sim_id


def test_unknown_name_raises_rather_than_returning_minus_one():
    """Case 3 must not collapse into case 2 (mapping/__init__.py)."""
    with pytest.raises(UnknownCardError, match="stale map"):
        mapping.resolve("Definitely Not A Clash Royale Card")


def test_slug_normalisation_survives_the_registrys_own_inconsistency():
    """id 13 is 'P.E.K.K.A.' and id 5 'Mini PEKKA': a scheme preserving dots would
    key one real card two ways.
    """
    assert mapping.slugify("P.E.K.K.A.") == "pekka"
    assert mapping.slugify("Mini PEKKA") == "minipekka"
    assert mapping.resolve("P.E.K.K.A.").sim_id == 13
    assert mapping.resolve("pekka").sim_id == 13


def test_detector_vocabulary_excludes_unverified_cards():
    """The Hero cards and Spirit Empress stay out of the label set: unconfirmed in
    the recorded build, and dead classes produce confident false positives on
    adjacent real cards.
    """
    vocabulary = set(mapping.detector_vocabulary())
    assert "hogrider" in vocabulary
    assert "heroknight" not in vocabulary
    assert "spiritempress" not in vocabulary


def test_deck_with_an_unimplemented_card_refuses_to_start():
    """A deck with an unimplemented card cannot be simulated at all; refuse, and
    name the card.
    """
    real_deck = ["Hog Rider", "Cannon", "Musketeer", "Archers",
                 "Knight", "Minions", "Fireball", "Valkyrie"]
    assert mapping.require_sim_deck(real_deck) == [15, 25, 6, 1, 0, 41, 7, 10]

    entries = mapping._entries()
    unimplemented = [e for e in entries.values() if e.sim_id is None]
    if not unimplemented:
        pytest.skip("no sim_id: null rows in the map yet -- add real-only cards")

    bad = real_deck[:-1] + [unimplemented[0].real_name]
    with pytest.raises(DeckNotRepresentableError, match="cannot be simulated"):
        mapping.require_sim_deck(bad)


def test_review_queue_is_reported_not_hidden():
    """Rows the generator could not fully determine must stay visible."""
    queue = mapping.review_queue()
    assert queue, "expected Evolution pairings and the >=165 ids to need review"
    slugs = {e.slug for e in queue}
    assert "heroknight" in slugs
