"""Scenario injection must not spend its budget on boards the deck cannot answer.

Two of the four scenarios (60% of the injection weight, 18% of ALL phase-1
episodes at SCENARIO_INJECTION_PROB = 0.30) build a board whose correct answer is
an area-damage spell -- a swarm in our half, troops hugging an enemy tower. With
no such spell in the trainee's deck they still fired, teaching a reflex the deck
cannot express. `giant_commit` was removed on 2026-08-19 for exactly this at
17% of the budget ("a dead entry here is worse than none"). The weights now
follow the deck: a spell scenario draws only if the deck holds a spell that can
answer it.
"""
import numpy as np

from python_ai import deck as D
from python_ai.envs import scenarios

SPELL_SCENARIOS = {"fireball_swarm", "fireball_tower_value"}


def _names(deck, n=400):
    rng = np.random.default_rng(0)
    return {scenarios.sample_scenario(rng, deck=deck)["name"] for _ in range(n)}


def test_the_shipped_deck_still_draws_every_scenario():
    assert _names(list(D.SHIPPED_DECK)) >= SPELL_SCENARIOS | {"bridge_push",
                                                              "bridge_push_supported"}


def test_a_deck_with_no_area_spell_never_draws_a_spell_scenario():
    deck = D.parse_deck("hog rider,musketeer,cannon,ice golem,skeletons,"
                        "ice spirit,the log,knight")
    drawn = _names(deck)
    assert not (drawn & SPELL_SCENARIOS), drawn
    assert drawn == {"bridge_push", "bridge_push_supported"}


def test_the_default_call_uses_the_trainee_deck():
    """Callers pass only an rng; the deck defaults to python_ai.deck."""
    rng = np.random.default_rng(1)
    assert scenarios.sample_scenario(rng)["name"]


def test_the_share_is_reported_for_the_deck_contract():
    assert scenarios.spell_scenario_share(list(D.SHIPPED_DECK)) > 0.5
    deck = D.parse_deck("hog rider,musketeer,cannon,ice golem,skeletons,"
                        "ice spirit,the log,knight")
    assert scenarios.spell_scenario_share(deck) == 0.0
