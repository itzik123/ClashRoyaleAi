"""The pre-flight runs under any deck, and one crashing check cannot hide the rest.

On 2026-09-27 `validate_advisor` raised `KeyError: 25` under a deck without a
Cannon: its value readout indexed `legal[CANNON_ID]`, and `legal` holds only
the cards this deck's advisor covers. The run aborted there, so the side null
never ran. Same class as the 2026-09-15 audit's findings: keyed to the 2.6 deck.
"""
import pytest

from python_ai.advisors import advisor_target as AT
from python_ai.advisors import tactics
from python_ai.deck import SHIPPED_DECK, parse_deck
from python_ai.tools import validate_pipeline as VP

# The deck that crashed it: neither a Cannon nor a Fireball.
CANNONLESS = ("evo:furnace,hero barbarian barrel,evo:bats,poison,"
              "giant skeleton,graveyard,berserker,goblin hut")

ADVISOR_CHECKS = {
    "no target puts mass on a cell the ENGINE refuses",
    "the engine legality probe actually ran",
    "the gate declines on an empty board",
    "the advisor speaks often enough to train on",
}


@pytest.fixture
def cannonless_deck(monkeypatch):
    deck = parse_deck(CANNONLESS)
    assert tactics.CANNON_ID not in deck and tactics.FIREBALL_ID not in deck
    # What CLASH_DECK does at import, for the globals validate_advisor reads.
    monkeypatch.setattr(VP, "DECK", list(deck))
    monkeypatch.setattr(AT, "ADVISOR_CARDS", AT.advisor_cards_for(deck))
    return deck


def test_value_readout_follows_the_deck():
    shipped_legal = {tactics.CANNON_ID: None, tactics.FIREBALL_ID: None}
    # Control that must fire: the shipped deck keeps both readouts.
    assert VP.value_readout_cards(SHIPPED_DECK, shipped_legal) == {
        tactics.CANNON_ID, tactics.FIREBALL_ID}
    assert VP.value_readout_cards(parse_deck(CANNONLESS), {}) == set()
    # In the deck but not in `legal` would still be a KeyError.
    assert VP.value_readout_cards(SHIPPED_DECK, {tactics.FIREBALL_ID: None}) == {
        tactics.FIREBALL_ID}


def test_the_advisor_check_runs_under_a_cannonless_deck(cannonless_deck,
                                                        monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(VP, "check",
                        lambda name, ok, detail="": seen.append((name, ok, detail)))
    VP.validate_advisor(episodes=1)

    # It reached the asserted part, not just past the crash site.
    assert {n for n, _, _ in seen} == ADVISOR_CHECKS, seen
    out = capsys.readouterr().out
    assert "Cannon value readout skipped" in out
    assert "Fireball value readout skipped" in out


def test_a_crashing_check_does_not_hide_the_ones_after_it(monkeypatch):
    ran = []
    validators = ("validate_pfsp", "validate_scenarios", "validate_spell_anneal",
                  "validate_search", "validate_cpp", "validate_side_null")
    for name in validators:
        monkeypatch.setattr(VP, name, lambda *a, _n=name: ran.append(_n))

    def crash(*_a):
        raise KeyError(25)

    monkeypatch.setattr(VP, "validate_advisor", crash)
    monkeypatch.setattr(VP, "RESULTS", [])

    rc = VP.main(["--advisor-episodes", "1", "--null-episodes", "1"])

    assert set(ran) == set(validators)          # the side null still ran
    assert rc == 1                              # ...and the crash is a failure
    crashed = [(n, d) for n, ok, d in VP.RESULTS if not ok]
    assert len(crashed) == 1 and "KeyError" in crashed[0][1], VP.RESULTS
