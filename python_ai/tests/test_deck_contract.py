"""`validate_deck`: every deck-dependent mechanism that would otherwise go SILENT.

The pre-launch audit (2026-09-15) found the same bug four times: a mechanism
keyed to the 2.6 deck that, under another deck, quietly contributed nothing --
the advisor-target term, the win-condition reward, the Fireball-keyed spell
terms, the Fireball-only scenarios. None raised or logged. The general antidote
is one preflight that says, at the top of the run log, what this deck turns on
and what it turns off.

ERROR  -- the run cannot train this deck (a Champion: ability sampling is not
          implemented, and base_trainer raises anyway -- say why, up front).
WARN   -- the run will train, with a named mechanism off or weak.
INFO   -- facts worth having in the log.
"""
import pytest

from python_ai import deck as D
from python_ai.envs import deck_contract as DC


def levels(report):
    return {lvl for lvl, _ in report}


def text(report):
    return "\n".join(f"{lvl} {msg}" for lvl, msg in report)


def test_the_shipped_deck_has_no_errors_and_names_its_win_condition():
    report = DC.validate_deck(list(D.SHIPPED_DECK), strict=False)
    assert "ERROR" not in levels(report), text(report)
    assert "Hog Rider" in text(report)


def test_a_champion_deck_trains_and_says_the_ability_path_is_new():
    """Refused until 2026-09-16, when ability training landed. Still flagged:
    the path is new and the mirror teacher uses the ability heuristically."""
    deck = D.parse_deck("musketeer,golden knight,ice golem,skeletons,ice spirit,"
                        "the log,fireball,miner")
    report = DC.validate_deck(deck, strict=False)
    assert "ERROR" not in levels(report), text(report)
    assert any(lvl == "WARN" and "ability" in msg.lower() for lvl, msg in report)
    DC.validate_deck(deck, strict=True)


def test_a_deck_without_fireball_keys_the_spell_terms_to_its_own_spell():
    """This deck WARNED "no Fireball: the spell terms contribute nothing" until
    2026-09-16. They follow the deck's own damage spell now -- here Lightning --
    so the report names it, with the numbers the terms actually use."""
    deck = D.parse_deck("royal giant,fisherman,hunter,electro spirit,skeletons,"
                        "lightning,the log,cannon")
    report = DC.validate_deck(deck, strict=False)
    spell_lines = [(lvl, msg) for lvl, msg in report if "spell reward terms" in msg]
    assert spell_lines, text(report)
    lvl, msg = spell_lines[0]
    assert lvl == "INFO" and "Lightning" in msg, msg
    assert "1057" in msg and "6 elixir" in msg, msg


def test_a_deck_with_no_damaging_spell_warns_that_the_spell_terms_are_off():
    """The Log is a ROLLER, which the damage-spell resolver declines (its value
    is a corridor, not a disc), so a Log-only deck has no finishing spell."""
    deck = D.parse_deck("hog rider,musketeer,cannon,ice golem,skeletons,"
                        "ice spirit,the log,knight")
    report = DC.validate_deck(deck, strict=False)
    assert any(lvl == "WARN" and "no damaging area spell" in msg
               for lvl, msg in report), text(report)


def test_advisor_coverage_is_reported():
    deck = D.parse_deck("miner,poison,bomber,musketeer,valkyrie,skeletons,"
                        "ice spirit,the log")
    report = DC.validate_deck(deck, strict=False)
    assert "advisor" in text(report).lower()


def test_a_deck_with_no_win_condition_warns():
    deck = D.parse_deck("knight,musketeer,valkyrie,archers,skeletons,"
                        "ice spirit,the log,fireball")
    report = DC.validate_deck(deck, strict=False)
    assert any(lvl in ("WARN", "ERROR") and "win condition" in msg.lower()
               for lvl, msg in report), text(report)
