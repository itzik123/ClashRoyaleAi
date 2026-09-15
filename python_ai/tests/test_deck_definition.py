"""The trainee's deck: one definition, settable without editing code.

Until 2026-09-15 the only way to change the deck was to edit the
`DEFAULT_DECK` literal in `envs/gym_wrapper.py` -- a file CLAUDE.md marks
read-only -- with no validation and nothing in the run's log recording which
deck produced it. `python_ai.deck` owns the definition now and reads
`CLASH_DECK` (ids or names), and a bad deck fails at parse time with the
engine's own reason instead of as a stack trace inside module import.
"""
import pytest

import clash_royale_env as E
from python_ai import deck as D


def test_the_shipped_deck_is_the_default():
    assert D.parse_deck(None) == list(D.SHIPPED_DECK)
    assert D.parse_deck("") == list(D.SHIPPED_DECK)


def test_a_deck_can_be_given_by_id():
    assert D.parse_deck("15, 6,25,40,24,72,33,7") == [15, 6, 25, 40, 24, 72, 33, 7]


def test_a_deck_can_be_given_by_name_ignoring_case_dots_and_spaces():
    deck = D.parse_deck("hog rider,musketeer,cannon,ice golem,skeletons,"
                        "ice spirit,the log,fireball")
    assert deck == [15, 6, 25, 40, 24, 72, 33, 7]
    assert D.parse_deck("pekka,minipekka,knight,archers,zap,arrows,"
                        "fireball,minions")[:2] == [13, 5]


def test_an_evolution_is_named_with_an_evo_prefix():
    """Evolutions reuse the base card's name verbatim (id 1 and id 128 are both
    'Archers'), so a bare name must mean the base card and an Evolution has to
    be asked for explicitly."""
    deck = D.parse_deck("evo:archers,knight,musketeer,skeletons,ice spirit,"
                        "the log,fireball,cannon")
    assert deck[0] == 128
    assert D.parse_deck("archers,knight,musketeer,skeletons,ice spirit,"
                        "the log,fireball,cannon")[0] == 1


def test_an_unknown_card_names_the_token():
    with pytest.raises(ValueError, match="no card named 'hog ryder'"):
        D.parse_deck("hog ryder,6,25,40,24,72,33,7")


def test_a_deck_the_engine_refuses_raises_with_the_engine_reason():
    """A Champion is only legal in deck slots 1 and 2."""
    with pytest.raises(ValueError, match="slot 0"):
        D.parse_deck("115,6,25,40,24,72,33,7")


def test_gym_wrapper_reexports_the_one_definition():
    from python_ai.envs import gym_wrapper
    assert gym_wrapper.DEFAULT_DECK == D.DEFAULT_DECK
