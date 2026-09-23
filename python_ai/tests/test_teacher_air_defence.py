"""The teacher answers a flying building-targeter with a card that can hit it.

A Balloon ignores every troop, so a ground-only card dropped under it is an
elixir gift. These pin the mechanism on real engine boards.
"""
import numpy as np
import pytest

import python_ai  # noqa: F401
import clash_royale_env as E

from python_ai.advisors import card_probes, tactics
from python_ai.opponents import teacher as T

ID = {E.get_card_info(c)["name"]: c for c in E.get_all_card_ids()}
DECK = [15, 6, 25, 40, 24, 72, 33, 7]      # the 2.6 deck: two anti-air troops


def _board(*units):
    """Team 0's observation of enemy units [(name, x, y)] on its half."""
    env = E.ClashRoyaleEnv(list(DECK), list(DECK), 3600)
    env.seed(1)
    for name, x, y in units:
        env.inject(ID[name], x, y, 1, -1.0, 300)
    env.step_self_play(4, 0.0, 0.0, 4, 0.0, 0.0, 1)
    return np.asarray(env.get_observation_for_team(0), np.float32)


@pytest.mark.parametrize("name", ["Musketeer", "Ice Spirit", "Archers", "Tesla",
                                  "Minions", "Fireball", "Arrows"])
def test_cards_that_hit_air_are_recognised(name):
    assert card_probes.damages_air(ID[name]), name


@pytest.mark.parametrize("name", ["Knight", "Skeletons", "Ice Golem", "Hog Rider",
                                  "Cannon", "Bomb Tower", "The Log", "Earthquake"])
def test_ground_only_cards_are_recognised(name):
    assert not card_probes.damages_air(ID[name]), name


def test_the_air_siege_map_sees_a_balloon_and_nothing_a_ground_unit_can_answer():
    assert tactics.air_siege_map(_board(("Balloon", 9.0, 11.0))).sum() > 0.0
    # A flyer that chases troops, and a walking building-targeter: neither
    # needs anti-air specifically.
    assert tactics.air_siege_map(_board(("Minions", 9.0, 11.0))).sum() == 0.0
    assert tactics.air_siege_map(_board(("Hog Rider", 9.0, 11.0))).sum() == 0.0


def _gate(obs, *names):
    """Drive the real rung-0 gate; return the card it plays, or None."""
    teacher = T.UtilityTeacher(list(DECK), team=0)
    cands = [T.Candidate.single(i, ID[n], 8.0, 10.0,
                                role=teacher.roles.get(ID[n], "melee"))
             for i, n in enumerate(names)]
    slot, _x, _y = teacher._rules_only(obs, cands)
    return None if slot >= E.ClashRoyaleEnv.HAND_SIZE else names[slot]


def test_a_balloon_is_answered_by_the_card_that_can_hit_it():
    obs = _board(("Balloon", 9.0, 11.0))
    assert _gate(obs, "Skeletons", "Musketeer") == "Musketeer"


def test_a_ground_only_card_is_held_back_from_a_lone_balloon():
    """With nothing that can hit it in hand, waiting beats gifting elixir."""
    obs = _board(("Balloon", 9.0, 11.0))
    assert _gate(obs, "Skeletons", "Ice Golem") is None


def test_a_ground_card_still_answers_the_ground_half_of_a_mixed_push():
    obs = _board(("Balloon", 9.0, 11.0), ("Hog Rider", 4.0, 9.0))
    assert _gate(obs, "Skeletons") == "Skeletons"


def test_with_no_air_threat_the_gate_decides_exactly_as_before():
    """Control: a lone Hog. Both cards score the old 3.0 and the first listed
    wins.
    """
    obs = _board(("Hog Rider", 9.0, 11.0))
    assert _gate(obs, "Skeletons", "Musketeer") == "Skeletons"
    assert _gate(obs, "Musketeer", "Skeletons") == "Musketeer"


def test_an_anti_air_card_is_aimed_at_the_balloon_not_the_deeper_ground_unit():
    teacher = T.UtilityTeacher(list(DECK), team=0)
    obs = _board(("Balloon", 12.0, 12.0), ("Knight", 4.0, 7.0))
    mx, my = teacher._cells_for("ranged", ID["Musketeer"], obs)[0]
    sx, sy = teacher._cells_for("melee", ID["Skeletons"], obs)[0]
    assert (mx, my) != (sx, sy)
    assert abs(mx - 12.0) <= 1.0, (mx, my)      # the Balloon's column
    assert abs(sx - 4.0) <= 1.0, (sx, sy)       # the deeper Knight's
