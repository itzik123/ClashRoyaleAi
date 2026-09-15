"""The teacher activates a Champion's ability -- in a fight, not on arrival.

Until 2026-09-16 the UtilityTeacher never issued an ability (no ability code at
all; gym_wrapper passed False for team 1), so the phase-1 MIRROR of a Champion
deck played the Champion as a plain troop. Readiness turns true the moment a
Champion is deployed with elixir to spare (measured: Golden Knight, Archer
Queen, Monk all ready from their first second), so "activate when ready" would
spend the ability on arrival; the rule also requires an enemy force on the board.
"""
import numpy as np

import clash_royale_env as E
from python_ai.opponents.teacher import UtilityTeacher

CE = E.ClashRoyaleEnv
H = CE.HAND_SIZE
DECK = [6, 116, 40, 24, 72, 33, 7, 52]          # Golden Knight in deck slot 1


def _deployed():
    env = CE(DECK, DECK, 3600)
    env.seed(4)
    assert env.set_hand_for_team(1, [116, 6, 40, 24])
    env.set_elixir_for_team(1, 10.0)
    env.step_self_play(H, 0, 0, 0, 9.0, 10.0, 10)
    env.set_elixir_for_team(1, 10.0)
    assert env.is_champion_ability_ready(1, 1)
    t = UtilityTeacher(DECK, team=1, seed=1)
    t.reset()
    return env, t


def test_a_ready_champion_with_nothing_to_fight_holds_its_ability():
    env, t = _deployed()
    obs = np.asarray(env.get_observation_for_team(1), np.float32)
    assert t.ability_flags(env, obs) == (False, False)


def test_a_ready_champion_uses_its_ability_against_a_real_push():
    env, t = _deployed()
    for x in (8.0, 9.0, 10.0):
        env.inject(13, x, 18.0, 0, -1.0, 0)          # P.E.K.K.A.s at the river
    env.step_self_play(H, 0, 0, H, 0, 0, 1)
    env.set_elixir_for_team(1, 10.0)
    obs = np.asarray(env.get_observation_for_team(1), np.float32)
    assert t.ability_flags(env, obs) == (True, False)
