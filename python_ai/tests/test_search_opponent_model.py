"""Search's rollout opponent is a parameter.

Rolling candidates forward with the raw engine's `step` drives the C++
HeuristicOpponent, so without a model search optimises against the heuristic
rather than phase 1's UtilityTeacher. During a rollout only our side no-ops;
the opponent always acts.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401
import clash_royale_env as E  # noqa: E402
from python_ai.envs.gym_wrapper import DEFAULT_DECK  # noqa: E402
from python_ai.search import search as S  # noqa: E402

CE = E.ClashRoyaleEnv
NOOP = CE.HAND_SIZE


def _warm_env(ticks=30):
    env = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), 3600)
    env.reset()
    for _ in range(ticks):
        env.step_self_play(NOOP, 0., 0., NOOP, 0., 0., 10,
                           False, False, False, False)
    return env


class _RecordingOpponent:
    """Never plays, but records that it was asked."""

    def __init__(self):
        self.calls = 0

    def reset(self, rng=None):
        pass

    def act(self, env, obs_own):
        self.calls += 1
        return NOOP, 0.0, 0.0


def test_the_rollout_asks_the_opponent_model_once_per_step():
    """The model must actually drive the rollout; a parameter accepted and then
    ignored would leave the heuristic in place.
    """
    env = _warm_env()
    opp = _RecordingOpponent()
    S.rollout(env.snapshot(), NOOP, 0.0, 0.0, horizon=5, opponent=opp)
    assert opp.calls == 5, f"opponent consulted {opp.calls} times, expected 5"


def test_without_a_model_the_rollout_keeps_the_old_behaviour():
    """The C++ heuristic path is preserved exactly, so earlier results stay
    reproducible.
    """
    env = _warm_env()
    a = S.rollout(env.snapshot(), NOOP, 0.0, 0.0, horizon=6, opponent=None)
    b = S.rollout(env.snapshot(), NOOP, 0.0, 0.0, horizon=6, opponent=None)
    assert np.allclose(np.asarray(a.observation, np.float32),
                       np.asarray(b.observation, np.float32)), (
        "the no-model rollout must stay deterministic")


def test_a_passive_model_and_the_heuristic_diverge():
    """A do-nothing opponent must reach a different board than the heuristic, or
    the opponent is not being modelled.
    """
    env = _warm_env()
    passive = S.rollout(env.snapshot(), NOOP, 0.0, 0.0, horizon=12,
                        opponent=_RecordingOpponent())
    heuristic = S.rollout(env.snapshot(), NOOP, 0.0, 0.0, horizon=12,
                          opponent=None)
    assert not np.allclose(
        np.asarray(passive.observation, np.float32),
        np.asarray(heuristic.observation, np.float32)), (
        "a passive opponent produced the same board as the C++ heuristic")


def test_the_teacher_is_usable_as_the_model_and_actually_plays():
    """A UtilityTeacher can be dropped in and actually spends elixir; a model that
    never plays is the passive rollout at extra cost.
    """
    from python_ai.opponents.teacher import UtilityTeacher

    env = _warm_env(ticks=60)
    teacher = UtilityTeacher(list(DEFAULT_DECK), team=1, epsilon=0.0,
                             horizon_ticks=0, k_cells=1, max_combos=0)
    teacher.reset()
    sim = env.snapshot()
    before = sim.get_elixir_for_team(1)
    S.rollout(sim, NOOP, 0.0, 0.0, horizon=20, opponent=teacher)
    # Elixir is not a perfect play detector (income accrues): require that the
    # bar did not simply run up untouched.
    after = sim.get_elixir_for_team(1)
    assert after < before + 20 * 10 * 0.035, (
        f"team 1 elixir went {before:.2f} -> {after:.2f}, i.e. it banked its "
        f"whole income and never played")
