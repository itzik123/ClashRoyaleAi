"""Search's rollout opponent, and why it needed to become a parameter.

WHAT WAS WRONG. `search_action` rolled candidates forward with `sim.step(...)`,
and the raw engine's `step` drives the C++ HeuristicOpponent. So search
optimised against the heuristic while phase 1's real opponent is the
UtilityTeacher, which forward-simulates. Measured 2026-09-06 on the ep-111k
policy, paired and seeded, greedy control constant at 0.844:

    horizon  4   greedy 0.844   search 0.531   -0.313
    horizon  8   greedy 0.844   search 0.312   -0.531
    horizon 12   greedy 0.844   search 0.375   -0.469

Every positive search result in this repo -- the +0.319, the +0.4025 horizon
sweep -- was measured against that same heuristic, which is why they held then
and do not now. The regime moved; the search did not.

NOTE THE DOCSTRING THAT MISLED. `SearchCfg` says "a candidate rollout assumes
BOTH SIDES NO-OP". Only OUR side no-ops; the opponent has always acted. The
first diagnosis of this bug was built on that sentence and was wrong until the
board was actually inspected -- 2 enemy bodies and 1302 tower HP inside a
rollout that was supposed to be empty.
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
    """The model has to actually drive the rollout. A parameter that is accepted
    and then ignored is the failure this test exists for -- it would leave the
    heuristic in place while every report said otherwise."""
    env = _warm_env()
    opp = _RecordingOpponent()
    S.rollout(env.snapshot(), NOOP, 0.0, 0.0, horizon=5, opponent=opp)
    assert opp.calls == 5, f"opponent consulted {opp.calls} times, expected 5"


def test_without_a_model_the_rollout_keeps_the_old_behaviour():
    """The C++ heuristic path is preserved exactly, so every earlier result
    stays reproducible and this change is additive."""
    env = _warm_env()
    a = S.rollout(env.snapshot(), NOOP, 0.0, 0.0, horizon=6, opponent=None)
    b = S.rollout(env.snapshot(), NOOP, 0.0, 0.0, horizon=6, opponent=None)
    assert np.allclose(np.asarray(a.observation, np.float32),
                       np.asarray(b.observation, np.float32)), (
        "the no-model rollout must stay deterministic")


def test_a_passive_model_and_the_heuristic_diverge():
    """The point of the parameter: a rollout with a do-nothing opponent must
    reach a DIFFERENT board than one driven by the heuristic. If these agreed,
    the opponent would not be being modelled at all."""
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
    """The real case. A UtilityTeacher must be droppable in as the opponent and
    must spend elixir doing it -- an opponent model that never plays is the
    passive rollout with extra cost."""
    from python_ai.opponents.teacher import UtilityTeacher

    env = _warm_env(ticks=60)
    teacher = UtilityTeacher(list(DEFAULT_DECK), team=1, epsilon=0.0,
                             horizon_ticks=0, k_cells=1, max_combos=0)
    teacher.reset()
    sim = env.snapshot()
    before = sim.get_elixir_for_team(1)
    S.rollout(sim, NOOP, 0.0, 0.0, horizon=20, opponent=teacher)
    # Elixir is not a perfect play detector on its own (income accrues), so
    # require that the bar did not simply run up untouched.
    after = sim.get_elixir_for_team(1)
    assert after < before + 20 * 10 * 0.035, (
        f"team 1 elixir went {before:.2f} -> {after:.2f}, i.e. it banked its "
        f"whole income and never played")
