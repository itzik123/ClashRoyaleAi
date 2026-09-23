"""Reactive rollouts: the rollout opponent answers an attacking candidate instead
of standing still.

A no-op rollout is blind to the cost of being answered, and its error runs
entirely in favour of attacking. The fix does not make the teacher pick better
plays; it stops it committing to pushes a real opponent would punish. The
counter is open-loop (it reads no observation inside the rollout): every
responder tried beat the no-op control and none beat another, so the cheapest
was kept.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env as E  # noqa: E402
from python_ai.advisors import tactics  # noqa: E402
from python_ai.envs import gym_wrapper  # noqa: E402
from python_ai.opponents import teacher as T  # noqa: E402

CE = E.ClashRoyaleEnv
DECK = list(gym_wrapper.DEFAULT_DECK)
HOG = 15


def _env(ticks=40, seed=17):
    env = CE(DECK, DECK, 3600)
    env.seed(seed)
    for _ in range(ticks):
        env.step_self_play(-1, 0, 0, -1, 0, 0, 10)
    return env


def _teacher(team=0, reactive=True, **kw):
    t = T.UtilityTeacher(DECK, team=team, horizon_ticks=100,
                         profile="balanced", reactive_rollout=reactive, **kw)
    t.reset(np.random.default_rng(0))
    return t


def _bridge_push(env, team):
    """A naked win-condition placement at the advisor's bridge cell, or None."""
    obs = np.asarray(env.get_observation_for_team(team), np.float32)
    hand = list(env.get_hand_for_team(team))
    if HOG not in hand[:CE.HAND_SIZE]:
        return None
    slot = hand.index(HOG)
    x, y, _ = tactics.best_hog_cell(obs)
    xi, yi = float(int(x)), float(int(y))
    mirror = float(CE.BOARD_HEIGHT - 1)
    y_abs = yi if team == 0 else mirror - yi
    if not env.is_valid_placement(HOG, xi, y_abs, team):
        return None
    return T.Candidate.single(slot, HOG, xi, yi, "wincon")


def test_the_rollout_opponent_spends_elixir_answering_an_attacking_candidate():
    """With reactive rollouts, a win condition walked to the bridge is answered
    inside the rollout and the answer costs the opponent elixir. Asserted on
    the opponent's spend, since damage is confounded by our own towers
    shooting.
    """
    env = _env()
    env.set_elixir_for_team(0, 10.0)
    env.set_elixir_for_team(1, 10.0)
    env.set_hand_for_team(0, [HOG, DECK[4], DECK[5], DECK[1]])
    push = _bridge_push(env, 0)
    assert push is not None, "fixture must offer a legal bridge push"

    t = _teacher(team=0, reactive=True)
    s = env.snapshot()
    before = s.get_elixir_spent(1)
    t.execute_steps(s, push, t.rollout_ticks())
    assert s.get_elixir_spent(1) > before


def test_a_card_played_in_our_own_half_draws_no_counter():
    """Only an attacking placement draws a counter; answering defence would price
    it as an attack.
    """
    t = _teacher(team=0, reactive=True)
    own_half = T.Candidate.single(0, DECK[4], 9.0, 3.0, "melee")
    assert t.counter_schedule(own_half) == {}

    attacking = T.Candidate.single(0, HOG, 4.0, float(tactics.BRIDGE_ROW),
                                   "wincon")
    assert t.counter_schedule(attacking) != {}


def test_reactive_rollout_False_is_the_old_no_op_rollout_exactly():
    """The off switch: with it off the rollout opponent never acts, so it spends
    nothing.
    """
    env = _env()
    env.set_elixir_for_team(0, 10.0)
    env.set_elixir_for_team(1, 10.0)
    env.set_hand_for_team(0, [HOG, DECK[4], DECK[5], DECK[1]])
    push = _bridge_push(env, 0)
    assert push is not None

    t = _teacher(team=0, reactive=False)
    assert t.counter_schedule(push) == {}
    s = env.snapshot()
    before = s.get_elixir_spent(1)
    t.execute_steps(s, push, t.rollout_ticks())
    assert s.get_elixir_spent(1) == before


@pytest.mark.parametrize("team", [0, 1])
def test_the_counter_is_placed_in_the_opponents_own_frame_for_either_side(team):
    """One class plays both sides, and the counter must mirror with it:
    `step_self_play` mirrors team 1's y, `is_valid_placement` takes absolute y.
    Wrong frames make every counter silently refused, so this asserts the
    opponent actually spends.
    """
    env = _env()
    env.set_elixir_for_team(0, 10.0)
    env.set_elixir_for_team(1, 10.0)
    env.set_hand_for_team(team, [HOG, DECK[4], DECK[5], DECK[1]])
    push = _bridge_push(env, team)
    assert push is not None

    t = _teacher(team=team, reactive=True)
    s = env.snapshot()
    before = s.get_elixir_spent(1 - team)
    t.execute_steps(s, push, t.rollout_ticks())
    assert s.get_elixir_spent(1 - team) > before, (
        f"team {team}'s rollout opponent never answered -- frame bug")


def test_the_noop_still_scores_exactly_zero():
    """The no-op must still score exactly 0.0 now that the opponent acts in
    candidate rollouts.
    """
    env = _env()
    t = _teacher(team=0, reactive=True)
    obs = np.asarray(env.get_observation_for_team(0), np.float32)
    noop = [c for c in t.candidates(env, obs) if c.slot == CE.HAND_SIZE][0]
    base = t.rollout_stats(env, noop)
    assert t.score(env, noop, base, obs) == 0.0


def test_the_reacting_rollout_still_does_not_touch_the_live_match():
    """The rollout opponent places cards, so snapshot() must keep deep-copying the
    stats collectors or the teacher corrupts the returns its student is scored
    against.
    """
    env = _env()
    before = (env.get_tower_damage_dealt(0), env.get_tower_damage_dealt(1),
              env.get_elixir_spent(0), env.get_elixir_spent(1),
              env.get_elixir_for_team(0), env.get_elixir_for_team(1))
    t = _teacher(team=0, reactive=True)
    t.act(env, np.asarray(env.get_observation_for_team(0), np.float32))
    after = (env.get_tower_damage_dealt(0), env.get_tower_damage_dealt(1),
             env.get_elixir_spent(0), env.get_elixir_spent(1),
             env.get_elixir_for_team(0), env.get_elixir_for_team(1))
    assert before == after


def test_the_counter_never_plays_a_card_the_opponent_cannot_afford():
    """The counter is filtered through the opponent's real elixir; a free answer
    would make every attack look punished.
    """
    env = _env()
    env.set_elixir_for_team(0, 10.0)
    env.set_elixir_for_team(1, 0.0)          # opponent is broke
    env.set_hand_for_team(0, [HOG, DECK[4], DECK[5], DECK[1]])
    push = _bridge_push(env, 0)
    assert push is not None

    t = _teacher(team=0, reactive=True)
    s = env.snapshot()
    slot, x, y = t.counter_action(s, (push.x, push.y))
    assert slot == -1, "a broke opponent must not be able to answer"


def test_reactivity_is_a_competence_axis_and_the_short_rungs_do_without_it():
    """Reactivity is on only at the top rung, the one rung whose horizon clears
    COUNTER_MIN_HORIZON_TICKS (see the too-short test below). That also keeps
    it off at cold start, when the modelled agent cannot defend at all.
    """
    # A property, not a literal list, so it survives the table changing length.
    reactive = [cfg["reactive"] for cfg in T.TEACHER_STAGES]
    top = len(T.TEACHER_STAGES) - 1
    assert reactive[top] is True, "the top rung is where +0.1500 was measured"
    assert not any(reactive[:top]), (
        "no rung below the top may react -- every one of them prices attacks "
        "below their true value")

    t = _teacher(team=0, reactive=True)
    t.set_stage(0)
    assert t.reactive_rollout is False
    t.set_stage(top)
    assert t.reactive_rollout is True


def test_a_naked_bridge_push_scores_LOWER_when_the_opponent_answers():
    """A naked push must score lower when the opponent answers. A differential on
    one candidate, since an absolute threshold would move with the board, deck
    and profile.
    """
    env = _env()
    env.set_elixir_for_team(0, 10.0)
    env.set_elixir_for_team(1, 10.0)
    env.set_hand_for_team(0, [HOG, DECK[4], DECK[5], DECK[1]])
    push = _bridge_push(env, 0)
    assert push is not None
    obs = np.asarray(env.get_observation_for_team(0), np.float32)

    scores = {}
    for on in (False, True):
        t = _teacher(team=0, reactive=on)
        base = t.rollout_stats(env, T.NOOP)
        scores[on] = t.score(env, push, base, obs)

    assert scores[True] < scores[False], (
        f"reacting must not make a naked push look BETTER: "
        f"off={scores[False]:.3f} on={scores[True]:.3f}")


def test_the_counter_does_not_fire_when_the_rollout_is_too_short_to_see_the_payoff():
    """The counter's cost lands at +10 ticks while a Hog's payoff needs ~130, so a
    rollout shorter than the crossing charges the answer and credits none of
    the push: an anti-attack bias that freezes the teacher against a passive
    opponent. Same argument as COMBO_MIN_HORIZON_TICKS.
    """
    push = T.Candidate.single(0, HOG, 2.0, float(tactics.BRIDGE_ROW), "wincon")

    short = T.UtilityTeacher(DECK, team=0, profile="balanced",
                             horizon_ticks=30, reactive_rollout=True)
    short.reset(np.random.default_rng(0))
    assert short.counter_schedule(push) == {}, (
        "a 30-tick rollout cannot see a Hog arrive, so charging it for the "
        "answer prices the push below its true value")

    full = T.UtilityTeacher(DECK, team=0, profile="balanced",
                            horizon_ticks=T.COUNTER_MIN_HORIZON_TICKS,
                            reactive_rollout=True)
    full.reset(np.random.default_rng(0))
    assert full.counter_schedule(push) != {}


def test_the_ladder_only_enables_reacting_where_it_was_measured():
    """`reactive` must not be on at a rung whose horizon the gate rejects, or the
    table claims a behaviour the code declines to perform.
    """
    for i, cfg in enumerate(T.TEACHER_STAGES):
        if cfg["reactive"]:
            assert cfg["horizon_ticks"] >= T.COUNTER_MIN_HORIZON_TICKS, (
                f"stage {i} enables reacting at horizon "
                f"{cfg['horizon_ticks']}, below the gate")
