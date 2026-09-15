"""The top rung's reactive counter must not assume an opponent who never answers.

Measured 2026-09-15 (audit 05, BUG 2). At rung 10 `counter_schedule` answers EVERY
attacking placement 10 ticks later with the opponent's cheapest affordable body.
Against a PASSIVE opponent sitting on 10 elixir that imagined answer is always
affordable, so every offensive candidate scored negative and the teacher held
forever: 30 of 116 top-rung matches against a do-nothing opponent froze, bar full
on 65-96% of decisions, six at zero tower damage. `classic_log_bait_inferno` went
1/3/0/3 crowns over seeds 1-4, 3/3/3/3 with the counter off.

Rung 10 is where the curriculum ENDS, and the states it froze in -- an agent
banking elixir and holding -- are exactly where defending a push is learned.

The counter itself is right and stays: it was adopted on +0.158 paired win rate
against an ACTIVE opponent. What was wrong is that it was unconditional. It now
switches off while the real opponent has not spent elixir for
COUNTER_PASSIVE_DECISIONS decisions, and back on the moment they play.
"""
import numpy as np
import pytest

import clash_royale_env as E
from python_ai.opponents import teacher as T

CE = E.ClashRoyaleEnv
HAND = CE.HAND_SIZE
TOP = len(T.TEACHER_STAGES) - 1


def _teacher(deck, seed=3):
    t = T.UtilityTeacher(list(deck), team=0, seed=seed)
    t.set_stage(TOP)
    t.reset()
    return t


def _attack():
    """One card placed at the bridge row -- an attacking placement."""
    step = T.PlacementStep(0, 15, 3.0, float(T.tactics.BRIDGE_ROW), 0)
    return T.Candidate((step,), "wincon", "single")


def test_the_counter_is_on_against_an_opponent_who_plays():
    """CONTROL: an opponent spending elixir still draws the modelled answer."""
    deck = [15, 6, 25, 40, 24, 72, 33, 7]
    env = CE(deck, deck, 3600)
    env.seed(3)
    t = _teacher(deck)
    for _ in range(3):
        obs = np.asarray(env.get_observation_for_team(0), np.float32)
        t.act(env, obs)
        env.step_self_play(HAND, 0.0, 0.0, 0, 9.0, 10.0, 10)   # opponent plays slot 0
    assert t.counter_schedule(_attack()), "the counter must stay on for an active opponent"


def test_the_counter_goes_quiet_against_an_opponent_who_never_plays():
    deck = [15, 6, 25, 40, 24, 72, 33, 7]
    env = CE(deck, deck, 3600)
    env.seed(3)
    t = _teacher(deck)
    for _ in range(T.COUNTER_PASSIVE_DECISIONS + 1):
        obs = np.asarray(env.get_observation_for_team(0), np.float32)
        t.act(env, obs)
        env.step_self_play(HAND, 0.0, 0.0, HAND, 0.0, 0.0, 10)
    assert t.counter_schedule(_attack()) == {}


@pytest.mark.slow
def test_the_top_rung_teacher_closes_a_match_against_a_passive_opponent():
    """The behavioural regression: the deck and seed that froze at 0 crowns."""
    from python_ai.opponents import deck_pool
    deck = next(d.card_ids for d in deck_pool.load_pool()
                if d.name == "classic_log_bait_inferno")
    env = CE(list(deck), list(deck), 3600)
    env.seed(3)
    t = _teacher(deck, seed=3)
    for _ in range(3600 // 10 + 5):
        obs = np.asarray(env.get_observation_for_team(0), np.float32)
        slot, x, y = t.act(env, obs)
        r = env.step_self_play(slot, x, y, HAND, 0.0, 0.0, 10)
        if env.is_game_over():
            break
    lost = sum(1 for s in (1, 2) if env.get_tower_hp(1, s) <= 0)
    assert lost >= 1, "the top-rung teacher failed to take a single tower from a passive opponent"
