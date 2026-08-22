"""REACTIVE ROLLOUTS: the rollout opponent answers instead of standing still.

WHY THIS FILE EXISTS
--------------------
`rollout_stats` rolled every candidate forward with BOTH SIDES NO-OPING, so the
scorer was structurally blind to the cost of being answered. Measured against a
ground-truth opponent (a full stage-5 UtilityTeacher playing the other side),
over 150 naked bridge pushes:

    tower damage dealt   no-op rollout predicts 587.5   truth 139.5   bias +448.0
    elixir lost          no-op rollout predicts   0.09  truth   2.84
    says PLAY            no-op rollout  90.7%           truth   7.3%

i.e. 125 false GO and 0 false HOLD -- the error was entirely one-directional and
entirely in favour of attacking.

WHAT THE FIX IS AND IS NOT. It does NOT make the teacher pick better plays:
`P(same | truth plays)` is pinned at 28.3% for every responder tried, at every
margin, including the blind one. What it does is stop the teacher COMMITTING to
attacks that a real opponent would punish -- false GO 27 -> 4 across all
decisions. That alone is worth the win rate.

WHY THE COUNTER IS OPEN-LOOP. Four responders were measured as paired
teacher-vs-teacher win rate, 150 seeded openings each, sides swapped, control
`noop vs noop == 0.5000` exactly:

    scripted (open-loop)  0.91x cost   +0.1583  [+0.1033, +0.2117]  p=1.9e-07
    reflex, stride 50     1.95x        +0.2050  [+0.1500, +0.2567]  p=3.3e-11
    reflex, stride 30     2.63x        +0.2450  [+0.1933, +0.2933]  p=1.4e-15
    reflex, stride 10     5.50x        +0.2217  [+0.1667, +0.2733]  p=4.4e-12

All four beat the control decisively. But paired ARM vs ARM on the same
openings, all six comparisons are NULL (p from 0.0857 to 0.832) -- the arms are
indistinguishable from each other, because ~60 of 150 openings tie. So the
decision collapsed to cost, and the cheapest arm won. The open-loop responder
reads NO observation inside the rollout and is measurably slightly FASTER than
the no-op control, because the counter shortens matches.

Note the non-monotonicity in that table, since it is the reason a "more is
better" reading was rejected: stride 10 costs 2.1x stride 30 and scores BELOW
it. Deciding every tick makes the rollout opponent an unrealistically good
defender -- it was observed dumping four cards onto a single Hog -- which
over-penalises attacking. Same shape as the neural search measuring horizon 20
worse than 12.
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
    """THE core behaviour. With reactive rollouts the opponent is no longer a
    statue: a candidate that walks a win condition to the bridge is answered
    inside the rollout, and the answer costs the opponent elixir.

    Asserted on the OPPONENT'S OWN SPEND rather than on damage, because damage
    is confounded by our own towers shooting -- the trap CLAUDE.md records for
    `get_troop_damage_dealt`."""
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
    """Only an ATTACKING placement is a threat the opponent must spend on.

    Charging a counter against a defensive placement would price defence as if
    it were an attack, which inverts the whole point: the measured value of
    this feature is suppressing naked ATTACKS, and an always-answering
    responder scored BELOW a less frequent one (+0.2217 vs +0.2450) precisely
    because over-answering over-penalises.
    """
    t = _teacher(team=0, reactive=True)
    own_half = T.Candidate.single(0, DECK[4], 9.0, 3.0, "melee")
    assert t.counter_schedule(own_half) == {}

    attacking = T.Candidate.single(0, HOG, 4.0, float(tactics.BRIDGE_ROW),
                                   "wincon")
    assert t.counter_schedule(attacking) != {}


def test_reactive_rollout_False_is_the_old_no_op_rollout_exactly():
    """The one-line off switch, in the shape `max_combos = 0` already has.

    A feature that changes what every phase-1 opponent does needs a way to be
    turned off without editing source -- both so an A/B can hold everything
    else byte-identical, and so a training run that misbehaves can be reverted
    without a rebuild. Asserted as the NEGATION of the core behaviour: with the
    switch off the rollout opponent spends nothing, because it never acts.
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
    """One class plays both sides, and the counter has to mirror with it.

    THE FRAME TRAP, which this project has paid for twice: `step_self_play`
    mirrors team 1's y itself, but `is_valid_placement` takes ABSOLUTE y for
    BOTH teams. Get it wrong and every counter is silently refused -- the
    symptom is not a crash but a rollout opponent that never answers, i.e. the
    feature quietly reverting to the no-op behaviour it exists to replace.

    Asserted on the opponent actually SPENDING, because that is the thing a
    rejected placement fails to do.
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
    """REGRESSION GUARD on the invariant the whole scorer rests on.

    The no-op baseline is what makes every other score MARGINAL. Reactive
    rollouts change what the baseline rollout contains -- the opponent now acts
    in candidate rollouts -- so it is worth re-pinning that the no-op itself
    still scores exactly 0.0 and the bot can still hold elixir.
    """
    env = _env()
    t = _teacher(team=0, reactive=True)
    obs = np.asarray(env.get_observation_for_team(0), np.float32)
    noop = [c for c in t.candidates(env, obs) if c.slot == CE.HAND_SIZE][0]
    base = t.rollout_stats(env, noop)
    assert t.score(env, noop, base, obs) == 0.0


def test_the_reacting_rollout_still_does_not_touch_the_live_match():
    """REGRESSION GUARD, and reactive rollouts raise the stakes on it.

    The rollout opponent now PLACES CARDS. If `snapshot()` ever stopped
    deep-copying the stats collectors, those hypothetical placements would post
    into the real match's statistics -- which feed the reward shaping, so the
    teacher would corrupt the returns its own student is scored against.
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
    """The counter is filtered through the opponent's REAL elixir.

    A rollout in which the opponent answers for free would be a different and
    worse bias than the one being fixed: it would make every attack look
    punished, which is the over-answering failure the stride sweep measured
    (+0.2217 at stride 10 against +0.2450 at stride 30).
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
    """Responder fidelity ramps with the ladder, like every other axis here.

    THE COLD-START ARGUMENT. In phase 1 the teacher models TEAM 0, which is the
    RL agent, and at episode 0 that agent cannot defend at all. A rollout that
    assumes a competent answer would price every attack as punished against an
    opponent who would not punish it -- the same shape as `play_margin = 3.0`
    freezing the bot against a PASSIVE opponent (14 plays across 6 matches,
    elixir pinned at 9.56), which is the zero-gradient environment the whole
    2026-08-19 curriculum pivot exists to remove.

    THE LADDER WAS FIRST WRITTEN AS [F, F, T, T, T, T] ON THAT ARGUMENT ALONE
    AND THAT WAS WRONG -- the cold-start reasoning is sound but it is not the
    binding constraint. See
    `test_the_counter_does_not_fire_when_the_rollout_is_too_short_to_see_the_payoff`:
    the counter is charged at +10 ticks while a Hog needs ~130 to arrive, so
    every rung below `COUNTER_MIN_HORIZON_TICKS` prices attacks below their
    true value and 3 of 20 openings froze at horizon 30. Only stage 5 clears
    the gate, and stage 5 is also the only rung the +0.1500 was measured at.

    So the honest ladder is "on where it was measured, off everywhere else",
    which happens to also satisfy the cold-start argument rather than resting
    on it.
    """
    reactive = [cfg["reactive"] for cfg in T.TEACHER_STAGES]
    assert reactive == [False, False, False, False, False, True]

    t = _teacher(team=0, reactive=True)
    t.set_stage(0)
    assert t.reactive_rollout is False
    t.set_stage(5)
    assert t.reactive_rollout is True


def test_a_naked_bridge_push_scores_LOWER_when_the_opponent_answers():
    """THE MECHANISM, pinned as a differential rather than an absolute.

    This is what the whole feature buys, and it is narrower than it sounds. It
    does NOT make the teacher pick better plays -- measured, `P(same | truth
    plays)` is 28.3% with the responder and 28.3% without. What it does is stop
    the teacher committing to a push a real opponent would punish. Over 150
    such pushes the no-op rollout said PLAY on 90.7% where the ground truth
    said 7.3%.

    A differential on ONE candidate in ONE state, because an absolute
    threshold here would be pinning a number that legitimately moves with the
    board, the deck and the profile.
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
    """THE ASYMMETRY THAT MAKES A SHORT REACTIVE ROLLOUT WORSE THAN A BLIND ONE.

    The counter's COST lands at +10 ticks. The attack's PAYOFF needs ~130 --
    a Hog has ~12 tiles to cross at Fast speed. So a rollout shorter than the
    crossing charges the answer in full and credits none of the push, which is
    a systematic anti-attack bias that gets worse the shorter the horizon.

    Measured against a PASSIVE opponent (which is what an episode-0 agent is),
    20 seeded openings, share of decisions that landed a card:

        horizon   OFF     ON      froze (<5 plays in 120 decisions)
           30    12.2%   9.8%     3/20
           50    11.6%  10.1%     3/20
           70    12.5%  11.3%     1/20
          100    11.8%  12.3%     0/20

    The +0.1500 win rate was measured at horizon 100, where the bias is gone.
    Turning it on at 30 would ship a regression into the exact regime the
    2026-08-19 curriculum pivot exists to prevent -- a teacher that freezes
    against a weak opponent, i.e. a zero-gradient environment.

    Same argument, and the same shape, as COMBO_MIN_HORIZON_TICKS: do not
    simulate half an interaction and score it as if it were whole.
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
    """`reactive` must not be on at a rung whose horizon the gate rejects.

    Otherwise the config claims a behaviour the code silently declines to
    perform -- the stage table would be documentation that is not true.
    """
    for i, cfg in enumerate(T.TEACHER_STAGES):
        if cfg["reactive"]:
            assert cfg["horizon_ticks"] >= T.COUNTER_MIN_HORIZON_TICKS, (
                f"stage {i} enables reacting at horizon "
                f"{cfg['horizon_ticks']}, below the gate")
