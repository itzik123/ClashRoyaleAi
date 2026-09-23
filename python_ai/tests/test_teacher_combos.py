"""Multi-card combo planning in the UtilityTeacher.

Deploy time made the escorted push correct and the naked push punished, and a
teacher proposing one card per decision could only play the naked push. A
0-tick `step_self_play` places nothing and `gym_wrapper.step` takes one (slot,
x, y) per decision, so a combo is a SEQUENCE across consecutive decisions.
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
HOG, MUSK, CANNON, ICE_GOLEM, SKELETONS, ICE_SPIRIT, LOG, FIREBALL = (
    15, 6, 25, 40, 24, 72, 33, 7)
PLANE = CE.BOARD_HEIGHT * CE.BOARD_WIDTH


# --- helpers ---
def _env(ticks=40, seed=0):
    """A staged mid-match board, seeded so assertions about which candidate wins
    are reproducible.
    """
    env = CE(DECK, DECK, 3600)
    env.seed(seed)
    for _ in range(ticks):
        env.step_self_play(-1, 0, 0, -1, 0, 0, 10)
    return env


def _teacher(team=0, horizon=30, k_cells=2, max_combos=4):
    t = T.UtilityTeacher(DECK, team=team, profile="balanced",
                         horizon_ticks=horizon, k_cells=k_cells, seed=0)
    t.reset()
    t.max_combos = max_combos
    return t


def _stage_hand(env, team, hand, elixir):
    """Deal an exact hand and bar."""
    env.set_hand_for_team(team, list(hand))
    env.set_elixir_for_team(team, float(elixir))


def _obs(env, team=0):
    return np.asarray(env.get_observation_for_team(team), np.float32)


def _occupied_cells(env, team=0):
    sp = _obs(env, team)[:3 * PLANE].reshape(3, CE.BOARD_HEIGHT, CE.BOARD_WIDTH)
    ys, xs = np.nonzero(sp.sum(axis=0))
    return sorted(set(zip(ys.tolist(), xs.tolist())))


# --- the action representation ---
def test_a_candidate_carries_a_placement_SEQUENCE_not_a_single_cell():
    """`.slot/.x/.y` now mean the first step of the sequence."""
    c = T.Candidate.single(1, HOG, 14.0, 15.0, "wincon")
    assert c.placements == [(1, 14.0, 15.0)]
    assert (c.slot, c.card_id, c.x, c.y) == (1, HOG, 14.0, 15.0)
    assert not c.is_combo

    combo = T.Candidate.combo(
        T.PlacementStep(0, ICE_GOLEM, 14.0, 15.0, 0),
        T.PlacementStep(1, HOG, 14.0, 14.0, T.COMBO_FOLLOWUP_DELAY_TICKS),
        kind="supported_push")
    assert combo.is_combo
    assert combo.placements == [(0, 14.0, 15.0), (1, 14.0, 14.0)]
    # The first step is what `act()` returns this decision.
    assert (combo.slot, combo.card_id, combo.x, combo.y) == (0, ICE_GOLEM, 14.0, 15.0)
    assert combo.cards == [ICE_GOLEM, HOG]
    assert combo.total_cost == pytest.approx(2.0 + 4.0)


def test_the_noop_is_still_an_empty_sequence_that_scores_zero():
    env = _env()
    t = _teacher()
    obs = _obs(env)
    assert T.NOOP.placements == []
    base = t.rollout_stats(env, T.NOOP)
    assert t.score(env, T.NOOP, base, obs) == 0.0


# --- the combo generator ---
def test_a_supported_push_is_proposed_when_tank_and_wincon_are_both_in_hand():
    """Tank first, win condition one decision later, same lane."""
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, ICE_SPIRIT], 10.0)
    t = _teacher()
    cands = t.candidates(env, _obs(env))
    pushes = [c for c in cands if c.kind == "supported_push"]
    assert pushes, "no supported push proposed with Ice Golem + Hog in hand"
    for c in pushes:
        assert c.cards == [ICE_GOLEM, HOG], (
            "the TANK must land first -- a win condition in front of its own "
            "escort is the naked push the engine now punishes")
        assert c.steps[1].delay_ticks == T.COMBO_FOLLOWUP_DELAY_TICKS
        # Same lane, the tank at or ahead of the win condition.
        assert c.steps[0].x == c.steps[1].x
        assert c.steps[0].y >= c.steps[1].y


def test_the_tank_is_derived_from_engine_HP_not_a_hardcoded_card_id():
    """The tank is read from engine HP, so a deck change cannot leave a stale
    literal.
    """
    hp = T.card_peak_hp(DECK)
    assert set(hp) <= set(DECK)
    assert hp[ICE_GOLEM] > hp[MUSK] > hp[ICE_SPIRIT] > hp[SKELETONS]
    assert T.tank_id(DECK) == ICE_GOLEM, (
        "the tank is the highest-HP troop that is NOT the win condition")
    assert FIREBALL not in hp and CANNON not in hp, "spells/buildings are not tanks"


def test_a_combo_is_never_proposed_when_the_PAIR_is_unaffordable():
    """Ice Golem (2) + Hog (4) needs 6, minus the 0.35 that regenerates between
    the two decisions.
    """
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, ICE_SPIRIT], 3.0)
    t = _teacher()
    combos = [c for c in t.candidates(env, _obs(env)) if c.is_combo]
    assert not combos, f"proposed {combos} on 3 elixir"

    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, ICE_SPIRIT], 6.0)
    combos = [c for c in t.candidates(env, _obs(env)) if c.is_combo]
    assert combos, "6 elixir covers Ice Golem + Hog and none was proposed"


def test_every_step_of_every_combo_is_a_legal_placement_for_either_team():
    """The team-1 frame trap extended to the second card: `is_valid_placement`
    takes absolute y, `step_self_play` mirrors team 1's.
    """
    for team in (0, 1):
        env = _env()
        _stage_hand(env, team, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
        t = _teacher(team=team)
        cands = t.candidates(env, _obs(env, team))
        assert any(c.is_combo for c in cands), f"team {team} proposed no combo"
        for c in cands:
            for st in c.steps:
                assert env.is_valid_placement(
                    st.card_id, st.x, t.to_absolute_y(st.y), team), (
                    f"team {team} proposed an illegal step {st} of {c}")


def test_combo_width_is_capped_so_the_latency_budget_survives():
    """Width is what search is expensive in: each extra candidate is a whole
    rollout. Curated combos, not the cross product.
    """
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    for cap in (0, 1, 2, 4):
        t = _teacher(max_combos=cap)
        combos = [c for c in t.candidates(env, _obs(env)) if c.is_combo]
        assert len(combos) <= cap


def test_combos_are_off_at_horizons_too_short_to_SEE_the_second_card():
    """A rollout that stops before the follow-up lands would charge both costs and
    simulate one card.
    """
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    for horizon in (0, 10):
        t = _teacher(horizon=horizon)
        assert not [c for c in t.candidates(env, _obs(env)) if c.is_combo]
    assert T.COMBO_MIN_HORIZON_TICKS > T.COMBO_FOLLOWUP_DELAY_TICKS


def test_a_defensive_stack_is_proposed_against_a_real_push():
    """Cannon (centre pull) plus a body to kill what it holds."""
    env = _env()
    _stage_hand(env, 0, [CANNON, SKELETONS, MUSK, ICE_SPIRIT], 10.0)
    for _ in range(6):
        env.inject_enemy(MUSK, 9.0, 12.0)
        env.step_self_play(-1, 0, 0, -1, 0, 0, 5)
    obs = _obs(env)
    assert tactics.threat_level(obs) > 0.0, "no threat staged"
    t = _teacher()
    stacks = [c for c in t.candidates(env, obs) if c.kind == "defensive_stack"]
    assert stacks, "no defensive stack proposed against an active push"
    for c in stacks:
        assert t.roles[c.cards[0]] == "building", "the building must land first"


# --- forward simulation ---
def test_a_zero_tick_step_places_nothing_which_is_why_a_combo_is_a_SEQUENCE():
    """The design rests on this: if a 0-tick placement ever works, a truly
    simultaneous combo becomes expressible.
    """
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    s = env.snapshot()
    s.step_self_play(0, 9.0, 10.0, -1, 0, 0, 0)
    assert s.get_elixir_spent(0) == 0.0
    s.step_self_play(0, 9.0, 10.0, -1, 0, 0, 1)
    assert s.get_elixir_spent(0) == pytest.approx(2.0)


def test_the_rollout_actually_plays_BOTH_cards_of_a_combo():
    """A rollout that plays only the first card scores the pair's cost against one
    card's value, and the teacher silently never escorts.
    """
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    t = _teacher(horizon=50)
    combo = next(c for c in t.candidates(env, _obs(env))
                 if c.kind == "supported_push")
    spent = t.rollout_stats(env, combo)["elixir_spent"]
    assert spent == pytest.approx(combo.total_cost), (
        f"rollout spent {spent}, combo costs {combo.total_cost}")


def test_the_second_card_lands_one_decision_after_the_first():
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    t = _teacher(horizon=50)
    combo = next(c for c in t.candidates(env, _obs(env))
                 if c.kind == "supported_push")
    # Two independent snapshots: `execute_steps` runs forward from wherever the
    # snapshot is, so reusing one would replay the first card.
    short = env.snapshot()
    t.execute_steps(short, combo, ticks=T.COMBO_FOLLOWUP_DELAY_TICKS)
    assert short.get_elixir_spent(0) == pytest.approx(
        E.get_card_info(combo.cards[0])["cost"]), "the follow-up landed early"
    long = env.snapshot()
    t.execute_steps(long, combo, ticks=T.COMBO_FOLLOWUP_DELAY_TICKS * 2)
    assert long.get_elixir_spent(0) == pytest.approx(combo.total_cost)


def test_both_cards_of_a_combo_get_their_OWN_deploy_time():
    """DEPLOY_TIME_TICKS is per entity (`CardFactories::applyCardMetadata`);
    shared, the second card would deploy instantly.
    """
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    env.step_self_play(0, 14.0, 15.0, -1, 0, 0, 1)     # tank at the bridge
    env.step_self_play(-1, 0, 0, -1, 0, 0, 9)
    env.step_self_play(1, 14.0, 14.0, -1, 0, 0, 1)     # win condition behind it
    at_landing = dict.fromkeys(_occupied_cells(env))
    assert (14, 14) in at_landing and (15, 14) in at_landing

    env.step_self_play(-1, 0, 0, -1, 0, 0, 9)          # still inside its deploy
    assert (14, 14) in dict.fromkeys(_occupied_cells(env)), (
        "the second card moved during its own deploy time")

    env.step_self_play(-1, 0, 0, -1, 0, 0, 30)
    rows = [y for (y, _x) in _occupied_cells(env)]
    assert max(rows) > 15, "nothing ever left the bridge; deploy never expired"


def test_a_combo_rollout_does_not_touch_the_live_match():
    """snapshot() deep-copies the stats collectors; otherwise a hypothetical hit
    lands in the real match's statistics, which feed the reward.
    """
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    before = (env.get_tower_damage_dealt(0), env.get_tower_damage_dealt(1),
              env.get_elixir_spent(0), env.get_elixir_for_team(0),
              list(env.get_hand_for_team(0)))
    t = _teacher(horizon=50)
    t.act(env, _obs(env))
    after = (env.get_tower_damage_dealt(0), env.get_tower_damage_dealt(1),
             env.get_elixir_spent(0), env.get_elixir_for_team(0),
             list(env.get_hand_for_team(0)))
    assert before == after


# --- scoring ---
def test_a_combo_is_charged_for_BOTH_cards():
    """`W_COST` is the opportunity cost the no-op baseline does not absorb, so a
    combo pays for both cards.
    """
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    t = _teacher(horizon=50)
    obs = _obs(env)
    combo = next(c for c in t.candidates(env, obs) if c.kind == "supported_push")
    assert t.sequence_cost(combo) == pytest.approx(6.0)


def test_the_elixir_charge_is_what_separates_a_combo_from_its_first_card():
    """Same first card, same board: the pair must cost strictly more, or `w_cost`
    is not reaching the second step.
    """
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    t = _teacher(horizon=50)
    obs = _obs(env)
    cands = t.candidates(env, obs)
    combo = next(c for c in cands if c.kind == "supported_push")
    solo = T.Candidate.single(combo.steps[0].slot, combo.steps[0].card_id,
                              combo.steps[0].x, combo.steps[0].y, "melee")
    assert t.sequence_cost(combo) > t.sequence_cost(solo)


# --- the plan: commit the first card, keep the second reachable ---
def test_choosing_a_combo_leaves_the_follow_up_reachable_next_decision():
    """`act()` returns one placement per decision, so the second half of the plan
    survives to the next one, where it is re-scored with the tank already on
    the board.
    """
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    t = _teacher(horizon=50)
    combo = next(c for c in t.candidates(env, _obs(env))
                 if c.kind == "supported_push")
    t.commit(combo)
    assert t.pending is not None and t.pending.card_id == HOG

    t.tick_plan()          # one decision passes, exactly as `act` does it
    follow = [c for c in t.candidates(env, _obs(env)) if c.kind == "followup"]
    assert follow, "the planned follow-up is not reachable next decision"
    assert follow[0].card_id == HOG
    assert (follow[0].x, follow[0].y) == (combo.steps[1].x, combo.steps[1].y)


def test_a_pending_follow_up_is_dropped_when_the_card_left_that_slot():
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    t = _teacher(horizon=50)
    combo = next(c for c in t.candidates(env, _obs(env))
                 if c.kind == "supported_push")
    t.commit(combo)
    _stage_hand(env, 0, [ICE_GOLEM, MUSK, CANNON, SKELETONS], 10.0)
    assert not [c for c in t.candidates(env, _obs(env)) if c.kind == "followup"]


def test_a_pending_follow_up_is_dropped_when_it_became_unaffordable():
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    t = _teacher(horizon=50)
    combo = next(c for c in t.candidates(env, _obs(env))
                 if c.kind == "supported_push")
    t.commit(combo)
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 1.0)
    assert not [c for c in t.candidates(env, _obs(env)) if c.kind == "followup"]


def test_a_plan_is_single_use_and_cannot_survive_two_decisions():
    """A plan is consumed on the decision it was planned for; a stale one was
    scored on a board that is gone.
    """
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    t = _teacher(horizon=50)
    combo = next(c for c in t.candidates(env, _obs(env))
                 if c.kind == "supported_push")
    t.commit(combo)
    planned = t.pending
    t.act(env, _obs(env))
    assert t.pending is not planned, (
        "the plan from the previous decision was carried forward unconsumed")


def test_reset_clears_a_pending_plan():
    """A plan must not survive into the next match."""
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    t = _teacher(horizon=50)
    combo = next(c for c in t.candidates(env, _obs(env))
                 if c.kind == "supported_push")
    t.commit(combo)
    assert t.pending is not None
    t.reset()
    assert t.pending is None


def test_act_still_returns_one_placement_in_the_teachers_own_frame():
    """`gym_wrapper.step` unpacks exactly three values; the action space grew
    inside the teacher only.
    """
    env = _env()
    t = _teacher(horizon=50)
    out = t.act(env, _obs(env))
    assert isinstance(out, tuple) and len(out) == 3
    slot, x, y = out
    assert 0 <= slot <= CE.HAND_SIZE


def test_the_teacher_actually_executes_a_planned_pair_end_to_end():
    """End to end, driven the way `gym_wrapper` drives it: `act` ->
    `step_self_play` twice, both cards reaching the board.

    Staged, because real matches choose a combo too rarely for a stable
    assertion; `eval/prove_combos.py` measures the rate.
    """
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    t = _teacher(horizon=50)
    spent0 = env.get_elixir_spent(0)

    slot, x, y = t.act(env, _obs(env))
    assert t.pending is not None, "no combo was chosen from a staged combo state"
    planned = t.pending.card_id
    env.step_self_play(slot, x, y, -1, 0, 0, 10)
    first = env.get_elixir_spent(0) - spent0
    assert first > 0.0, "the first card of the plan never landed"

    slot2, x2, y2 = t.act(env, _obs(env))
    assert slot2 < CE.HAND_SIZE
    assert list(env.get_hand_for_team(0))[slot2] == planned, (
        "the planned follow-up was not the action taken on the next decision")
    env.step_self_play(slot2, x2, y2, -1, 0, 0, 10)
    assert env.get_elixir_spent(0) - spent0 == pytest.approx(
        first + E.get_card_info(planned)["cost"])


# --- reachability ---
# A 5-6 elixir pair is rarely affordable, because the teacher spends
# continuously. A flat savings charge did not move the bar and was removed.
# What lowers the price is the follow-up delay: the pair's cost is paid across
# the gap, and a longer gap also lets the tank get further ahead.


def test_the_follow_up_delay_is_SEARCHED_not_fixed():
    """The follow-up delay is a searched candidate axis, like placement."""
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    t = _teacher(horizon=100, max_combos=12)
    delays = {c.steps[1].delay_ticks for c in t.candidates(env, _obs(env))
              if c.is_combo}
    assert len(delays) > 1, f"only one gap ever offered: {delays}"
    assert delays <= set(T.COMBO_FOLLOWUP_DELAYS)


def test_a_gap_is_only_offered_when_the_rollout_can_SEE_past_it():
    """A combo scored on a rollout that ends before its second card charges two
    cards and simulates one.
    """
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    for horizon in (30, 50, 100):
        t = _teacher(horizon=horizon, max_combos=12)
        for c in t.candidates(env, _obs(env)):
            if c.is_combo:
                assert c.max_delay + T.COMBO_FOLLOWUP_DELAY_TICKS <= horizon


def test_a_longer_gap_makes_a_pair_affordable_that_a_short_one_does_not():
    """Skeletons (1) + Hog (4) needs 4.65 at a one-second gap and 3.95 at three
    seconds.
    """
    env = _env()
    _stage_hand(env, 0, [SKELETONS, HOG, CANNON, MUSK], 4.1)
    t = _teacher(horizon=100, max_combos=12)
    combos = [c for c in t.candidates(env, _obs(env)) if c.is_combo]
    assert combos, "no combo affordable at 4.1 even with a long gap"
    assert all(c.steps[1].delay_ticks > T.COMBO_FOLLOWUP_DELAY_TICKS
               for c in combos), (
        "a one-second gap should still be unaffordable here")


def test_a_delayed_plan_is_held_until_its_own_decision_comes_round():
    """A plan with a three-second gap fires when it is due and never before, or
    the gap scored is not the gap played.
    """
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    t = _teacher(horizon=100, max_combos=12)
    combo = next(c for c in t.candidates(env, _obs(env))
                 if c.is_combo and c.steps[1].delay_ticks == 30)
    t.commit(combo)
    for _ in range(2):     # 30 ticks is THREE decisions, not two
        assert not [c for c in t.candidates(env, _obs(env))
                    if c.kind == "followup"]
        t.tick_plan()
    assert not [c for c in t.candidates(env, _obs(env)) if c.kind == "followup"]
    t.tick_plan()
    follow = [c for c in t.candidates(env, _obs(env)) if c.kind == "followup"]
    assert follow and follow[0].card_id == combo.cards[1]


def test_a_plan_reserves_the_elixir_its_own_follow_up_needs():
    """The plan reserve: having committed to a plan, do not spend the elixir its
    follow-up needs. It is live only for the few decisions a plan exists.
    """
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    t = _teacher(horizon=100, max_combos=12)
    combo = next(c for c in t.candidates(env, _obs(env))
                 if c.is_combo and c.steps[1].delay_ticks == 30)
    cheap = T.Candidate.single(3, SKELETONS, 9.0, 10.0, "melee")
    obs, hand = _obs(env), list(env.get_hand_for_team(0))
    assert t.plan_reserve_penalty(cheap, obs, hand, 3.5) == 0.0, (
        "nothing is committed yet")
    t.commit(combo)
    assert t.plan_reserve_penalty(cheap, obs, hand, 3.5) > 0.0


def test_the_plan_reserve_never_blocks_the_plans_own_follow_up():
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    t = _teacher(horizon=100, max_combos=12)
    combo = next(c for c in t.candidates(env, _obs(env)) if c.is_combo)
    t.commit(combo)
    follow = T.Candidate.single(t.pending.slot, t.pending.card_id,
                                t.pending.x, t.pending.y, "wincon",
                                kind="followup")
    assert t.plan_reserve_penalty(
        follow, _obs(env), list(env.get_hand_for_team(0)), 3.5) == 0.0


def test_the_plan_reserve_never_blocks_DEFENCE():
    """The reserve never blocks defence; holding elixir through an incoming push
    loses the tower. The threshold reuses `tactics.HOG_MAX_THREAT`.
    """
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    t = _teacher(horizon=100, max_combos=12)
    combo = next(c for c in t.candidates(env, _obs(env)) if c.is_combo)
    t.commit(combo)
    for _ in range(8):
        env.inject_enemy(MUSK, 9.0, 12.0)
        env.step_self_play(-1, 0, 0, -1, 0, 0, 5)
    obs = _obs(env)
    assert tactics.threat_level(obs) > tactics.HOG_MAX_THREAT
    cheap = T.Candidate.single(3, SKELETONS, 9.0, 10.0, "melee")
    assert t.plan_reserve_penalty(
        cheap, obs, list(env.get_hand_for_team(0)), 3.5) == 0.0


def test_the_plan_reserve_is_silent_when_the_spend_leaves_enough_anyway():
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    t = _teacher(horizon=100, max_combos=12)
    combo = next(c for c in t.candidates(env, _obs(env)) if c.is_combo)
    t.commit(combo)
    cheap = T.Candidate.single(3, SKELETONS, 9.0, 10.0, "melee")
    assert t.plan_reserve_penalty(
        cheap, _obs(env), list(env.get_hand_for_team(0)), 9.0) == 0.0


def test_the_teacher_records_WHICH_kind_of_play_it_just_chose():
    """The teacher records the kind of play it chose: a combo's first step is
    indistinguishable from the same card played alone, so
    `eval/prove_combos.py` cannot re-derive it.
    """
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    t = _teacher(horizon=50)
    t.act(env, _obs(env))
    assert t.last_kind in {"single", "followup", "noop"} | {
        c.kind for c in t.candidates(env, _obs(env))}
    # A staged combo state must report the combo family, not "single".
    assert t.pending is None or t.last_kind in {
        "supported_push", "counter_push", "defensive_stack",
        "spell_then_push", "push_then_spell"}


def test_last_kind_is_noop_when_the_teacher_holds():
    env = _env(ticks=0)
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 0.0)
    t = _teacher(horizon=50)
    slot, _x, _y = t.act(env, _obs(env))
    assert slot == CE.HAND_SIZE
    assert t.last_kind == "noop"


def test_a_cheap_two_body_defence_is_proposed_against_a_push():
    """A two-elixir defensive pair (Ice Spirit + Skeletons) for the states the
    expensive families cannot afford, needing no building in hand.
    """
    env = _env()
    _stage_hand(env, 0, [SKELETONS, ICE_SPIRIT, HOG, MUSK], 3.0)
    for _ in range(6):
        env.inject_enemy(MUSK, 9.0, 12.0)
        env.step_self_play(-1, 0, 0, -1, 0, 0, 5)
    t = _teacher(horizon=100, max_combos=12)
    cheap = [c for c in t.candidates(env, _obs(env)) if c.kind == "cheap_defence"]
    assert cheap, "no cheap two-body defence proposed against an active push"
    for c in cheap:
        assert c.total_cost <= 3.0, c
        assert t.wincon_id not in c.cards, (
            "the win condition is not a defensive body")


def test_the_cheap_defence_needs_no_building_and_no_threat_means_no_pair():
    env = _env()
    _stage_hand(env, 0, [SKELETONS, ICE_SPIRIT, HOG, MUSK], 10.0)
    t = _teacher(horizon=100, max_combos=12)
    assert tactics.threat_level(_obs(env)) <= 0.0
    assert not [c for c in t.candidates(env, _obs(env))
                if c.kind == "cheap_defence"], (
        "a two-body defence with nothing to defend against is elixir thrown away")


def test_a_single_combo_family_can_be_disabled_for_an_ABLATION():
    """`combo_families` allows a per-family ablation that keeps the arms
    byte-identical everywhere else.
    """
    env = _env()
    _stage_hand(env, 0, [SKELETONS, ICE_SPIRIT, HOG, MUSK], 4.0)
    for _ in range(6):
        env.inject_enemy(MUSK, 9.0, 12.0)
        env.step_self_play(-1, 0, 0, -1, 0, 0, 5)

    full = _teacher(horizon=100, max_combos=12)
    assert any(c.kind == "cheap_defence" for c in full.candidates(env, _obs(env)))

    ablated = _teacher(horizon=100, max_combos=12)
    ablated.combo_families = tuple(k for k in T.COMBO_FAMILIES
                                   if k != "cheap_defence")
    kinds = {c.kind for c in ablated.candidates(env, _obs(env))}
    assert "cheap_defence" not in kinds
    assert kinds & set(T.COMBO_FAMILIES), "the other families must survive"


def test_the_family_list_matches_what_the_generator_can_actually_emit():
    """A name no generator emits would make an ablation a silent no-op."""
    assert set(T.COMBO_FAMILIES) == {
        "supported_push", "counter_push", "defensive_stack", "cheap_defence",
        "spell_then_push", "push_then_spell"}
    t = _teacher()
    for name in T.COMBO_FAMILIES:
        assert hasattr(t, f"_combo_{name}"), name


# --- weight injection, for profile sweeps ---
def test_a_profile_can_be_given_as_an_explicit_WEIGHT_SET():
    """A profile can be an explicit weight mapping; the named form is unchanged.
    """
    weights = dict(T.PROFILES["balanced"], w_pos=7.5)
    t = T.UtilityTeacher(DECK, team=0, profile=weights, horizon_ticks=50)
    t.reset()
    assert t.profile["w_pos"] == 7.5
    assert t.profile["w_cost"] == T.PROFILES["balanced"]["w_cost"]


def test_an_injected_weight_set_SURVIVES_reset():
    """`reset()` re-draws the profile unless one was pinned; an injected set that
    reverted would make every sweep arm measure the same thing.
    """
    weights = dict(T.PROFILES["balanced"], w_pos=3.0)
    t = T.UtilityTeacher(DECK, team=1, profile=weights, horizon_ticks=50)
    for _ in range(3):
        t.reset()
        assert t.profile["w_pos"] == 3.0


def test_an_injected_weight_set_is_COPIED_not_aliased():
    """Two teachers built from one dict must not share it."""
    weights = dict(T.PROFILES["balanced"], w_pos=9.0)
    a = T.UtilityTeacher(DECK, team=0, profile=weights)
    a.reset()
    a.profile["w_pos"] = 1.0
    assert weights["w_pos"] == 9.0
    b = T.UtilityTeacher(DECK, team=1, profile=weights)
    b.reset()
    assert b.profile["w_pos"] == 9.0


def test_the_named_profiles_still_work_exactly_as_before():
    for name in T.PROFILES:
        t = T.UtilityTeacher(DECK, team=0, profile=name)
        t.reset()
        assert t.profile == T.PROFILES[name]


# --- play_margin: the anti-dumping guard ---
def test_play_margin_is_the_SWEPT_value_not_the_original_placeholder():
    """The marginal plays the margin guards against score around 1.4, so the old
    0.05 was an off switch. 3.0 was chosen by sweep and confirmed; 4.0 scores
    the same while wasting more income.
    """
    t = T.UtilityTeacher(DECK, team=0)
    assert t.play_margin == 3.0


def test_a_play_worth_less_than_the_margin_is_declined():
    """A candidate below the bar loses to holding, which makes the bar an economy
    control rather than a tiebreak.
    """
    env = _env()
    # Below the taper start, or this would test the taper instead of the gate.
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 6.0)
    t = _teacher(horizon=30)
    t.play_margin = 1e9          # nothing can clear this
    slot, _x, _y = t.act(env, _obs(env))
    assert slot == CE.HAND_SIZE, "an unreachable margin still let a card through"


def test_the_margin_does_not_reach_the_rules_only_rungs():
    """The rules-only rungs never consult `play_margin`, so the easiest rung stays
    beatable.
    """
    env = _env()
    _stage_hand(env, 0, [CANNON, SKELETONS, MUSK, ICE_SPIRIT], 6.0)
    # `_rules_only` answers a threat, so one must exist or this passes
    # vacuously.
    for _ in range(6):
        env.inject_enemy(MUSK, 9.0, 12.0)
        env.step_self_play(-1, 0, 0, -1, 0, 0, 5)
    assert tactics.threat_level(_obs(env)) > 0.0
    t = _teacher(horizon=0)
    t.play_margin = 1e9
    assert t.act(env, _obs(env))[0] < CE.HAND_SIZE, (
        "stage 0 declined to answer a push -- _rules_only is reading "
        "play_margin, which it must not")


def test_the_margin_TAPERS_as_the_bar_approaches_overflow():
    """A fixed margin freezes the teacher against a passive opponent, and an
    episode-0 agent is passive. The bar tapers from MARGIN_TAPER_START, below
    the overflow line, so it is relieved across the range where the freeze
    happened; below that it stays full.
    """
    t = T.UtilityTeacher(DECK, team=0)
    assert t.effective_play_margin(5.0) == pytest.approx(t.play_margin)
    # At ELIXIR_OVERFLOW_AT the bar is already partly relieved; an inequality,
    # so retuning MARGIN_TAPER_START needs no edit here.
    assert 0.0 < t.effective_play_margin(T.ELIXIR_OVERFLOW_AT) < t.play_margin
    assert t.effective_play_margin(T.MARGIN_TAPER_START) == pytest.approx(
        t.play_margin)
    mid = t.effective_play_margin(9.5)
    assert 0.0 < mid < t.play_margin
    assert t.effective_play_margin(10.0) == pytest.approx(0.0)


def test_the_taper_is_monotone_so_holding_never_gets_cheaper_as_elixir_rises():
    t = T.UtilityTeacher(DECK, team=0)
    xs = [0.0, 3.0, 6.0, 9.0, 9.25, 9.5, 9.75, 10.0]
    ms = [t.effective_play_margin(x) for x in xs]
    assert ms == sorted(ms, reverse=True), ms


def test_a_teacher_at_max_elixir_will_actually_play_something():
    """Full bar, nothing threatening: the teacher must play."""
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    t = _teacher(horizon=50)
    assert t.act(env, _obs(env))[0] < CE.HAND_SIZE, (
        "the teacher held a full bar with a free board -- it is wasting income")


def test_a_DUE_followup_is_not_charged_the_dumping_margin():
    """A due follow-up skips the dumping margin: the first card is already paid
    for, so holding wastes the commitment. It must still be the argmax over
    every other candidate.
    """
    t = T.UtilityTeacher(DECK, team=0)
    follow = T.Candidate.single(1, HOG, 14.0, 15.0, "wincon", kind="followup")
    single = T.Candidate.single(1, HOG, 14.0, 15.0, "wincon")
    assert t.margin_for(follow, 5.0) == 0.0
    assert t.margin_for(single, 5.0) == pytest.approx(t.play_margin)


def test_the_followup_exemption_does_not_leak_to_ordinary_plays():
    """The exemption must not switch the margin off for everything once a plan
    exists.
    """
    t = T.UtilityTeacher(DECK, team=0)
    for kind in ("single", "supported_push", "counter_push", "cheap_defence"):
        c = T.Candidate.single(0, SKELETONS, 9.0, 10.0, "melee", kind=kind)
        assert t.margin_for(c, 5.0) == pytest.approx(t.play_margin), kind
