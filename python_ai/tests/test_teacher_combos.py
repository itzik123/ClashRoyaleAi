"""Multi-card COMBO planning in the UtilityTeacher (TODO.md item 1).

WHY THIS FILE EXISTS
--------------------
The 2026-08-19 deploy-time change (`DEPLOY_TIME_TICKS = 10`) made the ESCORTED
push the correct play and the NAKED push the punished one, measured on the same
engine and the same harness:

    lone win condition                       -556.3 tower HP
    supported push (tank one decision ahead) +448.5 tower HP  [+137.3, +760.1]
    escorting, inside a punish window        +650   tower HP  [+429, +878]

`UtilityTeacher` could not make that play. `_cells_for` proposed cells for ONE
card per decision and `score` ranked single candidates, so its entire attack
repertoire was "send the win condition to a bridge, alone" -- exactly the play
the new physics correctly punishes. That, and not the engine, is why
`prove_environment.py`'s strategy arm still read attack 0.490 vs cycle 0.715.

THE ONE ENGINE FACT THAT DECIDES THE DESIGN, measured 2026-08-20 and pinned
below: `step_self_play(slot, x, y, ..., 0)` places NOTHING. Placement is
processed inside the tick loop, so a 0-tick call spends no elixir and puts no
unit on the board. There is therefore no such thing as a truly simultaneous
two-card placement, and -- more importantly -- `gym_wrapper.step` hands the
teacher exactly ONE `(slot, x, y)` per decision. So a combo is a SEQUENCE across
consecutive decisions, which is also the shape the +448.5 measurement was taken
at ("tank one decision ahead").
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


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _env(ticks=40, seed=0):
    """A staged mid-match board, SEEDED so the fixture is reproducible.

    It used to call a bare `reset()`, which left team 1's hand and the whole
    40-tick warm-up drawn from `std::random_device` -- so every invocation
    staged a different position and any assertion about which candidate WINS
    was a coin flip. That was invisible while the scorer was lenient and became
    a ~1-in-3 flake the moment reactive rollouts made scores tighter.

    `ClashRoyaleEnv.seed()` seeds both engine generators and re-deals (landed
    2026-08-21), so this is now available; CLAUDE.md's claim that the opening
    shuffle cannot be seeded is stale. Seed 0 is not cherry-picked: the staged
    state below chooses a combo in 33 of the first 40 seeds (82%).
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
    """Deal an exact hand and bar. Both setters exist in the shipped .pyd --
    CLAUDE.md records the 2026-08-19 stale-.pyd incident that hid them."""
    env.set_hand_for_team(team, list(hand))
    env.set_elixir_for_team(team, float(elixir))


def _obs(env, team=0):
    return np.asarray(env.get_observation_for_team(team), np.float32)


def _occupied_cells(env, team=0):
    sp = _obs(env, team)[:3 * PLANE].reshape(3, CE.BOARD_HEIGHT, CE.BOARD_WIDTH)
    ys, xs = np.nonzero(sp.sum(axis=0))
    return sorted(set(zip(ys.tolist(), xs.tolist())))


# ==========================================================================
# the action representation
# ==========================================================================
def test_a_candidate_carries_a_placement_SEQUENCE_not_a_single_cell():
    """`Candidate` used to be one (slot, x, y). Everything downstream reads
    `.slot/.x/.y`, so those stay -- they now mean THE FIRST STEP."""
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


# ==========================================================================
# the combo generator
# ==========================================================================
def test_a_supported_push_is_proposed_when_tank_and_wincon_are_both_in_hand():
    """THE combo the deploy-time change made correct: tank first, win condition
    one decision later, same lane."""
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
        # Same lane, and the tank at or ahead of the win condition.
        assert c.steps[0].x == c.steps[1].x
        assert c.steps[0].y >= c.steps[1].y


def test_the_tank_is_derived_from_engine_HP_not_a_hardcoded_card_id():
    """`DEFAULT_DECK` has changed twice. A literal 40 would silently mean
    'Ice Golem' forever -- the same failure `card_roles` exists to avoid."""
    hp = T.card_peak_hp(DECK)
    assert set(hp) <= set(DECK)
    assert hp[ICE_GOLEM] > hp[MUSK] > hp[ICE_SPIRIT] > hp[SKELETONS]
    assert T.tank_id(DECK) == ICE_GOLEM, (
        "the tank is the highest-HP troop that is NOT the win condition")
    assert FIREBALL not in hp and CANNON not in hp, "spells/buildings are not tanks"


def test_a_combo_is_never_proposed_when_the_PAIR_is_unaffordable():
    """Ice Golem (2) + Hog (4) needs 6, minus the 0.35 that regenerates during
    the one decision between them."""
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, ICE_SPIRIT], 3.0)
    t = _teacher()
    combos = [c for c in t.candidates(env, _obs(env)) if c.is_combo]
    assert not combos, f"proposed {combos} on 3 elixir"

    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, ICE_SPIRIT], 6.0)
    combos = [c for c in t.candidates(env, _obs(env)) if c.is_combo]
    assert combos, "6 elixir covers Ice Golem + Hog and none was proposed"


def test_every_step_of_every_combo_is_a_legal_placement_for_either_team():
    """The team-1 frame trap, extended to the second card. `is_valid_placement`
    takes ABSOLUTE y for both teams while `step_self_play` mirrors team 1's --
    getting that wrong reads as 'the teacher is weak', not as an exception."""
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
    """Width is what search is expensive in: one engine step is 0.015 ms but
    each extra candidate is a whole rollout. Curated combos, not the cross
    product."""
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    for cap in (0, 1, 2, 4):
        t = _teacher(max_combos=cap)
        combos = [c for c in t.candidates(env, _obs(env)) if c.is_combo]
        assert len(combos) <= cap


def test_combos_are_off_at_horizons_too_short_to_SEE_the_second_card():
    """The follow-up lands 10 ticks in. A rollout that stops at 10 ticks scores
    the pair without ever simulating half of it, which is worse than not
    proposing it -- it would charge both costs and credit only one card."""
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    for horizon in (0, 10):
        t = _teacher(horizon=horizon)
        assert not [c for c in t.candidates(env, _obs(env)) if c.is_combo]
    assert T.COMBO_MIN_HORIZON_TICKS > T.COMBO_FOLLOWUP_DELAY_TICKS


def test_a_defensive_stack_is_proposed_against_a_real_push():
    """Cannon (centre pull) plus a body to kill what it holds -- the other
    combo family, and the one that fires far more often than the push."""
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


# ==========================================================================
# forward simulation
# ==========================================================================
def test_a_zero_tick_step_places_nothing_which_is_why_a_combo_is_a_SEQUENCE():
    """Pinned because the whole design rests on it. If the engine ever starts
    honouring a 0-tick placement, a genuinely simultaneous combo becomes
    expressible and this test is the thing that should be re-read first."""
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    s = env.snapshot()
    s.step_self_play(0, 9.0, 10.0, -1, 0, 0, 0)
    assert s.get_elixir_spent(0) == 0.0
    s.step_self_play(0, 9.0, 10.0, -1, 0, 0, 1)
    assert s.get_elixir_spent(0) == pytest.approx(2.0)


def test_the_rollout_actually_plays_BOTH_cards_of_a_combo():
    """The failure this catches is silent and expensive: a rollout that plays
    only the first card scores the pair's COST against one card's VALUE, so
    every combo looks bad and the teacher quietly never learns to escort."""
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
    # Two independent snapshots: `execute_steps` runs a plan forward from
    # wherever the snapshot is, so re-running it on the same object would
    # replay the first card.
    short = env.snapshot()
    t.execute_steps(short, combo, ticks=T.COMBO_FOLLOWUP_DELAY_TICKS)
    assert short.get_elixir_spent(0) == pytest.approx(
        E.get_card_info(combo.cards[0])["cost"]), "the follow-up landed early"
    long = env.snapshot()
    t.execute_steps(long, combo, ticks=T.COMBO_FOLLOWUP_DELAY_TICKS * 2)
    assert long.get_elixir_spent(0) == pytest.approx(combo.total_cost)


def test_both_cards_of_a_combo_get_their_OWN_deploy_time():
    """`DEPLOY_TIME_TICKS = 10` is per-entity, assigned in
    `CardFactories::applyCardMetadata`. If it were global-per-tick or shared,
    the second card would deploy instantly and the escort would be scored
    against physics that do not exist."""
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
    """snapshot() deep-copies the stats collectors. Two placements is two
    chances for a hypothetical hit to land in the REAL match's statistics --
    and those feed the reward shaping."""
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


# ==========================================================================
# scoring
# ==========================================================================
def test_a_combo_is_charged_for_BOTH_cards():
    """`W_COST` is the opportunity cost the no-op baseline does NOT absorb. A
    combo that pays for one card is a bot that thinks escorting is free."""
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    t = _teacher(horizon=50)
    obs = _obs(env)
    combo = next(c for c in t.candidates(env, obs) if c.kind == "supported_push")
    assert t.sequence_cost(combo) == pytest.approx(6.0)


def test_the_elixir_charge_is_what_separates_a_combo_from_its_first_card():
    """Same first card, same board, same rollout arithmetic -- the pair must
    cost strictly more, or `w_cost` is not reaching the second step."""
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    t = _teacher(horizon=50)
    obs = _obs(env)
    cands = t.candidates(env, obs)
    combo = next(c for c in cands if c.kind == "supported_push")
    solo = T.Candidate.single(combo.steps[0].slot, combo.steps[0].card_id,
                              combo.steps[0].x, combo.steps[0].y, "melee")
    assert t.sequence_cost(combo) > t.sequence_cost(solo)


# ==========================================================================
# the plan: commit the first card, keep the second reachable
# ==========================================================================
def test_choosing_a_combo_leaves_the_follow_up_reachable_next_decision():
    """`act()` can only return ONE placement per decision, so the second half
    of the plan has to survive to the next one. It is re-SCORED there rather
    than blindly executed -- by then the tank is on the board, so a solo
    rollout of the win condition SEES the escort. Rules propose, simulation
    ranks; that contract does not get suspended for a plan."""
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
    """A stale plan is worse than none: the board it was scored on is gone.
    `act` consumes whatever plan it inherited before it makes a new one, so a
    follow-up is offered on exactly the decision it was planned for."""
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
    """A plan surviving into the next match would place a card against a board
    that no longer exists."""
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
    """`gym_wrapper.step` unpacks exactly three values. The action space grew
    INSIDE the teacher; the interface it is driven through did not."""
    env = _env()
    t = _teacher(horizon=50)
    out = t.act(env, _obs(env))
    assert isinstance(out, tuple) and len(out) == 3
    slot, x, y = out
    assert 0 <= slot <= CE.HAND_SIZE


def test_the_teacher_actually_executes_a_planned_pair_end_to_end():
    """The integration nothing above covers: `act` -> `step_self_play` -> `act`
    -> `step_self_play`, driven exactly the way `gym_wrapper` drives it, with
    BOTH cards reaching the board.

    STAGED rather than sampled from a real match, deliberately. Measured
    2026-08-20, a combo is actually chosen roughly twice per 950 decisions of
    teacher-vs-teacher play, because the bar reaches the pair's price on ~1.4%
    of decisions -- so a match-driven version of this assertion would be flaky
    for a reason that has nothing to do with the code path it is testing. The
    RATE is measured in `eval/prove_combos.py`; this pins the MECHANISM.
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


# ==========================================================================
# reachability -- the half the candidate generator alone does NOT solve
# ==========================================================================
# MEASURED 2026-08-20, teacher vs teacher at stage 5, ~2,400 decisions:
#
#     elixir mean 1.79   p90 3.30
#     states where ANY combo was affordable            2
#     ...and both were at the 5.00 opening bar
#
# So a pair priced at 5-6 elixir is not merely rare, it is UNREACHABLE. The bot
# spends continuously (`w_pos` credits any cheap troop for standing forward),
# and a generator that proposes a play the bot can never afford has not solved
# the problem it was built for.
#
# A FLAT SAVINGS CHARGE WAS TRIED FIRST AND IS MEASURED DEAD. Charging every
# marginal spend while a push was within six decisions of affordable moved the
# bar from 1.79 to 1.73 across reserves of 0.0 / 1.5 / 3.0 / 5.0 -- i.e. not at
# all, and if anything the wrong way. A per-decision charge cannot produce
# multi-second saving when the bot has many attractive cheap plays and the
# defence exemption keeps firing. It was removed rather than shipped, on the
# same principle CLAUDE.md already records for back-row structure penalties:
# a penalty cannot move a distribution with no mass to move.
#
# What DOES lower the price is the follow-up DELAY. The pair's cost is paid
# across the gap, not at the plan, so 3-5 s of regeneration is worth 1.05-1.75
# elixir -- and a longer gap is also the tactically better play, because the
# tank needs time to get 1-2 tiles ahead of the win condition rather than the
# half tile it manages in one second.


def test_the_follow_up_delay_is_SEARCHED_not_fixed():
    """Timing is a candidate axis like placement is. One second of gap leaves
    the tank barely half a tile ahead and forces both cards to be affordable at
    once; three to five seconds is both cheaper and the escort shape the deploy
    time actually rewards. The simulator picks."""
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    t = _teacher(horizon=100, max_combos=12)
    delays = {c.steps[1].delay_ticks for c in t.candidates(env, _obs(env))
              if c.is_combo}
    assert len(delays) > 1, f"only one gap ever offered: {delays}"
    assert delays <= set(T.COMBO_FOLLOWUP_DELAYS)


def test_a_gap_is_only_offered_when_the_rollout_can_SEE_past_it():
    """A combo scored on a rollout that ends before its second card charges two
    cards and simulates one."""
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    for horizon in (30, 50, 100):
        t = _teacher(horizon=horizon, max_combos=12)
        for c in t.candidates(env, _obs(env)):
            if c.is_combo:
                assert c.max_delay + T.COMBO_FOLLOWUP_DELAY_TICKS <= horizon


def test_a_longer_gap_makes_a_pair_affordable_that_a_short_one_does_not():
    """The whole reason the delay is searched. Skeletons (1) + Hog (4) needs
    4.65 at a one-second gap and 3.95 at three seconds -- and the bar's p90 is
    3.30, so that difference is the difference between never and sometimes."""
    env = _env()
    _stage_hand(env, 0, [SKELETONS, HOG, CANNON, MUSK], 4.1)
    t = _teacher(horizon=100, max_combos=12)
    combos = [c for c in t.candidates(env, _obs(env)) if c.is_combo]
    assert combos, "no combo affordable at 4.1 even with a long gap"
    assert all(c.steps[1].delay_ticks > T.COMBO_FOLLOWUP_DELAY_TICKS
               for c in combos), (
        "a one-second gap should still be unaffordable here")


def test_a_delayed_plan_is_held_until_its_own_decision_comes_round():
    """A plan with a three-second gap must not fire on the next decision. The
    follow-up is offered when it is DUE and never before -- otherwise the gap
    that was scored is not the gap that is played."""
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
    """The narrow reserve that replaced the flat one. It is not "save toward
    some push"; it is "you committed to a plan two seconds ago, do not spend
    the money it needs". Bounded by construction to the few decisions a plan is
    live, which is why it can work where the flat charge could not."""
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
    """A reserve that holds elixir through an incoming push does not save
    elixir, it loses the tower -- the same failure a FLAT solvency reserve
    produced. The threshold is `tactics.HOG_MAX_THREAT`, reused rather than
    restated, because it is already this project's calibrated "our half is
    clear enough to commit" line."""
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
    """Telemetry, and the reason it is on the teacher rather than recomputed by
    the harness: `eval/prove_combos.py` has to tell "the combo did not help"
    apart from "the combo never happened", and re-deriving the chosen kind from
    a returned `(slot, x, y)` cannot, because a combo's first step is
    indistinguishable from the same card played alone."""
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
    """The family added because AFFORDABILITY is the binding constraint.

    Measured: combos are chosen on 0.28% of decisions and the bar's p90 is
    3.30, so every family that needs 5-6 elixir is priced out of most states.
    Ice Spirit + Skeletons is TWO elixir and is a real 2.6 defensive pair --
    chip and stall, then bodies. It fires in the states the expensive families
    cannot reach, and unlike `defensive_stack` it needs no building in hand.
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
    """`combo_families` exists because the win-rate A/B came back with two of
    three runs pointing negative, and the only way to tell WHICH family is
    responsible is to remove one and re-measure. Ablating by configuration
    keeps the arms byte-identical everywhere else -- editing the family list in
    source between runs would not."""
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
    """A name in COMBO_FAMILIES that no generator emits would silently make an
    ablation a no-op, and the ablation would read as 'that family was harmless'."""
    assert set(T.COMBO_FAMILIES) == {
        "supported_push", "counter_push", "defensive_stack", "cheap_defence",
        "spell_then_push", "push_then_spell"}
    t = _teacher()
    for name in T.COMBO_FAMILIES:
        assert hasattr(t, f"_combo_{name}"), name


# ==========================================================================
# weight injection -- what a profile sweep needs
# ==========================================================================
def test_a_profile_can_be_given_as_an_explicit_WEIGHT_SET():
    """`PROFILES` has three named entries and a sweep needs to try weights that
    are not among them. Passing a mapping is the minimal way in; the named form
    keeps working unchanged because everything else in the tree uses it."""
    weights = dict(T.PROFILES["balanced"], w_pos=7.5)
    t = T.UtilityTeacher(DECK, team=0, profile=weights, horizon_ticks=50)
    t.reset()
    assert t.profile["w_pos"] == 7.5
    assert t.profile["w_cost"] == T.PROFILES["balanced"]["w_cost"]


def test_an_injected_weight_set_SURVIVES_reset():
    """`reset()` re-draws the profile every match unless one was pinned, and it
    used to re-read it out of `PROFILES` by name. A swept weight set that
    silently reverted on the first reset would make every arm of a sweep
    measure the same thing -- and the sweep would report a flat line and look
    like a null result rather than a broken harness."""
    weights = dict(T.PROFILES["balanced"], w_pos=3.0)
    t = T.UtilityTeacher(DECK, team=1, profile=weights, horizon_ticks=50)
    for _ in range(3):
        t.reset()
        assert t.profile["w_pos"] == 3.0


def test_an_injected_weight_set_is_COPIED_not_aliased():
    """Two teachers built from one dict must not share it -- a sweep builds
    both sides of a match from the same weights."""
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


# ==========================================================================
# play_margin -- the anti-dumping guard, finally set to a value that guards
# ==========================================================================
def test_play_margin_is_the_SWEPT_value_not_the_original_placeholder():
    """0.05 was an OFF SWITCH, and the number that says so was measured.

    `play_margin`'s own docstring says it exists "so rollout noise on a dead
    board cannot talk the bot into dumping". The marginal cheap plays it was
    meant to stop score a MEDIAN of 1.37 (measured 2026-08-20 over the plays
    the teacher actually chose while its win condition sat in hand and nothing
    threatened). A 0.05 guard is 27x below the thing it guards against, which
    is the same shape as HOG_DEFENSIVE_RESERVE's first value of 3.0 opening its
    gate on 0 of 542 states -- a constant chosen on plausibility that turns out
    to be a no-op.

    3.0 was selected by sweep and CONFIRMED on a fresh independent run:
    head-to-head against the shipped profile it scores 0.969 [0.917, 1.000],
    and 4.0 scores the same 0.969 while tripling wasted income (overflow
    4.3% -> 12.9%). So this is the knee, not the end of a monotone climb.
    """
    t = T.UtilityTeacher(DECK, team=0)
    assert t.play_margin == 3.0


def test_a_play_worth_less_than_the_margin_is_declined():
    """The gate itself, independent of the value. A candidate whose marginal
    utility is below the bar must lose to holding -- which is what makes the
    bar an economy control rather than a tiebreak."""
    env = _env()
    # BELOW the overflow line, or the taper zeroes the bar and the assertion
    # would be testing the taper instead of the gate.
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 6.0)
    t = _teacher(horizon=30)
    t.play_margin = 1e9          # nothing can clear this
    slot, _x, _y = t.act(env, _obs(env))
    assert slot == CE.HAND_SIZE, "an unreachable margin still let a card through"


def test_the_margin_does_not_reach_the_rules_only_rungs():
    """Stage 0 has horizon 0 and takes `_rules_only`, which ranks by role
    priority and never consults `play_margin`. Worth pinning because the
    curriculum's easiest rung must stay beatable: if raising the margin had
    silently made stage 0 hold as well, the ladder would have lost its bottom.
    """
    env = _env()
    _stage_hand(env, 0, [CANNON, SKELETONS, MUSK, ICE_SPIRIT], 6.0)
    # `_rules_only` ranks by ROLE PRIORITY against a threat, so a threat has to
    # exist or it correctly holds and this would pass vacuously.
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
    """A FIXED margin is wrong, and the measurement that shows it is the one
    where the opponent does nothing.

    Against an active opponent a high bar looks excellent -- the C++ heuristic
    constantly creates scoreable situations, and margin 3.0 beat the shipped
    profile 0.969 head to head. Against a PASSIVE opponent nothing scores above
    a fixed 3.0 at all, so the bot froze: 14 plays across 6 matches, elixir
    pinned at 9.56 (i.e. throwing away almost all income), tower damage more
    than halved (9143 -> 4063), and it dropped a match it should win trivially.

    An episode-0 agent IS passive, so a fixed high bar would hand phase 1 the
    zero-gradient environment the whole 2026-08-19 pivot exists to avoid.

    The taper reuses `score`'s own overflow relief: above ELIXIR_OVERFLOW_AT the
    bar is discarding income, so holding is NOT free and a marginal play stops
    needing to justify itself. Same threshold, same shape, one idea expressed
    once.
    """
    t = T.UtilityTeacher(DECK, team=0)
    assert t.effective_play_margin(5.0) == pytest.approx(t.play_margin)
    assert t.effective_play_margin(T.ELIXIR_OVERFLOW_AT) == pytest.approx(
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
    """The end-to-end version: full bar, nothing threatening, and the bot must
    not sit there. This is the state margin 3.0 froze in."""
    env = _env()
    _stage_hand(env, 0, [ICE_GOLEM, HOG, CANNON, SKELETONS], 10.0)
    t = _teacher(horizon=50)
    assert t.act(env, _obs(env))[0] < CE.HAND_SIZE, (
        "the teacher held a full bar with a free board -- it is wasting income")


def test_a_DUE_followup_is_not_charged_the_dumping_margin():
    """`play_margin` stops the bot DUMPING -- spending on a marginal play when
    holding was free. For a plan's second half, holding is NOT free: the first
    card is already on the board and already paid for, so declining the
    follow-up does not bank the elixir, it wastes the commitment.

    The exemption is narrow and this is the part that keeps it honest: the
    follow-up still has to be the ARGMAX over every other candidate. All it
    skips is the "beat holding by N" floor, whose premise is false here. It
    parallels `plan_reserve_penalty`'s exemption for the plan's own second
    half -- the same idea applied to the other gate.
    """
    t = T.UtilityTeacher(DECK, team=0)
    follow = T.Candidate.single(1, HOG, 14.0, 15.0, "wincon", kind="followup")
    single = T.Candidate.single(1, HOG, 14.0, 15.0, "wincon")
    assert t.margin_for(follow, 5.0) == 0.0
    assert t.margin_for(single, 5.0) == pytest.approx(t.play_margin)


def test_the_followup_exemption_does_not_leak_to_ordinary_plays():
    """If it did, `play_margin` would be off for everything the moment a plan
    existed -- and the economy control this whole session is about would be
    silently disabled."""
    t = T.UtilityTeacher(DECK, team=0)
    for kind in ("single", "supported_push", "counter_push", "cheap_defence"):
        c = T.Candidate.single(0, SKELETONS, 9.0, 10.0, "melee", kind=kind)
        assert t.margin_for(c, 5.0) == pytest.approx(t.play_margin), kind
