"""Tests for live/mirror.py and live/teacher_policy.py: the teacher playing as us.

Each case carries a control that must differ, because every failure mode here
is quiet: an ignored `hp=` leaves units at full, an ignored `deploy_ticks=0`
hands every visible unit a fresh deploy second, and an unchecked hand refusal
plays a card nobody chose. Design:
docs/design/specs/2026-08-24-live-teacher-play-design.md, section 7.
"""
from __future__ import annotations

import numpy as np
import pytest

from contracts import (
    UNKNOWN_CARD_SIM_ID,
    GameState,
    Phase,
    TowerObservation,
    UnitObservation,
)

DECK = [15, 6, 25, 40, 24, 72, 33, 7]          # 2.6 Hog Cycle
HOG, MUSKETEER, CANNON, ICE_GOLEM, SKELETONS = 15, 6, 25, 40, 24
HAND = (HOG, MUSKETEER, CANNON, ICE_GOLEM)
FULL = TowerObservation(hp_fraction=1.0, hp_measured=True)
NO_OP_CARD = 9


def board(*units, hand=HAND, elixir=7.0, seconds=30.0, **towers) -> GameState:
    t = dict(own_king=FULL, own_princess_left=FULL, own_princess_right=FULL,
             opp_king=FULL, opp_princess_left=FULL, opp_princess_right=FULL)
    t.update(towers)
    return GameState(units=tuple(units), my_elixir=elixir, my_hand=tuple(hand),
                     seconds_elapsed=seconds, phase=Phase.SINGLE, **t)


def unit(card_id, x, y, team=1, hp=1.0, measured=True) -> UnitObservation:
    return UnitObservation(card_sim_id=card_id, unit_name="", team=team,
                           tile_x=x, tile_y=y, hp_fraction=hp,
                           hp_measured=measured)


@pytest.fixture(scope="module")
def builder(engine):
    from live.mirror import MirrorBuilder
    return MirrorBuilder(DECK, engine=engine)


def _obs(env):
    return np.asarray(env.get_observation_for_team(0), np.float32)


def _advance(env, ticks):
    env.step_self_play(NO_OP_CARD, 0.0, 0.0, NO_OP_CARD, 0.0, 0.0, ticks)


def _enemy_cells(env):
    from forecast import occupancy, tower_cells
    return occupancy(_obs(env), 1) - tower_cells()


# --- the mirror ------------------------------------------------------------

def test_the_hand_is_set_and_reads_back_in_screen_order(builder):
    env, gaps = builder.build(board(hand=(ICE_GOLEM, CANNON, HOG, MUSKETEER)))
    assert gaps.hand_set, gaps.hand_reason
    assert list(env.get_hand_for_team(0)) == [ICE_GOLEM, CANNON, HOG, MUSKETEER]


@pytest.mark.parametrize("hand, why", [
    ((HOG, MUSKETEER, CANNON, UNKNOWN_CARD_SIM_ID), "unreadable"),
    ((HOG, MUSKETEER, CANNON), "perceived 3"),
    ((HOG, MUSKETEER, CANNON, 2), "refused"),            # Giant: not in the deck
    ((HOG, HOG, CANNON, ICE_GOLEM), "refused"),          # a duplicate
])
def test_a_hand_the_engine_cannot_hold_is_reported_not_ignored(builder, hand, why):
    _env, gaps = builder.build(board(hand=hand))
    assert not gaps.hand_set
    assert why in gaps.hand_reason
    assert any(f.startswith("hand_refused") for f in gaps.fabricated())


def test_visible_units_are_already_deployed(builder, engine):
    """deploy_ticks=0: a Hog perception can see moves on the next ticks. The
    control, a default inject, stands still for its deploy second."""
    env, _gaps = builder.build(board(unit(HOG, 3, 24)))
    # One tick first: an injected entity is not on the board until an update.
    _advance(env, 1)
    start = _enemy_cells(env)
    assert start, "the Hog never materialised"
    _advance(env, 7)
    moved = _enemy_cells(env)

    control = engine.ClashRoyaleEnv(DECK, DECK, 3600)
    control.seed(0)
    control.inject(HOG, 3.5, 24.5, 1)
    _advance(control, 1)
    c_start = _enemy_cells(control)
    _advance(control, 7)
    assert _enemy_cells(control) == c_start, "control moved: test is vacuous"
    assert moved != start


def test_a_measured_hp_fraction_reaches_the_engine(builder):
    """Half a bar injects half the card's own HP; the control is a full bar."""
    def hog_channel(env):
        # Channels 4-7 are the enemy's four type classes; a Hog is a
        # building-targeter (offset 2), so it never shares a channel with a
        # tower (offset 3).
        _advance(env, 1)
        return _obs(env)[:8 * 34 * 18].reshape(8, 34, 18)[6].max()

    full = hog_channel(builder.build(board(unit(HOG, 3, 24, hp=1.0)))[0])
    env, gaps = builder.build(board(unit(HOG, 3, 24, hp=0.5)))
    assert full > 0.0
    assert hog_channel(env) == pytest.approx(full * 0.5, rel=0.02)
    assert gaps.hp_assumed_full == 0


def test_an_unread_bar_is_full_and_counted(builder):
    _env, gaps = builder.build(board(unit(HOG, 3, 24, hp=1.0, measured=False)))
    assert gaps.hp_assumed_full == 1


def test_towers_take_a_fraction_of_the_engines_own_maximum(builder):
    env, gaps = builder.build(board(
        opp_princess_left=TowerObservation(0.4, True),
        opp_princess_right=TowerObservation(1.0, True, destroyed=True),
        own_king=TowerObservation(1.0, False),
        own_princess_right=TowerObservation(0.0, True)))
    assert env.get_tower_hp(1, 1) == round(0.4 * env.get_tower_max_hp(1, 1))
    assert env.get_tower_hp(1, 2) <= 0                        # destroyed
    assert env.get_tower_hp(1, 0) == env.get_tower_max_hp(1, 0)
    assert gaps.towers_assumed_full == ("own_king",)
    # A measured 0.0 that is not destroyed is a reading the engine refuses.
    assert gaps.towers_refused == ("own_princess_right",)
    assert env.get_tower_hp(0, 2) == env.get_tower_max_hp(0, 2)


def test_the_clock_and_elixir_come_from_perception(builder):
    env, gaps = builder.build(board(elixir=3.0, seconds=135.0))
    assert env.get_current_tick() == 1350
    assert env.get_elixir_multiplier() == pytest.approx(2.0)   # double from 2:00
    assert env.get_elixir_for_team(0) == pytest.approx(3.0)
    assert gaps.opp_elixir_assumed == pytest.approx(5.0)


def test_a_swarm_is_one_card_not_one_card_per_body(builder):
    skeletons = [unit(SKELETONS, 9, 20 + i) for i in range(3)]
    _env, gaps = builder.build(board(*skeletons))
    assert gaps.injected == 1
    assert builder.last_opp_bodies == {SKELETONS: 1}


def test_an_unmappable_unit_is_counted(builder):
    _env, gaps = builder.build(board(unit(UNKNOWN_CARD_SIM_ID, 9, 20)))
    assert gaps.unmappable_units == 1 and gaps.injected == 0


def test_two_builds_of_one_frame_are_identical(builder):
    gs = board(unit(HOG, 3, 24), unit(MUSKETEER, 8, 10, team=0, hp=0.6))
    a = _obs(builder.build(gs)[0]).copy()
    b = _obs(builder.build(gs)[0])
    assert np.array_equal(a, b)


# --- the policy --------------------------------------------------------------

class SpyTeacher:
    """Stands in for UtilityTeacher where the test is about the policy's
    handling, not the teacher's choice."""

    horizon_ticks = 0

    def __init__(self, answer=(0, 3.0, 10.0), raises=None):
        self.answer, self.raises, self.calls = answer, raises, 0
        self.last_kind = "single"
        self.reset()

    def reset(self):
        self._opp_spent_seen = None
        self._opp_idle_decisions = 0

    def act(self, env, obs):
        self.calls += 1
        if self.raises:
            raise self.raises
        return self.answer


def _policy(builder, **kw):
    from live.teacher_policy import TeacherPolicy
    kw.setdefault("tappable", lambda y: 0 < y < 33)
    return TeacherPolicy(DECK, builder=builder, **kw)


def test_the_real_teacher_answers_with_a_legal_play(builder):
    policy = _policy(builder)
    gs = board(unit(HOG, 3, 20), elixir=8.0)
    for now in range(3):
        d = policy.decide(gs, {0, 1, 2, 3}, float(now))
        if d.slot is not None:
            card = gs.my_hand[d.slot]
            env, _ = builder.build(gs)
            assert env.is_valid_placement(card, float(d.tile[0]),
                                          float(d.tile[1]), 0), d
    assert policy.stats.teacher_errors == 0
    assert policy.stats.decisions == 3


def test_a_refused_hand_holds_without_consulting_the_teacher(builder):
    spy = SpyTeacher()
    policy = _policy(builder, teacher=spy)
    d = policy.decide(board(hand=(HOG, MUSKETEER, CANNON, -1)), {0}, 0.0)
    assert d.slot is None and "hand not set" in d.why
    assert spy.calls == 0 and policy.stats.hand_refused == 1


def test_a_teacher_exception_is_a_hold_never_a_crash(builder):
    policy = _policy(builder, teacher=SpyTeacher(raises=RuntimeError("boom")))
    d = policy.decide(board(), {0}, 0.0)
    assert d.slot is None and "boom" in d.why
    assert policy.stats.teacher_errors == 1


def test_a_slot_the_screen_says_is_not_ready_is_held(builder):
    policy = _policy(builder, teacher=SpyTeacher(answer=(2, 9.0, 8.0)))
    assert policy.decide(board(), {0, 1}, 0.0).slot is None
    assert policy.decide(board(), {2}, 1.0).slot == 2


def test_an_untappable_row_is_held(builder):
    policy = _policy(builder, teacher=SpyTeacher(answer=(0, 9.0, 0.0)))
    d = policy.decide(board(), None, 0.0)
    assert d.slot is None and policy.stats.untappable == 1


def test_the_teachers_hold_is_a_hold(builder):
    policy = _policy(builder, teacher=SpyTeacher(answer=(4, 0.0, 0.0)))
    assert policy.decide(board(), None, 0.0).slot is None


def test_decision_interval_drift_is_measured(builder):
    policy = _policy(builder, teacher=SpyTeacher())
    policy.decide(board(), None, 0.0)
    assert "drift" not in policy.decide(board(), None, 1.0).why
    assert "drift" in policy.decide(board(), None, 3.0).why
    assert policy.stats.drift_events == 1
    assert policy.drift_s == pytest.approx(1.0)
    policy.on_screen_change(False)
    policy.on_screen_change(True)                 # a new match
    assert policy.drift_s == 0.0


def test_the_real_teacher_still_has_the_passivity_counter_this_feeds(engine):
    """The policy writes `_opp_spent_seen`; a rename in teacher.py would turn
    that write into a no-op, so it is pinned here."""
    from live.teacher_policy import TeacherPolicy  # noqa: F401  (pins the probe deck)
    from python_ai.opponents.teacher import UtilityTeacher
    import python_ai.opponents.teacher as T
    if T._PROBE_DECK is None:
        T._PROBE_DECK = list(DECK)
    t = UtilityTeacher(DECK, team=0)
    t.reset()
    assert hasattr(t, "_opp_spent_seen") and hasattr(t, "_opp_idle_decisions")
    assert hasattr(t, "_observe_opponent_spend")


def test_a_new_opponent_card_resets_the_passivity_count(builder, engine):
    """The mirror's opponent spend is always 0, so the teacher alone would call
    the opponent passive after COUNTER_PASSIVE_DECISIONS. The policy feeds it
    perception's signal: a new card on the board counts as spending."""
    from live.teacher_policy import TeacherPolicy
    import python_ai.opponents.teacher as T
    policy = TeacherPolicy(DECK, builder=builder, stage=1,
                           tappable=lambda y: True)
    quiet = board(unit(HOG, 3, 24))
    for now in range(T.COUNTER_PASSIVE_DECISIONS + 1):
        policy.decide(quiet, None, float(now))
    # Control: an unchanged board does read as idle.
    assert policy.teacher._opp_idle_decisions >= T.COUNTER_PASSIVE_DECISIONS
    policy.decide(board(unit(HOG, 3, 24), unit(MUSKETEER, 14, 26)), None, 20.0)
    assert policy.teacher._opp_idle_decisions == 0


def test_the_teacher_persists_across_decisions_and_resets_per_match(builder):
    spy = SpyTeacher()
    policy = _policy(builder, teacher=spy)
    policy.decide(board(), None, 0.0)
    spy._opp_idle_decisions = 5
    policy.decide(board(), None, 1.0)
    assert policy.teacher is spy and spy._opp_idle_decisions == 5
    policy.on_screen_change(True)
    assert spy._opp_idle_decisions == 0
