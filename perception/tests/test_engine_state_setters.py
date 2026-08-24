"""The state-estimator WRITE interface, from the side that consumes it.

WHY THESE LIVE HERE AND NOT IN THE C++ SUITE
--------------------------------------------
`tests/core/test_state_setters.cpp` pins the engine BEHAVIOUR -- the clamps,
the refusals, the King's wake latch. It cannot see the pybind surface at all,
and the binding is where this interface is actually used: a renamed keyword, a
dropped default, or a parameter silently reordered would leave the whole C++
suite green while every caller in `perception/` broke.

That gap is not hypothetical here. CLAUDE.md records `python_ai/`'s `.pyd`
going stale twice and hiding `set_elixir_for_team` / `set_hand_for_team` for
days, with the C++ suite passing throughout.

WHAT THIS INTERFACE IS FOR
--------------------------
The live loop rebuilds a mirror `ClashRoyaleEnv` from what perception can see
and hands it to `UtilityTeacher`, which ranks candidates by rolling each one
forward on `env.snapshot()`. Everything the mirror cannot be told is something
every candidate is then scored against wrongly -- see
`perception/UPSTREAM_REQUESTS.md` item 22.

EVERY CASE BELOW CARRIES AN INTERNAL CONTROL, deliberately. A setter that did
nothing at all would satisfy a naive "it refused an illegal value" assertion,
and an ignored `hp=` would satisfy a naive "it clamped to full" one. Both of
those were written, both passed against a do-nothing stub, and both had to be
strengthened before they could fail.
"""
from __future__ import annotations

import pytest

DECK = [15, 6, 25, 40, 24, 72, 33, 7]  # 2.6 Hog Cycle, DEFAULT_DECK

# Any card index outside [0, 4) skips the play branch in stepSelfPlay, which is
# how "advance time and do nothing" is expressed. Same sentinel forecast.py
# uses; named here rather than repeated as a bare 9.
NO_OP_CARD = 9

# CardStats.h's DEPLOY_TIME_TICKS. NOT derivable from Python -- it is not
# bound -- so this follows perception/geometry.py's documented fallback:
# hardcoded, with the header that owns it named. It is also a deliberate
# tripwire, the same way test_clash_env.cpp pins the observation size as a
# literal: deploy time is gameplay-affecting, so it changing at all should
# fail something, and updating this line is the acknowledgement.
DEPLOY_TIME_TICKS = 10


def _env(engine, max_ticks=3600, seed=7):
    env = engine.ClashRoyaleEnv(DECK, DECK, max_ticks)
    env.seed(seed)
    return env


def _clock_index(engine):
    """Index of the elapsed-time scalar, derived rather than restated."""
    cls = engine.ClashRoyaleEnv
    spatial = cls.BOARD_WIDTH * cls.BOARD_HEIGHT * cls.NUM_CHANNELS
    return spatial + 1 + cls.HAND_SIZE + cls.HAND_SIZE * cls.NUM_CARD_IDS


def _ticks_until_tower_damage(env, limit=300):
    for tick in range(1, limit + 1):
        env.step_self_play(NO_OP_CARD, 0, 0, NO_OP_CARD, 0, 0, 1)
        if env.get_tower_damage_dealt(1) > 0:
            return tick
    return None


# --------------------------------------------------------------------------
# The binding surface itself
# --------------------------------------------------------------------------

def test_every_item_22_binding_is_reachable_by_its_documented_name(engine):
    for name in ("set_tower_hp", "destroy_tower", "get_tower_hp",
                 "get_tower_max_hp", "set_current_tick"):
        assert hasattr(engine.ClashRoyaleEnv, name), name


def test_inject_still_accepts_its_original_four_argument_form(engine):
    """The back-compat guarantee item 22 was designed around.

    Every existing caller -- forecast.py, the prove_* harnesses, the audit
    tools -- passes four arguments. If the new parameters were not optional,
    or defaulted to anything other than today's behaviour, this interface
    would be a breaking change wearing an additive one's clothes.
    """
    a, b = _env(engine, seed=11), _env(engine, seed=11)
    a.inject(15, 9.0, 20.0, 1)
    b.inject(15, 9.0, 20.0, 1, -1.0, -1)
    a.step_self_play(NO_OP_CARD, 0, 0, NO_OP_CARD, 0, 0, 50)
    b.step_self_play(NO_OP_CARD, 0, 0, NO_OP_CARD, 0, 0, 50)
    assert a.get_observation_for_team(0) == b.get_observation_for_team(0)


# --------------------------------------------------------------------------
# Tower HP -- the fraction workflow perception will actually use
# --------------------------------------------------------------------------

def test_tower_max_hp_distinguishes_the_king_from_a_princess(engine):
    """A single shared ceiling would silently over-heal one of them.

    This is also what makes the fraction workflow expressible at all: the
    caller multiplies its measured fraction by THIS tower's maximum, so the
    level mismatch between this engine (level 9) and a real account (often
    level 4-5) never has to be represented on the engine side.
    """
    env = _env(engine)
    env.reset()
    king = env.get_tower_max_hp(1, 0)
    princess = env.get_tower_max_hp(1, 1)
    assert king > princess > 0
    assert env.get_tower_max_hp(1, 2) == princess


def test_a_measured_fraction_round_trips_through_set_tower_hp(engine):
    """The exact call MirrorBuilder makes: fraction x this tower's own max."""
    env = _env(engine)
    env.reset()
    for slot in (0, 1, 2):
        maximum = env.get_tower_max_hp(1, slot)
        assert env.set_tower_hp(1, slot, 0.37 * maximum) is True
        assert env.get_tower_hp(1, slot) == round(0.37 * maximum)


def test_set_tower_hp_refuses_zero_without_becoming_an_inert_setter(engine):
    """Refusal, with the control that makes the refusal meaningful.

    Without the positive write first, a setter that did nothing at all would
    pass this -- which is exactly the state the case exists to exclude.
    """
    env = _env(engine)
    env.reset()

    assert env.set_tower_hp(1, 1, 1234.0) is True      # control: it CAN write
    assert env.get_tower_hp(1, 1) == 1234

    assert env.set_tower_hp(1, 1, 0.0) is False
    assert env.get_tower_hp(1, 1) == 1234              # and wrote nothing
    assert env.get_towers_alive(1) == 3


def test_destroy_tower_is_the_only_way_to_take_a_tower_down(engine):
    env = _env(engine)
    env.reset()
    assert env.get_towers_alive(1) == 3

    assert env.destroy_tower(1, 1) is True
    assert env.get_towers_alive(1) == 2

    # Idempotent, and says so rather than silently succeeding -- a caller
    # re-sending a stale reading can tell the difference.
    assert env.destroy_tower(1, 1) is False
    assert env.get_towers_alive(1) == 2
    assert env.get_towers_alive(0) == 3


# --------------------------------------------------------------------------
# Unit HP and deploy state on inject
# --------------------------------------------------------------------------

def test_injected_hp_changes_what_the_unit_can_actually_do(engine):
    """Scored by the engine rather than read back, because that is the claim.

    A wounded Hog dies to the towers before it arrives; a full one takes a
    Princess Tower down. Reading `hp` back would only prove a field was
    written -- this proves the rollout the teacher scores actually differs.
    """
    full, wounded = _env(engine, seed=3), _env(engine, seed=3)
    full.inject(15, 9.0, 20.0, 1, -1.0, 0)
    wounded.inject(15, 9.0, 20.0, 1, 400.0, 0)
    full.step_self_play(NO_OP_CARD, 0, 0, NO_OP_CARD, 0, 0, 140)
    wounded.step_self_play(NO_OP_CARD, 0, 0, NO_OP_CARD, 0, 0, 140)

    assert full.get_tower_damage_dealt(1) > 0
    assert wounded.get_tower_damage_dealt(1) == 0


def test_injected_hp_is_clamped_to_the_cards_own_full_health(engine):
    """With the control that separates clamping from ignoring.

    "Clamped to full" and "the parameter was ignored" produce the SAME number,
    and ignoring it is the pre-item-22 behaviour -- so the below-maximum write
    has to be checked in the same case.
    """
    env = _env(engine)
    env.reset()
    lowered, over = _env(engine, seed=5), _env(engine, seed=5)

    lowered.inject(15, 9.0, 20.0, 1, 500.0, 0)
    over.inject(15, 9.0, 20.0, 1, 999999.0, 0)
    lowered.step_self_play(NO_OP_CARD, 0, 0, NO_OP_CARD, 0, 0, 140)
    over.step_self_play(NO_OP_CARD, 0, 0, NO_OP_CARD, 0, 0, 140)

    baseline = _env(engine, seed=5)
    baseline.inject(15, 9.0, 20.0, 1, -1.0, 0)
    baseline.step_self_play(NO_OP_CARD, 0, 0, NO_OP_CARD, 0, 0, 140)

    # Control: 500 hp really did land, so the parameter is read.
    assert lowered.get_tower_damage_dealt(1) != baseline.get_tower_damage_dealt(1)
    # ...and an impossible value is capped at the card's own health.
    assert over.get_tower_damage_dealt(1) == baseline.get_tower_damage_dealt(1)


def test_deploy_ticks_zero_recovers_exactly_the_deploy_second(engine):
    """The fourth gap, and the one nobody had written down.

    inject -> spawnEntity -> applyCardMetadata sets deployTicksRemaining
    unconditionally, so a mirror rebuilt from perception handed EVERY unit a
    fresh deploy second -- including one that had been walking for six. Every
    rollout then believed it had an extra second before anything could act.

    Measured on ARRIVAL rather than on damage dealt: over a long enough window
    the Hog deals its full damage either way, so a total-damage probe
    SATURATES and reports no difference at all. That probe was written first
    and measured exactly nothing.
    """
    default_arrival = _ticks_until_tower_damage(_deployed(engine, -1))
    deployed_arrival = _ticks_until_tower_damage(_deployed(engine, 0))

    assert default_arrival is not None and deployed_arrival is not None
    assert deployed_arrival < default_arrival
    assert default_arrival - deployed_arrival == DEPLOY_TIME_TICKS


def _deployed(engine, deploy_ticks):
    env = _env(engine)
    env.inject(15, 9.0, 20.0, 1, -1.0, deploy_ticks)
    return env


def test_the_deploy_subsidy_is_worth_a_whole_hog_hit_mid_push(engine):
    """Why the second matters, in the units the teacher is scored in.

    Chosen inside the window where arrival decides the outcome. Outside it the
    measurement saturates -- see the docstring above.
    """
    results = {}
    for deploy in (-1, 0):
        env = _deployed(engine, deploy)
        env.step_self_play(NO_OP_CARD, 0, 0, NO_OP_CARD, 0, 0, 110)
        results[deploy] = env.get_tower_damage_dealt(1)
    assert results[0] > results[-1]


# --------------------------------------------------------------------------
# The match clock
# --------------------------------------------------------------------------

def test_set_current_tick_moves_the_observations_elapsed_time_scalar(engine):
    env = _env(engine, max_ticks=1000)
    env.reset()
    idx = _clock_index(engine)
    assert env.get_observation_for_team(0)[idx] == pytest.approx(0.0)

    env.set_current_tick(500)

    # Both teams: the scalar is elapsed time, which is not mirrored.
    assert env.get_observation_for_team(0)[idx] == pytest.approx(0.5)
    assert env.get_observation_for_team(1)[idx] == pytest.approx(0.5)


def test_set_current_tick_clamps_into_a_range_the_match_can_reach(engine):
    env = _env(engine, max_ticks=1000)
    env.reset()
    idx = _clock_index(engine)

    env.set_current_tick(999999)
    assert env.get_observation_for_team(0)[idx] == pytest.approx(1.0)

    env.set_current_tick(-5)
    assert env.get_observation_for_team(0)[idx] == pytest.approx(0.0)


def test_the_injected_state_survives_into_the_snapshot_a_rollout_runs_on(engine):
    """The property the whole interface depends on.

    `UtilityTeacher.rollout_stats` scores every candidate on `env.snapshot()`.
    Anything the mirror was told that does not reach the snapshot is something
    every candidate is scored without -- i.e. the fresh board this item exists
    to replace, reintroduced one level down.
    """
    env = _env(engine, max_ticks=1000)
    env.reset()
    env.set_tower_hp(1, 1, 700.0)
    env.destroy_tower(1, 2)
    env.set_current_tick(300)

    copy = env.snapshot()

    assert copy.get_tower_hp(1, 1) == 700
    assert copy.get_towers_alive(1) == 2
    assert copy.get_observation_for_team(0)[_clock_index(engine)] == pytest.approx(0.3)
