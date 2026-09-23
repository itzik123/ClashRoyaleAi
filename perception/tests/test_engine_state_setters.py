"""The state-estimator write interface, from the side that consumes it.

`tests/core/test_state_setters.cpp` pins the engine behaviour (clamps,
refusals, the King's wake latch) but cannot see the pybind surface, where a
renamed keyword, dropped default or reordered parameter would break every
caller here with the C++ suite green.

The live loop rebuilds a mirror `ClashRoyaleEnv` from what perception sees and
hands it to `UtilityTeacher`, which rolls each candidate forward on
`env.snapshot()`. Anything the mirror cannot be told, every candidate is scored
against wrongly (UPSTREAM_REQUESTS.md item 22).

Every case carries an internal control: a setter that does nothing satisfies a
naive "it refused an illegal value", and an ignored `hp=` satisfies a naive "it
clamped to full".
"""
from __future__ import annotations

import pytest

DECK = [15, 6, 25, 40, 24, 72, 33, 7]  # 2.6 Hog Cycle, DEFAULT_DECK

# Any card index outside [0, 4) skips stepSelfPlay's play branch: "advance time
# and do nothing".
NO_OP_CARD = 9

# CardStats.h's DEPLOY_TIME_TICKS, not bound, so hardcoded with its header
# named. Also a tripwire: deploy time is gameplay-affecting, and updating this
# line acknowledges a change.
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


# --- the binding surface ---

def test_every_item_22_binding_is_reachable_by_its_documented_name(engine):
    for name in ("set_tower_hp", "destroy_tower", "get_tower_hp",
                 "get_tower_max_hp", "set_current_tick"):
        assert hasattr(engine.ClashRoyaleEnv, name), name


def test_inject_still_accepts_its_original_four_argument_form(engine):
    """Back-compat: existing callers pass four arguments, so the new parameters
    must be optional and default to the old behaviour.
    """
    a, b = _env(engine, seed=11), _env(engine, seed=11)
    a.inject(15, 9.0, 20.0, 1)
    b.inject(15, 9.0, 20.0, 1, -1.0, -1)
    a.step_self_play(NO_OP_CARD, 0, 0, NO_OP_CARD, 0, 0, 50)
    b.step_self_play(NO_OP_CARD, 0, 0, NO_OP_CARD, 0, 0, 50)
    assert a.get_observation_for_team(0) == b.get_observation_for_team(0)


# --- tower HP, the fraction workflow perception uses ---

def test_tower_max_hp_distinguishes_the_king_from_a_princess(engine):
    """A single shared ceiling would silently over-heal one of them. Per-tower
    maxima are also what make the fraction workflow expressible: the caller
    multiplies its fraction by this tower's maximum, so the engine (level 9)
    never represents a real account's level.
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
    """Refusal, with the positive write first as its control: otherwise a setter
    that did nothing would pass.
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

    # Idempotent, and says so, so a caller re-sending a stale reading can tell.
    assert env.destroy_tower(1, 1) is False
    assert env.get_towers_alive(1) == 2
    assert env.get_towers_alive(0) == 3


# --- unit HP and deploy state on inject ---

def test_injected_hp_changes_what_the_unit_can_actually_do(engine):
    """Scored by the engine rather than read back, because the claim is that the
    rollout differs: a wounded Hog dies to the towers before arriving, a full
    one takes a Princess Tower down.

    Scored as a monotone ladder rather than at one hand-picked hp, so a speed
    change moves the threshold instead of inverting the test.
    """
    def damage_at(hp):
        env = _env(engine, seed=3)
        env.inject(15, 9.0, 20.0, 1, hp, 0)
        env.step_self_play(NO_OP_CARD, 0, 0, NO_OP_CARD, 0, 0, 140)
        return env.get_tower_damage_dealt(1)

    ladder = [damage_at(hp) for hp in (150.0, 400.0, 700.0, 1200.0)]

    # A full-health Hog reaches the tower...
    assert damage_at(-1.0) > 0
    # ...and one wounded far enough dies on the way: injected hp reaches the
    # simulation.
    assert ladder[0] == 0
    # Strictly increasing across the ladder, so a setter that clamps every
    # value to full cannot pass.
    assert ladder == sorted(ladder), ladder
    assert ladder[-1] > ladder[0]
    assert len(set(ladder)) > 2, f"hp barely moves the outcome: {ladder}"


def test_injected_hp_is_clamped_to_the_cards_own_full_health(engine):
    """With the control that separates clamping from ignoring: "clamped to full"
    and "parameter ignored" give the same number, so a below-maximum write is
    checked in the same case.
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

    # Control: 500 hp landed, so the parameter is read.
    assert lowered.get_tower_damage_dealt(1) != baseline.get_tower_damage_dealt(1)
    # ...and an impossible value is capped at the card's own health.
    assert over.get_tower_damage_dealt(1) == baseline.get_tower_damage_dealt(1)


def test_deploy_ticks_zero_recovers_exactly_the_deploy_second(engine):
    """inject -> spawnEntity -> applyCardMetadata sets deployTicksRemaining
    unconditionally, so a mirror rebuilt from perception would hand every unit
    a fresh deploy second, including one walking for six.

    Measured on arrival rather than damage dealt: over a long window the Hog
    deals its full damage either way, so a total-damage probe saturates and
    reports no difference.
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

    The window is derived from the two arrival times, not written down: the
    damage difference oscillates between zero and one hit with the Hog's attack
    cooldown, so any fixed tick count is one balance change from a trough. One
    tick before the un-deployed Hog arrives, the deployed one has landed
    exactly one hit and the other none.
    """
    default_arrival = _ticks_until_tower_damage(_deployed(engine, -1))
    deployed_arrival = _ticks_until_tower_damage(_deployed(engine, 0))
    assert default_arrival is not None and deployed_arrival is not None
    assert deployed_arrival < default_arrival

    # One hit's worth, read off the engine: get_card_info does not expose
    # damage.
    probe = _deployed(engine, 0)
    probe.step_self_play(NO_OP_CARD, 0, 0, NO_OP_CARD, 0, 0, deployed_arrival)
    one_hit = probe.get_tower_damage_dealt(1)
    assert one_hit > 0

    window = default_arrival - 1
    results = {}
    for deploy in (-1, 0):
        env = _deployed(engine, deploy)
        env.step_self_play(NO_OP_CARD, 0, 0, NO_OP_CARD, 0, 0, window)
        results[deploy] = env.get_tower_damage_dealt(1)

    assert results[-1] == 0, "the window is past the un-deployed Hog's arrival"
    assert results[0] == one_hit
    assert results[0] - results[-1] == one_hit


# --- the match clock ---

def test_set_current_tick_moves_the_observations_elapsed_time_scalar(engine):
    env = _env(engine, max_ticks=1000)
    env.reset()
    idx = _clock_index(engine)
    assert env.get_observation_for_team(0)[idx] == pytest.approx(0.0)

    env.set_current_tick(500)

    # Both teams: elapsed time is not mirrored.
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
    """The property the interface depends on: `UtilityTeacher.rollout_stats`
    scores every candidate on `env.snapshot()`, so anything told to the mirror
    must reach the snapshot.
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
