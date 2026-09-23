"""The four scripted opponents, as free functions over an observation vector.
"""
import numpy as np
import pytest

from python_ai.engine_constants import (
    BOARD_H, BOARD_W, HAND_SIZE, N_CHANNELS, SPATIAL_SIZE,
)
from python_ai.envs.scripted_opponents import (
    DEFENSIVE_SCRIPTED_MIN_WEIGHT, DEFENSIVE_SCRIPTED_OPPONENTS,
    SCRIPTED_OPPONENTS, find_incursion, scripted_action,
)

MAX_X, MAX_Y = 17.0, 15.0
NO_OP = HAND_SIZE


def make_obs(elixir=10.0, costs=(1.0, 2.0, 3.0, 4.0), enemies=()):
    """A synthetic observation. `enemies` is [(channel, x, y)].

    Costs are in the engine's normalised units (cost / 10.0), so 0.3 is a
    3-elixir card; 0.0 marks an empty hand slot, never a free card.
    """
    obs = np.zeros(SPATIAL_SIZE + 1 + HAND_SIZE + 32, dtype=np.float32)
    spatial = obs[:SPATIAL_SIZE].reshape(N_CHANNELS, BOARD_H, BOARD_W)
    for ch, x, y in enemies:
        spatial[ch, int(y), int(x)] = 1.0
    obs[SPATIAL_SIZE] = elixir
    obs[SPATIAL_SIZE + 1:SPATIAL_SIZE + 1 + HAND_SIZE] = costs
    return obs


# --- the roster ---
def test_the_roster_is_the_four_permanent_pool_members():
    assert SCRIPTED_OPPONENTS == ["scripted:Rusher", "scripted:Defender",
                                  "scripted:Cycler", "scripted:Counter"]


def test_the_two_defensive_bots_hold_a_weight_FLOOR_above_the_pfsp_one():
    """PFSP drives a mastered opponent's weight to the floor, which works against
    seeing the defensive bots; they hold a higher floor.
    """
    from python_ai.envs.selfplay_env import PFSP_MIN_WEIGHT
    assert DEFENSIVE_SCRIPTED_OPPONENTS == {"scripted:Defender",
                                            "scripted:Counter"}
    assert DEFENSIVE_SCRIPTED_MIN_WEIGHT == 0.8
    assert DEFENSIVE_SCRIPTED_MIN_WEIGHT > PFSP_MIN_WEIGHT


# --- find_incursion ---
def test_an_empty_board_has_no_incursion():
    spatial = make_obs()[:SPATIAL_SIZE].reshape(N_CHANNELS, BOARD_H, BOARD_W)
    assert find_incursion(spatial, MAX_Y) is None


def test_a_channel_6_unit_is_heavy_and_a_channel_4_unit_is_not():
    """Channel 6 is the building-targeter / tank archetype. A squad troop in range
    does not set it, and Defender escalates on that distinction.
    """
    tank = make_obs(enemies=[(6, 5, 10)])[:SPATIAL_SIZE].reshape(
        N_CHANNELS, BOARD_H, BOARD_W)
    assert find_incursion(tank, MAX_Y)[2] is True
    squad = make_obs(enemies=[(4, 5, 10)])[:SPATIAL_SIZE].reshape(
        N_CHANNELS, BOARD_H, BOARD_W)
    assert find_incursion(squad, MAX_Y)[2] is False


def test_a_tank_outranks_a_squad_troop_that_is_deeper():
    spatial = make_obs(enemies=[(4, 2, 1), (6, 9, 20)])[:SPATIAL_SIZE].reshape(
        N_CHANNELS, BOARD_H, BOARD_W)
    x, _y, heavy = find_incursion(spatial, MAX_Y)
    assert heavy is True and x == 9.0


def test_the_scan_reaches_the_ENEMY_half_not_just_our_own():
    """The scan covers the enemy half too, so Defender/Counter react before a
    threat crosses the river.
    """
    deep = int(MAX_Y) + 8
    assert deep < BOARD_H
    spatial = make_obs(enemies=[(6, 4, deep)])[:SPATIAL_SIZE].reshape(
        N_CHANNELS, BOARD_H, BOARD_W)
    found = find_incursion(spatial, MAX_Y)
    assert found is not None, "a threat in the enemy half must still be seen"


def test_the_response_y_is_clamped_to_our_own_legal_half():
    """Placement is legal only on our own half, so a threat still crossing is met
    at the bridge.
    """
    spatial = make_obs(enemies=[(6, 4, BOARD_H - 2)])[:SPATIAL_SIZE].reshape(
        N_CHANNELS, BOARD_H, BOARD_W)
    _x, y, _heavy = find_incursion(spatial, MAX_Y)
    assert y <= MAX_Y


# --- the four policies ---
@pytest.mark.parametrize("kind", ["Rusher", "Defender", "Cycler", "Counter"])
def test_nothing_affordable_means_no_op(kind):
    """Cost 0 marks an empty slot, never a free card."""
    obs = make_obs(elixir=0.0, costs=(0.0, 0.0, 0.0, 0.0))
    action = scripted_action(kind, obs, "left", MAX_X, MAX_Y)
    assert action[0] == NO_OP


def test_rusher_plays_its_most_expensive_card_into_its_committed_lane():
    """Rusher and Counter commit to one lane for the whole episode."""
    obs = make_obs(elixir=1.0, costs=(0.1, 0.4, 0.2, 0.3))
    slot, x, y, a1, a2 = scripted_action("Rusher", obs, "left", MAX_X, MAX_Y)
    assert slot == 1, "the priciest affordable slot"
    assert x == pytest.approx(MAX_X * 0.2)
    assert y == MAX_Y
    assert (a1, a2) == (False, False), "no ability logic in these heuristics"
    _slot, x_right, _y, _a1, _a2 = scripted_action(
        "Rusher", obs, "right", MAX_X, MAX_Y)
    assert x_right == pytest.approx(MAX_X * 0.8)


def test_cycler_plays_its_cheapest_card_in_the_middle():
    obs = make_obs(elixir=1.0, costs=(0.4, 0.1, 0.3, 0.2))
    slot, x, y, _a1, _a2 = scripted_action("Cycler", obs, "left", MAX_X, MAX_Y)
    assert slot == 1
    assert x == pytest.approx(MAX_X * 0.5) and y == pytest.approx(MAX_Y * 0.5)


def test_defender_waits_when_nothing_is_attacking():
    """With no incursion there is nothing to answer; spending anyway would be a
    worse Cycler.
    """
    obs = make_obs(elixir=1.0)
    assert scripted_action("Defender", obs, "left", MAX_X, MAX_Y)[0] == NO_OP


def test_defender_escalates_to_its_strongest_card_against_a_TANK():
    """Against a tank, the Defender reaches for its strongest card; always playing
    the cheapest loses to any strong rush.
    """
    obs = make_obs(elixir=1.0, costs=(0.1, 0.4, 0.2, 0.3),
                   enemies=[(6, 7, 10)])
    slot, x, y, _a1, _a2 = scripted_action("Defender", obs, "left", MAX_X, MAX_Y)
    assert slot == 1, "the priciest affordable answer"
    assert (x, y) == (7.0, 10.0), "placed onto the threat"


def test_defender_takes_the_cheap_trade_against_a_squad_troop():
    obs = make_obs(elixir=1.0, costs=(0.1, 0.4, 0.2, 0.3),
                   enemies=[(4, 7, 10)])
    assert scripted_action("Defender", obs, "left", MAX_X, MAX_Y)[0] == 0


def test_counter_defends_when_threatened_and_pushes_when_it_is_not():
    obs_threat = make_obs(elixir=1.0, costs=(0.1, 0.4, 0.2, 0.3),
                          enemies=[(4, 6, 9)])
    slot, x, y, _a1, _a2 = scripted_action("Counter", obs_threat, "right",
                                           MAX_X, MAX_Y)
    assert slot == 1 and (x, y) == (6.0, 9.0)

    obs_quiet = make_obs(elixir=1.0, costs=(0.1, 0.4, 0.2, 0.3))
    slot, x, y, _a1, _a2 = scripted_action("Counter", obs_quiet, "right",
                                           MAX_X, MAX_Y)
    assert slot == 1 and x == pytest.approx(MAX_X * 0.8) and y == MAX_Y


def test_an_unknown_kind_is_a_no_op_rather_than_an_exception():
    """`opponent_kind` is "neural" on most steps; the dispatch must not raise on
    it.
    """
    obs = make_obs(elixir=1.0)
    assert scripted_action("neural", obs, "left", MAX_X, MAX_Y)[0] == NO_OP


def test_the_heuristics_never_read_a_card_ID():
    """They are handed a randomised deck, so they read only costs from the scalar
    tail and enemy positions from the spatial channels.
    """
    import inspect

    from python_ai.envs import scripted_opponents
    src = inspect.getsource(scripted_opponents.scripted_action)
    for forbidden in ("get_card_info", "card_id", "DEFAULT_DECK",
                      "hand_card_ids"):
        assert forbidden not in src, (
            f"{forbidden} would make these opponents deck-specific")
