"""The four scripted opponents, now that they are a policy and not a method.

They were 120 lines inside `MicroRoyaleSelfPlayEnv._scripted_opponent_action`,
where nothing could reach them without building a whole self-play environment
(a frozen MicroRoyaleNet included) just to ask what a Defender does about a
tank. As free functions over an observation vector they are directly testable,
which is what these are.
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

    Costs are in the engine's own normalized units (card->cost / 10.0), so 0.3
    is a 3-elixir card; a cost of 0.0 marks an EMPTY hand slot and never a free
    card.
    """
    obs = np.zeros(SPATIAL_SIZE + 1 + HAND_SIZE + 32, dtype=np.float32)
    spatial = obs[:SPATIAL_SIZE].reshape(N_CHANNELS, BOARD_H, BOARD_W)
    for ch, x, y in enemies:
        spatial[ch, int(y), int(x)] = 1.0
    obs[SPATIAL_SIZE] = elixir
    obs[SPATIAL_SIZE + 1:SPATIAL_SIZE + 1 + HAND_SIZE] = costs
    return obs


# ------------------------------------------------------------- the roster --
def test_the_roster_is_the_four_permanent_pool_members():
    assert SCRIPTED_OPPONENTS == ["scripted:Rusher", "scripted:Defender",
                                  "scripted:Cycler", "scripted:Counter"]


def test_the_two_defensive_bots_hold_a_weight_FLOOR_above_the_pfsp_one():
    """PFSP's own criterion works AGAINST seeing them: mastering an opponent
    drives its weight to the floor. 0.20 was tried first and confirmed too low
    -- in a ~98-member pool that is ~7.7% combined, statistically invisible."""
    from python_ai.envs.selfplay_env import PFSP_MIN_WEIGHT
    assert DEFENSIVE_SCRIPTED_OPPONENTS == {"scripted:Defender",
                                            "scripted:Counter"}
    assert DEFENSIVE_SCRIPTED_MIN_WEIGHT == 0.8
    assert DEFENSIVE_SCRIPTED_MIN_WEIGHT > PFSP_MIN_WEIGHT


# --------------------------------------------------------- find_incursion --
def test_an_empty_board_has_no_incursion():
    spatial = make_obs()[:SPATIAL_SIZE].reshape(N_CHANNELS, BOARD_H, BOARD_W)
    assert find_incursion(spatial, MAX_Y) is None


def test_a_channel_6_unit_is_heavy_and_a_channel_4_unit_is_not():
    """Channel 6 is the building-targeter / tank archetype -- the same category
    the scenarios treat as "the real threat". A plain squad troop does not set
    it even if it is also in range, and that distinction is what Defender
    escalates on."""
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
    """It previously scanned only up to int(MAX_Y), so Defender/Counter noticed
    a threat only once it had already crossed the river -- often most of the way
    to the tower by the time a response spawned. An isolated eval found 55-70%
    of head-to-head games ending in an early blowout regardless of exposure:
    sampling weight was not the bottleneck, REACTION LATENCY was."""
    deep = int(MAX_Y) + 8
    assert deep < BOARD_H
    spatial = make_obs(enemies=[(6, 4, deep)])[:SPATIAL_SIZE].reshape(
        N_CHANNELS, BOARD_H, BOARD_W)
    found = find_incursion(spatial, MAX_Y)
    assert found is not None, "a threat in the enemy half must still be seen"


def test_the_response_y_is_clamped_to_our_own_legal_half():
    """Placement is only ever legal in our own half, so a threat still crossing
    is met at the bridge -- the earliest legal interception point."""
    spatial = make_obs(enemies=[(6, 4, BOARD_H - 2)])[:SPATIAL_SIZE].reshape(
        N_CHANNELS, BOARD_H, BOARD_W)
    _x, y, _heavy = find_incursion(spatial, MAX_Y)
    assert y <= MAX_Y


# ------------------------------------------------------- the four policies --
@pytest.mark.parametrize("kind", ["Rusher", "Defender", "Cycler", "Counter"])
def test_nothing_affordable_means_no_op(kind):
    """cost 0 marks an EMPTY slot, never a free card -- so a hand of empties
    must not be read as four playable cards."""
    obs = make_obs(elixir=0.0, costs=(0.0, 0.0, 0.0, 0.0))
    action = scripted_action(kind, obs, "left", MAX_X, MAX_Y)
    assert action[0] == NO_OP


def test_rusher_plays_its_most_expensive_card_into_its_committed_lane():
    """Rusher and Counter commit to one lane for the whole episode -- real
    players do not re-decide their push lane on every card."""
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
    """It is a DEFENDER: with no incursion there is nothing to answer, and
    spending anyway would just be a worse Cycler."""
    obs = make_obs(elixir=1.0)
    assert scripted_action("Defender", obs, "left", MAX_X, MAX_Y)[0] == NO_OP


def test_defender_escalates_to_its_strongest_card_against_a_TANK():
    """A Defender that always reaches for its cheapest card regardless of what
    is attacking loses to any sufficiently strong rush no matter how often it
    is sampled -- which is what made facing it more often worth nothing."""
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
    """`opponent_kind` is "neural" on most steps, and the dispatch must not
    raise on it -- a worker crash mid-rollout is far worse than a wasted turn."""
    obs = make_obs(elixir=1.0)
    assert scripted_action("neural", obs, "left", MAX_X, MAX_Y)[0] == NO_OP


def test_the_heuristics_never_read_a_card_ID():
    """They are handed a RANDOMIZED deck, so they must be deck-agnostic by
    construction: only elixir/cost from the scalar tail and enemy positions
    from the spatial channels."""
    import inspect

    from python_ai.envs import scripted_opponents
    src = inspect.getsource(scripted_opponents.scripted_action)
    for forbidden in ("get_card_info", "card_id", "DEFAULT_DECK",
                      "hand_card_ids"):
        assert forbidden not in src, (
            f"{forbidden} would make these opponents deck-specific")
