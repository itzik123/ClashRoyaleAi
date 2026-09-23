"""The opponent card-cycle blocks, from the Python side.

`tests/core/test_card_cycle_observation.cpp` pins the engine. This pins what
the C++ suite cannot see: `MicroRoyaleNet` sizes its scalar input from the
engine, and every consumer of the observation tail still reads the tail now
that the cycle blocks sit behind it. A tail offset counted back from the end
would silently read card-recency floats, and `enemy_tower_hp` feeds the
reward's tower potential.
"""
import numpy as np
import pytest

import clash_royale_env
from python_ai import engine_constants as EC
from python_ai.envs import gym_wrapper
from python_ai.models.net import MicroRoyaleNet

_E = clash_royale_env.ClashRoyaleEnv
CE = clash_royale_env.ClashRoyaleEnv
DECK = list(gym_wrapper.DEFAULT_DECK)


@pytest.fixture(scope="module")
def env():
    e = CE(DECK, DECK, 3600)
    e.reset()
    return e


def _obs(e, team=0):
    return np.asarray(e.get_observation_for_team(team), dtype=np.float32)


def test_the_engine_exposes_the_cycle_block_geometry():
    """Bound, not restated."""
    assert _E.NUM_CYCLE_BLOCKS == 2
    assert _E.CYCLE_BLOCK_SIZE == _E.NUM_CYCLE_BLOCKS * _E.NUM_CARD_IDS
    assert _E.CYCLE_RECENCY_TAU_TICKS > 0


def test_observation_size_grew_by_exactly_the_cycle_blocks(env):
    obs = _obs(env)
    assert len(obs) == env.observation_size()
    expected = (_E.BOARD_WIDTH * _E.BOARD_HEIGHT * _E.NUM_CHANNELS
                + 1 + _E.HAND_SIZE + _E.HAND_SIZE * _E.NUM_CARD_IDS
                + _E.NUM_EXTRA_SCALARS + _E.CYCLE_BLOCK_SIZE)
    assert env.observation_size() == expected


def test_the_network_sizes_its_scalar_input_from_the_engine():
    """The net absorbs the new block automatically, rather than failing on shape
    or (worse) loading a stale checkpoint that feeds cycle floats into weights
    fitted without them.
    """
    net = MicroRoyaleNet(num_ability_slots=0)
    assert net.cycle_block_size == _E.CYCLE_BLOCK_SIZE
    assert net.spatial_size + net.scalar_size == CE(DECK, DECK, 100).observation_size()
    # The blocks sit after the extra scalars, so the offset into the tail is
    # unmoved.
    assert net.cycle_start == net.extra_start + net.num_extra_scalars


def test_the_tower_hp_scalars_still_read_as_tower_hp(env):
    """The regression that would have been silent. On a fresh board the tower
    scalars are strictly positive while the card-recency floats are all zero,
    so an offset that slid into the cycle blocks fails immediately.
    """
    obs = _obs(env)
    start = EC.EXTRA_SCALARS_START
    own = obs[start + 3:start + 6]
    enemy = obs[start + 6:start + 9]
    assert np.all(own > 0.0), f"own tower HP scalars read {own}, expected all > 0"
    assert np.all(enemy > 0.0), f"enemy tower HP scalars read {enemy}"
    recovered = float(own.sum()) * _E.MAX_BUILDING_HP
    assert recovered == pytest.approx(EC.OWN_TOWER_HP_TOTAL, rel=1e-5)


def test_a_fresh_board_has_an_entirely_empty_cycle(env):
    e = CE(DECK, DECK, 3600)
    e.reset()
    obs = _obs(e)
    block = obs[EC.CYCLE_START:EC.CYCLE_START + _E.CYCLE_BLOCK_SIZE]
    assert block.shape[0] == _E.CYCLE_BLOCK_SIZE
    assert np.all(block == 0.0), "nothing has been played, so nothing may be seen"


def test_a_noted_play_appears_in_the_opponents_view_only():
    """`note_played_card` lands in the other team's observation, never the
    player's own (the agent already sees its own hand).
    """
    e = CE(DECK, DECK, 3600)
    e.reset()
    card = DECK[0]
    e.note_played_card(1, card)

    seen_by_0 = _obs(e, 0)[EC.CYCLE_START + card]
    seen_by_1 = _obs(e, 1)[EC.CYCLE_START + card]
    assert seen_by_0 == 1.0, "team 0 must see what team 1 played"
    assert seen_by_1 == 0.0, "team 1 must not be handed its own cycle here"


def test_recency_decays_and_seen_does_not():
    e = CE(DECK, DECK, 3600)
    e.reset()
    card = DECK[0]
    e.note_played_card(1, card)
    played_at = e.get_last_played_tick(1, card)
    assert played_at >= 0

    tau = int(_E.CYCLE_RECENCY_TAU_TICKS)
    rec_idx = EC.CYCLE_START + _E.NUM_CARD_IDS + card

    e.set_current_tick(played_at)
    assert _obs(e)[rec_idx] == pytest.approx(1.0)

    e.set_current_tick(played_at + tau)
    assert _obs(e)[rec_idx] == pytest.approx(np.exp(-1.0), abs=1e-4)
    assert _obs(e)[EC.CYCLE_START + card] == 1.0, "seen must not decay"


def test_every_cycle_value_stays_in_the_unit_interval():
    """Every other channel is in [0, 1]; these must be too."""
    e = CE(DECK, DECK, 3600)
    e.reset()
    for i, card in enumerate(DECK):
        e.note_played_card(1, card)
        e.set_current_tick(i * 37)
    block = _obs(e)[EC.CYCLE_START:EC.CYCLE_START + _E.CYCLE_BLOCK_SIZE]
    assert np.all(np.isfinite(block))
    assert block.min() >= 0.0 and block.max() <= 1.0
