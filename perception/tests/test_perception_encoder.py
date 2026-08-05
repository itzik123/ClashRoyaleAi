"""Round-trip: a GameState encodes to the observation the ENGINE would emit.

WHY THIS TEST CAN BE EXACT
--------------------------
Most of perception can only be checked against hand labels, because there is no
ground truth for what a screen contains. The encoder is different: the engine
can be driven to a known board and asked for its own observation, and a
GameState describing that same board must encode to the identical vector. No
labels, no tolerance, no sampling -- byte equality or a bug.

That makes it the strongest test in the module, and the reason the encoder was
built against it rather than against a spec.

WHY THE GameState IS REBUILT FROM THE BOARD, NOT FROM THE INJECTION
-------------------------------------------------------------------
The obvious construction -- inject card X at (9,8), then assert a GameState
holding "X at (9,8)" matches -- is wrong, and failed exactly that way first.
A Minions placement is THREE bodies which the engine spreads across cells, so a
sensor would report three units at their real positions, not one at the
placement point. Reconstructing from the engine's board is what a perfect
sensor would have seen, which is the input the encoder is actually specified
against.
"""
from __future__ import annotations

import numpy as np
import pytest

from contracts import GameState, Phase, TowerObservation, UnitObservation

# No module-level `importorskip` for the engine: `python_ai/` reaches sys.path
# via conftest's `engine` fixture, which runs AFTER collection imports this
# file. An importorskip here therefore skips the whole module even when the
# binding is perfectly available -- which it did, silently, until the run
# reported "1 skipped" instead of the expected failures.

DECK = [10, 1, 41, 25, 7, 2, 6, 5]
FULL = TowerObservation(hp_fraction=1.0, hp_measured=True)


@pytest.fixture(scope="module")
def enc(engine):
    import perception_encoder  # noqa: PLC0415

    return perception_encoder


def _blank_state(env, units=()):
    return GameState(
        units=tuple(units), my_elixir=env.get_elixir(),
        my_hand=tuple(env.get_hand()), seconds_elapsed=0.1, phase=Phase.SINGLE,
        own_king=FULL, own_princess_left=FULL, own_princess_right=FULL,
        opp_king=FULL, opp_princess_left=FULL, opp_princess_right=FULL)


def _state_from_board(enc, env, injected_ids):
    """What a perfect sensor would report about the engine's current board."""
    obs = np.asarray(env.get_observation_for_team(0), np.float32)
    table = enc.card_table()
    tower_cells = {i for c in enc._tower_cells().values() for i in c.indices}
    plane = enc.PLANE
    units = []
    for team, base_ch in ((0, 0), (1, 4)):
        for offset in range(4):
            channel = base_ch + offset
            for i in range(channel * plane, (channel + 1) * plane):
                if obs[i] <= 1e-6 or i in tower_cells:
                    continue
                rem = i % plane
                row, col = divmod(rem, enc.BOARD_WIDTH)
                side = 0 if team == 0 else 1
                bodies = int(round(
                    obs[(enc.CH_COUNT + side) * plane + rem] * enc.MAX_CELL_UNITS))
                card_id = next((c for c in injected_ids
                                if table.get(c) and table[c].type_offset == offset),
                               None)
                if card_id is None:
                    continue
                attrs = table[card_id]
                fraction = float(obs[i]) * attrs.hp_divisor / attrs.max_hp
                for _ in range(max(bodies, 1)):
                    units.append(UnitObservation(
                        card_sim_id=card_id, unit_name="", team=team,
                        tile_x=col, tile_y=row, hp_fraction=fraction,
                        hp_measured=True))
    return _blank_state(env, units)


def _engine_with(engine, injects):
    env = engine.ClashRoyaleEnv(DECK, DECK)
    env.reset()
    for card_id, x, y, team in injects:
        env.inject(card_id, float(x), float(y), team)
    env.step(len(DECK) // 2, 0.0, 0.0, 1)      # no-op card index; advance a tick
    return env


def test_observation_size_matches_the_engine(enc, engine):
    env = engine.ClashRoyaleEnv(DECK, DECK)
    env.reset()
    assert enc.OBSERVATION_SIZE == env.observation_size()


def test_empty_board_is_bit_exact(enc, engine):
    """River and six towers, none of which perception reports as units. A
    units-only encoder leaves 34 floats at zero here -- that was the entire
    difference before towers were handled."""
    env = _engine_with(engine, [])
    truth = np.asarray(env.get_observation_for_team(0), np.float32)
    assert np.array_equal(enc.encode(_blank_state(env)), truth)


@pytest.mark.parametrize("tag,injects", [
    ("single troop", [(2, 9, 8, 0)]),
    ("multi-body card", [(41, 9, 8, 0)]),          # Minions: 3 bodies
    ("building", [(25, 9, 8, 0)]),                 # Cannon
    ("two types at once", [(2, 9, 8, 0), (6, 5, 10, 0)]),
    ("both teams", [(2, 9, 8, 0), (6, 5, 10, 0), (25, 12, 20, 1)]),
])
def test_round_trip_is_bit_exact(enc, engine, tag, injects):
    env = _engine_with(engine, injects)
    truth = np.asarray(env.get_observation_for_team(0), np.float32)
    state = _state_from_board(enc, env, [c for c, _, _, _ in injects])
    assert np.array_equal(enc.encode(state), truth), tag


def test_a_full_princess_is_not_encoded_as_1(enc, engine):
    """The conversion that a units-only reading would get wrong.

    Tower scalars are hp / MAX_BUILDING_HP, so an undamaged Princess reads
    0.632 (2534/4008). GameState carries a fraction of the tower's OWN maximum,
    where full is 1.0. Kings hide this -- they ARE 4008 -- so only the four
    Princesses ever disagreed.
    """
    env = _engine_with(engine, [])
    obs = enc.encode(_blank_state(env))
    extra = enc.SPATIAL_SIZE + 1 + enc.HAND_SIZE + enc.HAND_SIZE * enc.NUM_CARD_IDS
    king, princess = obs[extra + 3], obs[extra + 4]
    assert king == pytest.approx(1.0)
    assert 0.5 < princess < 0.8
    assert princess < king


def test_a_destroyed_tower_zeroes_its_cells_and_scalar(enc, engine):
    env = _engine_with(engine, [])
    dead = TowerObservation(hp_fraction=0.0, hp_measured=True, destroyed=True)
    state = GameState(
        units=(), my_elixir=5.0, my_hand=tuple(env.get_hand()),
        seconds_elapsed=0.1, phase=Phase.SINGLE,
        own_king=FULL, own_princess_left=dead, own_princess_right=FULL,
        opp_king=FULL, opp_princess_left=FULL, opp_princess_right=FULL)
    obs = enc.encode(state)
    for i in enc._tower_cells()["own_princess_left"].indices:
        assert obs[i] == 0.0
    extra = enc.SPATIAL_SIZE + 1 + enc.HAND_SIZE + enc.HAND_SIZE * enc.NUM_CARD_IDS
    assert obs[extra + 4] == 0.0


def test_unmeasured_opponent_spend_uses_the_named_constant(enc, engine):
    """None must not silently become an anonymous zero. The 900-episode result
    validated 0.0 specifically, so the value is named and changing it is
    understood to invalidate that measurement."""
    env = _engine_with(engine, [])
    obs = enc.encode(_blank_state(env))
    extra = enc.SPATIAL_SIZE + 1 + enc.HAND_SIZE + enc.HAND_SIZE * enc.NUM_CARD_IDS
    assert obs[extra + 2] == enc.OPP_SPEND_WHEN_UNMEASURED


def test_card_table_covers_the_playable_roster_minus_spells(enc, engine):
    table = enc.card_table()
    assert len(table) > 100
    assert all(not engine.get_card_info(c)["is_spell"] for c in table)
    giant = table[2]
    assert giant.type_offset == 2 and giant.units == 1
    minions = table[41]
    assert minions.units == 3 and minions.flying == 1.0
