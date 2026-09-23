"""Engine constants, derived once from the compiled bindings.

Never keep a second copy of an engine constant in Python: read it here. Where
no binding exposes a value, keep it in the module that needs it with a comment
naming the header it came from.
"""
import numpy as np

import clash_royale_env

# Observation layout (ClashEnv.h).
N_CHANNELS = clash_royale_env.ClashRoyaleEnv.NUM_CHANNELS

BOARD_H = clash_royale_env.ClashRoyaleEnv.BOARD_HEIGHT

BOARD_W = clash_royale_env.ClashRoyaleEnv.BOARD_WIDTH

SPATIAL_SIZE = N_CHANNELS * BOARD_H * BOARD_W

# The HP normalizers ClashEnv::extractObservation() divides by.
MAX_TROOP_HP = clash_royale_env.ClashRoyaleEnv.MAX_TROOP_HP

MAX_BUILDING_HP = clash_royale_env.ClashRoyaleEnv.MAX_BUILDING_HP

# Arena geometry, bound from ArenaLayout.h.
BOARD_CENTER_X = clash_royale_env.ARENA_CENTER_X

#: Princess Tower columns. Mirror images under (BOARD_W - 1) - x.
LEFT_LANE_X = clash_royale_env.ARENA_LEFT_LANE_X
RIGHT_LANE_X = clash_royale_env.ARENA_RIGHT_LANE_X

#: Bridge centres. A bridge is two tiles wide, so its centre sits on the seam
#: between them: 2.5 spans cells 2 and 3.
LEFT_BRIDGE_X = clash_royale_env.ARENA_LEFT_BRIDGE_X
RIGHT_BRIDGE_X = clash_royale_env.ARENA_RIGHT_BRIDGE_X
BRIDGE_Y = clash_royale_env.ARENA_BRIDGE_Y


def king_y(team):
    """Row of `team`'s King Tower."""
    return clash_royale_env.arena_king_y(team)


def princess_y(team):
    """Row of `team`'s two Princess Towers."""
    return clash_royale_env.arena_princess_y(team)

#: The card head's no-op column follows the hand slots.
HAND_SIZE = clash_royale_env.ClashRoyaleEnv.HAND_SIZE
NOOP_ACTION = HAND_SIZE

#: Scalars after the spatial block: elapsed time, both sides' cumulative elixir
#: spend, and the six tower HPs.
NUM_EXTRA_SCALARS = clash_royale_env.ClashRoyaleEnv.NUM_EXTRA_SCALARS

#: Forward offsets of the extra scalars and the opponent card-cycle blocks.
#: Never locate a section as `observation_size() - N`: sections are appended
#: over time, and a backward offset silently reads the wrong data after the
#: next append.
EXTRA_SCALARS_START = clash_royale_env.ClashRoyaleEnv.EXTRA_SCALARS_START
CYCLE_START = clash_royale_env.ClashRoyaleEnv.CYCLE_START
CYCLE_BLOCK_SIZE = clash_royale_env.ClashRoyaleEnv.CYCLE_BLOCK_SIZE

def _own_tower_hp_total():
    """Total starting HP of our three towers, read from a fresh board's tower
    scalars (hp / MAX_BUILDING_HP).
    """
    E = clash_royale_env.ClashRoyaleEnv
    probe = E(list(range(8)), list(range(8)), 100)
    obs = np.asarray(probe.reset(), dtype=np.float32)
    tail = E.EXTRA_SCALARS_START
    # tail: 0 time | 1-2 elixir spent | 3-5 own king/left/right | 6-8 enemy
    return float(obs[tail + 3:tail + 6].sum()) * E.MAX_BUILDING_HP

OWN_TOWER_HP_TOTAL = _own_tower_hp_total()

_CARD_NAME_CACHE = {}


def card_name(card_id):
    """Registry name for a card id, for diagnostic labels only.

    Evolutions share their base card's name (ids 1 and 128 are both "Archers"),
    so the id is appended.
    """
    if card_id not in _CARD_NAME_CACHE:
        try:
            name = clash_royale_env.get_card_info(card_id)["name"]
        except Exception:  # noqa: BLE001 -- unknown id is a label problem, not fatal
            name = "card"
        _CARD_NAME_CACHE[card_id] = f"{name.replace(' ', '_')}_{card_id}"
    return _CARD_NAME_CACHE[card_id]
