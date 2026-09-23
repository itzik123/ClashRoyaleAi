"""Four hand-written opponents, driven purely from the observation vector.

They read only elixir and costs from the scalars and enemy positions from the
spatial channels, so they play sensibly with the random deck
`set_scripted_opponent` hands them.
"""
import numpy as np

from python_ai.engine_constants import (
    BOARD_H, BOARD_W, HAND_SIZE, N_CHANNELS, SPATIAL_SIZE,
)

# Permanent members of the PFSP pool (scripted bots before self-play, as in
# OpenAI Five), tagged "scripted:<name>" so pool dispatch and win-rate keys
# work unchanged.
SCRIPTED_OPPONENTS = ["scripted:Rusher", "scripted:Defender", "scripted:Cycler", "scripted:Counter"]

# Defender and Counter get a sampling floor above PFSP's, which would otherwise
# shrink them as the agent masters them; they supply defensive pressure PFSP
# cannot see. 0.8 gives them ~25% of episodes in a ~98-member pool.
DEFENSIVE_SCRIPTED_OPPONENTS = {"scripted:Defender", "scripted:Counter"}

DEFENSIVE_SCRIPTED_MIN_WEIGHT = 0.8


def find_incursion(spatial, max_y):
    """(x, y, is_heavy) for the most urgent enemy incursion, or None.

    Channels 4-6 are the enemy's troops from this observer's mirrored view.
    Scans the whole board so a threat is met at the bridge rather than once it
    is deep in our half. `is_heavy` is True when a channel-6 (building-targeter
    / tank) unit is involved; Defender escalates on it.
    """
    def clamp_y(y):
        # Placement is legal only on our half, so meet a threat still crossing
        # at the bridge.
        return min(float(y), float(int(max_y)))

    tank_nz = np.nonzero(spatial[6])
    if tank_nz[0].size > 0:
        ys, xs = tank_nz
        # Smallest y is closest to our tower: most urgent.
        deepest = int(np.argmin(ys))
        return float(xs[deepest]), clamp_y(ys[deepest]), True
    other_nz = np.nonzero(spatial[4] + spatial[5])
    if other_nz[0].size == 0:
        return None
    ys, xs = other_nz
    deepest = int(np.argmin(ys))
    return float(xs[deepest]), clamp_y(ys[deepest]), False


def scripted_action(kind, obs1, lane, max_x, max_y):
    """(card_index, x, y, ability1, ability2) for one scripted opponent. Never
    activates an ability.
    """
    spatial = obs1[:SPATIAL_SIZE].reshape(N_CHANNELS, BOARD_H, BOARD_W)
    scalar = obs1[SPATIAL_SIZE:]
    elixir = float(scalar[0])
    costs = scalar[1:1 + HAND_SIZE]
    # cost <= 0 marks an empty hand slot.
    affordable = [i for i in range(HAND_SIZE)
                  if costs[i] > 0.0 and costs[i] <= elixir + 1e-6]

    NO_OP = (HAND_SIZE, 0.0, 0.0, False, False)

    def lane_x():
        return max_x * (0.2 if lane == "left" else 0.8)

    if kind == "Rusher":
        if not affordable:
            return NO_OP
        return max(affordable, key=lambda i: costs[i]), lane_x(), max_y, False, False

    if kind == "Cycler":
        if not affordable:
            return NO_OP
        return (min(affordable, key=lambda i: costs[i]),
                max_x * 0.5, max_y * 0.5, False, False)

    incursion = find_incursion(spatial, max_y)

    if kind == "Defender":
        if incursion is None or not affordable:
            return NO_OP
        x, y, is_heavy = incursion
        # Strongest affordable answer against a win-condition threat, cheapest
        # against an ordinary squad.
        slot = (max(affordable, key=lambda i: costs[i]) if is_heavy
                else min(affordable, key=lambda i: costs[i]))
        return slot, x, y, False, False

    if kind == "Counter":
        if not affordable:
            return NO_OP
        slot = max(affordable, key=lambda i: costs[i])
        if incursion is not None:
            x, y, _is_heavy = incursion
            return slot, x, y, False, False
        return slot, lane_x(), max_y, False, False

    return NO_OP
