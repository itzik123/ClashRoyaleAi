"""The context key, computable identically from a replay frame and from our
engine's observation.

The prior is collected against a replay-side key and served against an obs-side
key; if they disagree it is served into the wrong bucket while everything looks
healthy. Both implementations live here and end in the same `_classify`, so
only threat location can differ, which the paired test exercises.

The key is coarse because KataCR's detector undercounts badly (it misses the
ego's own Princess Towers for the first 20 s of every episode), biasing any
count or HP magnitude between the two sides. Presence and lane are robust: a
missed unit moves a count a lot and an argmax rarely. So the key uses only
where the nearest threat is, and when.

`phase` is a time bucket, not a claim that the two worlds share an economy.
"""
from __future__ import annotations

import numpy as np

import clash_royale_env as _E
from python_ai.advisors import tactics

LANE_NONE, LANE_LEFT, LANE_RIGHT = 0, 1, 2
HALF_NONE, HALF_FAR, HALF_NEAR = 0, 1, 2
PHASE_EARLY, PHASE_LATE = 0, 1

N_LANE, N_HALF, N_PHASE = 3, 3, 2
N_CONTEXTS = N_LANE * N_HALF * N_PHASE

#: Seconds after which a match counts as late: a coarse "how far in are we".
PHASE_LATE_SECONDS = 120.0

#: Board centre, the fixed point of the mirror: 8.5, since cell indices run
#: 0..17 and the centre sits on the seam.
CENTRE_X = _E.ARENA_CENTER_X

#: First row of the river, from the engine (tactics derives it from
#: get_own_half_max_y()).
RIVER_Y = tactics.RIVER_Y


def context_index(lane: int, half: int, phase: int) -> int:
    return (lane * N_HALF + half) * N_PHASE + phase


def decode(idx: int) -> tuple[int, int, int]:
    phase = idx % N_PHASE
    rest = idx // N_PHASE
    return rest // N_HALF, rest % N_HALF, phase


def _classify(threat_x, threat_y, seconds: float) -> int:
    """(nearest threat, match seconds) -> context index. The single definition of
    every threshold, shared by both callers.
    """
    phase = PHASE_LATE if seconds >= PHASE_LATE_SECONDS else PHASE_EARLY
    if threat_x is None or threat_y is None:
        return context_index(LANE_NONE, HALF_NONE, phase)
    lane = LANE_LEFT if float(threat_x) < CENTRE_X else LANE_RIGHT
    half = HALF_NEAR if float(threat_y) < RIVER_Y else HALF_FAR
    return context_index(lane, half, phase)


def threat_from_obs(obs):
    """Nearest enemy body in the team-0 frame, as (x, y) or (None, None). Nearest
    is the smallest y (our King is at y=2.5). Ties break on HP, which decides
    only the lane.
    """
    hp = tactics.enemy_hp_map(obs)
    ys, xs = np.nonzero(hp > 0.0)
    if not len(ys):
        return None, None
    front = ys.min()
    cand = xs[ys == front]
    weights = hp[front, cand]
    return float(cand[int(np.argmax(weights))]), float(front)


def context_from_obs(obs) -> int:
    x, y = threat_from_obs(obs)
    return _classify(x, y, tactics.elapsed_ticks(obs) / 10.0)


def threat_from_replay(episode, frame: int, transform):
    """The same quantity from a recorded frame: the nearest bel=1 body, mapped
    into the engine frame.
    """
    from .katacr_format import NON_BODY_CLASSES

    best = None
    for u in episode.state[frame]["unit_infos"]:
        if u["cls"] is None or u["xy"] is None or u.get("bel") != 1:
            continue
        if episode.idx2unit[u["cls"]] in NON_BODY_CLASSES:
            continue
        x, y = transform.to_engine(float(u["xy"][0]), float(u["xy"][1]))
        if best is None or y < best[1]:
            best = (x, y)
    return best if best is not None else (None, None)


def context_from_replay(episode, frame: int, transform) -> int:
    x, y = threat_from_replay(episode, frame, transform)
    return _classify(x, y, episode.seconds_at(frame))
