"""Their detected-pixel-cell frame -> our engine's board frame.

Their arena is 32 rows and ours is 34, their y runs from the OPPONENT's edge
downward while ours runs from team 0's edge upward, and their unit coordinates
are detected bounding-box centres rather than tile centres. So the two frames
differ by a flip, a scale and an offset.

Rather than hardcode that transform -- which is exactly the "second copy of a
constant" this repo has been bitten by six times, and which would go stale the
moment either side's arena moved -- it is FITTED PER EPISODE from landmarks
both sides can see: the four towers. Our side of each landmark comes from the
bound `ARENA_*` constants, so the engine remains the only source of truth.

The fit's residual is returned with it and is a first-class output: a large
residual means the episode's detections are unreliable and the episode should
be dropped, not silently reconstructed against a bad transform.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import clash_royale_env as _E


@dataclass
class BoardTransform:
    ax: float
    bx: float
    ay: float
    by: float
    residual_tiles: float
    n_landmarks: int

    def to_engine(self, x: float, y: float) -> tuple[float, float]:
        return self.ax * x + self.bx, self.ay * y + self.by


def _their_tower_landmarks(episode) -> dict[tuple[str, int, str], tuple[float, float]]:
    """Median detected centre for each (tower class, side, lane).

    Median over the whole episode, because a single frame's detection is noisy
    and a tower does not move.
    """
    buckets: dict[tuple[str, int, str], list[tuple[float, float]]] = {}
    for s in episode.state[:episode.n_frames]:
        for u in s["unit_infos"]:
            if u["cls"] is None or u["xy"] is None or u.get("bel") is None:
                continue
            name = episode.idx2unit[u["cls"]]
            if name not in ("king-tower", "queen-tower"):
                continue
            x, y = float(u["xy"][0]), float(u["xy"][1])
            lane = "c" if name == "king-tower" else ("l" if x < 9.0 else "r")
            buckets.setdefault((name, int(u["bel"]), lane), []).append((x, y))
    out = {}
    for k, v in buckets.items():
        a = np.asarray(v)
        if len(a) >= 20:            # a tower seen only briefly is a misdetection
            out[k] = (float(np.median(a[:, 0])), float(np.median(a[:, 1])))
    return out


def _our_tower_landmarks() -> dict[tuple[str, int, str], tuple[float, float]]:
    """The same landmarks in OUR frame, derived from the bound arena constants.

    bel=0 is the ego player, which we reconstruct as team 0.
    """
    out = {}
    for bel, team in ((0, 0), (1, 1)):
        out[("king-tower", bel, "c")] = (_E.ARENA_CENTER_X, _E.arena_king_y(team))
        out[("queen-tower", bel, "l")] = (_E.ARENA_LEFT_LANE_X, _E.arena_princess_y(team))
        out[("queen-tower", bel, "r")] = (_E.ARENA_RIGHT_LANE_X, _E.arena_princess_y(team))
    return out


def fit_transform(episode) -> BoardTransform:
    theirs, ours = _their_tower_landmarks(episode), _our_tower_landmarks()
    keys = sorted(set(theirs) & set(ours), key=str)
    if len(keys) < 4:
        raise ValueError(
            f"only {len(keys)} tower landmarks in {episode.path.name}; "
            "cannot fit a board transform")
    tx = np.array([theirs[k][0] for k in keys]); ox = np.array([ours[k][0] for k in keys])
    ty = np.array([theirs[k][1] for k in keys]); oy = np.array([ours[k][1] for k in keys])
    ax, bx = np.polyfit(tx, ox, 1)
    ay, by = np.polyfit(ty, oy, 1)
    res = np.hypot(ax * tx + bx - ox, ay * ty + by - oy)
    return BoardTransform(float(ax), float(bx), float(ay), float(by),
                          float(res.mean()), len(keys))
