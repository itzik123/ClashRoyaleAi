"""Board geometry in simulator tile coordinates.

Derived from the live engine wherever the bindings allow, hardcoded only where
they do not, with a test pinning each hardcoded value against its header. A
second copy of a geometric constant goes stale silently.

  width, height          ClashRoyaleEnv.BOARD_WIDTH / BOARD_HEIGHT
  river_y_start          get_own_half_max_y() + OWN_HALF_RIVER_BUFFER
                         (getOwnHalfMaxY() is getRiverStart() - 0.5)
  bridges, towers        the ARENA_* bindings from ArenaLayout.h
  river_y_end            not bound; hardcoded from Board.h's riverY_end
"""

from __future__ import annotations

import sys
from pathlib import Path

import engine as _engine_build  # noqa: F401 -- see engine.py; must precede
                                # any `import clash_royale_env` in the process
from contracts import BoardGeometry

# GameManager.h: OWN_HALF_RIVER_BUFFER.
OWN_HALF_RIVER_BUFFER = 0.5

# Board.h riverY_end. Not reachable through any binding.
RIVER_Y_END = 17.5

# The arena landmarks (bridge centres, towers) are defined below, once
# _engine_module() exists.

# Fallbacks for when the engine cannot be imported (ClashEnv.h BOARD_WIDTH /
# BOARD_HEIGHT), so calibration and its tests run without a built .pyd.
_FALLBACK_WIDTH = 18
_FALLBACK_HEIGHT = 34
_FALLBACK_OWN_HALF_MAX_Y = 15.0


def _engine_module():
    """Import the simulator, or return None. The .pyd is built for Python 3.11 and
    perception's venv may not load it; only bridge/sim_driver.py treats its
    absence as fatal.
    """
    python_ai = Path(__file__).resolve().parent.parent / "python_ai"
    if str(python_ai) not in sys.path:
        sys.path.insert(0, str(python_ai))
    if str(python_ai.parent) not in sys.path:
        sys.path.insert(0, str(python_ai.parent))
    try:
        import clash_royale_env  # noqa: PLC0415
    except Exception:
        return None
    return clash_royale_env


# --- arena landmarks ---
# Read live from the ArenaLayout.h bindings. The fallbacks are the current
# values, for a venv that cannot load the .pyd.
def _load_arena_landmarks():
    engine = _engine_module()
    if engine is None or not hasattr(engine, "ARENA_LEFT_BRIDGE_X"):
        cx, ll, rl, lb, rb, by = 8.5, 3.0, 14.0, 2.5, 14.5, 16.5
        king_y = (2.5, 30.5)
        princess_y = (6.0, 27.0)
    else:
        cx = engine.ARENA_CENTER_X
        ll = engine.ARENA_LEFT_LANE_X
        rl = engine.ARENA_RIGHT_LANE_X
        lb = engine.ARENA_LEFT_BRIDGE_X
        rb = engine.ARENA_RIGHT_BRIDGE_X
        by = engine.ARENA_BRIDGE_Y
        king_y = (engine.arena_king_y(0), engine.arena_king_y(1))
        princess_y = (engine.arena_princess_y(0), engine.arena_princess_y(1))
    return {
        # The single points river-crossing pathing uses. Observation channel 8
        # marks the bridge columns (a two-column band), which cannot locate a
        # seam-centred point as well as these.
        "LEFT_BRIDGE": (lb, by),
        "RIGHT_BRIDGE": (rb, by),
        "OWN_KING": (cx, king_y[0]),
        "OPP_KING": (cx, king_y[1]),
        "OWN_PRINCESS_LEFT": (ll, princess_y[0]),
        "OWN_PRINCESS_RIGHT": (rl, princess_y[0]),
        "OPP_PRINCESS_LEFT": (ll, princess_y[1]),
        "OPP_PRINCESS_RIGHT": (rl, princess_y[1]),
    }


globals().update(_load_arena_landmarks())


def load_geometry() -> BoardGeometry:
    """Board geometry, live from the engine where the bindings allow."""
    engine = _engine_module()

    if engine is None:
        width = _FALLBACK_WIDTH
        height = _FALLBACK_HEIGHT
        own_half_max_y = _FALLBACK_OWN_HALF_MAX_Y
    else:
        width = engine.ClashRoyaleEnv.BOARD_WIDTH
        height = engine.ClashRoyaleEnv.BOARD_HEIGHT
        # get_own_half_max_y() is an instance method; constructing a throwaway
        # env is cheap and has no global side effects.
        probe = engine.ClashRoyaleEnv([15, 25, 6, 1, 0, 41, 7, 10],
                                      [15, 25, 6, 1, 0, 41, 7, 10], 10)
        own_half_max_y = probe.get_own_half_max_y()

    return BoardGeometry(
        width=width,
        height=height,
        river_y_start=own_half_max_y + OWN_HALF_RIVER_BUFFER,
        river_y_end=RIVER_Y_END,
        left_bridge=LEFT_BRIDGE,
        right_bridge=RIGHT_BRIDGE,
        own_king=OWN_KING,
        opp_king=OPP_KING,
        own_princess_left=OWN_PRINCESS_LEFT,
        own_princess_right=OWN_PRINCESS_RIGHT,
        opp_princess_left=OPP_PRINCESS_LEFT,
        opp_princess_right=OPP_PRINCESS_RIGHT,
    )


def mirror_y(raw_y: float, geom: BoardGeometry) -> float:
    """Raw board y -> team 1's own frame, matching the engine exactly:
    extractObservationForTeam's `(BOARD_HEIGHT - 1 - rawY)`. Its own inverse,
    which is why stepSelfPlay converts team 1's action back with the same
    expression.
    """
    return (geom.height - 1) - raw_y


def calibration_anchors(geom: BoardGeometry) -> dict[str, tuple[float, float]]:
    """Landmarks to solve the screen->tile homography from: the four Princess
    towers.

    A homography is only as well conditioned as its quadrilateral. The Princess
    towers form a wide, near-rectangular quad spanning most of the arena; two
    bridges plus two Kings form a thin central diamond, badly conditioned near
    the corners where placement accuracy matters most. They are also fixed,
    hard-edged and present in every frame, while bridges are often occluded by
    crossing troops. Bridges and Kings are held out in validation_landmarks(),
    so the calibration error is out-of-sample.
    """
    return {
        "own_princess_left": geom.own_princess_left,
        "own_princess_right": geom.own_princess_right,
        "opp_princess_left": geom.opp_princess_left,
        "opp_princess_right": geom.opp_princess_right,
    }


def validation_landmarks(geom: BoardGeometry) -> dict[str, tuple[float, float]]:
    """Held-out landmarks for measuring calibration error. Never fitted."""
    return {
        "left_bridge": geom.left_bridge,
        "right_bridge": geom.right_bridge,
        "own_king": geom.own_king,
        "opp_king": geom.opp_king,
    }
