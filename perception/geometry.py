"""Board geometry in simulator tile coordinates.

Derived from the live engine wherever the bindings expose enough to derive it,
and hardcoded only where they do not -- with a test pinning every hardcoded
value against the header it came from. This follows the same anti-drift rule
the engine itself already applies to its Python side (see ClashEnv.h's comment
on why BOARD_WIDTH and friends are bound as readable constants rather than
copied into model.py): a second copy of a geometric constant is exactly the
kind of thing that goes stale silently and takes a training run with it.

WHAT IS DERIVABLE, AND HOW
--------------------------
  width, height          -> ClashRoyaleEnv.BOARD_WIDTH / BOARD_HEIGHT
  river_y_start          -> get_own_half_max_y() + OWN_HALF_RIVER_BUFFER.
                            GameManager::getOwnHalfMaxY() returns
                            getRiverStart() - 0.5, so the river start comes
                            back exactly.
  river_y_end            -> NOT derivable. Board::getRiverEnd() is not
                            reachable through ClashEnv, and no bound quantity
                            depends on it. Hardcoded from Board.h:17.
  bridges, tower centres -> NOT derivable. No binding exposes board entities
                            or their positions. Hardcoded from Board.h:17-18
                            and GameManager::reset().

THE RIVER IS NOT CENTRED ON THE BOARD
-------------------------------------
Worth knowing before trusting any mirrored coordinate. The towers are
perfectly symmetric under the engine's own mirror (y -> 33 - y): King 2.5
<-> 30.5, Princess 6.0 <-> 27.0. The river is not. Board.h puts it at
[16.0, 18.0), whose centre is 17.0, while the tower layout's centre is 16.5.

The consequence is in GameManager::isValidPlacement, which gates the two
teams on different edges of that band:

    team 0 invalid when  y > getRiverStart() - 0.5   ->  playable y <= 15.5
    team 1 invalid when  y < getRiverEnd()   + 0.5   ->  playable y >= 18.5

Mirrored into team 1's own frame that is y' <= 14.5, against team 0's 15.5.
Team 1 has one full row less placeable ground than team 0, in a game the
engine otherwise treats as symmetric. See perception/README.md, "Findings
reported upstream" -- flagged, not worked around, and not this module's to
fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

from contracts import BoardGeometry

# GameManager.h: OWN_HALF_RIVER_BUFFER.
OWN_HALF_RIVER_BUFFER = 0.5

# Board.h:17. Not reachable through any binding -- see module docstring.
RIVER_Y_END = 18.0

# Board.h:17-18. leftBridge{4.0, 17.0}, rightBridge{14.0, 17.0}. These are
# the single points river-crossing pathing actually uses. NOT to be confused
# with the x in {3,4} and {13,14} band that ClashEnv::extractObservationForTeam
# paints into observation channel 8 -- that is a wider visual hint for the
# network, not the geometry, and using it as a calibration anchor would put
# every bridge landmark half a tile off.
LEFT_BRIDGE = (4.0, 17.0)
RIGHT_BRIDGE = (14.0, 17.0)

# GameManager::reset(). Kings sit on half-integer coordinates so their 4x4
# footprint lands flush on tile boundaries; Princess towers are 3-wide and
# sit on integers.
OWN_KING = (8.5, 2.5)
OPP_KING = (8.5, 30.5)
OWN_PRINCESS_LEFT = (3.0, 6.0)
OWN_PRINCESS_RIGHT = (14.0, 6.0)
OPP_PRINCESS_LEFT = (3.0, 27.0)
OPP_PRINCESS_RIGHT = (14.0, 27.0)

# Fallbacks used only when the engine cannot be imported (ClashEnv.h:43,48).
# load_geometry() prefers the live values every time; these exist so
# calibration and its tests run on a machine with no built .pyd.
_FALLBACK_WIDTH = 18
_FALLBACK_HEIGHT = 34
_FALLBACK_OWN_HALF_MAX_Y = 15.5


def _engine_module():
    """Import the simulator, or return None.

    The .pyd lives next to the training code and is built against Python
    3.11; perception is a separate venv that may or may not be able to load
    it. Everything in this package that can work without the engine does, and
    only bridge/sim_driver.py treats its absence as fatal.
    """
    python_ai = Path(__file__).resolve().parent.parent / "python_ai"
    if str(python_ai) not in sys.path:
        sys.path.insert(0, str(python_ai))
    try:
        import clash_royale_env  # noqa: PLC0415
    except Exception:
        return None
    return clash_royale_env


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
        # get_own_half_max_y() is an instance method, so a throwaway env is
        # needed. Constructing one runs a full GameManager::reset(), which is
        # cheap (six towers) and has no global side effects.
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
    """Raw board y -> team 1's own frame, matching the engine exactly.

    ClashEnv::extractObservationForTeam:111 -- `(BOARD_HEIGHT - 1 - rawY)`.
    Its own inverse, which is why ClashEnv::stepSelfPlay can use the same
    expression to convert team 1's action back into board coordinates.
    """
    return (geom.height - 1) - raw_y


def calibration_anchors(geom: BoardGeometry) -> dict[str, tuple[float, float]]:
    """Landmarks to solve the screen->tile homography from.

    The four Princess towers, deliberately, and not the bridges.

    A homography is only as well conditioned as the quadrilateral it is
    solved from. The Princess towers form a wide, near-rectangular quad
    spanning most of the arena in both axes. The obvious-looking alternative
    -- two bridges plus two Kings -- is a thin diamond centred on the board,
    which leaves the solve badly conditioned near the corners where placement
    accuracy matters most.

    They are also the most unambiguous thing on screen: fixed structures with
    hard edges that are present in every frame of every match, whereas the
    bridges are frequently occluded by the troops crossing them.

    Bridges and Kings are returned by validation_landmarks() instead, and are
    held out of the solve entirely so stage 0's error figure is a real
    out-of-sample measurement rather than a restatement of the fit residual.
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
