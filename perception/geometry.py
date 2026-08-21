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

THE RIVER WAS NOT CENTRED ON THE BOARD -- FIXED UPSTREAM
---------------------------------------------------------
Previously worth knowing before trusting any mirrored coordinate, now fixed
at the source. The towers are perfectly symmetric under the engine's own
mirror (y -> 33 - y): King 2.5 <-> 30.5, Princess 6.0 <-> 27.0. The river
used to NOT be -- Board.h put it at [16.0, 18.0), whose centre is 17.0,
against the tower layout's 16.5.

That gave GameManager::isValidPlacement two different edges for the two
teams (team 0 playable y <= 15.5, team 1 playable y >= 18.5, which mirrors
to y' <= 14.5 -- team 1 had one full row less placeable ground). See
perception/README.md, "Findings reported upstream", item 1.

Fixed by re-centring the river on 16.5: Board.h now has
riverY_start=15.5/riverY_end=17.5, bridges at y=16.5. Both teams now get
y <= 15.0 in their own mirrored frame. RIVER_Y_END/LEFT_BRIDGE/RIGHT_BRIDGE
below are updated to match; river_y_start stays live-derived from
get_own_half_max_y() and needs no change here.

TWO MORE X-OFFSETS -- FIXED UPSTREAM 2026-07-30
------------------------------------------------
UPSTREAM_REQUESTS.md items 1-2, fitted from real-recording homography (8
landmarks, aggregated over 8 matches): the left Princess sat a full tile off
its own bridge (both at x=3.0/4.0 while the right side already agreed with
itself at 14.0/14.0), and the Kings sat at 8.5 instead of the board's
measured true centre 9.0. Both corrected directly in GameManager::reset();
OWN_KING/OPP_KING/OWN_PRINCESS_LEFT/OPP_PRINCESS_LEFT below are updated to
match. `tools/calibrate.py`'s corrected_tiles() hypothesis (built to
separate "the calibration is wrong" from "the engine disagrees with the
arena") now converges with engine_tiles() -- see that file's own docstring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import engine as _engine_build  # noqa: F401 -- see engine.py; must precede
                                # any `import clash_royale_env` in the process
from contracts import BoardGeometry

# GameManager.h: OWN_HALF_RIVER_BUFFER.
OWN_HALF_RIVER_BUFFER = 0.5

# Board.h:17 (post river-recentring fix -- was 18.0). Not reachable through
# any binding -- see module docstring.
RIVER_Y_END = 17.5

# Board.h:17-18 (post river-recentring fix -- was {4.0,17.0}/{14.0,17.0}).
# leftBridge{4.0, 16.5}, rightBridge{14.0, 16.5}. These are the single points
# river-crossing pathing actually uses. NOT to be confused with the x in
# {3,4} and {13,14} band that ClashEnv::extractObservationForTeam paints
# into observation channel 8 -- that is a wider visual hint for the
# network, not the geometry, and using it as a calibration anchor would put
# every bridge landmark half a tile off.
# Defined further down, once _engine_module() exists: as of 2026-08-21 these
# ARE derivable (ArenaLayout.h is bound), and every one of them had gone stale
# before that -- see _load_arena_landmarks().

# Fallbacks used only when the engine cannot be imported (ClashEnv.h:43,48).
# load_geometry() prefers the live values every time; these exist so
# calibration and its tests run on a machine with no built .pyd.
_FALLBACK_WIDTH = 18
_FALLBACK_HEIGHT = 34
_FALLBACK_OWN_HALF_MAX_Y = 15.0


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
    if str(python_ai.parent) not in sys.path:
        sys.path.insert(0, str(python_ai.parent))
    try:
        import clash_royale_env  # noqa: PLC0415
    except Exception:
        return None
    return clash_royale_env


# ---------------------------------------------------------------------------
# Arena landmarks.
#
# These were hardcoded here for as long as the file existed, under a comment
# saying no binding exposed them. That was true, and all six went STALE the
# moment the arena was corrected on 2026-08-21 -- this file still held King
# x=9.0 and left Princess x=4.0 against an engine that had moved to 8.5 and 3.0,
# which would have silently thrown off every calibration fit that uses them as
# landmarks.
#
# ArenaLayout.h is bound now (src/bindings.cpp), so they are read live. The
# fallbacks are the CURRENT values and exist only for the case this file already
# handles everywhere else: perception is a separate venv that may not be able to
# load a .pyd built for Python 3.11.
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
        # The single points river-crossing pathing actually uses. NOT the wider
        # band ClashEnv::extractObservationForTeam paints into observation
        # channel 8 -- that is a visual hint for the network, and using it as a
        # calibration anchor would put every bridge landmark half a tile off.
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
