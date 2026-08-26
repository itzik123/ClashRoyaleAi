"""Screen pixels <-> ENGINE tile coordinates, anchored on the Princess Towers.

WHY NOT THE CALIBRATION PROFILE
-------------------------------
`config/profile_gpg_1920x1080.json` maps screen -> tile through a homography
that was FIT against tile coordinates taken from `geometry.py` as it stood on
2026-07-30 -- when the left Princess Tower was believed to sit at x=4.0 and
the Kings at x=9.0. The engine REVERSED that on 2026-08-21 (left Princess back
to 3.0, centre back to 8.5; see CLAUDE.md, "Board geometry"). The profile was
never re-fit, so every tile x it produces is still in the old frame:

    profile says   own_princess_left -> x = 3.94,  own_princess_right -> 14.03
    engine says                         x = 3.00                        14.00

That is a seventh copy of the arena constants -- a FITTED one, which is why
the "derive, do not restate" rule did not catch it. Re-fitting the profile is
the real fix and belongs upstream of here; this module exists so a comparison
against the engine is not silently shifted while that happens.

WHAT THIS DOES INSTEAD
----------------------
Anchors on the two Princess Towers alone and reads the scale off their
separation. Both sides agree the towers exist where they are drawn, so
anchoring there isolates the thing actually being measured -- how units MOVE
-- from the question of where the arena's landmarks are.

Residuals of the resulting map against the profile's other anchors, which are
NOT used in the fit and so are an honest check:

    own_king    screen y 732 -> 1.99   (engine 2.5,  -0.51 rows)
    opp_king    screen y 197 -> 30.22  (engine 30.5, -0.28 rows)
    opp_princess           -> 27.00 / 3.00 / 14.07  (engine 27.0 / 3.0 / 14.0)

The Princess rows close to 0.00-0.07; the Kings do not, and that is reported
rather than fitted away -- a King sprite's visual centre and its footprint
centre are not obviously the same point, and averaging the two families would
hide a real disagreement inside a smaller mean error.
"""

from __future__ import annotations

from dataclasses import dataclass

# Hand-clicked Princess Tower centres, 1920x1080 desktop capture. Same source
# as profile_gpg_1920x1080.json's anchors_screen -- only the tile coordinates
# they are paired with differ.
OWN_PRINCESS_LEFT_PX = (819.0, 656.0)
OWN_PRINCESS_RIGHT_PX = (1101.0, 656.0)
OPP_PRINCESS_LEFT_PX = (817.0, 261.0)
OPP_PRINCESS_RIGHT_PX = (1105.0, 258.0)

# ArenaLayout.h. Hardcoded because this file has to run where the .pyd does
# not exist; the test pins them against the header.
ENGINE_PRINCESS_X = (3.0, 14.0)
ENGINE_PRINCESS_Y = (6.0, 27.0)


@dataclass(frozen=True)
class ArenaMap:
    x0_px: float      # screen x of engine x = ENGINE_PRINCESS_X[0]
    px_per_tile_x: float
    y0_px: float      # screen y of engine y = ENGINE_PRINCESS_Y[0]
    px_per_tile_y: float

    def to_tile(self, px: float, py: float) -> tuple[float, float]:
        x = ENGINE_PRINCESS_X[0] + (px - self.x0_px) / self.px_per_tile_x
        # Screen y grows downward; engine y grows toward the opponent.
        y = ENGINE_PRINCESS_Y[0] + (self.y0_px - py) / self.px_per_tile_y
        return x, y

    def to_px(self, x: float, y: float) -> tuple[float, float]:
        px = self.x0_px + (x - ENGINE_PRINCESS_X[0]) * self.px_per_tile_x
        py = self.y0_px - (y - ENGINE_PRINCESS_Y[0]) * self.px_per_tile_y
        return px, py


def default_map() -> ArenaMap:
    """The map for the 1920x1080 recordings in perception/videos/."""
    own_x = (OWN_PRINCESS_LEFT_PX[0] + OPP_PRINCESS_LEFT_PX[0]) / 2
    opp_x = (OWN_PRINCESS_RIGHT_PX[0] + OPP_PRINCESS_RIGHT_PX[0]) / 2
    span_x = ENGINE_PRINCESS_X[1] - ENGINE_PRINCESS_X[0]
    own_y = (OWN_PRINCESS_LEFT_PX[1] + OWN_PRINCESS_RIGHT_PX[1]) / 2
    opp_y = (OPP_PRINCESS_LEFT_PX[1] + OPP_PRINCESS_RIGHT_PX[1]) / 2
    span_y = ENGINE_PRINCESS_Y[1] - ENGINE_PRINCESS_Y[0]
    return ArenaMap(x0_px=own_x, px_per_tile_x=(opp_x - own_x) / span_x,
                    y0_px=own_y, px_per_tile_y=(own_y - opp_y) / span_y)


if __name__ == "__main__":
    m = default_map()
    print(f"px_per_tile: x {m.px_per_tile_x:.3f}  y {m.px_per_tile_y:.3f}")
    for name, px in [("own_king", (956.0, 732.0)), ("opp_king", (960.0, 197.0)),
                     ("own_princess_left", OWN_PRINCESS_LEFT_PX),
                     ("own_princess_right", OWN_PRINCESS_RIGHT_PX),
                     ("opp_princess_left", OPP_PRINCESS_LEFT_PX),
                     ("opp_princess_right", OPP_PRINCESS_RIGHT_PX),
                     ("left_bridge", (821.0, 465.5)),
                     ("right_bridge", (1102.5, 465.5))]:
        print(f"  {name:20s} -> ({m.to_tile(*px)[0]:6.2f}, {m.to_tile(*px)[1]:6.2f})")
