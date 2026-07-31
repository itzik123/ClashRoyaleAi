"""King Tower HP, which ClashRoyaleBuildABot does not read at all.

WHY THIS EXISTS
---------------
CRBAB's `Numbers` carries four fields -- both Princess towers per side -- and
nothing for the Kings. The engine's observation needs six: extra scalars 3-5
are our king/left/right and 6-8 are the enemy's (see ClashEnv.h,
"TOWER HP as explicit scalars"). Measured on a real ladder frame, our King sat
at 241 HP while the match was being lost; a policy blind to that cannot know it
is about to lose.

TWO THINGS DIFFER FROM THE PRINCESS BARS, AND BOTH BITE
------------------------------------------------------
1. AN ABSENT BAR MEANS FULL, NOT DESTROYED. Clash Royale draws no HP bar over
   an undamaged King -- only its level crown. CRBAB's `_calculate_hp` returns
   0.0 when it cannot find a bar, which is correct for a Princess (gone means
   destroyed) and exactly inverted for a King. A King at 0 ends the match, so
   0.0 is not even a reachable steady state; "no bar" is always full.

2. THE EMPTY SEGMENT IS A DIFFERENT COLOUR. Sampled off real frames, the
   King bar's unfilled portion is dark brown (~111,94,83) for BOTH sides,
   where `ALLY_HP_RHS_COLOUR` is a dark blue-grey (63,79,112). Reusing the
   Princess constants finds nothing on the ally King.

So rather than match exact RGB triples, each column is classified by hue
family -- clearly blue, clearly magenta, or neither. That also disposes of the
HP NUMERAL, which the game draws in white ON TOP of the bar and which would
otherwise punch a hole through the middle of any exact-colour match. This is
the same failure already documented for the elixir bar in readers/elixir.py.

COORDINATES
-----------
Screenshot space (368x652), the frame every CRBAB detector works in, so these
survive a device-resolution change -- which matters, because the emulator was
900x1600 when this was written and 720x1280 an hour later.

Bar bounds were measured, not guessed: a column profile over a frame with both
Kings damaged put the bar at x 164..218. Validated by cross-check -- the enemy
King read a 0.400 fill while displaying 1208 HP, and 1208/0.400 = 3020 against
a level-4 King's ~2928-3096, i.e. agreement to about one pixel of boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Bar interior, screenshot space. Both Kings sit on the board's centre line so
# they share x; only the row differs. Rows are the interior only -- the gold
# border above and below is excluded, since it is the one thing on screen that
# is present whether or not the bar is drawn.
KING_BAR_X0 = 164
KING_BAR_X1 = 218
ALLY_KING_BAR_Y = 491
ENEMY_KING_BAR_Y = 21
KING_BAR_HEIGHT = 6

# A column counts as filled/track if this fraction of its rows match.
COLUMN_FILL_RATIO = 0.34

# PRESENCE IS DECIDED ON THE EMPTY TRACK, NOT ON THE FILL. Measured on a real
# frame: at 241 HP the ally King's bar showed NO blue fill whatever -- the
# remaining sliver is under a pixel and the white "241" numeral is drawn
# starting at x=164, right where the fill would be. Deciding presence on fill
# therefore reported a King at 8% as "no bar", i.e. FULL, which is the most
# dangerous direction this function can be wrong in.
#
# The unfilled track is dark brown on BOTH sides (~111,94,83 fading to
# ~68,43,36) and covers most of the bar exactly when HP is low, so it is the
# reliable presence signal precisely when the fill is not.
MIN_TRACK_COLUMNS = 12


@dataclass(frozen=True)
class KingHp:
    fraction: float
    """0.0-1.0 of the King's own maximum. A FRACTION, never absolute HP: the
    engine models level-9 towers (2534 Princess) while real matches are played
    at whatever level the account has -- measured 1750 at level 4 and 1890 at
    level 5. Absolute values are not comparable across that gap; fractions are.
    """

    bar_present: bool
    """False means no bar was drawn, i.e. undamaged. Reported rather than
    folded into `fraction` so a caller can distinguish "certainly full" from
    "measured as full"."""

    columns: int


def _is_ally(rgb: np.ndarray) -> np.ndarray:
    """Clearly blue: blue dominant over both other channels and bright."""
    r, g, b = rgb[..., 0].astype(np.int16), rgb[..., 1].astype(np.int16), rgb[..., 2].astype(np.int16)
    return (b > r + 40) & (b > 120) & (g > r)


def _is_enemy(rgb: np.ndarray) -> np.ndarray:
    """Clearly magenta/pink: red dominant, blue well above green."""
    r, g, b = rgb[..., 0].astype(np.int16), rgb[..., 1].astype(np.int16), rgb[..., 2].astype(np.int16)
    return (r > 150) & (r > g + 60) & (b > g + 10)


def _is_track(rgb: np.ndarray) -> np.ndarray:
    """The bar's UNFILLED channel: a dark, desaturated brown, r > g > b.

    Excludes the gold border (too bright, r ~ 246) and the arena floor (green,
    so g > r). Same colour family on both sides -- measured identical browns
    under the ally and the enemy King.
    """
    r, g, b = rgb[..., 0].astype(np.int16), rgb[..., 1].astype(np.int16), rgb[..., 2].astype(np.int16)
    return (r > 40) & (r < 145) & (g < r) & (b < g) & (r - b > 15) & (r - b < 75)


def read_king_hp(image, ally: bool) -> KingHp:
    """King HP fraction from a screenshot-space RGB frame.

    `image` is anything array-convertible in the 368x652 frame CRBAB uses --
    a PIL Image or an ndarray.
    """
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"expected an HxWx3 RGB frame, got shape {arr.shape}")

    y0 = ALLY_KING_BAR_Y if ally else ENEMY_KING_BAR_Y
    band = arr[y0:y0 + KING_BAR_HEIGHT, KING_BAR_X0:KING_BAR_X1, :3]
    if band.size == 0:
        raise ValueError(
            f"King bar row {y0} falls outside a {arr.shape[1]}x{arr.shape[0]} "
            "frame -- these constants are screenshot space (368x652), not "
            "device pixels."
        )

    filled = (_is_ally(band) if ally else _is_enemy(band)).mean(axis=0) >= COLUMN_FILL_RATIO
    track = _is_track(band).mean(axis=0) >= COLUMN_FILL_RATIO
    columns = int(filled.sum())
    width = KING_BAR_X1 - KING_BAR_X0

    # Presence first, and on the TRACK -- see MIN_TRACK_COLUMNS. A bar is drawn
    # iff the King is damaged, so track+fill together should span most of it.
    if int(track.sum()) + columns < MIN_TRACK_COLUMNS:
        return KingHp(fraction=1.0, bar_present=False, columns=columns)

    if columns == 0:
        # Bar drawn but no fill resolvable: the King is at a few percent and
        # the numeral covers what little remains. Report the smallest value the
        # bar can express rather than 0.0 -- a King at 0 has ended the match,
        # so 0.0 would assert something that cannot be observed.
        return KingHp(fraction=1.0 / width, bar_present=True, columns=0)

    # Fill runs from the left edge, so the rightmost filled column is the
    # boundary. Using the count instead would under-read whenever the numeral
    # blanks a column in the middle of the filled run.
    boundary = int(np.nonzero(filled)[0].max()) + 1
    return KingHp(fraction=min(1.0, boundary / width), bar_present=True, columns=columns)
