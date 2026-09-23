"""King Tower HP, which ClashRoyaleBuildABot does not read.

CRBAB's `Numbers` holds the four Princess towers and nothing for the Kings,
while the observation's extra scalars 3-8 need all six.

Two differences from the Princess bars:

1. An absent bar means full, not destroyed. No HP bar is drawn over an
   undamaged King, only its crown; a King at 0 ends the match, so 0.0 is never
   a steady state. CRBAB's `_calculate_hp` returns 0.0 for no bar, right for a
   Princess and inverted for a King.

2. The empty segment is a different colour: dark brown (~111,94,83) for both
   sides, where `ALLY_HP_RHS_COLOUR` is blue-grey (63,79,112). So each column
   is classified by hue family (clearly blue, clearly magenta, or neither),
   which also ignores the white HP numeral drawn on top of the bar (the same
   problem as readers/elixir.py).

Coordinates are CRBAB's 368x652 screenshot space, which survives a
device-resolution change. Bar bounds were measured from a column profile with
both Kings damaged (x 164..218) and cross-checked: an enemy King read 0.400
while showing 1208 HP, i.e. 3020 max against a level-4 King's ~2928-3096.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Bar interior, screenshot space. Both Kings sit on the centre line and share
# x; only the row differs. The gold border is excluded: it is present whether
# or not a bar is drawn.
KING_BAR_X0 = 164
KING_BAR_X1 = 218
ALLY_KING_BAR_Y = 491
ENEMY_KING_BAR_Y = 21
KING_BAR_HEIGHT = 6

# A column counts as filled/track if this fraction of its rows match.
COLUMN_FILL_RATIO = 0.34

# Presence is decided on the empty track, not the fill. At very low HP the
# remaining fill can be under a pixel and covered by the white numeral, so a
# fill test reports "no bar", i.e. full, the most dangerous direction to be
# wrong. The dark-brown track (~111,94,83 fading to ~68,43,36) covers most of
# the bar exactly when HP is low.
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
    """The bar's unfilled channel: a dark, desaturated brown, r > g > b. Excludes
    the gold border (r ~ 246) and the green floor (g > r). The same browns on
    both sides.
    """
    r, g, b = rgb[..., 0].astype(np.int16), rgb[..., 1].astype(np.int16), rgb[..., 2].astype(np.int16)
    return (r > 40) & (r < 145) & (g < r) & (b < g) & (r - b > 15) & (r - b < 75)


def read_king_hp(image, ally: bool) -> KingHp:
    """King HP fraction from a screenshot-space RGB frame: anything
    array-convertible in CRBAB's 368x652 frame.
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

    # Presence first, on the track (see MIN_TRACK_COLUMNS): a bar is drawn iff
    # the King is damaged, and track plus fill should span most of it.
    if int(track.sum()) + columns < MIN_TRACK_COLUMNS:
        return KingHp(fraction=1.0, bar_present=False, columns=columns)

    if columns == 0:
        # Bar drawn but no fill resolvable: the King is at a few percent under
        # the numeral. Report the smallest value the bar can express; 0.0 would
        # assert a match-ending state.
        return KingHp(fraction=1.0 / width, bar_present=True, columns=0)

    # Fill runs from the left, so the rightmost filled column is the boundary;
    # a count would under-read where the numeral blanks a column.
    boundary = int(np.nonzero(filled)[0].max()) + 1
    return KingHp(fraction=min(1.0, boundary / width), bar_present=True, columns=columns)
