"""Per-unit HP, read from the bar above a detected unit.

STATUS: NOT READY. DO NOT WIRE INTO THE ADAPTER YET.
----------------------------------------------------
The approach is right and the pixel facts below are measured, but the
thresholds are not calibrated and the recall is bad. Two variants were tried
against 27 units from one ladder match:

    searching for the bar directly   44% found, widths mean 32 (true 38-56),
                                     most fractions pinned at 0.00
    anchoring on the level badge     widths correct (mean 38, max 46) but only
                                     11% found

The badge anchor is the better design -- it fixed the widths -- and it fails on
recall for a reason that is known rather than mysterious: `_is_badge` was
calibrated against **exactly one badge**, an ally Valkyrie's, and the enemy
magenta thresholds are a guess.

WHAT IT NEEDS, AND THE PART THAT IS NOT JUST TUNING
---------------------------------------------------
Badge colours sampled across many units and both teams, then thresholds set
from that distribution -- the same discipline that made king_hp.py work, where
a fixed ROI allowed exhaustive sampling.

But recall cannot honestly be tuned yet, because there is nothing to tune
against. The King bars could be validated against the HP NUMERAL the game
prints beside them; units have no numeral, only the bar. So "no bar found"
is genuinely ambiguous between *undamaged* (correct -- Clash Royale draws no
bar on a healthy unit) and *missed*, and 24 of those 27 units read that way.
Ground truth here means eyeballing crops of units known to be damaged and
labelling them. Until that exists, any threshold that improves the numbers is
being fitted to a metric nobody can see.

WHY THIS EXISTS
---------------
ClashRoyaleBuildABot reports no HP for units at all -- its `Position` carries
a bbox, a confidence and a tile, nothing more (upstream issue #245). But
observation channels 0-7 store `hp / MAX` per cell, which is **4,896 of the
13,606 floats** the policy consumes. Filling them with a constant would hand a
reinforcement learner fabricated data for a third of its input and make every
later evaluation ambiguous, so they are read instead.

The detector supplies the hard part -- where each unit is. This only has to
look just above that box.

WHAT THE PIXELS ACTUALLY LOOK LIKE
----------------------------------
Sampled off a native 720x1280 ladder frame, an ally unit's bar is:

    border   (0, 44, 98)      dark navy, one row above and below
    fill     (255,255,255)    near-WHITE -- not ALLY_HP_LHS_COLOUR (111,208,252)
    track    (65, 80, 113)    the unfilled remainder; matches ALLY_HP_RHS_COLOUR

Two consequences. Matching the fill against CRBAB's `*_LHS_COLOUR` constants
finds nothing, because at this size the fill saturates to white. And the TRACK
is the team-tinted part -- navy for ally, dark maroon for enemy -- so the track
is what identifies the bar, while brightness identifies the fill.

THE OFFSET IS SEARCHED, NOT ASSUMED
-----------------------------------
Measured on one frame: the bar sits ~12 px above the bbox top for a Valkyrie
and ~30-45 px above it for a Cannon. It scales with sprite height, so a fixed
offset is wrong for most units. A bounded window above the box is searched
instead, and when several bars fall inside it -- normal in a crowd -- the one
whose centre is nearest the unit's own centre wins.

AN ABSENT BAR MEANS FULL HP
---------------------------
Clash Royale draws no bar over an undamaged unit. Same convention as the King
Tower (see king_hp.py) and the opposite of a Princess tower, where an absent
bar means destroyed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Search window above the bbox top, in units of the bbox's own height. Scaled
# rather than absolute because the offset grows with sprite size.
SEARCH_ABOVE_FRACTION = 0.85
SEARCH_ABOVE_MIN_PX = 26
SEARCH_ABOVE_MAX_PX = 64

# Horizontal slack either side of the bbox. The badge sits left of the unit's
# centre and the bar extends right, so the pair is not centred on the box.
SEARCH_PAD_X = 26

# A bar must fall in this width range, in native pixels at 720x1280. Measured
# spans were 38-56; the bounds are loose around that. The MAXIMUM matters as
# much as the minimum -- without it, two adjacent units' bars merge into one
# 89-px run and the fill of one is divided by the width of both.
MIN_BAR_WIDTH = 20
MAX_BAR_WIDTH = 72

# A pixel counts as fill if this bright. The fill saturates to white; nothing
# else in the bar's row band is anywhere near it.
FILL_BRIGHTNESS = 185


@dataclass(frozen=True)
class UnitHp:
    fraction: float
    bar_found: bool
    """False means no bar was drawn, i.e. the unit is undamaged. Kept separate
    from `fraction` so a caller can tell 'certainly full' from 'measured full'
    -- and so a missed bar in a crowd is visible rather than silently 1.0."""

    width: int = 0


def _is_track(rgb: np.ndarray, ally: bool) -> np.ndarray:
    """The bar's unfilled remainder: dark, tinted to the team, NOT the border.

    Excluding the border is load-bearing. The bar is outlined in a much darker
    navy -- (0,44,98) against a track of (65,80,113) -- which passes every hue
    test the track passes, spans the bar's FULL width, and contains no fill at
    all. Scoring rows without excluding it selects the border row every time
    and reports a healthy unit as 0.00. Measured: that produced 0.00 for six of
    eight units, including one whose bar was over half full.

    Total intensity separates them cleanly: 258 for the track, 142 for the
    border.
    """
    rgb = rgb.astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    dark = (r < 170) & (g < 170) & (b < 190)
    not_border = rgb.sum(axis=-1) > 190
    if ally:
        # navy, e.g. (65,80,113)
        return dark & not_border & (b > r + 20) & (b > g + 15)
    # maroon, e.g. (90,49,68)
    return dark & not_border & (r > g + 20) & (r > b + 10)


def _is_fill(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.int16).mean(axis=-1) >= FILL_BRIGHTNESS


def _is_badge(rgb: np.ndarray, ally: bool) -> np.ndarray:
    """The LEVEL BADGE beside the bar -- the reliable anchor.

    Searching for the bar directly does not work. It is a ~40x6 strip of low
    contrast in a cluttered scene, its offset above the sprite varies with unit
    height, and neighbouring units' bars merge with it. Measured that way:
    44% of units found a bar at all, widths came out at a mean of 32 against a
    true 38-56, and most fractions pinned at 0.00.

    The badge does not have those problems. It is a saturated rounded rect
    outlined in bright cyan (77,187,238) for ally and bright magenta for enemy,
    it is the same size for every unit, and it stayed legible even in the most
    crowded frame examined -- where the sprites underneath were an
    unrecognisable pile. The bar is then simply what lies to its right, on its
    own rows.
    """
    rgb = rgb.astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if ally:
        return (b > 190) & (g > 140) & (r < 145) & (b - r > 80)
    return (r > 185) & (g < 115) & (b > 85) & (r - g > 95)


def read_unit_hp(frame, bbox, ally: bool) -> UnitHp:
    """HP fraction for one detected unit.

    `frame` is the NATIVE-resolution RGB frame (not the 368x652 detector
    frame -- the bar is only ~6 px tall there, and throwing away resolution on
    the one measurement that needs it would be perverse). `bbox` must be in the
    same coordinate space as `frame`.
    """
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"expected an HxWx3 RGB frame, got shape {arr.shape}")
    arr = arr[..., :3]
    height, width = arr.shape[:2]

    left, top, right, bottom = (int(v) for v in bbox)
    box_h = max(1, bottom - top)
    above = int(np.clip(box_h * SEARCH_ABOVE_FRACTION,
                        SEARCH_ABOVE_MIN_PX, SEARCH_ABOVE_MAX_PX))

    y0, y1 = max(0, top - above), min(height, top + 6)
    x0, x1 = max(0, left - SEARCH_PAD_X), min(width, right + SEARCH_PAD_X)
    if y1 - y0 < 3 or x1 - x0 < MIN_BAR_WIDTH:
        return UnitHp(1.0, bar_found=False)

    window = arr[y0:y1, x0:x1]
    badge = _is_badge(window, ally)
    if badge.sum() < 8:
        # No badge, so no unit UI here at all. Every live unit carries one, so
        # this means the window missed -- not that the unit is undamaged.
        return UnitHp(1.0, bar_found=False)

    # Badge column: the one nearest this unit's centre, so a neighbour's badge
    # in the same window does not capture us.
    unit_centre = (left + right) / 2.0 - x0
    cols = np.flatnonzero(badge.any(axis=0))
    anchor = int(cols[np.argmin(np.abs(cols - unit_centre))])
    # Widen to the whole badge blob around that column.
    keep = cols[np.abs(cols - anchor) <= 18]
    badge_right = int(keep.max())
    rows = np.flatnonzero(badge[:, keep].any(axis=1))
    row_lo, row_hi = int(rows.min()), int(rows.max())

    # The bar occupies the badge's own rows, starting just past its right edge.
    band = window[row_lo:row_hi + 1, badge_right + 1:]
    if band.shape[1] < MIN_BAR_WIDTH:
        return UnitHp(1.0, bar_found=False)

    track = _is_track(band, ally)
    fill = _is_fill(band)
    # Best row of the bar: most bar-like pixels, counting both states.
    scores = (track | fill).sum(axis=1)
    row = int(np.argmax(scores))
    runs = [r for r in _runs((track | fill)[row]) if r[0] <= 3]
    runs = [r for r in runs if MIN_BAR_WIDTH <= r[1] - r[0] <= MAX_BAR_WIDTH]
    if not runs:
        return UnitHp(1.0, bar_found=False)

    start, end = runs[0]
    segment = band[row, start:end]
    total = end - start
    filled = int(_is_fill(segment).sum())
    return UnitHp(fraction=min(1.0, filled / total), bar_found=True, width=total)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True spans of a 1-D mask, as [start, end) pairs."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))
