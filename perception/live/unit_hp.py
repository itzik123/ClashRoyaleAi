"""Per-unit HP, read from the bar above a detected unit.

STATUS: CALIBRATED, STILL UNDER-MATCHING. NOT WIRED IN.
-------------------------------------------------------
Thresholds are now measured rather than guessed -- see ALLY_HUE / ENEMY_HUE_*,
set from 179 badges harvested across 71 in-game ladder frames. Progress by
approach, on the same footage:

    search for the bar directly      44% found, widths mean 32 (true 38-56),
                                     fractions mostly pinned at 0.00
    badge anchor, 1 sample calibr.   widths right, 11% found
    badge anchor, measured thresh.   21% found, widths mean 39
    frame scan + match to units      37% matched, widths 30-54, fractions
                                     mean 0.60 spread 0.25-0.92

The remaining gap is ASSOCIATION, not detection. A frame yields ~2.5 badges and
~1.8 on-board units, yet only 0.67 units get a badge attached -- so the badges
are being found and then not matched. `match_badge`'s geometry is the suspect:
it requires the badge centre above `top + 12` and scores on a hand-weighted
distance, neither of which was fitted to anything.

WHAT IT STILL NEEDS
-------------------
Ground truth, which does not exist yet. King bars could be validated against
the HP NUMERAL printed beside them; units have no numeral, only the bar. So
"no bar found" stays ambiguous between *undamaged* -- correct, Clash Royale
draws no bar on a healthy unit -- and *missed*. Labelled crops of units known
to be damaged are needed before any threshold moves again, or the tuning is
fitted to a metric nobody can see.

A SIDE-EFFECT WORTH MORE THAN THE HP
------------------------------------
The badge hue gives the unit's TEAM directly, and it disagrees with CRBAB's
learned `side.onnx` on **31% of matched units** (11/16 agreed). Three
disagreements were checked by eye and the badge was right every time: our own
Valkyrie and our own Cannon were both reported as enemies. A side error is
worse than a miss -- it writes the unit into the opponent's channels 4-7
instead of ours 0-3 -- so `find_badges` is likely to be more valuable as a
side oracle than as an HP source.

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

# Badge geometry, measured over 179 harvested badges: ally 24x21, enemy 20x21.
BADGE_MIN_SIDE, BADGE_MAX_SIDE = 9, 26
BADGE_MIN_AREA = 70
BADGE_MIN_DIGIT_PIXELS = 8

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


# Badge hue in OpenCV's 0-179 scale, MEASURED over 179 badges harvested from
# 71 in-game ladder frames (79 ally, 100 enemy):
#
#   ally    hue median 101, p5 100, p95 105     sat median 173
#   enemy   hue median 171, wrapping past 179   sat median 150
#
# The bounds below are those distributions widened, not guesses. An earlier
# version of this function was calibrated against a SINGLE ally badge with the
# enemy values invented, and scored 11% recall.
ALLY_HUE = (94, 113)
ENEMY_HUE_LOW = (158, 179)
ENEMY_HUE_HIGH = (0, 8)      # magenta wraps around 0
BADGE_MIN_SAT = 110
BADGE_MIN_VAL = 100


def _badge_mask(bgr_window: np.ndarray, ally: bool) -> np.ndarray:
    """The LEVEL BADGE beside the bar -- the reliable anchor.

    Searching for the bar directly does not work. It is a ~40x6 strip of low
    contrast in a cluttered scene, its offset above the sprite varies with unit
    height, and neighbouring units' bars merge with it. Measured that way: 44%
    of units found a bar at all, widths came out at a mean of 32 against a true
    38-56, and most fractions pinned at 0.00.

    The badge has none of those problems. It is a saturated rounded rect with a
    white level digit, the same size for every unit, and it stayed legible even
    in the most crowded frame examined -- where the sprites underneath were an
    unrecognisable pile.
    """
    import cv2  # noqa: PLC0415

    hsv = cv2.cvtColor(bgr_window, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if ally:
        hue = (h >= ALLY_HUE[0]) & (h <= ALLY_HUE[1])
    else:
        hue = ((h >= ENEMY_HUE_LOW[0]) & (h <= ENEMY_HUE_LOW[1])) | \
              ((h >= ENEMY_HUE_HIGH[0]) & (h <= ENEMY_HUE_HIGH[1]))
    return hue & (s > BADGE_MIN_SAT) & (v > BADGE_MIN_VAL)


@dataclass(frozen=True)
class Badge:
    """One level badge found in a frame, with the bar beside it."""

    x: int
    y: int
    w: int
    h: int
    ally: bool
    hp: float
    bar_found: bool

    @property
    def anchor(self) -> tuple[float, float]:
        """Badge centre. A unit's badge sits above and slightly left of it."""
        return self.x + self.w / 2.0, self.y + self.h / 2.0


def find_badges(frame) -> list[Badge]:
    """Every level badge in a frame, scanned once.

    Scanning the frame and MATCHING to units beats searching a window per unit,
    which is what the per-unit version did. Measured: harvesting found 2.5
    badges per frame while the per-unit search associated only 0.67, so the
    badge test was never the problem -- the window was. One scan is also
    cheaper than N windows.
    """
    import cv2  # noqa: PLC0415

    arr = np.asarray(frame)[..., :3]
    out: list[Badge] = []
    for ally in (True, False):
        mask = _badge_mask(arr, ally).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, _lab, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, count):
            x, y, w, h, area = stats[i]
            if area < BADGE_MIN_AREA or not (BADGE_MIN_SIDE <= h <= BADGE_MAX_SIDE):
                continue
            # An enemy badge merges with its pink bar into one wide blob; the
            # badge is the left end. The ally badge stays square because its
            # bar fill is white, not cyan.
            bw = min(int(w), int(h) + 4)
            patch = arr[y:y + h, x:x + bw]
            grey = patch.astype(np.int16).mean(axis=-1)
            if int((grey >= 225).sum()) < BADGE_MIN_DIGIT_PIXELS:
                continue          # no white level digit -> not a badge
            hp, found = _bar_beside(arr, int(x + bw), int(y), int(h), ally)
            out.append(Badge(int(x), int(y), bw, int(h), ally, hp, found))
    return out


def _bar_beside(arr, x_from: int, y: int, h: int, ally: bool) -> tuple[float, bool]:
    """Read the HP bar immediately right of a badge, on the badge's own rows."""
    band = arr[y:y + h, x_from + 1:x_from + 1 + MAX_BAR_WIDTH + 6]
    if band.shape[1] < MIN_BAR_WIDTH:
        return 1.0, False
    track, fill = _is_track(band, ally), _is_fill(band)
    scores = (track | fill).sum(axis=1)
    row = int(np.argmax(scores))
    runs = [r for r in _runs((track | fill)[row]) if r[0] <= 3]
    runs = [r for r in runs if MIN_BAR_WIDTH <= r[1] - r[0] <= MAX_BAR_WIDTH]
    if not runs:
        return 1.0, False
    start, end = runs[0]
    segment = band[row, start:end]
    return min(1.0, int(_is_fill(segment).sum()) / (end - start)), True


def match_badge(badges: list[Badge], bbox, max_distance: float = 70.0) -> Badge | None:
    """The badge belonging to a detected unit, or None.

    Matched on the badge sitting ABOVE the unit -- Clash Royale draws it over
    the sprite's head -- and nearest horizontally. Returns None rather than a
    far-away badge, because in a crowd a wrong association gives a confident
    HP and side for the wrong unit, which is worse than admitting ignorance.
    """
    left, top, right, _bottom = (int(v) for v in bbox)
    centre_x = (left + right) / 2.0
    best, best_d = None, max_distance
    for badge in badges:
        bx, by = badge.anchor
        if by > top + 12:                 # must be above the sprite
            continue
        d = abs(bx - centre_x) + 0.6 * max(0.0, top - by)
        if d < best_d:
            best, best_d = badge, d
    return best


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
    badge = _badge_mask(window, ally)
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
