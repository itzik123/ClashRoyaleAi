"""Per-unit HP and per-unit TEAM, read from the bar above a detected unit.

STATUS: FITTED AND MEASURED AGAINST 60 HAND-LABELLED CROPS.
-----------------------------------------------------------
Ground truth now exists: 60 crops sampled across the three outcomes the code
can produce, labelled damaged / undamaged / not-a-unit by hand. Scored as an
"is this unit damaged" detector over 226 detections from 71 in-game ladder
frames, reweighted from the stratified sample back to the population:

                          precision   recall    F1
    before                    0.79      0.34    0.48
    after                     0.98      0.56    0.72

Three separate faults were behind the old numbers, and they were found in this
order -- each one had to be fixed before the next became visible:

  1. the ASSOCIATION window was inverted (see below), which was most of it;
  2. the FILL test was a brightness threshold, i.e. an accidental ally
     detector, so every enemy unit measured 0%;
  3. two units could claim the SAME badge, and two stacked widgets merged into
     one blob that the size filter discarded whole.

The remaining recall gap is badge DETECTION, not association: 8 of the 10
surviving misses have no badge found anywhere near the unit. That is the next
lever, and one concrete case is recorded -- crop #64 has a plainly visible bar
that `_bar_beside` reports as not-found.

Note what recall does NOT mean here. Clash Royale draws no widget at all over
an undamaged unit, confirmed by eye on the labels, so most unmatched units are
correctly undamaged; 12 of the 20 sampled "no badge" crops were labelled
undamaged. Recall is over units that are genuinely damaged.

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

THE WIDGET AND WHERE IT SITS
----------------------------
Clash Royale draws one widget per live unit: a square level BADGE with a white
digit, and an HP BAR extending to its right.

    [BADGE 5][=====fill=====|-----track-----]

The pair is centred over the unit, so the badge -- being the left end -- sits
LEFT of the unit's own centre, and low enough to overlap the top of the
detector's box. Measured over 15 units whose badge was confirmed by eye:

    dx = badge centre x - bbox centre x     mean -19.6, range [-24.5, -10.0]
    dy = bbox top       - badge centre y    mean  -6.0, range [-14.5,  +1.5]

Negative `dy` means the badge centre is BELOW the box top. That is the normal
case, not an anomaly, and getting it backwards is what broke the first version:
it rejected any badge more than 12 px below the top -- which threw away correct
matches at dy -13 to -15 -- while accepting badges up to 104 px ABOVE, which
belong to other units entirely. Every false positive in the labelled set came
in that way. The window below is the measured cluster widened to ~2.5 sigma.

THE FILL TEST IS PER-TEAM, NOT BRIGHTNESS
-----------------------------------------
An earlier version called a pixel "fill" if its mean channel value was >= 185.
That is very nearly a test for "is this the ALLY bar":

    ALLY_HP_LHS_COLOUR  (111, 208, 252)   mean 190.3  -- passes, barely
    ENEMY_HP_LHS_COLOUR (224,  35,  93)   mean 117.3  -- fails, always

So every enemy unit measured 0% fill, and enemy bars came back either 0.00 or
not-found. Fill and track are now decided by nearest colour against CRBAB's own
four constants, which is symmetric across teams by construction.

THE SIDE ORACLE DID NOT SURVIVE MEASUREMENT
-------------------------------------------
The badge hue states the team in the game's own UI, so the plan was to let it
override CRBAB's learned `side.onnx`. An earlier reading put the disagreement
at 31% with the badge right in every case checked. That figure does not hold:
it was measured with the broken matcher, so it was largely counting bad
ASSOCIATION rather than bad `side.onnx`.

Re-measured after the fixes above, over 67 associated detections, the two
disagree on **10%** (7 cases). Five were checked by eye:

    #64, #225   badge RIGHT -- our own Valkyrie reported as an enemy
    #108        badge WRONG -- an undamaged Mini P.E.K.K.A has no widget of
                its own, so the matcher took the neighbouring enemy badge
    #194        badge WRONG -- a false badge on a Giant's foot
    #42         junk either way -- the detection is not a unit

Two right, two wrong. Neither source dominates, so nothing is overridden:
`side_from_badge` is exposed and `adapter.py` carries its answer beside the
detector's in `team_from_badge`, lowering confidence and raising a flag when
they differ. A side error is worse than a missing HP -- it writes the unit into
the opponent's channels 4-7 instead of ours 0-3 -- which is the reason to
surface the disagreement rather than pick a winner on this evidence.

AN ABSENT BAR MEANS FULL HP
---------------------------
Clash Royale draws no bar over an undamaged unit. Same convention as the King
Tower (see king_hp.py) and the opposite of a Princess tower, where an absent
bar means destroyed. `UnitHp.bar_found` keeps "certainly full" distinguishable
from "measured full", so a miss in a crowd stays visible rather than silently
becoming 1.0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The bar's two colours per team, taken from CRBAB rather than copied. These
# are the single source; re-typing them here is how the fill test drifted.
from clashroyalebuildabot.constants import (
    ALLY_HP_LHS_COLOUR,
    ALLY_HP_RHS_COLOUR,
    ENEMY_HP_LHS_COLOUR,
    ENEMY_HP_RHS_COLOUR,
)

# Every pixel constant below was measured at this frame width. `scale` in
# `find_badges` / `match_badge` converts to whatever the live frame is.
CALIBRATION_WIDTH = 720

# A bar must fall in this width range. The MAXIMUM matters as much as the
# minimum -- without it, two adjacent units' bars merge into one 89-px run and
# the fill of one is divided by the width of both.
MIN_BAR_WIDTH = 20
MAX_BAR_WIDTH = 72

# Badge geometry, measured over 179 harvested badges: ally 24x21, enemy 20x21.
BADGE_MIN_SIDE, BADGE_MAX_SIDE = 9, 26
BADGE_MIN_AREA = 70
BADGE_MIN_DIGIT_PIXELS = 8

# Shortest vertical run that can only be a badge. The badge is ~21 px tall in a
# single column; the bar beside it is ~7. Used to pull two stacked widgets
# apart -- see _split_tall_blob.
BADGE_MIN_RUN = 15
# Width of the white level digit, i.e. how far apart the badge's two surviving
# stripes are after that run filter.
BADGE_DIGIT_GAP = 21

# Badge hue in OpenCV's 0-179 scale, measured over those same 179 badges
# (79 ally, 100 enemy):
#
#   ally    hue median 101, p5 100, p95 105     sat median 173
#   enemy   hue median 171, wrapping past 179   sat median 150
#
# An earlier version of this was calibrated against a SINGLE ally badge with
# the enemy values invented, and scored 11% recall.
ALLY_HUE = (94, 113)
ENEMY_HUE_LOW = (158, 179)
ENEMY_HUE_HIGH = (0, 8)      # magenta wraps around 0
BADGE_MIN_SAT = 110
BADGE_MIN_VAL = 100

# Association window, fitted to 15 confirmed badge/unit pairs (see the module
# docstring). Offsets are badge centre relative to the bbox.
MATCH_DX = (-32.0, -7.0)     # badge centre x - bbox centre x
MATCH_DY = (-28.0, +8.0)     # bbox top - badge centre y
MATCH_DX_CENTRE = -19.6
MATCH_DY_CENTRE = -6.0

# How far a pixel may sit from a bar colour, in RGB euclidean distance, and
# still count as that colour. The two colours of a team are 200+ apart, so this
# is nowhere near tight enough to confuse them; it exists to reject grass.
COLOUR_TOLERANCE = 70


@dataclass(frozen=True)
class UnitHp:
    fraction: float
    bar_found: bool
    """False means no bar was drawn, i.e. the unit is undamaged."""

    width: int = 0
    ally: bool | None = None
    """Team as read from the badge hue, or None when no badge was matched.
    Authoritative when present -- see the module docstring."""


def _bar_colours(ally: bool) -> tuple[np.ndarray, np.ndarray]:
    fill = ALLY_HP_LHS_COLOUR if ally else ENEMY_HP_LHS_COLOUR
    track = ALLY_HP_RHS_COLOUR if ally else ENEMY_HP_RHS_COLOUR
    return np.array(fill, np.int32), np.array(track, np.int32)


def _bar_classes(rgb: np.ndarray, ally: bool) -> tuple[np.ndarray, np.ndarray]:
    """(track, fill) masks, by nearest bar colour.

    Nearest-colour rather than a threshold per channel, because the two states
    differ per team in opposite directions -- the ally bar fills BRIGHTER than
    its track, the enemy bar fills more SATURATED but no brighter. Any single
    brightness rule is therefore a team detector, which is what the previous
    version accidentally was.
    """
    fill_c, track_c = _bar_colours(ally)
    px = rgb.astype(np.int32)
    d_fill = ((px - fill_c) ** 2).sum(axis=-1)
    d_track = ((px - track_c) ** 2).sum(axis=-1)
    tol = COLOUR_TOLERANCE ** 2
    fill = (d_fill <= tol) & (d_fill < d_track)
    track = (d_track <= tol) & (d_track <= d_fill)
    return track, fill


def _badge_mask(rgb_window: np.ndarray, ally: bool) -> np.ndarray:
    """The LEVEL BADGE beside the bar -- the reliable anchor.

    Searching for the bar directly does not work. It is a ~40x6 strip of low
    contrast in a cluttered scene, its offset above the sprite varies with unit
    height, and neighbouring units' bars merge with it. Measured that way: 44%
    of units found a bar at all, widths came out at a mean of 32 against a true
    38-56, and most fractions pinned at 0.00.

    The badge has none of those problems. It is a saturated rounded rect with a
    white level digit, the same size for every unit, and it stayed legible even
    in the most crowded frame examined.
    """
    import cv2  # noqa: PLC0415

    hsv = cv2.cvtColor(rgb_window, cv2.COLOR_RGB2HSV)
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
        """Badge centre. The unit it belongs to is below and to the right."""
        return self.x + self.w / 2.0, self.y + self.h / 2.0


def find_badges(frame) -> list[Badge]:
    """Every level badge in a frame, scanned once.

    Scanning the frame and MATCHING to units beats searching a window per unit.
    Measured: harvesting found 2.5 badges per frame while a per-unit search
    associated only 0.67, so the badge test was never the problem -- the window
    was. One scan is also cheaper than N windows.

    `frame` is the NATIVE-resolution RGB frame, not the 368x652 detector frame:
    the bar is only ~6 px tall there, and throwing away resolution on the one
    measurement that needs it would be perverse.
    """
    import cv2  # noqa: PLC0415

    arr = np.asarray(frame)[..., :3]
    scale = arr.shape[1] / CALIBRATION_WIDTH
    out: list[Badge] = []
    for ally in (True, False):
        mask = _badge_mask(arr, ally).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        # Pixels belonging to a vertical run at least BADGE_MIN_RUN long.
        # Opening with a vertical line is exactly that test. Used only to
        # rescue oversized blobs below -- see _split_tall_blob.
        tall = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN,
            np.ones((max(3, int(BADGE_MIN_RUN * scale)), 1), np.uint8))
        count, labels, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, count):
            x, y, w, h, area = stats[i]
            if area < BADGE_MIN_AREA * scale ** 2:
                continue
            if BADGE_MIN_SIDE * scale <= h <= BADGE_MAX_SIDE * scale:
                boxes = [(int(x), int(y), int(w), int(h))]
            else:
                boxes = _split_tall_blob(tall, labels, i, x, y, w, h, scale)
            for bx, by, bw_full, bh in boxes:
                # An enemy badge merges with its pink bar into one wide blob;
                # the badge is the left end. The ally badge merges too -- its
                # bar's track converts to a hue just inside the ally band.
                bw = min(bw_full, bh + int(4 * scale))
                patch = arr[by:by + bh, bx:bx + bw]
                if patch.size == 0:
                    continue
                grey = patch.astype(np.int16).mean(axis=-1)
                if int((grey >= 225).sum()) < BADGE_MIN_DIGIT_PIXELS:
                    continue      # no white level digit -> not a badge
                hp, found = _bar_beside(arr, bx + bw, by, bh, ally, scale)
                out.append(Badge(bx, by, bw, bh, ally, hp, found))
    return out


def _split_tall_blob(tall, labels, blob_id: int, x: int, y: int, w: int,
                     h: int, scale: float) -> list[tuple[int, int, int, int]]:
    """Badge boxes inside a blob too tall to be a single badge.

    Two units' widgets that stack within ~15 px merge into one component, and
    rejecting it whole was throwing away both badges -- 8 of the 12 units still
    missed after the match window was fixed failed exactly this way, including
    a Valkyrie whose badge is plainly visible in its crop.

    Neither blob size nor a per-column pixel count can separate them: stacked,
    a column holds a badge's 21 px AND a bar's 7. The longest vertical RUN can,
    because it stays 21 for the badge and 7 for the bar however they overlap.
    """
    import cv2  # noqa: PLC0415

    sub = ((labels[y:y + h, x:x + w] == blob_id) &
           (tall[y:y + h, x:x + w] > 0)).astype(np.uint8)
    if not sub.any():
        return []
    # The white level digit splits a badge into a left and a right stripe;
    # close horizontally to rejoin them into one box per badge.
    sub = cv2.morphologyEx(
        sub, cv2.MORPH_CLOSE,
        np.ones((1, max(3, int(BADGE_DIGIT_GAP * scale))), np.uint8))
    n, _lab, st, _c = cv2.connectedComponentsWithStats(sub, 8)
    boxes = []
    for j in range(1, n):
        sx, sy, sw, sh, sa = st[j]
        if sa < BADGE_MIN_AREA * scale ** 2 * 0.5:
            continue
        if not (BADGE_MIN_SIDE * scale <= sh <= BADGE_MAX_SIDE * scale):
            continue
        boxes.append((int(x + sx), int(y + sy), int(sw), int(sh)))
    return boxes


def _bar_beside(arr, x_from: int, y: int, h: int, ally: bool,
                scale: float = 1.0) -> tuple[float, bool]:
    """Read the HP bar immediately right of a badge, on the badge's own rows."""
    max_w = int(MAX_BAR_WIDTH * scale)
    min_w = int(MIN_BAR_WIDTH * scale)
    band = arr[y:y + h, x_from + 1:x_from + 1 + max_w + 6]
    if band.shape[1] < min_w:
        return 1.0, False
    track, fill = _bar_classes(band, ally)
    bar = track | fill
    # Best row of the bar: most bar-like pixels, counting both states. The
    # bright highlight along the bar's top edge scores low here and is skipped.
    row = int(np.argmax(bar.sum(axis=1)))
    runs = [r for r in _runs(bar[row]) if r[0] <= 3]
    runs = [r for r in runs if min_w <= r[1] - r[0] <= max_w]
    if not runs:
        return 1.0, False
    start, end = runs[0]
    filled = int(fill[row, start:end].sum())
    return min(1.0, filled / (end - start)), True


def match_badge(badges: list[Badge], bbox, scale: float = 1.0) -> Badge | None:
    """The badge belonging to a detected unit, or None.

    Matched inside the measured offset window rather than by nearest-anything.
    A crowd puts several badges within any generous radius, and a wrong
    association gives a confident HP and a confident TEAM for the wrong unit --
    worse than admitting ignorance, so this returns None instead of a
    best-effort guess.
    """
    left, top, right, _bottom = (float(v) for v in bbox)
    centre_x = (left + right) / 2.0
    best, best_cost = None, float("inf")
    for badge in badges:
        bx, by = badge.anchor
        dx = (bx - centre_x) / scale
        dy = (top - by) / scale
        if not (MATCH_DX[0] <= dx <= MATCH_DX[1]):
            continue
        if not (MATCH_DY[0] <= dy <= MATCH_DY[1]):
            continue
        cost = abs(dx - MATCH_DX_CENTRE) + abs(dy - MATCH_DY_CENTRE)
        if cost < best_cost:
            best, best_cost = badge, cost
    return best


def match_badges(badges: list[Badge], bboxes, scale: float = 1.0) -> list[Badge | None]:
    """Assign badges to a whole frame's units at once, each badge used once.

    Prefer this over calling `match_badge` per unit. Independently, two units
    can both pick the same badge -- two Giants stacked in a lane did exactly
    that, and the second one then reported the first one's HP and the first
    one's TEAM. Assigning greedily by cost, cheapest pair first, makes that
    impossible and costs nothing: the loser falls through to its own badge if
    it has one, or to None if it does not.
    """
    pairs = []
    for ui, bbox in enumerate(bboxes):
        left, top, right, _bottom = (float(v) for v in bbox)
        centre_x = (left + right) / 2.0
        for bi, badge in enumerate(badges):
            bx, by = badge.anchor
            dx, dy = (bx - centre_x) / scale, (top - by) / scale
            if not (MATCH_DX[0] <= dx <= MATCH_DX[1]):
                continue
            if not (MATCH_DY[0] <= dy <= MATCH_DY[1]):
                continue
            cost = abs(dx - MATCH_DX_CENTRE) + abs(dy - MATCH_DY_CENTRE)
            pairs.append((cost, ui, bi))
    pairs.sort()
    out: list[Badge | None] = [None] * len(list(bboxes))
    used_u: set[int] = set()
    used_b: set[int] = set()
    for _cost, ui, bi in pairs:
        if ui in used_u or bi in used_b:
            continue
        out[ui] = badges[bi]
        used_u.add(ui)
        used_b.add(bi)
    return out


def read_unit_hp(frame_badges: list[Badge], bbox, scale: float = 1.0) -> UnitHp:
    """HP and team for one detected unit, from a frame's pre-scanned badges.

    Takes the badge list rather than the frame so a caller scans once per frame
    and matches N times, which is the ordering that made this work at all. For
    a whole frame prefer `match_badges`, which additionally stops two units
    claiming the same badge.
    """
    badge = match_badge(frame_badges, bbox, scale)
    if badge is None:
        return UnitHp(1.0, bar_found=False, ally=None)
    return UnitHp(fraction=badge.hp, bar_found=badge.bar_found, ally=badge.ally)


def side_from_badge(frame_badges: list[Badge], bbox,
                    scale: float = 1.0) -> bool | None:
    """True for ally, False for enemy, None when no badge could be matched.

    Authoritative over CRBAB's `side.onnx` when it returns a value -- the badge
    is the game's own UI stating the team, while `side.onnx` is a 16x16 learned
    guess off the sprite crop. See the module docstring for the measured
    disagreement rate.
    """
    badge = match_badge(frame_badges, bbox, scale)
    return None if badge is None else badge.ally


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True spans of a 1-D mask, as [start, end) pairs."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))
