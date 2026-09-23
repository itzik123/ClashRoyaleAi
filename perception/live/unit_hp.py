"""Per-unit HP and team, read from the widget above a detected unit.

CRBAB reports no unit HP, but observation channels 0-7 store `hp / MAX` per
cell, a large share of the policy's input; a constant there would be fabricated
data, so it is read. The detector supplies where each unit is; this looks just
above that box.

The widget is a square level badge with a white digit, and an HP bar to its
right:

    [BADGE 5][=====fill=====|-----track-----]

The pair is centred over the unit, so the badge sits left of the unit's centre
and low enough to overlap the top of the detector's box. Measured over 15
confirmed pairs:

    dx = badge centre x - bbox centre x     mean -19.6, range [-24.5, -10.0]
    dy = bbox top       - badge centre y    mean  -6.0, range [-14.5,  +1.5]

Negative dy (badge centre below the box top) is the normal case. The match
window is that cluster widened to ~2.5 sigma.

Fill and track are decided by nearest colour against CRBAB's four bar
constants, which is symmetric across teams; a brightness threshold is in effect
an ally detector.

The badge's hue states the team. It is not allowed to override CRBAB's
`side.onnx`: the two disagree on ~10% of associated detections and neither was
consistently right when checked by eye. `adapter.py` carries both
(`team_from_badge`) and flags disagreement. A side error writes the unit into
the wrong team's channels, which is worse than a missing HP.

An absent bar means full HP (as for the King Tower, and unlike a Princess
Tower, where it means destroyed). `UnitHp.bar_found` keeps "certainly full"
distinct from "measured full", so a miss in a crowd stays visible.

Scored as an "is this unit damaged" detector on 226 detections from 71 ladder
frames (60 hand-labelled crops, reweighted): precision 0.98, recall 0.56. Most
unmatched units are correctly undamaged, since no widget is drawn over an
undamaged unit; the remaining misses are badge detection, not association.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The bar's colours per team, from CRBAB: the single source.
from clashroyalebuildabot.constants import (
    ALLY_HP_LHS_COLOUR,
    ALLY_HP_RHS_COLOUR,
    ENEMY_HP_LHS_COLOUR,
    ENEMY_HP_RHS_COLOUR,
)

# Every pixel constant below was measured at this frame width; `scale` in
# `find_badges` / `match_badge` converts.
CALIBRATION_WIDTH = 720

# A bar's width range. The maximum matters: without it two adjacent bars merge
# into one run and one fill is divided by both widths.
MIN_BAR_WIDTH = 20
MAX_BAR_WIDTH = 72

# Badge geometry over 179 harvested badges: ally 24x21, enemy 20x21.
BADGE_MIN_SIDE, BADGE_MAX_SIDE = 9, 26
BADGE_MIN_AREA = 70
BADGE_MIN_DIGIT_PIXELS = 8

# Shortest vertical run that can only be a badge (~21 px tall; the bar is ~7).
# Used to pull stacked widgets apart; see _split_tall_blob.
BADGE_MIN_RUN = 15
# Width of the white level digit: the gap between the badge's two surviving
# stripes after the run filter.
BADGE_DIGIT_GAP = 21

# Badge hue on OpenCV's 0-179 scale, over the same 179 badges (79 ally, 100
# enemy):
#
#   ally    hue median 101, p5 100, p95 105     sat median 173
#   enemy   hue median 171, wrapping past 179   sat median 150
ALLY_HUE = (94, 113)
ENEMY_HUE_LOW = (158, 179)
ENEMY_HUE_HIGH = (0, 8)      # magenta wraps around 0
BADGE_MIN_SAT = 110
BADGE_MIN_VAL = 100

# Association window from the 15 confirmed pairs (module docstring); badge
# centre relative to the bbox.
MATCH_DX = (-32.0, -7.0)     # badge centre x - bbox centre x
MATCH_DY = (-28.0, +8.0)     # bbox top - badge centre y
MATCH_DX_CENTRE = -19.6
MATCH_DY_CENTRE = -6.0

# RGB distance within which a pixel counts as a bar colour. A team's two
# colours are 200+ apart; this exists to reject grass.
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

    Not a per-channel threshold: the ally bar fills brighter than its track and
    the enemy bar more saturated but no brighter, so any single brightness rule
    is a team detector.
    """
    fill_c, track_c = _bar_colours(ally)
    px = rgb.astype(np.int32)
    d_fill = ((px - fill_c) ** 2).sum(axis=-1)
    d_track = ((px - track_c) ** 2).sum(axis=-1)
    tol = COLOUR_TOLERANCE ** 2
    fill = (d_fill <= tol) & (d_fill < d_track)
    track = (d_track <= tol) & (d_track <= d_fill)
    return track, fill


def _badge_mask(rgb_window: np.ndarray, ally: bool,
                hsv: np.ndarray | None = None) -> np.ndarray:
    """The level badge beside the bar: the reliable anchor.

    The bar itself is a ~40x6 low-contrast strip whose offset varies with unit
    height and which merges with neighbours' bars. The badge is a saturated
    rounded rect with a white digit, the same size for every unit, and legible
    even in crowded frames.
    """
    import cv2  # noqa: PLC0415

    # `hsv` is passed in when the caller already has it: the ally and enemy
    # masks differ only in hue band, and converting twice was a third of
    # find_badges' cost.
    if hsv is None:
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

    Scanning the frame and matching to units finds far more than searching a
    window per unit, and is cheaper. `frame` is the native-resolution RGB
    frame, not the 368x652 detector frame, where the bar is only ~6 px tall.
    """
    import cv2  # noqa: PLC0415

    arr = np.asarray(frame)[..., :3]
    scale = arr.shape[1] / CALIBRATION_WIDTH
    # Converted once and shared: the two passes differ only in hue band.
    hsv = cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_RGB2HSV)
    out: list[Badge] = []
    for ally in (True, False):
        mask = _badge_mask(arr, ally, hsv=hsv).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        # Pixels in a vertical run at least BADGE_MIN_RUN long (an opening with
        # a vertical line). Used only to rescue oversized blobs; see
        # _split_tall_blob.
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
                # A badge merges with its bar into one wide blob (the ally
                # bar's track converts to a hue just inside the ally band); the
                # badge is the left end.
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

    Two widgets stacked within ~15 px merge into one component. Neither blob
    size nor per-column pixel count separates them, since a column then holds a
    badge's 21 px and a bar's 7; the longest vertical run does.
    """
    import cv2  # noqa: PLC0415

    sub = ((labels[y:y + h, x:x + w] == blob_id) &
           (tall[y:y + h, x:x + w] > 0)).astype(np.uint8)
    if not sub.any():
        return []
    # The white digit splits a badge into two stripes; close horizontally to
    # rejoin them.
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
    # Best row of the bar: most bar-like pixels in either state. The bright
    # highlight on the top edge scores low and is skipped.
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

    Matched inside the measured offset window, not by nearest: a crowd puts
    several badges within any generous radius, and a wrong association gives a
    confident HP and team for the wrong unit.
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

    Prefer this to `match_badge` per unit, where two units can pick the same
    badge and the second reports the first's HP and team. Greedy by cost,
    cheapest pair first; the loser falls through to its own badge or None.
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
    """HP and team for one detected unit, from a frame's pre-scanned badges. Takes
    the badge list so a caller scans once and matches N times. For a whole
    frame prefer `match_badges`.
    """
    badge = match_badge(frame_badges, bbox, scale)
    if badge is None:
        return UnitHp(1.0, bar_found=False, ally=None)
    return UnitHp(fraction=badge.hp, bar_found=badge.bar_found, ally=badge.ally)


def side_from_badge(frame_badges: list[Badge], bbox,
                    scale: float = 1.0) -> bool | None:
    """True for ally, False for enemy, None when no badge matched. The badge is
    the game's own UI stating the team; see the module docstring for how often
    it disagrees with `side.onnx`.
    """
    badge = match_badge(frame_badges, bbox, scale)
    return None if badge is None else badge.ally


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True spans of a 1-D mask, as [start, end) pairs."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))
