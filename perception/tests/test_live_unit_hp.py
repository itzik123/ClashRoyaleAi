"""Tests for live/unit_hp.py.

Each pins a bug found against 60 hand-labelled crops. Synthetic pixels rather
than fixture images: the failures were in the decision rules (which badge
belongs to which unit, what counts as fill), so building the triggering pattern
states the rule more clearly than a screenshot.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("clashroyalebuildabot", reason="vendored bot not importable")
pytest.importorskip("cv2", reason="opencv not installed")

from live.unit_hp import (  # noqa: E402
    ALLY_HP_LHS_COLOUR,
    ENEMY_HP_LHS_COLOUR,
    MATCH_DX,
    MATCH_DX_CENTRE,
    MATCH_DY,
    MATCH_DY_CENTRE,
    Badge,
    _bar_classes,
    match_badge,
    match_badges,
    read_unit_hp,
)


def badge_at(cx: float, cy: float, ally: bool = True, hp: float = 0.5,
             bar: bool = True, w: int = 20, h: int = 21) -> Badge:
    """A Badge whose CENTRE is (cx, cy)."""
    return Badge(x=int(cx - w / 2), y=int(cy - h / 2), w=w, h=h,
                 ally=ally, hp=hp, bar_found=bar)


BBOX = (100, 200, 160, 270)          # centre x 130, top 200
CENTRE_X, TOP = 130.0, 200.0


def at_offset(dx: float, dy: float, **kw) -> Badge:
    """A badge at a given (dx, dy) from BBOX, in the module's own convention.
    """
    return badge_at(CENTRE_X + dx, TOP - dy, **kw)


# --- the window ---

def test_badge_below_the_box_top_is_the_normal_case():
    """Over 15 confirmed pairs dy runs -14.5 to +1.5: the badge centre is almost
    always below the box top, because the widget overlaps the sprite's head.
    Rejecting badges more than 12 px below discarded correct matches.
    """
    assert match_badge([at_offset(-20, -14)], BBOX) is not None
    assert match_badge([at_offset(-20, -1)], BBOX) is not None


def test_badge_far_above_the_box_belongs_to_another_unit():
    """A badge far above the box belongs to another unit. A cost of `|dx| +
    0.6*max(0, dy)` against a limit of 70 matched a badge 104 px above (dx
    +5.5, cost 67.9) while rejecting the unit's own badge 14 px below.
    """
    for dy in (38, 60, 81, 104):
        assert match_badge([at_offset(-10, dy)], BBOX) is None


def test_badge_right_of_centre_is_not_ours():
    """The widget is badge-then-bar centred over the unit, so the badge sits left
    of the unit's centre (mean dx -19.6, never positive).
    """
    assert match_badge([at_offset(+20, -6)], BBOX) is None


def test_window_bounds_are_the_measured_cluster():
    """Guards the constants: the fitted cluster must sit inside the window."""
    assert MATCH_DX[0] <= -24.5 and -10.0 <= MATCH_DX[1]
    assert MATCH_DY[0] <= -26.0 and 1.5 <= MATCH_DY[1]
    assert MATCH_DX[0] < MATCH_DX_CENTRE < MATCH_DX[1]
    assert MATCH_DY[0] < MATCH_DY_CENTRE < MATCH_DY[1]


def test_nearest_to_the_window_centre_wins():
    far = at_offset(-31, +7, hp=0.1)
    near = at_offset(-20, -6, hp=0.9)
    assert match_badge([far, near], BBOX) is near


# --- one-to-one ---

def test_two_units_cannot_share_one_badge():
    """Two Giants stacked in a lane must not match the same badge, or the second
    reports the first's HP and team.
    """
    one = (100, 200, 160, 270)
    two = (100, 260, 160, 330)          # directly below, same column
    badge = badge_at(CENTRE_X - 20, TOP + 6)
    got = match_badges([badge], [one, two])
    assert got.count(badge) == 1
    assert None in got


def test_each_unit_still_gets_its_own_badge():
    one = (100, 200, 160, 270)
    two = (300, 200, 360, 270)
    a = badge_at(130 - 20, 200 + 6)
    b = badge_at(330 - 20, 200 + 6)
    assert match_badges([a, b], [one, two]) == [a, b]


def test_match_badges_returns_one_entry_per_unit():
    boxes = [(0, 0, 10, 10)] * 5
    assert len(match_badges([], boxes)) == 5


# --- the per-team fill ---

@pytest.mark.parametrize("ally,fill_colour", [
    (True, ALLY_HP_LHS_COLOUR),
    (False, ENEMY_HP_LHS_COLOUR),
])
def test_fill_is_detected_for_both_teams(ally, fill_colour):
    """A brightness test (`mean(rgb) >= 185`) is an ally detector: ally fill
    (111,208,252) means 190.3 and passes, enemy fill (224,35,93) means 117.3
    and never can, so every enemy measured 0%.
    """
    band = np.tile(np.array(fill_colour, np.uint8), (6, 40, 1))
    _track, fill = _bar_classes(band, ally)
    assert fill.all()


def test_a_team_does_not_see_the_other_teams_fill():
    band = np.tile(np.array(ENEMY_HP_LHS_COLOUR, np.uint8), (6, 40, 1))
    _track, fill = _bar_classes(band, ally=True)
    assert not fill.any()


def test_grass_is_neither_fill_nor_track():
    band = np.tile(np.array((140, 158, 64), np.uint8), (6, 40, 1))
    for ally in (True, False):
        track, fill = _bar_classes(band, ally)
        assert not track.any() and not fill.any()


# --- the contract ---

def test_no_badge_means_undamaged_not_unknown():
    """No widget is drawn over an undamaged unit, so an absent badge is 1.0, but
    `hp_measured` stays False so a miss in a crowd is visible.
    """
    hp = read_unit_hp([], BBOX)
    assert hp.fraction == 1.0
    assert hp.bar_found is False
    assert hp.ally is None


def test_matched_badge_reports_its_team_and_fraction():
    hp = read_unit_hp([at_offset(-20, -6, ally=False, hp=0.42)], BBOX)
    assert hp.ally is False
    assert hp.fraction == pytest.approx(0.42)
    assert hp.bar_found is True
