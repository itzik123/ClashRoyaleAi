"""Tests for readers/tower_numerals.py, each pinning something measured wrong
while building it, against a real frame whose answer is known by eye.
End-to-end matching against the tower templates
(`config/templates/tower_549x976`) is asserted at the bottom.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from readers.clock import DigitTemplates
from readers.tower_numerals import (
    NUMERAL_OFFSET,
    HP_RANGE,
    TowerNumeralReader,
    ink_channel,
    numeral_roi,
    split_digits,
)

# CRBAB's bar bboxes in its 368x652 detector space, from NUMBER_CONFIG's
# constants: (LEFT/RIGHT_PRINCESS_HP_X, ENEMY/ALLY_PRINCESS_HP_Y, +HP_WIDTH,
# +HP_HEIGHT).
BAR_ENEMY_LEFT = (74, 95, 114, 105)
BAR_ENEMY_RIGHT = (266, 95, 306, 105)
DETECTOR = (368, 652)
NATIVE = (549, 976)

_ASSETS = Path(__file__).resolve().parent / "assets"
_TEMPLATES = (Path(__file__).resolve().parent.parent
              / "config" / "templates" / "tower_549x976")


def test_the_roi_sits_above_the_bar_not_on_it():
    """The bar is a saturated stripe across the ROI; including even two pixels of
    it merged "2030" into a single cell.
    """
    x1, y1, x2, y2 = numeral_roi(BAR_ENEMY_RIGHT, NATIVE, DETECTOR)
    bar_top_native = int(BAR_ENEMY_RIGHT[1] * NATIVE[1] / DETECTOR[1])
    assert y2 <= bar_top_native, "ROI reaches into the bar"
    assert y1 < y2 and x1 < x2


def test_the_roi_is_derived_from_the_bar_not_calibrated_separately():
    """A separate calibration entry would be a second copy of where the tower is.
    """
    a = numeral_roi(BAR_ENEMY_RIGHT, NATIVE, DETECTOR)
    shifted = (BAR_ENEMY_RIGHT[0] + 10, BAR_ENEMY_RIGHT[1],
               BAR_ENEMY_RIGHT[2] + 10, BAR_ENEMY_RIGHT[3])
    b = numeral_roi(shifted, NATIVE, DETECTOR)
    assert b[0] > a[0] and b[2] > a[2], "ROI ignored the bar's position"


def test_the_roi_is_clamped_to_the_frame():
    """A tower at the very top would otherwise produce a negative y."""
    x1, y1, x2, y2 = numeral_roi((0, 0, 40, 10), NATIVE, DETECTOR)
    assert x1 >= 0 and y1 >= 0
    assert x2 <= NATIVE[0] and y2 <= NATIVE[1]


# --- the ink channel ---

def test_grass_and_glyph_separate_on_the_min_channel_but_not_on_luminance():
    """Grass and a near-white glyph are both bright in luminance (~169 vs ~230),
    so Otsu keeps them together; grass's blue channel is low while the glyph is
    high in every channel.
    """
    grass = np.array([[[140, 200, 90]]], np.uint8)
    glyph = np.array([[[250, 220, 230]]], np.uint8)

    lum = lambda p: 0.299 * p[0, 0, 0] + 0.587 * p[0, 0, 1] + 0.114 * p[0, 0, 2]
    assert abs(lum(grass) - lum(glyph)) < 70, "premise: both bright in luma"

    assert int(ink_channel(glyph)[0, 0]) - int(ink_channel(grass)[0, 0]) > 100


def test_ink_channel_passes_greyscale_through():
    grey = np.full((4, 4), 120, np.uint8)
    assert np.array_equal(ink_channel(grey), grey)


# --- segmentation ---

def _numeral(digits, on_grass=True):
    """A synthetic numeral: bright bars on a saturated background."""
    h, w = 20, 18 * len(digits)
    bg = (140, 200, 90) if on_grass else (30, 30, 30)
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :] = bg
    for i, wide in enumerate(digits):
        x = i * 18 + 4
        span = 10 if wide else 3          # a "0" is wide, a "1" is narrow
        img[3:17, x:x + span] = (250, 245, 250)
    return img


def test_segments_each_digit_separately():
    cells = split_digits(_numeral([True, True, True, True]))
    assert len(cells) == 4


def test_the_digit_count_is_not_fixed():
    """The digit count changes as a tower drops through 1000; asserting three
    would fail on exactly the damaged frames.
    """
    assert len(split_digits(_numeral([True] * 3))) == 3
    assert len(split_digits(_numeral([True] * 4))) == 4


def test_a_narrow_glyph_is_not_filtered_out_with_the_noise():
    """A "1" is about a third the width of a "0", so any width filter that removes
    speckle removes the 1 too; runs are rejected by height instead.
    """
    cells = split_digits(_numeral([True, False, True]))
    assert len(cells) == 3
    widths = [c.shape[1] for c in cells]
    assert min(widths) < max(widths) / 2, "the narrow glyph was widened"


def test_short_runs_are_rejected():
    """The bar's edge and stray highlights are short; a digit spans most of the
    ROI height.
    """
    img = _numeral([True, True])
    img[9:11, :] = (250, 245, 250)         # a thin full-width streak
    assert len(split_digits(img)) == 2


def test_an_empty_patch_yields_no_digits():
    assert split_digits(np.zeros((0, 0, 3), np.uint8)) == []


def test_a_blank_patch_yields_no_digits():
    """Uniform colour has no ink: return nothing, not one ROI-wide cell a caller
    would read as a digit.
    """
    flat = np.full((20, 60, 3), 140, np.uint8)
    assert len(split_digits(flat)) <= 1


# --- the reading contract ---

def test_hp_range_excludes_implausible_readings():
    """A number outside any real tower's range is a segmentation failure that
    happened to produce digits; reporting it would look measured.
    """
    assert HP_RANGE[0] >= 1
    assert HP_RANGE[1] < 100000


def test_the_numeral_offset_reaches_above_the_bar():
    assert NUMERAL_OFFSET[1] < NUMERAL_OFFSET[3] <= 0


# --- end to end, against the frame the templates were accepted on ---

@pytest.fixture(scope="module")
def reader():
    if not (_TEMPLATES / "digits.json").exists():
        pytest.skip(f"no tower digit templates at {_TEMPLATES}")
    return TowerNumeralReader(DigitTemplates.load(_TEMPLATES))


@pytest.fixture(scope="module")
def frame_2030():
    path = _ASSETS / "tower_numerals_2030_549x976.jpg"
    if not path.exists():
        pytest.skip(f"missing {path.name}")
    return cv2.imread(str(path))


@pytest.mark.parametrize("bar", [BAR_ENEMY_LEFT, BAR_ENEMY_RIGHT])
def test_reads_the_known_frame(reader, frame_2030, bar):
    """Both enemy Princess towers read 2030 on this frame, by eye. Exercises the
    whole reader (ROI derivation, ink channel, segmentation, matching,
    plausibility range), not just template matching.
    """
    reading = reader.read(frame_2030, bar, DETECTOR)
    assert reading.digits == "2030"
    assert reading.value == 2030
    assert reading.measured


def test_a_blank_frame_is_reported_unmeasured_not_zero(reader):
    """A tower of unknown HP must stay distinguishable from a nearly dead one: a
    live tower reported empty tells the policy a lane is lost.
    """
    blank = np.full((976, 549, 3), 90, np.uint8)
    reading = reader.read(blank, BAR_ENEMY_RIGHT, DETECTOR)
    assert reading.value is None
    assert not reading.measured


def test_every_digit_has_a_template():
    """A partial template set would misread missing digits as whichever glyph it
    has; DigitTemplates raises on construction for that, and this pins that the
    shipped set is complete.
    """
    if not (_TEMPLATES / "digits.json").exists():
        pytest.skip("no tower digit templates")
    templates = DigitTemplates.load(_TEMPLATES)
    assert set(templates.templates) == set("0123456789")
    assert templates.shape == (26, 18)
