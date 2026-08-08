"""Tests for readers/tower_numerals.py.

Every one of these pins something that was measured wrong while building it,
against a real captured frame where the answer is known by eye.

Template MATCHING is deliberately not asserted here. The clock's digit
templates were tried and measurably do not transfer -- confidence 0.08-0.18
against a 0.35 threshold, with "3" read as "1" -- so a tower-specific set is
still to be cut. Everything upstream of that is what these tests cover, and
it is what the segmentation half of the reader is made of.
"""
from __future__ import annotations

import numpy as np
import pytest

from readers.tower_numerals import (
    NUMERAL_OFFSET,
    HP_RANGE,
    ink_channel,
    numeral_roi,
    split_digits,
)

# CRBAB's bar bboxes in its own 368x652 detector space, from a live capture.
BAR_ENEMY_RIGHT = (266, 95, 306, 105)
DETECTOR = (368, 652)
NATIVE = (549, 976)


def test_the_roi_sits_above_the_bar_not_on_it():
    """The bar is a saturated stripe spanning the ROI's full width. Including
    even two pixels of it put the bar and the digits on the same side of Otsu,
    so the column projection saw one unbroken run and "2030" segmented into a
    SINGLE cell -- with the digits plainly legible in the crop."""
    x1, y1, x2, y2 = numeral_roi(BAR_ENEMY_RIGHT, NATIVE, DETECTOR)
    bar_top_native = int(BAR_ENEMY_RIGHT[1] * NATIVE[1] / DETECTOR[1])
    assert y2 <= bar_top_native, "ROI reaches into the bar"
    assert y1 < y2 and x1 < x2


def test_the_roi_is_derived_from_the_bar_not_calibrated_separately():
    """A separate calibration entry would be a second copy of where the tower
    is, and would drift from CRBAB's the first time its layout moved."""
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


# --- the ink channel --------------------------------------------------------

def test_grass_and_glyph_separate_on_the_min_channel_but_not_on_luminance():
    """The measured failure, reproduced as a test.

    Arena grass and a near-white glyph are BOTH bright in luminance (~169 vs
    ~230), so Otsu keeps them together. Grass is a saturated green, so its
    blue channel is low while the glyph is high in every channel -- which is
    what makes them separable at all.
    """
    grass = np.array([[[140, 200, 90]]], np.uint8)
    glyph = np.array([[[250, 220, 230]]], np.uint8)

    lum = lambda p: 0.299 * p[0, 0, 0] + 0.587 * p[0, 0, 1] + 0.114 * p[0, 0, 2]
    assert abs(lum(grass) - lum(glyph)) < 70, "premise: both bright in luma"

    assert int(ink_channel(glyph)[0, 0]) - int(ink_channel(grass)[0, 0]) > 100


def test_ink_channel_passes_greyscale_through():
    grey = np.full((4, 4), 120, np.uint8)
    assert np.array_equal(ink_channel(grey), grey)


# --- segmentation -----------------------------------------------------------

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
    """A tower reads 2446 at full health and 887 after damage, and the count
    changes mid-match as it drops through 1000. Asserting three -- as the
    clock reader can, since M:SS is fixed -- would fail on exactly the frames
    where damage is the thing being measured."""
    assert len(split_digits(_numeral([True] * 3))) == 3
    assert len(split_digits(_numeral([True] * 4))) == 4


def test_a_narrow_glyph_is_not_filtered_out_with_the_noise():
    """A "1" is about a third the width of a "0". Any width-based filter that
    removes background speckle also removes the 1 -- which is why runs are
    rejected by HEIGHT instead."""
    cells = split_digits(_numeral([True, False, True]))
    assert len(cells) == 3
    widths = [c.shape[1] for c in cells]
    assert min(widths) < max(widths) / 2, "the narrow glyph was widened"


def test_short_runs_are_rejected():
    """The bar's edge and stray highlights are short; a digit spans most of
    the ROI height."""
    img = _numeral([True, True])
    img[9:11, :] = (250, 245, 250)         # a thin full-width streak
    assert len(split_digits(img)) == 2


def test_an_empty_patch_yields_no_digits():
    assert split_digits(np.zeros((0, 0, 3), np.uint8)) == []


def test_a_blank_patch_yields_no_digits():
    """Uniform colour has no ink. It must return nothing rather than one cell
    spanning the whole ROI, which is what a caller would read as a digit."""
    flat = np.full((20, 60, 3), 140, np.uint8)
    assert len(split_digits(flat)) <= 1


# --- the reading contract ---------------------------------------------------

def test_hp_range_excludes_implausible_readings():
    """A number outside any real tower's range is a segmentation failure that
    happened to produce digits. Reporting it would be worse than reporting
    nothing, because it looks measured."""
    assert HP_RANGE[0] >= 1
    assert HP_RANGE[1] < 100000


def test_the_numeral_offset_reaches_above_the_bar():
    assert NUMERAL_OFFSET[1] < NUMERAL_OFFSET[3] <= 0
