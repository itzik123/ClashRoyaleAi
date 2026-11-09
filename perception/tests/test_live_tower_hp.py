"""Tests for live/tower_hp.py: Princess Tower HP from the printed numeral.

The numeral is absolute and the contract wants a fraction of the tower's own
maximum, which is not on screen; the tracker learns it. Each case pins a way
that goes quietly wrong: a single misread setting the ceiling, an unreadable
numeral reported as damage, a frame the templates do not fit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2", reason="opencv not installed")
pytest.importorskip("clashroyalebuildabot", reason="vendored bot not importable")

import cv2  # noqa: E402

from contracts import TowerObservation  # noqa: E402
from live.tower_hp import (  # noqa: E402
    CONFIRMATIONS,
    TEMPLATE_FRAME_SIZE,
    TEMPLATES_DIR,
    TowerHpTracker,
)
from readers.tower_numerals import TowerNumeralReading  # noqa: E402

_ASSETS = Path(__file__).resolve().parent / "assets"
BAR = TowerObservation(1.0, False)                 # what the bar reader gives
FIELDS = ("own_princess_left", "own_princess_right",
          "opp_princess_left", "opp_princess_right")
FRAME = np.zeros((TEMPLATE_FRAME_SIZE[1], TEMPLATE_FRAME_SIZE[0], 3), np.uint8)


class ScriptedReader:
    """Returns the next scripted value for each tower, None for unreadable."""

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.boxes = {}

    def read(self, frame, bar, det_size):
        name = self.boxes[bar]
        value = self.script[name].pop(0) if self.script.get(name) else None
        if value is None:
            return TowerNumeralReading(None, "", 0.0)
        return TowerNumeralReading(value, str(value), 0.9)


def tracker(script):
    reader = ScriptedReader(script)
    boxes = {name: (i, 0, i + 1, 1) for i, name in enumerate(FIELDS)}
    reader.boxes = {box: name for name, box in boxes.items()}
    return TowerHpTracker(reader=reader, boxes=boxes, det_size=(368, 652))


def fallback():
    return {k: BAR for k in FIELDS}


def test_a_tower_is_its_own_maximum_once_confirmed():
    t = tracker({"opp_princess_left": [1750] * CONFIRMATIONS + [1400]})
    for _ in range(CONFIRMATIONS - 1):
        out = t.read(FRAME, fallback())
        assert out["opp_princess_left"] is BAR      # not confirmed yet
    out = t.read(FRAME, fallback())
    assert out["opp_princess_left"].hp_fraction == pytest.approx(1.0)
    assert out["opp_princess_left"].hp_measured
    out = t.read(FRAME, fallback())
    assert out["opp_princess_left"].hp_fraction == pytest.approx(1400 / 1750)


def test_one_misread_cannot_set_the_ceiling():
    """'2030' read once as '8030' would make every later reading look like a
    tower at a quarter HP, for the rest of the match."""
    t = tracker({"opp_princess_left": [2030, 8030, 2030, 1800]})
    for _ in range(3):
        t.read(FRAME, fallback())
    assert t.maxima()["opp_princess_left"] == 2030
    out = t.read(FRAME, fallback())
    assert out["opp_princess_left"].hp_fraction == pytest.approx(1800 / 2030)


def test_an_unreadable_numeral_keeps_the_bar_reading():
    t = tracker({"opp_princess_left": [2030, 2030, None]})
    t.read(FRAME, fallback())
    t.read(FRAME, fallback())
    out = t.read(FRAME, fallback())
    assert out["opp_princess_left"] is BAR
    assert not out["opp_princess_left"].destroyed


def test_a_frame_the_templates_do_not_fit_is_not_read():
    t = tracker({"opp_princess_left": [2030] * 4})
    wrong = np.zeros((1280, 720, 3), np.uint8)
    for _ in range(3):
        assert t.read(wrong, fallback())["opp_princess_left"] is BAR
    assert t.numeral_reads == 0


def test_a_new_match_forgets_the_maxima():
    t = tracker({"opp_princess_left": [2030, 2030, 1750, 1750]})
    t.read(FRAME, fallback())
    t.read(FRAME, fallback())
    t.reset()
    t.read(FRAME, fallback())
    out = t.read(FRAME, fallback())
    assert out["opp_princess_left"].hp_fraction == pytest.approx(1.0)
    assert t.maxima()["opp_princess_left"] == 1750


# --- end to end, against the frame the numeral templates were accepted on ----

@pytest.fixture(scope="module")
def frame_2030():
    path = _ASSETS / "tower_numerals_2030_549x976.jpg"
    if not path.exists() or not (TEMPLATES_DIR / "digits.json").exists():
        pytest.skip("reference frame or tower templates missing")
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def test_the_reference_frame_reads_both_enemy_princesses_full(frame_2030):
    t = TowerHpTracker()
    for _ in range(CONFIRMATIONS):
        out = t.read(frame_2030, fallback())
    assert t.maxima()["opp_princess_left"] == 2030
    assert t.maxima()["opp_princess_right"] == 2030
    for name in ("opp_princess_left", "opp_princess_right"):
        assert out[name].hp_measured and out[name].hp_fraction == 1.0


def test_the_adapter_uses_the_numeral_where_the_bar_failed(frame_2030):
    """The live failure this fixes: CRBAB's bar read 0.0 on a full Princess,
    which the adapter correctly reports as unmeasured -- and the mirror then
    assumed every tower full. With the tracker the same frame is measured."""
    from clashroyalebuildabot.constants import SCREENSHOT_HEIGHT, SCREENSHOT_WIDTH
    from live.adapter import build_game_state
    from test_live_adapter import state  # noqa: PLC0415  (its fake CRBAB State)

    small = cv2.resize(frame_2030, (SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT))
    st = state()
    for attr in ("left_enemy_princess_hp", "right_enemy_princess_hp"):
        getattr(st.numbers, attr).number = 0.0          # the bar failed
    t = TowerHpTracker()
    for _ in range(CONFIRMATIONS):
        gs, _ = build_game_state(st, frame_2030, small, tower_hp=t)
    assert gs.opp_princess_left.hp_measured and gs.opp_princess_right.hp_measured
    control, _ = build_game_state(st, frame_2030, small)
    assert not control.opp_princess_left.hp_measured
