"""Tests for live/adapter.py, the CRBAB State -> GameState join.

Driven by hand-built stand-ins for CRBAB's namespaces rather than the detector
(~4 s a frame): what is under test is the adapter's translations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("clashroyalebuildabot", reason="vendored bot not importable")
pytest.importorskip("cv2", reason="opencv not installed")

from contracts import UNKNOWN_CARD_SIM_ID, GameState, Phase  # noqa: E402
from live.adapter import (  # noqa: E402
    ENGINE_BOARD_HEIGHT,
    SIDE_DISAGREEMENT_CONFIDENCE,
    TILE_Y_OFFSET,
    build_game_state,
    to_engine_tile,
)
from live.board_filter import BOARD_HEIGHT  # noqa: E402


# --- stand-ins for the CRBAB namespaces ---
@dataclass
class FakeUnit:
    name: str


@dataclass
class FakePos:
    bbox: tuple
    conf: float
    tile_x: int
    tile_y: int


@dataclass
class FakeDetection:
    unit: FakeUnit
    position: FakePos


@dataclass
class FakeNumber:
    number: float


@dataclass
class FakeNumbers:
    left_enemy_princess_hp: FakeNumber
    right_enemy_princess_hp: FakeNumber
    left_ally_princess_hp: FakeNumber
    right_ally_princess_hp: FakeNumber
    elixir: FakeNumber


@dataclass
class FakeCard:
    name: str


@dataclass
class FakeState:
    allies: list
    enemies: list
    numbers: FakeNumbers
    cards: tuple


def unit(name, tile_x, tile_y, conf=0.9, bbox=(100, 200, 130, 240)):
    return FakeDetection(FakeUnit(name), FakePos(bbox, conf, tile_x, tile_y))


def state(allies=(), enemies=(), elixir=5.0, cards=("giant", "fireball",
                                                    "valkyrie", "archers"),
          next_card="minions"):
    """`cards` is the hand, four slots. The detector's own list has five entries
    (CARD_CONFIG[0] is the "Next" preview), so the preview is prepended here; a
    four-entry fake is what let the off-by-one in `_hand_ids` pass a full
    suite.
    """
    nums = FakeNumbers(FakeNumber(0.5), FakeNumber(0.5), FakeNumber(1.0),
                       FakeNumber(1.0), FakeNumber(elixir))
    return FakeState(list(allies), list(enemies), nums,
                     tuple(FakeCard(c) for c in (next_card, *cards)))


@pytest.fixture
def frames():
    """A native frame and its detector-space resize, both blank. With no badges
    anywhere every unit takes the "no widget drawn" path, isolating the adapter
    from the HP reader.
    """
    from clashroyalebuildabot.constants import SCREENSHOT_HEIGHT, SCREENSHOT_WIDTH
    native = np.zeros((1280, 720, 3), np.uint8)
    small = np.zeros((SCREENSHOT_HEIGHT, SCREENSHOT_WIDTH, 3), np.uint8)
    return native, small


# --- the tile frame ---

def test_engine_board_is_two_rows_taller_than_the_arena():
    """ClashEnv adds a row behind each King that the real board does not have.
    """
    assert ENGINE_BOARD_HEIGHT == BOARD_HEIGHT + 2
    assert ENGINE_BOARD_HEIGHT == 34


def test_tile_conversion_shifts_rows_only():
    assert to_engine_tile(0, 0) == (0, TILE_Y_OFFSET)
    assert to_engine_tile(17, 31) == (17, 31 + TILE_Y_OFFSET)


def test_converted_rows_stay_inside_the_engine_board():
    """Both ends: an off-by-one here is invisible in every metric."""
    for y in (0, BOARD_HEIGHT - 1):
        _x, ey = to_engine_tile(0, y)
        assert 0 <= ey < ENGINE_BOARD_HEIGHT


def test_offset_is_flagged_as_unverified(engine, frames):
    """Derived from the two heights, not measured against a calibration; a
    consumer must meet that caveat.
    """
    gs, _ = build_game_state(state(), *frames)
    assert "tile_offset_unverified" in gs.flags


# --- what gets dropped ---

def test_off_board_detections_are_dropped(engine, frames):
    """The two player avatar icons are detected as `knight`, outside the arena.
    """
    gs, rep = build_game_state(
        state(allies=[unit("knight", 19, 5), unit("knight", -2, 13),
                      unit("giant", 8, 10)]), *frames)
    assert rep.off_board == 2
    assert [u.unit_name for u in gs.units] == ["giant"]
    assert "off_board_detections" in gs.flags


def test_projectiles_are_not_board_presence(engine, frames):
    """ClashEnv skips anything !isTargetable(), so a spell in flight must not
    reach the spatial channels.
    """
    gs, rep = build_game_state(
        state(allies=[unit("giant_snowball", 8, 10), unit("giant", 8, 11)]),
        *frames)
    assert rep.not_board_presence == 1
    assert [u.unit_name for u in gs.units] == ["giant"]


# --- identity ---

def test_units_carry_engine_card_ids(engine, frames):
    gs, _ = build_game_state(state(allies=[unit("archer", 6, 12)]), *frames)
    assert engine.get_card_info(gs.units[0].card_sim_id)["name"] == "Archers"


def test_hand_uses_the_card_table_not_the_unit_table(engine, frames):
    """Fireball spawns no unit, so resolving the hand through the unit table would
    silently return the sentinel for it.
    """
    gs, _ = build_game_state(state(cards=("giant", "fireball", "valkyrie",
                                          "musketeer")), *frames)
    assert UNKNOWN_CARD_SIM_ID not in gs.my_hand
    assert engine.get_card_info(gs.my_hand[1])["name"] == "Fireball"


def test_hand_skips_the_next_card_preview(engine, frames):
    """CARD_CONFIG[0] is the "Next" box, not a hand slot. Reading `cards[:4]`
    shifts every real card one slot right; since the actuator taps physical
    slot `i`, every placement plays the card beside the one asked for.
    """
    gs, _ = build_game_state(
        state(next_card="giant",
              cards=("archers", "valkyrie", "minipekka", "cannon")), *frames)
    names = [engine.get_card_info(c)["name"] for c in gs.my_hand]
    assert names == ["Archers", "Valkyrie", "Mini PEKKA", "Cannon"]
    assert "Giant" not in names, "the Next preview leaked into the hand"


def test_hand_has_one_entry_per_playable_slot(engine, frames):
    """Four, matching the engine's HAND_SIZE and the actuator's slots 0-3."""
    gs, _ = build_game_state(state(), *frames)
    assert len(gs.my_hand) == 4


def test_unreadable_hand_slot_is_the_sentinel_not_an_error(engine, frames):
    gs, _ = build_game_state(state(cards=("blank", "", "valkyrie", "giant")),
                             *frames)
    assert gs.my_hand[0] == UNKNOWN_CARD_SIM_ID
    assert gs.my_hand[1] == UNKNOWN_CARD_SIM_ID


# --- HP and side ---

def test_absent_widget_means_undamaged_but_unmeasured(engine, frames):
    """No bar is drawn over a healthy unit, so 1.0 is right, but `hp_measured`
    stays False: a missed bar in a crowd looks the same on one frame.
    """
    gs, rep = build_game_state(state(allies=[unit("giant", 8, 10)]), *frames)
    u = gs.units[0]
    assert u.hp_fraction == 1.0
    assert u.hp_measured is False
    assert rep.hp_measured == 0


def test_team_comes_from_the_detector_not_the_badge(engine, frames):
    """Neither side signal dominates, so disagreement is surfaced rather than
    resolved (see unit_hp.py).
    """
    gs, _ = build_game_state(
        state(allies=[unit("giant", 8, 10)], enemies=[unit("valkyrie", 8, 20)]),
        *frames)
    assert [u.team for u in gs.units] == [0, 1]
    assert all(u.team_from_badge is None for u in gs.units)


def test_kings_are_read_and_princesses_pass_through(engine, frames):
    """CRBAB reads only the Princess bars; extra scalars 3 and 6 need the Kings.
    """
    gs, _ = build_game_state(state(), *frames)
    for tower in (gs.own_king, gs.opp_king):
        assert 0.0 <= tower.hp_fraction <= 1.0
        assert tower.hp_measured is True
    assert gs.own_princess_left.hp_fraction == pytest.approx(1.0)
    assert gs.opp_princess_left.hp_fraction == pytest.approx(0.5)


def test_a_zero_princess_is_unmeasured_not_destroyed(engine, frames):
    """0.0 means "could not read the bar", not "destroyed": `_calculate_hp`
    returns 0.0 when it cannot colour-match, and a live capture read a
    full-health Princess as 0.0 every frame. A live tower reported dead tells
    the policy a lane is lost. See BOT_REQUESTS.md item 7.
    """
    st = state()
    st.numbers.left_enemy_princess_hp = FakeNumber(0.0)
    gs, _ = build_game_state(st, *frames)
    assert gs.opp_princess_left.hp_measured is False
    assert gs.opp_princess_left.destroyed is False


def test_a_readable_princess_is_measured(engine, frames):
    """The contrast: a nonzero reading is trusted, and is already a fraction
    (`_calculate_hp` returns change_point/39), so nothing is rescaled.
    """
    st = state()
    st.numbers.left_enemy_princess_hp = FakeNumber(0.5)
    gs, _ = build_game_state(st, *frames)
    assert gs.opp_princess_left.hp_measured is True
    assert gs.opp_princess_left.hp_fraction == pytest.approx(0.5)


def test_elixir_is_never_negative(engine, frames):
    gs, _ = build_game_state(state(elixir=-1), *frames)
    assert gs.my_elixir == 0.0


# --- shape ---

def test_output_is_a_game_state(engine, frames):
    gs, rep = build_game_state(state(), *frames, seconds_elapsed=42.5,
                               phase=Phase.DOUBLE, frame_index=7)
    assert isinstance(gs, GameState)
    assert gs.seconds_elapsed == 42.5
    assert gs.phase is Phase.DOUBLE
    assert gs.frame_index == 7
    assert rep.detections == 0


def test_confidence_floor_is_below_one():
    """A disagreement must lower the number, or a numeric consumer sees nothing.
    """
    assert 0.0 < SIDE_DISAGREEMENT_CONFIDENCE < 1.0
