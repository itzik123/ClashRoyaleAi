"""ClashRoyaleBuildABot's `State` -> this project's `GameState`.

The join between the vendored detector and the contract the training side
consumes. Each translation fills a gap the detector leaves:

    off-board detections   the two player avatar icons read as `knight`,
                           outside the arena (board_filter.py)
    unit name -> card id   97 detector classes against the engine's cards,
                           many named differently (unit_to_card.py)
    per-unit HP            CRBAB reports none; observation channels 0-7
                           store hp/MAX per cell (unit_hp.py)
    King Tower HP          CRBAB reads only the Princess bars (king_hp.py)
    tile frame             the detector's arena is 18x32, the engine's
                           board 18x34 (TILE_Y_OFFSET)

The row offset is derived from the two heights, not measured: the engine adds
one row behind each King, so `engine_y = detector_y + 1`. `GameState.flags`
carries "tile_offset_unverified" so no consumer meets these coordinates without
the caveat.

Side is not overridden by the level badge: the two disagree on ~10% of
associated detections and neither is consistently right (see unit_hp.py).
`team` stays CRBAB's, the badge's reading rides beside it in `team_from_badge`,
and a disagreement lowers `confidence` and raises "side_disagreement".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contracts import (
    UNKNOWN_CARD_SIM_ID,
    GameState,
    Phase,
    TowerObservation,
    UnitObservation,
)
from live.board_filter import BOARD_HEIGHT, on_board
from live.king_hp import read_king_hp
from live.unit_hp import CALIBRATION_WIDTH, find_badges, match_badges
from live.unit_to_card import (
    UnitMappingError,
    card_id_for,
    hand_card_id_for,
    is_board_presence,
)

# The engine's board is one row taller at each end than the real arena;
# derived, not measured.
TILE_Y_OFFSET = 1
ENGINE_BOARD_HEIGHT = BOARD_HEIGHT + 2 * TILE_Y_OFFSET   # 34

# Confidence for a unit whose two side signals disagree. Not zero: something is
# still on that tile, even if which half it belongs to is in doubt.
SIDE_DISAGREEMENT_CONFIDENCE = 0.5


@dataclass(frozen=True)
class AdapterReport:
    """What the adapter dropped and why, for one frame."""

    detections: int = 0
    off_board: int = 0
    unmapped: int = 0
    not_board_presence: int = 0
    side_disagreements: int = 0
    hp_measured: int = 0

    @property
    def kept(self) -> int:
        return (self.detections - self.off_board - self.unmapped
                - self.not_board_presence)


def to_engine_tile(tile_x: int, tile_y: int) -> tuple[int, int]:
    """Detector arena tile (18x32) -> engine board tile (18x34). Only the row
    moves. A named function so the offset has one definition.
    """
    return tile_x, tile_y + TILE_Y_OFFSET


def build_game_state(
    state,
    frame,
    detector_frame,
    *,
    seconds_elapsed: float = 0.0,
    phase: Phase = Phase.SINGLE,
    frame_index: int = 0,
    wall_time_ms: float = 0.0,
    my_elixir_spent: float = 0.0,
    opp_elixir_spent: float | None = None,
) -> tuple[GameState, AdapterReport]:
    """One CRBAB `State` plus its two frames -> one `GameState`.

    The two readers are calibrated in different spaces:

      `frame`           native resolution, for the badge and unit-bar scan (in the
                        368x652 detector frame the bar is ~6 px tall).
      `detector_frame`  the 368x652 image the detector ran on, for king_hp.py,
                        whose ROI constants are in that space.

    The caller already paid for the resize to run the detector.
    """
    arr = np.asarray(frame)[..., :3]
    height, width = arr.shape[:2]
    scale = width / CALIBRATION_WIDTH

    # CRBAB's bboxes are in its own screenshot space; map them onto `frame`.
    from clashroyalebuildabot.constants import (  # noqa: PLC0415
        SCREENSHOT_HEIGHT,
        SCREENSHOT_WIDTH,
    )
    sx, sy = width / SCREENSHOT_WIDTH, height / SCREENSHOT_HEIGHT

    raw = [(0, u) for u in state.allies] + [(1, u) for u in state.enemies]
    detections = len(raw)
    off_board = unmapped = not_presence = disagreements = measured = 0

    kept = []
    for team, unit in raw:
        pos = unit.position
        if not on_board(pos.tile_x, pos.tile_y):
            off_board += 1
            continue
        if not is_board_presence(unit.unit.name):
            # A spell in flight is not an entity (ClashEnv skips anything not
            # isTargetable()), so it must not reach the spatial channels.
            not_presence += 1
            continue
        left, top, right, bottom = pos.bbox
        kept.append((team, unit, (left * sx, top * sy, right * sx, bottom * sy)))

    # Badges are scanned once per frame and assigned to all units at once, so
    # two units cannot claim the same badge.
    badges = find_badges(arr)
    matched = match_badges(badges, [k[2] for k in kept], scale)

    units = []
    for (team, unit, _bbox), badge in zip(kept, matched):
        name = unit.unit.name
        try:
            card_id = card_id_for(name)
        except UnitMappingError:
            # A wrong card id is worse than a missing one: it fills the
            # attribute channels with another card's damage, range and speed.
            card_id = UNKNOWN_CARD_SIM_ID
            unmapped += 1

        badge_team = None if badge is None else (0 if badge.ally else 1)
        confidence = float(unit.position.conf)
        if badge_team is not None and badge_team != team:
            disagreements += 1
            confidence = min(confidence, SIDE_DISAGREEMENT_CONFIDENCE)

        hp_measured = badge is not None and badge.bar_found
        if hp_measured:
            measured += 1
        tile_x, tile_y = to_engine_tile(unit.position.tile_x, unit.position.tile_y)
        units.append(UnitObservation(
            card_sim_id=card_id,
            unit_name=name,
            team=team,
            tile_x=tile_x,
            tile_y=tile_y,
            hp_fraction=badge.hp if hp_measured else 1.0,
            hp_measured=hp_measured,
            confidence=confidence,
            team_from_badge=badge_team,
        ))

    flags = ["tile_offset_unverified"]
    if disagreements:
        flags.append("side_disagreement")
    if off_board:
        flags.append("off_board_detections")
    if unmapped:
        flags.append("unmapped_unit")

    towers = _read_towers(state, detector_frame)
    game_state = GameState(
        units=tuple(units),
        my_elixir=max(0.0, float(state.numbers.elixir.number)),
        my_hand=_hand_ids(state),
        seconds_elapsed=seconds_elapsed,
        phase=phase,
        frame_index=frame_index,
        wall_time_ms=wall_time_ms,
        my_elixir_spent=my_elixir_spent,
        # None unless a caller has something better than units-appearing, which
        # over-counts 2.1x. See GameState.opp_elixir_spent.
        opp_elixir_spent=opp_elixir_spent,
        flags=tuple(flags),
        **towers,
    )
    report = AdapterReport(
        detections=detections, off_board=off_board, unmapped=unmapped,
        not_board_presence=not_presence, side_disagreements=disagreements,
        hp_measured=measured,
    )
    return game_state, report


def _hand_ids(state) -> tuple[int, ...]:
    """The four hand slots as simulator card ids, in on-screen order.

    `state.cards` has five entries: `CARD_CONFIG[0]` is the small "Next"
    preview (26x33 at x=21) and only `CARD_CONFIG[1:]` are hand slots (61x73 at
    x>=84, y=543). So the hand is `[1:5]`. CRBAB's `_detect_if_ready` iterates
    `crops[1:]`, so `state.ready` is already indexed to hand slots 0-3.
    """
    return tuple(hand_card_id_for(getattr(card, "name", ""))
                 for card in state.cards[1:5])


def _read_towers(state, detector_frame) -> dict:
    """The six towers as fractions.

    The Kings come from king_hp.py, since CRBAB reads only the Princess bars.
    CRBAB's Princess reader already returns a fraction (`change_point / 39`),
    which is the right unit: tower maxima depend on the account's tower level
    (1750 at level 4, 1890 at level 5, against the engine's level-9 2534), so
    an absolute number would be wrong by a different factor per player.

    A 0.0 reading is not "destroyed". `_calculate_hp` returns 0.0 both for an
    empty bar and for one it cannot colour-match, and live capture at 549x976
    read a full-health Princess as 0.0 on every frame. So 0.0 is reported as
    not measured, and `destroyed` is left to a caller holding history: a live
    tower reported dead tells the policy a lane is lost. Reading the printed HP
    numeral instead is tracked in BOT_REQUESTS.md.
    """
    nums = state.numbers
    out = {}
    for key, attr in (
        ("own_princess_left", "left_ally_princess_hp"),
        ("own_princess_right", "right_ally_princess_hp"),
        ("opp_princess_left", "left_enemy_princess_hp"),
        ("opp_princess_right", "right_enemy_princess_hp"),
    ):
        raw = float(getattr(nums, attr).number)
        readable = raw > 0.0
        out[key] = TowerObservation(
            hp_fraction=max(0.0, min(1.0, raw)) if readable else 1.0,
            hp_measured=readable,
            destroyed=False,
        )

    for key, ally in (("own_king", True), ("opp_king", False)):
        king = read_king_hp(detector_frame, ally=ally)
        # An absent King bar means full (validated 8/8 on hand-checked frames):
        # a determination, reported as measured.
        out[key] = TowerObservation(king.fraction, hp_measured=True)
    return out
