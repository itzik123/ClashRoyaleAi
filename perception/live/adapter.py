"""ClashRoyaleBuildABot's `State` -> this project's `GameState`.

This is the join between the vendored detector and the contract the training
side consumes. Everything it does is a translation the detector does not do
itself, and each one exists because of a measured gap:

    off-board detections   31% of raw detections sat outside the arena, all of
                           them the two player avatar icons read as `knight`
                           (board_filter.py)
    unit name -> card id   97 detector classes against 132 engine cards, with
                           27 names that did not resolve at all, including
                           `archer` and `minion` -- two of our own eight
                           (unit_to_card.py)
    per-unit HP            CRBAB reports none; observation channels 0-7 store
                           hp/MAX per cell, 4,896 of the 13,606 floats
                           (unit_hp.py)
    King Tower HP          CRBAB reads the two Princess bars and not the King;
                           extra scalars 3 and 6 need it (king_hp.py)
    tile frame             the detector's arena is 18x32, the engine's board is
                           18x34 -- see TILE_Y_OFFSET

THE ROW OFFSET IS DERIVED, NOT VERIFIED
---------------------------------------
`ClashEnv` adds one row behind each King Tower that the real arena does not
have, so the detector's 32 rows are the engine's rows 1..32 and the conversion
is `engine_y = detector_y + 1`. That is a derivation from the two heights, not
a measurement: it has NOT been checked against a 720x1280 calibration profile,
because none exists yet (`config/` holds only the 1920x1080 desktop profile).

This is flagged rather than assumed silently because a one-row displacement is
precisely the failure this project has already paid for once -- the team-1
observation bug of 2026-07-31 sat undetected through an entire coordinate audit
and every training metric. `GameState.flags` carries "tile_offset_unverified"
so a consumer cannot use these coordinates without meeting the caveat.

SIDE IS NOT OVERRIDDEN BY THE BADGE
-----------------------------------
The level badge states the team in the game's own UI, and the plan was to let
it override CRBAB's learned `side.onnx`. Measured against 60 hand-labelled
crops that turned out not to be supported: over 67 associated detections the
two disagree on 10%, and of five disagreements checked by eye the badge was
right twice (#64, #225 -- our own Valkyrie called an enemy) and wrong twice
(#108, a neighbour's badge on an undamaged unit that had none of its own;
#194, a false badge on a Giant's foot). One was a junk detection either way.

Neither source dominates, so `team` stays CRBAB's and the badge's reading is
carried beside it in `team_from_badge`, with `confidence` lowered and a
"side_disagreement" flag raised. An earlier 31% figure quoted for this was
measured with the old broken matcher and was largely counting bad
ASSOCIATION, not bad `side.onnx`.
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
from live.board_filter import BOARD_HEIGHT, BOARD_WIDTH, on_board
from live.king_hp import read_king_hp
from live.unit_hp import CALIBRATION_WIDTH, find_badges, match_badges
from live.unit_to_card import (
    UnitMappingError,
    card_id_for,
    hand_card_id_for,
    is_board_presence,
)

# The engine's board is one row taller at each end than the real arena. See the
# module docstring -- this is derived from the two heights, not measured.
TILE_Y_OFFSET = 1
ENGINE_BOARD_HEIGHT = BOARD_HEIGHT + 2 * TILE_Y_OFFSET   # 34

# Confidence assigned to a unit whose two side signals disagree. Not zero: the
# detection itself is still evidence that SOMETHING is on that tile, and the
# consumer may reasonably keep it in the count channels while distrusting which
# half of the board it belongs to.
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
    """Detector arena tile (18x32) -> engine board tile (18x34).

    Only the row moves; the engine adds no columns. Kept as a named function
    rather than an inline `+ 1` so the offset has one definition and one place
    to be corrected when it is finally measured.
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

    Both frames are required because the two readers are calibrated in
    different spaces and neither can be derived from the other for free:

      `frame`           NATIVE resolution, for the badge and unit-bar scan. In
                        the 368x652 detector frame the bar is ~6 px tall, and
                        throwing away resolution on the one measurement that
                        needs it would be perverse.
      `detector_frame`  the 368x652 image the detector was actually run on,
                        for king_hp.py, whose ROI constants are in that space.

    Passing both rather than resizing internally keeps the resize cost with the
    caller, which already paid it to run the detector.
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
            # A spell in flight is not an entity; ClashEnv skips anything that
            # is not isTargetable(), so it must not reach the spatial channels.
            not_presence += 1
            continue
        left, top, right, bottom = pos.bbox
        kept.append((team, unit, (left * sx, top * sy, right * sx, bottom * sy)))

    # Scan badges once per frame and assign them to all units at once, so two
    # units cannot claim the same badge.
    badges = find_badges(arr)
    matched = match_badges(badges, [k[2] for k in kept], scale)

    units = []
    for (team, unit, _bbox), badge in zip(kept, matched):
        name = unit.unit.name
        try:
            card_id = card_id_for(name)
        except UnitMappingError:
            # A wrong card id is worse than a missing one -- it fills the
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
        # Stays None unless a caller has something better than the 2.1x
        # over-count units-appearing gives. See GameState.opp_elixir_spent.
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

    `state.cards` is FIVE entries, not four: `CARD_CONFIG[0]` is the small
    bottom-left "Next" preview box (26x33 at x=21), and only `CARD_CONFIG[1:]`
    are hand slots (61x73 at x>=84, y=543). So the hand is `[1:5]`.

    Reading `[:4]` instead cost a whole match. It put the NEXT card -- one the
    player does not hold and cannot play -- in observation slot 0, shifted every
    real card one slot right, and dropped hand slot 3 entirely. The actuator
    taps physical slot `i`, so the agent asked for the card it could see in slot
    i and got the one beside it, every single placement.

    CRBAB's own `_detect_if_ready` iterates `crops[1:]`, so `state.ready` was
    already indexed to hand slots 0-3. `ready` and `cards` therefore disagreed
    by one about what "slot 1" meant, which is the kind of internal
    contradiction worth grepping for after finding one of these.
    """
    return tuple(hand_card_id_for(getattr(card, "name", ""))
                 for card in state.cards[1:5])


def _read_towers(state, detector_frame) -> dict:
    """The six towers as fractions.

    The Kings come from king_hp.py, because CRBAB does not read them at all --
    it reads the two Princess bars and stops, while extra scalars 3 and 6 need
    the Kings.

    CRBAB's Princess reader already returns a FRACTION (`change_point / 39`),
    not absolute HP, so nothing is normalised here. That is the right unit for
    this contract anyway: tower maxima depend on the account's tower level --
    measured 1750 at level 4 and 1890 at level 5 against the engine's level-9
    2534 -- so an absolute number would be wrong by ~30% and by a different
    factor per player.

    A 0.0 READING IS NOT "DESTROYED". `_calculate_hp` returns 0.0 both when the
    bar reads empty and when it cannot match the bar colours at all
    (`avg_min_dist > threshold`), and this adapter used to resolve that in
    favour of destroyed. Live capture at 549x976 falsified it: over 101 frames
    of a real match `right_ally_princess_hp` read 0.0 in **101 of 101** while
    the other three read 1.0, and the tower was verified standing at FULL
    health -- its own ROI shows the numeral 1890, which is a level-5 Princess
    at maximum. The earlier justification came from 71 ladder frames where the
    failure happened not to occur, so a real reading and a total reader failure
    looked alike.

    So 0.0 is now reported as NOT MEASURED, and `destroyed` is left to a caller
    that holds history. A tower wrongly marked destroyed is worse than one
    marked unknown: extra scalars 3-8 are tower HP, and a live tower reported
    dead tells the policy a lane is already lost.

    THE REAL FIX IS TO READ THE NUMERAL, NOT THE BAR. The HP number is printed
    beside every Princess bar and is plainly legible at this resolution --
    1759, 1890 -- which gives absolute HP with no colour matching, no occlusion
    ambiguity, and no need to know the tower's level. `readers/clock.py`
    already carries a digit-template classifier built for exactly this kind of
    glyph. Recorded in BOT_REQUESTS.md rather than done here.
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
        # An absent King bar means FULL -- validated 8/8 on hand-checked
        # frames -- so it is a determination, not a failure, and is reported
        # as measured either way.
        out[key] = TowerObservation(king.fraction, hp_measured=True)
    return out
