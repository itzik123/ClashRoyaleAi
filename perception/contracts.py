"""Data contract between the perception pipeline and everything downstream.

Dataclasses only -- no logic, no imports beyond the standard library. Every
other module in `perception/` depends on this one; this one depends on
nothing. That is deliberate: the contract has to be readable and diffable on
its own, without pulling in numpy/opencv, so the bridge and the trackers can
be unit-tested with no capture stack installed at all.

Two fields here look redundant and are not. Both exist because of a concrete
gap in the simulator, and both would be very easy to drop by accident:

  * `confidence`, on essentially everything. The engine has no
    representation of uncertainty whatsoever -- the observation vector is a
    dense float grid with no mask channel, no NaN, no "unknown" sentinel (see
    ClashEnv::extractObservationForTeam). A tile is either occupied or it is
    zero, and zero means "empty", never "I could not see". So uncertainty
    cannot be handed to the engine and cannot be represented inside it: it
    has to live in this layer, or it does not exist anywhere.

  * `wall_time_ms`, on PlacementEvent. The simulator applies a play on the
    exact tick it is told to, with no actuation delay model -- real input has
    roughly 100ms of it (touch -> render -> capture -> decode). Without a
    real-clock stamp taken at capture time, that offset can never be measured
    after the fact, only guessed. Keeping the raw timestamp costs 8 bytes and
    is the only thing that makes the calibration possible later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Phase(Enum):
    """Which elixir-rate regime the match is currently in.

    THE SIMULATOR HAS PHASES SINCE 2026-09-02, and this docstring's previous
    claim ("no multiplier, no overtime, and no phase concept anywhere in the
    engine") is no longer true. `GameManager::elixirMultiplierAtTick` runs the
    real schedule -- 1x, double from 2:00 (`DOUBLE_ELIXIR_TICK = 1200`), triple
    from 3:00 (`TRIPLE_ELIXIR_TICK = 1800`) -- the multiplier is observation
    scalar 9, and `elixir_multiplier_at_tick(tick)` is bound. See
    `perception/UPSTREAM_REQUESTS.md` item 26.

    What has NOT changed is that this enum is still not consumed by the
    engine, and should not be: the engine derives its phase from its OWN clock,
    so a mirror driven through `set_current_tick` is already correct and
    pushing a Phase into it would be a second, conflicting source of truth.
    This type remains the SENSOR's reading of what the screen shows.

    Two differences from the engine's schedule are real and deliberate:
    OVERTIME is a distinct member here because it differs in every respect
    other than elixir (sudden death, tower activation), and the engine has no
    overtime concept at all -- it runs a fixed 3600-tick match and resolves a
    timeout on tower HP. Do not read these two as the same axis.

    (`oppElixirMultiplier` is still a training-curriculum knob for handicapping
    the built-in opponent -- it is not a game phase, it COMPOSES with the phase
    rather than replacing it, and driving it from here would silently corrupt
    the opponent model.)

    This field is produced, carried, logged, and asserted on -- and then
    ignored by bridge/sim_driver.py, on purpose. It is here so that the
    perception layer is already correct on the day the engine grows phases,
    and so the gap is visible in the type rather than buried in a comment.
    See perception/README.md, "Known gaps".
    """

    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    OVERTIME = 4


class EventSource(Enum):
    """How a PlacementEvent was produced.

    Matters because the two sides of the board are labelled by completely
    different mechanisms with completely different error profiles: our own
    plays are read off the hand (a card vacating a slot is a near-certain
    placement), while the opponent's are detected from board pixels. Anything
    that aggregates confidence across both -- the divergence metric above
    all -- needs to be able to tell them apart.
    """

    OWN_HAND = "own_hand"          # inferred from our own hand changing
    OPPONENT_VISION = "opp_vision"  # detected on the board
    REPLAY_GROUND_TRUTH = "replay"  # read from a simulator replay JSON (tests)


# Sentinel for PlacementEvent.card_sim_id: a placement that was detected and
# localised, but whose card has no counterpart in the simulator's registry.
# Deliberately not None -- the field stays a plain int so consumers can never
# get a TypeError instead of the explicit branch they should have written.
# See mapping/card_map.json and bridge/sim_driver.py's own handling.
UNKNOWN_CARD_SIM_ID = -1


@dataclass(frozen=True)
class PlacementEvent:
    """One card being put on the board, by either player.

    Frozen: events are the append-only spine of the whole pipeline. Once a
    placement has been emitted and fed to the simulator, mutating it would
    desynchronise the estimator from its own history with no way to detect
    that it happened.
    """

    tick: int
    """Simulator ticks since match start. NOT seconds -- see timebase.py."""

    wall_time_ms: float
    """Capture-time wall clock, milliseconds. For latency calibration only;
    never fed to the engine. See this module's docstring."""

    card_sim_id: int
    """CardRegistry id, after mapping. UNKNOWN_CARD_SIM_ID if the card is
    real but absent from the simulator's roster."""

    card_real_name: str
    """Canonical real-game card name. Always populated, even when
    card_sim_id is UNKNOWN_CARD_SIM_ID -- it is the only thing that makes an
    unmapped placement diagnosable."""

    team: int
    """0 = us, 1 = opponent. Same convention as the engine."""

    tile_x: int
    tile_y: int
    """Board tile, already in ClashEnv convention: origin at the corner, y
    increasing toward the opponent, and ALREADY MIRRORED for team 1 the way
    ClashEnv::extractObservationForTeam mirrors it
    (y = BOARD_HEIGHT - 1 - rawY). Consumers never re-mirror."""

    confidence: float
    """[0,1]. For OWN_HAND events this is near 1.0 by construction."""

    source: EventSource = EventSource.OPPONENT_VISION

    is_evolution: bool = False
    """The evolved variant was played. Kept separate from card_sim_id rather
    than folded into it because the registry gives Evolutions their own ids
    (123-163) while reusing the base card's NAME verbatim -- id 1 and id 128
    are both "Archers". Name alone can therefore never disambiguate, so the
    flag has to be carried explicitly. See mapping/card_map.json."""


@dataclass
class ClockState:
    """Match clock, as believed right now."""

    tick: int
    seconds_remaining: float
    phase: Phase
    last_resync_tick: int
    """Tick of the last successful on-screen verification. A large gap
    between this and `tick` means the clock is free-running and its drift is
    unbounded -- consumers should discount accordingly."""

    drift_ms: float
    """Measured (predicted - observed) at the last resync. Signed: positive
    means the free-running clock was ahead of the screen."""

    confidence: float = 1.0


@dataclass
class CycleState:
    """A player's hand and the known part of their cycle order."""

    hand: tuple[int, ...]
    """Up to 4 simulator card ids. Shorter only before the deck is known."""

    next_card: int
    """The card shown in the "next" slot, or UNKNOWN_CARD_SIM_ID."""

    queue: tuple[int, ...]
    """Known cycle order behind `next_card`, nearest first. Empty when
    unknown. For the opponent this fills in gradually and is only fully
    determined once every one of their 8 cards has been seen once."""

    confidence: float = 1.0


@dataclass(frozen=True)
class UnitObservation:
    """One unit visible on the board, as the live sensor sees it.

    `hp_fraction` is relative to THIS CARD'S OWN maximum, not to any global
    constant. The bar on screen is a fraction of the unit's own health and
    nothing on screen states the absolute number, so converting here would
    require a max-HP table inside perception -- which is the consumer's, since
    only it knows what normalisation the observation wants.
    """

    card_sim_id: int
    """CardRegistry id. UNKNOWN_CARD_SIM_ID if the detector's class has no
    counterpart in the simulator's roster."""

    unit_name: str
    """The detector's own class name, kept for diagnosis. A card id alone
    cannot be traced back to what the model actually saw."""

    team: int
    """0 = us, 1 = opponent. Same convention as the engine."""

    tile_x: int
    tile_y: int
    """Board tile in ENGINE coordinates (18 x 34), already converted from the
    detector's 18 x 32 arena. See live/adapter.py for the conversion and why
    it carries a row offset."""

    hp_fraction: float
    """[0,1] of this card's own maximum. 1.0 when no bar was drawn, which in
    Clash Royale means undamaged."""

    hp_measured: bool
    """False means `hp_fraction` is the 1.0 default rather than a reading. Kept
    separate so 'certainly full' stays distinguishable from 'could not see' --
    the engine has no representation of uncertainty, so it has to live here."""

    confidence: float = 1.0
    """[0,1]. Lowered when independent signals disagree -- in particular when
    the badge hue and the detector's own side model name different teams."""

    team_from_badge: int | None = None
    """Team as read from the level badge hue, or None if no badge was matched.
    Carried ALONGSIDE `team` rather than replacing it: measured on 67
    associated detections the two disagree on 10%, and spot-checking those
    found the badge right twice and wrong twice. Neither source dominates, so
    the disagreement is surfaced instead of silently resolved."""


@dataclass(frozen=True)
class TowerObservation:
    """One of the six towers.

    `hp_fraction`, not absolute HP, for the same reason as UnitObservation --
    and additionally because tower maxima depend on the player's tower level,
    which is not on screen. The engine's own towers are level 9 (2534 HP) while
    the recordings are levels 4-5 (1750-1890), so an absolute number would be
    wrong by ~30% and by a DIFFERENT factor per player.
    """

    hp_fraction: float
    hp_measured: bool
    destroyed: bool = False


@dataclass
class GameState:
    """Everything the live sensor reads off one frame. Perception's only output.

    Deliberately NOT the 13,606-float observation vector. Encoding belongs to
    the training side, which owns the layout and has already changed it once
    (6253 -> 13606 on 2026-07-29). A compact state survives that; an encoder on
    this side of the boundary does not. A GameState can also be logged, diffed
    and eyeballed, which a float vector cannot.

    SHARED CONTRACT. Neither side changes it alone -- see the coordination
    section of docs/superpowers/specs/2026-07-30-perception-live-sensor-design.md.
    """

    units: tuple[UnitObservation, ...]

    my_elixir: float
    """[0,10]. Read from the elixir bar, the strongest reader in the pipeline
    (mean confidence 0.978-0.991)."""

    my_hand: tuple[int, ...]
    """Up to 4 simulator card ids, in on-screen slot order."""

    seconds_elapsed: float
    phase: Phase

    own_king: TowerObservation
    own_princess_left: TowerObservation
    own_princess_right: TowerObservation
    opp_king: TowerObservation
    opp_princess_left: TowerObservation
    opp_princess_right: TowerObservation

    my_elixir_spent: float = 0.0
    """Cumulative elixir WE have spent this match. Feeds extra scalar 1.

    Measured from the elixir bar rather than the hand: over one live match this
    recovered 27 cards against an affordable ceiling of 28, with a conservation
    residual of +14% that is one-sided and explained (regen while the bar sits
    at its 10 cap is invisible). Reading it from hand transitions instead
    scored 1/76 -- the card-icon template is the weakest reader in the pipeline
    and cannot carry a ledger."""

    opp_elixir_spent: float | None = None
    """Cumulative elixir the OPPONENT has spent, or None when not measured.

    None, never 0.0. A zero is indistinguishable from "they have spent
    nothing", and the consumer has to be able to tell those apart -- the engine
    has no representation of uncertainty, so it lives here. Same convention as
    `hp_measured` and `team_from_badge`.

    Currently always None. There is no opponent elixir bar, so spend can only
    be inferred from units appearing, and that over-counts 2.1x -- 64 detected
    placements against ~31 affordable, with implied elixir below zero for 98%
    of a match (BOT_REQUESTS.md item 8). Measured on a frozen checkpoint,
    zeroing this field costs the policy nothing: 450 episodes per arm against
    heuristic@1.35 gave a delta of +0.031 with 95% CI [-0.018, +0.080],
    excluding any degradation worse than 1.8 points.

    What the ENCODER writes into the observation slot when this is None is the
    training side's decision, not perception's. Perception reports that it
    could not measure it and stops."""

    frame_index: int = 0
    wall_time_ms: float = 0.0
    """Capture-time wall clock. For actuation-latency calibration; never fed to
    the engine. Same rationale as PlacementEvent.wall_time_ms."""

    flags: tuple[str, ...] = ()
    """Machine-readable anomaly markers raised for this frame, e.g.
    "side_disagreement" or "off_board_detections". Consumers gate on these
    rather than re-deriving the conditions from the numeric fields."""


@dataclass(frozen=True)
class BoardGeometry:
    """Board dimensions and landmarks, in simulator tile coordinates.

    Populated from the live engine where possible rather than hardcoded --
    see geometry.py. Exists as a dataclass so calibration code can be handed
    a geometry without importing the engine at all (which is what lets the
    homography tests run with no .pyd present).
    """

    width: int
    height: int
    river_y_start: float
    river_y_end: float
    left_bridge: tuple[float, float]
    right_bridge: tuple[float, float]
    own_king: tuple[float, float]
    opp_king: tuple[float, float]
    own_princess_left: tuple[float, float]
    own_princess_right: tuple[float, float]
    opp_princess_left: tuple[float, float]
    opp_princess_right: tuple[float, float]


@dataclass
class CalibrationProfile:
    """Screen -> tile mapping plus the fixed ROIs for one capture resolution.

    One profile per (device, resolution). A resolution change invalidates
    every pixel constant in here, which is why the resolution is part of the
    identity rather than a field that can quietly disagree with the frames
    being fed in -- see calib/homography.py's own size assertion.
    """

    name: str
    frame_width: int
    frame_height: int
    homography: tuple[float, ...]
    """Row-major 3x3, screen pixels -> board tiles. Flattened to 9 floats so
    the profile stays plain JSON with no numpy dependency."""

    anchors_screen: dict[str, tuple[float, float]] = field(default_factory=dict)
    """The named landmark pixel coordinates the homography was solved from,
    kept so a profile can be re-solved or audited without re-clicking."""

    rois: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    """Named (x, y, w, h) rectangles in frame pixels: elixir bar, clock, the
    four hand slots, the next-card slot, the six tower HP bars."""

    reprojection_error_tiles: float = 0.0
    """Worst-case landmark reprojection error from the solve, in tiles. The
    stage-0 acceptance number; stored so it can be asserted on load instead
    of being a one-off console print at calibration time."""
