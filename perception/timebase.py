"""The one place real seconds are converted to simulator ticks.
tests/test_timebase.py asserts the elixir constant against the live engine.

The engine never states its tick duration. CardStats::attackCooldown is stored
in ticks while the real game publishes the same figure ("hit speed") in
seconds, and across the registry it is the real hit speed times ten:

    Archers        9 ticks   <->  0.9 s
    Musketeer     10 ticks   <->  1.0 s
    Knight        12 ticks   <->  1.2 s
    Valkyrie      15 ticks   <->  1.5 s
    Hog Rider     16 ticks   <->  1.6 s
    King Tower    10 ticks   <->  1.0 s   (GameManager::addTower)

132 rows agreeing on one ratio define the tick more reliably than any single
constant.

The engine regenerates 0.035 elixir per tick: 2.857 s per elixir against the
real game's 2.8, about 2% slow. Over a match that would put a derived opponent
elixir ~1.3 low by three minutes, turning the "opponent elixir went negative =>
a placement was missed" alarm into false positives. So the two rates are kept
separate:

  * REAL_SECONDS_PER_ELIXIR_1X drives the opponent elixir model (the real
    opponent).
  * SIM_ELIXIR_REGEN_PER_TICK describes the engine.

Neither is corrected to match the other; that would be a gameplay change to the
simulator.
"""

from __future__ import annotations

# Ticks per real second. See the derivation above.
TICKS_PER_SECOND = 10

# GameManager.h ELIXIR_REGEN_RATE, applied once per tick in step(). Asserted
# against the live engine in tests/test_timebase.py.
SIM_ELIXIR_REGEN_PER_TICK = 0.035

# Real-game single-elixir rate.
REAL_SECONDS_PER_ELIXIR_1X = 2.8

# PlayerState::initializeDeck sets both players to 5.0 at match start.
STARTING_ELIXIR = 5.0

# Elixir cap (GameManager::step()'s std::min).
MAX_ELIXIR = 10.0

# Ratio by which the simulator's elixir economy runs slow (>1.0 = slower than
# the real game), exported so a comparison can state its correction.
SIM_ELIXIR_RATE_ERROR = (
    REAL_SECONDS_PER_ELIXIR_1X * SIM_ELIXIR_REGEN_PER_TICK * TICKS_PER_SECOND
)  # == 0.98: the simulator produces 98% of the real elixir per unit time.


def seconds_to_ticks(seconds: float) -> int:
    """Real seconds -> simulator ticks, rounded to nearest: truncation would bias
    every conversion earlier.
    """
    return int(round(seconds * TICKS_PER_SECOND))


def ticks_to_seconds(ticks: int) -> float:
    """Simulator ticks -> real seconds."""
    return ticks / TICKS_PER_SECOND


def frame_index_to_ticks(frame_index: int, fps: float) -> int:
    """Video frame number -> simulator ticks. Constant-frame-rate sources only:
    under VFR the index is not proportional to time, which is why
    capture/video.py refuses an unconfirmed fps.
    """
    if fps <= 0:
        raise ValueError(f"frame_index_to_ticks: non-positive fps {fps!r}")
    return seconds_to_ticks(frame_index / fps)


def elixir_regenerated(seconds: float, multiplier: float = 1.0) -> float:
    """Real elixir accrued over `seconds` at `multiplier` x rate (1/2/3 for
    single/double/triple). Models the real game, for track/opp_elixir.py.
    """
    if seconds < 0:
        raise ValueError(f"elixir_regenerated: negative duration {seconds!r}")
    return seconds * multiplier / REAL_SECONDS_PER_ELIXIR_1X
