"""Place a card on the real screen: tile coordinates -> two taps.

THE UNTESTED JOINT
------------------
Everything upstream of this has been measured against recorded frames. This
has not run at all. It is also the one stage where being wrong is invisible
from the inside: a tap landing on the wrong tile produces a perfectly valid
GameState on the next frame, showing a unit somewhere the agent did not intend,
and nothing in the pipeline flags it. That is why it is worth closing the loop
here before anything upstream gets polished further.

WHY TWO TAPS AND NOT A DRAG
---------------------------
Clash Royale places a card by selecting it in the hand and then tapping the
target. `ClashRoyaleBuildABot` does exactly that (`Bot.play_action`), and this
follows it rather than inventing a drag gesture, because the tap pair is what
upstream has running against the real game.

COORDINATES ARE ANDROID'S, NOT THE SCREEN'S
-------------------------------------------
`adb shell input tap` takes coordinates in the Android display space
(720x1280 here), NOT in captured-frame pixels and NOT in desktop pixels. Three
spaces are in play and confusing them silently misplaces every card:

    Android display   720 x 1280   what input tap wants, and what CRBAB's
                                   TILE_INIT_*/TILE_* constants are in
    captured frame    549 x 976    what the detector and readers see
    desktop           1920 x 1020  the WGC surface, physical pixels

The conversion is CRBAB's own `_get_tile_centre` inverse, imported rather than
re-derived -- a second copy of TILE_INIT_X here is precisely the duplicated
constant CLAUDE.md names as having gone stale twice.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from clashroyalebuildabot.constants import (
    DISPLAY_CARD_DELTA_X,
    DISPLAY_CARD_HEIGHT,
    DISPLAY_CARD_INIT_X,
    DISPLAY_CARD_WIDTH,
    DISPLAY_CARD_Y,
    DISPLAY_HEIGHT,
    TILE_HEIGHT,
    TILE_INIT_X,
    TILE_INIT_Y,
    TILE_WIDTH,
)

from live.adapter import TILE_Y_OFFSET

ADB = Path(r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe")

# Between the card tap and the tile tap. The card has to register as selected
# before the placement lands; without a gap the second tap can be swallowed.
TAP_GAP_S = 0.08


class ActuationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Tap:
    x: int
    y: int


def card_centre(slot: int) -> Tap:
    """Android coordinates of hand slot 0-3."""
    if not 0 <= slot <= 3:
        raise ActuationError(f"hand slot must be 0-3, got {slot}")
    x = DISPLAY_CARD_INIT_X + DISPLAY_CARD_WIDTH / 2 + slot * DISPLAY_CARD_DELTA_X
    y = DISPLAY_CARD_Y + DISPLAY_CARD_HEIGHT / 2
    return Tap(int(round(x)), int(round(y)))


def tile_centre(tile_x: int, tile_y: int) -> Tap:
    """Android coordinates of a DETECTOR-frame tile (18x32).

    Detector frame, not engine frame. The engine's board is 18x34 with an extra
    row behind each King, so anything holding engine coordinates must convert
    first -- `engine_tile_centre` does that. Mixing the two shifts every
    placement by one row, which is the bug class this project has already paid
    for once.
    """
    x = TILE_INIT_X + (tile_x + 0.5) * TILE_WIDTH
    y = DISPLAY_HEIGHT - TILE_INIT_Y - (tile_y + 0.5) * TILE_HEIGHT
    return Tap(int(round(x)), int(round(y)))


def engine_tile_centre(tile_x: int, tile_y: int) -> Tap:
    """Android coordinates of an ENGINE-frame tile (18x34).

    The policy emits engine coordinates, so this is the one a live loop wants.
    Note the offset is still unverified -- see adapter.TILE_Y_OFFSET -- so a
    systematic one-row placement error is possible and would show up here
    first, as cards landing a row nearer or further than intended.
    """
    return tile_centre(tile_x, tile_y - TILE_Y_OFFSET)


class AdbActuator:
    """Taps via adb. Constructed with `dry_run=True` it only records intent.

    Dry run is the default on purpose: this module can misplace real cards in a
    real match, so acting has to be asked for explicitly rather than being what
    happens if a caller forgets a flag.
    """

    def __init__(self, dry_run: bool = True, adb: Path = ADB, serial: str | None = None):
        self.dry_run = dry_run
        self.adb = Path(adb)
        self.serial = serial
        self.taps: list[Tap] = []
        if not dry_run and not self.adb.exists():
            raise ActuationError(f"adb not found at {self.adb}")

    def tap(self, point: Tap) -> None:
        self.taps.append(point)
        if self.dry_run:
            return
        cmd = [str(self.adb)]
        if self.serial:
            cmd += ["-s", self.serial]
        cmd += ["shell", "input", "tap", str(point.x), str(point.y)]
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if done.returncode != 0:
            raise ActuationError(
                f"adb tap failed ({done.returncode}): {done.stderr.strip()}")

    def play(self, slot: int, tile_x: int, tile_y: int, *,
             engine_frame: bool = True) -> tuple[Tap, Tap]:
        """Select a hand slot, then tap the target tile."""
        import time  # noqa: PLC0415

        card = card_centre(slot)
        target = (engine_tile_centre(tile_x, tile_y) if engine_frame
                  else tile_centre(tile_x, tile_y))
        self.tap(card)
        if not self.dry_run:
            time.sleep(TAP_GAP_S)
        self.tap(target)
        return card, target
