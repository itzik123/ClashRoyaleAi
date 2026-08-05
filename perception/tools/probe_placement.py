"""Place one card at a named ENGINE tile and report where it actually landed.

    perception/.venv-dml/Scripts/python.exe perception/tools/probe_placement.py \
        --slot 0 --x 4 --y 8

WHY A PROBE RATHER THAN WATCHING THE LOOP
-----------------------------------------
`mvp_loop --act` produces a match's worth of placements chosen by a policy,
under time pressure, with the board changing underneath. That is the worst
possible instrument for measuring a coordinate convention. This places ONE
named card at ONE named tile with the board otherwise quiet, which is what
makes the answer attributable.

It drives the real `AdbActuator`, deliberately. The thing under test is
`engine_tile_centre`, so recomputing the tap here would test nothing.

WHAT IT CAN AND CANNOT SETTLE
-----------------------------
The round trip is self-consistent BY CONSTRUCTION: the actuator subtracts
TILE_Y_OFFSET and the adapter adds it back, so a matching y does NOT confirm
the offset convention is right. What it does catch is everything else -- a
wrong x, a scaling error, a tap landing on a different tile, the card not
being placed at all, or the WRONG CARD being played, which is how the hand
slot off-by-one was found.

Judging the offset itself needs a landmark, so both frames are kept as PNGs.
Place at y=16 (the row just below the river, which sits at [15.5, 17.5)) and
look at whether the unit stands at the water's edge or one row back.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT.parent / "python_ai") not in sys.path:
    sys.path.insert(0, str(_ROOT.parent / "python_ai"))

from clashroyalebuildabot.constants import (  # noqa: E402
    SCREENSHOT_HEIGHT,
    SCREENSHOT_WIDTH,
)
from clashroyalebuildabot.detectors.detector import Detector  # noqa: E402
from clashroyalebuildabot.namespaces.cards import Cards  # noqa: E402
from live.actuator import ADB, AdbActuator, engine_tile_centre  # noqa: E402
from live.adapter import TILE_Y_OFFSET  # noqa: E402

DECK = [Cards.VALKYRIE, Cards.ARCHERS, Cards.MINIONS, Cards.CANNON,
        Cards.FIREBALL, Cards.GIANT, Cards.MUSKETEER, Cards.MINIPEKKA]

# How long to wait for the placed unit to exist and be detectable. A card is
# visible almost immediately but the deploy animation is not what the detector
# was trained on.
#
# THIS IS A CONFOUND FOR TROOPS, and it is not small: a Mini P.E.K.K.A placed at
# engine y=4 read back at y=5 after 1.2 s, which is exactly the distance it
# walks in that time. Measuring a coordinate convention with a unit that moves
# means measuring the settle time as well. Probe with the CANNON -- a building
# does not move, so its reading is the placement and nothing else.
SETTLE_S = 1.2


def screencap(adb: Path, path: Path) -> Image.Image:
    raw = subprocess.run([str(adb), "exec-out", "screencap", "-p"],
                         capture_output=True, timeout=30).stdout
    path.write_bytes(raw)
    return Image.open(path).convert("RGB")


def hand_of(state) -> list[str]:
    """Hand slots 0-3. `cards[0]` is the Next preview -- see adapter._hand_ids."""
    return [c.name for c in state.cards[1:5]]


def describe(state, label: str) -> None:
    print(f"  {label}: screen={state.screen.name} "
          f"elixir={state.numbers.elixir.number}")
    print(f"    next={state.cards[0].name}   hand={hand_of(state)}   "
          f"ready={state.ready}")
    for side, units in (("ALLY", state.allies), ("ENEMY", state.enemies)):
        for u in units:
            p = u.position
            print(f"    {side:<5} {u.unit.name:<16} engine "
                  f"({p.tile_x},{p.tile_y + TILE_Y_OFFSET})  conf {p.conf:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slot", type=int, required=True, help="hand slot 0-3")
    ap.add_argument("--x", type=int, required=True, help="engine tile x")
    ap.add_argument("--y", type=int, required=True, help="engine tile y")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--settle", type=float, default=SETTLE_S,
                    help="seconds before reading the board back. Only "
                         "meaningful for a building; a troop walks.")
    ap.add_argument("--adb", type=Path, default=ADB)
    ap.add_argument("--out", type=Path, default=Path.cwd())
    args = ap.parse_args()

    tag = args.tag or f"s{args.slot}_{args.x}_{args.y}"
    args.out.mkdir(parents=True, exist_ok=True)

    detector = Detector(DECK)
    actuator = AdbActuator(dry_run=False, adb=args.adb)

    before_img = screencap(args.adb, args.out / f"probe_{tag}_before.png")
    before = detector.run(before_img.resize((SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT),
                                            Image.LANCZOS))
    print("\nBEFORE")
    describe(before, "state")

    if before.screen.name != "in_game":
        print(f"\n  not in a match (screen={before.screen.name}) -- aborting "
              "rather than tapping into a menu")
        return 1
    if args.slot not in before.ready:
        print(f"\n  slot {args.slot} ({hand_of(before)[args.slot]}) is not "
              f"affordable; ready={before.ready}. Placing anyway would measure "
              "a tap that the game ignores.")
        return 1

    intended_card = hand_of(before)[args.slot]
    print(f"\nPLACING slot {args.slot} ({intended_card}) at ENGINE "
          f"({args.x},{args.y})")
    print(f"  -> detector tile ({args.x},{args.y - TILE_Y_OFFSET})")
    card_tap, tile_tap = actuator.play(args.slot, args.x, args.y)
    # play() is non-blocking now, so without this the frame below would be
    # photographed before the card had landed.
    actuator.flush()
    print(f"  taps: card={card_tap} tile={tile_tap}")

    time.sleep(args.settle)
    after_img = screencap(args.adb, args.out / f"probe_{tag}_after.png")
    after = detector.run(after_img.resize((SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT),
                                          Image.LANCZOS))
    print("\nAFTER")
    describe(after, "state")

    # By NAME COUNT, not by (name, tile): every unit on the board moves between
    # the two frames, so matching on position reports movers as new. An earlier
    # version did exactly that and named a Musketeer that had merely walked.
    gained = Counter(u.unit.name for u in after.allies)
    gained.subtract(Counter(u.unit.name for u in before.allies))
    appeared = [name for name, n in gained.items() if n > 0]

    print("\nRESULT")
    if not appeared:
        print("  no new ally unit. The tap missed, the card was not selected, "
              "or the unit spawns nothing visible (a spell).")
    for name in appeared:
        for u in after.allies:
            if u.unit.name == name:
                p = u.position
                gx, gy = p.tile_x, p.tile_y + TILE_Y_OFFSET
                print(f"  {name:<16} landed engine ({gx},{gy})   "
                      f"intended ({args.x},{args.y})   "
                      f"delta ({gx - args.x:+d},{gy - args.y:+d})")
                break

    played = set(hand_of(before)) - set(hand_of(after))
    if played and intended_card not in played:
        print(f"  !! WRONG CARD: asked for {intended_card!r}, "
              f"the hand lost {sorted(played)}")
    print(f"\n  frames: probe_{tag}_before.png / probe_{tag}_after.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
