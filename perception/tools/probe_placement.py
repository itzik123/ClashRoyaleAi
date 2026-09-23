"""Place one card at a named engine tile and report where it actually landed.

    perception/.venv-dml/Scripts/python.exe perception/tools/probe_placement.py \
        --slot 0 --x 4 --y 8

`mvp_loop --act` places many cards under time pressure with the board changing,
the worst instrument for a coordinate convention. This places one named card at
one named tile on a quiet board, through the real `AdbActuator`, since
`engine_tile_centre` is what is under test.

The round trip is self-consistent by construction (the actuator subtracts
TILE_Y_OFFSET and the adapter adds it back), so a matching y does not confirm
the offset. It catches everything else: a wrong x, a scaling error, a tap on a
different tile, no placement at all, or the wrong card played. Judging the
offset needs a landmark, so both frames are kept as PNGs: place at y=16, just
below the river ([15.5, 17.5)), and see whether the unit stands at the water's
edge or one row back.
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
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))

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

# How long to wait for the placed unit to exist and be detectable. A troop
# walks during the wait (a Mini P.E.K.K.A placed at y=4 read back at y=5 after
# 1.2 s), so probe with a building such as the Cannon, which does not move.
SETTLE_S = 1.2


def screencap(adb: Path, path: Path) -> Image.Image:
    # Strip the adb banner before writing or decoding
    # (match_nav.decode_screencap): the bytes are saved and read back, so an
    # unstripped write corrupts both.
    from match_nav import _PNG_MAGIC, decode_screencap  # noqa: PLC0415

    p = subprocess.run([str(adb), "exec-out", "screencap", "-p"],
                       capture_output=True, timeout=60)
    img = decode_screencap(p.stdout, p.returncode, p.stderr)
    path.write_bytes(p.stdout[p.stdout.find(_PNG_MAGIC):])
    return img


def hand_of(state) -> list[str]:
    """Hand slots 0-3. `cards[0]` is the Next preview (adapter._hand_ids)."""
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
    # play() is non-blocking, so without this the frame below would be taken
    # before the card lands.
    actuator.flush()
    print(f"  taps: card={card_tap} tile={tile_tap}")

    time.sleep(args.settle)
    after_img = screencap(args.adb, args.out / f"probe_{tag}_after.png")
    after = detector.run(after_img.resize((SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT),
                                          Image.LANCZOS))
    print("\nAFTER")
    describe(after, "state")

    # By name count, not (name, tile): every unit moves between the two frames,
    # so position matching reports movers as new.
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
