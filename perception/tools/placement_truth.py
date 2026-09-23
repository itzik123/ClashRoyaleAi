"""Does a placement at this engine cell actually deploy? Measured, one trial at a
time.

    perception/.venv/Scripts/python.exe perception/tools/placement_truth.py \
        --rows 1,2,3,8 --xs 0,4,9,14,17 --repeat 1

A live match confounds placement legality with elixir management: an agent
playing faster than it can afford fails taps whatever cell they name. Probing
only from a full bar removes the confound.

Elixir caps at 10, so from the cap:

    the bar is still 10 two seconds later   ->  nothing was spent, REFUSED
    the bar has left the cap                ->  a card was played, ACCEPTED

No regeneration term and no subset-sum over ambiguous drops, the things that
stop `ElixirLedger` answering this. Two secondary oracles are recorded beside
it (the hand slot cycling, a unit of the expected type appearing);
disagreements are printed, not voted away.
"""
from __future__ import annotations

import argparse
import json
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
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from clashroyalebuildabot.constants import (  # noqa: E402
    SCREENSHOT_HEIGHT,
    SCREENSHOT_WIDTH,
)
from live.actuator import ADB, AdbActuator, engine_tile_centre  # noqa: E402
from live.placement_confirm import expected_unit_names  # noqa: E402
from match_nav import ensure_in_match, screencap  # noqa: E402

# Long enough for the two taps (~900 ms) to land and the board to be drawn,
# short enough that regeneration cannot refill the bar: 2.2 s buys at most 0.79
# elixir, below any card's cost.
SETTLE_S = 2.2

CAP = 10.0


def read(detector, adb: Path):
    img = screencap(adb)
    return detector.run(img.resize((SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT),
                                   Image.LANCZOS))


def own_counts(state) -> Counter:
    return Counter(u.unit.name for u in state.allies)


def hand_names(state) -> list[str]:
    """Slots 0-3. cards[0] is the Next preview (adapter._hand_ids)."""
    return [c.name for c in state.cards[1:5]]


def wait_for_cap(detector, adb: Path, timeout_s: float = 45.0):
    """Block until the bar reads 10, or the match ends. Returns the State."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = read(detector, adb)
        if state is None:
            time.sleep(0.5)
            continue
        if state.screen.name != "in_game":
            return state
        if state.numbers.elixir.number >= CAP:
            return state
        time.sleep(1.0)
    return None


def pick_slot(state, want_spell: bool = False) -> int | None:
    """A hand slot holding a card that spawns a body (or a spell if asked), so all
    three oracles can speak.
    """
    for slot, name in enumerate(hand_names(state)):
        if name in ("", "blank"):
            continue
        spawns = expected_unit_names(name)
        if bool(spawns) != want_spell:
            return slot
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", default="1,2,3,5,8,12,15",
                    help="ENGINE rows to probe")
    ap.add_argument("--xs", default="0,4,9,14,17", help="engine columns")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--adb", type=Path, default=ADB)
    ap.add_argument("--settle", type=float, default=SETTLE_S)
    ap.add_argument("--out", type=Path, default=Path("placement_truth.jsonl"))
    ap.add_argument("--max-trials", type=int, default=0,
                    help="stop after this many trials (0 = no limit)")
    ap.add_argument("--pixel-scan", default=None,
                    help="calibration mode: 'X:Y0,Y1,Y2,...' taps those RAW "
                         "screen pixels instead of tile centres, which is the "
                         "only way to locate the arena's deployable edge "
                         "without assuming the tile grid that is under test")
    args = ap.parse_args()

    from clashroyalebuildabot.detectors.detector import Detector  # noqa: PLC0415
    from clashroyalebuildabot.namespaces.cards import Cards  # noqa: PLC0415
    deck = [Cards.VALKYRIE, Cards.ARCHERS, Cards.MINIONS, Cards.CANNON,
            Cards.FIREBALL, Cards.GIANT, Cards.MUSKETEER, Cards.MINIPEKKA]
    detector = Detector(deck)
    actuator = AdbActuator(dry_run=False, adb=args.adb)
    print(f"actuator backend: {actuator.backend}")

    pixel_mode = args.pixel_scan is not None
    if pixel_mode:
        col, ys = args.pixel_scan.split(":")
        trials = [(int(col), int(v)) for _ in range(args.repeat)
                  for v in ys.split(",") if v.strip()]
        print(f"{len(trials)} PIXEL trials at x={col}, y={ys}\n")
    else:
        rows = [int(v) for v in args.rows.split(",") if v.strip()]
        xs = [int(v) for v in args.xs.split(",") if v.strip()]
        trials = [(x, y) for _ in range(args.repeat) for y in rows for x in xs]
        print(f"{len(trials)} trials: rows {rows} x columns {xs} "
              f"x {args.repeat}\n")
    if args.max_trials:
        trials = trials[:args.max_trials]
    print("   #  cell     tap        card         elixir      "
          "spent hand unit  verdict")

    results = []
    for i, (x, y) in enumerate(trials):
        ensure_in_match(detector, args.adb, verbose=False)
        state = wait_for_cap(detector, args.adb)
        if state is None or state.screen.name != "in_game":
            # The match ended mid-wait; ensure_in_match starts a new one and
            # the trial is retried.
            print(f"  {i:>3}  ({x:>2},{y:>2})  -- match ended while waiting, "
                  f"re-queueing")
            trials.append((x, y))
            continue

        slot = pick_slot(state)
        if slot is None:
            print(f"  {i:>3}  ({x:>2},{y:>2})  -- no readable unit card in hand")
            continue
        card = hand_names(state)[slot]
        before_elixir = float(state.numbers.elixir.number)
        before_units = own_counts(state)
        before_hand = hand_names(state)

        if pixel_mode:
            from live.actuator import Tap  # noqa: PLC0415
            tap = Tap(x, y)
            actuator.play_pixel(slot, x, y)
        else:
            tap = engine_tile_centre(x, y)
            actuator.play(slot, x, y)
        actuator.flush()
        time.sleep(args.settle)

        after = read(detector, args.adb)
        if after is None or after.screen.name != "in_game":
            print(f"  {i:>3}  ({x:>2},{y:>2})  -- match ended during settle")
            trials.append((x, y))
            continue
        after_elixir = float(after.numbers.elixir.number)
        after_units = own_counts(after)
        after_hand = hand_names(after)

        # The primary oracle: started at the cap, so any departure from it is
        # spend.
        spent = after_elixir < CAP - 0.5
        hand_changed = (after_hand[slot] != before_hand[slot]
                        and after_hand[slot] not in ("", "blank")
                        and before_hand[slot] not in ("", "blank"))
        expect = expected_unit_names(card)
        gained = sum(max(0, after_units[n] - before_units[n]) for n in expect)
        unit_ok = gained >= 1

        verdict = "ACCEPTED" if spent else "REFUSED"
        flag = ""
        if spent != hand_changed:
            flag = "  <- elixir/hand DISAGREE"
        results.append({
            "x": x, "y": y, "tap_x": tap.x, "tap_y": tap.y, "slot": slot,
            "card": card, "before_elixir": before_elixir,
            "after_elixir": after_elixir, "spent": spent,
            "hand_changed": hand_changed, "unit_gained": gained,
            "unit_ok": unit_ok, "verdict": verdict,
        })
        print(f"  {i:>3}  ({x:>2},{y:>2})  ({tap.x:>4},{tap.y:>4})  "
              f"{card:<12} {before_elixir:>2.0f} -> {after_elixir:>2.0f}   "
              f"{str(spent)[:1]:<5} {str(hand_changed)[:1]:<4} "
              f"{gained:<4}  {verdict}{flag}")

        args.out.write_text("\n".join(json.dumps(r) for r in results))

    actuator.close()

    print("\n" + "=" * 72)
    by_row: dict[int, list[dict]] = {}
    for r in results:
        by_row.setdefault(r["y"], []).append(r)
    label = "SCREEN PIXEL y" if pixel_mode else "ENGINE row"
    print(f"acceptance by {label} (elixir oracle):")
    for y in sorted(by_row):
        rs = by_row[y]
        ok = sum(r["spent"] for r in rs)
        print(f"  {label} {y:>4}: {ok:>2}/{len(rs):<2} accepted"
              f"   ({ok / len(rs):.0%})"
              + ("   <<< REFUSED" if ok == 0 else ""))
    agree = sum(r["spent"] == r["hand_changed"] for r in results)
    print(f"\nelixir and hand oracles agree on {agree}/{len(results)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
