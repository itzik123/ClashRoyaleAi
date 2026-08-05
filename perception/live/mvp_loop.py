"""Minimal end-to-end loop: screen -> GameState -> decision -> tap.

WHY A STUPID POLICY ON PURPOSE
------------------------------
The trained policy consumes a 13,606-float observation, and the encoder that
would build one from a `GameState` does not exist yet. Waiting for it before
running anything end to end is how a pipeline accumulates six components that
have each been measured alone and have never been in the same process
together.

So the decision here is a hand-written rule, and every OTHER joint is real:
live capture, the detector, the adapter, the tile->screen conversion and the
tap. Those are the joints that can be structurally wrong. The policy is the one
part already known to work -- it just needs the encoder to reach it, and it
drops in behind `Policy.decide` when that lands.

WHAT THIS IS FOR
----------------
Finding integration faults early, in the open. Specifically the ones that are
invisible from inside a single component:

  * a tap landing on the wrong tile produces a perfectly valid GameState next
    frame, showing a unit somewhere nobody intended;
  * the detector has only ever been run on recorded frames, never against a
    live match while competing with capture for CPU;
  * the whole chain has to fit inside the 1 Hz the policy acts at, and each
    piece was timed alone.

DRY RUN IS THE DEFAULT
----------------------
This can place real cards in a real match. Acting has to be asked for with
`--act`, so a forgotten flag means the loop watches rather than plays.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# The engine binding lives in python_ai/ and is NOT importable by default.
# `live/unit_to_card.py` needs it for the card registry, and the tests only get
# away without this because conftest's `engine` fixture adds the path. Any real
# entry point has to do it itself -- which is exactly the kind of gap that only
# shows up the first time the pieces run in one process.
_ENGINE = _ROOT.parent / "python_ai"
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

from capture.window import WindowSource  # noqa: E402
from clashroyalebuildabot.constants import (  # noqa: E402
    SCREENSHOT_HEIGHT,
    SCREENSHOT_WIDTH,
)
from clashroyalebuildabot.detectors.detector import Detector  # noqa: E402
from clashroyalebuildabot.namespaces.cards import Cards  # noqa: E402
from live.actuator import AdbActuator  # noqa: E402
from live.adapter import build_game_state  # noqa: E402
from live.elixir_ledger import ElixirLedger  # noqa: E402

DECK = [Cards.VALKYRIE, Cards.ARCHERS, Cards.MINIONS, Cards.CANNON,
        Cards.FIREBALL, Cards.GIANT, Cards.MUSKETEER, Cards.MINIPEKKA]

# A defensive tile in our own half, in ENGINE coordinates. Deliberately fixed:
# the point is to exercise the placement path, not to play well.
DEFAULT_TILE = (9, 8)

# Below this the loop is acting on a board that has already changed.
DECISION_HZ = 1.0


@dataclass
class Decision:
    slot: int | None
    tile: tuple[int, int]
    why: str


class ScriptedPolicy:
    """Play the cheapest ready card, on a cooldown. A placeholder with a
    deliberately obvious name, so nobody mistakes it for the agent."""

    def __init__(self, cooldown_s: float = 3.0, tile=DEFAULT_TILE):
        self.cooldown_s = cooldown_s
        self.tile = tile
        self._last_play = -1e9

    def decide(self, gs, ready, now: float) -> Decision:
        if now - self._last_play < self.cooldown_s:
            return Decision(None, self.tile, "cooldown")
        if not ready:
            return Decision(None, self.tile, "nothing ready")
        slot = int(min(ready))
        self._last_play = now
        return Decision(slot, self.tile, f"cheapest ready slot {slot}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--act", action="store_true",
                    help="actually tap. Without it the loop only watches.")
    ap.add_argument("--window", default="BlueStacks App Player")
    ap.add_argument("--frames", type=Path, default=None,
                    help="replay a tools/record_match.py directory instead of "
                         "capturing live. The rest of the chain is identical, "
                         "so this is the integration test the live path cannot "
                         "be: deterministic, repeatable, and runnable with no "
                         "emulator.")
    args = ap.parse_args()

    if args.frames and args.act:
        raise SystemExit("--act makes no sense while replaying a recording")

    print(f"{'ACTING - will place real cards' if args.act else 'DRY RUN - watching only'}")
    if args.frames:
        from capture.frames import RecordingSource  # noqa: PLC0415
        source = RecordingSource(args.frames)
        print(f"replaying {source.frame_count} frames from {args.frames}")
    else:
        source = WindowSource(args.window)
    detector = Detector(DECK)
    actuator = AdbActuator(dry_run=not args.act)
    policy = ScriptedPolicy()
    ledger = ElixirLedger()

    print(f"capture {source.size[0]}x{source.size[1]}   "
          f"decision rate {DECISION_HZ} Hz   for {args.seconds:.0f}s\n")
    print("  t     screen     units  elix  spent  hand                    "
          "decision                 ms")

    # Replay walks the recording at its own pace; live paces itself to
    # DECISION_HZ. Everything downstream of `frame` is identical.
    replay = iter(source.sample_every(DECISION_HZ)) if args.frames else None

    t0 = time.perf_counter()
    next_due = t0
    n = 0
    slow = 0
    try:
        while time.perf_counter() - t0 < args.seconds:
            now = time.perf_counter()
            if replay is None:
                if now < next_due:
                    time.sleep(min(0.02, next_due - now))
                    continue
                next_due = now + 1.0 / DECISION_HZ

            step = time.perf_counter()
            if replay is not None:
                frame = next(replay, None)
                if frame is None:
                    print("  (recording exhausted)")
                    break
            else:
                frame = source.read_new(timeout_s=1.0)
                if frame is None:
                    print("  (no new frame)")
                    continue
            native = Image.fromarray(frame.image[:, :, ::-1])   # BGR -> RGB
            small = native.resize((SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT), Image.LANCZOS)
            state = detector.run(small)

            in_game = state is not None and state.screen.name == "in_game"
            if in_game:
                ledger.update(state.numbers.elixir.number)
            gs, report = build_game_state(
                state, np.array(native), np.array(small),
                seconds_elapsed=now - t0, frame_index=frame.index,
                wall_time_ms=frame.wall_time_ms,
                my_elixir_spent=ledger.spent)

            decision = (policy.decide(gs, state.ready, now) if in_game
                        else Decision(None, DEFAULT_TILE, "not in game"))
            if decision.slot is not None:
                actuator.play(decision.slot, *decision.tile)

            ms = (time.perf_counter() - step) * 1000.0
            if ms > 1000.0:
                slow += 1
            hand = (",".join(c.name[:6] for c in state.cards[:4])
                    if state else "-")
            print(f"  {now - t0:5.1f} {state.screen.name[:10]:<10} "
                  f"{len(gs.units):>5}  {gs.my_elixir:>4.0f}  {ledger.spent:>5.0f}  "
                  f"{hand:<30} {decision.why:<22} {ms:>5.0f}")
            n += 1
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        source.close()

    elapsed = time.perf_counter() - t0
    print(f"\n{n} iterations in {elapsed:.0f}s   "
          f"{n / max(elapsed, 1e-9):.2f} Hz   over budget on {slow}/{max(n,1)}")
    print(f"taps issued: {len(actuator.taps)}"
          f"{'' if args.act else ' (dry run - none sent)'}")
    print(f"our elixir spent: {ledger.spent:.0f} over {ledger.cards} cards, "
          f"residual {ledger.residual:+.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
