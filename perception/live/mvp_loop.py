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
from live.action_gate import MAX_STALENESS_MS, ActionGate  # noqa: E402
from live.actuator import AdbActuator  # noqa: E402
from live.adapter import build_game_state  # noqa: E402
from live.elixir_ledger import ElixirLedger  # noqa: E402
from live.pipeline import PerceptionWorker, Stages  # noqa: E402

DECK = [Cards.VALKYRIE, Cards.ARCHERS, Cards.MINIONS, Cards.CANNON,
        Cards.FIREBALL, Cards.GIANT, Cards.MUSKETEER, Cards.MINIPEKKA]

# A defensive tile in our own half, in ENGINE coordinates. Deliberately fixed:
# the point is to exercise the placement path, not to play well.
DEFAULT_TILE = (9, 8)

# Below this the loop is acting on a board that has already changed.
DECISION_HZ = 1.0

# MAX_STALENESS_MS is imported from action_gate rather than defined here, so the
# threshold that REPORTS staleness and the one that REFUSES to act on it cannot
# drift apart.


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


class NeuralPolicy:
    """The trained agent, reading a real screen.

    Everything it needs already exists: `perception_encoder` turns a GameState
    into the 13,606 floats it was trained on, and the masks it applies are its
    own methods reading that same vector -- so affordability and placement
    legality are computed exactly as they are in training rather than
    reimplemented here, which is what keeps rollout and deployment from
    drifting apart.

    HIDDEN STATE IS THE PART THAT IS EASY TO GET WRONG. The LSTM carries the
    match's history, so it must persist across frames and reset when a NEW
    match begins. Resetting every frame would silently reduce a recurrent
    policy to a reflex one, and every diagnostic would still look healthy --
    the same shape of failure as the team-1 observation bug.
    """

    def __init__(self, checkpoint: Path, deck_ability_slots: int = 0):
        import torch  # noqa: PLC0415

        import perception_encoder  # noqa: PLC0415
        from model import MicroRoyaleNet  # noqa: PLC0415

        self.torch = torch
        self.encoder = perception_encoder
        self.net = MicroRoyaleNet(num_ability_slots=deck_ability_slots)
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.net.load_state_dict(blob["model"] if "model" in blob else blob)
        self.net.eval()
        self.episodes = int(blob.get("episodes_completed", -1))
        self._hx = None
        self._cx = None
        self._was_in_game = False

    def reset_hidden(self) -> None:
        self._hx = self.torch.zeros(1, 256)
        self._cx = self.torch.zeros(1, 256)

    def decide(self, gs, ready, now: float) -> Decision:
        torch = self.torch
        if self._hx is None:
            self.reset_hidden()

        obs = torch.as_tensor(self.encoder.encode(gs)).unsqueeze(0)
        with torch.no_grad():
            feats, embeds, smap = self.net.extract_features(obs)
            logits, _a1, _a2, _v, (self._hx, self._cx) = self.net.step_lstm_and_card(
                feats, (self._hx, self._cx), self.net.affordability_mask(obs))
            card = torch.distributions.Categorical(logits=logits).sample()
            slot = int(card.item())
            if slot >= self.net.hand_size:
                # The last column is the always-legal no-op. Deliberately not
                # forced into a play: "hold elixir" is a real decision and the
                # affordability mask guarantees this column is never masked.
                return Decision(None, DEFAULT_TILE, "no-op")
            placement = self.net.placement_given_card(
                self._hx, embeds, card, obs, smap)
            placement = placement.masked_fill(
                ~self.net.placement_mask(obs, card), float("-inf"))
            cell = torch.distributions.Categorical(logits=placement).sample()
            x, y = self.net.cell_to_xy(cell)
        # cell_to_xy returns ENGINE board coordinates, which is what the
        # actuator's engine_tile_centre expects.
        return Decision(slot, (int(x.item()), int(y.item())),
                        f"net slot {slot} -> ({int(x.item())},{int(y.item())})")

    def on_screen_change(self, in_game: bool) -> None:
        """Reset the recurrent state when a match starts, never mid-match."""
        if in_game and not self._was_in_game:
            self.reset_hidden()
        self._was_in_game = in_game


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--act", action="store_true",
                    help="actually tap. Without it the loop only watches.")
    ap.add_argument("--window", default="BlueStacks App Player")
    ap.add_argument("--serial", action="store_true",
                    help="run perception inline instead of on its own thread. "
                         "The original design, kept for comparison: it pins "
                         "the decision RATE to the detector's LATENCY, "
                         "measured at 0.48 Hz on a quiet machine.")
    ap.add_argument("--ignore-staleness", action="store_true",
                    help="act on a board older than the staleness budget. "
                         "Idempotency still holds -- this only relaxes the age "
                         "check, which is the one that stops the agent playing "
                         "into a board the match has already moved past.")
    ap.add_argument("--policy", choices=("scripted", "neural"),
                    default="scripted",
                    help="scripted exercises the joints; neural is the agent")
    ap.add_argument("--checkpoint", type=Path,
                    default=Path(__file__).resolve().parents[2] / "python_ai"
                    / "model_weights_selfplay.pth")
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
    # Recorded in the run's own output: a timing log that does not say which
    # execution provider produced it cannot be compared against another.
    print(f"execution provider: "
          f"{detector.unit_detector.sess.get_providers()[0]}")
    actuator = AdbActuator(dry_run=not args.act)
    if args.policy == "neural":
        # Copied first: the live phase-2 run rewrites this file periodically
        # and reading it mid-write loads a truncated checkpoint.
        import shutil, tempfile  # noqa: PLC0415
        frozen = Path(tempfile.gettempdir()) / "mvp_policy_snapshot.pth"
        shutil.copy2(args.checkpoint, frozen)
        policy = NeuralPolicy(frozen)
        print(f"policy: TRAINED NET from {args.checkpoint.name} "
              f"(episode {policy.episodes})")
    else:
        policy = ScriptedPolicy()
        print("policy: scripted placeholder")
    ledger = ElixirLedger()
    gate = ActionGate(enforce_staleness=not args.ignore_staleness)

    print(f"capture {source.size[0]}x{source.size[1]}   "
          f"decision rate {DECISION_HZ} Hz   for {args.seconds:.0f}s\n")
    print("  t     screen     units  elix  spent  hand                    "
          "decision                 ms")

    # Offline this whole function medians 190 ms; the first live run implied
    # ~2700 ms. Timing it by stage is the only way to tell which stage grew,
    # and the answer decides whether the next lever is the execution provider,
    # the capture, or the adapter. Guessing cost a DirectML venv build that
    # would have addressed 4% of the budget.
    stages = Stages()

    def perceive(frame):
        """capture-frame -> (State, GameState). Runs on whichever thread owns
        perception: the worker when pipelined, the loop when --serial."""
        t = time.perf_counter()
        native = Image.fromarray(frame.image[:, :, ::-1])       # BGR -> RGB
        small = native.resize((SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT), Image.LANCZOS)
        t = stages.time("1 decode+resize", t)
        state = detector.run(small)
        t = stages.time("2 detector.run", t)
        if state is None:
            return None, None
        if state.screen.name == "in_game":
            ledger.update(state.numbers.elixir.number)
        gs, _report = build_game_state(
            state, np.array(native), np.array(small),
            frame_index=frame.index, wall_time_ms=frame.wall_time_ms,
            my_elixir_spent=ledger.spent)
        stages.time("3 build_game_state", t)
        return state, gs

    replay = iter(source.sample_every(DECISION_HZ)) if args.frames else None
    pipelined = replay is None and not args.serial
    worker = None
    if pipelined:
        # Replay is never pipelined: it is the deterministic integration test,
        # and a background thread racing a finite recording would reproduce
        # differently every run.
        worker = PerceptionWorker(source, detector, perceive)
        worker.start()
        print("perception: BACKGROUND THREAD (decisions use the newest board)")
    else:
        print("perception: inline" + (" [replay]" if replay else " [--serial]"))
    print()
    print("  t     screen     units  elix  spent  age   hand"
          "                     decision               ms")

    t0 = time.perf_counter()
    next_due = t0
    n = slow = stale = 0
    ages = []
    try:
        while time.perf_counter() - t0 < args.seconds:
            now = time.perf_counter()
            if replay is None:
                if now < next_due:
                    time.sleep(min(0.02, next_due - now))
                    continue
                next_due = now + 1.0 / DECISION_HZ

            step = time.perf_counter()
            age_ms = 0.0
            board_index = n
            if pipelined:
                snap = worker.latest()
                if snap is None:
                    if not worker.alive:
                        print(f"  perception thread DIED: {worker.last_error!r}")
                        break
                    continue
                state, gs = snap.state, snap.game_state
                age_ms = snap.age_ms(step)
                board_index = snap.index
            else:
                frame = (next(replay, None) if replay is not None
                         else source.read_new(timeout_s=1.0))
                if frame is None:
                    print("  (no frame)")
                    if replay is not None:
                        break
                    continue
                state, gs = perceive(frame)
                if state is None:
                    continue
                board_index = frame.index

            in_game = state.screen.name == "in_game"
            if hasattr(policy, "on_screen_change"):
                policy.on_screen_change(in_game)
            # The policy steps every tick whatever the gate decides: it is
            # recurrent and was trained stepping once per second, so skipping
            # steps to match the producer's rate would change the LSTM's
            # cadence away from training. Only the TAP is gated.
            decision = (policy.decide(gs, state.ready, now) if in_game
                        else Decision(None, DEFAULT_TILE, "not in game"))
            if decision.slot is not None:
                verdict = gate.check(board_index, age_ms)
                if verdict:
                    actuator.play(decision.slot, *decision.tile)
                    # Recorded only after the tap returns, so an adb failure
                    # leaves the board available to retry.
                    gate.record(board_index)
                else:
                    gate.refuse(verdict.reason)
                    decision = Decision(None, decision.tile,
                                        f"[{verdict.reason}] {decision.why}")

            ms = (time.perf_counter() - step) * 1000.0
            slow += ms > 1000.0
            stale += age_ms > MAX_STALENESS_MS
            ages.append(age_ms)
            # [1:5], not [:4] -- cards[0] is the "Next" preview. See
            # adapter._hand_ids.
            hand = ",".join(c.name[:6] for c in state.cards[1:5])
            print(f"  {now - t0:5.1f} {state.screen.name[:10]:<10} "
                  f"{len(gs.units):>5}  {gs.my_elixir:>4.0f}  {ledger.spent:>5.0f}  "
                  f"{age_ms:>4.0f}  {hand:<30} {decision.why:<22} {ms:>5.0f}")
            n += 1
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if worker is not None:
            worker.stop()
        actuator.close()
        source.close()

    elapsed = time.perf_counter() - t0
    print(f"\n" + f"{n} decisions in {elapsed:.0f}s   {n / max(elapsed, 1e-9):.2f} Hz"
          f"   decision over 1000ms: {slow}/{max(n, 1)}")
    if worker is not None:
        print(f"perception thread: {worker.frames} boards "
              f"({worker.frames / max(elapsed, 1e-9):.2f} Hz), errors {worker.errors}")
        if ages:
            a = np.array(ages)
            print(f"board age: mean {a.mean():.0f} ms  median {np.median(a):.0f}"
                  f"  p95 {np.percentile(a, 95):.0f}  max {a.max():.0f}"
                  f"   over {MAX_STALENESS_MS:.0f} ms on {stale}/{max(n, 1)}")
        print("\nproducer period, by stage (median):")
        print(worker.stages.report())
    print("\nperceive, by stage (median):")
    print(stages.report(total_key=None))
    print(f"\n{gate.summary()}")
    print(f"taps issued: {len(actuator.taps)}"
          f"{'' if args.act else ' (dry run - none sent)'}")
    if actuator.dropped or actuator.errors:
        # Dropped means a placement was still being tapped when the next was
        # chosen. Rare at 1 Hz against a ~900 ms placement, and a real signal
        # if it is not: the actuator has become the bottleneck again.
        print(f"actuator: {actuator.dropped} dropped (still tapping), "
              f"{actuator.errors} errors"
              + (f" - last {actuator.last_error!r}" if actuator.last_error else ""))
    print(f"our elixir spent: {ledger.spent:.0f} over {ledger.cards} cards, "
          f"residual {ledger.residual:+.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
