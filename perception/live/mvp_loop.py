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
from dataclasses import dataclass, replace
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


# The longest the loop will hold a decision back to get a fresher board. Caps
# the cadence jitter a wait introduces: 150 ms on a 1000 ms tick is 15%, and
# the phase-lock means it only happens while converging, not every tick.
FRESHNESS_WAIT_CAP_S = 0.15

# Added to the predicted arrival so a board landing slightly late is still
# caught. Without it the wait expires just before the thing it is waiting for.
FRESHNESS_WAIT_SLACK_S = 0.03


def wait_for_fresher(worker, snap):
    """Hold the decision briefly if a newer board is about to land.

    THE WASTE THIS REMOVES. The producer publishes at its own rate and the
    decision loop samples at 1 Hz on an unrelated phase, so the board being
    acted on has typically been sitting finished for part of a producer period.
    That time is pure loss: it ages the world model without buying anything.

    Waiting for the next board shrinks the gap between "when the board was
    captured" and "when the card lands", because the newer board is captured
    later while the card lands only slightly later. The gap is what decides
    whether a counter meets the Battle Ram or arrives behind it.

    WHY IT IS BOUNDED, AND CONDITIONAL. Waiting is not free -- the action does
    land later in wall time -- so it is only worth it when the board in hand is
    already stale and the next is imminent. If the board just arrived, the next
    one is a whole period away and waiting would trade a lot of delay for
    nothing. Hence: wait only when the predicted arrival is within
    FRESHNESS_WAIT_CAP_S.

    Returns (snapshot, waited_ms).
    """
    period = worker.period
    if period is None:
        # Not enough history to predict an arrival. Acting now is the safe
        # default; waiting on a guessed period waits for nothing.
        return snap, 0.0

    started = time.perf_counter()
    eta = period - snap.sat_ms(started) / 1000.0
    if not 0.0 < eta <= FRESHNESS_WAIT_CAP_S:
        return snap, 0.0

    deadline = started + min(eta + FRESHNESS_WAIT_SLACK_S, FRESHNESS_WAIT_CAP_S)
    while time.perf_counter() < deadline:
        newer = worker.latest()
        if newer is not None and newer.index != snap.index:
            return newer, (time.perf_counter() - started) * 1000.0
        time.sleep(0.004)
    return snap, (time.perf_counter() - started) * 1000.0


def deck_costs(deck) -> tuple[tuple[float, ...], list[str]]:
    """The distinct elixir costs in `deck`, and anything worth complaining about.

    The ledger explains an elixir drop by decomposing it into card costs, so its
    cost table has to be the DECK's. Left at its default of (3, 4, 5) -- which
    happens to be exactly this deck's profile -- a deck containing a 2 or a 6
    would produce drops matching no legal combination, and those placements
    would vanish from the ledger silently. Not an error, not a warning: simply
    absent, with `my_elixir_spent` drifting further off for the rest of the
    match. That is the whole reason this is derived rather than assumed.

    Costs come from the ENGINE registry, because that is what fills the
    observation's cost scalars and therefore what `affordability_mask` gates on.
    A ledger disagreeing with the mask about what a card costs would be a second
    source of truth for the same number.

    CRBAB carries its own cost per card, so the two are cross-checked. They
    describe the same real game and a disagreement means one of them is wrong
    about it -- worth saying out loud rather than silently preferring either.
    """
    import clash_royale_env as engine  # noqa: PLC0415

    from live.unit_to_card import (  # noqa: PLC0415
        UNKNOWN_CARD_SIM_ID,
        hand_card_id_for,
    )

    costs: set[float] = set()
    warnings: list[str] = []
    for card in deck:
        crbab = float(card.cost)
        sim_id = hand_card_id_for(card.name)
        engine_cost = None
        if sim_id != UNKNOWN_CARD_SIM_ID:
            try:
                engine_cost = float(engine.get_card_info(sim_id)["cost"])
            except Exception:                               # noqa: BLE001
                engine_cost = None
        if engine_cost is None:
            # Fall back rather than drop it: a missing cost removes a whole
            # card's worth of explanations from the table, which is the exact
            # silent loss this function exists to prevent.
            costs.add(crbab)
            warnings.append(f"{card.name}: not in the engine registry, "
                            f"using CRBAB's cost {crbab:.0f}")
        else:
            costs.add(engine_cost)
            if engine_cost != crbab:
                warnings.append(f"{card.name}: engine says {engine_cost:.0f}, "
                                f"CRBAB says {crbab:.0f}")
    return tuple(sorted(costs)), warnings


def hand_cost(gs, slot: int) -> float | None:
    """What the card in `slot` costs, from the engine's own registry.

    Via the binding rather than a cost table copied into this file -- the
    duplicated-constant drift CLAUDE.md names as having gone stale twice.
    Returns None for an unreadable slot rather than guessing, so an unknown
    card cannot silently debit the wrong amount.
    """
    try:
        import clash_royale_env as engine  # noqa: PLC0415

        card_id = gs.my_hand[slot]
        if card_id is None or card_id < 0:
            return None
        return float(engine.get_card_info(card_id)["cost"])
    except Exception:                                   # noqa: BLE001
        return None


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
        # Which placement cells the actuator can actually reach, computed once
        # from the tile geometry (see actuator.engine_row_is_tappable). Built
        # here rather than per decision because it is a constant of the screen
        # mapping, and asserted non-empty so a future geometry change that
        # masked EVERY cell would fail loudly instead of turning the agent into
        # a permanent no-op.
        from live.actuator import engine_row_is_tappable  # noqa: PLC0415
        cells = self.net.placement_cells
        width = self.net.board_width
        self._tappable_cells = torch.tensor(
            [engine_row_is_tappable(c // width) for c in range(cells)],
            dtype=torch.bool).unsqueeze(0)
        n_ok = int(self._tappable_cells.sum())
        if n_ok == 0:
            raise RuntimeError(
                "no placement cell is tappable -- the tile geometry and the "
                "engine frame disagree completely; check TILE_Y_OFFSET.")
        print(f"placement: {n_ok}/{cells} cells reachable by the actuator "
              f"({cells - n_ok} engine rows have no detector row)")

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
            # SECOND mask, live-only: model.py's placement_mask is built from
            # the ENGINE's bounds, which include rows this screen mapping cannot
            # reach. Engine row 0 converts to detector row -1, whose tap lands
            # below the arena -- the game drops the placement silently and the
            # elixir ledger reports it as "issued but never confirmed".
            #
            # Applied here and not in model.py on purpose: this is a property of
            # the ACTUATOR, not of the game. Narrowing the training action space
            # would change what the policy learns and invalidate its win-rate
            # history, to fix something that only exists on this screen.
            placement = placement.masked_fill(
                ~self._tappable_cells.to(placement.device), float("-inf"))
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
    if args.act:
        print(f"actuator backend: {actuator.backend}")
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
    costs, cost_warnings = deck_costs(DECK)
    for warning in cost_warnings:
        print(f"  !! card cost: {warning}")
    print(f"deck costs: {', '.join(f'{c:.0f}' for c in costs)}")
    ledger = ElixirLedger(costs=costs)
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
            # The reading's own capture time, not now(): the ledger models how
            # much elixir regenerated between samples, and at this rate a
            # frame's worth of latency is a quarter of an elixir.
            ledger.update(state.numbers.elixir.number,
                          now=frame.wall_time_ms / 1000.0)
        gs, _report = build_game_state(
            state, np.array(native), np.array(small),
            frame_index=frame.index, wall_time_ms=frame.wall_time_ms,
            my_elixir_spent=ledger.spent)
        # OPTIMISTIC DEBIT. The bar the agent is reading is ~1 s old and its
        # last tap needs another ~0.9 s to land, so cards it has already
        # committed are still shown as affordable -- and it spends the same
        # elixir two and three times over. Measured: four placements against a
        # single unchanged reading of 10, which is also what fills the
        # actuator queue and gets taps dropped.
        #
        # Subtracting what we have promised but not yet seen leave the bar is
        # not a fiction, it is the better estimate, and it is what a human does
        # without thinking about it. `my_elixir` drives affordability_mask, so
        # this is the value the policy is actually gated on.
        owed = ledger.unconfirmed_cost
        if owed:
            gs = replace(gs, my_elixir=max(0.0, gs.my_elixir - owed))
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
    waits: list[float] = []
    try:
        while time.perf_counter() - t0 < args.seconds:
            now = time.perf_counter()
            if replay is None:
                if now < next_due:
                    time.sleep(min(0.02, next_due - now))
                    continue

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
                snap, waited_ms = wait_for_fresher(worker, snap)
                if waited_ms:
                    waits.append(waited_ms)
                step = time.perf_counter()
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

            # Phase-locked, not on a fixed grid: the next tick is measured from
            # the decision that actually happened, so a wait shifts the whole
            # cadence rather than being repaid by a short interval afterwards.
            # Once aligned to the producer, boards arrive just before each tick
            # and the wait stops triggering by itself.
            if replay is None:
                next_due = time.perf_counter() + 1.0 / DECISION_HZ

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
                    # Ground truth: we know exactly which card and what it
                    # cost, which no amount of staring at the bar can recover
                    # once 3 and 4 quantise to the same drop.
                    # Only a tap that was actually SENT spends elixir. In dry
                    # run nothing reaches the game, and on a replay the elixir
                    # being read is a human's -- recording our imaginary plays
                    # there would attribute their drops to us and corrupt the
                    # very trace the ledger is validated against.
                    cost = (None if actuator.dry_run
                            else hand_cost(gs, decision.slot))
                    if cost is not None:
                        # No timestamp: the ledger stamps it with its own frame
                        # clock. Passing a wall clock here would mix time bases
                        # with the readings and nothing would ever expire.
                        ledger.record_play(cost)
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
        p = worker.period
        print(f"producer period: {p * 1000:.0f} ms" if p else
              "producer period: not measured")
        if waits:
            w = np.array(waits)
            print(f"freshness waits: {len(waits)}/{max(n, 1)} decisions, "
                  f"mean {w.mean():.0f} ms  max {w.max():.0f} ms")
        else:
            print("freshness waits: none (boards already arriving on phase)")
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
          f"residual {ledger.residual:+.0f}, "
          f"{ledger.rejected} issued plays never confirmed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
