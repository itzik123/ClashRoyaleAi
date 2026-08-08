"""Measure the real game's spell delay and troop deploy time.

Two constants the engine states but has never had checked against reality:

    spell delay    `spell(7, "Fireball", ..., 10, ...)` -- a FLAT 10 ticks
                   (1.0 s) regardless of where it is cast. Measured here at
                   two distances, because if the real delay scales with cast
                   distance then a constant is STRUCTURALLY wrong, not merely
                   mistuned, and no amount of retuning it helps.
    deploy time    the engine has NONE. `spawnEntity` makes an entity live
                   immediately, while the real game freezes a troop for about
                   a second after it lands.

Both matter more since 2026-08-07 than they did before it. Troops now move at
real-game speed, so a spell's 1.0 s window covers 5x less ground than it used
to -- the Fireball shaping was tuned in a world where its targets fled five
times too fast, and its calibration does not survive that unchanged.

WHY THE ELIXIR BAR IS THE CAST CLOCK
------------------------------------
It drops the instant the GAME accepts the play, so it is measured on the far
side of every latency this project has fought: ADB, the actuator queue, the
emulator's own input handling. A tap timestamp would measure our pipeline; the
elixir bar measures the game.

WHY A TOWER IS THE IMPACT CLOCK
-------------------------------
It cannot move, cannot be mistaken for another unit, and its HP is a clean
step. Scoring impact off troops would need the detector to re-identify a unit
across the frame it takes damage on, which is the association problem this
project deliberately avoids everywhere else.

STATUS: CAPTURE WORKS, THE IMPACT CLOCK DOES NOT (2026-08-07)
-------------------------------------------------------------
A live Training Camp session got as far as landing a Fireball exactly on the
opponent's left Princess Tower -- confirmed by eye in the captured frames --
at 20 fps, i.e. +/-25 ms. **No delay came out of it**, for one reason:

`TowerHpReader`/CRBAB's bar reader is not usable as an impact clock. Over one
trial, both of OUR princess towers read 0.00 for all 77 frames (0.0 means
"could not match the bar colours", not "destroyed" -- see live/adapter.py),
and the target tower read 1.00 -> 0.00 -> 0.62 -> 0.67 -> 1.00, none of which
was the Fireball. The elixir reader is better but also glitches: a true
10 -> 6 drop was read as 10 -> 0 -> 6, so the cast clock needs debouncing
against a stable post-drop value rather than the first sample that moves.

**The frames already contain the answer.** Tower HP is printed as a NUMERAL
beside each bar and is plainly legible at this capture size (2030, 2446).
live/adapter.py already argues this is the real fix -- absolute HP, no colour
matching, no occlusion ambiguity, no dependence on tower level -- and
readers/clock.py carries a digit-template classifier built for exactly this
kind of glyph. That is the next step, and it is worth doing anyway: it fixes
the tower reader for the whole project, not just this measurement.

A SECOND FINDING, WORTH MORE THAN THE MEASUREMENT
-------------------------------------------------
**The hand only cycles when you play.** Waiting for a specific card while
playing nothing is a deadlock, not a wait. Two entire matches were spent
watching one frozen hand -- `['minipekka','musketeer','valkyrie','minions']`,
unchanged for 150 s -- which looked exactly like a broken detector. Hence
`CYCLE_TILE`: if the wanted card is not dealt, spend one to advance the cycle.

CAPTURE AND ANALYSIS ARE SEPARATE PHASES ON PURPOSE
---------------------------------------------------
`--capture` records short bursts of raw frames around each play and writes
them to disk; `--analyse` runs the detector over them afterwards. Running the
detector live would compete with capture for the CPU and lower the frame rate,
which is the measurement's resolution. It also makes the live session short
and scripted rather than exploratory, and lets the same trial be re-analysed
without being re-played.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

_PERCEPTION_ROOT = Path(__file__).resolve().parent.parent
if str(_PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(_PERCEPTION_ROOT))

from loguru import logger  # noqa: E402

logger.remove()

from PIL import Image  # noqa: E402

# Engine tiles. The opponent's left Princess Tower is the deep target; a spell
# is legal anywhere, so `engine_frame=True` placement reaches it.
DEEP_TARGET = (4, 27)
# Just past our own bridge -- the shortest cast that still has an enemy-side
# structure nowhere near it, used only when a near target is available.
NEAR_TARGET = (4, 18)
# Own half, clear of both Princess Towers (x=4, x=14) and the King (x=9, y=2).
DEPLOY_TILE = (11, 9)
# Where cards get dumped purely to advance the deck cycle. Back corner of our
# own half: it is Training Camp, the match outcome is irrelevant, and the only
# requirement is that it does not sit on top of the tile a trial is measuring.
CYCLE_TILE = (1, 4)
# The dearest card in DEFAULT_DECK. Cycling is only attempted with at least
# this much elixir, so a dump never fails silently for being unaffordable --
# playCard returns false with no exception, which is exactly the failure the
# affordability mask exists to prevent on the training side.
MAX_DECK_COST = 5.0

PRINCESS_KEYS = ("left_enemy_princess_hp", "right_enemy_princess_hp",
                 "left_ally_princess_hp", "right_ally_princess_hp")


class BurstRecorder:
    """Captures frames on a thread into memory, for a bounded window.

    Bounded because raw frames are ~1.6 MB each: a whole session in memory is
    gigabytes, while the four seconds around one play is ~100 MB and is all
    the measurement ever looks at.
    """

    def __init__(self, source):
        self._source = source
        self.frames: list[tuple[float, np.ndarray]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.frames = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            frame = self._source.read_new(timeout_s=0.5)
            if frame is not None:
                self.frames.append((frame.wall_time_ms, frame.image.copy()))

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def save(self, out_dir: Path, label: str, meta: dict) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        stamps = []
        for i, (ms, image) in enumerate(self.frames):
            Image.fromarray(image[:, :, ::-1]).save(
                out_dir / f"{label}_{i:04d}.jpg", quality=95)
            stamps.append(ms)
        record = dict(meta)
        record["label"] = label
        record["wall_time_ms"] = stamps
        record["frames"] = len(stamps)
        gaps = np.diff(stamps) if len(stamps) > 1 else np.array([0.0])
        record["median_gap_ms"] = float(np.median(gaps))
        (out_dir / f"{label}.json").write_text(json.dumps(record, indent=2))
        return record


def capture(args):
    from capture.window import WindowSource
    from clashroyalebuildabot.detectors.detector import Detector
    from live.actuator import AdbActuator
    from live.mvp_loop import DECK

    out_dir = Path(args.out)
    source = WindowSource(args.window)
    detector = Detector(DECK)
    actuator = AdbActuator(dry_run=False)
    print(f"capture {source.size}, actuator backend {actuator.backend}")

    wanted = args.card.lower()
    target = {"deep": DEEP_TARGET, "near": NEAR_TARGET,
              "deploy": DEPLOY_TILE}[args.where]

    recorder = BurstRecorder(source)
    done = 0
    deadline = time.perf_counter() + args.timeout
    while done < args.trials and time.perf_counter() < deadline:
        frame = source.read_new(timeout_s=2.0)
        if frame is None:
            continue
        native = Image.fromarray(frame.image[:, :, ::-1])
        from clashroyalebuildabot.constants import (  # noqa: PLC0415
            SCREENSHOT_HEIGHT,
            SCREENSHOT_WIDTH,
        )
        small = native.resize((SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT),
                              Image.LANCZOS)
        state = detector.run(small)
        if state is None:
            if args.verbose:
                print("  poll: state None", flush=True)
            continue
        elixir = float(state.numbers.elixir.number)
        if args.verbose:
            print(f"  poll: screen={state.screen.name} elixir={elixir} "
                  f"hand={[getattr(c, 'name', '') for c in state.cards[1:5]]}",
                  flush=True)
        # state.cards is FIVE entries -- [0] is the "Next" preview box, so the
        # hand is [1:5]. Reading [:4] cost a whole match once; see adapter.py.
        hand = [h.lower() for h in
                (getattr(c, "name", "") for c in state.cards[1:5])]
        if wanted not in hand:
            # THE HAND ONLY CYCLES WHEN YOU PLAY. Waiting for a card without
            # playing anything is a deadlock, not a wait -- the first two
            # attempts at this sat through entire matches watching one frozen
            # hand, and looked exactly like a broken detector. If the card we
            # want is not dealt, spend one to advance the cycle.
            if elixir >= MAX_DECK_COST:
                actuator.play(0, *CYCLE_TILE)
                time.sleep(args.cycle_s)
            else:
                time.sleep(0.3)
            continue
        if elixir < args.min_elixir:
            time.sleep(0.3)
            continue
        slot = hand.index(wanted)

        print(f"  trial {done}: {wanted} in slot {slot}, elixir {elixir:.0f} "
              f"-> tile {target}", flush=True)
        recorder.start()
        time.sleep(args.pre_s)
        actuator.play(slot, target[0], target[1])
        time.sleep(args.post_s)
        recorder.stop()
        rec = recorder.save(out_dir, f"{args.where}_{wanted}_{done}", {
            "card": wanted, "slot": slot, "target": list(target),
            "elixir_before": elixir, "where": args.where})
        print(f"    {rec['frames']} frames, median gap "
              f"{rec['median_gap_ms']:.0f} ms", flush=True)
        done += 1
        # Let elixir rebuild and the board settle before the next trial.
        time.sleep(args.gap_s)

    actuator.close()
    source.close()
    print(f"captured {done} trials into {out_dir}")


def analyse(args):
    from clashroyalebuildabot.constants import (  # noqa: PLC0415
        SCREENSHOT_HEIGHT,
        SCREENSHOT_WIDTH,
    )
    from clashroyalebuildabot.detectors.detector import Detector
    from live.mvp_loop import DECK

    detector = Detector(DECK)
    out_dir = Path(args.out)
    results = []
    for meta_path in sorted(out_dir.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        label = meta["label"]
        stamps = meta["wall_time_ms"]
        series = []
        for i, ms in enumerate(stamps):
            path = out_dir / f"{label}_{i:04d}.jpg"
            if not path.exists():
                continue
            native = Image.open(path).convert("RGB")
            small = native.resize((SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT),
                                  Image.LANCZOS)
            state = detector.run(small)
            if state is None:
                continue
            row = {"t": ms, "elixir": float(state.numbers.elixir.number)}
            for key in PRINCESS_KEYS:
                row[key] = float(getattr(state.numbers, key).number)
            row["units"] = [(u.unit.name, u.position.tile_x, u.position.tile_y)
                            for u in state.allies + state.enemies]
            series.append(row)
        results.append({"meta": meta, "series": series})

    report = _report(results)
    print(report)
    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    (out_dir / "series.json").write_text(json.dumps(results, indent=2))
    return results


def _first_drop(series, key, minimum):
    """Timestamp of the first sample where `key` falls by at least `minimum`.

    Uses the first reading as the reference rather than a running previous
    value, so a single noisy frame cannot start the clock -- the elixir reader
    misreads occasionally and a one-frame dip would otherwise become the cast.
    """
    if not series:
        return None
    base = series[0][key]
    for row in series:
        if base - row[key] >= minimum:
            return row["t"]
    return None


def _report(results):
    lines = ["", "=" * 68, "REAL-GAME TIMING", "=" * 68]
    if not results:
        lines.append("no trials found")
        return "\n".join(lines)
    for entry in results:
        meta, series = entry["meta"], entry["series"]
        if not series:
            lines.append(f"{meta['label']}: no readable frames")
            continue
        gap = meta.get("median_gap_ms", 0.0)
        cast = _first_drop(series, "elixir", 2.0)
        lines.append(f"\n{meta['label']}  ({len(series)} frames, "
                     f"{gap:.0f} ms apart, +/-{gap / 2:.0f} ms)")
        if cast is None:
            lines.append("  no elixir drop seen -- the play never landed")
            continue
        impacts = {}
        for key in PRINCESS_KEYS:
            hit = _first_drop([r for r in series if r["t"] >= cast], key, 0.05)
            if hit is not None:
                impacts[key] = hit - cast
        if impacts:
            for key, delay in sorted(impacts.items(), key=lambda kv: kv[1]):
                lines.append(f"  impact on {key:<26} {delay:7.0f} ms")
        else:
            lines.append("  no tower HP drop seen (spell missed a tower, "
                         "or the HP reader failed -- 0.0 is 'unreadable', "
                         "not 'destroyed')")
    lines.append("")
    lines.append("Engine, for comparison: Fireball delay is a flat 10 ticks")
    lines.append("= 1000 ms at any distance; deploy time is 0 ms.")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["capture", "analyse"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--card", default="fireball")
    parser.add_argument("--where", default="deep",
                        choices=["deep", "near", "deploy"])
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--min-elixir", type=float, default=5.0)
    parser.add_argument("--pre-s", type=float, default=0.6,
                        help="capture this long before issuing the play")
    parser.add_argument("--post-s", type=float, default=3.5)
    parser.add_argument("--gap-s", type=float, default=6.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--window", default="BlueStacks App Player")
    parser.add_argument("--cycle-s", type=float, default=2.0,
                        help="pause after dumping a card to advance the cycle")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.mode == "capture":
        capture(args)
    else:
        analyse(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
