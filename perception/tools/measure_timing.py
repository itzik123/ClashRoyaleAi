"""Measure the real game's spell delay and troop deploy time against the engine's.

The engine models spell delay as a flat interval regardless of cast distance,
so trials cast at two distances: if the real delay scales with distance, a
constant is the wrong shape of model, not a mistuned one.

The elixir bar is the cast clock: it drops the instant the game accepts the
play, on the far side of ADB, the actuator queue and the emulator's input
handling.

The explosion sprite is the impact clock. A tower's HP looks like the obvious
choice, but the spell's own UI (radius circle, "<card> lvl.N" banner, "-N" drag
cursor) is drawn over the tower being aimed at and occluded its numeral for
exactly the window a ~1 s flight lands in. The numeral
(readers/tower_numerals.py) is kept as corroboration and as the validity gate:
a spell that spends elixir and damages nothing is a miss, and averaging it into
a delay would be worse than reporting nothing.

The burst detector has never been checked against a confirmed impact, since no
capture here contains one. Its three discriminators (compactness, round aspect,
decay) are the properties an ongoing melee lacks. Run with `--inspect` and look
at the strip before believing a number.

The hand only cycles when you play, so waiting for a card while playing nothing
is a deadlock. Hence `CYCLE_TILE`: if the wanted card is not dealt, spend one
to advance the cycle.

`--capture` records short bursts of raw frames around each play; `--analyse`
runs the detector afterwards. Running it live would compete with capture for
CPU and lower the frame rate, which is the measurement's resolution, and a
trial can be re-analysed without being re-played.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

_PERCEPTION_ROOT = Path(__file__).resolve().parent.parent
if str(_PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(_PERCEPTION_ROOT))

# loguru and PIL are capture-side dependencies, imported lazily, so the pure
# burst-detection helpers import (and are tested) in the perception venv.
def _pil():
    from PIL import Image  # noqa: PLC0415
    return Image


def _logger():
    from loguru import logger  # noqa: PLC0415
    logger.remove()
    return logger

# Engine tiles. The opponent's left Princess Tower is the deep target; a spell
# is legal anywhere.
DEEP_TARGET = (4, 27)
# Just past our own bridge: the shortest cast, used only when a near target is
# available.
NEAR_TARGET = (4, 18)
# Own half, clear of both Princess Towers and the King.
DEPLOY_TILE = (11, 9)
# Where cards are dumped purely to advance the cycle: a back corner of our own
# half, away from any tile a trial measures. Training Camp, so the outcome is
# irrelevant.
CYCLE_TILE = (1, 4)
# The dearest card in DEFAULT_DECK. Cycling needs at least this much elixir,
# since an unaffordable playCard fails silently.
MAX_DECK_COST = 5.0

PRINCESS_KEYS = ("left_enemy_princess_hp", "right_enemy_princess_hp",
                 "left_ally_princess_hp", "right_ally_princess_hp")


class BurstRecorder:
    """Captures frames on a thread into memory, for a bounded window: raw frames
    are ~1.6 MB each, and the four seconds around one play (~100 MB) are all
    the measurement uses.
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
            _pil().fromarray(image[:, :, ::-1]).save(
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
    _logger()   # silence loguru's default handler before CRBAB imports
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
        native = _pil().fromarray(frame.image[:, :, ::-1])
        from clashroyalebuildabot.constants import (  # noqa: PLC0415
            SCREENSHOT_HEIGHT,
            SCREENSHOT_WIDTH,
        )
        small = native.resize((SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT),
                              _pil().LANCZOS)
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
        # state.cards[0] is the "Next" preview; the hand is [1:5]. See
        # adapter.py.
        hand = [h.lower() for h in
                (getattr(c, "name", "") for c in state.cards[1:5])]
        if wanted not in hand:
            # The hand only cycles when you play: if the wanted card is not
            # dealt, spend one to advance it.
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


def _expected_native_xy(target_tile, native_wh):
    """Where a commanded engine tile should land, in native capture pixels.
    Derived from the actuator's own mapping, since the point is to catch the
    actuator aiming somewhere other than it claims.
    """
    try:
        from live.actuator import engine_tile_centre  # noqa: PLC0415
        from clashroyalebuildabot.constants import (  # noqa: PLC0415
            DISPLAY_HEIGHT, DISPLAY_WIDTH,
        )
    except Exception:
        # Fail soft: this hint only ranks candidate bursts, and the import
        # reaches into the capture toolchain (CRBAB pulls in `keyboard`), which
        # analysis should not need.
        return None
    tap = engine_tile_centre(*target_tile)
    nw, nh = native_wh
    return (tap.x * nw / DISPLAY_WIDTH, tap.y * nh / DISPLAY_HEIGHT)


def _tower_reader():
    """The absolute-HP numeral reader, or None if its templates are absent."""
    from readers.clock import DigitTemplates  # noqa: PLC0415
    from readers.tower_numerals import TowerNumeralReader  # noqa: PLC0415
    d = _PERCEPTION_ROOT / "config" / "templates" / "tower_549x976"
    if not (d / "digits.json").exists():
        return None
    return TowerNumeralReader(DigitTemplates.load(d))


def _tower_bars():
    from clashroyalebuildabot.constants import (  # noqa: PLC0415
        ENEMY_PRINCESS_HP_Y, HP_HEIGHT, HP_WIDTH, LEFT_PRINCESS_HP_X,
        RIGHT_PRINCESS_HP_X,
    )
    return {
        "left_enemy_princess": (LEFT_PRINCESS_HP_X, ENEMY_PRINCESS_HP_Y,
                                LEFT_PRINCESS_HP_X + HP_WIDTH,
                                ENEMY_PRINCESS_HP_Y + HP_HEIGHT),
        "right_enemy_princess": (RIGHT_PRINCESS_HP_X, ENEMY_PRINCESS_HP_Y,
                                 RIGHT_PRINCESS_HP_X + HP_WIDTH,
                                 ENEMY_PRINCESS_HP_Y + HP_HEIGHT),
    }


def analyse(args):
    from clashroyalebuildabot.constants import (  # noqa: PLC0415
        SCREENSHOT_HEIGHT,
        SCREENSHOT_WIDTH,
    )
    from clashroyalebuildabot.detectors.detector import Detector
    from live.mvp_loop import DECK

    detector = Detector(DECK)
    reader = _tower_reader()
    bars = _tower_bars()
    detector_wh = (SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT)
    out_dir = Path(args.out)
    results = []

    for meta_path in sorted(out_dir.glob("*.json")):
        if meta_path.name in {"series.json", "report.txt"}:
            continue
        meta = json.loads(meta_path.read_text())
        label = meta["label"]
        stamps_all = meta["wall_time_ms"]

        frames, stamps, series = [], [], []
        for i, ms in enumerate(stamps_all):
            path = out_dir / f"{label}_{i:04d}.jpg"
            if not path.exists():
                continue
            native = cv2.imread(str(path))
            if native is None:
                continue
            small = _pil().fromarray(native[:, :, ::-1]).resize(
                detector_wh, _pil().LANCZOS)
            state = detector.run(small)
            if state is None:
                continue
            frames.append(native)
            stamps.append(ms)
            row = {"t": ms, "elixir": float(state.numbers.elixir.number)}
            for key in PRINCESS_KEYS:
                row[key] = float(getattr(state.numbers, key).number)
            # Absolute HP beside the bar fraction; the fraction is kept only as
            # a control.
            if reader is not None:
                for name, bar in bars.items():
                    r = reader.read(native, bar, detector_wh)
                    row[f"{name}_hp"] = r.value
            row["units"] = [(u.unit.name, u.position.tile_x, u.position.tile_y)
                            for u in state.allies + state.enemies]
            series.append(row)

        entry = {"meta": meta, "series": series}
        entry.update(_score_trial(meta, series, frames, stamps, out_dir,
                                  inspect=args.inspect))
        results.append(entry)

    report = _report(results)
    print(report)
    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    (out_dir / "series.json").write_text(json.dumps(results, indent=2, default=str))
    return results


def _score_trial(meta, series, frames, stamps, out_dir, inspect=False):
    """Cast, impact and validity for one trial."""
    if not series:
        return {"valid": False, "why": "no readable frames"}

    cast = _stable_drop(series, "elixir", 2.0)
    if cast is None:
        return {"valid": False, "why": "no sustained elixir drop -- the play "
                                       "never reached the game"}

    nh, nw = frames[0].shape[:2]
    target = meta.get("target")
    expect = _expected_native_xy(tuple(target), (nw, nh)) if target else None
    onset, why = detect_explosion(frames, stamps, cast, expect)

    # Damage is the validity gate, not the clock: a cast that damages nothing
    # is a miss, discarded loudly rather than averaged into a delay.
    after = [r for r in series if r["t"] >= cast]
    damaged = {}
    for name in ("left_enemy_princess", "right_enemy_princess"):
        vals = [(r["t"], r.get(f"{name}_hp")) for r in after
                if r.get(f"{name}_hp") is not None]
        if len(vals) >= 2 and vals[0][1] - vals[-1][1] >= 50:
            damaged[name] = vals[0][1] - vals[-1][1]

    out = {"valid": False, "cast_ms": cast, "damage": damaged}
    if onset is None:
        out["why"] = why
        if not damaged:
            out["why"] += " (and no tower lost HP -- the spell missed)"
        return out

    out.update(valid=bool(damaged), impact_ms=onset["t"],
               delay_ms=onset["t"] - cast, impact_xy=onset["xy"],
               impact_area=onset["area"],
               miss_px=onset.get("dist"))
    if not damaged:
        out["valid"] = False
        out["why"] = ("burst found but no tower lost HP -- it landed away "
                      "from the target; delay reported for inspection only")
    if inspect:
        out["inspect_png"] = _dump_inspect(meta, frames, stamps, cast,
                                           onset, expect, out_dir)
    return out


def _dump_inspect(meta, frames, stamps, cast, onset, expect, out_dir):
    """An annotated strip around the detected burst. The detector has not been
    validated against a confirmed explosion, so every delay should be eyeballed
    here first.
    """
    i = onset["i"]
    picks = [j for j in range(i - 3, i + 5) if 0 <= j < len(frames)]
    tiles = []
    for j in picks:
        c = frames[j].copy()
        cv2.circle(c, (int(onset["xy"][0]), int(onset["xy"][1])), 26,
                   (0, 255, 255), 2)
        if expect:
            cv2.drawMarker(c, (int(expect[0]), int(expect[1])), (255, 0, 255),
                           cv2.MARKER_CROSS, 30, 2)
        tag = "BURST" if j == i else ""
        cv2.putText(c, f"{j}  t={stamps[j] - cast:+.0f}ms {tag}", (6, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(cv2.resize(c, (275, 488)))
    path = out_dir / f"{meta['label']}_inspect.png"
    cv2.imwrite(str(path), np.hstack(tiles))
    return str(path)


def _first_drop(series, key, minimum):
    """Timestamp of the first sample where `key` falls by at least `minimum`.
    Referenced to the first reading, not a running previous value, so a single
    noisy frame cannot start the clock.
    """
    if not series:
        return None
    base = series[0][key]
    for row in series:
        if base - row[key] >= minimum:
            return row["t"]
    return None


def _stable_drop(series, key, minimum, hold=3):
    """Like _first_drop, but the drop must stay dropped for `hold` samples: the
    elixir reader can read a true 10 -> 6 as 10 -> 0 -> 6.
    """
    if not series:
        return None
    base = series[0][key]
    for i, row in enumerate(series):
        if base - row[key] < minimum:
            continue
        window = series[i:i + hold]
        if len(window) < hold:
            return None
        if all(base - r[key] >= minimum for r in window):
            return row["t"]
    return None


# --- the impact clock: an explosion is compact, bright and transient ---
# The tower numeral cannot be the clock (module docstring). A naive warm-pixel
# count fires on every frame (red roofs, tan paths) and a baseline-differenced
# one on an ongoing melee, so the discriminators are the three properties a
# melee lacks:
FIRE_MIN_AREA = 260        # px in the largest blob, at 549x976
FIRE_MIN_FILL = 0.42       # blob area / its bounding box: a burst is a disc
FIRE_MAX_ASPECT = 2.2      # ... and roughly round, not a smear of skirmishers
FIRE_DECAY_MS = 1400       # ... and gone this soon; the melee never decays
FIRE_SEARCH_RADIUS_PX = 150


def _fire_mask(frame_bgr, baseline_bgr):
    """Pixels that are fire-coloured AND changed from the static board."""
    im = frame_bgr.astype(np.int16)
    b, g, r = im[..., 0], im[..., 1], im[..., 2]
    fire = (r > 200) & (g > 110) & (g < 225) & (b < 130) & (r - b > 90)
    changed = np.abs(im - baseline_bgr.astype(np.int16)).max(axis=2) > 55
    m = (fire & changed).astype(np.uint8) * 255
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _blob_score(mask):
    """(area, centroid, fill, aspect) of the largest connected component."""
    n, _lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return 0, None, 0.0, 99.0
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[i, cv2.CC_STAT_AREA])
    w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
    fill = area / max(1, w * h)
    aspect = max(w, h) / max(1, min(w, h))
    return area, (float(cents[i][0]), float(cents[i][1])), fill, aspect


def detect_explosion(frames, stamps, cast_ms, expect_xy=None):
    """Frame index and timestamp of the spell burst, or None with a reason.

    `frames` are native BGR arrays. The baseline is the per-pixel median of
    everything before the cast, so the static board and steady-state fire
    cancel.

    `expect_xy` (native px) ranks candidates and never gates them, and the best
    global candidate is always reported too: a detector looking only where the
    spell was aimed would report "nothing happened" instead of "it landed
    somewhere else".
    """
    pre = [f for f, t in zip(frames, stamps) if t < cast_ms]
    if len(pre) < 3:
        return None, "fewer than 3 pre-cast frames -- no baseline"
    baseline = np.median(np.stack(pre[-12:]), axis=0).astype(np.uint8)

    cands = []
    for i, (f, t) in enumerate(zip(frames, stamps)):
        if t < cast_ms:
            continue
        m = _fire_mask(f, baseline)
        m[:55, :] = 0          # the top banner
        m[790:, :] = 0         # the card tray and elixir bar
        area, cen, fill, aspect = _blob_score(m)
        if area < FIRE_MIN_AREA or fill < FIRE_MIN_FILL or aspect > FIRE_MAX_ASPECT:
            continue
        d = (float(np.hypot(cen[0] - expect_xy[0], cen[1] - expect_xy[1]))
             if expect_xy else 0.0)
        cands.append({"i": i, "t": t, "area": area, "xy": cen,
                      "fill": fill, "aspect": aspect, "dist": d})

    if not cands:
        return None, "no compact transient burst after the cast"

    # Transience: the peak must fall away. A melee's compact-ish blob persists
    # to the end of the capture.
    peak = max(cands, key=lambda c: c["area"])

    # The decay test must be falsifiable: if the capture ends before
    # FIRE_DECAY_MS past the peak, "it did not persist" is unearned.
    if max(stamps) < peak["t"] + FIRE_DECAY_MS:
        return None, (f"capture ends {max(stamps) - peak['t']:.0f} ms after the "
                      f"brightest frame -- too soon to tell a burst from "
                      f"sustained combat (need {FIRE_DECAY_MS} ms)")

    later = [c for c in cands if c["t"] > peak["t"] + FIRE_DECAY_MS]
    if later and max(c["area"] for c in later) > 0.6 * peak["area"]:
        return None, ("burst never decays -- this is sustained combat, "
                      "not a spell impact")

    onset = min((c for c in cands if c["t"] <= peak["t"]), key=lambda c: c["t"])
    return onset, None


def _report(results):
    lines = ["", "=" * 72, "REAL-GAME TIMING", "=" * 72]
    if not results:
        lines.append("no trials found")
        return "\n".join(lines)

    good = []
    for e in results:
        meta = e["meta"]
        gap = meta.get("median_gap_ms", 0.0)
        lines.append(f"\n{meta['label']}  ({len(e['series'])} frames, "
                     f"{gap:.0f} ms apart, +/-{gap / 2:.0f} ms, "
                     f"target {meta.get('target')})")
        if e.get("cast_ms") is None:
            lines.append(f"  REJECTED  {e.get('why')}")
            continue
        if e.get("delay_ms") is None:
            lines.append(f"  REJECTED  {e.get('why')}")
            continue
        miss = e.get("miss_px")
        lines.append(f"  cast -> burst   {e['delay_ms']:7.0f} ms"
                     f"   burst at ({e['impact_xy'][0]:.0f},"
                     f"{e['impact_xy'][1]:.0f}) px"
                     + (f", {miss:.0f} px from the commanded tile" if miss else ""))
        if e.get("damage"):
            for k, v in e["damage"].items():
                lines.append(f"  confirmed by {k} losing {v:.0f} HP")
        if e["valid"]:
            good.append(e)
        else:
            lines.append(f"  REJECTED  {e.get('why')}")
        if e.get("inspect_png"):
            lines.append(f"  inspect: {e['inspect_png']}")

    lines += ["", "-" * 72]
    if not good:
        lines.append("NO VALID TRIAL. Nothing is reported as a delay.")
        lines.append("A trial counts only if the elixir drop holds, a compact")
        lines.append("transient burst is found, AND a tower actually loses HP.")
        lines.append("A spell that spends elixir and damages nothing is a miss,")
        lines.append("not a measurement.")
        return "\n".join(lines)

    # Distance scaling is the question that matters: the engine's delay is
    # flat, so a delay growing with distance means the wrong shape of model.
    by_target = {}
    for e in good:
        by_target.setdefault(tuple(e["meta"].get("target") or ()), []).append(
            e["delay_ms"])
    lines.append(f"{len(good)} valid trial(s)")
    for tgt, ds in sorted(by_target.items()):
        a = np.array(ds)
        lines.append(f"  target {str(tgt):<10} n={len(ds)}  "
                     f"mean {a.mean():6.0f} ms   sd {a.std(ddof=1) if len(a) > 1 else 0:5.0f}"
                     f"   range {a.min():.0f}-{a.max():.0f}")
    if len(by_target) >= 2:
        means = {t: float(np.mean(v)) for t, v in by_target.items()}
        spread = max(means.values()) - min(means.values())
        lines.append(f"  spread across distances: {spread:.0f} ms")
        lines.append("  -> a flat 10-tick model is "
                     + ("QUESTIONABLE; delay appears to scale with distance"
                        if spread > 150 else
                        "consistent with these data"))
    else:
        lines.append("  only one distance measured -- run --where near as well "
                     "before drawing any conclusion about scaling")
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
    # A ~1 s flight plus 1.4 s of decay plus margin: detect_explosion cannot
    # test decay without FIRE_DECAY_MS of tail after the peak.
    parser.add_argument("--post-s", type=float, default=5.0)
    parser.add_argument("--gap-s", type=float, default=6.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--window", default="BlueStacks App Player")
    parser.add_argument("--cycle-s", type=float, default=2.0,
                        help="pause after dumping a card to advance the cycle")
    parser.add_argument("--inspect", action="store_true",
                        help="dump an annotated strip around each detected "
                             "burst -- the detector has no confirmed "
                             "explosion to validate against yet, so look "
                             "before believing a delay")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.mode == "capture":
        capture(args)
    else:
        analyse(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
