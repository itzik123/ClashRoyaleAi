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

WHY THE EXPLOSION SPRITE IS THE IMPACT CLOCK, AND A TOWER IS NOT
----------------------------------------------------------------
A tower looked like the obvious impact clock -- it cannot move, cannot be
mistaken for another unit, and its HP is a clean step. It was tried twice and
failed twice, for two unrelated reasons.

First, CRBAB's bar FRACTION is unusable: over one 77-frame trial it took ten
distinct values (0.0, 0.46, 0.49, 0.62 ... 1.0) on a tower whose HP never
changed once, because `_calculate_hp` returns 0.0 both for an empty bar and
for one it cannot colour-match at all.

That was fixed -- `readers/tower_numerals.py` plus the tower_549x976 templates
read absolute HP correctly -- and the tower STILL could not be the clock. The
spell's own UI (the radius circle, the "<card> lvl.N" banner, the "-N" drag
cursor) is drawn directly over the tower being aimed at, and on a real capture
it occluded that tower's numeral from 950 ms to 2150 ms after the cast, which
is precisely the window a ~1 s flight lands in. The reader did the right thing
and reported "unreadable" for those 35 frames rather than guessing; that is
correct behaviour and still no measurement.

So impact is scored from the burst sprite, which is large, central, and the
one thing the cast UI cannot cover -- and the tower numeral is kept as
CORROBORATION and as the validity gate instead.

STATUS: THE DETECTOR IS UNVALIDATED (2026-08-10)
-------------------------------------------------
No capture in this repo contains a confirmed spell impact, so the burst
detector below has never been checked against a real explosion. Its thresholds
were chosen against measured FAILURES rather than a positive example:

  a naive "count warm-coloured pixels" detector fired on ~12,000 pixels of
  every frame including pre-cast ones (red tower roofs, tan paths);
  baseline-differencing removed that but left a diffuse signal that rose
  monotonically across the whole board -- an ongoing melee.

Hence the three discriminators a melee does not satisfy: compactness, round
aspect, and decay. Until a real burst is captured, run with `--inspect` and
LOOK at the strip before believing any number this prints.

Worth knowing about the capture that motivated all this: the Fireball was
genuinely played -- elixir 10 -> 6 and held, the card left the hand and the
slot refilled -- the radius indicator sat squarely on the target tower, and
neither enemy tower lost a single HP. Where it actually landed is unknown.
That is why damage is now a hard validity gate: a spell that spends elixir and
damages nothing is a miss, and averaging it into a delay would be worse than
reporting nothing.

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

import cv2
import numpy as np

_PERCEPTION_ROOT = Path(__file__).resolve().parent.parent
if str(_PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(_PERCEPTION_ROOT))

# loguru and PIL are CAPTURE-side dependencies and are imported lazily, inside
# the functions that need them. Importing them at module scope made the whole
# module -- including the pure burst-detection helpers below -- unimportable in
# the perception venv, which is where its tests run. An analysis routine that
# cannot be exercised without the capture toolchain installed is one that never
# gets a test.
def _pil():
    from PIL import Image  # noqa: PLC0415
    return Image


def _logger():
    from loguru import logger  # noqa: PLC0415
    logger.remove()
    return logger

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


def _expected_native_xy(target_tile, native_wh):
    """Where a commanded ENGINE tile should land, in native capture pixels.

    Derived from the actuator's own mapping and then rescaled, rather than
    calibrated separately: the point of comparing against it is to catch the
    actuator aiming somewhere other than it claims, and a second independent
    copy of the mapping could not do that.
    """
    try:
        from live.actuator import engine_tile_centre  # noqa: PLC0415
        from clashroyalebuildabot.constants import (  # noqa: PLC0415
            DISPLAY_HEIGHT, DISPLAY_WIDTH,
        )
    except Exception:
        # Fail soft. This hint only RANKS candidate bursts and never gates
        # them, so losing it costs a "px from the commanded tile" column --
        # not the measurement. Worth catching because the import reaches into
        # the capture toolchain (CRBAB pulls in `keyboard`), and analysis is
        # meant to run without it.
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
            # Absolute HP beside the bar fraction. The fraction is kept only
            # as a control: on the capture that motivated this rewrite it took
            # ten distinct values on a tower whose HP never changed.
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

    # Damage is the VALIDITY GATE, not the clock. A cast that damages nothing
    # is a miss, and a miss must be discarded loudly rather than averaged into
    # a delay -- the capture that motivated this rewrite spent the elixir,
    # cycled the card out of hand, and never touched the tower it was aimed at.
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
    """An annotated strip around the detected burst.

    Exists because the detector cannot currently be validated against a
    confirmed explosion -- no capture in this repo contains one. Until a real
    burst is captured, every delay this tool prints should be eyeballed here
    before it is believed.
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


def _stable_drop(series, key, minimum, hold=3):
    """Like _first_drop, but the drop has to STAY dropped for `hold` samples.

    The elixir reader glitches through zero: a true 10 -> 6 was read as
    10 -> 0 -> 6 on a real capture, and _first_drop would date the cast from
    the spurious 0. Requiring the post-drop value to persist rejects that
    without needing to model the glitch.
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


# --- the impact clock: an explosion is COMPACT, BRIGHT and TRANSIENT --------
#
# A tower's HP numeral was the obvious impact clock and it does not work. The
# spell's own UI -- the radius circle, the "<card> lvl.N" banner and the "-N"
# drag cursor -- is drawn on top of the target tower, and on a real capture it
# occluded that tower's numeral from 950 ms to 2150 ms after the cast, which
# is exactly the window a ~1 s flight lands in. The numeral is kept below, as
# CORROBORATION and as the validity gate, but it cannot be the clock.
#
# Every threshold here was chosen against a measured failure on that capture,
# where a naive "count warm-coloured pixels" detector reported ~12,000 hits on
# EVERY frame including the pre-cast ones (arena decoration: red tower roofs,
# tan paths) and a baseline-differenced version still produced a diffuse
# signal that rose monotonically across the whole board (an ongoing melee).
#
# So the discriminators are the three properties a melee does NOT have:
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
    everything BEFORE the cast, so the static board and any steady-state fire
    cancel; only a new transient survives.

    `expect_xy` (native px) is used to RANK candidates, never to gate them --
    the best global candidate is always reported too. That is deliberate: on
    the capture that motivated this rewrite the spell was commanded onto a
    tower, the radius indicator sat on that tower, and nothing there ever took
    damage. A detector that only looks where the spell was aimed would have
    reported "nothing happened" instead of "it landed somewhere else", and the
    difference between those two is the whole diagnosis.
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

    # Transience: the peak must fall away again. A melee produces a large
    # compact-ish blob that persists to the end of the capture, and on the
    # motivating capture that is exactly what a non-transient detector locked
    # onto.
    peak = max(cands, key=lambda c: c["area"])

    # The decay test has to be FALSIFIABLE before it is trusted. If the
    # capture ends before the burst has had time to fade, "it did not persist"
    # is unearned -- there were simply no frames in which it could. Caught by
    # running this against the motivating capture: the melee peaked 650 ms
    # from the end, no frame existed past peak + FIRE_DECAY_MS, the emptiness
    # was read as decay, and the detector reported a confident 1799 ms delay
    # for an explosion that is not there.
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

    # Distance scaling is the question that matters. The engine models a FLAT
    # 10 ticks, so a delay that grows with cast distance is not a mistuned
    # constant, it is the wrong SHAPE of model and no retuning fixes it.
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
    # 3.5 s was too short: on the motivating capture the burst-shaped signal
    # peaked 350 ms before the recording ended, and detect_explosion cannot
    # test decay without FIRE_DECAY_MS of tail after the peak. A ~1 s flight
    # plus 1.4 s of decay plus margin needs 5 s.
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
