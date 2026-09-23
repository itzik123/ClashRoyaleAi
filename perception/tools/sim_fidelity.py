"""How closely does the engine's physics match the real game?

Takes a real board off a recording, steps it forward with the engine, and
compares with what the recording shows one horizon later. Three numbers per
horizon:

    floor       rebuild the board and step the minimum one tick. What it loses
                is reconstruction, not dynamics, and no prediction can beat it.
    stale       the board as it was, scored against the board that followed:
                what the live loop acts on today.
    predicted   the engine's forecast, scored the same way.

`predicted > stale` means the engine's dynamics carry information about the
next second and some of the ~1 s latency is recoverable. `predicted < stale`
means stepping this engine misinforms, and that the agent trains on physics the
real game does not share.

Scored as intersection-over-union of occupied cells with towers excluded:
forecast.py does not reconstruct unit HP, tower HP, elixir or the clock, and
towers never move (six guaranteed matches would dominate a quiet board).

Speed is measured on both sides: the engine's by injection, the real game's
from card classes that appear exactly once on a side in both frames of a pair,
where displacement needs no tracker.

Caveats on every number:
  * `TILE_Y_OFFSET` is derived, not measured; a constant row error shifts
    both sides equally, so it barely affects the speed comparison.
  * The detector's errors are in both `stale` and `predicted`, the right
    control since the live loop eats them too.
  * Every rebuilt unit gets a fresh deploy second (forecast.py does not
    pass deploy_ticks), so newly placed units are predicted worst.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PERCEPTION_ROOT = Path(__file__).resolve().parent.parent
if str(_PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(_PERCEPTION_ROOT))

# The detector logs a DEBUG line per frame, which would bury the report.
from loguru import logger  # noqa: E402

logger.remove()

from PIL import Image  # noqa: E402

from capture.video import VideoSource  # noqa: E402
from clashroyalebuildabot.constants import (  # noqa: E402
    SCREENSHOT_HEIGHT,
    SCREENSHOT_WIDTH,
)
from clashroyalebuildabot.detectors.detector import Detector  # noqa: E402
from forecast import (  # noqa: E402
    MIN_HORIZON_S,
    SimForecaster,
    agreement,
    occupancy,
    tower_cells,
)
from live.adapter import build_game_state  # noqa: E402
from live.mvp_loop import DECK, deck_costs  # noqa: E402

# The recordings are 1920x1080 desktop captures and the emulator occupies this
# sub-rectangle. Uncropped, the elixir reader returns 0 and the detector finds
# phantom enemies.
EMULATOR_CROP = (686, 40, 1236, 1012)      # left, top, right, bottom

# Menus and the end screen produce confident nonsense, and CRBAB's `in_game`
# hash was cut from another emulator profile (it reports `unknown` here).
# Elixir reads 0 off-match and almost never during play, so it is the gate.
SKIP_HEAD_FRAC = 0.15
SKIP_TAIL_FRAC = 0.10

DEFAULT_HORIZONS = (0.5, 1.0, 2.0)


def perceive(detector, frame):
    """The live loop's own perception path, minus the ledger and the debit: the
    same calls in the same order as `mvp_loop.perceive`, so this measures the
    pipeline that plays.
    """
    left, top, right, bottom = EMULATOR_CROP
    cropped = frame.image[top:bottom, left:right]
    native = Image.fromarray(cropped[:, :, ::-1])          # BGR -> RGB
    small = native.resize((SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT), Image.LANCZOS)
    state = detector.run(small)
    if state is None:
        return None, None
    if not (0 < float(state.numbers.elixir.number) <= 10):
        return None, None
    game_state, _report = build_game_state(
        state, np.array(native), np.array(small),
        frame_index=frame.index, wall_time_ms=frame.wall_time_ms)
    return state, game_state


def real_occupancy(game_state, team):
    """Occupied cells straight from perception, towers excluded. Built from the
    GameState: the encoder is byte-exact against the engine, so a round trip
    would give the same set, slower.
    """
    return {(u.tile_x, u.tile_y) for u in game_state.units
            if int(u.team) == int(team) and int(u.card_sim_id) >= 0}


def sole_positions(game_state):
    """{card name: (x, y)} for classes with exactly one body on one side:
    association without a tracker. Fewer samples of a number that means
    something.
    """
    by_class = defaultdict(list)
    for u in game_state.units:
        if int(u.card_sim_id) >= 0:
            by_class[(u.unit_name, int(u.team))].append((u.tile_x, u.tile_y))
    return {k: v[0] for k, v in by_class.items() if len(v) == 1}


def sanity_check_frame(game_state):
    """Our units low-y, theirs high-y, or the comparison is mirrored. A silently
    flipped or offset coordinate frame looks exactly like wrong physics.
    """
    mine = [u.tile_y for u in game_state.units if int(u.team) == 0]
    theirs = [u.tile_y for u in game_state.units if int(u.team) == 1]
    if not mine or not theirs:
        return None
    return float(np.mean(mine)), float(np.mean(theirs))


def run(paths, samples_per_video, horizons, time_scales=(1.0,), verbose=False):
    """Steps the engine by `h * scale` to predict `h`, for each scale given.

    If the engine's troops move N times too fast, otherwise-correct dynamics
    predict footage well at `scale = 1/N` and badly at 1.0. The sweep separates
    wrong physics (a rewrite) from right physics at the wrong rate (one
    constant).

    All scales share one pass: perception is ~99% of the cost and a forecast is
    microseconds, and scoring every scale on the same boards makes the
    comparison paired.
    """
    detector = Detector(DECK)
    costs, _warn = deck_costs(DECK)
    forecaster = SimForecaster([c for c in _engine_deck()])
    towers = tower_cells()

    # `stale` does not depend on the scale; scored once per horizon.
    stale_scores = {h: [] for h in horizons}
    # "Reconstruct but do not step", scored against the same truth. Without it
    # `pred` is compared with a `stale` that never paid the reconstruction
    # cost, so a perfect predictor still loses (the rebuilt board's ceiling is
    # ~0.52). Isolates the effect of stepping.
    rebuilt_scores = {h: [] for h in horizons}
    scores = {(s, h): [] for s in time_scales for h in horizons}
    floor: list[float] = []
    real_speed = defaultdict(list)
    frames_ok = frames_gated = 0
    frame_sanity: list[tuple[float, float]] = []
    extent: list[tuple[int, int, int, int]] = []

    for path in paths:
        video = VideoSource(path)
        try:
            fps = video._nominal_fps
            first = int(video.frame_count * SKIP_HEAD_FRAC)
            last = int(video.frame_count * (1 - SKIP_TAIL_FRAC))
            span = max(horizons)
            last -= int(span * fps) + 2
            if last <= first:
                continue
            anchors = np.linspace(first, last, samples_per_video).astype(int)
            print(f"  {Path(path).name}  {video.frame_count} frames "
                  f"@ {fps:.2f} fps -> {len(anchors)} anchors", flush=True)

            for anchor in anchors:
                _state, now = perceive(detector, video.read_frame_at(int(anchor)))
                if now is None:
                    frames_gated += 1
                    continue

                futures = {}
                for h in horizons:
                    idx = int(anchor + round(h * fps))
                    _s, later = perceive(detector, video.read_frame_at(idx))
                    if later is not None:
                        futures[h] = later
                if not futures:
                    frames_gated += 1
                    continue
                frames_ok += 1

                pair = sanity_check_frame(now)
                if pair:
                    frame_sanity.append(pair)
                if now.units:
                    xs = [u.tile_x for u in now.units]
                    ys = [u.tile_y for u in now.units]
                    extent.append((min(xs), max(xs), min(ys), max(ys)))

                scaled = {(s, h): max(MIN_HORIZON_S, h * s)
                          for s in time_scales for h in horizons}
                predictions = {f.horizon_s: f for f in forecaster.forecast(
                    now, [MIN_HORIZON_S, *scaled.values()])}

                # Reconstruction floor: the rebuilt board at the shortest
                # horizon, against the board it was built from.
                base = predictions[min(predictions)]
                rebuilt = ((occupancy(base.observation, 0) - towers)
                           | (occupancy(base.observation, 1) - towers))
                seen = real_occupancy(now, 0) | real_occupancy(now, 1)
                floor.append(agreement(rebuilt, seen))

                stale = real_occupancy(now, 0) | real_occupancy(now, 1)
                for h, later in futures.items():
                    truth = real_occupancy(later, 0) | real_occupancy(later, 1)
                    stale_scores[h].append(agreement(stale, truth))
                    rebuilt_scores[h].append(agreement(rebuilt, truth))
                    for s in time_scales:
                        target = scaled[(s, h)]
                        key = min(predictions, key=lambda k: abs(k - target))
                        pred = (
                            (occupancy(predictions[key].observation, 0) - towers)
                            | (occupancy(predictions[key].observation, 1) - towers))
                        scores[(s, h)].append(agreement(pred, truth))

                    # Real-game speed, keyed by horizon: at the short ones a
                    # ~0.75 tiles/s unit covers 0.4 tiles in 0.5 s, which
                    # quantises to 0 or 2 tiles/s. Only the longest horizon
                    # carries enough displacement.
                    a, b = sole_positions(now), sole_positions(later)
                    for cls in set(a) & set(b):
                        dx = b[cls][0] - a[cls][0]
                        dy = b[cls][1] - a[cls][1]
                        real_speed[(cls[0], h)].append(
                            float(np.hypot(dx, dy)) / h)
        finally:
            video.close()

    return {"scores": scores, "stale": stale_scores, "rebuilt": rebuilt_scores,
            "scales": list(time_scales),
            "floor": floor, "real_speed": real_speed,
            "frames_ok": frames_ok, "frames_gated": frames_gated,
            "sanity": frame_sanity, "extent": extent,
            "forecaster": forecaster, "costs": costs}


def _engine_deck():
    """DECK as engine card ids, via the live loop's mapping."""
    from live.unit_to_card import hand_card_id_for  # noqa: PLC0415
    return [hand_card_id_for(c.name) for c in DECK]


def report(result, horizons):
    out = []
    add = out.append
    add("")
    add("=" * 72)
    add("SIM FIDELITY")
    add("=" * 72)
    add(f"paired samples: {result['frames_ok']}   "
        f"gated (menus/unreadable): {result['frames_gated']}")

    if result["sanity"]:
        mine = np.mean([s[0] for s in result["sanity"]])
        theirs = np.mean([s[1] for s in result["sanity"]])
        ok = "OK" if mine < theirs else "*** MIRRORED -- ALL NUMBERS SUSPECT ***"
        add(f"frame check: own units mean y={mine:.1f}, "
            f"enemy mean y={theirs:.1f}   {ok}")
    if result["extent"]:
        xs = [e[0] for e in result["extent"]] + [e[1] for e in result["extent"]]
        ys = [e[2] for e in result["extent"]] + [e[3] for e in result["extent"]]
        # Speed in tiles/second means something only if a detector tile is an
        # engine tile; a mis-scaled grid would put units in a sub-range of the
        # board and look exactly like an engine speed bug. Units rarely go past
        # the King's row, so the check asks for the deepest normal position, a
        # unit engaging an enemy Princess Tower (row 27).
        span_ok = (min(xs) <= 2 and max(xs) >= 15
                   and min(ys) <= 4 and max(ys) >= 26)
        verdict = "OK" if span_ok else (
            "*** units do not span the board -- speeds may be scaled ***")
        add(f"tile grid : x {min(xs)}..{max(xs)} of 0..17, "
            f"y {min(ys)}..{max(ys)} of 0..33   {verdict}")

    if result["floor"]:
        add("")
        add(f"reconstruction floor (rebuild + 1 tick vs the board it came "
            f"from): IoU {np.mean(result['floor']):.3f}")
        add("  Everything below is bounded by this. What it loses is HP,")
        add("  unmappable classes and cell collisions -- not dynamics.")

    add("")
    add("OCCUPANCY AGREEMENT vs the board that actually followed  "
        "(1.0 = perfect)")
    add("  `rebuilt` = reconstructed but NOT stepped, the fair baseline for")
    add("  the scaled columns; `stale` pays no reconstruction cost at all.")
    add("  Each column steps the engine by horizon*SCALE. If the engine runs")
    add("  N times too fast, the best column is at scale 1/N -- that column")
    add("  IS the measurement of the speed error, and it needs no C++ change.")
    add("")
    scales = result["scales"]
    header = f"  {'horizon':>8}  {'stale':>7}  {'rebuilt':>8}"
    for sc in scales:
        header += f"  {('x' + format(sc, 'g')):>8}"
    add(header + "   best")
    for h in horizons:
        st = np.array(result["stale"][h], float)
        if not len(st):
            continue
        rb = np.array(result["rebuilt"][h], float)
        row = f"  {h:>7.1f}s  {st.mean():>7.3f}  {rb.mean():>8.3f}"
        means = {}
        for sc in scales:
            v = np.array(result["scores"][(sc, h)], float)
            means[sc] = v.mean() if len(v) else float("nan")
            row += f"  {means[sc]:>8.3f}"
        best = max(means, key=lambda k: means[k])
        # Judged against `rebuilt`, not `stale`: both paid the reconstruction
        # cost, so the difference is stepping alone.
        tag = f"x{best:g}"
        if means[best] > rb.mean():
            tag += " beats rebuilt"
            tag += " AND stale" if means[best] > st.mean() else " (not stale)"
        else:
            tag = "none (stepping never helps)"
        add(row + f"   {tag}")
    add(f"  n = {len(np.array(result['stale'][horizons[0]], float))} per cell")

    add("")
    add(f"SPEED, tiles/second -- engine vs real footage "
        f"(real measured over the {max(horizons):.1f}s horizon)")
    add("  `real p90` is the fair comparison: the engine figure is a")
    add("  FREE-WALKING unit, while the real samples include every unit that")
    add("  stopped to fight, which drags the median down and would flatter")
    add("  the engine. p90 approximates a real unit that walked the whole")
    add("  horizon, so `ratio p90` is the conservative number.")
    add("")
    add(f"  {'card':<16} {'engine':>7}  {'real med':>8}  {'real p90':>8}  "
        f"{'ratio p90':>9}  {'n':>5}")
    forecaster = result["forecaster"]
    from live.unit_to_card import card_id_for  # noqa: PLC0415
    longest = max(horizons)
    rows = []
    for (name, horizon), samples in sorted(result["real_speed"].items()):
        if horizon != longest or len(samples) < 5:
            continue
        try:
            engine = forecaster.measure_speed(card_id_for(name))
        except Exception:                                   # noqa: BLE001
            continue
        if not np.isfinite(engine) or engine <= 0.05:
            continue        # a building: not moving is not a speed gap
        med = float(np.median(samples))
        p90 = float(np.percentile(samples, 90))
        ratio = engine / p90 if p90 > 0 else float("inf")
        rows.append((name, engine, med, p90, ratio, len(samples)))
    for name, engine, med, p90, ratio, n in sorted(rows, key=lambda r: -r[4]):
        add(f"  {name:<16} {engine:>7.2f}  {med:>8.2f}  {p90:>8.2f}  "
            f"{ratio:>8.1f}x  {n:>5}")
    if rows:
        add("")
        add(f"  median ratio across {len(rows)} classes: "
            f"{np.median([r[4] for r in rows]):.1f}x  "
            f"(vs real median: {np.median([r[1] / r[2] for r in rows if r[2] > 0]):.1f}x)")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recordings", type=Path,
                        default=_PERCEPTION_ROOT / "assets" / "recordings")
    parser.add_argument("--samples", type=int, default=12,
                        help="anchor frames per recording")
    parser.add_argument("--horizons", type=str,
                        default=",".join(str(h) for h in DEFAULT_HORIZONS))
    parser.add_argument("--limit", type=int, default=0,
                        help="use only the first N recordings")
    parser.add_argument("--time-scale", type=str, default="1.0",
                        help="comma-separated scales. Steps the engine by "
                             "h*SCALE to predict h; all scales share one pass "
                             "over the recordings. 0.2 tests a 5x-too-fast "
                             "engine.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    horizons = tuple(float(h) for h in args.horizons.split(","))
    paths = sorted(args.recordings.glob("*.mp4"))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f"no recordings in {args.recordings}")

    scales = tuple(float(x) for x in str(args.time_scale).split(","))
    print(f"{len(paths)} recordings, {args.samples} anchors each, "
          f"horizons {horizons}, scales {scales}")
    result = run(paths, args.samples, horizons, time_scales=scales)
    text = report(result, horizons)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
