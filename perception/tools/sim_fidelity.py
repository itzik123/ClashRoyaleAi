"""How closely does the engine's physics match the real game?

Takes a real board off a recording, steps it forward with the ENGINE, and
compares against what the recording actually shows one horizon later. Three
numbers per horizon:

    floor       rebuilding the board in the engine and stepping the minimum
                one tick. This is the ceiling on every other number here --
                whatever it loses is reconstruction, not dynamics, and a
                prediction score cannot be read without it.
    stale       the board as it was, scored against the board that followed.
                This is what the live loop acts on today.
    predicted   the engine's forecast, scored the same way.

`predicted > stale` means the engine's dynamics carry real information about
the next second and the ~1 s of end-to-end latency is partly recoverable.
`predicted < stale` means stepping this engine forward actively misinforms,
and -- more importantly -- that the agent is training against physics the real
game does not share.

WHY OCCUPANCY, AND WHY TOWERS ARE EXCLUDED
------------------------------------------
Scored as intersection-over-union of occupied cells. Unit HP, tower HP, elixir
and the match clock cannot be reconstructed at all (no setters exist -- see
forecast.py), so any whole-observation distance would be dominated by those
resets rather than by dynamics. Occupancy survives all of them.

Towers are excluded because they never move: leaving them in adds six
guaranteed matches to every comparison, which on a quiet board is most of the
score.

SPEED IS MEASURED ON BOTH SIDES
-------------------------------
The engine's own tiles/second comes from injecting and measuring, not from
reading a constant. The real game's comes from card classes that appear
exactly once on a side in both frames of a pair, where displacement is
unambiguous without a tracker. Comparing the two is the single most direct
statement of where the physics differ, and it needs no forecast at all.

CALIBRATION CAVEATS THAT APPLY TO EVERY NUMBER BELOW
----------------------------------------------------
  * `TILE_Y_OFFSET` is derived from two board heights, not measured. A
    constant row error shifts both sides equally, so it damages the
    engine-vs-real speed comparison little, but it is not zero.
  * The detector is the same one the live loop uses, so its errors are in
    both `stale` and `predicted` -- which is the right control, since the
    live loop would eat them too.
  * The engine has NO deploy time; the real game freezes a troop ~1 s after
    it lands. Newly placed units are therefore predicted worst, and they are
    the ones that matter most.
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

# The detector logs a DEBUG line per frame, which at several hundred frames
# buries the report it is being run to produce.
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

# The recordings are 1920x1080 DESKTOP captures; the emulator occupies this
# sub-rectangle (CLAUDE.md, "Recording real matches"). Verified rather than
# assumed: uncropped, the elixir reader returns 0 and the detector finds five
# phantom enemies; cropped it reads a plausible 6 with sane unit counts.
EMULATOR_CROP = (686, 40, 1236, 1012)      # left, top, right, bottom

# Menus and the end screen produce confident nonsense, and the CRBAB screen
# classifier's `in_game` hash was cut from a different emulator profile so it
# reports `unknown` on these files. Elixir reads 0 off-match and essentially
# never during play, which makes it the more reliable gate here.
SKIP_HEAD_FRAC = 0.15
SKIP_TAIL_FRAC = 0.10

DEFAULT_HORIZONS = (0.5, 1.0, 2.0)


def perceive(detector, frame):
    """The live loop's own perception path, minus the ledger and the debit.

    Deliberately the same calls in the same order as `mvp_loop.perceive`: a
    harness that measured a DIFFERENT pipeline from the one that plays would
    be measuring nothing anyone runs.
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
    """Occupied cells straight from perception, towers not included.

    Built from the GameState rather than by encoding it and reading the
    observation back: the encoder is verified byte-exact against the engine,
    so the round trip would be a slower way to get the same set, and one more
    place for a layout assumption to hide.
    """
    return {(u.tile_x, u.tile_y) for u in game_state.units
            if int(u.team) == int(team) and int(u.card_sim_id) >= 0}


def sole_positions(game_state):
    """{card name: (x, y)} for classes with exactly one body on one side.

    Association without a tracker. If a side shows exactly one Musketeer in
    both frames of a pair, the displacement between them is unambiguous; two
    Musketeers and it is a guess. Restricting to the unambiguous case gives
    fewer samples of a number that means something, rather than more of one
    that does not.
    """
    by_class = defaultdict(list)
    for u in game_state.units:
        if int(u.card_sim_id) >= 0:
            by_class[(u.unit_name, int(u.team))].append((u.tile_x, u.tile_y))
    return {k: v[0] for k, v in by_class.items() if len(v) == 1}


def sanity_check_frame(game_state):
    """Our units low-y, theirs high-y, or the whole comparison is mirrored.

    Costs nothing and catches the one class of fault this project has already
    paid for twice -- a coordinate frame that is silently flipped or offset
    looks exactly like 'the engine's physics are wrong'.
    """
    mine = [u.tile_y for u in game_state.units if int(u.team) == 0]
    theirs = [u.tile_y for u in game_state.units if int(u.team) == 1]
    if not mine or not theirs:
        return None
    return float(np.mean(mine)), float(np.mean(theirs))


def run(paths, samples_per_video, horizons, time_scales=(1.0,), verbose=False):
    """Steps the engine by `h * scale` to predict `h`, for each scale given.

    The decisive control. If the engine's troops move N times too fast, then
    dynamics that are otherwise correct would predict real footage well at
    `scale = 1/N` and badly at 1.0. Sweeping it separates "the engine models
    the wrong physics" from "the engine models the right physics at the wrong
    rate", which need completely different responses -- the second is one
    constant, the first is a rewrite.

    ALL SCALES SHARE ONE PASS over the recordings. Perception is ~99% of the
    cost here and a forecast is microseconds, so re-running the detector per
    scale would multiply a 20-minute measurement by the number of scales for
    no new information. Sharing the pass also makes the comparison PAIRED:
    every scale is scored on exactly the same boards, so a difference between
    them cannot be sampling.
    """
    detector = Detector(DECK)
    costs, _warn = deck_costs(DECK)
    forecaster = SimForecaster([c for c in _engine_deck()])
    towers = tower_cells()

    # `stale` does not depend on the scale, so it is scored once per horizon
    # rather than recomputed identically for every scale.
    stale_scores = {h: [] for h in horizons}
    # "Reconstruct but do not step", scored against the SAME truth. Without
    # this, `pred` is compared against a `stale` that never paid the
    # reconstruction cost, so a perfect predictor still loses -- the ceiling
    # on any rebuilt board is the floor (~0.52), while `stale` has none.
    # This is the control that isolates the effect of STEPPING.
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
                # horizon, against the board it was built FROM.
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

                    # Real-game speed, from unambiguous single-instance
                    # classes. Keyed BY HORIZON, because the shortest ones
                    # cannot resolve it: a real unit at ~0.75 tiles/s covers
                    # 0.4 tiles in 0.5 s, which quantises to 0 or 1 cell --
                    # i.e. 0 or 2 tiles/s. Only the longest horizon carries
                    # enough displacement for the number to mean anything.
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
    """DECK as engine card ids -- the same mapping the live loop uses."""
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
        # Speed in tiles/second is only meaningful if a detector tile IS an
        # engine tile. If the grid were scaled wrong, units would occupy a
        # sub-range of the board and every speed below would be wrong by the
        # same factor -- which would look exactly like an engine speed bug.
        # The enemy Princess towers sit on row 27 and the King on row 30, and
        # rows past the King are behind everything worth attacking -- units
        # essentially never go there. An earlier version of this check demanded
        # max(y) >= 29 and raised a false alarm on a perfectly good grid
        # measuring 1..28. The deepest NORMAL position is a unit engaging an
        # enemy Princess tower, so that is what the threshold is set to.
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
        # cost, so the difference between them is the effect of stepping
        # alone. Beating `stale` too is the stronger, separate claim.
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
            continue        # a building; "does not move" is not a speed gap
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
