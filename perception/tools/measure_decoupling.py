"""Does stepping the engine forward beat acting on the stale board?

The board the policy acts on is old (a detector pass plus the rest of
perceive()), so temporal decoupling proposes spending ~1 ms of engine time
advancing it to now. That pays only if the engine's prediction is closer to
what the game did next than the stale board is, which is not obvious: rebuilt
units get a fresh deploy second, projectile speed is uncalibrated, and a
forecast assumes both sides no-op.

Both arms are reconstructed through the same reset+inject path and differ only
in whether the engine is then stepped, so this isolates the engine's dynamics;
detector misses affect both arms equally. The ground truth is the
reconstruction of a later frame, so a perfect score means "the engine predicts
what perception will see", the quantity the policy consumes, not that the
engine models the game (sim_fidelity.py asks that).

Forecasting does not make perception faster; it can only reduce staleness, the
age of the world model when a decision is made.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import engine as _engine_build  # noqa: F401,E402 -- fresh-build path first

from clashroyalebuildabot.constants import (  # noqa: E402
    SCREENSHOT_HEIGHT,
    SCREENSHOT_WIDTH,
)
from clashroyalebuildabot.detectors.detector import Detector  # noqa: E402
from forecast import MIN_HORIZON_S, SimForecaster, agreement, occupancy, tower_cells  # noqa: E402
from live.adapter import build_game_state  # noqa: E402
from live.mvp_loop import DECK, _training_deck_ids  # noqa: E402

CAPTURE_FPS = 5.0

# Horizons must be multiples of the sample spacing, or no ground-truth frame
# exists at t + horizon and the pair is dropped (at stride 2 on a 5 fps dump
# the spacing is 0.4 s).
HORIZON_STEPS = [1, 2, 3, 4, 5, 6, 8]


def collect(frames_dir: Path, stride: int, limit: int | None):
    """Detector output -> GameState, for in-match frames only."""
    detector = Detector(list(DECK))
    paths = sorted(frames_dir.glob("f*.png"))[::stride]
    if limit:
        paths = paths[:limit]

    out = []
    for index, path in enumerate(paths):
        native = Image.open(path).convert("RGB")
        small = native.resize((SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT), Image.LANCZOS)
        state = detector.run(small)
        if state is None or state.screen.name != "in_game":
            continue
        wall_ms = index * stride * (1000.0 / CAPTURE_FPS)
        gs, _report = build_game_state(
            state, np.array(native), np.array(small),
            frame_index=index, wall_time_ms=wall_ms, my_elixir_spent=0.0)
        out.append((wall_ms / 1000.0, gs))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path,
                        default=_ROOT / "assets" / "live" / "match_practice_01")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print("running the detector over the capture (this is the slow part) ...")
    samples = collect(args.frames, args.stride, args.limit)
    print(f"{len(samples)} in-match GameStates\n")
    if len(samples) < 10:
        print("not enough in-match frames to measure")
        return 1

    forecaster = SimForecaster(deck=_training_deck_ids())
    by_time = {round(t, 3): gs for t, gs in samples}
    times = sorted(by_time)

    # Cost first: the proposition is that this is cheap.
    costs_ms = []
    for _t, gs in samples[:40]:
        t0 = time.perf_counter()
        forecaster.forecast(gs, [1.0])
        costs_ms.append((time.perf_counter() - t0) * 1000.0)
    costs_ms.sort()
    print(f"forecast cost: p50 {costs_ms[len(costs_ms)//2]:.2f} ms   "
          f"p95 {costs_ms[int(len(costs_ms)*0.95)]:.2f} ms   "
          f"max {costs_ms[-1]:.2f} ms")
    print("(one perceive() cycle costs ~502 ms, of which detector.run is ~290)\n")

    print(f"{'horizon':>8} {'n':>5} {'STALE IoU':>10} {'FORECAST':>10} "
          f"{'delta':>8} {'better':>7} {'worse':>6}")
    print("-" * 60)

    spacing = args.stride / CAPTURE_FPS
    for steps in HORIZON_STEPS:
        horizon = round(steps * spacing, 3)
        stale_scores, fc_scores = [], []
        better = worse = 0
        for t in times:
            t_future = round(t + horizon, 3)
            actual_gs = by_time.get(t_future)
            if actual_gs is None:
                continue
            base_gs = by_time[t]

            # Ground truth: the later board reconstructed the same way, not
            # stepped.
            actual_obs = forecaster.forecast(actual_gs, [MIN_HORIZON_S])[0].observation
            stale_obs = forecaster.forecast(base_gs, [MIN_HORIZON_S])[0].observation
            fc_obs = forecaster.forecast(base_gs, [horizon])[0].observation

            # Both teams: the side classifier labels almost every detected unit
            # as team 1 here, so an own-units metric compares two empty sets.
            # The enemy's push is also what a defensive decision is made
            # against.
            towers = tower_cells()
            actual_cells = ((occupancy(actual_obs, 0) | occupancy(actual_obs, 1))
                            - towers)
            stale_cells = ((occupancy(stale_obs, 0) | occupancy(stale_obs, 1))
                           - towers)
            fc_cells = ((occupancy(fc_obs, 0) | occupancy(fc_obs, 1)) - towers)
            # An empty board either way says nothing about whether stepping
            # helped; it would only pad both arms with 1.0.
            if not actual_cells and not stale_cells and not fc_cells:
                continue

            s = agreement(stale_cells, actual_cells)
            f = agreement(fc_cells, actual_cells)
            stale_scores.append(s)
            fc_scores.append(f)
            if f > s + 1e-9:
                better += 1
            elif s > f + 1e-9:
                worse += 1

        if len(stale_scores) < 5:
            print(f"{horizon:>7.1f}s {len(stale_scores):>5}   (too few pairs)")
            continue
        s_mean = statistics.mean(stale_scores)
        f_mean = statistics.mean(fc_scores)
        flag = "  <-- WORSE" if f_mean < s_mean else ""
        print(f"{horizon:>7.1f}s {len(stale_scores):>5} {s_mean:>10.4f} "
              f"{f_mean:>10.4f} {f_mean - s_mean:>+8.4f} {better:>7} {worse:>6}{flag}")

    # Time-scale sweep: predict t+H by stepping k*H, scored against the board
    # at t+H. k=0 is the stale board, k=1 a faithful forecast. No k helping
    # means the engine moves things the wrong way; an interior k winning means
    # too far. sim_fidelity.py runs the same sweep against the recordings.
    sweep_h = round(3 * spacing, 3)
    print(f"\ntime-scale sweep at horizon {sweep_h}s "
          f"(k=0 is the stale board, k=1 a faithful forecast):")
    print(f"{'k':>6} {'IoU':>10}")
    for k in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
        scores = []
        for t in times:
            actual_gs = by_time.get(round(t + sweep_h, 3))
            if actual_gs is None:
                continue
            a_obs = forecaster.forecast(actual_gs, [MIN_HORIZON_S])[0].observation
            p_obs = forecaster.forecast(by_time[t], [max(k * sweep_h,
                                                         MIN_HORIZON_S)])[0].observation
            towers = tower_cells()
            a = (occupancy(a_obs, 0) | occupancy(a_obs, 1)) - towers
            p = (occupancy(p_obs, 0) | occupancy(p_obs, 1)) - towers
            if not a and not p:
                continue
            scores.append(agreement(p, a))
        if scores:
            print(f"{k:>6.2f} {statistics.mean(scores):>10.4f}")

    print("\nIoU of non-tower occupancy against the later board.")
    print("A POSITIVE delta means stepping the engine forward is closer to what")
    print("perception saw next than doing nothing. A negative one means the")
    print("stale board is the better estimate and decoupling should NOT ship.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
