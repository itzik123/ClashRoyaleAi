"""Watch a live run for the four quantities the placement work turns on.

WHY A SEPARATE PROCESS. `probe_card_discrimination` replays a multi-thousand
state bank through the LSTM -- 27 s for 3,611 states. Calling that from inside
`train.py`'s loop would stall the rollout every time it fired, and `train.py` is
the live training script. This reads the CHECKPOINT and the TensorBoard event
file only, never the trainer's memory, so it cannot perturb the run it watches
-- the same contract `tools/monitor_run.py` already keeps.

WHAT IT WATCHES, AND WHAT EACH IS READ AGAINST
-----------------------------------------------
  quantity            baseline it only means something against
  ------------------  ---------------------------------------------------
  hi/lo ratio         its own value at the run's start. A marginal that
                      rises with a FLAT ratio is SYSTEMIC DRIFT, not
                      learning -- the failure this project has recorded
                      six times, and the exact shape of the 2026-09-03
                      deck-pool result.
  quality_hi          the share of achievable placement value the head
                      expects to collect. Fireball 0.675, The Log 0.203 at
                      ep 32,875. This is the number 490 episodes of deck
                      pool did NOT move (signal 0.0).
  Aux/NextCard_CE     ln(185) = 5.22 is uniform. The stale-head failure
                      read 13.76 -- worse than uniform is "confidently
                      wrong", and at aux_card_scale it pulled the shared
                      trunk ~7x harder toward card identities than toward
                      winning.
  Decks/WinRate_*     POOL_WINRATE_FLOOR = 0.20. A deck below it should
                      LOSE episode share and climb back out on its own.

SIGNAL BEFORE TREND. Every card row carries |delta| over the step-to-step
scatter of the series so far. Below ~2 a movement is not separable from PPO
jitter, and this project has already been fooled once by a three-point read of
a series whose fourth point reversed it.

Usage:

    ... -m python_ai.tools.track_placement --bank python_ai/eval/banks/<b>.npz \\
        --weights model_weights_phase7.pth --run-dir runs/phase7 \\
        --csv python_ai/eval/banks/trend_phase7.csv --every 900
"""
import argparse
import csv
import glob
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402

UNIFORM_NEXT_CARD_CE = float(np.log(185))   # ln(NUM_CARD_IDS)
POOL_WINRATE_FLOOR = 0.20                   # opponents/deck_pool.py


def scalars(run_dir):
    """Every scalar series in the newest event file under `run_dir`."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator)
    except ImportError:
        return {}
    cands = [p for p in glob.glob(os.path.join(run_dir, "*"))
             if "tfevents" in os.path.basename(p)]
    cands += glob.glob(os.path.join(run_dir, "runs", "*"))
    if not cands:
        return {}
    ea = EventAccumulator(max(cands, key=os.path.getmtime))
    ea.Reload()
    return {tag: [(e.step, e.value) for e in ea.Scalars(tag)]
            for tag in ea.Tags().get("scalars", [])}


def last(series, n=1):
    if not series:
        return None
    return float(np.mean([v for _s, v in series[-n:]]))


def episodes_of(path):
    import torch
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    return ck.get("episodes_completed") if isinstance(ck, dict) else None


def score_checkpoint(weights, bank, csv_path):
    """Run the discrimination probe as a subprocess and append to the CSV.

    A subprocess rather than an import so a probe that dies -- a half-written
    checkpoint being the obvious way -- cannot take the watcher down with it.
    """
    cmd = [sys.executable, "-m", "python_ai.eval.probe_card_discrimination",
           "--bank", bank, "--weights", weights, "--csv", csv_path]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=python_ai.REPO_ROOT)
    if r.returncode != 0:
        print(f"  probe failed ({r.returncode}): {r.stderr.strip()[-300:]}",
              flush=True)
        return False
    return True


def read_trend(csv_path):
    """{card: [rows]} from the trend CSV, oldest first."""
    if not os.path.exists(csv_path):
        return {}
    out = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            out.setdefault(row["card"], []).append(row)
    return out


def signal(series):
    """|last - first| over the step-to-step scatter of the series.

    The project's own rule: below ~2 a trend is not separable from PPO jitter.
    Needs at least three points, because two define a line through any noise.
    """
    v = np.array([x for x in series if np.isfinite(x)], dtype=np.float64)
    if v.size < 3:
        return float("nan")
    jitter = float(np.abs(np.diff(v)).mean())
    if jitter < 1e-12:
        return float("inf") if abs(v[-1] - v[0]) > 1e-12 else 0.0
    return abs(v[-1] - v[0]) / jitter


def report(csv_path, run_dir):
    trend = read_trend(csv_path)
    print(f"\n=== {time.strftime('%H:%M:%S')} ===", flush=True)
    if trend:
        first_ep = trend[next(iter(trend))][0]["episode"]
        last_ep = trend[next(iter(trend))][-1]["episode"]
        print(f"  bank trend, ep {first_ep} -> {last_ep} "
              f"({len(next(iter(trend.values())))} points)")
        print(f"  {'card':<10}{'marginal':>18}{'sig':>6}{'hi/lo':>16}{'sig':>6}"
              f"{'place_q':>18}{'sig':>6}")
        for card, rows in trend.items():
            marg = [float(r["marginal"]) for r in rows]
            ratio = [float(r["ratio"]) for r in rows]
            qual = [float(r["quality_hi"]) for r in rows]
            q = ("n/a" if not np.isfinite(qual[-1])
                 else f"{qual[0]:.3f}->{qual[-1]:.3f}")
            print(f"  {card:<10}"
                  f"{f'{marg[0]:.4f}->{marg[-1]:.4f}':>18}"
                  f"{signal(marg):>6.1f}"
                  f"{f'{ratio[0]:.2f}->{ratio[-1]:.2f}':>16}"
                  f"{signal(ratio):>6.1f}"
                  f"{q:>18}"
                  f"{signal(qual):>6.1f}", flush=True)
        print("  READ THE RATIO, NOT THE MARGINAL: usage up with a flat ratio "
              "is systemic drift.")

    s = scalars(run_dir)
    ce = last(s.get("Aux/NextCard_CE", []), n=3)
    if ce is not None:
        verdict = ("OK" if ce < UNIFORM_NEXT_CARD_CE
                   else "*** ABOVE UNIFORM -- confidently wrong, poisoning the trunk ***")
        print(f"  Aux/NextCard_CE {ce:.3f}  vs uniform ln(185)="
              f"{UNIFORM_NEXT_CARD_CE:.2f}   {verdict}", flush=True)
    wr = last(s.get("Training/Win_Rate_100", []))
    spread = last(s.get("Decks/WinRate_Spread", []))
    wmin = last(s.get("Decks/WinRate_Min", []))
    if wr is not None:
        print(f"  win rate(100) {wr:.3f}   deck spread {spread if spread is None else round(spread,3)}"
              f"   worst deck {wmin if wmin is None else round(wmin,3)}"
              f"   (floor {POOL_WINRATE_FLOOR})", flush=True)
    per_deck = {t.split("/")[-1]: last(v) for t, v in s.items()
                if t.startswith("Decks/WinRate/")}
    if per_deck:
        below = {k: v for k, v in per_deck.items() if v is not None
                 and v < POOL_WINRATE_FLOOR}
        order = sorted(per_deck.items(), key=lambda kv: (kv[1] is None, kv[1]))
        worst = ", ".join(f"{k} {v:.2f}" for k, v in order[:3] if v is not None)
        print(f"  {len(per_deck)} decks tracked, {len(below)} under the floor"
              f"   worst: {worst}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--every", type=float, default=900.0,
                    help="seconds between checks")
    ap.add_argument("--min-episodes", type=int, default=150,
                    help="don't re-score until the checkpoint has advanced "
                         "this many episodes -- scoring the same weights twice "
                         "adds a point to the trend that carries no information "
                         "and deflates the jitter estimate `signal` divides by")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    weights = args.weights
    if not os.path.isabs(weights):
        weights = os.path.join(python_ai.PACKAGE_DIR, weights)

    seen = None
    trend = read_trend(args.csv)
    if trend:
        seen = int(next(iter(trend.values()))[-1]["episode"])
        print(f"resuming trend at ep {seen}")

    while True:
        ep = episodes_of(weights)
        if ep is None:
            print(f"  no readable checkpoint at {weights}", flush=True)
        elif seen is not None and ep - seen < args.min_episodes:
            print(f"  ep {ep} (+{ep - seen}) -- waiting for "
                  f"{args.min_episodes}", flush=True)
        else:
            if score_checkpoint(weights, args.bank, args.csv):
                seen = ep
                report(args.csv, args.run_dir)
        if args.once:
            break
        time.sleep(args.every)


if __name__ == "__main__":
    main()
