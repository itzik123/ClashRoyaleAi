"""Paired win-rate A/B: 1-ply decision-time search vs the raw policy.

The search lives in `search/search.py`; this is the experiment. At each
decision the search arm rolls K candidates forward on a copy of the
environment, scores each with the critic and plays the best; the policy arm
plays the same net greedily.

Each trial resets one environment and hands both arms a bit-identical snapshot
of it (same hand, same opponent lane), removing the opening from the variance.

Read the deviation rate first: if search almost never disagrees with greedy,
the arms are near-identical and the win-rate comparison carries no information
at any n. The paired CI is printed whether or not it excludes zero.

    python_ai/venv/Scripts/python.exe python_ai/eval/search_ab_test.py --trials 100
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

# Run as a script, the repo root is not on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env  # noqa: E402
from python_ai.envs.gym_wrapper import DEFAULT_DECK  # noqa: E402
from python_ai.models.net import MicroRoyaleNet  # noqa: E402
from python_ai.models.policy_io import load_state_dict_flexible  # noqa: E402
from python_ai.search.config import SearchCfg  # noqa: E402
from python_ai.search.search import outcome_score, play_episode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100, help="paired episodes (2 games each)")
    ap.add_argument("--weights", default="model_weights_selfplay.pth")
    ap.add_argument("--horizon", type=int, default=4, help="decision steps rolled forward (1 = 1s)")
    ap.add_argument("--k-cards", type=int, default=3)
    ap.add_argument("--k-cells", type=int, default=2)
    ap.add_argument("--terminal-weight", type=float, default=10.0)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--max-ticks", type=int, default=3600)
    # At 1.0 both arms saturate near 100% and the delta is pinned at zero by
    # the ceiling; 1.5 leaves headroom on both sides.
    ap.add_argument("--opp-elixir", type=float, default=1.5,
                    help="opponent elixir multiplier; 1.0 saturates at this skill level")
    ap.add_argument("--time-budget", type=float, default=0.0, help="seconds; 0 = no limit")
    args = ap.parse_args()
    # A real SearchCfg rather than the argparse namespace, so every caller
    # names the same settings.
    search = SearchCfg(horizon=args.horizon, k_cards=args.k_cards,
                       k_cells=args.k_cells,
                       terminal_weight=args.terminal_weight,
                       max_steps=args.max_steps)
    cfg = args

    device = torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    here = python_ai.PACKAGE_DIR
    weights_path = cfg.weights if os.path.isabs(cfg.weights) else os.path.join(here, cfg.weights)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    net = MicroRoyaleNet().to(device)
    clean = load_state_dict_flexible(net, state, weights_path)
    net.eval()

    episodes = ckpt.get("episodes_completed", "?") if isinstance(ckpt, dict) else "?"
    print(f"weights          : {os.path.basename(weights_path)} "
          f"(episodes_completed={episodes}, clean_load={clean})")
    print(f"search           : K<={search.max_candidates} candidates, "
          f"horizon={cfg.horizon} steps ({cfg.horizon}s), critic-scored")
    print(f"opponent         : C++ HeuristicOpponent at {cfg.opp_elixir}x elixir")
    print(f"pairing          : both arms start from one snapshot of the same reset")
    print(f"trials           : {cfg.trials} paired ({2 * cfg.trials} episodes)")
    print()

    diffs, a_scores, b_scores = [], [], []
    dev_total, dev_steps, cand_total = 0, 0, 0
    a_time = b_time = 0.0
    started = time.perf_counter()

    for trial in range(cfg.trials):
        if cfg.time_budget and (time.perf_counter() - started) > cfg.time_budget:
            print(f"\n[time budget reached after {trial} trials]")
            break

        root = clash_royale_env.ClashRoyaleEnv(list(DEFAULT_DECK), list(DEFAULT_DECK), cfg.max_ticks)
        root.set_opponent_elixir_multiplier(cfg.opp_elixir)
        root.reset()
        base = root.snapshot()  # the shared opening both arms will play

        t0 = time.perf_counter()
        r_a, steps_a, _, _ = play_episode(net, base.snapshot(), device, False, search)
        a_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        r_b, steps_b, dev, cands = play_episode(net, base.snapshot(), device, True, search)
        b_time += time.perf_counter() - t0

        dev_total += dev
        dev_steps += steps_b
        cand_total += cands

        sa, sb = outcome_score(r_a), outcome_score(r_b)
        a_scores.append(sa)
        b_scores.append(sb)
        diffs.append(sb - sa)

        if (trial + 1) % 10 == 0:
            n = len(diffs)
            print(f"  trial {trial + 1:4d} | policy {np.mean(a_scores):.3f} "
                  f"search {np.mean(b_scores):.3f} | delta {np.mean(diffs):+.3f} "
                  f"| deviation {dev_total / max(1, dev_steps):.1%} "
                  f"| {(time.perf_counter() - started) / n:.1f}s/trial")

    n = len(diffs)
    if n == 0:
        print("no trials completed")
        return

    diffs = np.asarray(diffs)
    mean_d = float(diffs.mean())
    se = float(diffs.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    lo, hi = mean_d - 1.96 * se, mean_d + 1.96 * se

    print("\n" + "=" * 68)
    print(f"trials (paired)        : {n}")
    print(f"policy   win rate      : {np.mean(a_scores):.4f}")
    print(f"search   win rate      : {np.mean(b_scores):.4f}")
    print(f"paired delta           : {mean_d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  significant?         : {'YES' if (lo > 0 or hi < 0) else 'no -- CI includes 0'}")

    wins = int((diffs > 0).sum())
    losses = int((diffs < 0).sum())
    ties = int((diffs == 0).sum())
    print(f"  search better/worse/same : {wins} / {losses} / {ties}")

    print(f"\ndeviation rate         : {dev_total / max(1, dev_steps):.2%} "
          f"({dev_total} of {dev_steps} decisions)")
    print(f"mean candidates/decision: {cand_total / max(1, dev_steps):.2f}")
    if dev_total == 0:
        print("  !! search NEVER disagreed with the greedy policy. The two arms are")
        print("     the same agent, so the win-rate comparison above measures nothing")
        print("     whatever its CI says.")

    print(f"\nwall clock             : policy {a_time / n:.2f}s/ep, search {b_time / n:.2f}s/ep "
          f"({b_time / max(1e-9, a_time):.1f}x)")

    # Power check.
    if n > 1 and se > 0:
        detectable = 1.96 * se
        print(f"\nsmallest effect this n could resolve: +/-{detectable:.4f} "
              f"({detectable * 100:.1f} win-rate points)")
        if abs(mean_d) < detectable:
            needed = int(math.ceil((1.96 * diffs.std(ddof=1) / max(1e-6, abs(mean_d))) ** 2))
            print(f"observed effect is INSIDE the noise floor. Resolving an effect this "
                  f"size would need ~{needed} paired trials.")


if __name__ == "__main__":
    main()
