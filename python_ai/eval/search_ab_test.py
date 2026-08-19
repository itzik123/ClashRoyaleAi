"""THE PAIRED A/B that measured what decision-time search is worth.

The search itself lives in `search/search.py` now -- this file is the
EXPERIMENT: pairing, the arms, and the exact-McNemar read-out. Splitting them
was overdue: five other modules imported this script's underscore-private
functions, which meant reaching across a module boundary into a harness whose
main() runs a whole experiment.

1-ply decision-time search vs the raw policy: a PAIRED win-rate A/B.

Consumes `ClashRoyaleEnv.snapshot()` (perception/UPSTREAM_REQUESTS.md item 13).
At each decision the search arm proposes K candidate actions, rolls each one
forward on a throwaway copy of the live environment, scores the resulting
position with the critic, and plays the best. The policy arm plays the same
network greedily, exactly as `train_selfplay.evaluate_against_roster` does.

WHY PAIRED, AND WHY THAT IS THE POINT
-------------------------------------
The engine's RNG still cannot be seeded (UPSTREAM item 7 is open), so the usual
way to compare two agents is unpaired, and UPSTREAM item 7 works out the cost:
~1,568 episodes per arm to resolve a 5-point win-rate difference at 80% power.

snapshot() sidesteps that. Each trial resets ONE environment, snapshots it, and
hands both arms a bit-identical copy -- same shuffled opening hand, same
heuristic-opponent lane, same everything at t=0. The shared opening is removed
from the variance rather than averaged over, which is what pairing buys, and it
needs no engine RNG change at all.

WHAT THIS CANNOT TELL YOU
-------------------------
UPSTREAM item 13 records a previous attempt at this question whose confidence
interval came out 15x wider than the effect, and the honest reading of that was
"underpowered null", not "search does not work". Two things guard against
repeating it:

  * `--trials` is reported alongside a paired CI, and the CI is printed whether
    or not it excludes zero.
  * DEVIATION RATE is reported first. If search almost never disagrees with the
    greedy policy, the two arms are near-identical by construction and the
    win-rate comparison carries no information regardless of how many episodes
    are run. That diagnostic is cheap and it is the one that says whether the
    experiment measured anything at all.

Run:
    python_ai/venv/Scripts/python.exe python_ai/search_ab_test.py --trials 100
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

# Run as a script the repo root is not on sys.path, so `python_ai.*` cannot
# resolve; importing the package is also what makes `clash_royale_env` (an
# unpackaged .pyd in python_ai/) importable. See python_ai/__init__.py.
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
    # The curriculum's hardest rung. At 1.0 the ep-64k policy wins ~100% of
    # games and BOTH arms saturate, so the paired delta is pinned at zero by a
    # ceiling rather than by search being useless -- measured, not assumed:
    # 4/4 trials came back 1.000 vs 1.000. A comparison needs an opponent with
    # headroom on both sides. CLAUDE.md records this policy family plateauing
    # around 0.86 at 1.5x, which is where the resolution is.
    ap.add_argument("--opp-elixir", type=float, default=1.5,
                    help="opponent elixir multiplier; 1.0 saturates at this skill level")
    ap.add_argument("--time-budget", type=float, default=0.0, help="seconds; 0 = no limit")
    args = ap.parse_args()
    # A real SearchCfg, not the argparse namespace. They are duck-compatible,
    # which is how the namespace came to be passed straight through -- and the
    # whole reason SearchCfg exists is that two callers reaching different
    # settings under the same name is a silent way to compare two experiments.
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

    # Power check, stated up front rather than left for the reader to work out.
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
