"""Per-decision latency budget for live decision-time search.

The emulator does not pause, so the tail matters, not the mean: a search whose
p99 exceeds the budget misses reaction windows. Measures the real
`search_action` path over mid-match states and reports p50/p95/p99 for a grid
of (horizon, K), plus the greedy failsafe as the floor.

Numbers taken while training runs are pessimistic (shared cores), the safe
direction for a budget.
"""
import argparse
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

HERE = python_ai.PACKAGE_DIR

import clash_royale_env as CE  # noqa: E402
from python_ai.envs.gym_wrapper import DEFAULT_DECK  # noqa: E402
from python_ai.search.config import SearchCfg  # noqa: E402
from python_ai.models.policy_io import load_net  # noqa: E402

E = CE.ClashRoyaleEnv
NOOP = E.HAND_SIZE


def percentile(xs, q):
    return float(np.percentile(np.asarray(xs), q))


@torch.no_grad()
def measure(net, device, cfg, n_decisions, opp_elixir, warmup=15):
    """Latency of one decision, with and without search, on live-ish states."""
    env = E(list(DEFAULT_DECK), list(DEFAULT_DECK), 3600)
    env.set_opponent_elixir_multiplier(opp_elixir)
    obs = env.reset()
    hidden = (torch.zeros(1, LSTM_HIDDEN, device=device),
              torch.zeros(1, LSTM_HIDDEN, device=device))

    greedy_ms, search_ms, ncands = [], [], []
    i = 0
    while len(search_ms) < n_decisions:
        obs_t = torch.tensor(np.asarray(obs, dtype=np.float32),
                             device=device).unsqueeze(0)

        t0 = time.perf_counter()
        card_logits, card_embeds, spatial_map, _value, hidden_next = policy_head(
            net, obs_t, hidden)
        gc, gx, gy, _ = greedy_from_logits(net, obs_t, card_logits, card_embeds,
                                            spatial_map, hidden_next)
        greedy = (gc, gx, gy)
        t1 = time.perf_counter()

        action, _, n = search_action(net, env, obs_t, card_logits, card_embeds,
                                      spatial_map, hidden_next, greedy, cfg, device)
        t2 = time.perf_counter()

        if i >= warmup:
            greedy_ms.append((t1 - t0) * 1000.0)
            # The live loop pays greedy plus the search, since greedy's logits
            # seed the candidates.
            search_ms.append((t2 - t0) * 1000.0)
            ncands.append(n)
        i += 1
        if i > 40 * n_decisions:      # guard against a state distribution that
            break                      # never produces a multi-candidate step

        res = env.step(action[0], action[1], action[2], 10)
        obs = res.observation
        hidden = hidden_next
        if res.done:
            obs = env.reset()
            hidden = (torch.zeros(1, LSTM_HIDDEN, device=device),
                      torch.zeros(1, LSTM_HIDDEN, device=device))
    return greedy_ms, search_ms, ncands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(HERE, "model_weights_selfplay.pth"))
    ap.add_argument("--decisions", type=int, default=60)
    ap.add_argument("--opp-elixir", type=float, default=1.4)
    ap.add_argument("--budget-ms", type=float, default=150.0)
    ap.add_argument("--threads", type=int, default=0,
                    help="torch threads; 0 = leave default")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    net = load_net(args.weights, device, verbose=False)
    for p in net.parameters():
        p.requires_grad_(False)

    print(f"torch threads = {torch.get_num_threads()}   budget = {args.budget_ms:.0f} ms")
    print(f"n = {args.decisions} decisions per configuration, opponent {args.opp_elixir}x")
    print()
    print(f"{'config':<26}{'n':>5}{'K':>5}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}   verdict")
    print("-" * 90)

    # Greedy alone: the failsafe and the floor.
    g, s, _ = measure(net, device, SearchCfg(horizon=1, k_cards=1, k_cells=1),
                      args.decisions, args.opp_elixir)
    print(f"{'GREEDY (failsafe)':<26}{len(g):>5}{1:>5}{percentile(g,50):>9.1f}"
          f"{percentile(g,95):>9.1f}{percentile(g,99):>9.1f}{max(g):>9.1f}   floor")
    print("-" * 90)

    rows = []
    for k_cards, k_cells in ((2, 1), (3, 2)):
        for horizon in (2, 4, 8, 12):
            cfg = SearchCfg(horizon=horizon, k_cards=k_cards, k_cells=k_cells)
            _, s, nc = measure(net, device, cfg, args.decisions, args.opp_elixir)
            s, nc = np.asarray(s), np.asarray(nc)
            # Only real searches: at K == 1 search_action returns without
            # rolling anything forward.
            real = s[nc >= 2]
            if real.size < 20:
                print(f"h={horizon} k_cards={k_cards} k_cells={k_cells}"
                      f"   only {real.size} multi-candidate steps -- SKIPPED")
                continue
            p99 = percentile(real, 99)
            ok = "OK" if p99 <= args.budget_ms else "OVER BUDGET"
            label = f"h={horizon} k_cards={k_cards} k_cells={k_cells}"
            print(f"{label:<26}{real.size:>5}{np.mean(nc[nc>=2]):>5.1f}"
                  f"{percentile(real,50):>9.1f}{percentile(real,95):>9.1f}"
                  f"{p99:>9.1f}{real.max():>9.1f}   {ok}")
            rows.append((horizon, k_cards, k_cells, p99, percentile(real, 50)))

    print()
    safe = [r for r in rows if r[3] <= args.budget_ms]
    if safe:
        best = max(safe, key=lambda r: (r[0], r[1] * r[2]))
        print(f"  DEEPEST CONFIG INSIDE BUDGET: horizon={best[0]} "
              f"k_cards={best[1]} k_cells={best[2]}  (p99 {best[3]:.1f} ms)")
    else:
        print("  NOTHING fits the budget -- ship greedy only.")
    print("\n  Tail, not mean: the budget is a promise about the worst case.")
    print("  n below is MULTI-CANDIDATE steps only; K=1 steps skip rollout entirely.")


if __name__ == "__main__":
    main()
