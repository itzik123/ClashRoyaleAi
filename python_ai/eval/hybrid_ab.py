"""Paired win-rate A/B: pure neural policy vs the commander/tactical-officer
hybrid.

Pre-registered: n is fixed on the command line and the result is reported
whatever it says. n=250 gives a 95% CI of roughly 0.17, enough for a ~10-point
effect and not a 3-point one. Both arms get a bit-identical opening via
env.snapshot().

    python_ai/venv/Scripts/python.exe python_ai/eval/hybrid_ab.py --n 250
"""
import argparse
import os
import sys
from math import comb

import numpy as np
import torch

# Run as a script, the repo root is not on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env  # noqa: E402
from python_ai.advisors import tactics  # noqa: E402
from python_ai.envs.gym_wrapper import DEFAULT_DECK  # noqa: E402
from python_ai.models.policy_io import load_net  # noqa: E402
from python_ai.advisors.hybrid_policy import HybridPolicy  # noqa: E402
from python_ai.search.search import outcome_score  # noqa: E402

CE = clash_royale_env.ClashRoyaleEnv
SKIP = 10
CHEAPEST = 3.0


@torch.no_grad()
def play(policy, env, device):
    policy.reset()
    obs = env.get_observation_for_team(0)
    reward, steps, done = 0.0, 0, False
    elix = []
    while not done and steps < 400:
        elix.append(tactics.own_elixir(np.asarray(obs, dtype=np.float32)))
        slot, x, y = policy.act(obs)
        r = env.step(slot, x, y, SKIP)
        obs, reward, done = r.observation, float(r.reward), r.done
        steps += 1
    e = np.array(elix)
    return dict(score=outcome_score(reward),
                bankrupt=float((e < CHEAPEST).mean()),
                mean_elixir=float(e.mean()),
                tower_lost=float(env.get_tower_damage_dealt(1)),
                tower_dealt=float(env.get_tower_damage_dealt(0)),
                spent=float(env.get_elixir_spent(0)),
                plays=policy.stats["plays"],
                init_cannon=policy.stats["initiated_cannon"],
                init_fireball=policy.stats["initiated_fireball"])


def report(rows, base_key, other_key, rng):
    keys = [("win rate", "score", "{:.3f}"),
            ("bankrupt <3 elixir", "bankrupt", "{:.1%}"),
            ("mean elixir", "mean_elixir", "{:.2f}"),
            ("elixir spent / ep", "spent", "{:.0f}"),
            ("plays / ep", "plays", "{:.1f}"),
            ("tower HP DEALT / ep", "tower_dealt", "{:.0f}"),
            ("tower HP lost / ep", "tower_lost", "{:.0f}")]
    print(f"\n{'metric':<22}{base_key:>14}{other_key:>14}")
    print("-" * 50)
    for label, k, fmt in keys:
        a = np.array([r[k] for r in rows[base_key]], dtype=float)
        b = np.array([r[k] for r in rows[other_key]], dtype=float)
        print(f"{label:<22}{fmt.format(a.mean()):>14}{fmt.format(b.mean()):>14}")
    for label in ("init_cannon", "init_fireball"):
        v = np.mean([r[label] for r in rows[other_key]])
        print(f"{label:<22}{'--':>14}{v:>14.2f}")

    print(f"\n--- {other_key} vs {base_key} (paired) ---")
    for label, k, _ in keys:
        a = np.array([r[k] for r in rows[base_key]], dtype=float)
        b = np.array([r[k] for r in rows[other_key]], dtype=float)
        d = b - a
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        better, worse = int((d > 0).sum()), int((d < 0).sum())
        m = better + worse
        p = (sum(comb(m, i) for i in range(min(better, worse) + 1)) * 2 / (2 ** m)
             if m else 1.0)
        star = "  *" if p < 0.05 else ""
        print(f"  {label:<22}{d.mean():+10.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"  {better}+/{worse}-  p={min(1.0, p):.3g}{star}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--net", default="model_weights_selfplay.pth")
    ap.add_argument("--opp-elixir", type=float, default=1.5)
    ap.add_argument("--ablate", action="store_true",
                    help="localise which component moves which metric")
    ap.add_argument("--per-card", action="store_true",
                    help="run the per-card override ablation: the override is "
                         "worth keeping only for cards where the advisor still "
                         "beats the placement head")
    args = ap.parse_args()

    here = python_ai.PACKAGE_DIR
    device = torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    net = load_net(os.path.join(here, args.net), device)

    def mk(**kw):
        base = dict(use_gate=False, use_advisor=False, initiate=False)
        base.update(kw)
        return HybridPolicy(net, device, **base)

    if args.ablate:
        # One component at a time, so a regression can be attributed.
        arms = {
            "neural": mk(),
            "gate": mk(use_gate=True),
            "place": mk(use_advisor=True),
            "initiate": mk(use_advisor=True, initiate=True),
            "full": mk(use_gate=True, use_advisor=True, initiate=True),
        }
    elif args.per_card:
        from python_ai.advisors.hybrid_policy import CANNON, FIREBALL
        # The solvency gate stays on in every arm, so it cannot confound the
        # placement question.
        arms = {
            "neural": mk(use_gate=True),
            "cannon_only": mk(use_gate=True, use_advisor=True,
                              override_cards=(CANNON,)),
            "fireball_only": mk(use_gate=True, use_advisor=True,
                                override_cards=(FIREBALL,)),
            "both": mk(use_gate=True, use_advisor=True,
                       override_cards=(CANNON, FIREBALL)),
        }
    else:
        arms = {"neural": mk(), "hybrid": HybridPolicy(net, device)}
    rows = {k: [] for k in arms}

    for i in range(args.n):
        root = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), 3600)
        root.set_opponent_elixir_multiplier(args.opp_elixir)
        root.reset()
        base = root.snapshot()
        for k, pol in arms.items():
            rows[k].append(play(pol, base.snapshot(), device))
        if (i + 1) % 25 == 0:
            line = "  ".join(f"{k} {np.mean([r['score'] for r in rows[k]]):.3f}"
                             for k in arms)
            print(f"  pair {i+1:>4}: {line}", flush=True)

    print(f"\n{args.n} paired openings, {args.net}, opponent {args.opp_elixir}x")
    rng = np.random.default_rng(0)
    for k in arms:
        if k != "neural":
            report(rows, "neural", k, rng)


if __name__ == "__main__":
    main()
