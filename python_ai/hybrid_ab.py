"""Paired win-rate A/B: pure neural policy vs the commander/tactical-officer hybrid.

PRE-REGISTERED: n is fixed on the command line before the run and the result is
reported whatever it says. The gate A/B at n=130 produced a 95% CI of width
0.243 on win rate, so n=250 is chosen to bring that to roughly 0.17 -- enough to
resolve a ~10-point effect and NOT enough for a 3-point one, which is stated up
front rather than discovered afterwards. No extending after seeing the numbers:
this project has already had one +0.105 at p=0.044 evaporate at 4x the power.

Both arms are handed a bit-identical opening by env.snapshot(), so every
difference is the policy and not the deal.

    python_ai/venv/Scripts/python.exe python_ai/hybrid_ab.py --n 250
"""
import argparse
import os
import sys
from math import comb

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clash_royale_env  # noqa: E402
import tactics  # noqa: E402
from gym_wrapper import DEFAULT_DECK  # noqa: E402
from expert_iteration import load_net  # noqa: E402
from hybrid_policy import HybridPolicy  # noqa: E402
from search_ab_test import outcome_score  # noqa: E402

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
                init_fireball=policy.stats["initiated_fireball"],
                init_giant=policy.stats["initiated_giant"])


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
    for label in ("init_cannon", "init_fireball", "init_giant"):
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
                    help="run the PER-CARD override ablation. The override is "
                         "justified only for as long as the advisor beats the "
                         "head, and that is now a per-card question: as of "
                         "2026-08-14 the head matches the advisor on Fireball "
                         "(p = 0.163) while still losing badly on Cannon "
                         "(p = 6.8e-22). Turning it off wholesale would give "
                         "back the Cannon's contribution to the measured +11.8 "
                         "win-rate points; leaving it on wholesale keeps a "
                         "layer that is doing nothing for Fireball.")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    net = load_net(os.path.join(here, args.net), device)

    def mk(**kw):
        base = dict(use_gate=False, use_advisor=False, initiate=False,
                    initiate_giant=False)
        base.update(kw)
        return HybridPolicy(net, device, **base)

    if args.ablate:
        # One component at a time, so a regression can be attributed. The full
        # hybrid lost 57% of its tower damage DEALT in smoke tests and the
        # opponent-aware reserve did not recover it, so the cause is elsewhere.
        arms = {
            "neural": mk(),
            "gate": mk(use_gate=True),
            "place": mk(use_advisor=True),
            "initiate": mk(use_advisor=True, initiate=True),
            "full": mk(use_gate=True, use_advisor=True, initiate=True,
                       initiate_giant=True),
        }
    elif args.per_card:
        from hybrid_policy import CANNON, FIREBALL, GIANT
        # The solvency gate stays ON in every arm: it is a separate, measured
        # component (bankruptcy 72.7% -> 41.7%) and leaving it to vary would
        # confound the placement question this ablation exists to answer.
        arms = {
            "neural": mk(use_gate=True),
            "cannon_only": mk(use_gate=True, use_advisor=True,
                              override_cards=(CANNON,)),
            "cannon_giant": mk(use_gate=True, use_advisor=True,
                               override_cards=(CANNON, GIANT)),
            "all_three": mk(use_gate=True, use_advisor=True,
                            override_cards=(CANNON, FIREBALL, GIANT)),
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
