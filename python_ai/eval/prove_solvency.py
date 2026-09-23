"""Did the solvency term stop the bot bankrupting itself before it defends?

The endpoint is the share of decisions below 3 elixir, the cheapest card in the
deck: below it the action space is empty.

The statistical unit is the episode: decisions within a match are heavily
correlated (elixir integrates the episode's own spending), so CIs bootstrap
over episodes. The arms cannot be paired, since each net must steer its own
trajectory; a per-episode rate is used as the primary endpoint because win rate
needs far more episodes.

    python_ai/venv/Scripts/python.exe python_ai/eval/prove_solvency.py \
        --nets seed=A.pth coverage=B.pth solvency=D.pth --episodes 30
"""
import argparse
import os
import sys

import numpy as np
import torch

# Run as a script, the repo root is not on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env as E  # noqa: E402
from python_ai.envs import gym_wrapper  # noqa: E402
from python_ai.advisors import tactics  # noqa: E402
from python_ai.eval import stats  # noqa: E402
from python_ai.models.policy_io import load_net  # noqa: E402
from python_ai.search.search import outcome_score  # noqa: E402
from python_ai.engine_constants import BOARD_W  # noqa: E402

CE = E.ClashRoyaleEnv
CHEAPEST = 3.0
BIG_THREAT = 8.0


@torch.no_grad()
def run_episode(net, deck, opp_elixir, max_steps=400):
    env = CE(deck, deck, 3600)
    env.set_opponent_elixir_multiplier(opp_elixir)
    env.reset()
    hx, cx = torch.zeros(1, 256), torch.zeros(1, 256)
    obs = env.get_observation_for_team(0)

    elix, threats, played, reward = [], [], [], 0.0
    spent0 = env.get_elixir_spent(0)
    for _t in range(max_steps):
        t = torch.tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        mask = net.affordability_mask(t)
        feats, embeds, sp = net.extract_features(t)
        logits, _, _, _, (hx2, cx2) = net.step_lstm_and_card(feats, (hx, cx), mask)
        gi = int(logits.argmax(-1).item())
        place = net.placement_given_card(hx2, embeds, torch.tensor([gi]), t, sp)
        cell = int(place.argmax(-1).item())

        elix.append(tactics.own_elixir(obs))
        threats.append(_threat_elixir(obs))
        played.append(gi != net.hand_size)

        hx, cx = hx2, cx2
        r = env.step(gi, float(cell % BOARD_W), float(cell // BOARD_W), 10)
        obs, reward = r.observation, float(r.reward)
        if r.done:
            break

    return dict(
        elixir=np.array(elix), threat=np.array(threats), played=np.array(played),
        spent=env.get_elixir_spent(0) - spent0,
        tower_hp_lost=env.get_tower_damage_dealt(1),
        score=outcome_score(reward), steps=len(elix))


def _threat_elixir(obs):
    """Enemy HP on our half, converted to elixir so the ">8 elixir" bucket stays
    comparable.

    The conversion is one constant applied to every arm, so it cannot favour
    any of them.
    """
    return float(tactics.threat_map(obs).sum()) / DECK_HP_PER_ELIXIR


# Mean HP per elixir over the deck's troops, from CardRegistry.h:
#   (1907+608+690+824+3968+721+1390) / (4+3+3+3+5+4+4) = 10108/26
# Hardcoded because get_card_info exposes no hitpoints.
DECK_HP_PER_ELIXIR = 10108.0 / 26.0


def summarize(name, eps):
    def per_ep(fn):
        return np.array([fn(e) for e in eps], dtype=float)

    bank = per_ep(lambda e: float((e["elixir"] < CHEAPEST).mean()))
    big = per_ep(lambda e: float(((e["elixir"] < CHEAPEST) & (e["threat"] > BIG_THREAT)).sum()
                                 / max(1, (e["threat"] > BIG_THREAT).sum())))
    meane = per_ep(lambda e: float(e["elixir"].mean()))
    plays = per_ep(lambda e: float(e["played"].sum()))
    spent = per_ep(lambda e: float(e["spent"]))
    lost = per_ep(lambda e: float(e["tower_hp_lost"]))
    score = per_ep(lambda e: e["score"])
    return dict(name=name, bankrupt=bank, big_bankrupt=big, mean_elixir=meane,
                plays=plays, spent=spent, tower_lost=lost, score=score)


def boot_ci(x, rng, n=10000):
    """(mean, lo, hi); see eval/stats.py."""
    return stats.bootstrap_ci(x, rng=rng, n=n)


def boot_diff(a, b, rng, n=10000):
    """(delta, lo, hi, p) for unpaired arms; see eval/stats.py."""
    return stats.unpaired_bootstrap_diff(a, b, rng=rng, n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", nargs="+", required=True, help="label=path ...")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--opp-elixir", type=float, default=1.5)
    ap.add_argument("--baseline", default=None,
                    help="label to compare the others against (default: first)")
    args = ap.parse_args()

    here = python_ai.PACKAGE_DIR
    dev = torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    deck = list(gym_wrapper.DEFAULT_DECK)
    rng = np.random.default_rng(0)

    results = {}
    for spec in args.nets:
        label, path = spec.split("=", 1)
        net = load_net(path if os.path.isabs(path) else os.path.join(here, path), dev)
        eps = [run_episode(net, deck, args.opp_elixir) for _ in range(args.episodes)]
        results[label] = summarize(label, eps)
        print(f"  {label}: {args.episodes} episodes done", flush=True)

    base = args.baseline or list(results)[0]
    print("\n" + "=" * 78)
    print(f"ELIXIR MANAGEMENT  ({args.episodes} episodes/net, "
          f"opponent {args.opp_elixir}x, CIs bootstrap over EPISODES)")
    print("=" * 78)
    rows = [("bankrupt <3 elixir", "bankrupt", "{:.1%}"),
            ("...during BIG push", "big_bankrupt", "{:.1%}"),
            ("mean elixir", "mean_elixir", "{:.2f}"),
            ("plays / episode", "plays", "{:.1f}"),
            ("elixir spent / ep", "spent", "{:.0f}"),
            ("tower HP lost / ep", "tower_lost", "{:.0f}"),
            ("win rate", "score", "{:.3f}")]
    hdr = f"{'metric':<22}" + "".join(f"{k:>17}" for k in results)
    print(hdr)
    print("-" * len(hdr))
    for label, key, fmt in rows:
        line = f"{label:<22}"
        for k in results:
            m, lo, hi = boot_ci(results[k][key], rng)
            line += f"{fmt.format(m):>17}"
        print(line)

    for k in results:
        if k == base:
            continue
        print(f"\n--- {k} vs {base} ---")
        for label, key, fmt in rows:
            d, lo, hi, p = boot_diff(results[base][key], results[k][key], rng)
            star = "  *" if p < 0.05 else ""
            print(f"  {label:<22} {d:+9.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
                  f"  p={p:.3g}{star}")


if __name__ == "__main__":
    main()
