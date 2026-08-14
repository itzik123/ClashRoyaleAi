"""Did the solvency term stop the bot bankrupting itself before it defends?

THE STATISTIC UNDER TEST
------------------------
Measured on `model_weights_dist_e3.pth` before any change (40 greedy episodes,
1.5x opponent elixir):

    below 3 elixir, all decisions                65.3%
    below 3 elixir, during a BIG push (>8)       60.8%
    mean elixir                                   2.53
    plays/episode                                 29.1
    elixir spent/episode  (income is ~98)        ~105

3.0 is the cost of the cheapest card in DEFAULT_DECK, so below it the action
space is empty and P(play) is 0.0% by arithmetic rather than by choice. That is
the number this term exists to move.

STATISTICAL UNIT IS THE EPISODE, NOT THE DECISION. A net produces ~280
decisions per episode and they are heavily correlated within a match (elixir is
an integral of that episode's own spending), so pooling decisions would give an
absurdly tight interval around a number whose real variance is between-episode.
CIs here bootstrap over episodes.

Unlike prove_placement.py this comparison CANNOT be paired: each net steers its
own trajectory, which is the entire point -- we are asking whether it manages
its elixir differently, so it must be allowed to. That costs power, which is why
the primary endpoint is a per-episode rate (low variance) rather than win rate
(needs ~1,568 episodes per arm to resolve 5 points).

    python_ai/venv/Scripts/python.exe python_ai/prove_solvency.py \
        --nets seed=A.pth coverage=B.pth solvency=D.pth --episodes 30
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clash_royale_env as E  # noqa: E402
import gym_wrapper  # noqa: E402
import tactics  # noqa: E402
from expert_iteration import load_net  # noqa: E402
from search_ab_test import outcome_score  # noqa: E402

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
        r = env.step(gi, float(cell % 18), float(cell // 18), 10)
        obs, reward = r.observation, float(r.reward)
        if r.done:
            break

    return dict(
        elixir=np.array(elix), threat=np.array(threats), played=np.array(played),
        spent=env.get_elixir_spent(0) - spent0,
        tower_hp_lost=env.get_tower_damage_dealt(1),
        score=outcome_score(reward), steps=len(elix))


def _threat_elixir(obs):
    """Enemy HP on our half, converted to the elixir scale of the original
    diagnosis so the ">8 elixir" bucket keeps its meaning.

    That diagnosis summed the elixir COST of enemy troops past the river, which
    needs the per-tick entity list. Here only the observation is available, so
    HP is divided by DECK_HP_PER_ELIXIR. The conversion is a constant and is
    applied identically to every arm, so it cannot favour one of them -- it only
    has to keep the bucket boundary comparable to the pre-change number.
    """
    return float(tactics.threat_map(obs).sum()) / DECK_HP_PER_ELIXIR


# Mean HP per elixir over DEFAULT_DECK's troops, from CardRegistry.h
# (Valkyrie 1907/4, Archers 2x304/3, Minions 3x230/3, Cannon 824/3,
#  Giant 3968/5, Musketeer 721/4, Mini PEKKA 1390/4):
#   (1907+608+690+824+3968+721+1390) / (4+3+3+3+5+4+4) = 10108/26
# Hardcoded with this derivation because pybind's get_card_info exposes no
# hitpoints field -- the fallback CLAUDE.md prescribes when a value is not
# derivable from the bindings.
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
    x = np.asarray(x, dtype=float)
    b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def boot_diff(a, b, rng, n=10000):
    """Unpaired bootstrap of mean(b) - mean(a), resampling episodes."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.array([b[rng.integers(0, len(b), len(b))].mean()
                  - a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)])
    lo, hi = np.percentile(d, [2.5, 97.5])
    # two-sided bootstrap p: how often the resampled difference crosses 0
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(b.mean() - a.mean()), float(lo), float(hi), float(min(1.0, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", nargs="+", required=True, help="label=path ...")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--opp-elixir", type=float, default=1.5)
    ap.add_argument("--baseline", default=None,
                    help="label to compare the others against (default: first)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
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
