"""Is the policy's HOG placement worse than chance, or is it a valuation?

The 2.6 run at ep 6,053 plays Hog Rider on 0.8% of decisions. That is the same
shape as the Giant/Cannon/Fireball collapse this project already cured once --
but low usage has TWICE turned out to mean different things here, and the two
need different responses:

  * BROKEN FUNCTION -- the placement head returns a fixed/bad cell, the card is
    genuinely worthless, and the card head is right to drop it. This is what
    PLACEMENT_COLLAPSE.md found: the policy's Cannon cell preserved 121 HP
    against a random legal cell's 396, i.e. significantly WORSE THAN CHANCE. A
    policy cannot be correctly valuing a card it places worse than random.

  * CORRECT VALUATION -- the card really is a bad play in this engine and the
    policy is right. This project has been burned assuming otherwise: forcing
    Fireball usage dropped win rate 97% -> 23%.

The discriminator is the same one that settled it before: score the policy's
OWN chosen cell against a RANDOM LEGAL cell, paired on identical states, with
the ENGINE doing the scoring. Better than random => the function works and the
low usage is a valuation. Worse than random => broken, and fixing the head is
the lever.

Metric is ENEMY TOWER DAMAGE over HORIZON ticks, because that is what a win
condition is for -- not value-killed, which scores a Hog's whole purpose at
zero (it ignores troops by design).
"""
import argparse
import os
import random
import sys

import numpy as np

from python_ai.eval import stats
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
from python_ai.models.net import MicroRoyaleNet  # noqa: E402
from python_ai.models.policy_io import LSTM_HIDDEN  # noqa: E402

E = CE.ClashRoyaleEnv
NOOP = E.HAND_SIZE
HOG = 15
HORIZON = 600          # ticks = 60 s: long enough to cross AND hit a tower
                       # (a lone Hog from the bridge measured 317 damage in 40 s)


def hog_damage(env, x, y, baseline):
    """Enemy tower damage caused by adding a Hog at (x,y), over its lifetime.

    Injection costs no elixir, so the rest of the match is untouched -- the same
    protocol prove_placement.py / prove_giant.py use. `baseline` is the same
    state run forward with NO Hog, so this isolates the Hog's contribution from
    whatever was already going to happen.
    """
    s = env.snapshot()
    before = s.get_tower_damage_dealt(0)
    s.inject(HOG, float(x), float(y), 0)
    for _ in range(HORIZON // 30):
        s.step(NOOP, 0.0, 0.0, 30)
        if s.is_game_over():
            break
    return (s.get_tower_damage_dealt(0) - before) - baseline


def hog_baseline(env):
    s = env.snapshot()
    before = s.get_tower_damage_dealt(0)
    for _ in range(HORIZON // 30):
        s.step(NOOP, 0.0, 0.0, 30)
        if s.is_game_over():
            break
    return s.get_tower_damage_dealt(0) - before


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(HERE, "model_weights.pth"))
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--opp-elixir", type=float, default=1.4)
    args = ap.parse_args()

    random.seed(0)
    np.random.seed(0)
    device = torch.device("cpu")
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    eps = ckpt.get("episodes_completed", "?") if isinstance(ckpt, dict) else "?"
    net = MicroRoyaleNet().to(device)
    net.load_state_dict(state, strict=False)
    net.eval()

    legal = net._placement_legal[HOG].nonzero().flatten().tolist()
    deck = list(DEFAULT_DECK)
    env = E(deck, deck, 3600)
    env.set_opponent_elixir_multiplier(args.opp_elixir)

    pol, rnd = [], []
    cells = []
    with torch.no_grad():
        for ep in range(args.episodes):
            obs_l = env.reset()
            hx = torch.zeros(1, LSTM_HIDDEN); cx = torch.zeros(1, LSTM_HIDDEN)
            while True:
                obs = torch.from_numpy(np.asarray(obs_l, np.float32)).view(1, -1)
                feats, embeds, spatial, hires = net.extract_features_hires(obs)
                mask = net.affordability_mask(obs)
                cl, _, _, _, (hx2, cx2) = net.step_lstm_and_card(feats, (hx, cx), mask)
                ids = net.hand_card_ids(obs)[0].tolist()

                # Score the Hog wherever it is IN HAND -- not only where the
                # policy chose to play it. At 0.8% usage, conditioning on the
                # play would give almost no states and would also select
                # exactly the atypical ones.
                if HOG in ids:
                    slot = ids.index(HOG)
                    pl = net.placement_given_card(hx2, embeds, torch.tensor([slot]),
                                                  obs, spatial, hires_map=hires)
                    cell = int(pl.argmax(1))
                    px, py = cell % E.BOARD_WIDTH, cell // E.BOARD_WIDTH
                    r = random.choice(legal)
                    rx, ry = r % E.BOARD_WIDTH, r // E.BOARD_WIDTH
                    base = hog_baseline(env)
                    pol.append(hog_damage(env, px, py, base))
                    rnd.append(hog_damage(env, rx, ry, base))
                    cells.append((px, py))

                card = int(cl.argmax(1))
                pl = net.placement_given_card(hx2, embeds, torch.tensor([card]),
                                              obs, spatial, hires_map=hires)
                c = int(pl.argmax(1))
                res = env.step(card, float(c % E.BOARD_WIDTH), float(c // E.BOARD_WIDTH), 10)
                obs_l = res.observation
                hx, cx = hx2, cx2
                if res.done:
                    break
            print(f"  ep{ep}: {len(pol)} hog states", flush=True)

    p = np.array(pol, float); r = np.array(rnd, float)
    d = p - r
    # THE SHARED bootstrap, not a sixth hand-rolled copy. It is seeded by
    # default, so re-running this measurement reproduces its own CI -- and this
    # script's verdict below BRANCHES on the CI bounds, so an unseeded resample
    # could flip "BETTER than chance" to "indistinguishable" between two runs
    # of identical data.
    _, ci_lo, ci_hi = stats.bootstrap_ci(d)
    better = int((d > 1e-9).sum()); worse = int((d < -1e-9).sum())
    from collections import Counter
    modal, cnt = Counter(cells).most_common(1)[0]

    print("\n" + "=" * 78)
    print(f"HOG PLACEMENT, ENGINE-SCORED  --  {os.path.basename(args.weights)} "
          f"(ep={eps}), opp {args.opp_elixir}x")
    print("=" * 78)
    print(f"  n = {len(p)} paired states   (enemy tower damage over {HORIZON} ticks)")
    print(f"    policy cell        {p.mean():8.1f}")
    print(f"    random legal cell  {r.mean():8.1f}")
    print(f"    delta              {d.mean():+8.1f}   95% CI "
          f"[{ci_lo:+.1f}, {ci_hi:+.1f}]")
    print(f"    {better} better / {worse} worse / {len(p)-better-worse} tied")
    print(f"  placement: modal={modal} {cnt/len(cells):.1%}  distinct={len(set(cells))}")
    print()
    if d.mean() > 0 and ci_lo > 0:
        print("  => BETTER than chance. The placement FUNCTION works; the low")
        print("     usage is a VALUATION by the card head, not a broken head.")
    elif ci_hi < 0:
        print("  => WORSE than chance. A policy cannot be correctly valuing a card")
        print("     it places worse than random. The head is the lever.")
    else:
        print("  => indistinguishable from chance at this n.")


if __name__ == "__main__":
    main()
