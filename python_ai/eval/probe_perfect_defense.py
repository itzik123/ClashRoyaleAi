"""Is the agent learning perfect defense, or just winning?

W_FLAWLESS_DEFENSE pays only on a win, scaled by the fraction of our own tower
HP still standing. Metrics:

  win rate                 the baseline the others need
  tower HP left | WIN      the quantity the bonus pays for; the headline
  flawless win rate        won with every tower untouched
  crowns conceded | WIN    towers lost in games still won

  Cannon centroid / modal / distinct cells
                           2.6 defends by pulling: a central Cannon drags a Hog off the tower lane into both Princess towers' crossfire, a back-corner one pulls nothing. Read modal share with distinct cells: a sharp head that moves its mode is healthy, one that returns one cell is not.

  per-card play share      watch for the win condition going unplayed

Greedy throughout, so the numbers describe the deployed policy.
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

# Run as a script, the repo root is not on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

HERE = python_ai.PACKAGE_DIR

import clash_royale_env as CE  # noqa: E402
from python_ai.envs.gym_wrapper import DEFAULT_DECK  # noqa: E402
from python_ai.models.net import MicroRoyaleNet  # noqa: E402
from python_ai.models.policy_io import LSTM_HIDDEN  # noqa: E402
from python_ai.engine_constants import BOARD_CENTER_X  # noqa: E402

# Half-width of the reported central pull pocket around BOARD_CENTER_X.
PULL_POCKET_HALF_WIDTH = 3.0

E = CE.ClashRoyaleEnv
NOOP = E.HAND_SIZE
CANNON = 25
HOG = 15


def own_tower_hp_fraction(env):
    """Fraction of our three towers' starting HP still standing.

    Reads the observation's tower scalars (tail + 3..5), hp / MAX_BUILDING_HP,
    the same quantity train._own_tower_hp_total() sums at reset.
    """
    obs = np.asarray(env.get_observation_for_team(0), np.float32)
    tail = E.EXTRA_SCALARS_START
    return float(obs[tail + 3:tail + 6].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(HERE, "model_weights.pth"))
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--opp-elixir", type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device("cpu")
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    eps_done = ckpt.get("episodes_completed", "?") if isinstance(ckpt, dict) else "?"
    net = MicroRoyaleNet().to(device)
    net.load_state_dict(state, strict=False)
    net.eval()

    deck = list(DEFAULT_DECK)
    env = E(deck, deck, 3600)
    env.set_opponent_elixir_multiplier(args.opp_elixir)

    full_hp = None
    wins = losses = draws = 0
    hp_on_win, crowns_on_win = [], []
    placements = defaultdict(list)
    plays = Counter()
    total_plays = 0

    with torch.no_grad():
        for _ in range(args.episodes):
            obs_l = env.reset()
            if full_hp is None:
                full_hp = own_tower_hp_fraction(env)
            hx = torch.zeros(1, LSTM_HIDDEN)
            cx = torch.zeros(1, LSTM_HIDDEN)
            reward = 0.0
            while True:
                obs = torch.from_numpy(np.asarray(obs_l, np.float32)).view(1, -1)
                feats, embeds, spatial, hires = net.extract_features_hires(obs)
                mask = net.affordability_mask(obs)
                cl, _, _, _, (hx2, cx2) = net.step_lstm_and_card(feats, (hx, cx), mask)
                card = int(cl.argmax(1))
                ids = net.hand_card_ids(obs)[0].tolist()
                pl = net.placement_given_card(hx2, embeds, torch.tensor([card]),
                                              obs, spatial, hires_map=hires)
                cell = int(pl.argmax(1))
                x, y = cell % E.BOARD_WIDTH, cell // E.BOARD_WIDTH

                if card < NOOP and 0 <= card < len(ids) and ids[card] >= 0:
                    plays[ids[card]] += 1
                    total_plays += 1
                    placements[ids[card]].append((x, y))

                hp_before = own_tower_hp_fraction(env)
                res = env.step(card, float(x), float(y), 10)
                obs_l = res.observation
                hx, cx = hx2, cx2
                if res.done:
                    reward = float(res.reward)
                    # The engine may already be terminal after done; use the
                    # last live reading.
                    final_hp = hp_before
                    break

            if reward > 0.5:
                wins += 1
                hp_on_win.append(final_hp / full_hp)
                crowns_on_win.append(3 - env.get_towers_alive(0))
            elif reward < -0.5:
                losses += 1
            else:
                draws += 1

    n = args.episodes
    print("=" * 78)
    print(f"PERFECT DEFENSE  --  {os.path.basename(args.weights)} "
          f"(episodes_completed={eps_done}), opp elixir {args.opp_elixir}x, n={n}")
    print("=" * 78)
    print(f"  win / loss / draw          {wins/n:.3f} / {losses/n:.3f} / {draws/n:.3f}")
    if hp_on_win:
        hp = np.array(hp_on_win)
        print(f"  tower HP left | WIN        {hp.mean():.3f}   "
              f"(the quantity W_FLAWLESS_DEFENSE pays for)")
        print(f"  flawless wins              {(hp >= 0.999).mean():.3f}   "
              f"({int((hp >= 0.999).sum())}/{wins} won with every tower untouched)")
        print(f"  crowns conceded | WIN      {np.mean(crowns_on_win):.3f}")
    else:
        print("  no wins yet -- defense metrics undefined")

    print(f"\n  card usage ({total_plays} plays)")
    for cid in deck:
        name = CE.get_card_info(cid)["name"]
        share = plays[cid] / total_plays if total_plays else 0.0
        flag = "   <-- WIN CONDITION" if cid == HOG else ""
        print(f"    {name:<12} {plays[cid]:5d}  {share:6.1%}{flag}")

    for cid, label in ((CANNON, "CANNON (the pull)"), (HOG, "HOG RIDER")):
        cells = placements.get(cid, [])
        if not cells:
            print(f"\n  {label}: never played")
            continue
        xs = np.array([c[0] for c in cells]); ys = np.array([c[1] for c in cells])
        modal, cnt = Counter(cells).most_common(1)[0]
        print(f"\n  {label}: n={len(cells)}  centroid=({xs.mean():.1f}, {ys.mean():.1f})  "
              f"modal={modal} {cnt/len(cells):.1%}  distinct={len(set(cells))}")
        if cid == CANNON:
            # A pull wants the board centre, ahead of the Princess towers (y ~
            # 6-11), not a back corner.
            half = PULL_POCKET_HALF_WIDTH
            central = np.mean((np.abs(xs - BOARD_CENTER_X) <= half)
                              & (ys >= 5) & (ys <= 12))
            print(f"    in the central pull pocket "
                  f"(|x-{BOARD_CENTER_X}|<={half}, 5<=y<=12): {central:.1%}")


if __name__ == "__main__":
    main()
