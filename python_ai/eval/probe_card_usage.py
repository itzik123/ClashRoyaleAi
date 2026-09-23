"""Per-card usage probe for phase 1, which logs no per-card diagnostic of its own.

Loads a checkpoint off disk and runs its own environments, so it is safe to run
beside a live trainer.

    python_ai/venv/Scripts/python.exe python_ai/eval/probe_card_usage.py --episodes 40 --greedy

Reports per card how often it was played and how often it was affordable. Read
them together: rarely played but often affordable is a policy choice; rarely
affordable is the cost curve.
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch
from torch.distributions import Categorical

# Run as a script, the repo root is not on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env as E  # noqa: E402
from python_ai.envs import gym_wrapper
from python_ai.models.net import MicroRoyaleNet
from python_ai.models.policy_io import load_state_dict_flexible

CE = E.ClashRoyaleEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="model_weights.pth")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--greedy", action="store_true",
                    help="argmax instead of sampling -- measures the policy's "
                         "committed preference rather than its exploration")
    args = ap.parse_args()

    here = python_ai.PACKAGE_DIR
    weights = args.weights if os.path.isabs(args.weights) else os.path.join(here, args.weights)

    deck = list(gym_wrapper.DEFAULT_DECK)
    names = {c: E.get_card_info(c)["name"] for c in deck}
    costs = {c: E.get_card_info(c)["cost"] for c in deck}

    net = MicroRoyaleNet(num_ability_slots=gym_wrapper.DEFAULT_DECK_ABILITY_SLOTS)
    if os.path.exists(weights):
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        # Flexible load: a checkpoint predating a newer branch is not a
        # mismatch.
        load_state_dict_flexible(net, sd, os.path.basename(weights))
        ep = ck.get("episodes_completed", "?") if isinstance(ck, dict) else "?"
        print(f"loaded {os.path.basename(weights)} (episode {ep})")
    else:
        print(f"!! {weights} not found -- probing an UNTRAINED net, so the numbers")
        print("   below are the mask's shape, not the policy's preference.")
        ep = 0
    net.eval()

    played = Counter()
    affordable = Counter()
    steps = 0
    noops = 0
    env = gym_wrapper.MicroRoyaleEnv()

    for _ in range(args.episodes):
        obs, _ = env.reset()
        hx = torch.zeros(1, 256)
        cx = torch.zeros(1, 256)
        for _t in range(400):
            t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mask = net.affordability_mask(t)
                feats, embeds, spatial = net.extract_features(t)
                logits, _, _, _, (hx, cx) = net.step_lstm_and_card(feats, (hx, cx), mask)
                idx = logits.argmax(-1) if args.greedy else Categorical(logits=logits).sample()
                place = net.placement_given_card(hx, embeds, idx, t, spatial)
                cell = place.argmax(-1) if args.greedy else Categorical(logits=place).sample()
            x, y = net.cell_to_xy(cell)

            # Read the hand from the observation the decision was made on; the
            # hand rotates as soon as a card is played.
            hand_ids = net.hand_card_ids(t)[0].tolist()
            for slot, cid in enumerate(hand_ids):
                if cid >= 0 and bool(mask[0, slot]):
                    affordable[cid] += 1

            slot = int(idx.item())
            if slot < net.hand_size and hand_ids[slot] >= 0:
                played[hand_ids[slot]] += 1
            else:
                noops += 1
            steps += 1

            obs, _r, term, trunc, _info = env.step({
                "card_index": np.array([slot]),
                "target_x": x.numpy().reshape(1, 1),
                "target_y": y.numpy().reshape(1, 1),
                "activate_ability_slot1": np.zeros(1, dtype=np.int64),
                "activate_ability_slot2": np.zeros(1, dtype=np.int64),
            })
            if term or trunc:
                break

    total_plays = sum(played.values())
    print(f"\n{args.episodes} episodes, {steps} decision steps, "
          f"{total_plays} card plays, {noops} no-ops "
          f"({noops / max(1, steps):.0%})   mode={'greedy' if args.greedy else 'sampled'}\n")
    # Affordability alone mislabels the deck (every card sits at a few
    # percent). Take-up is play share divided by affordability share: low
    # take-up with healthy affordability is a policy choice, low affordability
    # is the cost curve.
    print(f"{'card':<14}{'cost':>5}{'% of plays':>12}{'% steps aff.':>14}{'take-up':>9}")
    print("-" * 54)
    for c in sorted(deck, key=lambda k: -played[k]):
        share = played[c] / max(1, total_plays)
        aff = affordable[c] / max(1, steps)
        takeup = share / aff if aff > 0 else float("nan")
        median_aff = float(np.median([affordable[k] / max(1, steps) for k in deck]))
        flag = ""
        if takeup < 0.5 and aff >= 0.5 * median_aff:
            flag = "  <- AVOIDED (legal often enough; policy declines it)"
        elif aff < 0.5 * median_aff:
            flag = "  <- STARVED (rarely legal: cost curve)"
        print(f"{names[c]:<14}{costs[c]:>5.0f}{share:>11.1%}{aff:>13.1%}{takeup:>9.2f}{flag}")
    print("\ntake-up 1.0 = picked in proportion to how often it was legal.")

    used = sum(1 for c in deck if played[c] > 0)
    print(f"\ncards used at least once: {used}/8")
    giant = [c for c in deck if names[c] == "Giant"]
    if giant:
        g = giant[0]
        print(f"Giant: {played[g]} plays ({played[g]/max(1,total_plays):.1%} of plays), "
              f"affordable on {affordable[g]/max(1,steps):.1%} of steps")


if __name__ == "__main__":
    main()
