"""Per-card usage probe. Read-only: never touches a live training run.

Exists because phase 1 (`train.py`) has no per-card diagnostic at all -- the
Cards/Game counter only lives in phase 2 -- while the single open question about
the current DEFAULT_DECK is whether the 5-cost Giant is ever played. That
question was previously answered only by ad hoc probes, and the answer
(never, across four full runs) is why the deck was swapped away and back.

Reads `model_weights.pth` off disk and runs its own environments, so it is safe
to run against a live trainer. The only interaction is a file read of a
checkpoint the trainer writes atomically-enough (torch.save to a fresh path);
if it ever races, the failure is a clean load error, not corruption.

Usage (needs the 3.11 venv -- the .pyd is 3.11 only):

    python_ai/venv/Scripts/python.exe python_ai/probe_card_usage.py
    python_ai/venv/Scripts/python.exe python_ai/probe_card_usage.py --episodes 40 --greedy

Reports, per card: how often it was PLAYED, and -- separately -- how often it
was even AFFORDABLE. Those two must be read together: a card at 0% usage that
was affordable 40% of the time is a policy choice, while one that was
affordable 2% of the time is starved by the cost curve, and the fixes are
opposite.
"""

import argparse
import os
from collections import Counter

import numpy as np
import torch
from torch.distributions import Categorical

import clash_royale_env as E
import gym_wrapper
from model import MicroRoyaleNet
from policy_io import load_state_dict_flexible

CE = E.ClashRoyaleEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="model_weights.pth")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--greedy", action="store_true",
                    help="argmax instead of sampling -- measures the policy's "
                         "committed preference rather than its exploration")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    weights = args.weights if os.path.isabs(args.weights) else os.path.join(here, args.weights)

    deck = list(gym_wrapper.DEFAULT_DECK)
    names = {c: E.get_card_info(c)["name"] for c in deck}
    costs = {c: E.get_card_info(c)["cost"] for c in deck}

    net = MicroRoyaleNet(num_ability_slots=gym_wrapper.DEFAULT_DECK_ABILITY_SLOTS)
    if os.path.exists(weights):
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        # Flexible, not strict -- see probe_aux_robustness.py for why: a
        # checkpoint predating the `place_hires` branch is not a mismatch.
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

            # Which real card ids the mask said were affordable THIS step, read
            # from the same observation the decision was made on -- the hand
            # rotates the instant a card is played, so reading get_hand()
            # afterwards would report the NEXT card, not the chosen one.
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
    # play share and affordability share are NOT comparable directly -- measured
    # affordability sits at 3-9% for every card, so any absolute threshold on it
    # mislabels the whole deck. The meaningful quantity is TAKE-UP: play share
    # divided by opportunity share, i.e. "when this card was legal, how often
    # did the policy pick it relative to how often it picked anything". Take-up
    # near 0 with healthy affordability is a policy choice; low affordability is
    # the cost curve. They need opposite fixes, so they must be separable.
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
