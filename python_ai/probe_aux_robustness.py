"""Aux-elixir-head robustness to an undercounted opponent spend.

Answers BOT_REQUESTS.md item 4. That item's argument is structural and correct:
extra scalars 1-2 are *cumulative* per-match elixir spend, so unlike every other
(instantaneous) field a sensor error is never corrected by the next frame -- it
is baked in, and it is baked in ONE DIRECTION. A perception layer that misses an
opponent placement can only ever undercount. The auxiliary opponent-elixir head
is trained to infer hidden elixir *from* that scalar, so the error is not noise
on its input, it is a systematic bias in it.

Method -- a controlled ablation, not a re-run:

  1. Roll out episodes with CLEAN observations and record (obs, true opponent
     elixir) per step. The trajectory and the targets are therefore fixed.
  2. For each undercount level, replay the network over those SAME stored
     observations with only `opp_elixir_spent` scaled down, and measure the aux
     head's MAE against the same targets.

Re-running the game under corruption instead would change the trajectory and
the targets too, so MAEs across conditions would not be comparable -- which is
the whole quantity of interest.

Read-only; safe to run against a live trainer.

    python_ai/venv/Scripts/python.exe python_ai/probe_aux_robustness.py
"""

import argparse
import os

import numpy as np
import torch
from torch.distributions import Categorical

import clash_royale_env as E
import gym_wrapper
from model import MicroRoyaleNet

CE = E.ClashRoyaleEnv


def collect(net, n_episodes):
    """Clean rollouts. Returns a list of (obs_seq, true_opp_elixir_seq)."""
    env = gym_wrapper.MicroRoyaleEnv()
    out = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        hx = torch.zeros(1, 256)
        cx = torch.zeros(1, 256)
        o_seq, y_seq = [], []
        for _t in range(400):
            t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            o_seq.append(obs.copy())
            y_seq.append(env.game.get_elixir_for_team(1))
            with torch.no_grad():
                mask = net.affordability_mask(t)
                feats, emb, sp = net.extract_features(t)
                logits, _, _, _, (hx, cx) = net.step_lstm_and_card(feats, (hx, cx), mask)
                idx = Categorical(logits=logits).sample()
                place = net.placement_given_card(hx, emb, idx, t, sp)
                cell = Categorical(logits=place).sample()
            x, y = net.cell_to_xy(cell)
            obs, _r, term, trunc, _i = env.step({
                "card_index": np.array([int(idx.item())]),
                "target_x": x.numpy().reshape(1, 1),
                "target_y": y.numpy().reshape(1, 1),
                "activate_ability_slot1": np.zeros(1, dtype=np.int64),
                "activate_ability_slot2": np.zeros(1, dtype=np.int64),
            })
            if term or trunc:
                break
        if len(o_seq) > 5:
            out.append((np.array(o_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)))
    return out


def aux_mae(net, episodes, keep_fraction, spend_idx):
    """Replay the net over stored observations with opp spend scaled by
    keep_fraction (1.0 = clean, 0.7 = 30% of opponent plays never detected)."""
    errs = []
    for o_seq, y_seq in episodes:
        o = torch.tensor(o_seq).clone()
        o[:, spend_idx] *= keep_fraction
        hx = torch.zeros(1, 256)
        cx = torch.zeros(1, 256)
        with torch.no_grad():
            for i in range(o.shape[0]):
                t = o[i:i + 1]
                feats, _, _ = net.extract_features(t)
                _, _, _, _, (hx, cx) = net.step_lstm_and_card(
                    feats, (hx, cx), net.affordability_mask(t))
                pred = float(net.predict_opp_elixir(hx).item())
                errs.append(abs(pred - float(y_seq[i])))
    return float(np.mean(errs)), len(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="model_weights.pth")
    ap.add_argument("--episodes", type=int, default=12)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    w = args.weights if os.path.isabs(args.weights) else os.path.join(here, args.weights)

    net = MicroRoyaleNet(num_ability_slots=gym_wrapper.DEFAULT_DECK_ABILITY_SLOTS)
    if not os.path.exists(w):
        print(f"!! {w} not found -- an untrained aux head has nothing to ablate.")
        return
    ck = torch.load(w, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    net.load_state_dict(sd)
    net.eval()
    print(f"loaded {os.path.basename(w)} (episode {ck.get('episodes_completed','?')})")

    # absolute index of opp_elixir_spent: spatial block, then the scalar tail's
    # [time, own spend, OPP SPEND, 6 tower HPs] -- see ClashEnv's appended block.
    spend_idx = net.spatial_size + net.extra_start + 2

    eps = collect(net, args.episodes)
    n_steps = sum(len(y) for _, y in eps)
    all_y = np.concatenate([y for _, y in eps])
    mean_baseline = float(np.abs(all_y - all_y.mean()).mean())
    print(f"{len(eps)} episodes, {n_steps} steps | opponent elixir "
          f"mean {all_y.mean():.2f} std {all_y.std():.2f}")
    print(f"predict-the-mean baseline MAE = {mean_baseline:.3f} elixir "
          f"(the point where the head stops being useful)\n")

    print(f"{'opp plays detected':>20}{'spend scalar':>14}{'aux MAE':>10}{'vs clean':>10}")
    print("-" * 54)
    clean = None
    for keep in (1.0, 0.9, 0.8, 0.7, 0.5, 0.0):
        mae, _ = aux_mae(net, eps, keep, spend_idx)
        if clean is None:
            clean = mae
        delta = f"{mae - clean:+.3f}"
        label = "all" if keep == 1.0 else ("none" if keep == 0.0 else f"{keep:.0%}")
        flag = ""
        if mae >= mean_baseline:
            flag = "  <- no better than guessing the mean"
        print(f"{label:>20}{keep:>13.0%}{mae:>10.3f}{delta:>10}{flag}")

    worst, _ = aux_mae(net, eps, 0.0, spend_idx)
    print(f"\nclean {clean:.3f} -> total blackout {worst:.3f} elixir "
          f"({worst/max(clean,1e-9):.2f}x)")
    print("Read: if a 20-30% undercount barely moves MAE, the head is not leaning")
    print("on the accumulator and item 4 is a non-issue for deployment. If MAE")
    print("crosses the mean baseline, the head is relying on a field perception")
    print("cannot deliver, and the observation or its training needs to change.")


if __name__ == "__main__":
    main()
