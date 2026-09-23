"""Does the recurrent state carry the opponent's card cycle, or only the
observation?

Offline linear probes on a frozen checkpoint, over four feature sets, since a
probe's score means nothing without its floor and ceiling beside it:

    hx_trained     256   the checkpoint's LSTM state
    hx_untrained   256   a random-init net stepped over the same rollouts: the random-projection floor. If trained ~= untrained, training contributed nothing.
    cycle_raw      370   seen[] + recency[] from the observation: the information ceiling
    noncycle       754   everything else: how much is guessable without the cycle

Two targets:

    HAND    is deck card c in the opponent's hand now? (8 binary probes; deterministic given the play history)
    NEXT    which deck card does the opponent play next? (8-way; irreducible entropy, so `cycle_raw` sets the ceiling)

Ground truth comes from the engine, never from the channels being probed. The
split is by episode, never by row: consecutive decisions share almost all their
state and a row split leaks the answer.

    python_ai/venv/Scripts/python.exe -m python_ai.eval.probe_card_counting \
        --checkpoint stage_checkpoints/stage0_ep00000939.pth --episodes 40
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401
import clash_royale_env  # noqa: E402
from python_ai.envs.gym_wrapper import DEFAULT_DECK, MicroRoyaleEnv  # noqa: E402
from python_ai.models.net import MicroRoyaleNet  # noqa: E402
from python_ai.models.policy_io import load_state_dict_flexible  # noqa: E402

OPPONENT_TEAM = 1


def collect(checkpoint, episodes, seed=0, max_steps=400):
    """Roll the frozen policy out; record features and ENGINE ground truth."""
    torch.manual_seed(seed)
    net = MicroRoyaleNet(num_ability_slots=0)
    if checkpoint:
        sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(sd, dict):
            # Checkpoints nest the weights under "model". Weights that silently
            # fail to load would make the trained arm a second random net and
            # the probe would report no learning, so a miss raises.
            for k in ("model", "model_state_dict", "state_dict"):
                if k in sd and isinstance(sd[k], dict):
                    sd = sd[k]
                    break
            else:
                if not any(isinstance(v, torch.Tensor) for v in sd.values()):
                    raise KeyError(
                        f"no weight tensors in {checkpoint}; keys={list(sd)[:8]}")
        loaded = load_state_dict_flexible(net, sd, f"probe:{checkpoint}")
        matched = sum(1 for k, v in net.state_dict().items()
                      if k in sd and sd[k].shape == v.shape)
        if matched < 10:
            raise RuntimeError(
                f"only {matched} tensors matched from {checkpoint} -- the "
                "trained arm would be indistinguishable from the random "
                "control, which is exactly the result this probe reports")
        print(f"[probe] loaded {matched} tensors from {checkpoint}")
        del loaded
    net.eval()

    # Same architecture, never trained: the random-projection floor.
    torch.manual_seed(seed + 9999)
    rnd = MicroRoyaleNet(num_ability_slots=0)
    rnd.eval()

    cyc0 = net.spatial_size + net.cycle_start
    cyc1 = cyc0 + net.cycle_block_size
    ncy0, ncy1 = net.spatial_size, net.spatial_size + net.cycle_start
    deck_index = {c: i for i, c in enumerate(DEFAULT_DECK)}

    rows = []
    for ep in range(episodes):
        env = MicroRoyaleEnv()
        obs, _ = env.reset()
        g = env.game
        hx = torch.zeros(1, net.LSTM_HIDDEN); cx = torch.zeros_like(hx)
        rhx = torch.zeros(1, rnd.LSTM_HIDDEN); rcx = torch.zeros_like(rhx)
        last_tick = {c: g.get_last_played_tick(OPPONENT_TEAM, c)
                     for c in DEFAULT_DECK}
        ep_rows, plays = [], []

        for _ in range(max_steps):
            o = torch.as_tensor(obs, dtype=torch.float32).view(1, -1)
            with torch.no_grad():
                f, e, sp, hi = net.extract_features_hires(o)
                m = net.affordability_mask(o)
                cl, _, _, _, (hx, cx) = net.step_lstm_and_card(f, (hx, cx), m)
                rf, _, _, _ = rnd.extract_features_hires(o)
                _, _, _, _, (rhx, rcx) = rnd.step_lstm_and_card(rf, (rhx, rcx), m)
                idx = torch.distributions.Categorical(logits=cl).sample()
                pl = net.placement_given_card(hx, e, idx, o, sp, hires_map=hi)
                cell = torch.distributions.Categorical(logits=pl).sample()
            x, y = net.cell_to_xy(cell)

            hand = set(g.get_hand_for_team(OPPONENT_TEAM))
            ep_rows.append({
                "t": len(ep_rows),
                "hx": hx.numpy()[0].copy(),
                "hx_rnd": rhx.numpy()[0].copy(),
                "cycle_raw": o[0, cyc0:cyc1].numpy().copy(),
                "noncycle": o[0, ncy0:ncy1].numpy().copy(),
                "hand": np.array([1.0 if c in hand else 0.0
                                  for c in DEFAULT_DECK], dtype=np.float32),
            })

            obs, _, term, trunc, _ = env.step(
                {"card_index": int(idx), "target_x": int(x.item()),
                 "target_y": int(y.item())})

            # Ground truth from the engine.
            for c in DEFAULT_DECK:
                lt = g.get_last_played_tick(OPPONENT_TEAM, c)
                if lt > last_tick[c]:
                    last_tick[c] = lt
                    plays.append((len(ep_rows) - 1, deck_index[c]))
            if term or trunc:
                break

        # Label each decision with the next opponent play after it.
        j = 0
        for r in ep_rows:
            while j < len(plays) and plays[j][0] < r["t"]:
                j += 1
            if j < len(plays):
                r["next"] = plays[j][1]
                r["ep"] = ep
                rows.append(r)
    return rows


def _fit_probe(Xtr, ytr, Xte, yte, n_classes, l2=1e-2, steps=300):
    """Multinomial logistic probe, L2-regularised, standardised on TRAIN only."""
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    lin = torch.nn.Linear(Xtr.shape[1], n_classes)
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=steps,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lin(Xtr), ytr)
        loss = loss + l2 * lin.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return (lin(Xte).argmax(1) == yte).float().mean().item()


def run(rows, feature, target, n_classes, holdout_frac=0.3):
    eps = sorted({r["ep"] for r in rows})
    cut = eps[int(len(eps) * (1 - holdout_frac))]
    tr = [r for r in rows if r["ep"] < cut]
    te = [r for r in rows if r["ep"] >= cut]
    Xtr = torch.tensor(np.stack([r[feature] for r in tr]))
    Xte = torch.tensor(np.stack([r[feature] for r in te]))
    ytr = torch.tensor([target(r) for r in tr], dtype=torch.long)
    yte = torch.tensor([target(r) for r in te], dtype=torch.long)
    acc = _fit_probe(Xtr, ytr, Xte, yte, n_classes)
    marg = torch.bincount(ytr, minlength=n_classes).argmax()
    base = (yte == marg).float().mean().item()
    return acc, base, len(tr), len(te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="stage_checkpoints/stage0_ep00000939.pth")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)   # be polite to a live training run

    rows = collect(args.checkpoint, args.episodes)
    print(f"rows={len(rows)}  episodes={len({r['ep'] for r in rows})}  "
          f"checkpoint={args.checkpoint}\n")

    feats = ["hx", "hx_rnd", "cycle_raw", "noncycle"]
    label = {"hx": "hx TRAINED (the policy)", "hx_rnd": "hx UNTRAINED (rand proj)",
             "cycle_raw": "cycle_raw (ceiling)", "noncycle": "noncycle (confound)"}

    print("=== TARGET: opponent's NEXT played card (8-way) ===")
    print(f"{'features':<30}{'acc':>8}{'marginal':>10}{'lift':>8}")
    for f in feats:
        a, b, ntr, nte = run(rows, f, lambda r: r["next"], 8)
        print(f"{label[f]:<30}{a:>8.3f}{b:>10.3f}{a - b:>+8.3f}")

    print(f"\n=== TARGET: is card c in the opponent's HAND (8 binary probes) ===")
    print(f"{'features':<30}{'mean acc':>10}{'marginal':>10}{'lift':>8}")
    for f in feats:
        accs, bases = [], []
        for i in range(8):
            a, b, _, _ = run(rows, f, lambda r, i=i: int(r["hand"][i]), 2)
            accs.append(a); bases.append(b)
        print(f"{label[f]:<30}{np.mean(accs):>10.3f}{np.mean(bases):>10.3f}"
              f"{np.mean(accs) - np.mean(bases):>+8.3f}")
    print(f"\n(train/test split is BY EPISODE; {ntr} train / {nte} test rows)")


if __name__ == "__main__":
    main()
