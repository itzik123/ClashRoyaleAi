"""Does the policy actually COUNT CARDS, or does it merely carry the cycle
channels around? Offline linear probes on a frozen checkpoint.

WHAT IS BEING ASKED. Item 24 put the opponent's `seen[]`/`recency[]` into the
observation and Phase 4 gave that block a dedicated 24-dim scalar branch. Both
are plumbing. The question neither settles is whether the RECURRENT STATE ends
up carrying the opponent's cycle in a linearly decodable form.

THE CONTROL THAT DECIDES IT, AND THE MISTAKE IT AVOIDS. This project has
already read one auxiliary head as evidence of memory and been wrong: the
opponent-elixir head scored MAE 0.77 and looked like a capability, until an
ordinary least-squares fit on two scalars ALREADY IN THE OBSERVATION scored
0.0000. The lesson is that a probe's absolute score means nothing without the
ceiling and the floor beside it. So four feature sets are fitted, never one:

    hx_trained     256   the checkpoint's LSTM state -- what the policy carries
    hx_untrained   256   a RANDOM-INIT net stepped over the SAME rollouts.
                         The random-projection floor. If trained ~= untrained,
                         training contributed nothing and the score is just
                         Johnson-Lindenstrauss doing its job.
    cycle_raw      370   seen[] + recency[] straight from the observation.
                         The information CEILING actually available.
    noncycle       754   everything else (elixir, costs, hand one-hots, the
                         extra scalars). How much is guessable WITHOUT the
                         cycle at all -- the confound floor.

`hx_untrained` is the one that matters most, and it is the analogue of the OLS
control that overturned the elixir result.

TWO TARGETS, because "counting cards" is two different claims:

    HAND    is deck-card c in the opponent's hand right now? (8 binary probes)
            This is what a human literally tracks, and it is DETERMINISTIC
            given the play history, so a high ceiling is expected.
    NEXT    which deck card does the opponent play next? (8-way)
            Carries irreducible entropy -- the opponent chooses heuristically
            among the four it holds -- so the ceiling is well below 1.0 and
            `cycle_raw` is what measures where it sits.

Ground truth comes from the engine (`get_hand_for_team`, `get_last_played_tick`),
never from the cycle channels being probed.

SPLIT BY EPISODE, NEVER BY ROW. Consecutive decisions share almost all of their
state; a random row split leaks the answer across the split and every probe
scores high for a reason that has nothing to do with memory.

Run:
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
            # This repo's checkpoints nest the weights under "model". Probing a
            # checkpoint whose weights silently failed to load is the worst
            # possible failure here -- the "trained" arm becomes a second random
            # net and the probe reports no-learning with total confidence. So
            # the key is REQUIRED to resolve, and a miss raises.
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

            # Ground truth from the ENGINE, not from the channels being probed.
            for c in DEFAULT_DECK:
                lt = g.get_last_played_tick(OPPONENT_TEAM, c)
                if lt > last_tick[c]:
                    last_tick[c] = lt
                    plays.append((len(ep_rows) - 1, deck_index[c]))
            if term or trunc:
                break

        # Label each decision with the NEXT opponent play after it.
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
    torch.set_num_threads(args.threads)   # be polite to the live training run

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
