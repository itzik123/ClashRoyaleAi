"""Stage 2 of expert iteration: fit the policy heads to the search's value
distribution.

The trunk is frozen by default: it reaches nearly the conditional lift of full
fine-tuning with zero critic drift, and the critic is the expert, so moving it
degrades every future label (`expert_metrics.critic_drift` checks). The
temperature must be chosen from the target's own entropy, never from the
outcome; see `calibrate_temperature`.
"""
import os
import sys

import math

import numpy as np
import torch

from python_ai.rl.optim_step import clip_and_step

# Run as a script, the repo root is not on sys.path; importing the package also
# makes `clash_royale_env` importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env  # noqa: E402

from python_ai.models.policy_io import LSTM_HIDDEN  # noqa: E402
from python_ai.trainers import bc_pretrain  # noqa: E402

CE = clash_royale_env.ClashRoyaleEnv
BOARD_W, BOARD_H = CE.BOARD_WIDTH, CE.BOARD_HEIGHT

# Everything upstream of the action heads, frozen by default.
# `card_id_embed`/`noop_embed` are read by extract_features, so training them
# would move the critic's features. `cycle_id_head` is the detached cycle
# branch's only gradient, so training it here would reshape the branch.
TRUNK_MODULES = ("cnn_trunk", "scalar_mlp", "card_id_embed", "lstm",
                 "value_head", "aux_card_head", "cycle_id_head")

TRUNK_PARAMS = ("noop_embed",)

# The heads expert iteration may move.
POLICY_MODULES = ("card_head", "place_ctx", "place_up")

# --- stage 2: distil ---

def freeze_trunk(net):
    """Freeze everything except the action heads. Returns (trainable, frozen).

    Adam skips parameters without gradients, so requires_grad=False suffices.
    """
    for p in net.parameters():
        p.requires_grad = False
    for name in POLICY_MODULES:
        mod = getattr(net, name, None)
        if mod is None:
            raise AttributeError(
                f"MicroRoyaleNet has no module '{name}' -- POLICY_MODULES is out of "
                f"date with model.py and freezing would silently train nothing.")
        for p in mod.parameters():
            p.requires_grad = True
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in net.parameters() if not p.requires_grad)
    return trainable, frozen

#: Where the soft target should sit, as a fraction of log(n_candidates): near 0
#: is the hard argmax label (measured null), near 1 is uniform.
TARGET_ENTROPY_FRAC = 0.55


def target_entropy_frac(cand_value, cand_n, temperature):
    """Median entropy of softmax(values / T), as a fraction of log(n)."""
    fracs = []
    for row in range(len(cand_n)):
        k = int(cand_n[row])
        if k < 2:
            continue
        v = np.asarray(cand_value[row][:k], dtype=np.float64)
        z = (v - v.max()) / max(1e-9, temperature)
        p = np.exp(z)
        tot = p.sum()
        if not np.isfinite(tot) or tot <= 0:
            continue
        p = p / tot
        nz = p[p > 0]
        fracs.append(float(-(nz * np.log(nz)).sum() / math.log(k)))
    return float(np.median(fracs)) if fracs else 0.0


def calibrate_temperature(cand_value, cand_n, target_frac=None,
                          lo=1e-4, hi=10.0, iters=60):
    """The T whose target sits at `target_frac` of maximum entropy.

    Solved per dataset because T depends on the critic's value spread, which
    expert iteration changes. A badly chosen T gives a near-uniform target that
    looks like training. Entropy is monotone in T, so bisection is exact.
    """
    target_frac = TARGET_ENTROPY_FRAC if target_frac is None else target_frac

    spreads = [float(np.ptp(np.asarray(cand_value[r][:int(cand_n[r])])))
               for r in range(len(cand_n)) if int(cand_n[r]) >= 2]
    if not spreads or max(spreads) <= 1e-9:
        raise ValueError(
            "cannot calibrate a temperature: candidate value spread is zero, so "
            "every target is uniform at every T. This is the no-op duplication "
            "signature -- check the candidate set is not one action recorded twice.")

    for _ in range(iters):
        mid = math.sqrt(lo * hi)          # geometric: T spans orders of magnitude
        if target_entropy_frac(cand_value, cand_n, mid) < target_frac:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def candidate_target(values, n, temperature):
    """softmax(values / T) over the n real candidates; rows with n < 2 carry
    nothing.
    """
    v = values[:n]
    z = (v - v.max()) / max(1e-6, temperature)
    e = np.exp(z)
    return e / e.sum()

def train_distribution(data, net, device, epochs=4, lr=3e-4, batch_episodes=8,
                       temperature=0.25, episode_filter=None, verbose=True):
    """AlphaZero-style distillation: fit the policy to search's value distribution
    over K joint candidates.

    The policy's factorisation is composed explicitly, log p_i = log P(card_i)
    + log P(cell_i | card_i), or just log P(no-op) for the no-op (the engine
    ignores its placement). The loss is soft-target cross-entropy, sum_i -q_i
    log p_i.

    Unlike hard labels, the target carries the margin between candidates, and
    every row with >= 2 candidates trains the placement head, not only rows
    that played a card.
    """
    net = net.to(device).train()
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)

    ep_ids = np.unique(data["episode"])
    if episode_filter is not None:
        keep = set(int(e) for e in episode_filter)
        ep_ids = np.asarray([e for e in ep_ids if int(e) in keep])

    # Observations are converted per episode, not all at once, so the dataset
    # can exceed free memory.
    obs_np = data["obs"]
    cand_card = data["cand_card"]
    cand_cell = data["cand_cell"]
    cand_value = data["cand_value"]
    cand_n = data["cand_n"]

    history = []
    for epoch in range(epochs):
        np.random.shuffle(ep_ids)
        tot_loss, n_rows, n_batches = 0.0, 0, 0
        tot_agree = 0
        for b in range(0, len(ep_ids), batch_episodes):
            chunk = ep_ids[b:b + batch_episodes]
            loss = 0.0
            used = 0
            for e in chunk:
                idx = np.where(data["episode"] == e)[0]
                o = torch.from_numpy(obs_np[idx]).to(device)
                feats, embeds, spatial = net.extract_features(o)
                mask = net.affordability_mask(o)
                hx = torch.zeros(1, LSTM_HIDDEN, device=device)
                cx = torch.zeros(1, LSTM_HIDDEN, device=device)
                for t in range(len(idx)):
                    cl, _, _, _, (hx, cx) = net.step_lstm_and_card(
                        feats[t:t + 1], (hx, cx), mask[t:t + 1])
                    row = idx[t]
                    n = int(cand_n[row])
                    if n < 2:
                        continue  # nothing to rank; see search_action
                    q = torch.tensor(
                        candidate_target(cand_value[row], n, temperature),
                        dtype=torch.float32, device=device)
                    card_lp = torch.log_softmax(cl, dim=-1)[0]

                    # One placement forward per distinct card among the
                    # candidates.
                    cards = [int(cand_card[row, i]) for i in range(n)]
                    place_lp = {}
                    for c in set(cards):
                        if c >= net.hand_size:
                            continue
                        c_t = torch.tensor([c], device=device)
                        pl = net.placement_given_card(
                            hx, embeds[t:t + 1], c_t, o[t:t + 1], spatial[t:t + 1])
                        place_lp[c] = torch.log_softmax(pl, dim=-1)[0]

                    logp = []
                    for i in range(n):
                        c = cards[i]
                        if c >= net.hand_size:
                            logp.append(card_lp[net.hand_size])
                        else:
                            logp.append(card_lp[c] + place_lp[c][int(cand_cell[row, i])])
                    logp = torch.stack(logp)
                    # A mask-forbidden candidate has -inf log-prob; drop it and
                    # renormalise.
                    ok = torch.isfinite(logp)
                    if ok.sum() < 2:
                        continue
                    qq = q[ok] / q[ok].sum().clamp_min(1e-8)
                    loss = loss - (qq * logp[ok]).sum()
                    used += 1
                    n_rows += 1
                    tot_agree += int(int(logp[ok].argmax()) == int(qq.argmax()))
            if not torch.is_tensor(loss) or used == 0:
                continue
            loss = loss / used
            opt.zero_grad()
            loss.backward()
            clip_and_step(opt, net.parameters(), 1.0)
            tot_loss += float(loss.detach())
            n_batches += 1
        rec = {"epoch": epoch, "loss": tot_loss / max(1, n_batches),
               "rows": n_rows, "argmax_agree": tot_agree / max(1, n_rows)}
        history.append(rec)
        if verbose:
            print(f"  epoch {epoch}: loss {rec['loss']:.4f}  rows {n_rows}  "
                  f"policy-argmax == target-argmax {rec['argmax_agree']:.3f}")
    return net, history
