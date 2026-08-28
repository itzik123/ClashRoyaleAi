"""Stage 2: fit the policy heads to the search's value distribution.

FROZEN TRUNK IS THE DEFAULT AND THE MEASUREMENT SAYS SO. The frozen arm reaches
nearly the same conditional lift as the full-network arm (+0.1027 vs +0.1415)
with ZERO critic drift and a sharper selectivity ratio -- and the critic IS the
expert here, so drifting it degrades every future label. `critic_drift()` in
`expert_metrics.py` is what verifies the freeze actually held rather than
assuming it.

TEMPERATURE IS THE KNOB, AND IT MUST BE PICKED FROM THE TARGET'S OWN ENTROPY,
never tuned on the outcome. Candidate value spread is mean 0.221 / median 0.193,
at which T=0.25 puts the target at 94% of maximum entropy -- near-uniform, no
signal, while looking like it is training. T=0.05 puts it at ~55%.
"""
import os
import sys

import numpy as np
import torch

from python_ai.rl.optim_step import clip_and_step

# Run as a script the repo root is not on sys.path, so `python_ai.*` cannot
# resolve; importing the package is also what makes `clash_royale_env` (an
# unpackaged .pyd in python_ai/) importable. See python_ai/__init__.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env  # noqa: E402

from python_ai.models.policy_io import LSTM_HIDDEN  # noqa: E402
from python_ai.trainers import bc_pretrain  # noqa: E402

CE = clash_royale_env.ClashRoyaleEnv
BOARD_W, BOARD_H = CE.BOARD_WIDTH, CE.BOARD_HEIGHT

# Everything upstream of the action heads. Frozen by default -- see the module
# docstring. `card_id_embed`/`noop_embed` feed placement_given_card AND are read
# by extract_features, so they count as trunk: training them would move the
# features the critic sees, which is the thing freezing exists to prevent.
# `cycle_id_head` is trunk for the same reason `aux_card_head` is: it is not an
# action head, and it is the only gradient the (detached) cycle branch has, so
# leaving it trainable here would let a distillation run reshape the branch --
# the exact thing the detach exists to prevent.
TRUNK_MODULES = ("cnn_trunk", "scalar_mlp", "card_id_embed", "lstm",
                 "value_head", "aux_card_head", "cycle_id_head")

TRUNK_PARAMS = ("noop_embed",)

# The heads expert iteration is allowed to move.
POLICY_MODULES = ("card_head", "place_ctx", "place_up")

# --------------------------------------------------------------------------
# stage 2: distil
# --------------------------------------------------------------------------

def freeze_trunk(net):
    """Freeze everything except the action heads. Returns (trainable, frozen).

    Adam skips parameters whose .grad is None, so requires_grad=False is
    sufficient here and needs no change to bc_pretrain.train_bc.
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

def candidate_target(values, n, temperature):
    """softmax(values / T) over the n real candidates. Rows with n<2 are dead.

    Temperature is THE knob here and is deliberately not tuned against the
    outcome metric. Too cold and this collapses back to the argmax label that
    already failed; too hot and every candidate looks equally good and there is
    no signal. It is chosen from the TARGET's own entropy (reported by
    --target-entropy) so the choice is made before any result is seen, which is
    the same discipline the 8.0 upweight was picked with.
    """
    v = values[:n]
    z = (v - v.max()) / max(1e-6, temperature)
    e = np.exp(z)
    return e / e.sum()

def train_distribution(data, net, device, epochs=4, lr=3e-4, batch_episodes=8,
                       temperature=0.25, episode_filter=None, verbose=True):
    """AlphaZero-style distillation: fit the policy to search's VALUE ranking.

    Separate from bc_pretrain.train_bc rather than a flag on it, because the
    loss is a different shape, not a different weighting. train_bc fits one hard
    (card, cell) label per row; this fits a DISTRIBUTION over K joint candidate
    actions, which needs the policy's own factorisation composed explicitly:

        log p_i = log P(card_i) + log P(cell_i | card_i)

    and for the no-op candidate just log P(no-op), since the engine ignores
    placement when cardIndex >= HAND_SIZE.

    Loss is cross-entropy with soft targets, sum_i -q_i log p_i, i.e. KL(q||p)
    up to a constant in q.

    Two things this buys that the hard-label version could not:

      * MARGIN. The hard label says only "candidate 0 lost". The distribution
        says whether it lost by 0.01 or by 0.8, which is exactly the difference
        between "waiting is marginally better" and "playing here is a blunder"
        -- the signal a conditional rule needs and the one the ablation showed
        was missing.
      * PLACEMENT COVERAGE. The hard-label placement loss is masked to rows
        that played a card, which is 10.3% of rows and starved the head. Here
        every row with >= 2 candidates contributes placement gradient whenever
        candidates differ in cell, no-op rows included.
    """
    net = net.to(device).train()
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)

    ep_ids = np.unique(data["episode"])
    if episode_filter is not None:
        keep = set(int(e) for e in episode_filter)
        ep_ids = np.asarray([e for e in ep_ids if int(e) in keep])

    # NOT torch.tensor(data["obs"]) -- that doubles peak RAM (13606 floats/row,
    # ~1.1 GB per 80 episodes) for no benefit, since the recurrent replay below
    # only ever touches one episode at a time. Converting per episode is what
    # lets the dataset scale past the machine's ~5 GB of free memory.
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

                    # One placement forward per DISTINCT card among candidates,
                    # not per candidate -- k_cards is 3 by default while K is up
                    # to 7, so this roughly halves the cost of the dearest head.
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
                    # A candidate the mask forbids has -inf log-prob and would
                    # make the loss inf. Drop it and renormalise, same idea as
                    # train_bc dropping mask-illegal targets.
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
