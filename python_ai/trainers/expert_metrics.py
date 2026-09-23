"""How to tell whether a distilled student learned the conditional rule.

Most of the expert's overrides are "wait where greedy plays", so a policy that
simply no-ops more scores on `disagreement_match` for free. `conditional_lift`
conditions on rows where the original policy plays and compares P(policy waits
| expert waited) with P(policy waits | expert played): indiscriminate drift
moves both together, only a state-conditional rule separates them.
"""
import math
import os
import sys

import numpy as np
import torch

# Run as a script, the repo root is not on sys.path; importing the package also
# makes `clash_royale_env` importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env  # noqa: E402
from python_ai.models.policy_io import LSTM_HIDDEN  # noqa: E402

CE = clash_royale_env.ClashRoyaleEnv
BOARD_W = CE.BOARD_WIDTH

@torch.no_grad()
def critic_drift(net_a, net_b, obs, device, limit=512):
    """Mean |V_a - V_b| and the aux head's drift, over the same observations. ~0
    under a frozen trunk; anything more means the critic that generated the
    labels is moving.
    """
    o = torch.tensor(obs[:limit], dtype=torch.float32, device=device)
    out = []
    for net in (net_a, net_b):
        net.eval()
        feats, _, _ = net.extract_features(o)
        hx = torch.zeros(len(o), LSTM_HIDDEN, device=device)
        cx = torch.zeros(len(o), LSTM_HIDDEN, device=device)
        _, _, _, value, (hx2, _) = net.step_lstm_and_card(feats, (hx, cx))
        # The aux head outputs card logits, so this compares argmax cards: a
        # disagreement rate, still answering whether distillation moved it.
        out.append((value.squeeze(-1),
                    net.aux_card_head(hx2).argmax(-1).float()))
    return (float((out[0][0] - out[1][0]).abs().mean()),
            float((out[0][1] - out[1][1]).abs().mean()))

def modal_cell_baseline(data):
    """Always-guess-the-most-common-cell, on played rows only: the baseline
    placement match must be read against (not 1/612).
    """
    played = data["card"] != CE.HAND_SIZE
    if played.sum() == 0:
        return float("nan"), 0
    cells = data["cell"][played]
    _, counts = np.unique(cells, return_counts=True)
    return float(counts.max() / len(cells)), int(played.sum())

def noop_rate(cards):
    return float((cards == CE.HAND_SIZE).mean())

@torch.no_grad()
def _score_replay(net, data, greedy_card, device, episodes=None):
    """One recurrent replay of `episodes`, scoring both conditional metrics (see
    conditional_metrics).
    """
    net = net.to(device).eval()
    ep_ids = np.unique(data["episode"]) if episodes is None else np.asarray(sorted(episodes))

    dis_ok = dis_n = agr_ok = agr_n = 0
    pred_noop = pred_n = 0
    n1 = n0 = ok1 = ok0 = 0

    for e in ep_ids:
        idx = np.where(data["episode"] == e)[0]
        if len(idx) == 0:
            continue
        o = torch.tensor(data["obs"][idx]).to(device)
        ca = data["card"][idx]
        gc = greedy_card[idx]
        feats, _, _ = net.extract_features(o)
        mask = net.affordability_mask(o)
        hx = torch.zeros(1, LSTM_HIDDEN, device=device)
        cx = torch.zeros(1, LSTM_HIDDEN, device=device)
        for t in range(len(idx)):
            cl, _, _, _, (hx, cx) = net.step_lstm_and_card(
                feats[t:t + 1], (hx, cx), mask[t:t + 1])
            pred = int(cl.argmax(1))
            expert = int(ca[t])
            greedy = int(gc[t])

            # --- conditional_match_rate bookkeeping ---
            pred_noop += int(pred == net.hand_size)
            pred_n += 1
            if expert != greedy:
                dis_n += 1
                dis_ok += int(pred == expert)
            else:
                agr_n += 1
                agr_ok += int(pred == expert)

            # --- conditional_lift bookkeeping ---
            # Rows where greedy already waited have no restraint decision (and
            # still count for the match rate above).
            if greedy != net.hand_size:
                waited = int(pred == net.hand_size)
                if expert == net.hand_size:
                    n1 += 1
                    ok1 += waited
                else:
                    n0 += 1
                    ok0 += waited

    p1 = ok1 / max(1, n1)
    p0 = ok0 / max(1, n0)
    # SE of a difference of two independent proportions, so a lift inside its
    # noise is not read as an effect.
    se = math.sqrt(p1 * (1 - p1) / max(1, n1) + p0 * (1 - p0) / max(1, n0))

    return {
        "match_rate": {
            "disagreement_match": dis_ok / max(1, dis_n),
            "agreement_match": agr_ok / max(1, agr_n),
            "pred_noop_rate": pred_noop / max(1, pred_n),
            "disagreement_n": dis_n,
            "agreement_n": agr_n,
        },
        "lift": {"p1_expert_waited": p1, "p0_expert_played": p0, "lift": p1 - p0,
                 "se": se, "n1": n1, "n0": n0},
    }


def conditional_metrics(net, data, greedy_card, device, episodes=None):
    """Both conditional metrics from one replay: (match_rate, lift). Prefer this
    to calling the two separately on the same arguments.
    """
    r = _score_replay(net, data, greedy_card, device, episodes)
    return r["match_rate"], r["lift"]


@torch.no_grad()
def conditional_match_rate(net, data, greedy_card, device, episodes=None):
    """Split match rates for whether the policy learned the conditional.

      disagreement_match  rows where the expert overrode greedy. The original net
                          scores ~0 here by construction, so movement above 0
                          means a state-dependent rule was learned.
      agreement_match     rows where they already agreed: what moving the first
                          number costs.
      pred_noop_rate      the marginal-drift detector: rising toward the expert's
                          no-op rate without moving disagreement_match means the
                          policy learned the marginal, not the conditional.
    """
    return _score_replay(net, data, greedy_card, device, episodes)["match_rate"]

@torch.no_grad()
def conditional_lift(net, data, greedy_card, device, episodes=None):
    """Separate conditional restraint from indiscriminate no-op drift.

    On rows where the original policy plays:

        p1 = P(policy no-ops | greedy plays, expert WAITED)       should be high
        p0 = P(policy no-ops | greedy plays, expert ALSO PLAYED)  should be low
        lift = p1 - p0

    Drift moves p1 and p0 together (lift ~ 0 at any no-op rate); only a policy
    conditioning on the board separates them.
    """
    return _score_replay(net, data, greedy_card, device, episodes)["lift"]

def disagreement_weights(data, greedy_card, factor):
    """Per-row weights: `factor` on rows the expert overrode, 1.0 elsewhere; None
    keeps train_bc's unweighted path. ~8.6 would equalise the two populations'
    gradient mass.
    """
    if not factor or factor == 1.0:
        return None
    w = np.ones(len(data["card"]), dtype=np.float32)
    w[data["card"] != greedy_card] = float(factor)
    return w

def deviation_breakdown(data):
    """What search changed, split by which head would have to learn it.

    The card head trains on every row; the placement head only on rows that
    played a card, which the expert does ~1 time in 10. So whether
    disagreements are card-level or placement-level decides what distillation
    can reach.
    """
    if "greedy_card" not in data:
        return None
    exp_c, grd_c = data["card"], data["greedy_card"]
    exp_x, grd_x = data["cell"], data["greedy_cell"]
    card_diff = exp_c != grd_c
    # Cells only count where a card was played; the engine ignores placement on
    # no-op rows.
    played = (exp_c != CE.HAND_SIZE) & (grd_c == exp_c)
    cell_diff = played & (exp_x != grd_x)
    n = len(exp_c)
    return {
        "n": n,
        "card_diff": int(card_diff.sum()),
        "cell_diff": int(cell_diff.sum()),
        "agree": int((~card_diff & ~cell_diff).sum()),
        "expert_plays": int((exp_c != CE.HAND_SIZE).sum()),
        "greedy_plays": int((grd_c != CE.HAND_SIZE).sum()),
        "expert_waits_greedy_plays": int(((exp_c == CE.HAND_SIZE) & (grd_c != CE.HAND_SIZE)).sum()),
        "expert_plays_greedy_waits": int(((exp_c != CE.HAND_SIZE) & (grd_c == CE.HAND_SIZE)).sum()),
    }
