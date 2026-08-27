"""How to tell whether a distilled student actually learned the CONDITIONAL rule.

THE OBVIOUS METRIC IS AN ARTIFACT, and this module exists because of it. 87% of
the expert's overrides are "wait where greedy plays", so a policy that simply
no-ops more scores on `disagreement_match` for free. Four ablation configs came
out perfectly monotone in their no-op rate (0.872 / 0.874 / 0.899 / 0.967), with
each "improvement" predicted by that rate alone; the config that looked best
waited on 80% of rows REGARDLESS of what the expert did.

`conditional_lift` is the metric that survives. Condition on rows where the
ORIGINAL policy plays, then compare P(policy waits | expert waited) against
P(policy waits | expert played). Indiscriminate drift moves both together; only
a state-conditional rule separates them. It is the fourth time this project has
landed on the same lesson: an aggregate cannot see a conditional.
"""
import math
import os
import sys

import numpy as np
import torch

# Run as a script the repo root is not on sys.path, so `python_ai.*` cannot
# resolve; importing the package is also what makes `clash_royale_env` (an
# unpackaged .pyd in python_ai/) importable. See python_ai/__init__.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env  # noqa: E402
from python_ai.models.policy_io import LSTM_HIDDEN  # noqa: E402

CE = clash_royale_env.ClashRoyaleEnv
BOARD_W = CE.BOARD_WIDTH

@torch.no_grad()
def critic_drift(net_a, net_b, obs, device, limit=512):
    """Mean |V_a - V_b| and the aux head's drift, over the same observations.

    Under --freeze-trunk this MUST be exactly 0.0: the value head and every
    feature it reads are frozen, so any nonzero number means the freeze did not
    take and the critic that generated the labels is being moved underneath the
    experiment.
    """
    o = torch.tensor(obs[:limit], dtype=torch.float32, device=device)
    out = []
    for net in (net_a, net_b):
        net.eval()
        feats, _, _ = net.extract_features(o)
        hx = torch.zeros(len(o), LSTM_HIDDEN, device=device)
        cx = torch.zeros(len(o), LSTM_HIDDEN, device=device)
        _, _, _, value, (hx2, _) = net.step_lstm_and_card(feats, (hx, cx))
        # argmax card id rather than a scalar since the 2026-08-28 aux swap;
        # the caller reports |a - b| over it, which for a class index is a
        # disagreement count rather than a magnitude -- still the right
        # question (did distillation move the auxiliary head?).
        out.append((value.squeeze(-1),
                    net.aux_card_head(hx2).argmax(-1).float()))
    return (float((out[0][0] - out[1][0]).abs().mean()),
            float((out[0][1] - out[1][1]).abs().mean()))

def modal_cell_baseline(data):
    """Always-guess-the-most-common-cell, on played rows only.

    CLAUDE.md: compare placement against the marginal, never against 1/612.
    The scripted-teacher dataset scored 0.273 this way and a BC run that
    reached 0.073 was therefore far WORSE than guessing, not mildly behind.
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
    """One recurrent replay of `episodes`, scoring BOTH conditional metrics.

    conditional_match_rate and conditional_lift ran character-for-character
    identical replay loops -- same per-episode index select, same
    extract_features/affordability_mask, same zeroed hidden state, same
    step_lstm_and_card walk -- differing only in the bookkeeping inside the
    `t` loop. Two copies of a recurrent unroll is two places for a hidden-state
    or masking fix to be applied to one and not the other.

    Worse, expert_iteration called them back to back on IDENTICAL arguments
    (student + held), so the same rows were replayed twice for no reason. The
    unroll is the expensive part -- it is sequential by construction, one
    LSTM step per row -- so folding them halves that call site outright. Use
    conditional_metrics() where both are wanted.

    Returned dicts are byte-for-byte what the two public functions returned
    before; they are now thin selectors over this.
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
            # Skips rows where greedy already waited: no restraint decision to
            # make. That `continue` came last in the original loop, so it never
            # skipped the match-rate counters above -- preserved by making it a
            # plain conditional here rather than a `continue`.
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
    # SE of a difference of two independent proportions -- printed so a lift
    # inside its own noise is not read as a small positive effect.
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
    """Both conditional metrics from ONE replay: (match_rate, lift).

    Prefer this over calling conditional_match_rate and conditional_lift
    separately on the same arguments -- that replays every row twice.
    """
    r = _score_replay(net, data, greedy_card, device, episodes)
    return r["match_rate"], r["lift"]


@torch.no_grad()
def conditional_match_rate(net, data, greedy_card, device, episodes=None):
    """THE instrument for this question: does the policy learn the CONDITIONAL?

    `card_match_decisions` cannot answer it. ~90% of expert labels already equal
    what the policy does, so that metric is dominated by rows where agreeing
    costs nothing, and a policy that merely drifted toward the expert's MARGINAL
    behaviour ("no-op more often") scores well on it. That is precisely the
    failure the first distillation run is suspected of.

    Split the rows instead:

      disagreement_match -- rows where the expert overrode greedy. The original
        net scores ~0.0 here BY CONSTRUCTION: `greedy_card` was recorded as its
        own argmax at that step, so if it still argmaxes the same way it cannot
        match the expert. Any movement above 0 is the policy having learned a
        state-dependent rule, because the only way to get these right while
        keeping the others is to condition on the board.

      agreement_match -- rows where expert and greedy already agreed. This is
        what a policy PAYS to move the first number. Upweighting disagreements
        trades one against the other, and a lever that lifts disagreement_match
        while collapsing agreement_match has bought nothing.

      pred_noop_rate -- the marginal-drift detector. If a run raises the no-op
        rate toward the expert's 89.7% without moving disagreement_match, it
        learned the marginal and not the conditional, which is the whole
        hypothesis under test.
    """
    return _score_replay(net, data, greedy_card, device, episodes)["match_rate"]

@torch.no_grad()
def conditional_lift(net, data, greedy_card, device, episodes=None):
    """Separates CONDITIONAL restraint from indiscriminate no-op drift.

    `disagreement_match` alone cannot do it, and assuming otherwise is the trap
    this project has now hit four times. 87% of the expert's overrides are
    "wait where greedy plays", so a policy that simply no-ops MORE OFTEN, with
    no regard for the board, scores well on disagreement_match for free. The
    first ablation's four configs came out perfectly monotonic in their no-op
    rate, which is exactly what that artifact looks like.

    So condition on the rows where the disagreement lives -- rows where the
    ORIGINAL policy plays -- and ask whether the new policy's restraint is
    aimed:

        p1 = P(policy no-ops | greedy plays, expert WAITED)   <- should be high
        p0 = P(policy no-ops | greedy plays, expert ALSO PLAYED) <- should be low
        lift = p1 - p0

    Indiscriminate drift moves p1 and p0 together and gives lift ~ 0 no matter
    how high the no-op rate climbs. Only a policy conditioning on the board can
    hold p0 down while pushing p1 up. Lift is therefore the metric that answers
    the actual question, and it is scale-free with respect to the no-op rate.
    """
    return _score_replay(net, data, greedy_card, device, episodes)["lift"]

def disagreement_weights(data, greedy_card, factor):
    """Per-row weights: `factor` on rows the expert overrode, 1.0 elsewhere.

    factor=None returns None, which takes train_bc's original unweighted path.
    The balancing value is ~8.6 (89.54/10.46), which equalises total gradient
    mass between the two populations -- quoted so the choice is a stated
    hypothesis rather than a tuned constant.
    """
    if not factor or factor == 1.0:
        return None
    w = np.ones(len(data["card"]), dtype=np.float32)
    w[data["card"] != greedy_card] = float(factor)
    return w

def deviation_breakdown(data):
    """What search actually CHANGED, split by which head would have to learn it.

    This decides where distillation can possibly help. The two heads are trained
    on very different amounts of data: the card head sees every row, while the
    placement loss is masked to rows that actually played a card -- and the
    expert no-ops ~90% of the time, so the placement head sees ~1 row in 10.

    If most disagreements are card-level (play vs wait, or a different card),
    the card head is the whole story and placement starvation does not matter.
    If most are placement-level, the reachable data is 10x smaller than the row
    count suggests and that has to be said out loud rather than discovered from
    a flat cell_match.
    """
    if "greedy_card" not in data:
        return None
    exp_c, grd_c = data["card"], data["greedy_card"]
    exp_x, grd_x = data["cell"], data["greedy_cell"]
    card_diff = exp_c != grd_c
    # Cell only counts where a card was actually played -- on no-op rows the
    # engine ignores placement entirely, so a "different cell" there is not a
    # real disagreement about anything.
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
