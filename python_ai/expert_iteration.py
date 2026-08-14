"""Expert iteration: distil 1-ply decision-time search back into the policy.

THE MEASUREMENT THIS FOLLOWS FROM
---------------------------------
`search_ab_test.py` measured, over 160 paired trials at 1.5x opponent elixir:

    greedy policy 0.625  ->  policy + 1-ply search 0.944
    paired delta +0.319, 95% CI [+0.237, +0.401], exact McNemar p = 5.6e-12
    deviation rate 13.8%, cost 2.2x wall clock

Search is scored by *this same network's critic*, and the greedy action is
always candidate 0, so search only deviates when the critic disagrees with the
action head. Beating the network's own action selection by 32 points therefore
says the VALUE head is much better than the POLICY head is at exploiting it --
the underfit is in the policy head. That is exactly the gap expert iteration
closes: search is a policy-improvement operator, and its output is a free
supply of expert labels for the head that is behind.

WHAT SUCCESS LOOKS LIKE, AND WHAT IT DOES NOT
---------------------------------------------
The point of distillation is to stop paying the 2.2x search cost, so the
metric is DISTILLED GREEDY vs ORIGINAL GREEDY, paired, both without search.
"Distilled + search beats original" would be a different and much weaker claim.

The band to read the result against is fixed by the numbers above: the original
greedy policy sits at 0.625 and the expert it is imitating sits at 0.944, so a
perfect distillation lands at 0.944 and no distillation lands at 0.625.

THE TRAP THIS FILE IS BUILT AROUND
----------------------------------
86.2% of expert labels are IDENTICAL to what the policy already does (that is
what a 13.8% deviation rate means). So a network that learned nothing at all
already scores ~0.862 card-match on this dataset. Reporting "card match 0.88"
would look like success and mean nothing.

    THE NULL FOR action_match_rate IS THE ORIGINAL NET'S OWN SCORE ON THE
    SAME DATA, NOT ZERO.

`--train` therefore always evaluates BOTH nets on the same rows and prints them
side by side. This is the same correction CLAUDE.md already records for
placement (always-guess-the-modal-cell scores 0.273, so cell-match must be read
against 0.273 and not against 1/612).

WHY THE TRUNK IS FROZEN BY DEFAULT
----------------------------------
The measured finding is specifically that the POLICY HEAD is underfit relative
to the critic. Freezing the trunk/LSTM/value/aux and training only the action
heads tests that hypothesis directly, and preserves bit-exactly the critic that
makes the expert work in the first place -- a full fine-tune moves the shared
features the critic reads, so it can silently degrade the very thing that
generated the labels. `--full-finetune` runs the other arm; both report critic
drift, which must be exactly 0.0 under the default.

Run:
    python_ai/venv/Scripts/python.exe python_ai/expert_iteration.py --collect 120
    python_ai/venv/Scripts/python.exe python_ai/expert_iteration.py --train
    python_ai/venv/Scripts/python.exe python_ai/expert_iteration.py --eval --trials 120
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clash_royale_env  # noqa: E402
import bc_pretrain  # noqa: E402
from bc_pretrain import _cell_from_xy, action_match_rate, train_bc  # noqa: E402
from gym_wrapper import DEFAULT_DECK  # noqa: E402
from model import MicroRoyaleNet  # noqa: E402
from search_ab_test import (  # noqa: E402
    LSTM_HIDDEN, NOOP, _greedy_from_logits, _policy_head, _search_action,
    outcome_score, play_episode,
)
from train import load_state_dict_flexible  # noqa: E402

CE = clash_royale_env.ClashRoyaleEnv
BOARD_W, BOARD_H = CE.BOARD_WIDTH, CE.BOARD_HEIGHT

# Padding width for the per-decision candidate set. The default search emits at
# most 1 + k_cards*k_cells = 7; 8 leaves headroom without costing anything
# meaningful (K_MAX ints per row against a 13606-float observation).
K_MAX = 8

# Everything upstream of the action heads. Frozen by default -- see the module
# docstring. `card_id_embed`/`noop_embed` feed placement_given_card AND are read
# by extract_features, so they count as trunk: training them would move the
# features the critic sees, which is the thing freezing exists to prevent.
TRUNK_MODULES = ("cnn_trunk", "scalar_mlp", "card_id_embed", "lstm",
                 "value_head", "aux_elixir_head")
TRUNK_PARAMS = ("noop_embed",)
# The heads expert iteration is allowed to move.
POLICY_MODULES = ("card_head", "place_ctx", "place_up")


class SearchCfg:
    """The exact search configuration the +0.319 result was measured with.

    Kept as a class rather than reusing argparse's namespace so collection and
    evaluation cannot silently drift onto different search settings -- the
    labels are only expert labels for the configuration that produced them.
    """

    def __init__(self, horizon=4, k_cards=3, k_cells=2, terminal_weight=10.0,
                 max_steps=400):
        self.horizon = horizon
        self.k_cards = k_cards
        self.k_cells = k_cells
        self.terminal_weight = terminal_weight
        self.max_steps = max_steps


def load_net(path, device, verbose=True):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    net = MicroRoyaleNet().to(device)
    clean = load_state_dict_flexible(net, state, path)
    net.eval()
    if verbose:
        eps = ckpt.get("episodes_completed", "?") if isinstance(ckpt, dict) else "?"
        print(f"  loaded {os.path.basename(path)} (episodes_completed={eps}, clean_load={clean})")
        if not clean:
            # load_state_dict_flexible has just printed WHICH case this is:
            # tensors discarded (trained weights lost) or merely tensors the
            # checkpoint predates (nothing lost). Do not restate it as the
            # alarming case -- every checkpoint written before the 2026-08-14
            # `place_hires` branch takes this path harmlessly.
            print("    ^ see the line above for whether anything trained was lost.")
    return net


def verify_cell_roundtrip(net):
    """cell -> (x, y) -> cell must be the identity, over every cell.

    Not defensive boilerplate. bc_pretrain._cell_from_xy carries a comment about
    the one time this went wrong: round() instead of the engine's truncation put
    every Rusher placement one row outside the legal own-half, train_bc silently
    dropped 100% of them as mask-illegal, and placement imitation sat at ~0.9%
    while training appeared to run fine. Expert labels arrive as (x, y) from
    net.cell_to_xy and go back out as cell indices, so the same round trip is on
    the critical path here. One cheap loop makes that failure impossible.
    """
    n_cells = net.placement_cells if hasattr(net, "placement_cells") else BOARD_W * BOARD_H
    cells = torch.arange(n_cells)
    xs, ys = net.cell_to_xy(cells)
    bad = [(int(c), float(x), float(y))
           for c, x, y in zip(cells, xs, ys)
           if _cell_from_xy(float(x), float(y), BOARD_W, BOARD_H) != int(c)]
    if bad:
        raise RuntimeError(
            f"cell_to_xy/_cell_from_xy round trip is not the identity for "
            f"{len(bad)} of {n_cells} cells (first: {bad[0]}). Expert labels "
            f"would be recorded on the wrong cells and train_bc would drop them "
            f"silently as mask-illegal. See bc_pretrain._cell_from_xy.")
    return n_cells


def make_env(opp_elixir, max_ticks):
    env = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), max_ticks)
    env.set_opponent_elixir_multiplier(opp_elixir)
    env.reset()
    return env


# --------------------------------------------------------------------------
# stage 1: collect expert labels
# --------------------------------------------------------------------------

@torch.no_grad()
def collect_episode(net, env, device, cfg, episode_index):
    """Play one episode with search ON, recording the EXPERT's action per step.

    Records every decision, not only the ones where search deviated. Training
    only on deviations would both wreck the action distribution (every
    deviation is a specific play, so the no-op rate would collapse) and throw
    away the "keep doing what you already do here" signal that stops
    distillation from drifting off the states the labels came from.
    """
    hidden = (torch.zeros(1, LSTM_HIDDEN, device=device),
              torch.zeros(1, LSTM_HIDDEN, device=device))
    obs = env.get_observation_for_team(0)
    rows = {"obs": [], "card": [], "cell": [], "episode": [],
            "greedy_card": [], "greedy_cell": [],
            "cand_card": [], "cand_cell": [], "cand_value": [], "cand_n": []}
    reward, steps, deviations = 0.0, 0, 0
    done = False

    while not done and steps < cfg.max_steps:
        obs_np = np.asarray(obs, dtype=np.float32)
        obs_t = torch.tensor(obs_np, device=device).unsqueeze(0)
        card_logits, card_embeds, spatial_map, _, hidden_next = _policy_head(net, obs_t, hidden)
        greedy_card, gx, gy, _ = _greedy_from_logits(
            net, obs_t, card_logits, card_embeds, spatial_map, hidden_next)
        greedy = (greedy_card, gx, gy)

        action, deviated, _, details = _search_action(
            net, env, obs_t, card_logits, card_embeds, spatial_map,
            hidden_next, greedy, cfg, device, return_details=True)
        deviations += int(deviated)

        rows["obs"].append(obs_np)
        rows["card"].append(int(action[0]))
        rows["cell"].append(_cell_from_xy(action[1], action[2], BOARD_W, BOARD_H))
        rows["episode"].append(episode_index)
        rows["greedy_card"].append(int(greedy_card))
        rows["greedy_cell"].append(_cell_from_xy(gx, gy, BOARD_W, BOARD_H))

        # The full ranked candidate set, padded to K_MAX. This is the whole
        # point of the distribution schema: the argmax says only "this one
        # won", while the values say by how much, which is what separates
        # "waiting is marginally better" from "playing here is a blunder".
        c_card = np.full(K_MAX, -1, dtype=np.int64)
        c_cell = np.full(K_MAX, -1, dtype=np.int64)
        c_val = np.zeros(K_MAX, dtype=np.float32)
        n_c = 0
        if details is not None:
            cands, scores = details
            n_c = min(len(cands), K_MAX)
            for i in range(n_c):
                ci, cx, cy = cands[i]
                c_card[i] = int(ci)
                c_cell[i] = _cell_from_xy(cx, cy, BOARD_W, BOARD_H)
                c_val[i] = float(scores[i])
        rows["cand_card"].append(c_card)
        rows["cand_cell"].append(c_cell)
        rows["cand_value"].append(c_val)
        rows["cand_n"].append(n_c)

        hidden = hidden_next
        result = env.step(action[0], action[1], action[2])
        obs, reward, done = result.observation, float(result.reward), result.done
        steps += 1

    return rows, reward, steps, deviations


def collect_expert_labels(net, n_episodes, cfg, device, opp_elixir, max_ticks,
                          time_budget=0.0):
    verify_cell_roundtrip(net)
    acc = {k: [] for k in ("obs", "card", "cell", "episode", "greedy_card", "greedy_cell",
                           "cand_card", "cand_cell", "cand_value", "cand_n")}
    scores, dev_total, dev_steps = [], 0, 0
    started = time.perf_counter()

    for ep in range(n_episodes):
        if time_budget and (time.perf_counter() - started) > time_budget:
            print(f"  [time budget reached after {ep} episodes]")
            break
        env = make_env(opp_elixir, max_ticks)
        rows, reward, steps, dev = collect_episode(net, env, device, cfg, ep)
        for k in acc:
            acc[k].extend(rows[k])
        scores.append(outcome_score(reward))
        dev_total += dev
        dev_steps += steps
        if (ep + 1) % 10 == 0:
            elapsed = time.perf_counter() - started
            print(f"  episode {ep + 1:4d}/{n_episodes} | rows {len(acc['card']):6d} "
                  f"| expert win rate {np.mean(scores):.3f} "
                  f"| deviation {dev_total / max(1, dev_steps):.1%} "
                  f"| {elapsed / (ep + 1):.1f}s/ep")

    data = {
        "obs": np.asarray(acc["obs"], dtype=np.float32),
        "card": np.asarray(acc["card"], dtype=np.int64),
        "cell": np.asarray(acc["cell"], dtype=np.int64),
        "episode": np.asarray(acc["episode"], dtype=np.int64),
        "greedy_card": np.asarray(acc["greedy_card"], dtype=np.int64),
        "greedy_cell": np.asarray(acc["greedy_cell"], dtype=np.int64),
        "cand_card": np.asarray(acc["cand_card"], dtype=np.int64),
        "cand_cell": np.asarray(acc["cand_cell"], dtype=np.int64),
        "cand_value": np.asarray(acc["cand_value"], dtype=np.float32),
        "cand_n": np.asarray(acc["cand_n"], dtype=np.int64),
    }
    meta = {"expert_win_rate": float(np.mean(scores)) if scores else float("nan"),
            "deviation_rate": dev_total / max(1, dev_steps),
            "episodes": len(scores)}
    return data, meta


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
        out.append((value.squeeze(-1), net.aux_elixir_head(hx2).squeeze(-1)))
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
    net = net.to(device).eval()
    ep_ids = np.unique(data["episode"]) if episodes is None else np.asarray(sorted(episodes))
    dis_ok = dis_n = agr_ok = agr_n = 0
    pred_noop = pred_n = 0
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
            pred_noop += int(pred == net.hand_size)
            pred_n += 1
            if int(ca[t]) != int(gc[t]):
                dis_n += 1
                dis_ok += int(pred == int(ca[t]))
            else:
                agr_n += 1
                agr_ok += int(pred == int(ca[t]))
    return {
        "disagreement_match": dis_ok / max(1, dis_n),
        "agreement_match": agr_ok / max(1, agr_n),
        "pred_noop_rate": pred_noop / max(1, pred_n),
        "disagreement_n": dis_n,
        "agreement_n": agr_n,
    }



def merge_datasets(paths, verbose=True):
    """Concatenate several label files, offsetting episode ids so they stay distinct.

    DAgger needs this: round k trains on round k's states UNION every earlier
    round's, or the policy forgets the distribution it was originally correct on.
    Episode ids are the unit of recurrence (train_distribution replays the LSTM
    per episode), so they must not collide across files -- two different
    trajectories sharing an id would be replayed as one spliced sequence.
    """
    import numpy as _np
    keys = ("obs", "card", "cell", "episode", "greedy_card", "greedy_cell",
            "cand_card", "cand_cell", "cand_value", "cand_n")
    out = {k: [] for k in keys}
    offset = 0
    for path in paths:
        z = _np.load(path)
        missing = [k for k in keys if k not in z]
        if missing:
            raise SystemExit(f"{path}: missing {missing} -- recorded before the "
                             f"distribution schema. Re-collect it.")
        n_ep = int(z["episode"].max()) + 1 if len(z["episode"]) else 0
        for k in keys:
            out[k].append(z[k] + offset if k == "episode" else z[k])
        if verbose:
            print(f"    {os.path.basename(path)}: {len(z['card'])} rows, {n_ep} episodes")
        offset += n_ep
    merged = {k: _np.concatenate(out[k]) for k in keys}
    if verbose:
        print(f"    merged: {len(merged['card'])} rows, "
              f"{len(_np.unique(merged['episode']))} episodes")
    return merged


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
                        continue  # nothing to rank; see _search_action
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
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot_loss += float(loss.detach())
            n_batches += 1
        rec = {"epoch": epoch, "loss": tot_loss / max(1, n_batches),
               "rows": n_rows, "argmax_agree": tot_agree / max(1, n_rows)}
        history.append(rec)
        if verbose:
            print(f"  epoch {epoch}: loss {rec['loss']:.4f}  rows {n_rows}  "
                  f"policy-argmax == target-argmax {rec['argmax_agree']:.3f}")
    return net, history


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
    net = net.to(device).eval()
    ep_ids = np.unique(data["episode"]) if episodes is None else np.asarray(sorted(episodes))
    n1 = n0 = ok1 = ok0 = 0
    for e in ep_ids:
        idx = np.where(data["episode"] == e)[0]
        if len(idx) == 0:
            continue
        o = torch.tensor(data["obs"][idx]).to(device)
        ca, gc = data["card"][idx], greedy_card[idx]
        feats, _, _ = net.extract_features(o)
        mask = net.affordability_mask(o)
        hx = torch.zeros(1, LSTM_HIDDEN, device=device)
        cx = torch.zeros(1, LSTM_HIDDEN, device=device)
        for t in range(len(idx)):
            cl, _, _, _, (hx, cx) = net.step_lstm_and_card(
                feats[t:t + 1], (hx, cx), mask[t:t + 1])
            if int(gc[t]) == net.hand_size:
                continue  # greedy already waited: no restraint decision to make
            waited = int(int(cl.argmax(1)) == net.hand_size)
            if int(ca[t]) == net.hand_size:
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
    return {"p1_expert_waited": p1, "p0_expert_played": p0, "lift": p1 - p0,
            "se": se, "n1": n1, "n0": n0}


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


# --------------------------------------------------------------------------
# stage 3: paired greedy-vs-greedy evaluation
# --------------------------------------------------------------------------

def paired_greedy_ab(net_a, net_b, trials, cfg, device, opp_elixir, max_ticks,
                     label_a="original", label_b="distilled", time_budget=0.0):
    """Both nets play GREEDILY from a bit-identical opening. No search anywhere.

    This is the measurement that decides whether expert iteration worked: if
    distillation only helps when search is also running, it has not bought the
    thing it exists to buy.
    """
    diffs, a_scores, b_scores = [], [], []
    started = time.perf_counter()

    for trial in range(trials):
        if time_budget and (time.perf_counter() - started) > time_budget:
            print(f"\n  [time budget reached after {trial} trials]")
            break
        root = make_env(opp_elixir, max_ticks)
        base = root.snapshot()

        r_a, _, _, _ = play_episode(net_a, base.snapshot(), device, False, cfg)
        r_b, _, _, _ = play_episode(net_b, base.snapshot(), device, False, cfg)

        sa, sb = outcome_score(r_a), outcome_score(r_b)
        a_scores.append(sa)
        b_scores.append(sb)
        diffs.append(sb - sa)

        if (trial + 1) % 10 == 0:
            n = len(diffs)
            print(f"  trial {trial + 1:4d} | {label_a} {np.mean(a_scores):.3f} "
                  f"{label_b} {np.mean(b_scores):.3f} | delta {np.mean(diffs):+.3f} "
                  f"| {(time.perf_counter() - started) / n:.1f}s/trial")

    return np.asarray(diffs), np.asarray(a_scores), np.asarray(b_scores)


def report_paired(diffs, a_scores, b_scores, label_a, label_b):
    n = len(diffs)
    if n == 0:
        print("no trials completed")
        return
    mean_d = float(diffs.mean())
    se = float(diffs.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    lo, hi = mean_d - 1.96 * se, mean_d + 1.96 * se
    print("\n" + "=" * 68)
    print(f"trials (paired)        : {n}")
    print(f"{label_a:22s} : {a_scores.mean():.4f}")
    print(f"{label_b:22s} : {b_scores.mean():.4f}")
    print(f"paired delta           : {mean_d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  significant?         : {'YES' if (lo > 0 or hi < 0) else 'no -- CI includes 0'}")
    wins = int((diffs > 0).sum())
    losses = int((diffs < 0).sum())
    ties = int((diffs == 0).sum())
    print(f"  {label_b} better/worse/same : {wins} / {losses} / {ties}")
    disc = wins + losses
    if disc:
        p = 2 * sum(math.comb(disc, k) for k in range(0, min(wins, losses) + 1)) / 2 ** disc
        print(f"  exact McNemar p        : {min(1.0, p):.3e} ({disc} discordant pairs)")
    if n > 1 and se > 0:
        detectable = 1.96 * se
        print(f"\nsmallest effect this n could resolve: +/-{detectable:.4f} "
              f"({detectable * 100:.1f} win-rate points)")
        if abs(mean_d) < detectable:
            needed = int(math.ceil((1.96 * diffs.std(ddof=1) / max(1e-6, abs(mean_d))) ** 2))
            print(f"observed effect is INSIDE the noise floor; resolving it would need "
                  f"~{needed} paired trials.")


# --------------------------------------------------------------------------

def run_ablation(args, cfg, device, resolve):
    """Which lever, if any, forces the policy to learn the CONDITIONAL rule.

    Four configurations over one shared dataset and one shared train/held-out
    split, so nothing between them is confounded:

        A  frozen trunk,   w=1     the configuration that already measured
                                   +0.016 [-0.030, +0.061] on win rate
        B  frozen trunk,   w=F     lever 1 -- upweight disagreement rows
        C  full finetune,  w=1     lever 2 -- capacity
        D  full finetune,  w=F     both

    Deliberately NO win-rate evaluation here. The 800-trial run established that
    this comparison needs ~6,700 paired trials to resolve effects of the size on
    offer, so a per-config win rate at any affordable n would be noise wearing a
    number. These are learning-dynamics metrics only, used to decide where the
    big compute goes.

    The held-out split is what makes the numbers mean anything: unfreezing the
    trunk adds ~118x the trainable parameters, so it will fit the training rows
    better whether or not it generalises. Train and held-out are reported side
    by side and the gap IS the result for lever 2.
    """
    data = bc_pretrain.load_dataset(resolve(args.data))
    extras = np.load(resolve(args.data))
    greedy_card = extras["greedy_card"]

    ep_ids = np.unique(data["episode"])
    n_hold = max(1, int(round(len(ep_ids) * args.holdout_frac)))
    held = set(int(e) for e in ep_ids[-n_hold:])
    train_eps = [int(e) for e in ep_ids if int(e) not in held]
    print(f"dataset : {len(data['card'])} rows, {len(ep_ids)} episodes")
    print(f"split   : {len(train_eps)} train / {len(held)} held-out episodes")
    dis_all = int((data["card"] != greedy_card).sum())
    print(f"disagreement rows: {dis_all} ({dis_all / len(data['card']):.2%})")
    print(f"balancing weight : {(len(data['card']) - dis_all) / max(1, dis_all):.1f} "
          f"(using --upweight {args.upweight})\n")

    original = load_net(resolve(args.weights), device, verbose=False)
    base_tr = conditional_match_rate(original, data, greedy_card, device, train_eps)
    base_ho = conditional_match_rate(original, data, greedy_card, device, held)
    print(f"ORIGINAL net (the null):")
    print(f"  train    disagree {base_tr['disagreement_match']:.4f} "
          f"agree {base_tr['agreement_match']:.4f} noop {base_tr['pred_noop_rate']:.3f}")
    print(f"  held-out disagree {base_ho['disagreement_match']:.4f} "
          f"agree {base_ho['agreement_match']:.4f} noop {base_ho['pred_noop_rate']:.3f}")
    print(f"  expert no-op rate {noop_rate(data['card']):.3f}, "
          f"greedy {noop_rate(greedy_card):.3f}\n")

    configs = [
        ("A frozen  w=1", False, None),
        ("B frozen  w=%g" % args.upweight, False, args.upweight),
        ("C full    w=1", True, None),
        ("D full    w=%g" % args.upweight, True, args.upweight),
    ]
    rows = []
    for name, full_ft, w in configs:
        print("=" * 70)
        print(f"config {name}")
        student = load_net(resolve(args.weights), device, verbose=False)
        if full_ft:
            n_train = sum(p.numel() for p in student.parameters())
            n_froz = 0
        else:
            n_train, n_froz = freeze_trunk(student)
        weights = disagreement_weights(data, greedy_card, w)
        print(f"  {n_train:,} trainable / {n_froz:,} frozen; "
              f"weighting {'off' if weights is None else f'{w}x on disagreements'}")
        t0 = time.perf_counter()
        student.train()
        student, _ = train_bc(data, net=student, epochs=args.epochs, lr=args.lr,
                              batch_episodes=args.batch_episodes, device=device,
                              placement_weight=args.placement_weight, verbose=True,
                              sample_weights=weights, episode_filter=train_eps)
        student.eval()
        tr = conditional_match_rate(student, data, greedy_card, device, train_eps)
        ho = conditional_match_rate(student, data, greedy_card, device, held)
        vd, ad = critic_drift(original, student, data["obs"], device)
        took = time.perf_counter() - t0
        rows.append((name, tr, ho, vd, ad, took))
        print(f"  train    disagree {tr['disagreement_match']:.4f} "
              f"agree {tr['agreement_match']:.4f} noop {tr['pred_noop_rate']:.3f}")
        print(f"  held-out disagree {ho['disagreement_match']:.4f} "
              f"agree {ho['agreement_match']:.4f} noop {ho['pred_noop_rate']:.3f}")
        print(f"  critic drift |dV| {vd:.6f}  aux {ad:.6f}   ({took / 60:.1f} min)")
        torch.save({"model": student.state_dict(), "config": name},
                   resolve(f"exit_ablate_{name.split()[0]}.pth"))

    print("\n" + "=" * 78)
    print("ABLATION SUMMARY  (held-out is the one that matters)")
    print("=" * 78)
    print(f"{'config':16s} {'HO disagree':>12s} {'HO agree':>10s} {'HO noop':>9s} "
          f"{'TR disagree':>12s} {'critic dV':>10s}")
    print(f"{'-- null --':16s} {base_ho['disagreement_match']:12.4f} "
          f"{base_ho['agreement_match']:10.4f} {base_ho['pred_noop_rate']:9.3f} "
          f"{base_tr['disagreement_match']:12.4f} {0.0:10.6f}")
    for name, tr, ho, vd, _ad, _t in rows:
        print(f"{name:16s} {ho['disagreement_match']:12.4f} "
              f"{ho['agreement_match']:10.4f} {ho['pred_noop_rate']:9.3f} "
              f"{tr['disagreement_match']:12.4f} {vd:10.6f}")
    print(f"\nexpert no-op rate {noop_rate(data['card']):.3f} -- a config whose "
          f"HO noop climbs toward it")
    print("without HO disagree moving learned the MARGINAL, not the conditional.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--collect", type=int, metavar="N_EPISODES",
                    help="generate expert labels by playing with search on")
    ap.add_argument("--train", action="store_true", help="distil --data into --weights")
    ap.add_argument("--eval", action="store_true", help="paired greedy A/B, original vs distilled")
    ap.add_argument("--data", default="exit_labels.npz")
    ap.add_argument("--weights", default="model_weights_selfplay.pth")
    ap.add_argument("--out", default="model_weights_exit.pth")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-episodes", type=int, default=8)
    ap.add_argument("--placement-weight", type=float, default=1.0)
    ap.add_argument("--full-finetune", action="store_true",
                    help="train the trunk too (default freezes it; see module docstring)")
    ap.add_argument("--train-dist", action="store_true",
                    help="AlphaZero-style distillation of the candidate value distribution")
    ap.add_argument("--target-entropy", action="store_true",
                    help="report target entropy vs temperature; pick T before measuring outcomes")
    # 0.05, not the 0.25 first guessed. Chosen from --target-entropy on the
    # collected labels BEFORE any outcome was measured: candidate value spread
    # is mean 0.221 / median 0.187, at which 0.25 puts the target at 94% of
    # maximum entropy (near-uniform, no signal) while 0.05 puts it at ~50% --
    # the midpoint between a hard label and no information.
    ap.add_argument("--temperature", type=float, default=0.05,
                    help="softmax temperature on candidate values (see --target-entropy)")
    ap.add_argument("--lift", action="store_true",
                    help="conditional-lift analysis over the saved ablation checkpoints")
    ap.add_argument("--ablate", action="store_true",
                    help="grid over {frozen,full} x {w=1,w=upweight}; learning dynamics only")
    ap.add_argument("--upweight", type=float, default=8.0,
                    help="loss weight on rows where the expert overrode greedy "
                         "(~8.6 balances the two populations)")
    ap.add_argument("--holdout-frac", type=float, default=0.2,
                    help="fraction of EPISODES held out of training for the ablation")
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--k-cards", type=int, default=3)
    ap.add_argument("--k-cells", type=int, default=2)
    ap.add_argument("--terminal-weight", type=float, default=10.0)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--max-ticks", type=int, default=3600)
    ap.add_argument("--opp-elixir", type=float, default=1.5)
    ap.add_argument("--time-budget", type=float, default=0.0)
    args = ap.parse_args()

    device = torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    here = os.path.dirname(os.path.abspath(__file__))

    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(here, p)

    cfg = SearchCfg(args.horizon, args.k_cards, args.k_cells,
                    args.terminal_weight, args.max_steps)

    if args.collect:
        print("=== stage 1: collecting expert labels ===")
        net = load_net(resolve(args.weights), device)
        print(f"  search      : K<={1 + args.k_cards * args.k_cells} candidates, "
              f"horizon={args.horizon} steps")
        print(f"  opponent    : HeuristicOpponent at {args.opp_elixir}x elixir")
        data, meta = collect_expert_labels(
            net, args.collect, cfg, device, args.opp_elixir, args.max_ticks,
            args.time_budget)
        bc_pretrain.save_dataset(data, resolve(args.data))
        mb = data["obs"].nbytes / 2 ** 20
        print(f"\n  wrote {len(data['card'])} rows from {meta['episodes']} episodes "
              f"to {args.data} ({mb:.0f} MB uncompressed)")
        print(f"  expert win rate  : {meta['expert_win_rate']:.4f}")
        print(f"  deviation rate   : {meta['deviation_rate']:.2%}")
        print(f"  expert no-op rate: {noop_rate(data['card']):.2%} "
              f"(greedy {noop_rate(data['greedy_card']):.2%})")

    if args.train:
        print("\n=== stage 2: distilling search into the policy ===")
        data = bc_pretrain.load_dataset(resolve(args.data))
        extras = np.load(resolve(args.data))
        greedy_card = extras["greedy_card"] if "greedy_card" in extras else None

        original = load_net(resolve(args.weights), device)
        student = load_net(resolve(args.weights), device, verbose=False)

        if args.full_finetune:
            print("  mode: FULL FINE-TUNE -- the trunk moves, so the critic that")
            print("        generated these labels can drift underneath them.")
            trainable = sum(p.numel() for p in student.parameters())
            frozen = 0
        else:
            trainable, frozen = freeze_trunk(student)
            print(f"  mode: FROZEN TRUNK -- action heads only")
        print(f"  parameters: {trainable:,} trainable / {frozen:,} frozen")

        # The null, computed BEFORE training and printed next to the result.
        base_match = action_match_rate(original, data, device, limit_episodes=10)
        modal, n_played = modal_cell_baseline(data)
        agree = float((data["card"] == greedy_card).mean()) if greedy_card is not None else float("nan")
        print(f"\n  dataset: {len(data['card'])} rows, {len(np.unique(data['episode']))} episodes")
        print(f"    expert agrees with greedy on : {agree:.2%} of rows")
        print(f"    modal-cell baseline          : {modal:.3f} ({n_played} played rows)")
        bd = deviation_breakdown(extras)
        if bd:
            print(f"    what search CHANGED:")
            print(f"      different card   : {bd['card_diff']:6d} ({bd['card_diff'] / bd['n']:.2%})")
            print(f"        expert waits where greedy plays: {bd['expert_waits_greedy_plays']}")
            print(f"        expert plays where greedy waits: {bd['expert_plays_greedy_waits']}")
            print(f"      same card, cell  : {bd['cell_diff']:6d} ({bd['cell_diff'] / bd['n']:.2%})")
            print(f"      placement head trains on {bd['expert_plays']} rows "
                  f"({bd['expert_plays'] / bd['n']:.1%}); card head on all {bd['n']}")
        print(f"  NULL (original net, untrained on these labels):")
        print(f"    card_match_decisions {base_match['card_match_decisions']:.4f}  "
              f"cell_match {base_match['cell_match']:.4f}")

        student.train()
        student, hist = train_bc(data, net=student, epochs=args.epochs, lr=args.lr,
                                 batch_episodes=args.batch_episodes, device=device,
                                 placement_weight=args.placement_weight)
        student.eval()

        new_match = action_match_rate(student, data, device, limit_episodes=10)
        v_drift, aux_drift = critic_drift(original, student, data["obs"], device)
        print(f"\n  AFTER distillation:")
        print(f"    card_match_decisions {new_match['card_match_decisions']:.4f}  "
              f"(null {base_match['card_match_decisions']:.4f}, "
              f"delta {new_match['card_match_decisions'] - base_match['card_match_decisions']:+.4f})")
        print(f"    cell_match           {new_match['cell_match']:.4f}  "
              f"(null {base_match['cell_match']:.4f}, modal {modal:.3f}, "
              f"delta {new_match['cell_match'] - base_match['cell_match']:+.4f})")
        print(f"    critic drift |dV|    {v_drift:.6f}   aux |d| {aux_drift:.6f}")
        if not args.full_finetune and (v_drift != 0.0 or aux_drift != 0.0):
            print("    !! NONZERO under --freeze-trunk. The freeze did not take; the critic")
            print("       that generated these labels is moving underneath the experiment.")
        torch.save({"model": student.state_dict(),
                    "distilled_from": os.path.basename(args.weights),
                    "labels": os.path.basename(args.data),
                    "frozen_trunk": not args.full_finetune}, resolve(args.out))
        print(f"  wrote {args.out}")

    if args.eval:
        print("\n=== stage 3: paired greedy A/B, original vs distilled ===")
        print("  NO SEARCH in either arm -- the point of distillation is to stop")
        print("  paying for it, so search-on would measure the wrong thing.")
        net_a = load_net(resolve(args.weights), device)
        net_b = load_net(resolve(args.out), device)
        print(f"  opponent : HeuristicOpponent at {args.opp_elixir}x elixir")
        print(f"  reference: original greedy 0.625 -> search expert 0.944 "
              f"(search_ab_test, n=160)\n")
        diffs, a, b = paired_greedy_ab(net_a, net_b, args.trials, cfg, device,
                                       args.opp_elixir, args.max_ticks,
                                       time_budget=args.time_budget)
        report_paired(diffs, a, b, "original greedy", "distilled greedy")

    if args.ablate:
        run_ablation(args, cfg, device, resolve)

    if args.target_entropy:
        print("=== target-distribution entropy vs temperature ===")
        print("Chosen BEFORE any outcome is measured. A target at ~0 nats is the")
        print("argmax label that already failed; at ~log(K) it carries no signal.\n")
        z = np.load(resolve(args.data))
        cv, cn = z["cand_value"], z["cand_n"]
        live = np.where(cn >= 2)[0]
        print(f"rows with >=2 candidates: {len(live)} of {len(cn)} "
              f"({len(live) / max(1, len(cn)):.1%})")
        spread = np.array([cv[r][:cn[r]].max() - cv[r][:cn[r]].min() for r in live])
        print(f"candidate value spread  : mean {spread.mean():.4f} "
              f"median {np.median(spread):.4f}\n")
        print(f"{'T':>8s} {'mean entropy':>14s} {'max possible':>14s} {'frac of max':>12s}")
        for T in (0.05, 0.1, 0.25, 0.5, 1.0):
            ents, maxes = [], []
            for r in live[:4000]:
                n = int(cn[r])
                q = candidate_target(cv[r], n, T)
                ents.append(float(-(q * np.log(q + 1e-12)).sum()))
                maxes.append(math.log(n))
            print(f"{T:8.2f} {np.mean(ents):14.4f} {np.mean(maxes):14.4f} "
                  f"{np.mean(ents) / max(1e-9, np.mean(maxes)):12.3f}")

    if args.train_dist:
        print("\n=== distribution distillation (AlphaZero-style) ===")
        paths = [resolve(p.strip()) for p in args.data.split(",") if p.strip()]
        print("  datasets:")
        data = merge_datasets(paths)
        # Same guard load_dataset applies: a dataset recorded against a different
        # observation layout must not be silently reinterpreted.
        expected = make_env(1.0, 100).observation_size()
        if data["obs"].shape[1] != expected:
            raise SystemExit(f"observation size {data['obs'].shape[1]} != engine's {expected}")
        greedy_card = data["greedy_card"]

        ep_ids = np.unique(data["episode"])
        n_hold = max(1, int(round(len(ep_ids) * args.holdout_frac)))
        held = [int(e) for e in ep_ids[-n_hold:]]
        train_eps = [int(e) for e in ep_ids if int(e) not in held]

        original = load_net(resolve(args.weights), device, verbose=False)
        student = load_net(resolve(args.weights), device, verbose=False)
        if args.full_finetune:
            n_tr, n_fz = sum(p.numel() for p in student.parameters()), 0
        else:
            n_tr, n_fz = freeze_trunk(student)
        print(f"  {n_tr:,} trainable / {n_fz:,} frozen, T={args.temperature}, "
              f"{len(train_eps)} train / {len(held)} held-out episodes")

        base = conditional_lift(original, data, greedy_card, device, held)
        print(f"  NULL lift {base['lift']:+.4f} +/- {1.96 * base['se']:.4f}\n")

        student, _ = train_distribution(
            data, student, device, epochs=args.epochs, lr=args.lr,
            batch_episodes=args.batch_episodes, temperature=args.temperature,
            episode_filter=train_eps)
        student.eval()

        lift = conditional_lift(student, data, greedy_card, device, held)
        cond = conditional_match_rate(student, data, greedy_card, device, held)
        vd, ad = critic_drift(original, student, data["obs"], device)
        ci = 1.96 * lift["se"]
        print(f"\n  CONDITIONAL LIFT {lift['lift']:+.4f} +/- {ci:.4f}  "
              f"({'CONDITIONAL' if lift['lift'] - ci > 0 else 'not significant'})")
        print(f"    p1 (expert waited) {lift['p1_expert_waited']:.4f}  "
              f"p0 (expert played) {lift['p0_expert_played']:.4f}")
        print(f"  held-out disagree {cond['disagreement_match']:.4f}  "
              f"agree {cond['agreement_match']:.4f}  noop {cond['pred_noop_rate']:.3f} "
              f"(expert {noop_rate(data['card']):.3f})")
        print(f"  critic drift |dV| {vd:.6f}  aux {ad:.6f}")
        torch.save({"model": student.state_dict(), "temperature": args.temperature},
                   resolve(args.out))
        print(f"  wrote {args.out}")

    if args.lift:
        print("=== conditional lift: aimed restraint vs indiscriminate no-op drift ===")
        data = bc_pretrain.load_dataset(resolve(args.data))
        greedy_card = np.load(resolve(args.data))["greedy_card"]
        ep_ids = np.unique(data["episode"])
        n_hold = max(1, int(round(len(ep_ids) * args.holdout_frac)))
        held = [int(e) for e in ep_ids[-n_hold:]]

        nets = [("-- null --", resolve(args.weights))]
        for tag in ("A", "B", "C", "D"):
            p = resolve(f"exit_ablate_{tag}.pth")
            if os.path.exists(p):
                nets.append((f"{tag}", p))
        print(f"held-out episodes: {len(held)}   "
              f"expert no-op rate {noop_rate(data['card']):.3f}\n")
        print(f"{'config':10s} {'p1 (exp waited)':>16s} {'p0 (exp played)':>16s} "
              f"{'lift':>9s} {'+/-1.96se':>10s} {'verdict':>22s}")
        for tag, path in nets:
            net = load_net(path, device, verbose=False)
            r = conditional_lift(net, data, greedy_card, device, held)
            ci = 1.96 * r["se"]
            verdict = "CONDITIONAL" if r["lift"] - ci > 0 else "indiscriminate/none"
            print(f"{tag:10s} {r['p1_expert_waited']:16.4f} {r['p0_expert_played']:16.4f} "
                  f"{r['lift']:+9.4f} {ci:10.4f} {verdict:>22s}")
        print(f"\nn1={r['n1']} rows where greedy played and the expert waited; "
              f"n0={r['n0']} where both played.")
        print("lift ~ 0 means the extra waiting is untargeted -- the policy learned")
        print("the expert's MARGINAL restraint, not the state-dependent rule.")

    if not (args.collect or args.train or args.eval or args.ablate or args.lift
            or args.train_dist or args.target_entropy):
        ap.print_help()


if __name__ == "__main__":
    main()
