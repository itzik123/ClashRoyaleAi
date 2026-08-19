"""Stage 1 of expert iteration: play with search on and record what it chose.

The label is not just the argmax. `--train-dist` fits the search's full VALUE
DISTRIBUTION over candidates, and that is the difference between a lift of
+0.1027 and one of +0.0462: a single hard label strips the MARGIN, so it cannot
distinguish "waiting is marginally better" from "playing here is a blunder" --
which is exactly the distinction a state-conditional rule is made of.

A NO-OP DUPLICATION SILENTLY DESTROYED THE FIRST ATTEMPT, and it is the reason
`_dedupe`-style care matters here. `(NOOP, gx, gy)` and `(NOOP, 0, 0)` are the
SAME action -- `step()` ignores placement for the no-op -- but differ as tuples,
so the no-op was emitted twice and scored twice, identically. Harmless for a
win-rate A/B (the duplicate ties and argmax returns greedy) and fatal here:
99.1% of rows had a target whose median value spread was exactly 0.0.
"""
import time
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
from python_ai.envs.gym_wrapper import DEFAULT_DECK  # noqa: E402
from python_ai.models.policy_io import LSTM_HIDDEN  # noqa: E402
from python_ai.search.search import (  # noqa: E402
    greedy_from_logits, outcome_score, policy_head, search_action,
)
from python_ai.trainers.bc_pretrain import _cell_from_xy  # noqa: E402

CE = clash_royale_env.ClashRoyaleEnv
BOARD_W, BOARD_H = CE.BOARD_WIDTH, CE.BOARD_HEIGHT

# Padding width for the per-decision candidate set. The default search
# (k_cards=3, k_cells=2) emits at most 7 -- enumerated exhaustively over which
# columns topk selects and whether greedy is the no-op, NOT the naive
# 1 + k_cards*k_cells. 8 leaves headroom without costing anything meaningful
# (K_MAX ints per row against a 13606-float observation).
#
# The default is the ONLY configuration that fits. Enumerated maxima:
#     k_cards=3 k_cells=2 ->  7   (fits)
#     k_cards=4 k_cells=2 ->  9   (overflows)
#     k_cards=3 k_cells=3 -> 10   (overflows)
#     k_cards=5 k_cells=3 -> 13   (overflows)
# Overflow is no longer silent-and-harmful (candidates are kept by SCORE, see
# collect_episode) but it is still lossy, so _warn_truncation reports it once.
K_MAX = 8

# Module-level so the warning fires once per process, not once per decision.
_truncation_warned = False

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
        card_logits, card_embeds, spatial_map, _, hidden_next = policy_head(net, obs_t, hidden)
        greedy_card, gx, gy, _ = greedy_from_logits(
            net, obs_t, card_logits, card_embeds, spatial_map, hidden_next)
        greedy = (greedy_card, gx, gy)

        action, deviated, _, details = search_action(
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
            # Truncate by SCORE, not by generation order. This used to be
            # `for i in range(min(len(cands), K_MAX))`, which keeps whichever
            # candidates happened to be generated first -- so a configuration
            # producing more than K_MAX could silently drop the HIGHEST-scoring
            # candidate, including the search winner, out of the recorded soft
            # target. The distribution would then be fitted to a set that does
            # not contain the action search actually chose.
            #
            # K_MAX stays a fixed schema width so merge_datasets' np.concatenate
            # keeps working. Ordering within the row carries no meaning --
            # candidate_target softmaxes values[:n] and cand_card/cell/value are
            # read positionally in lockstep -- so re-ranking is safe.
            if len(cands) > K_MAX:
                global _truncation_warned
                if not _truncation_warned:
                    _truncation_warned = True
                    print(f"  WARNING: search emitted {len(cands)} candidates but "
                          f"K_MAX={K_MAX}; keeping the {K_MAX} highest-scoring and "
                          f"DISCARDING the rest. The recorded target is a partial "
                          f"ranking. Raise K_MAX (it is a schema width -- datasets "
                          f"recorded at different K_MAX cannot be merged) or lower "
                          f"--k-cards/--k-cells. Only k_cards=3,k_cells=2 fits 8.",
                          flush=True)
            order = np.argsort(-np.asarray(scores, dtype=np.float64))[:K_MAX]
            n_c = len(order)
            for j, i in enumerate(order):
                ci, cx, cy = cands[i]
                c_card[j] = int(ci)
                c_cell[j] = _cell_from_xy(cx, cy, BOARD_W, BOARD_H)
                c_val[j] = float(scores[i])
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
