"""Stage 1 of expert iteration: play with search on and record what it chose.

Records the full ranked candidate set with values, not just the argmax: a
distribution target keeps the margin that separates "waiting is marginally
better" from "playing here is a blunder", which a hard label cannot.
"""
import time
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
from python_ai.envs.gym_wrapper import DEFAULT_DECK  # noqa: E402
from python_ai.models.policy_io import LSTM_HIDDEN  # noqa: E402
from python_ai.search.search import (  # noqa: E402
    greedy_from_logits, outcome_score, policy_head, search_action,
)
from python_ai.trainers.bc_pretrain import _cell_from_xy  # noqa: E402

CE = clash_royale_env.ClashRoyaleEnv
BOARD_W, BOARD_H = CE.BOARD_WIDTH, CE.BOARD_HEIGHT

# Schema width for the per-decision candidate set (fixed, so datasets can be
# merged). Widened search can emit more; `select_candidate_indices` then
# subsamples.
K_MAX = 160

def select_candidate_indices(scores, k_max=None):
    """Which candidates to record, value-sorted best first, capped at `k_max`.

    Overflow takes a uniform stride over the sorted list, not the top-k: the
    top-k are the most similar candidates, and keeping only them would collapse
    the target's range. The argmax (index 0) always survives.
    """
    k_max = K_MAX if k_max is None else k_max
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    if len(order) <= k_max:
        return order
    # Endpoints pinned, the rest spread evenly between them.
    idx = np.linspace(0, len(order) - 1, k_max)
    return order[np.unique(np.round(idx).astype(int))]


# Warn once per process.
_truncation_warned = False

def verify_cell_roundtrip(net):
    """cell -> (x, y) -> cell must be the identity over every cell.

    Labels go through cell_to_xy and back; a rounding mismatch once made
    train_bc drop every label as mask-illegal while training appeared to run.
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
    """The original collection env: the deck mirror against the C++
    HeuristicOpponent. Still the default, because recorded datasets were
    collected on it; see `make_pool_env` for the training distribution.
    """
    env = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), max_ticks)
    env.set_opponent_elixir_multiplier(opp_elixir)
    env.reset()
    return env


class PoolTeacherEnv:
    """The raw-env surface `collect_episode` needs, backed by the phase-1 training
    env (UtilityTeacher on a pool deck).

    The mirror rarely produces the boards where cards like Cannon, The Log and
    Fireball matter, so labels collected there teach aiming for boards the
    agent no longer sees, silently. This adapts the gym wrapper to the raw
    `ClashRoyaleEnv` protocol (`get_observation_for_team` / `snapshot` / `step`
    returning `.observation`, `.reward`, `.done`).

    `snapshot()` returns a raw engine copy, so search's rollouts are still
    stepped by the C++ heuristic while the real episode is played by the
    teacher; that is a property of the scorer, not of the state distribution.
    """

    class _Result:
        __slots__ = ("observation", "reward", "done")

        def __init__(self, observation, reward, done):
            self.observation = observation
            self.reward = reward
            self.done = done

    def __init__(self, stage, max_ticks, seed, deck):
        from python_ai.envs import gym_wrapper

        self._gym = gym_wrapper.MicroRoyaleEnv({
            "ai_deck": list(DEFAULT_DECK),
            "opp_deck": list(deck.card_ids),
            # The key is `opponent`; gym_wrapper ignores `opponent_kind` and
            # would silently use the builtin opponent.
            "opponent": "teacher",
            "teacher_stage": int(stage),
            "max_ticks": int(max_ticks),
        })
        self._gym.game.seed(seed)
        self._gym.reset()
        self.deck_name = deck.name
        self.game = self._gym.game

    def get_observation_for_team(self, team):
        return self._gym.game.get_observation_for_team(team)

    def snapshot(self):
        return self._gym.game.snapshot()

    def observation_size(self):
        return self._gym.game.observation_size()

    def step(self, card, x, y):
        obs, reward, term, trunc, _ = self._gym.step(
            {"card_index": card, "target_x": x, "target_y": y})
        return self._Result(obs, float(reward), bool(term or trunc))


def make_pool_env(stage, max_ticks, seed, deck):
    """Phase 1's real distribution: UtilityTeacher at `stage`, playing `deck`.
    """
    return PoolTeacherEnv(stage, max_ticks, seed, deck)

# --- stage 1: collect expert labels ---

@torch.no_grad()
def collect_episode(net, env, device, cfg, episode_index):
    """Play one episode with search on, recording the expert's action at every
    step.

    Every decision, not only deviations: training on deviations alone would
    collapse the no-op rate and drop the "keep doing this" signal.
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

        # The full ranked candidate set, padded to K_MAX.
        c_card = np.full(K_MAX, -1, dtype=np.int64)
        c_cell = np.full(K_MAX, -1, dtype=np.int64)
        c_val = np.zeros(K_MAX, dtype=np.float32)
        n_c = 0
        if details is not None:
            cands, scores = details
            # Selected by score, never generation order, so the search winner
            # is always in the recorded target. Order within a row carries no
            # meaning.
            if len(cands) > K_MAX:
                global _truncation_warned
                if not _truncation_warned:
                    _truncation_warned = True
                    print(f"  NOTE: search emitted {len(cands)} candidates against "
                          f"K_MAX={K_MAX}; recording a uniform stride over the "
                          f"value-sorted set, so the argmax and BOTH endpoints of "
                          f"the spread survive. The target is a subsample of the "
                          f"ranking, not a truncation of it. K_MAX is a schema "
                          f"width -- datasets recorded at different K_MAX cannot "
                          f"be merged.", flush=True)
            order = select_candidate_indices(scores)
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
                          time_budget=0.0, pool_stage=None, seed0=0):
    """Play `n_episodes` with search on and return the labelled rows.

    `pool_stage=None` keeps the original mirror-vs-heuristic distribution
    bit-identical. A curriculum rung switches to the training distribution
    (UtilityTeacher at that rung, round-robin over the deck pool); see
    `PoolTeacherEnv`.
    """
    verify_cell_roundtrip(net)
    pool = None
    if pool_stage is not None:
        from python_ai.opponents import deck_pool
        # Round-robin, not PFSP: a labelling pass should cover the pool, not
        # the decks the policy loses to.
        pool = deck_pool.load_pool()
        print(f"  collecting on the TRAINING distribution: teacher rung "
              f"{pool_stage}, {len(pool)} pool decks round-robin")
    acc = {k: [] for k in ("obs", "card", "cell", "episode", "greedy_card", "greedy_cell",
                           "cand_card", "cand_cell", "cand_value", "cand_n")}
    scores, dev_total, dev_steps = [], 0, 0
    started = time.perf_counter()

    for ep in range(n_episodes):
        if time_budget and (time.perf_counter() - started) > time_budget:
            print(f"  [time budget reached after {ep} episodes]")
            break
        if pool is None:
            env = make_env(opp_elixir, max_ticks)
        else:
            env = make_pool_env(pool_stage, max_ticks, seed0 + ep,
                                pool[ep % len(pool)])
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
    """Concatenate label files, offsetting episode ids so they stay distinct.

    For DAgger, which trains on every earlier round's states too. Episode ids
    delimit LSTM replays, so ids colliding across files would splice two
    trajectories.
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
