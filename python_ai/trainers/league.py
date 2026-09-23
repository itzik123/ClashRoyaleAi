"""The PFSP opponent pool and the fixed Elo roster it is measured against.

  the pool      what PFSP samples from during training; grows forever
  the roster    a fixed set of anchors scored against every
                EVAL_INTERVAL_EPISODES; entries are never reordered or
                replaced, since an Elo trend needs a roster that does not move
"""
import glob
import math
import os
import re

import numpy as np
import torch

from python_ai.envs import gym_wrapper
from python_ai.envs.selfplay_env import MicroRoyaleSelfPlayEnv
from python_ai.models.policy_io import LSTM_HIDDEN
from python_ai.rl.checkpointing import HISTORICAL_CHECKPOINT_DIR

# Pipeline 2's own snapshots younger than this (relative to the live trainee)
# are ineligible, so the pool is not filled with coin-flip mirrors of the
# current agent. Pipeline 1 snapshots are always eligible. Kept at 3x
# HISTORICAL_CHECKPOINT_INTERVAL_EPISODES ("exclude the three newest"); change
# the two together.
MIN_OPPONENT_AGE_EPISODES = 6000

# Fixed-roster evaluation: the greedy policy against anchors that never change,
# so the score is comparable across the run. Win rate against the sampled pool
# moves with the pool.
EVAL_INTERVAL_EPISODES = 5000

# Games per anchor. Elo is 400*log10(1/score - 1), which explodes near a score
# of 1.0, so few games make it noise.
EVAL_GAMES_PER_OPPONENT = 50

# The roster fills up to this as eligible snapshots appear, then freezes.
REFERENCE_ROSTER_MAX_SIZE = 6

# Permanent anchors that are not copies of the trainee: the C++ heuristic at
# three elixir multipliers. A roster of only past selves measures "better than
# I was", not "good at the game". 1.0x is saturated and serves as a regression
# tripwire; 1.35x and 1.5x carry the resolution. The "builtin:" prefix is a
# dispatch tag, never a file path.
BUILTIN_ANCHORS = [
    ("builtin:heuristic@1.00", 1200),
    ("builtin:heuristic@1.35", 1500),
    ("builtin:heuristic@1.50", 1700),
]

# Anchor Elos are hand-assigned by pool position, not calibrated against each
# other: the trend across evaluations is meaningful, the absolute number is
# not.
REFERENCE_ROSTER_BASE_ELO = 1000

REFERENCE_ROSTER_ELO_STEP = 150

def discover_historical_checkpoints(current_episode=None,
                                    directory=None, since=None):
    """All eligible *.pth files in the pool directory, oldest first by mtime.

    Save order, not filenames: both pipelines write here on unrelated episode
    scales. With `current_episode`, pipeline 2's own snapshots younger than
    MIN_OPPONENT_AGE_EPISODES are excluded. `since` restricts to this lineage.
    `directory` lets a test point at a tmp dir; the real one is anchored, not
    cwd-relative.
    """
    directory = HISTORICAL_CHECKPOINT_DIR if directory is None else directory
    paths = glob.glob(os.path.join(directory, "*.pth"))
    # This lineage only (`since` is when its phase 1 started): the directory is
    # shared across runs, and a fresh run would otherwise build its pool and
    # roster from another run's policies. None/0 disables the filter (legacy
    # checkpoints).
    if since:
        paths = [p for p in paths if os.path.getmtime(p) >= float(since)]
    if current_episode is not None:
        eligible = []
        for p in paths:
            match = re.search(r"_pipeline2_ep(\d+)\.pth$", os.path.basename(p))
            if match and current_episode - int(match.group(1)) < MIN_OPPONENT_AGE_EPISODES:
                continue
            eligible.append(p)
        paths = eligible
    return sorted(paths, key=os.path.getmtime)

def update_reference_roster(reference_roster, historical_pool):
    """Grow the fixed evaluation roster in place, up to REFERENCE_ROSTER_MAX_SIZE;
    existing entries never change.

    New anchors are spread evenly across the pool rather than taking the
    oldest, so some sit in the informative 0.3-0.7 score band instead of
    saturating near 1.0, where one lost game swings the Elo by hundreds of
    points.
    """
    known_paths = {entry["path"] for entry in reference_roster}
    # Built-in anchors first, permanently.
    for descriptor, elo in BUILTIN_ANCHORS:
        if descriptor not in known_paths:
            reference_roster.append({"path": descriptor, "elo": elo})
            known_paths.add(descriptor)
    room = REFERENCE_ROSTER_MAX_SIZE - len(reference_roster)
    if room <= 0:
        return reference_roster
    candidates = [p for p in historical_pool if p not in known_paths]
    if not candidates:
        return reference_roster
    # Evenly spaced across the pool, oldest (weakest) to newest (strongest).
    n = min(room, len(candidates))
    picks = [candidates[round(i * (len(candidates) - 1) / max(1, n - 1))] for i in range(n)] if n > 1 \
        else [candidates[-1]]
    for path in dict.fromkeys(picks):          # dedupe, preserve order
        if len(reference_roster) >= REFERENCE_ROSTER_MAX_SIZE:
            break
        next_elo = REFERENCE_ROSTER_BASE_ELO + REFERENCE_ROSTER_ELO_STEP * len(reference_roster)
        reference_roster.append({"path": path, "elo": next_elo})
        known_paths.add(path)
    return reference_roster

def evaluate_against_roster(net, device, roster, n_games=10):
    """Play the current policy greedily (argmax masked card and cell) against each
    roster opponent for n_games each; the opponent plays normally.

    Returns (agent_elo, per_opponent_stats): agent_elo averages the Elo implied
    by the score against each anchor's fixed Elo (a trend, not a calibrated
    rating); per_opponent_stats maps path -> {wins, losses, draws, score}.
    """
    net.eval()
    per_opponent_stats = {}
    implied_elos = []
    try:
        for entry in roster:
            path, ref_elo = entry["path"], entry["elo"]
            # A builtin anchor needs MicroRoyaleEnv: stepSelfPlay never calls
            # opponentTurn(), so the C++ heuristic would not run in the
            # self-play env.
            if path.startswith("builtin:"):
                multiplier = float(path.split("@")[1])
                env = gym_wrapper.MicroRoyaleEnv()
                env.set_opponent_elixir_multiplier(multiplier)
            else:
                # Scenarios off: Elo measures clean-game strength.
                env = MicroRoyaleSelfPlayEnv({"scenarios_enabled": False})
                env.set_historical_opponent(path)
            wins = losses = draws = 0
            for _ in range(n_games):
                obs, _ = env.reset()
                hx = torch.zeros(1, LSTM_HIDDEN).to(device)
                cx = torch.zeros(1, LSTM_HIDDEN).to(device)
                done = False
                reward = 0.0
                while not done:
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        card_mask = net.affordability_mask(obs_t)
                        features, card_embeds, spatial_map = net.extract_features(obs_t)
                        (card_logits, _, _, _,
                         (hx, cx)) = net.step_lstm_and_card(features, (hx, cx), card_mask)
                        # argmax over masked logits, so the greedy pick is
                        # always a legal move.
                        card_idx_t = card_logits.argmax(dim=-1)
                        place_logits = net.placement_given_card(hx, card_embeds, card_idx_t, obs_t, spatial_map)
                        cell = place_logits.argmax(dim=-1)
                        x_t, y_t = net.cell_to_xy(cell)
                    card_idx = int(card_idx_t.item())
                    action = {
                        "card_index": np.array([card_idx]),
                        "target_x": np.array([x_t.item()]),
                        "target_y": np.array([y_t.item()]),
                        "activate_ability_slot1": np.array([0]),
                        "activate_ability_slot2": np.array([0]),
                    }
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                if reward > 0.5:
                    wins += 1
                elif reward < -0.5:
                    losses += 1
                else:
                    draws += 1
            score = (wins + 0.5 * draws) / n_games
            per_opponent_stats[path] = {"wins": wins, "losses": losses, "draws": draws, "score": score}
            score_clipped = min(max(score, 0.03), 0.97)
            implied_elos.append(ref_elo - 400.0 * math.log10(1.0 / score_clipped - 1.0))
    finally:
        net.train()
    agent_elo = float(np.mean(implied_elos)) if implied_elos else None
    return agent_elo, per_opponent_stats
