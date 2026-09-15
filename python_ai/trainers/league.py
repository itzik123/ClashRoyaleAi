"""The PFSP opponent pool and the fixed Elo roster it is measured against.

Two different jobs that both read `historical_checkpoints/`, kept together
because they share the eligibility rule:

  the POOL      what PFSP samples from during training. Grows forever.
  the ROSTER    a FIXED set of anchors the trainee is scored against every
                EVAL_INTERVAL_EPISODES. Entries are never reordered or replaced
                -- an Elo TREND only means anything against a roster that does
                not move under it.
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

# Pipeline #2's own snapshots (saved every HISTORICAL_CHECKPOINT_INTERVAL_EPISODES
# = 5000 episodes, see train.py) refill the SAME shared pool this script draws
# opponents from -- once the genuinely-old/weak pipeline #1 snapshots are used
# up, every remaining candidate is one of the trainee's own very recent
# selves. discover_historical_checkpoints() excludes pipeline #2 snapshots
# younger than this many episodes (relative to the live trainee's CURRENT
# episodes_completed) from eligibility, so the pool -- and therefore both PFSP
# sampling and the evaluation roster below -- always reflects genuinely-older,
# genuinely-distinguishable versions instead of coin-flip-parity mirrors.
# Pipeline #1 snapshots are always eligible (wholly separate, earlier, much
# weaker phase -- see discover_historical_checkpoints()'s docstring for why
# their episode count isn't even comparable to this one).
#
# Lowered 15000 -> 6000 on 2026-08-09, together with
# HISTORICAL_CHECKPOINT_INTERVAL_EPISODES (5000 -> 2000). These two are COUPLED
# and must move together: this gate has always been exactly 3x the snapshot
# interval, i.e. "exclude the three newest snapshots". Leaving it at 15000 while
# the interval dropped to 2000 would have silently turned it into "exclude the
# seven newest" -- a materially more conservative pool than was ever intended,
# arrived at by changing a constant nobody edited.
#
# The re-denomination points the same way on its own merits. Since the
# 2026-08-07 speed fix an episode carries ~2.2x more policy change, so 15,000
# episodes now represents MORE divergence than when that number was picked --
# the gate was becoming over-conservative, not under. 15000 / 2.2 ~= 6,800, and
# 3 x 2000 = 6,000; both say the same thing.
MIN_OPPONENT_AGE_EPISODES = 6000

# --- Fixed-roster Elo-style evaluation ---
# Win rate against "whichever opponent PFSP happened to sample this window"
# is not a stable progress signal (it swings with the pool composition, same
# root problem the ladder had). This periodically plays the CURRENT policy,
# GREEDY (argmax card, distribution mean placement -- no sampling, so the
# measurement isn't itself noisy from exploration), against a small FIXED
# roster of reference opponents that, once chosen, never changes -- so the
# resulting score is comparable across the whole run and actually answers
# "is this getting better" instead of "did this window's sampled mix happen
# to be easy or hard."
EVAL_INTERVAL_EPISODES = 5000

# Raised 10 -> 50. At 10 games the Elo readout was unusable. Elo is derived via
# 400*log10(1/score - 1), which explodes near a score of 1.0: flipping ONE game
# on an anchor sitting at 0.9 moved that anchor's implied Elo by 141 points and
# the reported average by ~28. Measured across 8 evaluations spanning 35,000
# episodes: mean 1755, std 53, range 164 -- no resolvable trend, so the metric
# could neither confirm nor rule out improvement. 50 games cuts the per-anchor
# standard error roughly in half.
EVAL_GAMES_PER_OPPONENT = 50

# Reference roster grows (see update_reference_roster()) as new age-eligible
# checkpoints appear, capped here -- once full, it's permanently frozen so
# the Elo scale stays comparable for the rest of the run.
REFERENCE_ROSTER_MAX_SIZE = 6

# PERMANENT, non-self-referential anchors: the C++ HeuristicOpponent at three
# elixir multipliers.
#
# Why these have to exist. Every anchor used to be a snapshot of the trainee's
# own past, which makes the whole Elo scale self-referential -- it measures
# "better than I used to be", not "good at the game". That failure was measured
# twice over: 4 of 5 historical anchors sat pinned at a score of 0.90-1.00 for
# 8 straight evaluations (no resolution left), and separately the heuristic
# opponent -- roughly 100 lines of hand-written rules -- beat a policy trained
# for 47,000 episodes 71% of the time. A scale built only from self-copies
# cannot see either problem.
#
# Multipliers chosen from measurement, not guessed: the current policy scores
# 100% vs 1.0x and 97% vs 1.2x, so 1.0x is already saturated and only useful as
# a floor/regression tripwire. 1.35x and 1.5x are where there is real
# resolution. Elos are hand-assigned and ordered by measured difficulty -- as
# with the historical anchors these are a comparable TREND, not a calibrated
# rating (see REFERENCE_ROSTER_ELO_STEP).
#
# The "builtin:" prefix is what evaluate_against_roster dispatches on; it is
# never a real file path, so torch.load is never attempted on it.
BUILTIN_ANCHORS = [
    ("builtin:heuristic@1.00", 1200),
    ("builtin:heuristic@1.35", 1500),
    ("builtin:heuristic@1.50", 1700),
]

# Anchor Elo values are HAND-ASSIGNED, evenly spaced by pool position at the
# time each anchor is added -- NOT empirically cross-calibrated against each
# other. This means the absolute number this produces is not a "real"
# competitive Elo rating; what's meaningful is the TREND across successive
# evaluation rounds against this exact fixed roster (is Eval/Elo trending up
# over tens of thousands of episodes), which is exactly the signal that was
# missing before.
REFERENCE_ROSTER_BASE_ELO = 1000

REFERENCE_ROSTER_ELO_STEP = 150

def discover_historical_checkpoints(current_episode=None,
                                    directory=None, since=None):
    """All ELIGIBLE *.pth files in HISTORICAL_CHECKPOINT_DIR, oldest-saved-first
    (mtime). Save order is the ordering signal, not filenames -- pipeline #1
    and this same script both drop snapshots into this one shared folder, on
    two unrelated episode-count scales, so parsing/comparing episode numbers
    out of the filename wouldn't give a meaningful weakest->strongest order.

    When current_episode is given, pipeline #2's OWN snapshots (filenames
    embed THIS script's own episodes_completed at save time -- same scale as
    current_episode, unlike pipeline #1's) younger than
    MIN_OPPONENT_AGE_EPISODES are excluded from eligibility -- see that
    constant's comment for why. Pipeline #1 snapshots are always eligible
    regardless of current_episode (their episode count lives on a wholly
    different, incomparable scale, and they're always from an earlier, weaker
    phase anyway).

    `directory` exists so a test can point at a tmp_path EXPLICITLY. Until
    2026-08-25 the pool directory was cwd-relative and tests isolated
    themselves with `monkeypatch.chdir`; anchoring it (so a run cannot be
    redirected by the directory it was launched from) took that away, and
    without an explicit override the suite would write random-init snapshots
    into the real PFSP pool -- i.e. silently hand the league eight untrained
    opponents. Mirrors `save_historical_snapshot`'s existing `directory=`.
    """
    directory = HISTORICAL_CHECKPOINT_DIR if directory is None else directory
    paths = glob.glob(os.path.join(directory, "*.pth"))
    # THIS LINEAGE ONLY. `since` is when the run's phase 1 started from scratch
    # (BaseTrainer.lineage_started_at). The directory is shared across runs, and
    # on 2026-09-15 it held 51 snapshots from the previous lineage: a fresh run's
    # phase 2 would have built its PFSP pool and its Elo roster from a different
    # deck's policies without saying so (audit 08). None/0 disables the filter,
    # which is what a legacy checkpoint without the stamp gets.
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
    """Grows the FIXED evaluation roster (in place), up to
    REFERENCE_ROSTER_MAX_SIZE. Entries already in the roster are NEVER
    reordered or replaced -- see REFERENCE_ROSTER_ELO_STEP's comment on why the
    roster has to stay fixed for the Elo trend to mean anything.

    Anchors are now sampled EVENLY ACROSS the pool (weakest to strongest)
    instead of taking the first N oldest. The old behaviour filled the roster
    with the weakest checkpoints available at the time, and measurement showed
    what that costs: across 8 evaluations, 4 of the 5 anchors sat permanently at
    a score of 0.90-1.00. A saturated anchor contributes no information about
    whether the trainee improved -- it can only ever move DOWN, and near score
    1.0 the Elo formula turns a single lost game into a ~222-point swing. The
    roster was simultaneously blind to progress and extremely noisy.

    Spreading the picks means the roster spans a real difficulty range, so at
    least some anchors sit in the informative 0.3-0.7 band where score changes
    actually track skill."""
    known_paths = {entry["path"] for entry in reference_roster}
    # Built-in anchors go in FIRST and stay forever -- they are the only part of
    # the scale that is not a copy of the trainee, and they never saturate away
    # the way a beaten historical snapshot does.
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
    # Evenly spaced indices across the pool, oldest(weakest) -> newest(strongest).
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
    """Plays the CURRENT live policy, GREEDY (argmax masked card, argmax
    placement cell -- no sampling), against each fixed roster opponent for
    n_games each. No coordinate bounds are needed anymore: the discrete
    placement head emits a board-cell index that MicroRoyaleNet.cell_to_xy()
    converts to in-bounds coordinates by construction.

    The opponent itself is NOT forced greedy -- it plays its own normal
    (stochastic) game, since the point is "how does the trainee do against
    this opponent's real behavior," and only the trainee's OWN sampling noise
    needs removing from the measurement.

    Returns (agent_elo, per_opponent_stats). agent_elo is the average of the
    trainee's Elo implied by its score against each anchor's fixed Elo (see
    REFERENCE_ROSTER_ELO_STEP's comment: a comparable trend, not a calibrated
    rating). per_opponent_stats maps checkpoint path -> {wins, losses, draws,
    score}, useful for per-opponent diagnostics beyond the single aggregate
    number.
    """
    net.eval()
    per_opponent_stats = {}
    implied_elos = []
    try:
        for entry in roster:
            path, ref_elo = entry["path"], entry["elo"]
            # Two kinds of anchor need two different envs, and the distinction is
            # load-bearing: MicroRoyaleSelfPlayEnv drives team 1 through
            # ClashEnv::stepSelfPlay, which deliberately does NOT call
            # opponentTurn() -- so the C++ HeuristicOpponent never runs there.
            # Evaluating against it requires the ordinary MicroRoyaleEnv, whose
            # step() goes through game.step() and therefore does.
            if path.startswith("builtin:"):
                multiplier = float(path.split("@")[1])
                env = gym_wrapper.MicroRoyaleEnv()
                env.set_opponent_elixir_multiplier(multiplier)
            else:
                # Scenarios OFF for evaluation: Elo must measure clean-game
                # strength, not defense of an injected handicap.
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
                        # argmax over MASKED logits -- an unaffordable card can
                        # never be the greedy pick, so the measurement reflects
                        # the policy best LEGAL move rather than a play the
                        # engine would silently drop.
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
