import os
import glob
import re
import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from torch.distributions import Categorical, Normal
from torch.utils.tensorboard import SummaryWriter
from collections import deque

import clash_royale_env
from gym_wrapper import DEFAULT_DECK
from model import MicroRoyaleNet
from train import (
    compute_shaping, building_hp_end, annotate_replay_with_agent_info,
    HISTORICAL_CHECKPOINT_DIR, HISTORICAL_CHECKPOINT_INTERVAL_EPISODES,
    DRAW_PENALTY, load_state_dict_flexible,
)

# --- Pipeline #2: Historical Self-Play (League / PFSP) ---
# Pipeline #1 (train.py) teaches the bot to beat a random-but-unskilled
# opponent -- a necessary bootstrap, but not a ceiling: nothing in that
# opponent ever punishes a bad trade, reads a push, or plays around a specific
# threat the way a real opponent would. This pipeline is what actually
# introduces that pressure: the trainee (still the same live policy, continuing
# gradient updates) plays against FROZEN snapshots of its own past selves.
# Both sides genuinely decide what to play (see ClashEnv::stepSelfPlay/
# extractObservationForTeam in the C++ layer) instead of team 1 being the
# built-in random C++ bot.
#
# Opponent selection is Prioritized Fictitious Self-Play (PFSP, as used by
# AlphaStar's league), NOT a linear ladder -- replaces an earlier design that
# advanced through the pool one opponent at a time, gated on a win-rate
# threshold. That design had two structural problems, both confirmed in
# practice (see training_selfplay_run1.log from this project's own history):
# (1) once the pool's genuinely-old/weak snapshots were exhausted, the "next"
# opponent was inevitably a near-mirror of the live trainee (the pool refills
# from the trainee's OWN recent snapshots), so the win-rate gate became
# structurally unreachable -- not because the trainee stopped improving, but
# because a near-mirror matchup settles near 50% by construction; (2) a
# linear ladder never revisits opponents once passed, so nothing prevents
# catastrophic forgetting of earlier matchups. PFSP fixes both: EVERY env,
# at EVERY episode reset, independently samples an opponent from the whole
# eligible pool with probability weighted toward whichever opponents it's
# currently doing WORST against (see PFSP_EXPONENT below) -- so old,
# already-beaten opponents keep appearing (just rarely), and there is no
# single "current opponent" or gate to get stuck against.

# Each of the num_envs worker processes keeps its OWN local per-opponent
# win-rate estimate (see MicroRoyaleSelfPlayEnv.pfsp_stats) and samples from
# it independently -- deliberately decentralized rather than synchronized
# across workers/processes, since AsyncVectorEnv workers are separate OS
# processes and cross-process synchronization of a growing stats dict every
# episode would add real complexity for little benefit at only 8 workers;
# each worker converges its own reasonable local estimate from its own
# experience over time, same as any decentralized-actor RL setup.
PFSP_EXPONENT = 2.0
# Floor on sampling weight even for an opponent the trainee already wins
# 100% against -- keeps EVERY eligible opponent in rotation forever (at low
# frequency) instead of fully forgetting it once its local win-rate estimate
# saturates. This is what actually prevents catastrophic forgetting; PFSP's
# difficulty-weighting alone would drive an already-mastered opponent's
# sampling weight toward (but never quite to) zero, so this floor gives it a
# guaranteed non-zero share.
PFSP_MIN_WEIGHT = 0.05
# Smoothing factor for each worker's local per-opponent win-rate EMA
# (outcome in {1.0 win, 0.5 draw, 0.0 loss}, new_stat = old*(1-a) + outcome*a).
# ~1/0.08 = 12-game effective window -- short enough to track a real trend as
# both the trainee and the sampling weights shift over a long run, without
# being so short that one lucky/unlucky game swings the weight.
PFSP_EMA_ALPHA = 0.08

# Same escape hatch as before for mutual passivity: once draws cross this
# share of the current 100-episode window (now aggregated across whatever mix
# of pool opponents each worker happened to sample, not one fixed opponent),
# re-boost exploration -- floored entropy is exactly what locks a passive
# equilibrium in place, since neither side has any remaining chance to
# stumble into a different joint strategy. Reward-shaping fixes (see
# train.py's W_ELIXIR_TRADE/DRAW_PENALTY/W_ELIXIR_OVERFLOW comments) address
# WHY passivity looked attractive; this addresses the fact that once both
# sides are already sitting in it, ordinary gradient descent has no
# exploration left to climb back out.
STALL_DRAW_RATE_THRESHOLD = 0.5
# Cooldown so this doesn't re-fire every single episode once the window is
# saturated with draws -- gives each boost a real window to actually take
# effect before deciding whether another one is needed.
STALL_REBOOST_COOLDOWN_EPISODES = 1000
# Escape hatch for the failure mode the draw-rate trigger above CAN'T see:
# draws near zero but the aggregate win rate across the pool still stuck
# losing, entropy floored the whole time with nothing ever refreshing it.
# Narrowed to also require win_rate < 0.5 (not just "below some target"):
# confirmed in practice (this project's own earlier ladder-based run) that
# firing an exploration re-boost while ALREADY decisively winning measurably
# HURTS -- there's no better joint policy for injected noise to find in a
# matchup that's already going well, only sampling noise for the entropy
# bonus to add. Below 0.5 the trainee is genuinely losing more than winning
# against the pool it's currently being sampled against, which IS worth
# reacting to.
ENTROPY_STALE_REBOOST_EPISODES = 1500
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
MIN_OPPONENT_AGE_EPISODES = 15000

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
EVAL_GAMES_PER_OPPONENT = 10
# Reference roster grows (see update_reference_roster()) as new age-eligible
# checkpoints appear, capped here -- once full, it's permanently frozen so
# the Elo scale stays comparable for the rest of the run.
REFERENCE_ROSTER_MAX_SIZE = 5
# Anchor Elo values are HAND-ASSIGNED, evenly spaced by pool position at the
# time each anchor is added -- NOT empirically cross-calibrated against each
# other. This means the absolute number this produces is not a "real"
# competitive Elo rating; what's meaningful is the TREND across successive
# evaluation rounds against this exact fixed roster (is Eval/Elo trending up
# over tens of thousands of episodes), which is exactly the signal that was
# missing before.
REFERENCE_ROSTER_BASE_ELO = 1000
REFERENCE_ROSTER_ELO_STEP = 150

WEIGHT_PATH = "model_weights_selfplay.pth"
# Pipeline #1's final artifact -- read ONCE, only to seed a from-scratch
# pipeline #2 run (bare weights only; pipeline #2 keeps its own separate
# episode count/optimizer state in WEIGHT_PATH from then on, so pipeline #1's
# own checkpoint is never overwritten by this script).
BOOTSTRAP_FROM_PATH = "model_weights.pth"


def discover_historical_checkpoints(current_episode=None):
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
    phase anyway)."""
    paths = glob.glob(os.path.join(HISTORICAL_CHECKPOINT_DIR, "*.pth"))
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
    """Grows the FIXED evaluation roster (in place) as new age-eligible
    checkpoints become available in historical_pool, up to
    REFERENCE_ROSTER_MAX_SIZE. Entries already in the roster are NEVER
    reordered or replaced -- see REFERENCE_ROSTER_ELO_STEP's comment on why
    the roster has to stay fixed for the Elo trend to mean anything. Iterates
    historical_pool oldest-first (its own sort order), so anchors get added
    weakest-first as the pool grows, same "weakest known so far" assumption
    used everywhere else in this file."""
    known_paths = {entry["path"] for entry in reference_roster}
    for path in historical_pool:
        if len(reference_roster) >= REFERENCE_ROSTER_MAX_SIZE:
            break
        if path in known_paths:
            continue
        next_elo = REFERENCE_ROSTER_BASE_ELO + REFERENCE_ROSTER_ELO_STEP * len(reference_roster)
        reference_roster.append({"path": path, "elo": next_elo})
        known_paths.add(path)
    return reference_roster


def evaluate_against_roster(net, device, roster, max_x, max_y, n_games=10):
    """Plays the CURRENT live policy, GREEDY (argmax card, distribution mean
    placement -- no sampling), against each fixed roster opponent for n_games
    each. The opponent itself is NOT forced greedy -- it plays its own normal
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
            env = MicroRoyaleSelfPlayEnv()
            env.set_historical_opponent(path)
            wins = losses = draws = 0
            for _ in range(n_games):
                obs, _ = env.reset()
                hx = torch.zeros(1, 256).to(device)
                cx = torch.zeros(1, 256).to(device)
                done = False
                reward = 0.0
                while not done:
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        features, card_embeds = net.extract_features(obs_t)
                        (card_logits, ability_slot1_logits, ability_slot2_logits, _,
                         (hx, cx)) = net.step_lstm_and_card(features, (hx, cx))
                        card_idx_t = card_logits.argmax(dim=-1)
                        mean, _ = net.placement_given_card(hx, card_embeds, card_idx_t)
                    card_idx = int(card_idx_t.item())
                    placement = torch.clamp(mean, 0.0, 1.0)
                    action = {
                        "card_index": np.array([card_idx]),
                        "target_x": np.array([(placement[0, 0] * max_x).item()]),
                        "target_y": np.array([(placement[0, 1] * max_y).item()]),
                        # Greedy, same as card/placement above -- no sampling.
                        "activate_ability_slot1": np.array([int(ability_slot1_logits.argmax(dim=-1).item())]),
                        "activate_ability_slot2": np.array([int(ability_slot2_logits.argmax(dim=-1).item())]),
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


class MicroRoyaleSelfPlayEnv(gym.Env):
    """Same action/observation shape as gym_wrapper.MicroRoyaleEnv, but team 1
    is a frozen copy of MicroRoyaleNet (never trained here -- eval()/no_grad
    only) instead of the built-in random C++ opponentTurn().

    Opponent selection is PFSP (see this module's own top-of-file comment):
    each instance keeps its own local pool (pfsp_pool) and per-opponent
    win-rate EMA (pfsp_stats), and samples a fresh opponent at every reset()
    weighted toward whichever it's currently doing worst against. The main
    process broadcasts pool updates via refresh_pfsp_pool(); set_historical_
    opponent() is still available directly for one-off uses (evaluation,
    replay generation) that want a SPECIFIC opponent rather than a sampled
    one.
    """

    def __init__(self, env_config=None):
        super().__init__()
        env_config = env_config or {}
        self.deck = env_config.get("deck", list(DEFAULT_DECK))
        max_ticks = env_config.get("max_ticks", 3600)
        # Tower Troops: per-match config, not a per-step action -- see
        # gym_wrapper.py's identical wiring. NONE (the default) reproduces
        # the original hardcoded Princess Tower unchanged.
        ai_tower_troop = env_config.get("ai_tower_troop", clash_royale_env.TowerTroopType.NONE)
        opp_tower_troop = env_config.get("opp_tower_troop", clash_royale_env.TowerTroopType.NONE)
        self.game = clash_royale_env.ClashRoyaleEnv(self.deck, self.deck, max_ticks, ai_tower_troop, opp_tower_troop)
        # Instance attributes (not class-level constants) -- pulled live from
        # the engine's own enforced placement bounds instead of a hardcoded
        # copy that could silently drift if board geometry ever changes.
        self.MAX_X = self.game.get_max_placement_x()
        self.MAX_Y = self.game.get_own_half_max_y()

        # Team 1's brain -- CPU is plenty for a single inference-only forward
        # pass per step per worker process, and keeps this off the GPU the
        # trainee's own updates use in the main process.
        self.device = torch.device("cpu")
        self.opponent_net = MicroRoyaleNet().to(self.device)
        self.opponent_net.eval()
        self.opponent_hx = torch.zeros(1, 256).to(self.device)
        self.opponent_cx = torch.zeros(1, 256).to(self.device)
        self.opponent_checkpoint_path = None

        # PFSP pool/stats -- see refresh_pfsp_pool()/_sample_pfsp_opponent().
        # Empty until the main process's first broadcast; reset() no-ops the
        # sampling step until then (should never actually happen in normal
        # operation, since train_selfplay_ppo() broadcasts before the first
        # envs.reset() call).
        self.pfsp_pool = []
        self.pfsp_stats = {}

        if env_config.get("historical_checkpoint_path"):
            self.set_historical_opponent(env_config["historical_checkpoint_path"])

        self.action_space = spaces.Dict({
            "card_index": spaces.Discrete(clash_royale_env.ClashRoyaleEnv.HAND_SIZE + 1),
            "target_x": spaces.Box(low=0.0, high=self.MAX_X, shape=(1,), dtype=np.float32),
            "target_y": spaces.Box(low=0.0, high=self.MAX_Y, shape=(1,), dtype=np.float32),
            # Same keys as gym_wrapper.MicroRoyaleEnv's action_space -- see
            # its own comment. Team 1 (the frozen historical opponent) samples
            # its own independent pair via _opponent_action(), so both sides
            # can actually use Champion abilities during self-play.
            "activate_ability_slot1": spaces.Discrete(2),
            "activate_ability_slot2": spaces.Discrete(2),
        })
        obs_size = self.game.observation_size()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)

    def set_historical_opponent(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        load_state_dict_flexible(self.opponent_net, state_dict, f"historical opponent {checkpoint_path}")
        self.opponent_net.eval()
        self.opponent_checkpoint_path = checkpoint_path

    def refresh_pfsp_pool(self, pool_paths):
        """Broadcast from the main process (train_selfplay_ppo, via
        envs.call) whenever the eligible pool changes -- at startup, and
        after every new historical snapshot is saved. New entries start at a
        neutral 0.5 win-rate prior; PFSP_EXPONENT's weighting naturally
        prioritizes them for extra games until their real difficulty is
        established. Pool only ever grows (an eligible checkpoint never
        becomes ineligible again), so this never needs to prune pfsp_stats."""
        self.pfsp_pool = list(pool_paths)
        for p in self.pfsp_pool:
            if p not in self.pfsp_stats:
                self.pfsp_stats[p] = 0.5

    def _sample_pfsp_opponent(self):
        if not self.pfsp_pool:
            return
        weights = np.array([
            max(PFSP_MIN_WEIGHT, (1.0 - self.pfsp_stats.get(p, 0.5)) ** PFSP_EXPONENT)
            for p in self.pfsp_pool
        ], dtype=np.float64)
        weights /= weights.sum()
        chosen = self.pfsp_pool[np.random.choice(len(self.pfsp_pool), p=weights)]
        self.set_historical_opponent(chosen)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._sample_pfsp_opponent()
        obs_list = self.game.reset()
        self.opponent_hx = torch.zeros(1, 256).to(self.device)
        self.opponent_cx = torch.zeros(1, 256).to(self.device)
        return np.array(obs_list, dtype=np.float32), {}

    def _opponent_action(self):
        """Team 1's own decision, from ITS OWN (mirrored) point of view --
        see ClashEnv::extractObservationForTeam. Sampled the same way rollout
        actions are everywhere else in this project, just with no_grad and no
        buffering: this network never gets updated here. Also samples its own
        two independent Champion-ability decisions (slot1/slot2) -- otherwise
        team 1 would systematically never use Champion abilities even once
        the trainee (team 0) does."""
        obs1 = np.array(self.game.get_observation_for_team(1), dtype=np.float32)
        with torch.no_grad():
            obs1_t = torch.tensor(obs1, dtype=torch.float32).unsqueeze(0).to(self.device)
            features1, card_embeds1 = self.opponent_net.extract_features(obs1_t)
            (logits1, ability_slot1_logits1, ability_slot2_logits1, _,
             (self.opponent_hx, self.opponent_cx)) = self.opponent_net.step_lstm_and_card(
                features1, (self.opponent_hx, self.opponent_cx))
            card_idx1_t = Categorical(logits=logits1).sample()
            mean1, log_std1 = self.opponent_net.placement_given_card(self.opponent_hx, card_embeds1, card_idx1_t)
            card_idx1 = card_idx1_t.item()
            placement1 = torch.clamp(Normal(mean1, log_std1.exp()).sample(), 0.0, 1.0)
            x1 = (placement1[0, 0] * self.MAX_X).item()
            y1 = (placement1[0, 1] * self.MAX_Y).item()
            ability_slot1_action1 = bool(Categorical(logits=ability_slot1_logits1).sample().item())
            ability_slot2_action1 = bool(Categorical(logits=ability_slot2_logits1).sample().item())
        return card_idx1, x1, y1, ability_slot1_action1, ability_slot2_action1

    def step(self, action, skip_frames=10):
        def _to_scalar(val):
            if hasattr(val, "item"):
                return val.item()
            if isinstance(val, (list, tuple, np.ndarray)):
                return val[0]
            return val

        card_idx0 = int(_to_scalar(action["card_index"]))
        x0 = float(_to_scalar(action["target_x"]))
        y0 = float(_to_scalar(action["target_y"]))
        activate_ability0_slot1 = bool(_to_scalar(action.get("activate_ability_slot1", 0)))
        activate_ability0_slot2 = bool(_to_scalar(action.get("activate_ability_slot2", 0)))
        card_idx1, x1, y1, activate_ability1_slot1, activate_ability1_slot2 = self._opponent_action()

        result = self.game.step_self_play(card_idx0, x0, y0, card_idx1, x1, y1, skip_frames,
                                           activate_ability0_slot1, activate_ability0_slot2,
                                           activate_ability1_slot1, activate_ability1_slot2)

        obs = np.array(result.observation0, dtype=np.float32)
        reward = float(result.reward0)
        terminated = bool(result.done)

        if terminated and self.opponent_checkpoint_path is not None:
            if reward > 0.5:
                outcome = 1.0
            elif reward < -0.5:
                outcome = 0.0
            else:
                outcome = 0.5
            prev = self.pfsp_stats.get(self.opponent_checkpoint_path, 0.5)
            self.pfsp_stats[self.opponent_checkpoint_path] = prev * (1.0 - PFSP_EMA_ALPHA) + outcome * PFSP_EMA_ALPHA

        # Same key names/shape as gym_wrapper.MicroRoyaleEnv.step()'s info dict
        # on purpose -- lets compute_shaping() from train.py be reused as-is.
        info = {
            "elixir": self.game.get_elixir(),
            "hand": self.game.get_hand(),
            "team0_troop_damage": self.game.get_troop_damage_dealt(0),
            "team1_troop_damage": self.game.get_troop_damage_dealt(1),
            "team0_building_damage": self.game.get_building_damage_dealt(0),
            "team1_building_damage": self.game.get_building_damage_dealt(1),
            "team0_elixir_spent": self.game.get_elixir_spent(0),
            "team1_elixir_spent": self.game.get_elixir_spent(1),
            "champion_ability_slot1_ready": self.game.is_champion_ability_ready(0, 1),
            "champion_ability_slot2_ready": self.game.is_champion_ability_ready(0, 2),
        }
        return obs, reward, terminated, False, info


def make_env():
    def _init():
        return MicroRoyaleSelfPlayEnv()
    return _init


def train_selfplay_ppo():
    os.makedirs("replays", exist_ok=True)
    os.makedirs(HISTORICAL_CHECKPOINT_DIR, exist_ok=True)
    log_dir = "runs/clash_royale_selfplay"

    # NOTE: historical_pool is discovered further below, only once
    # episodes_completed is resolved (either 0 on a from-scratch run or
    # restored from a resumed checkpoint) -- MIN_OPPONENT_AGE_EPISODES
    # filtering needs that value to know what's "too young" to be eligible.
    num_envs = 8
    print(f"Initializing {num_envs} self-play environments...")
    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(num_envs)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MicroRoyaleNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=3e-4)

    # Same PPO hyperparameters as train.py -- same architecture and algorithm,
    # only the opponent-selection mechanic differs, so there's no reason to
    # re-derive these from scratch.
    gamma = 0.99
    gae_lambda = 0.9
    eps_clip = 0.2
    update_timestep = 500
    ppo_epochs = 2
    num_minibatches = 1
    max_grad_norm = 0.5

    initial_entropy_coef = 0.1
    min_entropy_coef = 0.01
    entropy_decay_rate = 0.995

    # Pulled live from the engine (a throwaway bare env is enough -- these
    # don't depend on which deck is used, and this avoids building a whole
    # extra MicroRoyaleSelfPlayEnv, opponent net included, just to read two
    # floats) instead of a class-level hardcoded copy that could silently
    # drift if board geometry ever changes.
    _dim_probe = clash_royale_env.ClashRoyaleEnv(list(DEFAULT_DECK), list(DEFAULT_DECK), 100)
    MAX_X = _dim_probe.get_max_placement_x()
    MAX_Y = _dim_probe.get_own_half_max_y()
    del _dim_probe

    # Episode at which the entropy decay schedule last reset -- renamed from
    # the old ladder's "stage_start_episode" now that there's no per-opponent
    # "stage" concept under PFSP; only reset by the two re-boost triggers
    # below (draw-rate stall, losing-trend stale-plateau).
    entropy_reboost_episode = 0
    last_stall_reboost_episode = 0  # episodes_completed value at the last entropy re-boost
    last_eval_ep = 0  # episodes_completed value at the last fixed-roster evaluation
    episodes_completed = 0
    reference_roster = []  # list of {"path":, "elo":}, see update_reference_roster()
    outcome_history = deque(maxlen=100)
    outcome_history_long = deque(maxlen=500)

    full_resume = False
    resuming = os.path.exists(WEIGHT_PATH)
    if resuming:
        checkpoint = torch.load(WEIGHT_PATH, map_location=device, weights_only=False)
        clean_load = load_state_dict_flexible(net, checkpoint["model"], f"pipeline2 resume ({WEIGHT_PATH})")
        if clean_load:
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            print("Optimizer state NOT restored (architecture mismatch above) -- "
                  "starting the optimizer fresh; network weights were still warm-started where shapes matched.")
        episodes_completed = checkpoint["episodes_completed"]
        if not clean_load:
            # Part of the network just got reinitialized (architecture change --
            # see load_state_dict_flexible's warm-start above, e.g. right now:
            # placement_head's card-conditioning). Entropy is almost certainly
            # floored this deep into training, so without this the freshly-
            # random part would settle into another near-deterministic
            # "averaged" policy before ever exploring enough to discover it
            # can now behave differently per card -- same reasoning as every
            # curriculum-stage transition elsewhere in this project resetting
            # the entropy clock, just triggered by an architecture change
            # instead of a harder opponent.
            entropy_reboost_episode = episodes_completed
            print(f">>> Architecture changed on resume -- forcing a fresh entropy "
                  f"re-boost from episode {episodes_completed} so the reinitialized "
                  "part actually gets explored, not just re-converged under floored entropy.")
        else:
            # .get() with a fallback to the OLD ladder-era field name, then to
            # episodes_completed: lets a checkpoint saved before this PFSP
            # rewrite still resume sanely instead of KeyError-ing.
            entropy_reboost_episode = checkpoint.get(
                "entropy_reboost_episode", checkpoint.get("stage_start_episode", episodes_completed))
        last_stall_reboost_episode = checkpoint.get("last_stall_reboost_episode", episodes_completed)
        last_eval_ep = checkpoint.get("last_eval_ep", episodes_completed)
        reference_roster = checkpoint.get("reference_roster", [])
        outcome_history = deque(checkpoint["outcome_history"], maxlen=100)
        full_resume = True
        print(f"Resumed pipeline #2 (PFSP) from {WEIGHT_PATH}: episode {episodes_completed}, "
              f"reference roster size {len(reference_roster)}")
    elif os.path.exists(BOOTSTRAP_FROM_PATH):
        bootstrap = torch.load(BOOTSTRAP_FROM_PATH, map_location=device, weights_only=False)
        state_dict = bootstrap["model"] if isinstance(bootstrap, dict) and "model" in bootstrap else bootstrap
        load_state_dict_flexible(net, state_dict, f"pipeline2 bootstrap from pipeline1 ({BOOTSTRAP_FROM_PATH})")
        print(f"Seeded pipeline #2's trainee from pipeline #1's {BOOTSTRAP_FROM_PATH} "
              "(bare weights only -- pipeline #2 keeps its own separate episode count from here).")
    else:
        raise RuntimeError(
            f"Neither {WEIGHT_PATH} nor {BOOTSTRAP_FROM_PATH} exists -- pipeline #2 needs "
            "pipeline #1's finished bot to start from.")

    historical_pool = discover_historical_checkpoints(episodes_completed)
    if not historical_pool:
        raise RuntimeError(
            f"No historical snapshot in {HISTORICAL_CHECKPOINT_DIR}/ is at least "
            f"MIN_OPPONENT_AGE_EPISODES ({MIN_OPPONENT_AGE_EPISODES}) episodes older than "
            f"the current trainee (episode {episodes_completed}) -- pipeline #1 (train.py) "
            "needs to have run long enough to have saved at least one eligible snapshot "
            "(every HISTORICAL_CHECKPOINT_INTERVAL_EPISODES episodes) before pipeline #2 "
            "has anything old enough to play against.")

    envs.call("refresh_pfsp_pool", historical_pool)
    print(f"PFSP pool initialized with {len(historical_pool)} eligible opponent(s).")

    if not full_resume and os.path.exists(log_dir):
        import shutil
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    obs_buffer, card_actions_buffer, placement_actions_buffer = [], [], []
    ability1_actions_buffer, ability2_actions_buffer = [], []
    logprobs_buffer, values_buffer, rewards_buffer = [], [], []
    masks_buffer, valid_buffer = [], []

    reward_history = deque(maxlen=50)
    shaping_history = deque(maxlen=50)
    ep_len_history = deque(maxlen=50)
    ally_bldg_end_history = deque(maxlen=50)
    enemy_bldg_end_history = deque(maxlen=50)

    obs, _ = envs.reset()
    prev_stats = None
    prev_dones = np.zeros(num_envs, dtype=bool)
    hx = torch.zeros(num_envs, 256).to(device)
    cx = torch.zeros(num_envs, 256).to(device)

    last_save_ep = episodes_completed
    last_historical_save_ep = episodes_completed
    last_replay_ep = episodes_completed
    ep_rewards = np.zeros(num_envs)
    ep_shaping = np.zeros(num_envs)
    ep_steps = np.zeros(num_envs, dtype=np.int64)

    print(f"Self-play training started on {num_envs} CPU cores simultaneously!")

    while episodes_completed < 1000000:
        episodes_since_reboost = episodes_completed - entropy_reboost_episode
        current_entropy_coef = max(min_entropy_coef, initial_entropy_coef * (entropy_decay_rate ** episodes_since_reboost))

        hx0 = hx.detach().clone()
        cx0 = cx.detach().clone()

        for step in range(update_timestep):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

            with torch.no_grad():
                # Autoregressive placement: card must actually be SAMPLED
                # before placement can be conditioned on it -- see model.py's
                # own comment on why forward_from_features (card_idx already
                # known) doesn't fit the rollout case. Ability-slot logits
                # only depend on hx, exactly like card_logits/state_value.
                features, card_embeds = net.extract_features(obs_tensor)
                card_logits, ability1_logits, ability2_logits, state_value, (hx, cx) = net.step_lstm_and_card(
                    features, (hx, cx))
                card_dist = Categorical(logits=card_logits)
                card_idx = card_dist.sample()
                placement_mean, placement_log_std = net.placement_given_card(hx, card_embeds, card_idx)
                placement_dist = Normal(placement_mean, placement_log_std.exp())
                placement_sample = placement_dist.sample()
                placement_logprob = placement_dist.log_prob(placement_sample).sum(dim=-1)
                ability1_dist = Categorical(logits=ability1_logits)
                ability1_action = ability1_dist.sample()
                ability2_dist = Categorical(logits=ability2_logits)
                ability2_action = ability2_dist.sample()
                total_logprob = (card_dist.log_prob(card_idx) + placement_logprob
                                 + ability1_dist.log_prob(ability1_action)
                                 + ability2_dist.log_prob(ability2_action))

            placement_clamped = torch.clamp(placement_sample, 0.0, 1.0)
            target_x = placement_clamped[:, 0] * MAX_X
            target_y = placement_clamped[:, 1] * MAX_Y

            action = {
                "card_index": card_idx.cpu().numpy(),
                "target_x": target_x.cpu().numpy().reshape(num_envs, 1),
                "target_y": target_y.cpu().numpy().reshape(num_envs, 1),
                # AsyncVectorEnv's Dict-space iteration requires every
                # action_space key to exist in the dict (it doesn't fall back
                # to a default like MicroRoyaleSelfPlayEnv.step()'s own
                # action.get("activate_ability_slot1", 0) does for a direct,
                # non-vectorized call), so both real sampled arrays must be
                # supplied here rather than omitted.
                "activate_ability_slot1": ability1_action.cpu().numpy(),
                "activate_ability_slot2": ability2_action.cpu().numpy(),
            }

            next_obs, step_rewards, terminateds, truncateds, infos = envs.step(action)
            dones = terminateds | truncateds

            is_draw = dones & (np.abs(step_rewards) < 0.5)
            draw_penalty = DRAW_PENALTY * is_draw.astype(np.float32)

            zeros = np.zeros(num_envs, dtype=np.int64)
            zeros_f = np.zeros(num_envs, dtype=np.float32)
            stats = {
                "team0_troop_damage": infos.get("team0_troop_damage", zeros),
                "team1_troop_damage": infos.get("team1_troop_damage", zeros),
                "team0_building_damage": infos.get("team0_building_damage", zeros),
                "team1_building_damage": infos.get("team1_building_damage", zeros),
                "team0_elixir_spent": infos.get("team0_elixir_spent", zeros_f),
                "team1_elixir_spent": infos.get("team1_elixir_spent", zeros_f),
                "team0_elixir_current": infos.get("elixir", zeros_f),
            }

            shaping = compute_shaping(stats, prev_stats)
            shaping = shaping * (1.0 - prev_dones)
            shaped_rewards = step_rewards + shaping - draw_penalty
            ep_rewards += shaped_rewards
            ep_shaping += shaping
            ep_steps += 1

            valid = torch.tensor(1.0 - prev_dones, dtype=torch.float32).to(device)
            mask = torch.tensor(1.0 - dones, dtype=torch.float32).to(device) * valid

            obs_buffer.append(obs_tensor)
            card_actions_buffer.append(card_idx)
            placement_actions_buffer.append(placement_sample)
            ability1_actions_buffer.append(ability1_action)
            ability2_actions_buffer.append(ability2_action)
            logprobs_buffer.append(total_logprob)
            values_buffer.append(state_value.squeeze(-1))
            rewards_buffer.append(torch.tensor(shaped_rewards, dtype=torch.float32).to(device))
            masks_buffer.append(mask)
            valid_buffer.append(valid)

            mask_tensor = mask.unsqueeze(1)
            hx = hx * mask_tensor
            cx = cx * mask_tensor

            for i, done in enumerate(dones):
                if done:
                    reward_history.append(ep_rewards[i])
                    shaping_history.append(ep_shaping[i])
                    ep_len_history.append(ep_steps[i])
                    ally_end, enemy_end = building_hp_end(next_obs[i])
                    ally_bldg_end_history.append(ally_end)
                    enemy_bldg_end_history.append(enemy_end)
                    ep_rewards[i] = 0
                    ep_shaping[i] = 0
                    ep_steps[i] = 0
                    episodes_completed += 1

                    if step_rewards[i] > 0.5:
                        outcome_value = 1
                    elif step_rewards[i] < -0.5:
                        outcome_value = -1
                    else:
                        outcome_value = 0
                    outcome_history.append(outcome_value)
                    outcome_history_long.append(outcome_value)

                    if episodes_completed % 10 == 0:
                        outcomes = np.array(outcome_history)
                        wins = int((outcomes == 1).sum())
                        losses = int((outcomes == -1).sum())
                        draws = int((outcomes == 0).sum())
                        n = len(outcomes)
                        decided = wins + losses
                        decisive_wr = wins / decided if decided > 0 else 0.0
                        avg_reward = np.mean(reward_history)
                        avg_shaping = np.mean(shaping_history)
                        print(f"Episodes: {episodes_completed} | Avg(50): {avg_reward:.2f} | "
                              f"W/L/D: {wins/n:.2f}/{losses/n:.2f}/{draws/n:.2f} | Decisive: {decisive_wr:.2f} | "
                              f"Pool: {len(historical_pool)} | Entropy: {current_entropy_coef:.4f}")
                        writer.add_scalar("Training/Avg_Reward_50", avg_reward, episodes_completed)
                        writer.add_scalar("Reward/Episode_Shaping_Sum", avg_shaping, episodes_completed)
                        writer.add_scalar("Training/Win_Rate_100", wins / n, episodes_completed)
                        writer.add_scalar("Rates/Loss_100", losses / n, episodes_completed)
                        writer.add_scalar("Rates/Draw_100", draws / n, episodes_completed)
                        writer.add_scalar("Rates/Decisive_Win_100", decisive_wr, episodes_completed)
                        writer.add_scalar("Training/Decided_Count_100", decided, episodes_completed)
                        outcomes_long = np.array(outcome_history_long)
                        wins_long = int((outcomes_long == 1).sum())
                        losses_long = int((outcomes_long == -1).sum())
                        decided_long = wins_long + losses_long
                        if decided_long > 0:
                            writer.add_scalar("Rates/Decisive_Win_500", wins_long / decided_long, episodes_completed)
                        writer.add_scalar("Progress/Episode_Length_50", np.mean(ep_len_history), episodes_completed)
                        writer.add_scalar("Progress/Enemy_Building_HP_End_50", np.mean(enemy_bldg_end_history), episodes_completed)
                        writer.add_scalar("Progress/Ally_Building_HP_End_50", np.mean(ally_bldg_end_history), episodes_completed)
                        writer.add_scalar("Training/PFSP_Pool_Size", len(historical_pool), episodes_completed)
                        writer.add_scalar("Training/Entropy_Coef", current_entropy_coef, episodes_completed)

                    # --- Entropy re-boost heuristics only -- opponent
                    # SELECTION is PFSP's job now (see
                    # MicroRoyaleSelfPlayEnv._sample_pfsp_opponent, called
                    # independently by each of the 8 workers at every
                    # reset()). This block only manages exploration, reading
                    # the GLOBAL outcome_history across whatever mix of pool
                    # opponents each worker happened to sample this window.
                    window_full = len(outcome_history) == outcome_history.maxlen
                    if window_full:
                        outcomes_arr = np.array(outcome_history)
                        n_outcomes = len(outcomes_arr)
                        win_rate = int((outcomes_arr == 1).sum()) / n_outcomes
                        draw_rate = int((outcomes_arr == 0).sum()) / n_outcomes

                        if (draw_rate >= STALL_DRAW_RATE_THRESHOLD
                                and episodes_completed - last_stall_reboost_episode >= STALL_REBOOST_COOLDOWN_EPISODES):
                            entropy_reboost_episode = episodes_completed
                            last_stall_reboost_episode = episodes_completed
                            print(f">>> High draw rate ({draw_rate:.2f}) across the pool -- "
                                  "re-boosting exploration to try to break out of a passive equilibrium.")
                        elif (win_rate < 0.5
                                and episodes_completed - last_stall_reboost_episode >= ENTROPY_STALE_REBOOST_EPISODES):
                            entropy_reboost_episode = episodes_completed
                            last_stall_reboost_episode = episodes_completed
                            print(f">>> Losing trend (win rate {win_rate:.2f}) across the pool with no draw-stall "
                                  f"trigger in {ENTROPY_STALE_REBOOST_EPISODES}+ episodes -- "
                                  "periodically re-boosting exploration anyway.")

            obs = next_obs
            prev_stats = stats
            prev_dones = dones

        # --- PPO Update (identical shape to train.py's) ---
        obs_seq = torch.stack(obs_buffer)
        card_actions_seq = torch.stack(card_actions_buffer)
        placement_actions_seq = torch.stack(placement_actions_buffer)
        ability1_actions_seq = torch.stack(ability1_actions_buffer)
        ability2_actions_seq = torch.stack(ability2_actions_buffer)
        old_logprobs_seq = torch.stack(logprobs_buffer)
        values_seq = torch.stack(values_buffer)
        rewards_seq = torch.stack(rewards_buffer)
        masks_seq = torch.stack(masks_buffer)
        valid_seq = torch.stack(valid_buffer)

        with torch.no_grad():
            next_obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            # Value only depends on hx, never needs a card/placement -- skips
            # straight past card_logits and never touches placement (see
            # model.py's own comment on why this split exists).
            next_features, _ = net.extract_features(next_obs_tensor)
            _, _, _, next_value, _ = net.step_lstm_and_card(next_features, (hx, cx))
            next_value = next_value.squeeze(-1)

        advantages_seq = torch.zeros_like(rewards_seq)
        gae = torch.zeros(num_envs).to(device)
        for t in reversed(range(update_timestep)):
            next_val = next_value if t == update_timestep - 1 else values_seq[t + 1]
            delta = rewards_seq[t] + gamma * next_val * masks_seq[t] - values_seq[t]
            gae = delta + gamma * gae_lambda * masks_seq[t] * gae
            advantages_seq[t] = gae

        returns_seq = advantages_seq + values_seq
        adv_norm_seq = (advantages_seq - advantages_seq.mean()) / (advantages_seq.std() + 1e-8)

        env_indices = np.arange(num_envs)
        mb_size = max(1, num_envs // num_minibatches)
        actor_losses, critic_losses, entropy_bonuses, total_losses, clip_fracs = [], [], [], [], []

        for epoch in range(ppo_epochs):
            np.random.shuffle(env_indices)
            for mb_start in range(0, num_envs, mb_size):
                mb_env = env_indices[mb_start:mb_start + mb_size]
                if len(mb_env) == 0:
                    continue
                mb_env_t = torch.as_tensor(mb_env, dtype=torch.long, device=device)
                mb = mb_env_t.shape[0]

                mb_obs = obs_seq[:, mb_env_t]
                feats_seq, card_embeds_seq = net.extract_features(mb_obs.reshape(update_timestep * mb, -1))
                feats_seq = feats_seq.view(update_timestep, mb, -1)
                card_embeds_seq = card_embeds_seq.view(update_timestep, mb, net.hand_size + 1, -1)

                rhx = hx0[mb_env_t]
                rcx = cx0[mb_env_t]
                new_logprobs, new_values, new_entropies = [], [], []
                for t in range(update_timestep):
                    # card_actions_seq[t, mb_env_t] -- the STORED action from
                    # rollout, not a fresh sample -- conditions placement here
                    # exactly like the log-prob evaluation two lines below
                    # does. Using anything else here would evaluate placement
                    # under a DIFFERENT card than the one log-prob is scored
                    # against, silently breaking the PPO ratio.
                    (logits_t, mean_t, log_std_t, value_t, ability1_logits_t, ability2_logits_t,
                     (rhx, rcx)) = net.forward_from_features(
                        feats_seq[t], card_embeds_seq[t], (rhx, rcx), card_actions_seq[t, mb_env_t])
                    card_dist_t = Categorical(logits=logits_t)
                    place_dist_t = Normal(mean_t, log_std_t.exp())
                    ability1_dist_t = Categorical(logits=ability1_logits_t)
                    ability2_dist_t = Categorical(logits=ability2_logits_t)
                    lp_t = card_dist_t.log_prob(card_actions_seq[t, mb_env_t]) \
                        + place_dist_t.log_prob(placement_actions_seq[t, mb_env_t]).sum(dim=-1) \
                        + ability1_dist_t.log_prob(ability1_actions_seq[t, mb_env_t]) \
                        + ability2_dist_t.log_prob(ability2_actions_seq[t, mb_env_t])
                    ent_t = card_dist_t.entropy() + place_dist_t.entropy().sum(dim=-1) \
                        + ability1_dist_t.entropy() + ability2_dist_t.entropy()
                    new_logprobs.append(lp_t)
                    new_values.append(value_t.squeeze(-1))
                    new_entropies.append(ent_t)
                    reset_t = masks_seq[t, mb_env_t].unsqueeze(1)
                    rhx = rhx * reset_t
                    rcx = rcx * reset_t

                new_logprobs = torch.stack(new_logprobs)
                new_values = torch.stack(new_values)
                new_entropies = torch.stack(new_entropies)

                mb_adv = adv_norm_seq[:, mb_env_t]
                mb_ret = returns_seq[:, mb_env_t]
                mb_old_logprobs = old_logprobs_seq[:, mb_env_t]
                mb_old_values = values_seq[:, mb_env_t]
                mb_valid = valid_seq[:, mb_env_t]
                n_valid = mb_valid.sum().clamp(min=1.0)

                ratios = torch.exp(new_logprobs - mb_old_logprobs)
                surr1 = ratios * mb_adv
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * mb_adv

                value_clipped = mb_old_values + torch.clamp(new_values - mb_old_values, -eps_clip, eps_clip)
                critic_loss_unclipped = F.mse_loss(new_values, mb_ret, reduction="none")
                critic_loss_clipped = F.mse_loss(value_clipped, mb_ret, reduction="none")
                critic_loss_per_elem = torch.max(critic_loss_unclipped, critic_loss_clipped)

                actor_loss = -(torch.min(surr1, surr2) * mb_valid).sum() / n_valid
                critic_loss = (critic_loss_per_elem * mb_valid).sum() / n_valid
                entropy_bonus = (new_entropies * mb_valid).sum() / n_valid
                loss = actor_loss + 0.5 * critic_loss - (current_entropy_coef * entropy_bonus)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_bonuses.append(entropy_bonus.item())
                total_losses.append(loss.item())
                clipped = ((ratios - 1.0).abs() > eps_clip).float()
                clip_fracs.append(((clipped * mb_valid).sum() / n_valid).item())

        mean_actor_loss = np.mean(actor_losses)
        mean_critic_loss = np.mean(critic_losses)
        mean_entropy = np.mean(entropy_bonuses)
        mean_total_loss = np.mean(total_losses)
        mean_clip_frac = np.mean(clip_fracs)
        writer.add_scalar("Loss/Actor", mean_actor_loss, episodes_completed)
        writer.add_scalar("Loss/Critic", mean_critic_loss, episodes_completed)
        writer.add_scalar("Loss/Entropy", mean_entropy, episodes_completed)
        writer.add_scalar("Loss/Total", mean_total_loss, episodes_completed)
        writer.add_scalar("Loss/Clip_Fraction", mean_clip_frac, episodes_completed)
        print(f"  >> Update @ ep {episodes_completed} | Actor: {mean_actor_loss:.5f} | "
              f"Critic: {mean_critic_loss:.5f} | Entropy: {mean_entropy:.4f} | "
              f"ClipFrac: {mean_clip_frac:.4f}")

        obs_buffer.clear()
        card_actions_buffer.clear()
        placement_actions_buffer.clear()
        ability1_actions_buffer.clear()
        ability2_actions_buffer.clear()
        logprobs_buffer.clear()
        values_buffer.clear()
        rewards_buffer.clear()
        masks_buffer.clear()
        valid_buffer.clear()

        hx, cx = hx.detach(), cx.detach()

        if episodes_completed - last_save_ep >= 500:
            torch.save({
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episodes_completed": episodes_completed,
                "entropy_reboost_episode": entropy_reboost_episode,
                "last_stall_reboost_episode": last_stall_reboost_episode,
                "last_eval_ep": last_eval_ep,
                "reference_roster": reference_roster,
                "outcome_history": list(outcome_history),
            }, WEIGHT_PATH)
            print(f">>> Checkpoint saved to {WEIGHT_PATH} (episode {episodes_completed}, "
                  f"pool {len(historical_pool)})")
            last_save_ep = episodes_completed

        # This run's own new snapshots join the SAME shared historical pool
        # (see HISTORICAL_CHECKPOINT_DIR's comment in train.py). Refreshing
        # historical_pool here (main process) AND broadcasting it to every
        # worker keeps both PFSP sampling and the evaluation roster growing
        # as the trainee improves, without waiting for any single "current
        # opponent" to be beaten first (there isn't one anymore).
        if episodes_completed - last_historical_save_ep >= HISTORICAL_CHECKPOINT_INTERVAL_EPISODES:
            hist_path = os.path.join(
                HISTORICAL_CHECKPOINT_DIR,
                f"{int(time.time() * 1000)}_pipeline2_ep{episodes_completed:08d}.pth")
            torch.save({"model": net.state_dict()}, hist_path)
            print(f">>> Historical snapshot saved to {hist_path}")
            last_historical_save_ep = episodes_completed
            historical_pool = discover_historical_checkpoints(episodes_completed)
            envs.call("refresh_pfsp_pool", historical_pool)

        if episodes_completed - last_eval_ep >= EVAL_INTERVAL_EPISODES:
            update_reference_roster(reference_roster, historical_pool)
            if reference_roster:
                print(f"Evaluating current policy (greedy) against {len(reference_roster)} fixed "
                      f"reference opponent(s), {EVAL_GAMES_PER_OPPONENT} games each...")
                agent_elo, per_opponent = evaluate_against_roster(
                    net, device, reference_roster, MAX_X, MAX_Y, n_games=EVAL_GAMES_PER_OPPONENT)
                writer.add_scalar("Eval/Elo", agent_elo, episodes_completed)
                for i, entry in enumerate(reference_roster):
                    stats_i = per_opponent[entry["path"]]
                    writer.add_scalar(f"Eval/WinRate_vs_Anchor{i}", stats_i["score"], episodes_completed)
                print(f">>> Eval @ ep {episodes_completed}: Elo~{agent_elo:.0f} "
                      f"(vs {len(reference_roster)} fixed anchor(s) -- a comparable trend against a "
                      "never-changing roster, not a calibrated rating; see REFERENCE_ROSTER_ELO_STEP's comment)")
            last_eval_ep = episodes_completed

        if episodes_completed - last_replay_ep >= 1000 and historical_pool:
            print(f"Generating replay video for episode {episodes_completed}...")
            test_env = MicroRoyaleSelfPlayEnv()
            # Newest/strongest known pool entry -- purely for a representative
            # demo/monitoring replay, unrelated to PFSP sampling or eval.
            demo_opponent_path = historical_pool[-1]
            test_env.set_historical_opponent(demo_opponent_path)
            t_obs, _ = test_env.reset()
            t_hx = torch.zeros(1, 256).to(device)
            t_cx = torch.zeros(1, 256).to(device)
            t_done = False
            t_decisions = []
            REPLAY_SKIP_FRAMES = 10
            while not t_done:
                t_obs_tensor = torch.tensor(t_obs, dtype=torch.float32).unsqueeze(0).to(device)
                t_features, t_card_embeds = net.extract_features(t_obs_tensor)
                (t_logits, t_ability1_logits, t_ability2_logits, t_value,
                 (t_hx, t_cx)) = net.step_lstm_and_card(t_features, (t_hx, t_cx))
                t_idx = Categorical(logits=t_logits).sample()
                t_norm, _ = net.placement_given_card(t_hx, t_card_embeds, t_idx)
                t_card_idx = t_idx.item()
                t_ability1_action = Categorical(logits=t_ability1_logits).sample().item()
                t_ability2_action = Categorical(logits=t_ability2_logits).sample().item()
                t_hand = test_env.game.get_hand()
                t_card_id = t_hand[t_card_idx] if t_card_idx < len(t_hand) else -1
                t_action = {
                    "card_index": np.array([t_card_idx]),
                    "target_x": np.array([torch.clamp(t_norm[0, 0] * MAX_X, 0.0, MAX_X).item()]),
                    "target_y": np.array([torch.clamp(t_norm[0, 1] * MAX_Y, 0.0, MAX_Y).item()]),
                    "activate_ability_slot1": np.array([t_ability1_action]),
                    "activate_ability_slot2": np.array([t_ability2_action]),
                }
                t_decisions.append({
                    "stateValue": t_value.item(),
                    "actionCardId": t_card_id,
                    "actionX": float(t_action["target_x"][0]),
                    "actionY": float(t_action["target_y"][0]),
                })
                t_obs, _, t_terminated, t_truncated, _ = test_env.step(t_action, skip_frames=REPLAY_SKIP_FRAMES)
                t_done = t_terminated or t_truncated
            replay_path = f"replays/selfplay_replay_ep{episodes_completed}.json"
            test_env.game.save_log(replay_path)
            annotate_replay_with_agent_info(replay_path, t_decisions, REPLAY_SKIP_FRAMES)
            last_replay_ep = episodes_completed

    writer.close()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    train_selfplay_ppo()
