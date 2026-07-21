import os
import glob
import time
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
    DRAW_PENALTY,
)

# --- Pipeline #2: Historical Self-Play ---
# Pipeline #1 (train.py) teaches the bot to beat a random-but-unskilled
# opponent -- a necessary bootstrap, but not a ceiling: nothing in that
# opponent ever punishes a bad trade, reads a push, or plays around a specific
# threat the way a real opponent would. This pipeline is what actually
# introduces that pressure: the trainee (still the same live policy, continuing
# gradient updates) plays against FROZEN snapshots of its own past selves,
# weakest first, advancing to the next-strongest snapshot once it consistently
# beats the current one. Both sides genuinely decide what to play now (see
# ClashEnv::stepSelfPlay/extractObservationForTeam in the C++ layer) instead of
# team 1 being the built-in random C++ bot.

# Raw win rate (not decisive), same convention as train.py's curriculum gates.
# Lowered from 0.95 -- observed win rate against real pool opponents tops out
# around 0.80-0.85, so 0.95 was practically unreachable and every transition
# ended up relying on the MAX_EPISODES_PER_HISTORICAL_OPPONENT safety valve
# below instead of genuine mastery. Paired with MASTERY_CONFIRM_EPISODES
# further down so a lower bar doesn't just trade "never advances" for
# "advances on a lucky noisy window."
SELFPLAY_WIN_RATE_GATE = 0.85
# Safety valve, not the intended advancement trigger -- same role as pipeline
# #1's MAX_EPISODES_PER_RANDOM_DECK. A near-mirror matchup (the current
# trainee vs. a historical snapshot of itself from not very long ago) can
# converge toward mutual passivity once both sides are closely matched and
# entropy is floored -- games time out into draws instead of being decisively
# won, and win_rate (draws count against it, same as losses) never clears
# SELFPLAY_WIN_RATE_GATE. Confirmed happening in practice: stuck on the same
# opponent for 10,000+ episodes with draw rate pinned at 82-94%. Without this,
# such a stall blocks the entire opponent queue forever.
MAX_EPISODES_PER_HISTORICAL_OPPONENT = 5000
# Escape hatch for the SAME mutual-passivity problem, triggered much earlier
# than the timeout above: once draws cross this share of the current 100-
# episode window, re-boost exploration right away instead of waiting the
# whole MAX_EPISODES_PER_HISTORICAL_OPPONENT window out at floored entropy --
# floored entropy is exactly what locks a passive equilibrium in place, since
# neither side has any remaining chance to stumble into a different joint
# strategy. Reward-shaping fixes (see train.py's W_ELIXIR_TRADE/DRAW_PENALTY/
# W_ELIXIR_OVERFLOW comments) address WHY passivity looked attractive; this
# addresses the fact that once both sides are already sitting in it, ordinary
# gradient descent has no exploration left to climb back out.
STALL_DRAW_RATE_THRESHOLD = 0.5
# Cooldown so this doesn't re-fire every single episode once the window is
# saturated with draws -- gives each boost a real window to actually take
# effect before deciding whether another one is needed.
STALL_REBOOST_COOLDOWN_EPISODES = 1000
# Escape hatch for the failure mode the draw-rate trigger above CAN'T see:
# draws near zero (the passivity fix worked) but win rate still stuck below
# SELFPLAY_WIN_RATE_GATE, with entropy floored the whole time and nothing
# ever refreshing it. Confirmed happening in practice: stage_start_episode
# sitting unreset for 24,800+ episodes straight, measured policy entropy
# drifting UP the whole stretch (more random, not less) while win rate and
# tower-HP margins both got WORSE -- floored entropy for that long just lets
# ordinary gradient noise erode an already-good policy with no exploration
# boost ever pulling it back. Longer than STALL_REBOOST_COOLDOWN_EPISODES
# since this is a broader "nothing's converged in a while" signal, not an
# acute passivity relapse -- shares the same last_stall_reboost_episode
# cooldown clock as the draw-rate trigger (whichever fires resets it).
ENTROPY_STALE_REBOOST_EPISODES = 1500
# How long a >=SELFPLAY_WIN_RATE_GATE window has to hold CONTINUOUSLY before
# it counts as real mastery rather than one lucky 100-episode sample -- without
# this, a single noisy window crossing the (now-lower) gate for an instant
# would immediately unlock a harder opponent the trainee hasn't actually
# beaten reliably yet.
MASTERY_CONFIRM_EPISODES = 100

WEIGHT_PATH = "model_weights_selfplay.pth"
# Pipeline #1's final artifact -- read ONCE, only to seed a from-scratch
# pipeline #2 run (bare weights only; pipeline #2 keeps its own separate
# episode count/optimizer state in WEIGHT_PATH from then on, so pipeline #1's
# own checkpoint is never overwritten by this script).
BOOTSTRAP_FROM_PATH = "model_weights.pth"


def discover_historical_checkpoints():
    """All *.pth files in HISTORICAL_CHECKPOINT_DIR, oldest-saved-first (mtime).
    Save order is the ordering signal, not filenames -- pipeline #1 and this
    same script both drop snapshots into this one shared folder, on two
    unrelated episode-count scales, so parsing/comparing episode numbers out
    of the filename wouldn't give a meaningful weakest->strongest order."""
    paths = glob.glob(os.path.join(HISTORICAL_CHECKPOINT_DIR, "*.pth"))
    return sorted(paths, key=os.path.getmtime)


def load_state_dict_flexible(net, state_dict, context_label):
    """Loads state_dict into net. Returns True on a clean, fully-matching load.

    On an architecture mismatch (e.g. a card-roster change resizing the hand
    one-hot encoding, which is the only part of MicroRoyaleNet that depends on
    NUM_CARD_IDS -- see model.py's scalar_size), falls back to loading only
    the tensors whose shape still matches, leaving the rest at their fresh
    initialization instead of crashing outright. The CNN/LSTM/action heads are
    independent of NUM_CARD_IDS, so this warm-starts on everything except the
    one incompatible layer rather than discarding a whole checkpoint (and,
    upstream of this function, an entire opponent-history library) over it.

    Returns False when this fallback path was taken -- the caller should NOT
    then load a paired optimizer state dict, since Adam's per-parameter
    buffers would be stale/mismatched for whatever just got reinitialized.
    """
    try:
        net.load_state_dict(state_dict)
        return True
    except RuntimeError:
        own_state = net.state_dict()
        compatible = {k: v for k, v in state_dict.items()
                      if k in own_state and v.shape == own_state[k].shape}
        skipped = sorted(set(state_dict.keys()) - set(compatible.keys()))
        own_state.update(compatible)
        net.load_state_dict(own_state)
        print(f"[{context_label}] Architecture mismatch -- warm-started "
              f"{len(compatible)}/{len(state_dict)} tensor(s), re-initialized: {skipped}")
        return False


class MicroRoyaleSelfPlayEnv(gym.Env):
    """Same action/observation shape as gym_wrapper.MicroRoyaleEnv, but team 1
    is a frozen copy of MicroRoyaleNet (never trained here -- eval()/no_grad
    only) instead of the built-in random C++ opponentTurn(). Swapping which
    historical snapshot team 1 uses is a live, resumable-mid-episode operation
    (set_historical_opponent), the same shape as set_opponent_deck in
    gym_wrapper.py.
    """

    MAX_X = 17.0
    MAX_Y = 15.5  # riverStart(16.0) - OWN_HALF_RIVER_BUFFER(0.5), see train.py's MAX_Y_AI

    def __init__(self, env_config=None):
        super().__init__()
        env_config = env_config or {}
        self.deck = env_config.get("deck", list(DEFAULT_DECK))
        max_ticks = env_config.get("max_ticks", 3600)
        self.game = clash_royale_env.ClashRoyaleEnv(self.deck, self.deck, max_ticks)

        # Team 1's brain -- CPU is plenty for a single inference-only forward
        # pass per step per worker process, and keeps this off the GPU the
        # trainee's own updates use in the main process.
        self.device = torch.device("cpu")
        self.opponent_net = MicroRoyaleNet().to(self.device)
        self.opponent_net.eval()
        self.opponent_hx = torch.zeros(1, 256).to(self.device)
        self.opponent_cx = torch.zeros(1, 256).to(self.device)
        self.opponent_checkpoint_path = None
        if env_config.get("historical_checkpoint_path"):
            self.set_historical_opponent(env_config["historical_checkpoint_path"])

        self.action_space = spaces.Dict({
            "card_index": spaces.Discrete(5),
            "target_x": spaces.Box(low=0.0, high=self.MAX_X, shape=(1,), dtype=np.float32),
            "target_y": spaces.Box(low=0.0, high=self.MAX_Y, shape=(1,), dtype=np.float32),
        })
        obs_size = self.game.observation_size()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)

    def set_historical_opponent(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        load_state_dict_flexible(self.opponent_net, state_dict, f"historical opponent {checkpoint_path}")
        self.opponent_net.eval()
        self.opponent_checkpoint_path = checkpoint_path

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs_list = self.game.reset()
        self.opponent_hx = torch.zeros(1, 256).to(self.device)
        self.opponent_cx = torch.zeros(1, 256).to(self.device)
        return np.array(obs_list, dtype=np.float32), {}

    def _opponent_action(self):
        """Team 1's own decision, from ITS OWN (mirrored) point of view --
        see ClashEnv::extractObservationForTeam. Sampled the same way rollout
        actions are everywhere else in this project, just with no_grad and no
        buffering: this network never gets updated here."""
        obs1 = np.array(self.game.get_observation_for_team(1), dtype=np.float32)
        with torch.no_grad():
            obs1_t = torch.tensor(obs1, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits1, mean1, log_std1, _, (self.opponent_hx, self.opponent_cx) = self.opponent_net(
                obs1_t, (self.opponent_hx, self.opponent_cx))
            card_idx1 = Categorical(logits=logits1).sample().item()
            placement1 = torch.clamp(Normal(mean1, log_std1.exp()).sample(), 0.0, 1.0)
            x1 = (placement1[0, 0] * self.MAX_X).item()
            y1 = (placement1[0, 1] * self.MAX_Y).item()
        return card_idx1, x1, y1

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
        card_idx1, x1, y1 = self._opponent_action()

        result = self.game.step_self_play(card_idx0, x0, y0, card_idx1, x1, y1, skip_frames)

        obs = np.array(result.observation0, dtype=np.float32)
        reward = float(result.reward0)
        terminated = bool(result.done)

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

    historical_pool = discover_historical_checkpoints()
    if not historical_pool:
        raise RuntimeError(
            f"No historical snapshots found in {HISTORICAL_CHECKPOINT_DIR}/ -- "
            "pipeline #1 (train.py) needs to have run long enough to have saved "
            "at least one (every HISTORICAL_CHECKPOINT_INTERVAL_EPISODES episodes) "
            "before pipeline #2 has anything to play against.")

    num_envs = 8
    print(f"Initializing {num_envs} self-play environments against "
          f"{len(historical_pool)} known historical snapshot(s)...")
    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(num_envs)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MicroRoyaleNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=3e-4)

    # Same PPO hyperparameters as train.py -- same architecture and algorithm,
    # only the opponent-progression mechanic differs, so there's no reason to
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

    MAX_X = MicroRoyaleSelfPlayEnv.MAX_X
    MAX_Y = MicroRoyaleSelfPlayEnv.MAX_Y

    historical_opponent_index = 0
    stage_start_episode = 0
    opponent_episode_start = 0  # episodes_completed value when the CURRENT historical opponent started
    last_stall_reboost_episode = 0  # episodes_completed value at the last entropy re-boost (draw-stall or stale-plateau)
    mastery_streak_start = None  # episodes_completed value when the win-rate gate was first continuously cleared
    episodes_completed = 0
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
        historical_opponent_index = checkpoint["historical_opponent_index"]
        stage_start_episode = checkpoint["stage_start_episode"]
        # .get() with episodes_completed as the fallback: checkpoints saved
        # before this safety valve existed just restart the per-opponent
        # clock now, rather than retroactively counting already-elapsed
        # episodes against the new limit.
        opponent_episode_start = checkpoint.get("opponent_episode_start", episodes_completed)
        last_stall_reboost_episode = checkpoint.get("last_stall_reboost_episode", episodes_completed)
        mastery_streak_start = checkpoint.get("mastery_streak_start", None)
        outcome_history = deque(checkpoint["outcome_history"], maxlen=100)
        full_resume = True
        print(f"Resumed pipeline #2 from {WEIGHT_PATH}: episode {episodes_completed}, "
              f"historical_opponent_index {historical_opponent_index}")
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

    if historical_opponent_index >= len(historical_pool):
        historical_opponent_index = len(historical_pool) - 1
    current_opponent_path = historical_pool[historical_opponent_index]
    envs.call("set_historical_opponent", current_opponent_path)
    print(f"Initial historical opponent ({historical_opponent_index + 1}/{len(historical_pool)}): {current_opponent_path}")

    if not full_resume and os.path.exists(log_dir):
        import shutil
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    obs_buffer, card_actions_buffer, placement_actions_buffer = [], [], []
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
        episodes_in_stage = episodes_completed - stage_start_episode
        current_entropy_coef = max(min_entropy_coef, initial_entropy_coef * (entropy_decay_rate ** episodes_in_stage))

        hx0 = hx.detach().clone()
        cx0 = cx.detach().clone()

        for step in range(update_timestep):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

            with torch.no_grad():
                card_logits, placement_mean, placement_log_std, state_value, (hx, cx) = net(obs_tensor, (hx, cx))
                card_dist = Categorical(logits=card_logits)
                card_idx = card_dist.sample()
                placement_dist = Normal(placement_mean, placement_log_std.exp())
                placement_sample = placement_dist.sample()
                placement_logprob = placement_dist.log_prob(placement_sample).sum(dim=-1)
                total_logprob = card_dist.log_prob(card_idx) + placement_logprob

            placement_clamped = torch.clamp(placement_sample, 0.0, 1.0)
            target_x = placement_clamped[:, 0] * MAX_X
            target_y = placement_clamped[:, 1] * MAX_Y

            action = {
                "card_index": card_idx.cpu().numpy(),
                "target_x": target_x.cpu().numpy().reshape(num_envs, 1),
                "target_y": target_y.cpu().numpy().reshape(num_envs, 1),
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
                              f"Opponent: {historical_opponent_index + 1}/{len(historical_pool)} | Entropy: {current_entropy_coef:.4f}")
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
                        writer.add_scalar("Training/Historical_Opponent_Index", historical_opponent_index, episodes_completed)
                        writer.add_scalar("Training/Entropy_Coef", current_entropy_coef, episodes_completed)

                    # --- Historical-opponent advancement: beat the current
                    # snapshot ~SELFPLAY_WIN_RATE_GATE of the time, then move to
                    # the next-strongest known one. If the queue's exhausted,
                    # re-scan the shared folder -- pipeline #1 may still be
                    # running, and this same script also drops its own new
                    # snapshots in there as it improves (see the save below) --
                    # and if nothing new has appeared yet, just keep training
                    # against the current (strongest known) opponent rather
                    # than blocking.
                    #
                    # MAX_EPISODES_PER_HISTORICAL_OPPONENT is a safety valve,
                    # not the intended trigger -- see its own comment above. A
                    # near-mirror matchup can converge toward mutual passivity
                    # (draws, not decisive wins) and never clear the win-rate
                    # gate on its own; this makes sure that stalls the queue
                    # for at most one timeout instead of forever.
                    window_full = len(outcome_history) == outcome_history.maxlen
                    timed_out = episodes_completed - opponent_episode_start >= MAX_EPISODES_PER_HISTORICAL_OPPONENT
                    if window_full or timed_out:
                        n_outcomes = len(outcome_history)
                        if n_outcomes > 0:
                            outcomes_arr = np.array(outcome_history)
                            win_rate = int((outcomes_arr == 1).sum()) / n_outcomes
                            draw_rate = int((outcomes_arr == 0).sum()) / n_outcomes
                        else:
                            win_rate = 0.0
                            draw_rate = 0.0

                        # Gate-clearing win rate alone isn't enough to advance -- a
                        # single lucky 100-episode window can cross the (now-lower)
                        # gate by chance and immediately unlock a harder opponent
                        # the trainee hasn't actually mastered. Require the window-
                        # level win rate to stay >= the gate CONTINUOUSLY for a
                        # further MASTERY_CONFIRM_EPISODES before treating it as
                        # real mastery instead of noise.
                        gate_cleared_now = window_full and win_rate >= SELFPLAY_WIN_RATE_GATE
                        if gate_cleared_now:
                            if mastery_streak_start is None:
                                mastery_streak_start = episodes_completed
                        else:
                            mastery_streak_start = None
                        mastered = (mastery_streak_start is not None
                                    and episodes_completed - mastery_streak_start >= MASTERY_CONFIRM_EPISODES)

                        if mastered or timed_out:
                            refreshed_pool = discover_historical_checkpoints()
                            if len(refreshed_pool) > historical_opponent_index + 1:
                                historical_pool = refreshed_pool
                                historical_opponent_index += 1
                                current_opponent_path = historical_pool[historical_opponent_index]
                                envs.call("set_historical_opponent", current_opponent_path)
                                outcome_history.clear()
                                stage_start_episode = episodes_completed
                                opponent_episode_start = episodes_completed
                                mastery_streak_start = None
                                reason = (f"mastered it (win rate {win_rate:.2f}, held "
                                          f"{MASTERY_CONFIRM_EPISODES}+ episodes)" if mastered
                                          else f"hit the {MAX_EPISODES_PER_HISTORICAL_OPPONENT}-episode "
                                               f"safety cap without mastering it (win rate {win_rate:.2f})")
                                print(f">>> Beat opponent {historical_opponent_index}/{len(historical_pool)} "
                                      f"({reason}) -- advancing to "
                                      f"{historical_opponent_index + 1}/{len(historical_pool)}: {current_opponent_path}")
                                writer.add_scalar("Training/Historical_Opponent_Index", historical_opponent_index, episodes_completed)
                            elif timed_out:
                                # Timed out but nothing stronger is known yet -- re-boost
                                # exploration and give it a fresh attempt window against
                                # the SAME opponent instead of continuing indefinitely at
                                # floored entropy with zero further exploration pressure.
                                outcome_history.clear()
                                stage_start_episode = episodes_completed
                                opponent_episode_start = episodes_completed
                                mastery_streak_start = None
                                print(f">>> Opponent {historical_opponent_index + 1}/{len(historical_pool)} timed out "
                                      f"(win rate {win_rate:.2f}) but no stronger snapshot exists yet -- "
                                      "re-boosting exploration and retrying the same opponent.")
                            # else: nothing stronger known yet and not timed out --
                            # keep training against the current opponent,
                            # outcome_history keeps sliding so this re-checks
                            # every episode.
                        elif (window_full and draw_rate >= STALL_DRAW_RATE_THRESHOLD
                                and episodes_completed - last_stall_reboost_episode >= STALL_REBOOST_COOLDOWN_EPISODES):
                            # High draw rate well before either the win gate or the
                            # timeout -- re-boost exploration NOW rather than let it
                            # sit at floored entropy for the rest of the timeout
                            # window, since floored entropy is exactly what locks a
                            # passive equilibrium in place (see this constant's
                            # comment above). Doesn't touch opponent_episode_start
                            # or outcome_history -- still the same opponent, same
                            # win-rate/timeout clock, just a fresh exploration push.
                            stage_start_episode = episodes_completed
                            last_stall_reboost_episode = episodes_completed
                            print(f">>> High draw rate ({draw_rate:.2f}) against opponent "
                                  f"{historical_opponent_index + 1}/{len(historical_pool)} -- "
                                  "re-boosting exploration to try to break out of a passive equilibrium.")
                        elif (window_full
                                and episodes_completed - last_stall_reboost_episode >= ENTROPY_STALE_REBOOST_EPISODES):
                            # Covers the failure mode the draw-rate trigger above
                            # can't see: draws near zero but win rate still stuck
                            # below the mastery gate, entropy floored the whole
                            # stretch with nothing ever refreshing it (see this
                            # constant's own comment). Same shared cooldown clock
                            # as the draw-rate trigger -- whichever fires resets it.
                            stage_start_episode = episodes_completed
                            last_stall_reboost_episode = episodes_completed
                            print(f">>> No mastery/draw-stall trigger in {ENTROPY_STALE_REBOOST_EPISODES}+ episodes "
                                  f"against opponent {historical_opponent_index + 1}/{len(historical_pool)} "
                                  f"(win rate {win_rate:.2f}) -- periodically re-boosting exploration anyway.")

            obs = next_obs
            prev_stats = stats
            prev_dones = dones

        # --- PPO Update (identical shape to train.py's) ---
        obs_seq = torch.stack(obs_buffer)
        card_actions_seq = torch.stack(card_actions_buffer)
        placement_actions_seq = torch.stack(placement_actions_buffer)
        old_logprobs_seq = torch.stack(logprobs_buffer)
        values_seq = torch.stack(values_buffer)
        rewards_seq = torch.stack(rewards_buffer)
        masks_seq = torch.stack(masks_buffer)
        valid_seq = torch.stack(valid_buffer)

        with torch.no_grad():
            next_obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            _, _, _, next_value, _ = net(next_obs_tensor, (hx, cx))
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
                feats_seq = net.extract_features(mb_obs.reshape(update_timestep * mb, -1))
                feats_seq = feats_seq.view(update_timestep, mb, -1)

                rhx = hx0[mb_env_t]
                rcx = cx0[mb_env_t]
                new_logprobs, new_values, new_entropies = [], [], []
                for t in range(update_timestep):
                    logits_t, mean_t, log_std_t, value_t, (rhx, rcx) = net.forward_from_features(feats_seq[t], (rhx, rcx))
                    card_dist_t = Categorical(logits=logits_t)
                    place_dist_t = Normal(mean_t, log_std_t.exp())
                    lp_t = card_dist_t.log_prob(card_actions_seq[t, mb_env_t]) \
                        + place_dist_t.log_prob(placement_actions_seq[t, mb_env_t]).sum(dim=-1)
                    ent_t = card_dist_t.entropy() + place_dist_t.entropy().sum(dim=-1)
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
                "historical_opponent_index": historical_opponent_index,
                "stage_start_episode": stage_start_episode,
                "opponent_episode_start": opponent_episode_start,
                "last_stall_reboost_episode": last_stall_reboost_episode,
                "mastery_streak_start": mastery_streak_start,
                "outcome_history": list(outcome_history),
            }, WEIGHT_PATH)
            print(f">>> Checkpoint saved to {WEIGHT_PATH} (episode {episodes_completed}, "
                  f"opponent {historical_opponent_index + 1}/{len(historical_pool)})")
            last_save_ep = episodes_completed

        # This run's own new snapshots join the SAME shared historical pool
        # (see HISTORICAL_CHECKPOINT_DIR's comment in train.py) -- once the
        # trainee clears every currently-known opponent, these are what it
        # faces next: itself, a little while further into self-play.
        if episodes_completed - last_historical_save_ep >= HISTORICAL_CHECKPOINT_INTERVAL_EPISODES:
            hist_path = os.path.join(
                HISTORICAL_CHECKPOINT_DIR,
                f"{int(time.time() * 1000)}_pipeline2_ep{episodes_completed:08d}.pth")
            torch.save({"model": net.state_dict()}, hist_path)
            print(f">>> Historical snapshot saved to {hist_path}")
            last_historical_save_ep = episodes_completed

        if episodes_completed - last_replay_ep >= 1000:
            print(f"Generating replay video for episode {episodes_completed}...")
            test_env = MicroRoyaleSelfPlayEnv()
            test_env.set_historical_opponent(current_opponent_path)
            t_obs, _ = test_env.reset()
            t_hx = torch.zeros(1, 256).to(device)
            t_cx = torch.zeros(1, 256).to(device)
            t_done = False
            t_decisions = []
            REPLAY_SKIP_FRAMES = 10
            while not t_done:
                t_obs_tensor = torch.tensor(t_obs, dtype=torch.float32).unsqueeze(0).to(device)
                t_logits, t_norm, _, t_value, (t_hx, t_cx) = net(t_obs_tensor, (t_hx, t_cx))
                t_idx = Categorical(logits=t_logits).sample()
                t_card_idx = t_idx.item()
                t_hand = test_env.game.get_hand()
                t_card_id = t_hand[t_card_idx] if t_card_idx < len(t_hand) else -1
                t_action = {
                    "card_index": np.array([t_card_idx]),
                    "target_x": np.array([torch.clamp(t_norm[0, 0] * MAX_X, 0.0, MAX_X).item()]),
                    "target_y": np.array([torch.clamp(t_norm[0, 1] * MAX_Y, 0.0, MAX_Y).item()]),
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
