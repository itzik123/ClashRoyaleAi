import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from torch.distributions import Categorical, Normal
from torch.utils.tensorboard import SummaryWriter
from collections import deque

import gym_wrapper
from model import MicroRoyaleNet

# Reward-shaping weights (dense guidance on top of the sparse +1/-1 win/loss signal).
# Kept intentionally small so the cumulative shaping over an episode stays comparable
# to - not larger than - the terminal +1/-1. Watch Reward/Episode_Shaping_Sum against
# the win-rate in TensorBoard: if the shaping sum dwarfs +/-1, lower these.
W_BLDG = 0.5     # weight on building (tower) HP swings
W_TROOPS = 0.1   # weight on troop HP swings

# Spatial layout of the observation (must match ClashEnv.h):
# channels 0-2 ally troops (melee/ranged/tank), 3 ally buildings,
# channels 4-6 enemy troops, 7 enemy buildings, 8 river mask.
N_CHANNELS = 9
BOARD_H, BOARD_W = 32, 18
SPATIAL_SIZE = N_CHANNELS * BOARD_H * BOARD_W

def compute_shaping(obs, prev_obs, w_bldg=W_BLDG, w_troops=W_TROOPS):
    """
    Vectorized dense-reward shaping term based on per-step HP deltas.
    Rewards damage dealt to the enemy and penalizes damage taken.
    Returns the shaping term ONLY (num_envs,), excluding the sparse win/loss reward.
    obs / prev_obs: (num_envs, obs_size)
    """
    if prev_obs is None:
        return np.zeros(obs.shape[0], dtype=np.float32)

    num_envs = obs.shape[0]

    obs_spatial = obs[:, :SPATIAL_SIZE].reshape(num_envs, N_CHANNELS, BOARD_H, BOARD_W)
    prev_spatial = prev_obs[:, :SPATIAL_SIZE].reshape(num_envs, N_CHANNELS, BOARD_H, BOARD_W)

    d_ally_troops = obs_spatial[:, 0:3].sum(axis=(1, 2, 3)) - prev_spatial[:, 0:3].sum(axis=(1, 2, 3))
    d_enemy_troops = obs_spatial[:, 4:7].sum(axis=(1, 2, 3)) - prev_spatial[:, 4:7].sum(axis=(1, 2, 3))
    d_ally_bldg = obs_spatial[:, 3].sum(axis=(1, 2)) - prev_spatial[:, 3].sum(axis=(1, 2))
    d_enemy_bldg = obs_spatial[:, 7].sum(axis=(1, 2)) - prev_spatial[:, 7].sum(axis=(1, 2))

    shaping = (w_bldg * (d_ally_bldg - d_enemy_bldg)
               + w_troops * (d_ally_troops - d_enemy_troops))

    return shaping.astype(np.float32)

def building_hp_end(obs_vec):
    """Remaining normalized building HP (ally, enemy) in a single final observation --
    used as a per-episode offense/defense progress metric."""
    spatial = obs_vec[:SPATIAL_SIZE].reshape(N_CHANNELS, BOARD_H, BOARD_W)
    return float(spatial[3].sum()), float(spatial[7].sum())

def make_env():
    def _init():
        return gym_wrapper.MicroRoyaleEnv()
    return _init

def train_ppo():
    os.makedirs("replays", exist_ok=True)
    log_dir = "runs/clash_royale_experiment"
    if os.path.exists(log_dir):
        import shutil
        shutil.rmtree(log_dir)
        
    writer = SummaryWriter(log_dir=log_dir)
    
    num_envs = 4 # Based on user request (max 3-4 cores)
    print(f"Initializing {num_envs} Async Vectorized Environments...")
    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(num_envs)])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MicroRoyaleNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=3e-4)

    weight_path = "model_weights.pth"
    if os.path.exists(weight_path):
        try:
            net.load_state_dict(torch.load(weight_path, map_location=device))
            print(f"Loaded existing weights from {weight_path}")
        except RuntimeError:
            # Architecture changed since these weights were saved (e.g. the card head
            # grew from 4 to 5 actions). Keep them as a backup and start fresh.
            backup = weight_path + ".bak"
            os.replace(weight_path, backup)
            print(f"Saved weights are incompatible with the current architecture; moved to {backup}, starting fresh.")
    
    # PPO Hyperparameters
    gamma = 0.99
    gae_lambda = 0.95      # GAE(lambda) smoothing for advantage estimation
    eps_clip = 0.2
    update_timestep = 500  # Number of steps PER CORE before update
    ppo_epochs = 2         # How many times to reuse each rollout
    # Minibatches per epoch, split across the env (sequence) dimension. Each minibatch
    # replays its sequences through the LSTM, so cost ~= ppo_epochs * num_minibatches *
    # update_timestep sequential forward/backward passes. With only 4 envs a single
    # full-batch update is faster and lower-variance; raise this once num_envs grows.
    num_minibatches = 1
    max_grad_norm = 0.5    # Gradient clipping (stabilizes the long-horizon BPTT)

    initial_entropy_coef = 0.1
    min_entropy_coef = 0.001
    entropy_decay_rate = 0.9995
    
    # BOARD_MAX_X in the engine is 17.0 -- placements with x>17 are silently
    # rejected (isValidPlacement), so scaling by 18 wasted part of the action range.
    MAX_X = 17.0
    MAX_Y_AI = 14.5

    # --- Curriculum: once the agent's win-rate against the current opponent
    # settles above a threshold, escalate the opponent's elixir multiplier.
    # 1.0 = today's fully-random opponent; higher values make it play cards
    # faster/near-continuously. Extend this list to add more stages later.
    CURRICULUM_STAGES = [
        {"opp_elixir_multiplier": 1.0, "win_rate_threshold": 0.75},
        {"opp_elixir_multiplier": 1.75, "win_rate_threshold": 0.75},
        {"opp_elixir_multiplier": 3.0, "win_rate_threshold": None},  # final stage, no further auto-advance
    ]
    curriculum_stage = 0
    stage_start_episode = 0   # Entropy decays relative to the current stage's start (improvement #5)
    # Per-episode outcome: +1 win, -1 loss, 0 draw (timeout). The curriculum gate uses
    # the DECISIVE win rate W/(W+L): draws say "didn't close the game", not "can't beat
    # the opponent", so they shouldn't block stage advancement.
    outcome_history = deque(maxlen=100)
    MIN_DECIDED_FOR_ADVANCE = 30   # don't advance the curriculum off a handful of decided games

    obs_buffer = []
    card_actions_buffer = []
    placement_actions_buffer = []
    logprobs_buffer = []
    values_buffer = []
    rewards_buffer = []
    masks_buffer = []

    reward_history = deque(maxlen=50)
    shaping_history = deque(maxlen=50)   # Per-episode shaping sum, to watch it vs the +/-1 terminal (improvement #6)
    # Progress metrics: game length (winning FASTER = learning to close games) and
    # remaining building HP on each side at episode end (offense/defense quality).
    ep_len_history = deque(maxlen=50)
    ally_bldg_end_history = deque(maxlen=50)
    enemy_bldg_end_history = deque(maxlen=50)

    obs, _ = envs.reset()
    prev_obs = None
    prev_dones = np.zeros(num_envs, dtype=bool)   # Whether each env was reset on the previous step
    hx = torch.zeros(num_envs, 256).to(device)
    cx = torch.zeros(num_envs, 256).to(device)

    episodes_completed = 0
    last_save_ep = 0
    last_replay_ep = 0
    ep_rewards = np.zeros(num_envs)
    ep_shaping = np.zeros(num_envs)
    ep_steps = np.zeros(num_envs, dtype=np.int64)
    
    print(f"Training started on {num_envs} CPU cores simultaneously!")
    
    while episodes_completed < 50000:
        # Entropy decays within each curriculum stage, not over all time: advancing a
        # stage resets the clock (stage_start_episode) so exploration is boosted again
        # for the new, harder opponent instead of staying collapsed (improvement #5).
        episodes_in_stage = episodes_completed - stage_start_episode
        current_entropy_coef = max(min_entropy_coef, initial_entropy_coef * (entropy_decay_rate ** episodes_in_stage))

        # Hidden state at the start of this rollout; needed to correctly replay
        # the LSTM sequences during the multi-epoch update.
        hx0 = hx.detach().clone()
        cx0 = cx.detach().clone()

        for step in range(update_timestep):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

            # Rollout is pure data collection - no gradients here. Gradients are
            # produced later by replaying these transitions with the current params.
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
            target_y = placement_clamped[:, 1] * MAX_Y_AI

            action = {
                "card_index": card_idx.cpu().numpy(),
                "target_x": target_x.cpu().numpy().reshape(num_envs, 1),
                "target_y": target_y.cpu().numpy().reshape(num_envs, 1)
            }

            next_obs, step_rewards, terminateds, truncateds, _ = envs.step(action)
            dones = terminateds | truncateds

            # Dense shaping term. On the step right after an episode ended, the vector env
            # has auto-reset that env, so prev_obs belongs to the finished episode and the
            # HP delta would be a huge spurious spike (fresh full-HP board vs destroyed
            # board). Zero the shaping there so only the real +/-0 reset reward remains.
            shaping = compute_shaping(next_obs, prev_obs)
            shaping = shaping * (1.0 - prev_dones)
            shaped_rewards = step_rewards + shaping
            ep_rewards += shaped_rewards
            ep_shaping += shaping
            ep_steps += 1

            mask = torch.tensor(1.0 - dones, dtype=torch.float32).to(device)

            # Store the transition (everything detached) for the PPO update
            obs_buffer.append(obs_tensor)
            card_actions_buffer.append(card_idx)
            placement_actions_buffer.append(placement_sample)
            logprobs_buffer.append(total_logprob)
            values_buffer.append(state_value.squeeze(-1))
            rewards_buffer.append(torch.tensor(shaped_rewards, dtype=torch.float32).to(device))
            masks_buffer.append(mask)

            # Reset hidden states for environments whose episode just ended
            mask_tensor = mask.unsqueeze(1)
            hx = hx * mask_tensor
            cx = cx * mask_tensor

            for i, done in enumerate(dones):
                if done:

                    # Log the episode reward
                    reward_history.append(ep_rewards[i])
                    shaping_history.append(ep_shaping[i])
                    ep_len_history.append(ep_steps[i])
                    # next_obs at the done step is the episode's true final board
                    # (gymnasium next-step autoreset), so end-state metrics read from it.
                    ally_end, enemy_end = building_hp_end(next_obs[i])
                    ally_bldg_end_history.append(ally_end)
                    enemy_bldg_end_history.append(enemy_end)
                    ep_rewards[i] = 0
                    ep_shaping[i] = 0
                    ep_steps[i] = 0
                    episodes_completed += 1

                    # step_rewards carries the raw (unshaped) engine reward: +1 win, -1 loss, 0 timeout/draw
                    if step_rewards[i] > 0.5:
                        outcome_history.append(1)
                    elif step_rewards[i] < -0.5:
                        outcome_history.append(-1)
                    else:
                        outcome_history.append(0)

                    if episodes_completed % 10 == 0:
                        outcomes = np.array(outcome_history)
                        wins = int((outcomes == 1).sum())
                        losses = int((outcomes == -1).sum())
                        draws = int((outcomes == 0).sum())
                        n = len(outcomes)
                        decisive_wr = wins / (wins + losses) if (wins + losses) > 0 else 0.0
                        avg_reward = np.mean(reward_history)
                        avg_shaping = np.mean(shaping_history)
                        print(f"Episodes: {episodes_completed} | Avg(50): {avg_reward:.2f} | W/L/D: {wins/n:.2f}/{losses/n:.2f}/{draws/n:.2f} | Decisive: {decisive_wr:.2f} | Stage: {curriculum_stage} | Entropy: {current_entropy_coef:.4f}")
                        writer.add_scalar("Training/Avg_Reward_50", avg_reward, episodes_completed)
                        writer.add_scalar("Reward/Episode_Shaping_Sum", avg_shaping, episodes_completed)
                        writer.add_scalar("Training/Win_Rate_100", wins / n, episodes_completed)
                        # Full outcome split: a rising win share should come out of the
                        # LOSS share (getting stronger) or the DRAW share (closing games).
                        writer.add_scalar("Rates/Loss_100", losses / n, episodes_completed)
                        writer.add_scalar("Rates/Draw_100", draws / n, episodes_completed)
                        # The headline progress metric and the curriculum gate:
                        # of the games that got decided, how many did we win?
                        writer.add_scalar("Rates/Decisive_Win_100", decisive_wr, episodes_completed)
                        # Learning to close games shows up as shorter episodes and less
                        # enemy building HP left standing at the end.
                        writer.add_scalar("Progress/Episode_Length_50", np.mean(ep_len_history), episodes_completed)
                        writer.add_scalar("Progress/Enemy_Building_HP_End_50", np.mean(enemy_bldg_end_history), episodes_completed)
                        writer.add_scalar("Progress/Ally_Building_HP_End_50", np.mean(ally_bldg_end_history), episodes_completed)
                        writer.add_scalar("Training/Curriculum_Stage", curriculum_stage, episodes_completed)
                        writer.add_scalar("Training/Entropy_Coef", current_entropy_coef, episodes_completed)

                    # --- Curriculum advancement: escalate the opponent once the agent
                    # consistently wins the games that get DECIDED (draws excluded) ---
                    stage_threshold = CURRICULUM_STAGES[curriculum_stage]["win_rate_threshold"]
                    if (stage_threshold is not None
                            and len(outcome_history) == outcome_history.maxlen
                            and curriculum_stage + 1 < len(CURRICULUM_STAGES)):
                        outcomes = np.array(outcome_history)
                        wins = int((outcomes == 1).sum())
                        losses = int((outcomes == -1).sum())
                        decided = wins + losses
                        if decided >= MIN_DECIDED_FOR_ADVANCE and wins / decided >= stage_threshold:
                            curriculum_stage += 1
                            new_multiplier = CURRICULUM_STAGES[curriculum_stage]["opp_elixir_multiplier"]
                            envs.call("set_opponent_elixir_multiplier", new_multiplier)
                            outcome_history.clear()
                            stage_start_episode = episodes_completed   # Reset entropy decay clock -> re-boost exploration (improvement #5)
                            print(f">>> Curriculum advanced to stage {curriculum_stage} (opp_elixir_multiplier={new_multiplier}) - entropy re-boosted")
                            writer.add_scalar("Training/Curriculum_Stage", curriculum_stage, episodes_completed)
            
            obs = next_obs
            prev_obs = obs
            prev_dones = dones

        # --- PPO Update: GAE advantages + multiple epochs over env-minibatches ---
        obs_seq = torch.stack(obs_buffer)                              # (T, N, obs_dim)
        card_actions_seq = torch.stack(card_actions_buffer)            # (T, N)
        placement_actions_seq = torch.stack(placement_actions_buffer)  # (T, N, 2)
        old_logprobs_seq = torch.stack(logprobs_buffer)                # (T, N)
        values_seq = torch.stack(values_buffer)                        # (T, N)  (old critic values)
        rewards_seq = torch.stack(rewards_buffer)                      # (T, N)
        masks_seq = torch.stack(masks_buffer)                          # (T, N)

        # Watch how often the agent chooses to wait (action 4 = no-op). Near-1.0 means
        # it collapsed into total passivity (draw > loss); near-0.0 means it still
        # can't hold elixir. Healthy play should settle somewhere in between.
        writer.add_scalar("Policy/Noop_Fraction", (card_actions_seq == 4).float().mean().item(), episodes_completed)

        # Bootstrap value for the state right after the last stored step
        with torch.no_grad():
            next_obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            _, _, _, next_value, _ = net(next_obs_tensor, (hx, cx))
            next_value = next_value.squeeze(-1)

        # Generalized Advantage Estimation (GAE-lambda)
        advantages_seq = torch.zeros_like(rewards_seq)
        gae = torch.zeros(num_envs).to(device)
        for t in reversed(range(update_timestep)):
            next_val = next_value if t == update_timestep - 1 else values_seq[t + 1]
            delta = rewards_seq[t] + gamma * next_val * masks_seq[t] - values_seq[t]
            gae = delta + gamma * gae_lambda * masks_seq[t] * gae
            advantages_seq[t] = gae

        # Critic targets are the raw returns; only advantages are normalized (fix #3)
        returns_seq = advantages_seq + values_seq
        adv_norm_seq = (advantages_seq - advantages_seq.mean()) / (advantages_seq.std() + 1e-8)

        env_indices = np.arange(num_envs)
        mb_size = max(1, num_envs // num_minibatches)

        for epoch in range(ppo_epochs):
            np.random.shuffle(env_indices)
            for mb_start in range(0, num_envs, mb_size):
                mb_env = env_indices[mb_start:mb_start + mb_size]
                if len(mb_env) == 0:
                    continue
                mb_env_t = torch.as_tensor(mb_env, dtype=torch.long, device=device)
                mb = mb_env_t.shape[0]

                # Batch the (non-recurrent) CNN + scalar feature extraction over ALL
                # timesteps at once, then loop only the cheap LSTMCell. This avoids
                # calling the CNN update_timestep times on a tiny batch and is the main
                # speedup of the update step.
                mb_obs = obs_seq[:, mb_env_t]                                  # (T, mb, obs_dim)
                feats_seq = net.extract_features(mb_obs.reshape(update_timestep * mb, -1))
                feats_seq = feats_seq.view(update_timestep, mb, -1)           # (T, mb, feat_dim)

                # Replay this minibatch's env sequences through the LSTM with current params
                rhx = hx0[mb_env_t]
                rcx = cx0[mb_env_t]
                new_logprobs = []
                new_values = []
                new_entropies = []
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

                new_logprobs = torch.stack(new_logprobs)   # (T, mb)
                new_values = torch.stack(new_values)       # (T, mb)
                new_entropies = torch.stack(new_entropies)

                mb_adv = adv_norm_seq[:, mb_env_t]
                mb_ret = returns_seq[:, mb_env_t]
                mb_old_logprobs = old_logprobs_seq[:, mb_env_t]

                ratios = torch.exp(new_logprobs - mb_old_logprobs)
                surr1 = ratios * mb_adv
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * mb_adv

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(new_values, mb_ret)
                entropy_bonus = new_entropies.mean()
                loss = actor_loss + 0.5 * critic_loss - (current_entropy_coef * entropy_bonus)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                optimizer.step()

        obs_buffer.clear()
        card_actions_buffer.clear()
        placement_actions_buffer.clear()
        logprobs_buffer.clear()
        values_buffer.clear()
        rewards_buffer.clear()
        masks_buffer.clear()

        hx, cx = hx.detach(), cx.detach()
        
        # Saves
        if episodes_completed - last_save_ep >= 500:
            torch.save(net.state_dict(), weight_path)
            print(f">>> Weights saved successfully to {weight_path}")
            last_save_ep = episodes_completed

        # Generate Replay (Standalone test env to avoid corrupting async processes)
        if episodes_completed - last_replay_ep >= 1000:
            print(f"Generating replay video for episode {episodes_completed}...")
            test_env = gym_wrapper.MicroRoyaleEnv()
            t_obs, _ = test_env.reset()
            t_hx = torch.zeros(1, 256).to(device)
            t_cx = torch.zeros(1, 256).to(device)
            t_done = False
            while not t_done:
                t_obs_tensor = torch.tensor(t_obs, dtype=torch.float32).unsqueeze(0).to(device)
                t_logits, t_norm, _, _, (t_hx, t_cx) = net(t_obs_tensor, (t_hx, t_cx))
                t_idx = Categorical(logits=t_logits).sample()
                t_action = {
                    "card_index": np.array([t_idx.item()]),
                    "target_x": np.array([torch.clamp(t_norm[0,0]*MAX_X, 0.0, MAX_X).item()]),
                    "target_y": np.array([torch.clamp(t_norm[0,1]*MAX_Y_AI, 0.0, MAX_Y_AI).item()])
                }
                t_obs, _, t_terminated, t_truncated, _ = test_env.step(t_action)
                t_done = t_terminated or t_truncated
            test_env.game.save_log(f"replays/replay_ep{episodes_completed}.json")
            last_replay_ep = episodes_completed

    writer.close()

if __name__ == "__main__":
    # Prevent safe pickling errors in Windows multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    train_ppo()