import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from torch.distributions import Categorical, Normal
from torch.utils.tensorboard import SummaryWriter
from collections import deque

import gym_wrapper
from model import MicroRoyaleNet

def compute_dense_reward(obs, prev_obs, step_reward):
    """
    Vectorized dense reward computation.
    obs: (num_envs, obs_size)
    prev_obs: (num_envs, obs_size)
    step_reward: (num_envs,)
    """
    if prev_obs is None:
        return step_reward
        
    num_envs = obs.shape[0]
    spatial_size = 5 * 32 * 18
    
    obs_spatial = obs[:, :spatial_size].reshape(num_envs, 5, 32, 18)
    prev_spatial = prev_obs[:, :spatial_size].reshape(num_envs, 5, 32, 18)
    
    ally_troops_hp = obs_spatial[:, 0].sum(axis=(1, 2))
    enemy_troops_hp = obs_spatial[:, 1].sum(axis=(1, 2))
    ally_bldg_hp = obs_spatial[:, 2].sum(axis=(1, 2))
    enemy_bldg_hp = obs_spatial[:, 3].sum(axis=(1, 2))
    
    prev_ally_troops = prev_spatial[:, 0].sum(axis=(1, 2))
    prev_enemy_troops = prev_spatial[:, 1].sum(axis=(1, 2))
    prev_ally_bldg = prev_spatial[:, 2].sum(axis=(1, 2))
    prev_enemy_bldg = prev_spatial[:, 3].sum(axis=(1, 2))
    
    d_ally_bldg = ally_bldg_hp - prev_ally_bldg
    d_enemy_bldg = enemy_bldg_hp - prev_enemy_bldg
    d_ally_troops = ally_troops_hp - prev_ally_troops
    d_enemy_troops = enemy_troops_hp - prev_enemy_troops
    
    w_bldg = 0.5
    w_troops = 0.1
    
    dense_reward = step_reward + \
                   (w_bldg * d_ally_bldg) - (w_bldg * d_enemy_bldg) + \
                   (w_troops * d_ally_troops) - (w_troops * d_enemy_troops)
                   
    return dense_reward

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
        net.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"Loaded existing weights from {weight_path}")
    
    # PPO Hyperparameters
    gamma = 0.99
    eps_clip = 0.2
    update_timestep = 500  # Number of steps PER CORE before update
    
    initial_entropy_coef = 0.1
    min_entropy_coef = 0.001
    entropy_decay_rate = 0.9995
    
    MAX_X = 18.0
    MAX_Y_AI = 14.5
    
    logprobs_buffer = []
    values_buffer = []
    rewards_buffer = []
    masks_buffer = []
    entropies_buffer = []
    
    reward_history = deque(maxlen=50)
    
    obs, _ = envs.reset()
    prev_obs = None
    hx = torch.zeros(num_envs, 256).to(device)
    cx = torch.zeros(num_envs, 256).to(device)
    
    episodes_completed = 0
    last_save_ep = 0
    last_replay_ep = 0
    ep_rewards = np.zeros(num_envs)
    
    print(f"Training started on {num_envs} CPU cores simultaneously!")
    
    while episodes_completed < 50000:
        current_entropy_coef = max(min_entropy_coef, initial_entropy_coef * (entropy_decay_rate ** episodes_completed))
        
        for step in range(update_timestep):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            
            card_logits, placement_norm, state_value, (hx, cx) = net(obs_tensor, (hx, cx))
            
            card_dist = Categorical(logits=card_logits)
            card_idx = card_dist.sample()
            
            noise_x = torch.randn(num_envs).to(device) * 0.05
            noise_y = torch.randn(num_envs).to(device) * 0.05
            
            target_x = torch.clamp((placement_norm[:, 0] + noise_x) * MAX_X, 0.0, MAX_X)
            target_y = torch.clamp((placement_norm[:, 1] + noise_y) * MAX_Y_AI, 0.0, MAX_Y_AI)
            
            action = {
                "card_index": card_idx.detach().cpu().numpy(),
                "target_x": target_x.detach().cpu().numpy().reshape(num_envs, 1),
                "target_y": target_y.detach().cpu().numpy().reshape(num_envs, 1)
            }
            
            next_obs, step_rewards, terminateds, truncateds, _ = envs.step(action)
            dones = terminateds | truncateds
            
            shaped_rewards = compute_dense_reward(next_obs, prev_obs, step_rewards)
            ep_rewards += shaped_rewards
            
            logprobs_buffer.append(card_dist.log_prob(card_idx))
            values_buffer.append(state_value.squeeze(-1))
            rewards_buffer.append(torch.tensor(shaped_rewards, dtype=torch.float32).to(device))
            masks_buffer.append(torch.tensor(1.0 - dones, dtype=torch.float32).to(device))
            entropies_buffer.append(card_dist.entropy())
            
            # Out-of-place mask to reset hidden states for completed environments
            mask_tensor = torch.tensor(1.0 - dones, dtype=torch.float32).unsqueeze(1).to(device)
            hx = hx * mask_tensor
            cx = cx * mask_tensor

            for i, done in enumerate(dones):
                if done:
                    
                    # Log the episode reward
                    reward_history.append(ep_rewards[i])
                    ep_rewards[i] = 0
                    episodes_completed += 1
                    
                    if episodes_completed % 10 == 0:
                        avg_reward = np.mean(reward_history)
                        print(f"Episodes: {episodes_completed} | Avg(50): {avg_reward:.2f} | Entropy: {current_entropy_coef:.4f}")
                        writer.add_scalar("Training/Avg_Reward_50", avg_reward, episodes_completed)
                        writer.add_scalar("Training/Entropy_Coef", current_entropy_coef, episodes_completed)
            
            obs = next_obs
            prev_obs = obs
            
        # --- PPO Update ---
        returns = []
        discounted_sum = torch.zeros(num_envs).to(device)
        for reward, mask in zip(reversed(rewards_buffer), reversed(masks_buffer)):
            discounted_sum = reward + (gamma * discounted_sum * mask)
            returns.insert(0, discounted_sum)
        
        returns = torch.stack(returns) # shape: (update_timestep, num_envs)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        
        old_logprobs = torch.stack(logprobs_buffer).detach() # shape: (update_timestep, num_envs)
        old_values = torch.stack(values_buffer).detach()
        
        advantages = returns - old_values
        
        ratios = torch.exp(torch.stack(logprobs_buffer) - old_logprobs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
        
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = nn.MSELoss()(torch.stack(values_buffer), returns)
        
        mean_entropy = torch.stack(entropies_buffer).mean()
        loss = actor_loss + 0.5 * critic_loss - (current_entropy_coef * mean_entropy)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        logprobs_buffer.clear()
        values_buffer.clear()
        rewards_buffer.clear()
        masks_buffer.clear()
        entropies_buffer.clear()
        
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
                t_logits, t_norm, _, (t_hx, t_cx) = net(t_obs_tensor, (t_hx, t_cx))
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