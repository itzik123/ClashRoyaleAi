import gymnasium as gym
from gymnasium import spaces
import numpy as np

import clash_royale_env

class MicroRoyaleEnv(gym.Env):
    def __init__(self, env_config=None):
        super().__init__()
        
        # אם לא הועברה קונפיגורציה, ניצור מילון ריק
        if env_config is None:
            env_config = {}
            
        # משיכת ההגדרות עם ערכי ברירת מחדל הגיוניים, ללא Hard-coding מחייב
        ai_deck = env_config.get("ai_deck", [15, 6, 0, 25, 7, 24, 34, 29])
        opp_deck = env_config.get("opp_deck", [8, 2, 38, 13, 37, 32, 29, 12])
        max_ticks = env_config.get("max_ticks", 3600)
        
        self.game = clash_royale_env.ClashRoyaleEnv(ai_deck, opp_deck, max_ticks)
        
        self.action_space = spaces.Dict({
            "card_index": spaces.Discrete(4),
            "target_x": spaces.Box(low=0.0, high=18.0, shape=(1,), dtype=np.float32),
            "target_y": spaces.Box(low=0.0, high=14.5, shape=(1,), dtype=np.float32)
        })
        
        obs_size = self.game.observation_size()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        import random
        AVAILABLE_CARDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40]
        self.game.set_opponent_deck(random.sample(AVAILABLE_CARDS, 8))
        
        obs_list = self.game.reset()
        obs = np.array(obs_list, dtype=np.float32)
        return obs, {}

    def step(self, action, skip_frames=10):
        def _to_scalar(val):
            if hasattr(val, "item"):
                return val.item()
            if isinstance(val, (list, tuple, np.ndarray)):
                return val[0]
            return val

        card_idx = int(_to_scalar(action["card_index"]))
        target_x = float(_to_scalar(action["target_x"]))
        target_y = float(_to_scalar(action["target_y"]))
        
        step_result = self.game.step(card_idx, target_x, target_y, skip_frames)
        
        obs = np.array(step_result.observation, dtype=np.float32)
        reward = float(step_result.reward)
        terminated = bool(step_result.done)
        truncated = False
        
        info = {
            "elixir": self.game.get_elixir(),
            "hand": self.game.get_hand()
        }
        
        return obs, reward, terminated, truncated, info

    def set_opponent_deck(self, deck):
        self.game.set_opponent_deck(deck)