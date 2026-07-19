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
        # חפיסת Giant beatdown (טנק + תמיכה), לא חפיסת Hog cycle כמו קודם: נמדד
        # בפועל שעם חפיסת cycle (Hog Rider + Skeletons + Zap) האימון נתקע סביב
        # 40-50% win rate במשך 2800+ אפיזודות בלי מגמת שיפור - חלון המשחק הטוב
        # שם צר מדי (טיימינג cycle מדויק, קונטרות) כדי שחקירה אקראית תיתקל בו.
        # ענק+תמיכה נותן היוריסטיקה גסה ("טנק קדימה, תמיכה מאחוריו, דחוף") שכבר
        # מניבה תגמול סביר גם עם ביצוע לא-מושלם, ולכן קלה משמעותית ל-bootstrap.
        # 2=Giant, 11=Wizard, 6=Musketeer, 5=Mini PEKKA, 7=Fireball, 29=Zap,
        # 25=Cannon, 24=Skeletons.
        ai_deck = env_config.get("ai_deck", [2, 11, 6, 5, 7, 29, 25, 24])
        # ברירת מחדל: היריב משחק עם אותה חפיסה בדיוק (mirror match). נמדד אמפירית
        # שחפיסה רנדומלית מהמאגר חזקה בעשרות אחוזי win-rate מהחפיסה הקבועה (במנוע
        # הזה יחידות-ענק דורסות חפיסת cycle), כך שאימון מול חפיסות רנדומליות מציב
        # את הסוכן במשחק כמעט-אבוד מראש ואות הניצחון/הפסד נעלם. גיוון חפיסות שייך
        # לשלב קוריקולום מאוחר, אחרי שהסוכן לומד לנצח במשחק מאוזן.
        self.opp_deck = env_config.get("opp_deck", list(ai_deck))
        self.randomize_opp_deck = env_config.get("randomize_opp_deck", False)
        max_ticks = env_config.get("max_ticks", 3600)
        # וו לתכנית לימודים: מכפיל קצב האליקסיר של היריב (1.0 = רגיל, ערך גבוה מדמה יריב אגרסיבי/כמעט-בלתי-מוגבל)
        opp_elixir_multiplier = env_config.get("opp_elixir_multiplier", 1.0)

        self.game = clash_royale_env.ClashRoyaleEnv(ai_deck, self.opp_deck, max_ticks)
        self.game.set_opponent_elixir_multiplier(opp_elixir_multiplier)

        self.action_space = spaces.Dict({
            # אינדקס 4 = no-op (לא לשחק קלף הצעד הזה). המנוע מתעלם מ-cardIndex מחוץ
            # ל-[0,4), כך שהסוכן יכול סוף-סוף לאגור אליקסיר במקום להיות מאולץ לשחק.
            "card_index": spaces.Discrete(5),
            "target_x": spaces.Box(low=0.0, high=17.0, shape=(1,), dtype=np.float32),
            "target_y": spaces.Box(low=0.0, high=14.5, shape=(1,), dtype=np.float32)
        })
        
        obs_size = self.game.observation_size()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.randomize_opp_deck:
            import random
            AVAILABLE_CARDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40]
            self.game.set_opponent_deck(random.sample(AVAILABLE_CARDS, 8))
        else:
            self.game.set_opponent_deck(self.opp_deck)

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
            "hand": self.game.get_hand(),
            # Cumulative (this match, since reset) damage dealt BY each team,
            # split troop/building -- straight from the engine's MatchStatistics,
            # not inferred by diffing HP channels in the observation. train.py's
            # compute_shaping() diffs these itself to get a per-step delta, the
            # same way it used to diff raw HP.
            "team0_troop_damage": self.game.get_troop_damage_dealt(0),
            "team1_troop_damage": self.game.get_troop_damage_dealt(1),
            "team0_building_damage": self.game.get_building_damage_dealt(0),
            "team1_building_damage": self.game.get_building_damage_dealt(1),
        }

        return obs, reward, terminated, truncated, info

    def set_opponent_deck(self, deck):
        self.game.set_opponent_deck(deck)

    def set_opponent_elixir_multiplier(self, multiplier):
        self.game.set_opponent_elixir_multiplier(multiplier)