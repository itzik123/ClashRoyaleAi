import gymnasium as gym
from gymnasium import spaces
import numpy as np

import clash_royale_env

# Giant Beatdown (tank + support), not the old Hog cycle deck: measured in
# practice that the cycle deck (Hog Rider + Skeletons + Zap) got training stuck
# around 40-50% win rate for 2800+ episodes with no improving trend -- its good-
# play window (precise cycle timing, counters) was too narrow for random
# exploration to ever stumble into. Tank+support gives a coarse heuristic
# ("push the tank forward, support behind it") that already yields reasonable
# reward even with imperfect execution, so it's significantly easier to
# bootstrap. 2=Giant, 5=Mini PEKKA, 35=Electro Wizard, 7=Fireball, 33=The Log,
# 24=Skeletons, 40=Ice Golem, 25=Cannon.
# Replaces an earlier draft of this same archetype that used Wizard(11)/
# Musketeer(6) for anti-air and Zap(29) as the light spell -- both were quietly
# broken for the role: Musketeer/Wizard never got .withTargetsAir() in
# CardRegistry.h (ground-only here despite hitting air in the real game), and
# Zap here is damage-only with no stun (its actual niche in the real game).
# Electro Wizard/The Log are the two cards in this roster whose real-game
# signature mechanic (air-targeting+split+stun; ground-only roll) is actually
# modeled. Exposed at module level so other scripts (e.g. train_selfplay.py,
# which builds its ClashRoyaleEnv directly instead of through this wrapper)
# can import the same deck instead of duplicating/drifting from this literal.
DEFAULT_DECK = [2, 5, 35, 7, 33, 24, 40, 25]

class MicroRoyaleEnv(gym.Env):
    def __init__(self, env_config=None):
        super().__init__()
        
        # אם לא הועברה קונפיגורציה, ניצור מילון ריק
        if env_config is None:
            env_config = {}
            
        # משיכת ההגדרות עם ערכי ברירת מחדל הגיוניים, ללא Hard-coding מחייב
        ai_deck = env_config.get("ai_deck", list(DEFAULT_DECK))
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
            # 0..45 minus 16/37/38 (never defined in CardRegistry -- see
            # test_card_registry.cpp's "Card ids that were never defined" test).
            # Previously stopped at 40, silently excluding the 5 flying cards
            # (41-45: Minions, Minion Horde, Mega Minion, Baby Dragon, Balloon)
            # from ever showing up in a randomized opponent deck.
            AVAILABLE_CARDS = [i for i in range(46) if i not in (16, 37, 38)]
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
            # Cumulative elixir spent this match (sum of played cards' cost) --
            # feeds train.py's elixir-trade shaping term (reward for making the
            # enemy spend more than we do, independent of the damage itself).
            "team0_elixir_spent": self.game.get_elixir_spent(0),
            "team1_elixir_spent": self.game.get_elixir_spent(1),
        }

        return obs, reward, terminated, truncated, info

    def set_opponent_deck(self, deck):
        # Also updates self.opp_deck (not just the live game instance) --
        # otherwise the very next auto-reset (reset() always re-applies
        # self.opp_deck when randomize_opp_deck is False) would silently
        # revert this to whatever deck the env was constructed with.
        self.opp_deck = list(deck)
        self.game.set_opponent_deck(self.opp_deck)

    def set_opponent_elixir_multiplier(self, multiplier):
        self.game.set_opponent_elixir_multiplier(multiplier)