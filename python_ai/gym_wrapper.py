import gymnasium as gym
from gymnasium import spaces
import numpy as np

import clash_royale_env

# Real-meta Balloon Freeze deck, replacing the earlier Giant Beatdown archetype
# entirely -- deliberate full restart (fresh net, not resumed), not a tune-up.
# Reasons: (1) the old deck was picked back when CardRegistry had far fewer
# cards implemented and doesn't reflect the roster available now; (2) pipeline
# #1's phase 2 (randomized opponent decks, see train.py's PHASE2_WIN_RATE_GATE)
# never actually ran -- the old model only ever played mirror-deck self-play,
# a real overfitting risk; (3) the old checkpoint accumulated architecture
# drift across several unrelated engine changes this project went through
# (map geometry, champion-ability action space); (4) the old model never saw
# the enlarged board/back-row placement area at all -- trained entirely before
# that change.
#
# 44=Baby Dragon, 0=Knight, 56=Inferno Dragon, 45=Balloon (win condition),
# 22=Bowler, 106=Tornado, 107=Freeze, 101=Barbarian Barrel.
#
# All 8 confirmed against CardRegistry.h + a full green ClashRoyaleTests run
# (1759 assertions/376 cases) before committing to this deck. Two known,
# deliberate engine simplifications worth remembering if this deck's learning
# curve ever looks off: Bowler's hit does NOT knock enemies back here (real
# card does) -- AreaSpell's knockback has no equivalent on the troop-attack
# path this engine's splash/line-splash share, a structural limitation, not a
# bug specific to Bowler. Balloon's death-explosion fires unconditionally and
# instantly on death (real card: only if killed before its bomb-drop
# animation finishes, with a short delay) -- close enough to be a non-issue in
# practice, not a missing mechanic.
#
# Exposed at module level so other scripts (e.g. train_selfplay.py, which
# builds its ClashRoyaleEnv directly instead of through this wrapper) can
# import the same deck instead of duplicating/drifting from this literal.
#
# Classic Giant-beatdown archetype (Valkyrie, Archers, Minions, Cannon,
# Fireball, Giant, Musketeer, Mini PEKKA) -- deliberately replaces the old
# deck's two building-targeter tower-shredders (Inferno Dragon/Balloon) with
# a single, non-ramping win condition (Giant just tanks; it doesn't melt a
# tower and chain onto the next one the way Inferno Dragon's ramping damage
# did -- see the replay autopsy this responds to), and gives real cheap
# defensive tools (Cannon, Minions) that the old deck had none of at all.
DEFAULT_DECK = [10, 1, 41, 25, 7, 2, 6, 5]


def get_all_card_ids():
    """All ids CardRegistry currently has registered, derived live from the
    engine instead of a hardcoded range+exclusion list. That kind of list goes
    stale the moment a card is added to (or removed from) CardRegistry.h --
    already happened once: a hardcoded range(46) pool silently stopped covering
    new cards once the roster grew past 45, and a hand-maintained exclusion
    list is just as easy to get wrong in the other direction (mistaking real
    registered ids for gaps). See CardRegistry.h's getAllCardIds() free function."""
    return clash_royale_env.get_all_card_ids()


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
        # Tower Troops: per-match config like the deck itself, not a per-step
        # action -- see GameManager's constructor. NONE (the default)
        # reproduces the original hardcoded Princess Tower unchanged.
        ai_tower_troop = env_config.get("ai_tower_troop", clash_royale_env.TowerTroopType.NONE)
        opp_tower_troop = env_config.get("opp_tower_troop", clash_royale_env.TowerTroopType.NONE)

        self.game = clash_royale_env.ClashRoyaleEnv(ai_deck, self.opp_deck, max_ticks, ai_tower_troop, opp_tower_troop)
        self.game.set_opponent_elixir_multiplier(opp_elixir_multiplier)

        self.action_space = spaces.Dict({
            # אינדקס HAND_SIZE = no-op (לא לשחק קלף הצעד הזה). המנוע מתעלם מ-cardIndex
            # מחוץ ל-[0,HAND_SIZE), כך שהסוכן יכול סוף-סוף לאגור אליקסיר במקום להיות
            # מאולץ לשחק. הגבולות נשלפים חי מהמנוע (לא hardcoded) כדי שלא יהיה
            # צורך לסנכרן ידנית אם גודל הלוח/היד ישתנה בצד ה-C++.
            "card_index": spaces.Discrete(clash_royale_env.ClashRoyaleEnv.HAND_SIZE + 1),
            "target_x": spaces.Box(low=0.0, high=self.game.get_max_placement_x(), shape=(1,), dtype=np.float32),
            "target_y": spaces.Box(low=0.0, high=self.game.get_own_half_max_y(), shape=(1,), dtype=np.float32),
            # הפעלת יכולת צ'מפיון -- שתי משבצות עצמאיות (1=Heroic, 2=Wild Card,
            # ראו CardRegistry::validateDeckSlots), כי דק יכול להכיל עד 2
            # צ'מפיונים בו-זמנית. כל אחת: 0 = לא להפעיל, 1 = להפעיל עכשיו אם יש
            # צ'מפיון פרוס באותה משבצת, לא ב-cooldown, ויש מספיק אליקסיר (אחרת
            # no-op שקט, כמו ה-no-op של card_index). ברירת המחדל False בכל מקום
            # שלא מעביר את המפתח.
            "activate_ability_slot1": spaces.Discrete(2),
            "activate_ability_slot2": spaces.Discrete(2),
        })
        
        obs_size = self.game.observation_size()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.randomize_opp_deck:
            import random
            self.game.set_opponent_deck(random.sample(get_all_card_ids(), 8))
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
        # .get(..., 0): callers that build this dict by hand without these
        # keys (e.g. train.py's existing PPO loop) must not KeyError here.
        activate_ability_slot1 = bool(_to_scalar(action.get("activate_ability_slot1", 0)))
        activate_ability_slot2 = bool(_to_scalar(action.get("activate_ability_slot2", 0)))

        step_result = self.game.step(card_idx, target_x, target_y, skip_frames,
                                      activate_ability_slot1, activate_ability_slot2)
        
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
            # Not part of observation_space -- see ClashEnv.h's own comment
            # on why champion-ability state stays out of the flat
            # observation vector (would break model.py's fixed scalar_size
            # formula) rather than growing it. Two independent slots -- see
            # activate_ability_slot1/2's own comment above.
            "champion_ability_slot1_ready": self.game.is_champion_ability_ready(0, 1),
            "champion_ability_slot2_ready": self.game.is_champion_ability_ready(0, 2),
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