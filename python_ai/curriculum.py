import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import random

register(
    id='MicroRoyale-v0',
    entry_point='gym_wrapper:MicroRoyaleEnv',
)

def run_curriculum_episode(env, difficulty_level):
    """
    מריץ אפיזודת אימון בודדת עם גלי אויבים המותאמים לרמת הקושי.
    """
    obs, info = env.reset()
    done = False
    tick = 0
    
    print(f"--- Starting Episode: Level [{difficulty_level.upper()}] ---")
    
    while not done:
        # בשלב זה, הסוכן עדיין לא מאומן, אז נבצע פעולות אקראיות חוקיות.
        # בהמשך, השורה הזו תוחלף ב: action = model.predict(obs)
        action = env.action_space.sample() 
        
        # ביצוע הפעולה בסביבה
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- תכנית הלימודים: הזרקת אויבים מתוזמנת ---
        
        # רמה 1: קל - גובלינים בודדים כל 300 טיקים כדי ללמד תגובה בסיסית
        if difficulty_level == "easy":
            if tick > 0 and tick % 300 == 0:
                env.unwrapped.game.inject_enemy(4, 14.0, 25.0) # 4 = Goblins
                
        # רמה 2: בינוני - התקפות שריון (Knight) שדורשות נזק גבוה יותר
        elif difficulty_level == "medium":
            if tick > 0 and tick % 250 == 0:
                lane_x = random.choice([3.0, 14.0]) # תקיפה משני הנתיבים
                env.unwrapped.game.inject_enemy(0, lane_x, 25.0) # 0 = Knight
                
        # רמה 3: קשה - דחיפות משולבות (Tank + Support) ללמד תעדוף מטרות מרחבי
        elif difficulty_level == "hard":
            if tick > 0 and tick % 400 == 0:
                lane_x = random.choice([3.0, 14.0])
                env.unwrapped.game.inject_enemy(2, lane_x, 25.0) # 2 = Giant
                env.unwrapped.game.inject_enemy(6, lane_x, 28.0) # 6 = Musketeer (מאחורי הענק)
                
        done = terminated or truncated
        tick += 1
        
    print(f"Episode finished at tick {tick}.")

if __name__ == "__main__":
    # נגדיר קונפיגורציה ייעודית לאימון הגנתי.
    # אנחנו ממלאים את החפיסה של היריב בקלפים יקרים או חלשים (למשל סקלטים - 24)
    # כדי שהפונקציה המקורית opponentTurn() לא תפריע לגלים שאנחנו מזריקים.
    defense_config = {
        "ai_deck": [25, 0, 6, 24, 7, 34, 29, 15],  # חפיסה הגנתית (תותח, אביר, מוסקטר...)
        "opp_deck": [24, 24, 24, 24, 24, 24, 24, 24], # חפיסת "דמה" ליריב
        "max_ticks": 1800
    }
    
    # יצירת הסביבה עם הקונפיגורציה
    env = gym.make('MicroRoyale-v0', env_config=defense_config)
    
    # הרצת השלבים בזה אחר זה
    run_curriculum_episode(env, "easy")
    run_curriculum_episode(env, "medium")
    run_curriculum_episode(env, "hard")