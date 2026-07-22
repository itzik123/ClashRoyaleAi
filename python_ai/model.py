import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import clash_royale_env

class MicroRoyaleNet(nn.Module):
    # 9 ערוצים: 0-3 כוחות שלנו (קרבי/טווח/טנק/מבנים), 4-7 אותו דבר ליריב, 8 נהר/גשרים
    def __init__(self, channels=None, board_width=None, board_height=None, hand_size=None, num_card_ids=None):
        super(MicroRoyaleNet, self).__init__()

        # ברירות מחדל נשלפות חי מהמנוע המקומפל (לא hardcoded) -- כל שינוי גודל
        # לוח/מספר קלפים בצד ה-C++ מתפשט לכאן אוטומטית, בלי צורך בעדכון ידני
        # תואם בקובץ הזה. זו בדיוק סוג הסחיפה (drift) שגרמה לקריסות אימון
        # בפועל בפרויקט הזה יותר מפעם אחת לפני התיקון הזה.
        channels = channels if channels is not None else clash_royale_env.ClashRoyaleEnv.NUM_CHANNELS
        board_width = board_width if board_width is not None else clash_royale_env.ClashRoyaleEnv.BOARD_WIDTH
        board_height = board_height if board_height is not None else clash_royale_env.ClashRoyaleEnv.BOARD_HEIGHT
        hand_size = hand_size if hand_size is not None else clash_royale_env.ClashRoyaleEnv.HAND_SIZE
        num_card_ids = num_card_ids if num_card_ids is not None else clash_royale_env.ClashRoyaleEnv.NUM_CARD_IDS

        self.channels = channels
        self.board_width = board_width
        self.board_height = board_height

        # גודל המטריצה השטוחה המגיעה מ-ClashEnv
        self.spatial_size = channels * board_height * board_width
        # החלק הסקלרי: אליקסיר + 4 עלויות + 4 one-hot של זהות קלף (num_card_ids ערכים כל אחד)
        self.scalar_size = 1 + hand_size + hand_size * num_card_ids
        
        # ==========================================
        # 1. חילוץ תכונות מרחבי (CNN)
        # קלט: (Batch, 9, 34, 18)
        # ==========================================
        # ceil_mode=True בשני ה-MaxPool: board_height=34 לא מתחלק נקי פי 4
        # (34 -> 17 -> 8 עם floor רגיל, מה שהיה מוחק שורה שלמה -- בדיוק השורה
        # האחורית החדשה ליד מגדל המלך, שזה כל הטעם בשינוי הזה). עם ceil_mode
        # שום שורה/עמודה לא נופלת בשקט, רק התמונה המרחבית קצת יותר גדולה.
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),

            nn.Flatten()
        )

        # חישוב ממד הפלט של ה-CNN לאחר הפולינג (ceil פעמיים, תואם ceil_mode=True למעלה)
        pooled_h = math.ceil(math.ceil(board_height / 2) / 2)
        pooled_w = math.ceil(math.ceil(board_width / 2) / 2)
        self.cnn_out_dim = 32 * pooled_h * pooled_w
        
        # ==========================================
        # 2. חילוץ תכונות סקלרי (MLP)
        # קלט: אליקסיר + עלויות + זהות הקלפים ביד (one-hot לכל משבצת)
        # ==========================================
        self.scalar_mlp = nn.Sequential(
            nn.Linear(self.scalar_size, 64),
            nn.ReLU()
        )

        # ==========================================
        # 3. שכבת זיכרון (LSTM)
        # ==========================================
        self.lstm_input_dim = self.cnn_out_dim + 64
        # אנו משתמשים ב-LSTMCell כדי שנוכל לשלוט על הפעימות (Ticks) ידנית בלולאת הסביבה
        self.lstm = nn.LSTMCell(self.lstm_input_dim, 256)
        
        # ==========================================
        # 4. ראשי הפעולה - Actor Heads
        # ==========================================
        # א. ראש בחירת הקלף (התפלגות קטגוריאלית)
        # hand_size משבצות יד + פעולה אחת נוספת = no-op (המתנה/אגירת אליקסיר).
        # המנוע מתעלם מ-cardIndex מחוץ ל-[0,hand_size) כך שאין צורך בשינוי C++.
        self.card_head = nn.Linear(256, hand_size + 1)
        
        # ב. ראש המיקום במרחב (התפלגות גאוסיאנית / תחימה)
        # אנו מוציאים 2 ערכים, ונעביר אותם דרך פונקציית Sigmoid כדי לתחום אותם בין [0, 1]
        self.placement_head = nn.Linear(256, 2)
        # סטיית התקן של ההתפלגות הגאוסיאנית - פרמטר נלמד (state-independent), נדרש כדי
        # שיהיה ניתן לחשב log_prob ולתת gradient אמיתי לראש המיקום
        self.placement_log_std = nn.Parameter(torch.ones(2) * -2.0)
        
        # ==========================================
        # 5. ראש הערכת המצב - Critic Head
        # ==========================================
        self.value_head = nn.Linear(256, 1)

    def extract_features(self, obs):
        """
        חילוץ מאפיינים (CNN + MLP סקלרי) - החלק הלא-רקורנטי של הרשת.
        obs: (Batch, 2885) -> (Batch, lstm_input_dim)
        אפשר לקרוא לזה על באצ' ענק ומשוטח (T*N) כדי להריץ את ה-CNN פעם אחת
        במקום פעם לכל טיק - זהו הזירוז המרכזי של עדכון ה-PPO.
        """
        # פיצול הווקטור השטוח לחלק המרחבי ולחלק הסקלרי בהתאם לפונקציית observationSize() ב-C++
        spatial_obs = obs[:, :self.spatial_size].view(-1, self.channels, self.board_height, self.board_width)
        scalar_obs = obs[:, self.spatial_size:]

        cnn_features = self.cnn(spatial_obs)
        scalar_features = self.scalar_mlp(scalar_obs)

        return torch.cat((cnn_features, scalar_features), dim=1)

    def forward_from_features(self, features, hidden_state):
        """
        הצעד הרקורנטי (LSTM) + ראשי הפעולה, בהינתן מאפיינים שכבר חולצו.
        features: (Batch, lstm_input_dim)
        """
        hx, cx = self.lstm(features, hidden_state)

        card_logits = self.card_head(hx)
        placement_normalized = torch.sigmoid(self.placement_head(hx))
        placement_log_std = torch.clamp(self.placement_log_std, -4.0, 0.0).expand_as(placement_normalized)
        state_value = self.value_head(hx)

        return card_logits, placement_normalized, placement_log_std, state_value, (hx, cx)

    def forward(self, obs, hidden_state):
        """
        מעביר תצפית בודדת דרך הרשת (חילוץ מאפיינים ואז צעד רקורנטי).
        obs: טנזור בגודל (Batch, 2885)
        hidden_state: הסטייט הקודם של ה-LSTM -> (hx, cx)
        """
        return self.forward_from_features(self.extract_features(obs), hidden_state)