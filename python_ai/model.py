import torch
import torch.nn as nn
import torch.nn.functional as F

class MicroRoyaleNet(nn.Module):
    # 9 ערוצים: 0-3 כוחות שלנו (קרבי/טווח/טנק/מבנים), 4-7 אותו דבר ליריב, 8 נהר/גשרים
    def __init__(self, channels=9, board_width=18, board_height=32, hand_size=4, num_card_ids=120):
        super(MicroRoyaleNet, self).__init__()

        self.channels = channels
        self.board_width = board_width
        self.board_height = board_height

        # גודל המטריצה השטוחה המגיעה מ-ClashEnv
        self.spatial_size = channels * board_height * board_width
        # החלק הסקלרי: אליקסיר + 4 עלויות + 4 one-hot של זהות קלף (num_card_ids ערכים כל אחד)
        self.scalar_size = 1 + hand_size + hand_size * num_card_ids
        
        # ==========================================
        # 1. חילוץ תכונות מרחבי (CNN)
        # קלט: (Batch, 5, 32, 18)
        # ==========================================
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # מקטין ל- (16, 16, 9)
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # מקטין ל- (32, 8, 4) - בהתאמה לחלוקת השלמים ברוחב
            
            nn.Flatten()
        )
        
        # חישוב ממד הפלט של ה-CNN לאחר הפולנג: 32 ערוצים * גובה 8 * רוחב 4
        self.cnn_out_dim = 32 * (board_height // 4) * (board_width // 4)
        
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
        # 5 פעולות: 4 משבצות היד + פעולה 4 = no-op (המתנה/אגירת אליקסיר).
        # המנוע מתעלם מ-cardIndex מחוץ ל-[0,4) כך שאין צורך בשינוי C++.
        self.card_head = nn.Linear(256, 5)
        
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