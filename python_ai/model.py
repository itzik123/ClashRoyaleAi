import torch
import torch.nn as nn
import torch.nn.functional as F

class MicroRoyaleNet(nn.Module):
    def __init__(self, channels=5, board_width=18, board_height=32):
        super(MicroRoyaleNet, self).__init__()
        
        self.channels = channels
        self.board_width = board_width
        self.board_height = board_height
        
        # גודל המטריצה השטוחה המגיעה מ-ClashEnv
        self.spatial_size = channels * board_height * board_width
        
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
        # קלט: 5 ערכים (1 אליקסיר + 4 עלויות קלפים ביד)
        # ==========================================
        self.scalar_mlp = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU()
        )
        
        # ==========================================
        # 3. שכבת זיכרון (LSTM)
        # ==========================================
        self.lstm_input_dim = self.cnn_out_dim + 32
        # אנו משתמשים ב-LSTMCell כדי שנוכל לשלוט על הפעימות (Ticks) ידנית בלולאת הסביבה
        self.lstm = nn.LSTMCell(self.lstm_input_dim, 256)
        
        # ==========================================
        # 4. ראשי הפעולה - Actor Heads
        # ==========================================
        # א. ראש בחירת הקלף (התפלגות קטגוריאלית)
        self.card_head = nn.Linear(256, 4)
        
        # ב. ראש המיקום במרחב (התפלגות גאוסיאנית / תחימה)
        # אנו מוציאים 2 ערכים, ונעביר אותם דרך פונקציית Sigmoid כדי לתחום אותם בין [0, 1]
        self.placement_head = nn.Linear(256, 2)
        
        # ==========================================
        # 5. ראש הערכת המצב - Critic Head
        # ==========================================
        self.value_head = nn.Linear(256, 1)

    def forward(self, obs, hidden_state):
        """
        מעביר תצפית בודדת דרך הרשת.
        obs: טנזור בגודל (Batch, 2885)
        hidden_state: הסטייט הקודם של ה-LSTM -> (hx, cx)
        """
        # פיצול הווקטור השטוח לחלק המרחבי ולחלק הסקלרי בהתאם לפונקציית observationSize() ב-C++
        spatial_obs = obs[:, :self.spatial_size]
        scalar_obs = obs[:, self.spatial_size:]
        
        # עיצוב מחדש (Reshape) של הווקטור המרחבי למטריצה תלת-ממדית עבור ה-CNN
        # בסדר: Batch, Channels, Height, Width
        spatial_obs = spatial_obs.view(-1, self.channels, self.board_height, self.board_width)
        
        # חילוץ מאפיינים
        cnn_features = self.cnn(spatial_obs)
        scalar_features = self.scalar_mlp(scalar_obs)
        
        # שרשור (Concatenation) של כלל המאפיינים
        combined_features = torch.cat((cnn_features, scalar_features), dim=1)
        
        # העברה דרך הזיכרון
        hx, cx = self.lstm(combined_features, hidden_state)
        
        # חישוב הפלטים
        card_logits = self.card_head(hx)
        placement_normalized = torch.sigmoid(self.placement_head(hx))
        state_value = self.value_head(hx)
        
        return card_logits, placement_normalized, state_value, (hx, cx)