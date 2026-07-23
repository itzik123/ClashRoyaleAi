import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import clash_royale_env

# קטן בכוונה (באותה רוח שבה AlphaStar מגדיר embeddings לבחירת action-type) --
# צריך רק להבדיל בין כמה קלפים אפשריים במשבצת יד, לא לקודד את המשמעות
# האסטרטגית המלאה של קלף; ה-CNN/scalar MLP כבר נותנים ל-LSTM את כל השאר.
CARD_EMBED_DIM = 16

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
        self.hand_size = hand_size
        self.num_card_ids = num_card_ids

        # גודל המטריצה השטוחה המגיעה מ-ClashEnv
        self.spatial_size = channels * board_height * board_width
        # החלק הסקלרי: אליקסיר + hand_size עלויות + hand_size one-hot של זהות קלף (num_card_ids ערכים כל אחד)
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
        # 2ב. Embedding לזהות קלף -- למיקום אוטורגרסיבי (ראה placement_given_card)
        # ==========================================
        # פועל ישירות על ה-one-hot שכבר קיים ב-obs (לא על אינדקס המשבצת עצמו --
        # אינדקס המשבצת מסתובב, אותה משבצת היא קלף פיזי שונה כל מחזור, אז
        # embedding של האינדקס לא היה מלמד "Fireball על הצבר שלהם, Hog על
        # הגשר" בכלל). nn.Linear על וקטור one-hot אמיתי שקול מתמטית בדיוק ל-
        # embedding lookup רגיל, בלי מעגל דרך argmax.
        self.card_id_embed = nn.Linear(num_card_ids, CARD_EMBED_DIM, bias=False)
        # Embedding ייעודי לפעולת ה-no-op (card_index == hand_size, אין קלף
        # אמיתי במשבצת הזו) -- המנוע מתעלם לגמרי מ-target_x/target_y כש-
        # card_index מחוץ ל-[0, hand_size) (ראה ההערה המקבילה ב-gym_wrapper.py),
        # אז זה קיים אך ורק כדי לשמור על התפלגות מיקום מוגדרת-היטב לכל טיק
        # לצורך ה-loss -- פרמטר נלמד נפרד, כדי שגרדיאנט מ-timesteps של no-op
        # אף פעם לא "יזהם" את הלמידה של מיקום מותנה-קלף אמיתי כלשהו.
        self.noop_embed = nn.Parameter(torch.zeros(CARD_EMBED_DIM))

        # ==========================================
        # 3. שכבת זיכרון (LSTM)
        # ==========================================
        self.lstm_input_dim = self.cnn_out_dim + 64
        # אנו משתמשים ב-LSTMCell כדי שנוכל לשלוט על הפעימות (Ticks) ידנית בלולאת הסביבה
        self.lstm = nn.LSTMCell(self.lstm_input_dim, 256)

        # ==========================================
        # 4. ראשי הפעולה - Actor Heads
        # ==========================================
        # א. ראש בחירת הקלף (התפלגות קטגוריאלית) -- תלוי רק ב-hx, בדיוק כמו קודם.
        # hand_size משבצות יד + פעולה אחת נוספת = no-op (המתנה/אגירת אליקסיר).
        # המנוע מתעלם מ-cardIndex מחוץ ל-[0,hand_size) כך שאין צורך בשינוי C++.
        self.card_head = nn.Linear(256, hand_size + 1)

        # ב. ראש המיקום במרחב (התפלגות גאוסיאנית / תחימה) -- אוטורגרסיבי:
        # מותנה ב-hx *וגם* ב-embedding של הקלף שכבר נבחר (ראה
        # placement_given_card). קלט גדל מ-256 (רק hx) ל-256+CARD_EMBED_DIM.
        self.placement_head = nn.Linear(256 + CARD_EMBED_DIM, 2)
        # סטיית התקן של ההתפלגות הגאוסיאנית - פרמטר נלמד (state-independent), נדרש כדי
        # שיהיה ניתן לחשב log_prob ולתת gradient אמיתי לראש המיקום
        self.placement_log_std = nn.Parameter(torch.ones(2) * -2.0)

        # ==========================================
        # 5. ראש הערכת המצב - Critic Head -- תלוי רק ב-hx.
        # ==========================================
        self.value_head = nn.Linear(256, 1)

    def extract_features(self, obs):
        """
        חילוץ מאפיינים (CNN + MLP סקלרי + embeddings של זהות קלף) - החלק
        הלא-רקורנטי של הרשת. אפשר לקרוא לזה על באצ' ענק ומשוטח (T*N) כדי
        להריץ את ה-CNN (והחישוב הזול של embeddings הקלפים) פעם אחת במקום
        פעם לכל טיק - זהו הזירוז המרכזי של עדכון ה-PPO.

        obs: (Batch, obs_dim)
        מחזיר (combined, card_embeds):
          combined: (Batch, lstm_input_dim) -- בדיוק כמו קודם, מוזן ל-LSTM.
          card_embeds: (Batch, hand_size+1, CARD_EMBED_DIM) -- embedding לכל
            משבצת יד ממשית + אחד נוסף (no-op), נדגם לפי card_idx רק אחרי
            שנבחר -- ראה placement_given_card. הפיצול מ-combined הוא בדיוק מה
            שמאפשר את המיקום האוטורגרסיבי: אי אפשר "לערבב" את זהות הקלף לתוך
            ה-LSTM before הבחירה בלי לאבד את היכולת להתנות אחריה.
        """
        # פיצול הווקטור השטוח לחלק המרחבי ולחלק הסקלרי בהתאם לפונקציית observationSize() ב-C++
        spatial_obs = obs[:, :self.spatial_size].view(-1, self.channels, self.board_height, self.board_width)
        scalar_obs = obs[:, self.spatial_size:]

        cnn_features = self.cnn(spatial_obs)
        scalar_features = self.scalar_mlp(scalar_obs)
        combined = torch.cat((cnn_features, scalar_features), dim=1)

        # אותו layout שבו ClashEnv::extractObservationForTeam בונה את
        # scalar_obs: [elixir(1), costs(hand_size), onehots(hand_size*num_card_ids)].
        onehot_start = 1 + self.hand_size
        card_onehots = scalar_obs[:, onehot_start:onehot_start + self.hand_size * self.num_card_ids]
        card_onehots = card_onehots.view(-1, self.hand_size, self.num_card_ids)
        card_embeds = self.card_id_embed(card_onehots)  # (Batch, hand_size, CARD_EMBED_DIM)

        noop = self.noop_embed.view(1, 1, -1).expand(card_embeds.shape[0], 1, -1)
        card_embeds = torch.cat([card_embeds, noop], dim=1)  # (Batch, hand_size+1, CARD_EMBED_DIM)

        return combined, card_embeds

    def step_lstm_and_card(self, features, hidden_state):
        """
        חצי ראשון של הצעד הרקורנטי: מקדם את ה-LSTM ומחשב בחירת קלף + הערכת
        מצב -- שניהם תלויים רק ב-hx, לא בקלף שעוד ייבחר. מופרד מהמיקום כדי
        שקריאת ה-bootstrap value-only (ל-GAE) לעולם לא תצטרך לחשב מיקום
        שהיא סתם תזרוק.
        features: (Batch, lstm_input_dim)
        """
        hx, cx = self.lstm(features, hidden_state)
        card_logits = self.card_head(hx)
        state_value = self.value_head(hx)
        return card_logits, state_value, (hx, cx)

    def placement_given_card(self, hx, card_embeds, card_idx):
        """
        חצי שני: מיקום מותנה ב-card_idx (שנדגם עכשיו, בזמן rollout, או נשמר
        מהבאפר, בזמן עדכון PPO) -- זהו הצעד האוטורגרסיבי עצמו.
        hx: (Batch, 256). card_embeds: (Batch, hand_size+1, CARD_EMBED_DIM).
        card_idx: (Batch,) טנזור long, ערכים ב-[0, hand_size] כולל.
        """
        batch_idx = torch.arange(card_embeds.shape[0], device=card_embeds.device)
        chosen_embed = card_embeds[batch_idx, card_idx]  # (Batch, CARD_EMBED_DIM)
        placement_input = torch.cat((hx, chosen_embed), dim=-1)
        placement_normalized = torch.sigmoid(self.placement_head(placement_input))
        placement_log_std = torch.clamp(self.placement_log_std, -4.0, 0.0).expand_as(placement_normalized)
        return placement_normalized, placement_log_std

    def forward_from_features(self, features, card_embeds, hidden_state, card_idx):
        """
        עוטף את שני החצאים ביחד, לשימוש כש-card_idx כבר ידוע מראש (עדכון PPO,
        עם הפעולה השמורה מהבאפר -- קריטי: תמיד להעביר את card_idx *השמור*
        כאן, לא דגימה טרייה, אחרת יחס ה-PPO (ratio) בין old/new logprob
        מתקלקל). לא שימושי בזמן איסוף rollout (שם card_idx עוד לא ידוע לפני
        שדוגמים אותו מ-card_logits) -- שם קוראים ל-step_lstm_and_card ואז
        ל-placement_given_card בנפרד, ראה train.py/train_selfplay.py.
        """
        card_logits, state_value, (hx, cx) = self.step_lstm_and_card(features, hidden_state)
        placement_normalized, placement_log_std = self.placement_given_card(hx, card_embeds, card_idx)
        return card_logits, placement_normalized, placement_log_std, state_value, (hx, cx)
