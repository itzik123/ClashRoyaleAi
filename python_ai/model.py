import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import clash_royale_env

# קטן בכוונה (באותה רוח שבה AlphaStar מגדיר embeddings לבחירת action-type) --
# צריך רק להבדיל בין כמה קלפים אפשריים במשבצת יד, לא לקודד את המשמעות
# האסטרטגית המלאה של קלף; ה-CNN/scalar MLP כבר נותנים ל-LSTM את כל השאר.
CARD_EMBED_DIM = 16

# מספר שורות המיקום החוקיות בחצי שלנו, נשלף חי מהמנוע (כמו כל שאר הקבועים
# כאן) במקום עותק hardcoded. get_own_half_max_y() מחזיר 15.0 (לאחר תיקון
# מירכוז הנהר סביב 16.5, ראה Board.h -- קודם היה 15.5 עם נהר לא-ממורכז
# שנתן ל-team 0 שורה אחת יותר מ-team 1), כלומר שורות שלמות 0..15 -- 16
# שורות בכל מקרה. נדרש instance (לא static attr) אז נבנית פה פעם אחת
# בטעינת המודול, בדיוק כמו ה-_dim_probe ש-train.py כבר בונה.
_probe = clash_royale_env.ClashRoyaleEnv(list(range(8)), list(range(8)), 100)
OWN_HALF_MAX_Y = _probe.get_own_half_max_y()
MAX_PLACEMENT_X = _probe.get_max_placement_x()
del _probe
# השורה האחרונה בחצי שלנו שמותרת לכוחות (get_own_half_max_y = 15.0 -> שורה 15)
OWN_HALF_ROWS = int(OWN_HALF_MAX_Y) + 1
# ראש המיקום פורש עכשיו את **כל** הלוח, לא רק את החצי שלנו. הסיבה: המנוע
# פוטר לחשים ממגבלת החצי (GameManager::isValidPlacement בודק ללחש רק גבולות
# לוח + אזור מת), אבל מרחב הפעולות חסם את target_y ב-15.5 לכל קלף -- כלומר
# Fireball פיזית לא יכול היה לחצות את הנהר, ורבע מהחפיסה לא היה שמיש למטרתו
# בשום כמות אימון. עכשיו הראש מייצר את כל הלוח והחוקיות נאכפת ע"י מסכה
# מותנית-קלף (ראה placement_mask), לא ע"י כיווץ המרחב.
PLACEMENT_ROWS = clash_royale_env.ClashRoyaleEnv.BOARD_HEIGHT

# is_spell לכל card_id, נשלף חי מהמנוע (binding get_card_info) במקום רשימה
# קשיחה בפייתון -- בדיוק סוג ה-drift שהפרויקט הזה כבר נכווה ממנו.
_ALL_IDS = clash_royale_env.get_all_card_ids()
NUM_CARD_IDS_LIVE = clash_royale_env.ClashRoyaleEnv.NUM_CARD_IDS
_spell_flags = torch.zeros(NUM_CARD_IDS_LIVE)
for _cid in _ALL_IDS:
    if clash_royale_env.get_card_info(_cid)["is_spell"]:
        _spell_flags[_cid] = 1.0


class MicroRoyaleNet(nn.Module):
    # 9 ערוצים: 0-3 כוחות שלנו (קרבי/טווח/טנק/מבנים), 4-7 אותו דבר ליריב, 8 נהר/גשרים
    def __init__(self, channels=None, board_width=None, board_height=None, hand_size=None, num_card_ids=None,
                 placement_rows=None, num_ability_slots=0):
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

        # רשת המיקום היא קטגוריאלית על תאי לוח שלמים (ראה placement_head
        # למטה). placement_rows = כל גובה הלוח; החוקיות בפועל נאכפת במסכה
        # מותנית-קלף (placement_mask), כי היא שונה בין כוח ללחש.
        self.placement_rows = placement_rows if placement_rows is not None else PLACEMENT_ROWS
        self.placement_cells = self.placement_rows * board_width
        # שורות מותרות לכוח רגיל (לא לחש, לא deploy-anywhere) -- החצי שלנו בלבד.
        self.own_half_rows = min(OWN_HALF_ROWS, self.placement_rows)
        # (num_card_ids,) -- 1.0 אם הקלף הוא לחש. buffer ולא פרמטר: זו עובדה
        # על המנוע, לא משהו שנלמד, אבל היא חייבת לנוע יחד עם הרשת ל-device.
        self.register_buffer("spell_flags", _spell_flags[:num_card_ids].clone())

        # 0 = לחפיסה אין צ'מפיון, ולכן אין בכלל ראשי הפעלת יכולת. זה לא
        # אופטימיזציה קוסמטית: כשאין צ'מפיון, שני הראשים האלה דגמו רעש טהור
        # בכל טיק -- הוסיפו שונות ליחס ה-PPO (total_logprob) ותרמו עד
        # 2*log(2)=1.386 לבונוס האנטרופיה, כלומר המאמן *השקיע* מאמץ בשמירה
        # על מטבע אקראי הפוך. נמדד בפועל: DEFAULT_DECK לא מכילה צ'מפיון,
        # ו-is_champion_ability_ready החזיר False בשתי המשבצות לכל אורך
        # המשחק. הערך מועבר מפורשות מהמאמן לפי החפיסה בשימוש.
        self.num_ability_slots = num_ability_slots

        # גודל המטריצה השטוחה המגיעה מ-ClashEnv
        self.spatial_size = channels * board_height * board_width
        # החלק הסקלרי: אליקסיר + hand_size עלויות + hand_size one-hot של זהות
        # קלף (num_card_ids ערכים כל אחד) + הזנב שנוסף ב-ClashEnv
        # (NUM_EXTRA_SCALARS): שבר הזמן, אליקסיר שהוצא ע"י שני הצדדים, ו-6
        # ערכי HP של מגדלים.
        #
        # הזנב **מתווסף בסוף** ולא נדחף באמצע, וזה לא שרירותי: כל הקוד שקורא
        # את הווקטור הזה לפי היסטים (affordability_mask, hand_card_ids,
        # placement_mask כאן, והיריבים הסקריפטיים ב-train_selfplay.py) מודד
        # מתחילת הקטע הסקלרי. הוספה בסוף משאירה כל אחד מההיסטים האלה תקף
        # בלי שינוי; הוספה באמצע הייתה שוברת את כולם בשקט -- בלי חריגה, רק
        # מדיניות שמסתכלת על מספרים לא נכונים.
        self.num_extra_scalars = clash_royale_env.ClashRoyaleEnv.NUM_EXTRA_SCALARS
        self.scalar_size = 1 + hand_size + hand_size * num_card_ids + self.num_extra_scalars
        # ההיסט (בתוך scalar_obs) שבו מתחיל הזנב -- נחוץ לראש העזר ולאבחון.
        self.extra_start = 1 + hand_size + hand_size * num_card_ids

        # ==========================================
        # 1. חילוץ תכונות מרחבי (CNN)
        # קלט: (Batch, 9, 34, 18)
        # ==========================================
        # ceil_mode=True בשני ה-MaxPool: board_height=34 לא מתחלק נקי פי 4
        # (34 -> 17 -> 8 עם floor רגיל, מה שהיה מוחק שורה שלמה -- בדיוק השורה
        # האחורית החדשה ליד מגדל המלך, שזה כל הטעם בשינוי הזה). עם ceil_mode
        # שום שורה/עמודה לא נופלת בשקט, רק התמונה המרחבית קצת יותר גדולה.
        # מפוצל ל-trunk + flatten (במקום Sequential אחד שנגמר ב-Flatten):
        # מפת המאפיינים המרחבית *לפני* ההשטחה היא הקלט של ראש המיקום
        # הקונבולוציוני (ראה placement_given_card). זה אותו טנזור בדיוק, לא
        # חישוב נוסף -- ה-CNN עדיין רץ פעם אחת בלבד.
        self.cnn_trunk = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )
        self.cnn_flatten = nn.Flatten()

        # חישוב ממד הפלט של ה-CNN לאחר הפולינג (ceil פעמיים, תואם ceil_mode=True למעלה)
        pooled_h = math.ceil(math.ceil(board_height / 2) / 2)
        pooled_w = math.ceil(math.ceil(board_width / 2) / 2)
        self.pooled_h, self.pooled_w = pooled_h, pooled_w
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

        # ב. ראש המיקום במרחב -- קטגוריאלי על תאי לוח שלמים, לא גאוסיאן.
        # אוטורגרסיבי כמו קודם: מותנה ב-hx *וגם* ב-embedding של הקלף שנבחר.
        #
        # למה זה הוחלף: הגרסה הקודמת דגמה מ-Normal עם placement_log_std
        # נלמד. האנטרופיה של גאוסיאן היא log(sigma) + const, ולכן הנגזרת של
        # בונוס האנטרופיה לפי log_std היא *בדיוק 1, קבועה*, ללא תלות בנתונים
        # -- כוח קבוע שדוחף את sigma למעלה, שה-policy gradient הרועש מפסיד
        # לו. נמדד בפועל על הצ'קפוינטים: log_std התחיל ב--2.0 וטיפס
        # ל--1.86 אחרי 68,515 אפיזודות (סיגמא *גדלה* במקום להצטמצם), כלומר
        # sigma=0.156 ביחידות מנורמלות = 2.66 משבצות סטיית תקן ב-x ו-2.46
        # ב-y. שני הגשרים מרוחקים 10 משבצות זה מזה, אז הרעש העצמי של הסוכן
        # היה חצי מהמרחק שהוא אמור להבחין בו -- הוא פשוט לא היה מסוגל לכוון
        # לנתיב. התפלגות קטגוריאלית פותרת את זה משורש: האנטרופיה שלה חסומה
        # מלמעלה ב-log(placement_cells) ויורדת באופן טבעי ככל שהמדיניות
        # מתחדדת, המיקום מדויק עד משבצת, וניתן למסוך תאים לא חוקיים.
        #
        # ומה שהוחלף *עכשיו*: הגרסה הקודמת הייתה
        #     nn.Linear(256 + CARD_EMBED_DIM, placement_cells)   # 272 -> 612
        # כלומר שכבה צפופה אחת שמייצרת מפת לוגיטים על הלוח מתוך וקטור שכבר
        # עבר שני MaxPool והושטח. לשכבה כזו אין שום מבנה מרחבי: היא חייבת
        # *לשנן* בנפרד, במשקל נפרד, מה המשמעות של כל אחת מ-612 המשבצות, ואין
        # שום שיתוף בין משבצת (5,7) לשכנתה (5,8) -- גם אחרי שהרשת למדה
        # "להניח ליד הגשר השמאלי", שום דבר מזה לא מתפשט למשבצת הסמוכה.
        #
        # במקום זה: לוקחים את מפת המאפיינים המרחבית מה-CNN (B,32,9,5),
        # מוסיפים לה הקשר (hx + זהות הקלף) בשידור על כל התאים, ומרחיבים
        # בחזרה ל-34x18 עם ConvTranspose. הלוגיט של כל תא מחושב אז ע"י אותם
        # משקלים משותפים שפועלים על המאפיינים המקומיים *של אותו אזור לוח* --
        # וזו בדיוק ההטיה האינדוקטיבית הנכונה למשחק שבו ההחלטה היא "איפה".
        # אותו דפוס שבו AlphaStar מייצר ארגומנטים מרחביים.
        self.place_ctx = nn.Linear(256 + CARD_EMBED_DIM, 32)
        self.place_up = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),   # 9x5 -> 18x10
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),   # 18x10 -> 36x20
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),  # -> (B,1,36,20)
        )
        # ה-deconv מייצר 4*pooled_h x 4*pooled_w, שהוא >= גודל הלוח כי הפולינג
        # השתמש ב-ceil_mode. חותכים בחזרה לפינה השמאלית-עליונה: מכיוון ששני
        # הפולינגים הם stride 2 עם ceil, תא ממוזג i מכסה את המקוריים 2i,2i+1
        # בכל רמה, ולכן אינדקס j במפה המורחבת מתיישר בדיוק עם שורה/עמודה j
        # בלוח המקורי. החיתוך הוא יישור, לא קירוב.
        assert self.placement_rows == board_height, (
            "ראש המיקום הקונבולוציוני מייצר מפה בגודל הלוח וחותך אותה; "
            "placement_rows שונה מ-board_height יישבור את היישור הזה")

        # ==========================================
        # 5. ראש הערכת המצב - Critic Head -- תלוי רק ב-hx.
        # ==========================================
        self.value_head = nn.Linear(256, 1)

        # ==========================================
        # 6. ראשי הפעלת יכולת צ'מפיון (עד 2 צ'מפיונים בו-זמנית -- ראו
        # activate_ability_slot1/2 ב-gym_wrapper.py, ו-
        # CardRegistry::validateDeckSlots בצד ה-C++). כל אחד: 2 לוגיטים
        # (0=אל תפעיל, 1=הפעל).
        #
        # נוצרים *רק* אם num_ability_slots > 0 -- ראה ההערה על השדה הזה
        # למעלה. עם חפיסה בלי צ'מפיון הם היו רעש בלבד, ועכשיו הם פשוט לא
        # קיימים (גם לא ב-state_dict), אז אין פרמטרים מתים ואין תרומה
        # ל-logprob/entropy.
        # ==========================================
        self.ability_slot1_head = nn.Linear(256, 2) if num_ability_slots >= 1 else None
        self.ability_slot2_head = nn.Linear(256, 2) if num_ability_slots >= 2 else None

        # ==========================================
        # 7. ראש עזר: הערכת האליקסיר של היריב (auxiliary prediction head)
        # ==========================================
        # מנבא את האליקסיר הנוכחי של היריב (0..10, מנורמל ל-0..1) מתוך hx.
        #
        # למה זה קיים: ספירת אליקסיר היא הכישור המרכזי של שחקני קלאש חזקים,
        # והמידע הזה **מוסתר** -- אי אפשר לקרוא אותו מהמסך. לכן הוא בכוונה
        # לא נמצא בתצפית (ראה ClashEnv::getElixirForTeam): מדיניות שמותנית בו
        # לא הייתה ניתנת להפעלה מול יריב אמיתי דרך perception/. במקום זה
        # הרשת מקבלת בתצפית רק את מה ששחקן אנושי באמת רואה -- זמן שחלף
        # וההוצאה המצטברת של שני הצדדים -- ומתבקשת להסיק מהם את הערך המוסתר.
        #
        # הראש הזה לא משתתף בבחירת הפעולה בכלל. כל תפקידו הוא ה-loss: הוא
        # מכריח את מצב ה-LSTM לשמור בפועל את האינטגרל של ההוצאה לאורך המשחק,
        # במקום לקוות שהאות הדליל של ניצחון/הפסד ילמד את זה לבד. זה בדיוק
        # התפקיד של auxiliary tasks ב-UNREAL/IMPALA: לעצב את הייצוג, לא את
        # המדיניות.
        #
        # מכוון בכוונה כמתודה נפרדת ולא כפלט נוסף של step_lstm_and_card:
        # הוא נחוץ **רק בזמן אימון**, אף פעם לא בבחירת פעולה, אז אין סיבה
        # לשלם עליו בכל טיק של rollout ואין סיבה לשנות את החתימה של מסלול
        # הפעולה החם (ולסכן את כל מי שקורא לו).
        self.aux_elixir_head = nn.Linear(256, 1)

    def extract_features(self, obs):
        """
        חילוץ מאפיינים (CNN + MLP סקלרי + embeddings של זהות קלף) - החלק
        הלא-רקורנטי של הרשת. אפשר לקרוא לזה על באצ' ענק ומשוטח (T*N) כדי
        להריץ את ה-CNN (והחישוב הזול של embeddings הקלפים) פעם אחת במקום
        פעם לכל טיק - זהו הזירוז המרכזי של עדכון ה-PPO.

        obs: (Batch, obs_dim)
        מחזיר (combined, card_embeds, spatial_map):
          combined: (Batch, lstm_input_dim) -- בדיוק כמו קודם, מוזן ל-LSTM.
          card_embeds: (Batch, hand_size+1, CARD_EMBED_DIM) -- embedding לכל
            משבצת יד ממשית + אחד נוסף (no-op), נדגם לפי card_idx רק אחרי
            שנבחר -- ראה placement_given_card. הפיצול מ-combined הוא בדיוק מה
            שמאפשר את המיקום האוטורגרסיבי: אי אפשר "לערבב" את זהות הקלף לתוך
            ה-LSTM before הבחירה בלי לאבד את היכולת להתנות אחריה.
          spatial_map: (Batch, 32, pooled_h, pooled_w) -- מפת המאפיינים של
            ה-CNN *לפני* ההשטחה, הקלט של ראש המיקום הקונבולוציוני. זהו אותו
            טנזור שממנו נגזר combined, לא חישוב נוסף.
        """
        # פיצול הווקטור השטוח לחלק המרחבי ולחלק הסקלרי בהתאם לפונקציית observationSize() ב-C++
        spatial_obs = obs[:, :self.spatial_size].view(-1, self.channels, self.board_height, self.board_width)
        scalar_obs = obs[:, self.spatial_size:]

        spatial_map = self.cnn_trunk(spatial_obs)
        cnn_features = self.cnn_flatten(spatial_map)
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

        return combined, card_embeds, spatial_map

    def affordability_mask(self, obs):
        """
        מסכת פעולות: אילו משבצות יד באמת ניתנות לשחק *עכשיו*, לפי האליקסיר
        הנוכחי והעלויות -- שניהם כבר נמצאים בתוך וקטור התצפית עצמו, אז זה
        מחושב בפייתון בלבד בלי שום קריאה נוספת למנוע.

        זה התיקון המשמעותי ביותר בצינור כולו. GameManager::playCard מחזירה
        false בשקט כשאין מספיק אליקסיר -- בלי חריגה, בלי תגמול שלילי, בלי
        שום סימן לסוכן. נמדד על המדיניות המאומנת (243 צעדי החלטה אמיתיים):
        74.9% מכלל הצעדים היו ניסיון לשחק קלף שאין עליו אליקסיר, ורק 11.5%
        מהצעדים באמת הניחו קלף. כלומר ב-~87% מהדגימות בכל rollout הפעולה
        השמורה בבאפר לא השפיעה על העולם בכלל -- אותו next_state היה מתקבל
        מכל פעולה אחרת -- וה-advantage שיוחס להן היה רעש טהור שנכנס ישר
        לגרדיאנט. הסיבה מבנית ולא זמנית: התחדשות אליקסיר היא 0.035 לטיק
        ו-skip_frames=10, כלומר 0.35 אליקסיר לצעד החלטה מול קלף שעולה 3-5,
        אז בממוצע רק 0.55 מתוך 4 משבצות ניתנות לשחק ורק ב-27.2% מהצעדים יש
        ולו אפשרות חוקית אחת.

        המסכה חייבת להיות מיושמת *זהה* באיסוף ה-rollout ובעדכון ה-PPO,
        אחרת יחס ה-old/new logprob נשבר -- ולכן היא מחושבת מהתצפית עצמה
        (שנשמרת בבאפר ממילא) ולא נשמרת בנפרד: אותו obs מייצר בהכרח אותה
        מסכה בשתי הקריאות.

        obs: (Batch, obs_dim). מחזיר bool tensor (Batch, hand_size+1);
        העמודה האחרונה (no-op) תמיד True -- המתנה היא תמיד פעולה חוקית.
        """
        scalar_obs = obs[:, self.spatial_size:]
        # אותו layout שבו ClashEnv::extractObservationForTeam בונה את
        # scalar_obs: [elixir(1), costs(hand_size), onehots(...)]. שניהם
        # מחולקים ב-10 שם, אז היחס ביניהם נכון בלי להכפיל בחזרה.
        elixir = scalar_obs[:, 0:1]
        costs = scalar_obs[:, 1:1 + self.hand_size]
        # cost <= 0 מסמן משבצת ריקה/לא חוקית (ראה ClashEnv: card ? cost/10 : 0)
        # -- אף פעם לא קלף אמיתי בחינם.
        playable = (costs > 0.0) & (costs <= elixir + 1e-6)
        noop = torch.ones(obs.shape[0], 1, dtype=torch.bool, device=obs.device)
        return torch.cat([playable, noop], dim=1)

    def hand_card_ids(self, obs):
        """
        איזה card_id יושב בכל משבצת יד, לפי ה-one-hot שכבר נמצא ב-obs.
        (Batch, hand_size) long. משבצת ריקה -> -1.

        נדרש למדדי האבחון החיים (כמה קלפים שונים הבוט באמת משחק): את זהות
        הקלף חייבים לקרוא מה-obs שעליו התקבלה ההחלטה, לא מ-game.get_hand()
        אחרי הצעד -- היד מסתובבת ברגע ששוחק קלף, אז קריאה מאוחרת מחזירה
        את הקלף *הבא* ולא את זה שנבחר.
        """
        scalar_obs = obs[:, self.spatial_size:]
        onehot_start = 1 + self.hand_size
        onehots = scalar_obs[:, onehot_start:onehot_start + self.hand_size * self.num_card_ids]
        onehots = onehots.view(-1, self.hand_size, self.num_card_ids)
        ids = onehots.argmax(dim=-1)
        # שורת one-hot ריקה (סכום 0) היא משבצת בלי קלף -- argmax היה מחזיר 0
        # שהוא card_id חוקי (Knight), אז מסמנים אותה מפורשות.
        return torch.where(onehots.sum(dim=-1) > 0.5, ids, torch.full_like(ids, -1))

    def elixir_from_obs(self, obs):
        """אליקסיר נוכחי (0..10) מתוך ה-obs. ClashEnv מחלק ב-10 בבנייה."""
        return obs[:, self.spatial_size] * 10.0

    def step_lstm_and_card(self, features, hidden_state, card_mask=None):
        """
        חצי ראשון של הצעד הרקורנטי: מקדם את ה-LSTM ומחשב בחירת קלף + הערכת
        מצב -- שניהם תלויים רק ב-hx, לא בקלף שעוד ייבחר. מופרד מהמיקום כדי
        שקריאת ה-bootstrap value-only (ל-GAE) לעולם לא תצטרך לחשב מיקום
        שהיא סתם תזרוק.
        features: (Batch, lstm_input_dim)
        card_mask: (Batch, hand_size+1) bool מ-affordability_mask, או None
          (ללא מיסוך -- ההתנהגות הישנה, לשימוש רק היכן שאין תצפית זמינה).
        """
        hx, cx = self.lstm(features, hidden_state)
        card_logits = self.card_head(hx)
        if card_mask is not None:
            # -inf ולא ערך שלילי גדול-אך-סופי: Categorical מנרמל דרך
            # log_softmax, וערך סופי היה עדיין משאיר הסתברות זעירה אך אי-
            # אפסית לפעולה בלתי-חוקית, כלומר גם דגימה נדירה שלה וגם תרומה
            # לאנטרופיה. -inf נותן בדיוק אפס בשניהם. ה-no-op תמיד חוקי
            # (ראה affordability_mask) אז אף שורה לא יכולה לצאת כולה -inf.
            card_logits = card_logits.masked_fill(~card_mask, float("-inf"))
        state_value = self.value_head(hx)
        # ראשי הצ'מפיון תלויים רק ב-hx, בדיוק כמו card_logits/state_value --
        # לכן מחושבים כאן, לא ב-placement_given_card. None כשאין צ'מפיון
        # בחפיסה (ראה num_ability_slots), והמאמן מדלג עליהם לגמרי.
        ability_slot1_logits = self.ability_slot1_head(hx) if self.ability_slot1_head is not None else None
        ability_slot2_logits = self.ability_slot2_head(hx) if self.ability_slot2_head is not None else None
        return card_logits, ability_slot1_logits, ability_slot2_logits, state_value, (hx, cx)

    def placement_given_card(self, hx, card_embeds, card_idx, obs=None, spatial_map=None):
        """
        חצי שני: מיקום מותנה ב-card_idx (שנדגם עכשיו, בזמן rollout, או נשמר
        מהבאפר, בזמן עדכון PPO) -- זהו הצעד האוטורגרסיבי עצמו.
        hx: (Batch, 256). card_embeds: (Batch, hand_size+1, CARD_EMBED_DIM).
        card_idx: (Batch,) טנזור long, ערכים ב-[0, hand_size] כולל.

        מחזיר לוגיטים (Batch, placement_cells) על תאי לוח שלמים. ההמרה
        לקואורדינטות אמיתיות היא cell_to_xy() למטה.

        מפורק *כמפרק אחד* על כל התאים (ולא כמכפלה נפרדת של x ו-y): התפלגות
        מפורקת p(x)*p(y) לא יכולה לייצג "או ליד הגשר השמאלי או בפינה
        האחורית הימנית" בלי לפזר מסה גם על שתי הקומבינציות המעורבות, וזה
        בדיוק סוג ההחלטה הדו-מודאלית שהמשחק דורש. placement_cells קטן
        (18*16=288) אז השכבה זולה.
        """
        if spatial_map is None:
            raise ValueError(
                "placement_given_card דורש את spatial_map מ-extract_features. "
                "העברת None כאן הייתה מייצרת לוגיטים שונים מאלה שנוצרו ב-rollout, "
                "ויחס ה-PPO היה נשבר בשקט -- לכן זו שגיאה ולא ברירת מחדל.")

        batch_idx = torch.arange(card_embeds.shape[0], device=card_embeds.device)
        chosen_embed = card_embeds[batch_idx, card_idx]  # (Batch, CARD_EMBED_DIM)
        # הקשר -> 32 ערוצים, משודר על כל תא במפה המרחבית. חיבור ולא שרשור:
        # כך "מה המצב הכללי ואיזה קלף" מזיז את כל מפת הלוגיטים, בעוד המבנה
        # המקומי של הלוח נשאר במפה עצמה.
        ctx = self.place_ctx(torch.cat((hx, chosen_embed), dim=-1))     # (B, 32)
        h = spatial_map + ctx.view(-1, 32, 1, 1)
        logit_map = self.place_up(h)                                    # (B, 1, 4*ph, 4*pw)
        logits = logit_map[:, 0, :self.placement_rows, :self.board_width].reshape(
            -1, self.placement_cells)
        if obs is not None:
            # מיסוך חוקיות מותנה-קלף. -inf ולא ערך סופי, מאותה סיבה בדיוק
            # כמו ב-step_lstm_and_card: Categorical מנרמל דרך log_softmax,
            # אז ערך סופי היה משאיר הסתברות זעירה לתא לא חוקי ותרומה
            # לאנטרופיה. תמיד יש לפחות שורה חוקית אחת, אז אף שורה לא יוצאת
            # כולה -inf.
            logits = logits.masked_fill(~self.placement_mask(obs, card_idx), float("-inf"))
        return logits

    def predict_opp_elixir(self, hx):
        """
        הערכת האליקסיר של היריב מתוך מצב ה-LSTM. (Batch,) בסקאלה 0..10 --
        אותן יחידות כמו get_elixir_for_team, כדי שהשגיאה תהיה קריאה ישירות
        ביחידות אליקסיר ולא בסקאלה מנורמלת חסרת משמעות.

        נקרא רק מלולאת עדכון ה-PPO (ראה AUX_ELIXIR_COEF במאמנים). הגרדיאנט
        שלו זורם אחורה לתוך ה-LSTM וה-CNN -- זו כל המטרה.
        """
        return self.aux_elixir_head(hx).squeeze(-1) * 10.0

    def placement_mask(self, obs, card_idx):
        """
        אילו תאי לוח חוקיים לקלף שנבחר. (Batch, placement_cells) bool.

        המנוע (GameManager::isValidPlacement) מבחין בין שניים:
          * כוח רגיל -- רק החצי שלנו, y <= get_own_half_max_y().
          * לחש      -- כל הלוח; מגבלת החצי מדולגת לגמרי.
        עד עכשיו הפייתון כפה את המקרה המחמיר על שניהם (target_y נחסם ב-15.5
        תמיד), ולכן Fireball לא יכול היה לחצות את הנהר אף פעם. המסכה הזו
        מחזירה את ההבחנה למקום שבו היא שייכת.

        זהות הלחש נגזרת מה-one-hot שכבר יושב ב-obs כפול spell_flags, בלי
        קריאה למנוע ובלי argmax -- כלומר עובד על באצ' שלם ומשחזר בדיוק את
        אותה מסכה בעדכון ה-PPO כמו ב-rollout.

        no-op (card_idx == hand_size): המנוע מתעלם מהמיקום לגמרי, אז מחזירים
        את מסכת החצי שלנו רק כדי שההתפלגות תישאר מוגדרת-היטב ולא ריקה.
        """
        batch = obs.shape[0]
        scalar_obs = obs[:, self.spatial_size:]
        onehot_start = 1 + self.hand_size
        onehots = scalar_obs[:, onehot_start:onehot_start + self.hand_size * self.num_card_ids]
        onehots = onehots.view(batch, self.hand_size, self.num_card_ids)
        # (Batch, hand_size) -- 1.0 היכן שהמשבצת מחזיקה לחש
        slot_is_spell = (onehots * self.spell_flags.view(1, 1, -1)).sum(dim=-1)
        # הרחבה למשבצת ה-no-op (אף פעם לא לחש)
        slot_is_spell = torch.cat(
            [slot_is_spell, torch.zeros(batch, 1, device=obs.device, dtype=slot_is_spell.dtype)], dim=1)
        chosen_is_spell = slot_is_spell.gather(1, card_idx.view(-1, 1)).squeeze(1) > 0.5  # (Batch,)

        rows = torch.arange(self.placement_rows, device=obs.device).view(1, -1, 1)
        own_half = rows < self.own_half_rows                       # (1, rows, 1)
        full_board = torch.ones_like(own_half)
        allowed_rows = torch.where(chosen_is_spell.view(-1, 1, 1), full_board, own_half)
        mask = allowed_rows.expand(batch, self.placement_rows, self.board_width)
        return mask.reshape(batch, self.placement_cells)

    def cell_to_xy(self, cell_idx):
        """
        אינדקס תא -> קואורדינטות לוח אמיתיות שהמנוע מקבל.
        cell_idx: טנזור long כלשהו. מחזיר (x, y) טנזורי float באותה צורה.

        row-major, זהה לפריסה של placement_head. x=עמודה ו-y=שורה כערכים
        שלמים בדיוק: העמודה המקסימלית היא board_width-1 = get_max_placement_x()
        והשורה המקסימלית היא placement_rows-1 = 15 <= get_own_half_max_y()
        (15.5), כלומר כל תא נופל בתוך התחום שהמנוע אוכף -- אין צורך ב-clamp
        ואף תא לא "מתגלגל" בשקט לגבול.
        """
        row = torch.div(cell_idx, self.board_width, rounding_mode="floor")
        col = cell_idx % self.board_width
        return col.float(), row.float()

    def forward_from_features(self, features, card_embeds, hidden_state, card_idx,
                              card_mask=None, obs=None, spatial_map=None):
        """
        עוטף את שני החצאים ביחד, לשימוש כש-card_idx כבר ידוע מראש (עדכון PPO,
        עם הפעולה השמורה מהבאפר -- קריטי: תמיד להעביר את card_idx *השמור*
        כאן, לא דגימה טרייה, אחרת יחס ה-PPO (ratio) בין old/new logprob
        מתקלקל). לא שימושי בזמן איסוף rollout (שם card_idx עוד לא ידוע לפני
        שדוגמים אותו מ-card_logits) -- שם קוראים ל-step_lstm_and_card ואז
        ל-placement_given_card בנפרד, ראה train.py/train_selfplay.py.

        card_mask חייבת להיות אותה מסכה שהופעלה בזמן ה-rollout (בפועל:
        מחושבת מחדש מאותו obs שנשמר בבאפר -- ראה affordability_mask).
        """
        (card_logits, ability_slot1_logits, ability_slot2_logits, state_value,
         (hx, cx)) = self.step_lstm_and_card(features, hidden_state, card_mask)
        placement_logits = self.placement_given_card(hx, card_embeds, card_idx, obs, spatial_map)
        return (card_logits, placement_logits, state_value,
                ability_slot1_logits, ability_slot2_logits, (hx, cx))
