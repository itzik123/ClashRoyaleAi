import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import clash_royale_env

# קטן בכוונה (באותה רוח שבה AlphaStar מגדיר embeddings לבחירת action-type) --
# צריך רק להבדיל בין כמה קלפים אפשריים במשבצת יד, לא לקודד את המשמעות
# האסטרטגית המלאה של קלף; ה-CNN/scalar MLP כבר נותנים ל-LSTM את כל השאר.
CARD_EMBED_DIM = 16

# רוחב ענף הרזולוציה המלאה של ראש המיקום (ראה place_hires ב-__init__).
# מכוון בכוונה צר: הענף רץ ב-34x18 המלא, כלומר פי 12.24 תאים מהמפה המאוחדת
# 9x5, אז כל ערוץ שם עולה בערך פי 12 מערוץ בקצה הגס. 8 ערוצי הקשר + 8 ערוצי
# ביניים מספיקים כדי לבחור משבצת בתוך בלוק, וזה מה שהענף צריך לעשות -- את
# "איפה בערך" הקצה הגס כבר יודע.
HIRES_CTX_DIM = 8
HIRES_HIDDEN = 8

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



def _build_placement_legality(num_card_ids, placement_rows, board_width):
    """(num_card_ids + 1, rows*width) bool -- what the ENGINE will accept.

    Derived from ClashRoyaleEnv.is_valid_placement, never recomputed here. The
    predicate combines board bounds, Board::isBackRowDeadZone, the per-card
    placementRadius / isSpell / deployAnywhere, and the tower footprint
    clearance; a second copy of that geometry in Python is exactly the drift
    this project has already paid for twice (map geometry, NUM_CARD_IDS).

    Returns None when the binding is absent, and placement_mask then falls
    back to the row-only mask -- i.e. the old, leaky behaviour. That is
    deliberate and LOUD: it warns, because a silent fallback would let a stale
    .pyd quietly restore a defect that was costing 58.7% of the policy's card
    choices.

    Cost is one-off: ~113k pure predicate calls at construction, no per-step
    work at all, because legality does not depend on board state (measured:
    208/288 legal cells on an empty board and 208/288 with six troops down,
    zero cells changed).
    """
    try:
        import clash_royale_env as _env
        if not hasattr(_env.ClashRoyaleEnv, "is_valid_placement"):
            raise AttributeError("is_valid_placement")
    except Exception as exc:  # noqa: BLE001 -- any import/attr failure is the same story
        warnings.warn(
            f"clash_royale_env.is_valid_placement unavailable ({exc}); the "
            "placement mask falls back to own-half rows only. Measured on the "
            "ep~45,800 checkpoint, that let 58.7% of the policy's card choices "
            "be silently rejected by playCard, whose advantages are then pure "
            "noise in the gradient. Rebuild the .pyd -- see "
            "perception/UPSTREAM_REQUESTS.md item 12.",
            RuntimeWarning, stacklevel=2)
        return None, None

    deck = [10, 1, 41, 25, 7, 2, 6, 5]
    probe = _env.ClashRoyaleEnv(deck, deck, 20000)
    probe.reset()

    cells = placement_rows * board_width
    table = torch.zeros(num_card_ids + 1, cells, dtype=torch.bool)
    known = set(_env.get_all_card_ids())
    for cid in range(num_card_ids):
        if cid not in known:
            # Unknown id: leave the row all-False. It can never be the chosen
            # card (it is not in any hand), and an all-True row would be a
            # silent claim about a card the registry does not have.
            continue
        for cell in range(cells):
            y, x = divmod(cell, board_width)
            if probe.is_valid_placement(cid, float(x), float(y), 0):
                table[cid, cell] = True
    table[num_card_ids] = True          # the permissive no-op fallback row

    # --- cells our OWN dead towers hand back (2026-08-27) -----------------
    # The premise above -- "legality does not depend on board state" -- was
    # measured against TROOPS and does NOT hold for a destroyed tower. The
    # tower's 3x3 footprint clears when it dies:
    #
    #   own LEFT princess destroyed   242 -> 251 legal cells (+9)
    #   own RIGHT princess destroyed  242 -> 251 legal cells (+9)
    #   ENEMY princess destroyed      242 -> 242             ( 0)
    #
    # so only our own towers matter, and a table cached on a full board masks
    # those nine cells off forever -- exactly the ground a player defends after
    # losing a tower.
    #
    # Probed over a BOUNDED WINDOW around each tower rather than by re-running
    # the whole board for every card in every tower state: the base pass is
    # ~113k predicate calls and 2.35s, and two more full passes would triple
    # the cost of constructing a net. The window is +/-2 cells, i.e. 25 per
    # tower against an observed 3x3 footprint, and
    # tests/test_placement_mask_after_tower_loss.py does the exhaustive
    # all-cards/all-cells comparison to prove it is wide enough -- an
    # under-sized window would otherwise be a silent mask divergence.
    freed = torch.zeros(2, num_card_ids + 1, cells, dtype=torch.bool)
    centres = _own_princess_centres()
    for slot_i, (cx, cy) in enumerate(centres):
        probe2 = _env.ClashRoyaleEnv(deck, deck, 20000)
        probe2.reset()
        if not probe2.destroy_tower(0, slot_i + 1):   # slots 1=LEFT, 2=RIGHT
            continue
        window = [(x, y)
                  for y in range(max(0, cy - 2), min(placement_rows, cy + 3))
                  for x in range(max(0, cx - 2), min(board_width, cx + 3))]
        for cid in range(num_card_ids):
            if cid not in known:
                continue
            for x, y in window:
                cell = y * board_width + x
                if table[cid, cell]:
                    continue                      # already legal, no delta
                if probe2.is_valid_placement(cid, float(x), float(y), 0):
                    freed[slot_i, cid, cell] = True
    return table, freed


def _own_princess_centres():
    """[(x, y), ...] for team 0's LEFT and RIGHT Princess Towers, as CELLS.

    Derived from the bound ArenaLayout rather than restated -- CLAUDE.md's
    no-second-copies rule, and this geometry has already gone stale twice.
    """
    import clash_royale_env as _env
    y = int(_env.arena_princess_y(0))
    return [(int(_env.ARENA_LEFT_LANE_X), y),
            (int(_env.ARENA_RIGHT_LANE_X), y)]


class MicroRoyaleNet(nn.Module):
    # 9 ערוצים: 0-3 כוחות שלנו (קרבי/טווח/טנק/מבנים), 4-7 אותו דבר ליריב, 8 נהר/גשרים

    # The recurrent width, declared ONCE here because it is the shape every
    # caller needs before it has a net: a fresh (hx, cx) is zeros of this size,
    # and every harness that steps the policy manually builds one. It used to be
    # typed as a bare `256` in five separate files, which is the same duplicated-
    # constant failure CLAUDE.md forbids for engine constants -- `policy_io`
    # re-exports this attribute so nothing has to repeat the literal.
    LSTM_HIDDEN = 256

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

        # (num_card_ids + 1, placement_cells) bool -- אילו תאים המנוע באמת
        # מקבל לכל קלף. השורה האחרונה היא fallback מתירני ל-no-op.
        #
        # נבנה פעם אחת, כי הוא סטטי: נמדד 208/288 תאים חוקיים ללוח ריק ו-
        # 208/288 בדיוק עם שישה כוחות על הלוח, אפס תאים שהשתנו. לכן זו טבלה
        # קבועה ולא שאילתה בזמן ריצה, ואין לה עלות פר-צעד.
        #
        # buffer ולא פרמטר, ומסומן persistent=False: זו עובדה על המנוע ולא
        # משקל נלמד, ושמירתו בצ'קפוינט הייתה הופכת אותו לעותק שני שיכול
        # להתיישן מול המנוע -- בדיוק הסחיפה שהטבלה נועדה למנוע.
        _legal_table, _freed_table = _build_placement_legality(
            num_card_ids, self.placement_rows, board_width)
        self.register_buffer("_placement_legal", _legal_table,
                             persistent=False)
        #: (2, num_card_ids+1, cells) -- cells that become legal when our own
        #: LEFT/RIGHT Princess dies. Same persistent=False reasoning.
        self.register_buffer("_placement_freed", _freed_table,
                             persistent=False)
        #: Cell index of each own Princess centre, for reading its liveness out
        #: of the observation. Built once; cheap.
        # --- legality folded into ONE table indexed by tower state ----------
        # Four variants -- both Princesses alive / left dead / right dead /
        # both dead -- each already OR-ed with the base table. The mask then
        # does a single gather instead of a base gather plus one gather and one
        # OR per tower, which is what made the first version of this feature
        # 53% more expensive than the base mask. Built by pure tensor ORs, so it
        # costs no extra engine probing; 4 x 186 x 612 bools is ~455 KB.
        if _freed_table is not None:
            _by_state = torch.stack([
                _legal_table,                                          # 0: both alive
                _legal_table | _freed_table[0],                        # 1: LEFT dead
                _legal_table | _freed_table[1],                        # 2: RIGHT dead
                _legal_table | _freed_table[0] | _freed_table[1],      # 3: both dead
            ])
        else:
            _by_state = None
        self.register_buffer("_placement_legal_by_state", _by_state,
                             persistent=False)

        self._own_princess_cells = (
            [y * board_width + x for x, y in _own_princess_centres()]
            if _freed_table is not None else [])
        #: The same two centres as FLAT indices into the observation vector.
        #: The spatial half is channel-major, so channel 3 (ally buildings --
        #: the index rewards.shaping.building_hp_end reads) starts at
        #: 3*H*W. Precomputed so placement_mask can read two scalars instead of
        #: reshaping the whole spatial block on every call.
        _ch_ally_buildings = 3
        self._own_princess_flat = [
            _ch_ally_buildings * self.board_height * board_width + c
            for c in self._own_princess_cells]

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
        # שני בלוקים ברוחב num_card_ids עם מחזור הקלפים של היריב (item 24,
        # 2026-08-27): seen[] ו-recency[]. נגזר מה-binding ולא נכתב כאן כמספר
        # -- 2*185 בפייתון היה בדיוק העותק השני שהכלל בראש CLAUDE.md אוסר.
        # נוסף **אחרי** הזנב, כך שגם extra_start וגם כל ההיסטים שלפניו נשארים
        # תקפים בדיוק כפי שהיו.
        self.cycle_block_size = clash_royale_env.ClashRoyaleEnv.CYCLE_BLOCK_SIZE
        self.scalar_size = (1 + hand_size + hand_size * num_card_ids
                            + self.num_extra_scalars + self.cycle_block_size)
        # ההיסט (בתוך scalar_obs) שבו מתחיל הזנב -- נחוץ לראש העזר ולאבחון.
        self.extra_start = 1 + hand_size + hand_size * num_card_ids
        # ...ותחילת בלוקי המחזור, מיד אחרי הזנב. שני ההיסטים נמדדים קדימה
        # מתחילת הקטע הסקלרי ולא לאחור מסופו: מדידה לאחור נשברת בשקט בכל פעם
        # שמשהו נוסף בסוף, וזה בדיוק מה שקרה עכשיו.
        self.cycle_start = self.extra_start + self.num_extra_scalars

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
        self.lstm = nn.LSTMCell(self.lstm_input_dim, self.LSTM_HIDDEN)

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
        # בחזרה ל-34x18 (ראה place_up למטה). הלוגיט של כל תא מחושב אז ע"י אותם
        # משקלים משותפים שפועלים על המאפיינים המקומיים *של אותו אזור לוח* --
        # וזו בדיוק ההטיה האינדוקטיבית הנכונה למשחק שבו ההחלטה היא "איפה".
        # אותו דפוס שבו AlphaStar מייצר ארגומנטים מרחביים.
        self.place_ctx = nn.Linear(256 + CARD_EMBED_DIM, 32)
        # RESIZE + CONV, not ConvTranspose. השינוי הזה תוקן ב-2026-08-09 אחרי
        # שנמדד שהגרסה הקודמת --
        #     ConvTranspose2d(32,32,k=2,s=2) -> ReLU -> ConvTranspose2d(32,16,k=2,s=2)
        # -- מייצרת **הטיה מחזורית קבועה** על מפת הלוגיטים, זהה לכל קלף ולכל
        # מצב משחק.
        #
        # למה זה קורה: ctx מתווסף ב-broadcast, כלומר הוא **אחיד מרחבית**. ל-
        # ConvTranspose2d עם kernel_size=2, stride=2 יש משקל שונה לכל אחת מ-4
        # העמדות בבלוק הפלט, ולכן קלט אחיד *לא* מייצר פלט אחיד -- הוא מייצר
        # דפוס עם מחזור 2, ושתי שכבות כאלה נותנות מחזור 4. הדפוס הזה הוא
        # תכונה של המשקלים בלבד; הקלף יכול רק להזיז את כל המפה בקבוע, הוא לא
        # יכול לשנות איזה תא *בתוך* המחזור מנצח.
        #
        # מה שנמדד על צ'קפוינט ep~129k: 75.0% מהשונות של מפת הלוגיטים מוסברת
        # ע"י (x mod 4, y mod 4) לבדם, ובהתנהגות -- 73.0% מכלל ההנחות נחתו על
        # x = 3 (mod 4) מול null של 22.2% (chi^2 = 94.8, 3 df), 28.9% על שני
        # התאים (11,2)/(11,3), ורק 91 מתוך 288 תאים חוקיים נוצלו אי פעם.
        # זה יוחס בטעות לתמחור המבנים בפונקציית התגמול ("Cannon pathology"),
        # אבל הריכוז היה **בלתי תלוי בקלף** -- Giant 44.4%, Valkyrie 38.9%,
        # Cannon 27.1% -- ואסימטריית תגמול שנוגעת למבנים לא יכולה להסביר
        # Giant שהולך מאחורי מגדל המלך של עצמו.
        #
        # Upsample(nearest) + Conv2d(stride=1) פותר את זה מהשורש: קלט אחיד
        # מרחבית נשאר אחיד אחרי שתי הפעולות (למעט שוליים מה-padding), כי כל
        # תא פלט מחושב באותם משקלים בדיוק. Odena, Dumoulin & Olah,
        # "Deconvolution and Checkerboard Artifacts" (2016).
        #
        # ערוצים מצטמצמים 32->16->8 ולא 32->32->16: conv 3x3 ברזולוציה מלאה
        # יקר בהרבה מ-ConvTranspose עם k=2, וראש המיקום כבר צורך 41% מזמן
        # העדכון. הצמצום מחזיק את העלות קרוב למקור -- ראה את מספרי ה-benchmark
        # ב-git log של השינוי הזה.
        self.place_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),            # 9x5 -> 18x10
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),            # 18x10 -> 36x20
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),    # -> (B,1,36,20)
        )
        # ==========================================
        # 4ג. ענף רזולוציה-מלאה שיורי (2026-08-14)
        # ==========================================
        # מה שהיה חסר: place_up קורא מפה של 9x5 עבור לוח 34x18, כלומר תא מאוחד
        # אחד מכסה ~4x4 משבצות לוח, וההקשר (hx + קלף) נכנס כוקטור **אחיד
        # מרחבית**. לכן אוצר המילים המרחבי של הראש הוא בלוקים של 4 משבצות: הקלף
        # יכול להזיז את כל המפה בקבוע, אבל *איזו משבצת בתוך הבלוק מנצחת* נקבע
        # ע"י משקלים משותפים לכל המצבים. זו מגבלת **ייצוג**, לא כשל אימון --
        # וזה בדיוק מה ש-distill_tactics.py מדד ב-2026-08-14: cross-entropy מול
        # התא המדויק של היועץ ירד 180.9 -> 21.4 בעוד ההתאמה המדויקת (argmax)
        # נשארה **0.0%**. loss שיורד בזמן ש-argmax לא זז אף פעם הוא החתימה של
        # מטרה שהראש לא מסוגל לבטא.
        #
        # הענף כאן קורא את האקטיבציה של ה-trunk **לפני** הפולינג (16x34x18),
        # ברזולוציית משבצת בודדת, מותנה באותו הקשר בדיוק, ומתווסף ללוגיטים
        # הגסים כשארית.
        #
        # למה הקונבולוציה האחרונה מאותחלת ל-אפס: ה-handoff הציע לשרשר לתוך
        # place_up, מה שמשנה את הצורה שלו ולכן **זורק את ראש המיקום המאומן**
        # מכל צ'קפוינט (בדיוק המחיר שתיקון הצ'קרבורד ב-2026-08-09 נאלץ לשלם).
        # המחיר הזה מיותר: ענף שיורי מאותחל-אפס מחשב פונקציה **זהה בדיוק**
        # באתחול, ולכן צ'קפוינט קיים נטען ומתנהג bit-identical, הקלפים שעובדים
        # היום ממשיכים לעבוד, ורק פרמטרים חדשים באמת מתחילים מאפס.
        # הגרדיאנט עדיין זורם: לשכבה המאופסת עצמה יש גרדיאנט לא-אפסי (היא
        # רואה אקטיבציה חיה), אז היא יוצאת מאפס בצעד הראשון והשכבה שמתחתיה
        # מתחילה ללמוד בשני. התנהגות zero-conv סטנדרטית.
        self.place_ctx_hi = nn.Linear(256 + CARD_EMBED_DIM, HIRES_CTX_DIM)
        self.place_hires = nn.Sequential(
            nn.Conv2d(16 + HIRES_CTX_DIM, HIRES_HIDDEN,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(HIRES_HIDDEN, 1, kernel_size=3, stride=1, padding=1),
        )
        nn.init.zeros_(self.place_hires[-1].weight)
        nn.init.zeros_(self.place_hires[-1].bias)

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

    def extract_features_hires(self, obs):
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
          hires_map: (Batch, 16, board_height, board_width) -- האקטיבציה
            שלפני הפולינג הראשון, הקלט של ענף הרזולוציה המלאה
            (ראה place_hires). גם היא לא חישוב נוסף: spatial_map נגזר ממנה.
        """
        # פיצול הווקטור השטוח לחלק המרחבי ולחלק הסקלרי בהתאם לפונקציית observationSize() ב-C++
        spatial_obs = obs[:, :self.spatial_size].view(-1, self.channels, self.board_height, self.board_width)
        scalar_obs = obs[:, self.spatial_size:]

        # ה-trunk מורץ בשני חצאים במקום כ-Sequential אחד, כדי להוציא את
        # האקטיבציה שלפני הפולינג (16x34x18) לענף הרזולוציה המלאה. אותם
        # מודולים, אותו סדר, אותו חישוב -- לא מחושב שום דבר פעמיים, ו-
        # test_trunk_split_is_bit_identical_to_the_sequential מוודא שאין סטייה
        # ולו ב-ulp אחד (סטייה כזו הייתה מזיזה את המאפיינים מתחת לכל צ'קפוינט).
        hires_map = self.cnn_trunk[:2](spatial_obs)
        spatial_map = self.cnn_trunk[2:](hires_map)
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

        return combined, card_embeds, spatial_map, hires_map

    def extract_features(self, obs):
        """שלושת הערכים הישנים בלבד -- ראה extract_features_hires.

        נשמר כעטיפה במקום להרחיב את החתימה, כי לכל קורא קיים (שני המאמנים,
        exploiter, כל סקריפטי המדידה, לולאת ה-live) יש פריקה של בדיוק שלושה
        ערכים. הקוראים החמים בלבד עברו ל-extract_features_hires ומעבירים את
        המפה הלאה; כל השאר נותנים ל-placement_given_card לבנות אותה מחדש מ-obs,
        וזה **אותו חישוב בדיוק** (נבדק ב-test_recomputed_hires_equals_the_
        passed_one), רק בעלות של conv אחד נוסף בנתיב קר.
        """
        combined, card_embeds, spatial_map, _ = self.extract_features_hires(obs)
        return combined, card_embeds, spatial_map

    def hires_features(self, obs):
        """האקטיבציה של ה-trunk לפני הפולינג: (Batch, 16, 34, 18).

        זהו הקלט של ענף הרזולוציה המלאה. מחושב מ-obs כדי שקורא שלא החזיק את
        המפה יוכל לשחזר אותה בעצמו במקום לקבל ראש אחר בשקט.
        """
        spatial_obs = obs[:, :self.spatial_size].view(
            -1, self.channels, self.board_height, self.board_width)
        return self.cnn_trunk[:2](spatial_obs)

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

    def hand_costs_from_obs(self, obs):
        """עלויות קלפי היד (0..10) מתוך ה-obs: טנזור (Batch, hand_size).

        אותו היפוך של החלוקה ב-10 ש-elixir_from_obs עושה, על הסקלרים שמייד
        אחרי האליקסיר -- אותם היסטים בדיוק ש-affordability_mask קורא.

        מחזיר טנזור ולא רשימה, בעקבות elixir_from_obs שמעליו: קורא שרוצה
        רשימה שטוחה לשורה בודדת כותב `[0].tolist()` במפורש, במקום שהפונקציה
        תבליע בשקט את מימד ה-batch.

        קיים כאן כי היו לו שלושה עותקים מילוליים
        (advisors/hybrid_policy.py, eval/gate_ab.py, perception/live/mvp_loop.py),
        כל אחד עם ההיסט וה-10.0 כתובים ביד -- בדיוק דפוס ה"עותק שני של קבוע
        מנוע" ש-CLAUDE.md אוסר, ושכבר התיישן פעמיים בפרויקט הזה.

        לא מקפלים את זה לתוך SolvencyGate.mask: הבדיקה שם נבנית על
        np.zeros(SPATIAL+1), כך שקריאת עלויות הייתה מחזירה פרוסה ריקה ולא
        חריגה -- והבדיקה הייתה עוברת מבלי לבדוק דבר.
        """
        s = self.spatial_size
        return obs[:, s + 1:s + 1 + self.hand_size] * 10.0

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

    def placement_given_card(self, hx, card_embeds, card_idx, obs=None,
                             spatial_map=None, hires_map=None,
                             ctx=None, ctx_hi=None):
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
        בדיוק סוג ההחלטה הדו-מודאלית שהמשחק דורש.

        placement_cells = placement_rows * board_width = 34*18 = **612**, לא
        288. התיקון הזה נעשה ב-2026-08-27: הערך 18*16=288 היה נכון רק כשראש
        המיקום כיסה את החצי שלנו בלבד (16 שורות), ונשאר כאן אחרי שהראש הורחב
        לכל 34 השורות כדי שלחש יוכל לחצות את הנהר. CLAUDE.md מונה את השורה
        הזו בשמה כדוגמה לכלל "אל תשמור עותק שני של קבוע מנוע" -- וזה בדיוק
        אותו כשל: מספר שהיה מדיד פעם, נשאר אחרי שהדבר שהוא תיאר השתנה.

        שים לב ש-288 = 16*18 עדיין מופיע במקומות אחרים בקובץ (208/288 תאים
        חוקיים, 91/288 שנוצלו) ושם הוא **נכון**: זה אזור ההצבה החוקי לכוח
        רגיל, החצי שלנו בלבד. שני המספרים חיים זה לצד זה ומתארים דברים שונים,
        וזו הסיבה שהערבוב היה קל.
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
        # ctx/ctx_hi מסופקים מבחוץ רק ע"י מסלול דחיסת-השורות ב-forward_sequence,
        # שמחשב אותם על **כל** האצווה ואז חותך. הסיבה נמדדה: nn.Linear (GEMM)
        # **אינו** בלתי-תלוי בגודל האצווה על ה-backend הזה -- Linear(280->32)
        # נבדל ב-4.768e-07 ב-forward וב-2.289e-05 ב-grad_W בין אצווה 500 ל-167.
        # שתי השכבות האלה הן ~0.5% מעלות הראש, אז חישוב מלא + חיתוך כמעט חינם,
        # והוא מה שהופך את ה**לוגיטים** בשורות שנשמרות ל-bit-identical.
        #
        # **אבל זה לא הופך את המשקלים ל-bit-identical, ואסור לקרוא את זה כך.**
        # גם grad_W של Conv2d תלוי בגודל האצווה -- אך רק בחלק מהצורות, וזו
        # בדיוק המלכודת: בדיקה ראשונה על 18x10 החזירה "בלתי-תלוי" ונרשמה כאן
        # ככזו, וסריקה על שאר הצורות הפריכה אותה. נמדד 500->184, גרדיאנט נכנס
        # אפס מדויק בשורות שהושמטו:
        #
        #     place_up.1    Conv2d(32,16) 18x10   זהה ביט-לביט
        #     place_up.4    Conv2d(16,8)  36x20   נבדל ב-5.814e-03
        #     place_up.6    Conv2d(8,1)   36x20   נבדל ב-2.808e-03
        #     place_hires.0 Conv2d(24,8)  34x18   נבדל ב-4.883e-03
        #
        # זה חוסם **כל** סכימת דחיסת-שורות מלהיות bit-exact ברמת המשקלים. מה
        # שכן מובטח, ונמדד: הלוגיטים בשורות שנשמרות, וה-loss עצמו.
        if ctx is None:
            ctx = self.place_ctx(torch.cat((hx, chosen_embed), dim=-1))  # (B,32)
        h = spatial_map + ctx.view(-1, 32, 1, 1)
        logit_map = self.place_up(h)                                    # (B, 1, 4*ph, 4*pw)
        logits = logit_map[:, 0, :self.placement_rows, :self.board_width].reshape(
            -1, self.placement_cells)

        # --- ענף הרזולוציה המלאה, כשארית --------------------------------
        # הקצה הגס למעלה יודע "איפה בערך"; זה בוחר את המשבצת בתוך הבלוק.
        # מאותחל-אפס, אז באתחול השורה הזו מוסיפה אפס מדויק (לא "בקירוב"):
        # הקונבולוציה האחרונה מאופסת במשקל ובהטיה, ולכן הפלט הוא טנזור אפס
        # מדויק והחיבור השיורי מדויק. זה מה שמאפשר לצ'קפוינטים קיימים
        # להיטען ולהתנהג bit-identical.
        if hires_map is None:
            if obs is None:
                # אין נפילה שקטה לראש הגס-בלבד. פונקציה אחרת מזו שהריצה את
                # ה-rollout הייתה שוברת את יחס ה-PPO בשקט -- בדיוק מה
                # שהשמירה על spatial_map=None כבר קיימת בשבילו.
                raise ValueError(
                    "placement_given_card דורש hires_map או obs כדי לבנות "
                    "אותו (ראה hires_features). None בשניהם היה מחשב ראש "
                    "אחר מזה שהריץ את ה-rollout.")
            hires_map = self.hires_features(obs)
        if ctx_hi is None:
            ctx_hi = self.place_ctx_hi(torch.cat((hx, chosen_embed), dim=-1))
        h_hi = torch.cat(
            (hires_map,
             ctx_hi.view(-1, HIRES_CTX_DIM, 1, 1).expand(
                 -1, -1, hires_map.shape[2], hires_map.shape[3])), dim=1)
        fine = self.place_hires(h_hi)                                   # (B,1,H,W)
        logits = logits + fine[:, 0, :self.placement_rows, :self.board_width].reshape(
            -1, self.placement_cells)
        if obs is not None:
            # מיסוך חוקיות מותנה-קלף. -inf ולא ערך סופי, מאותה סיבה בדיוק
            # כמו ב-step_lstm_and_card: Categorical מנרמל דרך log_softmax,
            # אז ערך סופי היה משאיר הסתברות זעירה לתא לא חוקי ותרומה
            # לאנטרופיה. תמיד יש לפחות שורה חוקית אחת, אז אף שורה לא יוצאת
            # כולה -inf.
            logits = logits.masked_fill(~self.placement_mask(obs, card_idx), float("-inf"))
        return logits

    def forward_sequence(self, feats_seq, card_embeds_seq, spatial_seq, obs_seq,
                         card_mask_seq, card_idx_seq, reset_seq, hidden_state,
                         extra_card_idx_seq=None, hires_seq=None,
                         active_rows=None):
        """
        חלופה מאוחדת ל-forward_from_features בלולאה על timesteps.
        מתמטית **זהה** לחלוטין -- מוודא בבדיקת bit-identity ייעודית.

        למה זה קיים: רק ה-LSTM באמת רקורנטי. כל מה שאחריו (ראש קלף, ערך, עזר,
        מיקום) הוא נקודתי בזמן, אבל הלולאה הריצה אותו L פעמים על באצ' של B=8.
        על CPU זה נשלט ע"י תקורה ולא ע"י חישוב: נמדד 35.3ms ל-25 קריאות של
        ראשי הקלף/ערך/עזר, מול 0.7ms לקריאה אחת על L*B=200 -- פי 48. ראש
        המיקום מרוויח פי 1.3 (הוא חסום-חישוב, לא חסום-תקורה).

        בסך הכול זה חוסך ~5% מזמן העדכון, לא יותר -- ה-CNN וה-deconv הם עדיין
        84% מהעלות. זה שווה את זה רק בגלל שהשקילות ניתנת להוכחה.

        feats_seq/obs_seq/card_mask_seq/card_idx_seq/reset_seq: (L, B, ...).
        מחזיר card_logits (L,B,hand+1), place_logits (L,B,cells),
        values (L,B), aux_elixir (L,B), ומצב חבוי סופי.
        """
        L, B = feats_seq.shape[0], feats_seq.shape[1]
        hx, cx = hidden_state
        hx_steps = []
        for l in range(L):
            hx, cx = self.lstm(feats_seq[l], (hx, cx))
            # נאסף **לפני** ה-reset, בדיוק כמו בלולאה המקורית: הראשים בצעד l
            # משתמשים במצב שאחרי ה-LSTM ולפני איפוס סוף-אפיזודה.
            hx_steps.append(hx)
            reset = reset_seq[l].unsqueeze(1)
            hx = hx * reset
            cx = cx * reset

        flat_hx = torch.stack(hx_steps).reshape(L * B, -1)
        card_logits = self.card_head(flat_hx)
        mask_flat = card_mask_seq.reshape(L * B, -1)
        card_logits = card_logits.masked_fill(~mask_flat, float("-inf"))
        values = self.value_head(flat_hx).squeeze(-1)
        aux = self.aux_elixir_head(flat_hx).squeeze(-1) * 10.0

        # נבנית פעם אחת ומשותפת לשתי הקריאות למטה. בלי זה מעבר הכיסוי היה
        # משחזר את conv1 של ה-trunk בפעם השנייה על אותו קלט בדיוק.
        flat_obs = obs_seq.reshape(L * B, -1)
        flat_hires = (self.hires_features(flat_obs) if hires_seq is None
                      else hires_seq.reshape(L * B, *hires_seq.shape[2:]))
        flat_embeds = card_embeds_seq.reshape(L * B, *card_embeds_seq.shape[2:])
        flat_spatial = spatial_seq.reshape(L * B, *spatial_seq.shape[2:])

        # --- דחיסת שורות (2026-08-24) ----------------------------------------
        # ראש המיקום הוא ~41% מזמן העדכון והוא רץ כאן **פעמיים** (הקלף שנבחר +
        # משבצת הכיסוי). כל צרכני שני הפלטים ממוסכים ב-decision: actor_loss
        # ב-mb_decision, אנטרופיית המיקום ב-mb_placed (תת-קבוצה שלו), clip_frac
        # ב-mb_decision, ושני חצאי coverage_terms ב-decision. נמדד על
        # model_weights_selfplay.pth לאורך 1500 צעדים: decision דולק ב-0.368
        # מהשורות, כלומר **63.2% מהקונבולוציות האלה מוכפלות באפס מדויק**.
        #
        # active_rows=None משחזר בדיוק את ההתנהגות הקודמת, ולכן שום קורא קיים
        # לא מושפע.
        #
        # המילוי הוא **אפס ולא -inf**, וזו בחירה נושאת-משקל: שורה שכולה -inf
        # נותנת Categorical.entropy() = nan, ו-nan * 0.0 = nan היה מרעיל כל
        # סכום ממוסך בעדכון. כל מילוי **סופי** נותן finite * 0.0 == 0.0 בדיוק,
        # וזה בדיוק מה שהמסלול הישן ייצר שם -- ולכן כל רדוקציה ממוסכת שומרת
        # על הצורה, הסדר והערכים שלה, וה-loss יוצא bit-identical.
        #
        # שתי שכבות ההקשר הליניאריות מחושבות על **כל** האצווה ורק אז נחתכות,
        # כי GEMM אינו בלתי-תלוי בגודל אצווה. ראה placement_given_card -- שם
        # גם מתועד למה **המשקלים** בכל זאת אינם bit-identical (grad_W של
        # Conv2d תלוי-אצווה בצורות 36x20 ו-34x18), וזו מגבלת backend שאף
        # מימוש של דחיסת שורות לא יכול לעקוף.
        def _placement(idx_seq):
            flat_idx = idx_seq.reshape(L * B)
            if active_rows is None:
                return self.placement_given_card(
                    flat_hx, flat_embeds, flat_idx, flat_obs, flat_spatial,
                    hires_map=flat_hires)
            out = flat_hx.new_zeros(L * B, self.placement_cells)
            if active_rows.numel() == 0:
                # אצווה ריקה מגיעה ל-Conv2d; מדלגים לגמרי. זה לא מקרה קצה
                # תיאורטי -- chunk שכולו צעדים כפויים הוא בדיוק מה שסוכן
                # פושט-רגל מייצר, ונמדד P(אין מה להרשות) = 73.9%.
                return out
            rows = torch.arange(L * B, device=flat_hx.device)
            joint = torch.cat((flat_hx, flat_embeds[rows, flat_idx]), dim=-1)
            sub = self.placement_given_card(
                flat_hx[active_rows], flat_embeds[active_rows],
                flat_idx[active_rows], flat_obs[active_rows],
                flat_spatial[active_rows], hires_map=flat_hires[active_rows],
                ctx=self.place_ctx(joint)[active_rows],
                ctx_hi=self.place_ctx_hi(joint)[active_rows])
            return out.index_copy(0, active_rows, sub)

        place_logits = _placement(card_idx_seq)

        # --- placement COVERAGE pass (optional) -----------------------------
        # מפת המיקום של קלף *אחר* מזה שנבחר, על אותם flat_hx/spatial בדיוק.
        # קיים כדי לסגור חור-כיסוי בגרדיאנט: גם actor_loss וגם בונוס
        # האנטרופיה זורמים רק דרך placement_given_card של הקלף ש**נבחר**, ולכן
        # קלף שהמדיניות הפסיקה לשחק לא מקבל שום גרדיאנט מיקום לעולם והראש שלו
        # קופא. נמדד: Cannon/Fireball/Giant החזירו את התא הקבוע (11,0) ב-54%-91%
        # מהמצבים, והנחת Cannon בתא של המדיניות שימרה 121 HP של מגדלים מול 396
        # לתא אקראי חוקי -- גרוע מאקראי, כלומר פונקציה שבורה ולא הערכת ערך.
        #
        # מחזיר לוגיטים בלבד; מי שקורא מחליט מה לעשות איתם (train.py מוסיף
        # אנטרופיה). אין כאן שום פרמטר חדש -- הראש הוא אותו ראש -- ולכן שום
        # צ'קפוינט לא נפסל.
        extra_logits = None
        if extra_card_idx_seq is not None:
            extra_logits = _placement(extra_card_idx_seq).view(L, B, -1)

        return (card_logits.view(L, B, -1), place_logits.view(L, B, -1),
                values.view(L, B), aux.view(L, B), (hx, cx), extra_logits)

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
        mask = mask.reshape(batch, self.placement_cells)

        # ...ועכשיו גם מה שהמנוע באמת מקבל, ולא רק חצי-הלוח.
        #
        # המסכה למעלה מתירה כל שורה בחצי שלנו. GameManager::isValidPlacement
        # לא: הוא דוחה גם את Board::isBackRowDeadZone וגם את טביעת-הרגל של
        # המגדלים. playCard מחזיר false בשקט, בלי חריגה ובלי סיגנל, ולכן
        # פעולה כזו זהה ל-no-op בהשפעתה וה-advantage שלה הוא רעש טהור
        # שנכנס לגרדיאנט. זו בדיוק המחלה שמסכת-האפשרות (affordability) נבנתה
        # כדי לסגור, שנשארה פתוחה בציר המיקום.
        #
        # נמדד על צ'קפוינט ep~45,800, 1,340 צעדי החלטה: 58.7% מבחירות הקלף
        # נדחו כאן. 94.8% מהדחיות בשורה y=0. אפס דליפות באפשרות.
        #
        # הטבלה נגזרת מ-is_valid_placement של המנוע, לא מחושבת מחדש בפייתון:
        # הפרדיקט מרכיב גבולות לוח, dead-zone, placementRadius/isSpell/
        # deployAnywhere וטביעת-רגל של מגדלים. עותק שני של הגיאומטריה הזו הוא
        # בדיוק הסחיפה שהפרויקט כבר שילם עליה פעמיים.
        if self._placement_legal is not None:
            table = self._placement_legal.to(obs.device)
            slot_ids = onehots.argmax(dim=-1)                       # (B, hand)
            # משבצת ריקה: ה-one-hot כולו אפס ו-argmax מחזיר 0, שהוא מזהה קלף
            # חוקי. מסמנים אותה במפורש כדי שלא תיקרא בטעות כקלף 0.
            slot_empty = onehots.sum(dim=-1) <= 0.0                 # (B, hand)
            slot_ids = torch.where(slot_empty, torch.full_like(slot_ids, -1), slot_ids)
            noop = torch.full((batch, 1), -1, device=obs.device, dtype=slot_ids.dtype)
            slot_ids = torch.cat([slot_ids, noop], dim=1)
            chosen_id = slot_ids.gather(1, card_idx.view(-1, 1)).squeeze(1)
            # שורה אחרונה בטבלה = fallback מתירני ל-no-op/משבצת ריקה, כדי
            # שההתפלגות תישאר מוגדרת-היטב (אף שורה לא כולה -inf).
            chosen_id = torch.where(chosen_id < 0,
                                    torch.full_like(chosen_id, table.shape[0] - 1),
                                    chosen_id)
            # Legality, selected by which of OUR OWN Princess Towers are still
            # standing. A tower's 3x3 footprint clears when it dies (+9 cells
            # each, measured), so a table cached on a full board permanently
            # masks the policy out of the ground it defends after losing a
            # tower. ENEMY towers change nothing, which is why only team 0's two
            # Princesses index this.
            #
            # Liveness comes from the ally-building channel at each tower's own
            # centre cell: the encoder marks a tower at its centre only, at
            # hp/MAX_BUILDING_HP (2534/4008 = 0.632 at full) and 0 once dead,
            # and `hp <= 0` is refused by the engine's setters, so `> 0` is
            # exactly "alive". Read as two scalar columns straight out of the
            # flat vector via the precomputed offsets -- reshaping the whole
            # (channels, H, W) block to slice channel 3 copies 612 floats per
            # row for two numbers.
            #
            # ONE gather, into the state-indexed table built in __init__.
            # Doing it as base-gather + per-tower gather + OR measured 53% more
            # expensive than the base mask; this costs the same as the single
            # lookup it replaces.
            if self._placement_legal_by_state is not None:
                by_state = self._placement_legal_by_state.to(obs.device)
                dead_l = (obs[:, self._own_princess_flat[0]] <= 0.0).long()
                dead_r = (obs[:, self._own_princess_flat[1]] <= 0.0).long()
                legal = by_state[dead_l + 2 * dead_r, chosen_id]
            else:
                legal = table[chosen_id]

            mask = mask & legal
        return mask

    def freed_cells_for(self, tower_slot, card_id):
        """{(x, y)} that our own Princess `tower_slot` (1=LEFT, 2=RIGHT) hands
        back for `card_id` when it dies. Diagnostic/testing accessor."""
        if self._placement_freed is None:
            return set()
        row = self._placement_freed[tower_slot - 1, card_id]
        return {(c % self.board_width, c // self.board_width)
                for c in torch.nonzero(row, as_tuple=True)[0].tolist()}

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
