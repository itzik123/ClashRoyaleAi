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
# Hog cycle: Hog Rider (win condition), Cannon, Musketeer, Archers, Knight,
# Minions, Fireball, Valkyrie. Replaces the Giant beatdown deck for one measured
# reason: with the old deck the bot NEVER played Giant (5 elixir) in any probe
# across four full training runs -- only 5-7 of the 8 cards were ever used.
#
# The cause is an interaction between cost and the affordability action mask.
# Elixir regenerates 0.35 per decision step, so a 5-cost card is only legal
# after ~14 consecutive steps of not spending; the policy almost always spends
# on something cheaper first, so the expensive card's slot is masked out nearly
# every time it is checked and never accumulates gradient. The fix is not a
# bigger entropy bonus, it is a curve where no card is systematically starved:
#
#   old deck: costs 3-5, avg 3.75, spread 2   -> Giant effectively unreachable
#   this one: costs 3-4, avg 3.50, spread 1   -> worst case ~11 steps, and every
#                                                card competes on equal footing
#
# Hog Rider is a genuine building-targeter win condition (same archetype class
# train_selfplay.py's _WIN_CONDITION_IDS treats as "the real threat"), so the
# deck still has a way to actually close games. Fireball is kept and is finally
# usable: spells can now be aimed past the river (see MicroRoyaleNet's placement
# mask and the get_card_info binding added for it).
#
# ---------------------------------------------------------------------------
# 2026-07-30: SWITCHED BACK to the Giant-beatdown deck above, on purpose, for a
# reason that outranks the cost-curve argument: it is the deck actually played
# in the 8 real matches recorded in perception/assets/recordings/. Training on
# a different deck than the demonstrations would make the human data unusable
# for behaviour cloning (bc_pretrain.py), and human demonstrations are the
# highest-value unblocked item on the roadmap -- AlphaStar's supervised stage
# was load-bearing, not optional.
#
# The known risk is real and documented above, not hand-waved: costs 3-5, avg
# 3.75, spread 2, and Giant (5) was never played once across four full runs.
# Two things changed since that measurement, so the outcome is genuinely open:
#   * the affordability mask now makes "cannot afford Giant" an explicit,
#     observable fact rather than a silent playCard failure, so the no-op that
#     banks toward it is a representable choice instead of wasted motion;
#   * card-head entropy is now adaptively controlled toward 0.35 of maximum
#     (ENTROPY_TARGET_CARD), which exists specifically to fight the card-head
#     collapse that starves an expensive slot.
#
# WATCH: whether Giant (id 2) is ever played. If phase 2's Cards/Game sticks
# near 6/8 and a probe shows Giant at ~0 usage, the cost curve won again and
# the honest fix is a cheaper win condition, NOT more entropy (already tried,
# already measured to fail).
# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
# 2026-08-16: SWITCHED to the classic 2.6 HOG CYCLE.
#
#   Hog Rider(15) 4 | Musketeer(6) 4 | Cannon(25) 3 | Ice Golem(40) 2
#   Skeletons(24) 1 | Ice Spirit(72) 1 | The Log(33) 2 | Fireball(7) 4
#   total 21, average 2.625, spread 3 (1..4)
#
# This deliberately gives up the recorded-match tie the Giant deck was reverted
# FOR (2026-07-30, above): the 8 recordings play the Giant deck, so behaviour
# cloning from them is not available against this one. That is a real cost and
# it is being paid on purpose -- BC was blocked on the extraction step anyway,
# and the deck is being changed to test the LEARNING MECHANISM.
#
# It also removes the risk that reversion re-accepted. The Giant deck's failure
# mode was a starved 5-cost win condition: at 0.35 elixir per decision a 5-cost
# card is legal only after ~14 consecutive non-spending steps, and Giant was
# never played once across four full runs. NOTHING here costs more than 4, and
# two cards cost 1, so no slot can be systematically masked out of existence.
#
# MEASURED IN THIS ENGINE before switching (all 8 verified functional, not just
# present in the registry):
#   Hog Rider    building-targeter, crosses and deals 317 tower damage in 40 s
#   Cannon       prevents 3,823 HP against a lone enemy Hog
#   The Log      ground-only rolling spell, 12 -> 5 bodies on a Skeleton clump
#   Fireball     12 -> 3 bodies on the same clump
#   Skeletons    3 bodies | Ice Spirit 1, hits air | Ice Golem tank, ~17 s
#   Musketeer    ranged, hits air
#
# WATCH: Hog Rider (id 15) usage and Cannon-vs-Hog defence. The Giant deck's
# tell was a win condition that was never played; the equivalent tell here is
# Hog usage collapsing, and the honest response would again be the cost curve
# or the gradient path -- NOT more entropy.
# ---------------------------------------------------------------------------
#
# Verified against the live registry: all 8 ids exist, no Champions, and
# validate_deck_slots returns "" (legal).
DEFAULT_DECK = [15, 6, 25, 40, 24, 72, 33, 7]


def _find_win_condition(deck):
    """The deck's win condition = its BUILDING-TARGETER, found by asking the engine.

    A building-targeter is the only unit that walks past your defenders to hit
    a tower, which is what makes it the win condition -- and it is the reason
    the win-condition reward term can use `get_damage_dealt_by_card` at all:
    such a unit attacks nothing else, so its total damage IS tower damage.

    DERIVED, not written down. `get_card_info` exposes cost/is_spell/
    is_building but no archetype, so the class is recovered the same way
    perception_encoder.build_card_table recovers every other attribute: inject
    the card onto an empty board and see which type channel the observation
    lights up. ClashEnv.h's type offsets are 0 melee / 1 ranged / 2
    BUILDING-TARGETER / 3 building. A hardcoded `WIN_CONDITION_ID = 15` would
    silently mean "Hog Rider" forever and be wrong the next time the deck
    changes -- exactly the drift that made a test inject a Musketeer while its
    comment said Archers.

    Returns None if the deck has no building-targeter, which is a legitimate
    deck (the win-condition reward term then contributes nothing rather than
    crashing or, worse, crediting an arbitrary card).
    """
    import numpy as _np
    E = clash_royale_env.ClashRoyaleEnv
    plane = E.BOARD_WIDTH * E.BOARD_HEIGHT
    found = []
    for cid in deck:
        info = clash_royale_env.get_card_info(cid)
        if info["is_spell"] or info["is_building"]:
            continue
        env = E(deck, deck, 100)
        env.reset()
        env.inject(cid, 9.0, 8.0, 0)
        env.step(E.HAND_SIZE, 0.0, 0.0, 1)
        obs = _np.asarray(env.get_observation_for_team(0), _np.float32)
        # channel 2 = ally building-targeter ("tank") in ClashEnv's 0-3 block
        if obs[2 * plane:3 * plane].max() > 1e-6:
            found.append(cid)
    if not found:
        return None
    # More than one (e.g. Hog + Ice Golem, which also targets buildings): the
    # win condition is the one that actually threatens a tower, i.e. the
    # highest-cost such card. Ice Golem is a 2-cost shield, not a win condition.
    return max(found, key=lambda c: clash_royale_env.get_card_info(c)["cost"])


WIN_CONDITION_ID = _find_win_condition(DEFAULT_DECK)
# Card id of the deck's only spell -- see train.lethal_spell_potential.
train_FIREBALL_ID = 7

# How many Champion ability slots this deck actually has (0, 1 or 2 -- see
# CardRegistry::validateDeckSlots).
#
# Now DERIVED from the engine rather than hand-maintained: get_card_info was
# added to bindings.cpp precisely so this (and the spell flag below) stops
# being a constant somebody has to remember to update in lockstep with
# DEFAULT_DECK. Change the deck and this follows automatically.
#
# Why it matters: MicroRoyaleNet only builds its two Champion-ability heads
# when this is > 0. With a Champion-less deck those heads were pure noise --
# they widened the PPO ratio's variance via total_logprob and contributed up
# to 2*log(2)=1.386 to the entropy bonus, so the entropy budget was being
# spent keeping two irrelevant coin flips maximally random.
DEFAULT_DECK_ABILITY_SLOTS = sum(
    1 for _cid in DEFAULT_DECK
    if clash_royale_env.get_card_info(_cid)["is_champion"]
    or clash_royale_env.get_card_info(_cid)["is_hero"]
)

# Per-hand-slot spell flags are what let the placement mask open the enemy half
# for spells only -- see MicroRoyaleNet.placement_mask. Spells are exempt from
# GameManager::isValidPlacement's own-half restriction, but the action space
# used to cap target_y at getOwnHalfMaxY() for EVERY card, so a Fireball could
# physically never be thrown past the river. That made one of the deck's eight
# cards unusable for its actual purpose no matter how long training ran.
SPELL_BY_CARD_ID = {
    _cid: clash_royale_env.get_card_info(_cid)["is_spell"]
    for _cid in clash_royale_env.get_all_card_ids()
}


def get_all_card_ids():
    """All ids CardRegistry currently has registered, derived live from the
    engine instead of a hardcoded range+exclusion list. That kind of list goes
    stale the moment a card is added to (or removed from) CardRegistry.h --
    already happened once: a hardcoded range(46) pool silently stopped covering
    new cards once the roster grew past 45, and a hand-maintained exclusion
    list is just as easy to get wrong in the other direction (mistaking real
    registered ids for gaps). See CardRegistry.h's getAllCardIds() free function."""
    return clash_royale_env.get_all_card_ids()


def _to_scalar(val):
    """Unwrap whatever an action field arrived as into a plain Python scalar.

    Callers build the action dict several different ways -- gym's own vectorized
    space hands over 0-d/1-element numpy arrays, train.py's PPO loop passes
    `np.array([i])`, and the hand-written probes pass bare ints -- so this has to
    accept all three. Defined once at module level and imported by
    `train_selfplay.py` rather than nested inside both `step()` methods, which is
    where it used to live as two byte-identical copies.
    """
    if hasattr(val, "item"):
        return val.item()
    if isinstance(val, (list, tuple, np.ndarray)):
        return val[0]
    return val


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
            # Full board height, not get_own_half_max_y(). Spells are exempt
            # from isValidPlacement's own-half restriction, so capping the
            # DECLARED space at the own half made Fireball unable to cross the
            # river at all. Per-card legality is enforced by
            # MicroRoyaleNet.placement_mask instead -- the space describes what
            # the engine will accept from SOME card, the mask decides which
            # cells are legal for the card actually chosen this step.
            "target_y": spaces.Box(low=0.0, high=float(clash_royale_env.ClashRoyaleEnv.BOARD_HEIGHT - 1),
                                   shape=(1,), dtype=np.float32),
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
            # Correct-by-construction (not random.sample(get_all_card_ids(), 8)
            # + hope): that naive draw includes Champions/Evolutions, which
            # only some deck slots accept, so it would routinely violate
            # CardRegistry::validateDeckSlots -- see sampleRandomDeck's own
            # comment in ClashEnv.h.
            self.game.set_opponent_deck(clash_royale_env.sample_random_deck())
        else:
            self.game.set_opponent_deck(self.opp_deck)

        obs_list = self.game.reset()
        obs = np.array(obs_list, dtype=np.float32)
        return obs, {}

    def step(self, action, skip_frames=10):
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
            # Towers only. compute_shaping() needs tower damage and
            # deployed-building damage priced differently -- see
            # train.tower_potential.
            # --- inputs for the lethal-spell PBRS term (train.lethal_spell_potential)
            # Enemy tower HP in ABSOLUTE points. The observation carries these
            # normalized in its appended scalar tail (indices 6-8 = enemy
            # king/left/right), so this is a re-scale of data the net already
            # sees rather than a new engine call.
            # Heuristic-1 inputs (train.spell_value_shaping). BOTH are needed:
            # value-destroyed alone makes a whiffed spell free, which is the
            # guaranteed-zero trap that parked the Cannon in a back corner.
            "fireball_value_killed": self.game.get_elixir_value_killed_by(train_FIREBALL_ID, 0),
            "fireball_elixir_spent": self.game.get_elixir_spent_on_card(train_FIREBALL_ID, 0),
            "enemy_tower_hp": np.asarray(obs[-clash_royale_env.ClashRoyaleEnv.NUM_EXTRA_SCALARS:][6:9], dtype=np.float32) * clash_royale_env.ClashRoyaleEnv.MAX_BUILDING_HP,
            "fireball_in_hand": float(train_FIREBALL_ID in list(self.game.get_hand())),
            "team0_tower_damage": self.game.get_tower_damage_dealt(0),
            "team1_tower_damage": self.game.get_tower_damage_dealt(1),
            "team1_building_damage": self.game.get_building_damage_dealt(1),
            # Cumulative elixir spent this match (sum of played cards' cost) --
            # feeds train.py's elixir-trade shaping term (reward for making the
            # enemy spend more than we do, independent of the damage itself).
            "team0_elixir_spent": self.game.get_elixir_spent(0),
            "team1_elixir_spent": self.game.get_elixir_spent(1),
            # Surviving tower counts, for the discrete crown term in
            # compute_shaping() -- see W_TOWER_DESTROYED.
            "team0_towers_alive": self.game.get_towers_alive(0),
            "team1_towers_alive": self.game.get_towers_alive(1),
            # Cumulative damage dealt by the deck's WIN CONDITION -- input to
            # train.compute_shaping's win-condition term. See WIN_CONDITION_ID
            # below for how the card is identified, and ClashEnv::
            # getDamageDealtByCard for what this counter does and does not
            # measure (it is damage to anything, which for a building-targeter
            # is tower damage plus any enemy deployed building in the way).
            "team0_wincon_damage": (self.game.get_damage_dealt_by_card(WIN_CONDITION_ID, 0)
                                    if WIN_CONDITION_ID is not None else 0),
            # SUPERVISION TARGET for the network's auxiliary elixir head, and
            # nothing else. Deliberately delivered through info -- NOT through
            # the observation -- because the opponent's current elixir is
            # hidden information a human cannot read off the screen. Putting
            # it in the observation would train a policy that silently depends
            # on something perception/ can never supply from a real match.
            # See MicroRoyaleNet.predict_opp_elixir.
            "opp_elixir": self.game.get_elixir_for_team(1),
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