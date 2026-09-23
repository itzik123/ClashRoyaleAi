import random

import gymnasium as gym
from gymnasium import spaces
import numpy as np

import clash_royale_env
from python_ai.advisors import card_probes
from python_ai.envs import scenario_offense, scenarios
from python_ai.opponents import deck_pool

# Real-meta Balloon Freeze deck, replacing the earlier Giant Beatdown archetype
# entirely -- deliberate full restart (fresh net, not resumed), not a tune-up.
# Reasons: (1) the old deck was picked back when CardRegistry had far fewer
# cards implemented and doesn't reflect the roster available now; (2) pipeline
# #1's phase 2 (randomized opponent decks, see trainers/train.py's PHASE2_WIN_RATE_GATE)
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
# envs/scenarios.py's _WIN_CONDITION_IDS treats as "the real threat"), so the
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
#
# The deck itself now lives in `python_ai.deck`, which reads CLASH_DECK -- ids or
# names -- and validates it against the engine. Re-exported here because ~40
# call sites import it from this module. To train a different deck, set
# CLASH_DECK; do not edit this line.
from python_ai.deck import DEFAULT_DECK  # noqa: E402


def _find_win_condition(deck):
    """The deck's win condition, or None -- delegated to the ONE resolver.

    This used to be its own copy: "the most expensive BUILDING-TARGETER", found
    by injection. The injection technique was right; the ranking was not, and
    it was a second copy of a question the teacher also answers. Measured
    2026-09-15 (audit 07 F2, audit 05):

      * it returned None for every deck whose route to a tower is a siege
        building, a spawning spell or a deploy-anywhere troop (Mortar, X-Bow,
        Graveyard, Goblin Barrel, Miner) -- silently switching off
        `W_WIN_CONDITION_DAMAGE`, the term added precisely so the win condition
        is worth playing;
      * on a deck with no real building-targeting win condition it promoted a
        2-elixir Ice Golem, so the reward paid the agent to throw its tank at
        towers -- a wrong objective, which is worse than a missing one.

    `teacher.resolve_win_condition` ranks eligible cards (building-targeter,
    deploy-anywhere, siege building, spawning spell) by MEASURED tower damage
    per elixir from where each is actually played, with a floor, so the agent's
    reward and the teacher's plan now name the same card.

    Returns None when nothing qualifies, which is a legitimate deck; the
    win-condition term then contributes nothing, and `validate_deck` says so
    loudly at startup rather than letting it go silent.
    """
    from python_ai.opponents import teacher as _teacher
    return next((c for c, r in _teacher.card_roles(list(deck)).items()
                 if r == "wincon"), None)


WIN_CONDITION_ID = _find_win_condition(DEFAULT_DECK)


def deck_spell_info(game, spell):
    """The five `spell_*` info keys for team 0's damage spell, or all zeros.

    `spell` is `card_probes.damage_spell(deck)` -- (card_id, tower_damage, cost)
    or None. It replaces `train_FIREBALL_ID = 7`, which keyed both spell reward
    terms to Fireball whatever the deck (TODO.md 00.3): with a Rocket in its
    place they read exactly zero on every one of 2,372 measured steps.

    ONE builder for both envs. Pipeline 2 builds its ClashRoyaleEnv directly
    rather than through this module, and a key only one pipeline supplies is a
    term that silently contributes zero for a whole phase -- the shape
    `team0_wincon_damage`'s comment in selfplay_env already records.
    """
    if spell is None:
        return {"spell_in_hand": 0.0, "spell_value_killed": 0.0,
                "spell_elixir_spent": 0.0, "spell_damage": 0.0, "spell_cost": 0.0}
    card_id, damage, cost = spell
    return {
        "spell_in_hand": float(card_id in list(game.get_hand())),
        "spell_value_killed": game.get_elixir_value_killed_by(card_id, 0),
        "spell_elixir_spent": game.get_elixir_spent_on_card(card_id, 0),
        "spell_damage": float(damage),
        "spell_cost": float(cost),
    }


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
        
        # No config passed: fall back to an empty dict.
        if env_config is None:
            env_config = {}
            
        # Every setting is read with a sensible default; nothing is hard-coded.
        ai_deck = env_config.get("ai_deck", list(DEFAULT_DECK))
        # Default: the opponent plays exactly the same deck (a mirror match).
        # Measured: a random deck from the registry wins tens of percentage points
        # more than the fixed deck (in this engine, giant units crush a cycle deck),
        # so training against random decks starts the agent in a nearly lost game
        # and the win/loss signal disappears. The phase-1 trainer overrides this
        # default with the meta-deck pool (`deck_pool`, below).
        self.ai_deck = list(ai_deck)
        # The card the two spell reward terms follow, measured by injection and
        # cached per process -- see deck_spell_info.
        self._damage_spell = card_probes.damage_spell(self.ai_deck)
        self.opp_deck = env_config.get("opp_deck", list(ai_deck))
        self.current_opp_deck = list(self.opp_deck)
        self.randomize_opp_deck = env_config.get("randomize_opp_deck", False)

        # --- THE META-DECK POOL (2026-09-03) --------------------------------
        # `deck_pool` is None (off, and every existing caller is unchanged),
        # True (the whole enabled pool), or a list of deck NAMES to restrict to.
        # It takes precedence over randomize_opp_deck, which draws uniformly
        # from the 132-card registry and is a completely different distribution
        # -- see opponents/deck_pool.py's docstring for why "random deck" was
        # never a substitute for "real deck".
        pool_cfg = env_config.get("deck_pool", None)
        self.deck_pool = None
        self.current_deck_name = None
        #: Per-worker local win-rate estimate, {deck name: rate}. Decentralized
        #: exactly like selfplay_env's `pfsp_stats`: 8 workers each converge a
        #: reasonable local estimate rather than synchronizing every episode.
        self.deck_pool_stats = {}
        self._deck_pool_counts = {}
        if pool_cfg:
            all_decks = deck_pool.load_pool()
            if pool_cfg is not True:
                want = set(pool_cfg)
                all_decks = [d for d in all_decks if d.name in want]
                if not all_decks:
                    raise ValueError(f"deck_pool={pool_cfg!r} matched no deck")
            self.deck_pool = all_decks
            # Seeded from the pool file's measured priors, not from a flat 0.5
            # -- see deck_pool.NEUTRAL_PRIOR. A uniform start feeds a fresh net
            # the unwinnable decks as often as the mirror for the few thousand
            # episodes the EWMA needs to separate them.
            self.deck_pool_stats = {d.name: d.prior_win_rate for d in all_decks}
            #: Matches this worker has actually played per deck, for the
            #: count-weighted update in `_record_deck_outcome`.
            self._deck_pool_counts = {d.name: 0 for d in all_decks}
            # `is None`, not `or`: worker 0's seed is 0, and `0 or X` would
            # silently hand exactly one worker per run an unseeded stream --
            # the kind of off-by-falsy that makes a "seeded" run irreproducible
            # in one env out of eight and nowhere else.
            seed_cfg = env_config.get("scenario_seed")
            self._deck_rng = random.Random(
                random.randrange(1 << 30) if seed_cfg is None else seed_cfg)
        max_ticks = env_config.get("max_ticks", 3600)
        # Curriculum hook: the opponent's elixir-rate multiplier (1.0 = normal; higher simulates an aggressive, nearly unlimited opponent).
        opp_elixir_multiplier = env_config.get("opp_elixir_multiplier", 1.0)
        # Tower Troops: per-match config like the deck itself, not a per-step
        # action -- see GameManager's constructor. NONE (the default)
        # reproduces the original hardcoded Princess Tower unchanged.
        ai_tower_troop = env_config.get("ai_tower_troop", clash_royale_env.TowerTroopType.NONE)
        opp_tower_troop = env_config.get("opp_tower_troop", clash_royale_env.TowerTroopType.NONE)

        self.game = clash_royale_env.ClashRoyaleEnv(ai_deck, self.opp_deck, max_ticks, ai_tower_troop, opp_tower_troop)
        self.game.set_opponent_elixir_multiplier(opp_elixir_multiplier)

        # --- WHO PLAYS TEAM 1 ------------------------------------------------
        # "builtin" (the default) is unchanged: game.step() runs the C++
        # HeuristicOpponent from inside ClashEnv::opponentTurn.
        #
        # "teacher" routes through stepSelfPlay instead, which DELIBERATELY
        # never calls opponentTurn() -- so the C++ heuristic is absent on that
        # path, which is exactly what is wanted. Both entry points accumulate
        # calculateReward() identically, so the reward stream is unchanged.
        #
        # WHY THE OPPONENT CHANGED AT ALL: the elixir-multiplier curriculum
        # priced the win condition negatively (CLAUDE.md's 1.5x hypothesis,
        # monotone across 1.0/1.25/1.5x). Deleting the multiplier alone is not
        # enough -- at 1.0x the C++ heuristic is beaten ~100%, so that converts a
        # mispriced environment into a zero-gradient one. Difficulty moves to
        # COMPETENCE (teacher.TEACHER_STAGES) at a symmetric 1.0x economy.
        self.opponent_kind = env_config.get("opponent", "builtin")
        self.teacher = None
        # Offensive scenario injection ("Proposal A"), DEFAULT OFF -- see
        # scenario_offense.py for why it is demoted to an accelerator rather
        # than a mechanism. It changes the state distribution, not the payoff,
        # so it can only help once the payoff is right.
        self.offensive_scenario_prob = float(
            env_config.get("offensive_scenario_prob",
                           scenario_offense.OFFENSIVE_SCENARIO_PROB))
        self._scenario_rng = np.random.default_rng(
            env_config.get("scenario_seed", None))
        self.last_scenario = None

        # DEFENSIVE scenario injection in phase 1. Pipeline 2 has had this
        # since 2026-08-09; `trainers/train.py`'s own comment recorded that it
        # "never runs here", and the 2026-08-28 run showed what that costs: in
        # 32,680 episodes the agent never met a manufactured "defend or lose
        # the tower" moment, and its Cannon / Log / Fireball play probabilities
        # settled at 0.0053 / 0.0091 / 0.0011. Those cards are near-worthless
        # on a quiet board, so declining them was CORRECT -- the fix belongs in
        # the state distribution, not in the card head.
        #
        # DEFAULT 0.0, so an existing phase-1 run's distribution is unchanged
        # unless a caller opts in. `envs/scenarios.SCENARIO_INJECTION_PROB`
        # (0.30) is the value pipeline 2 uses and the one to pass here.
        self.defensive_scenario_prob = float(
            env_config.get("defensive_scenario_prob", 0.0))
        #: Truncation window in bot-steps for the active scenario, or None.
        self.scenario_max_steps = None
        self.scenario_steps_taken = 0
        self.scenario_defensive = False
        if self.opponent_kind == "teacher":
            from python_ai.opponents.teacher import UtilityTeacher
            self.teacher = UtilityTeacher(self.opp_deck, team=1)
            self.teacher.set_stage(int(env_config.get("teacher_stage", 0)))
            self.teacher.reset()

        self.action_space = spaces.Dict({
            # Index HAND_SIZE is the no-op: play no card this step. The engine ignores
            # a cardIndex outside [0, HAND_SIZE), so the agent can bank elixir instead
            # of being forced to play. The bounds are read live from the engine, not
            # hard-coded, so nothing needs syncing by hand if the C++ side changes the
            # board or hand size.
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
            # Champion ability activation: two independent slots (1 = Heroic, 2 = Wild
            # Card; see CardRegistry::validateDeckSlots), because a deck can hold up to
            # two Champions at once. Each is 0 = do not activate, 1 = activate now if a
            # Champion is deployed in that slot, off cooldown, with enough elixir
            # (otherwise a silent no-op, like card_index's no-op). Defaults to False
            # wherever the key is not passed.
            "activate_ability_slot1": spaces.Discrete(2),
            "activate_ability_slot2": spaces.Discrete(2),
        })
        
        obs_size = self.game.observation_size()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # super().reset() seeds the WRAPPER's np_random, which this env reads
        # NOWHERE -- so until this line existed the argument was accepted and
        # silently dropped, which is worse than refusing it: gymnasium's
        # contract says a caller may rely on it. ClashEnv::seed (2026-08-21)
        # seeds BOTH engine generators (HeuristicOpponent, and the opening
        # hand / cycle order) and ends in reset(), so it is a drop-in for the
        # self.game.reset() below rather than an extra one.
        #
        # GUARDED on `is not None` deliberately: the gymnasium convention is
        # that a seed is passed ONCE and later reset()s continue the stream.
        # Seeding unconditionally would make every episode of a run identical
        # -- a far worse failure than the one being fixed.
        #
        # NOT sufficient for reproducibility when randomize_opp_deck is True:
        # the deck comes from clash_royale_env.sample_random_deck(), whose
        # generator is a function-local static that nothing can seed. See
        # perception/UPSTREAM_REQUESTS.md item 23, section C.
        if seed is not None:
            self.game.seed(int(seed))
        if self.deck_pool is not None:
            # THE META-DECK POOL (2026-09-03). Sampled here, per worker, per
            # episode -- the same decentralized shape selfplay_env uses for
            # PFSP opponents and for the same reason: AsyncVectorEnv workers
            # are separate OS processes, so a per-index call from the trainer
            # does not exist and synchronizing a stats dict every episode would
            # cost more than the local estimate is worth at 8 workers.
            #
            # Per EPISODE and not per rung: a PPO batch then contains a mix of
            # matchups, which is what stops the policy specialising into
            # whichever deck the current rung happens to be showing it. That is
            # the same argument phase 2's league makes one level up.
            picked = deck_pool.sample_deck(
                self.deck_pool, self._deck_pool_weights(), self._deck_rng)
            self.current_deck_name = picked.name
            random_deck = list(picked.card_ids)
            self.game.set_opponent_deck(random_deck)
            # Recorded so a caller (and the teacher-sync test) can see WHICH
            # deck this episode is actually against -- self.opp_deck is the
            # fixed fallback and deliberately stays unchanged here.
            self.current_opp_deck = random_deck
            # Same reason as set_opponent_deck() below -- and this path is the
            # easier one to miss, because it writes straight to self.game.
            if self.teacher is not None:
                self.teacher.set_deck(random_deck)
        elif self.randomize_opp_deck:
            # Correct-by-construction (not random.sample(get_all_card_ids(), 8)
            # + hope): that naive draw includes Champions/Evolutions, which
            # only some deck slots accept, so it would routinely violate
            # CardRegistry::validateDeckSlots -- see sampleRandomDeck's own
            # comment in ClashEnv.h.
            random_deck = list(clash_royale_env.sample_random_deck())
            self.game.set_opponent_deck(random_deck)
            # Recorded so a caller (and the teacher-sync test) can see WHICH
            # deck this episode is actually against -- self.opp_deck is the
            # fixed fallback and deliberately stays unchanged here.
            self.current_opp_deck = random_deck
            # Same reason as set_opponent_deck() below -- and this path is the
            # easier one to miss, because it writes straight to self.game.
            if self.teacher is not None:
                self.teacher.set_deck(random_deck)
        else:
            self.game.set_opponent_deck(self.opp_deck)
            self.current_opp_deck = list(self.opp_deck)

        obs_list = self.game.reset()
        # New match: drop the played-tick baseline so the next poll re-seeds
        # against THIS episode's clock instead of comparing the new match's
        # ticks to the previous one's and reporting a phantom play on step 1.
        self._opp_last_tick = {}
        # Scenario injection rewrites the freshly-reset state, so the
        # observation has to be RE-READ afterwards -- reset()'s return value
        # describes the position before the rewrite.
        self.last_scenario = None
        self.scenario_max_steps = None
        self.scenario_steps_taken = 0
        self.scenario_defensive = False
        if (self.defensive_scenario_prob > 0.0
                and self._scenario_rng.random() < self.defensive_scenario_prob):
            obs_list = self._apply_defensive_scenario()
        if self.offensive_scenario_prob > 0.0:
            self.last_scenario = scenario_offense.apply_offensive_scenario(
                self.game, self._scenario_rng, list(self.game.get_hand())
                and self.ai_deck, self.offensive_scenario_prob)
            if self.last_scenario is not None:
                obs_list = self.game.get_observation_for_team(0)
        # New match, new weight profile and lane bias. A FULLY deterministic
        # opponent is memorizable in one counter-line, which is the
        # single-opponent version of the echo chamber this curriculum exists to
        # avoid -- see UtilityTeacher.reset.
        if self.teacher is not None:
            self.teacher.reset()
        obs = np.array(obs_list, dtype=np.float32)
        return obs, {}

    def _apply_defensive_scenario(self):
        """Rewrite the freshly-reset state into a defensive emergency.

        Mirrors `MicroRoyaleSelfPlayEnv.reset`'s injection deliberately, so the
        two pipelines present the SAME distribution of threats and a phase-2
        policy is not meeting them for the first time at handoff.

        Returns the re-read observation: `game.reset()`'s return value
        describes the position BEFORE this rewrite.
        """
        scenario = scenarios.sample_scenario(self._scenario_rng)

        warmup = scenario.get("warmup_ticks", 0)
        noop = clash_royale_env.ClashRoyaleEnv.HAND_SIZE
        if warmup:
            # Banked elixir, BEFORE the spawns so injected units do not walk
            # during the warm-up.
            self.game.step_self_play(noop, 0.0, 0.0, noop, 0.0, 0.0, warmup)

        for card_id, x, y in scenario["spawns"]:
            self.game.inject_enemy(card_id, x, y)

        # inject_enemy only QUEUES units into pendingEntities; they are absent
        # from the observation until a step commits them. One 1-tick no-op
        # makes the threat visible in the very first observation the net acts
        # on -- otherwise a Hog gets a full step of travel before the policy
        # has ever seen it.
        #
        # step_self_play, not step: it never calls opponentTurn(), so this
        # commit tick cannot hand the teacher a free extra decision. Same
        # reason pipeline 2 uses it here.
        self.game.step_self_play(noop, 0.0, 0.0, noop, 0.0, 0.0, 1)

        self.last_scenario = scenario["name"]
        self.scenario_max_steps = scenario["max_steps"]
        self.scenario_defensive = bool(scenario.get("defensive", False))
        return self.game.get_observation_for_team(0)

    def _poll_opponent_play(self):
        """Which card team 1 played since the last poll, or -1 for none.

        Baseline-and-diff over `get_last_played_tick`, one call per card in
        the opponent's CURRENT deck (8, re-read each call because
        `randomize_opp_deck` changes it per episode).

        The first poll of an episode always returns -1 and only establishes
        the baseline. That is correct rather than a lost sample: with the
        baseline empty there is no way to distinguish "played just now" from
        "the engine's initial value", and inventing a label there would put a
        wrong answer into the supervision set.

        When a 10-tick step contains more than one play, the EARLIEST is
        returned -- the label is "the next card they play", so the first one
        after the observation is the answer.
        """
        last = getattr(self, "_opp_last_tick", None)
        if last is None:
            last = self._opp_last_tick = {}
        # current_opp_deck, NOT opp_deck: under randomize_opp_deck the latter
        # stays the fixed fallback (reset() says so explicitly), so polling it
        # would watch eight cards the opponent does not hold and report -1 for
        # every play of the entire episode -- a silently empty supervision set.
        deck = getattr(self, "current_opp_deck", None) or self.opp_deck
        best_tick, best_card = None, -1
        for card in set(deck):
            tick = self.game.get_last_played_tick(1, card)
            prev = last.get(card)
            last[card] = tick
            if prev is not None and tick > prev:
                if best_tick is None or tick < best_tick:
                    best_tick, best_card = tick, card
        return best_card

    def step(self, action, skip_frames=10):
        card_idx = int(_to_scalar(action["card_index"]))
        target_x = float(_to_scalar(action["target_x"]))
        target_y = float(_to_scalar(action["target_y"]))
        # .get(..., 0): callers that build this dict by hand without these
        # keys (e.g. train.py's existing PPO loop) must not KeyError here.
        activate_ability_slot1 = bool(_to_scalar(action.get("activate_ability_slot1", 0)))
        activate_ability_slot2 = bool(_to_scalar(action.get("activate_ability_slot2", 0)))

        if self.teacher is None:
            step_result = self.game.step(card_idx, target_x, target_y, skip_frames,
                                          activate_ability_slot1, activate_ability_slot2)
            obs = np.array(step_result.observation, dtype=np.float32)
            reward = float(step_result.reward)
            terminated = bool(step_result.done)
        else:
            # Team 1's move comes from ITS OWN mirrored observation and is
            # returned in ITS OWN frame; stepSelfPlay mirrors the y back itself
            # (realY1 = BOARD_HEIGHT - 1 - y1). Nothing here converts frames --
            # doing so would double-mirror and put every opponent placement in
            # its own back corner.
            obs1 = np.asarray(self.game.get_observation_for_team(1), dtype=np.float32)
            slot1, x1, y1 = self.teacher.act(self.game, obs1)
            opp_ab1, opp_ab2 = self.teacher.ability_flags(self.game, obs1)
            step_result = self.game.step_self_play(
                card_idx, target_x, target_y, slot1, x1, y1, skip_frames,
                activate_ability_slot1, activate_ability_slot2, opp_ab1, opp_ab2)
            obs = np.array(step_result.observation0, dtype=np.float32)
            reward = float(step_result.reward0)
            terminated = bool(step_result.done)
        truncated = False

        # A scenario window expiring is NOT the world ending. `base_trainer`
        # bootstraps V(final_obs) on `truncated` and 0.0 on `terminated`,
        # reading the FLAG and never the reward's magnitude -- so reporting
        # this as terminated would teach the critic that successfully holding
        # a defence is worth zero, which is the exact value the scenario
        # exists to teach.
        if self.scenario_max_steps is not None and not terminated:
            self.scenario_steps_taken += 1
            if self.scenario_steps_taken >= self.scenario_max_steps:
                truncated = True
        
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
            # Same key and encoding as selfplay_env's, so ONE trainer-side
            # rule reads both pipelines. A scenario episode must stay out
            # of the curriculum's win-rate window: it is a 15-25 step
            # window that rarely ends in a crown, so recording it would
            # cap the achievable win rate near 1 - injection_prob against
            # a 0.80 gate and freeze the curriculum permanently.
            "is_scenario": 1.0 if self.last_scenario is not None else 0.0,
            "scenario_defensive": 1.0 if self.scenario_defensive else 0.0,
            "team0_building_damage": self.game.get_building_damage_dealt(0),
            # Towers only. compute_shaping() needs tower damage and
            # deployed-building damage priced differently -- see
            # train.tower_potential.
            # --- inputs for the lethal-spell PBRS term (train.lethal_spell_potential)
            # Enemy tower HP in ABSOLUTE points. The observation carries these
            # normalized in its appended scalar tail (indices 6-8 = enemy
            # king/left/right), so this is a re-scale of data the net already
            # sees rather than a new engine call.
            # Inputs to both spell terms (rewards.shaping.spell_value_shaping and
            # lethal_spell_potential), for THIS deck's damage spell -- see
            # deck_spell_info.
            **deck_spell_info(self.game, self._damage_spell),
            "enemy_tower_hp": np.asarray(obs[clash_royale_env.ClashRoyaleEnv.EXTRA_SCALARS_START + 6:clash_royale_env.ClashRoyaleEnv.EXTRA_SCALARS_START + 9], dtype=np.float32) * clash_royale_env.ClashRoyaleEnv.MAX_BUILDING_HP,
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
            # SUPERVISION TARGET for the auxiliary head, and nothing else.
            # Deliberately delivered through info -- NOT through the
            # observation -- because WHICH CARD the opponent played during
            # this step is, at the moment the agent chose its action, still in
            # the future. Putting it in the observation would be handing the
            # policy the answer it is being asked to predict.
            #
            # -1 means "the opponent played nothing during this step". The
            # trainer turns this per-step stream into a NEXT-card label by
            # scanning forward within the episode; a step with no future play
            # gets no label and is masked out of the loss entirely.
            #
            # Detected by watching get_last_played_tick per deck card rather
            # than by diffing elixir: elixir cannot identify WHICH card, and a
            # 10-tick step can contain more than one play. The EARLIEST new
            # play in the window is the right answer, since the label is "the
            # next card they play". See MicroRoyaleNet.predict_opp_next_card.
            "opp_played_card": self._poll_opponent_play(),
            # Kept for diagnostics only -- no head consumes it since the
            # 2026-08-28 aux swap. It is still the cheapest sanity read on
            # whether the opponent is spending at all.
            "opp_elixir": self.game.get_elixir_for_team(1),
            # Not part of observation_space -- see ClashEnv.h's own comment
            # on why champion-ability state stays out of the flat
            # observation vector (would break model.py's fixed scalar_size
            # formula) rather than growing it. Two independent slots -- see
            # activate_ability_slot1/2's own comment above.
            "champion_ability_slot1_ready": self.game.is_champion_ability_ready(0, 1),
            "champion_ability_slot2_ready": self.game.is_champion_ability_ready(0, 2),
        }

        # One finished MATCH updates this worker's per-deck estimate. Gated on
        # `terminated` and not `truncated`: a truncation is a scenario window or
        # a step cap, which is not a match result and would enter a phantom loss
        # for whichever deck happened to be up. Same rule train.py applies to
        # the curriculum's own window, for the same reason.
        if terminated and self.deck_pool is not None:
            self._record_deck_outcome(
                self.game.get_towers_alive(0) > self.game.get_towers_alive(1))

        return obs, reward, terminated, truncated, info

    def _deck_pool_weights(self):
        """PFSP weights over the pool from this worker's own win-rate estimates."""
        return deck_pool.pfsp_weights(self.deck_pool_stats)

    def _record_deck_outcome(self, won):
        """Fold one finished match into this worker's local estimate.

        An EWMA rather than a window: a window would need per-deck deques in
        every worker and 16 decks x 8 workers of them, for an estimate that only
        has to be roughly right -- PFSP weights are a sampling prior, not a
        gate.

        THE RATE IS COUNT-WEIGHTED EARLY: `alpha = max(0.05, 1/(n+1))` is the
        running mean for the first few matches, decaying into the 0.05 EWMA. A
        deck's estimate therefore reflects THIS policy within ~10 matches
        instead of ~60.

        That is an ESTIMATOR argument and deliberately not a win-rate claim. The
        shipped priors come from a TRAINED policy, so for a fresh net their
        ordering is right and their level is far too high, and a prior taken
        from a different policy must not outlive contact with the current one --
        which is exactly what `prior_win_rate`'s own contract promises. Measured
        over 8 simulated seeds, the downstream effect on which decks actually
        get sampled is INSIDE THE NOISE against a flat 0.05 (6.8-17.8% of
        episodes on unwinnable decks either way), so do not cite this as a fix
        for the cold start -- see CLAUDE.md, which records that a fresh policy
        won 0 of its first 50-70 episodes against the pool with and without it.
        """
        name = self.current_deck_name
        if name is None or name not in self.deck_pool_stats:
            return
        n = self._deck_pool_counts.get(name, 0) + 1
        self._deck_pool_counts[name] = n
        alpha = max(0.05, 1.0 / (n + 1))
        prev = self.deck_pool_stats[name]
        self.deck_pool_stats[name] = (1 - alpha) * prev + alpha * (1.0 if won else 0.0)

    def set_deck_pool_stats(self, stats, counts=None):
        """Seed this worker's estimates from a resumed checkpoint.

        WHY THIS EXISTS. `deck_pool_stats` was initialised from
        `prior_win_rate` on every env construction and lived only in worker
        memory, so EVERY RESTART discarded what the run had learned about the
        pool. Measured on the 2026-09-04 run after 12,589 episodes: the agent
        had learned xbow_30_cycle 0.181 and mortar_cycle 0.176, against JSON
        priors of 0.867 and 0.800 -- and PFSP weights by `(1 - rate)^2`, so
        those two decks' sampling weight was wrong by a factor of ~37 on resume.
        The 100-episode win rate fell 0.42 -> 0.143 and took ~500 episodes to
        recover, every restart, invisibly.

        COUNTS ARE NOT OPTIONAL. `alpha = max(0.05, 1/(n+1))`, so restoring the
        rates while leaving counts at zero gives the very next match alpha = 1.0
        and overwrites the restored estimate with a single game's outcome -- a
        restore that erases itself on contact, which would look exactly like the
        bug it was meant to fix.
        """
        if not self.deck_pool_stats:
            return          # pool disabled for this worker; nothing to seed
        for name, rate in (stats or {}).items():
            if name in self.deck_pool_stats:
                self.deck_pool_stats[name] = float(rate)
        for name, n in (counts or {}).items():
            if name in self._deck_pool_counts:
                self._deck_pool_counts[name] = int(n)

    def get_deck_pool_counts(self):
        """Matches played per deck by this worker -- the other half of a resume."""
        return dict(self._deck_pool_counts)

    def get_deck_pool_stats(self):
        """This worker's per-deck estimates, for the trainer's read-out.

        Phase 1 had NO per-deck diagnostic at all, which is half of why the
        2026-08-28 deck collapse ran for 30,000 episodes unseen. `envs.call`
        returns one dict per worker and the trainer averages them.
        """
        return dict(self.deck_pool_stats)

    def set_opponent_deck(self, deck):
        # Also updates self.opp_deck (not just the live game instance) --
        # otherwise the very next auto-reset (reset() always re-applies
        # self.opp_deck when randomize_opp_deck is False) would silently
        # revert this to whatever deck the env was constructed with.
        self.opp_deck = list(deck)
        self.game.set_opponent_deck(self.opp_deck)
        # The teacher PLAYS this deck -- its role table (which card is the win
        # condition) and its cycle tracker are per-deck, so a deck change that
        # skipped this would leave it reasoning about the previous one.
        if self.teacher is not None:
            self.teacher.set_deck(self.opp_deck)

    def set_teacher_stage(self, stage):
        """Advance the teacher one rung of the competence ladder.

        No-op when the opponent is the C++ heuristic, so `envs.call(...)` from
        the trainer is safe regardless of how the env was configured."""
        if self.teacher is not None:
            self.teacher.set_stage(int(stage))

    def set_opponent_elixir_multiplier(self, multiplier):
        self.game.set_opponent_elixir_multiplier(multiplier)