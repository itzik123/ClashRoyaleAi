import random

import gymnasium as gym
from gymnasium import spaces
import numpy as np

import clash_royale_env
from python_ai.advisors import card_probes
from python_ai.envs import scenario_offense, scenarios
from python_ai.opponents import deck_pool

# The trainee deck lives in `python_ai.deck` (set CLASH_DECK); re-exported here
# because many call sites import it from this module.
from python_ai.deck import DEFAULT_DECK  # noqa: E402


def _find_win_condition(deck):
    """The deck's win condition, or None, from the one resolver the teacher also
    uses.

    `teacher.resolve_win_condition` ranks eligible cards (building-targeter,
    deploy-anywhere, siege building, spawning spell) by measured tower damage
    from where each is actually played, so the reward and the teacher name the
    same card. None is a legitimate deck: the win-condition term is then zero,
    and `validate_deck` says so at startup.
    """
    from python_ai.opponents import teacher as _teacher
    return next((c for c, r in _teacher.card_roles(list(deck)).items()
                 if r == "wincon"), None)


WIN_CONDITION_ID = _find_win_condition(DEFAULT_DECK)


def deck_spell_info(game, spell):
    """The five `spell_*` info keys for team 0's damage spell, or all zeros.

    `spell` is `card_probes.damage_spell(deck)`: (card_id, tower_damage, cost)
    or None. Both envs use this one builder, so neither pipeline can silently
    lack a key.
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


# Champion ability slots in the deck (0, 1 or 2), derived from the engine.
# MicroRoyaleNet builds ability heads only when this is > 0.
DEFAULT_DECK_ABILITY_SLOTS = sum(
    1 for _cid in DEFAULT_DECK
    if clash_royale_env.get_card_info(_cid)["is_champion"]
    or clash_royale_env.get_card_info(_cid)["is_hero"]
)

# Per-card spell flags, from the engine.
SPELL_BY_CARD_ID = {
    _cid: clash_royale_env.get_card_info(_cid)["is_spell"]
    for _cid in clash_royale_env.get_all_card_ids()
}


def get_all_card_ids():
    """All registered card ids, from the engine (a hardcoded range once silently
    stopped covering new cards).
    """
    return clash_royale_env.get_all_card_ids()


def _to_scalar(val):
    """Unwrap an action field into a plain Python scalar.

    Callers pass 0-d/1-element numpy arrays (gym's vector space),
    `np.array([i])` (the PPO loop) or bare ints (probes). train_selfplay
    imports it.
    """
    if hasattr(val, "item"):
        return val.item()
    if isinstance(val, (list, tuple, np.ndarray)):
        return val[0]
    return val


class MicroRoyaleEnv(gym.Env):
    def __init__(self, env_config=None):
        super().__init__()
        
        if env_config is None:
            env_config = {}
            
        ai_deck = env_config.get("ai_deck", list(DEFAULT_DECK))
        # By default the opponent plays the same deck (a mirror). Uniformly
        # random registry decks crush the agent (giant units beat a cycle deck
        # in this engine), so the win/loss signal disappears; the phase-1
        # trainer uses the meta-deck pool (`deck_pool`) instead.
        self.ai_deck = list(ai_deck)
        # The deck's damage spell, which the spell reward terms follow; see
        # deck_spell_info.
        self._damage_spell = card_probes.damage_spell(self.ai_deck)
        self.opp_deck = env_config.get("opp_deck", list(ai_deck))
        self.current_opp_deck = list(self.opp_deck)
        self.randomize_opp_deck = env_config.get("randomize_opp_deck", False)

        # The meta-deck pool: None (off), True (every enabled deck) or a list
        # of deck names. Takes precedence over randomize_opp_deck, which draws
        # uniformly from the registry, a different distribution (see
        # opponents/deck_pool.py).
        pool_cfg = env_config.get("deck_pool", None)
        self.deck_pool = None
        self.current_deck_name = None
        #: Per-worker local win-rate estimate, {deck name: rate}, like
        #: selfplay_env's `pfsp_stats`: each worker converges its own estimate.
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
            # Seeded from the pool file's measured priors.
            self.deck_pool_stats = {d.name: d.prior_win_rate for d in all_decks}
            #: Matches this worker has played per deck, for the count-weighted
            #: update.
            self._deck_pool_counts = {d.name: 0 for d in all_decks}
            # `is None`, not `or`: worker 0's seed is 0.
            seed_cfg = env_config.get("scenario_seed")
            self._deck_rng = random.Random(
                random.randrange(1 << 30) if seed_cfg is None else seed_cfg)
        max_ticks = env_config.get("max_ticks", 3600)
        # The opponent's elixir-rate multiplier (1.0 = normal). Measurement
        # harnesses only; the curriculum does not use it.
        opp_elixir_multiplier = env_config.get("opp_elixir_multiplier", 1.0)
        # Tower Troops are per-match config like the deck. NONE is the plain
        # Princess Tower.
        ai_tower_troop = env_config.get("ai_tower_troop", clash_royale_env.TowerTroopType.NONE)
        opp_tower_troop = env_config.get("opp_tower_troop", clash_royale_env.TowerTroopType.NONE)

        self.game = clash_royale_env.ClashRoyaleEnv(ai_deck, self.opp_deck, max_ticks, ai_tower_troop, opp_tower_troop)
        self.game.set_opponent_elixir_multiplier(opp_elixir_multiplier)

        # Who plays team 1. "builtin" (default): game.step() runs the C++
        # HeuristicOpponent. "teacher": stepSelfPlay, which never calls
        # opponentTurn(), with the UtilityTeacher choosing team 1's moves. The
        # reward stream is the same either way.
        self.opponent_kind = env_config.get("opponent", "builtin")
        self.teacher = None
        # Offensive scenario injection, default off; see scenario_offense.py.
        self.offensive_scenario_prob = float(
            env_config.get("offensive_scenario_prob",
                           scenario_offense.OFFENSIVE_SCENARIO_PROB))
        self._scenario_rng = np.random.default_rng(
            env_config.get("scenario_seed", None))
        self.last_scenario = None

        # Defensive scenario injection (a manufactured "defend or lose the
        # tower" start), as pipeline 2 does. Default 0.0; pass
        # `envs/scenarios.SCENARIO_INJECTION_PROB` to enable.
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
            # Index HAND_SIZE is the no-op (bank elixir); the engine ignores a
            # cardIndex outside [0, HAND_SIZE). Bounds come from the engine.
            "card_index": spaces.Discrete(clash_royale_env.ClashRoyaleEnv.HAND_SIZE + 1),
            "target_x": spaces.Box(low=0.0, high=self.game.get_max_placement_x(), shape=(1,), dtype=np.float32),
            # Full board height: spells may cross the river. Per-card legality
            # is MicroRoyaleNet.placement_mask.
            "target_y": spaces.Box(low=0.0, high=float(clash_royale_env.ClashRoyaleEnv.BOARD_HEIGHT - 1),
                                   shape=(1,), dtype=np.float32),
            # Champion ability activation for deck slots 1 and 2 (a deck can
            # hold two Champions). 1 activates if the Champion is deployed, off
            # cooldown and affordable; otherwise a silent no-op. Missing keys
            # default to 0.
            "activate_ability_slot1": spaces.Discrete(2),
            "activate_ability_slot2": spaces.Discrete(2),
        })
        
        obs_size = self.game.observation_size()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # super().reset() seeds only the wrapper's np_random, which nothing
        # reads; ClashEnv::seed seeds both engine generators (heuristic
        # opponent, opening hand) and resets. Only when a seed is given: later
        # resets continue the stream. sample_random_deck() is not seedable, so
        # randomize_opp_deck runs are not reproducible (UPSTREAM_REQUESTS.md
        # item 23C).
        if seed is not None:
            self.game.seed(int(seed))
        if self.deck_pool is not None:
            # Sampled per worker, per episode: workers are separate processes,
            # and per-episode sampling keeps each PPO batch a mix of matchups.
            picked = deck_pool.sample_deck(
                self.deck_pool, self._deck_pool_weights(), self._deck_rng)
            self.current_deck_name = picked.name
            random_deck = list(picked.card_ids)
            self.game.set_opponent_deck(random_deck)
            # The deck actually faced this episode; self.opp_deck stays the
            # fixed fallback.
            self.current_opp_deck = random_deck
            # The teacher plays this deck too.
            if self.teacher is not None:
                self.teacher.set_deck(random_deck)
        elif self.randomize_opp_deck:
            # The engine's sampler respects Champion/Evolution slot rules,
            # which a naive random.sample would violate.
            random_deck = list(clash_royale_env.sample_random_deck())
            self.game.set_opponent_deck(random_deck)
            # The deck actually faced this episode; self.opp_deck stays the
            # fixed fallback.
            self.current_opp_deck = random_deck
            # The teacher plays this deck too.
            if self.teacher is not None:
                self.teacher.set_deck(random_deck)
        else:
            self.game.set_opponent_deck(self.opp_deck)
            self.current_opp_deck = list(self.opp_deck)

        obs_list = self.game.reset()
        # New match: reset the played-tick baseline, or step 1 reports a
        # phantom play.
        self._opp_last_tick = {}
        # Scenario injection rewrites the reset state, so the observation is
        # re-read afterwards.
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
        # New match, new profile and lane bias for the teacher, so it cannot be
        # memorised in one counter-line.
        if self.teacher is not None:
            self.teacher.reset()
        obs = np.array(obs_list, dtype=np.float32)
        return obs, {}

    def _apply_defensive_scenario(self):
        """Rewrite the freshly reset state into a defensive emergency.

        Mirrors `MicroRoyaleSelfPlayEnv.reset`, so both pipelines see the same
        threats. Returns the re-read observation.
        """
        scenario = scenarios.sample_scenario(self._scenario_rng)

        warmup = scenario.get("warmup_ticks", 0)
        noop = clash_royale_env.ClashRoyaleEnv.HAND_SIZE
        if warmup:
            # Bank elixir before the spawns, so injected units do not walk
            # during the warm-up.
            self.game.step_self_play(noop, 0.0, 0.0, noop, 0.0, 0.0, warmup)

        for card_id, x, y in scenario["spawns"]:
            self.game.inject_enemy(card_id, x, y)

        # inject_enemy only queues units; one 1-tick step commits them so the
        # threat is in the first observation. step_self_play so the teacher
        # gets no free decision.
        self.game.step_self_play(noop, 0.0, 0.0, noop, 0.0, 0.0, 1)

        self.last_scenario = scenario["name"]
        self.scenario_max_steps = scenario["max_steps"]
        self.scenario_defensive = bool(scenario.get("defensive", False))
        return self.game.get_observation_for_team(0)

    def _poll_opponent_play(self):
        """Which card team 1 played since the last poll, or -1 for none.

        Diffs `get_last_played_tick` for each card in the opponent's current
        deck. The first poll of an episode only sets the baseline and returns
        -1. With several plays in one step, the earliest is returned: the label
        is the next card they play.
        """
        last = getattr(self, "_opp_last_tick", None)
        if last is None:
            last = self._opp_last_tick = {}
        # current_opp_deck, not the fixed fallback opp_deck, or every play
        # reads as -1.
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
        # Hand-built action dicts may omit these keys.
        activate_ability_slot1 = bool(_to_scalar(action.get("activate_ability_slot1", 0)))
        activate_ability_slot2 = bool(_to_scalar(action.get("activate_ability_slot2", 0)))

        if self.teacher is None:
            step_result = self.game.step(card_idx, target_x, target_y, skip_frames,
                                          activate_ability_slot1, activate_ability_slot2)
            obs = np.array(step_result.observation, dtype=np.float32)
            reward = float(step_result.reward)
            terminated = bool(step_result.done)
        else:
            # Team 1's move comes from its own mirrored observation, in its own
            # frame; stepSelfPlay mirrors it back. Converting here would
            # double-mirror.
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

        # A scenario window expiring is a truncation, not the end of the world:
        # the trainer bootstraps V(final_obs) on `truncated` and 0 on
        # `terminated`.
        if self.scenario_max_steps is not None and not terminated:
            self.scenario_steps_taken += 1
            if self.scenario_steps_taken >= self.scenario_max_steps:
                truncated = True
        
        info = {
            "elixir": self.game.get_elixir(),
            "hand": self.game.get_hand(),
            # Cumulative per-match damage dealt by each team, from
            # MatchStatistics; compute_shaping diffs them.
            "team0_troop_damage": self.game.get_troop_damage_dealt(0),
            "team1_troop_damage": self.game.get_troop_damage_dealt(1),
            # Scenario episodes stay out of the curriculum's win-rate window:
            # they rarely end in a crown, and would cap the achievable win
            # rate.
            "is_scenario": 1.0 if self.last_scenario is not None else 0.0,
            "scenario_defensive": 1.0 if self.scenario_defensive else 0.0,
            "team0_building_damage": self.game.get_building_damage_dealt(0),
            # Towers only (tower_damage) vs towers plus deployed buildings
            # (building_damage): the shaping prices them differently. Then the
            # deck spell's inputs, and enemy tower HP in absolute points (a
            # rescale of the observation's scalars) for the lethal-spell term.
            **deck_spell_info(self.game, self._damage_spell),
            "enemy_tower_hp": np.asarray(obs[clash_royale_env.ClashRoyaleEnv.EXTRA_SCALARS_START + 6:clash_royale_env.ClashRoyaleEnv.EXTRA_SCALARS_START + 9], dtype=np.float32) * clash_royale_env.ClashRoyaleEnv.MAX_BUILDING_HP,
            "team0_tower_damage": self.game.get_tower_damage_dealt(0),
            "team1_tower_damage": self.game.get_tower_damage_dealt(1),
            "team1_building_damage": self.game.get_building_damage_dealt(1),
            # Cumulative elixir spent (sum of card costs), for the elixir-trade
            # term.
            "team0_elixir_spent": self.game.get_elixir_spent(0),
            "team1_elixir_spent": self.game.get_elixir_spent(1),
            # Surviving tower counts, for the crown term.
            "team0_towers_alive": self.game.get_towers_alive(0),
            "team1_towers_alive": self.game.get_towers_alive(1),
            # Cumulative damage by the win-condition card (to anything; for a
            # building-targeter, towers plus any building in the way).
            "team0_wincon_damage": (self.game.get_damage_dealt_by_card(WIN_CONDITION_ID, 0)
                                    if WIN_CONDITION_ID is not None else 0),
            # Supervision for the next-card auxiliary head, deliberately in
            # info and not the observation: when the agent acted, this play was
            # still in the future. -1 means nothing was played this step.
            "opp_played_card": self._poll_opponent_play(),
            # Diagnostics only.
            "opp_elixir": self.game.get_elixir_for_team(1),
            # Champion readiness stays out of the observation vector, whose
            # layout is fixed; the trainer masks the ability heads with it.
            "champion_ability_slot1_ready": self.game.is_champion_ability_ready(0, 1),
            "champion_ability_slot2_ready": self.game.is_champion_ability_ready(0, 2),
        }

        # A finished match updates this worker's deck estimate. Not on
        # truncation, which is not a result.
        if terminated and self.deck_pool is not None:
            self._record_deck_outcome(
                self.game.get_towers_alive(0) > self.game.get_towers_alive(1))

        return obs, reward, terminated, truncated, info

    def _deck_pool_weights(self):
        """PFSP weights over the pool from this worker's own win-rate estimates."""
        return deck_pool.pfsp_weights(self.deck_pool_stats)

    def _record_deck_outcome(self, won):
        """Fold one finished match into this worker's local estimate.

        An EWMA with a count-weighted start: alpha = max(0.05, 1/(n+1)) is the
        running mean for the first matches, so a prior taken from a different
        policy stops mattering within ~10 matches. An estimator argument, not a
        win-rate claim.
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
        """Restore this worker's estimates from a resumed checkpoint.

        Otherwise every restart resets them to the priors, and PFSP samples the
        pool wrongly until they re-converge. Counts must be restored too: with
        n = 0, alpha = 1 and the next match overwrites the restored rate.
        """
        if not self.deck_pool_stats:
            return          # pool disabled for this worker
        for name, rate in (stats or {}).items():
            if name in self.deck_pool_stats:
                self.deck_pool_stats[name] = float(rate)
        for name, n in (counts or {}).items():
            if name in self._deck_pool_counts:
                self._deck_pool_counts[name] = int(n)

    def get_deck_pool_counts(self):
        """Matches played per deck by this worker, the other half of a resume.
        """
        return dict(self._deck_pool_counts)

    def get_deck_pool_stats(self):
        """This worker's per-deck estimates, for the trainer's read-out (which
        averages the workers).
        """
        return dict(self.deck_pool_stats)

    def set_opponent_deck(self, deck):
        # Update opp_deck too, or the next auto-reset reverts to the
        # constructor's deck.
        self.opp_deck = list(deck)
        self.game.set_opponent_deck(self.opp_deck)
        # The teacher's role table and cycle tracker are per-deck.
        if self.teacher is not None:
            self.teacher.set_deck(self.opp_deck)

    def set_teacher_stage(self, stage):
        """Set the teacher's rung. A no-op against the C++ heuristic, so
        `envs.call(...)` is always safe.
        """
        if self.teacher is not None:
            self.teacher.set_stage(int(stage))

    def set_opponent_elixir_multiplier(self, multiplier):
        self.game.set_opponent_elixir_multiplier(multiplier)