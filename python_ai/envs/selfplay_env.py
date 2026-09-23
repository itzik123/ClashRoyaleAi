"""Pipeline 2's environment: team 1 is another policy, not the C++ bot.

Same action/observation shape as `envs/gym_wrapper.MicroRoyaleEnv`. Team 1 is a
frozen `MicroRoyaleNet` snapshot, one of four scripted heuristics, or (through
`game.step()`) the C++ HeuristicOpponent.

Opponents are drawn by PFSP (prioritized fictitious self-play), not a ladder:
every reset samples from the whole pool, weighted toward the opponents this
worker does worst against, so mastered opponents still recur and nothing is
forgotten. A ladder stalls once the next opponent is a near-mirror (sitting at
~50% by construction) and never revisits. Each worker keeps its own estimates;
the main process broadcasts pool updates via `refresh_pfsp_pool`, and
`set_historical_opponent` picks a specific opponent for evaluation or replays.
"""

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch.distributions import Categorical

import clash_royale_env
from python_ai.envs import scenarios, scripted_opponents
from python_ai.advisors import card_probes
from python_ai.envs.gym_wrapper import (
    DEFAULT_DECK, DEFAULT_DECK_ABILITY_SLOTS, WIN_CONDITION_ID, _to_scalar,
    deck_spell_info,
)
from python_ai.models.net import MicroRoyaleNet
from python_ai.models.policy_io import LSTM_HIDDEN, load_state_dict_flexible

PFSP_EXPONENT = 2.0

# Sampling floor, so every opponent stays in rotation forever.
PFSP_MIN_WEIGHT = 0.05

# EMA factor for each worker's per-opponent win rate (win 1, draw 0.5, loss 0):
# ~12-game window.
PFSP_EMA_ALPHA = 0.08

# The C++ heuristic as a training opponent, not only an evaluation anchor:
# stepSelfPlay never runs it, so otherwise phase 2 never trains against what it
# is measured on. Only the multipliers with headroom (the agent already wins
# ~100% at 1.00x).
BUILTIN_TRAINING_OPPONENTS = ["builtin:heuristic@1.35", "builtin:heuristic@1.50"]

# Floor above PFSP's own, which would taper these off as the agent improves;
# they are the measurement target. Sized to stay a minority of episodes (~29%
# at a pool of ~20, shrinking as the pool grows).
BUILTIN_MIN_WEIGHT = 0.5

class MicroRoyaleSelfPlayEnv(gym.Env):
    """Team 1 is a frozen MicroRoyaleNet (inference only), a scripted bot or the
    C++ heuristic, chosen by PFSP at every reset; see the module docstring.
    """

    def __init__(self, env_config=None):
        super().__init__()
        env_config = env_config or {}
        self.deck = env_config.get("deck", list(DEFAULT_DECK))
        # The card the spell reward terms follow; see
        # gym_wrapper.deck_spell_info.
        self._damage_spell = card_probes.damage_spell(self.deck)
        max_ticks = env_config.get("max_ticks", 3600)
        # Tower Troops, as in gym_wrapper.
        ai_tower_troop = env_config.get("ai_tower_troop", clash_royale_env.TowerTroopType.NONE)
        opp_tower_troop = env_config.get("opp_tower_troop", clash_royale_env.TowerTroopType.NONE)
        self.game = clash_royale_env.ClashRoyaleEnv(self.deck, self.deck, max_ticks, ai_tower_troop, opp_tower_troop)
        # The engine's placement bounds.
        self.MAX_X = self.game.get_max_placement_x()
        self.MAX_Y = self.game.get_own_half_max_y()

        # Team 1's network runs on CPU: one inference pass per step per worker.
        self.device = torch.device("cpu")
        self.opponent_net = MicroRoyaleNet(num_ability_slots=DEFAULT_DECK_ABILITY_SLOTS).to(self.device)
        self.opponent_net.eval()
        self.opponent_hx = torch.zeros(1, LSTM_HIDDEN).to(self.device)
        self.opponent_cx = torch.zeros(1, LSTM_HIDDEN).to(self.device)
        self.opponent_checkpoint_path = None
        # "neural", "builtin", or a scripted bot's name ("Rusher", "Defender",
        # "Cycler", "Counter").
        self.opponent_kind = "neural"
        # Rusher/Counter commit to one lane per episode.
        self.opponent_lane = None

        # Filled by the main process's broadcast before the first reset.
        self.pfsp_pool = []
        self.pfsp_stats = {}
        #: Games this worker has scored per opponent. Weights the cross-worker
        #: merge at checkpoint time, so a worker still holding the 0.5 prior
        #: does not dilute measured estimates.
        self.pfsp_counts = {}

        # Scenario injection; scenarios_enabled=False (the replay env) keeps
        # replays to full games.
        self.scenarios_enabled = env_config.get("scenarios_enabled", True)
        #: The env's only stochastic source: scenario injection, the attacking
        #: lane and the PFSP draw all read it. `scenario_seed` is the same key
        #: pipeline 1 takes; None means OS entropy. rl/seeding.py derives
        #: per-worker seeds.
        self.rng = np.random.default_rng(env_config.get("scenario_seed", None))
        #: Alias used by the scenario code; must be the same generator.
        self.scenario_rng = self.rng
        self.scenario_active = None       # name of the current episode's scenario, or None
        self.scenario_defensive = False   # is ScenDef a meaningful test for it?
        self.scenario_max_steps = None    # truncation window in bot-steps, or None
        self.scenario_steps_taken = 0

        if env_config.get("historical_checkpoint_path"):
            self.set_historical_opponent(env_config["historical_checkpoint_path"])

        self.action_space = spaces.Dict({
            "card_index": spaces.Discrete(clash_royale_env.ClashRoyaleEnv.HAND_SIZE + 1),
            "target_x": spaces.Box(low=0.0, high=self.MAX_X, shape=(1,), dtype=np.float32),
            # Full board height, as in gym_wrapper. self.MAX_Y is still used by
            # the scripted opponents, which are own-half only.
            "target_y": spaces.Box(low=0.0, high=float(clash_royale_env.ClashRoyaleEnv.BOARD_HEIGHT - 1),
                                   shape=(1,), dtype=np.float32),
            # As in gym_wrapper. Team 1 samples its own ability decisions in
            # _opponent_action.
            "activate_ability_slot1": spaces.Discrete(2),
            "activate_ability_slot2": spaces.Discrete(2),
        })
        obs_size = self.game.observation_size()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)

    def set_historical_opponent(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        load_state_dict_flexible(self.opponent_net, state_dict, f"historical opponent {checkpoint_path}")
        self.opponent_net.eval()
        self.opponent_checkpoint_path = checkpoint_path
        self.opponent_kind = "neural"
        # Undo a previous scripted opponent's random deck, which
        # set_opponent_deck does not reset on its own.
        self.game.set_opponent_deck(self.deck)
        self._reset_opponent_elixir()

    def set_scripted_opponent(self, name):
        """Team 1 becomes a scripted heuristic bot, with a random deck: the
        heuristics read only elixir, costs and positions, so they are
        deck-agnostic.
        """
        self.opponent_kind = name
        self.opponent_checkpoint_path = f"scripted:{name}"
        # The engine's sampler respects deck-slot rules.
        self.game.set_opponent_deck(clash_royale_env.sample_random_deck())
        self._reset_opponent_elixir()
        if name in ("Rusher", "Counter"):
            self.opponent_lane = ("left" if self.rng.random() < 0.5 else "right")

    def _reset_opponent_elixir(self):
        """Reset the opponent's elixir multiplier, which does not reset itself: a
        1.5x heuristic episode would otherwise hand the next opponent 50% extra
        elixir.
        """
        self.game.set_opponent_elixir_multiplier(1.0)

    def set_builtin_opponent(self, descriptor):
        """Team 1 becomes the C++ HeuristicOpponent at a given elixir multiplier;
        see BUILTIN_TRAINING_OPPONENTS.

        It runs inside game.step(), which step() dispatches to for this kind.
        Both sides play self.deck, matching how evaluate_against_roster builds
        these anchors.
        """
        self.opponent_kind = "builtin"
        self.opponent_checkpoint_path = descriptor
        self.game.set_opponent_deck(self.deck)
        self.game.set_opponent_elixir_multiplier(float(descriptor.split("@")[1]))

    def _set_opponent(self, descriptor):
        """Dispatch a sampled descriptor: a checkpoint path, "scripted:<name>", or
        "builtin:heuristic@<mult>" (the spelling evaluate_against_roster uses;
        never a file path).
        """
        if descriptor.startswith("scripted:"):
            self.set_scripted_opponent(descriptor[len("scripted:"):])
        elif descriptor.startswith("builtin:"):
            self.set_builtin_opponent(descriptor)
        else:
            self.set_historical_opponent(descriptor)

    def refresh_pfsp_pool(self, pool_paths):
        """Replace the pool (broadcast by the main process at startup and after
        each new snapshot). New entries start at a 0.5 prior; the pool only
        grows.
        """
        self.pfsp_pool = list(pool_paths)
        for p in self.pfsp_pool:
            if p not in self.pfsp_stats:
                self.pfsp_stats[p] = 0.5

    def get_pfsp_stats(self):
        """(estimates, counts) this worker holds; the trainer merges them."""
        return dict(self.pfsp_stats), dict(self.pfsp_counts)

    def set_pfsp_stats(self, stats, counts=None):
        """Seed this worker from a resumed checkpoint's pooled estimates.

        Otherwise every resume resets every opponent to 0.5 and PFSP
        mis-samples until each worker re-meets each opponent ~12 times. Keys
        need not be in the pool yet: `refresh_pfsp_pool` only fills missing
        entries.
        """
        for key, rate in (stats or {}).items():
            self.pfsp_stats[key] = float(rate)
        for key, n in (counts or {}).items():
            self.pfsp_counts[key] = int(n)

    def _sample_pfsp_opponent(self):
        """Draw this episode's opponent, dropping any entry that will not load.

        A truncated checkpoint in the pool would otherwise crash phase 2 at a
        random reset hours in. Skip it and say so, but raise if the whole pool
        fails: that is an architecture mismatch, and continuing would silently
        reduce the league to its scripted bots.
        """
        if not self.pfsp_pool:
            return

        def floor_for(p):
            if p in scripted_opponents.DEFENSIVE_SCRIPTED_OPPONENTS:
                return scripted_opponents.DEFENSIVE_SCRIPTED_MIN_WEIGHT
            if p.startswith("builtin:"):
                return BUILTIN_MIN_WEIGHT
            return PFSP_MIN_WEIGHT

        while self.pfsp_pool:
            weights = np.array([
                max(floor_for(p),
                    (1.0 - self.pfsp_stats.get(p, 0.5)) ** PFSP_EXPONENT)
                for p in self.pfsp_pool
            ], dtype=np.float64)
            weights /= weights.sum()
            chosen = self.pfsp_pool[self.rng.choice(len(self.pfsp_pool),
                                                    p=weights)]
            try:
                self._set_opponent(chosen)
                return
            except Exception as exc:   # noqa: BLE001 -- any load failure is one story
                print(f"  [WARN] dropping unreadable PFSP pool entry "
                      f"{chosen}: {type(exc).__name__}: {exc}", flush=True)
                self.pfsp_pool.remove(chosen)
                self.pfsp_stats.pop(chosen, None)

        raise RuntimeError(
            "every entry in the PFSP pool failed to load. One bad file is a "
            "truncated checkpoint; ALL of them is an architecture mismatch -- "
            "the pool is not compatible with the current network. Continuing "
            "would silently reduce the league to its scripted bots.")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._sample_pfsp_opponent()
        self.game.reset()
        # New match: reset the played-tick baseline so the first poll re-seeds
        # against this episode's clock.
        self._opp_last_tick = {}

        self.scenario_active = None
        self.scenario_max_steps = None
        self.scenario_steps_taken = 0
        self.scenario_defensive = False
        if self.scenarios_enabled and self.scenario_rng.random() < scenarios.SCENARIO_INJECTION_PROB:
            scenario = scenarios.sample_scenario(self.scenario_rng)

            # A scenario may require a card in the trainee's opening hand. The
            # shuffle cannot be set or read, so re-roll the reset until it
            # appears (a 4-of-8 card takes ~2 tries), capped so a reset can
            # never hang. No current scenario uses it; a new one must use a
            # card the deck can contain.
            want = scenario.get("require_own_card")
            if want is not None:
                for _ in range(24):
                    if want in self.game.get_hand():
                        break
                    self.game.reset()

            # Bank elixir on both sides before the spawns, so injected units do
            # not walk during the warm-up.
            warmup = scenario.get("warmup_ticks", 0)
            if warmup:
                noop_w = clash_royale_env.ClashRoyaleEnv.HAND_SIZE
                self.game.step_self_play(noop_w, 0.0, 0.0, noop_w, 0.0, 0.0, warmup)

            for card_id, x, y in scenario["spawns"]:
                self.game.inject_enemy(card_id, x, y)
            # inject_enemy only queues units; a 1-tick no-op step commits them
            # so the threat is in the first observation.
            noop = clash_royale_env.ClashRoyaleEnv.HAND_SIZE
            self.game.step_self_play(noop, 0.0, 0.0, noop, 0.0, 0.0, 1)
            self.scenario_active = scenario["name"]
            self.scenario_max_steps = scenario["max_steps"]
            # Whether surviving without a big loss is a meaningful success test
            # for this scenario (true only when it puts a threat in our half).
            self.scenario_defensive = bool(scenario.get("defensive", False))

        # Re-read so the observation includes any injected units.
        obs_list = self.game.get_observation_for_team(0)
        self.opponent_hx = torch.zeros(1, LSTM_HIDDEN).to(self.device)
        self.opponent_cx = torch.zeros(1, LSTM_HIDDEN).to(self.device)
        return np.array(obs_list, dtype=np.float32), {}

    def _opponent_action(self):
        """Team 1's decision from its own mirrored observation, sampled like a
        rollout action but under no_grad, plus its own two Champion-ability
        decisions.
        """
        obs1 = np.array(self.game.get_observation_for_team(1), dtype=np.float32)
        if self.opponent_kind != "neural":
            return self._scripted_opponent_action(obs1)
        with torch.no_grad():
            obs1_t = torch.tensor(obs1, dtype=torch.float32).unsqueeze(0).to(self.device)
            # The same affordability mask as the trainee; without it the frozen
            # opponent would waste most turns on refused plays and not
            # reproduce its checkpoint.
            card_mask1 = self.opponent_net.affordability_mask(obs1_t)
            # Thread the hi-res map into placement_given_card so the trunk's
            # first conv runs once per decision.
            (features1, card_embeds1, spatial_map1,
             hires_map1) = self.opponent_net.extract_features_hires(obs1_t)
            (logits1, ab1_1, ab2_1, _,
             (self.opponent_hx, self.opponent_cx)) = self.opponent_net.step_lstm_and_card(
                features1, (self.opponent_hx, self.opponent_cx), card_mask1)
            card_idx1_t = Categorical(logits=logits1).sample()
            place_logits1 = self.opponent_net.placement_given_card(
                self.opponent_hx, card_embeds1, card_idx1_t, obs1_t,
                spatial_map1, hires_map=hires_map1)
            cell1 = Categorical(logits=place_logits1).sample()
            x1_t, y1_t = self.opponent_net.cell_to_xy(cell1)
            card_idx1 = card_idx1_t.item()
            x1 = x1_t.item()
            y1 = y1_t.item()
            # Its own Champion decisions, under its own readiness
            # (rl/abilities.py).
            from python_ai.rl import abilities as ability_mod
            slots = ability_mod.ability_engine_slots(self.deck)
            logits = [l for l in (ab1_1, ab2_1) if l is not None]
            flags = {1: False, 2: False}
            if slots and logits:
                ready = torch.tensor([[bool(self.game.is_champion_ability_ready(1, s))
                                       for s in slots]])
                acts, _, _ = ability_mod.sample(logits, ready)
                for k, slot in enumerate(slots):
                    flags[slot] = bool(acts[0, k])
        return card_idx1, x1, y1, flags[1], flags[2]

    def _scripted_opponent_action(self, obs1):
        """Delegate to the scripted heuristics in `envs/scripted_opponents.py`.
        """
        return scripted_opponents.scripted_action(
            self.opponent_kind, obs1, self.opponent_lane, self.MAX_X, self.MAX_Y)

    def _poll_opponent_play(self):
        """Which card team 1 played since the last poll, or -1 for none.

        Same baseline-and-diff as
        gym_wrapper.MicroRoyaleEnv._poll_opponent_play, over self.deck (both
        sides play it here). The first poll of an episode only seeds the
        baseline.
        """
        last = getattr(self, "_opp_last_tick", None)
        if last is None:
            last = self._opp_last_tick = {}
        best_tick, best_card = None, -1
        for card in set(self.deck):
            tick = self.game.get_last_played_tick(1, card)
            prev = last.get(card)
            last[card] = tick
            if prev is not None and tick > prev:
                if best_tick is None or tick < best_tick:
                    best_tick, best_card = tick, card
        return best_card

    def step(self, action, skip_frames=10):
        card_idx0 = int(_to_scalar(action["card_index"]))
        x0 = float(_to_scalar(action["target_x"]))
        y0 = float(_to_scalar(action["target_y"]))
        activate_ability0_slot1 = bool(_to_scalar(action.get("activate_ability_slot1", 0)))
        activate_ability0_slot2 = bool(_to_scalar(action.get("activate_ability_slot2", 0)))

        if self.opponent_kind == "builtin":
            # The C++ heuristic runs inside game.step()'s opponentTurn(): the
            # phase-1 path.
            result = self.game.step(card_idx0, x0, y0, skip_frames,
                                    activate_ability0_slot1, activate_ability0_slot2)
            obs = np.array(result.observation, dtype=np.float32)
            reward = float(result.reward)
        else:
            card_idx1, x1, y1, activate_ability1_slot1, activate_ability1_slot2 = self._opponent_action()
            result = self.game.step_self_play(card_idx0, x0, y0, card_idx1, x1, y1, skip_frames,
                                               activate_ability0_slot1, activate_ability0_slot2,
                                               activate_ability1_slot1, activate_ability1_slot2)
            obs = np.array(result.observation0, dtype=np.float32)
            reward = float(result.reward0)
        terminated = bool(result.done)

        # A scenario window expiring truncates the episode without terminating
        # it; the trainer bootstraps V(final_obs).
        truncated = False
        if self.scenario_max_steps is not None and not terminated:
            self.scenario_steps_taken += 1
            if self.scenario_steps_taken >= self.scenario_max_steps:
                truncated = True

        # Scenario episodes do not update pfsp_stats: they measure the
        # handicap, not the opponent.
        if terminated and self.opponent_checkpoint_path is not None and self.scenario_active is None:
            if reward > 0.5:
                outcome = 1.0
            elif reward < -0.5:
                outcome = 0.0
            else:
                outcome = 0.5
            prev = self.pfsp_stats.get(self.opponent_checkpoint_path, 0.5)
            self.pfsp_stats[self.opponent_checkpoint_path] = prev * (1.0 - PFSP_EMA_ALPHA) + outcome * PFSP_EMA_ALPHA
            self.pfsp_counts[self.opponent_checkpoint_path] = (
                self.pfsp_counts.get(self.opponent_checkpoint_path, 0) + 1)

        # Same keys as gym_wrapper.MicroRoyaleEnv.step()'s info dict, so
        # compute_shaping is shared.
        info = {
            "elixir": self.game.get_elixir(),
            "hand": self.game.get_hand(),
            "team0_troop_damage": self.game.get_troop_damage_dealt(0),
            "team1_troop_damage": self.game.get_troop_damage_dealt(1),
            "team0_building_damage": self.game.get_building_damage_dealt(0),
            # Towers only vs towers plus deployed buildings, then the deck
            # spell's inputs (the same builder as pipeline 1) and enemy tower
            # HP in absolute points.
            **deck_spell_info(self.game, self._damage_spell),
            "enemy_tower_hp": np.asarray(obs[clash_royale_env.ClashRoyaleEnv.EXTRA_SCALARS_START + 6:clash_royale_env.ClashRoyaleEnv.EXTRA_SCALARS_START + 9], dtype=np.float32) * clash_royale_env.ClashRoyaleEnv.MAX_BUILDING_HP,
            "team0_tower_damage": self.game.get_tower_damage_dealt(0),
            "team1_tower_damage": self.game.get_tower_damage_dealt(1),
            "team1_building_damage": self.game.get_building_damage_dealt(1),
            "team0_elixir_spent": self.game.get_elixir_spent(0),
            "team1_elixir_spent": self.game.get_elixir_spent(1),
            # Surviving tower counts, for the crown term.
            "team0_towers_alive": self.game.get_towers_alive(0),
            "team1_towers_alive": self.game.get_towers_alive(1),
            # Supplied here too, or the win-condition term is silently zero for
            # all of self-play.
            "team0_wincon_damage": (self.game.get_damage_dealt_by_card(WIN_CONDITION_ID, 0)
                                    if WIN_CONDITION_ID is not None else 0),
            # Next-card supervision, via info (it had not happened when the
            # agent acted).
            "opp_played_card": self._poll_opponent_play(),
            # Diagnostics only.
            "opp_elixir": self.game.get_elixir_for_team(1),
            "champion_ability_slot1_ready": self.game.is_champion_ability_ready(0, 1),
            "champion_ability_slot2_ready": self.game.is_champion_ability_ready(0, 2),
            # 1.0 while the episode started from an injected scenario, so the
            # trainer scores scenario defences separately.
            "is_scenario": 1.0 if self.scenario_active is not None else 0.0,
            # Split out because a scenario with no threat in our half
            # (fireball_tower_value spawns at the enemy tower) counts as a
            # "successful defence" whatever the agent does, and would drag
            # ScenDef toward 1.0.
            "scenario_defensive": 1.0 if getattr(self, "scenario_defensive", False) else 0.0,
            # Per-card elixir killed and spent, for the ROI read-out (the half
            # of the strategy metrics that noise pushes down). Filled only on
            # the terminal step, but always present with shape (8,): a key that
            # appears only on some steps makes the vector env emit a companion
            # mask array.
            "ep_killed_by_card": self._episode_economy(0, terminated or truncated),
            "ep_spent_by_card": self._episode_economy(1, terminated or truncated),
        }
        return obs, reward, terminated, truncated, info

    # Indexed by DEFAULT_DECK, which the trainee (team 0) plays.
    def _episode_economy(self, which, ended):
        if not ended:
            return np.zeros(len(DEFAULT_DECK), dtype=np.float32)
        fn = (self.game.get_elixir_value_killed_by if which == 0
              else self.game.get_elixir_spent_on_card)
        return np.asarray([fn(c, 0) for c in DEFAULT_DECK], dtype=np.float32)


def make_env(seed=None, **env_config):
    """A worker factory. `seed` is this worker's own seed (from
    `rl.seeding.worker_seeds`), not the run's, so workers stay independent and
    reproducible.
    """
    config = dict(env_config)
    if seed is not None:
        config["scenario_seed"] = seed

    def _init():
        return MicroRoyaleSelfPlayEnv(config)
    return _init
