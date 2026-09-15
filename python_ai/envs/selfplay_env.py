"""Pipeline 2's environment: team 1 is another policy, not the C++ bot.

Same action/observation shape as `envs/gym_wrapper.MicroRoyaleEnv`, but team 1
is driven from Python -- a frozen `MicroRoyaleNet` snapshot, one of four
scripted heuristics, or (dispatching back through `game.step()`) the built-in
C++ HeuristicOpponent.

OPPONENT SELECTION IS PFSP, not a ladder. Each instance keeps its own local pool
and per-opponent win-rate EMA and samples a fresh opponent at EVERY reset,
weighted toward whichever it is currently doing worst against. The main process
broadcasts pool updates via `refresh_pfsp_pool`; `set_historical_opponent` stays
available for the one-off uses (evaluation, replay generation) that want a
SPECIFIC opponent rather than a sampled one.

Deliberately DECENTRALIZED: AsyncVectorEnv workers are separate OS processes,
and synchronizing a growing stats dict across them every episode would add real
complexity for little benefit at only 8 workers. Each worker converges its own
reasonable local estimate, as in any decentralized-actor setup.
"""

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch.distributions import Categorical

import clash_royale_env
from python_ai.envs import scenarios, scripted_opponents
from python_ai.envs.gym_wrapper import (
    DEFAULT_DECK, DEFAULT_DECK_ABILITY_SLOTS, WIN_CONDITION_ID, _to_scalar,
)
from python_ai.models.net import MicroRoyaleNet
from python_ai.models.policy_io import LSTM_HIDDEN, load_state_dict_flexible
from python_ai.rewards.weights import FIREBALL_CARD_ID

# --- Pipeline #2: Historical Self-Play (League / PFSP) ---
# Pipeline #1 (train.py) teaches the bot to beat a random-but-unskilled
# opponent -- a necessary bootstrap, but not a ceiling: nothing in that
# opponent ever punishes a bad trade, reads a push, or plays around a specific
# threat the way a real opponent would. This pipeline is what actually
# introduces that pressure: the trainee (still the same live policy, continuing
# gradient updates) plays against FROZEN snapshots of its own past selves.
# Both sides genuinely decide what to play (see ClashEnv::stepSelfPlay/
# extractObservationForTeam in the C++ layer) instead of team 1 being the
# built-in random C++ bot.
#
# Opponent selection is Prioritized Fictitious Self-Play (PFSP, as used by
# AlphaStar's league), NOT a linear ladder -- replaces an earlier design that
# advanced through the pool one opponent at a time, gated on a win-rate
# threshold. That design had two structural problems, both confirmed in
# practice (see training_selfplay_run1.log from this project's own history):
# (1) once the pool's genuinely-old/weak snapshots were exhausted, the "next"
# opponent was inevitably a near-mirror of the live trainee (the pool refills
# from the trainee's OWN recent snapshots), so the win-rate gate became
# structurally unreachable -- not because the trainee stopped improving, but
# because a near-mirror matchup settles near 50% by construction; (2) a
# linear ladder never revisits opponents once passed, so nothing prevents
# catastrophic forgetting of earlier matchups. PFSP fixes both: EVERY env,
# at EVERY episode reset, independently samples an opponent from the whole
# eligible pool with probability weighted toward whichever opponents it's
# currently doing WORST against (see PFSP_EXPONENT below) -- so old,
# already-beaten opponents keep appearing (just rarely), and there is no
# single "current opponent" or gate to get stuck against.

# Each of the num_envs worker processes keeps its OWN local per-opponent
# win-rate estimate (see MicroRoyaleSelfPlayEnv.pfsp_stats) and samples from
# it independently -- deliberately decentralized rather than synchronized
# across workers/processes, since AsyncVectorEnv workers are separate OS
# processes and cross-process synchronization of a growing stats dict every
# episode would add real complexity for little benefit at only 8 workers;
# each worker converges its own reasonable local estimate from its own
# experience over time, same as any decentralized-actor RL setup.
PFSP_EXPONENT = 2.0

# Floor on sampling weight even for an opponent the trainee already wins
# 100% against -- keeps EVERY eligible opponent in rotation forever (at low
# frequency) instead of fully forgetting it once its local win-rate estimate
# saturates. This is what actually prevents catastrophic forgetting; PFSP's
# difficulty-weighting alone would drive an already-mastered opponent's
# sampling weight toward (but never quite to) zero, so this floor gives it a
# guaranteed non-zero share.
PFSP_MIN_WEIGHT = 0.05

# Smoothing factor for each worker's local per-opponent win-rate EMA
# (outcome in {1.0 win, 0.5 draw, 0.0 loss}, new_stat = old*(1-a) + outcome*a).
# ~1/0.08 = 12-game effective window -- short enough to track a real trend as
# both the trainee and the sampling weights shift over a long run, without
# being so short that one lucky/unlucky game swings the weight.
PFSP_EMA_ALPHA = 0.08

# The C++ HeuristicOpponent, as a TRAINING opponent rather than only an
# evaluation anchor. Added 2026-07-31 after 13 evals over 65,000 phase-2
# episodes showed no measurable movement against it.
#
# The diagnosis those evals support: phase 2 never trains against the opponent
# it is measured on. stepSelfPlay deliberately never calls opponentTurn(), so
# the entire phase-2 pool is neural past-selves plus deck-agnostic Python bots,
# and the agent got correspondingly good at exactly that -- 0.84 against the
# pool, 0.92-1.00 against the neural anchors -- while its score against the
# heuristic stayed flat (chi2 13.1 on 12 df for @1.50, 17.4 on 12 df for @1.35,
# both well under the 21.03 critical value). Phase 1 DID train against it, and
# phase 1 is where the current strength came from. An exploiter of a mirror
# specialist is just another mirror specialist, which is why three bursts found
# nothing either.
#
# Only the two multipliers with headroom. The agent scores 1.00 against
# @1.00, so episodes there would teach nothing; the measured gap is 0.83 at
# @1.35 and 0.60 at @1.50.
BUILTIN_TRAINING_OPPONENTS = ["builtin:heuristic@1.35", "builtin:heuristic@1.50"]

# Same reasoning as DEFENSIVE_SCRIPTED_MIN_WEIGHT below, and the same override
# of PFSP's own criterion: the reason to keep facing these is not "the trainee
# is currently losing to them" but that they are the measurement target, and
# PFSP would taper them off exactly as the agent improved.
#
# Sized so they stay a meaningful minority rather than taking the run over. At
# the current pool of ~20, with the other members at PFSP_MIN_WEIGHT except
# Defender/Counter at 0.8: 2*0.5 / (16*0.05 + 2*0.8 + 2*0.5) = ~29%. The pool
# only grows, so this share decays on its own -- ~23% at 40 members. Setting it
# to 0.8 like the defensive bots would have made it ~64% and effectively
# reverted phase 2 into phase 1.
BUILTIN_MIN_WEIGHT = 0.5

class MicroRoyaleSelfPlayEnv(gym.Env):
    """Same action/observation shape as gym_wrapper.MicroRoyaleEnv, but team 1
    is a frozen copy of MicroRoyaleNet (never trained here -- eval()/no_grad
    only) instead of the built-in random C++ opponentTurn().

    Opponent selection is PFSP (see this module's own top-of-file comment):
    each instance keeps its own local pool (pfsp_pool) and per-opponent
    win-rate EMA (pfsp_stats), and samples a fresh opponent at every reset()
    weighted toward whichever it's currently doing worst against. The main
    process broadcasts pool updates via refresh_pfsp_pool(); set_historical_
    opponent() is still available directly for one-off uses (evaluation,
    replay generation) that want a SPECIFIC opponent rather than a sampled
    one.
    """

    def __init__(self, env_config=None):
        super().__init__()
        env_config = env_config or {}
        self.deck = env_config.get("deck", list(DEFAULT_DECK))
        max_ticks = env_config.get("max_ticks", 3600)
        # Tower Troops: per-match config, not a per-step action -- see
        # gym_wrapper.py's identical wiring. NONE (the default) reproduces
        # the original hardcoded Princess Tower unchanged.
        ai_tower_troop = env_config.get("ai_tower_troop", clash_royale_env.TowerTroopType.NONE)
        opp_tower_troop = env_config.get("opp_tower_troop", clash_royale_env.TowerTroopType.NONE)
        self.game = clash_royale_env.ClashRoyaleEnv(self.deck, self.deck, max_ticks, ai_tower_troop, opp_tower_troop)
        # Instance attributes (not class-level constants) -- pulled live from
        # the engine's own enforced placement bounds instead of a hardcoded
        # copy that could silently drift if board geometry ever changes.
        self.MAX_X = self.game.get_max_placement_x()
        self.MAX_Y = self.game.get_own_half_max_y()

        # Team 1's brain -- CPU is plenty for a single inference-only forward
        # pass per step per worker process, and keeps this off the GPU the
        # trainee's own updates use in the main process.
        self.device = torch.device("cpu")
        self.opponent_net = MicroRoyaleNet(num_ability_slots=DEFAULT_DECK_ABILITY_SLOTS).to(self.device)
        self.opponent_net.eval()
        self.opponent_hx = torch.zeros(1, LSTM_HIDDEN).to(self.device)
        self.opponent_cx = torch.zeros(1, LSTM_HIDDEN).to(self.device)
        self.opponent_checkpoint_path = None
        # "neural" (self.opponent_net drives team 1) or one of SCRIPTED_
        # OPPONENTS' bare names ("Rusher"/"Defender"/"Cycler"/"Counter") --
        # see _set_opponent's dispatch and _scripted_opponent_action.
        self.opponent_kind = "neural"
        # Rusher/Counter commit to one lane for the whole episode (real
        # players don't re-decide their push lane every single card) --
        # set once per episode in set_scripted_opponent, read in
        # _scripted_opponent_action.
        self.opponent_lane = None

        # PFSP pool/stats -- see refresh_pfsp_pool()/_sample_pfsp_opponent().
        # Empty until the main process's first broadcast; reset() no-ops the
        # sampling step until then (should never actually happen in normal
        # operation, since the Phase2Trainer broadcasts before the first
        # envs.reset() call).
        self.pfsp_pool = []
        self.pfsp_stats = {}

        # Scenario injection (see SCENARIOS / sample_scenario). Independent
        # per-worker RNG so the vectorized envs don't all inject the same
        # scenario in lockstep. scenarios_enabled=False (used by the replay
        # env) keeps replays representative of full, un-engineered games.
        self.scenarios_enabled = env_config.get("scenarios_enabled", True)
        #: THE env's only stochastic source: scenario injection, the attacking
        #: lane, and the PFSP opponent draw all read this one generator.
        #:
        #: They used to be three separate uncontrolled streams -- a bare
        #: `default_rng()`, the stdlib global via `random.choice`, and the numpy
        #: global via `np.random.choice` -- which is what made pipeline 2
        #: unreproducible: a crash could not be re-run and a paired A/B could
        #: not hold the opponent and scenario draws fixed across arms.
        #:
        #: `scenario_seed` is the SAME config key pipeline 1's env already
        #: takes, rather than a second convention. Default None keeps the old
        #: behaviour exactly -- fresh OS entropy, independent per worker -- so
        #: seeding is opt-in and no existing run changes. `rl/seeding.py`
        #: derives the per-worker seeds that keep workers independent AND
        #: reproducible.
        self.rng = np.random.default_rng(env_config.get("scenario_seed", None))
        #: Kept as an alias: `scenario_rng` is the name the scenario code and
        #: its tests already use, and it must be the SAME object -- two
        #: generators would re-open the reproducibility hole this closes.
        self.scenario_rng = self.rng
        self.scenario_active = None       # name of the current episode's scenario, or None
        self.scenario_defensive = False   # is ScenDef a meaningful test for it?
        self.scenario_max_steps = None    # truncation window in bot-steps, or None for full game
        self.scenario_steps_taken = 0

        if env_config.get("historical_checkpoint_path"):
            self.set_historical_opponent(env_config["historical_checkpoint_path"])

        self.action_space = spaces.Dict({
            "card_index": spaces.Discrete(clash_royale_env.ClashRoyaleEnv.HAND_SIZE + 1),
            "target_x": spaces.Box(low=0.0, high=self.MAX_X, shape=(1,), dtype=np.float32),
            # Full board height -- see gym_wrapper.MicroRoyaleEnv's identical
            # comment: spells are exempt from the own-half restriction, and
            # per-card legality is enforced by MicroRoyaleNet.placement_mask.
            # self.MAX_Y is still used by the scripted opponents' heuristics,
            # which really are own-half-only.
            "target_y": spaces.Box(low=0.0, high=float(clash_royale_env.ClashRoyaleEnv.BOARD_HEIGHT - 1),
                                   shape=(1,), dtype=np.float32),
            # Same keys as gym_wrapper.MicroRoyaleEnv's action_space -- see
            # its own comment. Team 1 (the frozen historical opponent) samples
            # its own independent pair via _opponent_action(), so both sides
            # can actually use Champion abilities during self-play.
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
        # Revert a previous scripted opponent's randomized deck, if any --
        # set_opponent_deck() has no auto-reset of its own (see
        # gym_wrapper.py's identical reset()-time re-apply), so without this
        # a neural opponent sampled right after a scripted one would
        # silently keep playing that random deck instead of self.deck.
        self.game.set_opponent_deck(self.deck)
        self._reset_opponent_elixir()

    def set_scripted_opponent(self, name):
        """Team 1 becomes a hand-written heuristic bot instead of a frozen
        checkpoint -- see SCRIPTED_OPPONENTS. Also randomizes team 1's deck
        (scoped to scripted opponents only, never neural ones: these
        heuristics read nothing card-ID-specific -- only elixir/cost from
        the observation and enemy positions from the spatial channels -- so
        they're deck-agnostic by construction, unlike a historical
        checkpoint, which only ever learned to play self.deck)."""
        self.opponent_kind = name
        self.opponent_checkpoint_path = f"scripted:{name}"
        # Correct-by-construction (not rejection-sampling against
        # validate_deck_slots) -- see sampleRandomDeck's own comment in
        # ClashEnv.h.
        self.game.set_opponent_deck(clash_royale_env.sample_random_deck())
        self._reset_opponent_elixir()
        if name in ("Rusher", "Counter"):
            self.opponent_lane = ("left" if self.rng.random() < 0.5 else "right")

    def _reset_opponent_elixir(self):
        """Undo any elixir multiplier a previous builtin opponent left behind.

        Exactly the same hazard the set_opponent_deck() re-apply above guards
        against, and worse if missed: set_opponent_elixir_multiplier() has no
        auto-reset, so a 1.5x heuristic episode would silently hand the NEXT
        sampled opponent -- a frozen snapshot, in a supposedly fair mirror
        matchup -- 50% extra elixir. That would corrupt both the trainee's
        gradient and the pfsp_stats win rate that drives sampling, and would
        look like nothing more than a sudden unexplained dip in win rate."""
        self.game.set_opponent_elixir_multiplier(1.0)

    def set_builtin_opponent(self, descriptor):
        """Team 1 becomes the C++ HeuristicOpponent at a given elixir
        multiplier -- see BUILTIN_TRAINING_OPPONENTS.

        Unlike every other opponent kind, this one is not driven from Python at
        all: it runs inside game.step(), which step_self_play() deliberately
        never calls. step() below dispatches on opponent_kind for that reason.

        Deck stays self.deck on both sides, matching how evaluate_against_roster
        builds these anchors (gym_wrapper.MicroRoyaleEnv defaults opp_deck to
        ai_deck) -- so what is trained against here is exactly what is measured
        against, which is the entire point of adding them."""
        self.opponent_kind = "builtin"
        self.opponent_checkpoint_path = descriptor
        self.game.set_opponent_deck(self.deck)
        self.game.set_opponent_elixir_multiplier(float(descriptor.split("@")[1]))

    def _set_opponent(self, descriptor):
        """Dispatch for whatever _sample_pfsp_opponent() (or a direct
        override) picked -- a real checkpoint path, one of SCRIPTED_OPPONENTS'
        "scripted:<name>" tags, or a "builtin:heuristic@<mult>" tag. The
        "builtin:" spelling is deliberately the same one
        evaluate_against_roster already dispatches on, and is never a real
        file path, so torch.load is never attempted on it."""
        if descriptor.startswith("scripted:"):
            self.set_scripted_opponent(descriptor[len("scripted:"):])
        elif descriptor.startswith("builtin:"):
            self.set_builtin_opponent(descriptor)
        else:
            self.set_historical_opponent(descriptor)

    def refresh_pfsp_pool(self, pool_paths):
        """Broadcast from the main process (Phase2Trainer, via
        envs.call) whenever the eligible pool changes -- at startup, and
        after every new historical snapshot is saved. New entries start at a
        neutral 0.5 win-rate prior; PFSP_EXPONENT's weighting naturally
        prioritizes them for extra games until their real difficulty is
        established. Pool only ever grows (an eligible checkpoint never
        becomes ineligible again), so this never needs to prune pfsp_stats."""
        self.pfsp_pool = list(pool_paths)
        for p in self.pfsp_pool:
            if p not in self.pfsp_stats:
                self.pfsp_stats[p] = 0.5

    def _sample_pfsp_opponent(self):
        """Draw this episode's opponent, dropping any entry that will not load.

        THIS RUNS ON EVERY RESET, and the pool is sampled at random, so an
        unreadable checkpoint used to crash phase 2 at an unpredictable point --
        typically hours in, with a traceback naming torch.load rather than the
        pool. The pool was written NON-ATOMICALLY for this project's whole
        history (`atomic_save` only landed 2026-08-26), so a file truncated by
        an OOM kill or a SIGKILL can already be sitting there; fixing the writer
        does not clean up what the old writer left.

        Both directions are failures and they pull opposite ways. Crashing over
        one bad file wastes a run for an opponent that could have been skipped.
        Silently swallowing load errors is worse: an architecture change makes
        EVERY checkpoint unloadable at once, the pool empties to the scripted
        bots, and the league stops being self-play with nothing said -- which is
        the shape CLAUDE.md already records for an empty pool ("silent, and it
        degrades the opponent distribution rather than crashing"). So: skip the
        entry, say so, and refuse to continue if the pool drains entirely.
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
        # New match: drop the played-tick baseline so the first poll re-seeds
        # against this episode's clock instead of comparing it to the previous
        # match's. Clearing once here covers the scenario warm-up's repeated
        # reset() loop below too -- the baseline is empty either way, and the
        # first poll of the episode is what seeds it.
        self._opp_last_tick = {}

        self.scenario_active = None
        self.scenario_max_steps = None
        self.scenario_steps_taken = 0
        self.scenario_defensive = False
        if self.scenarios_enabled and self.scenario_rng.random() < scenarios.SCENARIO_INJECTION_PROB:
            scenario = scenarios.sample_scenario(self.scenario_rng)

            # A scenario may require a specific card in the trainee's opening
            # hand. The opening shuffle is an unseeded mt19937 that cannot be
            # set and whose queue cannot be read, so the supported way to
            # control the hand is to re-roll until it comes up. reset() costs
            # 0.135 ms and its shuffle is uniform over all 70 hand-sets, so a
            # 4-of-8 card arrives in ~2 tries. Capped, and falling through on
            # exhaustion rather than looping: a scenario that occasionally runs
            # without its card is a diluted scenario, but a reset that can hang
            # is a stalled worker.
            #
            # NO SCENARIO CURRENTLY SETS THIS. Its only user, giant_commit, was
            # removed on 2026-08-19 -- it asked for a card DEFAULT_DECK cannot
            # contain, so the cap was reached on every single draw. The hook is
            # kept because it is the mechanism, not the dead config, and the
            # next scenario that needs a named card should use it -- but check
            # the card is actually in the deck, which is the failure that made
            # giant_commit inert for three days without anything noticing.
            want = scenario.get("require_own_card")
            if want is not None:
                for _ in range(24):
                    if want in self.game.get_hand():
                        break
                    self.game.reset()

            # Banked elixir, BEFORE the spawns so injected units do not walk
            # during the warm-up. Both sides regenerate together here (no-op on
            # both, and step_self_play never calls opponentTurn), so this is a
            # quiet mid-match moment rather than a handout.
            warmup = scenario.get("warmup_ticks", 0)
            if warmup:
                noop_w = clash_royale_env.ClashRoyaleEnv.HAND_SIZE
                self.game.step_self_play(noop_w, 0.0, 0.0, noop_w, 0.0, 0.0, warmup)

            for card_id, x, y in scenario["spawns"]:
                self.game.inject_enemy(card_id, x, y)
            # inject_enemy only QUEUES units into pendingEntities -- they aren't
            # in the observation until a game.step() commits them. Run a single
            # 1-tick no-op self-play step (card index HAND_SIZE = no-op on both
            # sides, no opponentTurn) so the threat is visible in the very first
            # observation the trainee acts on; otherwise a Hog would be a step's
            # worth of travel toward the tower before the net ever sees it. One
            # tick of drift is negligible.
            noop = clash_royale_env.ClashRoyaleEnv.HAND_SIZE
            self.game.step_self_play(noop, 0.0, 0.0, noop, 0.0, 0.0, 1)
            self.scenario_active = scenario["name"]
            self.scenario_max_steps = scenario["max_steps"]
            # Whether "the episode did not end in a big loss" is a meaningful
            # success test for THIS scenario. It is for the ones that put a
            # threat in our half; it is vacuous for the ones that do not, and
            # mixing them makes ScenDef unreadable -- see the info dict below.
            self.scenario_defensive = bool(scenario.get("defensive", False))

        # get_observation_for_team(0) == game.reset()'s own return for a normal
        # reset, but re-read here so it reflects any just-injected units.
        obs_list = self.game.get_observation_for_team(0)
        self.opponent_hx = torch.zeros(1, LSTM_HIDDEN).to(self.device)
        self.opponent_cx = torch.zeros(1, LSTM_HIDDEN).to(self.device)
        return np.array(obs_list, dtype=np.float32), {}

    def _opponent_action(self):
        """Team 1's own decision, from ITS OWN (mirrored) point of view --
        see ClashEnv::extractObservationForTeam. Sampled the same way rollout
        actions are everywhere else in this project, just with no_grad and no
        buffering: this network never gets updated here. Also samples its own
        two independent Champion-ability decisions (slot1/slot2) -- otherwise
        team 1 would systematically never use Champion abilities even once
        the trainee (team 0) does."""
        obs1 = np.array(self.game.get_observation_for_team(1), dtype=np.float32)
        if self.opponent_kind != "neural":
            return self._scripted_opponent_action(obs1)
        with torch.no_grad():
            obs1_t = torch.tensor(obs1, dtype=torch.float32).unsqueeze(0).to(self.device)
            # Team 1 gets the SAME affordability mask treatment as the trainee --
            # obs1 is team 1 own mirrored observation, so its elixir/cost scalars
            # are its own. Without this the frozen opponent would spend ~75% of
            # its turns attempting plays the engine silently refuses, i.e. it
            # would be a far weaker (and differently-behaved) opponent than the
            # checkpoint it is supposed to be reproducing.
            card_mask1 = self.opponent_net.affordability_mask(obs1_t)
            # `_hires`, and the map is THREADED into placement_given_card below.
            # The three-value wrapper computes `cnn_trunk[:2](spatial_obs)` and
            # discards it, and placement_given_card(hires_map=None) rebuilds it
            # from the same obs -- so the trunk's most expensive layer ran TWICE
            # per opponent decision. net.py accepts that duplication in COLD
            # paths (the eval harnesses) by design; this is not one. It runs
            # once per env per step for the whole of pipeline 2, exactly like
            # the trainee's own path in base_trainer.collect_rollout, which was
            # measured at 67.7% of the rollout's network time.
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
            # Its OWN Champion decisions, under its own readiness -- otherwise
            # the league's neural opponents would never use a Champion the
            # trainee is learning to use (rl/abilities.py).
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
        """Delegate to the hand-written heuristics.

        The heuristic itself lives in `envs/scripted_opponents.py`: it is a
        POLICY and this class is an ENVIRONMENT, and 120 lines of it inside a
        method here was the clearest case in the tree of one class doing two
        unrelated jobs.
        """
        return scripted_opponents.scripted_action(
            self.opponent_kind, obs1, self.opponent_lane, self.MAX_X, self.MAX_Y)

    def _poll_opponent_play(self):
        """Which card team 1 played since the last poll, or -1 for none.

        Same baseline-and-diff as gym_wrapper.MicroRoyaleEnv._poll_opponent_
        play; kept as a sibling rather than shared because the two envs hold
        their decks differently (this one plays self.deck on BOTH sides) and a
        shared helper would need the deck passed in anyway. The first poll of
        an episode returns -1 and only seeds the baseline.
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
            # The C++ HeuristicOpponent lives inside game.step()'s
            # opponentTurn(), which step_self_play() never calls -- so this is
            # the phase-1 code path, byte-for-byte what gym_wrapper's step()
            # does. No Python-side opponent action exists to compute.
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

        # Scenario truncation: end the focused defensive window WITHOUT marking
        # the game terminated (it isn't -- no king died). The training loop
        # bootstraps V(final_obs) for this, so the critic isn't told the world
        # ends here. Only applies while a scenario with a finite window is
        # active and the game hasn't already ended on its own.
        truncated = False
        if self.scenario_max_steps is not None and not terminated:
            self.scenario_steps_taken += 1
            if self.scenario_steps_taken >= self.scenario_max_steps:
                truncated = True

        # PFSP difficulty tracking must measure the OPPONENT's strength, not the
        # extra handicap of a free injected threat -- so scenario episodes never
        # update pfsp_stats (self.scenario_active is None only on normal games).
        if terminated and self.opponent_checkpoint_path is not None and self.scenario_active is None:
            if reward > 0.5:
                outcome = 1.0
            elif reward < -0.5:
                outcome = 0.0
            else:
                outcome = 0.5
            prev = self.pfsp_stats.get(self.opponent_checkpoint_path, 0.5)
            self.pfsp_stats[self.opponent_checkpoint_path] = prev * (1.0 - PFSP_EMA_ALPHA) + outcome * PFSP_EMA_ALPHA

        # Same key names/shape as gym_wrapper.MicroRoyaleEnv.step()'s info dict
        # on purpose -- lets rewards.shaping.compute_shaping be reused as-is.
        info = {
            "elixir": self.game.get_elixir(),
            "hand": self.game.get_hand(),
            "team0_troop_damage": self.game.get_troop_damage_dealt(0),
            "team1_troop_damage": self.game.get_troop_damage_dealt(1),
            "team0_building_damage": self.game.get_building_damage_dealt(0),
            # Towers only. compute_shaping() needs tower damage and
            # deployed-building damage priced differently -- see
            # rewards.shaping.tower_potential.
            # --- inputs for the lethal-spell PBRS term (rewards.shaping.lethal_spell_potential)
            # Enemy tower HP in ABSOLUTE points. The observation carries these
            # normalized in its appended scalar tail (indices 6-8 = enemy
            # king/left/right), so this is a re-scale of data the net already
            # sees rather than a new engine call.
            # Heuristic-1 inputs (rewards.shaping.spell_value_shaping). BOTH are needed:
            # value-destroyed alone makes a whiffed spell free, which is the
            # guaranteed-zero trap that parked the Cannon in a back corner.
            "fireball_value_killed": self.game.get_elixir_value_killed_by(FIREBALL_CARD_ID, 0),
            "fireball_elixir_spent": self.game.get_elixir_spent_on_card(FIREBALL_CARD_ID, 0),
            "enemy_tower_hp": np.asarray(obs[clash_royale_env.ClashRoyaleEnv.EXTRA_SCALARS_START + 6:clash_royale_env.ClashRoyaleEnv.EXTRA_SCALARS_START + 9], dtype=np.float32) * clash_royale_env.ClashRoyaleEnv.MAX_BUILDING_HP,
            "fireball_in_hand": float(FIREBALL_CARD_ID in list(self.game.get_hand())),
            "team0_tower_damage": self.game.get_tower_damage_dealt(0),
            "team1_tower_damage": self.game.get_tower_damage_dealt(1),
            "team1_building_damage": self.game.get_building_damage_dealt(1),
            "team0_elixir_spent": self.game.get_elixir_spent(0),
            "team1_elixir_spent": self.game.get_elixir_spent(1),
            # Surviving tower counts, for the discrete crown term in
            # compute_shaping() -- see W_TOWER_DESTROYED.
            "team0_towers_alive": self.game.get_towers_alive(0),
            "team1_towers_alive": self.game.get_towers_alive(1),
            # Cumulative damage by the deck's win condition -- input to
            # compute_shaping's win-condition term. Pipeline 2 builds its
            # ClashRoyaleEnv directly rather than through gym_wrapper, so this
            # key has to be supplied here too or the term silently contributes
            # zero for the whole of self-play.
            "team0_wincon_damage": (self.game.get_damage_dealt_by_card(WIN_CONDITION_ID, 0)
                                    if WIN_CONDITION_ID is not None else 0),
            # Supervision target for the auxiliary NEXT-CARD head -- see the
            # identical key in gym_wrapper.MicroRoyaleEnv.step()'s info dict
            # and MicroRoyaleNet.predict_opp_next_card. Hidden information (at
            # the moment the agent acted it had not happened yet), so it
            # travels through info and never through the observation.
            # Both sides play self.deck here, so the poll watches that.
            "opp_played_card": self._poll_opponent_play(),
            # Diagnostics only since the 2026-08-28 aux swap; no head reads it.
            "opp_elixir": self.game.get_elixir_for_team(1),
            "champion_ability_slot1_ready": self.game.is_champion_ability_ready(0, 1),
            "champion_ability_slot2_ready": self.game.is_champion_ability_ready(0, 2),
            # 1.0 while the current episode started from an injected scenario --
            # lets the training loop score scenario defenses separately from
            # normal-matchup win/loss (see Scenario/Defense_Success_Rate).
            "is_scenario": 1.0 if self.scenario_active is not None else 0.0,
            # Split out because ScenDef means "did the agent survive a threat",
            # and fireball_tower_value puts no threat in our half at all -- it
            # spawns at the ENEMY tower. There, "did not take a big hit" is true
            # no matter what the agent does -- including nothing -- so counting
            # it drags ScenDef toward 1.0 and hides real defensive failures.
            # (giant_commit, which spawned nothing, was the other such case and
            # was removed on 2026-08-19.)
            "scenario_defensive": 1.0 if getattr(self, "scenario_defensive", False) else 0.0,
            # Per-card realized elixir economy, for the strategy readout's ROI
            # column. This is the falsifiable half of "did un-choking the
            # entropy controller help": raising exploration alone SPREADS
            # placements and LOWERS return per elixir, so a spread that rises
            # while ROI also rises is learning, and a spread that rises while
            # ROI falls is just noise. Engine counters, so nothing here can be
            # gamed by the policy.
            #
            # Filled only on the terminal step -- 16 registry calls per EPISODE
            # rather than per step -- but the KEYS are always present and the
            # shape is always (8,). A key that appears on some steps and not
            # others makes the vector env's info aggregation emit a companion
            # mask array instead of a plain stack, which is a needless trap for
            # the reader; zeros mid-episode cost nothing and the trainer only
            # reads these on a real episode end.
            "ep_killed_by_card": self._episode_economy(0, terminated or truncated),
            "ep_spent_by_card": self._episode_economy(1, terminated or truncated),
        }
        return obs, reward, terminated, truncated, info

    # Deck order is DEFAULT_DECK: phase 2's neural opponents and the trainee all
    # play it (see CLAUDE.md). A scripted bot's randomised deck belongs to team
    # 1, which these team-0 counters never touch.
    def _episode_economy(self, which, ended):
        if not ended:
            return np.zeros(len(DEFAULT_DECK), dtype=np.float32)
        fn = (self.game.get_elixir_value_killed_by if which == 0
              else self.game.get_elixir_spent_on_card)
        return np.asarray([fn(c, 0) for c in DEFAULT_DECK], dtype=np.float32)


def make_env(seed=None, **env_config):
    """A worker factory. `seed` is this worker's OWN seed, not the run's.

    `rl.seeding.worker_seeds` derives one per worker from the run seed, so the
    envs stay statistically independent -- they must not inject the same
    scenario in lockstep -- while the run as a whole stays reproducible.
    """
    config = dict(env_config)
    if seed is not None:
        config["scenario_seed"] = seed

    def _init():
        return MicroRoyaleSelfPlayEnv(config)
    return _init
