import os
import glob
import re
import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import deque, defaultdict

import clash_royale_env
import gym_wrapper
from gym_wrapper import (
    DEFAULT_DECK, DEFAULT_DECK_ABILITY_SLOTS, _to_scalar, train_FIREBALL_ID,
    WIN_CONDITION_ID,
)
from model import MicroRoyaleNet
from train import (
    compute_shaping, building_hp_end, annotate_replay_with_agent_info,
    HISTORICAL_CHECKPOINT_DIR, HISTORICAL_CHECKPOINT_INTERVAL_EPISODES,
    DRAW_PENALTY,
    PLACEMENT_COVERAGE_COEF, placement_coverage_slots,
    spell_value_weight,
    W_FLAWLESS_DEFENSE, OWN_TOWER_HP_TOTAL, flawless_defense_bonus,
)
from policy_io import load_state_dict_flexible
import advisor_target as AT
import exploiter as exploiter_mod

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

# Same escape hatch as before for mutual passivity: once draws cross this
# share of the current 100-episode window (now aggregated across whatever mix
# of pool opponents each worker happened to sample, not one fixed opponent),
# re-boost exploration -- floored entropy is exactly what locks a passive
# equilibrium in place, since neither side has any remaining chance to
# stumble into a different joint strategy. Reward-shaping fixes (see
# train.py's W_ELIXIR_TRADE/DRAW_PENALTY/W_ELIXIR_OVERFLOW comments) address
# WHY passivity looked attractive; this addresses the fact that once both
# sides are already sitting in it, ordinary gradient descent has no
# exploration left to climb back out.
STALL_DRAW_RATE_THRESHOLD = 0.5
# Cooldown so this doesn't re-fire every single episode once the window is
# saturated with draws -- gives each boost a real window to actually take
# effect before deciding whether another one is needed.
STALL_REBOOST_COOLDOWN_EPISODES = 1000
# Escape hatch for the failure mode the draw-rate trigger above CAN'T see:
# draws near zero but the aggregate win rate across the pool still stuck
# losing, entropy floored the whole time with nothing ever refreshing it.
# Narrowed to also require win_rate < 0.5 (not just "below some target"):
# confirmed in practice (this project's own earlier ladder-based run) that
# firing an exploration re-boost while ALREADY decisively winning measurably
# HURTS -- there's no better joint policy for injected noise to find in a
# matchup that's already going well, only sampling noise for the entropy
# bonus to add. Below 0.5 the trainee is genuinely losing more than winning
# against the pool it's currently being sampled against, which IS worth
# reacting to.
ENTROPY_STALE_REBOOST_EPISODES = 1500
# Pipeline #2's own snapshots (saved every HISTORICAL_CHECKPOINT_INTERVAL_EPISODES
# = 5000 episodes, see train.py) refill the SAME shared pool this script draws
# opponents from -- once the genuinely-old/weak pipeline #1 snapshots are used
# up, every remaining candidate is one of the trainee's own very recent
# selves. discover_historical_checkpoints() excludes pipeline #2 snapshots
# younger than this many episodes (relative to the live trainee's CURRENT
# episodes_completed) from eligibility, so the pool -- and therefore both PFSP
# sampling and the evaluation roster below -- always reflects genuinely-older,
# genuinely-distinguishable versions instead of coin-flip-parity mirrors.
# Pipeline #1 snapshots are always eligible (wholly separate, earlier, much
# weaker phase -- see discover_historical_checkpoints()'s docstring for why
# their episode count isn't even comparable to this one).
#
# Lowered 15000 -> 6000 on 2026-08-09, together with
# HISTORICAL_CHECKPOINT_INTERVAL_EPISODES (5000 -> 2000). These two are COUPLED
# and must move together: this gate has always been exactly 3x the snapshot
# interval, i.e. "exclude the three newest snapshots". Leaving it at 15000 while
# the interval dropped to 2000 would have silently turned it into "exclude the
# seven newest" -- a materially more conservative pool than was ever intended,
# arrived at by changing a constant nobody edited.
#
# The re-denomination points the same way on its own merits. Since the
# 2026-08-07 speed fix an episode carries ~2.2x more policy change, so 15,000
# episodes now represents MORE divergence than when that number was picked --
# the gate was becoming over-conservative, not under. 15000 / 2.2 ~= 6,800, and
# 3 x 2000 = 6,000; both say the same thing.
MIN_OPPONENT_AGE_EPISODES = 6000

# --- Fixed-roster Elo-style evaluation ---
# Win rate against "whichever opponent PFSP happened to sample this window"
# is not a stable progress signal (it swings with the pool composition, same
# root problem the ladder had). This periodically plays the CURRENT policy,
# GREEDY (argmax card, distribution mean placement -- no sampling, so the
# measurement isn't itself noisy from exploration), against a small FIXED
# roster of reference opponents that, once chosen, never changes -- so the
# resulting score is comparable across the whole run and actually answers
# "is this getting better" instead of "did this window's sampled mix happen
# to be easy or hard."
EVAL_INTERVAL_EPISODES = 5000
# Raised 10 -> 50. At 10 games the Elo readout was unusable. Elo is derived via
# 400*log10(1/score - 1), which explodes near a score of 1.0: flipping ONE game
# on an anchor sitting at 0.9 moved that anchor's implied Elo by 141 points and
# the reported average by ~28. Measured across 8 evaluations spanning 35,000
# episodes: mean 1755, std 53, range 164 -- no resolvable trend, so the metric
# could neither confirm nor rule out improvement. 50 games cuts the per-anchor
# standard error roughly in half.
EVAL_GAMES_PER_OPPONENT = 50
# Reference roster grows (see update_reference_roster()) as new age-eligible
# checkpoints appear, capped here -- once full, it's permanently frozen so
# the Elo scale stays comparable for the rest of the run.
REFERENCE_ROSTER_MAX_SIZE = 6

# PERMANENT, non-self-referential anchors: the C++ HeuristicOpponent at three
# elixir multipliers.
#
# Why these have to exist. Every anchor used to be a snapshot of the trainee's
# own past, which makes the whole Elo scale self-referential -- it measures
# "better than I used to be", not "good at the game". That failure was measured
# twice over: 4 of 5 historical anchors sat pinned at a score of 0.90-1.00 for
# 8 straight evaluations (no resolution left), and separately the heuristic
# opponent -- roughly 100 lines of hand-written rules -- beat a policy trained
# for 47,000 episodes 71% of the time. A scale built only from self-copies
# cannot see either problem.
#
# Multipliers chosen from measurement, not guessed: the current policy scores
# 100% vs 1.0x and 97% vs 1.2x, so 1.0x is already saturated and only useful as
# a floor/regression tripwire. 1.35x and 1.5x are where there is real
# resolution. Elos are hand-assigned and ordered by measured difficulty -- as
# with the historical anchors these are a comparable TREND, not a calibrated
# rating (see REFERENCE_ROSTER_ELO_STEP).
#
# The "builtin:" prefix is what evaluate_against_roster dispatches on; it is
# never a real file path, so torch.load is never attempted on it.
BUILTIN_ANCHORS = [
    ("builtin:heuristic@1.00", 1200),
    ("builtin:heuristic@1.35", 1500),
    ("builtin:heuristic@1.50", 1700),
]
# Anchor Elo values are HAND-ASSIGNED, evenly spaced by pool position at the
# time each anchor is added -- NOT empirically cross-calibrated against each
# other. This means the absolute number this produces is not a "real"
# competitive Elo rating; what's meaningful is the TREND across successive
# evaluation rounds against this exact fixed roster (is Eval/Elo trending up
# over tens of thousands of episodes), which is exactly the signal that was
# missing before.
REFERENCE_ROSTER_BASE_ELO = 1000
REFERENCE_ROSTER_ELO_STEP = 150

# --- Auxiliary task: opponent elixir estimation ------------------------------
# Weight on the auxiliary loss that trains MicroRoyaleNet.predict_opp_elixir
# (see that method for why the head exists at all). The head predicts the
# opponent's CURRENT elixir, which is hidden information and deliberately
# absent from the observation; the net has to infer it from elapsed time and
# both sides' cumulative spend, which ARE in the observation.
#
# Its gradient flows back into the shared LSTM/CNN trunk -- that is the entire
# point. This is representation shaping, not an extra output: the sparse
# win/loss signal gives the recurrent state almost no reason to integrate
# opponent spending over a whole match, and "how much elixir do they have
# right now" is the single most load-bearing latent variable in the game.
#
# 0.5 chosen so the term is a real but minority contributor: the target is in
# elixir units (0-10) and a well-fit head sits around 1.0-1.5 MAE, i.e. an MSE
# of ~1-2, versus an actor loss of order 0.1. Scaled by 0.02 below to bring
# the two into the same range before weighting.
AUX_ELIXIR_COEF = 0.5
# Converts the elixir-unit MSE into the same numeric range as the other loss
# terms. Kept explicit (rather than folded into AUX_ELIXIR_COEF) so the LOGGED
# diagnostic stays in interpretable elixir units.
AUX_ELIXIR_SCALE = 0.02

WEIGHT_PATH = "model_weights_selfplay.pth"
# Pipeline #1's final artifact -- read ONCE, only to seed a from-scratch
# pipeline #2 run (bare weights only; pipeline #2 keeps its own separate
# episode count/optimizer state in WEIGHT_PATH from then on, so pipeline #1's
# own checkpoint is never overwritten by this script).
BOOTSTRAP_FROM_PATH = "model_weights.pth"

# --- Diverse scripted opponents (industry precedent: OpenAI Five bootstrapped
# against scripted bots before self-play -- pure self-play alone tends to
# converge onto whatever beats a narrow, self-similar pool rather than
# generalizing). Added as PERMANENT members of the SAME PFSP pool/weighting
# used for historical checkpoints (see _sample_pfsp_opponent) -- not a
# separate curriculum stage, since PFSP's own (1-winrate)^exponent weighting
# with a floor already does exactly what's wanted here: never fully drops
# out, gets sampled more if the trainee is currently weak against it. Tagged
# "scripted:<name>" (never a real file path) so _sample_pfsp_opponent's
# dispatch and pfsp_stats' win-rate keying both work unchanged -- see
# _set_opponent's prefix check.
SCRIPTED_OPPONENTS = ["scripted:Rusher", "scripted:Defender", "scripted:Cycler", "scripted:Counter"]

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

# Defender/Counter specifically model defensive play. PFSP's own win-rate-
# based weighting works AGAINST deliberately seeing more of them: once the
# trainee reliably beats a given opponent, (1-winrate)^PFSP_EXPONENT pushes
# its weight toward PFSP_MIN_WEIGHT same as anything else already mastered --
# so simply having them in the pool doesn't increase exposure on its own,
# and if anything actively suppresses it the better the trainee gets against
# them. The actual reason to keep facing them isn't "the trainee is
# currently weak against them" (PFSP's own criterion) -- it's that defensive
# pressure looked underrepresented in what's deciding average game length
# (see the ~300-tick average game length discussion this responds to), a
# property PFSP has no way to see or weight for on its own. This floor
# overrides PFSP_MIN_WEIGHT for these two specifically, independent of
# measured win-rate.
#
# First tried at 4x the base floor (0.20). Confirmed NOT enough: with a
# ~98-member pool where the other 96 sit at PFSP_MIN_WEIGHT once mastered
# (empirically true here -- aggregate decisive win rate stayed >=0.77
# throughout), 0.20 each gives Defender+Counter only ~7.7% combined sampling
# share -- against ep_len_history's maxlen=50 window, that's ~1-4 games/
# window, statistically invisible in Progress/Episode_Length_Ticks_50 (no
# trend after 3000+ episodes at that setting). A follow-up isolated eval
# (current greedy policy vs ONLY Defender/Counter, bypassing PFSP sampling
# entirely) confirmed the escalation logic itself does work -- AvgTicks 240
# and 339 respectively vs the live aggregate's ~250-290, Counter games up to
# 1000 ticks -- so the fix here is exposure, not the opponent logic. 0.8
# retargets combined share to ~25% (2*0.8 / (96*0.05 + 2*0.8) under the same
# all-others-floored assumption) -- large enough to actually move the
# aggregate if the isolated-eval numbers hold at scale. Even at 25% exposure
# the games are individually bimodal (most of both isolated runs still ended
# fast, 110-230 ticks) -- so if THIS still doesn't move AvgTicks, the next
# suspect is the bots' purely-reactive posture (no proactive early-game
# stance), not sampling weight again.
DEFENSIVE_SCRIPTED_OPPONENTS = {"scripted:Defender", "scripted:Counter"}
DEFENSIVE_SCRIPTED_MIN_WEIGHT = 0.8

# Pulled live from the engine (see model.py's own identical pattern) so a
# board-geometry or channel-layout change on the C++ side propagates here
# automatically -- the scripted opponents below parse the raw observation
# vector directly (same one their neural counterparts already consume via
# get_observation_for_team), not through model.py, so they need their own
# copy of this layout math.
_N_CH = clash_royale_env.ClashRoyaleEnv.NUM_CHANNELS
_BOARD_H = clash_royale_env.ClashRoyaleEnv.BOARD_HEIGHT
_BOARD_W = clash_royale_env.ClashRoyaleEnv.BOARD_WIDTH
_SPATIAL_SIZE = _N_CH * _BOARD_H * _BOARD_W
_HAND_SIZE = clash_royale_env.ClashRoyaleEnv.HAND_SIZE
# The card head's extra column: "play nothing this step". Named because the
# placement entropy term has to be able to exclude it -- see mb_placed in the
# PPO update.
_NOOP_ACTION = _HAND_SIZE

_CARD_NAME_CACHE = {}


def _card_name(card_id):
    """Registry name for a card id, for diagnostic labels only.

    Evolutions share their base card's name verbatim (id 1 and id 128 are both
    "Archers"), so the id is appended -- a TensorBoard series that silently
    merged two cards would be worse than no series at all.
    """
    if card_id not in _CARD_NAME_CACHE:
        try:
            name = clash_royale_env.get_card_info(card_id)["name"]
        except Exception:  # noqa: BLE001 -- unknown id is a label problem, not fatal
            name = "card"
        _CARD_NAME_CACHE[card_id] = f"{name.replace(' ', '_')}_{card_id}"
    return _CARD_NAME_CACHE[card_id]

# --- Scenario injection (start-state distribution design) ------------------
# Industry precedent: reshaping the START-STATE distribution is how rare-but-
# critical situations get learned when normal play visits them too seldom for
# the credit-assignment horizon to connect cause and effect (robotics resets
# from curated states; AlphaGo trained on curated positions; "Backplay"). The
# problem this targets here: a win-condition (Hog/Giant/...) dropped on the
# bridge is a "defend in the next couple of seconds or lose the tower" moment,
# but in a full 3600-tick game the causal link between that drop and the tower
# loss ~40 ticks later is buried under a long, noisy GAE trace and is a rare
# event -- so the reflex never gets enough gradient. We fix that by STARTING a
# fraction of episodes already in that state so the net sees it constantly.
#
# Design choices that keep it from becoming a different game (the isolation
# failure mode): the opponent is NOT frozen -- team 1 keeps playing its normal
# PFSP policy on top of the injected threat; it's the real engine/board/towers;
# and the existing reward (win/loss + compute_shaping's HP/elixir-trade terms)
# already scores "defend efficiently + keep something alive to counter-push",
# so no bespoke scenario reward is needed. The one artificial edge -- an
# optional short truncation window (max_steps) that focuses each episode on the
# critical moment -- is handled with a proper value BOOTSTRAP (see the training
# loop's is_terminal/needs_boot split), never a terminal, so the critic doesn't
# learn a biased "the world ends here" value.
SCENARIO_INJECTION_PROB = 0.30

# Building-targeter win-conditions -- every id here confirmed against
# CardRegistry.h directly (not from memory) as Archetype::MeleeBuildingTargeter/
# RangedBuildingTargeter/a building with a persistent tower-damage role, i.e.
# guaranteed to beeline for a tower ignoring troops in its path, matching the
# scenario's own premise ("defend or lose the tower in the next few seconds").
# Miner (52) deliberately excluded despite being a real-game win condition --
# this engine registers him as plain Archetype::MeleeSquad (no building-
# targeter/dig-anywhere behavior implemented), so injecting him wouldn't
# actually exercise the "must answer a beelining threat" reflex this scenario
# is for. Goblin Barrel (109) / Graveyard (110) also excluded for now -- both
# are spell(...)-registered (PeriodicSpawnEffect), and inject_enemy's
# card->spawnEntity(...) path is only confirmed exercised (via Hog/Royal
# Giant) for a troop/building CardDefinition; using it for a spell-shaped one
# is unverified, not worth risking on a data-fill task.
_WIN_CONDITION_IDS = [
    15,  # Hog Rider
    18,  # Royal Giant
    45,  # Balloon
    2,   # Giant
    19,  # Golem
    81,  # Battle Ram
    82,  # Royal Hogs
    83,  # Wall Breakers
    84,  # Electro Giant
    87,  # Ram Rider
    88,  # Goblin Giant
    89,  # Skeleton Barrel
    91,  # Lava Hound
]
# Ranged units commonly played to escort/protect a win-condition push (the
# "supported" scenario's second spawn) -- confirmed RangedSquad/ranged-role
# troops, a mix of cheap chip support and real mid-fight damage.
_SUPPORT_IDS = [
    6,   # Musketeer
    1,   # Archers
    11,  # Wizard
    44,  # Baby Dragon
    63,  # Magic Archer
    36,  # Executioner
    20,  # Dart Goblin
]
# Real board coords for inject_enemy (team 1, low-y-bound), which bypasses
# isValidPlacement so an on-the-bridge spawn inside the river band is allowed.
#
# These are BOARD coordinates. Do not justify them from
# ClashEnv::extractObservationForTeam's `riverRow = 17` / x-band 3-4 & 13-14 --
# that is the OBSERVATION channel-8 marker, a wider visual hint painted for the
# network, and it is a different frame. perception/geometry.py warns against
# exactly this conflation. The board's own geometry (Board.h) is river
# [15.5, 17.5) with bridges at x = 4.0 and 14.0.
#
# The values below are nonetheless correct and must not be "corrected":
# y = 17.0 is inside the band, injectEnemy applies no clamp, and
# Board::getNextWaypoint classifies 17.0 as neither bank and routes to the
# bridge exit -- which is precisely the on-the-bridge spawn this wants.
_RIVER_Y = 17.0
_BRIDGE_LANES = [3.5, 13.5]


def _scenario_bridge_push(rng):
    """The exact case: one enemy win-condition on a random bridge, nothing
    else engineered. Short window -- the defense itself resolves in ~2-4 steps,
    the rest lets a counter-push start and get shaped-rewarded."""
    lane = rng.choice(_BRIDGE_LANES)
    return {
        "name": "bridge_push",
        "defensive": True,
        "spawns": [(int(rng.choice(_WIN_CONDITION_IDS)), lane, _RIVER_Y)],
        "max_steps": 15,
    }


def _scenario_bridge_push_supported(rng):
    """Win-condition + a ranged support just behind it (same lane) -- a tankier,
    two-part threat that a single cheap defender can't fully answer. Longer
    window for the bigger commitment."""
    lane = rng.choice(_BRIDGE_LANES)
    return {
        "name": "bridge_push_supported",
        "defensive": True,
        "spawns": [
            (int(rng.choice(_WIN_CONDITION_IDS)), lane, _RIVER_Y),
            (int(rng.choice(_SUPPORT_IDS)), lane, _RIVER_Y + 3.0),
        ],
        "max_steps": 25,
    }


# --- Fireball target practice ----------------------------------------------
# Added 2026-08-09 after measuring WHY the agent almost never casts Fireball.
# Over 120 trials per constructed situation, with Fireball affordable in 100%
# of them, the policy put only 0.03-0.11 probability on it against a 0.20
# uniform baseline -- and, critically, when it DID cast, the mean distance from
# the target centroid was 2.8 tiles in its own half and 7.7-11.5 tiles at an
# enemy tower, against a blast radius of 2.5.
#
# So the low usage is not timidity and not an exploration failure: it is an
# ACCURATE valuation of the agent's own aim. spell_value_shaping charges a full
# -1.00 * w for a cast that kills nothing, and a policy that cannot aim is
# correctly declining to pay it. Raising the cast incentive without fixing the
# aim would make it cast more and miss more -- which is the mechanism behind
# the recorded 97% -> 23% win-rate collapse when Fireball use was forced.
#
# Scenario injection is the right lever because it changes the START-STATE
# DISTRIBUTION, not the reward: it buys dense practice at the aiming problem
# without biasing the optimum. Exactly the argument that justified the
# bridge-push scenarios -- a high-value Fireball moment is rare and its credit
# is buried in a long GAE trace.
#
# Low-HP bodies only. Fireball does 689, so these die to one well-placed cast
# and a whiff is genuinely punished; Barbarians (691 HP) are deliberately NOT
# here, since surviving by 2 HP would teach that a perfect cast still failed.
_FIREBALL_SWARM_IDS = [
    41,  # Minions        (3 bodies)
    1,   # Archers        (2 bodies)
    64,  # Firecracker    (304 hp)
    6,   # Musketeer
]


def _scenario_fireball_swarm(rng):
    """A cheap swarm already inside our half -- the defensive value cast.

    Placed past the river and short of the Princess Towers, so it is a live
    threat the agent must answer THIS second rather than a distant one it can
    ignore. Several separate cards so the cluster is many bodies, which is what
    makes one Fireball a large positive elixir trade.
    """
    lane = rng.choice(_BRIDGE_LANES)
    n = int(rng.integers(3, 6))
    cx, cy = lane, 12.0
    spawns = []
    for _ in range(n):
        spawns.append((int(rng.choice(_FIREBALL_SWARM_IDS)),
                       float(cx + rng.uniform(-1.1, 1.1)),
                       float(cy + rng.uniform(-1.1, 1.1))))
    # Defensive: the swarm is inside OUR half, so "did not take a big hit"
    # is a real question with a real answer.
    return {"name": "fireball_swarm", "spawns": spawns, "max_steps": 12,
            "defensive": True}


def _scenario_fireball_tower_value(rng):
    """Enemy troops hugging their OWN princess tower -- the two-for-one cast.

    One Fireball centred here hits the troops and the tower together, which is
    the case spell_value_shaping was written to pay for and the one the agent
    currently misses by 7.7-11.5 tiles. Spawned just in front of the tower so
    both fall inside a single 2.5 radius.
    """
    tower_x = 4.0 if rng.random() < 0.5 else 14.0
    n = int(rng.integers(2, 5))
    cx, cy = tower_x, 25.6
    spawns = []
    for _ in range(n):
        spawns.append((int(rng.choice(_FIREBALL_SWARM_IDS)),
                       float(cx + rng.uniform(-0.9, 0.9)),
                       float(cy + rng.uniform(-0.9, 0.9))))
    # NOT defensive: the troops are at THEIR tower, nothing threatens us, so
    # "did not take a big hit" is true whatever the agent does -- including
    # doing nothing. Scoring it on that axis inflates ScenDef toward 1.0 and
    # says nothing about whether the cast was made or aimed.
    return {"name": "fireball_tower_value", "spawns": spawns, "max_steps": 15,
            "defensive": False}


# --- Giant: unmasking the win condition -------------------------------------
#
# The Giant is not undervalued, it is UNAFFORDABLE. Measured on the ep-17k
# checkpoint over 1,839 decision steps: the Giant is in hand on 80.6% of them
# but legal on only 4.7% (5.9% of in-hand), and when it IS legal the policy
# picks it at P = 0.157 against a uniform 0.200. It was sampled on 0.54% of
# steps, and 1 of 900 logged placements across the run was a Giant.
#
# That is a masked-slot problem, and it is the exact mechanism already on
# record for this deck: at 0.35 elixir per decision a 5-cost card is legal only
# after ~14 consecutive non-spending steps, so its slot is masked nearly every
# time it is checked and never accumulates gradient. No opponent, reward or
# entropy change reaches an action that is never sampled -- which is why this
# is a START-STATE change and not any of those.
#
# (builder_fn, weight). Extend freely -- offensive/punish/endgame scenarios
# drop in here with the same machinery. Set a scenario's "max_steps" to None
# to run it to the natural end of the game instead of a focused window.
#
# The two Fireball scenarios take half the injection budget, which is a
# REALLOCATION rather than an addition: ScenDef has been running at 0.99, so
# the bridge-push scenarios are saturated and no longer teaching the reflex
# they were added for. Their share drops from 100% to 50% of injected episodes
# (SCENARIO_INJECTION_PROB itself is unchanged at 0.30).
# giant_commit was REMOVED on 2026-08-19 and this is GAMEPLAY-AFFECTING.
# It required card id 2 (Giant) in our own hand, and DEFAULT_DECK became 2.6
# Hog Cycle on 2026-08-16 -- a deck that cannot contain it. So every time it was
# drawn it spent 24 futile `reset()` calls hunting for the card, fell through,
# and then ran a quiet board with banked elixir and no spawns: the one thing its
# own docstring said was NOT the point ("the whole point is an unmasked Giant
# slot"). At weight 1.0 of 6.0 it was diluting 17% of the injection budget into
# a no-op. Removing it redistributes that share back to the four live
# scenarios, so injected-episode composition changes and ScenDef/ScenOff are
# not comparable across this edit.
#
# Total is now 5.0: Fireball 60%, bridge 40%.
SCENARIOS = [
    (_scenario_bridge_push, 1.0),
    (_scenario_bridge_push_supported, 1.0),
    (_scenario_fireball_swarm, 1.5),
    (_scenario_fireball_tower_value, 1.5),
]


def sample_scenario(rng):
    builders, weights = zip(*SCENARIOS)
    weights = np.array(weights, dtype=np.float64)
    weights /= weights.sum()
    return builders[rng.choice(len(builders), p=weights)](rng)


def discover_historical_checkpoints(current_episode=None):
    """All ELIGIBLE *.pth files in HISTORICAL_CHECKPOINT_DIR, oldest-saved-first
    (mtime). Save order is the ordering signal, not filenames -- pipeline #1
    and this same script both drop snapshots into this one shared folder, on
    two unrelated episode-count scales, so parsing/comparing episode numbers
    out of the filename wouldn't give a meaningful weakest->strongest order.

    When current_episode is given, pipeline #2's OWN snapshots (filenames
    embed THIS script's own episodes_completed at save time -- same scale as
    current_episode, unlike pipeline #1's) younger than
    MIN_OPPONENT_AGE_EPISODES are excluded from eligibility -- see that
    constant's comment for why. Pipeline #1 snapshots are always eligible
    regardless of current_episode (their episode count lives on a wholly
    different, incomparable scale, and they're always from an earlier, weaker
    phase anyway)."""
    paths = glob.glob(os.path.join(HISTORICAL_CHECKPOINT_DIR, "*.pth"))
    if current_episode is not None:
        eligible = []
        for p in paths:
            match = re.search(r"_pipeline2_ep(\d+)\.pth$", os.path.basename(p))
            if match and current_episode - int(match.group(1)) < MIN_OPPONENT_AGE_EPISODES:
                continue
            eligible.append(p)
        paths = eligible
    return sorted(paths, key=os.path.getmtime)


def update_reference_roster(reference_roster, historical_pool):
    """Grows the FIXED evaluation roster (in place), up to
    REFERENCE_ROSTER_MAX_SIZE. Entries already in the roster are NEVER
    reordered or replaced -- see REFERENCE_ROSTER_ELO_STEP's comment on why the
    roster has to stay fixed for the Elo trend to mean anything.

    Anchors are now sampled EVENLY ACROSS the pool (weakest to strongest)
    instead of taking the first N oldest. The old behaviour filled the roster
    with the weakest checkpoints available at the time, and measurement showed
    what that costs: across 8 evaluations, 4 of the 5 anchors sat permanently at
    a score of 0.90-1.00. A saturated anchor contributes no information about
    whether the trainee improved -- it can only ever move DOWN, and near score
    1.0 the Elo formula turns a single lost game into a ~222-point swing. The
    roster was simultaneously blind to progress and extremely noisy.

    Spreading the picks means the roster spans a real difficulty range, so at
    least some anchors sit in the informative 0.3-0.7 band where score changes
    actually track skill."""
    known_paths = {entry["path"] for entry in reference_roster}
    # Built-in anchors go in FIRST and stay forever -- they are the only part of
    # the scale that is not a copy of the trainee, and they never saturate away
    # the way a beaten historical snapshot does.
    for descriptor, elo in BUILTIN_ANCHORS:
        if descriptor not in known_paths:
            reference_roster.append({"path": descriptor, "elo": elo})
            known_paths.add(descriptor)
    room = REFERENCE_ROSTER_MAX_SIZE - len(reference_roster)
    if room <= 0:
        return reference_roster
    candidates = [p for p in historical_pool if p not in known_paths]
    if not candidates:
        return reference_roster
    # Evenly spaced indices across the pool, oldest(weakest) -> newest(strongest).
    n = min(room, len(candidates))
    picks = [candidates[round(i * (len(candidates) - 1) / max(1, n - 1))] for i in range(n)] if n > 1 \
        else [candidates[-1]]
    for path in dict.fromkeys(picks):          # dedupe, preserve order
        if len(reference_roster) >= REFERENCE_ROSTER_MAX_SIZE:
            break
        next_elo = REFERENCE_ROSTER_BASE_ELO + REFERENCE_ROSTER_ELO_STEP * len(reference_roster)
        reference_roster.append({"path": path, "elo": next_elo})
        known_paths.add(path)
    return reference_roster


def evaluate_against_roster(net, device, roster, n_games=10):
    """Plays the CURRENT live policy, GREEDY (argmax masked card, argmax
    placement cell -- no sampling), against each fixed roster opponent for
    n_games each. No coordinate bounds are needed anymore: the discrete
    placement head emits a board-cell index that MicroRoyaleNet.cell_to_xy()
    converts to in-bounds coordinates by construction.

    The opponent itself is NOT forced greedy -- it plays its own normal
    (stochastic) game, since the point is "how does the trainee do against
    this opponent's real behavior," and only the trainee's OWN sampling noise
    needs removing from the measurement.

    Returns (agent_elo, per_opponent_stats). agent_elo is the average of the
    trainee's Elo implied by its score against each anchor's fixed Elo (see
    REFERENCE_ROSTER_ELO_STEP's comment: a comparable trend, not a calibrated
    rating). per_opponent_stats maps checkpoint path -> {wins, losses, draws,
    score}, useful for per-opponent diagnostics beyond the single aggregate
    number.
    """
    net.eval()
    per_opponent_stats = {}
    implied_elos = []
    try:
        for entry in roster:
            path, ref_elo = entry["path"], entry["elo"]
            # Two kinds of anchor need two different envs, and the distinction is
            # load-bearing: MicroRoyaleSelfPlayEnv drives team 1 through
            # ClashEnv::stepSelfPlay, which deliberately does NOT call
            # opponentTurn() -- so the C++ HeuristicOpponent never runs there.
            # Evaluating against it requires the ordinary MicroRoyaleEnv, whose
            # step() goes through game.step() and therefore does.
            if path.startswith("builtin:"):
                multiplier = float(path.split("@")[1])
                env = gym_wrapper.MicroRoyaleEnv()
                env.set_opponent_elixir_multiplier(multiplier)
            else:
                # Scenarios OFF for evaluation: Elo must measure clean-game
                # strength, not defense of an injected handicap.
                env = MicroRoyaleSelfPlayEnv({"scenarios_enabled": False})
                env.set_historical_opponent(path)
            wins = losses = draws = 0
            for _ in range(n_games):
                obs, _ = env.reset()
                hx = torch.zeros(1, 256).to(device)
                cx = torch.zeros(1, 256).to(device)
                done = False
                reward = 0.0
                while not done:
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        card_mask = net.affordability_mask(obs_t)
                        features, card_embeds, spatial_map = net.extract_features(obs_t)
                        (card_logits, _, _, _,
                         (hx, cx)) = net.step_lstm_and_card(features, (hx, cx), card_mask)
                        # argmax over MASKED logits -- an unaffordable card can
                        # never be the greedy pick, so the measurement reflects
                        # the policy best LEGAL move rather than a play the
                        # engine would silently drop.
                        card_idx_t = card_logits.argmax(dim=-1)
                        place_logits = net.placement_given_card(hx, card_embeds, card_idx_t, obs_t, spatial_map)
                        cell = place_logits.argmax(dim=-1)
                        x_t, y_t = net.cell_to_xy(cell)
                    card_idx = int(card_idx_t.item())
                    action = {
                        "card_index": np.array([card_idx]),
                        "target_x": np.array([x_t.item()]),
                        "target_y": np.array([y_t.item()]),
                        "activate_ability_slot1": np.array([0]),
                        "activate_ability_slot2": np.array([0]),
                    }
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                if reward > 0.5:
                    wins += 1
                elif reward < -0.5:
                    losses += 1
                else:
                    draws += 1
            score = (wins + 0.5 * draws) / n_games
            per_opponent_stats[path] = {"wins": wins, "losses": losses, "draws": draws, "score": score}
            score_clipped = min(max(score, 0.03), 0.97)
            implied_elos.append(ref_elo - 400.0 * math.log10(1.0 / score_clipped - 1.0))
    finally:
        net.train()
    agent_elo = float(np.mean(implied_elos)) if implied_elos else None
    return agent_elo, per_opponent_stats


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
        self.opponent_hx = torch.zeros(1, 256).to(self.device)
        self.opponent_cx = torch.zeros(1, 256).to(self.device)
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
        # operation, since train_selfplay_ppo() broadcasts before the first
        # envs.reset() call).
        self.pfsp_pool = []
        self.pfsp_stats = {}

        # Scenario injection (see SCENARIOS / sample_scenario). Independent
        # per-worker RNG so the vectorized envs don't all inject the same
        # scenario in lockstep. scenarios_enabled=False (used by the replay
        # env) keeps replays representative of full, un-engineered games.
        self.scenarios_enabled = env_config.get("scenarios_enabled", True)
        self.scenario_rng = np.random.default_rng()
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
            self.opponent_lane = random.choice(["left", "right"])

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
        """Broadcast from the main process (train_selfplay_ppo, via
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
        if not self.pfsp_pool:
            return
        def floor_for(p):
            if p in DEFENSIVE_SCRIPTED_OPPONENTS:
                return DEFENSIVE_SCRIPTED_MIN_WEIGHT
            if p.startswith("builtin:"):
                return BUILTIN_MIN_WEIGHT
            return PFSP_MIN_WEIGHT

        weights = np.array([
            max(floor_for(p), (1.0 - self.pfsp_stats.get(p, 0.5)) ** PFSP_EXPONENT)
            for p in self.pfsp_pool
        ], dtype=np.float64)
        weights /= weights.sum()
        chosen = self.pfsp_pool[np.random.choice(len(self.pfsp_pool), p=weights)]
        self._set_opponent(chosen)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._sample_pfsp_opponent()
        self.game.reset()

        self.scenario_active = None
        self.scenario_max_steps = None
        self.scenario_steps_taken = 0
        self.scenario_defensive = False
        if self.scenarios_enabled and self.scenario_rng.random() < SCENARIO_INJECTION_PROB:
            scenario = sample_scenario(self.scenario_rng)

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
        self.opponent_hx = torch.zeros(1, 256).to(self.device)
        self.opponent_cx = torch.zeros(1, 256).to(self.device)
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
            features1, card_embeds1, spatial_map1 = self.opponent_net.extract_features(obs1_t)
            (logits1, _, _, _,
             (self.opponent_hx, self.opponent_cx)) = self.opponent_net.step_lstm_and_card(
                features1, (self.opponent_hx, self.opponent_cx), card_mask1)
            card_idx1_t = Categorical(logits=logits1).sample()
            place_logits1 = self.opponent_net.placement_given_card(
                self.opponent_hx, card_embeds1, card_idx1_t, obs1_t, spatial_map1)
            cell1 = Categorical(logits=place_logits1).sample()
            x1_t, y1_t = self.opponent_net.cell_to_xy(cell1)
            card_idx1 = card_idx1_t.item()
            x1 = x1_t.item()
            y1 = y1_t.item()
        # Deck has no Champion -- see gym_wrapper.DEFAULT_DECK_ABILITY_SLOTS.
        return card_idx1, x1, y1, False, False

    def _scripted_opponent_action(self, obs1):
        """Team 1's move for one of SCRIPTED_OPPONENTS, computed purely from
        the same observation vector its neural counterpart already gets --
        no card-ID-specific logic anywhere here (only elixir/cost and enemy
        spatial position), so these behave sensibly regardless of the
        randomized deck set_scripted_opponent gave them. Never activates a
        Champion ability (no ability logic in these heuristics at all) --
        always (False, False) for the two slots.
        """
        spatial = obs1[:_SPATIAL_SIZE].reshape(_N_CH, _BOARD_H, _BOARD_W)
        scalar = obs1[_SPATIAL_SIZE:]
        elixir = float(scalar[0])
        costs = scalar[1:1 + _HAND_SIZE]
        # cost <= 0 marks an empty/invalid hand slot (see ClashEnv::
        # extractObservationForTeam: card ? card->cost/10.0f : 0.0f) --
        # never a real, free card.
        affordable = [i for i in range(_HAND_SIZE) if costs[i] > 0.0 and costs[i] <= elixir + 1e-6]

        def lane_x():
            return self.MAX_X * (0.2 if self.opponent_lane == "left" else 0.8)

        def find_incursion():
            """Returns (x, y, is_heavy) for the most urgent enemy incursion
            anywhere on the board, or None. Channels 4-6 = enemy (team 0)
            melee/ranged/tank troops, from THIS observer's own mirrored
            point of view -- see model.py's identical channel-layout
            comment.

            Scans the WHOLE board, not just this observer's own half.
            Previously scanned only rows up to int(self.MAX_Y) (the
            enforced own-half placement bound, same one everything else in
            this file pulls live) -- meaning Defender/Counter only ever
            noticed a threat once it had already crossed the river into
            their own territory, often most of the way to the tower by the
            time a response spawned and reached it. An isolated eval of the
            live policy vs. these two scripted bots (bypassing PFSP
            sampling) found 55-70% of even head-to-head games still ended
            in an early blowout (110-230 ticks) regardless of exposure --
            i.e. sampling weight (see DEFENSIVE_SCRIPTED_MIN_WEIGHT) wasn't
            the bottleneck, reaction latency was. Detecting the threat the
            moment it's placed, anywhere, lets escalation (is_heavy) and
            the response fire as early as possible.

            is_heavy: True iff the incursion includes a channel-6 (building-
            targeter/tank archetype) unit -- the same archetype category
            _WIN_CONDITION_IDS/scenario injection already treats as "the
            real threat" (see that constant's own comment: Hog/Giant/
            Golem/Balloon/... are all this archetype). A plain melee/ranged
            squad troop (channels 4-5) doesn't set this even if it's also
            in range -- the distinction is what Defender escalates on.
            """
            river_row = int(self.MAX_Y)

            def clamp_y(y):
                # Placement is only ever legal within our own half (see
                # self.MAX_Y's own docstring) -- for a threat still crossing
                # from the enemy's half this meets it right at the bridge,
                # the earliest legal interception point, instead of only
                # reacting once it's already deep in our own territory (the
                # old own-half-only scan's implicit behavior).
                return min(float(y), float(river_row))

            tank_nz = np.nonzero(spatial[6])
            if tank_nz[0].size > 0:
                ys, xs = tank_nz
                deepest = int(np.argmin(ys))  # smallest y = closest to team 1's own tower = most urgent
                return float(xs[deepest]), clamp_y(ys[deepest]), True
            other_nz = np.nonzero(spatial[4] + spatial[5])
            if other_nz[0].size == 0:
                return None
            ys, xs = other_nz
            deepest = int(np.argmin(ys))
            return float(xs[deepest]), clamp_y(ys[deepest]), False

        NO_OP = (_HAND_SIZE, 0.0, 0.0, False, False)
        kind = self.opponent_kind

        if kind == "Rusher":
            if not affordable:
                return NO_OP
            slot = max(affordable, key=lambda i: costs[i])
            return slot, lane_x(), self.MAX_Y, False, False

        if kind == "Cycler":
            if not affordable:
                return NO_OP
            slot = min(affordable, key=lambda i: costs[i])
            return slot, self.MAX_X * 0.5, self.MAX_Y * 0.5, False, False

        incursion = find_incursion()

        if kind == "Defender":
            if incursion is None or not affordable:
                return NO_OP
            x, y, is_heavy = incursion
            # Escalate to the strongest affordable answer against a real
            # win-condition-style threat (channel 6 -- see find_incursion's
            # own comment); an ordinary squad troop still just gets the
            # cheapest efficient trade, same as before. A Defender that
            # always reaches for its cheapest card regardless of what's
            # actually attacking loses to any sufficiently strong rush no
            # matter how often it's sampled -- this is what actually makes
            # facing it more often (see DEFENSIVE_SCRIPTED_MIN_WEIGHT) worth
            # anything.
            slot = max(affordable, key=lambda i: costs[i]) if is_heavy else min(affordable, key=lambda i: costs[i])
            return slot, x, y, False, False

        if kind == "Counter":
            if not affordable:
                return NO_OP
            if incursion is not None:
                slot = max(affordable, key=lambda i: costs[i])
                x, y, _is_heavy = incursion
                return slot, x, y, False, False
            slot = max(affordable, key=lambda i: costs[i])
            return slot, lane_x(), self.MAX_Y, False, False

        return NO_OP

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
        # on purpose -- lets compute_shaping() from train.py be reused as-is.
        info = {
            "elixir": self.game.get_elixir(),
            "hand": self.game.get_hand(),
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
            # Supervision target for the auxiliary elixir head -- see the
            # identical key in gym_wrapper.MicroRoyaleEnv.step()'s info dict
            # and MicroRoyaleNet.predict_opp_elixir. Hidden information, so it
            # travels through info and never through the observation.
            "opp_elixir": self.game.get_elixir_for_team(1),
            "champion_ability_slot1_ready": self.game.is_champion_ability_ready(0, 1),
            "champion_ability_slot2_ready": self.game.is_champion_ability_ready(0, 2),
            # 1.0 while the current episode started from an injected scenario --
            # lets the training loop score scenario defenses separately from
            # normal-matchup win/loss (see Scenario/Defense_Success_Rate).
            "is_scenario": 1.0 if self.scenario_active is not None else 0.0,
            # Split out because ScenDef means "did the agent survive a threat",
            # and two scenarios put no threat in our half at all:
            # fireball_tower_value spawns at the ENEMY tower and giant_commit
            # spawns nothing. In those, "did not take a big hit" is true no
            # matter what the agent does -- including nothing -- so counting
            # them drags ScenDef toward 1.0 and hides real defensive failures.
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


def make_env():
    def _init():
        return MicroRoyaleSelfPlayEnv()
    return _init


def train_selfplay_ppo():
    os.makedirs("replays", exist_ok=True)
    os.makedirs(HISTORICAL_CHECKPOINT_DIR, exist_ok=True)
    log_dir = "runs/clash_royale_selfplay"

    # NOTE: historical_pool is discovered further below, only once
    # episodes_completed is resolved (either 0 on a from-scratch run or
    # restored from a resumed checkpoint) -- MIN_OPPONENT_AGE_EPISODES
    # filtering needs that value to know what's "too young" to be eligible.
    # Overridable for the same reason train.py's is: an experiment that has to
    # share the box with a second arm needs to say so without editing code
    # between arms. Leave unset for real runs.
    num_envs = int(os.environ.get("CLASH_NUM_ENVS", 8))
    print(f"Initializing {num_envs} self-play environments...")
    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(num_envs)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MicroRoyaleNet(num_ability_slots=DEFAULT_DECK_ABILITY_SLOTS).to(device)
    optimizer = optim.Adam(net.parameters(), lr=3e-4)
    # Same guard as train.py -- this loop no longer samples Champion abilities
    # at all, so a Champion in the deck must fail loudly instead of silently
    # never being used.
    if DEFAULT_DECK_ABILITY_SLOTS > 0:
        raise NotImplementedError(
            "DEFAULT_DECK contains a Champion (DEFAULT_DECK_ABILITY_SLOTS > 0), but this "
            "training loop ability sampling was removed when the deck had none. Restore "
            "the ability_dist sampling/buffering/log-prob branches before training this deck.")

    # Same PPO hyperparameters as train.py -- same architecture and algorithm,
    # only the opponent-selection mechanic differs, so there's no reason to
    # re-derive these from scratch.
    gamma = 0.99
    gae_lambda = 0.9
    eps_clip = 0.2
    update_timestep = 500
    # Truncated BPTT -- same values and same reasoning as train.py; see the long
    # bptt_chunk comment there for the two measurements that motivated it
    # (Clip_Fraction 0.0000 across 228 updates, and ~79% of epoch time spent in
    # the backward pass of a single 500-long chain).
    bptt_chunk = 25
    ppo_epochs = 4
    num_minibatches = 8
    max_grad_norm = 0.5
    assert update_timestep % bptt_chunk == 0, \
        "update_timestep must be divisible by bptt_chunk so every chunk is full-length"

    # Rescaled for the discrete placement head, and floor/decay calibrated from
    # the measured entropy-vs-coefficient data points -- same values and same
    # reasoning as train.py; see the long comment there.
    initial_entropy_coef = 0.05
    min_entropy_coef = 0.02
    entropy_decay_rate = 0.9997
    # --- Per-head entropy (normalized) ------------------------------------
    # The entropy bonus used to be a SINGLE coefficient on the raw SUM
    # H_card + H_placement. Those two heads have very different scales --
    # log(5)=1.61 vs log(288)=5.66 -- so the optimizer could satisfy almost the
    # whole entropy term through placement alone and let card selection collapse
    # for free. Measured at episode 20,245 of the run this replaces:
    #
    #   placement entropy 1.472 / 5.663  (26% of max)
    #   card      entropy 0.007 / 1.609  (0.4% of max -- effectively a constant)
    #
    # and behaviorally: hand slot 3 never chosen once, and only 5 of the deck's
    # 8 cards ever played across 456 decisions -- never Giant (the win
    # condition), Cannon (the only defensive building), or Fireball.
    #
    # Note what the real asymmetry was: the raw sum already applied the SAME
    # weight to both heads (d(bonus)/dH = 1 for each). Card selection did not
    # collapse because it was under-weighted in the loss -- it collapsed
    # because WHICH CARD you play moves the return far more than which exact
    # cell you drop it on, so the policy gradient drives the card head toward
    # determinism much harder, and equal counter-pressure simply lost.
    #
    # Fix: divide each head's entropy by its OWN maximum, so both terms live in
    # [0,1] and a coefficient means the same thing for each, THEN weight them
    # explicitly by how badly each head actually collapsed. Normalizing alone
    # would be a regression for placement -- it multiplies the absolute
    # placement pressure by 1/log(288)=0.18 -- which is why the scales below
    # are not both 1.0.
    #
    # Normalization is by the head's FULL action count, not by the per-step
    # legal-action count (which the affordability mask makes vary between 2 and
    # 5). Per-step normalization would divide by log(2)=0.69 on the tightest
    # steps and blow those samples up; the point here is rebalancing the two
    # heads against each other, which the fixed divisor already achieves.
    #
    # Scales chosen so that BOTH heads end up with more absolute entropy
    # pressure than the 0.006 floor that measurably failed, weighted by how far
    # each had actually collapsed (card to 0.4% of its maximum, placement to
    # 26%). Effective per-head coefficient at the floor is
    # min_entropy_coef * SCALE / log(n):
    #
    #   card:      0.02 * 4.0 / 1.609 = 0.0497   (8.3x the 0.006 that failed)
    # Raised 2.0 -> 4.0 after measurement: at 2.0 the card head settled at
    # Policy/Entropy_Card_Frac ~= 0.10 while placement sat at ~0.75, still a
    # 7.5x imbalance and only ~1.2 effective card choices. Cross-stage probes
    # confirmed the card head keeps narrowing (H_card 0.448 -> 0.202 -> 0.121
    # over stages 0/1/2 measured against a FIXED 1.0x opponent, so it is not
    # an artifact of the opponent getting harder).
    #   placement: 0.02 * 3.0 / 5.663 = 0.0106   (1.8x)
    #
    # i.e. card gets roughly twice the pressure placement does, rather than the
    # equal weighting that let it collapse. These are the two knobs to turn if
    # Policy/Entropy_Card_Frac or Policy/Entropy_Placement_Frac (logged below)
    # heads toward zero again -- raise the corresponding scale.
    LOG_N_CARD = math.log(net.hand_size + 1)
    LOG_N_PLACEMENT = math.log(net.placement_cells)
    # Seeded at the last hand-tuned effective values so the controller starts
    # from a known-reasonable point rather than hunting from zero.
    ent_coef_card = 0.05
    ent_coef_place = 0.06
    # Overwritten from the checkpoint on a resume -- see the block below. These
    # are the never-resumed defaults, used only on a genuinely fresh start.
    resumed_exploiter_burst_ep = None
    resumed_exploiter_burst_index = 0
    # --- Adaptive per-head entropy coefficients ---------------------------
    # Replaces a hand-tuned fixed coefficient per head. Two runs showed why
    # fixed values do not work here: the heads are COUPLED, so correcting one
    # breaks the other.
    #
    #   card scale 2.0 -> card head collapsed to 10% of its max entropy
    #   card scale 4.0 -> card recovered to ~35%, but placement fell from
    #                     H=4.38 to H=2.63 at the same stage (40 -> 20 cells,
    #                     top-5 share 36% -> 70%, left lane 33% -> 13%)
    #
    # Instead of picking coefficients, pick the ENTROPY LEVEL each head should
    # hold and let a controller find the coefficient. Same idea as SAC's
    # automatic temperature tuning: the coefficient is not a hyperparameter to
    # guess, it is whatever value happens to sustain the target.
    #
    # Multiplicative control on the normalized entropy fraction:
    #     coef *= exp(rate * (target - measured))
    # measured below target -> coefficient rises -> more exploration pressure.
    # Updated once per PPO update (~27s of wall clock), so it moves far slower
    # than training and cannot fight the policy gradient step-for-step.
    #
    # Targets come from the measured healthy/unhealthy bands across runs D/E:
    #   card      0.10 collapsed (5/8 cards), 0.35 kept 6/8 -> target 0.35
    #   placement 0.46 too narrow (20 cells), 0.77 was wide  -> target 0.65
    # ANNEALED, not fixed. Measured pathology this replaces: with the target
    # pinned at 0.65 for the whole run, the placement entropy coefficient rose
    # monotonically (0.1286 -> 0.1476 across 50,000 self-play episodes) --
    # i.e. the policy was trying to sharpen its placement the entire time and
    # the controller kept forcing it back open. 0.65 * log(612) = 4.17 nats is
    # a spread over ~65 board cells; a strong player commits to a tile.
    #
    # Note the existing comment on ENTROPY_ADAPT_RATE below already concluded
    # "placement spread is a poor proxy for skill and should not be optimized
    # directly" -- and yet a fixed target optimizes exactly that, forever.
    # Annealing resolves that contradiction: wide while the policy is still
    # discovering where things go, tight once it is refining.
    #
    # 0.25 * log(612) = 1.60 nats ~= 5 effective cells: committed, but not a
    # collapsed point mass. Deliberately NOT annealed to 0 -- some placement
    # noise is genuinely correct in a game with a live opponent.
    #
    # The CARD target is deliberately left FIXED. The measured failure mode
    # there is the opposite one: card entropy 0.10 collapsed the policy to
    # 5 of 8 cards. Narrowing card choice is the known danger, so only the
    # placement head -- where the evidence says the controller is fighting the
    # policy -- gets annealed.
    ENTROPY_TARGET_CARD = 0.35
    ENTROPY_TARGET_PLACEMENT_START = 0.50
    ENTROPY_TARGET_PLACEMENT_FINAL = 0.25
    # Sized to one long run on this machine: the last full pipeline reached
    # ~60k episodes in phase 1 and ~50k in phase 2. Past the horizon the
    # target simply stays at FINAL.
    ENTROPY_ANNEAL_EPISODES = 60000

    def placement_entropy_target(eps_done):
        frac = min(1.0, max(0.0, eps_done / ENTROPY_ANNEAL_EPISODES))
        return (ENTROPY_TARGET_PLACEMENT_START
                + frac * (ENTROPY_TARGET_PLACEMENT_FINAL - ENTROPY_TARGET_PLACEMENT_START))
    # Reverted 0.15 -> 0.5 after a matched-depth measurement contradicted the
    # earlier reasoning. Lowering the gain DID smooth the controller (mean
    # |entropy - target| fell, placement no longer free-fell), but the resulting
    # policy measured WORSE at stage 3: 81.7% [74-88] vs scripted opponents
    # against 96.7% [92-99] for the high-gain run, CIs not overlapping.
    #
    # The likely mechanism, though unproven: the large swings the high gain
    # produces act as periodic exploration re-boosts that shake the policy out
    # of local optima, and smoothing them away removes that. Note also that the
    # high-gain run had the WORST mid-training placement spread and still the
    # best final strength -- i.e. placement spread is a poor proxy for skill and
    # should not be optimized directly.
    #
    # CAVEAT: one run per configuration. RL run-to-run variance is large, so
    # this ordering may not survive replication with multiple seeds.
    ENTROPY_ADAPT_RATE = 0.5
    # ...and the placement head gets its OWN, much lower gain, because on
    # 2026-08-11 the 0.5 above drove it into windup and dissolved the policy.
    #
    # The reversion argument recorded above is void for THIS head. It was
    # measured against a placement-entropy signal that was ~85% no-op steps
    # (see mb_placed) -- a smooth, nearly-constant quantity sitting at 0.85-0.97
    # that the controller barely had to act on. Correcting the signal made it
    # far more responsive, and a gain tuned on the numbed version overreacted:
    # from measured 0.09 against target 0.25, each update multiplied the
    # coefficient by exp(0.5 * 0.16) = 1.083, compounding ~55x over 50 updates.
    # It went 0.0100 -> 0.433, blew measured entropy past target to 0.671, and
    # six of eight cards reached 0.93-1.00 of MAXIMUM placement entropy -- i.e.
    # uniform over every legal cell. ROI fell 0.96 -> 0.87 (r = -0.70 over 2,620
    # episodes) and win rate 0.67 -> 0.51 (r = -0.68). Same dissolution the
    # exploiter suffered at 86.7% of max; this reached ~95%.
    #
    # The general lesson, worth more than the number: FIXING A SENSOR
    # INVALIDATES ANY GAIN TUNED AGAINST THE BROKEN ONE. The two changes have
    # to ship together.
    ENTROPY_ADAPT_RATE_PLACEMENT = 0.10
    # Hard cap on how far one update may move a coefficient, independent of
    # gain or error size. The gain change above addresses the cause; this
    # addresses the failure MODE, so no future retune of a target or a
    # measurement can compound into a 40x excursion again. Binds only on large
    # excursions -- exactly when it should.
    ENTROPY_COEF_STEP_MAX = 0.10
    # Bounds keep a runaway controller from either silencing the entropy term
    # or drowning the policy gradient if a target is briefly unreachable.
    # Raised 0.002 -> 0.01. Healthy measured coefficients are 0.05-0.22, so
    # 0.002 was not a floor but an off switch: once there, the entropy term
    # stopped opposing the policy gradient at all and the head was free to
    # collapse until the controller noticed. 0.01 keeps real pressure at all
    # times, so the controller corrects from a slowed slide rather than a
    # free-fall.
    ENTROPY_COEF_FLOOR = 0.01
    ENTROPY_COEF_CEIL = 0.5
    # Separate, much lower ceiling for the placement head only. 0.433 already
    # dissolved the policy, so 0.5 was never a safety net for this head.
    #
    # Deliberately NOT applied to the card head, which shares the name above.
    # That controller is doing the right thing: card entropy measured 0.084 of
    # max against a 0.35 target -- a real collapse to ~5 of 8 cards -- and it
    # was operating at 0.24-0.43 to fight it. A global cut to 0.20 would
    # throttle the one controller that is working. The dissolution was
    # placement-only, so the cap is placement-only.
    ENTROPY_COEF_CEIL_PLACEMENT = 0.20

    # --- Value-clip range, scaled to the RETURN distribution ---------------
    # This used to reuse eps_clip (0.2) directly. That number is a bound on the
    # POLICY's probability RATIO -- a dimensionless quantity -- and reusing it
    # as an ABSOLUTE bound on how far the critic may move is a unit mismatch:
    # it only makes sense relative to how large returns actually are.
    #
    # Measured on a live 500-step rollout at episode 14,666 (stage 3):
    #   GAE return   std 0.530
    #   |return - V| median 0.181, p90 0.539
    #   46.1% of samples needed the critic to move MORE than 0.2
    #
    # i.e. on nearly half the batch the critic was forbidden from correcting its
    # own error in one update, no matter how many optimizer steps it got.
    #
    # Expressed as a fraction of the batch's own return spread instead, so it
    # stays correctly scaled if the reward shaping is ever retuned (which would
    # otherwise silently make this either crippling or a no-op). 1.0 std covers
    # roughly the p85 correction. Floored at eps_clip so it can never become
    # TIGHTER than the old behavior.
    VF_CLIP_STD_FRAC = 1.0

    # Pulled live from the engine (a throwaway bare env is enough -- these
    # don't depend on which deck is used, and this avoids building a whole
    # extra MicroRoyaleSelfPlayEnv, opponent net included, just to read two
    # floats) instead of a class-level hardcoded copy that could silently
    # drift if board geometry ever changes.
    # No coordinate bounds are needed in this scope anymore -- the discrete
    # placement head emits a board-cell index and MicroRoyaleNet.cell_to_xy()
    # converts it to coordinates already inside the engine enforced bounds.
    # MicroRoyaleSelfPlayEnv keeps its own self.MAX_X/self.MAX_Y -- those are
    # still genuinely used, by the scripted opponents and the action_space.

    # Episode at which the entropy decay schedule last reset -- renamed from
    # the old ladder's "stage_start_episode" now that there's no per-opponent
    # "stage" concept under PFSP; only reset by the two re-boost triggers
    # below (draw-rate stall, losing-trend stale-plateau).
    entropy_reboost_episode = 0
    last_stall_reboost_episode = 0  # episodes_completed value at the last entropy re-boost
    last_eval_ep = 0  # episodes_completed value at the last fixed-roster evaluation
    episodes_completed = 0
    reference_roster = []  # list of {"path":, "elo":}, see update_reference_roster()
    outcome_history = deque(maxlen=100)
    outcome_history_long = deque(maxlen=500)

    full_resume = False
    resuming = os.path.exists(WEIGHT_PATH)
    if resuming:
        checkpoint = torch.load(WEIGHT_PATH, map_location=device, weights_only=False)
        clean_load = load_state_dict_flexible(net, checkpoint["model"], f"pipeline2 resume ({WEIGHT_PATH})")
        if clean_load:
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            print("Optimizer state NOT restored (architecture mismatch above) -- "
                  "starting the optimizer fresh; network weights were still warm-started where shapes matched.")
        episodes_completed = checkpoint["episodes_completed"]
        if not clean_load:
            # Part of the network just got reinitialized (architecture change --
            # see load_state_dict_flexible's warm-start above, e.g. right now:
            # placement_head's card-conditioning). Entropy is almost certainly
            # floored this deep into training, so without this the freshly-
            # random part would settle into another near-deterministic
            # "averaged" policy before ever exploring enough to discover it
            # can now behave differently per card -- same reasoning as every
            # curriculum-stage transition elsewhere in this project resetting
            # the entropy clock, just triggered by an architecture change
            # instead of a harder opponent.
            entropy_reboost_episode = episodes_completed
            print(f">>> Architecture changed on resume -- forcing a fresh entropy "
                  f"re-boost from episode {episodes_completed} so the reinitialized "
                  "part actually gets explored, not just re-converged under floored entropy.")
        else:
            # .get() with a fallback to the OLD ladder-era field name, then to
            # episodes_completed: lets a checkpoint saved before this PFSP
            # rewrite still resume sanely instead of KeyError-ing.
            entropy_reboost_episode = checkpoint.get(
                "entropy_reboost_episode", checkpoint.get("stage_start_episode", episodes_completed))
        last_stall_reboost_episode = checkpoint.get("last_stall_reboost_episode", episodes_completed)
        last_eval_ep = checkpoint.get("last_eval_ep", episodes_completed)
        reference_roster = checkpoint.get("reference_roster", [])
        outcome_history = deque(checkpoint["outcome_history"], maxlen=100)
        # The adaptive entropy controller's converged state. train.py has always
        # persisted these; this file did not, so every phase-2 resume silently
        # threw the controller back to 0.05/0.06 and spent thousands of episodes
        # walking back. Observed on the 2026-07-30 restart: placement reset from
        # a converged 0.0132 to 0.06, and took ~5,600 episodes to return to
        # 0.0205. Nothing warned; the only visible trace was the EntCoef field
        # in the console line jumping back to its initial value.
        ent_coef_card = checkpoint.get("ent_coef_card", ent_coef_card)
        ent_coef_place = checkpoint.get("ent_coef_place", ent_coef_place)
        # Exploiter schedule. Falls back to episodes_completed rather than None,
        # because None means "never burst" and, past
        # EXPLOITER_FIRST_BURST_EPISODE, should_run_burst() reads that as "due
        # now" -- so a legacy checkpoint would fire an unscheduled 75-minute
        # burst immediately on every resume.
        # NOTE the fallback above does NOT fire for the checkpoints this run
        # actually has. dict.get returns the default only when the key is
        # ABSENT, and every checkpoint written while the exploiter was disabled
        # carries the key PRESENT with value None -- verified by reading one.
        # So resumed_exploiter_burst_ep is None here, and None is meaningful:
        # should_run_burst reads it as "never bursted, due now", which is
        # correct for an exploiter that has genuinely never run.
        #
        # It is only the MESSAGE that could not cope. Enabling the exploiter on
        # 2026-08-09 took this branch for the first time and crashed the resume
        # outright with `NoneType + int`, before a single episode ran -- a
        # latent fault that was unreachable for as long as the feature was off.
        resumed_exploiter_burst_ep = checkpoint.get(
            "last_exploiter_burst_episode", episodes_completed)
        resumed_exploiter_burst_index = checkpoint.get("exploiter_burst_index", 0)
        full_resume = True
        if not exploiter_mod.EXPLOITER_ENABLED:
            burst_note = "exploiter disabled (see EXPLOITER_ENABLED)"
        elif resumed_exploiter_burst_ep is None:
            burst_note = "exploiter enabled, no burst on record -- first burst is due now"
        else:
            burst_note = ("next exploiter burst at ep "
                          f"{resumed_exploiter_burst_ep + exploiter_mod.EXPLOITER_CYCLE_EPISODES}")
        print(f"Resumed pipeline #2 (PFSP) from {WEIGHT_PATH}: episode {episodes_completed}, "
              f"reference roster size {len(reference_roster)}, "
              f"ent coef c/p {ent_coef_card:.4f}/{ent_coef_place:.4f}, "
              + burst_note)
    elif os.path.exists(BOOTSTRAP_FROM_PATH):
        bootstrap = torch.load(BOOTSTRAP_FROM_PATH, map_location=device, weights_only=False)
        state_dict = bootstrap["model"] if isinstance(bootstrap, dict) and "model" in bootstrap else bootstrap
        load_state_dict_flexible(net, state_dict, f"pipeline2 bootstrap from pipeline1 ({BOOTSTRAP_FROM_PATH})")
        print(f"Seeded pipeline #2's trainee from pipeline #1's {BOOTSTRAP_FROM_PATH} "
              "(bare weights only -- pipeline #2 keeps its own separate episode count from here).")
    else:
        raise RuntimeError(
            f"Neither {WEIGHT_PATH} nor {BOOTSTRAP_FROM_PATH} exists -- pipeline #2 needs "
            "pipeline #1's finished bot to start from.")

    historical_pool = discover_historical_checkpoints(episodes_completed)
    if not historical_pool:
        raise RuntimeError(
            f"No historical snapshot in {HISTORICAL_CHECKPOINT_DIR}/ is at least "
            f"MIN_OPPONENT_AGE_EPISODES ({MIN_OPPONENT_AGE_EPISODES}) episodes older than "
            f"the current trainee (episode {episodes_completed}) -- pipeline #1 (train.py) "
            "needs to have run long enough to have saved at least one eligible snapshot "
            "(every HISTORICAL_CHECKPOINT_INTERVAL_EPISODES episodes) before pipeline #2 "
            "has anything old enough to play against.")

    # SCRIPTED_OPPONENTS are permanent PFSP-pool members (see that constant's
    # comment) -- broadcast for TRAINING sampling only, appended on top of
    # historical_pool rather than mixed into the variable itself, since
    # historical_pool alone also feeds update_reference_roster/evaluate_
    # against_roster below, both of which torch.load() every entry they're
    # given (a "scripted:X" tag would crash there, not just misbehave).
    envs.call("refresh_pfsp_pool", historical_pool + SCRIPTED_OPPONENTS + BUILTIN_TRAINING_OPPONENTS)
    print(f"PFSP pool initialized with {len(historical_pool)} historical + "
          f"{len(SCRIPTED_OPPONENTS)} scripted + "
          f"{len(BUILTIN_TRAINING_OPPONENTS)} builtin-heuristic opponent(s) "
          f"[{', '.join(BUILTIN_TRAINING_OPPONENTS)} at min weight {BUILTIN_MIN_WEIGHT}].")

    if not full_resume and os.path.exists(log_dir):
        import shutil
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    obs_buffer, card_actions_buffer, placement_actions_buffer = [], [], []
    logprobs_buffer, values_buffer, rewards_buffer = [], [], []
    masks_buffer, valid_buffer = [], []
    # Ground-truth opponent elixir per step -- supervision for the auxiliary
    # head only, never an input. See AUX_ELIXIR_COEF.
    aux_elixir_buffer = []
    # See train.py identical decision_buffer comment: 1 only where the
    # affordability mask left a real choice, used to normalize the actor and
    # entropy terms so forced no-ops do not shrink the effective step size.
    decision_buffer = []
    # LSTM state entering each timestep -- the resume point for whichever
    # truncated-BPTT chunk starts there.
    hx_in_buffer, cx_in_buffer = [], []
    # Correct-bootstrap GAE bookkeeping (see the is_terminal/needs_boot split in
    # the rollout): boot_nonterminal = 0 only on TRUE terminals (bootstrap
    # otherwise), trunc_flag marks steps whose next-state value must come from
    # the captured trunc_boot rather than the next (already-reset) episode's V.
    boot_nonterminal_buffer, trunc_flag_buffer, trunc_boot_buffer = [], [], []
    # --- advisor-targeted placement coverage --------------------------------
    # The coverage SLOT is drawn here, once per timestep, instead of inside the
    # PPO epoch loop where it used to be. Two reasons, and the first is the one
    # that matters: an advisor target has to be computed against the observation
    # the slot was drawn on, which only exists during the rollout. The second is
    # that a slot resampled every epoch meant the coverage entropy fed to the
    # controller was averaged over a shifting set of cards.
    coverage_slot_buffer, coverage_target_buffer, coverage_has_buffer = [], [], []
    advisor_legal = AT.build_legal_table(net) if AT.enabled() else {}

    reward_history = deque(maxlen=50)
    shaping_history = deque(maxlen=50)
    ep_len_history = deque(maxlen=50)
    ally_bldg_end_history = deque(maxlen=50)
    enemy_bldg_end_history = deque(maxlen=50)
    # Scenario defenses scored separately from normal-matchup win/loss: success
    # = the injected episode did NOT end in a tower/game loss (survived the
    # threat, or truncated out of the focused window still alive).
    scenario_success_history = deque(maxlen=200)
    # Non-defensive scenarios (fireball_tower_value, giant_commit) tracked
    # separately rather than discarded: 'did this episode avoid a big
    # negative' is still worth watching there, it just is not DEFENCE and
    # must not be averaged into ScenDef.
    scenario_other_history = deque(maxlen=200)
    # --- Live strategy diagnostics -------------------------------------
    # These track the two specific degenerate behaviours measured in phase 1,
    # which win/loss alone cannot see:
    #   * the greedy policy had abandoned 2 of its 8 cards entirely (never the
    #     win condition, never the spell) and won purely by cheap defence
    #   * it never accumulated elixir -- mean 3.6/10 at the moment it acted --
    #     so it could never afford a real push
    # Both are fatal against a competent opponent and invisible in the win rate
    # against a weak one, so they need their own live readout.
    cards_per_game_history = deque(maxlen=50)     # distinct cards actually played per game
    elixir_at_play_history = deque(maxlen=50)     # mean elixir held at the moment of a play
    plays_per_game_history = deque(maxlen=50)     # how many cards get played at all
    # Per-env accumulators for the episode currently in flight.
    ep_card_ids = [set() for _ in range(num_envs)]
    ep_elixir_at_play = [[] for _ in range(num_envs)]
    ep_play_count = np.zeros(num_envs, dtype=np.int64)
    # Placements landing in the forward band (y >= FORWARD_ROW_Y, i.e. the
    # bridge-adjacent rows rather than the tower pocket). Reported as a rate so
    # it does not move with match length.
    #
    # Read this ONLY alongside ROI. Raising placement entropy spreads the
    # marginal toward the middle of the legal region, which lifts this number
    # on its own -- it is evidence of aggression only when return per elixir
    # holds up at the same time.
    FORWARD_ROW_Y = 12
    ep_forward_placements = np.zeros(num_envs, dtype=np.int64)
    # Realized elixir economy per deck card, summed over recent episodes, so
    # ROI is a ratio of totals rather than a mean of noisy per-episode ratios
    # (episodes where a card was never played would otherwise contribute 0/0).
    econ_killed = deque(maxlen=50)
    econ_spent = deque(maxlen=50)
    forward_rate_history = deque(maxlen=50)
    tower_dmg_rate_history = deque(maxlen=50)

    obs, _ = envs.reset()
    prev_stats = None
    prev_dones = np.zeros(num_envs, dtype=bool)
    hx = torch.zeros(num_envs, 256).to(device)
    cx = torch.zeros(num_envs, 256).to(device)

    last_save_ep = episodes_completed
    last_historical_save_ep = episodes_completed
    # League exploiter state -- see exploiter.py. Kept across bursts so a burst
    # can continue the previous exploiter rather than always restarting, and
    # reset on the module's own re-seed cadence.
    #
    # The schedule (index + last burst episode) now survives a restart; the
    # exploiter's own WEIGHTS deliberately do not. Persisting them would double
    # every checkpoint write for a marginal gain, and a resume that re-seeds
    # from the current main agent is the more interpretable option anyway --
    # a re-seeded exploiter starts as a bit-exact copy of its target, which is
    # what makes 0.50 the exact null its win rate is read against.
    exploiter_state = None
    exploiter_burst_index = resumed_exploiter_burst_index
    last_exploiter_burst_ep = resumed_exploiter_burst_ep
    last_replay_ep = episodes_completed
    ep_rewards = np.zeros(num_envs)
    ep_shaping = np.zeros(num_envs)
    ep_steps = np.zeros(num_envs, dtype=np.int64)

    print(f"Self-play training started on {num_envs} CPU cores simultaneously!")

    while episodes_completed < 1000000:
        episodes_since_reboost = episodes_completed - entropy_reboost_episode
        # NOTE: the old decaying entropy schedule no longer drives anything --
        # the per-head coefficients are set by the controller in the update step
        # below (see ENTROPY_TARGET_*). It is kept computed ONLY so the resume
        # path and the stage bookkeeping that reference stage_start_episode keep
        # working unchanged; nothing reads it into the loss. The console and
        # TensorBoard readouts report the real adaptive coefficients instead.
        current_entropy_coef = max(min_entropy_coef, initial_entropy_coef * (entropy_decay_rate ** episodes_since_reboost))

        # NOTE: the rollout-start hidden state snapshot is gone -- truncated
        # BPTT resumes each chunk from its own recorded state (hx_in_buffer),
        # and hx_in_seq[0] is exactly what this snapshot was.

        for step in range(update_timestep):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

            # Affordability mask -- see model.py affordability_mask docstring.
            # Derived from obs_tensor, which is what obs_buffer stores, so the
            # PPO update can recompute a bit-identical mask.
            card_mask = net.affordability_mask(obs_tensor)

            # --- advisor-targeted coverage ---------------------------------
            # One affordable slot per env, and the advisor's own score surface
            # for it where the advisor has something to say about this board.
            # Computed here rather than in the update because the target is a
            # function of the OBSERVATION, and recomputing it once per epoch per
            # minibatch would cost ~8x for an identical answer.
            hand_ids_now = net.hand_card_ids(obs_tensor)
            cov_w = AT.slot_weights_for(hand_ids_now)
            cov_slot = placement_coverage_slots(
                card_mask.unsqueeze(0), net.hand_size,
                slot_weights=None if cov_w is None else cov_w.unsqueeze(0))[0]
            coverage_slot_buffer.append(cov_slot)
            if AT.enabled():
                slot_ids = torch.cat(
                    [hand_ids_now,
                     torch.full((num_envs, 1), -1, dtype=torch.long)], dim=1)
                cov_ids = slot_ids.gather(1, cov_slot.view(-1, 1)).squeeze(1)
                tgt_np, has_np = AT.targets_for_batch(
                    obs_tensor.cpu().numpy(), cov_ids.cpu().numpy(),
                    advisor_legal)
                coverage_target_buffer.append(torch.from_numpy(tgt_np))
                coverage_has_buffer.append(
                    torch.from_numpy(has_np.astype(np.float32)))

            # Captured BEFORE step_lstm_and_card advances (hx, cx).
            hx_in_buffer.append(hx)
            cx_in_buffer.append(cx)

            with torch.no_grad():
                # Autoregressive placement: card must actually be SAMPLED
                # before placement can be conditioned on it -- see model.py
                # own comment on why forward_from_features (card_idx already
                # known) does not fit the rollout case.
                features, card_embeds, spatial_map = net.extract_features(obs_tensor)
                card_logits, _, _, state_value, (hx, cx) = net.step_lstm_and_card(
                    features, (hx, cx), card_mask)
                card_dist = Categorical(logits=card_logits)
                card_idx = card_dist.sample()
                placement_logits = net.placement_given_card(hx, card_embeds, card_idx, obs_tensor, spatial_map)
                placement_dist = Categorical(logits=placement_logits)
                placement_cell = placement_dist.sample()
                total_logprob = card_dist.log_prob(card_idx) + placement_dist.log_prob(placement_cell)

            target_x, target_y = net.cell_to_xy(placement_cell)

            action = {
                "card_index": card_idx.cpu().numpy(),
                "target_x": target_x.cpu().numpy().reshape(num_envs, 1),
                "target_y": target_y.cpu().numpy().reshape(num_envs, 1),
                # AsyncVectorEnv's Dict-space iteration requires every
                # action_space key to exist in the dict (it doesn't fall back
                # to a default like MicroRoyaleSelfPlayEnv.step()'s own
                # action.get("activate_ability_slot1", 0) does for a direct,
                # non-vectorized call), so both real sampled arrays must be
                # supplied here rather than omitted.
                # Deck has no Champion -- see DEFAULT_DECK_ABILITY_SLOTS.
                "activate_ability_slot1": np.zeros(num_envs, dtype=np.int64),
                "activate_ability_slot2": np.zeros(num_envs, dtype=np.int64),
            }

            next_obs, step_rewards, terminateds, truncateds, infos = envs.step(action)
            dones = terminateds | truncateds

            # TRUE terminal (a king died: raw reward +/-1) vs TRUNCATION (natural
            # max-tick timeout OR a scenario-window cutoff, raw reward ~0). Only
            # true terminals get value 0 bootstrapped; truncations must bootstrap
            # V(final_obs) or the critic learns a biased "world ends here" value.
            # Derived from the raw engine reward sign so it needs no extra signal
            # from the wrapper -- a scenario cutoff arrives as truncated=True with
            # reward ~0, a timeout as terminated=True with reward ~0, both -> boot.
            is_terminal = dones & (np.abs(step_rewards) > 0.5)
            needs_boot = dones & ~is_terminal
            trunc_boot_val = torch.zeros(num_envs, dtype=torch.float32, device=device)
            if needs_boot.any():
                # next_obs at a done step is the episode's TRUE final observation
                # (gymnasium next-step autoreset), and (hx, cx) here is the hidden
                # state that would process it (post this step's forward, pre the
                # done-mask reset below) -- so this is exactly V(final_obs).
                with torch.no_grad():
                    boot_feats, _, _ = net.extract_features(torch.tensor(next_obs, dtype=torch.float32).to(device))
                    _, _, _, boot_v, _ = net.step_lstm_and_card(boot_feats, (hx, cx))
                    boot_v = boot_v.squeeze(-1)
                needs_boot_t = torch.as_tensor(needs_boot, dtype=torch.bool, device=device)
                trunc_boot_val = torch.where(needs_boot_t, boot_v, trunc_boot_val)

            # DRAW_PENALTY punishes a real game that timed out (passivity) --
            # keyed on `terminateds` (engine game-over with no winner), NOT on
            # `dones`, so a scenario-window truncation (arrives as truncated=True,
            # reward ~0) is NOT mistaken for a draw and a SUCCESSFUL defense that
            # simply ran out its focused window isn't spuriously penalized.
            is_draw = terminateds & (np.abs(step_rewards) < 0.5)
            draw_penalty = DRAW_PENALTY * is_draw.astype(np.float32)

            zeros = np.zeros(num_envs, dtype=np.int64)
            zeros_f = np.zeros(num_envs, dtype=np.float32)
            stats = {
                "team0_troop_damage": infos.get("team0_troop_damage", zeros),
                "team1_troop_damage": infos.get("team1_troop_damage", zeros),
                "team0_building_damage": infos.get("team0_building_damage", zeros),
                "team0_tower_damage": infos.get("team0_tower_damage", zeros),
                # Lethal-spell PBRS inputs. Defaults are the "no opportunity"
                # state, so a missing key can only ever zero the term, never
                # fabricate one.
                "enemy_tower_hp": np.asarray(infos.get(
                    "enemy_tower_hp", np.zeros((len(zeros), 3), dtype=np.float32)),
                    dtype=np.float32).reshape(len(zeros), 3),
                "fireball_in_hand": np.asarray(
                    infos.get("fireball_in_hand", zeros), dtype=np.float32),
                "fireball_value_killed": np.asarray(
                    infos.get("fireball_value_killed", zeros), dtype=np.float32),
                "fireball_elixir_spent": np.asarray(
                    infos.get("fireball_elixir_spent", zeros), dtype=np.float32),
                "team1_tower_damage": infos.get("team1_tower_damage", zeros),
                "team1_building_damage": infos.get("team1_building_damage", zeros),
                "team0_elixir_spent": infos.get("team0_elixir_spent", zeros_f),
                "team1_elixir_spent": infos.get("team1_elixir_spent", zeros_f),
                "team0_elixir_current": infos.get("elixir", zeros_f),
                # Default 3 (a full set) so the rare all-envs-reset step, where
                # gymnasium omits the key entirely, yields a zero delta rather
                # than a phantom three-crown swing.
                "team0_towers_alive": infos.get("team0_towers_alive", np.full(num_envs, 3, dtype=np.int64)),
                "team1_towers_alive": infos.get("team1_towers_alive", np.full(num_envs, 3, dtype=np.int64)),
                # Win-condition damage -- see train.W_WIN_CONDITION_DAMAGE.
                "team0_wincon_damage": infos.get("team0_wincon_damage", zeros),
            }

            # Which envs ACTUALLY got a card down this step. The engine silently
            # refuses illegal/unaffordable plays, so "the policy chose a card" is
            # not the same as "a card was played" -- only a rise in cumulative
            # elixir_spent proves it. prev_dones masks the phantom post-autoreset
            # step, where the counter restarts at 0 and the delta is meaningless.
            if prev_stats is not None:
                spent_delta = stats["team0_elixir_spent"] - prev_stats["team0_elixir_spent"]
                really_played = (spent_delta > 1e-6) & (~prev_dones)
                if really_played.any():
                    # Card identity and elixir must come from the observation the
                    # action was taken on -- the hand rotates the instant a card is
                    # played, so reading it afterwards returns the wrong card.
                    hand_ids_np = net.hand_card_ids(obs_tensor).cpu().numpy()
                    elixir_np = net.elixir_from_obs(obs_tensor).cpu().numpy()
                    card_idx_np = card_idx.cpu().numpy()
                    for i in np.nonzero(really_played)[0]:
                        slot = int(card_idx_np[i])
                        if slot < net.hand_size:
                            cid = int(hand_ids_np[i, slot])
                            if cid >= 0:
                                ep_card_ids[i].add(cid)
                        ep_elixir_at_play[i].append(float(elixir_np[i]))
                        ep_play_count[i] += 1
                        # Row of the cell that actually reached the board.
                        # placement_cell is row-major over board_width, the
                        # same layout cell_to_xy inverts.
                        if int(placement_cell[i].item()) // net.board_width >= FORWARD_ROW_Y:
                            ep_forward_placements[i] += 1

            # gamma passed explicitly: the tower term is potential-based
            # (gamma*Phi(s') - Phi(s)) and its policy-invariance guarantee only
            # holds if this is the SAME gamma the GAE/returns use below.
            # w_spell passed explicitly since 2026-08-14 -- see train.py's
            # spell_value_weight for the dead-code bug this closes.
            shaping = compute_shaping(stats, prev_stats, gamma=gamma,
                                      w_spell=spell_value_weight(episodes_completed))
            shaping = shaping * (1.0 - prev_dones)

            # "Perfect defense" -- the SAME function train.py calls, not a
            # second copy of the arithmetic.
            flawless_bonus = flawless_defense_bonus(dones, step_rewards,
                                                    stats, prev_stats)
            shaped_rewards = step_rewards + shaping - draw_penalty + flawless_bonus
            ep_rewards += shaped_rewards
            ep_shaping += shaping
            ep_steps += 1

            valid = torch.tensor(1.0 - prev_dones, dtype=torch.float32).to(device)
            mask = torch.tensor(1.0 - dones, dtype=torch.float32).to(device) * valid
            # Bootstrap coefficient: 1 except on TRUE terminals (0). Folded with
            # valid so the throwaway post-autoreset step matches the trace mask
            # exactly -- keeps this a strict, provable generalization of the old
            # single-mask GAE (identical when nothing truncates).
            boot_nonterminal = torch.as_tensor(1.0 - is_terminal.astype(np.float32),
                                               dtype=torch.float32, device=device) * valid
            trunc_flag = torch.as_tensor(needs_boot.astype(np.float32), dtype=torch.float32, device=device)

            obs_buffer.append(obs_tensor)
            card_actions_buffer.append(card_idx)
            placement_actions_buffer.append(placement_cell)
            decision_buffer.append((card_mask.sum(dim=1) > 1).float() * valid)
            logprobs_buffer.append(total_logprob)
            values_buffer.append(state_value.squeeze(-1))
            rewards_buffer.append(torch.tensor(shaped_rewards, dtype=torch.float32).to(device))
            masks_buffer.append(mask)
            valid_buffer.append(valid)
            aux_elixir_buffer.append(
                torch.tensor(np.asarray(infos.get("opp_elixir", np.zeros(num_envs, dtype=np.float32)),
                                        dtype=np.float32)).to(device))
            boot_nonterminal_buffer.append(boot_nonterminal)
            trunc_flag_buffer.append(trunc_flag)
            trunc_boot_buffer.append(trunc_boot_val)

            mask_tensor = mask.unsqueeze(1)
            hx = hx * mask_tensor
            cx = cx * mask_tensor

            is_scenario_arr = infos.get("is_scenario", np.zeros(num_envs, dtype=np.float32))
            scenario_def_arr = infos.get("scenario_defensive",
                                         np.zeros(num_envs, dtype=np.float32))
            for i, done in enumerate(dones):
                if done:
                    episodes_completed += 1
                    if is_scenario_arr[i] > 0.5:
                        # Scenario episodes are kept OUT of the matchup histories
                        # so the headline W/L/D and the length/building-HP curves
                        # stay pure normal-game signals.
                        #
                        # Only DEFENSIVE scenarios reach ScenDef. Success there
                        # means "the episode did not end in a tower/game loss",
                        # which is only a question worth asking when something
                        # was threatening us. fireball_tower_value spawns at the
                        # ENEMY tower and giant_commit spawns nothing, so both
                        # pass that test by default -- including when the agent
                        # does nothing at all. Counting them pushed ScenDef
                        # toward 1.0 and would have masked a genuine collapse in
                        # the defensive reflex the metric exists to watch.
                        if scenario_def_arr[i] > 0.5:
                            scenario_success_history.append(
                                1.0 if step_rewards[i] > -0.5 else 0.0)
                        else:
                            scenario_other_history.append(
                                1.0 if step_rewards[i] > -0.5 else 0.0)
                    else:
                        reward_history.append(ep_rewards[i])
                        shaping_history.append(ep_shaping[i])
                        ep_len_history.append(ep_steps[i])
                        ally_end, enemy_end = building_hp_end(next_obs[i])
                        ally_bldg_end_history.append(ally_end)
                        enemy_bldg_end_history.append(enemy_end)
                        if step_rewards[i] > 0.5:
                            outcome_value = 1
                        elif step_rewards[i] < -0.5:
                            outcome_value = -1
                        else:
                            outcome_value = 0
                        outcome_history.append(outcome_value)
                        outcome_history_long.append(outcome_value)
                    # Strategy diagnostics close out with the episode, on the same
                    # normal-game/scenario split as everything else above.
                    if is_scenario_arr[i] <= 0.5:
                        cards_per_game_history.append(len(ep_card_ids[i]))
                        plays_per_game_history.append(int(ep_play_count[i]))
                        if ep_elixir_at_play[i]:
                            elixir_at_play_history.append(float(np.mean(ep_elixir_at_play[i])))
                        if ep_play_count[i] > 0:
                            forward_rate_history.append(
                                float(ep_forward_placements[i]) / float(ep_play_count[i]))
                        killed_i = np.asarray(infos["ep_killed_by_card"][i], dtype=np.float64)
                        spent_i = np.asarray(infos["ep_spent_by_card"][i], dtype=np.float64)
                        if spent_i.sum() > 0.0:
                            econ_killed.append(killed_i)
                            econ_spent.append(spent_i)
                        # Tower damage per 1000 engine ticks. Length-normalized
                        # because a slower match accumulates more of everything
                        # -- AvgTicks moved 24% in one hour after the placement
                        # mask fix, which would masquerade as more pressure.
                        ticks_i = float(ep_steps[i]) * 10.0
                        if ticks_i > 0.0:
                            tower_dmg_rate_history.append(
                                float(infos["team0_tower_damage"][i]) * 1000.0 / ticks_i)
                    ep_card_ids[i] = set()
                    ep_elixir_at_play[i] = []
                    ep_play_count[i] = 0
                    ep_forward_placements[i] = 0
                    ep_rewards[i] = 0
                    ep_shaping[i] = 0
                    ep_steps[i] = 0

                    if episodes_completed % 10 == 0 and outcome_history:
                        outcomes = np.array(outcome_history)
                        wins = int((outcomes == 1).sum())
                        losses = int((outcomes == -1).sum())
                        draws = int((outcomes == 0).sum())
                        n = len(outcomes)
                        decided = wins + losses
                        decisive_wr = wins / decided if decided > 0 else 0.0
                        avg_reward = np.mean(reward_history)
                        avg_shaping = np.mean(shaping_history)
                        scenario_sr = np.mean(scenario_success_history) if scenario_success_history else float("nan")
                        scenario_or = (np.mean(scenario_other_history)
                                       if scenario_other_history else float("nan"))
                        # ep_len_history is in bot-steps (each step == skip_frames ticks,
                        # 10 by default -- see envs.step(action) above, which never
                        # overrides it), so *10 converts to real engine ticks. A short
                        # average here (well under a few hundred ticks) means most
                        # recent games are ending fast -- worth knowing whether that's
                        # decisive, well-played games or a degenerate/exploited shortcut,
                        # not just inferring it from the win-rate number alone.
                        avg_ticks_50 = np.mean(ep_len_history) * 10 if ep_len_history else float("nan")
                        # Strategy readout. Cards/Game is the headline: 8.0 means the
                        # bot is using its whole deck, ~6.0 was the phase-1 failure
                        # mode (win condition and spell abandoned). Elixir is the
                        # average bar level when it actually commits a card --
                        # phase 1 sat at ~3.6, i.e. spending the instant it could.
                        cards_pg = np.mean(cards_per_game_history) if cards_per_game_history else float("nan")
                        plays_pg = np.mean(plays_per_game_history) if plays_per_game_history else float("nan")
                        elix_pl = np.mean(elixir_at_play_history) if elixir_at_play_history else float("nan")
                        # --- strategic progress, as opposed to spread ---------
                        # ROI is a ratio of SUMS over the window, not a mean of
                        # per-episode ratios: a card that went unplayed in an
                        # episode contributes 0/0, and averaging those makes the
                        # number jump around for reasons that have nothing to do
                        # with how well the card was used.
                        roi_all = float("nan")
                        roi_worst = float("nan")
                        worst_name = "-"
                        if econ_spent:
                            k_tot = np.sum(econ_killed, axis=0)
                            s_tot = np.sum(econ_spent, axis=0)
                            if s_tot.sum() > 0:
                                roi_all = float(k_tot.sum() / s_tot.sum())
                            played = s_tot > 0
                            if played.any():
                                per_card = np.where(played, k_tot / np.maximum(s_tot, 1e-9), np.inf)
                                w = int(np.argmin(per_card))
                                roi_worst = float(per_card[w])
                                worst_name = _card_name(list(DEFAULT_DECK)[w]).split("_")[0][:9]
                                for ci, cid in enumerate(DEFAULT_DECK):
                                    if played[ci]:
                                        writer.add_scalar(
                                            f"Economy/ROI_ByCard/{_card_name(cid)}",
                                            float(k_tot[ci] / s_tot[ci]), episodes_completed)
                        fwd_rate = (np.mean(forward_rate_history)
                                    if forward_rate_history else float("nan"))
                        twr_rate = (np.mean(tower_dmg_rate_history)
                                    if tower_dmg_rate_history else float("nan"))
                        print(f"Episodes: {episodes_completed} | Avg(50): {avg_reward:.2f} | "
                              f"W/L/D: {wins/n:.2f}/{losses/n:.2f}/{draws/n:.2f} | Decisive: {decisive_wr:.2f} | "
                              f"ScenDef: {scenario_sr:.2f} | ScenOff: {scenario_or:.2f} | AvgTicks: {avg_ticks_50:.0f} | "
                              f"Cards/Game: {cards_pg:.2f}/8 | Plays: {plays_pg:.1f} | Elixir@Play: {elix_pl:.2f} | "
                              f"ROI: {roi_all:.2f} | Worst: {worst_name} {roi_worst:.2f} | "
                              f"Fwd: {100*fwd_rate:.0f}% | TwrDmg/1k: {twr_rate:.0f} | "
                              f"Pool: {len(historical_pool)} | EntCoef c/p: {ent_coef_card:.4f}/{ent_coef_place:.4f}")
                        writer.add_scalar("Economy/ROI_All_Cards", roi_all, episodes_completed)
                        writer.add_scalar("Economy/ROI_Worst_Card", roi_worst, episodes_completed)
                        writer.add_scalar("Strategy/Forward_Placement_Rate", fwd_rate, episodes_completed)
                        writer.add_scalar("Strategy/Tower_Damage_Per_1k_Ticks", twr_rate, episodes_completed)
                        writer.add_scalar("Strategy/Distinct_Cards_Per_Game", cards_pg, episodes_completed)
                        writer.add_scalar("Strategy/Plays_Per_Game", plays_pg, episodes_completed)
                        writer.add_scalar("Strategy/Elixir_At_Play", elix_pl, episodes_completed)
                        writer.add_scalar("Progress/Episode_Length_Ticks_50", avg_ticks_50, episodes_completed)
                        if scenario_success_history:
                            writer.add_scalar("Scenario/Defense_Success_Rate", scenario_sr, episodes_completed)
                            writer.add_scalar("Scenario/Sample_Count", len(scenario_success_history), episodes_completed)
                        if scenario_other_history:
                            writer.add_scalar("Scenario/Offensive_No_Loss_Rate",
                                              scenario_or, episodes_completed)
                            writer.add_scalar("Scenario/Offensive_Sample_Count",
                                              len(scenario_other_history), episodes_completed)
                        writer.add_scalar("Training/Avg_Reward_50", avg_reward, episodes_completed)
                        writer.add_scalar("Reward/Episode_Shaping_Sum", avg_shaping, episodes_completed)
                        writer.add_scalar("Training/Win_Rate_100", wins / n, episodes_completed)
                        writer.add_scalar("Rates/Loss_100", losses / n, episodes_completed)
                        writer.add_scalar("Rates/Draw_100", draws / n, episodes_completed)
                        writer.add_scalar("Rates/Decisive_Win_100", decisive_wr, episodes_completed)
                        writer.add_scalar("Training/Decided_Count_100", decided, episodes_completed)
                        outcomes_long = np.array(outcome_history_long)
                        wins_long = int((outcomes_long == 1).sum())
                        losses_long = int((outcomes_long == -1).sum())
                        decided_long = wins_long + losses_long
                        if decided_long > 0:
                            writer.add_scalar("Rates/Decisive_Win_500", wins_long / decided_long, episodes_completed)
                        writer.add_scalar("Progress/Episode_Length_50", np.mean(ep_len_history), episodes_completed)
                        writer.add_scalar("Progress/Enemy_Building_HP_End_50", np.mean(enemy_bldg_end_history), episodes_completed)
                        writer.add_scalar("Progress/Ally_Building_HP_End_50", np.mean(ally_bldg_end_history), episodes_completed)
                        writer.add_scalar("Training/PFSP_Pool_Size", len(historical_pool), episodes_completed)
                        writer.add_scalar("Training/Entropy_Coef_Card", ent_coef_card, episodes_completed)
                        writer.add_scalar("Training/Entropy_Coef_Placement", ent_coef_place, episodes_completed)

                    # --- Entropy re-boost heuristics only -- opponent
                    # SELECTION is PFSP's job now (see
                    # MicroRoyaleSelfPlayEnv._sample_pfsp_opponent, called
                    # independently by each of the 8 workers at every
                    # reset()). This block only manages exploration, reading
                    # the GLOBAL outcome_history across whatever mix of pool
                    # opponents each worker happened to sample this window.
                    window_full = len(outcome_history) == outcome_history.maxlen
                    if window_full:
                        outcomes_arr = np.array(outcome_history)
                        n_outcomes = len(outcomes_arr)
                        win_rate = int((outcomes_arr == 1).sum()) / n_outcomes
                        draw_rate = int((outcomes_arr == 0).sum()) / n_outcomes

                        if (draw_rate >= STALL_DRAW_RATE_THRESHOLD
                                and episodes_completed - last_stall_reboost_episode >= STALL_REBOOST_COOLDOWN_EPISODES):
                            entropy_reboost_episode = episodes_completed
                            last_stall_reboost_episode = episodes_completed
                            print(f">>> High draw rate ({draw_rate:.2f}) across the pool -- "
                                  "re-boosting exploration to try to break out of a passive equilibrium.")
                        elif (win_rate < 0.5
                                and episodes_completed - last_stall_reboost_episode >= ENTROPY_STALE_REBOOST_EPISODES):
                            entropy_reboost_episode = episodes_completed
                            last_stall_reboost_episode = episodes_completed
                            print(f">>> Losing trend (win rate {win_rate:.2f}) across the pool with no draw-stall "
                                  f"trigger in {ENTROPY_STALE_REBOOST_EPISODES}+ episodes -- "
                                  "periodically re-boosting exploration anyway.")

            obs = next_obs
            prev_stats = stats
            prev_dones = dones

        # --- PPO Update (identical shape to train.py's) ---
        obs_seq = torch.stack(obs_buffer)
        card_actions_seq = torch.stack(card_actions_buffer)
        placement_actions_seq = torch.stack(placement_actions_buffer)   # board-cell indices
        decision_seq = torch.stack(decision_buffer)
        hx_in_seq = torch.stack(hx_in_buffer)      # (T, N, 256) -- chunk resume states
        cx_in_seq = torch.stack(cx_in_buffer)
        old_logprobs_seq = torch.stack(logprobs_buffer)
        values_seq = torch.stack(values_buffer)
        rewards_seq = torch.stack(rewards_buffer)
        masks_seq = torch.stack(masks_buffer)
        valid_seq = torch.stack(valid_buffer)
        aux_elixir_seq = torch.stack(aux_elixir_buffer)                # (T, N) -- opponent elixir ground truth
        boot_nonterminal_seq = torch.stack(boot_nonterminal_buffer)
        trunc_flag_seq = torch.stack(trunc_flag_buffer)
        trunc_boot_seq = torch.stack(trunc_boot_buffer)
        coverage_slot_seq = torch.stack(coverage_slot_buffer)          # (T, N)
        if AT.enabled():
            coverage_target_seq = torch.stack(coverage_target_buffer)  # (T,N,cells)
            coverage_has_seq = torch.stack(coverage_has_buffer)        # (T, N)
        else:
            coverage_target_seq = None
            coverage_has_seq = torch.zeros_like(coverage_slot_seq, dtype=torch.float32)

        with torch.no_grad():
            next_obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            # Value only depends on hx, never needs a card/placement -- skips
            # straight past card_logits and never touches placement (see
            # model.py's own comment on why this split exists).
            next_features, _, _ = net.extract_features(next_obs_tensor)
            _, _, _, next_value, _ = net.step_lstm_and_card(next_features, (hx, cx))
            next_value = next_value.squeeze(-1)

        advantages_seq = torch.zeros_like(rewards_seq)
        gae = torch.zeros(num_envs).to(device)
        for t in reversed(range(update_timestep)):
            base_next_val = next_value if t == update_timestep - 1 else values_seq[t + 1]
            # On a truncation/timeout step, values_seq[t+1] belongs to the NEXT
            # (already-reset) episode, so it must NOT be used as this step's
            # next-state value -- swap in the captured V(final_obs) bootstrap.
            # boot_nonterminal zeroes the whole bootstrap term on true terminals.
            next_val = torch.where(trunc_flag_seq[t] > 0.5, trunc_boot_seq[t], base_next_val)
            delta = rewards_seq[t] + gamma * next_val * boot_nonterminal_seq[t] - values_seq[t]
            # Trace still cut at every episode boundary (masks_seq = 1-done, folded
            # with valid) -- unchanged.
            gae = delta + gamma * gae_lambda * masks_seq[t] * gae
            advantages_seq[t] = gae

        returns_seq = advantages_seq + values_seq
        adv_norm_seq = (advantages_seq - advantages_seq.mean()) / (advantages_seq.std() + 1e-8)

        # Critic diagnostics. Raw MSE is uninterpretable on its own -- it is
        # bounded below by the irreducible noise in the returns, so a "flat"
        # critic loss says nothing about whether the critic is any good.
        # Explained variance does: 1 - Var(return - V)/Var(return), where ~1 is
        # a good critic, 0 is no better than predicting the mean, and <0 is
        # worse than the mean. Measured at +0.64 when this was added, i.e. the
        # critic was healthy all along and the flat Loss/Critic curve had been
        # misread as a failure to learn.
        with torch.no_grad():
            _vm = valid_seq > 0.5
            _r = returns_seq[_vm]
            _v = values_seq[_vm]
            _rv = _r.var()
            explained_variance = (1.0 - (_r - _v).var() / _rv.clamp(min=1e-8)) if _rv > 0 else torch.tensor(0.0)
            # See VF_CLIP_STD_FRAC: scaled to this batch's own return spread,
            # never tighter than eps_clip.
            vf_clip_range = torch.clamp(VF_CLIP_STD_FRAC * _r.std(), min=eps_clip).item()
        writer.add_scalar("Loss/Critic_Explained_Variance", explained_variance.item(), episodes_completed)
        writer.add_scalar("Loss/Value_Clip_Range", vf_clip_range, episodes_completed)

        # Every (chunk-start, env) pair is one independent training segment --
        # see train.py identical construction and its bptt_chunk comment.
        chunk_starts = np.arange(0, update_timestep, bptt_chunk)
        segments = np.array([(cs, e) for cs in chunk_starts for e in range(num_envs)], dtype=np.int64)
        n_segments = len(segments)
        seg_mb_size = max(1, n_segments // num_minibatches)
        chunk_offsets = torch.arange(bptt_chunk, dtype=torch.long, device=device).unsqueeze(1)
        actor_losses, critic_losses, entropy_bonuses, total_losses, clip_fracs = [], [], [], [], []
        aux_losses, aux_maes = [], []
        # Placement entropy of affordable-but-UNCHOSEN cards -- the freeze
        # detector. See train.py's PLACEMENT_COVERAGE_COEF.
        coverage_ents = []
        # KL to the advisor's surface on the rows that carry one, and how many
        # rows those are. Read as a pair: a falling KL on a rising row count is
        # the head learning the advisor's geometry; a falling KL on a collapsing
        # row count is just the advisor going quiet.
        coverage_kls, coverage_hits = [], []
        # Logged separately so a collapsing head is visible in TensorBoard
        # directly, instead of only showing up in an offline behavioral probe.
        ent_card_log, ent_place_log = [], []
        # H(placement | card) as a fraction of that card's own maximum, per card
        # id, plus the no-op arm for contrast. The aggregate provably cannot see
        # a per-card collapse -- a mixture of eight sharp, well-separated modes
        # has high entropy even when every component is a delta, which is
        # exactly the state this run was in. Accumulated on epoch 0 only, from
        # tensors the update already computed, so it costs no extra simulation.
        percard_place_ent = defaultdict(list)
        ent_place_noop_log = []

        for epoch in range(ppo_epochs):
            seg_perm = np.random.permutation(n_segments)
            for mb_start in range(0, n_segments, seg_mb_size):
                mb_seg = segments[seg_perm[mb_start:mb_start + seg_mb_size]]
                if len(mb_seg) == 0:
                    continue
                t0 = torch.as_tensor(mb_seg[:, 0], dtype=torch.long, device=device)
                ev = torch.as_tensor(mb_seg[:, 1], dtype=torch.long, device=device)
                B = t0.shape[0]
                tt = t0.unsqueeze(0) + chunk_offsets          # (L, B)
                ee = ev.unsqueeze(0).expand(bptt_chunk, B)    # (L, B)

                mb_obs_flat = obs_seq[tt, ee].reshape(bptt_chunk * B, -1)
                # Pre-pool activation taken from the same call, for the
                # high-resolution placement branch -- see the matching comment
                # in train.py. Identical arithmetic to letting
                # placement_given_card rebuild it; this just avoids re-running
                # the trunk's first conv once per minibatch per epoch.
                (feats_seq, card_embeds_seq, spatial_seq,
                 hires_seq) = net.extract_features_hires(mb_obs_flat)
                feats_seq = feats_seq.view(bptt_chunk, B, -1)
                hires_seq = hires_seq.view(bptt_chunk, B, *hires_seq.shape[1:])
                # Same (T,B,...) split for the CNN's spatial map, which the
                # convolutional placement head consumes. Recomputed here from
                # the SAME stored observations rather than buffered, for the
                # identical reason card_mask_seq is: anything that can drift
                # apart from what the rollout used silently corrupts the PPO
                # ratio, and deriving it makes drift impossible.
                spatial_seq = spatial_seq.view(bptt_chunk, B, *spatial_seq.shape[1:])
                # Same observations the rollout acted on, reshaped per timestep so
                # placement legality is recomputed identically (see placement_mask).
                mb_obs_seq = mb_obs_flat.view(bptt_chunk, B, -1)
                card_embeds_seq = card_embeds_seq.view(bptt_chunk, B, net.hand_size + 1, -1)
                # Recomputed from the same stored observations the rollout acted
                # on -- bit-identical to the sampling-time mask by construction.
                card_mask_seq = net.affordability_mask(mb_obs_flat).view(
                    bptt_chunk, B, net.hand_size + 1)

                mb_card_actions = card_actions_seq[tt, ee]
                mb_place_actions = placement_actions_seq[tt, ee]
                mb_masks = masks_seq[tt, ee]

                # Resume from the state recorded at this chunk first timestep.
                rhx = hx_in_seq[t0, ev]
                rcx = cx_in_seq[t0, ev]
                # ONE batched pass over the whole chunk instead of a per-timestep
                # loop. Only the LSTM is genuinely recurrent; the card/value/aux/
                # placement heads are pointwise in time, and running them L times
                # at batch B is dominated by call overhead on CPU. Measured 1.82x
                # on the forward pass, and verified equal to the looped path to
                # within float32 round-off (max abs logit delta 1.1e-08 vs an
                # eps of 1.19e-07, identical -inf masks, PPO ratio exactly 1.0)
                # -- see MicroRoyaleNet.forward_sequence.
                #
                # mb_card_actions is the STORED action from rollout, not a fresh
                # sample: placement must be conditioned on exactly the card the
                # log-prob is scored against, or the ratio breaks silently.
                # Placement coverage pass -- see train.py's PLACEMENT_COVERAGE_COEF
                # for the measured defect this closes. Identical here because the
                # defect is identical: both loops score placement only through the
                # chosen card, so both starve an unplayed card's map of gradient.
                # Read from the rollout, NOT resampled: the advisor target
                # buffered alongside it was computed for exactly this slot on
                # exactly this observation, and a fresh draw would silently pair
                # one card's logits with another card's target.
                cf_idx = coverage_slot_seq[tt, ee]
                (cl_seq, pl_seq, new_values, new_aux_elixir, _, cf_pl_seq) = net.forward_sequence(
                    feats_seq, card_embeds_seq, spatial_seq, mb_obs_seq,
                    card_mask_seq, mb_card_actions, mb_masks, (rhx, rcx),
                    extra_card_idx_seq=cf_idx, hires_seq=hires_seq)
                card_dist_t = Categorical(logits=cl_seq)
                place_dist_t = Categorical(logits=pl_seq)
                new_logprobs = (card_dist_t.log_prob(mb_card_actions)
                                + place_dist_t.log_prob(mb_place_actions))
                new_ent_card = card_dist_t.entropy()
                new_ent_place = place_dist_t.entropy()

                # --- normalize each head by the entropy it can ACTUALLY reach.
                #
                # The per-card diagnostic below (place_frac_elem) has always
                # divided by log(n_legal) and its comment already gives the
                # reason: "a spell sees 588 cells, a plain troop 242, the
                # Cannon 208 ... a raw nat count is not comparable". The
                # OBJECTIVE and the CONTROLLER never got the same treatment --
                # they divided by log(total arms) instead, which is a ceiling
                # no masked step can reach.
                #
                # It matters most on the card head. MEASURED on
                # model_weights_cured.pth over 706 decision steps: 54.1% leave
                # exactly two legal arms (one affordable card + the no-op),
                # where the reachable maximum is log(2) = 0.693 nats, so the
                # 0.35 * log(5) = 0.5633 target is 81.3% of it -- the play/wait
                # decision was being held near a coin flip on the majority of
                # decisions. The controller read 0.3125 against its 0.35 target
                # and kept RAISING the coefficient while the policy's real
                # randomness was 0.4906 of reachable. Consequence: elixir spent
                # on sight, mean elixir 2.25/10, nothing affordable on 73.9% of
                # steps (78.5% under a big push), and P(play) FLAT against
                # threat -- 0.1008 with no threat vs 0.1016 under the largest.
                #
                # clamp(min=2) only guards log(1)=0; single-arm rows carry zero
                # entropy and are excluded by mb_decision regardless.
                n_card_legal = card_mask_seq.sum(-1).clamp(min=2).float()
                n_place_legal = torch.isfinite(pl_seq).sum(-1).clamp(min=2).float()
                new_ent_card = new_ent_card / torch.log(n_card_legal)
                new_ent_place = new_ent_place / torch.log(n_place_legal)

                mb_adv = adv_norm_seq[tt, ee]
                mb_ret = returns_seq[tt, ee]
                mb_old_logprobs = old_logprobs_seq[tt, ee]
                mb_old_values = values_seq[tt, ee]
                mb_valid = valid_seq[tt, ee]
                mb_decision = decision_seq[tt, ee]
                n_valid = mb_valid.sum().clamp(min=1.0)
                n_decision = mb_decision.sum().clamp(min=1.0)

                ratios = torch.exp(new_logprobs - mb_old_logprobs)
                surr1 = ratios * mb_adv
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * mb_adv

                value_clipped = mb_old_values + torch.clamp(
                    new_values - mb_old_values, -vf_clip_range, vf_clip_range)
                critic_loss_unclipped = F.mse_loss(new_values, mb_ret, reduction="none")
                critic_loss_clipped = F.mse_loss(value_clipped, mb_ret, reduction="none")
                critic_loss_per_elem = torch.max(critic_loss_unclipped, critic_loss_clipped)

                # Actor/entropy normalized by decision steps, critic by all real
                # steps -- see train.py identical comment for why.
                actor_loss = -(torch.min(surr1, surr2) * mb_decision).sum() / n_decision
                critic_loss = (critic_loss_per_elem * mb_valid).sum() / n_valid
                ent_card_mean = (new_ent_card * mb_decision).sum() / n_decision
                # Placement entropy is measured and rewarded ONLY on steps that
                # actually placed a card. mb_decision means "a card was
                # AFFORDABLE", not "a card was PLAYED", and the placement head
                # is sampled on every decision step -- including the ones where
                # the policy chose the no-op and the sampled cell never reaches
                # the board. Averaging over those let the head earn the entropy
                # bonus for free on steps that cost nothing, while the
                # distribution that actually places cards collapsed underneath.
                #
                # Measured on the ep~62,200 checkpoint, 549 decision steps:
                # reported 0.462 of max, decomposing into 0.850 on no-op steps
                # against 0.090 on real placements -- with the target at 0.25.
                # So the controller read "too much exploration" and drove the
                # coefficient to its 0.01 floor (57% of updates in the hour
                # before this change) while the policy that places cards sat
                # ~3x BELOW target. Conditioned on a card, Cannon was at 0.017
                # of max with 96.4% of its mass on one cell, and Fireball and
                # Giant were pinned to that same cell -- unchanged from a
                # pre-mask-fix control, so it is this metric and not the action
                # space. Same blind spot as the checkerboard bias and the
                # team-1 observation bug: an aggregate cannot see a conditional
                # collapse, and the per-card breakdown below is the detector.
                mb_placed = mb_decision * (mb_card_actions != _NOOP_ACTION).float()
                n_placed = float(mb_placed.sum())
                if n_placed > 0.0:
                    ent_place_mean = (new_ent_place * mb_placed).sum() / n_placed
                else:
                    # A chunk with no placement at all: fall back to the old
                    # denominator rather than feed the controller a 0, which it
                    # would chase as a total collapse.
                    ent_place_mean = (new_ent_place * mb_decision).sum() / n_decision

                if epoch == 0:
                    with torch.no_grad():
                        # Normalize each sample by ITS OWN legal-cell count: a
                        # spell sees 588 cells, a plain troop 242, the Cannon
                        # 208 (placementRadius clearance), so a raw nat count is
                        # not comparable across cards.
                        # new_ent_place is ALREADY divided by log(n_legal) at
                        # the top of this loop, so this is now the identity --
                        # kept as a named alias so the per-card diagnostic below
                        # keeps reading the same quantity it always did.
                        place_frac_elem = new_ent_place
                        slot_ids = net.hand_card_ids(mb_obs_flat).view(
                            bptt_chunk, B, net.hand_size)
                        slot_ids = torch.cat(
                            [slot_ids, torch.full((bptt_chunk, B, 1), -1,
                                                  dtype=slot_ids.dtype,
                                                  device=slot_ids.device)], dim=2)
                        played_id = slot_ids.gather(
                            2, mb_card_actions.unsqueeze(-1)).squeeze(-1)
                        sel = mb_placed > 0
                        for cid, frac in zip(played_id[sel].tolist(),
                                             place_frac_elem[sel].tolist()):
                            if cid >= 0:
                                percard_place_ent[cid].append(frac)
                        mb_noop = mb_decision * (mb_card_actions == _NOOP_ACTION).float()
                        n_noop = float(mb_noop.sum())
                        if n_noop > 0.0:
                            ent_place_noop_log.append(
                                float((place_frac_elem * mb_noop).sum() / n_noop))

                # Each head normalized by its own maximum, then weighted -- see
                # the LOG_N_CARD comment above for why the raw sum was wrong.
                #
                # The coverage half now splits by row: where the advisor has a
                # rule for the sampled card it supplies a TARGET and the row
                # gets KL to it; everywhere else the old entropy bonus stands.
                # They are mutually exclusive because they ask for opposite
                # things -- see advisor_target.coverage_terms.
                mb_cov_targets = (coverage_target_seq[tt, ee]
                                  if coverage_target_seq is not None else None)
                cov_delta, cov_ent_frac, cov_kl, cov_n = AT.coverage_terms(
                    cf_pl_seq, mb_cov_targets, coverage_has_seq[tt, ee],
                    mb_decision, PLACEMENT_COVERAGE_COEF, LOG_N_PLACEMENT)
                coverage_ents.append(float(cov_ent_frac))
                coverage_kls.append(float(cov_kl))
                coverage_hits.append(float(cov_n))
                # ent_*_mean are ALREADY fractions of each head's reachable
                # maximum (divided per-step above), so no second division here.
                entropy_bonus = (ent_coef_card * ent_card_mean
                                 + ent_coef_place * ent_place_mean)
                # entropy_bonus already carries its per-head coefficients.
                # Auxiliary opponent-elixir loss. Masked by mb_valid for the same
                # reason the critic loss is: phantom auto-reset steps carry an
                # observation from the NEXT episode paired with stale
                # bookkeeping, and regressing on those teaches noise.
                aux_err = (new_aux_elixir - aux_elixir_seq[tt, ee])
                aux_loss = ((aux_err ** 2) * mb_valid).sum() / n_valid
                aux_mae = ((aux_err.abs()) * mb_valid).sum() / n_valid
                aux_losses.append(aux_loss.item())
                aux_maes.append(aux_mae.item())

                # cov_delta already carries both its signs: the entropy half is
                # a bonus (negative) and the advisor KL is a penalty (positive).
                loss = (actor_loss + 0.5 * critic_loss - entropy_bonus + cov_delta
                        + AUX_ELIXIR_COEF * AUX_ELIXIR_SCALE * aux_loss)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_bonuses.append((ent_card_mean + ent_place_mean).item())
                ent_card_log.append(ent_card_mean.item())
                ent_place_log.append(ent_place_mean.item())
                total_losses.append(loss.item())
                clipped = ((ratios - 1.0).abs() > eps_clip).float()
                clip_fracs.append(((clipped * mb_decision).sum() / n_decision).item())

        mean_actor_loss = np.mean(actor_losses)
        mean_critic_loss = np.mean(critic_losses)
        mean_entropy = np.mean(entropy_bonuses)
        mean_total_loss = np.mean(total_losses)
        mean_clip_frac = np.mean(clip_fracs)
        writer.add_scalar("Loss/Actor", mean_actor_loss, episodes_completed)
        writer.add_scalar("Loss/Critic", mean_critic_loss, episodes_completed)
        writer.add_scalar("Loss/Entropy", mean_entropy, episodes_completed)
        # Per-head, both raw and as a fraction of that head's own maximum --
        # the fraction is what makes "is this head collapsing" readable at a
        # glance regardless of how many actions the head has.
        mean_ent_card = np.mean(ent_card_log)
        mean_ent_place = np.mean(ent_place_log)
        # --- entropy controller step (see ENTROPY_TARGET_* above) ---
        # Already per-step fractions of the REACHABLE maximum -- see the
        # normalization block in the update loop.
        card_frac = mean_ent_card
        place_frac = mean_ent_place
        # Measured on the anneal clock SINCE THE LAST STALL RE-BOOST, not on
        # the raw episode counter. Until 2026-08-09 this read
        # episodes_completed, which made the stall re-boost dead code: the
        # only state it updates is entropy_reboost_episode, and the only thing
        # that reads THAT is current_entropy_coef -- which the note at the top
        # of the loop already records as feeding nothing into the loss. So the
        # re-boost printed ">>> ... re-boosting exploration" and changed
        # nothing. It fired twice in the pipeline-2 run to ep 17,530, at a
        # measured pool win rate of 0.49 both times, and the placement entropy
        # target carried on annealing straight through both.
        #
        # Backward compatible: entropy_reboost_episode is 0 on a fresh run, so
        # this is identical to the old expression until a re-boost fires.
        #
        # Deliberately NOT paired with a change to ENTROPY_COEF_FLOOR or
        # ENTROPY_ADAPT_RATE. Measured over 478 updates of that same run,
        # placement entropy sat ABOVE its target in 99.4% of them, mean +0.127
        # -- 42 effective cells of 612. The coefficient is pinned at the floor
        # because the controller is trying to REDUCE exploration and 0.01 is
        # the least it may apply, so raising the floor would push against the
        # anneal schedule rather than rescue a collapsed head. What a re-boost
        # should do is stop the target sharpening during a stall, which is
        # exactly what this restores.
        #
        # NOTE (2026-08-11): that "99.4% above target" measurement is itself a
        # reading of the BROKEN metric -- it was mostly no-op steps. It is left
        # here because the re-boost reasoning it supports still stands, but it
        # is not evidence about the real placement policy.
        ent_target_place = placement_entropy_target(
            episodes_completed - entropy_reboost_episode)
        def _adapt_coef(coef, rate, target, measured, ceil):
            """One controller step: gain, then a hard per-update step limit,
            then the absolute bounds.

            The step limit is what makes this safe against a future change to
            the target or the measurement. Gain alone is not: any multiplicative
            controller compounds, so a signal that suddenly reads far from
            target walks the coefficient exponentially before the policy can
            respond. Capping the RATIO bounds that walk at
            (1 +/- STEP_MAX)^n_updates regardless of how wrong the error is.
            """
            step = math.exp(rate * (target - measured))
            step = min(1.0 + ENTROPY_COEF_STEP_MAX,
                       max(1.0 - ENTROPY_COEF_STEP_MAX, step))
            return float(np.clip(coef * step, ENTROPY_COEF_FLOOR, ceil))

        ent_coef_card = _adapt_coef(ent_coef_card, ENTROPY_ADAPT_RATE,
                                    ENTROPY_TARGET_CARD, card_frac,
                                    ENTROPY_COEF_CEIL)
        ent_coef_place = _adapt_coef(ent_coef_place, ENTROPY_ADAPT_RATE_PLACEMENT,
                                     ent_target_place, place_frac,
                                     ENTROPY_COEF_CEIL_PLACEMENT)
        writer.add_scalar("Policy/Entropy_Coef_Card", ent_coef_card, episodes_completed)
        writer.add_scalar("Policy/Entropy_Coef_Placement", ent_coef_place, episodes_completed)
        writer.add_scalar("Loss/Entropy_Card", mean_ent_card, episodes_completed)
        writer.add_scalar("Loss/Entropy_Placement", mean_ent_place, episodes_completed)
        writer.add_scalar("Policy/Entropy_Card_Frac", mean_ent_card, episodes_completed)
        writer.add_scalar("Policy/Entropy_Placement_Frac", mean_ent_place, episodes_completed)
        writer.add_scalar("Loss/Total", mean_total_loss, episodes_completed)
        # Auxiliary elixir head, reported in ELIXIR UNITS so it is directly
        # interpretable: MAE is "how many elixir off is our estimate of what
        # the opponent is holding". A always-guess-the-mean baseline sits near
        # the spread of opponent elixir (~2.5); anything meaningfully below
        # that means the recurrent state genuinely learned to count.
        writer.add_scalar("Aux/OppElixir_MAE", float(np.mean(aux_maes)), episodes_completed)
        # The annealed target next to the measured value, so "is the controller
        # fighting the policy" stays answerable at a glance -- that comparison
        # is exactly what diagnosed the fixed-target pathology in the first place.
        writer.add_scalar("Entropy/Placement_Target", ent_target_place, episodes_completed)
        # NOTE: since 2026-08-11 this is measured over REAL placements only (see
        # mb_placed). It is not comparable to the same series before that date,
        # which averaged in no-op steps and read roughly 0.37 higher.
        writer.add_scalar("Entropy/Placement_Measured", place_frac, episodes_completed)
        writer.add_scalar("Entropy/Placement_Coverage",
                          float(np.mean(coverage_ents)), episodes_completed)
        # The cure's own instrumentation. Advisor_KL is the quantity being
        # minimized; Advisor_Rows says how much of the batch it applied to.
        # Neither is interpretable alone -- KL can fall simply because the
        # advisor stopped speaking.
        writer.add_scalar("Advisor/KL", float(np.mean(coverage_kls)),
                          episodes_completed)
        writer.add_scalar("Advisor/Rows", float(np.mean(coverage_hits)),
                          episodes_completed)
        writer.add_scalar("Advisor/Coef", AT.ADVISOR_COVERAGE_COEF,
                          episodes_completed)
        # Proof that the anneal wired in on 2026-08-14 actually runs.
        writer.add_scalar("Shaping/SpellValueWeight",
                          spell_value_weight(episodes_completed),
                          episodes_completed)
        if ent_place_noop_log:
            # Kept purely as the contrast that makes the fix legible: if these
            # two ever converge, the no-op arm stopped being a free ride.
            writer.add_scalar("Entropy/Placement_Measured_NoOp",
                              float(np.mean(ent_place_noop_log)), episodes_completed)
        # Per-card conditional placement entropy -- the detector the aggregate
        # cannot be. A card sitting near 0 here is pinned to one cell no matter
        # how healthy Entropy/Placement_Measured looks.
        if percard_place_ent:
            per_card = {cid: float(np.mean(v)) for cid, v in percard_place_ent.items()}
            for cid, val in per_card.items():
                writer.add_scalar(f"Entropy/Placement_ByCard/{_card_name(cid)}",
                                  val, episodes_completed)
            worst_id = min(per_card, key=per_card.get)
            writer.add_scalar("Entropy/Placement_ByCard_Min", per_card[worst_id],
                              episodes_completed)
            shown = sorted(per_card.items(), key=lambda kv: kv[1])
            line = ("     H(place|card): "
                    + "  ".join(f"{_card_name(c)[:9]} {v:.3f}" for c, v in shown))
            if ent_place_noop_log:
                line += f"  | no-op arm {float(np.mean(ent_place_noop_log)):.3f}"
            print(line)
        writer.add_scalar("Aux/OppElixir_MSE", float(np.mean(aux_losses)), episodes_completed)
        writer.add_scalar("Loss/Clip_Fraction", mean_clip_frac, episodes_completed)
        print(f"  >> Update @ ep {episodes_completed} | Actor: {mean_actor_loss:.5f} | "
              f"Critic: {mean_critic_loss:.5f} | Entropy: {mean_entropy:.4f} | "
              f"ClipFrac: {mean_clip_frac:.4f} | OppElixirMAE: {float(np.mean(aux_maes)):.2f}")

        obs_buffer.clear()
        card_actions_buffer.clear()
        placement_actions_buffer.clear()
        coverage_slot_buffer.clear()
        coverage_target_buffer.clear()
        coverage_has_buffer.clear()
        decision_buffer.clear()
        hx_in_buffer.clear()
        cx_in_buffer.clear()
        logprobs_buffer.clear()
        values_buffer.clear()
        rewards_buffer.clear()
        masks_buffer.clear()
        valid_buffer.clear()
        aux_elixir_buffer.clear()
        boot_nonterminal_buffer.clear()
        trunc_flag_buffer.clear()
        trunc_boot_buffer.clear()

        hx, cx = hx.detach(), cx.detach()

        if episodes_completed - last_save_ep >= 500:
            torch.save({
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episodes_completed": episodes_completed,
                "entropy_reboost_episode": entropy_reboost_episode,
                "last_stall_reboost_episode": last_stall_reboost_episode,
                "last_eval_ep": last_eval_ep,
                "reference_roster": reference_roster,
                "outcome_history": list(outcome_history),
                # Adaptive entropy controller + exploiter schedule. Without
                # these a resume silently restarts both -- see the resume block.
                "ent_coef_card": ent_coef_card,
                "ent_coef_place": ent_coef_place,
                "last_exploiter_burst_episode": last_exploiter_burst_ep,
                "exploiter_burst_index": exploiter_burst_index,
            }, WEIGHT_PATH)
            print(f">>> Checkpoint saved to {WEIGHT_PATH} (episode {episodes_completed}, "
                  f"pool {len(historical_pool)})")
            last_save_ep = episodes_completed

        # This run's own new snapshots join the SAME shared historical pool
        # (see HISTORICAL_CHECKPOINT_DIR's comment in train.py). Refreshing
        # historical_pool here (main process) AND broadcasting it to every
        # worker keeps both PFSP sampling and the evaluation roster growing
        # as the trainee improves, without waiting for any single "current
        # opponent" to be beaten first (there isn't one anymore).
        if episodes_completed - last_historical_save_ep >= HISTORICAL_CHECKPOINT_INTERVAL_EPISODES:
            hist_path = os.path.join(
                HISTORICAL_CHECKPOINT_DIR,
                f"{int(time.time() * 1000)}_pipeline2_ep{episodes_completed:08d}.pth")
            torch.save({"model": net.state_dict()}, hist_path)
            print(f">>> Historical snapshot saved to {hist_path}")
            last_historical_save_ep = episodes_completed
            historical_pool = discover_historical_checkpoints(episodes_completed)
            envs.call("refresh_pfsp_pool", historical_pool + SCRIPTED_OPPONENTS + BUILTIN_TRAINING_OPPONENTS)

        # --- League exploiter burst ------------------------------------
        # Trains a SEPARATE agent whose only job is to beat the main agent as
        # it is right now, then drops its snapshot into the same pool the main
        # agent samples from. See exploiter.py for why the existing pool could
        # not produce this pressure on its own: every neural opponent in it is
        # a past self, so self-play was free to cycle rather than improve --
        # which is what the flat behavioural diagnostics of the previous run
        # actually showed.
        #
        # Note the exploiter's snapshot is deliberately eligible IMMEDIATELY:
        # discover_historical_checkpoints only age-gates filenames matching
        # _pipeline2_ep<N>, and an exploiter snapshot is named differently on
        # purpose. Age-gating it would defeat the point -- its whole value is
        # that it targets the CURRENT main agent, and it goes stale as the
        # main agent patches the hole, not as it gets older.
        if exploiter_mod.should_run_burst(episodes_completed, last_exploiter_burst_ep):
            print(f">>> Starting exploiter burst #{exploiter_burst_index} "
                  f"at episode {episodes_completed}...")
            exploiter_state, ex_stats = exploiter_mod.run_exploiter_burst(
                net, device, episodes_completed, DEFAULT_DECK_ABILITY_SLOTS,
                exploiter_state=exploiter_state,
                burst_index=exploiter_burst_index, writer=writer)
            exploiter_burst_index += 1
            last_exploiter_burst_ep = episodes_completed
            # Refresh so the brand-new exploiter snapshot actually enters
            # rotation; without this it would sit unused until the next
            # historical save happened to refresh the pool anyway.
            historical_pool = discover_historical_checkpoints(episodes_completed)
            envs.call("refresh_pfsp_pool", historical_pool + SCRIPTED_OPPONENTS + BUILTIN_TRAINING_OPPONENTS)

        if episodes_completed - last_eval_ep >= EVAL_INTERVAL_EPISODES:
            update_reference_roster(reference_roster, historical_pool)
            if reference_roster:
                print(f"Evaluating current policy (greedy) against {len(reference_roster)} fixed "
                      f"reference opponent(s), {EVAL_GAMES_PER_OPPONENT} games each...")
                agent_elo, per_opponent = evaluate_against_roster(
                    net, device, reference_roster, n_games=EVAL_GAMES_PER_OPPONENT)
                writer.add_scalar("Eval/Elo", agent_elo, episodes_completed)
                for i, entry in enumerate(reference_roster):
                    stats_i = per_opponent[entry["path"]]
                    writer.add_scalar(f"Eval/WinRate_vs_Anchor{i}", stats_i["score"], episodes_completed)
                print(f">>> Eval @ ep {episodes_completed}: Elo~{agent_elo:.0f} "
                      f"(vs {len(reference_roster)} fixed anchor(s) -- a comparable trend against a "
                      "never-changing roster, not a calibrated rating; see REFERENCE_ROSTER_ELO_STEP's comment)")
            last_eval_ep = episodes_completed

        if episodes_completed - last_replay_ep >= 1000 and historical_pool:
            print(f"Generating replay video for episode {episodes_completed}...")
            # Scenarios OFF: a demo replay should show a normal full game.
            test_env = MicroRoyaleSelfPlayEnv({"scenarios_enabled": False})
            # Newest/strongest known pool entry -- purely for a representative
            # demo/monitoring replay, unrelated to PFSP sampling or eval.
            demo_opponent_path = historical_pool[-1]
            test_env.set_historical_opponent(demo_opponent_path)
            t_obs, _ = test_env.reset()
            t_hx = torch.zeros(1, 256).to(device)
            t_cx = torch.zeros(1, 256).to(device)
            t_done = False
            t_decisions = []
            REPLAY_SKIP_FRAMES = 10
            while not t_done:
                t_obs_tensor = torch.tensor(t_obs, dtype=torch.float32).unsqueeze(0).to(device)
                t_mask = net.affordability_mask(t_obs_tensor)
                t_features, t_card_embeds, t_spatial = net.extract_features(t_obs_tensor)
                (t_logits, _, _, t_value,
                 (t_hx, t_cx)) = net.step_lstm_and_card(t_features, (t_hx, t_cx), t_mask)
                t_idx = Categorical(logits=t_logits).sample()
                t_place_logits = net.placement_given_card(t_hx, t_card_embeds, t_idx, t_obs_tensor, t_spatial)
                t_cell = Categorical(logits=t_place_logits).sample()
                t_x, t_y = net.cell_to_xy(t_cell)
                t_card_idx = t_idx.item()
                t_hand = test_env.game.get_hand()
                t_card_id = t_hand[t_card_idx] if t_card_idx < len(t_hand) else -1
                t_action = {
                    "card_index": np.array([t_card_idx]),
                    "target_x": np.array([t_x.item()]),
                    "target_y": np.array([t_y.item()]),
                    "activate_ability_slot1": np.array([0]),
                    "activate_ability_slot2": np.array([0]),
                }
                t_decisions.append({
                    "stateValue": t_value.item(),
                    "actionCardId": t_card_id,
                    "actionX": float(t_action["target_x"][0]),
                    "actionY": float(t_action["target_y"][0]),
                })
                t_obs, _, t_terminated, t_truncated, _ = test_env.step(t_action, skip_frames=REPLAY_SKIP_FRAMES)
                t_done = t_terminated or t_truncated
            replay_path = f"replays/selfplay_replay_ep{episodes_completed}.json"
            test_env.game.save_log(replay_path)
            annotate_replay_with_agent_info(replay_path, t_decisions, REPLAY_SKIP_FRAMES)
            last_replay_ep = episodes_completed

    writer.close()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    train_selfplay_ppo()
