import os
import sys
import json
import random
import subprocess
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from torch.distributions import Categorical, Normal
from torch.utils.tensorboard import SummaryWriter
from collections import deque

import clash_royale_env
import gym_wrapper
from model import MicroRoyaleNet

# Reward-shaping weights (dense guidance on top of the sparse +1/-1 win/loss signal).
# Kept intentionally small so the cumulative shaping over an episode stays comparable
# to - not larger than - the terminal +1/-1. Watch Reward/Episode_Shaping_Sum against
# the win-rate in TensorBoard: if the shaping sum dwarfs +/-1, lower these.
W_BLDG = 0.5     # weight on building (tower) HP swings
W_TROOPS = 0.1   # weight on troop HP swings
# Reward forcing the ENEMY to spend elixir -- the classic CR "won the trade"
# concept (e.g. a 2-elixir Skeletons stopping a 5-elixir Giant), independent of
# and additive to the damage terms above, which already separately reward/
# punish the damage itself but can't distinguish an efficient answer from a
# wasteful one.
#
# Deliberately one-sided (no symmetric -ally_elixir_spent term anymore, unlike
# this term's original version): that symmetric version taxed the agent the
# INSTANT it spent elixir, while the payoff for a good trade (troop/building
# damage) only lands gradually over many future ticks, discounted by gamma
# across a ~360-decision episode -- doing nothing at all was therefore always
# a perfectly safe, guaranteed-zero outcome. Confirmed causing exactly this in
# self-play (train_selfplay.py): both sides converged to holding elixir and
# never acting, running out the clock into a draw every game. Bad spends are
# still punished via the ally_troops_damage/ally_bldg_damage terms below (and
# the terminal loss) -- this term no longer ALSO taxes acting itself.
W_ELIXIR_TRADE = 0.15
# Continuous (not one-time) pressure against sitting on a full elixir bar --
# a real player never intentionally caps out (it wastes ongoing regen), and
# unlike DRAW_PENALTY below this is felt every single step it's true, not
# discounted away over a long episode. Same role as W_ELIXIR_TRADE's fix
# above: makes passivity actively cost something instead of being free.
ELIXIR_OVERFLOW_THRESHOLD = 9.0
W_ELIXIR_OVERFLOW = 0.1

# One-time penalty applied at episode end when the game times out without a
# decisive winner (raw engine reward ~0 at a done step). Draws don't teach the
# agent to close games, so nudge it away from stalling into the timeout on top
# of the existing +1/-1 win/loss signal. Applied once at the terminal step (not
# accumulated per-step).
#
# Raised from 0.2 -- that was too weak (and too temporally distant, heavily
# discounted by gamma over a long episode) to outweigh a whole game's worth of
# guaranteed per-step "safe to do nothing" incentive once self-play converged
# toward mutual passivity (see the elixir-trade comment above). 1.0 makes a
# draw as costly as an outright loss, matching real high-level play where a
# scoreless draw basically never happens -- someone always eventually finds
# the chip damage.
DRAW_PENALTY = 1.0

# Historical self-play (pipeline #2, train_selfplay.py) needs a library of past
# versions of this same policy to play against, weakest to strongest -- these
# are saved here as bare weights-only snapshots (never resumed-from for further
# gradient training, so no optimizer/training-state needed) periodically
# throughout THIS training run, not just at the end, so the library already
# spans a useful weak->strong range by the time pipeline #2 starts. Filenames
# are timestamp-prefixed rather than keyed by this run's own episode count --
# train_selfplay.py sorts the shared folder by save order (mtime) to build a
# single weakest-to-strongest queue, since it saves its own new (and by then
# much stronger) snapshots into the same folder as pipeline #2 progresses, and
# those two runs' episode counters aren't on the same scale.
HISTORICAL_CHECKPOINT_DIR = "historical_checkpoints"
HISTORICAL_CHECKPOINT_INTERVAL_EPISODES = 5000

# Spatial layout of the observation (must match ClashEnv.h):
# channels 0-2 ally troops (melee/ranged/tank), 3 ally buildings,
# channels 4-6 enemy troops, 7 enemy buildings, 8 river mask.
# Pulled live from the compiled engine's own exposed constants instead of a
# hardcoded copy -- a board-geometry or channel-layout change on the C++ side
# now propagates here automatically instead of silently drifting out of sync
# (confirmed painful in practice: this exact kind of drift crashed training
# more than once this project's history before these were queryable).
N_CHANNELS = clash_royale_env.ClashRoyaleEnv.NUM_CHANNELS
BOARD_H = clash_royale_env.ClashRoyaleEnv.BOARD_HEIGHT
BOARD_W = clash_royale_env.ClashRoyaleEnv.BOARD_WIDTH
SPATIAL_SIZE = N_CHANNELS * BOARD_H * BOARD_W

# Same blanket HP normalizers ClashEnv::extractObservation() divides by when
# building the observation (MAX_TROOP_HP/MAX_BUILDING_HP in ClashEnv.h) --
# reused here purely to keep the shaping magnitude identical to the old
# HP-channel-diffing version below, now that the raw damage numbers come from
# the engine's MatchStatistics (via gym_wrapper's info dict) instead. Pulled
# live for the same drift-safety reason as N_CHANNELS/BOARD_H/BOARD_W above.
MAX_TROOP_HP = clash_royale_env.ClashRoyaleEnv.MAX_TROOP_HP
MAX_BUILDING_HP = clash_royale_env.ClashRoyaleEnv.MAX_BUILDING_HP
# A full elixir bar -- a generous upper bound for what either side can spend in
# a single skip_frames-wide step (at most one or two card plays), keeping this
# term's per-step magnitude comparable to the HP-normalized damage terms above.
MAX_ELIXIR_PER_STEP = 10.0

def compute_shaping(stats, prev_stats, w_bldg=W_BLDG, w_troops=W_TROOPS, w_elixir=W_ELIXIR_TRADE,
                     w_overflow=W_ELIXIR_OVERFLOW):
    """
    Vectorized dense-reward shaping term based on per-step damage-dealt and
    elixir-spent deltas, read from the engine's authoritative MatchStatistics
    (via gym_wrapper's info dict) instead of inferred by diffing HP channels in
    the observation. Rewards damage dealt to the enemy and elixir forced out of
    them, penalizes damage taken and sitting on a near-full elixir bar.
    Returns the shaping term ONLY (num_envs,), excluding the sparse win/loss reward.
    stats / prev_stats: dict of (num_envs,) arrays, keys 'team0_troop_damage',
    'team1_troop_damage', 'team0_building_damage', 'team1_building_damage',
    'team0_elixir_spent', 'team1_elixir_spent' -- cumulative totals this match,
    team0 = ally/AI, team1 = enemy/opponent. 'team0_elixir_current' is an
    instantaneous (not cumulative) reading, only used from `stats`, never
    diffed against `prev_stats`.
    """
    if prev_stats is None:
        return np.zeros(stats["team0_troop_damage"].shape[0], dtype=np.float32)

    # These counters only ever increase within a live episode, so a negative
    # delta means the underlying env auto-reset between steps (a "phantom"
    # transition -- see valid_buffer/prev_dones at the call site), which
    # restarts them at 0 for the new episode. Clamping to >=0 makes that step
    # contribute zero shaping instead of a large bogus negative spike -- the
    # same role the old HP-diffing version's max(0, -delta) clamp played for
    # the equivalent case (HP jumping back up to full at reset).
    def delta(key):
        return np.maximum(0, stats[key] - prev_stats[key])

    enemy_troops_damage = delta("team0_troop_damage") / MAX_TROOP_HP
    ally_troops_damage = delta("team1_troop_damage") / MAX_TROOP_HP
    enemy_bldg_damage = delta("team0_building_damage") / MAX_BUILDING_HP
    ally_bldg_damage = delta("team1_building_damage") / MAX_BUILDING_HP
    enemy_elixir_spent = delta("team1_elixir_spent") / MAX_ELIXIR_PER_STEP

    ally_elixir_current = stats["team0_elixir_current"]
    overflow = np.maximum(0.0, ally_elixir_current - ELIXIR_OVERFLOW_THRESHOLD) / (10.0 - ELIXIR_OVERFLOW_THRESHOLD)

    shaping = (w_bldg * (enemy_bldg_damage - ally_bldg_damage)
               + w_troops * (enemy_troops_damage - ally_troops_damage)
               + w_elixir * enemy_elixir_spent
               - w_overflow * overflow)

    return shaping.astype(np.float32)

def building_hp_end(obs_vec):
    """Remaining normalized building HP (ally, enemy) in a single final observation --
    used as a per-episode offense/defense progress metric."""
    spatial = obs_vec[:SPATIAL_SIZE].reshape(N_CHANNELS, BOARD_H, BOARD_W)
    return float(spatial[3].sum()), float(spatial[7].sum())

def make_env():
    def _init():
        return gym_wrapper.MicroRoyaleEnv()
    return _init

def annotate_replay_with_agent_info(filepath, decisions, skip_frames):
    """Merge per-decision agent internals (critic's state value, chosen action) into
    an already-saved replay JSON, one skip_frames-wide tick window per decision, so
    the viewer can show what the network was "thinking" at any scrubbed tick without
    needing its own copy of the model."""
    with open(filepath, "r") as f:
        data = json.load(f)

    card_names = data.get("cardNames", {})
    for i, tick in enumerate(data["ticks"]):
        decision = decisions[min(i // skip_frames, len(decisions) - 1)]
        tick["stateValue"] = decision["stateValue"]
        tick["actionCardId"] = decision["actionCardId"]
        tick["actionCardName"] = card_names.get(str(decision["actionCardId"]), "No-op")
        tick["actionX"] = decision["actionX"]
        tick["actionY"] = decision["actionY"]

    with open(filepath, "w") as f:
        json.dump(data, f)

# Per-process cache of context_labels already warned about (see
# load_state_dict_flexible below) -- train_selfplay.py's PFSP calls
# set_historical_opponent, and therefore this function, on EVERY episode
# reset in EVERY worker, so without this a single genuinely-mismatched
# checkpoint floods the log with an identical line every episode for as long
# as PFSP keeps sampling it (observed in practice: one mismatched checkpoint
# alone produced ~4000 repeats of the same line). context_label already
# encodes the checkpoint path, so this naturally dedupes per unique
# checkpoint, not just per call.
_warned_mismatches = set()

def load_state_dict_flexible(net, state_dict, context_label):
    """Loads state_dict into net. Returns True on a clean, fully-matching load.

    On an architecture mismatch (e.g. a card-roster change resizing the hand
    one-hot encoding, which is the only part of MicroRoyaleNet that depends on
    NUM_CARD_IDS -- see model.py's scalar_size), falls back to loading only
    the tensors whose shape still matches, leaving the rest at their fresh
    initialization instead of crashing outright. The CNN/LSTM/action heads are
    independent of NUM_CARD_IDS, so this warm-starts on everything except the
    one incompatible layer rather than discarding a whole checkpoint (and,
    upstream of this function, an entire opponent-history library, for
    train_selfplay.py's callers) over it.

    Shared between both pipelines (train.py's own resume, and everything
    train_selfplay.py uses it for) rather than defined twice -- lives here
    since train_selfplay.py already imports from train.py, and the reverse
    would be a circular import.

    Returns False when this fallback path was taken -- the caller should NOT
    then load a paired optimizer state dict, since Adam's per-parameter
    buffers would be stale/mismatched for whatever just got reinitialized.
    """
    try:
        net.load_state_dict(state_dict)
        return True
    except RuntimeError:
        own_state = net.state_dict()
        compatible = {k: v for k, v in state_dict.items()
                      if k in own_state and v.shape == own_state[k].shape}
        skipped = sorted(set(state_dict.keys()) - set(compatible.keys()))
        own_state.update(compatible)
        net.load_state_dict(own_state)
        if context_label not in _warned_mismatches:
            _warned_mismatches.add(context_label)
            print(f"[{context_label}] Architecture mismatch -- warm-started "
                  f"{len(compatible)}/{len(state_dict)} tensor(s), re-initialized: {skipped}")
        return False

def train_ppo():
    os.makedirs("replays", exist_ok=True)
    os.makedirs(HISTORICAL_CHECKPOINT_DIR, exist_ok=True)
    weight_path = "model_weights.pth"
    resuming = os.path.exists(weight_path)
    log_dir = "runs/clash_royale_experiment"

    # 8 workers + 1 CPU-bound main process (no CUDA here, so the update step itself
    # needs real cores too) comfortably fits 12 logical processors with headroom.
    # Doubling from 4 halves the variance of the per-rollout advantage normalization
    # and the critic's return targets -- the flat/noisy Loss/Critic and the rising
    # (not falling) Loss/Entropy both pointed at that noise as the likely culprit.
    num_envs = 8
    print(f"Initializing {num_envs} Async Vectorized Environments...")
    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(num_envs)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MicroRoyaleNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=3e-4)

    # PPO Hyperparameters
    gamma = 0.99
    # Lowered from 0.95: every prior fix (num_envs, value clipping, entropy decay,
    # reward-shaping) left Loss/Critic on the same noisy, non-decreasing plateau
    # (confirmed across ~2650 real episodes post-shaping-fix). High lambda leans GAE
    # on multi-step Monte-Carlo-style returns instead of the value bootstrap; with a
    # critic that isn't converging, that keeps the ADVANTAGE TARGET itself noisy no
    # matter how the critic's own update is regularized -- which is why clipping the
    # critic's movement alone didn't help. Leaning more on the (imperfect but at
    # least consistent) bootstrap should break that loop.
    gae_lambda = 0.9
    eps_clip = 0.2
    update_timestep = 500  # Number of steps PER CORE before update
    ppo_epochs = 2         # How many times to reuse each rollout
    # Minibatches per epoch, split across the env (sequence) dimension. Each minibatch
    # replays its sequences through the LSTM, so cost ~= ppo_epochs * num_minibatches *
    # update_timestep sequential forward/backward passes. With only 4 envs a single
    # full-batch update is faster and lower-variance; raise this once num_envs grows.
    num_minibatches = 1
    max_grad_norm = 0.5    # Gradient clipping (stabilizes the long-horizon BPTT)

    initial_entropy_coef = 0.1
    # Raised from 0.001: stage 2 (opp 1.2x) sat with entropy pinned at the old floor
    # for 6000+ episodes (confirmed in the log) while oscillating in a stable
    # 0.50-0.68 win-rate band that stopped trending toward the 0.80 gate -- i.e. zero
    # exploration pressure for a very long stretch while stuck. A small persistent
    # floor keeps a little exploration alive instead of fully exploiting a plateaued
    # policy forever.
    min_entropy_coef = 0.01
    # 0.9995 decays over EPISODES (not updates), and at that rate reaching the floor
    # takes ~9200 episodes (0.1 * 0.9995^9200 ~= 0.001) -- far beyond any training
    # budget actually run so far. Confirmed by measurement: at episode 458 the coef
    # was still ~0.0795 (80% of initial), matching Loss/Entropy climbing instead of
    # falling and Loss/Critic never converging -- the loss was still explicitly
    # rewarding maximum entropy this whole time, so the policy never had room to
    # settle into a confident, predictable strategy for the critic to track. 0.995
    # reaches the floor by ~episode 1000 and is already down to ~2% of initial by
    # episode 300, matching realistic training budgets instead of a 9000-episode one.
    entropy_decay_rate = 0.995
    
    # Pulled live from the engine's own enforced placement bounds (a throwaway
    # instance is enough -- these don't depend on which deck is used) instead
    # of a hardcoded copy of BOARD_WIDTH-1 / riverStart-OWN_HALF_RIVER_BUFFER
    # that could silently drift if either changes on the C++ side.
    _dim_probe = clash_royale_env.ClashRoyaleEnv(list(range(8)), list(range(8)), 100)
    MAX_X = _dim_probe.get_max_placement_x()
    MAX_Y_AI = _dim_probe.get_own_half_max_y()
    del _dim_probe

    # --- Curriculum: once the agent's win-rate against the current opponent
    # settles above a threshold, escalate the opponent's elixir multiplier.
    # 1.0 = today's fully-random opponent; higher values make it play cards
    # faster/near-continuously.
    # Gradual 0.1 steps up to 1.5x, not the old 1.0 -> 1.75 -> 3.0 jump: at 1.75x the
    # agent went 0-for-2000+ episodes with zero improvement (confirmed by measurement,
    # not assumption) -- the opponent's elixir advantage was simply overwhelming at
    # that multiplier, no amount of extra training time was fixing it.
    CURRICULUM_STAGES = [
        {"opp_elixir_multiplier": 1.0, "win_rate_threshold": 0.80},
        {"opp_elixir_multiplier": 1.1, "win_rate_threshold": 0.80},
        {"opp_elixir_multiplier": 1.2, "win_rate_threshold": 0.80},
        {"opp_elixir_multiplier": 1.3, "win_rate_threshold": 0.80},
        {"opp_elixir_multiplier": 1.4, "win_rate_threshold": 0.80},
        {"opp_elixir_multiplier": 1.5, "win_rate_threshold": None},  # final stage, no further auto-advance
    ]

    # --- Phase 2: once the agent is consistently strong against the mirror-
    # deck opponent at the final curriculum stage, switch to randomized
    # opponent decks -- pipeline #1 (this file) is "beat a random-but-fixed-
    # deck opponent"; the eventual goal is a bot that beats a real player, and
    # this phase is what actually exposes it to card interactions it's never
    # seen (it only ever played its own 8 cards against themselves up to this
    # point), and doubles as an overfitting check: if mirror-deck performance
    # doesn't transfer at all, that's a sign the policy memorized this one
    # matchup rather than learning transferable play.
    # Raw win rate (not decisive), matching the CURRICULUM_STAGES gate above.
    # Also doubles as "deck mastered" gate at the per-deck curriculum's final
    # stage, below.
    PHASE2_WIN_RATE_GATE = 0.90
    # Phase 2 replays the SAME CURRICULUM_STAGES gated progression (win rate
    # threshold -> escalate elixir multiplier) against each random deck, from
    # stage 0 (1.0x, normal speed), instead of a flat elixir speed for a flat
    # episode count. This makes rotation performance-gated rather than
    # timer-gated: a deck the agent already handles well clears every stage
    # (each needs just one 100-episode window at/above threshold) and rotates
    # out quickly; a deck it has no answer for stalls at whatever stage it's
    # failing, and keeps accumulating real training time there for as long as
    # it takes -- training effort lands on whatever the agent still can't
    # handle, instead of being capped by an arbitrary count regardless of
    # difficulty. Confirmed via generalization.log that a flat count (the
    # previous design) wasn't giving the agent a real chance to adapt per deck
    # at all (see the entropy-floor finding from that analysis).
    #
    # MAX_EPISODES_PER_RANDOM_DECK is a safety valve, not the intended
    # rotation trigger: some random 8-card draws may be pathological (no real
    # win condition, or a genuinely overwhelming one) and could otherwise
    # stall forever. Deliberately generous -- "a few thousand" episodes is
    # explicitly fine for a deck that's hard-but-learnable; this only cuts in
    # for the rare deck that isn't converging at all.
    MAX_EPISODES_PER_RANDOM_DECK = 5000
    # Derived live from CardRegistry rather than a hardcoded range+exclusion
    # list -- see gym_wrapper.get_all_card_ids's comment for why that drifts
    # stale (already happened once when the roster grew past the old range(46)).
    RANDOM_DECK_POOL = gym_wrapper.get_all_card_ids()

    # Unlike phase 1 (which naturally terminates via the stage-5 + PHASE2_
    # WIN_RATE_GATE transition into phase 2), phase 2 itself has no completion
    # condition of its own -- it just keeps rotating random decks forever
    # (mastered or timed out, on to the next one), since generalization is a
    # continuous process with no natural "done" point the way a single fixed
    # curriculum is. User decision: once total episodes_completed reaches this
    # many (while in phase 2), that's judged as enough random-opponent
    # exposure to hand off to pipeline #2 (self-play/PFSP) -- at that point
    # training stops itself and automatically launches train_selfplay.py, so
    # this doesn't depend on anyone watching for the right moment.
    PHASE2_TOTAL_EPISODE_CAP = 150000

    def sample_random_deck():
        return random.sample(RANDOM_DECK_POOL, 8)

    # Defaults for a fresh run; overwritten below if resuming from a checkpoint.
    curriculum_stage = 0
    stage_start_episode = 0   # Entropy decays relative to the current stage's start (improvement #5)
    episodes_completed = 0
    # Pipeline #1's two phases: "mirror" (opponent plays the same 8-card deck)
    # then "random_opponent" (opponent plays a rotating random deck) once the
    # PHASE2_WIN_RATE_GATE is cleared at the final curriculum stage.
    phase = "mirror"
    phase_deck_episode_start = 0   # episodes_completed value when the CURRENT random deck started (phase=="random_opponent" only)
    current_random_deck = None     # the random deck currently in play (phase=="random_opponent" only), kept for logging/resume
    deck_curriculum_stage = 0      # this deck's own progress through CURRICULUM_STAGES (phase=="random_opponent" only)
    # Per-episode outcome: +1 win, -1 loss, 0 draw (timeout). The curriculum gate uses
    # the DECISIVE win rate W/(W+L): draws say "didn't close the game", not "can't beat
    # the opponent", so they shouldn't block stage advancement.
    outcome_history = deque(maxlen=100)
    # Same outcomes, wider window -- purely for a smoother TensorBoard trend line.
    # A 100-episode decisive win rate has a sampling-noise band of roughly +/-0.1
    # around the true rate, which can look like "learning then forgetting" when it's
    # just noise; 500 episodes narrows that band enough to see real trend changes.
    outcome_history_long = deque(maxlen=500)

    # True only when a full training-state checkpoint was actually restored (so
    # episodes_completed continues from a real prior value). Gates the TensorBoard
    # log wipe below: file-exists-but-incompatible or legacy-weights-only both still
    # restart episodes_completed at 0, so they need a clean log dir just like a
    # from-scratch run -- otherwise the new run's scalars overlap/interleave with the
    # old run's at the same episode numbers and the graphs become unreadable.
    full_resume = False

    if resuming:
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        try:
            if isinstance(checkpoint, dict) and "model" in checkpoint and "optimizer" in checkpoint:
                clean_load = load_state_dict_flexible(net, checkpoint["model"], f"pipeline1 resume ({weight_path})")
                if clean_load:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                else:
                    print("Optimizer state NOT restored (architecture mismatch above) -- "
                          "starting the optimizer fresh; network weights were still warm-started where shapes matched.")
                curriculum_stage = checkpoint["curriculum_stage"]
                stage_start_episode = checkpoint["stage_start_episode"]
                episodes_completed = checkpoint["episodes_completed"]
                outcome_history = deque(checkpoint["outcome_history"], maxlen=100)
                # .get() with the "mirror" default: checkpoints saved before this
                # phase mechanism existed simply resume into phase 1, same as a
                # fresh run would start.
                phase = checkpoint.get("phase", "mirror")
                phase_deck_episode_start = checkpoint.get("phase_deck_episode_start", 0)
                current_random_deck = checkpoint.get("current_random_deck", None)
                deck_curriculum_stage = checkpoint.get("deck_curriculum_stage", 0)
                full_resume = True
                # Phase 2 runs its OWN CURRICULUM_STAGES progression (deck_curriculum_stage)
                # per random deck, independent of phase 1's curriculum_stage -- overrides
                # the stage-based branch below, which would otherwise still apply phase 1's
                # final (1.5x) multiplier regardless of where this deck's own progress is.
                if phase == "random_opponent":
                    envs.call("set_opponent_elixir_multiplier", CURRICULUM_STAGES[deck_curriculum_stage]["opp_elixir_multiplier"])
                elif curriculum_stage > 0:
                    mult = CURRICULUM_STAGES[curriculum_stage]["opp_elixir_multiplier"]
                    envs.call("set_opponent_elixir_multiplier", mult)
                if phase == "random_opponent" and current_random_deck is not None:
                    envs.call("set_opponent_deck", current_random_deck)
                print(f"Resumed from {weight_path}: episode {episodes_completed}, "
                      f"curriculum stage {curriculum_stage}, phase {phase}")
            else:
                # Legacy checkpoint: bare model state_dict, no training state to restore.
                load_state_dict_flexible(net, checkpoint, f"pipeline1 legacy resume ({weight_path})")
                print(f"Loaded legacy weights-only checkpoint from {weight_path} "
                      f"(training state starts fresh).")
        except RuntimeError:
            # Genuinely unexpected/corrupt checkpoint -- load_state_dict_flexible
            # itself already handles ordinary architecture-shape mismatches
            # (e.g. NUM_CARD_IDS growing) without raising, so reaching here means
            # something else is wrong. Keep it as a backup and start fresh.
            backup = weight_path + ".bak"
            os.replace(weight_path, backup)
            print(f"Saved weights are incompatible with the current architecture; moved to {backup}, starting fresh.")

    if not full_resume and os.path.exists(log_dir):
        import shutil
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    obs_buffer = []
    card_actions_buffer = []
    placement_actions_buffer = []
    logprobs_buffer = []
    values_buffer = []
    rewards_buffer = []
    masks_buffer = []
    valid_buffer = []   # 0 on phantom auto-reset steps (see below), 1 on real transitions

    reward_history = deque(maxlen=50)
    shaping_history = deque(maxlen=50)   # Per-episode shaping sum, to watch it vs the +/-1 terminal (improvement #6)
    # Progress metrics: game length (winning FASTER = learning to close games) and
    # remaining building HP on each side at episode end (offense/defense quality).
    ep_len_history = deque(maxlen=50)
    ally_bldg_end_history = deque(maxlen=50)
    enemy_bldg_end_history = deque(maxlen=50)

    obs, _ = envs.reset()
    prev_stats = None
    prev_dones = np.zeros(num_envs, dtype=bool)   # Whether each env was reset on the previous step
    hx = torch.zeros(num_envs, 256).to(device)
    cx = torch.zeros(num_envs, 256).to(device)

    last_save_ep = episodes_completed
    last_replay_ep = episodes_completed
    # Not resume-tracked (unlike last_save_ep/last_replay_ep) -- these are
    # best-effort snapshots for pipeline #2's opponent library, not correctness-
    # critical, so losing track across a resume (saving one a bit early/late
    # near the boundary) is harmless.
    last_historical_save_ep = episodes_completed
    ep_rewards = np.zeros(num_envs)
    ep_shaping = np.zeros(num_envs)
    ep_steps = np.zeros(num_envs, dtype=np.int64)
    
    print(f"Training started on {num_envs} CPU cores simultaneously!")
    
    # Raised from 50000: that cap was hit mid-session while training was still
    # working well (cleared the entire curriculum, stages 0-5, right around the old
    # cap) -- extending it so remaining time isn't wasted on an arbitrary limit.
    # Second condition: see PHASE2_TOTAL_EPISODE_CAP's own comment -- phase 2
    # has no natural stopping point, so this is what actually ends the run.
    while episodes_completed < 1000000 and not (
            phase == "random_opponent" and episodes_completed >= PHASE2_TOTAL_EPISODE_CAP):
        # Entropy decays within each curriculum stage, not over all time: advancing a
        # stage resets the clock (stage_start_episode) so exploration is boosted again
        # for the new, harder opponent instead of staying collapsed (improvement #5).
        episodes_in_stage = episodes_completed - stage_start_episode
        current_entropy_coef = max(min_entropy_coef, initial_entropy_coef * (entropy_decay_rate ** episodes_in_stage))

        # Hidden state at the start of this rollout; needed to correctly replay
        # the LSTM sequences during the multi-epoch update.
        hx0 = hx.detach().clone()
        cx0 = cx.detach().clone()

        for step in range(update_timestep):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

            # Rollout is pure data collection - no gradients here. Gradients are
            # produced later by replaying these transitions with the current params.
            with torch.no_grad():
                # Autoregressive placement: card must actually be SAMPLED before
                # placement can be conditioned on it, so this can't be a single
                # net(...) call -- see model.py's own comment on why
                # forward_from_features (which takes card_idx already known)
                # doesn't fit the rollout case.
                features, card_embeds = net.extract_features(obs_tensor)
                card_logits, state_value, (hx, cx) = net.step_lstm_and_card(features, (hx, cx))

                card_dist = Categorical(logits=card_logits)
                card_idx = card_dist.sample()

                placement_mean, placement_log_std = net.placement_given_card(hx, card_embeds, card_idx)
                placement_dist = Normal(placement_mean, placement_log_std.exp())
                placement_sample = placement_dist.sample()

                placement_logprob = placement_dist.log_prob(placement_sample).sum(dim=-1)
                total_logprob = card_dist.log_prob(card_idx) + placement_logprob

            placement_clamped = torch.clamp(placement_sample, 0.0, 1.0)
            target_x = placement_clamped[:, 0] * MAX_X
            target_y = placement_clamped[:, 1] * MAX_Y_AI

            action = {
                "card_index": card_idx.cpu().numpy(),
                "target_x": target_x.cpu().numpy().reshape(num_envs, 1),
                "target_y": target_y.cpu().numpy().reshape(num_envs, 1),
                # No network head samples this yet (see gym_wrapper.py's own
                # comment on the same key) -- always "don't activate". Must
                # still be present: AsyncVectorEnv's Dict-space iteration
                # requires every action_space key to exist in the dict, it
                # doesn't fall back to a default like MicroRoyaleEnv.step()'s
                # own action.get("activate_ability", 0) does for a direct
                # (non-vectorized) call.
                "activate_ability": np.zeros(num_envs, dtype=np.int64),
            }

            next_obs, step_rewards, terminateds, truncateds, infos = envs.step(action)
            dones = terminateds | truncateds

            # Draw = episode ended (done) with a near-zero raw reward (same +1/-1/~0
            # convention used for outcome_history below). gym_wrapper never sets
            # truncated=True, so this is the only way to tell "timed out" apart from
            # "mid-episode" from here.
            is_draw = dones & (np.abs(step_rewards) < 0.5)
            draw_penalty = DRAW_PENALTY * is_draw.astype(np.float32)

            # gymnasium's info-batching only creates a key at all if at least one env
            # actually reported it this step (see AsyncVectorEnv._add_info) --
            # gym_wrapper's reset() returns {} for info, so on the (rare, but
            # real: e.g. several envs timing out at the same tick early in
            # training) step where EVERY env happens to auto-reset at once, the
            # whole key is simply absent rather than present with defaults.
            # Falling back to zeros here is safe: the same "cumulative counter
            # can't be smaller than last step" clamp in compute_shaping() below
            # already turns that into a delta of exactly 0, identical to how a
            # single reset env's own (zero-filled) slot is already handled.
            zeros = np.zeros(num_envs, dtype=np.int64)
            zeros_f = np.zeros(num_envs, dtype=np.float32)  # elixir_spent is a float (card costs), not an int count
            stats = {
                "team0_troop_damage": infos.get("team0_troop_damage", zeros),
                "team1_troop_damage": infos.get("team1_troop_damage", zeros),
                "team0_building_damage": infos.get("team0_building_damage", zeros),
                "team1_building_damage": infos.get("team1_building_damage", zeros),
                "team0_elixir_spent": infos.get("team0_elixir_spent", zeros_f),
                "team1_elixir_spent": infos.get("team1_elixir_spent", zeros_f),
                # Instantaneous elixir reading (not cumulative) -- feeds the
                # overflow-penalty term in compute_shaping(). infos["elixir"]
                # is a scalar per env from gym_wrapper.MicroRoyaleEnv.step().
                "team0_elixir_current": infos.get("elixir", zeros_f),
            }

            # Dense shaping term. On the step right after an episode ended, the vector env
            # has auto-reset that env, so prev_stats belongs to the finished episode and the
            # damage-dealt delta would be a huge spurious negative spike (fresh all-zero
            # counters vs the finished episode's accumulated totals). Zero the shaping
            # there so only the real +/-0 reset reward remains.
            shaping = compute_shaping(stats, prev_stats)
            shaping = shaping * (1.0 - prev_dones)
            shaped_rewards = step_rewards + shaping - draw_penalty
            ep_rewards += shaped_rewards
            ep_shaping += shaping
            ep_steps += 1

            # prev_dones marks envs whose PREVIOUS step ended the episode. Under
            # gymnasium's NEXT_STEP autoreset (AsyncVectorEnv's default), such envs
            # don't execute the sampled action this step at all -- the worker just
            # calls reset() and returns (fresh obs, reward=0, terminated=False,
            # truncated=False), silently discarding whatever card/placement the
            # policy sampled from the OLD episode's final board (which is what `obs`
            # still held this step). So this step is not a real transition: mark it
            # invalid (excluded from the PPO loss below) and also treat it as a
            # trajectory break (mask=0) so GAE doesn't bootstrap through it and the
            # LSTM state carried into the new episode's real first step starts clean
            # instead of being contaminated by the dead board.
            valid = torch.tensor(1.0 - prev_dones, dtype=torch.float32).to(device)
            mask = torch.tensor(1.0 - dones, dtype=torch.float32).to(device) * valid

            # Store the transition (everything detached) for the PPO update
            obs_buffer.append(obs_tensor)
            card_actions_buffer.append(card_idx)
            placement_actions_buffer.append(placement_sample)
            logprobs_buffer.append(total_logprob)
            values_buffer.append(state_value.squeeze(-1))
            rewards_buffer.append(torch.tensor(shaped_rewards, dtype=torch.float32).to(device))
            masks_buffer.append(mask)
            valid_buffer.append(valid)

            # Reset hidden states for environments whose episode just ended
            mask_tensor = mask.unsqueeze(1)
            hx = hx * mask_tensor
            cx = cx * mask_tensor

            for i, done in enumerate(dones):
                if done:

                    # Log the episode reward
                    reward_history.append(ep_rewards[i])
                    shaping_history.append(ep_shaping[i])
                    ep_len_history.append(ep_steps[i])
                    # next_obs at the done step is the episode's true final board
                    # (gymnasium next-step autoreset), so end-state metrics read from it.
                    ally_end, enemy_end = building_hp_end(next_obs[i])
                    ally_bldg_end_history.append(ally_end)
                    enemy_bldg_end_history.append(enemy_end)
                    ep_rewards[i] = 0
                    ep_shaping[i] = 0
                    ep_steps[i] = 0
                    episodes_completed += 1

                    # step_rewards carries the raw (unshaped) engine reward: +1 win, -1 loss, 0 timeout/draw
                    if step_rewards[i] > 0.5:
                        outcome_value = 1
                    elif step_rewards[i] < -0.5:
                        outcome_value = -1
                    else:
                        outcome_value = 0
                    outcome_history.append(outcome_value)
                    outcome_history_long.append(outcome_value)

                    if episodes_completed % 10 == 0:
                        outcomes = np.array(outcome_history)
                        wins = int((outcomes == 1).sum())
                        losses = int((outcomes == -1).sum())
                        draws = int((outcomes == 0).sum())
                        n = len(outcomes)
                        decided = wins + losses
                        decisive_wr = wins / decided if decided > 0 else 0.0
                        avg_reward = np.mean(reward_history)
                        avg_shaping = np.mean(shaping_history)
                        deck_stage_str = f"/{deck_curriculum_stage}" if phase == "random_opponent" else ""
                        print(f"Episodes: {episodes_completed} | Avg(50): {avg_reward:.2f} | W/L/D: {wins/n:.2f}/{losses/n:.2f}/{draws/n:.2f} | Decisive: {decisive_wr:.2f} | Stage: {curriculum_stage}{deck_stage_str} | Phase: {phase} | Entropy: {current_entropy_coef:.4f}")
                        writer.add_scalar("Training/Avg_Reward_50", avg_reward, episodes_completed)
                        writer.add_scalar("Reward/Episode_Shaping_Sum", avg_shaping, episodes_completed)
                        writer.add_scalar("Training/Win_Rate_100", wins / n, episodes_completed)
                        writer.add_scalar("Training/Phase", 0 if phase == "mirror" else 1, episodes_completed)
                        # Full outcome split: a rising win share should come out of the
                        # LOSS share (getting stronger) or the DRAW share (closing games).
                        writer.add_scalar("Rates/Loss_100", losses / n, episodes_completed)
                        writer.add_scalar("Rates/Draw_100", draws / n, episodes_completed)
                        # The headline progress metric and the curriculum gate:
                        # of the games that got decided, how many did we win?
                        writer.add_scalar("Rates/Decisive_Win_100", decisive_wr, episodes_completed)
                        # How many of the last 100 games actually got decided -- low
                        # values explain why the curriculum gate (needs >=30 decided)
                        # hasn't advanced yet.
                        writer.add_scalar("Training/Decided_Count_100", decided, episodes_completed)
                        # Same decisive win rate over a 500-episode window: noisier-but-
                        # real short-term swings average out, so a genuine trend (as
                        # opposed to sampling noise around a plateau) is visible here.
                        outcomes_long = np.array(outcome_history_long)
                        wins_long = int((outcomes_long == 1).sum())
                        losses_long = int((outcomes_long == -1).sum())
                        decided_long = wins_long + losses_long
                        if decided_long > 0:
                            writer.add_scalar("Rates/Decisive_Win_500", wins_long / decided_long, episodes_completed)
                        # Learning to close games shows up as shorter episodes and less
                        # enemy building HP left standing at the end.
                        writer.add_scalar("Progress/Episode_Length_50", np.mean(ep_len_history), episodes_completed)
                        writer.add_scalar("Progress/Enemy_Building_HP_End_50", np.mean(enemy_bldg_end_history), episodes_completed)
                        writer.add_scalar("Progress/Ally_Building_HP_End_50", np.mean(ally_bldg_end_history), episodes_completed)
                        writer.add_scalar("Training/Curriculum_Stage", curriculum_stage, episodes_completed)
                        writer.add_scalar("Training/Entropy_Coef", current_entropy_coef, episodes_completed)

                    # --- Curriculum advancement: escalate the opponent once the agent
                    # actually WINS most games -- raw win rate (wins / all 100 games in
                    # the window), not decisive rate (wins / decided). Decisive rate lets
                    # a high draw rate hide a mediocre bot (e.g. 40% win / 13% loss / 47%
                    # draw reads as 75% decisive while only actually winning 40% of games)
                    # ---
                    stage_threshold = CURRICULUM_STAGES[curriculum_stage]["win_rate_threshold"]
                    if (stage_threshold is not None
                            and len(outcome_history) == outcome_history.maxlen
                            and curriculum_stage + 1 < len(CURRICULUM_STAGES)):
                        outcomes = np.array(outcome_history)
                        wins = int((outcomes == 1).sum())
                        win_rate = wins / len(outcomes)
                        if win_rate >= stage_threshold:
                            curriculum_stage += 1
                            new_multiplier = CURRICULUM_STAGES[curriculum_stage]["opp_elixir_multiplier"]
                            envs.call("set_opponent_elixir_multiplier", new_multiplier)
                            outcome_history.clear()
                            stage_start_episode = episodes_completed   # Reset entropy decay clock -> re-boost exploration (improvement #5)
                            print(f">>> Curriculum advanced to stage {curriculum_stage} (opp_elixir_multiplier={new_multiplier}) - entropy re-boosted")
                            writer.add_scalar("Training/Curriculum_Stage", curriculum_stage, episodes_completed)

                    # --- Phase transition: once the final curriculum stage's win rate
                    # is consistently strong against the mirror-deck opponent, move to
                    # phase 2 (random opponent decks) -- see PHASE2_WIN_RATE_GATE's
                    # comment above for why. Raw win rate, matching the curriculum gate.
                    if (phase == "mirror"
                            and curriculum_stage == len(CURRICULUM_STAGES) - 1
                            and len(outcome_history) == outcome_history.maxlen):
                        outcomes = np.array(outcome_history)
                        wins = int((outcomes == 1).sum())
                        win_rate = wins / len(outcomes)
                        if win_rate >= PHASE2_WIN_RATE_GATE:
                            phase = "random_opponent"
                            current_random_deck = sample_random_deck()
                            envs.call("set_opponent_deck", current_random_deck)
                            deck_curriculum_stage = 0
                            envs.call("set_opponent_elixir_multiplier", CURRICULUM_STAGES[0]["opp_elixir_multiplier"])
                            outcome_history.clear()
                            phase_deck_episode_start = episodes_completed
                            stage_start_episode = episodes_completed  # re-boost exploration for the new opponent variety
                            print(f">>> Phase advanced to random_opponent (deck={current_random_deck}) "
                                  f"- mirror win rate {win_rate:.2f} reached the {PHASE2_WIN_RATE_GATE} gate")
                            writer.add_scalar("Training/Phase", 1, episodes_completed)

                    # --- Phase 2 per-deck curriculum: the SAME gated stage progression
                    # as phase 1 (win-rate threshold -> escalate elixir multiplier),
                    # replayed fresh against each random deck -- see MAX_EPISODES_PER_
                    # RANDOM_DECK's comment above for why rotation is performance-gated
                    # (deck mastered, or the safety-valve timeout) instead of a flat count.
                    elif phase == "random_opponent":
                        win_rate = None
                        if len(outcome_history) == outcome_history.maxlen:
                            win_rate = int((np.array(outcome_history) == 1).sum()) / len(outcome_history)

                        at_final_stage = deck_curriculum_stage == len(CURRICULUM_STAGES) - 1
                        mastered = at_final_stage and win_rate is not None and win_rate >= PHASE2_WIN_RATE_GATE
                        timed_out = episodes_completed - phase_deck_episode_start >= MAX_EPISODES_PER_RANDOM_DECK
                        deck_stage_threshold = CURRICULUM_STAGES[deck_curriculum_stage]["win_rate_threshold"]
                        can_advance = (not at_final_stage and deck_stage_threshold is not None
                                       and win_rate is not None and win_rate >= deck_stage_threshold)

                        if mastered or timed_out:
                            current_random_deck = sample_random_deck()
                            envs.call("set_opponent_deck", current_random_deck)
                            deck_curriculum_stage = 0
                            envs.call("set_opponent_elixir_multiplier", CURRICULUM_STAGES[0]["opp_elixir_multiplier"])
                            outcome_history.clear()
                            phase_deck_episode_start = episodes_completed
                            stage_start_episode = episodes_completed   # re-boost exploration for the new opponent variety
                            reason = (f"mastered it (>={PHASE2_WIN_RATE_GATE:.0%} at the final stage)" if mastered
                                      else f"hit the {MAX_EPISODES_PER_RANDOM_DECK}-episode safety cap without mastering it")
                            print(f">>> New random opponent deck: {current_random_deck} (previous deck {reason})")
                        elif can_advance:
                            deck_curriculum_stage += 1
                            new_multiplier = CURRICULUM_STAGES[deck_curriculum_stage]["opp_elixir_multiplier"]
                            envs.call("set_opponent_elixir_multiplier", new_multiplier)
                            outcome_history.clear()
                            stage_start_episode = episodes_completed   # re-boost exploration for the harder version of the SAME deck
                            print(f">>> Deck curriculum advanced to stage {deck_curriculum_stage} "
                                  f"(opp_elixir_multiplier={new_multiplier}) against current random deck")

            obs = next_obs
            prev_stats = stats
            prev_dones = dones

        # --- PPO Update: GAE advantages + multiple epochs over env-minibatches ---
        obs_seq = torch.stack(obs_buffer)                              # (T, N, obs_dim)
        card_actions_seq = torch.stack(card_actions_buffer)            # (T, N)
        placement_actions_seq = torch.stack(placement_actions_buffer)  # (T, N, 2)
        old_logprobs_seq = torch.stack(logprobs_buffer)                # (T, N)
        values_seq = torch.stack(values_buffer)                        # (T, N)  (old critic values)
        rewards_seq = torch.stack(rewards_buffer)                      # (T, N)
        masks_seq = torch.stack(masks_buffer)                          # (T, N)
        valid_seq = torch.stack(valid_buffer)                          # (T, N) -- 0 on phantom auto-reset steps

        # Watch how often the agent chooses to wait (action 4 = no-op), among real
        # steps only. Near-1.0 means it collapsed into total passivity (draw > loss);
        # near-0.0 means it still can't hold elixir. Healthy play should settle
        # somewhere in between.
        noop_frac = ((card_actions_seq == 4).float() * valid_seq).sum() / valid_seq.sum().clamp(min=1.0)
        writer.add_scalar("Policy/Noop_Fraction", noop_frac.item(), episodes_completed)

        # Bootstrap value for the state right after the last stored step --
        # value only depends on hx, never needs a card/placement at all, so
        # this skips straight past step_lstm_and_card's card_logits and never
        # touches placement (see model.py's own comment on why this split
        # exists).
        with torch.no_grad():
            next_obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            next_features, _ = net.extract_features(next_obs_tensor)
            _, next_value, _ = net.step_lstm_and_card(next_features, (hx, cx))
            next_value = next_value.squeeze(-1)

        # Generalized Advantage Estimation (GAE-lambda)
        advantages_seq = torch.zeros_like(rewards_seq)
        gae = torch.zeros(num_envs).to(device)
        for t in reversed(range(update_timestep)):
            next_val = next_value if t == update_timestep - 1 else values_seq[t + 1]
            delta = rewards_seq[t] + gamma * next_val * masks_seq[t] - values_seq[t]
            gae = delta + gamma * gae_lambda * masks_seq[t] * gae
            advantages_seq[t] = gae

        # Critic targets are the raw returns; only advantages are normalized (fix #3)
        returns_seq = advantages_seq + values_seq
        adv_norm_seq = (advantages_seq - advantages_seq.mean()) / (advantages_seq.std() + 1e-8)

        env_indices = np.arange(num_envs)
        mb_size = max(1, num_envs // num_minibatches)

        # Per-minibatch loss diagnostics, averaged and logged once per update below --
        # direct visibility into whether the network is still learning (shrinking
        # critic loss, non-collapsing clip fraction) instead of inferring it indirectly
        # from noisy episode-outcome stats.
        actor_losses, critic_losses, entropy_bonuses, total_losses, clip_fracs = [], [], [], [], []

        for epoch in range(ppo_epochs):
            np.random.shuffle(env_indices)
            for mb_start in range(0, num_envs, mb_size):
                mb_env = env_indices[mb_start:mb_start + mb_size]
                if len(mb_env) == 0:
                    continue
                mb_env_t = torch.as_tensor(mb_env, dtype=torch.long, device=device)
                mb = mb_env_t.shape[0]

                # Batch the (non-recurrent) CNN + scalar feature extraction over ALL
                # timesteps at once, then loop only the cheap LSTMCell. This avoids
                # calling the CNN update_timestep times on a tiny batch and is the main
                # speedup of the update step.
                mb_obs = obs_seq[:, mb_env_t]                                  # (T, mb, obs_dim)
                feats_seq, card_embeds_seq = net.extract_features(mb_obs.reshape(update_timestep * mb, -1))
                feats_seq = feats_seq.view(update_timestep, mb, -1)           # (T, mb, feat_dim)
                card_embeds_seq = card_embeds_seq.view(update_timestep, mb, net.hand_size + 1, -1)

                # Replay this minibatch's env sequences through the LSTM with current params
                rhx = hx0[mb_env_t]
                rcx = cx0[mb_env_t]
                new_logprobs = []
                new_values = []
                new_entropies = []
                for t in range(update_timestep):
                    # card_actions_seq[t, mb_env_t] -- the STORED action from
                    # rollout, not a fresh sample -- conditions placement here
                    # exactly like the log-prob evaluation two lines below
                    # does. Using anything else here would evaluate placement
                    # under a DIFFERENT card than the one log-prob is scored
                    # against, silently breaking the PPO ratio.
                    logits_t, mean_t, log_std_t, value_t, (rhx, rcx) = net.forward_from_features(
                        feats_seq[t], card_embeds_seq[t], (rhx, rcx), card_actions_seq[t, mb_env_t])
                    card_dist_t = Categorical(logits=logits_t)
                    place_dist_t = Normal(mean_t, log_std_t.exp())
                    lp_t = card_dist_t.log_prob(card_actions_seq[t, mb_env_t]) \
                        + place_dist_t.log_prob(placement_actions_seq[t, mb_env_t]).sum(dim=-1)
                    ent_t = card_dist_t.entropy() + place_dist_t.entropy().sum(dim=-1)
                    new_logprobs.append(lp_t)
                    new_values.append(value_t.squeeze(-1))
                    new_entropies.append(ent_t)
                    reset_t = masks_seq[t, mb_env_t].unsqueeze(1)
                    rhx = rhx * reset_t
                    rcx = rcx * reset_t

                new_logprobs = torch.stack(new_logprobs)   # (T, mb)
                new_values = torch.stack(new_values)       # (T, mb)
                new_entropies = torch.stack(new_entropies)

                mb_adv = adv_norm_seq[:, mb_env_t]
                mb_ret = returns_seq[:, mb_env_t]
                mb_old_logprobs = old_logprobs_seq[:, mb_env_t]
                mb_old_values = values_seq[:, mb_env_t]
                mb_valid = valid_seq[:, mb_env_t]
                n_valid = mb_valid.sum().clamp(min=1.0)

                ratios = torch.exp(new_logprobs - mb_old_logprobs)
                surr1 = ratios * mb_adv
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * mb_adv

                # Value clipping (PPO2-style): cap how far the critic's prediction can
                # move from its rollout-time value in a single update, same idea as the
                # policy ratio clip above. Take the worse (larger) of the clipped/
                # unclipped loss so the critic can't dodge the penalty by jumping back
                # and forth outside the trust region -- this is what actually stabilizes
                # a noisy critic instead of just producing a smoother-looking loss curve.
                value_clipped = mb_old_values + torch.clamp(new_values - mb_old_values, -eps_clip, eps_clip)
                critic_loss_unclipped = F.mse_loss(new_values, mb_ret, reduction="none")
                critic_loss_clipped = F.mse_loss(value_clipped, mb_ret, reduction="none")
                critic_loss_per_elem = torch.max(critic_loss_unclipped, critic_loss_clipped)

                # Phantom auto-reset steps (mb_valid=0) carry an action that was never
                # actually executed in the env -- excluded from every loss term instead
                # of being averaged in as if it were a real transition.
                actor_loss = -(torch.min(surr1, surr2) * mb_valid).sum() / n_valid
                critic_loss = (critic_loss_per_elem * mb_valid).sum() / n_valid
                entropy_bonus = (new_entropies * mb_valid).sum() / n_valid
                loss = actor_loss + 0.5 * critic_loss - (current_entropy_coef * entropy_bonus)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_bonuses.append(entropy_bonus.item())
                total_losses.append(loss.item())
                # Fraction of (real) samples where the PPO ratio hit the clip range --
                # near 0 means the policy barely moved this update (possible plateau/too
                # low LR), consistently high means updates may be too aggressive.
                clipped = ((ratios - 1.0).abs() > eps_clip).float()
                clip_fracs.append(((clipped * mb_valid).sum() / n_valid).item())

        mean_actor_loss = np.mean(actor_losses)
        mean_critic_loss = np.mean(critic_losses)
        mean_entropy = np.mean(entropy_bonuses)
        mean_total_loss = np.mean(total_losses)
        mean_clip_frac = np.mean(clip_fracs)
        writer.add_scalar("Loss/Actor", mean_actor_loss, episodes_completed)
        writer.add_scalar("Loss/Critic", mean_critic_loss, episodes_completed)
        writer.add_scalar("Loss/Entropy", mean_entropy, episodes_completed)
        writer.add_scalar("Loss/Total", mean_total_loss, episodes_completed)
        writer.add_scalar("Loss/Clip_Fraction", mean_clip_frac, episodes_completed)
        # Printed (not just logged to TensorBoard) so progress can be monitored from
        # the console/log file alone, without needing the TensorBoard UI open.
        print(f"  >> Update @ ep {episodes_completed} | Actor: {mean_actor_loss:.5f} | "
              f"Critic: {mean_critic_loss:.5f} | Entropy: {mean_entropy:.4f} | "
              f"ClipFrac: {mean_clip_frac:.4f}")

        obs_buffer.clear()
        card_actions_buffer.clear()
        placement_actions_buffer.clear()
        logprobs_buffer.clear()
        values_buffer.clear()
        rewards_buffer.clear()
        masks_buffer.clear()
        valid_buffer.clear()

        hx, cx = hx.detach(), cx.detach()

        # Saves
        if episodes_completed - last_save_ep >= 500:
            torch.save({
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episodes_completed": episodes_completed,
                "curriculum_stage": curriculum_stage,
                "stage_start_episode": stage_start_episode,
                "outcome_history": list(outcome_history),
                "phase": phase,
                "phase_deck_episode_start": phase_deck_episode_start,
                "current_random_deck": current_random_deck,
                "deck_curriculum_stage": deck_curriculum_stage,
            }, weight_path)
            print(f">>> Checkpoint saved to {weight_path} (episode {episodes_completed}, stage {curriculum_stage})")
            last_save_ep = episodes_completed

        # Historical snapshot for pipeline #2 (train_selfplay.py) -- bare
        # weights only, see HISTORICAL_CHECKPOINT_DIR's comment above.
        if episodes_completed - last_historical_save_ep >= HISTORICAL_CHECKPOINT_INTERVAL_EPISODES:
            hist_path = os.path.join(
                HISTORICAL_CHECKPOINT_DIR,
                f"{int(time.time() * 1000)}_pipeline1_ep{episodes_completed:08d}.pth")
            torch.save({"model": net.state_dict()}, hist_path)
            print(f">>> Historical snapshot saved to {hist_path}")
            last_historical_save_ep = episodes_completed

        # Generate Replay (Standalone test env to avoid corrupting async processes)
        if episodes_completed - last_replay_ep >= 1000:
            print(f"Generating replay video for episode {episodes_completed}...")
            test_env = gym_wrapper.MicroRoyaleEnv()
            # Match the standalone replay env to the actual curriculum stage in
            # progress -- otherwise it silently records against the default 1.0x
            # opponent regardless of how far training has actually advanced.
            test_env.set_opponent_elixir_multiplier(CURRICULUM_STAGES[curriculum_stage]["opp_elixir_multiplier"])
            # Same for phase 2 -- otherwise this would silently keep recording
            # mirror-deck replays even once training has moved on to random
            # opponent decks, and at the wrong (phase 1) elixir multiplier.
            if phase == "random_opponent" and current_random_deck is not None:
                test_env.set_opponent_deck(current_random_deck)
                test_env.set_opponent_elixir_multiplier(CURRICULUM_STAGES[deck_curriculum_stage]["opp_elixir_multiplier"])
            t_obs, _ = test_env.reset()
            t_hx = torch.zeros(1, 256).to(device)
            t_cx = torch.zeros(1, 256).to(device)
            t_done = False
            t_decisions = []
            REPLAY_SKIP_FRAMES = 10
            while not t_done:
                t_obs_tensor = torch.tensor(t_obs, dtype=torch.float32).unsqueeze(0).to(device)
                t_features, t_card_embeds = net.extract_features(t_obs_tensor)
                t_logits, t_value, (t_hx, t_cx) = net.step_lstm_and_card(t_features, (t_hx, t_cx))
                t_idx = Categorical(logits=t_logits).sample()
                t_norm, _ = net.placement_given_card(t_hx, t_card_embeds, t_idx)
                t_card_idx = t_idx.item()
                t_hand = test_env.game.get_hand()
                t_card_id = t_hand[t_card_idx] if t_card_idx < len(t_hand) else -1
                t_action = {
                    "card_index": np.array([t_card_idx]),
                    "target_x": np.array([torch.clamp(t_norm[0,0]*MAX_X, 0.0, MAX_X).item()]),
                    "target_y": np.array([torch.clamp(t_norm[0,1]*MAX_Y_AI, 0.0, MAX_Y_AI).item()])
                }
                t_decisions.append({
                    "stateValue": t_value.item(),
                    "actionCardId": t_card_id,
                    "actionX": float(t_action["target_x"][0]),
                    "actionY": float(t_action["target_y"][0]),
                })
                t_obs, _, t_terminated, t_truncated, _ = test_env.step(t_action, skip_frames=REPLAY_SKIP_FRAMES)
                t_done = t_terminated or t_truncated
            replay_path = f"replays/replay_ep{episodes_completed}.json"
            test_env.game.save_log(replay_path)
            annotate_replay_with_agent_info(replay_path, t_decisions, REPLAY_SKIP_FRAMES)
            last_replay_ep = episodes_completed

    # Guaranteed fresh save right at the stop point (not just whatever the
    # periodic 500-episode cadence happened to catch) -- pipeline #2 bootstraps
    # from this exact file next, so it should reflect the truly-latest trained
    # state, not one up to 500 episodes stale.
    torch.save({
        "model": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episodes_completed": episodes_completed,
        "curriculum_stage": curriculum_stage,
        "stage_start_episode": stage_start_episode,
        "outcome_history": list(outcome_history),
        "phase": phase,
        "phase_deck_episode_start": phase_deck_episode_start,
        "current_random_deck": current_random_deck,
        "deck_curriculum_stage": deck_curriculum_stage,
    }, weight_path)
    envs.close()
    writer.close()
    print(f">>> Pipeline #1 stopped at episode {episodes_completed} (phase={phase}) -- "
          f"final checkpoint saved to {weight_path}.")

    # Automatic handoff to pipeline #2 (self-play/PFSP) -- see
    # PHASE2_TOTAL_EPISODE_CAP's comment for why this doesn't wait for anyone
    # to notice and launch it manually. sys.executable guarantees the same
    # venv interpreter this script itself is running under.
    selfplay_out = open("training_selfplay_pfsp.log", "w")
    selfplay_err = open("training_selfplay_pfsp_err.log", "w")
    subprocess.Popen([sys.executable, "train_selfplay.py"], stdout=selfplay_out, stderr=selfplay_err)
    print(">>> Launched train_selfplay.py (pipeline #2) -- see training_selfplay_pfsp.log / "
          "training_selfplay_pfsp_err.log")

if __name__ == "__main__":
    # Prevent safe pickling errors in Windows multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    train_ppo()