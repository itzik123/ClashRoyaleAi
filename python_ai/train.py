import os
import math
import sys
import json
import subprocess
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import deque

import clash_royale_env
import gym_wrapper
from model import MicroRoyaleNet
from elixir_shaping import W_SOLVENCY, solvency_shaping
import advisor_target

# Reward-shaping weights (dense guidance on top of the sparse +1/-1 win/loss signal).
# Kept intentionally small so the cumulative shaping over an episode stays comparable
# to - not larger than - the terminal +1/-1. Watch Reward/Episode_Shaping_Sum against
# the win-rate in TensorBoard: if the shaping sum dwarfs +/-1, lower these.
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
# MEASURED, then cut 0.15 -> 0.03. A reward decomposition over 12 real games
# (12 wins/2 losses, mean 112 steps) put this term at +0.3585 of the DISCOUNTED
# episode return -- larger than the +0.2781 the agent got for actually winning.
# It is also the one term that rewards something orthogonal to winning: it pays
# out whenever the OPPONENT spends elixir, which efficient defence maximises
# perfectly without ever threatening a tower.
#
# That single fact explains the behaviour every other lever failed to move: the
# greedy policy never played its win condition or its spell across 47,000
# self-play episodes, and beat an ATTACKING scripted opponent 13-0-2 without one.
# The bot was optimising correctly -- for an objective that valued trading over
# winning. Kept non-zero (not deleted) because punishing bad spends is still
# useful; it just must not dominate.
W_ELIXIR_TRADE = 0.03

# DELIBERATE BIAS, and the only term here that is intentionally NOT
# potential-based. Fires once, undiscounted, each time a tower changes hands.
#
# Why it has to exist: making the tower term potential-based was mathematically
# right and strategically wrong. Potential-based shaping is policy-invariant BY
# CONSTRUCTION -- it cannot change which policy is optimal, only how fast the
# agent finds it. Measured consequence over 8,300 episodes: win-condition usage
# rose to 7.3% early and then decayed back to 0.7%, with Episode_Shaping_Sum
# hovering at ~0.0 exactly as the telescoping property predicts. Removing the
# bias revealed that against this engine's opponent, pure defence genuinely WAS
# optimal, so the agent correctly converged to it.
#
# The fix is not to un-do the PBRS term (it still gives unbiased dense guidance)
# but to add an explicit, honest bias next to it: destroying a tower is the
# thing we actually want and it should be paid for directly. Set above the
# discounted value of a win (~0.28 at these episode lengths) so taking a crown
# is never worth less than the trade that led to it.
W_TOWER_DESTROYED = 0.6
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
# Lowered 5000 -> 2000 on 2026-08-09, as a RE-DENOMINATION rather than a change
# of intent. The 2026-08-07 movement-speed fix left gradient steps per hour
# unchanged but cut episodes per hour 2,873 -> 1,301, so one episode now carries
# ~2.2x more transitions and ~2.2x more policy change. 5,000 episodes had come
# to mean what ~11,000 used to; 5000 / 2.2 ~= 2,270, rounded to 2,000.
#
# Deliberately NOT 1,000, which was the other candidate: that would make the
# pool ~5x denser than the original design, and 5,000 was itself a judgement
# call rather than a measured optimum, so there is nothing to justify
# overshooting it. Two costs bound this from above -- every extra pool member
# dilutes the PFSP share of every other (which is what already forced
# DEFENSIVE_SCRIPTED_MIN_WEIGHT up to 0.8), and each worker keeps its OWN local
# per-opponent win-rate estimate, so more members means fewer games each and a
# noisier (1 - winrate)^2 weighting.
#
# MIN_OPPONENT_AGE_EPISODES in train_selfplay.py is kept at 3x this value; see
# its comment. The two are coupled and changing one alone silently changes
# which snapshots are eligible.
HISTORICAL_CHECKPOINT_INTERVAL_EPISODES = 2000

# Diagnostic-only snapshots, one per curriculum-stage transition: the exact
# policy that just cleared a stage's 80%-over-100-episodes gate, captured
# BEFORE the opponent gets harder.
#
# Deliberately a SEPARATE directory from HISTORICAL_CHECKPOINT_DIR, not extra
# files in it: that one is pipeline #2's PFSP opponent pool, ordered by mtime
# as a weakest->strongest ladder, so dropping additional checkpoints in would
# silently change which opponents self-play samples and how often. These are
# for offline analysis only and nothing ever trains against them.
#
# The motivating question, which could not be answered on the previous run
# because the checkpoints had already been deleted: the narrow policy measured
# at stage 4 (8 of 288 placement cells, 5 of 8 deck cards, one lane) was
# measured ONLY against that stage's 1.4x-elixir opponent. Whether that
# narrowness is a pathological collapse or a rational response to being
# out-elixired is decidable -- but only by replaying ONE fixed policy against
# SEVERAL stages' opponents, which requires having kept the per-stage weights.
STAGE_CHECKPOINT_DIR = "stage_checkpoints"


def save_stage_snapshot(net, directory, stage, episodes_completed, opp_elixir_multiplier, reason):
    """Weights-only snapshot tagged with the curriculum state it was taken at.

    Weights only (no optimizer/training state) on purpose -- these are never
    resumed from, only loaded for offline behavioral probes, so carrying Adam's
    buffers would triple the file size for nothing. The metadata is what makes
    a probe reproducible: it records which opponent strength this policy was
    actually trained against, so a later comparison can replay it against a
    DIFFERENT multiplier and attribute the difference correctly.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"stage{stage}_ep{episodes_completed:08d}.pth")
    torch.save({
        "model": net.state_dict(),
        "curriculum_stage": stage,
        "episodes_completed": episodes_completed,
        "opp_elixir_multiplier": opp_elixir_multiplier,
        "reason": reason,
    }, path)
    print(f">>> Stage snapshot saved to {path} ({reason}, opp_elixir_multiplier={opp_elixir_multiplier})")
    return path

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

# --- Lethal spell cycling (heuristic 2) -------------------------------------
# Fireball's damage, read from the registry rather than copied, so a balance
# change can never leave this silently wrong. See CLAUDE.md's rule about second
# copies of engine constants in Python.
FIREBALL_CARD_ID = 7
FIREBALL_DAMAGE = float(clash_royale_env.get_card_info(FIREBALL_CARD_ID)["damage"]) \
    if "damage" in clash_royale_env.get_card_info(FIREBALL_CARD_ID) else 689.0
FIREBALL_COST = float(clash_royale_env.get_card_info(FIREBALL_CARD_ID)["cost"])
W_LETHAL_SPELL = 0.15


def lethal_spell_potential(stats, w=W_LETHAL_SPELL):
    """Phi(s): 1 when a finishing spell is genuinely available AND an enemy
    tower is inside its damage, 0 otherwise.

    STRICTLY potential-based, and it is worth being explicit about what that
    buys and what it does NOT. PBRS telescopes over an episode to
    gamma^T*Phi(s_T) - Phi(s_0); both ends are 0 here (no tower is in Fireball
    range at the start, and the game is over at the end), so this term's total
    contribution to any episode's return is EXACTLY ZERO. By Ng et al. that
    makes it policy-invariant: it cannot make the agent value Fireball more at
    the optimum, and it cannot be farmed by cycling in and out of the state.

    What it does is redistribute credit. The sparse signal for "cycle the spell
    into hand while their tower is low, then finish" is otherwise buried at the
    end of a long GAE trace; this puts a gradient on entering that state at the
    moment it becomes reachable. If the win is genuinely there, this shortens
    the path to finding it. If Fireball is genuinely negative-EV in this
    matchup (see perception/UPSTREAM_REQUESTS.md item 8), this will correctly
    change nothing -- which is the safety property, not a failure.

    All three conditions matter. Tower-in-range alone would reward states the
    agent cannot act on; requiring the card in hand and the elixir to cast it
    makes the potential track an ACTIONABLE opportunity.
    """
    hp = stats["enemy_tower_hp"]                       # (num_envs, 3) absolute
    in_range = np.any((hp > 0.0) & (hp <= FIREBALL_DAMAGE), axis=1)
    actionable = (stats["fireball_in_hand"] > 0.5) & \
                 (stats["team0_elixir_current"] >= FIREBALL_COST)
    return w * (in_range & actionable).astype(np.float32)


# --- Value Fireball (heuristic 1) -------------------------------------------
# NOT potential-based, deliberately, and therefore biasing by construction --
# the same eyes-open trade as W_TOWER_DESTROYED. That is the point: PBRS cannot
# change an optimum, and this term exists precisely to change one. It anneals to
# zero so the policy finishes trained on the true objective.
W_SPELL_VALUE_START = 0.08
W_SPELL_VALUE_FINAL = 0.0
SPELL_VALUE_ANNEAL_EPISODES = int(os.environ.get(
    "CLASH_SPELL_ANNEAL_EPISODES", 40000))
# Episode at which the anneal BEGINS. 0 reproduces the originally-intended
# schedule exactly and is what a from-scratch run wants; see the docstring for
# the only reason it is not always 0.
SPELL_VALUE_ANNEAL_START = int(os.environ.get("CLASH_SPELL_ANNEAL_START", 0))
# Elixir that must remain after a cast for its POSITIVE reward to count. Set to
# Fireball's own cost: enough to answer with one more card.
SPELL_SOLVENCY_RESERVE = 4.0


def spell_value_weight(eps_done, start=None, length=None):
    """The Fireball-value weight at `eps_done`, annealing START -> FINAL.

    WIRED IN 2026-08-14, and that is a GAMEPLAY-AFFECTING change: every win rate
    measured before it was earned under a constant w_spell = 0.08.

    It had been dead code since the term was written. Both trainers called
    `compute_shaping(stats, prev_stats, gamma=gamma)` with no `w_spell`, so the
    weight sat at `W_SPELL_VALUE_START` for the whole of training and the anneal
    the comment block above describes never ran. Nothing detected it because no
    test ever varied the argument -- `test_compute_shaping_actually_responds_to_
    w_spell` is the regression that now would.

    The anneal matters for the reason that block gives: this term is NOT
    potential-based, so it biases the optimum by construction, deliberately, and
    it has to reach zero for the policy to finish trained on the true objective.
    A term that never anneals is a permanent bias nobody chose.

    `start` slides the schedule onto a run that resumes mid-life.
    `model_weights_selfplay.pth` is at episode 64,309 against a 40,000-episode
    horizon, so a faithful wiring pins a resumed run at FINAL from its first
    step -- correct by the schedule, and it makes the anneal unobservable, which
    matters when the anneal is one of the things being validated. Both knobs are
    env-overridable (`CLASH_SPELL_ANNEAL_START`, `CLASH_SPELL_ANNEAL_EPISODES`)
    so a run can set them without editing code between arms.
    """
    start = SPELL_VALUE_ANNEAL_START if start is None else start
    length = SPELL_VALUE_ANNEAL_EPISODES if length is None else length
    frac = min(1.0, max(0.0, (eps_done - start) / float(max(1, length))))
    return W_SPELL_VALUE_START + frac * (W_SPELL_VALUE_FINAL - W_SPELL_VALUE_START)


def spell_value_shaping(stats, prev_stats, w):
    """Pays for the elixir a Fireball actually destroys, charges for casting it.

    The cast term is load-bearing and is the whole reason this is not simply
    "reward value destroyed". Rewarding only successful hits makes a WHIFFED
    Fireball cost exactly zero, and guaranteed-zero beats risky-positive -- the
    identical failure that put the Cannon in a back corner (see
    tower_potential). Charging one unit per cast makes the quantity a TRADE
    RATIO centred on break-even:

        killed 8 elixir with a 4-cost spell ->  8/4 - 1 = +1.00
        killed 3 elixir (a Minions squad)   ->  3/4 - 1 = -0.25
        killed nothing                      ->  0/4 - 1 = -1.00

    So it agrees with the measured EV rather than fighting it: the -1 trade that
    is Fireball's common case scores negative, and only genuine two-for-ones
    pay. Nothing here has to detect "bad timing" -- a mistimed cast earns its
    penalty automatically by killing nothing.

    The solvency gate is asymmetric ON PURPOSE. A good trade made while broke
    earns nothing; a bad trade costs regardless. That is what makes the
    "Fireball at our own bridge with 4 elixir left and a push incoming" case
    unprofitable at best rather than merely less profitable, which is the
    spam-failure this whole term has to avoid.
    """
    killed = np.maximum(0.0, stats["fireball_value_killed"] - prev_stats["fireball_value_killed"])
    spent = np.maximum(0.0, stats["fireball_elixir_spent"] - prev_stats["fireball_elixir_spent"])
    casts = spent / FIREBALL_COST
    traded = killed / FIREBALL_COST - casts
    solvent = (stats["team0_elixir_current"] >= SPELL_SOLVENCY_RESERVE).astype(np.float32)
    return (w * np.where(traded > 0.0, traded * solvent, traded)).astype(np.float32)


# --- elixir solvency --------------------------------------------------------
# Potential-based, therefore policy-invariant: it CANNOT change which policy is
# optimal, only how fast the agent finds it. That is the right tool here because
# the failure is credit assignment, not a mis-specified objective -- decision-
# time search optimises this same reward and gains +0.319 win rate with 87% of
# its overrides being "wait where greedy plays", so waiting more is already
# better under the current objective and the policy simply has not found it.
#
# Full derivation, the measurement, and the refuted alternative explanation
# (the card-entropy target is NOT forcing the spending -- the policy carries
# 45.1% play probability where the target only requires 18.1%) are in
# elixir_shaping.py.
#
# Both knobs are env-overridable so the term can be ablated against itself
# without editing code between arms.
# Periodic-checkpoint interval. Overridable ONLY so a short controlled run
# produces matched artifacts: a resume sets last_save_ep to the resumed episode,
# so at the 500 default an experiment shorter than 500 episodes finishes having
# written nothing at all -- and `timeout` kills the process before the
# end-of-loop save, so the whole run is unmeasurable. Leave unset for real runs.
SAVE_EVERY_EPISODES = int(os.environ.get("CLASH_SAVE_EVERY", 500))
SOLVENCY_ENABLED = os.environ.get("CLASH_SOLVENCY", "1") != "0"
SOLVENCY_COEF = float(os.environ.get("CLASH_SOLVENCY_COEF", W_SOLVENCY))


# --- placement coverage -----------------------------------------------------
# THE BUG THIS EXISTS FOR, measured 2026-08-14 on model_weights_dist_e3.pth.
#
# Both the actor loss and the placement entropy bonus flow through
# `placement_given_card` for the card that was CHOSEN and no other. A card the
# policy has stopped playing therefore receives ZERO placement gradient from
# either term, forever. Its conditional map freezes at whatever it happened to
# be and drifts only as the shared trunk moves under it.
#
# That is a self-sustaining deadlock, not a transient: the frozen cell makes the
# card worthless, worthlessness keeps the card head from selecting it, and not
# being selected keeps the head frozen. No amount of additional training escapes
# it, which is why "train it longer" had not worked.
#
# Measured, 40 greedy episodes at 1.5x opponent elixir:
#
#   card       modal cell   modal share   plays   H(place|card)
#   Cannon       (11,0)        91.0%        24        0.098
#   Fireball     (11,0)        58.4%         2        0.141
#   Giant        (11,0)        54.1%         2        0.147
#   Mini PEKKA   (14,15)       19.0%       230        0.086
#
# Mini PEKKA has the LOWEST entropy of the four and is the most-played card, so
# entropy does not separate them -- modal-cell stability across states does. And
# the collapse is not a valuation: scored by tower HP preserved over a Cannon's
# full 300-tick life across 449 threatened states, the policy's own cell saved
# 121 HP against 396 for a RANDOM legal cell (paired -274 HP, 95% CI
# [-328, -221]). A policy cannot be correctly valuing a card it places
# significantly worse than chance.
#
# Dating it: the phase-1 net from 2026-08-09 placed Cannon at (3,15) with a 9.5%
# modal share and H=0.368 -- healthy and state-dependent. The collapse appears in
# the 2026-08-11 net, bracketing the entropy-masking change of that day
# (e16cdd7, "Measure placement entropy on real placements, not on no-ops").
# That change was CORRECT for the defect it targeted, and it had an unmeasured
# side effect: the no-op steps it stopped rewarding were the only thing holding
# open the placement maps of cards that are never played. Note the signature --
# in the pre-fix net Cannon and Giant had the two HIGHEST per-card placement
# entropies; after it they have the lowest. The rank order inverted for exactly
# the unplayed cards, which is what this mechanism predicts and little else does.
#
# The fix restores a placement gradient for affordable-but-unchosen cards
# without reintroducing the measurement bug: the REPORTED
# Entropy/Placement_Measured still averages over real placements only.
#
# Cost is real: the placement head is ~41% of update time and this runs it a
# second time. Measured wall clock is in the commit message. It is the cheapest
# correct option -- covering all 4 slots every step would be ~4x, and sampling
# one slot uniformly reaches every card ~1000 times per rollout, which is ample.
#
# Overridable from the environment ONLY so the fix can be ablated against
# itself: the A/B that justifies this term sets CLASH_PLACEMENT_COVERAGE_COEF=0
# for the control arm and leaves the default for the treatment. Both arms then
# run byte-identical code, which is the only way the comparison attributes the
# difference to the term rather than to two different scripts.
PLACEMENT_COVERAGE_COEF = float(os.environ.get("CLASH_PLACEMENT_COVERAGE_COEF", 0.02))


def placement_coverage_slots(card_mask_seq, hand_size, slot_weights=None):
    """(L,B) long: one sampled AFFORDABLE hand slot per timestep.

    Rows with nothing affordable fall back to the no-op slot. Those rows are
    masked out of the coverage term anyway (mb_decision is 0 exactly there), so
    the fallback only has to be a legal index, never a meaningful one.

    `slot_weights` (L,B,hand_size) multiplies the per-slot sampling probability,
    for concentrating the coverage budget where it is needed. Left None the
    draw is uniform over affordable slots, which is the original behaviour.

    WHY THE WEIGHTS EXIST, measured 2026-08-14. The coverage budget is one slot
    per step, and the cards it has to reach -- Cannon(3), Fireball(4), Giant(5)
    -- are exactly the ones affordability hides: the agent sits under 3 elixir
    on 65.3% of decisions, so a uniform draw over AFFORDABLE slots is biased
    toward the cheap cards, which are also the ones already receiving actor
    gradient because they are the ones being played. Weighting toward the cards
    with an advisor rule spends a scarce budget on the starved cards instead.
    """
    L, B, _ = card_mask_seq.shape
    playable = card_mask_seq[..., :hand_size].reshape(L * B, hand_size).float()
    if slot_weights is not None:
        playable = playable * slot_weights.reshape(L * B, hand_size).float()
        # A row where every AFFORDABLE slot got weight 0 would be indistinguishable
        # from a row with nothing affordable. Restore the unweighted mask there so
        # the weights can only ever re-rank, never remove, a candidate.
        empty = playable.sum(-1) <= 0
        if bool(empty.any()):
            base = card_mask_seq[..., :hand_size].reshape(L * B, hand_size).float()
            playable = torch.where(empty.unsqueeze(-1), base, playable)
    none = playable.sum(-1) <= 0
    # Uniform over the affordable slots; the all-zero rows get a valid dummy
    # distribution so multinomial cannot raise, then are overwritten below.
    probs = torch.where(none.unsqueeze(-1), torch.ones_like(playable), playable)
    idx = torch.multinomial(probs, 1).squeeze(-1)
    idx = torch.where(none, torch.full_like(idx, hand_size), idx)
    return idx.view(L, B)


def tower_potential(stats, w_bldg=W_BLDG):
    """Phi(s): the TOWER-damage differential, normalized.

    This is the quantity that actually tracks progress toward winning -- towers
    only ever lose HP, and the match ends when a King Tower dies, so a rising
    differential IS the game being won. Used as the potential for the
    potential-based shaping in compute_shaping() below.

    Towers only, since 2026-08-06. This used to read team*_building_damage,
    which the engine defines as towers PLUS deployed buildings, so damage to
    the agent's own Cannon was charged at the Princess-Tower rate. That is the
    wrong price for a sacrificial card: losing the Cannon's 824 HP cost
    0.5 * 824/4008 = 0.1028, while killing a troop with it paid only
    0.1 * hp/4256 -- it had to kill 5.3x its own HP to break even. Parking it in
    a back corner cost exactly zero instead, because decay emits no damage event
    at all (Building::update). Measured on the ep-130,306 checkpoint: 27.9% of
    Cannons went behind its own King at (11,2)/(11,3), mean placement y = 6.3,
    i.e. behind its own Princess Towers.

    Deployed-building damage is not discarded -- compute_shaping() now folds it
    into the troop term, where a building is priced like any other unit that
    trades HP, making the break-even 1:1 instead of 5.3:1.
    """
    return w_bldg * (stats["team0_tower_damage"] - stats["team1_tower_damage"]) / MAX_BUILDING_HP


def compute_shaping(stats, prev_stats, gamma=0.99, w_bldg=W_BLDG, w_troops=W_TROOPS,
                     w_elixir=W_ELIXIR_TRADE, w_overflow=W_ELIXIR_OVERFLOW,
                     w_tower=W_TOWER_DESTROYED, w_spell=W_SPELL_VALUE_START):
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

    # Deployed buildings (Cannon, Tesla, ...) are priced HERE, with the troops,
    # not in the tower potential -- see tower_potential's docstring. A defensive
    # building is a unit that trades HP, so it belongs on the same scale as one:
    # this makes its break-even 1:1 (kill at least what you lose) instead of the
    # 5.3:1 that the tower rate imposed. The engine's building counter is towers
    # PLUS deployed buildings, so the deployed part is the difference.
    def deployed_building(team):
        return (stats[f"team{team}_building_damage"] - stats[f"team{team}_tower_damage"],
                prev_stats[f"team{team}_building_damage"] - prev_stats[f"team{team}_tower_damage"])

    e_now, e_prev = deployed_building(0)
    a_now, a_prev = deployed_building(1)
    enemy_troops_damage = (delta("team0_troop_damage") + np.maximum(0, e_now - e_prev)) / MAX_TROOP_HP
    ally_troops_damage = (delta("team1_troop_damage") + np.maximum(0, a_now - a_prev)) / MAX_TROOP_HP
    enemy_elixir_spent = delta("team1_elixir_spent") / MAX_ELIXIR_PER_STEP

    ally_elixir_current = stats["team0_elixir_current"]
    overflow = np.maximum(0.0, ally_elixir_current - ELIXIR_OVERFLOW_THRESHOLD) / (10.0 - ELIXIR_OVERFLOW_THRESHOLD)

    # POTENTIAL-BASED shaping for the tower term: F = gamma*Phi(s') - Phi(s).
    #
    # The previous form was w_bldg * (Phi(s') - Phi(s)) -- the same difference
    # WITHOUT the gamma. That looks almost identical and is not: Ng et al.'s
    # policy-invariance result requires the discounted form, and with gamma<1 the
    # undiscounted difference does change which policy is optimal. It was
    # therefore free to trade "win the game" against "accumulate shaping", which
    # is exactly what the measured behaviour showed.
    #
    # In the discounted form the whole episode's tower shaping telescopes to
    # gamma^T*Phi(s_T) - Phi(s_0), so it can guide the agent toward tower damage
    # without ever paying it to prolong a game for extra shaping.
    tower_shaping = gamma * tower_potential(stats, w_bldg) - tower_potential(prev_stats, w_bldg)

    # Heuristic 2, same discounted form and for the same reason -- see
    # lethal_spell_potential's docstring for why this cannot bias the optimum.
    lethal_shaping = (gamma * lethal_spell_potential(stats)
                      - lethal_spell_potential(prev_stats))

    # Discrete crown events. Counts only ever go DOWN within an episode, so a
    # negative delta is a tower falling; np.maximum(0, ...) also makes the
    # phantom post-autoreset step (counts jump back to 3) contribute nothing,
    # the same guard delta() applies to the cumulative counters above.
    towers_taken = np.maximum(0, prev_stats["team1_towers_alive"] - stats["team1_towers_alive"])
    towers_lost = np.maximum(0, prev_stats["team0_towers_alive"] - stats["team0_towers_alive"])
    tower_events = w_tower * (towers_taken - towers_lost)

    # Elixir solvency, same discounted potential-based form and for the same
    # reason -- see elixir_shaping.py for the measurement (below 3 elixir on
    # 65.3% of decisions, 60.8% during a big push) and for why the obvious
    # entropy-normalization explanation was tested and refuted.
    solvency = (solvency_shaping(stats, prev_stats, gamma, w=SOLVENCY_COEF)
                if SOLVENCY_ENABLED else 0.0)

    shaping = (tower_shaping
               + lethal_shaping
               + spell_value_shaping(stats, prev_stats, w_spell)
               + solvency
               + tower_events
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
            # Two very different situations reach this branch and only one of
            # them loses trained weights:
            #   * `skipped` non-empty -- the checkpoint carried a tensor this
            #     net cannot use. Something trained was DISCARDED.
            #   * `skipped` empty -- every tensor the checkpoint had was loaded;
            #     the net simply has parameters that postdate it (e.g. the
            #     zero-initialized `place_hires` branch added 2026-08-14, which
            #     is an exact no-op at init). Nothing trained was lost.
            # Reporting both as "re-initialized" is how a harmless load gets
            # read as a discarded placement head.
            missing = sorted(set(own_state.keys()) - set(state_dict.keys()))
            if skipped:
                print(f"[{context_label}] Architecture mismatch -- warm-started "
                      f"{len(compatible)}/{len(state_dict)} tensor(s), "
                      f"DISCARDED (shape changed): {skipped}")
            else:
                print(f"[{context_label}] Checkpoint predates this architecture "
                      f"-- all {len(compatible)} of its tensor(s) loaded; "
                      f"fresh (not in checkpoint): {missing}")
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
    # Overridable so several arms of a controlled experiment fit on one machine
    # at once. Every arm must use the SAME value -- it sets the rollout batch
    # width B, which changes advantage-normalization variance and the number of
    # BPTT segments per minibatch. Keep it a divisor-friendly value: the update
    # splits (update_timestep/bptt_chunk)*num_envs segments across
    # num_minibatches, so 4 gives 20*4/8 = 10 segments per minibatch.
    num_envs = int(os.environ.get("CLASH_NUM_ENVS", 8))
    print(f"Initializing {num_envs} Async Vectorized Environments...")
    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(num_envs)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MicroRoyaleNet(num_ability_slots=gym_wrapper.DEFAULT_DECK_ABILITY_SLOTS).to(device)
    optimizer = optim.Adam(net.parameters(), lr=3e-4)
    # This loop no longer samples/stores/scores Champion abilities at all --
    # with a Champion-less deck they were pure noise in the PPO ratio and the
    # entropy bonus (see gym_wrapper.DEFAULT_DECK_ABILITY_SLOTS). Fail loudly
    # rather than silently ignoring a Champion that IS in the deck: the net
    # would build the heads but nothing here would ever sample them, so the
    # bot would simply never use its Champion and nothing would say why.
    if gym_wrapper.DEFAULT_DECK_ABILITY_SLOTS > 0:
        raise NotImplementedError(
            "DEFAULT_DECK contains a Champion (DEFAULT_DECK_ABILITY_SLOTS > 0), but this "
            "training loop's ability sampling was removed when the deck had none. Restore "
            "the ability_dist sampling/buffering/log-prob branches before training this deck.")

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

    # --- Truncated BPTT ---------------------------------------------------
    # The update used to replay each env's WHOLE 500-step rollout through the
    # LSTM as one sequence, with num_minibatches=1 and ppo_epochs=2 -- which is
    # exactly 2 optimizer steps per 4000 collected transitions. Two measured
    # consequences:
    #   * Loss/Clip_Fraction was 0.0000 across ALL 228 updates of an 18,740-
    #     episode run. That is structural, not a plateau: epoch 0 evaluates the
    #     policy that generated the data, so its ratio is exactly 1 by
    #     construction, leaving a single Adam step before epoch 1 -- far too
    #     little movement to ever reach the 0.2 clip boundary.
    #   * Profiling put ~5.0s of the ~7s epoch in the backward pass alone,
    #     i.e. 79%, all of it unrolling one 500-long chain.
    #
    # Splitting the rollout into fixed-length chunks fixes both at once. Each
    # (chunk, env) pair becomes an independent training segment starting from
    # the hidden state that was actually stored at that timestep during the
    # rollout (see hx_in_buffer), which is the standard stored-state approach
    # for recurrent PPO. 500/25 = 20 chunks x 8 envs = 160 segments per
    # rollout, so a minibatch is a real batch instead of 8 sequences.
    #
    # This is FASTER, not just more thorough: sequential LSTMCell calls per
    # epoch drop from 500 (one 500-long chain, batch 8) to 8 x 25 = 200 (batch
    # 20 each), and the Python-loop overhead per step is amortized over a wider
    # batch. Same total timesteps processed, far fewer serial steps.
    bptt_chunk = 25
    ppo_epochs = 4
    # 160 segments / 8 = 20 segments per minibatch, 8 optimizer steps per
    # epoch, 32 per rollout -- 16x the previous 2.
    num_minibatches = 8
    max_grad_norm = 0.5    # Gradient clipping (stabilizes BPTT)
    assert update_timestep % bptt_chunk == 0,         "update_timestep must be divisible by bptt_chunk so every chunk is full-length"

    initial_entropy_coef = 0.05
    # Raised from 0.001: stage 2 (opp 1.2x) sat with entropy pinned at the old floor
    # for 6000+ episodes (confirmed in the log) while oscillating in a stable
    # 0.50-0.68 win-rate band that stopped trending toward the 0.80 gate -- i.e. zero
    # exploration pressure for a very long stretch while stuck. A small persistent
    # floor keeps a little exploration alive instead of fully exploiting a plateaued
    # policy forever.
    min_entropy_coef = 0.02
    # 0.9995 decays over EPISODES (not updates), and at that rate reaching the floor
    # takes ~9200 episodes (0.1 * 0.9995^9200 ~= 0.001) -- far beyond any training
    # budget actually run so far. Confirmed by measurement: at episode 458 the coef
    # was still ~0.0795 (80% of initial), matching Loss/Entropy climbing instead of
    # falling and Loss/Critic never converging -- the loss was still explicitly
    # rewarding maximum entropy this whole time, so the policy never had room to
    # settle into a confident, predictable strategy for the critic to track. 0.995
    # reaches the floor by ~episode 1000 and is already down to ~2% of initial by
    # episode 300, matching realistic training budgets instead of a 9000-episode one.
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
    ENTROPY_TARGET_PLACEMENT_START = 0.65
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
    
    # Pulled live from the engine's own enforced placement bounds (a throwaway
    # instance is enough -- these don't depend on which deck is used) instead
    # of a hardcoded copy of BOARD_WIDTH-1 / riverStart-OWN_HALF_RIVER_BUFFER
    # that could silently drift if either changes on the C++ side.
    # Placement coordinates no longer come from scaling a normalized [0,1]
    # sample by these bounds -- the discrete head emits a board-cell index and
    # MicroRoyaleNet.cell_to_xy() converts it to coordinates inside the
    # engine's enforced bounds by construction. The bounds are still read live
    # from the engine, but inside model.py now (PLACEMENT_ROWS/MAX_PLACEMENT_X).

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
    # Lowered 0.90 -> 0.80 to match the per-stage gate every other transition
    # uses. 0.90 was measurably a dead end: the final stage's opponent gets 1.5x
    # elixir, and the strongest policy produced so far tops out around 0.79 raw
    # win rate there after ~76k episodes of dedicated stage-5 training. So the
    # bot would have sat at stage 5 indefinitely and NEVER reached phase 2
    # (random opponent decks) or, past it, pipeline #2 self-play -- the only two
    # sources of genuinely harder opposition left, and the only path to skill
    # that transfers to a human.
    #
    # Using the same 0.80 the stage gates use is also the internally consistent
    # choice: there is no principled reason the phase transition should demand a
    # strictly higher bar than the stage transitions leading up to it.
    PHASE2_WIN_RATE_GATE = 0.80

    # Which curriculum stage is enough to leave the mirror phase. Was implicitly
    # the FINAL stage (5, opponent at 1.5x elixir); now 4.
    #
    # The reasoning is about what each remaining obstacle actually teaches. A
    # 1.5x-elixir opponent is a resource handicap that does not exist in the real
    # game -- clearing it trains "survive being out-resourced", not "play Clash
    # Royale". Random opponent decks and, past them, pipeline #2 self-play are
    # the things that teach transferable skill, and the stage-5 requirement was
    # holding the bot behind the obstacle that teaches least.
    #
    # It was also close to unreachable in practice: two 0.80 windows were needed
    # (stage 4 -> 5, then stage 5 -> phase 2), the second against a HARDER
    # opponent, while measurement put the strongest policy so far at ~0.79 peak
    # at stage 5 after ~76k episodes. Run I sat at stage 4 for 13k episodes with
    # Win_Rate_100 climbing 0.555 -> 0.624 (max 0.73) -- real progress, but on a
    # trajectory that would spend many more hours to clear a gate whose reward
    # is a harder version of an artificial handicap.
    PHASE2_MIN_CURRICULUM_STAGE = 4

    # Win rate required to LEAVE the mirror phase. Split out from
    # PHASE2_WIN_RATE_GATE, which one constant was doing two unrelated jobs for:
    # this decides "stop training against a resource-handicapped clone", while
    # PHASE2_WIN_RATE_GATE also decides "this random deck is mastered, rotate to
    # the next one" inside phase 2. Lowering one should not silently change the
    # other, and it did.
    #
    # Set to 0.60 deliberately, which the policy already clears (measured
    # 0.62-0.69 at stage 4), so the transition happens on the next full window
    # rather than after hours of grinding. That is not a lowered standard, it is
    # the recognition that this gate was never measuring the right thing:
    # beating a bot that gets 1.4x elixir and plays random cards at random
    # positions is not a prerequisite for learning from varied decks, it is a
    # different and less useful skill. The real tests come after it.
    # Overridable from the environment for CONTROLLED EXPERIMENTS only. A
    # mirror -> random_opponent flip swaps the opponent's whole deck mid-run,
    # which silently changes the task underneath any A/B that is measuring
    # something else. Setting CLASH_PHASE2_ENTRY_WIN_RATE above 1.0 makes the
    # gate unreachable and pins the run to the mirror deck for its duration.
    # Not a training knob -- leave it unset for real runs.
    PHASE2_ENTRY_WIN_RATE = float(os.environ.get("CLASH_PHASE2_ENTRY_WIN_RATE", 0.60))
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
    # stall forever. This only cuts in for the rare deck that isn't converging.
    #
    # Lowered 5000 -> 1250 on 2026-08-09, when the phase's total budget became
    # RANDOM_OPPONENT_EPISODE_BUDGET = 5000 (see below). At the old value the
    # safety valve EQUALLED the whole phase budget, so a single pathological
    # first draw could consume all of it and the agent would leave phase 1
    # having faced exactly ONE random deck -- which is the precise opposite of
    # what the phase is for. 1250 guarantees at least 4 distinct decks even in
    # the worst case. It rarely binds: rotation is normally performance-gated,
    # and mastering a deck takes ~600 episodes (6 stages x one 100-episode
    # window each), so a comfortable budget still turns over ~8 decks.
    MAX_EPISODES_PER_RANDOM_DECK = 1250
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
    # Lowered 150000 -> 40000. Phase 2's job here is narrow: confirm the policy
    # is not overfitted to the mirror matchup. That question is already answered
    # -- the greedy policy took 95% against randomly drawn opponent decks on
    # FIRST contact, before any phase-2 training, and 97.5%/93.8% against the
    # scripted opponents with randomized decks in earlier runs. There is no
    # overfitting to grind out.
    #
    # Spending the remaining ~113k episodes here would buy very little: random
    # decks vary WHAT the opponent plays but it still plays randomly at random
    # positions, so it cannot punish the degenerate no-win-condition strategy
    # the policy has settled into. Pipeline #2 self-play can, because there the
    # opponent is a frozen copy of the bot itself and actually defends.
    #
    # ---- 2026-08-09: replaced by a BUDGET measured inside the phase ----------
    # A cap on TOTAL episodes was the wrong quantity, and it was arbitrary in a
    # way that mattered: how long the agent spends against random decks depended
    # entirely on how fast it cleared the mirror curriculum. Clear the stages in
    # 3k episodes and you get 37k of random decks; take 35k and you get 5k. The
    # phase's value has nothing to do with either number.
    #
    # Kept rather than deleted, at a deliberately small budget, because the
    # measured case for deleting it does not currently hold:
    #
    #  * The "95% on first contact" evidence above predates BOTH the 2026-08-07
    #    movement-speed fix and the 2026-08-09 placement-head fix. CLAUDE.md is
    #    explicit that no win rate from before the speed fix survives it, so the
    #    overfitting question is once again unanswered.
    #  * This is the only overfitting check that runs BEFORE the handoff. The
    #    league costs 30+ hours; finding out there that the policy memorized
    #    DEFAULT_DECK is the expensive way to learn it.
    #  * It is not substitutable by scenario injection, which is a pipeline-2
    #    feature (train_selfplay.py) and never runs here. Scenario injection also
    #    TELEPORTS 1-2 units from a hand-picked list of ~13 ids onto the bridge;
    #    it never has an opponent play an unfamiliar deck through a real match,
    #    so it exercises no elixir management, no cycle and no placement pattern.
    #
    # The counter-argument is real and is why the budget is 5000 and not more:
    # pipeline 2's four scripted bots DO get randomized decks
    # (set_scripted_opponent -> sample_random_deck), and two of them hold a 0.8
    # PFSP weight floor, so random-deck exposure continues there at a far larger
    # total volume than this phase can provide. This phase is a cheap pre-flight
    # check, not the place generalization is actually learned.
    RANDOM_OPPONENT_EPISODE_BUDGET = 5000

    def sample_random_deck():
        # Correct-by-construction (not a raw random.sample over every
        # registered card id, which would routinely violate CardRegistry::
        # validateDeckSlots since Champions/Evolutions only fit some deck
        # slots) -- see sampleRandomDeck's own comment in ClashEnv.h.
        return clash_royale_env.sample_random_deck()

    # Defaults for a fresh run; overwritten below if resuming from a checkpoint.
    curriculum_stage = 0
    stage_start_episode = 0   # Entropy decays relative to the current stage's start (improvement #5)
    episodes_completed = 0
    # Pipeline #1's two phases: "mirror" (opponent plays the same 8-card deck)
    # then "random_opponent" (opponent plays a rotating random deck) once the
    # PHASE2_WIN_RATE_GATE is cleared at the final curriculum stage.
    phase = "mirror"
    phase_deck_episode_start = 0   # episodes_completed value when the CURRENT random deck started (phase=="random_opponent" only)
    # episodes_completed when the random_opponent PHASE began -- the budget in
    # RANDOM_OPPONENT_EPISODE_BUDGET is measured from here, not from episode 0.
    # Distinct from phase_deck_episode_start, which restarts on every new deck.
    random_phase_episode_start = 0
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
                # Fall back to phase_deck_episode_start, not 0: a checkpoint
                # written before this key existed was already in the phase, and
                # defaulting to 0 would make the elapsed budget look like the
                # full episode count and hand off immediately on resume.
                random_phase_episode_start = checkpoint.get(
                    "random_phase_episode_start", phase_deck_episode_start)
                current_random_deck = checkpoint.get("current_random_deck", None)
                deck_curriculum_stage = checkpoint.get("deck_curriculum_stage", 0)
                # Controller state; falls back to the seed values for
                # checkpoints written before the controller existed.
                ent_coef_card = checkpoint.get("ent_coef_card", ent_coef_card)
                ent_coef_place = checkpoint.get("ent_coef_place", ent_coef_place)
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
    placement_actions_buffer = []   # discrete board-cell indices now, not 2-vectors
    # 1 on steps where the agent actually had a CHOICE (>=2 legal actions), 0
    # where the affordability mask left only the forced no-op. On a forced step
    # the sampled action has probability exactly 1, so its log-prob is exactly
    # 0, the PPO ratio is a constant 1, and the actor/entropy terms contribute
    # exactly zero gradient -- including them in the loss DENOMINATOR anyway
    # would silently scale the actor gradient down by ~3.7x, since only ~27% of
    # steps have any affordable card (measured). Critic loss still uses `valid`:
    # the value function must be learned on every real state, choice or not.
    decision_buffer = []
    # LSTM state as it was ENTERING each timestep -- the starting state for
    # whichever truncated-BPTT chunk begins there. Captured before the forward
    # pass (so it is the post-episode-reset state from t-1) and already
    # detached, since the rollout runs under no_grad.
    hx_in_buffer = []
    cx_in_buffer = []
    logprobs_buffer = []
    values_buffer = []
    rewards_buffer = []
    masks_buffer = []
    valid_buffer = []   # 0 on phantom auto-reset steps (see below), 1 on real transitions
    # Ground-truth opponent elixir per step -- supervision for the auxiliary
    # head only, never an input. See AUX_ELIXIR_COEF.
    aux_elixir_buffer = []
    # --- advisor-targeted placement coverage --------------------------------
    # Slot drawn here, once per timestep, rather than inside the PPO epoch loop.
    # The advisor target is a function of the OBSERVATION, so it can only be
    # computed while that observation is the live one.
    coverage_slot_buffer, coverage_target_buffer, coverage_has_buffer = [], [], []
    advisor_legal = advisor_target.build_legal_table(net) if advisor_target.enabled() else {}

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
    # Second condition: see RANDOM_OPPONENT_EPISODE_BUDGET's own comment -- the
    # random-deck phase has no natural stopping point, so this is what ends
    # pipeline 1 and triggers the handoff to train_selfplay.py.
    #
    # Counted from when the phase STARTED, not from episode 0: the budget is
    # "how much random-deck exposure is enough", which has nothing to do with
    # how many episodes the mirror curriculum happened to take.
    while episodes_completed < 1000000 and not (
            phase == "random_opponent"
            and episodes_completed - random_phase_episode_start
                >= RANDOM_OPPONENT_EPISODE_BUDGET):
        # Entropy decays within each curriculum stage, not over all time: advancing a
        # stage resets the clock (stage_start_episode) so exploration is boosted again
        # for the new, harder opponent instead of staying collapsed (improvement #5).
        episodes_in_stage = episodes_completed - stage_start_episode
        # NOTE: the old decaying entropy schedule no longer drives anything --
        # the per-head coefficients are set by the controller in the update step
        # below (see ENTROPY_TARGET_*). It is kept computed ONLY so the resume
        # path and the stage bookkeeping that reference stage_start_episode keep
        # working unchanged; nothing reads it into the loss. The console and
        # TensorBoard readouts report the real adaptive coefficients instead.
        current_entropy_coef = max(min_entropy_coef, initial_entropy_coef * (entropy_decay_rate ** episodes_in_stage))

        # NOTE: the rollout-start hidden state used to be snapshotted here, as
        # the single resume point for replaying each env's whole 500-step
        # sequence. Truncated BPTT resumes each chunk from its OWN recorded
        # state (hx_in_buffer, captured every timestep below), and hx_in_seq[0]
        # is exactly what this snapshot was -- so it is redundant.

        for step in range(update_timestep):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

            # Rollout is pure data collection - no gradients here. Gradients are
            # produced later by replaying these transitions with the current params.
            # Affordability mask -- see model.py's affordability_mask docstring
            # for the measured justification (74.9% of all steps used to be a
            # card play the engine silently refused). Derived purely from
            # obs_tensor, which is exactly what gets stored in obs_buffer, so
            # the PPO update below can recompute a bit-identical mask without
            # storing it separately.
            card_mask = net.affordability_mask(obs_tensor)

            # --- advisor-targeted coverage ---------------------------------
            # See train_selfplay.py's identical block; the defect and the fix
            # are the same in both loops.
            hand_ids_now = net.hand_card_ids(obs_tensor)
            cov_w = advisor_target.slot_weights_for(hand_ids_now)
            cov_slot = placement_coverage_slots(
                card_mask.unsqueeze(0), net.hand_size,
                slot_weights=None if cov_w is None else cov_w.unsqueeze(0))[0]
            coverage_slot_buffer.append(cov_slot)
            if advisor_target.enabled():
                slot_ids = torch.cat(
                    [hand_ids_now,
                     torch.full((num_envs, 1), -1, dtype=torch.long)], dim=1)
                cov_ids = slot_ids.gather(1, cov_slot.view(-1, 1)).squeeze(1)
                tgt_np, has_np = advisor_target.targets_for_batch(
                    obs_tensor.cpu().numpy(), cov_ids.cpu().numpy(),
                    advisor_legal)
                coverage_target_buffer.append(torch.from_numpy(tgt_np))
                coverage_has_buffer.append(
                    torch.from_numpy(has_np.astype(np.float32)))

            # Captured BEFORE step_lstm_and_card advances (hx, cx) -- this is
            # the state a chunk starting at this timestep must resume from.
            hx_in_buffer.append(hx)
            cx_in_buffer.append(cx)

            with torch.no_grad():
                # Autoregressive placement: card must actually be SAMPLED before
                # placement can be conditioned on it, so this can't be a single
                # net(...) call -- see model.py's own comment on why
                # forward_from_features (which takes card_idx already known)
                # doesn't fit the rollout case.
                features, card_embeds, spatial_map = net.extract_features(obs_tensor)
                card_logits, _, _, state_value, (hx, cx) = net.step_lstm_and_card(
                    features, (hx, cx), card_mask)

                card_dist = Categorical(logits=card_logits)
                card_idx = card_dist.sample()

                # Discrete placement over whole board cells, no Gaussian and no
                # clamping -- every cell index maps to coordinates the engine
                # already accepts (see model.cell_to_xy).
                placement_logits = net.placement_given_card(hx, card_embeds, card_idx, obs_tensor, spatial_map)
                placement_dist = Categorical(logits=placement_logits)
                placement_cell = placement_dist.sample()

                total_logprob = card_dist.log_prob(card_idx) + placement_dist.log_prob(placement_cell)

            target_x, target_y = net.cell_to_xy(placement_cell)

            action = {
                "card_index": card_idx.cpu().numpy(),
                "target_x": target_x.cpu().numpy().reshape(num_envs, 1),
                "target_y": target_y.cpu().numpy().reshape(num_envs, 1),
                # Deck has no Champion -> nothing to activate. Explicit zeros
                # rather than omitted: AsyncVectorEnv's Dict-space iteration
                # requires every action_space key to be present.
                "activate_ability_slot1": np.zeros(num_envs, dtype=np.int64),
                "activate_ability_slot2": np.zeros(num_envs, dtype=np.int64),
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
                # Instantaneous elixir reading (not cumulative) -- feeds the
                # overflow-penalty term in compute_shaping(). infos["elixir"]
                # is a scalar per env from gym_wrapper.MicroRoyaleEnv.step().
                "team0_elixir_current": infos.get("elixir", zeros_f),
                # Default 3 (a full set) so the rare all-envs-reset step, where
                # gymnasium omits the key entirely, yields a zero delta rather
                # than a phantom three-crown swing.
                "team0_towers_alive": infos.get("team0_towers_alive", np.full(num_envs, 3, dtype=np.int64)),
                "team1_towers_alive": infos.get("team1_towers_alive", np.full(num_envs, 3, dtype=np.int64)),
            }

            # Dense shaping term. On the step right after an episode ended, the vector env
            # has auto-reset that env, so prev_stats belongs to the finished episode and the
            # damage-dealt delta would be a huge spurious negative spike (fresh all-zero
            # counters vs the finished episode's accumulated totals). Zero the shaping
            # there so only the real +/-0 reset reward remains.
            # gamma passed explicitly: the tower term is potential-based
            # (gamma*Phi(s') - Phi(s)) and its policy-invariance guarantee only
            # holds if this is the SAME gamma the GAE/returns use below.
            # w_spell passed explicitly since 2026-08-14. It used to be omitted,
            # which silently pinned the Fireball-value term at its START weight
            # forever instead of annealing it to zero -- see spell_value_weight.
            shaping = compute_shaping(stats, prev_stats, gamma=gamma,
                                      w_spell=spell_value_weight(episodes_completed))
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
            placement_actions_buffer.append(placement_cell)
            # >=2 legal card options means a real decision was made here.
            decision_buffer.append((card_mask.sum(dim=1) > 1).float() * valid)
            logprobs_buffer.append(total_logprob)
            values_buffer.append(state_value.squeeze(-1))
            rewards_buffer.append(torch.tensor(shaped_rewards, dtype=torch.float32).to(device))
            masks_buffer.append(mask)
            valid_buffer.append(valid)
            aux_elixir_buffer.append(
                torch.tensor(np.asarray(infos.get("opp_elixir", np.zeros(num_envs, dtype=np.float32)),
                                        dtype=np.float32)).to(device))

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
                        print(f"Episodes: {episodes_completed} | Avg(50): {avg_reward:.2f} | W/L/D: {wins/n:.2f}/{losses/n:.2f}/{draws/n:.2f} | Decisive: {decisive_wr:.2f} | Stage: {curriculum_stage}{deck_stage_str} | Phase: {phase} | EntCoef c/p: {ent_coef_card:.4f}/{ent_coef_place:.4f}")
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
                        writer.add_scalar("Training/Entropy_Coef_Card", ent_coef_card, episodes_completed)
                        writer.add_scalar("Training/Entropy_Coef_Placement", ent_coef_place, episodes_completed)

                    # --- Phase transition. Deliberately checked BEFORE curriculum
                    # advancement, and that ordering is load-bearing: the stage gate
                    # calls outcome_history.clear() when it fires, so if it ran first
                    # a window that satisfies BOTH gates would always be consumed by
                    # the stage advance and the phase transition could never see it.
                    # Reaching phase 2 would then still require an extra full window
                    # at the harder stage -- exactly the behavior this change exists
                    # to remove.
                    #
                    # Raw win rate, matching the curriculum gate.
                    phase_just_advanced = False
                    if (phase == "mirror"
                            and curriculum_stage >= PHASE2_MIN_CURRICULUM_STAGE
                            and len(outcome_history) == outcome_history.maxlen):
                        outcomes = np.array(outcome_history)
                        wins = int((outcomes == 1).sum())
                        win_rate = wins / len(outcomes)
                        if win_rate >= PHASE2_ENTRY_WIN_RATE:
                            # The end of phase 1: strongest mirror-deck policy,
                            # before random opponent decks change the problem.
                            save_stage_snapshot(
                                net, STAGE_CHECKPOINT_DIR, curriculum_stage, episodes_completed,
                                CURRICULUM_STAGES[curriculum_stage]["opp_elixir_multiplier"],
                                f"entered phase 2 at win_rate={win_rate:.2f} (end of mirror phase)")
                            phase = "random_opponent"
                            current_random_deck = sample_random_deck()
                            envs.call("set_opponent_deck", current_random_deck)
                            deck_curriculum_stage = 0
                            envs.call("set_opponent_elixir_multiplier", CURRICULUM_STAGES[0]["opp_elixir_multiplier"])
                            outcome_history.clear()
                            phase_deck_episode_start = episodes_completed
                            random_phase_episode_start = episodes_completed
                            stage_start_episode = episodes_completed  # re-boost exploration for the new opponent variety
                            phase_just_advanced = True
                            print(f">>> Phase advanced to random_opponent from stage {curriculum_stage} "
                                  f"(deck={current_random_deck}) - mirror win rate {win_rate:.2f} "
                                  f"reached the {PHASE2_ENTRY_WIN_RATE} entry threshold")
                            print(f">>> Random-deck budget: {RANDOM_OPPONENT_EPISODE_BUDGET} episodes "
                                  f"(handoff to train_selfplay.py at episode "
                                  f"{episodes_completed + RANDOM_OPPONENT_EPISODE_BUDGET})")
                            writer.add_scalar("Training/Phase", 1, episodes_completed)

                    # --- Curriculum advancement: escalate the opponent once the agent
                    # actually WINS most games -- raw win rate (wins / all 100 games in
                    # the window), not decisive rate (wins / decided). Decisive rate lets
                    # a high draw rate hide a mediocre bot (e.g. 40% win / 13% loss / 47%
                    # draw reads as 75% decisive while only actually winning 40% of games)
                    #
                    # Guarded on phase == "mirror": this block used to run in BOTH
                    # phases, which was harmless only because phase 2 previously
                    # required the final stage, making `curriculum_stage + 1 <
                    # len(CURRICULUM_STAGES)` permanently false once there. Now that
                    # phase 2 can start from an earlier stage, an unguarded block
                    # would keep escalating curriculum_stage DURING phase 2 and fight
                    # the per-deck curriculum below for control of the opponent's
                    # elixir multiplier.
                    if phase == "mirror" and not phase_just_advanced:
                        stage_threshold = CURRICULUM_STAGES[curriculum_stage]["win_rate_threshold"]
                        if (stage_threshold is not None
                                and len(outcome_history) == outcome_history.maxlen
                                and curriculum_stage + 1 < len(CURRICULUM_STAGES)):
                            outcomes = np.array(outcome_history)
                            wins = int((outcomes == 1).sum())
                            win_rate = wins / len(outcomes)
                            if win_rate >= stage_threshold:
                                # Snapshot BEFORE incrementing: captures the policy
                                # that actually cleared THIS stage, still labeled with
                                # the multiplier it trained against.
                                save_stage_snapshot(
                                    net, STAGE_CHECKPOINT_DIR, curriculum_stage, episodes_completed,
                                    CURRICULUM_STAGES[curriculum_stage]["opp_elixir_multiplier"],
                                    f"cleared stage {curriculum_stage} gate at win_rate={win_rate:.2f}")
                                curriculum_stage += 1
                                new_multiplier = CURRICULUM_STAGES[curriculum_stage]["opp_elixir_multiplier"]
                                envs.call("set_opponent_elixir_multiplier", new_multiplier)
                                outcome_history.clear()
                                stage_start_episode = episodes_completed
                                print(f">>> Curriculum advanced to stage {curriculum_stage} (opp_elixir_multiplier={new_multiplier})")
                                writer.add_scalar("Training/Curriculum_Stage", curriculum_stage, episodes_completed)

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
        placement_actions_seq = torch.stack(placement_actions_buffer)  # (T, N) -- board-cell indices
        old_logprobs_seq = torch.stack(logprobs_buffer)                # (T, N)
        values_seq = torch.stack(values_buffer)                        # (T, N)  (old critic values)
        rewards_seq = torch.stack(rewards_buffer)                      # (T, N)
        masks_seq = torch.stack(masks_buffer)                          # (T, N)
        valid_seq = torch.stack(valid_buffer)                          # (T, N) -- 0 on phantom auto-reset steps
        aux_elixir_seq = torch.stack(aux_elixir_buffer)                # (T, N) -- opponent elixir ground truth
        decision_seq = torch.stack(decision_buffer)                    # (T, N) -- 1 only where a real choice existed
        hx_in_seq = torch.stack(hx_in_buffer)                          # (T, N, 256) -- chunk resume states
        cx_in_seq = torch.stack(cx_in_buffer)
        coverage_slot_seq = torch.stack(coverage_slot_buffer)          # (T, N)
        if advisor_target.enabled():
            coverage_target_seq = torch.stack(coverage_target_buffer)  # (T,N,cells)
            coverage_has_seq = torch.stack(coverage_has_buffer)        # (T, N)
        else:
            coverage_target_seq = None
            coverage_has_seq = torch.zeros_like(coverage_slot_seq, dtype=torch.float32)

        # Watch how often the agent chooses to wait (no-op) when it ACTUALLY had
        # a choice -- restricted to decision steps, because counting the forced
        # no-ops (~73% of all steps, where nothing was affordable) would make
        # this read ~0.8 no matter what the policy does.
        n_decisions = decision_seq.sum().clamp(min=1.0)
        noop_frac = ((card_actions_seq == net.hand_size).float() * decision_seq).sum() / n_decisions
        writer.add_scalar("Policy/Noop_Fraction", noop_frac.item(), episodes_completed)
        # What fraction of real steps offered any choice at all -- directly
        # measures the structural sparsity the affordability mask exists for
        # (~27% before this change).
        writer.add_scalar("Policy/Decision_Fraction",
                          (decision_seq.sum() / valid_seq.sum().clamp(min=1.0)).item(), episodes_completed)

        # Bootstrap value for the state right after the last stored step --
        # value only depends on hx, never needs a card/placement at all, so
        # this skips straight past step_lstm_and_card's card_logits and never
        # touches placement (see model.py's own comment on why this split
        # exists).
        with torch.no_grad():
            next_obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            next_features, _, _ = net.extract_features(next_obs_tensor)
            _, _, _, next_value, _ = net.step_lstm_and_card(next_features, (hx, cx))
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

        # Every (chunk-start, env) pair is one independent training segment of
        # bptt_chunk timesteps -- see the bptt_chunk comment above.
        chunk_starts = np.arange(0, update_timestep, bptt_chunk)
        segments = np.array([(cs, e) for cs in chunk_starts for e in range(num_envs)], dtype=np.int64)
        n_segments = len(segments)
        seg_mb_size = max(1, n_segments // num_minibatches)
        # Offsets within a chunk, reused every minibatch to build (L, B) time indices.
        chunk_offsets = torch.arange(bptt_chunk, dtype=torch.long, device=device).unsqueeze(1)

        # Per-minibatch loss diagnostics, averaged and logged once per update below --
        # direct visibility into whether the network is still learning (shrinking
        # critic loss, non-collapsing clip fraction) instead of inferring it indirectly
        # from noisy episode-outcome stats.
        actor_losses, critic_losses, entropy_bonuses, total_losses, clip_fracs = [], [], [], [], []
        aux_losses, aux_maes = [], []
        # Normalized placement entropy of affordable-but-unchosen cards. This is
        # the freeze detector: the aggregate Entropy/Placement_Measured provably
        # cannot see a per-card collapse, and this series can, because the cards
        # at risk are precisely the ones it samples that the other series never
        # reaches.
        coverage_ents = []
        # KL to the advisor's surface on the rows carrying one, and how many
        # rows those are. Read as a pair -- KL can fall simply because the
        # advisor went quiet.
        coverage_kls, coverage_hits = [], []
        # Logged separately so a collapsing head is visible in TensorBoard
        # directly, instead of only showing up in an offline behavioral probe.
        ent_card_log, ent_place_log = [], []

        for epoch in range(ppo_epochs):
            seg_perm = np.random.permutation(n_segments)
            for mb_start in range(0, n_segments, seg_mb_size):
                mb_seg = segments[seg_perm[mb_start:mb_start + seg_mb_size]]
                if len(mb_seg) == 0:
                    continue
                t0 = torch.as_tensor(mb_seg[:, 0], dtype=torch.long, device=device)   # (B,)
                ev = torch.as_tensor(mb_seg[:, 1], dtype=torch.long, device=device)   # (B,)
                B = t0.shape[0]

                # (L, B) gather indices: row l selects timestep t0+l for each segment env.
                tt = t0.unsqueeze(0) + chunk_offsets          # (L, B)
                ee = ev.unsqueeze(0).expand(bptt_chunk, B)    # (L, B)

                # Batch the (non-recurrent) CNN + scalar feature extraction over the
                # whole chunk at once, then loop only the cheap LSTMCell.
                mb_obs_flat = obs_seq[tt, ee].reshape(bptt_chunk * B, -1)
                # ...and the trunk's PRE-pool activation with it, for the
                # high-resolution placement branch. Taken from the same call
                # rather than rebuilt inside placement_given_card: identical
                # arithmetic either way (pinned by
                # test_recomputed_hires_equals_the_passed_one), but rebuilding
                # would re-run the trunk's first conv -- the most expensive
                # layer in it -- once per minibatch per epoch.
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
                # Recomputed from the SAME stored observations the rollout acted
                # on, so it is bit-identical to the mask applied when the action
                # was sampled. Deriving it (rather than storing it) makes it
                # impossible for the two to drift apart, which would silently
                # corrupt the PPO ratio.
                card_mask_seq = net.affordability_mask(mb_obs_flat).view(
                    bptt_chunk, B, net.hand_size + 1)

                mb_card_actions = card_actions_seq[tt, ee]        # (L, B)
                mb_place_actions = placement_actions_seq[tt, ee]  # (L, B)
                mb_masks = masks_seq[tt, ee]                      # (L, B)

                # Resume from the hidden state actually recorded at this chunk
                # first timestep (stored-state truncated BPTT), not from the
                # start of the whole rollout.
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
                # One AFFORDABLE-but-not-necessarily-chosen slot per timestep,
                # sampled uniformly -- the coverage pass. See
                # placement_coverage_slots for why this is not free and why it
                # is nonetheless the cheapest correct option.
                # Read from the rollout, NOT resampled: the advisor target
                # buffered alongside it belongs to exactly this slot on exactly
                # this observation, and a fresh draw would pair one card's
                # logits with another card's target.
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

                # Value clipping (PPO2-style): cap how far the critic's prediction can
                # move from its rollout-time value in a single update, same idea as the
                # policy ratio clip above. Take the worse (larger) of the clipped/
                # unclipped loss so the critic can't dodge the penalty by jumping back
                # and forth outside the trust region -- this is what actually stabilizes
                # a noisy critic instead of just producing a smoother-looking loss curve.
                value_clipped = mb_old_values + torch.clamp(
                    new_values - mb_old_values, -vf_clip_range, vf_clip_range)
                critic_loss_unclipped = F.mse_loss(new_values, mb_ret, reduction="none")
                critic_loss_clipped = F.mse_loss(value_clipped, mb_ret, reduction="none")
                critic_loss_per_elem = torch.max(critic_loss_unclipped, critic_loss_clipped)

                # Phantom auto-reset steps (mb_valid=0) carry an action that was never
                # actually executed in the env -- excluded from every loss term instead
                # of being averaged in as if it were a real transition.
                # Actor and entropy are normalized by mb_decision (a real choice
                # was available), the critic by mb_valid (every real state). On a
                # forced step the masked distribution is a point mass: log-prob 0,
                # ratio constant 1, entropy 0 -- it contributes nothing to either
                # term but WOULD inflate the denominator, shrinking the actor
                # effective step size by the ~3.7x all-steps/decision-steps ratio.
                actor_loss = -(torch.min(surr1, surr2) * mb_decision).sum() / n_decision
                critic_loss = (critic_loss_per_elem * mb_valid).sum() / n_valid
                ent_card_mean = (new_ent_card * mb_decision).sum() / n_decision
                # Placement entropy is measured and rewarded ONLY on steps that
                # actually placed a card -- mb_decision means "a card was
                # AFFORDABLE", and the placement head is also sampled on the
                # steps where the policy chose the no-op, where the sampled cell
                # never reaches the board. Averaging those in let the head earn
                # the bonus for free while the distribution that places cards
                # collapsed underneath it. Measured in pipeline 2 at ep~62,200:
                # 0.462 reported = 0.850 on no-op steps vs 0.090 on real
                # placements, against a 0.25 target, which pinned the
                # coefficient to its floor. Full evidence in train_selfplay.py's
                # copy of this block; the defect and the fix are identical here.
                mb_placed = mb_decision * (mb_card_actions != net.hand_size).float()
                n_placed = float(mb_placed.sum())
                if n_placed > 0.0:
                    ent_place_mean = (new_ent_place * mb_placed).sum() / n_placed
                else:
                    # No placement anywhere in the chunk: keep the old
                    # denominator rather than feed the controller a 0, which it
                    # would chase as a total collapse.
                    ent_place_mean = (new_ent_place * mb_decision).sum() / n_decision
                # --- placement coverage -------------------------------------
                # Entropy of the placement map for a card that was AFFORDABLE
                # this step, chosen or not. This is a regularizer, not part of
                # the PPO objective: it never touches new_logprobs, so the
                # ratio is unaffected and the update stays a valid PPO step.
                #
                # Its coefficient is deliberately FIXED rather than tied to the
                # adaptive ent_coef_place. The controller lowers that
                # coefficient when REAL placements are sharp enough, which is
                # exactly the condition under which an unplayed card is
                # freezing -- tying the two would switch coverage off precisely
                # when it is needed.
                #
                # Split by row since 2026-08-14: where the advisor has a rule
                # for the sampled card it supplies a TARGET and the row gets KL
                # to it, everywhere else the entropy bonus stands. They are
                # mutually exclusive because they ask for opposite things --
                # see advisor_target.coverage_terms.
                mb_cov_targets = (coverage_target_seq[tt, ee]
                                  if coverage_target_seq is not None else None)
                cov_delta, cov_ent_frac, cov_kl, cov_n = advisor_target.coverage_terms(
                    cf_pl_seq, mb_cov_targets, coverage_has_seq[tt, ee],
                    mb_decision, PLACEMENT_COVERAGE_COEF, LOG_N_PLACEMENT)
                coverage_ents.append(float(cov_ent_frac))
                coverage_kls.append(float(cov_kl))
                coverage_hits.append(float(cov_n))
                # Each head normalized by its own maximum, then weighted -- see
                # the LOG_N_CARD comment above for why the raw sum was wrong.
                entropy_bonus = (ent_coef_card * ent_card_mean / LOG_N_CARD
                                 + ent_coef_place * ent_place_mean / LOG_N_PLACEMENT)
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

                # cov_delta carries both signs already: the entropy half is a
                # bonus (negative), the advisor KL a penalty (positive).
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
                # Fraction of (real) samples where the PPO ratio hit the clip range --
                # near 0 means the policy barely moved this update (possible plateau/too
                # low LR), consistently high means updates may be too aggressive.
                # Measured over decision steps only -- forced steps have ratio
                # exactly 1.0 by construction and would dilute this toward 0
                # regardless of how much the policy actually moved.
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
        card_frac = mean_ent_card / LOG_N_CARD
        place_frac = mean_ent_place / LOG_N_PLACEMENT
        ent_target_place = placement_entropy_target(episodes_completed)
        ent_coef_card = float(np.clip(
            ent_coef_card * math.exp(ENTROPY_ADAPT_RATE * (ENTROPY_TARGET_CARD - card_frac)),
            ENTROPY_COEF_FLOOR, ENTROPY_COEF_CEIL))
        ent_coef_place = float(np.clip(
            ent_coef_place * math.exp(ENTROPY_ADAPT_RATE * (ent_target_place - place_frac)),
            ENTROPY_COEF_FLOOR, ENTROPY_COEF_CEIL))
        writer.add_scalar("Policy/Entropy_Coef_Card", ent_coef_card, episodes_completed)
        writer.add_scalar("Policy/Entropy_Coef_Placement", ent_coef_place, episodes_completed)
        writer.add_scalar("Loss/Entropy_Card", mean_ent_card, episodes_completed)
        writer.add_scalar("Loss/Entropy_Placement", mean_ent_place, episodes_completed)
        writer.add_scalar("Policy/Entropy_Card_Frac", mean_ent_card / LOG_N_CARD, episodes_completed)
        writer.add_scalar("Policy/Entropy_Placement_Frac", mean_ent_place / LOG_N_PLACEMENT, episodes_completed)
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
        writer.add_scalar("Entropy/Placement_Measured", place_frac, episodes_completed)
        # Affordable-but-unchosen cards. Read this NEXT TO Placement_Measured:
        # Measured going down while Coverage stays up is a policy sharpening on
        # the cards it plays; BOTH going down is the freeze this term exists to
        # prevent, and it is the shape that produced the (11,0) collapse.
        writer.add_scalar("Entropy/Placement_Coverage",
                          float(np.mean(coverage_ents)), episodes_completed)
        # The cure's own instrumentation -- read KL and Rows together, since KL
        # can fall simply because the advisor stopped speaking.
        writer.add_scalar("Advisor/KL", float(np.mean(coverage_kls)),
                          episodes_completed)
        writer.add_scalar("Advisor/Rows", float(np.mean(coverage_hits)),
                          episodes_completed)
        writer.add_scalar("Advisor/Coef", advisor_target.ADVISOR_COVERAGE_COEF,
                          episodes_completed)
        # Proof that the anneal wired in on 2026-08-14 actually runs.
        writer.add_scalar("Shaping/SpellValueWeight",
                          spell_value_weight(episodes_completed),
                          episodes_completed)
        writer.add_scalar("Aux/OppElixir_MSE", float(np.mean(aux_losses)), episodes_completed)
        writer.add_scalar("Loss/Clip_Fraction", mean_clip_frac, episodes_completed)
        # Printed (not just logged to TensorBoard) so progress can be monitored from
        # the console/log file alone, without needing the TensorBoard UI open.
        print(f"  >> Update @ ep {episodes_completed} | Actor: {mean_actor_loss:.5f} | "
              f"Critic: {mean_critic_loss:.5f} | Entropy: {mean_entropy:.4f} | "
              f"ClipFrac: {mean_clip_frac:.4f} | OppElixirMAE: {float(np.mean(aux_maes)):.2f}")

        obs_buffer.clear()
        coverage_slot_buffer.clear()
        coverage_target_buffer.clear()
        coverage_has_buffer.clear()
        card_actions_buffer.clear()
        placement_actions_buffer.clear()
        decision_buffer.clear()
        hx_in_buffer.clear()
        cx_in_buffer.clear()
        logprobs_buffer.clear()
        values_buffer.clear()
        rewards_buffer.clear()
        masks_buffer.clear()
        valid_buffer.clear()
        aux_elixir_buffer.clear()

        hx, cx = hx.detach(), cx.detach()

        # Saves
        if episodes_completed - last_save_ep >= SAVE_EVERY_EPISODES:
            torch.save({
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episodes_completed": episodes_completed,
                "curriculum_stage": curriculum_stage,
                "stage_start_episode": stage_start_episode,
                "outcome_history": list(outcome_history),
                "phase": phase,
                "phase_deck_episode_start": phase_deck_episode_start,
                "random_phase_episode_start": random_phase_episode_start,
                "current_random_deck": current_random_deck,
                "deck_curriculum_stage": deck_curriculum_stage,
                "ent_coef_card": ent_coef_card,
                "ent_coef_place": ent_coef_place,
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
        "random_phase_episode_start": random_phase_episode_start,
        "current_random_deck": current_random_deck,
        "deck_curriculum_stage": deck_curriculum_stage,
    }, weight_path)
    envs.close()
    writer.close()
    print(f">>> Pipeline #1 stopped at episode {episodes_completed} (phase={phase}) -- "
          f"final checkpoint saved to {weight_path}.")

    # Automatic handoff to pipeline #2 (self-play/PFSP) -- see
    # RANDOM_OPPONENT_EPISODE_BUDGET's comment for why this doesn't wait for
    # anyone to notice and launch it manually. sys.executable guarantees the
    # same venv interpreter this script itself is running under.
    selfplay_out = open("training_selfplay_pfsp.log", "w")
    selfplay_err = open("training_selfplay_pfsp_err.log", "w")
    # -u (unbuffered) is not optional here. Python block-buffers stdout when it
    # is redirected to a file, so without it the handoff produces a 0-byte log
    # for a long stretch and the live strategy readout (Cards/Game, Elixir@Play)
    # -- whose entire purpose is watching the run as it happens -- is invisible
    # until a buffer happens to flush.
    subprocess.Popen([sys.executable, "-u", "train_selfplay.py"],
                     stdout=selfplay_out, stderr=selfplay_err)
    print(">>> Launched train_selfplay.py (pipeline #2) -- see training_selfplay_pfsp.log / "
          "training_selfplay_pfsp_err.log")

if __name__ == "__main__":
    # Prevent safe pickling errors in Windows multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    train_ppo()