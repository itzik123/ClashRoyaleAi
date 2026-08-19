"""Reward-shaping WEIGHTS, and the measurements that fixed each one.

Split out from the terms themselves (`rewards/shaping.py`) so that retuning a
weight is a one-line change in a file with no logic in it, and so the paragraph
justifying a number sits next to the number rather than 600 lines above the
function that uses it.

Kept intentionally small so the cumulative shaping over an episode stays
comparable to -- not larger than -- the terminal +/-1. Watch
`Reward/Episode_Shaping_Sum` against the win rate in TensorBoard: if the shaping
sum dwarfs +/-1, lower these.

WHICH TERMS ARE POLICY-INVARIANT, because it is the property that matters most
here and it is not uniform:

  potential-based (cannot change the optimum, only the speed of finding it)
      the tower term (W_BLDG), the lethal-spell term (W_LETHAL_SPELL), the
      solvency term (SOLVENCY_COEF)
  DELIBERATELY BIASING (changes the optimum, eyes open)
      W_TOWER_DESTROYED, W_FLAWLESS_DEFENSE, W_WIN_CONDITION_DAMAGE,
      W_SPELL_VALUE_START, DRAW_PENALTY

Every biasing term is there because a policy-invariant version was measured and
found to leave PURE DEFENCE as the true optimum -- win-condition usage decayed
to 0.7% over 8,300 episodes. Each one should anneal toward zero once the
behaviour it buys is established; `W_SPELL_VALUE_START` is the only one that
currently does, and it sat dead for a whole training era before being wired in.
"""
import os

import clash_royale_env

from python_ai.rewards.elixir_shaping import W_SOLVENCY

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

# --- the "perfect defense" standard --------------------------------------
# Paid ONLY on a win, scaled by the fraction of our own tower HP still
# standing: a flawless win pays 1 + W_FLAWLESS_DEFENSE, a win that gave up
# both Princess towers pays barely more than 1.
#
# WHY IT IS SHAPED THIS WAY, and it is the whole design. The tower term in
# compute_shaping() is LINEAR and SYMMETRIC -- 100 HP chipped off the enemy
# pays exactly what 100 HP taken costs -- so the reward is indifferent between
# "trade 500 for 500" and "take 0, deal 0". "Zero tower damage is the standard"
# is simply not expressible in it.
#
# The obvious alternative, weighting damage TAKEN above damage DEALT, is the
# one thing that must NOT be done here: this file already records that
# policy-invariant tower shaping left PURE DEFENCE as the true optimum and
# win-condition usage decayed to 0.7% over 8,300 episodes. Tilting the
# per-step trade further toward defence walks straight back into that.
#
# Gating on the WIN is what makes this safe. A turtle that stalls into a
# timeout collects nothing and still pays DRAW_PENALTY; a policy that loses
# collects nothing. So the term cannot reorder win/loss/draw at all -- it can
# only rank WINS against each other, which is exactly "among the policies that
# win, prefer the ones that took no damage".
#
# NOT potential-based, and therefore biasing by construction -- the same
# eyes-open trade as W_TOWER_DESTROYED, stated rather than hidden.
W_FLAWLESS_DEFENSE = 0.5

# Require the win to be DECISIVE -- at least one enemy tower actually
# destroyed -- before the flawless bonus pays. Added 2026-08-17, and it is a
# correction to the term above rather than an extension of it.
#
# The original gate was "any win", argued safe because a turtle that stalls
# into a timeout collects nothing. That argument was incomplete. Against a weak
# opponent the agent wins ~100% of games anyway, so "win" is nearly free, and
# conditional on winning the ONLY remaining gradient was preserve-tower-HP --
# i.e. never spend elixir on offence. The arithmetic is stark: conceding one
# Princess costs 2534/9076 of the bonus = 0.140 reward, while a fully
# successful Hog pays 0.5 * 470/4008 = 0.059 through tower_potential. The term
# punished a defensive slip 2.4x harder than a perfect attack paid.
#
# Measured consequence at ep 6,053: Hog Rider fell to 0.8% of plays, Fireball
# 0.6%, Cannon 1.6%, with 78.6% of all plays on four cheap defensive cards --
# and prove_hog.py showed the Hog's PLACEMENT was fine (+28.5 over a random
# legal cell, 155 distinct cells), so this was a valuation shift, not a broken
# head. That is the "lazy local optimum" this gate exists to close.
#
# Requiring a crown makes the two terms ALIGNED instead of opposed: you must
# attack to collect, and you are still paid for keeping your own towers. It is
# also what "perfect defense" means in Clash -- take a crown, give none.
FLAWLESS_REQUIRES_CROWN = True

# --- teach the deck's WIN CONDITION that it is the win condition -----------
# Extra reward per point of damage dealt BY the win-condition card, on top of
# the tower term every source of damage already earns.
#
# WHY THIS FORM. The alternative on the table was forcing the card head toward
# the Hog. That is the thing this project has already measured to be harmful:
# forcing Fireball usage dropped win rate 97% -> 23%, because the low weighting
# was a CORRECT valuation. This term never touches the action distribution. It
# is OUTCOME-GATED -- a Hog thrown into a PEKKA deals no damage and earns
# nothing -- so unlike forced usage it cannot pay for a suicide. It pays only
# for a win condition that actually connected.
#
# SIZING, from the same arithmetic as FLAWLESS_REQUIRES_CROWN. A connecting Hog
# deals ~470-630 damage. At 1.0 this adds ~470/4008 = 0.117, so a successful
# Hog is worth ~0.176 all-in against the 0.140 that conceding a Princess costs.
# That is deliberately just past break-even, not overwhelming: the point is to
# make the win condition WORTH PLAYING, not to make it worth spamming.
#
# THIS IS NOT POTENTIAL-BASED AND THEREFORE BIASES THE OPTIMUM BY CONSTRUCTION.
# That is the point and it is the cost: a policy-invariant version could not
# change a valuation, which is the entire objective here. The honest reading is
# that we are asserting the Hog is worth more than this engine's return says it
# is, because a 2.6 agent that never plays its win condition cannot transfer to
# a real opponent. It should be ANNEALED TOWARD ZERO once Hog usage recovers --
# see W_SPELL_VALUE_START, whose anneal sat dead for a whole training era.
# Every win rate earned under this is not comparable to one earned without it.
W_WIN_CONDITION_DAMAGE = float(os.environ.get("CLASH_W_WINCON_DAMAGE", 1.0))

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
SOLVENCY_ENABLED = os.environ.get("CLASH_SOLVENCY", "1") != "0"
SOLVENCY_COEF = float(os.environ.get("CLASH_SOLVENCY_COEF", W_SOLVENCY))
