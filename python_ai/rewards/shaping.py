"""The dense reward terms themselves. Weights live in `rewards/weights.py`.

Every function here is a pure function of the engine's own MatchStatistics
(delivered through the env's info dict as (num_envs,) arrays) and returns a
(num_envs,) float32. Nothing here touches torch, a network, or a trainer --
which is what makes the terms testable in isolation and what makes
`compute_shaping` reusable between the two pipelines instead of copied.
"""
import numpy as np

from python_ai.engine_constants import (
    BOARD_H, BOARD_W, MAX_BUILDING_HP, MAX_TROOP_HP, N_CHANNELS,
    OWN_TOWER_HP_TOTAL, SPATIAL_SIZE,
)
from python_ai.rewards.elixir_shaping import solvency_shaping
from python_ai.rewards.weights import (
    ELIXIR_OVERFLOW_THRESHOLD,
    FLAWLESS_REQUIRES_CROWN, MAX_ELIXIR_PER_STEP, SOLVENCY_COEF,
    SOLVENCY_ENABLED, SPELL_VALUE_ANNEAL_EPISODES,
    SPELL_VALUE_ANNEAL_START, W_BLDG, W_ELIXIR_OVERFLOW, W_ELIXIR_TRADE,
    W_FLAWLESS_DEFENSE, W_LETHAL_SPELL, W_SPELL_VALUE_FINAL,
    W_SPELL_VALUE_START, W_TOWER_DESTROYED, W_TROOPS,
    W_WIN_CONDITION_DAMAGE,
)

#: Counters the engine only ever INCREASES within one episode. A decrease is
#: not physically possible mid-match, so it is a reliable witness that the
#: vector env auto-reset between the two readings.
_MONOTONE_UP = (
    "team0_troop_damage", "team1_troop_damage",
    "team0_building_damage", "team1_building_damage",
    "team0_tower_damage", "team1_tower_damage",
    "team0_wincon_damage",
    "team0_elixir_spent", "team1_elixir_spent",
    "spell_value_killed", "spell_elixir_spent",
)
#: ...and the ones that only ever DECREASE, for which a reset looks like a rise.
#: Needed on its own: an episode that ended with towers already lost restores
#: none of the counters above to a LOWER value, so the damage keys alone would
#: miss it.
_MONOTONE_DOWN = ("team0_towers_alive", "team1_towers_alive")


def auto_reset_mask(stats, prev_stats):
    """(num_envs,) bool: True where these two readings straddle an auto-reset.

    Every cumulative counter restarts at 0 when a vector env resets, so the
    POTENTIAL-based terms -- which read raw values, not deltas -- evaluate
    `gamma*Phi(new episode) - Phi(finished episode)` and emit a large spurious
    reward on a step where nothing happened. Measured at -0.3038 for a match
    that was 3000 tower damage ahead, against a sparse win reward of +/-1.

    Detected here rather than trusted to the caller. The guard used to live in
    `base_trainer` and `exploiter` as two copies of `* (1 - prev_dones)`, and
    the exploiter shipped WITHOUT its copy through burst #0 -- this exact bug,
    in production, once already. A monotone counter moving the wrong way is
    proof of a reset needing no information the caller has to remember to pass.
    """
    n = np.asarray(stats["team0_troop_damage"]).shape[0]
    reset = np.zeros(n, dtype=bool)
    for key in _MONOTONE_UP:
        if key in stats and key in prev_stats:
            reset |= np.asarray(stats[key]) < np.asarray(prev_stats[key])
    for key in _MONOTONE_DOWN:
        if key in stats and key in prev_stats:
            reset |= np.asarray(stats[key]) > np.asarray(prev_stats[key])
    return reset


def flawless_defense_bonus(dones, step_rewards, stats, prev_stats,
                           w=None):
    """The 'perfect defense' term: rank WINS by how little we gave up.

    Returns (num_envs,) float32, zero everywhere except on a step that ENDED a
    won episode, where it is w * (fraction of our own tower HP still standing).

    Gated on the win on purpose -- see W_FLAWLESS_DEFENSE. A stalled timeout
    collects nothing and still pays DRAW_PENALTY, a loss collects nothing, so
    this cannot reorder win/loss/draw; it only separates a clean win from a
    scraped one.

    np.maximum(prev, cur) rather than cur: on a done step the vector env has
    already auto-reset, so the cumulative counter may read 0 for the NEW
    episode. Both counters are monotone WITHIN an episode, so the running max
    is exactly "the most damage this finished episode ever recorded" whichever
    of the two the info dict happens to carry -- the same reasoning behind
    compute_shaping's delta() clamp.
    """
    if w is None:
        w = W_FLAWLESS_DEFENSE
    is_win = dones & (step_rewards > 0.5)
    if FLAWLESS_REQUIRES_CROWN:
        # A win with all three enemy towers still standing is a timeout win on
        # tower HP -- exactly the turtle this bonus must not pay. See
        # FLAWLESS_REQUIRES_CROWN for the measured reason.
        is_win = is_win & (np.asarray(stats["team1_towers_alive"]) < 3)
    taken = stats["team1_tower_damage"]
    if prev_stats is not None:
        taken = np.maximum(prev_stats["team1_tower_damage"], taken)
    hp_left = np.clip(1.0 - taken / OWN_TOWER_HP_TOTAL, 0.0, 1.0)
    return (w * is_win.astype(np.float32) * hp_left.astype(np.float32)).astype(np.float32)

def lethal_spell_potential(stats, w=W_LETHAL_SPELL):
    """Phi(s): 1 when a finishing spell is genuinely available AND an enemy
    tower is inside its damage, 0 otherwise.

    Potential-based, and it is worth being explicit about what that buys and
    what it does NOT. PBRS telescopes over an episode to
    gamma^T*Phi(s_T) - Phi(s_0). Phi(s_0) is 0 (no tower is in spell range at
    the start). Phi(s_T) is USUALLY 0 but not always -- "the game is over at
    the end" does not zero a potential, and a match that ends with a surviving
    tower inside the window and the spell in hand leaves +W_LETHAL_SPELL on the
    table: measured nonzero in 2 of 16 seeded mirror matches, up to +0.125
    (TODO 00.4; see weights.py's policy-invariance note). Where it is 0 the term
    is policy-invariant by Ng et al.: it cannot make the agent value the spell
    more at the optimum, and it cannot be farmed by cycling in and out of the
    state.

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

    THE SPELL IS THE DECK'S, NOT FIREBALL'S, since 2026-09-16. `spell_damage`
    and `spell_cost` ride in on the stats dict from whichever card
    `card_probes.damage_spell` named, so a Rocket deck gets a Rocket-sized
    lethal window. A deck with no damaging spell publishes 0.0 for both, and
    `hp > 0.0` then makes `hp <= 0.0` false for every tower -- the term is
    structurally zero rather than zero by luck.

    `spell_damage` is the spell's measured damage TO A CROWN TOWER, not to a
    troop. They are equal in this engine today and 15-30% apart in the real
    game; see `card_probes.spell_tower_damage` for why the window is keyed to
    the one that stays right if the engine is corrected.

    The keys are REQUIRED. A default would have to be some spell's constants,
    and a silent fallback to Fireball's is exactly the defect this replaced.
    """
    hp = stats["enemy_tower_hp"]                       # (num_envs, 3) absolute
    damage = np.reshape(np.asarray(stats["spell_damage"], dtype=np.float32), (-1, 1))
    cost = np.asarray(stats["spell_cost"], dtype=np.float32)
    in_range = np.any((hp > 0.0) & (hp <= damage), axis=1)
    actionable = (stats["spell_in_hand"] > 0.5) & \
                 (stats["team0_elixir_current"] >= cost)
    return w * (in_range & actionable).astype(np.float32)

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

    THE DENOMINATOR IS THE DECK'S SPELL COST, since 2026-09-16, and it is the
    half of this that a hardcoded 4.0 got WRONG rather than merely approximate:
    the quantity is a ratio centred on break-even, so pricing a 6-cost Rocket at
    4 turns an even trade into a +0.5 reward and teaches the policy that Rocket
    spam pays. The solvency reserve moves with it for the same reason -- it
    means "enough left to answer with one more card", which is the spell's own
    cost; the retired `SPELL_SOLVENCY_RESERVE = 4.0` was that number for
    Fireball specifically.

    A deck with no damaging spell publishes `spell_cost = 0.0`, and that is
    GUARDED rather than divided by: the counters are also zero so the arithmetic
    would give 0/0 -> nan, and a nan reward propagates into the advantage,
    through the optimizer, and kills the net silently. The term is then exactly
    zero.
    """
    cost = np.asarray(stats["spell_cost"], dtype=np.float32)
    killed = np.maximum(0.0, stats["spell_value_killed"] - prev_stats["spell_value_killed"])
    spent = np.maximum(0.0, stats["spell_elixir_spent"] - prev_stats["spell_elixir_spent"])
    have_spell = cost > 0.0
    safe_cost = np.where(have_spell, cost, 1.0)
    casts = spent / safe_cost
    traded = (killed / safe_cost - casts) * have_spell.astype(np.float32)
    solvent = (stats["team0_elixir_current"] >= safe_cost).astype(np.float32)
    return (w * np.where(traded > 0.0, traded * solvent, traded)).astype(np.float32)

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

def compute_shaping(stats, prev_stats, gamma, w_bldg=W_BLDG, w_troops=W_TROOPS,
                     w_elixir=W_ELIXIR_TRADE, w_overflow=W_ELIXIR_OVERFLOW,
                     w_tower=W_TOWER_DESTROYED, w_spell=W_SPELL_VALUE_START,
                     w_wincon=W_WIN_CONDITION_DAMAGE):
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

    THE DISCOUNT IS REQUIRED, NOT DEFAULTED, and that is the whole guarantee.
    The tower term is `gamma*Phi(s') - Phi(s)`, and Ng et al.'s policy-
    invariance result holds only when this gamma is the SAME one GAE discounts
    with. It used to default to a literal 0.99 -- harmless only while
    `PPOConfig.gamma` also read 0.99, and a silent invariance break the moment
    it did not. Deriving the default was not available either: `rewards/` is an
    enforced leaf layer that may import neither `rl/` nor `trainers/`
    (`tests/test_package_layout.py`). Requiring the argument satisfies the
    no-second-copies rule by ABSENCE rather than by derivation, which is the
    stronger form -- there is no copy here to go stale, and no caller can
    compute potential-based shaping without stating the discount it is for.
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

    # WIN-CONDITION damage, on top of the tower term it already earns. Plain
    # delta of a cumulative counter, clamped >=0 by delta() for the same
    # post-autoreset reason as every other counter here. Deliberately NOT
    # potential-based -- see W_WIN_CONDITION_DAMAGE. Contributes exactly zero
    # for a deck with no building-targeter, where the key is absent.
    wincon_damage = (delta("team0_wincon_damage") / MAX_BUILDING_HP
                     if "team0_wincon_damage" in stats and "team0_wincon_damage" in prev_stats
                     else 0.0)

    shaping = (tower_shaping
               + lethal_shaping
               + spell_value_shaping(stats, prev_stats, w_spell)
               + solvency
               + tower_events
               + w_wincon * wincon_damage
               + w_troops * (enemy_troops_damage - ally_troops_damage)
               + w_elixir * enemy_elixir_spent
               - w_overflow * overflow)

    # A step that straddles an auto-reset is not a transition and carries no
    # shaping signal. Both live callers already zero it, so this changes
    # nothing for them -- it makes the function safe for the NEXT caller, which
    # is the one that has historically forgotten.
    shaping = np.where(auto_reset_mask(stats, prev_stats), 0.0, shaping)

    return shaping.astype(np.float32)

def building_hp_end(obs_vec):
    """Remaining normalized building HP (ally, enemy) in a single final observation --
    used as a per-episode offense/defense progress metric."""
    spatial = obs_vec[:SPATIAL_SIZE].reshape(N_CHANNELS, BOARD_H, BOARD_W)
    return float(spatial[3].sum()), float(spatial[7].sum())
