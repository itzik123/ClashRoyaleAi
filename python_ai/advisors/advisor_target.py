"""The advisor's score surface as a TRAINING TARGET for the coverage term.

WHY THIS EXISTS -- the mathematical conflict it resolves
--------------------------------------------------------
`PLACEMENT_COVERAGE_COEF` (train.py) restores a placement gradient to cards the
policy has stopped playing, by adding an ENTROPY BONUS on one uniformly-sampled
affordable slot per step. For such a card that bonus is the only placement
gradient in the entire objective, and it pushes the map toward UNIFORM.

A distilled placement map is exactly such a card's map. So the coverage term and
the high-resolution distillation of `prove_hires.py` pull in opposite
directions: distillation puts mass on the right cell, coverage flattens it, and
coverage runs for all of training. The predicted consequence is that the
measured hi-res gain (Fireball 5.4x better than a random legal cell) erodes back
toward uniform under PPO.

The fix is the conclusion this project has now reached three separate times:

    closing a coverage hole needs a TARGET, not noise.

Entropy is a MARGINAL objective. It says "be spread out"; it does not say
"depend on the board". A card with no other gradient has nothing telling it
WHICH cell is right in WHICH state, which is why the measured 3-arm coverage A/B
moved the frozen cell (11,0) -> (6,0) without breaking the lock: the argmax of a
flat map is an arbitrary constant.

So wherever the advisor has a rule for the sampled card, this module supplies
its own score surface as target logits and the trainer applies KL to it. Where
the advisor has nothing to say, the trainer keeps the entropy bonus. The two are
mutually exclusive per row -- applying both is asking the head to be spread out
and concentrated at the same time.

THE GATE IS THE LOAD-BEARING PART
---------------------------------
`tactics` always returns a cell. With no threat on the board that cell is a
default -- the defensive pocket for a building, an arbitrarily tie-broken lane
for the Giant -- and training on defaults teaches a CONSTANT, which is precisely
the pathology being repaired. `target_logits_for` returns None in those states
and the caller falls back to entropy. This mirrors `distill_tactics.collect`,
which keeps only states where `cover > 0 or catch > 0` for the same reason.

WHAT IS AND IS NOT CLAIMED
--------------------------
The advisor is measured better than the head for all three cards
(`prove_placement.py`: Cannon 539.1 HP preserved vs the net's 102.2, Fireball
2.647 elixir killed vs 0.062), so it is a sound target. It is NOT a ceiling
worth reaching exactly -- for the Cannon its argmax is row-major noise off a
large exact-tie plateau, which is why buildings get the whole standardized
surface rather than a delta.
"""
import os

import numpy as np

from python_ai.advisors import tactics

CANNON_ID = tactics.CANNON_ID
FIREBALL_ID = tactics.FIREBALL_ID
HOG_ID = tactics.HOG_ID
BOARD_W = tactics.BOARD_W
BOARD_H = tactics.BOARD_H
N_CELLS = BOARD_H * BOARD_W

# Which advisor rule produces each card's surface. "cell" means the rule yields
# a single cell rather than a scored map, so its target is a delta.
ADVISOR_CARDS = {
    CANNON_ID: "building",
    FIREBALL_ID: "spell",
    # The GIANT entry was removed on 2026-08-17. It is not in DEFAULT_DECK --
    # the deck became 2.6 Hog Cycle on 2026-08-16 -- so the rule could never
    # fire, and a dead entry here is worse than none: it makes the advisor look
    # like it covers a win condition when it covers nothing.
    HOG_ID: "wincon",
}

# Temperature on the STANDARDIZED score. Read this off the target's own entropy,
# never tuned on the outcome -- CLAUDE.md's expert-iteration entry records a run
# lost to a T that put the target at 94% of maximum entropy while looking busy.
# At 0.25 the measured target entropies are Cannon 68.3% of max, Fireball 41.8%.
TARGET_TEMPERATURE = float(os.environ.get("CLASH_ADVISOR_TARGET_T", 0.25))

_NEG_INF = float("-inf")


def _standardize(flat_scores, legal, T):
    """Score vector -> target logits, -inf off `legal`.

    Standardized over the legal cells first, so T means the same thing for the
    Cannon's coverage sums and the Fireball's caught-damage sums, which live on
    completely different scales. Bit-identical to
    `prove_hires.soft_target_logits`, which is pinned by test rather than by
    import: the trainers must not pull the offline harness (and through it
    expert_iteration) into their rollout path.
    """
    out = np.full(N_CELLS, _NEG_INF, dtype=np.float32)
    v = flat_scores[legal]
    # ddof=1 to match torch's Tensor.std(), which is UNBIASED by default while
    # numpy's is not. The gap is ~0.1% of a logit and invisible by eye; the
    # equality test against soft_target_logits is what found it, which is the
    # whole reason that test exists rather than a comment claiming they agree.
    std = float(v.std(ddof=1)) if v.size > 1 else 0.0
    if not np.isfinite(std) or std < 1e-8:
        # A flat surface carries no preference, and a uniform target over the
        # legal cells is the honest encoding of that. It also stops such rows
        # dominating the loss with arbitrary noise.
        out[legal] = 0.0
    else:
        out[legal] = ((v - v.mean()) / std / T).astype(np.float32)
    return out


def target_logits_for(obs, card_id, legal, T=None):
    """(612,) float32 target logits for `card_id`, or None if the advisor
    has nothing to say about this board.

    obs:   (obs_dim,) float32, one state, team-0 frame.
    legal: (612,) bool -- the net's own `_placement_legal[card_id]`. Passing the
           net's table rather than recomputing legality is what keeps the
           surface being distilled and the cell being played on one set of
           cells; a second copy of that geometry is drift this project has
           already paid for twice.
    """
    kind = ADVISOR_CARDS.get(int(card_id))
    if kind is None:
        return None
    legal = np.asarray(legal, dtype=bool).reshape(-1)
    if not legal.any():
        return None
    T = TARGET_TEMPERATURE if T is None else T

    if kind == "building":
        # _building_score returns (map, kind) from ONE implementation shared
        # with best_building_cell, so the surface distilled here and the cell
        # the advisor plays can never come from two drifting copies. Its `kind`
        # is the gate: "coverage" means a real threat was scored, while
        # "pocket"/"none" are the defaults that would teach a constant.
        score, why = tactics._building_score(obs, legal=legal)
        if why != "coverage":
            return None
        flat = np.asarray(score, dtype=np.float32).reshape(-1).copy()
        flat[~legal] = _NEG_INF
        if not np.isfinite(flat[legal]).all():
            return None
        return _standardize(flat, legal, T)

    if kind == "wincon":
        # The win condition is the one card whose rule is as much about WHEN as
        # WHERE, so the gate lives in tactics.hog_advice and returning None here
        # is the normal case, not an error path. Committing 4 elixir into an
        # active push, while broke, or against a banked opponent are all trades
        # the measurement says to refuse -- and a rule that always speaks would
        # teach the head a CONSTANT bridge cell regardless of board, which is
        # precisely the collapse the 2026-08-14 cure undid.
        advice = tactics.hog_advice(obs, legal=legal)
        if advice is None:
            return None
        x, y = advice
        cell = int(y) * tactics.BOARD_W + int(x)
        if not legal[cell]:
            return None
        flat = np.full(legal.shape, _NEG_INF, dtype=np.float32)
        flat[cell] = 0.0
        return flat

    if kind == "spell":
        flat = tactics.spell_catch_map(obs).reshape(-1).astype(np.float32).copy()
        if float(flat[legal].max()) <= 0.0:
            return None                      # nothing worth spelling anywhere
        flat[~legal] = _NEG_INF
        return _standardize(flat, legal, T)

    # kind == "cell": a delta at the advisor's cell. masked_kl against a delta
    # is exactly cross-entropy to that cell, so this needs no separate
    # hard-label path in the trainers.
    if float(tactics.enemy_hp_map(obs).sum()) <= 0.0:
        # The Giant rule picks the less-defended lane; with nothing on the board
        # that is an arbitrary fixed tie-break, i.e. a constant.
        return None
    gx, gy, _ = tactics.best_giant_cell(obs, legal=legal)
    cell = int(gy) * BOARD_W + int(gx)
    if not legal[cell]:
        return None
    out = np.full(N_CELLS, _NEG_INF, dtype=np.float32)
    out[cell] = 0.0
    return out


def build_legal_table(net):
    """{card_id: (612,) bool} for every card the advisor has a rule for."""
    return {cid: net._placement_legal[cid].numpy().astype(bool)
            for cid in ADVISOR_CARDS}


# --- the trainer-facing half ------------------------------------------------
# Both trainers use these rather than each growing its own copy. The two PPO
# loops are already near-duplicates and every divergence between them has cost
# this project a measurement at some point.

# How hard the advisor target pulls, as a fraction of log(n_cells) so it is
# directly comparable to PLACEMENT_COVERAGE_COEF. 0 disables the target
# entirely -- no advisor call is made at rollout time either -- which is the
# control arm of the A/B and costs nothing to run.
ADVISOR_COVERAGE_COEF = float(os.environ.get("CLASH_ADVISOR_COVERAGE_COEF", 0.10))

_LOG_N_CELLS = float(np.log(N_CELLS))

# How much more likely a slot holding an advisor card is to win the coverage
# draw. 1.0 = uniform (the pre-2026-08-14 behaviour). Default 5.0 because the
# measured constraint is affordability, not the advisor declining: over 258
# decision steps the sampled slot held an advisor card 32.6% of the time and
# the advisor then spoke on 90% of those -- so the way to raise coverage is to
# draw those cards more often, not to loosen the gate.
ADVISOR_SLOT_WEIGHT = float(os.environ.get("CLASH_ADVISOR_SLOT_WEIGHT", 5.0))


def enabled():
    return ADVISOR_COVERAGE_COEF > 0.0


def slot_weights_for(hand_ids):
    """(B, hand_size) float: ADVISOR_SLOT_WEIGHT on advisor cards, 1 elsewhere.

    hand_ids: (B, hand_size) long from `net.hand_card_ids`, -1 for empty.
    Returns None when the weighting is off, which the sampler reads as uniform.
    """
    import torch
    if ADVISOR_SLOT_WEIGHT == 1.0 or not enabled():
        return None
    w = torch.ones(hand_ids.shape, dtype=torch.float32)
    for cid in ADVISOR_CARDS:
        w[hand_ids == cid] = ADVISOR_SLOT_WEIGHT
    return w


def targets_for_batch(obs_np, card_ids, legal_table, T=None):
    """One rollout step's advisor targets.

    obs_np:    (B, obs_dim) float32 -- the observations the policy just acted on
    card_ids:  (B,) int -- the card in each env's sampled COVERAGE slot, or -1
    returns:   targets (B, N_CELLS) float32, has_target (B,) bool

    Rows without a target are left as zeros and flagged False; the caller must
    never read them, and the loss masks them out.
    """
    B = obs_np.shape[0]
    out = np.zeros((B, N_CELLS), dtype=np.float32)
    has = np.zeros(B, dtype=bool)
    for b in range(B):
        cid = int(card_ids[b])
        legal = legal_table.get(cid)
        if legal is None:
            continue
        t = target_logits_for(obs_np[b], cid, legal, T)
        if t is None:
            continue
        out[b] = t
        has[b] = True
    return out, has


def masked_kl_elementwise(new_logits, target_logits):
    """KL(target || new) per row, over the LEGAL cells only. (...,) tensor.

    Same arithmetic as `distill_tactics.masked_kl` without its final `.mean()`,
    so the caller can mask rows before averaging -- which this one must, since
    only some rows carry a target.

    Zeroing the non-finite terms is correct rather than a patch. `-inf` appears
    on illegal cells in BOTH distributions, torch evaluates
    0 * (-inf - -inf) = nan there, and the true contribution of a cell carrying
    no probability mass in either distribution is exactly zero.

    SANITIZED BEFORE THE ARITHMETIC, NOT AFTER IT. The previous form computed
    `term` and then zeroed the non-finite entries with `torch.where`, which
    fixes the FORWARD and does not fix the BACKWARD: `where` still
    differentiates the branch it did not select, and `0 * nan = nan`. Tracing
    it, with `a = lo.exp()` and `b = (lo - ln)`:

        grad_a = grad_term * b = 0 * nan = NAN
        grad_b = grad_term * a = 0 * 0   = 0

    so a NaN really was produced. It stayed harmless only because `grad_a`
    flows into `log_softmax(target_logits)` and the target is a CONSTANT -- the
    advisor's surface, buffered from numpy during the rollout. Two implicit
    properties were holding it up: that the target never requires grad, and
    that no row is entirely `-inf`. The second one is demonstrably load-bearing
    -- a fully-masked row DID put NaN straight into `new_logits.grad`, forward
    still reading a clean 0.0.

    Replacing the operands first means no NaN is ever created, so neither
    property has to hold. Bit-identical on every reachable input: at a masked
    cell the substituted operands give `exp(0) * (0 - 0) * 0 = 0`, which is what
    the old zeroing produced, and at a legal cell nothing is substituted.
    """
    import torch
    ln = torch.log_softmax(new_logits, -1)
    lo = torch.log_softmax(target_logits, -1)
    keep = torch.isfinite(lo) & torch.isfinite(ln)
    zero = torch.zeros_like(lo)
    lo_s = torch.where(keep, lo, zero)
    ln_s = torch.where(keep, ln, zero)
    term = lo_s.exp() * (lo_s - ln_s) * keep
    return term.sum(-1)


def coverage_terms(cf_logits, targets, has_target, decision, entropy_coef,
                   log_n_placement):
    """The coverage half of the loss, for one minibatch.

    Returns (loss_delta, ent_frac, kl_mean, n_target) where `loss_delta` is
    ADDED to the total loss -- it already carries both signs, since the entropy
    half is a bonus (negative) and the KL half is a penalty (positive).

    THE RESOLUTION OF THE CONFLICT IS THE ROW MASK. A row either has an advisor
    target and gets KL to it, or it does not and gets the entropy bonus. Never
    both: entropy says "be spread out" and KL says "be here", and a row carrying
    both is asking the head for two incompatible things at once. That opposition
    is not hypothetical -- it is why the measured 3-arm coverage A/B moved the
    frozen cell rather than unfreezing it.
    """
    import torch
    from torch.distributions import Categorical

    has = has_target * decision
    no_t = (1.0 - has_target) * decision
    n_has = has.sum()
    n_no = no_t.sum()

    cf_ent = Categorical(logits=cf_logits).entropy()
    # Denominator is the rows the term actually applies to. Dividing by all
    # decision rows instead would silently scale the bonus down as advisor
    # coverage rises, i.e. weaken the fallback exactly where it still matters.
    ent_mean = (cf_ent * no_t).sum() / n_no.clamp(min=1.0)
    ent_frac = ent_mean / log_n_placement
    loss_delta = -entropy_coef * ent_frac

    kl_mean = torch.zeros((), device=cf_logits.device)
    if float(n_has) > 0.0 and ADVISOR_COVERAGE_COEF > 0.0:
        kl_elems = masked_kl_elementwise(cf_logits, targets)
        kl_mean = (kl_elems * has).sum() / n_has.clamp(min=1.0)
        loss_delta = loss_delta + ADVISOR_COVERAGE_COEF * kl_mean / log_n_placement

    # Only loss_delta carries the graph. The other three are diagnostics and are
    # detached here so callers can log them without torch warning about
    # converting a grad-tracking tensor to a scalar on every minibatch.
    return loss_delta, ent_frac.detach(), kl_mean.detach(), n_has.detach()
