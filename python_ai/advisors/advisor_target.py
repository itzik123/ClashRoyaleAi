"""The advisor's score surface as a training target for the coverage term.

The placement coverage term gives a card the policy stopped playing an entropy
bonus on its map, which says "be spread out" but never "depend on the board",
so the map's argmax stays an arbitrary constant. Where the advisor has a rule
for the sampled card, this module supplies its score surface as target logits
and the trainer applies KL to it instead; elsewhere the entropy bonus stays.
Never both on one row.

The gate matters most: `tactics` always returns a cell, and with nothing on the
board that cell is a default, which would teach a constant. `target_logits_for`
returns None in those states and the row falls back to entropy.

The advisor is a sound target (it beats the head for its cards) but not an
exact one: for buildings its argmax is arbitrary within a plateau, so they get
the whole standardized surface.
"""
import os

import numpy as np

from python_ai.advisors import card_probes, human_prior, tactics
from python_ai.deck import DEFAULT_DECK

CANNON_ID = tactics.CANNON_ID
FIREBALL_ID = tactics.FIREBALL_ID
HOG_ID = tactics.HOG_ID
BOARD_W = tactics.BOARD_W
BOARD_H = tactics.BOARD_H
N_CELLS = BOARD_H * BOARD_W

# Which advisor rule fits each card, derived from what the card does
# (`card_probes`):
#
#   spell     a damaging area spell castable on the enemy half, no bodies;
#             scored with its own measured radius and damage
#   building  a building that attacks and has no route to a tower
#   wincon    the deck's win condition, when it walks to the tower (Hog, Giant,
#             Balloon); a Miner or siege building is not a bridge commit
def advisor_cards_for(deck):
    """{card_id: "building" | "spell" | "wincon"} for the cards a rule fits."""
    from python_ai.opponents.teacher import card_roles
    wincon = next((c for c, r in card_roles(list(deck)).items() if r == "wincon"), None)
    out = {}
    for cid in deck:
        cid = int(cid)
        if card_probes.spell_effect(cid) is not None:
            out[cid] = "spell"
        elif card_probes.building_defends(cid):
            out[cid] = "building"
        elif cid == wincon and card_probes.walking_building_targeter(cid):
            out[cid] = "wincon"
    return out


ADVISOR_CARDS = advisor_cards_for(DEFAULT_DECK)

# Temperature on the standardized score. Choose it from the target's own
# entropy, never from the outcome: at 0.25 the targets sit at 68% (Cannon) and
# 42% (Fireball) of maximum entropy.
TARGET_TEMPERATURE = float(os.environ.get("CLASH_ADVISOR_TARGET_T", 0.25))

_NEG_INF = float("-inf")


def _standardize(flat_scores, legal, T):
    """Score vector -> target logits, -inf off `legal`.

    Standardized over the legal cells first, so T means the same thing for
    surfaces on different scales. Must equal `prove_hires.soft_target_logits`
    (pinned by a test, not an import, to keep offline harnesses off the rollout
    path).
    """
    out = np.full(N_CELLS, _NEG_INF, dtype=np.float32)
    v = flat_scores[legal]
    # ddof=1 to match torch's unbiased Tensor.std().
    std = float(v.std(ddof=1)) if v.size > 1 else 0.0
    if not np.isfinite(std) or std < 1e-8:
        # A flat surface carries no preference: a uniform target over the legal
        # cells.
        out[legal] = 0.0
    else:
        out[legal] = ((v - v.mean()) / std / T).astype(np.float32)
    return out


def _advisor_logits_for(obs, card_id, legal, T=None):
    """(612,) float32 target logits from the hand-written rules, or None when the
    rule declines to speak.

    obs:   (obs_dim,) float32, one state, team-0 frame.
    legal: (612,) bool, the net's own `_placement_legal[card_id]`, so the
    distilled surface and the played cell use one legality table.
    """
    kind = ADVISOR_CARDS.get(int(card_id))
    if kind is None:
        return None
    legal = np.asarray(legal, dtype=bool).reshape(-1)
    if not legal.any():
        return None
    T = TARGET_TEMPERATURE if T is None else T

    if kind == "building":
        # One implementation shared with best_building_cell. Its `kind` is the
        # gate: only "coverage" scored a real threat; "pocket"/"none" are
        # defaults.
        score, why = tactics._building_score(obs, legal=legal)
        if why != "coverage":
            return None
        flat = np.asarray(score, dtype=np.float32).reshape(-1).copy()
        flat[~legal] = _NEG_INF
        if not np.isfinite(flat[legal]).all():
            return None
        return _standardize(flat, legal, T)

    if kind == "wincon":
        # The win-condition rule is about when as much as where, so the gate is
        # tactics.hog_advice, and None is the normal case. A rule that always
        # spoke would teach a constant bridge cell.
        advice = tactics.hog_advice(
            obs, legal=legal,
            cost=float(tactics.E.get_card_info(int(card_id))["cost"]))
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
        # The card's own measured area and damage.
        radius, damage = card_probes.spell_effect(int(card_id))
        flat = (tactics.spell_catch_map(obs, radius, 0, damage)
                .reshape(-1).astype(np.float32).copy())
        if float(flat[legal].max()) <= 0.0:
            return None                      # nothing worth a spell anywhere
        flat[~legal] = _NEG_INF
        return _standardize(flat, legal, T)

    # Otherwise a delta at the advisor's cell; KL against a delta is
    # cross-entropy to that cell.
    if float(tactics.enemy_hp_map(obs).sum()) <= 0.0:
        # With nothing on the board the lane choice is an arbitrary tie-break,
        # i.e. a constant.
        return None
    gx, gy, _ = tactics.best_giant_cell(obs, legal=legal)
    cell = int(gy) * BOARD_W + int(gx)
    if not legal[cell]:
        return None
    out = np.full(N_CELLS, _NEG_INF, dtype=np.float32)
    out[cell] = 0.0
    return out


def target_logits_for(obs, card_id, legal, T=None):
    """(612,) float32 target logits for `card_id`, or None.

    A hand-written rule wins where it speaks; the mined human prior fills the
    gaps. The rules are validated against this engine, while the prior comes
    from the real game, whose physics this engine does not reproduce, so the
    prior never overrides a rule.
    """
    out = _advisor_logits_for(obs, card_id, legal, T)
    if out is not None:
        return out
    # obs lets the prior honour the quiet-board gate.
    return human_prior.logits_for(card_id, legal, obs)


def target_and_weight(obs, card_id, legal, T=None):
    """(logits, weight): weight 1.0 for a rule, HUMAN_PRIOR_COEF for the prior.

    The weight rides in the buffer's `coverage_has` float, so the prior needs
    no new field.
    """
    out = _advisor_logits_for(obs, card_id, legal, T)
    if out is not None:
        return out, 1.0
    out = human_prior.logits_for(card_id, legal, obs)
    if out is not None:
        return out, human_prior.HUMAN_PRIOR_COEF
    return None, 0.0


def target_cards():
    """Every card some source can produce a target for."""
    return set(ADVISOR_CARDS) | human_prior.prior_cards()


def build_legal_table(net):
    """{card_id: (612,) bool} for every card any target source covers."""
    return {cid: net._placement_legal[cid].numpy().astype(bool)
            for cid in sorted(target_cards())}


# --- the trainer-facing half ---

# How hard the advisor target pulls, as a fraction of log(n_cells) (comparable
# to PLACEMENT_COVERAGE_COEF). 0 disables it, with no advisor calls at rollout
# time.
ADVISOR_COVERAGE_COEF = float(os.environ.get("CLASH_ADVISOR_COVERAGE_COEF", 0.10))

_LOG_N_CELLS = float(np.log(N_CELLS))

# How much more likely a slot holding an advisor card is to win the coverage
# draw (1.0 = uniform). The binding constraint was how often the sampled slot
# held an advisor card, not the advisor declining.
ADVISOR_SLOT_WEIGHT = float(os.environ.get("CLASH_ADVISOR_SLOT_WEIGHT", 5.0))


def enabled():
    return ADVISOR_COVERAGE_COEF > 0.0


def slot_weights_for(hand_ids):
    """(B, hand_size) float: ADVISOR_SLOT_WEIGHT on cards with a target, 1
    elsewhere.

    hand_ids: (B, hand_size) long from `net.hand_card_ids`, -1 for empty. None
    when the weighting is off (uniform).
    """
    import torch
    if ADVISOR_SLOT_WEIGHT == 1.0 or not enabled():
        return None
    # Every card with any target source. With the prior on, all deck cards
    # qualify and the draw is uniform again; with it off this reduces to the
    # rule cards.
    w = torch.ones(hand_ids.shape, dtype=torch.float32)
    for cid in target_cards():
        w[hand_ids == cid] = ADVISOR_SLOT_WEIGHT
    return w


def targets_for_batch(obs_np, card_ids, legal_table, T=None):
    """One rollout step's advisor targets.

    obs_np:    (B, obs_dim) float32, the observations just acted on
    card_ids:  (B,) int, the card in each env's sampled coverage slot, or -1
    returns:   targets (B, N_CELLS) float32, weight (B,) float32

    `weight` is 0 without a target (the row is zeros and must not be read), 1
    for a rule, HUMAN_PRIOR_COEF for the prior.
    """
    B = obs_np.shape[0]
    out = np.zeros((B, N_CELLS), dtype=np.float32)
    has = np.zeros(B, dtype=np.float32)
    for b in range(B):
        cid = int(card_ids[b])
        legal = legal_table.get(cid)
        if legal is None:
            continue
        t, w = target_and_weight(obs_np[b], cid, legal, T)
        if t is None or w <= 0.0:
            continue
        out[b] = t
        has[b] = w
    return out, has


def masked_kl_elementwise(new_logits, target_logits):
    """KL(target || new) per row, over the legal cells only. (...,) tensor.

    `distill_tactics.masked_kl` without the final mean, so the caller can mask
    rows first. The -inf operands on illegal cells are replaced before the
    arithmetic, not after: torch.where after the fact fixes the forward but
    still differentiates the unselected branch, putting NaN in the gradient (it
    did, for a fully masked row). Bit-identical on every reachable input.
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

    Returns (loss_delta, ent_frac, kl_mean, n_target). `loss_delta` is added to
    the total loss and already carries both signs (entropy bonus negative, KL
    penalty positive). Each row gets KL to its target or the entropy bonus,
    never both: one says "be here", the other "be spread out".
    """
    import torch
    from torch.distributions import Categorical

    # has_target is a weight (1.0 rule, HUMAN_PRIOR_COEF prior), so "has a
    # target" and "how hard it pulls" are separate. Deriving the entropy mask
    # as `1 - has_target` would give a 0.1-weight row both terms.
    has_row = (has_target > 0).to(has_target.dtype)
    has = has_target * decision           # weighted, for the KL
    no_t = (1.0 - has_row) * decision     # masked, for the entropy
    n_has = (has_row * decision).sum()    # rows, not summed weight; see below
    n_no = no_t.sum()

    cf_ent = Categorical(logits=cf_logits).entropy()
    # Averaged over the rows it applies to, or the fallback would weaken as
    # advisor coverage rises.
    ent_mean = (cf_ent * no_t).sum() / n_no.clamp(min=1.0)
    ent_frac = ent_mean / log_n_placement
    loss_delta = -entropy_coef * ent_frac

    kl_mean = torch.zeros((), device=cf_logits.device)
    if float(n_has) > 0.0 and ADVISOR_COVERAGE_COEF > 0.0:
        kl_elems = masked_kl_elementwise(cf_logits, targets)
        # Divide by the row count, not the summed weight, which would normalise
        # a uniform weight away and make the coefficient do nothing. With all
        # weights 1.0 the two agree.
        kl_mean = (kl_elems * has).sum() / n_has.clamp(min=1.0)
        loss_delta = loss_delta + ADVISOR_COVERAGE_COEF * kl_mean / log_n_placement

    # Only loss_delta carries the graph; the diagnostics are detached.
    return loss_delta, ent_frac.detach(), kl_mean.detach(), n_has.detach()
