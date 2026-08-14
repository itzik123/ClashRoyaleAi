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

import tactics

CANNON_ID = tactics.CANNON_ID
FIREBALL_ID = tactics.FIREBALL_ID
GIANT_ID = tactics.GIANT_ID
BOARD_W = tactics.BOARD_W
BOARD_H = tactics.BOARD_H
N_CELLS = BOARD_H * BOARD_W

# Which advisor rule produces each card's surface. "cell" means the rule yields
# a single cell rather than a scored map, so its target is a delta.
ADVISOR_CARDS = {
    CANNON_ID: "building",
    FIREBALL_ID: "spell",
    GIANT_ID: "cell",
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
