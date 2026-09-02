"""A floor under P(play card | card in hand), per deck card.

WHAT THIS EXISTS TO CATCH, AND WHY NOTHING ELSE DID
---------------------------------------------------
Across the 2026-08-28 phase-1 run the agent used five of its eight cards for
30,000 consecutive episodes. Every instrument in the loop read healthy:

  * `EntropyController` held `target_card = 0.35` almost exactly (0.339 ->
    0.350) and never approached its 0.5 ceiling. It measures entropy over the
    four hand SLOTS plus no-op, per decision -- which is blind to the marginal
    distribution over the DECK. A policy that no-ops ~81% of the time and
    spreads the rest over five cards sits at that target forever.
  * `Entropy/Placement_ByCard/*` only logs cards that were PLAYED, so a card at
    zero usage vanishes from the readout instead of alarming.
  * The advisor coverage term (`advisors/advisor_target.py`) does act on
    unplayed cards, but only on the three that have hand-written rules, and its
    KL to the advisor surface still sat at 0.73 after 24,000 episodes.

Measured on the final checkpoint (ep 32,484), P(play card | card in hand):

    Skeletons   0.3429      Hog Rider   0.0704      The Log    0.0091
    Ice Spirit  0.3247      Musketeer   0.0662      Cannon     0.0053
    Ice Golem   0.1868                              Fireball   0.0011

Five live cards at 0.066-0.343, three dead ones at 0.0011-0.0091, and a factor
of ~7 of empty space between the two groups.

THE SHAPE OF THE HINGE, AND THE MEASUREMENT THAT FORCED IT
----------------------------------------------------------
The first version hinged on the probability directly, `relu(floor - p)`. That
is wrong in a way no coefficient can repair: the gradient reaching the card's
logit carries a softmax factor `p*(1-p)`, so it VANISHES exactly as the card
dies. Measured on the real distribution:

    p_dead     d(linear)/d(logit)     d(log)/d(logit)
    0.0011              0.000275              0.2497     <- Fireball, measured
    0.0150              0.003694              0.2463

Seventeen times weaker at the value that actually needed help. Three paired
18-minute arms resuming the ep-32,484 checkpoint confirmed it live -- change in
MinCardProb was -0.0007 at coef 0, +0.0018 at coef 2 and +0.0004 at coef 8:
non-monotone in the coefficient and inside the update-to-update noise, i.e. a
null. Quadrupling the coefficient did nothing because the term was fighting its
own shape.

Hinging on `log p` instead gives `d(-log p)/d(logit) = (1 - p)`, which is ~1 for
any card worth rescuing and flat in p, so the push does not fade as the card
dies. It also reads naturally: `relu(log(floor/p))` is "how many e-folds below
the floor is this card", and it is exactly zero at and above the floor.

A HINGE, NOT A TARGET -- and that distinction is the whole design
-----------------------------------------------------------------
`DECK_COVERAGE_FLOOR` sits inside that gap, so the term is EXACTLY ZERO for
every healthy card and only ever acts on a frozen one. It must not be an
entropy-style target: pushing card use toward uniform would be a strictly worse
policy (a 2.6 Hog deck genuinely should cycle Skeletons far more often than it
Fireballs), and it would fight the policy gradient everywhere instead of only
where a head has died.

It is also NOT part of the PPO objective. Like the advisor coverage term it
never touches `new_logprobs`, so the importance ratio is untouched and the
update stays a valid PPO step -- see `rl/ppo.py`'s comment at the coverage call
site for why that separation is load-bearing.

THE DENOMINATOR IS "IN HAND", NOT "AFFORDABLE"
-----------------------------------------------
`card_logits` arrive already `masked_fill(-inf)` on unaffordable arms, so an
unaffordable card contributes p = 0 to its own mean and drags it down. That is
deliberate and matches the table above, which was measured the same way -- but
it has two consequences worth knowing before reading the metric:

  * the term is slightly stronger for EXPENSIVE cards, which are unaffordable
    more often. Fireball is the most-affected and also the deadest, so the bias
    points the right way here, but it is a bias and not a neutral choice.
  * `Deck/MinCardProb` is therefore NOT the probe's `take-up`, which divides by
    affordability instead. The two answer different questions and will not
    agree; compare a run against itself, never across the two instruments.

The GRADIENT is unaffected by the dilution: a -inf arm has zero gradient, so the
push lands only on rows where the card could actually have been played.

WHAT IT CANNOT DO
-----------------
A floor keeps a card's logit off zero; it cannot make the card good. If the
placement head for that card is poor the card stays -EV and the floor merely
pays a small, bounded price to keep exploring it. The companion fix is on the
state-distribution side (defensive scenario injection in phase 1), which is
what makes the exploration worth anything.
"""
import math
import os

import torch
import torch.nn.functional as F

#: Minimum P(play card | card in hand) below which a card is being pushed up.
#: Read off the measured separation above, not tuned on an outcome -- the same
#: discipline CLAUDE.md records for the advisor target's temperature, where a
#: value picked on the outcome put the target at 94% of maximum entropy while
#: looking busy. 0.02 is 3.3x below the weakest live card and 2.2x above the
#: strongest dead one.
DECK_COVERAGE_FLOOR = float(os.environ.get("CLASH_DECK_COVERAGE_FLOOR", 0.02))

#: Floating-point guard on log(p). Never reached by a live card; it exists so a
#: card that was unaffordable across an ENTIRE minibatch cannot produce -inf.
_EPS = 1e-9

#: Weight on the penalty. MUCH smaller than the linear form's, because the log
#: hinge is ~150x larger in magnitude: one dead card at p = 0.0011 contributes
#: log(0.02/0.0011)/4 = 0.725 raw, against 0.0047 before.
#:
#: DEFAULT 0.0 -- THE TERM IS OFF, BECAUSE IT IS MEASURED HARMFUL.
#:
#: It works mechanically: the cards climb. It costs more win rate than the
#: cards are worth. Paired arms resuming the ep-32,484 checkpoint at stage 3,
#: all from an identical copy, scenario injection ON in every arm:
#:
#:     coef   Win_Rate_100      MinCardProb          H_card
#:     0.00   0.600 -> 0.520    0.00126 -> 0.00058   0.34 -> 0.32
#:     0.05   0.600 -> 0.340    0.00093 -> 0.00189   0.38 -> 0.43
#:     0.20   0.620 -> 0.120    0.00177 -> 0.00892   0.38 -> 0.57
#:
#: A clean dose-response in the WRONG direction: the harder the floor pushes,
#: the better the deck coverage and the worse the agent plays. Confirmed live
#: on a full launch, which reproduced the 0.20 arm exactly -- 0.60 -> 0.11 over
#: 180 episodes, monotone, reward +1.96 -> -1.46. The 0.00 control isolates the
#: cause: scenario injection alone costs 0.08, the floor costs the other 0.42.
#:
#: WHY, and it is this module's own stated caveat turning out to be decisive
#: rather than marginal (see WHAT IT CANNOT DO): a floor keeps a card's logit
#: off zero, it cannot make the card GOOD. Cannon placement captures 27.5% of
#: achievable value and Log/Fireball are worse, so forcing those cards into
#: play just spends elixir on placements that do not work. The card head was
#: not broken -- it was correctly pricing a broken placement head.
#:
#: THE ORDER IS WRONG, NOT THE MECHANISM. Fix what makes those cards worth
#: playing (scenario injection creating the states, and a placement head that
#: can exploit them), and the policy gradient should revive the cards on its
#: own. Re-enable this only to accelerate a revival that is already happening,
#: and re-measure WIN RATE when you do -- never the deck metric alone.
DECK_COVERAGE_COEF = float(os.environ.get("CLASH_DECK_COVERAGE_COEF", 0.0))


def enabled():
    return DECK_COVERAGE_COEF > 0.0


def deck_coverage_penalty(card_logits, hand_ids, decision, floor=None):
    """mean over deck cards of `relu(log(floor / P(play card | card in hand)))`.

    card_logits: (N, hand_size + 1) -- the final column is the no-op arm.
    hand_ids:    (N, hand_size) long, -1 for an empty slot.
    decision:    (N,) float; rows at 0 are padding and contribute nothing.

    Returns (penalty, min_p, n_cards). `penalty` carries gradient; `min_p` and
    `n_cards` are plain Python numbers for logging.

    The no-op arm is excluded from the coverage set deliberately: it is not a
    deck card, and flooring it would be a standing pressure to play rather than
    wait -- the opposite of what this run needs, where waiting more is already
    better under the current objective.
    """
    floor = DECK_COVERAGE_FLOOR if floor is None else floor
    hand_size = hand_ids.shape[1]

    # Softmax over ALL arms including no-op -- these are the real action
    # probabilities -- then drop the no-op column from the coverage set.
    probs = torch.softmax(card_logits, dim=-1)[:, :hand_size]

    valid = (hand_ids >= 0) & (decision.reshape(-1, 1) > 0)
    if not bool(valid.any()):
        # A minibatch of pure padding. Return a real zero that still carries a
        # grad_fn, so a caller adding this to its loss cannot get a NaN from a
        # degenerate batch -- the failure `test_rl_ppo_degenerate_batches.py`
        # exists to prevent.
        return probs.sum() * 0.0, 0.0, 0

    ids = hand_ids[valid]
    p = probs[valid]

    uniq, inv = torch.unique(ids, return_inverse=True)
    zeros = torch.zeros(uniq.numel(), dtype=p.dtype, device=p.device)
    sums = zeros.index_add(0, inv, p)
    counts = zeros.index_add(0, inv, torch.ones_like(p))
    p_card = sums / counts

    # LOG space, not probability space -- see THE SHAPE OF THE HINGE above.
    # clamp_min guards log(0): p_card is a MEAN over rows and an unaffordable
    # arm contributes an exact 0.0, so a card that was never affordable in the
    # whole minibatch would otherwise produce -inf and poison the update.
    shortfall = F.relu(math.log(floor) - torch.log(p_card.clamp_min(_EPS)))
    # .detach() before the scalar read: min_p is a logging value, and pulling
    # it off the graph as a bare float otherwise warns and keeps the subgraph
    # alive for no reason.
    return shortfall.mean(), float(p_card.min().detach()), int(uniq.numel())
