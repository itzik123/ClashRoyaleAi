"""Elixir solvency shaping -- the fix for the measured bankruptcy.

THE MEASUREMENT THIS EXISTS FOR
-------------------------------
`model_weights_dist_e3.pth`, 40 greedy episodes at 1.5x opponent elixir:

* it spends ~105 elixir per episode against ~98 of income;
* it sits below 3 elixir -- the cheapest card in the deck -- on **65.3%** of all
  decisions;
* during a BIG enemy push, **60.8%** of decisions are below 3 elixir, where
  P(play) is 0.0% by arithmetic rather than by choice.

Conditioned on affordability its threat response is real (27.8% -> 34.9%), so
this is not apathy and not hoarding. It is insolvency: the answer is unaffordable
at the moment it is needed.

TWO EXPLANATIONS WERE TESTED. THE OBVIOUS ONE IS WRONG.
-------------------------------------------------------
The tempting mechanical explanation was that the card-head entropy controller
forces the spending: `LOG_N_CARD = log(5)` but the affordability-masked
distribution often has only 2-3 legal options, so a 0.35-of-log-5 target could
be a floor on P(play). That is the same normalization defect this project has
paid for twice before.

Measured over 601 decision steps, it is NOT what is happening:

    controller target        0.35 of log 5  = 0.5633 nats
    measured card entropy    0.4910 of log 5 = 0.7905 nats   (ABOVE target)
    P(play) the target forces           18.1%
    P(play) the policy actually carries 45.1%

The policy is voluntarily 2.5x more play-happy than the entropy target requires,
and is already above the target it is being regulated toward. The controller is
not the cause; the policy genuinely over-values spending.

WHY POTENTIAL-BASED, AND NOT A PLAIN PENALTY
--------------------------------------------
That distinction decides the tool. Over-valuing spending is a CREDIT-ASSIGNMENT
failure, not a mis-specified objective, and the proof is already in CLAUDE.md:
decision-time search optimises the SAME reward and gains +0.319 win rate, with
87% of its overrides being "wait where greedy plays" (1,847 vs 137). Under the
existing objective, waiting more is already better -- the policy simply has not
found it. The cost of spending now lands seconds later, discounted across a
~281-decision episode, and never reaches the action that caused it.

Potential-based shaping is exactly the tool for that case and is the only one
that is SAFE here. By Ng et al., F = gamma*Phi(s') - Phi(s) cannot change which
policy is optimal -- it only redistributes credit. So:

* if over-spending really is optimal, this provably changes nothing (the same
  safety property `lethal_spell_potential` relies on);
* it cannot manufacture the guaranteed-zero trap this codebase has hit three
  times (the symmetric elixir term drove both sides to mutual passivity; a
  free timeout made stalling safe; a zero-cost decaying Cannon made a back
  corner optimal). A plain "hold a reserve" penalty could, because doing
  nothing would then pay directly.

The whole episode's contribution telescopes to gamma^T*Phi(s_T) - Phi(s_0), and
both ends are ~0 at ordinary elixir levels, so the term adds nothing to the
return. What it changes is WHEN the agent is told: dipping below the reserve is
charged at the instant of the spend instead of implicitly, many steps later,
through a tower it could not defend.
"""
import numpy as np

# Elixir the agent should still hold after acting. Set to 4.0 from the deck, not
# tuned: the cheapest card is 3 (Archers/Minions/Cannon) and every dedicated
# defensive answer in DEFAULT_DECK -- Valkyrie, Musketeer, Mini PEKKA -- costs 4.
# Below this the agent cannot answer anything that matters, which is precisely
# the 60.8%-of-a-big-push state measured above.
SOLVENCY_RESERVE = 4.0

# Scale of the potential, in the same units as the other shaping weights and
# numerically equal to rewards/weights.py's W_ELIXIR_OVERFLOW: this term is that one's
# conceptual mirror image. Overflow penalises sitting above 9 elixir (wasting
# regen); this penalises sitting below 4 (unable to answer). Together they
# define a healthy band of 4-9 in which the agent is charged nothing and is
# free to play the game.
#
# DELIBERATELY NOT SHARED, and do not "fix" this into an import. Two reasons:
# (1) train.py imports W_SOLVENCY FROM this module, so the dependency already
#     runs elixir_shaping -> train; importing back is circular in both entry
#     orders at module level.
# (2) They are different quantities that happen to be tuned to the same value --
#     this one is a potential-based (policy-invariant) band-keeping term, that
#     one is a per-step penalty. They are free to diverge, and an equality test
#     asserting otherwise would be asserting a coincidence.
#
# Magnitude matters only for LEARNING SPEED here, not for the optimum, since the
# term is potential-based. Kept modest anyway: policy-invariance is an
# asymptotic statement about the optimum, and under function approximation a
# large potential can still distort a finitely-trained policy.
W_SOLVENCY = 0.1


def solvency_potential(elixir, reserve=SOLVENCY_RESERVE, w=W_SOLVENCY):
    """Phi(s): 0 at or above the reserve, falling to -w at zero elixir.

    Linear rather than a step, so the gradient distinguishes "one elixir short"
    from "completely broke". A step function would make every state below the
    reserve identical and give the agent no reason to prefer 3.9 to 0.1.

    `elixir` is an array of the CURRENT elixir reading (never a delta).
    """
    shortfall = np.maximum(0.0, reserve - np.asarray(elixir, dtype=np.float32))
    return (-w * shortfall / reserve).astype(np.float32)


def solvency_shaping(stats, prev_stats, gamma,
                     reserve=SOLVENCY_RESERVE, w=W_SOLVENCY):
    """F = gamma*Phi(s') - Phi(s), the discounted form Ng et al.'s invariance needs.

    Needs, not guarantees: invariance also requires Phi(terminal) = 0, which
    nothing here enforces. A match that ends with the bar below the reserve
    leaves gamma^T * Phi(s_T) < 0 behind -- measured small (at most -0.019, in
    2 of 16 mirror matches); see `rewards/weights.py`'s policy-invariance note.

    THE DISCOUNT IS REQUIRED, NOT DEFAULTED, and that is the whole guarantee.
    This term and the tower term in `shaping.py` are the two potential-based
    ones, and Ng et al.'s policy-invariance result holds for either only when
    this gamma is the SAME one GAE discounts with. It used to default to a literal 0.99 -- harmless only while
    `PPOConfig.gamma` also read 0.99, and a silent invariance break the moment
    it did not. Deriving the default was not available either: `rewards/` is an
    enforced leaf layer that may import neither `rl/` nor `trainers/`
    (`tests/test_package_layout.py`). Requiring the argument satisfies the
    no-second-copies rule by ABSENCE rather than by derivation, which is the
    stronger form -- there is no copy here to go stale, and no caller can
    compute potential-based shaping without stating the discount it is for.


    The gamma is load-bearing and is not decoration: Ng et al.'s invariance
    result requires the discounted difference, and the undiscounted version is a
    DIFFERENT, biased shaping that looks almost identical. CLAUDE.md records this
    project shipping exactly that mistake once already on the tower term.
    """
    return (gamma * solvency_potential(stats["team0_elixir_current"], reserve, w)
            - solvency_potential(prev_stats["team0_elixir_current"], reserve, w)
            ).astype(np.float32)


def bankruptcy_rate(elixir, floor=3.0):
    """Share of decisions below `floor` elixir -- the statistic to watch.

    THE 3.0 IS A CONSTRAINED-ECONOMY THRESHOLD, NOT AN EMPTY ACTION SPACE. This
    docstring used to justify it as "the cost of the cheapest card in
    DEFAULT_DECK, so below it the action space is literally empty and P(play) is
    0.0% by arithmetic". That was true of the Giant deck it was written for and
    is FALSE of the 2.6 Hog Cycle adopted 2026-08-16, whose costs are
    [4, 4, 3, 2, 1, 1, 2, 4]: the cheapest card costs 1, and at 2.0 elixir the
    agent can still play Skeletons, Ice Spirit, Ice Golem or The Log.

    The number is KEPT at 3.0 anyway, deliberately: it is the Cannon's cost and
    the point below which the deck's defensive answer is unaffordable, it is
    still a meaningful "economically constrained" line, and the 65.3% overall /
    60.8%-during-a-push baselines in CLAUDE.md were measured against it.
    Changing the default would silently redefine a number those figures are
    quoted for. Only the JUSTIFICATION was wrong.

    `tests/test_elixir_shaping.py` now pins the real cheapest cost against the
    engine, so this cannot go stale a second time -- it is the same
    second-copy-of-an-engine-constant drift CLAUDE.md already records for
    model.py's "18*16=288" and calibrate.py's y=17.0.
    """
    return float(np.mean(np.asarray(elixir, dtype=np.float32) < floor))
