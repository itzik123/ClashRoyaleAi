"""The shipping agent: which weights, which search settings, and why.

> **WARNING (2026-09-06): THE SEARCH SETTINGS BELOW ARE MEASURED NEGATIVE
> AGAINST THE CURRENT OPPONENT. Ship the policy GREEDY until this is
> re-validated.**
>
> Every number in this file was measured against the C++ HeuristicOpponent.
> Re-measured against the UtilityTeacher on the 16-deck pool with the ep-111k
> policy (`eval/search_vs_greedy_pool_ab.py`, paired, n=30, seeded teacher),
> this exact configuration -- horizon 12 -- scores:
>
>     greedy 0.700   search 0.267   delta -0.433 [-0.633, -0.233]  p = 0.00098
>
> and it is negative at horizon 4 and 8 as well. The cause is not a regression
> in search: a candidate rollout assumes BOTH SIDES NO-OP, which models the C++
> heuristic passably and a forward-simulating teacher badly. See TODO 0e for the
> full table and for the two hypotheses that were tested and refuted.
>
> The horizon sweep below is therefore a record of what was true against the
> heuristic, not a current recommendation.


ONE PLACE that names the deployable configuration, so an evaluation and a
deployment cannot silently drift onto different settings -- the same reason
`search.config.SearchCfg` is a frozen dataclass rather than an argparse
namespace.

READ THIS FIRST: THE WEIGHTS CHANGED AND THE EVIDENCE DID NOT MOVE WITH THEM
---------------------------------------------------------------------------
Until the 2026-08-19 cleanup this named `model_weights_cured.pth` (episode
78,214). That net was trained on the GIANT deck, which `DEFAULT_DECK` left on
2026-08-16 for the 2.6 Hog Cycle, and it was retired along with it. The
surviving baseline is `model_weights_selfplay.pth` (episode 31,312), the 2.6
net.

Everything measured below -- the horizon sweep, the +0.4025 against the C++
heuristic, the head-to-head against v1.2.0 -- was measured ON THE RETIRED NET,
against a deck that is no longer the default, and on an engine that has since
gained a 1.0 s deploy time and lost the opponent-elixir curriculum. **None of
those numbers describe the current configuration.** They are kept because the
STRUCTURAL findings behind them (depth is cheap, width is not; past ~12 a
no-op rollout stops resembling the game) are properties of the search, not of
the weights -- but the win rates are historical and must be re-measured before
anyone quotes them again.

This file is deliberately not deleted along with the checkpoint it named. Its
job is to be the single place a deployment reads, and leaving it pointing at a
missing file is how an evaluation and a deployment drift apart -- which is the
exact failure it exists to prevent.

WHAT IS STILL BELIEVED, AND ON WHAT BASIS
-----------------------------------------
SEARCH_HORIZON = 12 was SELECTED on an n=80 sweep and then CONFIRMED on a fresh
independent run, because selecting and reporting on the same data is how this
project once manufactured a +0.105 that a 4x-power rerun collapsed to +0.016.
The sweep (retired cured net vs heuristic@1.5x, n=80 each, paired):

    horizon  4 (default)  policy 0.483 -> search 0.667   1.6x cost
    horizon  4, K<=13     policy 0.450 -> search 0.788   1.7x cost
    horizon  8            policy 0.525 -> search 0.925   1.5x cost
    horizon 12            policy 0.563 -> search 0.963   1.5x cost   <-- chosen
    horizon 20            policy 0.488 -> search 0.875   1.8x cost

DEPTH IS NEARLY FREE AND WIDTH IS NOT. One engine step costs 0.015 ms and one
scored candidate costs a network row at 0.13 ms, so horizon buys lookahead at
~1/9 the price of an extra candidate -- which is why horizon 12 was both the
strongest and the CHEAPEST setting there. Past ~12 it degraded: a candidate
rollout assumes both sides no-op, and 20 s of that stops resembling the game.
That cost argument is engine arithmetic and survives the weight change; the win
rates attached to it do not.

READ THIS BEFORE CLAIMING SEARCH MAKES A NET STRONGER. On the retired net,
giving BOTH sides search erased the whole advantage: cured scored 0.4425, CI
[0.3825, 0.5025] (n=100, sides swapped) against v1.2.0 -- no difference
resolved, point estimate favouring v1.2.0. The +0.72 headline was SEARCH, not
the weights. Search and the 2026-08-14 placement cure repaired the same
weakness and did not stack.

WHAT IS NOT INCLUDED, deliberately:
  * The tactical placement override (`hybrid_policy.py`). Measured redundant on
    the cured net -- +0.003 / -0.028 / -0.028 across three arms, all null,
    against +11.8 points at v1.2.0. Never re-measured on the 2.6 net, so this
    is an inherited default, not a fresh result.
  * `SolvencyGate` is left available but UNMEASURED on top of search. It fixed
    bankruptcy outright (72.7% -> 39.2%) while moving win rate not at all, and
    nothing has tested whether it still helps once search is choosing actions.
    Do not assume it composes; measure it before turning it on.
"""
import os
import sys

# Run as a script (`python python_ai/shipping.py`) the repo root is not on
# sys.path. See python_ai/__init__.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import python_ai  # noqa: E402,F401

# --------------------------------------------------------------------------
# the configuration
# --------------------------------------------------------------------------

# The surviving 2.6 Hog Cycle baseline (episode 31,312). Replace this with the
# Episode 0 run's output once that run has been measured -- not before, and not
# by assuming the numbers in the docstring carry over.
#: model_weights_selfplay.pth was DELETED in the 2026-08-19 cleanup, so this
#: pointed at a missing file and every deployment path raised
#: FileNotFoundError. Frozen copy of phase 9 at ep 115,173 -- the only
#: checkpoint that loads clean against the current observation layout, and
#: a COPY so a resumed training run cannot mutate what ships.
SHIPPING_WEIGHTS = "model_weights_live.pth"

#: THE SHIPPING SWITCH. False since 2026-09-06: search is measured NEGATIVE
#: against the opponent phase 1 actually trains on, so the deployable agent
#: runs its policy GREEDY.
#:
#: Paired, seeded, UtilityTeacher rung 3 on the 16-deck pool, ep-111k policy,
#: `eval/search_vs_greedy_pool_ab.py`. The greedy control reads 0.844 in all
#: three, which is what makes the pairing credible:
#:
#:     horizon  4   greedy 0.844   search 0.531   -0.313 [-0.531, -0.125]
#:     horizon  8   greedy 0.844   search 0.312   -0.531 [-0.719, -0.313]
#:     horizon 12   greedy 0.844   search 0.375   -0.469 [-0.688, -0.250]
#:
#: and on THIS file's own configuration, -0.433 [-0.633, -0.233], p = 0.00098.
#:
#: A FLAG rather than a deletion, because the fault is not in the search. A
#: candidate rollout is stepped by the C++ HeuristicOpponent (`sim.step` runs
#: it), while the real opponent forward-simulates -- so search optimises against
#: a materially different and weaker opponent than the one it then faces. Every
#: positive search result in this repo, including the +0.319 and the horizon
#: sweep below, was measured against that same heuristic, which is why they held
#: at the time and do not now. Give the rollout the right opponent and this can
#: come back on -- after a re-measurement, which
#: `test_shipping_does_not_use_search_until_it_is_re_validated` exists to force.
#:
#: Widening needs no separate switch: with search off it never runs.
USE_SEARCH = False

SEARCH_HORIZON = 12        # decision steps rolled forward; 1 step = 1 s
SEARCH_K_CARDS = 3         # top-k card head arms expanded
SEARCH_K_CELLS = 2         # top-k cells per expanded card
SEARCH_TERMINAL_WEIGHT = 10.0   # a finished rollout uses its OUTCOME, weighted
                                # to dominate any bootstrapped critic value

USE_TACTICAL_OVERRIDE = False   # measured redundant -- see module docstring
USE_SOLVENCY_GATE = False       # unmeasured on top of search -- see docstring


def search_cfg():
    """The validated SearchCfg, for the search and expert-iteration callers."""
    from python_ai.search.config import SearchCfg
    return SearchCfg(horizon=SEARCH_HORIZON,
                     k_cards=SEARCH_K_CARDS,
                     k_cells=SEARCH_K_CELLS,
                     terminal_weight=SEARCH_TERMINAL_WEIGHT)


def load_shipping_net(device=None):
    """The shipping net, loaded from the package directory the .pth files
    live in -- never from the caller's cwd, which is what made a harness and a
    deployment able to load two different checkpoints under one name."""
    import torch
    from python_ai.models.policy_io import load_net
    net = load_net(os.path.join(python_ai.PACKAGE_DIR, SHIPPING_WEIGHTS),
                   device or torch.device("cpu"))
    for p in net.parameters():
        p.requires_grad_(False)
    return net


if __name__ == "__main__":
    print(__doc__)
    print(f"weights          : {SHIPPING_WEIGHTS}")
    print(f"search horizon   : {SEARCH_HORIZON} decision steps "
          f"({SEARCH_HORIZON} s of lookahead)")
    print(f"candidates       : k_cards={SEARCH_K_CARDS} k_cells={SEARCH_K_CELLS} "
          f"(K <= {1 + SEARCH_K_CARDS * SEARCH_K_CELLS})")
    print(f"tactical override: {USE_TACTICAL_OVERRIDE}")
    print(f"solvency gate    : {USE_SOLVENCY_GATE}")
