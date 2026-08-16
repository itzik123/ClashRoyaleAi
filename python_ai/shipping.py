"""The shipping agent: which weights, which search settings, and why.

ONE PLACE that names the deployable configuration, so an evaluation and a
deployment cannot silently drift onto different settings -- the same reason
`expert_iteration.SearchCfg` is a class rather than an argparse namespace.

WHAT SHIPS, and the honest version of why. The 2026-08-15 placement cure left
`model_weights_cured.pth` beating v1.2.0 head-to-head (0.6125) while LOSING to
the C++ HeuristicOpponent it had stopped training against (0.51 vs v1.2.0's
0.6225) -- specialization, not degradation. The fix is NOT new weights. It is
decision-time search, which was validated in 2026-08-11 and never wired into
anything that plays.

The weights are UNCHANGED from `model_weights_cured.pth`. Everything below is
inference configuration. That is stated plainly because it is the whole result:
the gain is not in the parameters and re-training will not reproduce it.

SEARCH_HORIZON = 12 was SELECTED on an n=80 sweep and then CONFIRMED on a fresh
independent run, because selecting and reporting on the same data is how this
project once manufactured a +0.105 that a 4x-power rerun collapsed to +0.016.
The sweep (cured net vs heuristic@1.5x, n=80 each, paired):

    horizon  4 (default)  policy 0.483 -> search 0.667   1.6x cost
    horizon  4, K<=13     policy 0.450 -> search 0.788   1.7x cost
    horizon  8            policy 0.525 -> search 0.925   1.5x cost
    horizon 12            policy 0.563 -> search 0.963   1.5x cost   <-- chosen
    horizon 20            policy 0.488 -> search 0.875   1.8x cost

DEPTH IS NEARLY FREE AND WIDTH IS NOT. One engine step costs 0.015 ms and one
scored candidate costs a network row at 0.13 ms, so horizon buys lookahead at
~1/9 the price of an extra candidate -- which is why horizon 12 is both the
strongest and the CHEAPEST setting here. Past ~12 it degrades: a candidate
rollout assumes both sides no-op, and 20 s of that stops resembling the game.

CONFIRMED (fresh runs, not the sweep):

    vs C++ heuristic@1.5x, n=400 paired   policy 0.5200 -> search 0.9225
                                          delta +0.4025 CI [+0.3486, +0.4564]
                                          174 better / 13 worse / 213 tied
    h2h vs v1.2.0 as shipped, n=150       0.7200  CI [0.6750, 0.7650]
    inference cost                        1.5x wall clock, 13.2% of decisions
                                          overridden

READ THIS BEFORE CLAIMING THE CURED WEIGHTS ARE STRONGER. Give BOTH sides
search and the advantage disappears: cured scores 0.4425, CI [0.3825, 0.5025]
(n=100, sides swapped) against v1.2.0 -- no difference resolved, and the point
estimate favours v1.2.0. The +0.72 above is SEARCH, not the weights. Search and
the 2026-08-14 placement cure repair the same weakness and do not stack. The
deployable agent is better; the network is not.

WHAT IS NOT INCLUDED, deliberately:
  * The tactical placement override (`hybrid_policy.py`). Measured redundant on
    the cured net -- +0.003 / -0.028 / -0.028 across three arms, all null,
    against +11.8 points at v1.2.0. There is no longer a hole for it to fill.
  * `SolvencyGate` is left available but UNMEASURED on top of search. It fixed
    bankruptcy outright (72.7% -> 39.2%) while moving win rate not at all, and
    nothing has tested whether it still helps once search is choosing actions.
    Do not assume it composes; measure it before turning it on.
"""
import os

# --------------------------------------------------------------------------
# the configuration
# --------------------------------------------------------------------------

SHIPPING_WEIGHTS = "model_weights_cured.pth"

SEARCH_HORIZON = 12        # decision steps rolled forward; 1 step = 1 s
SEARCH_K_CARDS = 3         # top-k card head arms expanded
SEARCH_K_CELLS = 2         # top-k cells per expanded card
SEARCH_TERMINAL_WEIGHT = 10.0   # a finished rollout uses its OUTCOME, weighted
                                # to dominate any bootstrapped critic value

USE_TACTICAL_OVERRIDE = False   # measured redundant -- see module docstring
USE_SOLVENCY_GATE = False       # unmeasured on top of search -- see docstring


def search_cfg():
    """The validated SearchCfg, for search_ab_test / expert_iteration callers."""
    from expert_iteration import SearchCfg
    return SearchCfg(horizon=SEARCH_HORIZON,
                     k_cards=SEARCH_K_CARDS,
                     k_cells=SEARCH_K_CELLS,
                     terminal_weight=SEARCH_TERMINAL_WEIGHT)


def load_shipping_net(device=None):
    """The shipping net, loaded from this file's own directory."""
    import torch
    from expert_iteration import load_net
    here = os.path.dirname(os.path.abspath(__file__))
    net = load_net(os.path.join(here, SHIPPING_WEIGHTS),
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
