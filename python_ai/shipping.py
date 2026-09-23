"""The shipping agent: which weights and which search settings.

One place names the deployable configuration, so an evaluation and a deployment
cannot drift onto different settings.

Search is off. Against the UtilityTeacher on the deck pool it loses to the
greedy policy at every horizon tried (horizon 12: 0.700 -> 0.267, p = 0.001),
because a candidate rollout is stepped by the C++ HeuristicOpponent rather than
by an opponent like the one it faces. The search settings below were chosen
against that heuristic (horizon 12 was the strongest and cheapest; past it a
no-op rollout stops resembling the game) and need re-measuring before search is
turned back on. See docs/TODO.md item 0e.

Not included: the tactical placement override (measured redundant on an older
net) and SolvencyGate (never measured on top of search).
"""
import os
import sys

# Run as a script, the repo root is not on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import python_ai  # noqa: E402,F401

# A frozen copy of a training checkpoint, so a resumed run cannot change what
# ships.
SHIPPING_WEIGHTS = "model_weights_live.pth"

#: Off until search is re-validated against the teacher; see the module
#: docstring. test_shipping_does_not_use_search_until_it_is_re_validated
#: enforces it.
USE_SEARCH = False

SEARCH_HORIZON = 12        # decision steps rolled forward; 1 step = 1 s
SEARCH_K_CARDS = 3         # top-k card head arms expanded
SEARCH_K_CELLS = 2         # top-k cells per expanded card
SEARCH_TERMINAL_WEIGHT = 10.0   # a finished rollout scores its outcome,
                                # weighted above any critic bootstrap

USE_TACTICAL_OVERRIDE = False   # see module docstring
USE_SOLVENCY_GATE = False       # see module docstring


def search_cfg():
    """The validated SearchCfg, for the search and expert-iteration callers."""
    from python_ai.search.config import SearchCfg
    return SearchCfg(horizon=SEARCH_HORIZON,
                     k_cards=SEARCH_K_CARDS,
                     k_cells=SEARCH_K_CELLS,
                     terminal_weight=SEARCH_TERMINAL_WEIGHT)


def load_shipping_net(device=None):
    """The shipping net, loaded from the package directory, never the caller's
    cwd.
    """
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
