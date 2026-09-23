"""What a decision-time search is configured with. Kept import-cheap:
`shipping.py` reads it.
"""
import os
from dataclasses import dataclass


#: Placement top-1 probability below which a card's head has nothing useful for
#: search to rank, so its cell proposals are widened instead. Sits in the
#: largest gap between the sharp and diffuse heads of a trained policy. Not a
#: dead-card detector: cheap cycle cards are legitimately flat.
WIDE_PROPOSAL_TOP1 = float(os.environ.get("CLASH_WIDE_PROPOSAL_TOP1", 0.25))

#: Board stride for the widened sweep.
WIDE_PROPOSAL_STRIDE = (2, 2)

#: Cap on cells proposed for one widened card, so `max_candidates` is a real
#: bound (the stride grid alone is 153 cells).
WIDE_PROPOSAL_MAX_CELLS = int(os.environ.get("CLASH_WIDE_PROPOSAL_MAX_CELLS", 48))


@dataclass(frozen=True)
class SearchCfg:
    """The exact search configuration a result was measured with; frozen so
    collection and evaluation cannot drift apart.

    Depth is nearly free and width is not: one engine step costs ~0.015 ms, one
    scored candidate a network row at ~0.13 ms. A rollout idles our side after
    the candidate; the opponent is whatever drives it (the C++ heuristic by
    default, since `sim.step` runs it, or a model passed to `search.rollout`).
    That is why search measured well against the heuristic and negative against
    the UtilityTeacher.
    """

    #: Decision steps rolled forward; 1 step = 1 s of engine time.
    horizon: int = 4
    #: Top-k card-head arms to expand.
    k_cards: int = 3
    #: Top-k cells per expanded card.
    k_cells: int = 2
    #: A finished rollout is scored by its outcome, weighted above any critic
    #: bootstrap.
    terminal_weight: float = 10.0
    #: Hard cap on episode length, so a pathological draw cannot stall a sweep.
    max_steps: int = 400

    @property
    def max_candidates(self):
        """Greedy plus the expansion grid: the worst case, every expanded card
        widened. The bound `K_MAX` is checked against.
        """
        per_card = max(self.k_cells, WIDE_PROPOSAL_MAX_CELLS)
        return 1 + self.k_cards * per_card
