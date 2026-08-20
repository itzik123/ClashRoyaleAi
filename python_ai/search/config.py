"""What a decision-time search is configured with.

MOVED OUT OF `trainers/expert_iteration.py`, where it lived until 2026-08-20.
That mattered: `shipping.py` -- the one file that names the deployable
configuration -- had to import a 1,162-line experiment harness (and through it
`bc_pretrain`, `search_ab_test`, numpy, torch and a card registry probe) purely
to reach this dataclass. A configuration object should be the cheapest thing in
the tree to import, not the most expensive.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchCfg:
    """The exact search configuration a result was measured with.

    A frozen object rather than an argparse namespace so collection and
    evaluation cannot silently drift onto different settings -- expert labels
    are only expert labels for the configuration that produced them.

    HORIZON IS THE LEVER, AND DEPTH IS NEARLY FREE. One engine step costs
    0.015 ms; one scored candidate costs a network row at 0.13 ms, so lookahead
    is ~9x cheaper than width. Measured sweep (cured net vs heuristic@1.5x,
    n=80 paired): horizon 4 -> 0.667, 8 -> 0.925, 12 -> 0.963, 20 -> 0.875.

    It degrades past ~12 because a candidate rollout assumes BOTH SIDES NO-OP,
    and 20 s of that stops resembling the game. That is the same reason
    `opponents/teacher.py` caps its own lookahead at 10 s.
    """

    #: Decision steps rolled forward. 1 step = 1 s of engine time.
    horizon: int = 4
    #: Top-k arms of the card head to expand.
    k_cards: int = 3
    #: Top-k cells per expanded card.
    k_cells: int = 2
    #: A rollout that ENDED is not a position to be valued -- the critic's
    #: estimate of a finished game is meaningless -- so its real outcome is used
    #: instead, weighted to dominate any bootstrapped value.
    terminal_weight: float = 10.0
    #: Hard cap on episode length, so a pathological draw cannot stall a sweep.
    max_steps: int = 400

    @property
    def max_candidates(self):
        """Greedy plus the expansion grid. The bound `K_MAX` is checked against."""
        return 1 + self.k_cards * self.k_cells
