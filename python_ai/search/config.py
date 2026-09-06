"""What a decision-time search is configured with.

MOVED OUT OF `trainers/expert_iteration.py`, where it lived until 2026-08-20.
That mattered: `shipping.py` -- the one file that names the deployable
configuration -- had to import a 1,162-line experiment harness (and through it
`bc_pretrain`, `search_ab_test`, numpy, torch and a card registry probe) purely
to reach this dataclass. A configuration object should be the cheapest thing in
the tree to import, not the most expensive.
"""
import os
from dataclasses import dataclass


#: Placement top-1 probability below which a card's head is treated as having
#: nothing useful for search to RANK, so its cell proposals are widened instead.
#:
#: MEASURED SITING. Mean top-1 per deck card on the ep-32,484 policy: Hog
#: 0.7654, Musketeer 0.3939, Skeletons 0.3527 | Ice Golem 0.1259, Fireball
#: 0.1074, Ice Spirit 0.0987, Cannon 0.0970, The Log 0.0288. The largest gap
#: inside the diffuse region is 0.227 and 0.25 sits in it.
#:
#: NOT a dead-card detector: Ice Golem and Ice Spirit are heavily played and
#: their heads are just as flat, because for a cheap cycle card placement
#: genuinely matters less. It selects "search has nothing here to rank".
WIDE_PROPOSAL_TOP1 = float(os.environ.get("CLASH_WIDE_PROPOSAL_TOP1", 0.25))

#: Board stride for the widened sweep.
WIDE_PROPOSAL_STRIDE = (2, 2)

#: Hard cap on cells proposed for ONE widened card.
#:
#: This exists so `max_candidates` is a PROOF rather than the largest sample
#: anyone happened to see. Unbounded, the stride grid alone is
#: ceil(34/2) * ceil(18/2) = 153 cells per card, so three expanded cards could
#: emit 460 candidates -- while the observed max was 150. Sizing a schema width
#: on that observation would have been sizing it on luck.
WIDE_PROPOSAL_MAX_CELLS = int(os.environ.get("CLASH_WIDE_PROPOSAL_MAX_CELLS", 48))


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

    CORRECTED 2026-09-06: a candidate rollout does NOT assume both sides no-op.
    OUR side no-ops, and the OPPONENT is whatever drives the rollout -- the C++
    HeuristicOpponent by default, because `sim.step` runs it, or an explicit
    model passed to `search.rollout`. Verified on the board rather than argued:
    a rollout of 400 ticks with our side idle put 2 enemy bodies out and took
    1302 of our tower hp.
    
    The sentence this replaces cost a wrong diagnosis. The real mismatch is that
    search optimised against the HEURISTIC while phase 1's opponent became the
    UtilityTeacher, which is why every horizon in this sweep now measures
    negative against the teacher (-0.313 at 4, -0.531 at 8, -0.469 at 12) while
    these numbers still reproduce against the heuristic they were taken from.
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
        """Greedy plus the expansion grid. The bound `K_MAX` is checked against.

        ACCOUNTS FOR WIDENING since 2026-09-03. It returned
        `1 + k_cards*k_cells` = 7 while the real search emitted up to 150, so
        the padding-width test it feeds passed for a schema that could not hold
        a row -- a guard that had stopped guarding without failing, the same
        shape as the receptive-field test that kept measuring 10x10 across the
        change that invalidated its docstring.

        A widened card contributes at most `WIDE_PROPOSAL_MAX_CELLS`; a sharp
        one at most `k_cells`. The bound takes the worst case, which is every
        expanded card being flat.
        """
        per_card = max(self.k_cells, WIDE_PROPOSAL_MAX_CELLS)
        return 1 + self.k_cards * per_card
