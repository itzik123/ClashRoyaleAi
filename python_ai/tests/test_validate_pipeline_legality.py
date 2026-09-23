"""The pre-flight gate's placement-legality check asks the engine, not the table
the target was built from.

Checking `isfinite(t[~legal])` could never fail: `advisor_target` writes only
cells that are in `legal`. A stub engine drives both expressions with the same
input, including one the real engine would never produce.
"""
import numpy as np
import pytest

from python_ai.tools import validate_pipeline as VP
from python_ai.engine_constants import BOARD_W

N_CELLS = 612  # 18 x 34; checked against the board below
CANNON = 25


class _StubEngine:
    """Refuses exactly the cells it was told to refuse, and records what it was
    asked.
    """

    def __init__(self, refuse=()):
        self.refuse = {(float(x), float(y)) for x, y in refuse}
        self.asked = []

    def is_valid_placement(self, card_id, x, y, team=0):
        self.asked.append((card_id, x, y, team))
        return (float(x), float(y)) not in self.refuse


def _target_with_mass_at(cell):
    """A target logit vector shaped like target_logits_for's output."""
    t = np.full(N_CELLS, -np.inf, dtype=np.float32)
    t[cell] = 1.0
    return t


def test_the_cell_count_matches_the_engines_board():
    # If this fails, N_CELLS is stale and so is every flat-cell index here.
    assert N_CELLS % BOARD_W == 0


def test_the_old_check_could_not_see_a_real_violation():
    """Same input, both expressions: the old one reports clean."""
    cell = 5 * BOARD_W + 7          # (x=7, y=5)
    x, y = cell % BOARD_W, cell // BOARD_W

    legal = np.zeros(N_CELLS, dtype=bool)
    legal[cell] = True              # the net believes this cell is placeable
    target = _target_with_mass_at(cell)

    # The retired expression, verbatim: False by construction, not because the
    # board is fine.
    assert not np.isfinite(target[~legal]).any()

    # The engine refuses that exact cell.
    engine = _StubEngine(refuse=[(x, y)])
    assert VP.cells_the_engine_refuses(engine, CANNON, target) == [(cell, x, y)]


def test_a_target_the_engine_accepts_is_clean():
    """Control: the check must not fire on a legal target, or the test above would
    pass for a function that flags everything.
    """
    cell = 5 * BOARD_W + 7
    target = _target_with_mass_at(cell)
    engine = _StubEngine(refuse=[])
    assert VP.cells_the_engine_refuses(engine, CANNON, target) == []
    # ...and it really consulted the engine about that cell.
    assert (CANNON, float(cell % BOARD_W), float(cell // BOARD_W), 0) in engine.asked


def test_only_cells_carrying_mass_are_asked_about():
    """-inf cells are not placements and must not cost a pybind call each."""
    cell = 3 * BOARD_W + 2
    target = _target_with_mass_at(cell)
    engine = _StubEngine(refuse=[])
    VP.cells_the_engine_refuses(engine, CANNON, target)
    assert len(engine.asked) == 1


def test_every_finite_cell_is_checked_not_just_the_argmax():
    """Every cell carrying mass is validated: the invariant is "no mass on an
    illegal cell", not "the best cell is legal".
    """
    cells = [2 * BOARD_W + 1, 4 * BOARD_W + 3, 6 * BOARD_W + 5]
    target = np.full(N_CELLS, -np.inf, dtype=np.float32)
    for i, c in enumerate(cells):
        target[c] = float(i)        # the last one is the argmax
    # Refuse a cell that is not the argmax.
    victim = cells[0]
    engine = _StubEngine(refuse=[(victim % BOARD_W, victim // BOARD_W)])
    bad = VP.cells_the_engine_refuses(engine, CANNON, target)
    assert bad == [(victim, victim % BOARD_W, victim // BOARD_W)]
    assert len(engine.asked) == len(cells)
