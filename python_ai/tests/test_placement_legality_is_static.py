"""What the placement-legality cache may assume about board state.

`net._build_placement_legality` probes `is_valid_placement` for every (card,
cell) at construction: a base table plus a per-tower delta for our own two
Princesses (a destroyed own Princess frees its footprint; see
test_placement_mask_after_tower_loss.py). The assumptions left, each a tripwire
for an engine change:

  * an enemy tower dying changes nothing (else the delta table needs a team-1 half);
  * troops change nothing (else legality is per-step and no cached table works);
  * the cached table agrees with the engine cell for cell.
"""
import pytest

import clash_royale_env as E
from python_ai.models.net import MicroRoyaleNet

DECK = [10, 1, 41, 25, 7, 2, 6, 5]
BOARD_W = E.ClashRoyaleEnv.BOARD_WIDTH
BOARD_H = E.ClashRoyaleEnv.BOARD_HEIGHT


def _legal(game, card):
    return {(x, y)
            for y in range(BOARD_H)
            for x in range(BOARD_W)
            if game.is_valid_placement(card, float(x), float(y), 0)}


@pytest.fixture
def game():
    g = E.ClashRoyaleEnv(DECK, DECK, 20000)
    g.reset()
    return g


@pytest.mark.parametrize("slot", [1, 2])
def test_destroying_an_ENEMY_princess_does_not_change_legality(game, slot):
    """Tripwire: the delta table covers our own towers only. If the engine ever
    extends the deploy zone on taking a tower (the real game does), this fails
    and the table must grow a team-1 half.
    """
    before = _legal(game, DECK[0])
    assert game.destroy_tower(1, slot) is True
    after = _legal(game, DECK[0])
    gained, lost = sorted(after - before), sorted(before - after)
    assert not gained and not lost, (
        f"destroying ENEMY tower slot {slot} changed legality "
        f"(+{len(gained)} / -{len(lost)}). net._build_placement_legality's "
        "delta table tracks team 0 only -- extend it to team 1.")


def test_troops_on_the_board_do_not_change_legality(game):
    """If troops mattered, legality would be per-step and no cached table could
    work.
    """
    before = _legal(game, DECK[0])
    for cid, x, y in ((15, 4, 20), (6, 9, 22), (24, 13, 21),
                      (40, 8, 19), (33, 5, 23), (72, 12, 24)):
        try:
            game.inject(cid, float(x), float(y), 1, -1.0, 0)
        except TypeError:
            game.inject(cid, float(x), float(y), 1)
    for _ in range(5):
        game.step(0, 0, 0)
    assert _legal(game, DECK[0]) == before


def test_the_cached_table_agrees_with_the_engine_cell_for_cell():
    """The cache must match the engine on an intact board, the state the base
    table is built for.
    """
    net = MicroRoyaleNet(num_ability_slots=0)
    table = net._placement_legal
    assert table is not None, "no cached legality table; is the .pyd current?"

    g = E.ClashRoyaleEnv(DECK, DECK, 20000)
    g.reset()
    width = net.board_width
    for card in DECK:
        for cell in range(net.placement_cells):
            y, x = divmod(cell, width)
            assert bool(table[card, cell]) is bool(
                g.is_valid_placement(card, float(x), float(y), 0)), (
                f"cached legality disagrees with the engine at "
                f"card={card} cell=({x},{y})")


def test_the_delta_table_is_populated_for_our_own_towers():
    """An empty delta would make the mask fix a no-op while every intact-board
    test still passed.
    """
    net = MicroRoyaleNet(num_ability_slots=0)
    assert net._placement_freed is not None
    assert int(net._placement_freed.sum()) > 0
    for slot in (1, 2):
        assert net.freed_cells_for(slot, DECK[0]), (
            f"no cells recorded as freed by our own tower slot {slot}")
