"""The placement mask tracks the board after the agent loses a tower.

When our own Princess dies its 3x3 footprint becomes legal (an enemy tower
dying changes nothing). A table cached on a full board would forbid those cells
forever, exactly the ground defended after losing a tower.
"""
import numpy as np
import pytest
import torch

import clash_royale_env as E
from python_ai.models.net import MicroRoyaleNet

DECK = [10, 1, 41, 25, 7, 2, 6, 5]


@pytest.fixture(scope="module")
def net():
    return MicroRoyaleNet(num_ability_slots=0)


def _game():
    g = E.ClashRoyaleEnv(DECK, DECK, 20000)
    g.reset()
    return g


def _obs(g, net):
    return torch.tensor(
        np.asarray(g.get_observation_for_team(0), dtype=np.float32)).unsqueeze(0)


def _force_hand(g, cards):
    """Put `cards` in team 0's hand so a specific card can be masked for; the
    opening hand is a random subset of the deck.
    """
    ok = g.set_hand_for_team(0, list(cards))
    if not ok:
        pytest.skip(f"engine refused the hand {cards}")
    return ok


def _mask_cells_for_slot(net, obs, slot):
    """Cells the mask permits for the card in hand slot `slot`."""
    idx = torch.tensor([slot])
    m = net.placement_mask(obs, idx)[0]
    return {(c % net.board_width, c // net.board_width)
            for c in range(net.placement_cells) if bool(m[c])}


def _assert_mask_matches_engine(net, g, label):
    """Compare mask against engine for every occupied hand slot."""
    obs = _obs(g, net)
    hand = net.hand_card_ids(obs)[0].tolist()
    checked = 0
    for slot, card in enumerate(hand):
        if card < 0:
            continue
        checked += 1
        got = _mask_cells_for_slot(net, obs, slot)
        want = _engine_cells(g, card, net)
        missing, extra = want - got, got - want
        assert not missing, (
            f"[{label}] card {card} (slot {slot}): mask FORBIDS "
            f"{len(missing)} cells the engine accepts: {sorted(missing)}")
        assert not extra, (
            f"[{label}] card {card} (slot {slot}): mask PERMITS "
            f"{len(extra)} cells the engine rejects: {sorted(extra)} -- the "
            "worse direction, it injects gradient noise")
    assert checked, "no card was actually checked"


def _engine_cells(g, card_id, net):
    return {(x, y)
            for y in range(net.placement_rows)
            for x in range(net.board_width)
            if g.is_valid_placement(card_id, float(x), float(y), 0)}


@pytest.mark.parametrize("slot,expect_centre", [(1, (3, 6)), (2, (14, 6))])
def test_the_engine_frees_the_footprint_of_our_own_dead_princess(slot,
                                                                 expect_centre,
                                                                 net):
    """Pins the engine behaviour the fix tracks, independently of our
    implementation.
    """
    g = _game()
    before = _engine_cells(g, DECK[0], net)
    assert g.destroy_tower(0, slot) is True
    after = _engine_cells(g, DECK[0], net)

    gained = after - before
    assert len(gained) == 9, f"expected a 3x3 footprint, got {sorted(gained)}"
    assert not (before - after), "destroying a tower should never REMOVE a cell"
    cx, cy = expect_centre
    assert gained == {(cx + dx, cy + dy)
                      for dx in (-1, 0, 1) for dy in (-1, 0, 1)}


def test_an_enemy_tower_dying_changes_nothing(net):
    """The asymmetry is why only our own towers are tracked."""
    g = _game()
    before = _engine_cells(g, DECK[0], net)
    g.destroy_tower(1, 1)
    assert _engine_cells(g, DECK[0], net) == before


@pytest.mark.parametrize("slot", [1, 2])
def test_the_MASK_follows_the_engine_after_our_tower_falls(slot, net):
    """The regression test: mask and engine agree in the destroyed state."""
    g = _game()
    g.destroy_tower(0, slot)
    for _ in range(2):
        g.step(0, 0, 0)

    _assert_mask_matches_engine(net, g, f"own princess slot {slot} destroyed")


def test_the_mask_still_matches_on_a_FULL_board(net):
    """Nothing loosens while the towers stand."""
    g = _game()
    _assert_mask_matches_engine(net, g, "intact board")
    # ...and with the other half of the deck in hand, so every card is
    # exercised.
    _force_hand(g, DECK[4:])
    _assert_mask_matches_engine(net, g, "intact board, second half of deck")


def test_both_princesses_down_frees_both_footprints(net):
    g = _game()
    g.destroy_tower(0, 1)
    g.destroy_tower(0, 2)
    for _ in range(2):
        g.step(0, 0, 0)
    _assert_mask_matches_engine(net, g, "both princesses down")


def test_every_deck_card_agrees_in_the_destroyed_state(net):
    """Per card: a clear footprint still has to satisfy each card's own placement
    rules.
    """
    g = _game()
    g.destroy_tower(0, 1)
    for _ in range(2):
        g.step(0, 0, 0)
    _assert_mask_matches_engine(net, g, "left princess down, dealt hand")
    _force_hand(g, DECK[4:])
    _assert_mask_matches_engine(net, g, "left princess down, second half")


@pytest.mark.slow
def test_the_probe_WINDOW_is_wide_enough_for_every_registered_card(net):
    """The freed-cell table probes only a window around each tower; this
    exhaustive comparison over every registered card catches an undersized
    window.
    """
    ids = [c for c in E.get_all_card_ids()]
    for slot in (1, 2):
        g = _game()
        base = {c: _engine_cells(g, c, net) for c in ids}
        g.destroy_tower(0, slot)
        for c in ids:
            gained = _engine_cells(g, c, net) - base[c]
            if not gained:
                continue
            recorded = net.freed_cells_for(slot, c)
            assert gained <= recorded, (
                f"card {c} gains {sorted(gained - recorded)} outside the "
                f"recorded window for tower slot {slot}; widen it")
