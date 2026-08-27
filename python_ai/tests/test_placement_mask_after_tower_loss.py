"""The placement mask must track the board after the agent LOSES a tower.

`_build_placement_legality` probes `is_valid_placement` once at construction and
caches it, on the stated premise that legality is board-state-independent. That
premise was measured against TROOPS on the board. It does not hold for a
DESTROYED TOWER:

    own LEFT princess destroyed   242 -> 251 legal cells   (+9, its footprint)
    own RIGHT princess destroyed  242 -> 251 legal cells   (+9)
    enemy princess destroyed      242 -> 242               ( 0, no change)

The tower's own 3x3 footprint is cleared when it dies. The cached table, built
on a full board, keeps masking those nine cells off forever -- so the policy can
never deploy in its own fallen tower's footprint, which is precisely the ground
you defend after losing a tower.

Direction matters for severity: this mask FORBIDS legal cells, it does not
permit illegal ones, so it costs options rather than injecting gradient noise
the way the affordability bug did. But it is a real divergence between the mask
and the engine, in a state most matches reach.

GAMEPLAY-AFFECTING: it widens the action space in a state the agent reaches
often, so win rates from before it are not strictly comparable.
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
    """Put `cards` in team 0's hand, so a specific card can be masked for.

    The opening hand is a shuffled subset of the deck, so `hand.index(card)`
    raises for the half of the deck that happens not to be drawn -- which is
    what a first draft of this file did.
    """
    ok = g.set_hand_for_team(0, list(cards))
    if not ok:
        pytest.skip(f"engine refused the hand {cards}")
    return ok


def _mask_cells_for_slot(net, obs, slot):
    """Cells the MASK permits for the card in hand slot `slot`."""
    idx = torch.tensor([slot])
    m = net.placement_mask(obs, idx)[0]
    return {(c % net.board_width, c // net.board_width)
            for c in range(net.placement_cells) if bool(m[c])}


def _assert_mask_matches_engine(net, g, label):
    """Compare mask against engine for EVERY occupied hand slot."""
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
    """Pins the engine behaviour the fix exists to track, so the test suite
    states the fact independently of our implementation of it."""
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
    """The asymmetry is the reason only our OWN towers are tracked."""
    g = _game()
    before = _engine_cells(g, DECK[0], net)
    g.destroy_tower(1, 1)
    assert _engine_cells(g, DECK[0], net) == before


@pytest.mark.parametrize("slot", [1, 2])
def test_the_MASK_follows_the_engine_after_our_tower_falls(slot, net):
    """THE regression test: mask and engine must agree in the destroyed state."""
    g = _game()
    g.destroy_tower(0, slot)
    for _ in range(2):
        g.step(0, 0, 0)

    _assert_mask_matches_engine(net, g, f"own princess slot {slot} destroyed")


def test_the_mask_still_matches_on_a_FULL_board(net):
    """The change must not loosen anything while the towers stand."""
    g = _game()
    _assert_mask_matches_engine(net, g, "intact board")
    # ...and again with the other half of the deck forced into hand, so every
    # card in DECK is actually exercised rather than whichever four were dealt.
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
    """Per-card, because the freed cells still have to satisfy each card's own
    placement rules -- a footprint being clear does not make it legal for
    everything."""
    g = _game()
    g.destroy_tower(0, 1)
    for _ in range(2):
        g.step(0, 0, 0)
    _assert_mask_matches_engine(net, g, "left princess down, dealt hand")
    _force_hand(g, DECK[4:])
    _assert_mask_matches_engine(net, g, "left princess down, second half")


@pytest.mark.slow
def test_the_probe_WINDOW_is_wide_enough_for_every_registered_card(net):
    """The expensive guard on the cheap implementation.

    The freed-cell table is built by probing only a bounded window around each
    tower, rather than re-probing the whole board for every card in every tower
    state. This test does the full, exhaustive comparison for EVERY registered
    card and every cell, so an under-sized window is caught here rather than
    becoming a silent mask divergence.
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
