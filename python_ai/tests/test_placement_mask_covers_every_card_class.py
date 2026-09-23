"""The placement mask agrees with the engine for every class of card a deck can
hold.

* Deploy-anywhere troops (Miner 52, Goblin Drill 98 / 153): the table allows
  the enemy half, so no row rule may delete it.
* Evolutions (ids 123-163): `get_all_card_ids()` filters them out, so the
  legality table and the spell flags must include them explicitly.

Not circular: besides comparing mask and predicate, every test plays a real
placement through `step_self_play` and checks the engine took the elixir.
"""
import numpy as np
import pytest
import torch

import clash_royale_env as E
from python_ai.models.net import MicroRoyaleNet

CE = E.ClashRoyaleEnv
BOARD_W, BOARD_H = CE.BOARD_WIDTH, CE.BOARD_HEIGHT
NOOP = CE.HAND_SIZE
#: Eight ordinary cards, so a deck built around one of them still has eight.
FILLER = [6, 25, 40, 24, 72, 33, 7, 0]


@pytest.fixture(scope="module")
def net():
    return MicroRoyaleNet(num_ability_slots=0)


def _deck_with(card, index):
    """A legal 8-card deck holding `card` at deck `index`."""
    rest = [c for c in FILLER if c != card]
    deck = rest[:index] + [card] + rest[index:]
    deck = deck[:8]
    assert E.validate_deck_slots(deck) == "", E.validate_deck_slots(deck)
    return deck


def _game_holding(card, index):
    deck = _deck_with(card, index)
    g = CE(deck, [15, 6, 25, 40, 24, 72, 33, 7], 3600)
    g.seed(7)
    assert g.set_hand_for_team(0, [card] + [c for c in deck if c != card][:3])
    return g


def _mask_for_slot0(net, game):
    obs = torch.tensor(np.asarray(game.get_observation_for_team(0), np.float32))
    assert int(net.hand_card_ids(obs.unsqueeze(0))[0, 0]) == game.get_hand()[0]
    mask = net.placement_mask(obs.unsqueeze(0), torch.tensor([0]))[0]
    return mask.view(BOARD_H, BOARD_W).numpy()


def _engine_accepts(card, index, x, y):
    """Ground truth: play the card and see whether the engine took the elixir.
    """
    g = _game_holding(card, index)
    g.set_elixir_for_team(0, 10.0)
    before = g.get_elixir_spent_on_card(card, 0)
    g.step_self_play(0, float(x), float(y), NOOP, 0.0, 0.0, 1)
    return g.get_elixir_spent_on_card(card, 0) > before


# --- deploy-anywhere troops ---

@pytest.mark.parametrize("card", [52, 98])     # Miner, Goblin Drill
def test_a_deploy_anywhere_troop_keeps_the_enemy_half(net, card):
    g = _game_holding(card, 0)
    mask = _mask_for_slot0(net, g)
    engine = np.array([[g.is_valid_placement(card, float(x), float(y), 0)
                        for x in range(BOARD_W)] for y in range(BOARD_H)])
    lost = np.argwhere(engine & ~mask)
    assert lost.size == 0, (
        f"card {card}: mask forbids {len(lost)} cells the engine allows, "
        f"rows {sorted({int(r) for r, _ in lost})}")
    assert not np.argwhere(mask & ~engine).size


@pytest.mark.parametrize("card", [52, 98])
def test_the_engine_really_accepts_a_deploy_anywhere_troop_deep_in_the_enemy_half(net, card):
    """The non-circular half: a cell the mask allows on the enemy half, played
    through the engine.
    """
    g = _game_holding(card, 0)
    mask = _mask_for_slot0(net, g)
    enemy = [(int(x), int(y)) for y, x in np.argwhere(mask) if y >= 20]
    assert enemy, f"card {card}: mask allows no cell at y >= 20"
    x, y = enemy[len(enemy) // 2]
    assert _engine_accepts(card, 0, x, y), f"engine refused card {card} at {(x, y)}"


def test_an_ordinary_troop_still_cannot_cross_the_river(net):
    """Control: the enemy half stays closed to troops that are not
    deploy-anywhere.
    """
    g = _game_holding(6, 0)                        # Musketeer
    mask = _mask_for_slot0(net, g)
    assert mask[:16].any()
    assert not mask[18:].any(), "a Musketeer was allowed onto the enemy half"


# --- Evolutions ---

EVO_ARCHERS, EVO_ZAP = 128, 124


def test_an_evolution_has_a_real_legality_row(net):
    legal = net._placement_legal[EVO_ARCHERS]
    assert int(legal.sum()) > 0, "Evolution Archers has an all-False legality row"


def test_an_evolution_troop_in_hand_can_be_placed_where_the_engine_plays_it(net):
    g = _game_holding(EVO_ARCHERS, 0)
    mask = _mask_for_slot0(net, g)
    assert mask.sum() > 0, "no legal cell for Evolution Archers"
    y, x = np.argwhere(mask)[len(np.argwhere(mask)) // 2]
    assert _engine_accepts(EVO_ARCHERS, 0, int(x), int(y))


def test_an_evolution_spell_is_flagged_as_a_spell(net):
    """An Evolution spell gets the spell rule (whole board), not a troop's own
    half.
    """
    assert float(net.spell_flags[EVO_ZAP]) == 1.0
    g = _game_holding(EVO_ZAP, 0)
    mask = _mask_for_slot0(net, g)
    assert mask[25:].any(), "Evolution Zap cannot reach the enemy half"
