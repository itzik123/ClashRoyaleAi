"""The two damage-spell reward terms follow the deck's own spell, not card id 7.
Pinned, in order of importance:

1. the resolver picks the deck's damage spell, measured by injection and ranked
   by Crown Tower damage;
2. the probe reads a spell's whole effect, including damage over time;
3. the shaping reads the spell's damage and cost from the stats dict, and a
   missing key raises rather than falling back to Fireball;
4. a deck with no damage spell contributes exactly zero (no NaN, no division by
   zero, no Fireball-shaped phantom);
5. the 2.6 deck is unchanged: the control that stops this being a silent retune
   of the deck the baselines were earned on.

Card ids are checked against get_card_info rather than trusted.
"""
import numpy as np
import pytest

import python_ai  # noqa: F401  -- puts the .pyd on sys.path
import clash_royale_env as E

from python_ai.advisors import card_probes
from python_ai.rewards import shaping, weights as W
from python_ai.tests.helpers import shaping_stats

FIREBALL, ROCKET, ZAP, POISON, GOBLIN_CURSE = 7, 30, 29, 32, 102
HOG, MUSKETEER, CANNON, ICE_GOLEM = 15, 6, 25, 40
SKELETONS, ICE_SPIRIT, LOG, KNIGHT = 24, 72, 33, 0

HOG_26 = (HOG, MUSKETEER, CANNON, ICE_GOLEM, SKELETONS, ICE_SPIRIT, LOG, FIREBALL)
#: The same deck with a Rocket for the Fireball: a bigger, dearer spell, so
#: every derived quantity moves visibly.
ROCKET_26 = HOG_26[:-1] + (ROCKET,)
#: The Log is a roller that `spell_effect` declines (a corridor, not a disc),
#: so this deck has no damage spell.
NO_SPELL = HOG_26[:-1] + (KNIGHT,)


def test_the_ids_this_file_uses_are_the_cards_it_says():
    names = {FIREBALL: "Fireball", ROCKET: "Rocket", ZAP: "Zap",
             POISON: "Poison", GOBLIN_CURSE: "Goblin Curse", LOG: "The Log",
             KNIGHT: "Knight"}
    for cid, name in names.items():
        assert E.get_card_info(cid)["name"] == name, (cid, name)


# --- 1. the resolver ---

def test_the_resolver_names_the_decks_own_spell_not_card_7():
    assert card_probes.damage_spell(HOG_26)[0] == FIREBALL
    assert card_probes.damage_spell(ROCKET_26)[0] == ROCKET
    assert card_probes.damage_spell(NO_SPELL) is None


def test_the_resolver_prefers_the_bigger_spell_when_a_deck_carries_two():
    """With Zap and Rocket the lethal term keys to the Rocket; picking by cost or
    hand order would pick the Zap.
    """
    two = (HOG, MUSKETEER, CANNON, ICE_GOLEM, ZAP, SKELETONS, LOG, ROCKET)
    assert card_probes.damage_spell(two)[0] == ROCKET


def test_the_resolver_reports_the_measured_tower_damage_and_the_real_cost():
    cid, damage, cost = card_probes.damage_spell(ROCKET_26)
    assert cost == pytest.approx(float(E.get_card_info(cid)["cost"]))
    # The Crown Tower number, which is what "tower hp <= damage" is about; see
    # spell_tower_damage.
    assert damage == pytest.approx(card_probes.spell_tower_damage(ROCKET))
    assert damage > card_probes.spell_tower_damage(FIREBALL)


def test_the_tower_probe_reads_what_the_tower_actually_loses():
    """Independent of the probe's plumbing: cast the spell on the tower by hand,
    run past any effect, read the tower.
    """
    import python_ai.engine_constants as EC
    for cid in (FIREBALL, ROCKET, POISON):
        e = E.ClashRoyaleEnv(list(HOG_26), list(HOG_26), 3600)
        e.seed(1)
        before = e.get_tower_hp(1, 1)
        e.inject(cid, float(EC.LEFT_LANE_X), float(EC.princess_y(1)), 0, -1.0, 0)
        for _ in range(29):
            e.step_self_play(4, 0.0, 0.0, 4, 0.0, 0.0, 10)
        assert card_probes.spell_tower_damage(cid) == pytest.approx(
            before - e.get_tower_hp(1, 1)), E.get_card_info(cid)["name"]


# --- 2. the probe reads the whole effect ---

def test_a_damage_over_time_spell_is_read_to_completion():
    """Poison is 92 every 10 ticks for 8 pulses = 736 (CardRegistry.h); Goblin
    Curse (43 x 6 = 258) is a second, independent control.
    """
    assert card_probes.spell_effect(POISON)[1] == pytest.approx(92 * 8)
    assert card_probes.spell_effect(GOBLIN_CURSE)[1] == pytest.approx(43 * 6)


def test_reading_longer_adds_nothing_for_any_damaging_spell():
    """General form over every spell the probe accepts: reading for the target's
    whole hold window must find no more damage.
    """
    for cid in sorted(E.get_all_card_ids()):
        eff = card_probes.spell_effect(cid)
        if eff is None:
            continue
        e = card_probes._env()
        e.inject(card_probes._TANK_ID, card_probes._CX, card_probes._CY, 1,
                 -1.0, card_probes._HOLD_TICKS)
        e.step_self_play(4, 0.0, 0.0, 4, 0.0, 0.0, 1)
        before = e.get_troop_damage_dealt(0)
        e.inject(cid, card_probes._CX, card_probes._CY, 0, -1.0, 0)
        card_probes._idle(e, card_probes._HOLD_TICKS - 10)
        assert eff[1] == pytest.approx(e.get_troop_damage_dealt(0) - before), \
            E.get_card_info(cid)["name"]


# --- 3. the shaping reads the deck's spell ---

def _lethal_stats(tower_hp, *, damage, cost, elixir=10.0):
    cur, _ = shaping_stats(0.0)
    cur["enemy_tower_hp"] = np.array([[tower_hp, 0.0, 0.0]], dtype=np.float32)
    cur["spell_in_hand"] = np.array([1.0], dtype=np.float32)
    cur["team0_elixir_current"] = np.array([elixir], dtype=np.float32)
    cur["spell_damage"] = np.array([damage], dtype=np.float32)
    cur["spell_cost"] = np.array([cost], dtype=np.float32)
    return cur


def test_the_lethal_window_is_the_decks_spell_damage():
    """A tower on 900 HP is lethal to a Rocket and not to a Fireball."""
    _, fb_dmg, fb_cost = card_probes.damage_spell(HOG_26)
    _, rk_dmg, rk_cost = card_probes.damage_spell(ROCKET_26)
    assert fb_dmg < 900.0 < rk_dmg, "fixture no longer straddles the two spells"

    fireball = shaping.lethal_spell_potential(
        _lethal_stats(900.0, damage=fb_dmg, cost=fb_cost))
    rocket = shaping.lethal_spell_potential(
        _lethal_stats(900.0, damage=rk_dmg, cost=rk_cost))
    assert float(fireball[0]) == 0.0
    assert float(rocket[0]) == pytest.approx(W.W_LETHAL_SPELL)


def test_the_castability_gate_is_the_decks_spell_cost():
    """5 elixir casts a Fireball and not a Rocket."""
    _, fb_dmg, fb_cost = card_probes.damage_spell(HOG_26)
    _, rk_dmg, rk_cost = card_probes.damage_spell(ROCKET_26)
    assert fb_cost <= 5.0 < rk_cost, "fixture no longer straddles the two costs"
    broke = shaping.lethal_spell_potential(
        _lethal_stats(100.0, damage=rk_dmg, cost=rk_cost, elixir=5.0))
    rich = shaping.lethal_spell_potential(
        _lethal_stats(100.0, damage=fb_dmg, cost=fb_cost, elixir=5.0))
    assert float(broke[0]) == 0.0
    assert float(rich[0]) == pytest.approx(W.W_LETHAL_SPELL)


def test_the_trade_ratio_is_priced_in_the_decks_own_spell_cost():
    """6 elixir killed is a win for a 4-cost spell and break-even for a 6-cost
    one; the term is a ratio centred on break-even.
    """
    def traded(cost):
        cur, prev = shaping_stats(0.0)
        for d in (cur, prev):
            d["spell_cost"] = np.array([cost], dtype=np.float32)
        cur["spell_value_killed"] = np.array([6.0], dtype=np.float32)
        cur["spell_elixir_spent"] = np.array([cost], dtype=np.float32)
        cur["team0_elixir_current"] = np.array([10.0], dtype=np.float32)
        prev["spell_value_killed"] = np.zeros(1, dtype=np.float32)
        prev["spell_elixir_spent"] = np.zeros(1, dtype=np.float32)
        return float(shaping.spell_value_shaping(cur, prev, 1.0)[0])

    assert traded(4.0) == pytest.approx(6.0 / 4.0 - 1.0)   # +0.50
    assert traded(6.0) == pytest.approx(6.0 / 6.0 - 1.0)   # 0.00, break-even
    assert traded(8.0) < 0.0                                # a losing trade


def test_the_solvency_reserve_is_the_spells_own_cost():
    """A winning Rocket trade with 5 elixir left is not solvent: the reserve is
    one more Rocket, 6.
    """
    cur, prev = shaping_stats(0.0)
    for d in (cur, prev):
        d["spell_cost"] = np.array([6.0], dtype=np.float32)
    cur["spell_value_killed"] = np.array([12.0], dtype=np.float32)
    cur["spell_elixir_spent"] = np.array([6.0], dtype=np.float32)
    cur["team0_elixir_current"] = np.array([5.0], dtype=np.float32)
    assert float(shaping.spell_value_shaping(cur, prev, 1.0)[0]) == 0.0
    cur["team0_elixir_current"] = np.array([6.0], dtype=np.float32)
    assert float(shaping.spell_value_shaping(cur, prev, 1.0)[0]) == pytest.approx(1.0)


@pytest.mark.parametrize("key", ["spell_damage", "spell_cost"])
def test_a_missing_spell_key_raises_rather_than_falling_back_to_fireball(key):
    """A silent Fireball default would be the removed defect by another road;
    `extract_engine_stats` always supplies the keys, so a missing one is a bug
    to hear about.
    """
    cur, prev = shaping_stats(8.0)
    del cur[key]
    with pytest.raises(KeyError):
        shaping.compute_shaping(cur, prev, gamma=0.99)


# --- 4. no damage spell: exactly zero ---

def test_a_deck_with_no_damage_spell_contributes_exactly_zero_not_a_nan():
    """Control that must fire: zero cost must not divide, and zero damage must not
    make every tower lethal.
    """
    cur, prev = shaping_stats(0.0)
    for d in (cur, prev):
        d["spell_damage"] = np.zeros(1, dtype=np.float32)
        d["spell_cost"] = np.zeros(1, dtype=np.float32)
        d["spell_in_hand"] = np.ones(1, dtype=np.float32)
    cur["enemy_tower_hp"] = np.array([[2534.0, 2534.0, 4008.0]], dtype=np.float32)
    cur["spell_value_killed"] = np.array([5.0], dtype=np.float32)

    lethal = shaping.lethal_spell_potential(cur)
    value = shaping.spell_value_shaping(cur, prev, 1.0)
    for name, arr in (("lethal", lethal), ("value", value)):
        assert np.isfinite(arr).all(), f"{name} produced a non-finite reward"
        assert float(arr[0]) == 0.0, f"{name} fabricated a reward with no spell"


# --- 5. the 2.6 control ---

def test_the_26_deck_resolves_to_exactly_the_constants_it_used_to_hardcode():
    """The control: the 2.6 deck resolves to card 7 at exactly the damage and cost
    the terms used to read from constants.
    """
    cid, damage, cost = card_probes.damage_spell(HOG_26)
    assert cid == W.FIREBALL_CARD_ID
    assert damage == W.FIREBALL_DAMAGE == 689.0
    assert cost == W.FIREBALL_COST == 4.0   # the retired SPELL_SOLVENCY_RESERVE


# --- the envs publish it ---

_SPELL_KEYS = ("spell_in_hand", "spell_value_killed", "spell_elixir_spent",
               "spell_damage", "spell_cost")


def _gym_info(deck):
    from python_ai.envs.gym_wrapper import MicroRoyaleEnv
    env = MicroRoyaleEnv({"ai_deck": list(deck), "opp_deck": list(deck)})
    try:
        env.reset(seed=3)
        _, _, _, _, info = env.step(
            {"card_index": 4, "target_x": 0.0, "target_y": 0.0})
    finally:
        env.close()
    return info


def _selfplay_info(deck):
    from python_ai.envs.selfplay_env import MicroRoyaleSelfPlayEnv
    env = MicroRoyaleSelfPlayEnv({"deck": list(deck), "scenarios_enabled": False,
                                  "scenario_seed": 3})
    try:
        env.reset(seed=3)
        _, _, _, _, info = env.step(
            {"card_index": 4, "target_x": 0.0, "target_y": 0.0})
    finally:
        env.close()
    return info


@pytest.mark.parametrize("info_of", [_gym_info, _selfplay_info],
                         ids=["phase1", "phase2"])
def test_both_envs_publish_the_spell_keys_for_their_own_deck(info_of):
    """Measured through the real env, where both halves meet; both pipelines,
    since a key only one supplies silences the term for a whole phase.
    """
    info = info_of(ROCKET_26)
    for key in _SPELL_KEYS:
        assert key in info, f"env published no {key}"
    assert info["spell_damage"] == pytest.approx(card_probes.spell_tower_damage(ROCKET))
    assert info["spell_cost"] == pytest.approx(float(E.get_card_info(ROCKET)["cost"]))


@pytest.mark.parametrize("info_of", [_gym_info, _selfplay_info],
                         ids=["phase1", "phase2"])
def test_an_env_with_no_damage_spell_publishes_zeros(info_of):
    info = info_of(NO_SPELL)
    for key in _SPELL_KEYS:
        assert info[key] == 0.0, key
