"""The two damage-spell reward terms must follow the DECK, not card id 7.

The 2026-09-15 audit's finding (`TODO.md` item 00.3) was that `W_LETHAL_SPELL`
and `W_SPELL_VALUE_START` are keyed to Fireball, so a deck without it trains
under a silently smaller objective -- no exception, no log line. Measured on
2026-09-16 over 12 seeded matches through the trainer's own reward path: a Hog
deck carrying Rocket instead of Fireball had both terms at exactly zero on every
one of 2,372 steps.

What is pinned here, in the order that matters:

1. the resolver picks the deck's own damage spell, measured by injection, and
   ranks it by what it does to a CROWN TOWER;
2. the probe underneath reads a spell's WHOLE effect (it cut damage-over-time
   spells off halfway: Poison read 368 of its 736);
3. the shaping READS the spell's damage and cost out of the stats dict, so a
   Rocket deck gets a Rocket-sized lethal window -- and a missing key raises
   rather than falling back to Fireball;
4. a deck with NO damage spell contributes exactly zero -- not a nan, not a
   division by zero, and not a Fireball-shaped phantom;
5. the 2.6 Hog Cycle is UNCHANGED across the change, which is the control that
   stops this being a silent retune of a deck that already worked.

(5) is the load-bearing one. Everything else here could pass while the change
quietly moved the reward for the deck every measured baseline was earned on.
It is pinned here by the resolved constants; the end-to-end version -- identical
seeded matches through extract_engine_stats -> compute_shaping on the old and
the new code, compared bit for bit -- was run when this landed and is recorded
in the commit.

Card ids are the ENGINE'S, checked against get_card_info below rather than
trusted: the first draft of this file had Rocket as 32 (Poison), Zap as 12
(Skeleton Army) and Arrows as 8 (Barbarians).
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
#: Same deck with the Fireball swapped for a Rocket -- a bigger, dearer spell,
#: so every quantity this term derives moves in a direction a test can see.
ROCKET_26 = HOG_26[:-1] + (ROCKET,)
#: Log is a ROLLER: `spell_effect` declines it (its value is a corridor, not a
#: disc), so this deck has no damage spell at all.
NO_SPELL = HOG_26[:-1] + (KNIGHT,)


def test_the_ids_this_file_uses_are_the_cards_it_says():
    names = {FIREBALL: "Fireball", ROCKET: "Rocket", ZAP: "Zap",
             POISON: "Poison", GOBLIN_CURSE: "Goblin Curse", LOG: "The Log",
             KNIGHT: "Knight"}
    for cid, name in names.items():
        assert E.get_card_info(cid)["name"] == name, (cid, name)


# --- 1. the resolver --------------------------------------------------------

def test_the_resolver_names_the_decks_own_spell_not_card_7():
    assert card_probes.damage_spell(HOG_26)[0] == FIREBALL
    assert card_probes.damage_spell(ROCKET_26)[0] == ROCKET
    assert card_probes.damage_spell(NO_SPELL) is None


def test_the_resolver_prefers_the_bigger_spell_when_a_deck_carries_two():
    """A deck with Zap AND Rocket must key the LETHAL term to the Rocket.

    Both are damaging spells; only one can finish a tower. Picking by cost or by
    hand order (Zap sits first here) would pick the Zap, so this is not a
    tautology.
    """
    two = (HOG, MUSKETEER, CANNON, ICE_GOLEM, ZAP, SKELETONS, LOG, ROCKET)
    assert card_probes.damage_spell(two)[0] == ROCKET


def test_the_resolver_reports_the_measured_tower_damage_and_the_real_cost():
    cid, damage, cost = card_probes.damage_spell(ROCKET_26)
    assert cost == pytest.approx(float(E.get_card_info(cid)["cost"]))
    # The CROWN TOWER number, which is what "tower hp <= damage" is about. It
    # equals the troop number in today's engine; see spell_tower_damage.
    assert damage == pytest.approx(card_probes.spell_tower_damage(ROCKET))
    assert damage > card_probes.spell_tower_damage(FIREBALL)


def test_the_tower_probe_reads_what_the_tower_actually_loses():
    """Independent of the probe's own plumbing: cast the spell on the tower by
    hand, run well past any effect, and read the tower. A probe that measured
    the wrong tower, the wrong team or a truncated window disagrees here."""
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


# --- 2. the probe reads the WHOLE effect ------------------------------------

def test_a_damage_over_time_spell_is_read_to_completion():
    """Poison is `spell(32, ..., 92, ...).withRepeats(8, 10)` in CardRegistry.h
    -- 92 every 10 ticks for 8 pulses = 736. A fixed 40-tick window read 368,
    and this module's docstring quoted the 368 as proof the probe reproduced
    the registry. Goblin Curse (43 x 6 = 258) is the second, independent
    control: it read 129."""
    assert card_probes.spell_effect(POISON)[1] == pytest.approx(92 * 8)
    assert card_probes.spell_effect(GOBLIN_CURSE)[1] == pytest.approx(43 * 6)


def test_reading_longer_adds_nothing_for_any_damaging_spell():
    """The general form of the check above, over every registered spell the
    probe accepts: whatever the settle rule decided, running the same cast for
    the target's whole hold window must not find more damage. A settle rule that
    stopped early fails here for the spell it stopped on."""
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


# --- 3. the shaping reads the deck's spell ----------------------------------

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
    """5 elixir casts a Fireball and does not cast a Rocket."""
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
    """6 elixir killed is a WIN for a 4-cost spell and break-even for a 6-cost
    one. This is the whole point of the term -- it is a ratio centred on
    break-even -- and a hardcoded 4.0 denominator paid a Rocket +0.5 for an even
    trade."""
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
    """A winning Rocket trade made with 5 elixir left is NOT solvent (the reserve
    is one more Rocket, 6), where the retired constant 4.0 would have paid it."""
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
    """The first draft read `stats.get("spell_damage", FIREBALL_DAMAGE)`. A
    silent Fireball default is the defect this file exists to remove, arriving
    by a different road; `extract_engine_stats` always supplies the keys, so a
    caller that lacks them is a bug to hear about."""
    cur, prev = shaping_stats(8.0)
    del cur[key]
    with pytest.raises(KeyError):
        shaping.compute_shaping(cur, prev, gamma=0.99)


# --- 4. no damage spell -> exactly zero ---------------------------------------

def test_a_deck_with_no_damage_spell_contributes_exactly_zero_not_a_nan():
    """The control that must fire: zero cost must not divide, and zero damage
    must not make every tower lethal."""
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


# --- 5. the 2.6 control ---------------------------------------------------------

def test_the_26_deck_resolves_to_exactly_the_constants_it_used_to_hardcode():
    """THE CONTROL. The 2.6 deck's reward must not move: its resolved spell must
    be card 7 at exactly the damage and cost the terms used to read from
    constants, and its cost must equal the retired SPELL_SOLVENCY_RESERVE."""
    cid, damage, cost = card_probes.damage_spell(HOG_26)
    assert cid == W.FIREBALL_CARD_ID
    assert damage == W.FIREBALL_DAMAGE == 689.0
    assert cost == W.FIREBALL_COST == 4.0   # the retired SPELL_SOLVENCY_RESERVE


# --- the envs publish it ----------------------------------------------------------

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
    """Measured through the real env, because the info dict is where the two
    halves of this meet and a unit test on either half alone cannot see it.
    Both pipelines, because a key only one supplies is a term that silently
    contributes zero for a whole phase."""
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
