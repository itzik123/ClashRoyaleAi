"""The teacher aims each spell with ITS OWN radius and damage, not Fireball's.

`_top_spell_cells`, both spell combos and the rung-0 rules gate called
`tactics.spell_catch_map(obs)` with its Fireball defaults whatever spell was in
hand. Measured on 320 mid-match boards against eight pool decks, engine-scored
by elixir value killed: Rocket +38% (better on 75 boards, worse on 4), Zap +11%,
Poison +7%, Arrows +4%, Lightning level -- and Fireball's cell identical on all
320, which is the control that the 2.6 mirror does not move. The measurement is
the evidence for "better"; these tests pin the MECHANISM, on real engine boards.
"""
import numpy as np
import pytest

import python_ai  # noqa: F401
import clash_royale_env as E

from python_ai.advisors import card_probes, tactics
from python_ai.opponents import teacher as T

FIREBALL, ROCKET, ZAP, POISON, LOG = 7, 30, 29, 32, 33
SKELETONS, MUSKETEER = 24, 6
DECK = [15, 6, 25, 40, 24, 72, 33, 7]


def _board(*units):
    """Team 0's observation of enemy `units` [(card_id, x, y)] on our half."""
    e = E.ClashRoyaleEnv(list(DECK), list(DECK), 3600)
    e.seed(1)
    for cid, x, y in units:
        e.inject(cid, x, y, 1, -1.0, 300)
    e.step_self_play(4, 0.0, 0.0, 4, 0.0, 0.0, 1)
    return np.asarray(e.get_observation_for_team(0), np.float32)


def test_fireball_is_aimed_exactly_as_before():
    """THE CONTROL: the 2.6 mirror's only area spell keeps its old geometry."""
    assert T.spell_geometry(FIREBALL) == (tactics.FIREBALL_RADIUS,
                                          tactics.FIREBALL_DAMAGE)
    obs = _board((SKELETONS, 6.0, 10.0), (MUSKETEER, 12.0, 11.0))
    assert np.array_equal(T.spell_catch_for(obs, FIREBALL),
                          tactics.spell_catch_map(obs))


@pytest.mark.parametrize("cid", [ROCKET, ZAP, POISON])
def test_every_other_area_spell_is_aimed_with_its_measured_geometry(cid):
    assert T.spell_geometry(cid) == card_probes.spell_effect(cid)
    assert T.spell_geometry(cid) != T.spell_geometry(FIREBALL)


def test_a_roller_keeps_the_disc_it_was_aimed_with_before():
    """The Log's value is a corridor; no disc describes it, and the probe
    declines it. It is not in scope here, so it must not move."""
    assert card_probes.spell_effect(LOG) is None
    assert T.spell_geometry(LOG) == T.spell_geometry(FIREBALL)


def test_suppression_uses_the_cards_own_disc():
    """The second candidate must be a different decision under the spell being
    cast: a 3.5-radius Poison suppresses a wider disc than Fireball's 2.5."""
    teacher = T.UtilityTeacher(list(DECK), team=0)
    obs = _board((SKELETONS, 4.0, 9.0), (SKELETONS, 13.0, 9.0),
                 (MUSKETEER, 8.0, 12.0))
    cells = teacher._top_spell_cells(obs, 2, POISON)
    assert len(cells) == 2
    (x0, y0), (x1, y1) = cells
    assert np.hypot(x1 - x0, y1 - y0) > T.spell_geometry(POISON)[0]


def _rules_gate_casts(obs, cid):
    """Drive the REAL rung-0 gate (`_rules_only`) with one spell candidate."""
    teacher = T.UtilityTeacher(list(DECK), team=0)
    cand = T.Candidate.single(2, cid, 8.0, 10.0, role="spell")
    slot, _x, _y = teacher._rules_only(obs, [cand])
    return slot == 2


def test_the_rules_gate_fires_a_zap_on_a_skeleton_clump():
    """243 HP of Skeletons is a full-damage Zap. Fireball's 689 bar held the
    rung-0 teacher's Zap back from exactly this board."""
    obs = _board((SKELETONS, 8.0, 10.0))
    assert float(tactics.spell_catch_map(obs).max()) < tactics.FIREBALL_DAMAGE, \
        "fixture: the OLD gate must have refused this board"
    assert _rules_gate_casts(obs, ZAP)


def test_the_rules_gate_holds_a_rocket_back_from_a_lone_musketeer():
    """A 6-elixir Rocket on one 4-elixir Musketeer is a losing trade. Fireball's
    689 bar let it through; the Rocket's own 1485 does not."""
    obs = _board((MUSKETEER, 8.0, 10.0))
    assert float(tactics.spell_catch_map(obs).max()) >= tactics.FIREBALL_DAMAGE, \
        "fixture: the OLD gate must have cast on this board"
    assert not _rules_gate_casts(obs, ROCKET)


def test_the_rules_gate_is_unchanged_for_fireball_on_both_boards():
    """The control for the two above: Fireball decides exactly as it did."""
    clump = _board((SKELETONS, 8.0, 10.0))
    lone = _board((MUSKETEER, 8.0, 10.0))
    assert not _rules_gate_casts(clump, FIREBALL)
    assert _rules_gate_casts(lone, FIREBALL)
