"""The 2x2 that separates 'the game refused it' from 'the ledger lost it'.

These run against synthetic GameStates rather than an emulator, because the
whole value of the cross-tab is that its two axes are independent -- and that is
a property of the code, testable offline, not of any particular match.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from contracts import UNKNOWN_CARD_SIM_ID
from live.elixir_ledger import PLAY_CONFIRM_WINDOW_S, ElixirLedger
from live.placement_confirm import PlacementConfirmer, expected_unit_names


@dataclass(frozen=True)
class FakeUnit:
    unit_name: str
    team: int


@dataclass(frozen=True)
class FakeState:
    units: tuple = ()
    my_hand: tuple = (2, 1, 41, 25)
    my_elixir: float = 10.0


def place(conf, gs, *, slot=0, card="minions", tile=(9, 8), now=0.0):
    return conf.issue(gs, slot=slot, card_name=card, card_sim_id=41,
                      tile=tile, tap=(360, 900), now=now)


def test_expected_units_comes_from_crbabs_own_table():
    assert expected_unit_names("minions") == frozenset({"minion"})
    assert expected_unit_names("giant") == frozenset({"giant"})
    # A spell spawns no board presence at all, which is why it must be excluded
    # from the unit axis rather than scored as a failure.
    assert expected_unit_names("fireball") == frozenset()


def test_unit_appearing_confirms_the_placement():
    conf = PlacementConfirmer()
    empty = FakeState()
    place(conf, empty, now=0.0)
    conf.observe(FakeState(units=(FakeUnit("minion", 0),)), now=1.0)
    assert conf.issued[0].unit_confirmed
    assert conf.issued[0].best_delta == 1


def test_only_our_own_units_count():
    """An enemy Minions played at the same moment must not confirm ours."""
    conf = PlacementConfirmer()
    place(conf, FakeState(), now=0.0)
    conf.observe(FakeState(units=(FakeUnit("minion", 1),)), now=1.0)
    assert not conf.issued[0].unit_confirmed


def test_a_unit_already_on_the_board_does_not_confirm():
    """The baseline is taken at issue time, so standing units are not evidence.

    Without this, playing a second Minions while the first three are still
    alive would confirm itself instantly.
    """
    conf = PlacementConfirmer()
    standing = FakeState(units=(FakeUnit("minion", 0),) * 3)
    place(conf, standing, now=0.0)
    conf.observe(standing, now=1.0)
    assert not conf.issued[0].unit_confirmed
    conf.observe(FakeState(units=(FakeUnit("minion", 0),) * 6), now=1.5)
    assert conf.issued[0].unit_confirmed
    assert conf.issued[0].best_delta == 3


def test_evidence_after_the_window_is_ignored():
    conf = PlacementConfirmer(window_s=2.0)
    place(conf, FakeState(), now=0.0)
    conf.observe(FakeState(units=(FakeUnit("minion", 0),)), now=5.0)
    assert not conf.issued[0].unit_confirmed
    assert conf.issued[0].observations == 0


def test_hand_cycling_confirms_independently_of_units():
    """The oracle that covers spells, where the unit axis is blind."""
    conf = PlacementConfirmer()
    conf.issue(FakeState(my_hand=(2, 1, 41, 25)), slot=1, card_name="fireball",
               card_sim_id=7, tile=(9, 8), tap=(360, 900), now=0.0)
    conf.observe(FakeState(my_hand=(2, 6, 41, 25)), now=1.0)
    assert conf.issued[0].hand_changed
    assert conf.issued[0].is_spell


def test_an_unreadable_slot_is_not_a_hand_change():
    """A card reader going blind must not be read as a card being played.

    This is the failure the ledger already has -- inferring an event from a
    reading that merely got worse -- and repeating it here would confirm
    placements from detector noise.
    """
    conf = PlacementConfirmer()
    conf.issue(FakeState(my_hand=(2, 1, 41, 25)), slot=1, card_name="archers",
               card_sim_id=1, tile=(9, 8), tap=(360, 900), now=0.0)
    conf.observe(FakeState(my_hand=(2, UNKNOWN_CARD_SIM_ID, 41, 25)), now=1.0)
    assert not conf.issued[0].hand_changed


def test_refused_requires_both_oracles_silent():
    conf = PlacementConfirmer()
    place(conf, FakeState(), now=0.0)          # 0: nothing happens -> refused
    place(conf, FakeState(), slot=1, now=0.0)  # 1: hand cycles -> played
    conf.observe(FakeState(my_hand=(2, 6, 41, 25)), now=1.0)
    refused = conf.refused()
    assert [r.seq for r in refused] == [0]


def test_two_placements_of_one_card_do_not_share_the_same_body():
    """One Minion appearing is evidence for ONE placement, not for both.

    Sharing it would bias the whole measurement towards "it landed", which is
    the direction that hides refusals.
    """
    conf = PlacementConfirmer()
    first = place(conf, FakeState(), now=0.0)
    second = place(conf, FakeState(), slot=1, now=0.1)
    conf.observe(FakeState(units=(FakeUnit("minion", 0),)), now=1.0)
    assert first.unit_confirmed
    assert not second.unit_confirmed
    # A second body then confirms the second placement too.
    conf.observe(FakeState(units=(FakeUnit("minion", 0),) * 2), now=1.5)
    assert second.unit_confirmed


def test_cross_tab_separates_the_ledgers_failure_from_the_games():
    """The whole point: a placement the ledger wrote off but a unit confirms is
    a METRIC bug, not a refused placement."""
    conf = PlacementConfirmer()
    landed = place(conf, FakeState(), card="minions", now=0.0)
    refused = place(conf, FakeState(), slot=1, card="giant", now=0.0)
    conf.observe(FakeState(units=(FakeUnit("minion", 0),)), now=1.0)
    conf.apply_ledger(confirmed_tags=[], rejected_tags=[landed.seq, refused.seq])
    tab = conf.cross_tab()
    assert tab["unit_yes_ledger_no"] == 1     # the ledger lost a real play
    assert tab["unit_no_ledger_no"] == 1      # the game really refused this one
    assert tab["unit_yes_ledger_yes"] == 0


def test_spells_are_excluded_from_the_cross_tab():
    """Counting a Fireball as 'no unit appeared' would manufacture a refusal
    every time the agent played its only spell."""
    conf = PlacementConfirmer()
    rec = conf.issue(FakeState(), slot=0, card_name="fireball", card_sim_id=7,
                     tile=(9, 8), tap=(360, 900), now=0.0)
    conf.apply_ledger(confirmed_tags=[], rejected_tags=[rec.seq])
    assert sum(conf.cross_tab().values()) == 0
    assert conf.refused() == []


# -- the ledger side of the join ------------------------------------------


def test_ledger_reports_which_plays_it_confirmed_not_just_how_many():
    led = ElixirLedger(costs=(3.0, 4.0, 5.0))
    for t, v in ((0.0, 10), (0.1, 10), (0.2, 10)):
        led.update(v, now=t)
    led.record_play(4.0, now=0.2, tag="A")
    for t, v in ((0.3, 6), (0.4, 6), (0.5, 6)):
        led.update(v, now=t)
    assert led.confirmed_tags == ["A"]
    assert led.rejected_tags == []


def test_ledger_reports_which_plays_it_wrote_off():
    led = ElixirLedger(costs=(3.0, 4.0, 5.0))
    for t, v in ((0.0, 10), (0.1, 10), (0.2, 10)):
        led.update(v, now=t)
    led.record_play(4.0, now=0.2, tag="A")
    late = 0.2 + PLAY_CONFIRM_WINDOW_S + 1.0
    for t in (late, late + 0.1, late + 0.2):
        led.update(10, now=t)
    assert led.rejected_tags == ["A"]
    assert led.confirmed_tags == []


def test_untagged_plays_still_work():
    """Backwards compatibility: every existing caller passes no tag."""
    led = ElixirLedger(costs=(3.0, 4.0, 5.0))
    for t, v in ((0.0, 10), (0.1, 10), (0.2, 10)):
        led.update(v, now=t)
    led.record_play(4.0, now=0.2)
    for t, v in ((0.3, 6), (0.4, 6), (0.5, 6)):
        led.update(v, now=t)
    assert led.confirmed_tags == [None]
    assert led.cards == 1
