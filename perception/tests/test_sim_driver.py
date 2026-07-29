"""Bridge correctness, isolated from vision and from fixture noise.

THE CONTROL EXPERIMENT
----------------------
test_zero_error_events_reproduce_truth_exactly builds its own ground truth
rather than using a training replay, and the reason is worth stating because
it was learned the hard way.

A training replay looks like ideal test input -- it has labelled placements
(train.py stamps actionCardId/actionX/actionY on every tick). But it only
labels the LEARNER's plays. The built-in heuristic opponent's plays are
logged nowhere at all, so they have to be reconstructed from entities
appearing on the board, which introduces:

  * ±1 tick, because an entity is visible the tick AFTER it was played;
  * ±0.5 tile, because a multi-unit card's aim point has to be recovered as
    the centroid of its squad.

Feeding those approximations back in produced up to 530 HP/tower of
divergence and, in one run, killed a King Tower ~200 ticks before the real
match ended. That is not a bridge defect -- it is combat being chaotically
sensitive to placement, which is exactly the sensitivity the divergence
metric exists to expose. But it makes a replay useless as a CONTROL, because
it cannot distinguish "the bridge is wrong" from "the input was approximate".

So the control generates a truth trajectory with exact events, feeds those
same exact events through SimDriver, and requires the divergence to be
identically zero. Any non-zero result is unambiguously the bridge's fault.
"""

from __future__ import annotations

import random

import pytest

from bridge.sim_driver import SimDriver
from contracts import UNKNOWN_CARD_SIM_ID, EventSource, PlacementEvent

DECK = [15, 25, 6, 1, 0, 41, 7, 10]

# Own-half tile rows only. GameManager::isValidPlacement rejects a non-spell
# above getOwnHalfMaxY() == 15.0 (river re-centred on 16.5, see geometry.py's
# own docstring), so 15 is the last legal row.
OWN_ROWS = (5, 8, 11, 14)
# Opponent placements are injected raw, but the events carry mirrored y (see
# contracts.PlacementEvent), so these are in the mirrored frame too.
OPP_ROWS = (5, 8, 11, 14)


def _build_truth(engine, seed: int, n_events: int = 14):
    """Run a match with scripted plays, recording EXACT events.

    Own plays go through step_self_play by hand index, so they are always
    legal by construction and the card id is known exactly. Opponent plays go
    through inject_enemy, the same primitive the bridge uses, so the control
    isolates the bridge's bookkeeping rather than re-testing the engine.
    """
    rng = random.Random(seed)
    env = engine.ClashRoyaleEnv(DECK, DECK, 3600)
    env.reset()
    opening_hand = tuple(env.get_hand())

    events: list[tuple[PlacementEvent, int | None]] = []
    tick = 0
    for i in range(n_events):
        gap = rng.randint(40, 90)
        env.step_self_play(-1, 0.0, 0.0, -1, 0.0, 0.0, gap)
        tick += gap

        if i % 2 == 0:
            hand = list(env.get_hand())
            index = rng.randrange(len(hand))
            card = hand[index]
            x = rng.randrange(2, 16)
            y = rng.choice(OWN_ROWS)
            env.step_self_play(index, float(x), float(y), -1, 0.0, 0.0, 1)
            after = list(env.get_hand())
            tick += 1
            if after == hand:
                continue  # engine refused (elixir); not an event at all
            events.append((
                PlacementEvent(
                    tick=tick - 1, wall_time_ms=0.0, card_sim_id=card,
                    card_real_name=f"card{card}", team=0,
                    tile_x=x, tile_y=y, confidence=1.0,
                    source=EventSource.REPLAY_GROUND_TRUTH,
                ),
                after[index],
            ))
        else:
            card = rng.choice(DECK)
            x = rng.randrange(2, 16)
            mirrored_y = rng.choice(OPP_ROWS)
            raw_y = float(33 - mirrored_y)
            env.inject_enemy(card, float(x), raw_y)
            events.append((
                PlacementEvent(
                    tick=tick, wall_time_ms=0.0, card_sim_id=card,
                    card_real_name=f"card{card}", team=1,
                    tile_x=x, tile_y=mirrored_y, confidence=1.0,
                    source=EventSource.REPLAY_GROUND_TRUTH,
                ),
                None,
            ))

    # Settle, then read the truth state.
    env.step_self_play(-1, 0.0, 0.0, -1, 0.0, 0.0, 60)
    tick += 60
    return env, opening_hand, events, tick


def _tower_hp_from_env(env, engine):
    """Same readout SimDriver.tower_hp uses, applied to any env."""
    obs = env.get_observation_for_team(0)
    W = engine.ClashRoyaleEnv.BOARD_WIDTH
    H = engine.ClashRoyaleEnv.BOARD_HEIGHT
    mx = engine.ClashRoyaleEnv.MAX_BUILDING_HP
    plane = W * H

    def read(ch, x, y):
        return int(round(obs[ch * plane + int(y) * W + int(x)] * mx))

    return {
        "own_king": read(3, 8.5, 2.5),
        "own_princess_left": read(3, 3.0, 6.0),
        "own_princess_right": read(3, 14.0, 6.0),
        "opp_king": read(7, 8.5, 30.5),
        "opp_princess_left": read(7, 3.0, 27.0),
        "opp_princess_right": read(7, 14.0, 27.0),
    }


@pytest.mark.parametrize("seed", [1, 7, 23])
def test_zero_error_events_reproduce_truth_exactly(engine, seed):
    """The control. Exact events in => exact state out, divergence 0."""
    truth_env, opening_hand, events, final_tick = _build_truth(engine, seed)
    expected = _tower_hp_from_env(truth_env, engine)

    driver = SimDriver(my_deck=DECK, opp_deck=DECK)
    driver.start(my_opening_hand=opening_hand)

    for event, incoming in events:
        result = driver.apply(event, incoming_card=incoming)
        assert result.applied, f"bridge failed to apply {event}: {result.reason}"

    driver.advance_to(final_tick)

    assert driver.report.own_plays_cycle_desync == 0
    assert driver.report.own_plays_rejected_by_engine == 0
    assert driver.tower_hp() == expected, (
        "bridge did not reproduce ground truth from exact events -- this is a "
        "bridge defect, not input noise"
    )
    assert driver.divergence(expected) == 0.0


@pytest.mark.parametrize("seed", [1, 7, 23])
def test_candidate_pool_identifies_the_real_cycle(engine, seed):
    """The pool collapses to our actual cycle within a few plays.

    This is the property that makes own-side injection possible at all
    without an engine change -- see sim_driver.py's docstring.
    """
    truth_env, opening_hand, events, _ = _build_truth(engine, seed)

    driver = SimDriver(my_deck=DECK, opp_deck=DECK)
    initial = driver.start(my_opening_hand=opening_hand)
    assert initial > 1, "the pool should start ambiguous, or it proves nothing"

    for event, incoming in events:
        driver.apply(event, incoming_card=incoming)

    # APPLIED plays, not attempted. A play the engine refuses deals no card,
    # so it reveals nothing about the queue and cannot count toward pinning
    # it down -- see SimDriver.cycle_identified.
    applied = driver.report.own_plays_applied
    if applied >= 4:
        assert driver.cycle_identified, (
            f"the cycle was not pinned down after {applied} applied plays; "
            f"{driver.report.candidates_final} candidates left"
        )
        assert driver.report.cycle_identified_after is not None
        assert driver.report.cycle_identified_after <= driver.report.own_plays_attempted

    # Hand MEMBERSHIP must match the truth env. Not slot order -- see
    # SimDriver._cycle_agreed for why arrangement is a free permutation that
    # nothing observable depends on.
    assert set(driver.primary.get_hand()) == set(truth_env.get_hand())


def test_start_rejects_a_hand_that_is_not_in_the_deck(engine):
    driver = SimDriver(my_deck=DECK, opp_deck=DECK)
    with pytest.raises(RuntimeError, match="no reset produced"):
        driver.start(my_opening_hand=(99, 100, 101, 102))


def test_unmapped_card_is_reported_not_injected(engine):
    driver = SimDriver(my_deck=DECK, opp_deck=DECK)
    driver.start()
    before = driver.tower_hp()

    result = driver.apply(PlacementEvent(
        tick=10, wall_time_ms=0.0, card_sim_id=UNKNOWN_CARD_SIM_ID,
        card_real_name="Some Unimplemented Card", team=1,
        tile_x=8, tile_y=10, confidence=0.9,
    ))

    assert result.applied is False
    assert "no simulator id" in result.reason
    assert driver.report.unmapped_skipped == 1
    assert driver.tower_hp() == before


def test_opponent_y_is_unmirrored_before_injection(engine):
    """A team-1 event's tile_y is in ClashEnv (mirrored) convention.

    Getting this backwards puts every opponent card in their own back rows
    instead of near the river -- plausible-looking and completely wrong. The
    check is that a card placed just past the river on their side actually
    damages OUR tower, which can only happen if it spawned near the river.
    """
    driver = SimDriver(my_deck=DECK, opp_deck=DECK)
    driver.start()

    # Mirrored y=14 is raw y=19: their side, just past the river, in the
    # right-hand lane. A Hog Rider from there reaches our tower quickly.
    driver.apply(PlacementEvent(
        tick=1, wall_time_ms=0.0, card_sim_id=15, card_real_name="Hog Rider",
        team=1, tile_x=14, tile_y=14, confidence=1.0,
    ))
    driver.advance_to(400)

    hp = driver.tower_hp()
    assert hp["own_princess_right"] < 2534, (
        "opponent Hog Rider never reached our tower -- the mirror is inverted"
    )


def test_events_must_arrive_in_tick_order(engine):
    driver = SimDriver(my_deck=DECK, opp_deck=DECK)
    driver.start()
    driver.apply(PlacementEvent(50, 0.0, 15, "Hog Rider", 1, 8, 10, 1.0))
    with pytest.raises(ValueError, match="tick order"):
        driver.apply(PlacementEvent(20, 0.0, 15, "Hog Rider", 1, 8, 10, 1.0))


def test_divergence_excludes_king_towers(engine):
    """King HP is excluded because the engine's King has no activation.

    Tower.h gives the King no dormancy condition at all -- it fires from tick
    0, while the real King is inert until activated. Including it would add a
    systematic error to every measurement and bury the signal.
    """
    driver = SimDriver(my_deck=DECK, opp_deck=DECK)
    driver.start()
    predicted = driver.tower_hp()

    observed = dict(predicted)
    observed["own_king"] = 0
    observed["opp_king"] = 0
    assert driver.divergence(observed) == 0.0

    observed = dict(predicted)
    observed["own_princess_left"] -= 100
    assert driver.divergence(observed) == pytest.approx(25.0)  # 100 over 4 towers
