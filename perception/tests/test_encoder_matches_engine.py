"""The live encoder must produce the same observation the engine does.

`perception_encoder.encode(GameState)` and `ClashEnv::getObservationForTeam(0)`
are two implementations of one layout on opposite sides of the
perception/training boundary; the policy trains on the second and is deployed
on the first. A disagreement puts the live agent off-distribution with only
confusing downstream symptoms.

Towers-only is the strongest state to pin: the one configuration whose ground
truth both sides know with no detector involved, so a failure here is a layout
drift.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python_ai"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perception"))

pytest.importorskip("clash_royale_env",
                    reason="engine bindings not importable from this interpreter")

import clash_royale_env as cre  # noqa: E402
from python_ai.models import perception_encoder as pe  # noqa: E402
from contracts import GameState, Phase, TowerObservation  # noqa: E402

# Not gym_wrapper.DEFAULT_DECK, which imports gymnasium. The test is about
# layout, not a deck: both sides get the same sampled deck.
DECK = cre.sample_random_deck()


def _towers_only_state(env):
    """A GameState matching the engine immediately after reset(). Elixir and hand
    are read from the engine, so the observations differ only where the
    encoders differ.
    """
    full = TowerObservation(hp_fraction=1.0, hp_measured=True, destroyed=False)
    return GameState(
        units=(),
        my_elixir=float(env.get_elixir()),
        my_hand=tuple(env.get_hand()),
        seconds_elapsed=0.0,
        phase=Phase.SINGLE,
        own_king=full, own_princess_left=full, own_princess_right=full,
        opp_king=full, opp_princess_left=full, opp_princess_right=full,
    )


def test_encoder_reproduces_the_engine_observation_exactly():
    env = cre.ClashRoyaleEnv(list(DECK), list(DECK))
    env.reset()
    sim = np.asarray(env.get_observation_for_team(0), dtype=np.float32)
    live = np.asarray(pe.encode(_towers_only_state(env)), dtype=np.float32)

    assert live.shape == sim.shape == (env.observation_size(),)

    differing = np.flatnonzero(np.abs(sim - live) > 1e-4)
    if differing.size:
        plane = cre.ClashRoyaleEnv.BOARD_WIDTH * cre.ClashRoyaleEnv.BOARD_HEIGHT
        first = int(differing[0])
        where = (f"spatial channel {first // plane}, cell {first % plane}"
                 if first < cre.ClashRoyaleEnv.NUM_CHANNELS * plane
                 else f"scalar index {first - cre.ClashRoyaleEnv.NUM_CHANNELS * plane}")
        pytest.fail(
            f"{differing.size} of {sim.size} floats differ; first at {where} "
            f"(sim={sim[first]:.6f} live={live[first]:.6f}). The live agent is "
            f"running off-distribution.")


def test_tower_channels_are_actually_populated():
    """Guards the test above from passing on two identically empty vectors."""
    env = cre.ClashRoyaleEnv(list(DECK), list(DECK))
    env.reset()
    live = np.asarray(pe.encode(_towers_only_state(env)), dtype=np.float32)
    plane = cre.ClashRoyaleEnv.BOARD_WIDTH * cre.ClashRoyaleEnv.BOARD_HEIGHT
    spatial = live[:cre.ClashRoyaleEnv.NUM_CHANNELS * plane]
    assert np.count_nonzero(spatial) >= 6, "expected at least the six towers encoded"
