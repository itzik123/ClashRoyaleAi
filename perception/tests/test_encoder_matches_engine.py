"""The live encoder must produce the SAME observation the engine does.

WHY THIS IS WORTH A TEST
------------------------
`perception_encoder.encode(GameState)` and `ClashEnv::getObservationForTeam(0)`
are two independent implementations of one 13,606-float layout, on opposite
sides of the perception/training boundary. The policy is trained on the second
and deployed on the first. If they disagree, the live agent is running
off-distribution and every symptom is downstream and confusing -- which is
exactly the hypothesis this test was written to settle when a live match placed
60% of its cards on the back two rows against 19-21% in simulation.

Measured: they agree EXACTLY on the towers-only baseline (0 differing cells of
12,852 spatial and 0 of 754 scalar), which localised that discrepancy to unit
detection rather than to the encoding.

Towers-only is the strongest state to pin. It is the one configuration whose
ground truth both sides know without any detector involvement, so a failure
here is unambiguously a layout drift -- the class of bug CLAUDE.md records
twice already (the 6253 -> 13606 change, and the team-1 row displacement).
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

# Deliberately NOT gym_wrapper.DEFAULT_DECK: that module imports gymnasium,
# which the perception venv has no reason to carry, and this test is about the
# observation LAYOUT rather than about any particular deck. Both sides are
# handed the same sampled deck, so the assertion holds for any of them -- and
# resampling each run is slightly better coverage than pinning one.
DECK = cre.sample_random_deck()


def _towers_only_state(env):
    """A GameState matching the engine immediately after reset().

    Elixir and hand are read FROM the engine rather than hardcoded, so the two
    observations differ only where the encoders differ -- otherwise a mismatched
    hand would light up 4 x NUM_CARD_IDS one-hot slots and drown the signal.
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
    """Guards the test above from passing on two identically-empty vectors.

    An encoder that returned zeros and an engine that returned zeros would agree
    perfectly and prove nothing, which is the failure mode a pure equality
    assertion invites.
    """
    env = cre.ClashRoyaleEnv(list(DECK), list(DECK))
    env.reset()
    live = np.asarray(pe.encode(_towers_only_state(env)), dtype=np.float32)
    plane = cre.ClashRoyaleEnv.BOARD_WIDTH * cre.ClashRoyaleEnv.BOARD_HEIGHT
    spatial = live[:cre.ClashRoyaleEnv.NUM_CHANNELS * plane]
    assert np.count_nonzero(spatial) >= 6, "expected at least the six towers encoded"
