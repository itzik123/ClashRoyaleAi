"""tactics.py locates the extra scalars by forward offset.

The observation grows only by appending, so an offset counted back from the end
is right until the next append and then silently wrong. These pin the behaviour
(the clock tracks the engine) and the structure (every read lands inside the
extra-scalar block); the structural check catches a reintroduction before it is
measured wrong.
"""
import numpy as np
import pytest

import clash_royale_env as E
from python_ai.advisors import tactics
from python_ai.envs.gym_wrapper import DEFAULT_DECK

CE = E.ClashRoyaleEnv


def _env():
    env = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), max_ticks=3600)
    env.seed(1)
    env.reset()
    return env


def test_scalar_offsets_are_forward_and_inside_the_extra_scalar_block():
    """The structural guard: every index must sit in [START, START + N)."""
    start = CE.EXTRA_SCALARS_START
    end = start + CE.NUM_EXTRA_SCALARS
    for name in ("IDX_ELAPSED", "IDX_OWN_SPEND", "IDX_OPP_SPEND"):
        idx = getattr(tactics, name)
        assert start <= idx < end, f"{name}={idx} outside [{start}, {end})"
        # ...and not in the cycle blocks appended behind them.
        assert idx < CE.CYCLE_START, f"{name} reads the opponent-cycle block"


@pytest.mark.parametrize("steps", [10, 30, 60])
def test_elapsed_ticks_tracks_the_engine_clock(steps):
    env = _env()
    for _ in range(steps):
        env.step_self_play_fast(4, 0, 0, 4, 0, 0, skip_frames=10)
    obs = np.asarray(env.get_observation_for_team(0), dtype=np.float32)
    assert tactics.elapsed_ticks(obs) == pytest.approx(env.get_current_tick(), rel=1e-4)


def test_elapsed_ticks_is_not_pinned_at_zero():
    """The exact symptom of the old bug, a clock that never advances; the test
    above would also pass on a board where no time had passed.
    """
    env = _env()
    for _ in range(50):
        env.step_self_play_fast(4, 0, 0, 4, 0, 0, skip_frames=10)
    obs = np.asarray(env.get_observation_for_team(0), dtype=np.float32)
    assert tactics.elapsed_ticks(obs) > 100.0


def test_opp_spend_index_reads_the_OPPONENT_not_us():
    """In the mirrored team-1 frame, our spend must appear at IDX_OPP_SPEND.
    `opp_elixir_estimate`, and through it hog_advice's gate, depends on telling
    own from opponent spend.
    """
    env = _env()
    for _ in range(60):
        env.step_self_play_fast(4, 0, 0, 4, 0, 0, skip_frames=10)
    hand = env.get_hand_for_team(0)
    cost = float(E.get_card_info(hand[0])["cost"])
    env.step_self_play_fast(0, 8.0, 8.0, 4, 0, 0, skip_frames=10)
    for _ in range(3):
        env.step_self_play_fast(4, 0, 0, 4, 0, 0, skip_frames=10)

    own = np.asarray(env.get_observation_for_team(0), dtype=np.float32)
    opp = np.asarray(env.get_observation_for_team(1), dtype=np.float32)
    expected = cost / CE.MAX_MATCH_ELIXIR
    assert own[tactics.IDX_OWN_SPEND] == pytest.approx(expected, abs=1e-6)
    assert own[tactics.IDX_OPP_SPEND] == pytest.approx(0.0, abs=1e-6)
    assert opp[tactics.IDX_OPP_SPEND] == pytest.approx(expected, abs=1e-6)
