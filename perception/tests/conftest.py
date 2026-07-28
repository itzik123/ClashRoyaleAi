"""Shared pytest setup.

Puts perception/ on sys.path so modules import by their plain names
(`from contracts import ...`) both in tests and in normal use, without the
package needing to be installed.

Nothing here reaches outside perception/ except to READ python_ai/ for the
simulator binding and its replays. Nothing in this package ever writes there.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PERCEPTION_ROOT = Path(__file__).resolve().parent.parent
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

ASSETS = PERCEPTION_ROOT / "tests" / "assets"


def _engine_or_none():
    sys.path.insert(0, str(PERCEPTION_ROOT.parent / "python_ai"))
    try:
        import clash_royale_env
        return clash_royale_env
    except Exception:
        return None


@pytest.fixture(scope="session")
def engine():
    """The compiled simulator, or skip.

    Skipped rather than failed when absent: the .pyd is a build artefact of a
    separate toolchain, and the calibration/reader half of this package is
    fully testable without it. A hard failure here would make an unrelated
    build problem look like a perception regression.
    """
    module = _engine_or_none()
    if module is None:
        pytest.skip("clash_royale_env not importable (needs the Python 3.11 .pyd)")
    return module


@pytest.fixture(scope="session")
def replay_path():
    """A frozen replay fixture, or skip.

    Frozen copies, not python_ai/replays/ -- that directory is rewritten
    continuously by the training run and was observed dropping from eight
    files to one mid-session, which would make these tests pass or fail
    depending on timing.
    """
    paths = sorted(ASSETS.glob("*.json"))
    if not paths:
        pytest.skip(f"no replay fixture in {ASSETS}")
    return paths[0]


@pytest.fixture(scope="session")
def replay(replay_path):
    from simlog import Replay
    return Replay(replay_path)
