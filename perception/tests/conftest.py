"""Shared pytest setup.

Puts perception/ on sys.path so modules import by plain names (`from contracts
import ...`) in tests and in normal use, without installing the package.
Reaches outside perception/ only to read python_ai/ for the simulator binding
and replays; never writes there.
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
    # The repo root too: python_ai is a package, and tests comparing against
    # its observation encoder import it by that path.
    sys.path.insert(0, str(PERCEPTION_ROOT.parent))
    try:
        import clash_royale_env
        return clash_royale_env
    except Exception:
        return None


@pytest.fixture(scope="session")
def engine():
    """The compiled simulator, or skip. The .pyd is a separate toolchain's
    artifact and the calibration/reader half of this package is testable
    without it; failing would make a build problem look like a perception
    regression.
    """
    module = _engine_or_none()
    if module is None:
        pytest.skip("clash_royale_env not importable (needs the Python 3.11 .pyd)")
    return module


@pytest.fixture(scope="session")
def replay_path():
    """A frozen replay fixture, or skip. Frozen copies, since the training run
    rewrites and prunes python_ai/replays/ continuously.
    """
    # `replay_*.json`, not `*.json`: this directory also holds ground-truth
    # label sets, and a labels file sorting first would be loaded as the
    # replay.
    paths = sorted(ASSETS.glob("replay_*.json"))
    if not paths:
        pytest.skip(f"no replay fixture in {ASSETS}")
    return paths[0]


@pytest.fixture(scope="session")
def replay(replay_path):
    from simlog import Replay
    return Replay(replay_path)
