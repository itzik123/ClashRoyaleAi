"""pytest bootstrap: puts the repo root on sys.path so `python_ai.*` resolves from
any cwd, and imports `python_ai` so `clash_royale_env` does too.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import python_ai  # noqa: E402,F401


def pytest_configure(config):
    """Register the suite's markers so an unknown-mark warning stays meaningful.
    """
    config.addinivalue_line(
        "markers",
        "slow: exercises a real environment or a real training update "
        "(seconds, not milliseconds). Deselect with -m 'not slow'.")
