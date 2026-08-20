"""pytest bootstrap for the whole `python_ai` package.

`pytest python_ai/tests -q` is normally run from the repository root, where the
root is already on `sys.path` -- but pytest also inserts each test file's own
rootdir-relative directory, and a run started from inside `python_ai/` would not
have the root at all. Putting it here, at the package root, makes
`from python_ai.rl.ppo import ...` resolve identically from either cwd.

Importing `python_ai` is also what puts the compiled engine's directory on
`sys.path`, so `import clash_royale_env` works in every test without each file
repeating the incantation. See python_ai/__init__.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import python_ai  # noqa: E402,F401


def pytest_configure(config):
    """Register the markers the suite uses, so an unknown-mark warning stays a
    real signal instead of six lines of noise on every run."""
    config.addinivalue_line(
        "markers",
        "slow: exercises a real environment or a real training update "
        "(seconds, not milliseconds). Deselect with -m 'not slow'.")
