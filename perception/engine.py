"""The single place that decides WHICH `clash_royale_env` build gets imported.

WHY THIS IS NOT JUST `import clash_royale_env`
----------------------------------------------
Two copies of the binding exist at once:

    python_ai/clash_royale_env.pyd            what the build COPIES to
    build_python/Release/...cp311-...pyd      what the build PRODUCES

The post-build copy fails with MSB3073 whenever a Python process has the module
mapped -- Windows will not overwrite a loaded DLL -- so during any training run
python_ai/ holds a stale binary and build_python/Release/ holds the current one.
The failure is silent at the call site: a binding added an hour ago simply is
not an attribute, which reads as "it was never bound" rather than "the copy was
blocked".

`sys.modules` caches by NAME, so the FIRST import anywhere in the process
decides which binary every later importer gets. Eleven modules under
perception/ import it directly, which made the winner a function of import
order. Fixing one of them (forecast.py) fixed nothing unless it happened to run
first -- the path has to be installed before ANY of them, which is what
importing this module does.

Copying over python_ai/ instead would be the obvious alternative and is exactly
what must not happen: overwriting that file mid-run is the one action that could
disturb a multi-hour training job, and perception has no reason to take it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BUILD_OUTPUT = Path(__file__).resolve().parent.parent / "build_python" / "Release"


def install_path() -> Path | None:
    """Put the fresh build ahead of python_ai/ on sys.path. Idempotent.

    Returns the directory that was promoted, or None if it does not exist --
    a source checkout that has never been built is a normal state, and the
    stale-copy problem cannot arise there.
    """
    if not _BUILD_OUTPUT.is_dir():
        return None
    path = str(_BUILD_OUTPUT)
    # Remove-then-insert rather than a membership test: another module may
    # already have appended it BEHIND python_ai/, which would still lose.
    while path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
    return _BUILD_OUTPUT


def load():
    """The bindings, from the freshest build available."""
    install_path()
    try:
        import clash_royale_env  # noqa: PLC0415
    except ImportError as exc:                              # pragma: no cover
        raise ImportError(
            "could not import clash_royale_env. The .pyd is built for Python "
            "3.11; run this with perception/.venv/Scripts/python.exe or "
            "py -3.11."
        ) from exc
    return clash_royale_env


# Installed on import, so that `import engine` anywhere -- even without calling
# load() -- is enough to make a later bare `import clash_royale_env` correct.
install_path()
