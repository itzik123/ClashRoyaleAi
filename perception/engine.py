"""The single place that decides which `clash_royale_env` build gets imported.

Two copies of the binding exist:

    python_ai/clash_royale_env.pyd            what the build copies to
    build_python/Release/...cp311-...pyd      what the build produces

The post-build copy fails (MSB3073) while any Python process has the module
mapped, so during a training run python_ai/ holds a stale binary; a binding
added an hour ago is simply not an attribute. `sys.modules` caches by name, so
the first import in the process decides which binary everyone gets; importing
this module installs the path before any of them. Copying over python_ai/
instead could disturb a running training job.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BUILD_OUTPUT = Path(__file__).resolve().parent.parent / "build_python" / "Release"


def install_path() -> Path | None:
    """Put the fresh build ahead of python_ai/ on sys.path. Idempotent. Returns
    the promoted directory, or None if it does not exist (a never-built
    checkout, where the stale-copy problem cannot arise).
    """
    if not _BUILD_OUTPUT.is_dir():
        return None
    path = str(_BUILD_OUTPUT)
    # Remove-then-insert: another module may already have appended it behind
    # python_ai/.
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


# Installed on import, so `import engine` alone makes a later bare `import
# clash_royale_env` correct.
install_path()
