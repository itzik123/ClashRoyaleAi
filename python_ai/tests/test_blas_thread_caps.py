"""The BLAS thread cap, and the import-order invariant it depends on.

`import numpy` (scipy-openblas) commits ~32 MB of private memory per BLAS
thread, one thread per core by default, invisible in the working set.
`python_ai/__init__.py` sets OPENBLAS_NUM_THREADS=1, but OpenBLAS reads it when
the library loads, so the cap is a silent no-op if anything pulls numpy before
`import python_ai`. The static test below keeps the entry points in the right
order.
"""
import ast
import os
import subprocess
import sys

import pytest

import python_ai

#: Entry points that must import `python_ai` before anything that pulls numpy.
ENTRY_POINTS = (
    "trainers/train.py",
    "trainers/train_selfplay.py",
)

#: Modules whose import loads numpy (and so OpenBLAS) transitively.
_NUMPY_PULLERS = {"numpy", "gymnasium", "torch", "scipy", "pandas",
                  "clash_royale_env"}


def _first_import_positions(path):
    """(lineno of first python_ai import, [(lineno, module) pulled earlier]).
    """
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)

    python_ai_line = None
    pullers = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            root = name.split(".")[0]
            if root == "python_ai" and python_ai_line is None:
                python_ai_line = node.lineno
            elif root in _NUMPY_PULLERS:
                pullers.append((node.lineno, root))
    return python_ai_line, pullers


@pytest.mark.parametrize("rel", ENTRY_POINTS)
def test_python_ai_is_imported_before_anything_that_loads_numpy(rel):
    """The regression detector: once numpy is loaded the cap cannot take effect,
    and nothing fails.
    """
    path = os.path.join(python_ai.PACKAGE_DIR, *rel.split("/"))
    python_ai_line, pullers = _first_import_positions(path)
    assert python_ai_line is not None, f"{rel} never imports python_ai"

    too_early = [(ln, mod) for ln, mod in pullers if ln < python_ai_line]
    assert not too_early, (
        f"{rel} imports {too_early} BEFORE `import python_ai` (line "
        f"{python_ai_line}). python_ai/__init__.py sets OPENBLAS_NUM_THREADS, "
        "and OpenBLAS reads it at load time -- so anything that pulls numpy "
        "first makes the cap a silent no-op worth ~353 MB per process.")


def test_the_package_caps_openblas_on_import():
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "1"
    assert "OPENBLAS_NUM_THREADS" in python_ai.BLAS_THREAD_VARS


def test_the_cap_is_a_default_not_an_override(monkeypatch):
    """An explicit setting from the caller wins."""
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "4")
    python_ai._apply_blas_caps()
    assert os.environ["OPENBLAS_NUM_THREADS"] == "4"


def test_torch_intra_op_threads_are_NOT_capped():
    """OMP_NUM_THREADS is left alone: the conv-bound PPO update runs on torch's
    OpenMP threads, and those cost little memory.
    """
    assert "OMP_NUM_THREADS" not in python_ai.BLAS_THREAD_VARS
    import torch
    assert torch.get_num_threads() > 1 or (os.cpu_count() or 1) == 1


_CHILD = r'''
import ctypes, os
from ctypes import wintypes

class P(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t)]

_ps = ctypes.WinDLL("psapi.dll")
_ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(P),
                                     wintypes.DWORD]
_ps.GetProcessMemoryInfo.restype = wintypes.BOOL

def priv():
    c = P(); c.cb = ctypes.sizeof(P)
    if not _ps.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return c.PrivateUsage / 1048576.0

import sys
sys.path.insert(0, REPO)
base = priv()
MODE
import numpy as np
a = np.random.rand(64, 64)
for _ in range(3):
    (a @ a).sum()
print(f"{priv() - base:.1f}")
'''


def _child_commit(mode):
    src = _CHILD.replace("REPO", repr(python_ai.REPO_ROOT)).replace("MODE", mode)
    env = dict(os.environ)
    env.pop("OPENBLAS_NUM_THREADS", None)
    out = subprocess.run([sys.executable, "-c", src], env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    return float(out.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(sys.platform != "win32", reason="uses psapi")
@pytest.mark.slow
def test_importing_the_package_actually_cuts_the_commit():
    """End to end, in a child process (OpenBLAS is already loaded here). Compares
    against an uncapped control on the same machine, so no core count is
    hardcoded.
    """
    capped = _child_commit("import python_ai")
    control = _child_commit("pass")
    assert capped < control * 0.5, (
        f"import python_ai left numpy's commit at {capped:.1f} MB against an "
        f"uncapped {control:.1f} MB -- the cap did not take effect")
