"""The BLAS thread cap, and the import-order invariant it depends on.

WHY THIS EXISTS. `import numpy` (scipy-openblas) commits 32.1 MB of private
memory PER BLAS THREAD, and with the variable unset OpenBLAS defaults to one
thread per core. Measured on this machine (12 logical processors):

    OPENBLAS_NUM_THREADS   private commit added by `import numpy`
    unset                  397.4 MB
    1                       44.1 MB
    2                       76.1 MB
    4                      140.4 MB
    8                      268.8 MB
    12                     397.5 MB          (== unset, confirming the default)

i.e. `private_MB ~= 44.1 + 32.1 * (threads - 1)`. The WORKING SET is flat at
~16 MB across every setting, so this is invisible in Task Manager's default
column and shows only in private commit -- it looks like nothing until the
commit limit is hit. A phase-1 run is 1 main + `num_envs` workers, so at the
default num_envs=8 that is nine processes each paying ~353 MB it cannot use.

THE TRAP THIS FILE REALLY GUARDS. The variable is read by OpenBLAS when the
library LOADS, so setting it after numpy is imported does exactly nothing --
measured, and bit-identical to never setting it at all:

    set before numpy                      43.9 MB
    gymnasium (=> numpy) first, then set  403.3 MB
    gymnasium first, never set            403.3 MB

Both trainers used to `import gymnasium` BEFORE `import python_ai`, so putting
the cap in `python_ai/__init__.py` alone would have been a silent no-op that
measures as a fix and changes nothing. The static test below is the part that
keeps it working: it fails if any entry point ever again imports a
numpy-pulling module ahead of `python_ai`.
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

#: Modules whose import loads numpy (and therefore OpenBLAS) transitively.
_NUMPY_PULLERS = {"numpy", "gymnasium", "torch", "scipy", "pandas",
                  "clash_royale_env"}


def _first_import_positions(path):
    """(lineno of first python_ai import, [(lineno, module) pulled earlier])."""
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
    """THE regression detector for the cap.

    OpenBLAS reads its thread count at library load. Once numpy is in, the
    cap can no longer take effect, and it fails SILENTLY -- the run simply
    keeps paying ~353 MB per process. A static check is the only thing that
    catches the reordering that causes it.
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
    """An explicit setting from the caller must win -- someone benchmarking
    numpy, or a box where a wider BLAS genuinely helps, sets this deliberately.
    """
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "4")
    python_ai._apply_blas_caps()
    assert os.environ["OPENBLAS_NUM_THREADS"] == "4"


def test_torch_intra_op_threads_are_NOT_capped():
    """OMP_NUM_THREADS is deliberately left alone.

    The main process spends ~87% of its wall clock in the PPO update, which is
    conv-bound, and torch's intra-op parallelism runs on OpenMP. Capping that
    to 1 to save memory would trade the update's throughput for RAM the update
    does not use -- torch costs ~166 MB at import regardless of thread count
    and only ~4.4 MB per OMP thread, so there is nothing to win there.
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
    """The end-to-end proof, in a CHILD process.

    Not assertable in-process: OpenBLAS is already loaded here, so the only
    honest measurement spawns a fresh interpreter. Compares `import python_ai`
    first against the uncapped control on the SAME machine, so the assertion
    does not hardcode this box's core count.
    """
    capped = _child_commit("import python_ai")
    control = _child_commit("pass")
    assert capped < control * 0.5, (
        f"import python_ai left numpy's commit at {capped:.1f} MB against an "
        f"uncapped {control:.1f} MB -- the cap did not take effect")
