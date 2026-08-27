"""The reinforcement-learning half of ClashRoyaleEnv: model, environments,
trainers, advisors and the measurement harnesses that keep them honest.

WHY THIS FILE EXISTS AT ALL, and it is not just a package marker.

`clash_royale_env` is a compiled pybind11 extension that lives at
`python_ai/clash_royale_env.pyd` with no package of its own, so that directory
has to be on `sys.path` before ANY module in here can `import clash_royale_env`.
Until 2026-08-20 every module carried its own
`sys.path.insert(0, dirname(__file__))` line to arrange that -- ~30 copies of
one fact. Importing this package now does it once, which is the same rule
CLAUDE.md applies to engine constants: one derivation point, no second copies.

The directory is APPENDED rather than prepended: prepending would let a file in
here shadow a stdlib or site-packages module of the same name, and nothing in
this package needs to win that race.

Layout -- one package per responsibility:

    models/     MicroRoyaleNet, checkpoint IO, the observation encoder
    envs/       gym environments, self-play, scripted opponents, scenarios
    opponents/  the UtilityTeacher (phase 1's sparring partner)
    advisors/   the deterministic tactical advisor and its training target
    rewards/    reward shaping: weights, terms, potentials
    rl/         the PPO algorithm itself -- buffer, GAE, update, controllers
    trainers/   the two pipelines that compose rl/ into a training run
    search/     decision-time lookahead
    eval/       measurement harnesses (prove_*, probe_*, *_ab)
    tools/      operational scripts (validate, monitor, setup)
    tests/      the test suite
"""
import os as _os
import sys as _sys

#: Thread-count variables capped on import, BEFORE numpy is loaded.
#:
#: `import numpy` (scipy-openblas) commits 32.1 MB of private memory per BLAS
#: thread, and unset OpenBLAS defaults to one thread per core. Measured here on
#: 12 logical processors: 397.4 MB unset against 44.1 MB at 1, fitting
#: `44.1 + 32.1*(threads-1)`. The WORKING SET is flat at ~16 MB either way, so
#: it is invisible in Task Manager and shows only in private commit. A phase-1
#: run is 1 main + num_envs workers, so the default num_envs=8 pays ~3.2 GB of
#: commit that nothing in this package can use: every numpy array here is
#: (num_envs,)-shaped reward/stat bookkeeping, for which a multi-threaded BLAS
#: is pure overhead.
#:
#: OMP_NUM_THREADS IS DELIBERATELY NOT IN THIS LIST. That one governs torch's
#: intra-op parallelism, and the main process spends ~87% of its wall clock in
#: the conv-bound PPO update. Capping it would trade real update throughput for
#: memory the update does not use -- torch costs ~166 MB at import regardless of
#: thread count, plus only ~4.4 MB per OMP thread, so there is nothing to
#: reclaim there.
BLAS_THREAD_VARS = ("OPENBLAS_NUM_THREADS",)


def _apply_blas_caps():
    """`setdefault` each cap. Deliberately not an override: someone
    benchmarking numpy, or on a box where a wider BLAS genuinely pays, sets
    these on purpose and must win.
    """
    for _var in BLAS_THREAD_VARS:
        _os.environ.setdefault(_var, "1")


# MUST run before anything imports numpy: OpenBLAS reads the variable when the
# library LOADS, so a later set is measurably a no-op (403.3 MB, bit-identical
# to never setting it). That is why both trainers import THIS package before
# they import gymnasium/numpy/torch, and why
# tests/test_blas_thread_caps.py pins that ordering statically -- a reordering
# would not raise, it would just quietly cost the memory again.
_apply_blas_caps()

#: Absolute path of this package -- also where `clash_royale_env.pyd` and the
#: `*.pth` checkpoints live.
PACKAGE_DIR = _os.path.dirname(_os.path.abspath(__file__))

#: The repository root, one level up. Entry-point scripts put this on sys.path
#: so `python_ai.*` resolves when they are run as files rather than as modules.
REPO_ROOT = _os.path.dirname(PACKAGE_DIR)

if PACKAGE_DIR not in _sys.path:
    _sys.path.append(PACKAGE_DIR)
