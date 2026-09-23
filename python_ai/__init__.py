"""The reinforcement-learning half of ClashRoyaleEnv.

Importing this package puts its own directory on `sys.path`, which is what
makes the unpackaged `clash_royale_env.pyd` importable everywhere. It is
appended rather than prepended so nothing here can shadow a stdlib module.

    models/     MicroRoyaleNet, checkpoint IO, the observation encoder
    envs/       gym environments, self-play, scripted opponents, scenarios
    opponents/  the UtilityTeacher (phase 1's opponent) and the deck pool
    advisors/   the deterministic tactical advisor and its training target
    rewards/    reward shaping: weights, terms, potentials
    rl/         the PPO algorithm: buffer, GAE, update, controllers
    trainers/   the two pipelines that compose rl/ into a training run
    search/     decision-time lookahead
    eval/       measurement harnesses (prove_*, probe_*, *_ab)
    tools/      operational scripts (validate, monitor, setup)
    tests/      the test suite
"""
import os as _os
import sys as _sys

#: Capped before numpy loads. Each OpenBLAS thread commits ~32 MB of private
#: memory, ~3 GB across a phase-1 run's worker processes, and nothing here does
#: BLAS-sized work. OMP_NUM_THREADS is left alone: torch's update uses it.
BLAS_THREAD_VARS = ("OPENBLAS_NUM_THREADS",)


def _apply_blas_caps():
    """`setdefault`, so an explicit setting still wins."""
    for _var in BLAS_THREAD_VARS:
        _os.environ.setdefault(_var, "1")


# Must run before anything imports numpy: OpenBLAS reads the variable when it
# loads. tests/test_blas_thread_caps.py pins the import order.
_apply_blas_caps()

#: This package's directory: where `clash_royale_env.pyd` and the `*.pth`
#: checkpoints live.
PACKAGE_DIR = _os.path.dirname(_os.path.abspath(__file__))

#: The repository root. Entry-point scripts put it on sys.path so `python_ai.*`
#: resolves when run as files.
REPO_ROOT = _os.path.dirname(PACKAGE_DIR)

if PACKAGE_DIR not in _sys.path:
    _sys.path.append(PACKAGE_DIR)
