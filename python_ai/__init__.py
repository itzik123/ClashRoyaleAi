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

#: Absolute path of this package -- also where `clash_royale_env.pyd` and the
#: `*.pth` checkpoints live.
PACKAGE_DIR = _os.path.dirname(_os.path.abspath(__file__))

#: The repository root, one level up. Entry-point scripts put this on sys.path
#: so `python_ai.*` resolves when they are run as files rather than as modules.
REPO_ROOT = _os.path.dirname(PACKAGE_DIR)

if PACKAGE_DIR not in _sys.path:
    _sys.path.append(PACKAGE_DIR)
