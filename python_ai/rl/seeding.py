"""One seed, every stochastic source. The single entry point for determinism.

WHAT WAS UNSEEDED. Before this module there was no seed anywhere in the
training path -- no `PPOConfig.seed`, no `CLASH_SEED`, and nothing calling
`torch.manual_seed`. The stochastic surface of a run was:

    torch                 network initialisation, and every action sampled
    np.random (global)    the PPO minibatch permutation (`rl/ppo.py`), and
                          pipeline 2's PFSP opponent draw
    random (global)       pipeline 2's attacking-lane choice
    per-env Generator     scenario injection -- seedable in pipeline 1,
                          constructed as bare `default_rng()` in pipeline 2
    the C++ engine        the opening-hand shuffle (`ClashEnv::seed`)

So a phase-2 result could not be reproduced, a phase-2 crash could not be
re-run, and a paired A/B could not hold the opponent and scenario draws fixed
across arms -- which is most of what `env.snapshot()` was built to enable.

SEEDING IS OPT-IN. `seed is None` leaves every generator exactly as it was, on
fresh OS entropy. That matters more than it looks: making runs deterministic by
default would silently change what every existing configuration does, and every
win rate in CLAUDE.md was earned unseeded.

WHY `SeedSequence` FOR THE WORKERS. The vectorized envs must not inject the
same scenario in lockstep -- that is the reason the original code gave for
leaving the generator unseeded, and it is a real concern. But `default_rng()`
buys independence by giving up reproducibility, when both are available:
`SeedSequence(root).spawn(n)` yields streams that are reproducible from `root`
AND statistically independent of each other. Seeding workers `n, n+1, n+2`
would be reproducible and still correlated, which is the trap this avoids.

WHAT THIS DOES NOT BUY. Torch on CPU is deterministic for these ops, but a
seeded run is only bit-reproducible on the SAME machine and library versions --
and `AsyncVectorEnv` interleaves workers by process scheduling, so the ORDER
episodes complete in is not fixed by any seed. Seeding pins the draws, not the
wall clock.
"""
import random

import numpy as np
import torch


def seed_everything(seed):
    """Seed torch, numpy's global RNG and the stdlib's. Returns `seed`.

    All three, because seeding two of the three is the failure that makes a run
    LOOK reproducible right up until the one you forgot is the one that matters.
    `None` is a no-op, so callers can pass an unset config through unguarded.
    """
    if seed is None:
        return None
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def worker_seeds(seed, n):
    """`n` reproducible, mutually independent seeds -- or `n` Nones.

    `[None] * n` when unseeded, so `make_env(seed=...)` takes the same code
    path either way and the default behaviour is untouched.

    These seed each env's SCENARIO generator. The engine's own shuffle is a
    separate stream -- see `engine_seeds`.
    """
    if seed is None:
        return [None] * n
    return [int(s.generate_state(1)[0])
            for s in np.random.SeedSequence(int(seed)).spawn(n)]


def engine_seeds(seed, n):
    """`n` seeds for the ENGINE's own RNG, one per worker.

    Passed to `envs.reset(seed=...)`, which is the only route to
    `ClashEnv::seed` and therefore the only way the opening-hand shuffle is
    pinned. `BaseTrainer.setup` used to call `envs.reset()` with no seed at all,
    so a run with CLASH_SEED set reproduced its network initialisation and its
    minibatch permutation while dealing DIFFERENT opening hands every time --
    and printed "Deterministic run" regardless. Measured at seed 4242:

        network init identical : True
        opening hands run A    : [[7, 24, 6, 33], [24, 25, 6, 7]]
        opening hands run B    : [[24, 33, 25, 72], [40, 15, 25, 24]]

    The opening hand decides what the agent is ABLE to play, so this was not a
    minor stochastic source; it is most of an episode's variance, and it is
    exactly what a paired A/B needs held fixed across arms.

    A SEPARATE STREAM from `worker_seeds`, not the same integers reused. Handing
    one number to both would tie which scenario is injected to which hand is
    dealt -- a correlation between two things an experiment varies
    independently, invisible because both would still look properly seeded.
    The distinct entropy comes from the two-element SeedSequence key, which
    leaves `worker_seeds`' existing stream bit-identical.
    """
    if seed is None:
        return [None] * n
    return [int(s.generate_state(1)[0])
            for s in np.random.SeedSequence([int(seed), 0xE1]).spawn(n)]
