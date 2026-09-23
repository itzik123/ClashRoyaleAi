"""One seed for every stochastic source.

Covers torch (initialisation, action sampling), numpy's global RNG (minibatch
order, PFSP draws), stdlib random (lane choice), each env's scenario Generator,
and the C++ engine's opening-hand shuffle. Opt-in: `seed is None` leaves every
generator on OS entropy.

Worker seeds come from `SeedSequence(root).spawn(n)`, which is reproducible and
statistically independent; seeding workers n, n+1, ... would be reproducible
but correlated.

A seeded run is bit-reproducible only on the same machine and library versions,
and AsyncVectorEnv's episode completion order still depends on scheduling.
"""
import random

import numpy as np
import torch


def seed_everything(seed):
    """Seed torch, numpy's global RNG and the stdlib's; returns `seed`. None is a
    no-op.
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
    """`n` reproducible, mutually independent seeds for each env's scenario
    generator, or `n` Nones.
    """
    if seed is None:
        return [None] * n
    return [int(s.generate_state(1)[0])
            for s in np.random.SeedSequence(int(seed)).spawn(n)]


def engine_seeds(seed, n):
    """`n` seeds for the engine's own RNG, one per worker, passed to
    `envs.reset(seed=...)`, the only route to the opening-hand shuffle.

    A separate stream from `worker_seeds` (via a two-element SeedSequence key),
    so which scenario is injected is not tied to which hand is dealt.
    """
    if seed is None:
        return [None] * n
    return [int(s.generate_state(1)[0])
            for s in np.random.SeedSequence([int(seed), 0xE1]).spawn(n)]
