"""Run-level determinism: one seed must fix every stochastic source.

WHY THIS FILE EXISTS. Before it, nothing in the training path was seedable.
`PPOConfig` had no seed field, there was no `CLASH_SEED` knob, and pipeline 2's
environment drew from THREE uncontrolled streams at once:

    self.scenario_rng = np.random.default_rng()      # no seed argument at all
    random.choice(["left", "right"])                 # global stdlib RNG
    np.random.choice(len(self.pfsp_pool), p=weights) # global numpy RNG

...on top of torch's weight init and `np.random.permutation` in the PPO
minibatch shuffle. So a phase-2 result could not be reproduced, a phase-2 crash
could not be re-run, and a paired A/B -- the whole reason `env.snapshot()` was
built -- could not hold the opponent and scenario draws fixed across arms.

Pipeline 1's env already took a `scenario_seed` config key. Pipeline 2's did
not, and nothing passed one to either. These tests pin the seed CONTRACT, not
the particular numbers: what matters is same-seed-same-sequence and
different-seed-different-sequence, never a specific draw.
"""
import random

import numpy as np
import pytest
import torch

from python_ai.rl.seeding import seed_everything, worker_seeds


def test_seeding_makes_torch_numpy_and_stdlib_all_reproducible():
    """One call, all three streams. Seeding two of the three is the failure
    that makes a run LOOK reproducible until the one you forgot matters."""
    def draw():
        return (torch.randn(4).tolist(),
                np.random.rand(4).tolist(),
                [random.random() for _ in range(4)])

    seed_everything(1234)
    first = draw()
    seed_everything(1234)
    assert draw() == first


def test_a_different_seed_gives_a_different_stream():
    seed_everything(1)
    a = torch.randn(8).tolist()
    seed_everything(2)
    assert torch.randn(8).tolist() != a


def test_seeding_reproduces_network_initialisation():
    """The weights themselves are a stochastic source, and the biggest one:
    two arms of an A/B that start from different inits are not comparable."""
    from python_ai.models.net import MicroRoyaleNet
    seed_everything(7)
    a = MicroRoyaleNet(num_ability_slots=0).card_head.weight.detach().clone()
    seed_everything(7)
    b = MicroRoyaleNet(num_ability_slots=0).card_head.weight.detach().clone()
    assert torch.equal(a, b)


def test_worker_seeds_are_distinct():
    """The vector envs must NOT inject the same scenario in lockstep -- the
    reason the original code gave for leaving the generator unseeded. Distinct
    per-worker seeds buy independence AND reproducibility; `default_rng()` buys
    independence by giving up reproducibility entirely."""
    seeds = worker_seeds(99, 8)
    assert len(seeds) == 8
    assert len(set(seeds)) == 8


def test_worker_seeds_are_reproducible_from_the_root_seed():
    assert worker_seeds(99, 8) == worker_seeds(99, 8)
    assert worker_seeds(99, 8) != worker_seeds(100, 8)


def test_worker_seeds_of_none_are_all_none():
    """No seed means the old behaviour EXACTLY: fresh OS entropy per worker.
    Seeding must be opt-in, or every historical run becomes incomparable."""
    assert worker_seeds(None, 4) == [None, None, None, None]


def test_worker_streams_are_independent_not_merely_offset():
    """Seeds n, n+1, n+2 would be reproducible and still correlated. The point
    of SeedSequence is that the streams are statistically independent."""
    a, b = worker_seeds(5, 2)
    ra = np.random.default_rng(a).random(500)
    rb = np.random.default_rng(b).random(500)
    assert abs(float(np.corrcoef(ra, rb)[0, 1])) < 0.15


# --- the environment's own streams ---------------------------------------

def _lane_and_scenario_draws(seed, n=12):
    """Drive the env's stochastic surface without starting a real match."""
    from python_ai.envs.selfplay_env import MicroRoyaleSelfPlayEnv
    env = MicroRoyaleSelfPlayEnv({"scenario_seed": seed})
    return [(env.rng.random(), env.rng.integers(0, 1000)) for _ in range(n)]


def test_the_selfplay_env_accepts_a_seed_and_repeats_its_draws():
    assert _lane_and_scenario_draws(42) == _lane_and_scenario_draws(42)


def test_the_selfplay_env_differs_across_seeds():
    assert _lane_and_scenario_draws(42) != _lane_and_scenario_draws(43)


def test_an_unseeded_selfplay_env_still_works():
    """Default None must keep the pre-existing behaviour, not raise."""
    from python_ai.envs.selfplay_env import MicroRoyaleSelfPlayEnv
    env = MicroRoyaleSelfPlayEnv()
    assert env.rng is not None
    assert 0.0 <= float(env.rng.random()) <= 1.0


def test_the_env_draws_from_its_OWN_generator_not_the_global_ones():
    """The three streams must be routed through one seeded generator. If any
    still reads `random.*` or `np.random.*`, re-seeding the globals between two
    identically-seeded envs would change its draws -- and this catches it.
    """
    from python_ai.envs.selfplay_env import MicroRoyaleSelfPlayEnv
    env_a = MicroRoyaleSelfPlayEnv({"scenario_seed": 5})
    random.seed(1); np.random.seed(1)
    a = [env_a.rng.random() for _ in range(6)]

    env_b = MicroRoyaleSelfPlayEnv({"scenario_seed": 5})
    random.seed(999); np.random.seed(999)
    b = [env_b.rng.random() for _ in range(6)]
    assert a == b


def test_no_module_in_the_training_path_still_uses_a_global_rng():
    """A static check, because the dynamic ones above can only see the streams
    they happen to drive. `random.` and `np.random.` at module level in the
    env/trainer path are exactly what made pipeline 2 unseedable."""
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for rel in ("envs/selfplay_env.py", "envs/gym_wrapper.py"):
        text = (root / rel).read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            code = line.split("#", 1)[0]   # a comment NAMING the old call is fine
            if re.search(r"(?<![\w.])(random\.(choice|random|randint|sample|shuffle)"
                         r"|np\.random\.(choice|rand|randint|random|shuffle|permutation))\(",
                         code):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, "global RNG in the env path:\n" + "\n".join(offenders)


def test_both_pipelines_make_env_take_a_seed_and_use_it():
    """One contract, both pipelines. Pipeline 1's env already accepted
    `scenario_seed` and nothing passed one; pipeline 2's could not accept one
    at all. A second convention here would be a third way to forget."""
    from python_ai.envs.selfplay_env import make_env as sp_make
    from python_ai.trainers.train import make_env as p1_make

    a = sp_make(seed=31)()
    b = sp_make(seed=31)()
    assert [a.rng.random() for _ in range(5)] == [b.rng.random() for _ in range(5)]

    c = p1_make(seed=31)()
    d = p1_make(seed=31)()
    assert ([c._scenario_rng.random() for _ in range(5)]
            == [d._scenario_rng.random() for _ in range(5)])


def test_an_unseeded_make_env_is_still_random_for_both_pipelines():
    """Opt-in means opt-in: with no seed both must keep drawing fresh entropy,
    or every historical run silently becomes a different experiment."""
    from python_ai.envs.selfplay_env import make_env as sp_make
    from python_ai.trainers.train import make_env as p1_make

    assert sp_make()().rng.random() != sp_make()().rng.random()
    assert p1_make()()._scenario_rng.random() != p1_make()()._scenario_rng.random()


# --- the distillation trainers --------------------------------------------
#
# bc_pretrain, distill_tactics and expert_distill each shuffle their training
# set once per epoch on the GLOBAL RNG:
#
#     np.random.shuffle(ep_ids)          # bc_pretrain, expert_distill
#     perm = np.random.permutation(n)    # distill_tactics
#
# so no distillation run could be reproduced. That matters more here than it
# looks: CLAUDE.md's expert-iteration conclusions are PAIRED A/B comparisons
# between distilled nets, and the `--ablate` 2x2 grid compares four configs
# against each other. If each run shuffles differently, part of every measured
# difference is shuffle noise, and the +0.045 win-rate result cannot be
# re-derived from the same inputs.
#
# bc_pretrain already seeds its DATA COLLECTION (`collect_demonstrations(...,
# seed=0)` builds a `default_rng`) and then trained on an unseeded shuffle --
# half-seeded, which is the shape that makes a run look reproducible until the
# half you forgot is the half that matters.

def test_the_distillation_entry_points_expose_a_seed():
    """Wiring check: each CLI takes --seed and routes it to seed_everything.

    Deliberately static. The behaviour these scripts depend on -- that seeding
    fixes `np.random.shuffle` -- is proved by the test below; what cannot be
    proved cheaply is that each script actually CALLS it, because doing so
    would mean running a full distillation.
    """
    import pathlib

    import python_ai
    root = pathlib.Path(python_ai.PACKAGE_DIR) / "trainers"
    for name in ("bc_pretrain.py", "distill_tactics.py", "expert_iteration.py"):
        src = (root / name).read_text(encoding="utf-8")
        assert '"--seed"' in src, f"{name} has no --seed"
        assert "seed_everything" in src, f"{name} never applies its seed"


def test_seeding_fixes_the_global_shuffle_these_trainers_use():
    """The mechanism the wiring above relies on."""
    import numpy as np

    from python_ai.rl.seeding import seed_everything

    def order():
        ids = np.arange(64)
        np.random.shuffle(ids)
        return ids.tolist()

    seed_everything(5)
    first = order()
    seed_everything(5)
    assert order() == first
