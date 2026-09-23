"""Run-level determinism: one seed fixes every stochastic source.

These pin the contract (same seed, same sequence; different seed, different
sequence), never particular draws.
"""
import random

import numpy as np
import pytest
import torch

from python_ai.rl.seeding import seed_everything, worker_seeds


def test_seeding_makes_torch_numpy_and_stdlib_all_reproducible():
    """One call seeds torch, numpy and stdlib; seeding two of three looks
    reproducible until the third matters.
    """
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
    """Network initialisation is the biggest stochastic source: two A/B arms from
    different inits are not comparable.
    """
    from python_ai.models.net import MicroRoyaleNet
    seed_everything(7)
    a = MicroRoyaleNet(num_ability_slots=0).card_head.weight.detach().clone()
    seed_everything(7)
    b = MicroRoyaleNet(num_ability_slots=0).card_head.weight.detach().clone()
    assert torch.equal(a, b)


def test_worker_seeds_are_distinct():
    """Distinct per-worker seeds give independence and reproducibility, so the
    vector envs do not inject scenarios in lockstep.
    """
    seeds = worker_seeds(99, 8)
    assert len(seeds) == 8
    assert len(set(seeds)) == 8


def test_worker_seeds_are_reproducible_from_the_root_seed():
    assert worker_seeds(99, 8) == worker_seeds(99, 8)
    assert worker_seeds(99, 8) != worker_seeds(100, 8)


def test_worker_seeds_of_none_are_all_none():
    """No seed means fresh OS entropy per worker, as before; seeding is opt-in.
    """
    assert worker_seeds(None, 4) == [None, None, None, None]


def test_worker_streams_are_independent_not_merely_offset():
    """Seeds n, n+1, n+2 would be reproducible and still correlated; SeedSequence
    streams are independent.
    """
    a, b = worker_seeds(5, 2)
    ra = np.random.default_rng(a).random(500)
    rb = np.random.default_rng(b).random(500)
    assert abs(float(np.corrcoef(ra, rb)[0, 1])) < 0.15


# --- the environment's own streams ---

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
    """Default None keeps the old behaviour."""
    from python_ai.envs.selfplay_env import MicroRoyaleSelfPlayEnv
    env = MicroRoyaleSelfPlayEnv()
    assert env.rng is not None
    assert 0.0 <= float(env.rng.random()) <= 1.0


def test_the_env_draws_from_its_OWN_generator_not_the_global_ones():
    """All draws route through one seeded generator: re-seeding the globals
    between two identically seeded envs must not change them.
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
    """Static check: the dynamic tests only see the streams they drive, and a
    global `random.` / `np.random.` call in the env path makes the pipeline
    unseedable.
    """
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for rel in ("envs/selfplay_env.py", "envs/gym_wrapper.py"):
        text = (root / rel).read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            code = line.split("#", 1)[0]   # a comment naming the old call is fine
            if re.search(r"(?<![\w.])(random\.(choice|random|randint|sample|shuffle)"
                         r"|np\.random\.(choice|rand|randint|random|shuffle|permutation))\(",
                         code):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, "global RNG in the env path:\n" + "\n".join(offenders)


def test_both_pipelines_make_env_take_a_seed_and_use_it():
    """One seeding contract for both pipelines' make_env."""
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
    """With no seed both pipelines keep drawing fresh entropy."""
    from python_ai.envs.selfplay_env import make_env as sp_make
    from python_ai.trainers.train import make_env as p1_make

    assert sp_make()().rng.random() != sp_make()().rng.random()
    assert p1_make()()._scenario_rng.random() != p1_make()()._scenario_rng.random()


# --- the distillation trainers ---
# bc_pretrain, distill_tactics and expert_distill shuffle once per epoch on the
# global RNG, so a distillation run is reproducible only if the seed is
# applied; paired comparisons between distilled nets depend on it.

def test_the_distillation_entry_points_expose_a_seed():
    """Wiring check: each CLI takes --seed and calls seed_everything. Static,
    since proving it by behaviour means running a full distillation.
    """
    import pathlib

    import python_ai
    root = pathlib.Path(python_ai.PACKAGE_DIR) / "trainers"
    for name in ("bc_pretrain.py", "distill_tactics.py", "expert_iteration.py"):
        src = (root / name).read_text(encoding="utf-8")
        assert '"--seed"' in src, f"{name} has no --seed"
        assert "seed_everything" in src, f"{name} never applies its seed"


def test_seeding_fixes_the_global_shuffle_these_trainers_use():
    """The mechanism the wiring relies on."""
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
