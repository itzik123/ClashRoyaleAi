"""`_disc_offsets` is loop-invariant and pure -- it must be neither recomputed
per enemy cell nor recomputed per call.

`spell_catch_map` called `_disc_offsets(radius)` INSIDE its
`for y, x in zip(ys, xs)` loop, rebuilding the identical offset list once per
occupied enemy cell. The very next scatter loop in this same module (in
`_reach_cover`, ~line 701) hoists it correctly -- so the module already knew the
pattern; one site just did not follow it.

The module also computed `_FIREBALL_DISC = _disc_offsets(FIREBALL_RADIUS)` at
import and then NEVER READ IT: the cache was built, left unwired, and the hot
loop recomputed the same thing anyway. Memoizing the function supersedes that
module-level global, which is removed.

Caching hands the SAME object to every caller, so the return type is now a
tuple. Both call sites only iterate it, and a tuple makes accidental mutation
an error instead of a cross-caller corruption.

The output is a pure function of `radius` -- integer offsets from a hypot test
-- so memoizing cannot change any downstream number. The tests below pin that
directly rather than assuming it.
"""
import numpy as np
import pytest

from python_ai.advisors import tactics


def test_the_offsets_are_a_pure_function_of_the_radius():
    a = tactics._disc_offsets(2.5)
    b = tactics._disc_offsets(2.5)
    assert tuple(a) == tuple(b)


def test_memoization_returns_the_identical_object():
    """Cheap proof the cache is actually engaged, not just correct."""
    assert tactics._disc_offsets(2.5) is tactics._disc_offsets(2.5)


def test_the_result_is_immutable():
    """It is shared now, so it must not be mutable by one caller."""
    offs = tactics._disc_offsets(2.5)
    assert isinstance(offs, tuple)
    with pytest.raises((AttributeError, TypeError)):
        offs.append((0, 0))


def test_distinct_radii_do_not_collide_in_the_cache():
    small = tactics._disc_offsets(1.0)
    big = tactics._disc_offsets(4.0)
    assert len(big) > len(small)
    assert set(small).issubset(set(big))


@pytest.mark.parametrize("radius", [0.5, 1.0, 2.5, 3.0, 4.5, 7.0])
def test_the_geometry_is_unchanged(radius):
    """Recompute the original definition inline and compare exactly. This is
    what makes the optimization safe to believe."""
    r = int(np.ceil(radius))
    expected = [(dy, dx)
                for dy in range(-r, r + 1)
                for dx in range(-r, r + 1)
                if np.hypot(dx + 0.5, dy + 0.5) <= radius]
    assert list(tactics._disc_offsets(radius)) == expected


def test_the_dead_module_level_cache_is_gone():
    """It was assigned at import and never read once. Leaving it would keep a
    second, unmemoized copy of the same computation in the module."""
    assert not hasattr(tactics, "_FIREBALL_DISC")


def test_spell_catch_map_calls_disc_offsets_once_per_call(monkeypatch):
    """THE regression test: the call must not be inside the per-cell loop.

    Driven with several occupied enemy cells, so a per-cell call is clearly
    distinguishable from a hoisted one.
    """
    import clash_royale_env
    from python_ai.envs.gym_wrapper import DEFAULT_DECK

    deck = list(DEFAULT_DECK)
    env = clash_royale_env.ClashRoyaleEnv(deck, deck, 3600)
    env.reset()
    for cid, x, y in ((15, 4, 22), (6, 9, 24), (24, 13, 21), (24, 5, 25)):
        try:
            env.inject(cid, float(x), float(y), 1, -1.0, 0)
        except TypeError:
            env.inject(cid, float(x), float(y), 1)
    for _ in range(3):
        env.step(0, 0, 0)

    obs = np.asarray(env.get_observation_for_team(0), dtype=np.float32)

    calls = []
    real = tactics._disc_offsets

    def counting(radius):
        calls.append(radius)
        return real(radius)

    monkeypatch.setattr(tactics, "_disc_offsets", counting)
    out = tactics.spell_catch_map(obs)

    occupied = int(np.count_nonzero(
        np.minimum(tactics.enemy_hp_map(obs),
                   tactics.FIREBALL_DAMAGE * np.maximum(
                       1.0, tactics.spatial(obs)[tactics.CH_ENEMY_COUNT]
                       * tactics.MAX_CELL_UNITS))))
    if occupied < 2:
        pytest.skip("need >=2 occupied enemy cells to tell the shapes apart")

    assert len(calls) <= 1, (
        f"_disc_offsets was called {len(calls)} times for {occupied} occupied "
        "cells -- it is still inside the scatter loop")
    assert out.shape == (tactics.BOARD_H, tactics.BOARD_W)


def test_spell_catch_map_output_is_unchanged_by_the_hoist():
    """Compare against an independent reimplementation of the original
    per-cell-call shape, so the optimization is proven value-preserving."""
    import clash_royale_env
    from python_ai.envs.gym_wrapper import DEFAULT_DECK

    deck = list(DEFAULT_DECK)
    env = clash_royale_env.ClashRoyaleEnv(deck, deck, 3600)
    env.reset()
    for cid, x, y in ((15, 4, 22), (6, 9, 24), (24, 13, 21)):
        try:
            env.inject(cid, float(x), float(y), 1, -1.0, 0)
        except TypeError:
            env.inject(cid, float(x), float(y), 1)
    for _ in range(3):
        env.step(0, 0, 0)
    obs = np.asarray(env.get_observation_for_team(0), dtype=np.float32)

    got = tactics.spell_catch_map(obs)

    # the original shape, recomputing offsets per cell
    hp = tactics.enemy_hp_map(obs)
    count = np.maximum(
        1.0, tactics.spatial(obs)[tactics.CH_ENEMY_COUNT] * tactics.MAX_CELL_UNITS)
    effective = np.minimum(hp, tactics.FIREBALL_DAMAGE * count)
    want = np.zeros((tactics.BOARD_H, tactics.BOARD_W), dtype=np.float32)
    ys, xs = np.nonzero(effective)
    for y, x in zip(ys, xs):
        v = effective[y, x]
        r = int(np.ceil(tactics.FIREBALL_RADIUS))
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if np.hypot(dx + 0.5, dy + 0.5) > tactics.FIREBALL_RADIUS:
                    continue
                ay, ax = y - dy, x - dx
                if 0 <= ay < tactics.BOARD_H and 0 <= ax < tactics.BOARD_W:
                    want[ay, ax] += v

    assert np.array_equal(got, want)
