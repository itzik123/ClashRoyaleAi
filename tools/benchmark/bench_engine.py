"""Engine speed and determinism benchmark, through the public Python bindings.

Reproduces docs/RESEARCH.md, Table 2.6 and the determinism/fork checks of §2.5.

Protocol (docs/RESEARCH.md, Appendix B.3):
  * N seeded matches (default 20, seeds 1000..1019), 2.6 Hog Cycle both sides.
    Each side, each decision: with p=0.5 a random AFFORDABLE card at a random
    LEGAL cell, otherwise wait. The action stream is seeded, so a seed is the
    same workload on every repeat.
  * Each match is replayed R times (default 7) with the two step functions
    interleaved and their order alternated. step_self_play_fast advances the
    engine without building observations; step_self_play also builds both
    teams' observations (not converted to Python unless read).
  * Reported: the per-match MINIMUM over repeats (the engine is deterministic,
    so noise is additive), then the MEDIAN across matches.
  * Micro-costs (snapshot, seed, reset, observation) are min/median of 300 calls.

Absolute times depend on clock state (thermal, power plan, battery). Quote the
fast/full RATIO with confidence and absolute figures as a range.

Run with the training venv (the engine loads on Python 3.11 only):

    python_ai/venv/Scripts/python.exe tools/benchmark/bench_engine.py
    python_ai/venv/Scripts/python.exe tools/benchmark/bench_engine.py --seeds 5 --reps 3 --json out.json

Read-only: builds engines in memory and writes nothing unless --json is given.
"""
import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import struct
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
import python_ai  # noqa: E402,F401  (puts the compiled engine on sys.path)
import clash_royale_env as cr  # noqa: E402

DECK = [15, 6, 25, 40, 24, 72, 33, 7]          # 2.6 Hog Cycle
MAX_TICKS = 3600
SKIP = 10
TICKS_PER_SECOND = 10                          # docs/RESEARCH.md §2.2
H = cr.ClashRoyaleEnv.BOARD_HEIGHT
W = cr.ClashRoyaleEnv.BOARD_WIDTH
COST = {c: cr.get_card_info(c)["cost"] for c in DECK}


def _legal_cells():
    """Legal cells per (card, team), in the team's OWN frame: step_self_play
    takes team 1's target mirrored in y, while is_valid_placement takes absolute
    y for both teams."""
    probe = cr.ClashRoyaleEnv(DECK, DECK, MAX_TICKS)
    probe.seed(0)
    out = {}
    for c in DECK:
        for team in (0, 1):
            cells = []
            for y in range(H):
                for x in range(W):
                    by = y if team == 0 else (H - 1) - y
                    if probe.is_valid_placement(c, float(x), float(by), team):
                        cells.append((float(x), float(y)))
            out[(c, team)] = cells
    return out


LEGAL = _legal_cells()


def pick(env, team, rng):
    """A random affordable card at a random legal cell with p=0.5, else wait."""
    if rng.random() < 0.5:
        return 4, 0.0, 0.0
    hand = env.get_hand_for_team(team)
    elixir = env.get_elixir_for_team(team)
    slots = [i for i, c in enumerate(hand) if COST.get(c, 99) <= elixir + 1e-6]
    if not slots:
        return 4, 0.0, 0.0
    i = rng.choice(slots)
    x, y = rng.choice(LEGAL[(hand[i], team)])
    return i, x, y


def play_match(seed, fast=True):
    """One seeded match. Returns (ticks, seconds spent inside engine step calls,
    final-state signature)."""
    env = cr.ClashRoyaleEnv(DECK, DECK, MAX_TICKS)
    env.seed(seed)
    rng = random.Random(seed)
    step = env.step_self_play_fast if fast else env.step_self_play
    t_step, done = 0.0, False
    while not done:
        a0, a1 = pick(env, 0, rng), pick(env, 1, rng)
        s = time.perf_counter()
        r = step(*a0, *a1, SKIP)
        t_step += time.perf_counter() - s
        done = r.done
    ticks = env.get_current_tick()
    sig = (ticks, tuple(env.get_tower_hp(t, s) for t in (0, 1) for s in (0, 1, 2)),
           env.get_elixir_spent(0), env.get_elixir_spent(1))
    return ticks, t_step, sig


def obs_hash(obs):
    return hashlib.sha256(struct.pack(f"{len(obs)}f", *obs)).hexdigest()


def time_calls(fn, n=300):
    ts = []
    for _ in range(n):
        s = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - s)
    return min(ts) * 1e3, statistics.median(ts) * 1e3


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--json", default=None, help="write the results here")
    args = ap.parse_args()

    res = {"python": sys.version.split()[0], "platform": platform.platform(),
           "processor": platform.processor(), "seeds": args.seeds, "reps": args.reps,
           "seed_base": args.seed_base}
    print(f"python {res['python']}  {res['platform']}")

    # Warm-up, both arms.
    play_match(999, True)
    play_match(999, False)

    per = {"fast": {}, "full": {}}
    ticks = {}
    for r in range(args.reps):
        for i in range(args.seeds):
            seed = args.seed_base + i
            order = [("fast", True), ("full", False)]
            if (r + i) % 2:
                order.reverse()
            for name, fast in order:
                t, t_step, _ = play_match(seed, fast)
                per[name].setdefault(seed, []).append(t_step / t * 1e6)
                ticks[seed] = t

    mean_ticks = statistics.mean(ticks.values())
    for name in ("fast", "full"):
        mins = [min(v) for v in per[name].values()]
        us = statistics.median(mins)
        res[f"{name}_us_per_tick"] = us
        res[f"{name}_us_per_tick_range_of_match_minima"] = [min(mins), max(mins)]
        res[f"{name}_ms_per_match"] = us * mean_ticks / 1e3
        res[f"{name}_realtime_factor"] = (1e6 / TICKS_PER_SECOND) / us
    res["mean_ticks_per_match"] = mean_ticks
    res["full_over_fast"] = res["full_us_per_tick"] / res["fast_us_per_tick"]

    print(f"\n{args.seeds} seeded random-play matches, per-match minimum of {args.reps} "
          f"interleaved repeats, median across matches (mean {mean_ticks:.0f} ticks/match):")
    for name, label in (("fast", "step_self_play_fast (no observations)"),
                        ("full", "step_self_play (both observations)")):
        print(f"  {label:42s} {res[name + '_us_per_tick']:6.2f} us/tick   "
              f"{res[name + '_ms_per_match']:6.2f} ms/match   "
              f"{res[name + '_realtime_factor']:8,.0f}x real time")
    print(f"  full / fast = {res['full_over_fast']:.3f}")

    # Determinism: same seed twice; a different seed must differ.
    a, b, c = play_match(4242)[2], play_match(4242)[2], play_match(4243)[2]
    res["determinism_same_seed_identical"] = a == b
    res["determinism_other_seed_differs"] = a != c
    print(f"\ndeterminism: same seed identical {a == b}; different seed differs {a != c}")

    # Snapshot costs, bit-identity and independence at a mid-match state.
    env = cr.ClashRoyaleEnv(DECK, DECK, MAX_TICKS)
    env.seed(7)
    rng = random.Random(7)
    for _ in range(60):
        env.step_self_play_fast(*pick(env, 0, rng), *pick(env, 1, rng), SKIP)
    res["snapshot_tick"] = env.get_current_tick()
    res["snapshot_ms"] = time_calls(env.snapshot)

    def fork_step():
        f = env.snapshot()
        s = time.perf_counter()
        f.step_self_play_fast(4, 0.0, 0.0, 4, 0.0, 0.0, SKIP)
        return time.perf_counter() - s
    fs = [fork_step() for _ in range(300)]
    res["fork_10_tick_step_ms"] = (min(fs) * 1e3, statistics.median(fs) * 1e3)

    fork = env.snapshot()
    r1, r2 = random.Random(11), random.Random(11)
    identical = True
    for _ in range(40):
        o1 = env.step_self_play(*pick(env, 0, r1), *pick(env, 1, r1), SKIP)
        o2 = fork.step_self_play(*pick(fork, 0, r2), *pick(fork, 1, r2), SKIP)
        if (obs_hash(o1.observation0) != obs_hash(o2.observation0)
                or obs_hash(o1.observation1) != obs_hash(o2.observation1)):
            identical = False
            break
        if o1.done:
            break
    res["fork_bit_identical_40_decisions"] = identical

    base = env.snapshot()
    before = (base.get_current_tick(),
              [base.get_tower_hp(t, s) for t in (0, 1) for s in (0, 1, 2)])
    child = base.snapshot()
    for _ in range(30):
        child.step_self_play_fast(*pick(child, 0, rng), *pick(child, 1, rng), SKIP)
    after = (base.get_current_tick(),
             [base.get_tower_hp(t, s) for t in (0, 1) for s in (0, 1, 2)])
    res["stepping_fork_leaves_parent_untouched"] = before == after

    e = cr.ClashRoyaleEnv(DECK, DECK, MAX_TICKS)
    res["reset_ms"] = time_calls(e.reset)
    seeds = iter(range(10 ** 6))
    res["seed_ms"] = time_calls(lambda: e.seed(next(seeds)))
    res["observation_to_python_ms"] = time_calls(lambda: e.get_observation_for_team(0))

    print(f"\nsnapshot at tick {res['snapshot_tick']} (min/median ms): {res['snapshot_ms'][0]:.4f} / {res['snapshot_ms'][1]:.4f}")
    print(f"10-tick step on a fork (ms):             {res['fork_10_tick_step_ms'][0]:.4f} / {res['fork_10_tick_step_ms'][1]:.4f}")
    print(f"fork bit-identical over 40 decisions:     {identical}")
    print(f"stepping a fork leaves parent untouched:  {res['stepping_fork_leaves_parent_untouched']}")
    for k in ("reset_ms", "seed_ms", "observation_to_python_ms"):
        print(f"{k:42s} {res[k][0]:.4f} / {res[k][1]:.4f}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
