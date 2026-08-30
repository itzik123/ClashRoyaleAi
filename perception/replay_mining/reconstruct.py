"""Replay an event stream through our engine and sample the resulting board.

Placements are `inject`ed rather than played from hand. That is deliberate: we
are reproducing WHAT a human did, not re-deriving whether they could afford it.
The economy is already baked into the timing of the events, and routing through
the hand would need the shuffled opening hand to match a real player's, which it
cannot.

One consequence worth knowing: `inject` bypasses `playCard`, so the cycle
observation blocks stay empty in a reconstruction. That is irrelevant to
divergence but would matter to anything that later mined observations for BC.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import clash_royale_env as E

TICKS_PER_SECOND = 10
NOOP = 4  # card_head index HAND_SIZE == no-op


@dataclass
class Sample:
    t: float
    units: tuple[int, int]           # (ego, opponent) living body count
    centroid: tuple[tuple[float, float] | None, tuple[float, float] | None]
    tower_hp: tuple[list[int], list[int]]


@dataclass
class Reconstruction:
    samples: list[Sample] = field(default_factory=list)
    injected: dict[str, int] = field(default_factory=dict)
    towers_alive: tuple[int, int] = (3, 3)
    ended_early: bool = False
    end_t: float = 0.0


def _read_board(env) -> tuple[tuple[int, int], tuple, tuple]:
    H, W, C = env.BOARD_HEIGHT, env.BOARD_WIDTH, env.NUM_CHANNELS
    obs = np.asarray(env.get_observation_for_team(0), dtype=np.float32)
    grid = obs[: C * H * W].reshape(C, H, W)
    own, enemy = grid[env.CH_COUNT], grid[env.CH_COUNT + 1]
    counts, centroids = [], []
    for ch in (own, enemy):
        n = float(ch.sum()) * 5.0          # channel is count/5
        ys, xs = np.nonzero(ch > 1e-6)
        centroids.append((float(xs.mean()), float(ys.mean())) if len(xs) else None)
        counts.append(int(round(n)))
    hp = ([env.get_tower_hp(0, s) for s in range(3)],
          [env.get_tower_hp(1, s) for s in range(3)])
    return (counts[0], counts[1]), tuple(centroids), hp


def reconstruct(resolved_events, sample_times, deck, seed=0,
                max_seconds=320.0) -> Reconstruction:
    env = E.ClashRoyaleEnv(list(deck), list(deck),
                           max_ticks=int(max_seconds * TICKS_PER_SECOND) + 100)
    env.seed(seed)
    env.reset()

    out = Reconstruction()
    plan: list[tuple[int, str, object]] = []
    for ev, cid in resolved_events:
        plan.append((int(round(ev.t * TICKS_PER_SECOND)), "inject", (ev, cid)))
    for t in sample_times:
        plan.append((int(round(t * TICKS_PER_SECOND)), "sample", t))
    plan.sort(key=lambda p: (p[0], p[1] != "inject"))   # inject before sampling

    tick = 0
    for target, kind, payload in plan:
        while tick < target:
            if env.is_game_over():
                break
            step = min(10, target - tick)
            env.step_self_play_fast(NOOP, 0.0, 0.0, NOOP, 0.0, 0.0, skip_frames=step)
            tick += step
        if env.is_game_over():
            out.ended_early = True
            out.end_t = tick / TICKS_PER_SECOND
            break
        if kind == "inject":
            ev, cid = payload
            x = float(np.clip(ev.x, 0.0, E.ARENA_WIDTH - 1.0))
            y = float(np.clip(ev.y, 0.0, E.ARENA_HEIGHT - 1.0))
            env.inject(cid, x, y, ev.team)
            out.injected[ev.card] = out.injected.get(ev.card, 0) + 1
        else:
            units, centroid, hp = _read_board(env)
            out.samples.append(Sample(payload, units, centroid, hp))

    out.towers_alive = (env.get_towers_alive(0), env.get_towers_alive(1))
    if not out.ended_early:
        out.end_t = tick / TICKS_PER_SECOND
    return out


def default_deck() -> list[int]:
    """`DEFAULT_DECK`, read from its source rather than restated here.

    Importing `gym_wrapper` would pull in gymnasium, which perception's venv
    does not have; copying the eight ids would be the "second copy of an engine
    constant" this project forbids. Parsing the assignment is neither.
    """
    import ast
    from pathlib import Path

    import python_ai

    src = Path(python_ai.PACKAGE_DIR) / "envs" / "gym_wrapper.py"
    for node in ast.parse(src.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "DEFAULT_DECK" for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise RuntimeError(f"DEFAULT_DECK not found in {src}")
