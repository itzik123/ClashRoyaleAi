"""Dry-run `--policy teacher` over a recorded match, with no emulator.

`mvp_loop.py --policy teacher` needs a live emulator or a recorded frame
directory. This drives the same `TeacherPolicy` from a simulator replay JSON
instead: each second of the match becomes the GameState perception would have
reported, including the sensor's known blind spots, and the policy decides on
it exactly as it would live. What it checks, per decision:

  * every play is legal for that card at that tile (engine predicate, on a
    fresh mirror) and on a tappable row;
  * the hand, elixir and slot the policy plays all come from the frame;
  * decision latency against the 1 s period;
  * what each mirror build had to fabricate (MirrorGaps);
  * with --stall-prob, that skipped decisions are caught up on the plan clock.

The sensor model, so the gaps look like a live run's rather than a perfect
one's:

  unit HP     a damaged unit's bar is read with probability --hp-recall
              (perception measured 0.34-0.56); otherwise it reports full and
              hp_measured=False, as the live adapter does
  units       each body is missed with probability --drop
  spawned     bodies with no card id (Goblin Barrel's goblins, Golemites, ...)
              arrive as UNKNOWN_CARD_SIM_ID, as unmapped detector classes do
  towers      fraction of the engine's own maximum, always measured; a tower
              absent from the frame is destroyed

What it cannot check is the part only a live run can: real detector error,
the actuator, and the game accepting the tap.

    python perception/tools/teacher_dry_run.py [replay.json ...] [--stage 10]
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for p in (_ROOT, _ROOT.parent, _ROOT.parent / "python_ai"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from contracts import (  # noqa: E402
    UNKNOWN_CARD_SIM_ID,
    GameState,
    Phase,
    TowerObservation,
    UnitObservation,
)
from simlog import Replay  # noqa: E402
from timebase import TICKS_PER_SECOND  # noqa: E402

DEFAULT_REPLAY = _ROOT / "tests" / "assets" / "replay_ep1026.json"

# GameState field per (team, slot); slots are board coordinates, as in
# live/mirror.TOWER_SLOTS.
_TOWER_FIELDS = {(team, slot): name for name, team, slot in
                 __import__("live.mirror", fromlist=["TOWER_SLOTS"]).TOWER_SLOTS}


def replay_deck(replay: Replay) -> list[int]:
    """Our deck, as the union of every hand the replay shows. A replay does not
    record the deck itself; eight distinct cards across the hands is it."""
    seen: list[int] = []
    for t in replay.ticks:
        for c in t.ai_hand:
            if c not in seen:
                seen.append(int(c))
    if len(seen) != 8:
        raise ValueError(f"{replay.path.name}: hands show {len(seen)} distinct "
                         f"cards, not a deck of 8: {seen}")
    return seen


class SensorModel:
    """Turns one replay tick into what perception would report."""

    def __init__(self, engine, rng, hp_recall=0.5, drop=0.0):
        from python_ai.models import perception_encoder as enc  # noqa: PLC0415
        self._table = enc.card_table()
        self._rng = rng
        self.hp_recall = float(hp_recall)
        self.drop = float(drop)
        env = engine.ClashRoyaleEnv([15, 6, 25, 40, 24, 72, 33, 7],
                                    [15, 6, 25, 40, 24, 72, 33, 7], 100)
        self._tower_max = {(t, s): float(env.get_tower_max_hp(t, s))
                           for t in (0, 1) for s in range(3)}
        self._centre = float(engine.ARENA_CENTER_X) if hasattr(
            engine, "ARENA_CENTER_X") else None

    def _tower_slot(self, e) -> int:
        if e.symbol == "R":
            return 0
        centre = self._centre if self._centre is not None else 8.5
        return 1 if e.x < centre else 2

    def game_state(self, tick) -> GameState:
        units = []
        towers = {}
        for e in tick.entities:
            if e.is_tower:
                towers[(e.team, self._tower_slot(e))] = e
                continue
            if self._rng.random() < self.drop:
                continue
            card = int(e.card_id) if e.card_id >= 0 else UNKNOWN_CARD_SIM_ID
            row = self._table.get(card) if card >= 0 else None
            full = row.max_hp if row is not None and row.max_hp > 0 else None
            frac = min(1.0, e.hp / full) if full else 1.0
            measured = frac < 1.0 and self._rng.random() < self.hp_recall
            units.append(UnitObservation(
                card_sim_id=card, unit_name="", team=int(e.team),
                tile_x=int(e.x), tile_y=int(e.y),
                hp_fraction=frac if measured else 1.0,
                hp_measured=measured or frac >= 1.0))
        tower_obs = {}
        for key, name in _TOWER_FIELDS.items():
            e = towers.get(key)
            if e is None or e.hp <= 0:
                tower_obs[name] = TowerObservation(0.0, True, destroyed=True)
            else:
                tower_obs[name] = TowerObservation(
                    min(1.0, e.hp / self._tower_max[key]), True)
        return GameState(units=tuple(units), my_elixir=float(tick.elixir_ai),
                         my_hand=tuple(int(c) for c in tick.ai_hand),
                         seconds_elapsed=tick.tick / TICKS_PER_SECOND,
                         phase=Phase.SINGLE, **tower_obs)


def dry_run(path, engine, stage=None, hp_recall=0.5, drop=0.0, stall_prob=0.0,
            seed=0, verbose=False):
    from live.actuator import engine_row_is_tappable  # noqa: PLC0415
    from live.mirror import MirrorBuilder  # noqa: PLC0415
    from live.teacher_policy import TeacherPolicy  # noqa: PLC0415

    replay = Replay(path)
    deck = replay_deck(replay)
    rng = np.random.default_rng(seed)
    sensor = SensorModel(engine, rng, hp_recall=hp_recall, drop=drop)
    policy = TeacherPolicy(deck, stage=stage, seed=seed)
    checker = MirrorBuilder(deck, engine=engine)
    policy.on_screen_change(True)

    illegal, untappable, plays_by_card = [], 0, Counter()
    gaps_sum: Counter = Counter()
    wall_ms = []
    decided = 0
    for tick in replay.ticks:
        if tick.tick % 10:
            continue
        if stall_prob and rng.random() < stall_prob:
            continue                    # perception stalled: no decision now
        gs = sensor.game_state(tick)
        costs = [engine.get_card_info(c)["cost"] for c in gs.my_hand]
        ready = {i for i, c in enumerate(costs) if c <= gs.my_elixir + 1e-6}
        t0 = time.perf_counter()
        d = policy.decide(gs, ready, tick.tick / TICKS_PER_SECOND)
        wall_ms.append((time.perf_counter() - t0) * 1000.0)
        decided += 1
        g = policy.last_gaps
        gaps_sum.update(unmappable=g.unmappable_units,
                        hp_assumed_full=g.hp_assumed_full,
                        hp_ceiling_unknown=g.hp_ceiling_unknown,
                        towers_refused=len(g.towers_refused),
                        injected=g.injected)
        if d.slot is None:
            if verbose:
                print(f"  t={tick.tick:5d} hold  {d.why}")
            continue
        card = gs.my_hand[d.slot]
        plays_by_card[engine.get_card_info(card)["name"]] += 1
        env, _ = checker.build(gs)
        if not env.is_valid_placement(card, float(d.tile[0]), float(d.tile[1]), 0):
            illegal.append((tick.tick, card, d.tile))
        if not engine_row_is_tappable(d.tile[1]):
            untappable += 1
        if verbose:
            print(f"  t={tick.tick:5d} PLAY  {d.why}")

    w = np.array(wall_ms) if wall_ms else np.zeros(1)
    return {
        "replay": Path(path).name,
        "deck": deck,
        "decisions": decided,
        "stats": policy.stats,
        "illegal": illegal,
        "untappable_plays": untappable,
        "plays_by_card": dict(plays_by_card),
        "gaps_per_decision": {k: v / max(decided, 1) for k, v in gaps_sum.items()},
        "decide_ms": (float(np.median(w)), float(np.percentile(w, 95)),
                      float(w.max())),
        "drift_s": policy.drift_s,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replays", nargs="*", type=Path, default=[DEFAULT_REPLAY])
    ap.add_argument("--stage", type=int, default=None,
                    help="teacher rung; default the top rung")
    ap.add_argument("--hp-recall", type=float, default=0.5)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--stall-prob", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    import engine as engine_mod  # noqa: PLC0415
    engine = engine_mod.load()
    failed = False
    for path in args.replays:
        r = dry_run(path, engine, stage=args.stage, hp_recall=args.hp_recall,
                    drop=args.drop, stall_prob=args.stall_prob, seed=args.seed,
                    verbose=args.verbose)
        p50, p95, mx = r["decide_ms"]
        print(f"{r['replay']}: deck {r['deck']}")
        print(f"  {r['stats'].summary()}")
        print(f"  decide() p50 {p50:.1f} ms, p95 {p95:.1f} ms, max {mx:.1f} ms "
              f"(period 1000 ms); residual plan drift {r['drift_s']:+.2f} s")
        print("  plays: " + (", ".join(f"{k} {v}" for k, v in
                                       sorted(r["plays_by_card"].items()))
                             or "none"))
        print("  per build: " + ", ".join(f"{k} {v:.2f}" for k, v in
                                          sorted(r["gaps_per_decision"].items())))
        print(f"  illegal plays {len(r['illegal'])}, untappable plays "
              f"{r['untappable_plays']}")
        failed |= bool(r["illegal"]) or r["stats"].teacher_errors > 0 \
            or r["untappable_plays"] > 0 or mx > 1000.0
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
