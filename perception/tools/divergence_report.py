"""Stage 5's deliverable: how far the estimator drifts from ground truth.

Replays a match through SimDriver and plots predicted against actual tower HP
over time.

  --replay  Ground truth from a simulator replay JSON. No vision in the loop, so
            any divergence belongs to the bridge (injection semantics, tick
            alignment, cycle reconstruction). The control: it must be near zero
            before a vision number means anything.
  --all     Every replay in python_ai/replays, reported together.
  --json    Write the per-tick series out instead of only summarising.

Not implemented: a video mode, taking ground truth from `readers/towers.py` so
every vision error folds into the number. It needs the calibration
`readers/towers.py` raises `TowerCalibrationMissing` for.

No target is asserted: a threshold invented before the first measurement would
later be treated as a requirement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PERCEPTION_ROOT = Path(__file__).resolve().parent.parent
if str(_PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(_PERCEPTION_ROOT))

from bridge.sim_driver import SimDriver  # noqa: E402
from contracts import EventSource, PlacementEvent  # noqa: E402
from simlog import Replay, available_replays  # noqa: E402
from timebase import ticks_to_seconds  # noqa: E402
from track.opp_deck import OpponentDeckTracker  # noqa: E402
from track.opp_elixir import OpponentElixirTracker  # noqa: E402

TOWER_KEYS = ("own_princess_left", "own_princess_right",
              "opp_princess_left", "opp_princess_right")


def _observed_tower_hp(tick, geom) -> dict[str, int]:
    """Tower HP from a replay tick, keyed the same way SimDriver reports it."""
    by_pos = tick.tower_hp_by_position()
    lookup = {
        "own_king": (0, *geom.own_king),
        "own_princess_left": (0, *geom.own_princess_left),
        "own_princess_right": (0, *geom.own_princess_right),
        "opp_king": (1, *geom.opp_king),
        "opp_princess_left": (1, *geom.opp_princess_left),
        "opp_princess_right": (1, *geom.opp_princess_right),
    }
    # A destroyed tower is absent from the entity list, which is 0 HP, not
    # missing data; defaulting to 0 keeps the curve defined to the end of the
    # match.
    return {name: by_pos.get(key, 0) for name, key in lookup.items()}


def run_replay(path: Path, verbose: bool = True) -> dict:
    replay = Replay(path)
    if not replay.has_action_labels:
        raise SystemExit(
            f"{path.name} has no action labels -- it was not written by "
            "train.py, so it carries no placement ground truth."
        )

    deck0 = list(replay.starting_deck(0))
    deck1 = list(replay.starting_deck(1))
    opening = replay.ticks[0].ai_hand

    driver = SimDriver(my_deck=deck0, opp_deck=deck1)
    driver.start(my_opening_hand=opening)

    hands_by_tick = {t.tick: t.ai_hand for t in replay.ticks}
    opp_deck = OpponentDeckTracker()
    opp_elixir = OpponentElixirTracker()

    events: list[tuple[PlacementEvent, int | None]] = []
    for p in replay.iter_placements():
        after = hands_by_tick.get(p.tick + 1)
        before = hands_by_tick.get(p.tick)
        incoming = None
        if before and after and p.card_id in before:
            incoming = after[list(before).index(p.card_id)]
        events.append((
            PlacementEvent(
                tick=p.tick, wall_time_ms=ticks_to_seconds(p.tick) * 1000.0,
                card_sim_id=p.card_id, card_real_name=p.card_name, team=0,
                tile_x=int(round(p.x)), tile_y=int(round(p.y)),
                confidence=1.0, source=EventSource.REPLAY_GROUND_TRUTH,
            ),
            incoming,
        ))
    for p in replay.infer_opponent_placements(deck=tuple(deck1)):
        # Opponent y arrives raw from the replay, and PlacementEvent is in
        # mirrored convention (contracts.py).
        mirrored_y = (replay.board_height - 1) - p.y
        events.append((
            PlacementEvent(
                tick=p.tick, wall_time_ms=ticks_to_seconds(p.tick) * 1000.0,
                card_sim_id=p.card_id, card_real_name=p.card_name, team=1,
                tile_x=int(round(p.x)), tile_y=int(round(mirrored_y)),
                confidence=1.0, source=EventSource.REPLAY_GROUND_TRUTH,
            ),
            None,
        ))
    events.sort(key=lambda e: e[0].tick)

    ticks_by_index = {t.tick: t for t in replay.ticks}
    curve: list[dict] = []
    next_event = 0
    sample_every = 10  # one sample per simulated second

    for tick_no in range(0, replay.ticks[-1].tick + 1):
        while next_event < len(events) and events[next_event][0].tick <= tick_no:
            event, incoming = events[next_event]
            driver.apply(event, incoming_card=incoming)
            if event.team == 1:
                opp_deck.on_play(event.card_sim_id, event.tick)
                opp_elixir.advance_to(event.tick)
                opp_elixir.on_play(driver.card_cost(event.card_sim_id), event.tick)
            next_event += 1

        driver.advance_to(tick_no)

        if tick_no % sample_every == 0 and tick_no in ticks_by_index:
            observed = _observed_tower_hp(ticks_by_index[tick_no], driver.geom)
            predicted = driver.tower_hp()
            curve.append({
                "tick": tick_no,
                "seconds": round(ticks_to_seconds(tick_no), 1),
                "divergence": round(driver.divergence(observed), 1),
                "predicted": predicted,
                "observed": observed,
            })

    report = driver.report.as_dict()
    report["replay"] = path.name
    report["ticks"] = replay.ticks[-1].tick
    report["opp_deck_known"] = sorted(opp_deck.deck)
    report["opp_deck_complete_tick"] = opp_deck.complete_tick
    report["opp_elixir_final"] = round(opp_elixir.value, 2)
    report["opp_elixir_flags"] = list(opp_elixir.flags())
    report["curve"] = curve

    if curve:
        finals = curve[-1]
        report["final_divergence"] = finals["divergence"]
        report["max_divergence"] = max(c["divergence"] for c in curve)

    if verbose:
        _print(report)
    return report


def _print(report: dict) -> None:
    print(f"\n=== {report['replay']} ===")
    print(f"  ticks simulated          {report['ticks']}")
    print(f"  candidates initial       {report['candidates_initial']}")
    print(f"  cycle identified after   {report['cycle_identified_after']} own plays")
    print(f"  own plays applied        {report['own_plays_applied']}/{report['own_plays_attempted']}")
    print(f"  own plays engine-refused {report['own_plays_rejected_by_engine']}")
    print(f"  opponent injections      {report['opp_plays_injected']}")
    print(f"  unmapped skipped         {report['unmapped_skipped']}")
    print(f"  opponent deck found      {report['opp_deck_known']} @ tick {report['opp_deck_complete_tick']}")
    print(f"  opponent elixir final    {report['opp_elixir_final']}  {report['opp_elixir_flags']}")
    print()

    curve = report["curve"]
    if not curve:
        return
    print(f"  final divergence  {report['final_divergence']:.1f} HP/tower")
    print(f"  max divergence    {report['max_divergence']:.1f} HP/tower")
    print()
    print("   time    diverg   own PL   own PR  |  opp PL   opp PR   (pred/obs)")
    step = max(1, len(curve) // 24)
    for row in curve[::step]:
        p, o = row["predicted"], row["observed"]
        print(
            f"  {row['seconds']:6.1f}  {row['divergence']:7.1f}"
            f"  {p['own_princess_left']:5d}/{o['own_princess_left']:<5d}"
            f" {p['own_princess_right']:5d}/{o['own_princess_right']:<5d}"
            f" | {p['opp_princess_left']:5d}/{o['opp_princess_left']:<5d}"
            f" {p['opp_princess_right']:5d}/{o['opp_princess_right']:<5d}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    paths = (
        [args.replay] if args.replay
        else list(available_replays()) if args.all
        else list(available_replays())[:1]
    )
    if not paths:
        raise SystemExit("no replays available -- see perception/simlog.py")

    reports = [run_replay(p) for p in paths]
    if args.json:
        args.json.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
