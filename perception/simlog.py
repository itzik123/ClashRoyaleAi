"""Reader for the simulator's own replay JSON.

WHY THIS EXISTS AND WHY IT IS WORTH MORE THAN IT LOOKS
------------------------------------------------------
python_ai/replays/ holds match logs the simulator produced during training.
GameLogger::save() writes per-tick entity positions, HP, elixir and both
hands -- and python_ai/rl/replay.py's annotator adds four more fields per tick that
GameLogger itself never writes:

    "actionCardId", "actionCardName", "actionX", "actionY"

That is a labelled placement stream: which card, which tile, which tick. It
is, field for field, the thing PlacementEvent is a contract for.

Which means the entire back half of this pipeline -- cycle tracking, deck
discovery, opponent elixir derivation, and the simulator bridge -- can be
built and tested to completion against real ground truth with no video, no
calibration, and no vision at all. Vision only has to reproduce a stream this
module can already produce perfectly, and every consumer of that stream can
be proven correct before the first frame is ever read.

Read-only. Nothing here writes to python_ai/.

TWO PROPERTIES OF THE FORMAT THAT WILL BITE
-------------------------------------------
1. The action fields are stamped on EVERY tick of the decision window, not
   just the tick the card was played on. train.py writes one decision per
   skip_frames=10 tick block and repeats it across the block, so reading them
   naively yields ten copies of every placement. iter_placements() collapses
   each run to its first tick, which is the tick the engine actually applied
   the play on (ClashEnv::step only plays when i == 0).

2. actionCardId is -1 for the no-op action, which is most ticks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

REPLAY_DIR = Path(__file__).resolve().parent.parent / "python_ai" / "replays"

# GameManager.h: negative sentinel ids for towers, which are built directly
# rather than through CardRegistry.
TOWER_KING_ID = -2
TOWER_PRINCESS_ID = -3


@dataclass(frozen=True)
class ReplayEntity:
    id: int
    x: float
    y: float
    hp: int
    team: int
    symbol: str
    is_flying: bool
    card_id: int

    @property
    def is_tower(self) -> bool:
        return self.card_id in (TOWER_KING_ID, TOWER_PRINCESS_ID)


@dataclass(frozen=True)
class ReplayTick:
    tick: int
    elixir_ai: float
    elixir_opp: float
    ai_hand: tuple[int, ...]
    opp_hand: tuple[int, ...]
    entities: tuple[ReplayEntity, ...]
    action_card_id: int | None
    action_x: float | None
    action_y: float | None

    def towers(self, team: int) -> tuple[ReplayEntity, ...]:
        return tuple(e for e in self.entities if e.is_tower and e.team == team)

    def tower_hp_by_position(self) -> dict[tuple[int, float, float], int]:
        """Tower HP keyed by (team, x, y).

        Position-keyed rather than id-keyed because that is the only key
        vision can also produce -- entity ids exist solely inside the
        simulator. Keeping the two comparable is the whole point of this
        being the reference implementation for the divergence metric.
        """
        return {(e.team, e.x, e.y): e.hp for e in self.entities if e.is_tower}


@dataclass(frozen=True)
class ReplayPlacement:
    """One card play, deduplicated to the tick the engine applied it on."""

    tick: int
    card_id: int
    card_name: str
    x: float
    y: float
    team: int = 0
    """Always 0. train.py only logs the learner's own action; the built-in
    opponent's plays are not recorded anywhere in the replay. See
    Replay.opponent_placements_are_unlogged."""


class Replay:
    """One simulator replay JSON, parsed."""

    # train.py logs only the agent's own decisions. The heuristic opponent
    # (HeuristicOpponent::act, driven from inside ClashEnv::step) places cards
    # with no logging hook at all, so its plays appear in the replay ONLY as
    # entities materialising on the board. Tests that need an opponent
    # placement stream must recover it from entity appearances -- see
    # infer_opponent_placements().
    opponent_placements_are_unlogged = True

    def __init__(self, path: str | Path):
        self.path = Path(path)
        body = json.loads(self.path.read_text(encoding="utf-8"))

        self.board_width: int = body["boardWidth"]
        self.board_height: int = body["boardHeight"]
        self.card_names: dict[int, str] = {
            int(k): v for k, v in body.get("cardNames", {}).items()
        }
        self.card_meta: dict[int, dict] = {
            int(k): v for k, v in body.get("cardMeta", {}).items()
        }

        self.ticks: tuple[ReplayTick, ...] = tuple(
            ReplayTick(
                tick=t["tick"],
                elixir_ai=t["elixirAI"],
                elixir_opp=t["elixirOpp"],
                ai_hand=tuple(t.get("aiHand", ())),
                opp_hand=tuple(t.get("oppHand", ())),
                entities=tuple(
                    ReplayEntity(
                        id=e["id"], x=e["x"], y=e["y"], hp=e["hp"], team=e["team"],
                        symbol=e["symbol"], is_flying=e["isFlying"], card_id=e["cardId"],
                    )
                    for e in t.get("entities", ())
                ),
                action_card_id=t.get("actionCardId"),
                action_x=t.get("actionX"),
                action_y=t.get("actionY"),
            )
            for t in body["ticks"]
        )

    def __len__(self) -> int:
        return len(self.ticks)

    @property
    def has_action_labels(self) -> bool:
        """Whether this replay carries train.py's action annotations.

        A replay written by anything that calls ClashEnv::saveLog directly
        (main.cpp, a test) has only GameLogger's own fields, and every
        placement-derived test has to skip rather than silently see zero
        placements and pass vacuously.
        """
        return any(t.action_card_id is not None for t in self.ticks)

    def iter_placements(self) -> Iterator[ReplayPlacement]:
        """Own-team placements, one per actual play. See property 1 above."""
        previous: tuple | None = None
        for tick in self.ticks:
            card_id = tick.action_card_id
            if card_id is None or card_id < 0:
                previous = None
                continue
            current = (card_id, tick.action_x, tick.action_y)
            if current != previous:
                yield ReplayPlacement(
                    tick=tick.tick,
                    card_id=card_id,
                    card_name=self.card_names.get(card_id, f"card{card_id}"),
                    x=float(tick.action_x),
                    y=float(tick.action_y),
                )
            previous = current

    def infer_opponent_placements(
        self, deck: tuple[int, ...] | None = None
    ) -> Iterator[ReplayPlacement]:
        """Opponent plays, recovered from entities appearing on the board.

        The only route available -- the opponent's decisions are never
        logged (see opponent_placements_are_unlogged). An entity id is
        allocated monotonically by Board::allocateId and never reused, so the
        first tick an id is seen is the tick its card was played.

        Deliberately approximate, and only used to exercise downstream code
        with a realistic two-sided event stream:
          * multi-unit cards (Goblins, Barbarians) appear as several entities
            at once and are collapsed by (tick, cardId) to one placement at
            the squad centroid, which is where the card was aimed;
          * units spawned by other units (Witch's skeletons, a Golem's
            Golemites) are indistinguishable from a genuine play here, so
            they show up as phantom placements. `deck` filters those out when
            the caller knows what the opponent is actually playing.

        Projectiles are excluded unconditionally. Every arrow a tower fires is
        a fresh Entity with a fresh id, and GameLogger records it -- it logs
        everything alive, not everything targetable -- so without this filter
        an ordinary match yields dozens of "placements" at the tower
        positions. They carry cardId -1 (no CardRegistry entry), which is
        what distinguishes them.
        """
        allowed = set(deck) if deck is not None else None
        seen: set[int] = set()
        for tick in self.ticks:
            fresh: dict[int, list[ReplayEntity]] = {}
            for entity in tick.entities:
                if entity.id in seen or entity.team != 1 or entity.is_tower:
                    continue
                seen.add(entity.id)
                if entity.card_id < 0:
                    continue  # projectile / unregistered -- see docstring
                if allowed is not None and entity.card_id not in allowed:
                    continue  # spawned by another unit, not played
                fresh.setdefault(entity.card_id, []).append(entity)
            for card_id, group in sorted(fresh.items()):
                yield ReplayPlacement(
                    tick=tick.tick,
                    card_id=card_id,
                    card_name=self.card_names.get(card_id, f"card{card_id}"),
                    x=sum(e.x for e in group) / len(group),
                    y=sum(e.y for e in group) / len(group),
                    team=1,
                )

    def starting_deck(self, team: int) -> tuple[int, ...]:
        """The eight card ids that team cycles through.

        Recovered as the union of every card seen in hand across the match,
        which is complete as soon as the whole deck has cycled once -- the
        same condition the opponent-deck tracker works under against real
        vision, and reached within the first minute in practice.
        """
        seen: set[int] = set()
        for tick in self.ticks:
            seen.update(tick.ai_hand if team == 0 else tick.opp_hand)
        return tuple(sorted(seen))


# Replays copied into perception/ so tests have a stable input. The live
# directory is NOT stable: the training run rewrites it continuously and
# prunes old files (observed dropping from eight replays to one within
# minutes), so a test bound to python_ai/replays/ passes or fails depending
# on when it happens to run.
FIXTURE_DIR = Path(__file__).resolve().parent / "tests" / "assets"


def available_replays(include_live: bool = False) -> tuple[Path, ...]:
    """Replays available to work with.

    Defaults to the frozen fixtures. `include_live` also pulls in whatever
    the training run currently has on disk -- useful for ad hoc analysis over
    more matches, never for tests.
    """
    paths = list(FIXTURE_DIR.glob("*.json")) if FIXTURE_DIR.is_dir() else []
    if include_live and REPLAY_DIR.is_dir():
        paths.extend(REPLAY_DIR.glob("*.json"))
    return tuple(sorted(paths))
