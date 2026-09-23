"""Reader for the simulator's own replay JSON.

GameLogger::save() writes per-tick entity positions, HP, elixir and both hands,
and python_ai/rl/replay.py's annotator adds four fields per tick:

    "actionCardId", "actionCardName", "actionX", "actionY"

That is a labelled placement stream (card, tile, tick): exactly what
PlacementEvent is a contract for. So the back half of this pipeline (cycle
tracking, deck discovery, opponent elixir, the simulator bridge) can be built
and tested against real ground truth with no video or vision.

Read-only. Nothing here writes to python_ai/.

Two properties of the format:

1. The action fields are stamped on every tick of the decision window (one
   decision per skip_frames=10 block, repeated across it). iter_placements()
   collapses each run to its first tick, the one the engine applied the play on
   (ClashEnv::step plays only when i == 0).
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
        """Tower HP keyed by (team, x, y): position is the only key vision can
        also produce, and this is the reference implementation for the
        divergence metric.
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

    # Only the agent's own decisions are logged. The heuristic opponent
    # (HeuristicOpponent::act, inside ClashEnv::step) has no logging hook, so
    # its plays appear only as entities materialising; see
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
        """Whether this replay carries the training annotations. A replay from
        ClashEnv::saveLog directly (main.cpp, a test) has only GameLogger's
        fields, and placement-derived tests must skip rather than pass
        vacuously on zero placements.
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

        The only route: the opponent's decisions are never logged. Entity ids
        are allocated monotonically (Board::allocateId) and never reused, so
        the first tick an id is seen is the tick its card was played.

        Approximate, and used only to exercise downstream code with a two-sided
        stream:
          * multi-unit cards appear as several entities at once and are collapsed by
            (tick, cardId) to one placement at the squad centroid;
          * units spawned by other units (Witch's skeletons, Golemites) look like
            plays; `deck` filters them when the caller knows the opponent's deck.

        Projectiles are excluded unconditionally: GameLogger logs everything
        alive, and every tower arrow is a fresh entity. They carry cardId -1.
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
                    continue  # projectile / unregistered; see docstring
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
        """The eight card ids that team cycles through: the union of every card
        seen in hand, complete once the deck has cycled (within the first
        minute in practice).
        """
        seen: set[int] = set()
        for tick in self.ticks:
            seen.update(tick.ai_hand if team == 0 else tick.opp_hand)
        return tuple(sorted(seen))


# Replays copied into perception/ so tests have a stable input: the training
# run rewrites and prunes python_ai/replays/ continuously.
FIXTURE_DIR = Path(__file__).resolve().parent / "tests" / "assets"


def available_replays(include_live: bool = False) -> tuple[Path, ...]:
    """Replays available to work with: the frozen fixtures by default.
    `include_live` adds whatever the training run has on disk, for ad hoc
    analysis, never tests.
    """
    paths = list(FIXTURE_DIR.glob("*.json")) if FIXTURE_DIR.is_dir() else []
    if include_live and REPLAY_DIR.is_dir():
        paths.extend(REPLAY_DIR.glob("*.json"))
    return tuple(sorted(paths))
