"""Did the card we tapped actually reach the board?

`ElixirLedger` answers "did a drop appear that our pending costs explain",
which conflates three events: the game refused the tap, the tap never arrived,
or the ledger could not reconcile the elixir trace. The third is real: it
matches drops by subset-sum over pending costs against an integer bar, with the
opponent also spending.

This uses two oracles that do not share the ledger's assumption, both read off
the `GameState` the loop already has:

    unit    an own-team unit of the type this card spawns appeared that was
            not there when we issued the tap
    hand    the card in the tapped slot changed (a played card is replaced by
            the next in the cycle)

`unit` cannot see a spell and `hand` depends on the weakest reader, but they
fail for unrelated reasons, so the 2x2 against the ledger separates a refusal
from a lost metric:

    |                  | ledger confirmed | ledger rejected      |
    |------------------|------------------|----------------------|
    | unit appeared    | healthy          | LEDGER METRIC BUG    |
    | no unit          | impossible-ish   | THE GAME REFUSED IT  |

Only the bottom-right cell is a placement problem. Everything landing there is
logged with its card, tile and the exact pixel tapped: ground truth for the
real legality rule, which `is_valid_placement` cannot provide.

Counts, not positions: every unit moves between two observations a second
apart, so matching by tile reports movers as new. Counting units of the
expected type and asking whether the count rose also handles multi-body cards
(Minions, Archers).
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from contracts import UNKNOWN_CARD_SIM_ID

# How long an issued placement is watched before its verdict is final. A
# legitimate confirmation lands ~1.6 s late (~900 ms tap, ~700 ms until the
# board is seen); 3.5 s allows a slow producer frame while keeping two
# placements of the same card from overlapping at 1 Hz.
CONFIRM_WINDOW_S = 3.5


def expected_unit_names(card_name: str) -> frozenset[str]:
    """Detector unit classes this card puts on the board; empty for a spell. From
    CRBAB's own card->units table, which follows upstream retraining.
    """
    import dataclasses  # noqa: PLC0415

    from clashroyalebuildabot.namespaces.cards import Cards  # noqa: PLC0415

    for f in dataclasses.fields(Cards):
        card = getattr(Cards, f.name)
        if card.name == card_name:
            return frozenset(u.name for u in card.units)
    return frozenset()


@dataclass
class Issued:
    """One placement we asked the game for, and what became of it."""

    seq: int
    t_issue: float
    slot: int
    card_name: str
    card_sim_id: int
    tile: tuple[int, int]
    tap: tuple[int, int]
    elixir_seen: float
    unconfirmed_owed: float
    expect: frozenset[str]
    baseline_count: int
    hand_at_issue: tuple[int, ...]

    best_delta: int = 0
    hand_changed: bool = False
    observations: int = 0
    closed: bool = False
    ledger_confirmed: bool | None = None

    @property
    def is_spell(self) -> bool:
        return not self.expect

    @property
    def unit_confirmed(self) -> bool:
        """A spell is never unit-confirmable, so it reports via `is_spell` and is
        excluded from the 2x2; counting it would score every Fireball as "the
        game refused it".
        """
        return self.best_delta >= 1

    def as_row(self) -> dict:
        return {
            "seq": self.seq, "t": round(self.t_issue, 2), "slot": self.slot,
            "card": self.card_name, "sim_id": self.card_sim_id,
            "tile_x": self.tile[0], "tile_y": self.tile[1],
            "tap_x": self.tap[0], "tap_y": self.tap[1],
            "elixir_seen": self.elixir_seen,
            "unconfirmed_owed": self.unconfirmed_owed,
            "spell": self.is_spell,
            "unit_delta": self.best_delta,
            "unit_confirmed": self.unit_confirmed,
            "hand_changed": self.hand_changed,
            "ledger_confirmed": self.ledger_confirmed,
            "observations": self.observations,
            "hand_at_issue": list(self.hand_at_issue),
        }


@dataclass
class PlacementConfirmer:
    """Watches issued placements for evidence they reached the board."""

    window_s: float = CONFIRM_WINDOW_S
    issued: list[Issued] = field(default_factory=list)
    _open: list[Issued] = field(default_factory=list)
    _seq: int = 0

    def issue(self, gs, *, slot: int, card_name: str, card_sim_id: int,
              tile: tuple[int, int], tap: tuple[int, int], now: float,
              owed: float = 0.0) -> Issued:
        expect = expected_unit_names(card_name)
        rec = Issued(
            seq=self._seq, t_issue=now, slot=slot, card_name=card_name,
            card_sim_id=card_sim_id, tile=tile, tap=tap,
            elixir_seen=float(getattr(gs, "my_elixir", 0.0)),
            unconfirmed_owed=float(owed),
            expect=expect,
            baseline_count=self._count(gs, expect),
            hand_at_issue=self._hand_ids(gs),
        )
        self._seq += 1
        self.issued.append(rec)
        self._open.append(rec)
        return rec

    def observe(self, gs, now: float) -> None:
        """Fold one perceived board into every still-open placement.

        Evidence is allocated, not shared. Two placements of the same card can
        be in flight at once, and both would see the same new body and call
        themselves confirmed, biasing the measurement toward "it landed", the
        direction that hides refusals. Within one observation the surplus goes
        out in issue order, one body per record.
        """
        claimed: Counter = Counter()
        still_open = []
        for rec in self._open:
            if now - rec.t_issue > self.window_s:
                rec.closed = True
                continue
            rec.observations += 1
            if rec.expect:
                delta = (self._count(gs, rec.expect) - rec.baseline_count
                         - claimed[rec.expect])
                rec.best_delta = max(rec.best_delta, delta)
                if delta >= 1:
                    claimed[rec.expect] += 1
            hand = self._hand_ids(gs)
            if (rec.slot < len(hand) and rec.slot < len(rec.hand_at_issue)
                    and hand[rec.slot] != UNKNOWN_CARD_SIM_ID
                    and rec.hand_at_issue[rec.slot] != UNKNOWN_CARD_SIM_ID
                    and hand[rec.slot] != rec.hand_at_issue[rec.slot]):
                rec.hand_changed = True
            still_open.append(rec)
        self._open = still_open

    def close_all(self) -> None:
        for rec in self._open:
            rec.closed = True
        self._open = []

    def apply_ledger(self, confirmed_tags, rejected_tags) -> None:
        """Attach the ledger's own verdict, keyed by the seq we tagged it with.
        """
        confirmed, rejected = set(confirmed_tags), set(rejected_tags)
        for rec in self.issued:
            if rec.seq in confirmed:
                rec.ledger_confirmed = True
            elif rec.seq in rejected:
                rec.ledger_confirmed = False

    # --- reporting ---

    def cross_tab(self) -> dict:
        """The 2x2 that separates a refused placement from a lost metric. Spells
        are excluded: they spawn no board presence.
        """
        cells = Counter()
        for rec in self.issued:
            if rec.is_spell or rec.ledger_confirmed is None:
                continue
            cells[(rec.unit_confirmed, bool(rec.ledger_confirmed))] += 1
        return {
            "unit_yes_ledger_yes": cells[(True, True)],
            "unit_yes_ledger_no": cells[(True, False)],
            "unit_no_ledger_yes": cells[(False, True)],
            "unit_no_ledger_no": cells[(False, False)],
        }

    def refused(self) -> list[Issued]:
        """Placements with no evidence they reached the board: both oracles
        silent. A card that spawned no detected unit but whose slot cycled was
        played, and calling it a refusal would fill the "what the game rejects"
        dataset with detector misses.
        """
        return [r for r in self.issued
                if not r.is_spell and not r.unit_confirmed and not r.hand_changed]

    def summary(self) -> str:
        n = len(self.issued)
        if not n:
            return "placements: none issued"
        spells = sum(r.is_spell for r in self.issued)
        troops = [r for r in self.issued if not r.is_spell]
        unit_ok = sum(r.unit_confirmed for r in troops)
        hand_ok = sum(r.hand_changed for r in self.issued)
        either = sum(r.unit_confirmed or r.hand_changed for r in troops)
        ct = self.cross_tab()
        lines = [
            f"placements issued: {n}  ({spells} spells, {len(troops)} unit-spawning)",
            f"  unit appeared     : {unit_ok}/{len(troops)}"
            + (f"  ({unit_ok / len(troops):.0%})" if troops else ""),
            f"  hand slot cycled  : {hand_ok}/{n}  ({hand_ok / n:.0%})",
            f"  either oracle     : {either}/{len(troops)}"
            + (f"  ({either / len(troops):.0%})" if troops else ""),
            "",
            "  unit x ledger (unit-spawning cards with a ledger verdict):",
            f"    unit YES / ledger YES : {ct['unit_yes_ledger_yes']:>3}   healthy",
            f"    unit YES / ledger NO  : {ct['unit_yes_ledger_no']:>3}   "
            f"LEDGER METRIC BUG - it landed",
            f"    unit NO  / ledger YES : {ct['unit_no_ledger_yes']:>3}   "
            f"detector missed the body",
            f"    unit NO  / ledger NO  : {ct['unit_no_ledger_no']:>3}   "
            f"THE GAME REFUSED IT",
        ]
        bad = self.refused()
        if bad:
            lines.append("")
            lines.append(f"  NO EVIDENCE IT LANDED ({len(bad)}) - "
                         f"card, engine tile, tapped pixel:")
            for r in bad:
                lines.append(
                    f"    {r.card_name:<12} tile=({r.tile[0]:>2},{r.tile[1]:>2})"
                    f"  tap=({r.tap[0]:>4},{r.tap[1]:>4})"
                    f"  elixir_seen={r.elixir_seen:.0f} owed={r.unconfirmed_owed:.0f}"
                    f"  obs={r.observations}")
        return "\n".join(lines)

    def write_log(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for rec in self.issued:
                fh.write(json.dumps(rec.as_row()) + "\n")

    # --- internals ---

    @staticmethod
    def _count(gs, expect: frozenset[str]) -> int:
        if not expect:
            return 0
        return sum(1 for u in getattr(gs, "units", ())
                   if u.team == 0 and u.unit_name in expect)

    @staticmethod
    def _hand_ids(gs) -> tuple[int, ...]:
        """Hand slots as simulator card ids, as `GameState` carries them.

        An unreadable slot is UNKNOWN_CARD_SIM_ID, and the change test skips
        those at both ends: "unknown -> Giant" and "Giant -> unknown" are the
        reader waking up or going blind, not a play.
        """
        return tuple(int(c) for c in getattr(gs, "my_hand", ()) or ())
