"""Episode -> a single time-ordered stream of placement events for both sides.

The ego side is labelled (KataCR reads the player's own hand: slot, cell,
time). The opponent's hand is invisible, so their placements are inferred from
the detected unit grid: a body class appearing on their side after being absent
counts as a card just played.

That inference is the weakest link and is kept crude on purpose; the
scrambled-time control in `divergence.py` decides whether it carries signal.
Two known errors:

  - A body is not a card. 'skeleton' is Skeletons, Skeleton Army, a Tombstone's
    output or a Graveyard's; `_UNIT_TO_CARD` picks one.
  - Detection flicker re-emits a unit that never left. `_DEBOUNCE_FRAMES` and
    `_REAPPEAR_COOLDOWN_S` suppress the fast cases, nothing the slow ones.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .katacr_format import NON_BODY_CLASSES

_DEBOUNCE_FRAMES = 2        # a class must persist this long to count as arrived
_REAPPEAR_COOLDOWN_S = 2.0  # ...and must have been absent this long before that

# Their detector names bodies, our registry names cards; a body belonging to
# several cards maps to the cheapest or most common.
_UNIT_TO_CARD = {
    "skeleton": "skeletons", "barbarian": "barbarians",
    "phoenix-big": "phoenix", "phoenix-small": "phoenix", "phoenix-egg": "phoenix",
    "minion": "minions", "goblin": "goblins", "spear-goblin": "spear-goblins",
    "bat": "bats", "archer": "archers", "royal-recruit": "royal-recruits",
    "royal-hog": "royal-hogs", "rascal-boy": "rascals", "rascal-girl": "rascals",
    "three-musketeer": "three-musketeers", "elite-barbarian": "elite-barbarians",
    "golemite": "golem", "lava-pup": "lava-hound", "guard": "guards",
    "skeleton-king-skeleton": "skeleton-king", "mini-pekka": "mini-pekka",
    "the-log": "the-log", "ice-golemite": "ice-golem",
}


@dataclass
class Event:
    t: float          # match seconds
    team: int         # 0 = ego, 1 = opponent
    card: str         # OUR card name
    x: float          # engine frame
    y: float
    inferred: bool    # True for the opponent side


def _engine_card_names() -> dict[str, int]:
    import clash_royale_env as E
    from python_ai.engine_constants import card_name
    out: dict[str, int] = {}
    for cid in E.get_all_card_ids():
        nm = card_name(cid).rsplit("_", 1)[0].lower().replace("_", "-")
        out.setdefault(nm, cid)
    return out


def unit_to_card_name(unit: str) -> str | None:
    if unit in NON_BODY_CLASSES:
        return None
    return _UNIT_TO_CARD.get(unit, unit)


def ego_events(episode, transform) -> list[Event]:
    out = []
    for i, a in enumerate(episode.action[:episode.n_frames]):
        slot = int(a["card_id"])
        if slot == 0 or a["xy"] is None:
            continue
        card = episode.card_name_at(i, slot)
        if card == "empty":
            continue                      # slot read as empty: a perception miss
        x, y = transform.to_engine(float(a["xy"][0]), float(a["xy"][1]))
        out.append(Event(episode.seconds_at(i), 0, card, x, y, inferred=False))
    return out


def opponent_events(episode, transform) -> list[Event]:
    """Infer opponent placements from first-appearances of their bodies."""
    n = episode.n_frames
    present: list[dict[str, tuple[float, float]]] = []
    for s in episode.state[:episode.n_frames]:
        cur: dict[str, list[tuple[float, float]]] = {}
        for u in s["unit_infos"]:
            if u["cls"] is None or u["xy"] is None or u.get("bel") != 1:
                continue
            nm = episode.idx2unit[u["cls"]]
            if nm in NON_BODY_CLASSES:
                continue
            cur.setdefault(nm, []).append((float(u["xy"][0]), float(u["xy"][1])))
        present.append({k: tuple(np.median(np.asarray(v), axis=0)) for k, v in cur.items()})

    cooldown_frames = int(round(_REAPPEAR_COOLDOWN_S * episode.fps))
    last_seen: dict[str, int] = {}
    out = []
    for i in range(n):
        for nm, xy in present[i].items():
            persists = all(nm in present[j] for j in range(i, min(n, i + _DEBOUNCE_FRAMES)))
            if not persists:
                continue
            if nm in last_seen and i - last_seen[nm] < cooldown_frames:
                last_seen[nm] = i
                continue
            fresh = nm not in last_seen or (i - last_seen[nm]) >= cooldown_frames
            last_seen[nm] = i
            if not fresh:
                continue
            card = unit_to_card_name(nm)
            if card is None:
                continue
            x, y = transform.to_engine(*xy)
            out.append(Event(episode.seconds_at(i), 1, card, x, y, inferred=True))
        for nm in present[i]:
            last_seen[nm] = i
    return out


def build_events(episode, transform):
    """Merged, time-ordered stream plus a per-card census (control C1)."""
    ego, opp = ego_events(episode, transform), opponent_events(episode, transform)
    import clash_royale_env as E

    names = _engine_card_names()
    census = {"ego": {}, "opp": {}, "unmapped": {}, "unaffordable": {}}
    ego_r: list[tuple[Event, int]] = []
    opp_r: list[tuple[Event, int]] = []
    for ev in sorted(ego + opp, key=lambda e: e.t):
        cid = names.get(ev.card)
        if cid is None:
            census["unmapped"][ev.card] = census["unmapped"].get(ev.card, 0) + 1
            continue
        (opp_r if ev.team else ego_r).append((ev, cid))

    costs: dict[int, float] = {}

    def cost_of(cid: int) -> float:
        if cid not in costs:
            costs[cid] = float(E.get_card_info(cid)["cost"])
        return costs[cid]

    # Only the inferred side is gated; ego plays come from the player's own
    # hand.
    opp_kept, opp_dropped = gate_by_elixir(opp_r, cost_of)
    for ev, _ in ego_r:
        census["ego"][ev.card] = census["ego"].get(ev.card, 0) + 1
    for ev, _ in opp_kept:
        census["opp"][ev.card] = census["opp"].get(ev.card, 0) + 1
    for ev, _ in opp_dropped:
        census["unaffordable"][ev.card] = census["unaffordable"].get(ev.card, 0) + 1
    resolved = sorted(ego_r + opp_kept, key=lambda p: p[0].t)
    return resolved, census


def _elixir_multiplier(t: float) -> float:
    """The real game's schedule: 1x for the first 2:00, 2x for the last minute
    of regular time, 3x in overtime."""
    if t < 120.0:
        return 1.0
    if t < 180.0:
        return 2.0
    return 3.0


def gate_by_elixir(events, cost_of, start_elixir: float = 5.0, cap: float = 10.0):
    """Drop inferred placements the opponent could not have afforded.

    A Golem splitting, a Graveyard ticking out Skeletons, a Phoenix hatching
    and every hut's output look like placements to a first-appearance detector,
    each injecting a whole fresh card. Detection cannot tell them from real
    plays; the opponent's economy can, since they cannot spend faster than
    elixir accrues. What the budget rejects measures how much the inference
    over-fires.
    """
    from perception.timebase import elixir_regenerated

    kept, dropped = [], []
    bar, last_t = start_elixir, None
    for ev, cid in events:
        if last_t is not None:
            bar += elixir_regenerated(max(0.0, ev.t - last_t), _elixir_multiplier(ev.t))
        bar = min(bar, cap)
        last_t = ev.t
        c = cost_of(cid)
        if c <= bar:
            bar -= c
            kept.append((ev, cid))
        else:
            dropped.append((ev, cid))
    return kept, dropped
