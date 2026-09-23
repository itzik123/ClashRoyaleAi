"""The metrics, and the two controls that stop them lying.

Every failure mode of this probe (a parser emitting no opponent events, a card
map dropping half the deck, a timestamp offset aligning the empty opening)
makes the reconstruction look better, not worse. So the measurement carries
controls that must fire:

  C1  an injection census: parsed, injected and dropped, per side. Both sides
      non-zero, or the episode is void.
  C2  a scrambled-time arm: the same events with timestamps shuffled. The true
      arm must beat it, or M1/M2/M3 mean nothing.

M1 and M3 are robust to an imperfect board transform; M2 is not, so it is read
against the fit residual.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .katacr_format import NON_BODY_CLASSES

SAMPLE_TIMES = (30.0, 60.0, 90.0, 120.0, 150.0, 180.0)


@dataclass
class EpisodeResult:
    name: str
    fit_residual: float
    census: dict
    n_events: tuple[int, int]
    m1: dict[float, tuple[int, int]]      # t -> (|d ego|, |d opp|)
    m2: dict[float, float | None]         # t -> centroid distance (tiles)
    m3: tuple[str, str]                   # (recorded outcome, ours)
    recorded_units: dict[float, tuple[int, int]]
    our_units: dict[float, tuple[int, int]]
    alive: dict[float, bool]
    our_end_t: float = 0.0
    recorded_end_t: float = 0.0


def _frame_at(episode, t: float) -> int:
    t0 = episode.seconds_at(0)
    return int(np.clip(round((t - t0) * episode.fps), 0, episode.n_frames - 1))


def recorded_units(episode, t: float) -> tuple[int, int]:
    s = episode.state[_frame_at(episode, t)]
    n = [0, 0]
    for u in s["unit_infos"]:
        if u["cls"] is None or u.get("bel") is None:
            continue
        if episode.idx2unit[u["cls"]] in NON_BODY_CLASSES:
            continue
        n[int(u["bel"])] += 1
    return n[0], n[1]


def recorded_centroid(episode, t, bel, transform):
    s = episode.state[_frame_at(episode, t)]
    pts = [transform.to_engine(float(u["xy"][0]), float(u["xy"][1]))
           for u in s["unit_infos"]
           if u["cls"] is not None and u["xy"] is not None and u.get("bel") == bel
           and episode.idx2unit[u["cls"]] not in NON_BODY_CLASSES]
    if not pts:
        return None
    a = np.asarray(pts)
    return float(a[:, 0].mean()), float(a[:, 1].mean())


_TERMINAL_REWARD_THRESHOLD = 0.5


def recorded_outcome(episode) -> str:
    """Winner from KataCR's own terminal reward. Tower presence does not work on
    this corpus (the ego's Princess Towers go undetected for the first 20 s,
    and all six read alive at the end, so every outcome came back "draw");
    `RewardBuilder` OCRs tower HP and emits a terminal bonus of about +/-1,
    which separates cleanly.

    The corpus is a strong player's own uploads and heavily win-biased, so M3
    must beat the base rate, reported beside it, not 50%.
    """
    nz = np.nonzero(episode.reward)[0]
    if not len(nz):
        return "draw"
    v = float(episode.reward[nz[-1]])
    if v > _TERMINAL_REWARD_THRESHOLD:
        return "ego"
    if v < -_TERMINAL_REWARD_THRESHOLD:
        return "opp"
    return "draw"


def our_outcome(rec) -> str:
    ego_alive, opp_alive = rec.towers_alive
    if opp_alive < ego_alive:
        return "ego"
    if ego_alive < opp_alive:
        return "opp"
    return "draw"


def scramble(resolved_events, rng: random.Random):
    """C2: keep the same placements, destroy only their timing."""
    times = [ev.t for ev, _ in resolved_events]
    rng.shuffle(times)
    out = []
    for (ev, cid), t in zip(resolved_events, times):
        out.append((type(ev)(t, ev.team, ev.card, ev.x, ev.y, ev.inferred), cid))
    return sorted(out, key=lambda p: p[0].t)


def score(episode, transform, resolved, rec, census) -> EpisodeResult:
    m1, m2, rec_u, our_u, alive = {}, {}, {}, {}, {}
    by_t = {s.t: s for s in rec.samples}
    for t in SAMPLE_TIMES:
        if t > episode.seconds_at(episode.n_frames - 1):
            continue
        r_ego, r_opp = recorded_units(episode, t)
        rec_u[t] = (r_ego, r_opp)
        s = by_t.get(t)
        if s is None:
            m1[t] = (r_ego, r_opp)     # our match ended: full mismatch
            m2[t] = None
            our_u[t] = (0, 0)
            alive[t] = False
            continue
        our_u[t] = s.units
        alive[t] = True
        m1[t] = (abs(s.units[0] - r_ego), abs(s.units[1] - r_opp))
        ds = []
        for bel in (0, 1):
            rc = recorded_centroid(episode, t, bel, transform)
            oc = s.centroid[bel]
            if rc and oc:
                ds.append(float(np.hypot(rc[0] - oc[0], rc[1] - oc[1])))
        m2[t] = float(np.mean(ds)) if ds else None
    n_ego = sum(1 for e, _ in resolved if e.team == 0)
    return EpisodeResult(
        episode.path.name, transform.residual_tiles, census,
        (n_ego, len(resolved) - n_ego), m1, m2,
        (recorded_outcome(episode), our_outcome(rec)), rec_u, our_u, alive,
        rec.end_t, episode.seconds_at(episode.n_frames - 1))
