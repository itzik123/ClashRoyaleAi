"""The promo video's story beats as short vertical clips, one MP4 each.

Each scene is staged in the engine (bodies injected, both players holding),
checked against what its caption will claim, and rendered through
web/viewer.html by export_viewer.py:

  cannon_corner     the reward loophole: an enemy Hog runs from the bridge and,
                    at the last tick a Cannon in the middle would still pull
                    it, the Cannon lands behind the King instead (recreated;
                    slow motion into the drop, and cannon_corner.beats.json
                    gives the drop's time in the clip for a sound effect)
  giant_walk_now    a Giant from the bridge to the tower, at today's speed
  giant_walk_bug    the same walk at the pre-fix speed: bridge to tower in
                    3.5 s (recreated)
  giant_stuck       the bridge-mouth orbit, RECORDED before the 2026-09-24
                    fix: a Fireball knocks a Giant off the bridge and it
                    vibrates in place (tools/promo/recordings/)
  mortar_r          the King and the Mortar share the letter R, which is how
                    the engine once found the King (recreated)
  fireball_one      a 4-elixir Fireball that catches a single troop, which
                    survives on a sliver of HP until our Princess Tower
                    finishes it (beats: fireball, dies)

    python_ai/venv/Scripts/python.exe tools/promo/scenes.py
    python_ai/venv/Scripts/python.exe tools/promo/scenes.py cannon_corner giant_stuck
    python_ai/venv/Scripts/python.exe tools/promo/scenes.py --list

Clips land in tools/promo/out/scenes/. Every clip carries a small burned-in
label saying whether it is recorded or recreated; keep it in the edit.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from common import OUT_DIR, RIVER_Y_END, engine
from export_viewer import export, frame_ticks

HERE = os.path.dirname(os.path.abspath(__file__))
RECORDINGS = os.path.join(HERE, "recordings")
SCENE_DIR = OUT_DIR / "scenes"
SKIP = 10
FPS = 30

#: Bridge-to-tower time of a Giant before the 2026-08-07 movement-speed fix,
#: as .claude/CLAUDE.md records it ("a Giant crossed bridge-to-tower in ~3.5 s").
PRE_FIX_WALK_SECONDS = 3.5

#: cannon_corner: the tick the enemy Hog is placed (the half second before it
#: is an empty lane), and the game speed while it runs into the drop. A Hog
#: must be answered within a second of leaving the bridge, so at real time
#: the build-up is over before the music gets going.
HOG_PLACED = 5
SLOW_MO = 0.3


def _ids():
    E = engine()
    return {E.get_card_info(c)["name"]: c for c in E.get_all_card_ids()}


def _env(seed=1, max_ticks=3600):
    E = engine()
    from python_ai.deck import SHIPPED_DECK
    env = E.ClashRoyaleEnv(list(SHIPPED_DECK), list(SHIPPED_DECK), max_ticks)
    env.seed(seed)
    return env


def _hold(env, ticks):
    """Step `ticks` ticks with both players holding."""
    hold = engine().ClashRoyaleEnv.HAND_SIZE
    return env.step_self_play(hold, 0.0, 0.0, hold, 0.0, 0.0, ticks)


def _stage(name, bodies, ticks, seed=1):
    """A fresh board with `bodies` injected, run for `ticks` with both players
    holding; saves the replay and returns (path, env).

    Each body is (card, x, y, team, deploy_ticks) or, to arrive later,
    (at_tick, card, x, y, team, deploy_ticks). deploy_ticks None keeps the
    engine's deploy second.
    """
    env = _env(seed)
    pending = sorted((b if len(b) == 6 else (0,) + tuple(b)) for b in bodies)
    t = 0
    while t < ticks:
        while pending and pending[0][0] <= t:
            _, card, x, y, team, deploy = pending.pop(0)
            env.inject(card, x, y, team, -1.0, -1 if deploy is None else deploy)
        # Stop at the next arrival, so a body can land on any tick.
        n = min(SKIP, ticks - t, pending[0][0] - t if pending else SKIP)
        if _hold(env, n).done:
            break
        t += n
    os.makedirs(SCENE_DIR, exist_ok=True)
    path = str(SCENE_DIR / f"{name}.json")
    env.save_log(path)
    return path, env


def _entity(path, card_id, team):
    """Id of the first entity of `card_id` on `team` in a replay."""
    with open(path, "r", encoding="utf-8") as fh:
        for tick in json.load(fh)["ticks"]:
            for e in tick["entities"]:
                if e["cardId"] == card_id and e["team"] == team:
                    return e["id"]
    raise SystemExit(f"{os.path.basename(path)}: no card {card_id} on team {team}")


def _legal(card, x, y, team, env):
    """The legal cell nearest (x, y), searched outward in half-tile steps."""
    best = None
    for dx in range(-6, 7):
        for dy in range(-6, 7):
            cx, cy = x + dx * 0.5, y + dy * 0.5
            if env.is_valid_placement(card, cx, cy, team):
                d = dx * dx + dy * dy
                if best is None or d < best[0]:
                    best = (d, cx, cy)
    if best is None:
        raise SystemExit(f"no legal cell for card {card} near ({x}, {y})")
    return best[1], best[2]


# --- scenes -------------------------------------------------------------------
# Each returns (export kwargs, facts) and raises if the footage would not show
# what its caption claims.

def _last_pull_tick(hog, cannon, x, y, window=90):
    """The last tick at which a Cannon dropped at (x, y) still keeps the Hog
    off the tower: the drop is tried on a snapshot of every tick from the
    Hog's placement on, and the tower must take no damage for `window` ticks.
    """
    at, card, hx, hy, team, deploy = hog
    env = _env()
    _hold(env, at)
    env.inject(card, hx, hy, team, -1.0, -1 if deploy is None else deploy)
    saves = []
    for _ in range(window):
        s = env.snapshot()
        before = s.get_tower_damage_dealt(1)
        s.inject(cannon, x, y, 0, -1.0, -1)
        _hold(s, window)
        saves.append(s.get_tower_damage_dealt(1) == before)
        _hold(env, 1)
    if env.get_tower_damage_dealt(1) == 0:  # the control: no Cannon, a hit tower
        raise SystemExit(f"cannon_corner: the Hog never reached the tower in {window} ticks")
    n = saves.index(False) if False in saves else len(saves)
    if n == 0 or any(saves[n:]):
        raise SystemExit(f"cannon_corner: a Cannon at ({x}, {y}) has no single pull "
                         f"window against this Hog: {saves}")
    return at + n - 1


def cannon_corner(ids, out_kw):
    """An enemy Hog runs from the bridge; at the last moment a Cannon in the
    middle would still pull it, the Cannon lands behind the King instead."""
    E = engine()
    cannon = ids["Cannon"]
    probe = _env(max_ticks=10)
    # Where the old policy parked it: beside and behind its own King, back row,
    # far from both lanes (27.9% of its Cannons went there).
    cx, cy = _legal(cannon, E.ARENA_CENTER_X + 3.0, 2.0, 0, probe)
    # Where a player drops one against a Hog: mid-arena, four tiles back from
    # the river, to pull it off its lane.
    px, py = _legal(cannon, E.ARENA_CENTER_X, probe.get_own_half_max_y() - 4.0, 0, probe)
    # The Hog on the enemy's first row at the left bridge: the far lane.
    hog = (HOG_PLACED, ids["Hog Rider"], E.ARENA_LEFT_BRIDGE_X, RIVER_Y_END + 0.5, 1, None)
    drop = _last_pull_tick(hog, cannon, px, py)
    path, env = _stage("cannon_corner", [hog, (drop, cannon, cx, cy, 0, None)], drop + 90)
    cannon_dmg = env.get_damage_dealt_by_card(cannon, 0)
    tower_dmg = env.get_tower_damage_dealt(1)
    if cannon_dmg != 0 or tower_dmg <= 0:
        raise SystemExit(f"cannon_corner does not show the loophole: Cannon dealt "
                         f"{cannon_dmg}, the Hog dealt {tower_dmg} to towers")

    with open(path, "r", encoding="utf-8") as fh:
        ticks = json.load(fh)["ticks"]
    hog_id, cannon_id = _entity(path, ids["Hog Rider"], 1), _entity(path, cannon, 0)

    def find(t, eid):
        return next((e for e in ticks[t]["entities"] if e["id"] == eid), None)

    def towers(t):
        return sum(e["hp"] for e in ticks[t]["entities"]
                   if e["cardId"] in (-2, -3) and e["team"] == 0)

    placed = next(t for t in range(len(ticks)) if find(t, hog_id))
    first = find(placed, hog_id)
    runs = next(t for t in range(placed, len(ticks))
                if (find(t, hog_id)["x"], find(t, hog_id)["y"]) != (first["x"], first["y"]))
    lands = next(t for t in range(len(ticks)) if find(t, cannon_id))
    hit = next(t for t in range(1, len(ticks)) if towers(t) < towers(t - 1))
    h = find(lands, hog_id)
    tower = min((e for e in ticks[lands]["entities"] if e["cardId"] == -3 and e["team"] == 0),
                key=lambda e: (e["x"] - h["x"]) ** 2 + (e["y"] - h["y"]) ** 2)
    gap = ((tower["x"] - h["x"]) ** 2 + (tower["y"] - h["y"]) ** 2) ** 0.5

    # Real time while the Hog deploys, slow motion from its first stride
    # through the drop, then back to real time before it reaches the tower.
    ramp = [(runs - 2, 1.0), (runs + 2, SLOW_MO), (lands + 4, SLOW_MO), (lands + 14, 1.0)]
    return (dict(replay=path, start=max(0, placed - 5), end=min(len(ticks) - 1, hit + 35),
                 speed=ramp, select=cannon_id, label="Recreated in the engine",
                 beats={"cannon_drop": lands, "hog_first_hit": hit}),
            f"a Cannon mid-arena at ({px}, {py}) still pulls the Hog if dropped by tick "
            f"{drop}, with the Hog {gap:.1f} tiles from the tower; one tick later it "
            f"does not. It lands at ({cx}, {cy}) instead, deals 0, and the Hog deals "
            f"{tower_dmg} to the towers")


def _giant_walk(ids):
    E = engine()
    lane = E.ARENA_RIGHT_BRIDGE_X
    path, env = _stage("giant_walk", [(ids["Giant"], lane, 14.0, 0, 0)], 200)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    gid = _entity(path, ids["Giant"], 0)
    # From the bridge mouth to the first swing at the tower.
    on_bridge = hit = None
    tower_hp = None
    for t, tick in enumerate(data["ticks"]):
        g = next((e for e in tick["entities"] if e["id"] == gid), None)
        towers = sum(e["hp"] for e in tick["entities"] if e["cardId"] in (-2, -3) and e["team"] == 1)
        if g and on_bridge is None and g["y"] >= 15.5:
            on_bridge = t
        if tower_hp is not None and towers < tower_hp and hit is None:
            hit = t
        tower_hp = towers
    if on_bridge is None or hit is None:
        raise SystemExit("giant_walk: the Giant never reached the tower")
    return path, gid, on_bridge, hit


def giant_walk_now(ids, out_kw):
    path, gid, a, b = _giant_walk(ids)
    return (dict(replay=path, start=max(0, a - 5), end=b + 8, speed=1.0, select=gid,
                 label="After the fix"),
            f"bridge to first hit: {(b - a) / 10:.1f} s at today's speed")


def giant_walk_bug(ids, out_kw):
    path, gid, a, b = _giant_walk(ids)
    speed = (b - a) / 10.0 / PRE_FIX_WALK_SECONDS
    return (dict(replay=path, start=max(0, a - 5), end=b + 8, speed=speed, select=gid,
                 label="Recreated: pre-fix speed"),
            f"played at {speed:.2f}x, so bridge to tower takes {PRE_FIX_WALK_SECONDS} s")


def giant_stuck(ids, out_kw):
    path = os.path.join(RECORDINGS, "giant_stuck_at_bridge_pre_fix.json")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}: the pre-fix recording cannot be "
                         "re-simulated on a fixed engine")
    with open(path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)["promo"]
    return (dict(replay=path, start=meta["start"], end=meta["end"], speed=1.0,
                 crop=tuple(meta["crop"]), select=meta["giant_id"],
                 label="Recorded before the fix"),
            meta["note"])


def mortar_r(ids, out_kw):
    E = engine()
    mid = ids["Mortar"]
    cx, cy = E.ARENA_CENTER_X, E.arena_king_y(1) - 7.0
    path, env = _stage("mortar_r", [
        (mid, cx, cy, 1, 0),
        (ids["Knight"], E.ARENA_LEFT_BRIDGE_X, 14.0, 0, 0),
    ], 160)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    king = next(e for e in data["ticks"][0]["entities"] if e["cardId"] == -2 and e["team"] == 1)
    mortar = next(e for t in data["ticks"] for e in t["entities"] if e["cardId"] == mid)
    if king["symbol"] != mortar["symbol"]:
        raise SystemExit(f"mortar_r: the King is {king['symbol']!r} and the Mortar "
                         f"{mortar['symbol']!r}, so the scene no longer shows the clash")
    return (dict(replay=path, start=0, end=150, speed=1.5, select=mortar["id"],
                 label="Recreated in the engine"),
            f"King and Mortar both carry the symbol {king['symbol']!r}")


def fireball_one(ids, out_kw):
    E = engine()
    x = E.ARENA_LEFT_LANE_X
    # A lone Musketeer walks down the lane; a Fireball lands on it 2 s later.
    # It survives on a sliver of HP, and our Princess Tower finishes it.
    path, env = _stage("fireball_one", [
        (ids["Musketeer"], x, 23.0, 1, 0),
        (20, ids["Fireball"], x, 21.0, 0, None),
    ], 110)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    ticks = data["ticks"]
    fb = next((t for t, tick in enumerate(ticks)
               if any(e["cardId"] == ids["Fireball"] for e in tick["entities"])), None)
    if fb is None:
        raise SystemExit("fireball_one: the Fireball never appeared")
    mid = _entity(path, ids["Musketeer"], 1)
    hp = [next((e["hp"] for e in tick["entities"] if e["id"] == mid), None) for tick in ticks]
    alive = [t for t, h in enumerate(hp) if h is not None]
    dies = alive[-1] + 1
    hit = next((t for t in alive if hp[t] < hp[alive[0]]), None)
    if hit is None or dies >= len(ticks):
        raise SystemExit("fireball_one: the Musketeer was never hit, or never died")
    if dies <= hit + 1:
        raise SystemExit("fireball_one: the Fireball killed the Musketeer outright; "
                         "the scene is meant to leave it for the tower")
    return (dict(replay=path, start=max(0, fb - 15), end=min(len(ticks) - 1, dies + 25),
                 speed=1.0, label="Illustration", beats={"fireball": fb, "dies": dies}),
            f"one Fireball, one troop: it survives on {hp[hit]:.0f} HP and the tower "
            f"finishes it {(dies - hit) / 10:.1f} s later")


SCENES = {f.__name__: f for f in (cannon_corner, giant_walk_now, giant_walk_bug,
                                   giant_stuck, mortar_r, fireball_one)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("names", nargs="*", help="scenes to render (default: all)")
    ap.add_argument("--list", action="store_true", help="list the scenes and exit")
    ap.add_argument("--size", default="vertical")
    ap.add_argument("--still", type=float, default=None,
                    help="one PNG per scene at this many seconds, no video")
    ap.add_argument("--no-label", action="store_true",
                    help="leave the recorded/recreated label off the video; "
                         "edit/edit_short.py draws it on top, where zooms cannot crop it")
    ap.add_argument("--no-select", action="store_true",
                    help="no yellow selection ring on the scene's key unit; "
                         "edit/edit_short.py marks it with its own graphics")
    ap.add_argument("--ffmpeg", default=None)
    ap.add_argument("--browser", default=None)
    args = ap.parse_args()

    if args.list:
        print(__doc__.split("\n\n")[1])
        return
    names = args.names or list(SCENES)
    unknown = [n for n in names if n not in SCENES]
    if unknown:
        raise SystemExit(f"unknown scene(s) {unknown}; choose from {list(SCENES)}")

    ids = _ids()
    for name in names:
        t0 = time.time()
        kw, fact = SCENES[name](ids, {})
        replay = kw.pop("replay")
        beats = kw.pop("beats", None)
        label = kw.pop("label", None)
        if args.no_select:
            kw.pop("select", None)
        out = str(SCENE_DIR / f"{name}.mp4")
        print(f"[{name}] {fact}")
        path = export(replay, out, size=args.size, still=args.still, fps=FPS,
                      label=None if args.no_label else label,
                      ffmpeg=args.ffmpeg, browser=args.browser, quiet=True, **kw)
        print(f"[{name}] wrote {path} in {time.time() - t0:.0f}s")
        if args.still is None:
            # The label travels with the clip, burned in or not, so the editor
            # can never show recreated footage unlabelled by accident.
            with open(SCENE_DIR / f"{name}.clip.json", "w", encoding="utf-8") as fh:
                json.dump({"label": label, "burned": bool(label) and not args.no_label},
                          fh, indent=2)
        if beats:
            _write_beats(name, beats, kw)


def _write_beats(name, beats, kw):
    """<scene>.beats.json: the clip time of each named moment, for cutting
    music and sound effects to it."""
    shown = frame_ticks(kw["start"], kw["end"], kw["speed"], FPS)
    out = {"fps": FPS}
    for beat, tick in beats.items():
        frame = next((f for f, t in enumerate(shown) if t >= tick), None)
        if frame is None:
            continue
        out[beat] = {"frame": frame, "seconds": round(frame / FPS, 3), "tick": tick}
        print(f"[{name}] {beat}: {frame / FPS:.2f} s into the clip (frame {frame})")
    with open(SCENE_DIR / f"{name}.beats.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
