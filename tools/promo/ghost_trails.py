"""Ghost trails: the futures the lookahead search looked at, drawn on the board.

Plays one match with the network plus 1-ply search (python_ai/search/search.py,
the search behind "win rate 0.625 -> 0.944") against the built-in C++ bot at
1.5x elixir, the condition that number was measured in. At each decision the
search forks the game (env.snapshot()), plays every candidate move 4 seconds
ahead, scores each future with the network's critic, and picks the best.

For a few of those decisions this re-plays every candidate on its own
snapshot, records where the new units go, and renders a clip:

  1. the real match up to the decision;
  2. the board freezes and each candidate's future grows as a coloured trail
     from its placement, ending in the critic's score for it;
  3. the chosen future lights up, the rest fade, and the real match plays on
     with the move search actually made.

    python_ai/venv/Scripts/python.exe tools/promo/ghost_trails.py
    python_ai/venv/Scripts/python.exe tools/promo/ghost_trails.py --seed 4 --clips 3
    python_ai/venv/Scripts/python.exe tools/promo/ghost_trails.py --still

Clips land in tools/promo/out/ghost_trails/. The engine is deterministic, so a
re-played candidate is the same future the search scored (checked per clip).

The search runs as it was MEASURED (2026-08-11): greedy plus the top 3 cards'
top 2 cells, at most 7 futures. The widened proposals added on 2026-09-03 (a
48-cell sweep for any card whose placement head is flat) put ~97 futures on
one board, which is neither readable nor the search the win rate belongs to;
--wide-proposals shows them anyway.

A STAGED board instead of a real match:

    python_ai/venv/Scripts/python.exe tools/promo/ghost_trails.py --board tools/promo/boards/triple_elixir.json

The board file sets the clock, both decks, tower HP, our hand and elixir, the
troops on the board (where they stand --lead seconds before the decision) and
the plays to compare. The engine plays the lead, the network reads the board
and names its own pick (it is always one of the options, at its own cell),
and every play is rolled forward and scored by the critic exactly as the
search does. The scores on screen are each play's score minus doing nothing's.
The clip says "Staged in the engine" throughout.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import time

from PIL import ImageFilter

from common import (BG, OUT_DIR, REPO_ROOT, TEAM, BoardPainter, Image, ImageDraw,
                    Replay, VideoWriter, composite, ease_in_out, engine, font,
                    load_arena, parse_size)

SKIP = 10
CLIP_DIR = OUT_DIR / "ghost_trails"
# Candidate colours: none is a team colour, all read on grass and water.
COLORS = [(250, 204, 21), (34, 211, 238), (244, 114, 182), (163, 230, 53),
          (251, 146, 60), (255, 255, 255), (167, 139, 250), (45, 212, 191)]
# A staged board's plays keep one colour per card, so a spell reads as fire.
CARD_COLORS = {"Fireball": (251, 146, 60), "Cannon": (34, 211, 238),
               "Musketeer": (250, 204, 21), "Hog Rider": (244, 114, 182),
               "The Log": (163, 230, 53), "Ice Golem": (167, 139, 250),
               "Skeletons": (255, 255, 255), "Ice Spirit": (45, 212, 191)}
# An enemy troop counts as pulled by a play when it ends this far (tiles) from
# where it ends if we do nothing.
MOVED_TILES = 1.2


# --- the match ------------------------------------------------------------------
def play_match(args, workdir):
    """One searched match. Returns (replay path, [decision]), where a decision
    holds the snapshot taken before the search and what the search did."""
    E = engine()
    import numpy as np
    import torch
    from python_ai.envs.gym_wrapper import DEFAULT_DECK
    from python_ai.models.policy_io import LSTM_HIDDEN, load_net
    from python_ai.search.config import SearchCfg
    from python_ai.search.search import greedy_from_logits, policy_head, search_action

    device = torch.device("cpu")
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    weights = args.weights if os.path.isabs(args.weights) else \
        str(REPO_ROOT / "python_ai" / args.weights)
    net = load_net(weights, device)
    cfg = SearchCfg()

    env = E.ClashRoyaleEnv(list(DEFAULT_DECK), list(DEFAULT_DECK), args.max_ticks)
    env.set_opponent_elixir_multiplier(args.opp_elixir)
    env.seed(args.seed)
    hidden = (torch.zeros(1, LSTM_HIDDEN), torch.zeros(1, LSTM_HIDDEN))
    obs = env.get_observation_for_team(0)
    decisions, done, steps, reward = [], False, 0, 0.0
    with torch.no_grad():
        while not done and steps < cfg.max_steps:
            obs_t = torch.tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
            logits, embeds, spatial, _, hidden_next = policy_head(net, obs_t, hidden)
            greedy = greedy_from_logits(net, obs_t, logits, embeds, spatial, hidden_next)[:3]
            snap = env.snapshot()
            hand = list(env.get_hand())
            action, deviated, n, details = search_action(
                net, env, obs_t, logits, embeds, spatial, hidden_next, greedy, cfg,
                device, return_details=True)
            if details is not None:
                cands, scores = details
                decisions.append({
                    "tick": env.get_current_tick(), "snap": snap, "hand": hand,
                    "cands": [tuple(c) for c in cands],
                    "scores": [float(s) for s in scores],
                    "chosen": [tuple(c) for c in cands].index(tuple(action)),
                    "deviated": bool(deviated), "horizon": cfg.horizon})
            hidden = hidden_next
            r = env.step(action[0], action[1], action[2], SKIP)
            obs, reward, done = r.observation, float(r.reward), r.done
            steps += 1
    path = os.path.join(workdir, "match.json")
    env.save_log(path)
    outcome = "won" if reward > 0.5 else "lost" if reward < -0.5 else "drew"
    print(f"match: {steps} decisions, {len(decisions)} with a choice to make, "
          f"{sum(d['deviated'] for d in decisions)} where search overrode the "
          f"network; the searched agent {outcome}")
    return path, decisions


def staged_match(args, workdir):
    """A hand-built board (--board): the engine plays the lead, the network
    names its pick, and each play in the file is scored as the search scores
    its candidates. Returns (replay path, decision) like play_match."""
    E = engine()
    import numpy as np
    import torch
    from python_ai.envs.gym_wrapper import DEFAULT_DECK
    from python_ai.models.policy_io import LSTM_HIDDEN, load_net
    from python_ai.search.config import SearchCfg
    from python_ai.search.search import (NOOP, greedy_from_logits, policy_head, rollout,
                                         terminal_score)

    with open(args.board, "r", encoding="utf-8") as fh:
        board = json.load(fh)
    ids = {E.get_card_info(c)["name"]: c for c in E.get_all_card_ids()}

    def card(name):
        if name not in ids:
            raise SystemExit(f"{args.board}: no card named {name!r}")
        return ids[name]

    weights = args.weights if os.path.isabs(args.weights) else \
        str(REPO_ROOT / "python_ai" / args.weights)
    net = load_net(weights, torch.device("cpu"))
    cfg = SearchCfg()
    horizon = int(board.get("horizon", cfg.horizon))
    hold = E.ClashRoyaleEnv.HAND_SIZE

    env = E.ClashRoyaleEnv(list(DEFAULT_DECK), [card(n) for n in board["enemy_deck"]],
                           args.max_ticks)
    env.seed(int(board.get("seed", args.seed)))
    lead = round(args.lead * SKIP)
    env.set_current_tick(int(board["tick"]) - lead)
    for team in (0, 1):
        for slot, frac in enumerate(board["towers"][str(team)]):
            if not env.set_tower_hp(team, slot, frac * env.get_tower_max_hp(team, slot)):
                raise SystemExit(f"{args.board}: tower {team}/{slot} refused hp {frac}")
    if not env.set_hand_for_team(0, [card(n) for n in board["hand"]]):
        raise SystemExit(f"{args.board}: the engine refused the hand {board['hand']}")
    for team in (0, 1):
        env.set_elixir_for_team(team, float(board["elixir"][team]))
    horizon = int(getattr(args, "horizon", None) or horizon)
    for name, x, y, team in board["bodies"]:
        env.inject(card(name), float(x), float(y), int(team), -1.0, 0)

    def obs():
        return torch.tensor(np.asarray(env.get_observation_for_team(0),
                                       dtype=np.float32)).unsqueeze(0)

    # The lead: both sides hold, the troops play. The network reads every
    # second of it, so its memory holds the fight when it decides.
    hidden = (torch.zeros(1, LSTM_HIDDEN), torch.zeros(1, LSTM_HIDDEN))
    with torch.no_grad():
        done = 0
        while done < lead:
            n = min(SKIP, lead - done)
            _, _, _, _, hidden = policy_head(net, obs(), hidden)
            env.step_self_play(hold, 0.0, 0.0, hold, 0.0, 0.0, n)
            done += n
        for team in (0, 1):
            env.set_elixir_for_team(team, float(board["elixir"][team]))
        obs_t = obs()
        logits, embeds, spatial, _, hidden_next = policy_head(net, obs_t, hidden)
        greedy = greedy_from_logits(net, obs_t, logits, embeds, spatial, hidden_next)[:3]
    hand = list(env.get_hand())
    name_of = {i: E.get_card_info(c)["name"] for i, c in enumerate(hand)}
    network = name_of.get(greedy[0], "Wait")

    def aim(cid):
        """Where a spell does the most damage: tried on every enemy troop."""
        tmp = os.path.join(workdir, "aim.json")
        env.save_log(tmp)
        now = Replay(tmp)
        spots = [(x, y) for eid, x, y, _ in now.ticks[-1]
                 if now.bodies[eid].team == 1 and now.bodies[eid].kind == "troop"]
        best = None
        for x, y in spots:
            if not env.is_valid_placement(cid, x, y, 0):
                continue
            sim = env.snapshot()
            sim.step_self_play(hand.index(cid), x, y, hold, 0.0, 0.0, 2 * SKIP)
            dmg = sim.get_damage_dealt_by_card(cid, 0)
            if best is None or dmg > best[0]:
                best = (dmg, x, y)
        if best is None:
            raise SystemExit(f"{args.board}: no enemy troop to aim {E.get_card_info(cid)['name']} at")
        return best[1], best[2]

    def network_cell(cid):
        """The network's own favourite cell for this card."""
        with torch.no_grad():
            idx = torch.tensor([hand.index(cid)])
            place = net.placement_given_card(hidden_next[0], embeds, idx, obs_t, spatial)
            x_t, y_t = net.cell_to_xy(place.argmax(dim=-1))
        return float(x_t.item()), float(y_t.item())

    # The network's pick first, at its own cell; then the file's plays. A play
    # at "aim" goes where it does the most damage (for a spell); at "net",
    # where the network would put that card.
    cands = [(NOOP, 0.0, 0.0) if greedy[0] >= hold else tuple(greedy)]
    for name, *at in board["plays"]:
        if name == network:
            continue
        if card(name) not in hand:
            raise SystemExit(f"{args.board}: {name} is not in the hand {board['hand']}")
        x, y = aim(card(name)) if at == ["aim"] else network_cell(card(name)) \
            if at == ["net"] else at
        if not env.is_valid_placement(card(name), float(x), float(y), 0):
            raise SystemExit(f"{args.board}: {name} cannot be placed at ({x}, {y})")
        cands.append((hand.index(card(name)), float(x), float(y)))
    wait = (NOOP, 0.0, 0.0)
    scored = cands + ([wait] if wait not in cands else [])

    # Scored exactly as search.search_action scores its candidates: each play
    # rolled `horizon` decisions ahead on its own snapshot, the end state read
    # by the critic from the decision's memory, a finished game by its result.
    snap = env.snapshot()
    final, terminal = [], []
    for c in scored:
        sim = snap.snapshot()
        r = rollout(sim, c[0], c[1], c[2], horizon)
        final.append(sim.get_observation_for_team(0))
        terminal.append((r.done, float(r.reward)))
    with torch.no_grad():
        feats, _, _ = net.extract_features(torch.tensor(np.asarray(final, dtype=np.float32)))
        k = len(scored)
        _, _, _, values, _ = net.step_lstm_and_card(
            feats, (hidden_next[0].expand(k, LSTM_HIDDEN).contiguous(),
                    hidden_next[1].expand(k, LSTM_HIDDEN).contiguous()))
    scores = values.squeeze(-1).tolist()
    for i, (finished, reward) in enumerate(terminal):
        if finished:
            scores[i] = terminal_score(reward, cfg.terminal_weight)
    base = scores[scored.index(wait)]
    scores = scores[:len(cands)]
    chosen = max(range(len(cands)), key=lambda i: scores[i])

    # The clip plays on with the chosen move, as its future did.
    env.step(*cands[chosen], SKIP)
    for _ in range(round(args.follow) + 1):
        env.step(NOOP, 0.0, 0.0, SKIP)
    path = os.path.join(workdir, "staged.json")
    env.save_log(path)
    names = [name_of.get(c[0], "Wait") for c in cands]
    print(f"staged board {os.path.basename(args.board)}: the network wanted {network}; "
          f"the critic scored " + ", ".join(f"{n} {s:+.3f}" for n, s in zip(names, scores))
          + f", doing nothing {base:+.3f}")
    return path, {"tick": lead, "snap": snap, "hand": hand, "cands": cands,
                  "scores": scores, "shown": [s - base for s in scores], "chosen": chosen,
                  "deviated": chosen != 0, "horizon": horizon, "staged": True,
                  "title": board.get("title", "Staged board"),
                  "net": net, "hidden": hidden_next}


def futures(decision, workdir, tag):
    """Re-play each candidate on its own copy of the pre-search snapshot and
    return one Replay of its future per candidate. Also sets
    decision["baseline"], the future where we do nothing."""
    from python_ai.search.search import NOOP, rollout
    out = []
    for i, (card, x, y) in enumerate(decision["cands"]):
        sim = decision["snap"].snapshot()
        rollout(sim, card, x, y, decision["horizon"])
        path = os.path.join(workdir, f"{tag}_cand{i}.json")
        sim.save_log(path)
        out.append(Replay(path))
    sim = decision["snap"].snapshot()
    rollout(sim, NOOP, 0.0, 0.0, decision["horizon"])
    path = os.path.join(workdir, f"{tag}_wait.json")
    sim.save_log(path)
    decision["baseline"] = Replay(path)
    # Determinism check: a second copy of candidate 0 must land identically,
    # or the trails would not be the futures that were scored.
    sim = decision["snap"].snapshot()
    card, x, y = decision["cands"][0]
    rollout(sim, card, x, y, decision["horizon"])
    check = os.path.join(workdir, f"{tag}_check.json")
    sim.save_log(check)
    again = Replay(check)
    if again.ticks[-1] != out[0].ticks[-1]:
        raise SystemExit("a re-played future differs from the first: the engine is "
                         "not deterministic from a snapshot, so the trails would lie")
    return out


def consequences(fut, base, enemies):
    """What a future does to the enemy that doing nothing does not: the paths
    of the enemy troops it pulls off course, and where the ones it kills fall.
    `enemies` is the enemy troops alive at the decision."""
    def paths(rep):
        out = {}
        for tick in rep.ticks:
            for eid, x, y, _ in tick:
                if eid in enemies:
                    out.setdefault(eid, []).append((x, y))
        return out

    pf, pb = paths(fut), paths(base)
    end_f, end_b = {e[0] for e in fut.ticks[-1]}, {e[0] for e in base.ticks[-1]}
    moved, kills = [], []
    for eid, path in pf.items():
        if eid in end_b and eid not in end_f:
            kills.append(path[-1])
        elif eid in end_f and eid in pb and math.dist(path[-1], pb[eid][-1]) > MOVED_TILES:
            moved.append(path)
    return moved, kills


def describe(decision, main):
    """Per candidate: name, placement, the trail of each unit it adds, and
    what it does to the enemy that doing nothing would not."""
    E = engine()
    now = main.ticks[min(decision["tick"], len(main) - 1)]
    before = {e[0] for e in now}
    enemies = {e[0] for e in now if main.bodies[e[0]].team == 1
               and main.bodies[e[0]].kind == "troop"}
    base = decision.get("baseline")
    rows = []
    for (card, x, y), fut in zip(decision["cands"], decision["futures"]):
        if card >= len(decision["hand"]):
            rows.append({"name": "Wait", "wait": True, "trails": [], "at": None,
                         "moved": [], "kills": []})
            continue
        cid = decision["hand"][card]
        name = E.get_card_info(cid)["name"]
        paths = {}
        for tick in fut.ticks:
            for eid, ex, ey, _ in tick:
                b = fut.bodies[eid]
                if eid not in before and b.team == 0 and b.kind in ("troop", "building", "spell"):
                    paths.setdefault(eid, []).append((ex, ey))
        moved, kills = consequences(fut, base, enemies) if base else ([], [])
        if not E.get_card_info(cid)["is_building"]:
            # Pulling troops off course is what a building is FOR; after a
            # troop or a spell it is just defenders retargeting, which reads
            # as noise.
            moved = []
        spell = next((fut.bodies[e].radius for e in paths if fut.bodies[e].kind == "spell"), 0.0)
        rows.append({"name": name, "wait": False, "at": (x, y), "trails": list(paths.values()),
                     "bodies": {eid: fut.bodies[eid] for eid in paths},
                     "moved": moved, "kills": kills, "spell": spell})
    return rows


def pick(decisions, main, n, gap=80):
    """The decisions worth a clip: search overrode the network, and at least
    two candidates put a moving unit on the board. Spread-out in time."""
    scored = []
    for d in decisions:
        moving = sum(1 for c in d["cands"] if c[0] < len(d["hand"]))
        if len(d["cands"]) < 3 or moving < 2:
            continue
        spread = max(d["scores"]) - min(d["scores"])
        scored.append(((d["deviated"], moving, spread), d))
    scored.sort(key=lambda s: s[0], reverse=True)
    chosen = []
    for _, d in scored:
        if 60 <= d["tick"] <= len(main) - 60 and all(abs(d["tick"] - c["tick"]) >= gap for c in chosen):
            chosen.append(d)
        if len(chosen) == n:
            break
    return sorted(chosen, key=lambda d: d["tick"])


def one_per_card(d, rows):
    """Keep one future per card (and Wait): the network's own pick, the
    search's pick, and otherwise each card's best-scored cell. The same card at
    two cells draws two near-identical trails; the point is how different the
    plays are."""
    keep = {}
    for i, row in enumerate(rows):
        name = row["name"]
        pinned = i in (0, d["chosen"])
        if name not in keep or pinned or (d["scores"][i] > d["scores"][keep[name]]
                                          and keep[name] not in (0, d["chosen"])):
            keep[name] = i
    idx = sorted(keep.values())
    sub = {**d, "cands": [d["cands"][i] for i in idx], "scores": [d["scores"][i] for i in idx],
           "chosen": idx.index(d["chosen"])}
    return sub, [rows[i] for i in idx]


def showcase(d, main, want):
    """How well a decision tells the story on screen, and why.

    Rewards: search chose `want` over what the network wanted; the options are
    different KINDS of play (a spell, a building, several troops, not one card
    at two cells); the board is busy, with enemies on our side; and the winner
    wins clearly."""
    E = engine()
    names, kinds = [], set()
    for card, _, _ in d["cands"]:
        if card >= len(d["hand"]):
            names.append("Wait")
            continue
        info = E.get_card_info(d["hand"][card])
        names.append(info["name"])
        kinds.add("spell" if info["is_spell"] else "building" if info["is_building"] else info["name"])
    chosen, greedy = names[d["chosen"]], names[0]
    row = main.ticks[min(d["tick"], len(main) - 1)]
    troops = [e for e in row if main.bodies[e[0]].kind == "troop"]
    enemies_here = sum(1 for e in troops if main.bodies[e[0]].team == 1 and e[2] < 17.5)
    ranked = sorted(d["scores"], reverse=True)
    margin = ranked[0] - ranked[1] if len(ranked) > 1 else 0.0
    distinct = len(set(n for n in names if n != "Wait"))
    score = (3.0 * (chosen == want) + 2.0 * (d["deviated"] and greedy != chosen)
             + 1.5 * distinct + 1.0 * ("spell" in kinds) + 1.0 * ("building" in kinds)
             + 0.25 * min(len(troops), 10) + 0.4 * min(enemies_here, 4) + 2.0 * min(margin, 0.5))
    return score, {"tick": d["tick"], "chosen": chosen, "network": greedy, "options": names,
                   "scores": [round(v, 2) for v in d["scores"]], "troops": len(troops),
                   "enemies_on_our_side": enemies_here, "margin": round(margin, 2)}


def spread(d, main):
    """How far apart the futures end up: the mean distance (tiles) between the
    candidates' new units at the end of the horizon. Big = they branch."""
    ends = []
    for row in describe(d, main):
        pts = [trail[-1] for trail in row["trails"] if trail]
        if pts:
            ends.append((sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)))
    pairs = [(a, b) for i, a in enumerate(ends) for b in ends[i + 1:]]
    return sum(math.dist(a, b) for a, b in pairs) / len(pairs) if pairs else 0.0


def sweep(args, workdir):
    """Play --sweep matches and rank every decision as a showcase."""
    found = []
    for seed in range(args.seed, args.seed + args.sweep):
        args_seed = argparse.Namespace(**{**vars(args), "seed": seed})
        sub = os.path.join(workdir, f"s{seed}")
        os.makedirs(sub, exist_ok=True)
        path, decisions = play_match(args_seed, sub)
        main = Replay(path)
        ranked = sorted((showcase(d, main, args.want) + (d,) for d in decisions
                         if 60 <= d["tick"] <= len(main) - 60 and len(d["cands"]) >= 3),
                        key=lambda r: r[0], reverse=True)[:3]
        for k, (score, why, d) in enumerate(ranked):
            d["futures"] = futures(d, sub, f"t{d['tick']}")
            branch = spread(d, main)
            why.update(seed=seed, spread=round(branch, 1), score=round(score + 0.3 * min(branch, 8), 2))
            found.append(why)
        print(f"seed {seed}: best so far " + (f"{max(f['score'] for f in found):.2f}" if found else "-"))
    found.sort(key=lambda f: f["score"], reverse=True)
    print("\nbest showcases (render one with --seed S --tick T):")
    for f in found[:12]:
        opts = ", ".join(f"{n} {v:+.2f}" for n, v in zip(f["options"], f["scores"]))
        print(f"  {f['score']:5.2f}  --seed {f['seed']} --tick {f['tick']}: network wanted "
              f"{f['network']}, search chose {f['chosen']} | {opts} | {f['troops']} troops, "
              f"{f['enemies_on_our_side']} enemies on our side, futures {f['spread']} tiles apart")
    os.makedirs(CLIP_DIR, exist_ok=True)
    with open(CLIP_DIR / "sweep.json", "w", encoding="utf-8") as fh:
        json.dump(found, fh, indent=1)
    print(f"all {len(found)} ranked in {CLIP_DIR / 'sweep.json'}")


# --- drawing --------------------------------------------------------------------
def score_text(s):
    if s >= 5:
        return "WIN"
    if s <= -5:
        return "LOSS"
    return f"{s:+.2f}"


def draw_futures(frame, painter, ox, oy, t, rows, decision, grow, reveal, focus,
                 only=None):
    """Overlay every candidate's future (or just candidate `only`).

    grow   0..1: how much of each 4-second future is drawn
    reveal 0..1: the score labels fading in
    focus  0..1: the chosen future brightening, the others fading back

    Each future shows the units its play adds (a trail each), a spell's blast,
    the enemy troops it pulls off course (thin dashed trails) and the ones it
    kills (small crosses). The lines glow; the labels sit on top, crisp.
    """
    lines = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    labels = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lines)
    dl = ImageDraw.Draw(labels)
    chosen = decision["chosen"]
    shown = decision.get("shown", decision["scores"])
    f_score = font(t * 0.62)
    f_small = font(t * 0.46)
    wait_slot = 0
    placed = []        # label boxes already drawn, for collision nudging
    named = []         # (name, x, y) of placement tags already drawn

    def free_box(box):
        """Shift `box` down until it overlaps no earlier label."""
        l, tp, r, b = box
        for _ in range(8):
            hit = next((p for p in placed if l < p[2] and r > p[0] and tp < p[3] and b > p[1]), None)
            if hit is None:
                break
            dy = hit[3] - tp + t * 0.08
            tp, b = tp + dy, b + dy
        placed.append((l, tp, r, b))
        return tp - box[1]

    def upto(path, u):
        return path[:max(2, round(len(path) * u))]

    order = sorted(range(len(rows)), key=lambda i: i == chosen)  # chosen on top
    for i in order:
        if only is not None and i != only:
            continue
        row = rows[i]
        col = COLORS[i % len(COLORS)]
        if decision.get("staged"):
            col = CARD_COLORS.get(row["name"], col)
        dim = 1.0 - 0.72 * focus if i != chosen else 1.0
        a = lambda v: max(0, min(255, round(v * dim)))
        width = max(2, round(t * (0.16 if i != chosen else 0.16 + 0.1 * focus)))
        label = score_text(shown[i]) + ("  BEST" if i == chosen and focus > 0.5 else "")

        if row["wait"]:
            # No unit to trail: a badge on the blue half's back line.
            x, y = painter.to_px(ox, oy, t, 1.2, 3.0 + 1.6 * wait_slot)
            wait_slot += 1
            text = f"Wait  {label}" if reveal > 0 else "Wait"
            l, tp, r, b = dl.textbbox((x, y), text, font=f_score)
            pad = t * 0.25
            dl.rounded_rectangle((l - pad, tp - pad, r + pad, b + pad), radius=pad * 2,
                                 fill=(11, 11, 12, a(200 * max(grow, 0.3))),
                                 outline=col + (a(255),), width=width)
            dl.text((x, y), text, font=f_score, fill=col + (a(255),))
            continue

        px, py = painter.to_px(ox, oy, t, *row["at"])
        head = None
        if row.get("spell"):
            # A spell: its blast, growing out from where it lands.
            r = row["spell"] * t * (0.25 + 0.75 * min(1.0, grow * 1.6))
            d.ellipse((px - r, py - r, px + r, py + r), fill=col + (a(70),),
                      outline=col + (a(235),), width=width)
            head = (px + r * 0.72, py - r * 0.72)
        # The enemy troops this play pulls off course: thin and dashed.
        for path in row.get("moved", []):
            pts = [painter.to_px(ox, oy, t, x, y) for x, y in upto(path, grow)]
            for k in range(1, len(pts)):
                if (k // 3) % 2 == 0:
                    d.line((pts[k - 1], pts[k]), fill=col + (a(200),), width=max(2, width // 2))
            if len(pts) > 1:
                ex, ey = pts[-1]
                rr = t * 0.2
                d.ellipse((ex - rr, ey - rr, ex + rr, ey + rr), fill=col + (a(220),))
        ring = t * 0.62
        d.ellipse((px - ring, py - ring, px + ring, py + ring), outline=col + (a(255),),
                  width=width, fill=(11, 11, 12, a(110)))
        dl.text((px, py), str(i + 1), font=f_small, fill=col + (a(255),), anchor="mm")
        # One name tag per card per spot: two cells of the same card sit side
        # by side and their tags would print over each other.
        if i == chosen or not any(n == row["name"] and math.hypot(px - nx, py - ny) < t * 2.5
                                  for n, nx, ny in named):
            named.append((row["name"], px, py))
            # A spell's tag sits under its blast, clear of what it hits.
            tx, ty, anchor = (px, py + row["spell"] * t + t * 0.35, "mt") if row.get("spell") \
                else (px + ring * 1.25, py, "lm")
            tag = dl.textbbox((tx, ty), row["name"], font=f_small, anchor=anchor)
            dy = free_box(tag)
            dl.text((tx, ty + dy), row["name"], font=f_small, anchor=anchor,
                    fill=(255, 255, 255, a(230)), stroke_width=max(1, round(t * 0.05)),
                    stroke_fill=(0, 0, 0, a(200)))
        if not row.get("spell"):
            for path in row["trails"]:
                pts = [painter.to_px(ox, oy, t, x, y) for x, y in upto(path, grow)]
                for k in range(1, len(pts)):
                    fade = 0.35 + 0.65 * k / max(1, len(pts) - 1)
                    d.line((pts[k - 1], pts[k]), fill=col + (a(230 * fade),), width=width)
                if pts:
                    hx, hy = pts[-1]
                    r = t * 0.45
                    d.ellipse((hx - r, hy - r, hx + r, hy + r), fill=col + (a(150),),
                              outline=(255, 255, 255, a(220)), width=max(1, width // 2))
                    if head is None or hy < head[1]:
                        head = (hx, hy)
        # The enemy troops it kills, once its future has run that far.
        if grow > 0.6:
            k_alpha = min(1.0, (grow - 0.6) / 0.3)
            for kx, ky in row.get("kills", []):
                cx, cy = painter.to_px(ox, oy, t, kx, ky)
                s = t * 0.26
                d.ellipse((cx - s * 1.5, cy - s * 1.5, cx + s * 1.5, cy + s * 1.5),
                          fill=(11, 11, 12, a(170 * k_alpha)), outline=col + (a(255 * k_alpha),),
                          width=max(2, width // 2))
                for sx in (-1, 1):
                    d.line((cx - s * 0.75, cy - sx * s * 0.75, cx + s * 0.75, cy + sx * s * 0.75),
                           fill=(255, 255, 255, a(240 * k_alpha)), width=max(2, width // 2))
        if reveal > 0 and head is not None:
            hx, hy = head
            text = label
            pad = t * 0.18
            l, tp, r, b = dl.textbbox((hx + t * 0.7, hy), text, font=f_score, anchor="lm")
            dy = free_box((l - pad, tp - pad, r + pad, b + pad))
            dl.rounded_rectangle((l - pad, tp - pad + dy, r + pad, b + pad + dy), radius=pad * 2,
                                 fill=(11, 11, 12, a(210 * reveal)))
            dl.text((hx + t * 0.7, hy + dy), text, font=f_score, anchor="lm",
                    fill=col + (a(255 * reveal),))
    # A soft glow under the lines, blurred at quarter size to stay cheap.
    small = lines.resize((max(1, lines.width // 4), max(1, lines.height // 4)), Image.BILINEAR)
    halo = small.filter(ImageFilter.GaussianBlur(max(1.0, t * 0.09))).resize(lines.size,
                                                                            Image.BILINEAR)
    frame.alpha_composite(halo)
    frame.alpha_composite(lines)
    frame.alpha_composite(labels)


def pill(frame, text, W, H):
    f = font(round(min(W, H) * 0.028))
    d = ImageDraw.Draw(frame, "RGBA")
    x, y = round(W * 0.05), round(H * 0.11)
    l, t, r, b = d.textbbox((x, y), text, font=f)
    pad = round(f.size * 0.45)
    d.rounded_rectangle((l - pad, t - pad, r + pad, b + pad), radius=pad * 2,
                        fill=(11, 11, 12, 200), outline=(255, 255, 255, 60), width=2)
    d.text((x, y), text, font=f, fill=(245, 245, 245, 255))


def note(frame, text, W, H):
    """The small honesty note, top right, level with the pill."""
    f = font(round(min(W, H) * 0.022))
    d = ImageDraw.Draw(frame, "RGBA")
    x, y = round(W * 0.95), round(H * 0.11 + min(W, H) * 0.003)
    l, t, r, b = d.textbbox((x, y), text, font=f, anchor="ra")
    pad = round(f.size * 0.45)
    d.rounded_rectangle((l - pad, t - pad, r + pad, b + pad), radius=pad * 2,
                        fill=(11, 11, 12, 150))
    d.text((x, y), text, font=f, fill=(235, 235, 235, 255), anchor="ra")


# --- the clip -------------------------------------------------------------------
def render_clip(main, decision, rows, out, args, painter, still=False):
    W, H = parse_size(args.size)
    SS = 2
    bw, bh = painter.size_tiles
    t = min(W * 0.96 / bw, H * 0.96 / bh) * SS
    ox, oy = (W * SS - bw * t) / 2, (H * SS - bh * t) / 2
    fps, T = args.fps, decision["tick"]
    n_opts = len(rows)
    title = f"Lookahead: {n_opts} options × {decision['horizon']} s"

    lead = round(args.lead * fps)        # the real match, up to the decision
    grow = round(1.8 * args.think * fps)     # futures grow
    reveal = round(0.7 * args.think * fps)   # scores appear
    focus = round(0.9 * args.think * fps)    # the chosen one lights up
    hold = round(0.8 * args.think * fps)
    follow = round(args.follow * fps)    # the real match plays on
    per_tick = 10.0 / fps                # 1x real time

    def board(q):
        img = Image.new("RGBA", (W * SS, H * SS), BG + (255,))
        size = (round(bw * t), round(bh * t))
        img.paste(painter.background(t, size), (round(ox), round(oy)))
        painter.draw(img, round(ox), round(oy), t, main, main.at(q))
        return img

    frames = []
    for f in range(lead):
        frames.append(("play", T - (lead - f) * per_tick, None))
    think = grow + reveal + focus + hold
    for f in range(think):
        g = ease_in_out(f / grow) if f < grow else 1.0
        rv = ease_in_out((f - grow) / reveal) if f >= grow else 0.0
        fc = ease_in_out((f - grow - reveal) / focus) if f >= grow + reveal else 0.0
        frames.append(("think", T, (g, rv, fc)))
    for f in range(follow):
        frames.append(("follow", T + f * per_tick, (1.0, 1.0, 1.0, 1.0 - min(1.0, f / fps))))

    frozen = None

    def render(kind, q, p):
        nonlocal frozen
        if kind == "think":
            if frozen is None:
                frozen = board(T)
                shade = Image.new("RGBA", frozen.size, (0, 0, 0, 70))
                frozen.alpha_composite(shade)
            img = frozen.copy()
            draw_futures(img, painter, round(ox), round(oy), t, rows, decision, *p)
        else:
            img = board(q)
            if kind == "follow" and p[3] > 0:
                # The chosen future stays up briefly, fading, as the real one
                # plays out beside it.
                ghost = Image.new("RGBA", img.size, (0, 0, 0, 0))
                draw_futures(ghost, painter, round(ox), round(oy), t, rows, decision,
                             1.0, 1.0, 1.0, only=decision["chosen"])
                alpha = ghost.getchannel("A").point(lambda v: round(v * p[3] * 0.6))
                ghost.putalpha(alpha)
                img.alpha_composite(ghost)
        out_img = img.reduce(SS).convert("RGB")
        played = rows[decision["chosen"]]["name"]
        pill(out_img, {"play": decision.get("title", "Real match"), "think": title,
                       "follow": f"Played: {played}"}[kind], W, H)
        if decision.get("staged"):
            note(out_img, "Staged in the engine", W, H)
        return out_img

    if still:
        # Two stills: every option with its score, then the pick lit up.
        path = os.path.splitext(out)[0] + ".png"
        render(*frames[lead + grow + reveal - 1]).save(os.path.splitext(out)[0] + "_options.png")
        render(*frames[lead + grow + reveal + focus]).save(path)
        return path
    with VideoWriter(out, (W, H), fps, args.ffmpeg, args.crf) as vid:
        for kind, q, p in frames:
            vid.write(render(kind, q, p))
    # The phase boundaries, for syncing a voiceover to them (edit/edit_short.py).
    beats = {"freeze": lead, "best": lead + grow + reveal, "resume": lead + think}
    with open(os.path.splitext(out)[0] + ".beats.json", "w", encoding="utf-8") as fh:
        json.dump({"fps": fps, **{k: {"frame": f, "seconds": round(f / fps, 3)}
                                  for k, f in beats.items()}}, fh, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", default="model_weights_final_ep83128.pth",
                    help="checkpoint in python_ai/ (or an absolute path)")
    ap.add_argument("--seed", type=int, default=0, help="which match to play")
    ap.add_argument("--opp-elixir", type=float, default=1.5,
                    help="the built-in bot's elixir multiplier (1.5 = the "
                         "condition the 0.625 -> 0.944 result was measured in)")
    ap.add_argument("--max-ticks", type=int, default=3600)
    ap.add_argument("--clips", type=int, default=3, help="decisions to render")
    ap.add_argument("--size", default="vertical")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--lead", type=float, default=2.0,
                    help="seconds of real match before each decision")
    ap.add_argument("--follow", type=float, default=3.0,
                    help="seconds of real match after it")
    ap.add_argument("--think", type=float, default=1.0,
                    help="stretch the frozen 'thinking' part (futures growing, scores, "
                         "the pick) by this factor, e.g. 2.5 under a long voice line")
    ap.add_argument("--sweep", type=int, default=0,
                    help="play this many matches (from --seed on) and rank every "
                         "decision as a showcase, instead of rendering")
    ap.add_argument("--want", default="Hog Rider",
                    help="--sweep: the card the showcase should end on")
    ap.add_argument("--one-per-card", action="store_true",
                    help="draw one future per card (its best cell) instead of every "
                         "candidate: fewer, more different trails")
    ap.add_argument("--tick", type=int, default=None,
                    help="render the decision at this tick (from --sweep) instead of "
                         "picking automatically")
    ap.add_argument("--board", default=None,
                    help="a staged board file (tools/promo/boards/*.json) instead of a "
                         "real match; see the module docstring")
    ap.add_argument("--horizon", type=int, default=None,
                    help="--board: seconds each future is played ahead (default: the "
                         "board's \"horizon\", else the measured search's 4)")
    ap.add_argument("--still", action="store_true",
                    help="one PNG per clip at the moment the choice lights up")
    ap.add_argument("--wide-proposals", action="store_true",
                    help="use today's widened candidate sweep (~97 futures) "
                         "instead of the measured top-k search (<= 7)")
    ap.add_argument("--ffmpeg", default=None)
    ap.add_argument("--crf", type=int, default=18)
    args = ap.parse_args()

    if not args.wide_proposals:
        # Read by python_ai/search/config.py at import, which play_match does
        # lazily, so this lands before it. 0 means no head is ever "flat".
        os.environ["CLASH_WIDE_PROPOSAL_TOP1"] = "0"

    workdir = tempfile.mkdtemp(prefix="ghost-trails-")
    try:
        t0 = time.time()
        if args.sweep:
            sweep(args, workdir)
            return
        if args.board:
            path, decision = staged_match(args, workdir)
            main_replay = Replay(path)
            painter = BoardPainter(load_arena())
            os.makedirs(CLIP_DIR, exist_ok=True)
            decision["futures"] = futures(decision, workdir, "staged")
            rows = describe(decision, main_replay)
            name = os.path.splitext(os.path.basename(args.board))[0]
            written = render_clip(main_replay, decision, rows, str(CLIP_DIR / f"staged_{name}.mp4"),
                                  args, painter, still=args.still)
            for row, shown in zip(rows, decision["shown"]):
                print(f"  {row['name']:10s} {score_text(shown):>6s} vs doing nothing | "
                      f"pulls {len(row['moved'])} enemy troop(s), kills {len(row['kills'])}")
            print(f"  wrote {written}")
            return
        path, decisions = play_match(args, workdir)
        main_replay = Replay(path)
        if args.tick is not None:
            chosen = [d for d in decisions if d["tick"] == args.tick]
            if not chosen:
                near = sorted(decisions, key=lambda d: abs(d["tick"] - args.tick))[:3]
                raise SystemExit(f"no decision at tick {args.tick} with seed {args.seed}; "
                                 f"nearest: {[d['tick'] for d in near]}")
        else:
            chosen = pick(decisions, main_replay, args.clips)
        if not chosen:
            raise SystemExit("no decision in this match had two moving candidates "
                             "to compare; try another --seed")
        painter = BoardPainter(load_arena())
        os.makedirs(CLIP_DIR, exist_ok=True)
        print(f"played in {time.time() - t0:.0f}s; rendering {len(chosen)} clip(s)")
        for k, d in enumerate(chosen, 1):
            d["futures"] = futures(d, workdir, f"d{k}")
            rows = describe(d, main_replay)
            if args.one_per_card:
                d, rows = one_per_card(d, rows)
            out = str(CLIP_DIR / (f"seed{args.seed}_tick{d['tick']}.mp4" if args.tick is not None
                                  else f"seed{args.seed}_clip{k}_tick{d['tick']}.mp4"))
            opts = ", ".join(f"{i + 1}:{r['name']} {score_text(s)}"
                             for i, (r, s) in enumerate(zip(rows, d["scores"])))
            print(f"  tick {d['tick']}: network wanted #1, search chose "
                  f"#{d['chosen'] + 1} ({rows[d['chosen']]['name']}) | {opts}")
            written = render_clip(main_replay, d, rows, out, args, painter, still=args.still)
            print(f"  wrote {written}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
