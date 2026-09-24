"""Extract the MEASURED data behind the RESEARCH.md figures.

Writes docs/results/figures/data/*.json. make_research_figures.py then renders
the SVGs from those files plus the recorded values it carries (each tagged
with its source in DECISIONS.md). Split in two so the plotting step needs
nothing but the standard library.

Run with the training venv (engine + torch + tensorboard, Python 3.11):

    python_ai/venv/Scripts/python.exe tools/figures/extract_figure_data.py

Read-only with respect to the repository's training state: it loads
python_ai/model_weights_live.pth for inference and reads runs/phase9 logs; it
never writes a checkpoint.
"""
import copy
import glob
import json
import os
import random
import statistics
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
import python_ai  # noqa: E402,F401
import clash_royale_env as cr  # noqa: E402

OUT = os.path.join(REPO_ROOT, "docs", "results", "figures", "data")
DECK = [15, 6, 25, 40, 24, 72, 33, 7]
FIREBALL = 7
E = cr.ClashRoyaleEnv
H, W, C = E.BOARD_HEIGHT, E.BOARD_WIDTH, E.NUM_CHANNELS
PLANE = H * W


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))
    print(f"  wrote {os.path.relpath(path, REPO_ROOT)}")


# The benchmark's action sampler: a random affordable card at a random legal
# cell with p=0.5, else wait. Imported rather than copied.
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "benchmark"))
from bench_engine import pick  # noqa: E402


def seeded_env(seed):
    env = E(DECK, DECK, 3600)
    env.seed(seed)
    return env, random.Random(seed)


def entity_count(obs):
    """Units on the board, from the COUNT channels (1/5 per unit, capped at 5
    per cell); includes towers and buildings."""
    n = 0.0
    for ch in (E.CH_COUNT, E.CH_COUNT + 1):
        base = ch * PLANE
        n += sum(obs[base:base + PLANE]) * 5.0
    return round(n)


# --- Figure 3.2: one real observation ---------------------------------------
def fig_observation(seeds=40, first=30, last=150):
    """The busiest board among seeds 1..`seeds`, decisions `first`..`last`, so
    the channel panels show a real fight rather than an empty arena."""
    best = (-1, None, None)
    for seed in range(1, seeds + 1):
        env, rng = seeded_env(seed)
        for k in range(last):
            if k >= first:
                n = entity_count(env.get_observation_for_team(0))
                if n > best[0]:
                    best = (n, seed, k)
            if env.step_self_play_fast(*pick(env, 0, rng), *pick(env, 1, rng), 10).done:
                break
    _, seed, k = best
    env, rng = seeded_env(seed)
    for _ in range(k):
        env.step_self_play_fast(*pick(env, 0, rng), *pick(env, 1, rng), 10)
    obs = env.get_observation_for_team(0)
    chans = [[obs[c * PLANE + y * W: c * PLANE + (y + 1) * W] for y in range(H)]
             for c in range(C)]
    sections = [
        ["Spatial channels (21 x 34 x 18)", 0, PLANE * C],
        ["Own elixir", PLANE * C, 1],
        ["Hand costs", PLANE * C + 1, E.HAND_SIZE],
        ["Hand identity (4 x 185)", PLANE * C + 1 + E.HAND_SIZE, E.HAND_SIZE * E.NUM_CARD_IDS],
        ["Extra scalars", E.EXTRA_SCALARS_START, E.NUM_EXTRA_SCALARS],
        ["Opponent card cycle (2 x 185)", E.CYCLE_START, E.CYCLE_BLOCK_SIZE],
    ]
    save("fig3_2_observation.json", {
        "seed": seed, "tick": env.get_current_tick(), "units": best[0], "size": len(obs),
        "channels": chans, "sections": sections,
        "cycle_seen": obs[E.CYCLE_START:E.CYCLE_START + E.NUM_CARD_IDS],
    })


# --- Figure 2.3(a): per-tick cost against live entities ---------------------
def fig_tick_cost(seeds=20, reps=3):
    """Per-decision cost of step_self_play_fast (10 ticks) against the number of
    units on the board, minimum over interleaved repeats of the same seeded
    match (same workload each repeat), then pooled into entity-count bins."""
    per_key = {}
    for r in range(reps):
        for i in range(seeds):
            env, rng = seeded_env(1000 + i)
            k, done = 0, False
            while not done:
                n = entity_count(env.get_observation_for_team(0))
                a0, a1 = pick(env, 0, rng), pick(env, 1, rng)
                s = time.perf_counter()
                res = env.step_self_play_fast(*a0, *a1, 10)
                dt = (time.perf_counter() - s) / 10 * 1e6
                done = res.done
                key = (i, k)
                prev = per_key.get(key)
                per_key[key] = (n, dt if prev is None else min(prev[1], dt))
                k += 1
    bins = {}
    for n, us in per_key.values():
        b = min(n // 4 * 4, 40)
        bins.setdefault(b, []).append(us)
    rows = []
    for b in sorted(bins):
        v = sorted(bins[b])
        if len(v) < 30:
            continue
        rows.append({"entities_lo": b, "n": len(v), "median_us": statistics.median(v),
                     "p25_us": v[len(v) // 4], "p75_us": v[(3 * len(v)) // 4]})
    save("fig2_3_tick_cost.json", {"seeds": seeds, "reps": reps, "bins": rows})


# --- Figure 3.3: the placement head -----------------------------------------
def phase_r2(grid, period=4, border=4):
    """Share of the interior's variance explained by (x mod p, y mod p), and the
    interior's range.

    border=4: in the shipped head, two nearest-upsamples followed by three
    zero-padded 3x3 convolutions let the padding reach 4 cells in from every
    edge of the 36x20 map (and the 34x18 crop keeps the top/left edge), so only
    rows 4..29, columns 4..13 see a truly constant neighbourhood.
    """
    pts = [(x, y, grid[y][x]) for y in range(border, len(grid) - border)
           for x in range(border, len(grid[0]) - border)]
    mean = statistics.fmean(v for _, _, v in pts)
    tot = sum((v - mean) ** 2 for _, _, v in pts)
    groups = {}
    for x, y, v in pts:
        groups.setdefault((x % period, y % period), []).append(v)
    gm = {k: statistics.fmean(v) for k, v in groups.items()}
    expl = sum((gm[(x % period, y % period)] - mean) ** 2 for x, y, _ in pts)
    return (expl / tot) if tot > 1e-18 else 0.0, max(v for *_, v in pts) - min(v for *_, v in pts)


def fig_heads():
    import torch
    import torch.nn as nn
    from python_ai.models.policy_io import load_net

    weights = os.path.join(REPO_ROOT, "python_ai", "model_weights_live.pth")
    net = load_net(weights, "cpu")
    torch.manual_seed(0)

    # (a) Constant input through the coarse path: the shipped (trained) head
    # against a replica of the pre-2026-08-09 transposed-convolution head.
    const = torch.full((1, 32, net.pooled_h, net.pooled_w), 0.5)
    ctx = torch.randn(1, 32) * 0.5
    with torch.no_grad():
        new = net.place_up(const + ctx.view(1, 32, 1, 1))[0, 0, :H, :W]
        replica = nn.Sequential(nn.ConvTranspose2d(32, 16, 2, 2), nn.ReLU(),
                                nn.ConvTranspose2d(16, 1, 2, 2))
        old = replica(const + ctx.view(1, 32, 1, 1))[0, 0, :H, :W]
    new_l, old_l = new.tolist(), old.tolist()
    r2_new, rng_new = phase_r2(new_l)
    r2_old, rng_old = phase_r2(old_l)

    # (b) A trained head on a real board: Fireball's placement distribution with
    # the full-resolution branch, and with that branch's output zeroed.
    coarse = copy.deepcopy(net)
    with torch.no_grad():
        coarse.place_hires[-1].weight.zero_()
        coarse.place_hires[-1].bias.zero_()

    board = None
    for seed in range(1, 200):
        env, rng = seeded_env(seed)
        hid = (torch.zeros(1, net.LSTM_HIDDEN), torch.zeros(1, net.LSTM_HIDDEN))
        done = False
        while not done:
            obs = env.get_observation_for_team(0)
            ot = torch.tensor([obs], dtype=torch.float32)
            with torch.no_grad():
                feats, embeds, smap, hmap = net.extract_features_hires(ot)
                _, _, _, _, hid = net.step_lstm_and_card(feats, hid, net.affordability_mask(ot))
            hand = env.get_hand_for_team(0)
            enemy = sum(obs[c * PLANE + i] for c in (4, 5, 6) for i in range(PLANE))
            if (env.get_current_tick() >= 400 and FIREBALL in hand
                    and env.get_elixir_for_team(0) >= 4.0 and enemy > 0.6):
                slot = hand.index(FIREBALL)
                with torch.no_grad():
                    cyc = net.cycle_features(ot, detached=True)
                    idx = torch.tensor([slot])
                    full = net.placement_given_card(hid[0], embeds, idx, ot, smap,
                                                    hires_map=hmap, cycle_feat=cyc)
                    crs = coarse.placement_given_card(hid[0], embeds, idx, ot, smap,
                                                      hires_map=hmap, cycle_feat=cyc)
                pf = torch.softmax(full[0], -1).view(H, W).tolist()
                pc = torch.softmax(crs[0], -1).view(H, W).tolist()
                enemy_map = [[sum(obs[c * PLANE + y * W + x] for c in (4, 5, 6))
                              for x in range(W)] for y in range(H)]
                board = {"seed": seed, "tick": env.get_current_tick(), "slot": slot,
                         "p_full": pf, "p_coarse": pc, "enemy_troop_hp": enemy_map,
                         "top1_full": max(max(r) for r in pf),
                         "top1_coarse": max(max(r) for r in pc)}
                break
            res = env.step_self_play_fast(*pick(env, 0, rng), *pick(env, 1, rng), 10)
            done = res.done
        if board:
            break

    save("fig3_3_heads.json", {
        "weights": "python_ai/model_weights_live.pth",
        "constant_input": {"shipped_head": new_l, "replica_convtranspose": old_l,
                           "phase_r2_shipped": r2_new, "phase_r2_replica": r2_old,
                           "interior_range_shipped": rng_new,
                           "interior_range_replica": rng_old},
        "fireball_board": board,
    })


# --- Figure 4.3: the phase-9 curriculum ------------------------------------
def fig_curriculum():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    files = sorted(glob.glob(os.path.join(REPO_ROOT, "runs", "phase9",
                                          "events.out.tfevents.*")))
    runs = []
    for f in files:
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        runs.append({t: [(e.step, e.value) for e in ea.Scalars(t)]
                     for t in ea.Tags()["scalars"]})

    # Each file is one process; a resume restarts from the last checkpoint, so
    # a file's tail past the next file's first step was discarded work.
    def stitch(tag):
        out = []
        for i, r in enumerate(runs):
            pts = r.get(tag, [])
            nxt = None
            for later in runs[i + 1:]:
                if later.get(tag):
                    nxt = later[tag][0][0]
                    break
            out += [(s, v) for s, v in pts if nxt is None or s < nxt]
        return out

    decks = sorted({t.split("/", 2)[2] for r in runs for t in r
                    if t.startswith("Decks/WinRate/")})
    per_deck = {d: dict(stitch(f"Decks/WinRate/{d}")) for d in decks}
    steps = sorted(set().union(*[set(v) for v in per_deck.values()]))
    unweighted, worst = [], []
    for s in steps:
        vals = [per_deck[d][s] for d in decks if s in per_deck[d]]
        if len(vals) == len(decks):
            unweighted.append((s, statistics.fmean(vals)))
            worst.append((s, min(vals)))
    save("fig4_3_curriculum.json", {
        "source": "runs/phase9 TensorBoard logs",
        "rung": stitch("Training/Curriculum_Stage"),
        "win_rate_100": stitch("Training/Win_Rate_100"),
        "per_deck_unweighted_mean": unweighted,
        "per_deck_worst": worst,
        "plateau_advances": [s for s, _ in stitch("Training/Curriculum_PlateauAdvances")],
        "demotions": [s for s, _ in stitch("Training/Curriculum_Demotions")],
    })


if __name__ == "__main__":
    only = set(sys.argv[1:])
    for name, fn in (("observation", fig_observation), ("tick_cost", fig_tick_cost),
                     ("heads", fig_heads), ("curriculum", fig_curriculum)):
        if not only or name in only:
            print(f"[{name}]")
            fn()
