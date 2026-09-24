"""Render the twelve RESEARCH.md figures as light and dark SVGs.

Standard library only. Two kinds of input, and each figure's caption in
RESEARCH.md says which it uses:

  * MEASURED for the report: docs/results/figures/data/fig*.json, written by
    extract_figure_data.py (engine, shipped checkpoint, phase-9 logs), and
    docs/results/2026-09-05-deck-matchups-ep83128.csv (JSON).
  * RECORDED: numbers already published in docs/DECISIONS.md or docs/TODO.md,
    re-plotted from the RECORDED dict below. Each entry names its source. They
    are re-plotted rather than re-measured where the checkpoint or engine
    version that produced them no longer exists.

    python tools/figures/make_research_figures.py      # any Python 3.8+

Writes docs/results/figures/fig-*-{light,dark}.svg and
docs/results/figures/data/recorded_values.json (the table view of every
re-plotted number).
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from svgplot import Axes, Canvas, THEMES, diverging, fmt, ramp  # noqa: E402

OUT = os.path.join(REPO_ROOT, "docs", "results", "figures")
DATA = os.path.join(OUT, "data")
H, W = 34, 18

RECORDED = {
    "fig2_2_time_scale_sweep": {
        "source": "DECISIONS.md, UPSTREAM item 9, 'The time-scale sweep' (pre-fix, n = 158 per cell)",
        "scales": [0.1, 0.15, 0.2, 0.25, 0.33, 0.5, 1.0],
        "iou_1.0s": [0.187, 0.187, 0.216, 0.216, 0.203, 0.141, 0.114],
        "iou_2.0s": [0.138, 0.134, 0.143, 0.152, 0.122, 0.111, 0.091],
    },
    "fig2_2_speed_ratio": {
        "source": "DECISIONS.md, UPSTREAM item 9, 'Verified three ways after the change'",
        "median_before_after": [6.5, 1.3], "p90_before_after": [3.8, 0.8],
        "time_scale_optimum_before_after": [0.2, 1.0],
    },
    "fig2_3_collision_bench": {
        "source": "DECISIONS.md, '2026-08-26 (perf)': resolveCollisions + resolvePositionAgainstBuildings, us per tick",
        "entities": [20, 32, 48], "old_us": [4.190, 10.657, 19.696], "new_us": [1.638, 4.085, 6.482],
    },
    "fig3_4_aux_gradient": {
        "source": "CLAUDE.md, 2026-09-15 audit: aux gradient vs all other terms, 12 updates from random init",
        "lstm_ratio_update1_update12": [0.62, 1.67], "trunk_ratio_update1_update12": [0.63, 2.01],
        "lstm_cosine_update1_update12": [-0.37, -0.84], "trunk_cosine_update12": -0.90,
    },
    "fig3_4_next_card_ce": {
        "source": "CLAUDE.md, 'THE STALE AUX HEAD' (2026-09-03)",
        "mirror_run_range": [1.45, 1.58], "pool_resume": 13.76, "after_reset_first_update": 4.30,
        "ln8": math.log(8), "ln185": math.log(185),
    },
    "fig5_1_cannon": {
        "source": "DECISIONS.md, 2026-08-06/09 (ep ~130k) and 2026-08-14/15 (Table 5.3: n = 1,582 paired states)",
        "stages": [
            {"label": "A. Reward mispricing + checkerboard (ep ~130k)", "cells": [[11, 2], [11, 3]], "share": 27.9},
            {"label": "B. Coverage freeze (seed)", "cells": [[11, 0]], "share": 44.9, "top1": 0.140},
            {"label": "C. + coverage entropy (80 updates)", "cells": [[11, 0]], "share": 27.1, "top1": 0.050},
            {"label": "D. + advisor target (524 updates)", "cells": [[16, 15]], "share": 17.0, "top1": 0.062},
        ],
        "hp_preserved": {"B. seed": 276.0, "C. coverage entropy": 403.0, "D. advisor target": 553.1},
        "random_cell": 411.3, "advisor": 687.9,
    },
    "fig5_2_catch_distribution": {
        "source": "CLAUDE.md, elixir-phase table (same policy, same 24 episodes)",
        "p_catch_ge3_flat_phased": [15.5, 21.8], "p_catch_ge4_flat_phased": [6.1, 10.4],
        "median_catch_flat_phased": [1, 1],
    },
    "fig5_3_marginal_value": {
        "source": "DECISIONS.md, '2026-08-19: the curriculum pivot' (gate sweep) and '2026-08-19 (later): DEPLOY TIME'",
        "rows": [
            ["Lone Hog, commit at opp. elixir <= 7.0", -298.2, -494, -107, "old engine"],
            ["Lone Hog, commit at opp. elixir <= 3.0", -268.6, -455, -90, "old engine"],
            ["Lone Hog, commit at opp. elixir <= 1.5", -585.4, -816, -360, "old engine"],
            ["Supported push, no deploy time", -73.7, -349.5, 195.3, "controlled A/B"],
            ["Supported push, 1 s deploy time", 448.5, 137.3, 760.1, "controlled A/B"],
            ["Escort vs lone Hog, punish window", 650.0, 429, 878, "deploy time on"],
        ],
    },
    "fig5_4_search": {
        "source": "DECISIONS.md, 'Search horizon sweep' (heuristic @1.5x, n = 80 paired each); TODO.md 0e (teacher, rung 3, n = 32; follow-up harness n = 30)",
        "heuristic_h": [4, 8, 12, 20], "heuristic_greedy": [0.483, 0.525, 0.563, 0.488],
        "heuristic_search": [0.667, 0.925, 0.963, 0.875],
        "teacher_h": [4, 8, 12], "teacher_delta": [-0.313, -0.531, -0.469],
        "teacher_ci": [[-0.531, -0.125], [-0.719, -0.313], [-0.688, -0.250]],
        "followup": {"h4": {"heuristic_rollout": [-0.167, -0.400, 0.067], "teacher_rules_rollout": [-0.067, -0.233, 0.100]},
                     "h12": {"heuristic_rollout": [-0.469, None, None], "teacher_rules_rollout": [-0.333, -0.600, -0.067]}},
    },
    "fig5_5_bankruptcy": {
        "source": "DECISIONS.md, 2026-08-16 ('disconnected zombie') and the ep-600 first read",
        "p_play_given_affordable": {"before (old normaliser, Giant deck)": [0.3655, 0.4719],
                                    "after (fixed, 2.6 deck, ep 600)": [0.4606, 0.8601]},
        "legal_arms_share_pct": {"2": 54.1, "3": 15.6, "4": 25.8, "5": 4.5},
        "old_target_nats": 0.35 * math.log(5),
    },
}


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


def write(name, draw):
    os.makedirs(OUT, exist_ok=True)
    for theme in ("light", "dark"):
        cv = draw(theme)
        with open(os.path.join(OUT, f"{name}-{theme}.svg"), "w", encoding="utf-8") as f:
            f.write(cv.svg())
    print(f"  {name}")


def board(cv, x, y, cell, grid_fn, outline=True, flip=True):
    """Draw a 34 x 18 board; grid_fn(xc, yc) -> colour or None. Own back row
    (y = 0) at the bottom when flip."""
    t = cv.t
    if outline:
        cv.rect(x, y, W * cell, H * cell, t["surface"])
    for yc in range(H):
        for xc in range(W):
            col = grid_fn(xc, yc)
            if col is None:
                continue
            py = y + ((H - 1 - yc) if flip else yc) * cell
            cv.rect(x + xc * cell, py, cell, cell, col)
    if outline:
        cv.add(f'<rect x="{x:.2f}" y="{y:.2f}" width="{W * cell:.2f}" height="{H * cell:.2f}" '
               f'fill="none" stroke="{t["axis"]}" stroke-width="1"/>')


def river(cv, x, y, cell):
    """The river band, rows 15.5-17.5 -> cells 16 and 17 (y from the bottom)."""
    t = cv.t
    for yc in (16, 17):
        cv.rect(x, y + (H - 1 - yc) * cell, W * cell, cell, t["river"])


# ---------------------------------------------------------------------------
def fig_2_2(theme):
    r1, r2 = RECORDED["fig2_2_time_scale_sweep"], RECORDED["fig2_2_speed_ratio"]
    cv = Canvas(880, 420, theme, "Figure 2.2 - movement-speed calibration",
                "Pre-fix time-scale sweep and the engine:real speed ratio before and after the fix.")
    t = cv.t
    cv.heading("The engine ran 4-5x too fast, and footage said so three ways",
               sub="Recorded measurements (DECISIONS.md, UPSTREAM item 9)")
    ax = Axes(cv, 80, 100, 430, 240, (0, len(r1["scales"]) - 1), (0.08, 0.24))
    ax.title("(a) Agreement with footage vs engine time-scale, before the fix")
    ax.grid_y([0.10, 0.14, 0.18, 0.22], lambda v: f"{v:.2f}", "IoU with real footage")
    step = ax.w / (len(r1["scales"]) - 1)
    for i, s in enumerate(r1["scales"]):
        cv.text(ax.x + i * step, ax.y + ax.h + 17, f"x{s:g}", size=11, color=t["muted"], anchor="middle")
    cv.text(ax.x + ax.w / 2, ax.y + ax.h + 37, "time-scale factor (x1 = the engine as it was)", size=12,
            color=t["ink2"], anchor="middle")
    for key, col, lab in (("iou_1.0s", t["series"][0], "1.0 s horizon"), ("iou_2.0s", t["series"][1], "2.0 s horizon")):
        pts = [(ax.x + i * step, ax.sy(v)) for i, v in enumerate(r1[key])]
        cv.polyline(pts, col)
        for (px, py), v in zip(pts, r1[key]):
            cv.dot(px, py, col, r=4, tip=f"{lab}: {v:.3f}")
        cv.text(pts[0][0] + 6, pts[0][1] - 12, lab, size=11, color=t["ink2"])
    opt_x = ax.x + 2.5 * step
    cv.text(opt_x, ax.y + 14, "peak at x0.2-0.25: a 4-5x divisor", size=11, color=t["ink2"], anchor="middle")
    cv.legend(ax.x, ax.y + ax.h + 62, [("1.0 s horizon", t["series"][0], "line"), ("2.0 s horizon", t["series"][1], "line")])

    bx = Axes(cv, 620, 100, 220, 240, (-0.5, 1.5), (0, 7))
    bx.title("(b) Engine : real speed ratio")
    bx.grid_y([0, 1, 2, 4, 6], lambda v: f"{v:g}x")
    bx.xcats(["before", "after fix"])
    cv.line(bx.x, bx.sy(1), bx.x + bx.w, bx.sy(1), t["axis"], 1.5)
    cv.text(bx.x + 4, bx.sy(1) + 14, "real game = 1x", size=11, color=t["ink2"])
    for key, col, lab in (("median_before_after", t["series"][0], "median"), ("p90_before_after", t["series"][1], "p90")):
        v = r2[key]
        pts = [(bx.x + bx.w * 0.25, bx.sy(v[0])), (bx.x + bx.w * 0.75, bx.sy(v[1]))]
        cv.polyline(pts, col)
        for (px, py), vv in zip(pts, v):
            cv.dot(px, py, col, tip=f"{lab}: {vv}x")
        cv.text(pts[0][0] - 10, pts[0][1] + 4, f"{lab} {v[0]}x", size=11, color=t["ink2"], anchor="end")
        cv.text(pts[1][0] + 10, pts[1][1] + (-4 if key.startswith("median") else 12), f"{v[1]}x", size=11, color=t["ink2"])
    cv.text(bx.x, bx.y + bx.h + 62, "Time-scale optimum at 1.0 s: x0.2 -> x1.0 after the fix", size=11, color=t["ink2"])
    return cv


def fig_2_3(theme):
    d = load("fig2_3_tick_cost.json")
    rec = RECORDED["fig2_3_collision_bench"]
    cv = Canvas(880, 420, theme, "Figure 2.3 - per-tick cost",
                "Measured per-tick engine cost by live-entity count, and the recorded collision hot-path rework.")
    t = cv.t
    cv.heading("Per-tick cost grows with the board; the collision rework cut it 2.6-3x",
               sub="(a) measured 2026-09-24 through the bindings  ·  (b) recorded, tools/audit/collision_bench.cpp")
    bins = d["bins"]
    ax = Axes(cv, 80, 100, 400, 240, (0, len(bins)), (0, 10))
    ax.title("(a) Engine cost per tick vs units on the board")
    ax.grid_y([0, 2, 4, 6, 8, 10], lambda v: f"{v:g} µs", None)
    step = ax.xcats([f"{b['entities_lo']}-{b['entities_lo'] + 3}" for b in bins],
                    "units on the board (towers included)")
    col = t["series"][0]
    pts = []
    for i, b in enumerate(bins):
        cx = ax.x + step * (i + 0.5)
        cv.line(cx, ax.sy(b["p25_us"]), cx, ax.sy(b["p75_us"]), col, 2, cap="round")
        pts.append((cx, ax.sy(b["median_us"])))
    cv.polyline(pts, col)
    for (px, py), b in zip(pts, bins):
        cv.dot(px, py, col, tip=f"{b['entities_lo']}-{b['entities_lo'] + 3} units: median {b['median_us']:.2f} us (n={b['n']})")
        cv.text(px + 9, py - 8, f"{b['median_us']:.1f}", size=11, color=t["ink2"])
    cv.text(ax.x, ax.y + ax.h + 60, "Dot = median, bar = interquartile range; per-decision minimum of 3 repeats, 20 seeded matches.",
            size=11, color=t["ink2"])

    bx = Axes(cv, 580, 100, 260, 240, (0, 3), (0, 22))
    bx.title("(b) Collision stages, µs per tick")
    bx.grid_y([0, 5, 10, 15, 20], lambda v: f"{v:g}")
    s2 = bx.xcats([f"{n} entities" for n in rec["entities"]])
    for i, (o, n) in enumerate(zip(rec["old_us"], rec["new_us"])):
        cx = bx.x + s2 * (i + 0.5)
        cv.bar_v(cx - 23, bx.sy(0), 20, bx.sy(o), t["series"][1], tip=f"old: {o:.2f} us")
        cv.bar_v(cx + 3, bx.sy(0), 20, bx.sy(n), t["series"][0], tip=f"new: {n:.2f} us")
        cv.text(cx + 13, bx.sy(n) - 6, f"{o / n:.1f}x", size=11, color=t["ink2"], anchor="middle")
    bx.baseline(0)
    cv.legend(bx.x, bx.y + bx.h + 60, [("old", t["series"][1], "bar"), ("new", t["series"][0], "bar")])
    return cv


CH_NAMES = ["ally melee", "ally ranged", "ally bldg-target.", "ally building",
            "enemy melee", "enemy ranged", "enemy bldg-target.", "enemy building", "river & bridges",
            "ally count", "enemy count", "ally flying", "enemy flying", "ally anti-air", "enemy anti-air",
            "ally DPS", "enemy DPS", "ally range", "enemy range", "ally speed", "enemy speed"]


def fig_3_2(theme):
    d = load("fig3_2_observation.json")
    cv = Canvas(880, 800, theme, "Figure 3.2 - the observation",
                "The 13,977-float observation as sections, and all 21 spatial channels of one real state.")
    t = cv.t
    cv.heading("One observation: 13,977 floats, 92% of them the board",
               sub=f"(a) layout  ·  (b) all 21 channels: seed {d['seed']}, tick {d['tick']}, {d['units']} units on the board, "
                   "team 0's view (own back row at the bottom) — measured")
    total = d["size"]
    x0, y0, wbar = 24, 92, 832
    shades = [t["series"][0], t["series"][1], t["series"][2], t["seq"][2], t["series"][1], t["series"][2]]
    cx = x0
    for (name, off, size), col in zip(d["sections"], shades):
        w = max(2.0, size / total * wbar)
        cv.rect(cx, y0, w - 2, 18, col, rx=3, tip=f"{name}: offset {off:,}, {size:,} floats")
        cx += w
    cv.text(x0, y0 + 36, "spatial 12,852", size=11, color=t["ink2"])
    cv.text(x0 + wbar, y0 + 36, "scalars 1,125 (zoomed below)", size=11, color=t["ink2"], anchor="end")
    tail = total - d["sections"][0][2]
    cx, ty = x0, y0 + 48
    for (name, off, size), col in zip(d["sections"][1:], shades[1:]):
        w = max(3.0, size / tail * wbar)
        cv.rect(cx, ty, w - 2, 14, col, rx=3, tip=f"{name}: offset {off:,}, {size:,} floats")
        if w > 90:
            cv.text(cx + 4, ty + 28, f"{name} ({size:,})", size=10.5, color=t["ink2"])
        cx += w
    cell, cols = 4.6, 7
    gx, gy = 24, 196
    pw, ph = W * cell, H * cell
    for c, grid in enumerate(d["channels"]):
        r_, k = divmod(c, cols)
        px = gx + k * (pw + 38)
        py = gy + r_ * (ph + 36)
        vmax = max(max(row) for row in grid) or 1.0
        board(cv, px, py, cell, lambda xc, yc, g=grid, vm=vmax: (ramp(t["seq"], 0.15 + 0.85 * g[yc][xc] / vm)
                                                                  if g[yc][xc] > 0 else None))
        cv.text(px, py - 6, f"{c}  {CH_NAMES[c]}", size=10.5, color=t["ink2"])
    cv.text(24, 785, "Each channel scaled to its own maximum; empty cells show the surface.",
            size=11, color=t["ink2"])
    return cv


def fig_3_3(theme):
    d = load("fig3_3_heads.json")
    ci, fb = d["constant_input"], d["fireball_board"]
    cv = Canvas(880, 480, theme, "Figure 3.3 - the placement head",
                "Constant-input logit surfaces for a replica transposed-convolution head and the shipped head; "
                "Fireball placement with and without the full-resolution branch.")
    t = cv.t
    cv.heading("The period-4 bias, and what the full-resolution branch adds",
               sub="Measured: shipped head from model_weights_live.pth (ep 115,173); replica = the pre-2026-08-09 design, random init")
    cell = 6.5
    top = 124
    cv.text(24, 98, "(a) Constant input, coarse path", size=12.5, weight="600")
    panels = [
        (24, "Replica transposed conv", ci["replica_convtranspose"],
         ["phase explains", f"{ci['phase_r2_replica'] * 100:.0f}% of variance"]),
        (214, "Shipped upsample + conv", ci["shipped_head"],
         ["interior range", f"{ci['interior_range_shipped']:.0f}: exactly flat"]),
    ]
    for px, name, grid, note in panels:
        flat = [v for row in grid for v in row]
        mean = sum(flat) / len(flat)
        span = max(abs(v - mean) for v in flat) or 1.0
        board(cv, px, top, cell, lambda xc, yc, g=grid: diverging(theme, (g[yc][xc] - mean) / span), flip=False)
        cv.text(px, top - 8, name, size=11)
        cv.text(px, top + H * cell + 16, note[0], size=11, color=t["ink2"])
        cv.text(px, top + H * cell + 30, note[1], size=11, color=t["ink2"])
    cv.text(24, top + H * cell + 52, "Same constant input and context to both; colour = deviation from the",
            size=10.5, color=t["muted"])
    cv.text(24, top + H * cell + 66, "map's mean, blue below, red above. The shipped head's border cells see",
            size=10.5, color=t["muted"])
    cv.text(24, top + H * cell + 80, "zero padding; its interior (rows 4-29, columns 4-13) is exactly flat.",
            size=10.5, color=t["muted"])

    enemy = fb["enemy_troop_hp"]
    cv.text(470, 98, "(b) Fireball placement on one real board", size=12.5, weight="600")
    for i, (key, name) in enumerate((("p_coarse", "Full-res branch zeroed"), ("p_full", "Shipped (both paths)"))):
        px = 470 + i * 200
        grid = fb[key]
        vmax = max(max(r) for r in grid)
        board(cv, px, top, cell, lambda xc, yc: None, outline=True)
        river(cv, px, top, cell)
        board(cv, px, top, cell, lambda xc, yc, g=grid: (ramp(t["seq"], 0.12 + 0.88 * g[yc][xc] / vmax)
                                                           if g[yc][xc] > 1e-4 else None), outline=False)
        for yc in range(H):
            for xc in range(W):
                if enemy[yc][xc] > 0.05:
                    cxp = px + (xc + 0.5) * cell
                    cyp = top + (H - 1 - yc + 0.5) * cell
                    cv.add(f'<circle cx="{cxp:.2f}" cy="{cyp:.2f}" r="{cell * 0.62:.2f}" fill="none" '
                           f'stroke="{t["ink"]}" stroke-width="1.2"/>')
        top1 = fb["top1_" + key.split("_")[1]]
        cv.text(px, top - 8, name, size=11)
        cv.text(px, top + H * cell + 16, f"top-1 probability {top1:.3f}", size=11, color=t["ink2"])
    cv.text(470, top + H * cell + 52, f"Seed {fb['seed']}, tick {fb['tick']}. Rings = enemy troops; band = river;",
            size=10.5, color=t["muted"])
    shade = "darker" if theme == "light" else "brighter"
    cv.text(470, top + H * cell + 66, f"{shade} = more probability; own back row at the bottom. Zeroing the",
            size=10.5, color=t["muted"])
    cv.text(470, top + H * cell + 80, "branch is an ablation of a net trained with both paths.",
            size=10.5, color=t["muted"])
    return cv


def fig_3_4(theme):
    g, ce = RECORDED["fig3_4_aux_gradient"], RECORDED["fig3_4_next_card_ce"]
    cv = Canvas(880, 420, theme, "Figure 3.4 - the auxiliary next-card loss",
                "Aux gradient magnitude relative to all other terms at updates 1 and 12; next-card cross-entropy in three regimes.")
    t = cv.t
    cv.heading("A fresh auxiliary head out-pulls and opposes the policy",
               sub="Recorded measurements (CLAUDE.md, 2026-09-03 and 2026-09-15)")
    ax = Axes(cv, 80, 100, 340, 240, (0, 2), (0, 2.4))
    ax.title("(a) Aux gradient ÷ all other terms combined")
    ax.grid_y([0, 0.5, 1.0, 1.5, 2.0], lambda v: f"{v:.1f}x")
    step = ax.xcats(["LSTM", "CNN trunk"])
    for i, (key, cos_txt) in enumerate((("lstm_ratio_update1_update12", "cosine -0.37 -> -0.84"),
                                        ("trunk_ratio_update1_update12", "cosine -> -0.90"))):
        v = g[key]
        cx = ax.x + step * (i + 0.5)
        cv.bar_v(cx - 23, ax.sy(0), 20, ax.sy(v[0]), t["series"][0], tip=f"update 1: {v[0]}x")
        cv.bar_v(cx + 3, ax.sy(0), 20, ax.sy(v[1]), t["series"][1], tip=f"update 12: {v[1]}x")
        cv.text(cx - 13, ax.sy(v[0]) - 6, f"{v[0]}", size=11, color=t["ink2"], anchor="middle")
        cv.text(cx + 13, ax.sy(v[1]) - 6, f"{v[1]}", size=11, color=t["ink2"], anchor="middle")
        cv.text(cx, ax.y + ax.h + 34, cos_txt, size=10.5, color=t["muted"], anchor="middle")
    cv.line(ax.x, ax.sy(1), ax.x + ax.w, ax.sy(1), t["axis"], 1.5)
    ax.baseline(0)
    cv.legend(ax.x, ax.y + ax.h + 62, [("update 1", t["series"][0], "bar"), ("update 12", t["series"][1], "bar")])

    bx = Axes(cv, 520, 100, 230, 240, (0, 3), (0, 15))
    bx.title("(b) Next-card cross-entropy (nats)")
    bx.grid_y([0, 5, 10, 15], lambda v: f"{v:g}")
    s2 = bx.xcats(["mirror run", "pool resume", "after reset"])
    vals = [ce["mirror_run_range"][1], ce["pool_resume"], ce["after_reset_first_update"]]
    for v, lab in ((ce["ln185"], ["ln 185 = 5.22", "uniform guess"]), (ce["ln8"], ["ln 8 = 2.08", "no cycle knowledge"])):
        cv.line(bx.x, bx.sy(v), bx.x + bx.w, bx.sy(v), t["axis"], 1.5)
        cv.text(bx.x + bx.w + 8, bx.sy(v) - 2, lab[0], size=10.5, color=t["ink2"])
        cv.text(bx.x + bx.w + 8, bx.sy(v) + 11, lab[1], size=10.5, color=t["muted"])
    for i, v in enumerate(vals):
        cx = bx.x + s2 * (i + 0.5)
        cv.bar_v(cx - 10, bx.sy(0), 20, bx.sy(v), t["series"][0], tip=f"{v}")
        lab = "1.45-1.58" if i == 0 else f"{v:.2f}"
        cv.text(cx, bx.sy(v) - 6, lab, size=11, color=t["ink2"], anchor="middle")
    bx.baseline(0)
    return cv


def fig_4_2(theme):
    cv = Canvas(880, 400, theme, "Figure 4.2 - what a win is worth",
                "Discounted value at episode start of a +1 terminal reward, by episode length, for gamma 0.99 and 0.999.")
    t = cv.t
    cv.heading("At γ = 0.99 a win at the end of a match is worth 0.027; a crown pays 0.6",
               sub="Exact: γ^T, the value at episode start of +1 received after T decisions")
    ax = Axes(cv, 80, 96, 740, 230, (0, 400), (0, 1.0))
    ax.grid_y([0, 0.25, 0.5, 0.75, 1.0], lambda v: f"{v:.2f}", "discounted value of +1")
    ax.grid_x([0, 100, 200, 300, 400], lambda v: f"{v:g}", "episode length T (decisions; one decision = 1 s)")
    for gamma, col in ((0.999, t["series"][0]), (0.99, t["series"][1])):
        pts = [(ax.sx(T), ax.sy(gamma ** T)) for T in range(0, 401, 4)]
        cv.polyline(pts, col)
    cv.line(ax.sx(0), ax.sy(0.6), ax.sx(400), ax.sy(0.6), t["axis"], 1.5)
    cv.text(ax.sx(400), ax.sy(0.6) - 6, "one tower destroyed: +0.6, paid when it happens", size=11,
            color=t["ink2"], anchor="end")
    for T, gamma, col, dy in ((112, 0.99, t["series"][1], -10), (360, 0.99, t["series"][1], -10),
                              (360, 0.999, t["series"][0], -10)):
        v = gamma ** T
        cv.dot(ax.sx(T), ax.sy(v), col, tip=f"gamma {gamma}, T {T}: {v:.4f}")
        cv.text(ax.sx(T) + 8, ax.sy(v) + dy, f"{v:.3f}", size=11, color=t["ink2"])
    cv.line(ax.sx(360), ax.y, ax.sx(360), ax.y + ax.h, t["axis"], 1)
    cv.text(ax.sx(360) - 6, ax.y + 12, "full match (360)", size=11, color=t["ink2"], anchor="end")
    cv.text(ax.sx(112) - 10, ax.sy(0.99 ** 112) + 16, "old episodes (~112 decisions)", size=10.5,
            color=t["muted"], anchor="end")
    cv.legend(ax.x, ax.y + ax.h + 56, [("γ = 0.999 (now)", t["series"][0], "line"), ("γ = 0.99 (until 2026-08-27)", t["series"][1], "line")])
    return cv


def fig_4_3(theme):
    d = load("fig4_3_curriculum.json")
    cv = Canvas(880, 520, theme, "Figure 4.3 - the phase-9 curriculum",
                "Rung and three win-rate signals over episodes 83,140 to 115,420 of the phase-9 run.")
    t = cv.t
    cv.heading("The readable win rate stayed flat while every deck improved",
               sub="Measured: runs/phase9 TensorBoard logs (episodes 83,140-115,420)")
    x0, x1 = 83000, 115500
    rung = d["rung"]
    ax = Axes(cv, 80, 92, 740, 90, (x0, x1), (0.5, 4.5))
    ax.title("Rung")
    ax.grid_y([1, 2, 3, 4], lambda v: f"{int(v)}")
    pts = []
    for (s, v), nxt in zip(rung, rung[1:] + [(x1, rung[-1][1])]):
        pts += [(ax.sx(s), ax.sy(v)), (ax.sx(nxt[0]), ax.sy(v))]
    cv.polyline(pts, t["series"][0], 2)
    for s in d["plateau_advances"]:
        cv.path(f"M{ax.sx(s):.2f},{ax.y + ax.h + 2:.2f} l-5,9 l10,0 Z", t["ink2"])
    for s in d["demotions"]:
        cv.path(f"M{ax.sx(s):.2f},{ax.y + ax.h + 11:.2f} l-5,-9 l10,0 Z", t["series"][1])
    cv.text(ax.x + ax.w, ax.y - 12, "▲ plateau advance   ▼ demotion", size=11, color=t["ink2"], anchor="end")

    bx = Axes(cv, 80, 236, 740, 200, (x0, x1), (0, 1))
    bx.title("Win rate")
    bx.grid_y([0, 0.25, 0.5, 0.75, 1.0], lambda v: f"{v:.2f}")
    bx.grid_x([85000, 90000, 95000, 100000, 105000, 110000, 115000], lambda v: f"{v / 1000:g}k", "episode")
    series = [("win_rate_100", t["series"][1], "readable (PFSP-weighted, 100 ep.)", 1.2),
              ("per_deck_unweighted_mean", t["series"][0], "unweighted per-deck mean", 2),
              ("per_deck_worst", t["series"][2], "worst deck", 2)]
    for key, col, lab, wdt in series:
        pts = [(bx.sx(s), bx.sy(v)) for s, v in d[key]]
        cv.polyline(pts, col, wdt)
        sx, sv = d[key][-1]
        cv.dot(bx.sx(sx), bx.sy(sv), col, r=4, tip=f"{lab}: {sv:.3f} at ep {sx}")
    first, last = d["per_deck_unweighted_mean"][0], d["per_deck_unweighted_mean"][-1]
    cv.text(bx.sx(first[0]) + 6, bx.sy(first[1]) + 16, f"{first[1]:.2f}", size=11, color=t["ink2"])
    cv.text(bx.sx(last[0]) + 8, bx.sy(last[1]) + 4, f"{last[1]:.2f}", size=11, color=t["ink2"])
    cv.legend(bx.x, bx.y + bx.h + 60, [(lab, col, "line") for _, col, lab, _ in series])
    return cv


def fig_5_1(theme):
    r = RECORDED["fig5_1_cannon"]
    cv = Canvas(880, 420, theme, "Figure 5.1 - where the Cannon went",
                "Modal Cannon cells at four stages on a schematic of the agent's half, and tower HP preserved.")
    t = cv.t
    cv.heading("The Cannon's modal cell moved from its own back row to the bridge",
               sub="Recorded measurements (DECISIONS.md, 2026-08-06 to 08-15). Those checkpoints were retired with the Giant deck.")
    cell, rows = 12, 18
    bx0, by0 = 40, 96
    cv.rect(bx0, by0, W * cell, rows * cell, t["surface"])
    for yc in (16, 17):
        cv.rect(bx0, by0 + (rows - 1 - yc) * cell, W * cell, cell, t["river"])
    for xc in range(W + 1):
        cv.line(bx0 + xc * cell, by0, bx0 + xc * cell, by0 + rows * cell, t["board_line"], 0.6)
    for yc in range(rows + 1):
        cv.line(bx0, by0 + yc * cell, bx0 + W * cell, by0 + yc * cell, t["board_line"], 0.6)
    for (tx, ty, rr) in ((3.0, 6.0, 1.5), (14.0, 6.0, 1.5), (8.5, 2.5, 2.0)):
        cv.add(f'<circle cx="{bx0 + (tx + 0.5) * cell:.2f}" cy="{by0 + (rows - 1 - ty + 0.5) * cell:.2f}" '
               f'r="{rr * cell:.2f}" fill="none" stroke="{t["tower"]}" stroke-width="1.5"/>')
    letters = ["A", "B", "C", "D"]
    cols = [t["series"][1], t["series"][1], t["series"][1], t["series"][0]]
    offsets = {("B", 11, 0): -7, ("C", 11, 0): 7}
    for st, letter, col in zip(r["stages"], letters, cols):
        for (xc, yc) in st["cells"]:
            cxp = bx0 + (xc + 0.5) * cell + offsets.get((letter, xc, yc), 0)
            cyp = by0 + (rows - 1 - yc + 0.5) * cell
            cv.dot(cxp, cyp, col, r=4.5, tip=f"{letter}: ({xc},{yc}) {st['share']}%")
            cv.text(cxp, cyp - 9, letter, size=10.5, weight="600", anchor="middle")
    cv.text(bx0, by0 + rows * cell + 18, "Own half and river; circles = own towers (schematic,", size=10.5, color=t["muted"])
    cv.text(bx0, by0 + rows * cell + 32, "current arena layout). Orange = pathological, blue = cured.", size=10.5, color=t["muted"])

    lx = bx0 + W * cell + 28
    for i, (st, letter) in enumerate(zip(r["stages"], letters)):
        ly = by0 + 12 + i * 46
        cv.text(lx, ly, f"{st['label']}", size=11.5)
        cells = ", ".join(f"({a},{b})" for a, b in st["cells"])
        extra = f", top-1 {st['top1']:.3f}" if "top1" in st else ""
        cv.text(lx, ly + 16, f"{cells}: {st['share']}% of placements{extra}", size=11, color=t["ink2"])

    hp = r["hp_preserved"]
    ax = Axes(cv, 640, 250, 180, 110, (0, 3), (0, 800))
    ax.title("Tower HP preserved (n = 1,582)")
    ax.grid_y([0, 400, 800], lambda v: f"{v:g}")
    step = ax.w / 3
    for v, lab in ((r["random_cell"], "random 411"), (r["advisor"], "advisor 688")):
        cv.line(ax.x, ax.sy(v), ax.x + ax.w, ax.sy(v), t["axis"], 1.5)
        cv.text(ax.x + 2, ax.sy(v) - 5, lab, size=10.5, color=t["ink2"])
    for i, (k, v) in enumerate(hp.items()):
        cx = ax.x + step * (i + 0.5)
        cv.bar_v(cx - 12, ax.sy(0), 24, ax.sy(v), t["series"][0] if k.startswith("D") else t["series"][1],
                 tip=f"{k}: {v}")
        cv.text(cx, ax.sy(v) - 6, f"{v:g}", size=11, color=t["ink2"], anchor="middle")
        cv.text(cx, ax.y + ax.h + 15, k.split(". ")[0], size=11, color=t["ink2"], anchor="middle")
    ax.baseline(0)
    return cv


VOID = {"dart_bait_cycle", "xbow_30_cycle", "graveyard_control", "classic_log_bait_inferno", "mortar_cycle"}


def fig_5_2(theme):
    with open(os.path.join(REPO_ROOT, "docs", "results", "2026-09-05-deck-matchups-ep83128.csv"), encoding="utf-8") as f:
        res = json.load(f)
    rows = sorted(res["rows"], key=lambda r: r["fb_catch_mean"])
    rec = RECORDED["fig5_2_catch_distribution"]
    cv = Canvas(880, 560, theme, "Figure 5.2 - Fireball opportunity by opponent deck",
                "Best-case Fireball catch per decision and the agent's win rate against each of the 16 pool decks.")
    t = cv.t
    cv.heading("The mirror offers Fireball the least; win rate follows the deck, not the opportunity",
               sub=f"Measured: docs/results/2026-09-05-deck-matchups-ep83128.csv (ep {res['episode']:,}, rung {res['stage']}, "
                   f"{res['episodes_per_deck']} episodes per deck)")
    top, rh = 96, 22
    vmax = max(r["fb_catch_mean"] for r in rows) * 1.08
    ax = Axes(cv, 230, top, 300, rh * len(rows), (0, vmax), (0, 1))
    ax.title("(a) Best-case Fireball catch, enemy HP per decision")
    bx = Axes(cv, 610, top, 230, rh * len(rows), (0, 1), (0, 1))
    bx.title("(b) Agent win rate")
    for v in (0, 250, 500, 750):
        if v <= vmax:
            cv.line(ax.sx(v), ax.y, ax.sx(v), ax.y + ax.h, t["grid"], 1)
            cv.text(ax.sx(v), ax.y + ax.h + 16, f"{v}", size=11, color=t["muted"], anchor="middle")
    for v in (0, 0.5, 1.0):
        cv.line(bx.sx(v), bx.y, bx.sx(v), bx.y + bx.h, t["grid"], 1)
        cv.text(bx.sx(v), bx.y + bx.h + 16, f"{v:g}", size=11, color=t["muted"], anchor="middle")
    for i, r in enumerate(rows):
        y = top + i * rh + 4
        mirror = r["deck"] == "hog_26_mirror"
        col = t["series"][1] if mirror else t["series"][0]
        name = r["deck"].replace("_", " ") + (" †" if r["deck"] in VOID else "")
        cv.text(ax.x - 8, y + 11, name, size=11, color=t["ink"] if mirror else t["ink2"], anchor="end",
                weight="600" if mirror else "normal")
        cv.bar_h(ax.sx(0), y, 14, ax.sx(r["fb_catch_mean"]), col, tip=f"{r['deck']}: {r['fb_catch_mean']:.0f}")
        cv.text(ax.sx(r["fb_catch_mean"]) + 5, y + 11, f"{r['fb_catch_mean']:.0f}", size=10.5, color=t["ink2"])
        cv.bar_h(bx.sx(0), y, 14, bx.sx(max(r["win_rate"], 0.004)), col, tip=f"{r['deck']}: {r['win_rate']:.3f}")
        cv.text(bx.sx(r["win_rate"]) + 5, y + 11, f"{r['win_rate']:.2f}", size=10.5, color=t["ink2"])
    ax.vbaseline(0)
    bx.vbaseline(0)
    yb = top + rh * len(rows) + 36
    cv.text(24, yb, "Orange = the 2.6 mirror.  † = teacher could not yet pilot the deck (fixed 2026-09-06); win rate void.",
            size=11, color=t["ink2"])
    cv.text(24, yb + 18, f"Across the elixir-phase change (same policy, 24 episodes): median best-case catch "
                         f"{rec['median_catch_flat_phased'][0]} -> {rec['median_catch_flat_phased'][1]} unit;  "
                         f"P(catch >= 3) {rec['p_catch_ge3_flat_phased'][0]}% -> {rec['p_catch_ge3_flat_phased'][1]}%;  "
                         f"P(catch >= 4) {rec['p_catch_ge4_flat_phased'][0]}% -> {rec['p_catch_ge4_flat_phased'][1]}% (recorded).",
            size=11, color=t["ink2"])
    return cv


def fig_5_3(theme):
    rows = RECORDED["fig5_3_marginal_value"]["rows"]
    cv = Canvas(880, 400, theme, "Figure 5.3 - marginal value of committing the win condition",
                "Paired tower-HP deltas with 95% CIs: the gate sweep on the old engine, and the deploy-time A/B.")
    t = cv.t
    cv.heading("Tighter timing made attacking worse; one second of deploy time made it pay",
               sub="Recorded, DECISIONS.md (2026-08-19). Paired tower-HP delta with 95% CI; > 0 favours committing.")
    ax = Axes(cv, 330, 96, 510, 36 * len(rows), (-900, 950), (0, 1))
    for v in (-800, -400, 0, 400, 800):
        cv.line(ax.sx(v), ax.y, ax.sx(v), ax.y + ax.h, t["grid"] if v else t["axis"], 1.5 if v == 0 else 1)
        cv.text(ax.sx(v), ax.y + ax.h + 16, f"{v:+,}" if v else "0", size=11, color=t["muted"], anchor="middle")
    cv.text(ax.x + ax.w / 2, ax.y + ax.h + 36, "tower HP (paired delta)", size=12, color=t["ink2"], anchor="middle")
    for i, (lab, m, lo, hi, tag) in enumerate(rows):
        y = ax.y + 36 * i + 18
        col = t["series"][1] if hi < 0 else (t["series"][0] if lo > 0 else t["muted"])
        cv.text(ax.x - 12, y + 4, lab, size=11.5, anchor="end")
        cv.text(ax.x - 12, y + 18, tag, size=10, color=t["muted"], anchor="end")
        cv.line(ax.sx(lo), y, ax.sx(hi), y, col, 2, cap="round")
        cv.dot(ax.sx(m), y, col, tip=f"{lab}: {m:+.1f} [{lo:+}, {hi:+}]")
        cv.text(ax.sx(hi) + 8, y + 4, f"{m:+.1f}", size=11, color=t["ink2"])
    cv.legend(ax.x, ax.y + ax.h + 62, [("CI below 0", t["series"][1], "dot"), ("CI spans 0", t["muted"], "dot"),
                                       ("CI above 0", t["series"][0], "dot")], gap=14)
    return cv


def fig_5_4(theme):
    r = RECORDED["fig5_4_search"]
    cv = Canvas(880, 460, theme, "Figure 5.4 - search vs greedy, by horizon and opponent",
                "Win rates against the C++ heuristic and deltas against the UtilityTeacher.")
    t = cv.t
    cv.heading("Search's gain belongs to its opponent model",
               sub="Recorded: DECISIONS.md horizon sweep (heuristic @1.5x, n = 80 each) and TODO.md 0e (teacher)")
    ax = Axes(cv, 80, 104, 300, 230, (2, 22), (0.4, 1.0))
    ax.title("(a) vs the C++ heuristic @1.5x")
    ax.grid_y([0.4, 0.6, 0.8, 1.0], lambda v: f"{v:.1f}", "win rate")
    ax.grid_x(r["heuristic_h"], lambda v: f"{v:g}", "horizon (decisions)")
    for key, col, lab in (("heuristic_search", t["series"][0], "search"), ("heuristic_greedy", t["series"][1], "greedy")):
        pts = [(ax.sx(h), ax.sy(v)) for h, v in zip(r["heuristic_h"], r[key])]
        cv.polyline(pts, col)
        for (px, py), v in zip(pts, r[key]):
            cv.dot(px, py, col, tip=f"{lab}: {v:.3f}")
        cv.text(pts[-1][0] + 8, pts[-1][1] + 4, lab, size=11, color=t["ink2"])

    yl = (-0.8, 0.2)
    bx = Axes(cv, 470, 104, 170, 230, (2, 14), yl)
    bx.title("(b) vs the teacher, rung 3")
    bx.grid_y([-0.8, -0.6, -0.4, -0.2, 0, 0.2], lambda v: f"{v:+.1f}" if v else "0", None)
    bx.grid_x(r["teacher_h"], lambda v: f"{v:g}", "horizon")
    cv.line(bx.x, bx.sy(0), bx.x + bx.w, bx.sy(0), t["axis"], 1.5)
    for h, dlt, (lo, hi) in zip(r["teacher_h"], r["teacher_delta"], r["teacher_ci"]):
        cv.line(bx.sx(h), bx.sy(lo), bx.sx(h), bx.sy(hi), t["series"][1], 2, cap="round")
        cv.dot(bx.sx(h), bx.sy(dlt), t["series"][1], tip=f"h{h}: {dlt:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        cv.text(bx.sx(h) + 8, bx.sy(dlt) + 4, f"{dlt:+.2f}", size=11, color=t["ink2"])
    cv.text(bx.x, bx.y + bx.h + 56, "Δ = search − greedy, 95% CI;", size=11, color=t["ink2"])
    cv.text(bx.x, bx.y + bx.h + 70, "greedy read 0.844 in every arm (n = 32)", size=11, color=t["ink2"])

    cx_ = Axes(cv, 710, 104, 130, 230, (0, 2), yl)
    cx_.title("(c) Rollout opponent")
    for v in (-0.8, -0.6, -0.4, -0.2, 0, 0.2):
        cv.line(cx_.x, cx_.sy(v), cx_.x + cx_.w, cx_.sy(v), t["grid"] if v else t["axis"], 1.5 if v == 0 else 1)
    step = cx_.xcats(["h4", "h12"])
    for i, h in enumerate(("h4", "h12")):
        for j, (key, col) in enumerate((("heuristic_rollout", t["series"][1]), ("teacher_rules_rollout", t["series"][0]))):
            m, lo, hi = r["followup"][h][key]
            x = cx_.x + step * (i + 0.5) + (j - 0.5) * 22
            if lo is not None:
                cv.line(x, cx_.sy(lo), x, cx_.sy(hi), col, 2, cap="round")
            cv.dot(x, cx_.sy(m), col, tip=f"{h} {key}: {m:+.3f}")
    cv.legend(cx_.x - 10, cx_.y + cx_.h + 56, [("heuristic", t["series"][1], "dot")], gap=6)
    cv.legend(cx_.x - 10, cx_.y + cx_.h + 74, [("teacher rules", t["series"][0], "dot")], gap=6)
    cv.text(cx_.x - 10, cx_.y + cx_.h + 92, "separate harness, n = 30", size=10.5, color=t["muted"])
    return cv


def fig_5_5(theme):
    r = RECORDED["fig5_5_bankruptcy"]
    cv = Canvas(880, 400, theme, "Figure 5.5 - apathy that was bankruptcy",
                "P(play | affordable) with no threat and under the largest threat, before and after the normaliser fix; "
                "legal-arm counts and the old entropy target as a share of the reachable maximum.")
    t = cv.t
    cv.heading("The entropy normaliser demanded a coin flip; fixing it restored the reflex",
               sub="Recorded, DECISIONS.md (2026-08-16). Note: the deck changed with the fix, so (a) is confounded.")
    ax = Axes(cv, 80, 100, 360, 220, (0, 2), (0, 1))
    ax.title("(a) P(play | affordable)")
    ax.grid_y([0, 0.25, 0.5, 0.75, 1.0], lambda v: f"{v:.2f}")
    step = ax.xcats(["no threat", "largest threat"])
    for j, (lab, v) in enumerate(r["p_play_given_affordable"].items()):
        col = t["series"][1] if j == 0 else t["series"][0]
        for i, vv in enumerate(v):
            x = ax.x + step * (i + 0.5) + (j - 1) * 26 + 3
            cv.bar_v(x, ax.sy(0), 22, ax.sy(vv), col, tip=f"{lab}: {vv}")
            cv.text(x + 11, ax.sy(vv) - 6, f"{vv:.2f}", size=11, color=t["ink2"], anchor="middle")
    ax.baseline(0)
    cv.legend(ax.x, ax.y + ax.h + 56, [("before (old normaliser)", t["series"][1], "bar"), ("after (fixed)", t["series"][0], "bar")])

    bx = Axes(cv, 540, 100, 300, 220, (0, 4), (0, 60))
    bx.title("(b) Legal arms on a decision step")
    bx.grid_y([0, 20, 40, 60], lambda v: f"{v:g}%")
    s2 = bx.xcats(["2", "3", "4", "5"], "legal arms (affordable cards + wait)")
    for i, (n, share) in enumerate(r["legal_arms_share_pct"].items()):
        cx = bx.x + s2 * (i + 0.5)
        cv.bar_v(cx - 12, bx.sy(0), 24, bx.sy(share), t["series"][0], tip=f"{n} arms: {share}% of decisions")
        frac = r["old_target_nats"] / math.log(int(n))
        cv.text(cx, bx.sy(share) - 20, f"{share}%", size=11, color=t["ink2"], anchor="middle")
        cv.text(cx, bx.sy(share) - 6, f"target {frac * 100:.0f}%", size=10, color=t["muted"], anchor="middle")
    bx.baseline(0)
    cv.text(bx.x, bx.y + bx.h + 56, "'target' = the old goal, 0.35·ln 5, as a share of ln(arms)", size=11, color=t["ink2"])
    return cv


FIGURES = [
    ("fig-2-2-speed-calibration", fig_2_2), ("fig-2-3-tick-cost", fig_2_3),
    ("fig-3-2-observation", fig_3_2), ("fig-3-3-placement-head", fig_3_3),
    ("fig-3-4-aux-loss", fig_3_4), ("fig-4-2-discount", fig_4_2),
    ("fig-4-3-curriculum", fig_4_3), ("fig-5-1-cannon", fig_5_1),
    ("fig-5-2-deck-opportunity", fig_5_2), ("fig-5-3-marginal-value", fig_5_3),
    ("fig-5-4-search-horizon", fig_5_4), ("fig-5-5-bankruptcy", fig_5_5),
]

if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "recorded_values.json"), "w", encoding="utf-8") as f:
        json.dump(RECORDED, f, indent=2)
    for name, fn in FIGURES:
        write(name, fn)
