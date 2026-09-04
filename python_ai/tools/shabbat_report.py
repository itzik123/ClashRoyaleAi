"""Build the training-session report from the run's own artifacts.

Re-runnable: reads the TensorBoard event files, the frozen-bank trend CSV and
the trainer log, and writes a self-contained HTML page. Nothing here computes a
number that is not already recorded somewhere -- the point is to assemble
evidence, not to generate it, so a claim in the report can always be traced to
the series it came from.

    python_ai/venv/Scripts/python.exe -m python_ai.tools.shabbat_report \\
        --run-dir runs/phase7 --csv python_ai/eval/banks/trend_phase7.csv \\
        --out report.html
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402
from python_ai.tools.track_placement import scalars, last, signal  # noqa: E402

#: Where each rung boundary fell, discovered from the log rather than assumed.
ADVANCE_MARKERS = ("Curriculum advanced to stage", "DEMOTED", "PLATEAU")


def log_events(path):
    """Curriculum events with their episode, straight out of the trainer log."""
    if not os.path.exists(path):
        return []
    out, last_ep = [], None
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith("Episodes:"):
                try:
                    last_ep = int(line.split()[1])
                except (IndexError, ValueError):
                    pass
            elif any(m in line for m in ADVANCE_MARKERS) or "Resumed from" in line:
                out.append((last_ep, line.strip()))
    return out


def trend(csv_path):
    rows = {}
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.setdefault(r["card"], []).append(r)
    # De-duplicate by episode, keeping the last measurement written for it.
    for card, rs in rows.items():
        seen = {}
        for r in rs:
            seen[int(r["episode"])] = r
        rows[card] = [seen[k] for k in sorted(seen)]
    return rows


def series(s, tag, n=1):
    return last(s.get(tag, []), n=n)


def spark(values, w=260, h=44, lo=None, hi=None):
    """An inline SVG sparkline. No CDN, no library -- the page must render
    from a file:// URL and inside an artifact sandbox alike."""
    v = [x for x in values if x is not None and np.isfinite(x)]
    if len(v) < 2:
        return ""
    lo = min(v) if lo is None else lo
    hi = max(v) if hi is None else hi
    rng = (hi - lo) or 1.0
    pts = " ".join(
        f"{w * i / (len(v) - 1):.1f},{h - (h - 6) * (x - lo) / rng - 3:.1f}"
        for i, x in enumerate(v))
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
            f'<polyline points="{pts}"/></svg>')


def fmt(x, nd=3, dash="--"):
    try:
        return dash if x is None or not np.isfinite(x) else f"{x:.{nd}f}"
    except TypeError:
        return dash


def build(args):
    s = scalars(args.run_dir)
    tr = trend(args.csv)
    events = log_events(args.log)

    wr = [v for _st, v in s.get("Training/Win_Rate_100", [])]
    wr_steps = [st for st, _v in s.get("Training/Win_Rate_100", [])]
    ep_now = wr_steps[-1] if wr_steps else None
    decks = {t.split("/")[-1]: [v for _st, v in vs]
             for t, vs in s.items() if t.startswith("Decks/WinRate/")}

    parts = []
    A = parts.append
    A(f"<h1>Clash Royale RL — training session report</h1>")
    A(f'<p class="sub">Generated {args.stamp} · run <code>{args.run_dir}</code> · '
      f'episode <strong>{ep_now}</strong></p>')

    # ---- headline -------------------------------------------------------
    A('<div class="cards">')
    for label, val, note in [
        ("Episodes", f"{ep_now}", f"from {args.start_ep} at session start"),
        ("Curriculum rung", fmt(series(s, "Training/Curriculum_Stage"), 0),
         "of 11"),
        ("Win rate (100 ep)", fmt(series(s, "Training/Win_Rate_100", n=3)),
         "PFSP-weighted; not a mirror win rate"),
        ("Critic expl. var.", fmt(series(s, "Loss/Critic_Explained_Variance", n=3)),
         "1.0 is perfect; underfitting falls"),
    ]:
        A(f'<div class="card"><div class="k">{label}</div>'
          f'<div class="v">{val}</div><div class="n">{note}</div></div>')
    A('</div>')

    # ---- the placement result ------------------------------------------
    A("<h2>Placement quality — the session's main result</h2>")
    A('<p><code>place_q</code> is the share of achievable value a card\'s '
      'placement distribution expects to collect, measured on a <strong>frozen '
      f'{args.bank_states}-state bank</strong> so every checkpoint is scored on '
      'identical boards. <code>hi/lo</code> is P(play | board offers a lot) '
      'divided by P(play | board offers nothing) — a marginal that rises with a '
      'flat ratio is systemic drift, not learning.</p>')
    A('<table><thead><tr><th>card</th><th>place_q</th><th></th>'
      '<th>hi/lo</th><th>marginal</th><th>read</th></tr></thead><tbody>')
    for card, rs in tr.items():
        q = [float(r["quality_hi"]) for r in rs]
        ratio = [float(r["ratio"]) for r in rs]
        marg = [float(r["marginal"]) for r in rs]
        qs, rs_, ms = signal(q), signal(ratio), signal(marg)
        if np.isfinite(q[-1]):
            qcell = (f'<strong>{q[0]:.3f} → {q[-1]:.3f}</strong>'
                     f'<span class="sig">sig {qs:.1f}</span>')
        else:
            qcell = '<span class="dim">n/a — no cell map</span>'
        verdict = ("conditional" if rs_ > 2 and ms < 2 else
                   "drift" if ms > 2 and rs_ < 2 else "mixed / below noise")
        A(f"<tr><td>{card}</td><td>{qcell}</td><td>{spark(q)}</td>"
          f'<td>{ratio[0]:.2f} → {ratio[-1]:.2f}<span class="sig">sig {rs_:.1f}</span></td>'
          f'<td>{marg[0]:.4f} → {marg[-1]:.4f}<span class="sig">sig {ms:.1f}</span></td>'
          f"<td>{verdict}</td></tr>")
    A("</tbody></table>")
    A('<p class="note">Signal is |last − first| over the series\' own '
      'step-to-step scatter. Below ~2 a movement is not separable from PPO '
      'jitter — this project has been fooled once by a three-point read whose '
      'fourth point reversed it.</p>')

    # ---- per-deck -------------------------------------------------------
    A("<h2>Opponent pool — 16 real ladder decks</h2>")
    A('<p>PFSP samples by <code>(1 − win_rate)²</code>, so the run concentrates '
      'on its worst matchups and the readable win rate is regulated toward the '
      'hard end of the pool. A deck under the 0.20 floor is <em>parked</em>, not '
      'broken.</p>')
    A('<table><thead><tr><th>deck</th><th>win rate</th><th>trend</th>'
      '<th></th></tr></thead><tbody>')
    for name, vs in sorted(decks.items(), key=lambda kv: -kv[1][-1]):
        cls = "bad" if vs[-1] < 0.2 else ("good" if vs[-1] > 0.5 else "")
        A(f'<tr class="{cls}"><td>{name}</td><td>{vs[-1]:.3f}</td>'
          f"<td>{vs[0]:.3f} → {vs[-1]:.3f}</td><td>{spark(vs, w=180, lo=0, hi=1)}</td></tr>")
    A("</tbody></table>")

    # ---- curriculum timeline -------------------------------------------
    A("<h2>Curriculum timeline</h2><ul class='timeline'>")
    for ep, line in events[-24:]:
        A(f'<li><code>{ep}</code> {line[:170]}</li>')
    A("</ul>")

    # ---- diagnostics ----------------------------------------------------
    A("<h2>Health diagnostics</h2><table><thead><tr><th>metric</th>"
      "<th>value</th><th>read against</th></tr></thead><tbody>")
    for tag, label, ref in [
        ("Loss/Critic_Explained_Variance", "Critic explained variance",
         "improving = not underfitting"),
        ("Aux/NextCard_CE", "Next-card CE", "uniform ln(185) = 5.22"),
        ("Loss/Clip_Fraction", "Clip fraction", "healthy run peaked ~0.28"),
        ("Loss/NonFinite_Skips", "Non-finite skips", "must stay 0"),
        ("Progress/Episode_Length_50", "Episode length", "decisions; 2x elixir at 120"),
        ("Decks/WinRate_Spread", "Deck spread", "max − min across the pool"),
    ]:
        v = series(s, tag, n=3)
        if v is not None:
            A(f"<tr><td>{label}</td><td><strong>{fmt(v, 3)}</strong></td>"
              f"<td>{ref}</td></tr>")
    A("</tbody></table>")

    if args.notes and os.path.exists(args.notes):
        A("<h2>Findings this session</h2>")
        A(open(args.notes, encoding="utf-8").read())

    return TEMPLATE.replace("{{BODY}}", "\n".join(parts))


TEMPLATE = """<title>Training Session Report</title>
<style>
:root{
  --bg:#faf9f7; --fg:#1c1b19; --dim:#6b6862; --line:#e2ded7;
  --card:#ffffff; --accent:#8a5a2b; --good:#2f6f4f; --bad:#8d3a3a;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#16151a; --fg:#eceaf0; --dim:#9c98a6; --line:#2e2c36;
  --card:#1e1d24; --accent:#d0a070; --good:#7fc7a0; --bad:#e08585; } }
:root[data-theme="dark"]{
  --bg:#16151a; --fg:#eceaf0; --dim:#9c98a6; --line:#2e2c36;
  --card:#1e1d24; --accent:#d0a070; --good:#7fc7a0; --bad:#e08585; }
body{background:var(--bg);color:var(--fg);font:15px/1.6 ui-sans-serif,-apple-system,
  "Segoe UI",system-ui,sans-serif;margin:0;padding:38px 26px 90px;}
.wrap{max-width:940px;margin:0 auto}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.02em}
h2{font-size:17px;margin:38px 0 10px;padding-bottom:7px;border-bottom:1px solid var(--line)}
.sub{color:var(--dim);margin:0 0 26px;font-size:13px}
p{margin:0 0 12px;max-width:74ch}
code{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-size:.88em;
  background:color-mix(in srgb,var(--fg) 7%,transparent);padding:1px 5px;border-radius:4px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:20px 0 8px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:13px 15px}
.card .k{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.07em}
.card .v{font-size:26px;font-weight:600;margin:3px 0 2px;letter-spacing:-.02em}
.card .n{color:var(--dim);font-size:11.5px;line-height:1.4}
table{width:100%;border-collapse:collapse;margin:14px 0;font-size:13.5px;
  display:block;overflow-x:auto}
th{text-align:left;font-weight:600;color:var(--dim);font-size:11px;
  text-transform:uppercase;letter-spacing:.06em;padding:7px 10px 7px 0;
  border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:8px 10px 8px 0;border-bottom:1px solid var(--line);vertical-align:middle}
tr.good td:nth-child(2){color:var(--good);font-weight:600}
tr.bad td:nth-child(2){color:var(--bad);font-weight:600}
.sig{color:var(--dim);font-size:11px;margin-left:7px}
.dim{color:var(--dim)}
.note{color:var(--dim);font-size:12.5px;font-style:italic}
.spark{width:200px;height:34px;display:block}
.spark polyline{fill:none;stroke:var(--accent);stroke-width:1.6;
  vector-effect:non-scaling-stroke;stroke-linejoin:round}
.timeline{list-style:none;padding:0;font-size:13px}
.timeline li{padding:5px 0 5px 2px;border-bottom:1px solid var(--line);color:var(--dim)}
.timeline code{color:var(--accent);margin-right:8px}
ul{max-width:74ch}
</style>
<div class="wrap">
{{BODY}}
</div>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/phase7")
    ap.add_argument("--csv", default="python_ai/eval/banks/trend_phase7.csv")
    ap.add_argument("--log", default="")
    ap.add_argument("--out", default="shabbat_report.html")
    ap.add_argument("--notes", default="")
    ap.add_argument("--start-ep", default="32875")
    ap.add_argument("--bank-states", default="3611")
    ap.add_argument("--stamp", default="")
    args = ap.parse_args()
    if not args.stamp:
        import datetime
        args.stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    html = build(args)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {args.out} ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
