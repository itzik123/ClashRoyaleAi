"""CLI: measure how fast our engine's board diverges from a real match.

    perception/.venv/Scripts/python.exe -m perception.replay_mining.run_probe \
        --episodes 20 --data-dir <cache>
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import python_ai  # noqa: F401  (puts the .pyd on sys.path)

from perception.replay_mining.divergence import (SAMPLE_TIMES, scramble, score)
from perception.replay_mining.events import build_events
from perception.replay_mining.geometry import fit_transform
from perception.replay_mining.katacr_format import load_episode, load_unit_labels
from perception.replay_mining.reconstruct import reconstruct, default_deck

RAW = "https://raw.githubusercontent.com"
DATASET = f"{RAW}/wty-yy/Clash-Royale-Replay-Dataset/master"
LABELS = f"{RAW}/wty-yy/KataCR/master/katacr/constants/label_list.py"
API = ("https://api.github.com/repos/wty-yy/Clash-Royale-Replay-Dataset"
       "/git/trees/master?recursive=1")


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def episode_paths(data_dir: Path, n: int) -> list[Path]:
    import json
    tree = json.loads(urllib.request.urlopen(API).read())["tree"]
    names = sorted(x["path"] for x in tree
                   if x["type"] == "blob" and x["path"].startswith("fast_hog_2.6/"))
    out = []
    for name in names[:n]:
        out.append(fetch(f"{DATASET}/{name}", data_dir / name))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--data-dir", default=os.environ.get("CLASH_REPLAY_CACHE", "./_replay_cache"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    DEFAULT_DECK = default_deck()

    data_dir = Path(args.data_dir)
    labels = fetch(LABELS, data_dir / "label_list.py")
    idx2unit = load_unit_labels(labels)
    print(f"fetching up to {args.episodes} episodes into {data_dir} ...")
    paths = episode_paths(data_dir, args.episodes)
    print(f"{len(paths)} episodes cached\n")

    rng = random.Random(args.seed)
    results, scrambled, void = [], [], []
    for p in paths:
        ep = load_episode(p, idx2unit)
        try:
            T = fit_transform(ep)
        except ValueError as e:
            void.append((p.name, str(e)))
            continue
        resolved, census = build_events(ep, T)
        n_ego = sum(1 for e, _ in resolved if e.team == 0)
        n_opp = len(resolved) - n_ego
        if n_ego == 0 or n_opp == 0:      # C1: a one-sided board is not a match
            void.append((p.name, f"C1 one-sided: ego={n_ego} opp={n_opp}"))
            continue
        rec = reconstruct(resolved, SAMPLE_TIMES, DEFAULT_DECK, seed=args.seed)
        results.append(score(ep, T, resolved, rec, census))
        srec = reconstruct(scramble(resolved, rng), SAMPLE_TIMES, DEFAULT_DECK,
                           seed=args.seed)
        scrambled.append(score(ep, T, resolved, srec, census))

    if not results:
        print("no usable episodes."); 
        for n, r in void: print("  void:", n, r)
        return 1

    report(results, scrambled, void)
    return 0


def _paired_live(results, scrambled, t):
    """Mean total |ours - recorded| for both arms, over episodes where both arms
    still had a running match at t.

    Restricting to live samples is load-bearing: a dead arm scores the full
    recorded population by construction, so once a reconstruction ends early
    both arms agree trivially and the C2 ratio pins at 1.00, reading as "the
    control found nothing" when nothing was compared.
    """
    a, b = [], []
    for r, sc in zip(results, scrambled):
        if not (r.alive.get(t) and sc.alive.get(t)):
            continue
        a.append(sum(r.m1[t])); b.append(sum(sc.m1[t]))
    if not a:
        return float("nan"), float("nan"), 0
    return float(np.mean(a)), float(np.mean(b)), len(a)


def report(results, scrambled, void):
    print("=" * 78)
    print("C1  INJECTION CENSUS")
    print("=" * 78)
    ego = {}; opp = {}; unmapped = {}
    for r in results:
        for k, v in r.census["ego"].items(): ego[k] = ego.get(k, 0) + v
        for k, v in r.census["opp"].items(): opp[k] = opp.get(k, 0) + v
        for k, v in r.census["unmapped"].items(): unmapped[k] = unmapped.get(k, 0) + v
    tot_ego, tot_opp = sum(ego.values()), sum(opp.values())
    tot_un = sum(unmapped.values())
    print(f"  episodes usable {len(results)}   void {len(void)}")
    for n, why in void: print(f"     void: {n}  ({why})")
    print(f"  ego events   {tot_ego:6d}   {dict(sorted(ego.items(), key=lambda k: -k[1]))}")
    print(f"  opp events   {tot_opp:6d}   {dict(list(sorted(opp.items(), key=lambda k: -k[1]))[:10])}")
    drop = tot_un / max(1, tot_ego + tot_opp + tot_un)
    print(f"  UNMAPPED     {tot_un:6d}  ({drop:.1%} of all events)  {dict(list(sorted(unmapped.items(), key=lambda k:-k[1]))[:8])}")
    unaff = {}
    for r in results:
        for k, v in r.census.get("unaffordable", {}).items():
            unaff[k] = unaff.get(k, 0) + v
    tot_unaff = sum(unaff.values())
    share = tot_unaff / max(1, tot_opp + tot_unaff)
    print(f"  OPP EVENTS REJECTED AS UNAFFORDABLE: {tot_unaff} "
          f"({share:.1%} of inferred opponent plays)")
    print(f"     {dict(list(sorted(unaff.items(), key=lambda k:-k[1]))[:8])}")
    print(f"  board-transform residual: mean {np.mean([r.fit_residual for r in results]):.3f} tiles"
          f"  max {max(r.fit_residual for r in results):.3f}")

    print()
    print("=" * 78)
    print("M1 / C2  UNIT-POPULATION DIVERGENCE  (mean |ours - recorded|, both sides)")
    print("=" * 78)
    print(f"  {'t (s)':>6} {'TRUE':>8} {'SCRAMBLED':>10} {'ratio':>7} {'n live':>7}   "
          f"{'recorded u':>11} {'our u':>10}")
    true_by_t, scr_by_t = {}, {}
    for t in SAMPLE_TIMES:
        a, b, n = _paired_live(results, scrambled, t)
        ru = [r.recorded_units[t] for r in results if t in r.recorded_units]
        ou = [r.our_units[t] for r in results if r.alive.get(t)]
        if not ru:
            continue
        rstr = f"{np.mean([x[0] for x in ru]):.1f}/{np.mean([x[1] for x in ru]):.1f}"
        ostr = f"{np.mean([x[0] for x in ou]):.1f}/{np.mean([x[1] for x in ou]):.1f}" if ou else "-"
        if n:
            true_by_t[t], scr_by_t[t] = a, b
            ratio = b / a if a > 1e-9 else float("nan")
            print(f"  {t:>6.0f} {a:>8.2f} {b:>10.2f} {ratio:>7.2f} {n:>7d}   {rstr:>11} {ostr:>10}")
        else:
            print(f"  {t:>6.0f} {'-':>8} {'-':>10} {'-':>7} {0:>7d}   {rstr:>11} {ostr:>10}")

    print()
    print("  SURVIVAL: our reconstruction vs the real match")
    surv = [(r.our_end_t, r.recorded_end_t) for r in results]
    print(f"    our match ends at   mean {np.mean([a for a, _ in surv]):6.1f} s")
    print(f"    real match ends at  mean {np.mean([b for _, b in surv]):6.1f} s")
    print(f"    reconstructions ending early: "
          f"{sum(1 for a, b in surv if a < b - 5):d}/{len(surv)}")

    print()
    print("=" * 78)
    print("M2  POSITIONAL DIVERGENCE (mean centroid distance, tiles)")
    print("=" * 78)
    for t in SAMPLE_TIMES:
        v = [r.m2[t] for r in results if r.m2.get(t) is not None]
        if v: print(f"  t={t:>5.0f}s   {np.mean(v):.2f} tiles   (n={len(v)})")

    print()
    print("=" * 78)
    print("M3  OUTCOME AGREEMENT")
    print("=" * 78)
    from collections import Counter
    agree = sum(1 for r in results if r.m3[0] == r.m3[1])
    rec_c = Counter(r.m3[0] for r in results)
    base = max(rec_c.values()) / len(results)
    print(f"  agreement {agree}/{len(results)} = {agree/len(results):.1%}")
    print(f"  BASE RATE (always guess the commonest recorded outcome): {base:.1%}")
    print(f"  recorded: {dict(rec_c)}")
    print(f"  ours:     {dict(Counter(r.m3[1] for r in results))}")

    print()
    print("=" * 78)
    print("VERDICT AGAINST THE KILL CRITERIA")
    print("=" * 78)
    ratios = [scr_by_t[t] / true_by_t[t] for t in true_by_t if true_by_t[t] > 1e-9]
    c2 = float(np.mean(ratios)) if ratios else float("nan")
    print(f"  C2 scrambled/true ratio (>1 means the true arm is better): {c2:.3f}")
    from collections import Counter as _C
    _rc = _C(r.m3[0] for r in results)
    _base = max(_rc.values()) / len(results)
    print(f"  M3 outcome agreement: {agree/len(results):.1%}  vs base rate {_base:.1%}")
    horizon = None
    for t in sorted(true_by_t):
        rec_tot = np.mean([sum(r.recorded_units[t]) for r in results if t in r.recorded_units])
        if true_by_t[t] > 0.5 * max(1.0, rec_tot):
            break
        horizon = t
    print(f"  usable horizon (divergence < 50% of the real population): "
          f"{horizon if horizon else '< 30'} s")


if __name__ == "__main__":
    raise SystemExit(main())
