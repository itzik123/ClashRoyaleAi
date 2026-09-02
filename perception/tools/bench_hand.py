"""Benchmark hand identity: the incumbent CRBAB hash against the deck pool.

    perception/.venv/Scripts/python.exe perception/tools/bench_hand.py

THREE MEASUREMENTS, BECAUSE ONE WOULD BE MISLEADING
---------------------------------------------------
1. **Accuracy against hand labels.** 120 slot crops sampled with a fixed seed
   from `assets/live/match_practice_01`, labelled BLIND -- the crops were
   shuffled and shown with numeric ids only, so neither classifier's opinion
   was visible while labelling. This is the only absolute accuracy figure
   here; everything else is a constraint check.

   **It is reported per STRATUM, and that is not decoration.** The first
   version of this label set sampled only slots carrying a magenta cost
   badge, which excluded every unaffordable card and every empty slot -- 20%
   of in-match slots, and precisely the population `DeckHandDetector`'s
   presence rule was wrong about. It scored 100.0% while reporting `blank`
   for one slot in seven. A sample drawn through the same predicate the
   classifier gates on cannot test that gate.

2. **The game's own structural invariants**, which need no labels at all:
   no off-deck card, no duplicate within a hand, and no early return (a card
   that leaves the hand cannot come back until four OTHERS are played). The
   third is the strongest, and the only one that catches a classifier which is
   *stably* wrong rather than noisy. Same instrument as `audit_hand_id.py`,
   imported rather than reimplemented.

3. **Latency**, INTERLEAVED, with an identical-arms control that must read
   ~1.00. This box thermally throttles -- CLAUDE.md records a 3x absolute
   drift within one session -- so a sequential "arm A then arm B" comparison
   measures the machine warming up. Arms alternate, start order rotates, and
   a third arm identical to the first bounds the noise floor. A control
   outside 0.90-1.10 invalidates the run and says so.

WHAT THE POOL-SIZE HYPOTHESIS PREDICTED, AND WHY IT IS NOT TESTED HERE
----------------------------------------------------------------------
The proposal this benchmark was written to evaluate was to shrink the
template pool from "the whole card database" to the eight cards of our deck.
`CardDetector` already does that -- it loads only the cards it is handed,
plus five `blank` entries -- so there is no 115-template arm to compare
against; it does not exist in the code. The off-deck column below is what
that restriction already buys, and it is already near zero for BOTH arms.

WHAT THE ERROR ACTUALLY IS
--------------------------
Stable within-deck confusion, not fabrication. Both arms report `blank`
correctly on 100% of genuinely empty slots; the incumbent simply reads
Musketeer as Mini P.E.K.K.A most of the time (46.2% correct on that card),
and a consistently-wrong card produces a hand history the 8-card FIFO rules
out -- which is why the invariant in section 2 moves so much further than the
accuracy in section 1.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from clashroyalebuildabot.constants import (  # noqa: E402
    SCREENSHOT_HEIGHT,
    SCREENSHOT_WIDTH,
)
from clashroyalebuildabot.detectors.card_detector import CardDetector  # noqa: E402
from clashroyalebuildabot.detectors.screen_detector import ScreenDetector  # noqa: E402
from clashroyalebuildabot.namespaces.cards import Cards  # noqa: E402
from live.deck_hand import DeckHandDetector, load_pool  # noqa: E402
from tools.audit_hand_id import audit, report  # noqa: E402

ASSETS = _ROOT / "tests" / "assets" / "hand"

#: The two benchmarked recordings. A pool, a frame dump and a label set are one
#: unit -- they are all a specific deck, and mixing them across decks measures
#: what an unseen card least mismatches.
SUITES = {
    "giant": {
        "frames": _ROOT / "assets" / "live" / "match_practice_01",
        "labels": ASSETS / "hand_labels_match_practice_01.json",
        "pool": _ROOT / "config" / "templates" / "deck_pool_giant",
        "fps": 5.0,
        "hold_s": 9.3,
    },
    "hog26": {
        "frames": _ROOT / "assets" / "live" / "match_hog26_01",
        "labels": ASSETS / "hand_labels_match_hog26_01.json",
        "pool": _ROOT / "config" / "templates" / "deck_pool_hog26",
        "fps": 3.0,
        # No measured hold time for this deck. 2.6 Hog Cycle averages 2.625
        # elixir a card against the Giant deck's 4.1, so it cycles far faster
        # by design and the Giant figure would flag a correct reading.
        "hold_s": None,
    },
}


def deck_from_pool(pool_dir):
    """The deck a pool was built for, read off the pool.

    Derived rather than listed so a benchmark cannot be run with a deck that
    disagrees with its own templates -- which would silently measure the
    refusal path instead of the classifier.
    """
    names = set(load_pool(pool_dir))
    return [c for c in vars(Cards).values()
            if getattr(c, "name", None) in names]


def _load(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize(
        (SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT), Image.LANCZOS)


def build_arms(pool_dir: Path, deck=None):
    """Fresh detector per arm. CardDetector MUTATES the list it is handed.

    `CardDetector.__init__` does `self.cards = cards` with no copy and then
    `self.cards.extend([BLANK] * 5)`, so constructing two of them from one
    list gives the second thirteen cards and the third eighteen. Every arm
    therefore gets its own `list(deck)`.
    """
    deck = list(deck if deck is not None else deck_from_pool(pool_dir))
    return {
        "crbab (incumbent, 8x8 grey hash vs stock jpgs)":
            CardDetector(list(deck)),
        "deck_pool argmax (live-domain 48x40 templates)":
            DeckHandDetector(list(deck), pool_dir, assign="argmax"),
        "deck_pool hungarian (same, forced bijection)":
            DeckHandDetector(list(deck), pool_dir, assign="hungarian"),
    }


def score_labels(arms, labels, frames_dir: Path) -> dict:
    """Per-slot accuracy against the blind labels."""
    frames = sorted({e["frame"] for e in labels})
    images = {f: _load(frames_dir / f"f{f:05d}.png") for f in frames}

    def stratum(entry):
        if entry["empty"]:
            return "empty (card-back)"
        return "dimmed (unaffordable)" if entry["dimmed"] else "affordable"

    out = {}
    for name, det in arms.items():
        hits = 0
        confusion: Counter = Counter()
        per_card_total: Counter = Counter()
        per_card_hit: Counter = Counter()
        strat_total: Counter = Counter()
        strat_hit: Counter = Counter()
        for entry in labels:
            cards, _ready = det.run(images[entry["frame"]])
            got = cards[entry["slot"]].name
            truth = entry["truth"]
            key = stratum(entry)
            per_card_total[truth] += 1
            strat_total[key] += 1
            if got == truth:
                hits += 1
                per_card_hit[truth] += 1
                strat_hit[key] += 1
            else:
                confusion[(truth, got)] += 1
        out[name] = {
            "n": len(labels),
            "acc": hits / len(labels),
            "confusion": confusion,
            "per_card": {c: (per_card_hit[c], per_card_total[c])
                         for c in per_card_total},
            "per_stratum": {k: (strat_hit[k], strat_total[k])
                            for k in strat_total},
        }
    return out


def score_invariants(arms, frames_dir: Path, stride: int, fps: float,
                     deck=None, real_hold_seconds=None) -> dict:
    paths = sorted(frames_dir.glob("f*.png"))[::stride]
    screens = ScreenDetector()
    images, gated = [], []
    for path in paths:
        image = _load(path)
        images.append((path.name, image))
        gated.append(screens.run(image).name == "in_game")

    out = {}
    for name, det in arms.items():
        readings = []
        for (fname, image), keep in zip(images, gated):
            if not keep:
                continue
            cards, _ready = det.run(image)
            readings.append((fname, tuple(c.name for c in cards[1:5])))
        out[name] = audit(readings, fps / max(stride, 1), deck,
                          real_hold_seconds)
        out[name]["_freq"] = Counter(c for _n, h in readings for c in h)
    return out


def score_latency(arms, pool_dir: Path, frames_dir: Path,
                  n_frames: int, repeats: int):
    """Interleaved, order-rotated, with an identical-arms control.

    The control is a SECOND, independently constructed copy of arm 1. Two
    identical arms must read the same time; CLAUDE.md records that they do
    not when anything else is loading the box, and that a fixed arm order
    alone is not enough either -- with two identical arms the second read 6.8%
    slower every round, because it inherits the cache the first just evicted.
    Hence rotation as well as interleaving.
    """
    paths = sorted(frames_dir.glob("f*.png"))
    step = max(1, len(paths) // n_frames)
    images = [_load(p) for p in paths[::step][:n_frames]]

    names = list(arms)
    control_name = names[0] + "  [CONTROL, 2nd copy of arm 1]"
    lanes = {**arms, control_name: build_arms(pool_dir)[names[0]]}

    # WARM-UP, DISCARDED. The first pass over any lane pays page faults, lazy
    # numpy/scipy imports and a cold cache, and it lands on whichever lane
    # happens to go first. Measured without it, the identical-arms control
    # read 1.426 -- a 43% "difference" between two copies of the same code,
    # which would have been reported as a real regression.
    for det in lanes.values():
        for image in images[:8]:
            det.run(image)

    samples: dict[str, list[float]] = {k: [] for k in lanes}
    order = list(lanes)
    for r in range(repeats):
        rotated = order[r % len(order):] + order[:r % len(order)]
        for name in rotated:
            det = lanes[name]
            start = time.perf_counter()
            for image in images:
                det.run(image)
            samples[name].append(
                (time.perf_counter() - start) * 1000.0 / len(images))

    return {k: statistics.median(v) for k, v in samples.items()}, control_name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=sorted(SUITES), default="hog26",
                        help="which deck's recording to benchmark")
    parser.add_argument("--pool", type=Path, default=None,
                        help="override the suite's template pool")
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--frames", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--latency-frames", type=int, default=60)
    parser.add_argument("--latency-repeats", type=int, default=15)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--only-latency", action="store_true")
    parser.add_argument("--min-frame", type=int, default=None,
                        help="score only labels at or after this frame index; "
                             "use with a pool built from EARLIER frames for a "
                             "held-out evaluation")
    args = parser.parse_args()

    suite = SUITES[args.suite]
    pool = args.pool or suite["pool"]
    frames_dir = args.frames or suite["frames"]
    labels_path = args.labels or suite["labels"]
    fps = args.fps if args.fps is not None else suite["fps"]

    if not frames_dir.exists():
        raise SystemExit(
            f"no frame dump at {frames_dir}. assets/live/ is gitignored; "
            f"regenerate it from the recording.")

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if args.min_frame is not None:
        labels = [e for e in labels if e["frame"] >= args.min_frame]
    arms = build_arms(pool)

    print(f"suite {args.suite}   pool {pool.name}   frames {frames_dir.name}")
    print(f"deck: {', '.join(sorted(c.name for c in deck_from_pool(pool)))}")
    if args.min_frame is not None:
        print(f"HELD OUT: scoring only labels from frame >= {args.min_frame}")

    if args.only_latency:
        med, control_name = score_latency(arms, pool, frames_dir,
                                          args.latency_frames,
                                          args.latency_repeats)
        base = list(arms)[0]
        print("\nLATENCY -- interleaved, order-rotated, warmed up")
        for name, ms in med.items():
            print(f"  {name:<52} {ms:7.3f} ms/frame")
        ratio = med[control_name] / med[base]
        print(f"\n  identical-arms control ratio: {ratio:.3f}  "
              + ("OK" if 0.90 <= ratio <= 1.10 else "*** OUT OF BAND ***"))
        for name in list(arms)[1:]:
            print(f"  vs incumbent: {med[name]/med[base]:.2f}x  ({name})")
        return 0

    print("\n" + "=" * 74)
    print(f"1. ACCURACY vs {len(labels)} BLIND-LABELLED SLOT CROPS")
    print("=" * 74)
    acc = score_labels(arms, labels, frames_dir)
    for name, res in acc.items():
        print(f"\n  {name}")
        print(f"    accuracy {res['acc']*100:5.1f}%   "
              f"({int(round(res['acc']*res['n']))}/{res['n']})")
        for key in ("affordable", "dimmed (unaffordable)", "empty (card-back)"):
            if key not in res["per_stratum"]:
                continue
            hit, tot = res["per_stratum"][key]
            print(f"      [{key:<21}] {hit:>3}/{tot:<3} {100*hit/tot:5.1f}%")
        worst = sorted(res["per_card"].items(), key=lambda kv: kv[1][0] / kv[1][1])
        for card, (hit, tot) in worst:
            bar = "#" * int(20 * hit / tot)
            print(f"      {card:<11} {hit:>2}/{tot:<3} {100*hit/tot:5.1f}%  {bar}")
        if res["confusion"]:
            print("      top confusions (truth -> predicted):")
            for (truth, got), count in res["confusion"].most_common(6):
                print(f"        {truth:<11} -> {got:<11} x{count}")

    print("\n" + "=" * 74)
    print("2. STRUCTURAL INVARIANTS (no labels needed), in-match frames only")
    print("=" * 74)
    inv = score_invariants(arms, frames_dir, args.stride, fps,
                           deck_from_pool(pool),
                           suite.get("hold_s"))
    for name, stats in inv.items():
        report(name, stats)
        total = sum(stats["_freq"].values()) or 1
        skew = max(abs(100 * c / total - 12.5) for c in stats["_freq"].values())
        print(f"  worst card-frequency skew vs 12.5% prior: {skew:5.1f} pts")

    if not args.skip_latency:
        print("\n" + "=" * 74)
        print("3. LATENCY -- interleaved, order-rotated, control must read ~1.00")
        print("=" * 74)
        med, control_name = score_latency(arms, pool, frames_dir,
                                          args.latency_frames,
                                          args.latency_repeats)
        base = list(arms)[0]
        for name, ms in med.items():
            print(f"  {name:<52} {ms:7.3f} ms/frame")
        ratio = med[control_name] / med[base]
        print(f"\n  identical-arms control ratio: {ratio:.3f}", end="  ")
        print("OK" if 0.90 <= ratio <= 1.10
              else "*** OUT OF BAND -- this run's timings are contended ***")
        for name in list(arms)[1:]:
            print(f"  vs incumbent: {med[name]/med[base]:.2f}x  ({name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
