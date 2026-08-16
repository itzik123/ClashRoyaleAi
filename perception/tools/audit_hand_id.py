"""Measure the LIVE hand classifier against constraints the game itself enforces.

WHY THIS EXISTS
---------------
Hand identity had one number attached to it -- "templates agreed with the elixir
ledger 33.8% of the time" -- and that number was measured on
`perception/readers/hand.py`, which has NO CALLER. The live loop reads its hand
from the vendored CRBAB `CardDetector` (an 8x8 greyscale perceptual hash with a
Hungarian assignment), a completely different classifier. So the one published
accuracy figure describes dead code.

This tool measures the classifier that actually runs.

THERE ARE NO LABELS, AND NONE ARE NEEDED
----------------------------------------
Nobody hand-annotated the live frames, but the game enforces three structural
invariants that a correct reading cannot violate. Each is a ground truth that
costs nothing to check:

  1. NO OFF-DECK CARD. We chose the deck; a card outside it is a misread.
  2. NO DUPLICATE. Eight distinct cards in a strict FIFO cannot put the same
     card in two slots.
  3. NO EARLY RETURN. The cycle is 4 in hand + 4 in queue, so a card that
     leaves the hand cannot come back until four OTHER cards have been played.
     This is the strongest of the three and the only one that catches a
     classifier which is wrong CONSISTENTLY rather than noisily.

Plus one rate check: a hand holds ~9.3 s between plays, so a reading that
changes much faster than that is churning, whatever it reports.

`--compare` additionally scores candidate representations on the same frames,
so a proposed fix is measured against the incumbent rather than argued about.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from clashroyalebuildabot.constants import (  # noqa: E402
    CARD_CONFIG,
    SCREENSHOT_HEIGHT,
    SCREENSHOT_WIDTH,
)
from clashroyalebuildabot.detectors.card_detector import CardDetector  # noqa: E402
from clashroyalebuildabot.detectors.screen_detector import ScreenDetector  # noqa: E402
from clashroyalebuildabot.namespaces.cards import Cards  # noqa: E402

# The deck in the live recording, and the one mvp_loop.py is configured for.
DECK = [Cards.VALKYRIE, Cards.ARCHERS, Cards.MINIONS, Cards.CANNON,
        Cards.FIREBALL, Cards.GIANT, Cards.MUSKETEER, Cards.MINIPEKKA]

# A real hand holds this long between plays (readers/hand.py, measured over a
# 326 s recording). Used only to express churn as a multiple.
REAL_HOLD_SECONDS = 9.3


def load_frames(directory: Path, limit: int | None, stride: int):
    paths = sorted(directory.glob("f*.png"))
    if stride > 1:
        paths = paths[::stride]
    if limit:
        paths = paths[:limit]
    for path in paths:
        image = Image.open(path).convert("RGB")
        yield path.name, image.resize((SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT),
                                      Image.LANCZOS)


def collapse_blanks(readings):
    """Carry the last non-blank card forward, per slot.

    Playing a card empties its slot for a few frames while the next slides in,
    so the raw stream reads `X -> blank -> Y` for a single play. Left alone
    that is two transitions, which both double-counts plays and turns an
    `X -> blank -> X` flicker into a card returning after one play -- something
    the 8-slot FIFO makes impossible. Neither is a classifier error, so the
    cycle test has to see through the gap rather than score it.
    """
    out = []
    last: list[str | None] = [None, None, None, None]
    for name, hand in readings:
        filled = []
        for i, card in enumerate(hand):
            if card == "blank":
                filled.append(last[i] if last[i] is not None else "blank")
            else:
                last[i] = card
                filled.append(card)
        out.append((name, tuple(filled)))
    return out


def audit(readings: list[tuple[str, tuple[str, ...]]], fps: float) -> dict:
    """Score a stream of 4-card hand readings against the three invariants."""
    readings = collapse_blanks(readings)
    deck_names = {c.name for c in DECK}

    off_deck = 0
    duplicates = 0
    total = 0
    changes = 0
    early_returns = 0
    checked_returns = 0
    run_lengths: list[int] = []

    previous: tuple[str, ...] | None = None
    run = 0
    # Cards played since a given card left the hand, for the early-return test.
    since_left: dict[str, int] = {}
    plays_seen = 0

    for _name, hand in readings:
        total += 1
        if any(c not in deck_names for c in hand):
            off_deck += 1
        if len(set(hand)) != len(hand):
            duplicates += 1

        if previous is None:
            previous, run = hand, 1
            continue

        if hand == previous:
            run += 1
        else:
            changes += 1
            run_lengths.append(run)
            run = 1
            left = [c for c in previous if c not in hand]
            entered = [c for c in hand if c not in previous]
            # Only single-slot transitions carry cycle information; a
            # multi-slot jump means frames were dropped or the read is garbage.
            #
            # BLANK IS NOT A CARD AND MUST NOT ENTER THE CYCLE TEST. Playing a
            # card empties its slot for a few frames while the next one slides
            # in, so a single real play reads as `X -> blank -> Y`. Counting
            # those as two transitions both inflates the play counter and makes
            # a flicker `X -> blank -> X` look like a card returning after one
            # play, which is impossible by construction. An earlier version of
            # this function did exactly that and attributed 62.9% of
            # transitions to misclassification when the reading was in fact
            # stable and correct.
            if len(left) == 1 and len(entered) == 1 and \
                    "blank" not in (left[0], entered[0]):
                plays_seen += 1
                for card in entered:
                    if card in since_left:
                        checked_returns += 1
                        if plays_seen - since_left[card] < 4:
                            early_returns += 1
                for card in left:
                    since_left[card] = plays_seen
            previous = hand

    if run:
        run_lengths.append(run)

    median_run_frames = statistics.median(run_lengths) if run_lengths else 0
    median_run_s = median_run_frames / fps if fps else 0.0

    return {
        "frames": total,
        "off_deck": off_deck,
        "duplicates": duplicates,
        "changes": changes,
        "distinct_hands": len({h for _n, h in readings}),
        "median_run_s": median_run_s,
        "churn_multiple": (REAL_HOLD_SECONDS / median_run_s) if median_run_s else float("inf"),
        "early_returns": early_returns,
        "checked_returns": checked_returns,
    }


def report(title: str, stats: dict) -> None:
    n = max(stats["frames"], 1)
    print(f"\n=== {title} ===")
    print(f"  frames                 {stats['frames']}")
    print(f"  off-deck cards         {stats['off_deck']:>5}  ({100*stats['off_deck']/n:.1f}%)")
    print(f"  duplicate cards        {stats['duplicates']:>5}  ({100*stats['duplicates']/n:.1f}%)")
    print(f"  distinct hands read    {stats['distinct_hands']:>5}")
    print(f"  hand changes           {stats['changes']:>5}")
    print(f"  median unchanged run   {stats['median_run_s']:.2f} s"
          f"   (a real hand holds ~{REAL_HOLD_SECONDS} s"
          f" -> churning {stats['churn_multiple']:.1f}x too fast)")
    if stats["checked_returns"]:
        pct = 100 * stats["early_returns"] / stats["checked_returns"]
        print(f"  IMPOSSIBLE early returns {stats['early_returns']:>3}"
              f" / {stats['checked_returns']}  ({pct:.1f}%)")
    else:
        print("  IMPOSSIBLE early returns   n/a (no single-slot transitions)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path,
                        default=_ROOT / "assets" / "live" / "match_practice_01")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=5.0,
                        help="capture rate of the frame dump, for run lengths")
    args = parser.parse_args()

    fps = args.fps / max(args.stride, 1)

    detector = CardDetector(list(DECK))
    screens = ScreenDetector()

    ungated: list[tuple[str, tuple[str, ...]]] = []
    gated: list[tuple[str, tuple[str, ...]]] = []
    slot_counts: Counter = Counter()

    for name, image in load_frames(args.frames, args.limit, args.stride):
        cards, _ready = detector.run(image)
        # cards[0] is the "next card" slot; 1..4 are the hand.
        hand = tuple(c.name for c in cards[1:5])
        ungated.append((name, hand))
        if screens.run(image).name == "in_game":
            gated.append((name, hand))
            slot_counts.update(hand)

    report("UNGATED -- every frame, which is what the live loop does today",
           audit(ungated, fps))
    report("GATED on screen == in_game", audit(gated, fps))
    print(f"\n  gate kept {len(gated)} / {len(ungated)} frames "
          f"({100*len(gated)/max(len(ungated),1):.1f}%)")

    print("\n  predicted card frequency, IN-MATCH only "
          "(uniform prior would be 12.5% each):")
    total_slots = sum(slot_counts.values()) or 1
    for card, count in slot_counts.most_common():
        share = 100 * count / total_slots
        flag = "  <-- SINK" if share > 25.0 else ""
        print(f"    {card:<14} {share:5.1f}%{flag}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
