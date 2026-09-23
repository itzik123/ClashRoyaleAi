"""Score candidate card-icon representations against the same in-match frames.

CRBAB's classifier reduces a 61x73 colour crop to an 8x8 grey hash (a 70x
reduction that discards colour) and matches by L1. This holds everything else
fixed (crops, Hungarian assignment, blank padding) and varies only the feature.

Scored without labels, from the game's rules:

  * Impossible early returns. A card that leaves the hand cannot return until
    four others are played. The primary metric: it catches a classifier wrong
    consistently rather than noisily, and relabelling cannot game it.
  * Median hold. A real hand holds several seconds between plays; faster
    reading is churn.
  * Frequency spread. Over a match every card is drawn about equally, so
    max/min share is ~1.0 for a good reader.

A representation wins only if it moves the first; the other two are gamed by
always answering the same hand.

Any feature must be invariant to affordability dimming (an unaffordable card is
dark and low-contrast) or it classifies affordability instead of identity.
Contrast normalisation and hue histograms are; raw RGB is not, and is included
as a control.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from clashroyalebuildabot.constants import (  # noqa: E402
    CARD_CONFIG,
    IMAGES_DIR,
    SCREENSHOT_HEIGHT,
    SCREENSHOT_WIDTH,
)
from clashroyalebuildabot.detectors.screen_detector import ScreenDetector  # noqa: E402
from clashroyalebuildabot.namespaces.cards import Cards  # noqa: E402

DECK = [Cards.VALKYRIE, Cards.ARCHERS, Cards.MINIONS, Cards.CANNON,
        Cards.FIREBALL, Cards.GIANT, Cards.MUSKETEER, Cards.MINIPEKKA]
BLANK_SLOTS = 5
REAL_HOLD_SECONDS = 9.3

MULTI_HASH_SCALE = 0.355
MULTI_HASH_INTERCEPT = 163


# --- features ---
# Each maps a BGR crop to a 1-D float vector; distance is L1.

def f_grey8(bgr: np.ndarray) -> np.ndarray:
    """The incumbent: 8x8 bilinear grey, raw values."""
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grey, (8, 8), interpolation=cv2.INTER_LINEAR).astype(np.float32).ravel()


def f_grey16_norm(bgr: np.ndarray) -> np.ndarray:
    """16x16 grey, contrast-normalised -- 4x the detail, dimming removed."""
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
    return ((small - small.mean()) / (small.std() + 1e-6)).ravel()


def f_grey32_norm(bgr: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    return ((small - small.mean()) / (small.std() + 1e-6)).ravel()


def f_rgb8(bgr: np.ndarray) -> np.ndarray:
    """CONTROL, not a candidate: colour with no dimming invariance."""
    return cv2.resize(bgr, (8, 8), interpolation=cv2.INTER_LINEAR).astype(np.float32).ravel()


def f_hue_hist16(bgr: np.ndarray) -> np.ndarray:
    """py-clash-bot's approach: a 16-bin hue histogram, invariant to affordability
    dimming by construction.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Mask near-grey and near-black pixels, whose hue is numerically defined
    # but meaningless.
    mask = ((hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 40)).astype(np.uint8)
    hist = cv2.calcHist([hsv], [0], mask, [16], [0, 180]).ravel().astype(np.float32)
    return hist / (hist.sum() + 1e-6) * 100.0


def f_grey16_hue(bgr: np.ndarray) -> np.ndarray:
    """Shape AND colour: contrast-normalised structure plus a hue histogram."""
    return np.concatenate([f_grey16_norm(bgr), f_hue_hist16(bgr)])


FEATURES = {
    "grey8 raw (INCUMBENT)": (f_grey8, True),
    "grey16 contrast-norm": (f_grey16_norm, False),
    "grey32 contrast-norm": (f_grey32_norm, False),
    "rgb8 raw (control)": (f_rgb8, True),
    "hue-hist16 (py-clash-bot)": (f_hue_hist16, False),
    "grey16-norm + hue-hist16": (f_grey16_hue, False),
}


def build_templates(fn, multi: bool):
    """Reference vectors, one per candidate card, in DECK+blanks order."""
    cards = list(DECK) + [Cards.BLANK] * BLANK_SLOTS
    out = []
    for card in cards:
        path = Path(IMAGES_DIR) / "cards" / f"{card.name}.jpg"
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        base = fn(bgr)
        if multi:
            # The incumbent's dimming handling: light and dark variants, min
            # distance over the three. Meaningful only for raw-value features.
            light = MULTI_HASH_SCALE * base + MULTI_HASH_INTERCEPT
            dark = (base - MULTI_HASH_INTERCEPT) / MULTI_HASH_SCALE
            out.append(np.stack([base, light, dark]))
        else:
            out.append(base[None, :])
    return cards, out


def classify(crops, cards, templates):
    """Hungarian assignment of 5 slots over the candidate set, as CRBAB does."""
    cost = np.zeros((len(crops), len(cards)), dtype=np.float32)
    for i, crop in enumerate(crops):
        for j, tmpl in enumerate(templates):
            cost[i, j] = np.min(np.mean(np.abs(tmpl - crop[None, :]), axis=1))
    _rows, idx = linear_sum_assignment(cost)
    return [cards[j] for j in idx]


def audit(hands: list[tuple[str, ...]], fps: float) -> dict:
    deck_names = {c.name for c in DECK}
    early = checked = changes = off_deck = 0
    runs: list[int] = []
    previous, run = None, 0
    since_left: dict[str, int] = {}
    plays = 0
    counts: Counter = Counter()

    for hand in hands:
        counts.update(hand)
        if any(c not in deck_names for c in hand):
            off_deck += 1
        if previous is None:
            previous, run = hand, 1
            continue
        if hand == previous:
            run += 1
            continue
        changes += 1
        runs.append(run)
        run = 1
        left = [c for c in previous if c not in hand]
        entered = [c for c in hand if c not in previous]
        if len(left) == 1 and len(entered) == 1:
            plays += 1
            for card in entered:
                if card in since_left:
                    checked += 1
                    if plays - since_left[card] < 4:
                        early += 1
            for card in left:
                since_left[card] = plays
        previous = hand
    if run:
        runs.append(run)

    deck_shares = [counts[c.name] for c in DECK]
    lo = min(deck_shares) or 1
    median_run = (statistics.median(runs) / fps) if runs else 0.0
    return {
        "early_pct": (100 * early / checked) if checked else float("nan"),
        "early": early,
        "checked": checked,
        "median_run_s": median_run,
        "off_deck_pct": 100 * off_deck / max(len(hands), 1),
        "spread": max(deck_shares) / lo,
        "changes": changes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path,
                        default=_ROOT / "assets" / "live" / "match_practice_01")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--fps", type=float, default=5.0)
    args = parser.parse_args()
    fps = args.fps / max(args.stride, 1)

    screens = ScreenDetector()
    paths = sorted(args.frames.glob("f*.png"))[::args.stride]

    print(f"loading {len(paths)} frames, keeping in-match only ...")
    kept = []
    for path in paths:
        pil = Image.open(path).convert("RGB").resize(
            (SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT), Image.LANCZOS)
        if screens.run(pil).name != "in_game":
            continue
        bgr = np.array(pil)[:, :, ::-1]
        kept.append([bgr[t:b, l:r] for (l, t, r, b) in CARD_CONFIG])
    print(f"kept {len(kept)} in-match frames\n")

    print(f"{'representation':<28} {'impossible':>11} {'hold':>7} "
          f"{'offdeck':>8} {'spread':>7}")
    print(f"{'':<28} {'returns':>11} {'(s)':>7} {'':>8} {'max/min':>7}")
    print("-" * 66)

    for label, (fn, multi) in FEATURES.items():
        cards, templates = build_templates(fn, multi)
        hands = []
        for crops in kept:
            vecs = [fn(c) for c in crops]
            assigned = classify(vecs, cards, templates)
            hands.append(tuple(c.name for c in assigned[1:5]))
        s = audit(hands, fps)
        print(f"{label:<28} {s['early_pct']:>9.1f}%  {s['median_run_s']:>6.2f} "
              f"{s['off_deck_pct']:>7.1f}% {s['spread']:>7.1f}"
              f"   ({s['early']}/{s['checked']})")

    print(f"\n  a real hand holds ~{REAL_HOLD_SECONDS} s; spread would be ~1.0 "
          "for a reader that sees every card equally often")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
