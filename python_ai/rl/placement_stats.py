"""Per-card MODAL SHARE of placements over a rolling window of updates.

THE conditional-collapse detector, per CLAUDE.md -- and until 2026-09-15 it was
not logged anywhere (audit 08). `Entropy/Placement_ByCard_Min` is the wrong
statistic: measured 2026-08-14 it flagged the healthiest card (most played,
lowest entropy, modal share 19%) and cleared one sitting at 91% of its mass on a
single cell. A good head is sharp but MOVES its mode with the board; a collapsed
one returns the same cell regardless of it. Counting how often each card's
placements land on its single most common cell sees exactly that.

Computed on the SAMPLED placements of the rollout, so it reads a little below a
greedy argmax count at the same policy -- but a head that has collapsed samples
the same cell too, and that is the failure this exists to catch.
"""
from collections import Counter, deque


class ModalShareWindow:
    def __init__(self, window_updates=10, min_plays=30):
        self.window_updates = int(window_updates)
        #: Below this many plays a share is noise: three plays read >= 33%.
        self.min_plays = int(min_plays)
        self._updates = deque(maxlen=self.window_updates)

    def add_update(self, card_ids, cells):
        per_card = {}
        for cid, cell in zip(card_ids, cells):
            cid = int(cid)
            if cid < 0:
                continue
            per_card.setdefault(cid, Counter())[int(cell)] += 1
        self._updates.append(per_card)

    def shares(self):
        """{card_id: modal share} for cards with at least `min_plays` plays."""
        merged = {}
        for upd in self._updates:
            for cid, counter in upd.items():
                merged.setdefault(cid, Counter()).update(counter)
        out = {}
        for cid, counter in merged.items():
            n = sum(counter.values())
            if n >= self.min_plays:
                out[cid] = counter.most_common(1)[0][1] / n
        return out
