"""Per-card modal share of placements over a rolling window of updates.

The conditional-collapse detector. Entropy is the wrong statistic: the
most-played card is legitimately the sharpest. A good head is sharp but moves
its mode with the board; a collapsed one returns the same cell regardless,
which is what the share of a card's single most common cell measures. Computed
on sampled placements, so it reads a little below a greedy count.
"""
from collections import Counter, deque


class ModalShareWindow:
    def __init__(self, window_updates=10, min_plays=30):
        self.window_updates = int(window_updates)
        #: Below this many plays a share is noise.
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
