"""The phase-1 opponent deck pool: real meta decks, loaded from a data file.

A mirror match never asks the questions some cards exist to answer (a Cannon
answers a tank, Fireball a medium-HP cluster, The Log a ground swarm), so
against its own deck the agent correctly stops playing them. Uniformly random
registry decks (`sample_random_deck`) are no substitute: they produce
combinations no player queues into and crush a cycle deck. Curated ladder
decks, sampled by how winnable they are, are the distribution to train against.

The file names cards, resolved over the playable set (`get_all_card_ids()`, no
Evolutions): an Evolution shares its base card's name, so a name is unambiguous
only there. Champions and Heroes are refused unless `allow_champions=True`.
"""
import json
import os
import sys

# Run as a file, the repo root is not on sys.path (tests/test_package_layout.py
# checks this bootstrap).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402

import clash_royale_env as E  # noqa: E402

#: `CLASH_DECK_POOL` overrides this with an absolute path, so an experiment arm
#: can use its own pool.
DEFAULT_POOL_PATH = os.path.join(
    python_ai.PACKAGE_DIR, "opponents", "decks", "meta_decks.json")

DECK_SIZE = 8

#: Prior for a deck with no `prior_win_rate`. The shipped file carries none:
#: priors measured against one trained policy were wrong for a fresh net on
#: another deck, and a wrongly high prior seals itself (a high estimate earns a
#: low PFSP weight, hence no samples to correct it). The old numbers survive in
#: the JSON as `prior_win_rate_vs_26hog_2026_09_03`, which the loader ignores.
NEUTRAL_PRIOR = 0.5


class DeckPoolError(ValueError):
    """A pool file that cannot be trusted; names the deck and the card.

    Raised rather than skipping: a silently dropped deck would change the
    opponent distribution while everything still runs.
    """


def _playable_by_name():
    """{name: id} over the playable set only; see the module docstring."""
    table = {}
    for cid in E.get_all_card_ids():
        table[E.get_card_info(cid)["name"]] = cid
    return table


def _closest(name, known):
    """A suggestion for a misspelt card name."""
    import difflib
    hits = difflib.get_close_matches(name, known, n=3, cutoff=0.6)
    return f"  did you mean: {', '.join(hits)}?" if hits else ""


class Deck:
    """One resolved opponent deck.

    `tags` (e.g. `anti_26`, the human meta's view) are commentary and never
    gate anything; measured win rates decide sampling.
    """

    def __init__(self, name, archetype, card_ids, tags, cards,
                 prior_win_rate=None):
        self.name = name
        self.archetype = archetype
        self.card_ids = list(card_ids)
        self.tags = list(tags)
        self.card_names = list(cards)
        #: Seed for the PFSP estimate; see NEUTRAL_PRIOR.
        self.prior_win_rate = (NEUTRAL_PRIOR if prior_win_rate is None
                               else float(prior_win_rate))

    @property
    def avg_elixir(self):
        return sum(E.get_card_info(c)["cost"] for c in self.card_ids) / DECK_SIZE

    def __repr__(self):
        return f"<Deck {self.name} avg={self.avg_elixir:.2f}>"


def load_pool(path=None, *, allow_champions=False, include_disabled=False):
    """Read, validate and resolve the deck file. Returns [Deck]; every problem
    raises DeckPoolError.
    """
    path = path or os.environ.get("CLASH_DECK_POOL") or DEFAULT_POOL_PATH
    if not os.path.exists(path):
        raise DeckPoolError(f"deck pool file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        blob = json.load(fh)

    by_name = _playable_by_name()
    known = sorted(by_name)
    decks, seen = [], set()

    for entry in blob.get("decks", []):
        name = entry.get("name")
        if not name:
            raise DeckPoolError(f"a deck in {path} has no 'name'")
        if name in seen:
            raise DeckPoolError(f"duplicate deck name {name!r} in {path}")
        seen.add(name)
        if not entry.get("enabled", True) and not include_disabled:
            continue

        cards = entry.get("cards", [])
        if len(cards) != DECK_SIZE:
            raise DeckPoolError(
                f"deck {name!r} has {len(cards)} cards, expected {DECK_SIZE}")
        if len(set(cards)) != DECK_SIZE:
            dupes = sorted({c for c in cards if cards.count(c) > 1})
            raise DeckPoolError(f"deck {name!r} repeats {dupes}")

        ids = []
        for card in cards:
            if card not in by_name:
                raise DeckPoolError(
                    f"deck {name!r}: unknown card {card!r}.{_closest(card, known)}")
            cid = by_name[card]
            info = E.get_card_info(cid)
            if not allow_champions and (info["is_champion"] or info["is_hero"]):
                kind = "Champion" if info["is_champion"] else "Hero"
                raise DeckPoolError(
                    f"deck {name!r}: {card!r} is a {kind}. The teacher never "
                    "activates an ability, so it would play a strictly worse "
                    "card than the deck is named for. Swap it, disable the "
                    "deck, or pass allow_champions=True.")
            ids.append(cid)

        decks.append(Deck(name, entry.get("archetype", ""), ids,
                          entry.get("tags", []), cards,
                          entry.get("prior_win_rate")))

    if not decks:
        raise DeckPoolError(f"{path} resolved to zero enabled decks")
    return decks


# --- sampling ---
# Floor on any deck's sampling weight, so a mastered deck stays in rotation and
# is not forgotten (phase 2's PFSP_MIN_WEIGHT, one level down).
POOL_MIN_WEIGHT = 0.05

#: Off (0.0); `pfsp_weights(floor=...)` restores it. It parked decks below a
#: measured win rate, but a win rate against this pool confounds deck
#: difficulty with whether the teacher can pilot the deck, and it sent most of
#: the run to decks the teacher could not play while starving the hard ones. A
#: matchup that is too hard is handled by lowering the rung instead.
POOL_WINRATE_FLOOR = 0.0

#: Keep-alive weight for a deck below that floor, smaller than POOL_MIN_WEIGHT:
#: with 16 decks, a shared 0.05 gave the unwinnable ones ~20% of episodes; 0.02
#: gives ~9%. Never zero, so a deck returns on its own as the agent improves.
POOL_UNWINNABLE_WEIGHT = 0.02


#: Below this every deck's win rate is noise and PFSP has nothing to rank with,
#: so sampling falls back to uniform. Unlike the retired floor, it only breaks
#: the all-near-zero tie of a fresh run.
POOL_SIGNAL_FLOOR = 0.05


def pfsp_weights(win_rates, *, min_weight=POOL_MIN_WEIGHT,
                 floor=POOL_WINRATE_FLOOR, signal_floor=POOL_SIGNAL_FLOOR):
    """{deck: weight} from {deck: measured win rate}, PFSP-style.

    `(1 - wr)^2`, phase 2's curve, concentrates on decks that are hard but not
    hopeless. A deck below `floor` gets the small keep-alive weight.
    """
    rates = {n: min(max(float(w), 0.0), 1.0) for n, w in win_rates.items()}

    # Nothing at or above the signal floor (a fresh run): no ranking to use.
    # `(1 - wr)^2` would hand a losing policy the matchup it loses hardest.
    if rates and not any(wr >= max(floor, signal_floor) for wr in rates.values()):
        # Uniform, not easiest-first: below the signal floor the differences
        # are noise, and the rung, not the deck mix, carries a cold start.
        return {k: 1.0 / len(rates) for k in rates}

    weights = {}
    for name, wr in rates.items():
        if wr < floor:
            w = POOL_UNWINNABLE_WEIGHT
        else:
            w = max(min_weight, (1.0 - wr) ** 2)
        weights[name] = w
    total = sum(weights.values())
    if total <= 0:
        return {k: 1.0 / len(weights) for k in weights}
    return {k: v / total for k, v in weights.items()}


def sample_deck(decks, weights, rng):
    """Draw one deck. `weights` is {name: w}; missing names get the floor."""
    ws = [weights.get(d.name, POOL_MIN_WEIGHT) for d in decks]
    total = sum(ws)
    if total <= 0:
        return decks[rng.randrange(len(decks))]
    r = rng.random() * total
    for deck, w in zip(decks, ws):
        r -= w
        if r <= 0:
            return deck
    return decks[-1]


def _main():
    """Validate the pool and print it; run this after editing the file."""
    decks = load_pool()
    print(f"{len(decks)} decks resolved from "
          f"{os.environ.get('CLASH_DECK_POOL') or DEFAULT_POOL_PATH}\n")
    print(f"{'deck':<28} {'avg':>5}  {'tags':<26} cards")
    for d in decks:
        print(f"{d.name:<28} {d.avg_elixir:>5.2f}  {','.join(d.tags):<26} "
              f"{', '.join(d.card_names)}")


if __name__ == "__main__":
    _main()
