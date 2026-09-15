"""The phase-1 opponent deck pool: real meta decks, loaded from a data file.

WHY THIS EXISTS
---------------
Phase 1's opponent played OUR OWN deck (`opp_deck` defaults to `ai_deck`) for
every episode of every run this project has done. `gym_wrapper`'s own comment
says why that was chosen -- a uniformly random deck out of the 132-card registry
beats a cycle deck "by tens of win-rate points" in this engine, so random decks
were a near-lost game and the win/loss signal vanished.

Both halves of that are right and the conclusion does not follow. The fix for
"random decks are too strong" is not "play yourself forever", it is *real decks,
measured, and sampled by how winnable they are*. `sample_random_deck()` draws 8
cards uniformly out of 132, which produces Golem + P.E.K.K.A. + Mega Knight +
Sparky far more often than any human would ever queue into; a RoyaleAPI ladder
deck is a curated, elixir-balanced object and is a completely different
distribution.

WHAT THE MIRROR COST, MEASURED
------------------------------
Three of the eight `DEFAULT_DECK` cards were dead at P(play | in hand) <= 0.009
for 30,000 consecutive episodes, and the 2026-08-29 autopsy established the card
head was *correctly* pricing them: forced through `env.step`, a Cannon at the
policy's own cell was worth +185 tower HP net over the policy's own action, in
the best state the card ever gets. A coin flip.

`measure_deck_matchups.py` re-runs that question with the OPPONENT'S DECK as the
independent variable, which is the one thing the autopsy held fixed. The three
dead cards are 2.6's answers to threats a 2.6 mirror does not produce:

    Cannon    answers a tank walking at your tower -- the mirror has one Hog
    Fireball  answers medium-HP clusters -- the mirror has one 4-cost support
    The Log   answers ground swarm and bait -- the mirror has 1-elixir Skeletons

A card is worth its elixir against the deck it was put in the deck for. Nothing
in the mirror asks those three questions, so a policy that stopped answering
them is not broken; it is correct about a distribution we chose.

NAMES, NOT IDS, AND WHY THAT IS THE SAFE DIRECTION
--------------------------------------------------
The data file names cards. Resolution goes through `get_all_card_ids()`, which
filters Evolutions out -- load-bearing, because an Evolution reuses its base
card's name verbatim (id 1 and id 128 are both "Archers", per CLAUDE.md). So a
name is AMBIGUOUS over the full registry and UNAMBIGUOUS over the playable set,
and resolving against the playable set is what makes a named data file safe at
all. Ids would also have made the file unreadable and un-editable by hand, which
is the entire point of it being a data file.

Champions and Heroes are rejected by default (`is_champion` / `is_hero` off
`get_card_info`), because the teacher never activates an ability -- a champion it
holds is a strictly worse card than the real one and the deck stops being the
deck it is named after. Set `allow_champions=True` to lift it once abilities are
wired.
"""
import json
import os
import sys

# Run as a file (`python python_ai/opponents/deck_pool.py`) the repo root is not
# on sys.path, so `python_ai.*` cannot resolve -- and importing the package is
# also what puts the unpackaged `clash_royale_env` .pyd on the path. Same
# four-line bootstrap every runnable module here carries; pinned by
# tests/test_package_layout.py, which is what caught its absence.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402

import clash_royale_env as E  # noqa: E402

#: The pool ships here. `CLASH_DECK_POOL` overrides with an absolute path, the
#: same escape hatch `CLASH_WEIGHTS` gives the checkpoint -- an experiment arm
#: needs its own pool without editing a file the live run is reading.
DEFAULT_POOL_PATH = os.path.join(
    python_ai.PACKAGE_DIR, "opponents", "decks", "meta_decks.json")

DECK_SIZE = 8

#: Prior for a deck the file gives no `prior_win_rate` for. 0.5 is the same
#: neutral value `refresh_pfsp_pool` gives a new league member, and it makes an
#: unmeasured deck moderately likely to be drawn so its real difficulty is
#: established quickly.
#:
#: WHY THE FILE CARRIES MEASURED PRIORS AT ALL. At a uniform 0.5 the pool starts
#: uniform, and the EWMA needs ~20 matches per deck to move -- which at 16 decks
#: and 8 workers is a few thousand episodes of a fresh net being fed P.E.K.K.A.
#: bridge spam (measured 0.067) as often as the mirror (1.000). That is not a
#: cold start, it is the zero-gradient state the curriculum pivot exists to
#: avoid, arriving in the first three hours of every run. The priors are the
#: 2026-09-03 sweep's own numbers, so PFSP is approximately right from episode 0
#: and the live EWMA corrects it from there.
#:
#: They are a SEED and never a gate: nothing reads `prior_win_rate` after the
#: first episode of a worker's life.
#:
#: AND AS OF 2026-09-15 THE SHIPPED FILE CARRIES NONE. Those priors described a
#: TRAINED 2.6 Hog policy. For the from-scratch run on a new deck they were
#: wrong twice over, and simulated across 8 workers (audit 04, C3) they bought
#: no episode-share benefit over uniform -- a fresh net loses to everything and
#: every estimate collapses within ~10 matches regardless of the start -- while
#: costing a 3.2x larger estimate error at episode 3,000. Worse, a wrongly-HIGH
#: prior self-seals: a high estimate buys a low PFSP weight and so no samples to
#: correct it (xbow_30_cycle shipped at 0.867 and read 0.36 against a true
#: 0.02). The numbers are kept in the JSON as
#: `prior_win_rate_vs_26hog_2026_09_03`, which the loader does not read.
NEUTRAL_PRIOR = 0.5


class DeckPoolError(ValueError):
    """A pool file that cannot be trusted. Always names the deck and the card.

    Loud on purpose. A silently-dropped deck is the failure mode that matters
    here: the pool would still load, training would still run, and the arm would
    quietly be measuring a different opponent distribution than the file says.
    """


def _playable_by_name():
    """{name: id} over the PLAYABLE set only. See the module docstring."""
    table = {}
    for cid in E.get_all_card_ids():
        table[E.get_card_info(cid)["name"]] = cid
    return table


def _closest(name, known):
    """Cheap suggestion for a misspelt card, so the error is actionable."""
    import difflib
    hits = difflib.get_close_matches(name, known, n=3, cutoff=0.6)
    return f"  did you mean: {', '.join(hits)}?" if hits else ""


class Deck:
    """One resolved opponent deck.

    `card_ids` is the engine-ready list. `tags` are a PRIOR and never a gate --
    `anti_26` records that the human meta calls this deck a 2.6 counter, which
    `measure_deck_matchups.py` is free to contradict. Encoding a matchup belief
    as a hard filter is how you end up training against a distribution chosen by
    opinion; the measured win rate is the gate.
    """

    def __init__(self, name, archetype, card_ids, tags, cards,
                 prior_win_rate=None):
        self.name = name
        self.archetype = archetype
        self.card_ids = list(card_ids)
        self.tags = list(tags)
        self.card_names = list(cards)
        #: Seed for the PFSP prior, from `measure_deck_matchups.py`. See
        #: `NEUTRAL_PRIOR` for why this is worth carrying in the data file.
        self.prior_win_rate = (NEUTRAL_PRIOR if prior_win_rate is None
                               else float(prior_win_rate))

    @property
    def avg_elixir(self):
        return sum(E.get_card_info(c)["cost"] for c in self.card_ids) / DECK_SIZE

    def __repr__(self):
        return f"<Deck {self.name} avg={self.avg_elixir:.2f}>"


def load_pool(path=None, *, allow_champions=False, include_disabled=False):
    """Read, validate and resolve the deck file. Returns [Deck].

    Every failure raises rather than skipping -- see `DeckPoolError`.
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


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------
#: Floor on any deck's sampling weight, so a deck the agent masters can never
#: leave rotation. Phase 2's league uses `PFSP_MIN_WEIGHT = 0.05` for exactly
#: this reason and this is the same argument one level down: a deck that stops
#: being sampled is a deck the policy is free to forget.
POOL_MIN_WEIGHT = 0.05

#: OFF since 2026-09-06 (0.0). The mechanism is kept and `pfsp_weights(floor=)`
#: restores it in one argument; only the default moved.
#:
#: WHAT IT WAS FOR: below this measured win rate a deck is not teaching, it is
#: only losing, and the curriculum pivot's finding is that a mispriced opponent
#: inverts the ranking of strategy classes.
#:
#: WHY IT IS OFF. Measured 2026-09-06, the floor was answering the wrong
#: question. Its input is a win rate, and a win rate against this pool confounds
#: two things: how hard the deck is, and whether the TEACHER can pilot it. It
#: could not pilot five of the sixteen -- with mortar, xbow, both bait decks and
#: graveyard it never spent one elixir on the card the deck is named for. So the
#: floor read "the agent beats this deck" as "this deck is winnable" when the
#: cause was a broken opponent, and read "the agent loses to this deck" as
#: "structurally lost" when the cause was a teacher that happens to pilot heavy
#: decks correctly (send the expensive building-targeter to the bridge IS the
#: Royal Giant plan).
#:
#: The episode shares it produced, against the 2026-09-05 per-deck measurement:
#:
#:     dart_bait_cycle  0.20 win -> 26.7% of episodes   teacher never played the Barrel
#:     mortar_cycle     0.20 win -> 26.7%               teacher never played the Mortar
#:     rg_fisherman     0.00 win ->  0.84%
#:     royal_hogs       0.00 win ->  0.84%
#:     mega_knight_ram  0.00 win ->  0.84%
#:
#: Over half the run went to two decks the opponent could not play, and the
#: three that beat it 40-0 got 0.84% each. A 2.6 cycle deck is BUILT to answer a
#: big push with minimal elixir -- two Cannons, a full cycle inside one defence
#: -- so those matchups are the only place that skill can be learned, and they
#: were the ones being skipped.
#:
#: THE SAFETY ARGUMENT IS NOW CARRIED BY THE RUNG, WHICH IS THE RIGHT AXIS.
#: The floor's real worry is a zero-gradient matchup. Difficulty in this
#: curriculum is the teacher's lookahead, and at a low rung (2 s horizon, 0.15
#: epsilon) a Mega Knight deck is winnable. Resuming a strong checkpoint at a
#: LOW rung with the whole pool live raises one axis and lowers the other,
#: which is exactly what CLAUDE.md's "DROP THE RUNG FIRST" section prescribes.
POOL_WINRATE_FLOOR = 0.0

#: Keep-alive weight for a deck BELOW that floor -- smaller than
#: `POOL_MIN_WEIGHT`, and the difference is the whole point.
#:
#: Phase 2 can use one floor for everything because its pool has ~98 members, so
#: a floored member's share is ~1%. This pool has 16, and the measured spread is
#: 0.07-1.00, so several decks sit under the floor at once: at a shared 0.05 the
#: unwinnable ones took 20% of all training episodes. At 0.02 they take ~9%,
#: which is enough that the policy is not blindsided by an archetype it will
#: meet in phase 2 and on a real ladder, and not so much that a fifth of the run
#: is spent losing.
#:
#: NOT zero, ever, and not an exclusion list either. As the agent improves its
#: measured rate crosses the floor and the deck re-enters rotation on its own --
#: which a static "these decks counter us" list cannot do.
POOL_UNWINNABLE_WEIGHT = 0.02


#: The signal level below which PFSP has nothing to rank with. NOT the retired
#: `POOL_WINRATE_FLOOR`: that was a GATE (park a deck below it) and was removed
#: because a win rate confounds deck difficulty with teacher competence. This is
#: a TIE-BREAK for when literally every deck reads near zero -- there is then no
#: ranking to confound, and `(1 - wr)^2` would hand a policy that is losing
#: everything the matchup it loses hardest.
#:
#: MEASURED (audit 04, C2, 2026-09-15): with the gate at 0.0 the fallback below
#: had become UNREACHABLE -- rates are clamped to [0, 1], so `wr >= 0.0` always
#: held -- and every test of it passed an explicit `floor=0.20` that production
#: never uses. A random-init net got the four hardest decks 59.9% of the time.
POOL_SIGNAL_FLOOR = 0.05


def pfsp_weights(win_rates, *, min_weight=POOL_MIN_WEIGHT,
                 floor=POOL_WINRATE_FLOOR, signal_floor=POOL_SIGNAL_FLOOR):
    """{deck: weight} from {deck: measured win rate}, PFSP-style.

    `(1 - wr)^2` is phase 2's own curve, reused deliberately rather than
    reinvented: it concentrates on decks that are hard but not hopeless. The
    floor is the piece phase 2 does not need -- its league members are past
    selves and so are winnable by construction, while a meta deck can be a
    structural loss in this engine, which is the exact zero-gradient state the
    curriculum pivot exists to avoid.

    A deck below `floor` keeps `min_weight` and no more, so it stays visible
    (no forgetting) without spending the run on it.
    """
    rates = {n: min(max(float(w), 0.0), 1.0) for n, w in win_rates.items()}

    # THE DEGENERATE CASE, and it is the one a fresh run starts in: if NOTHING
    # is at or above the floor, PFSP has no signal to work with. `(1 - wr)^2`
    # would rank a 0.02 deck above a 0.19 one -- i.e. hand a policy that is
    # losing everything the matchup it loses hardest -- and the flat
    # unwinnable weight would make the draw uniform, which is no better.
    #
    # Fall back to UNIFORM (see below). The rung ladder is the difficulty axis;
    # when the deck axis has nothing to say it should say nothing, rather than
    # rank on noise in either direction.
    if rates and not any(wr >= max(floor, signal_floor) for wr in rates.values()):
        # UNIFORM, not easiest-first. Below signal_floor the differences
        # between decks are one to four wins per hundred -- noise, not a
        # ranking -- so the honest encoding of "no signal" is equal shares.
        # The old easiest-first rule (bare `wr**2`) also sent the hardest deck
        # to 0.1% of episodes, an exclusion in all but name and against the
        # standing rule that hard decks are the point. The RUNG is the axis
        # that should carry a cold start, not the deck mix.
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
    """Validate the pool and print it. The one command to run after editing."""
    decks = load_pool()
    print(f"{len(decks)} decks resolved from "
          f"{os.environ.get('CLASH_DECK_POOL') or DEFAULT_POOL_PATH}\n")
    print(f"{'deck':<28} {'avg':>5}  {'tags':<26} cards")
    for d in decks:
        print(f"{d.name:<28} {d.avg_elixir:>5.2f}  {','.join(d.tags):<26} "
              f"{', '.join(d.card_names)}")


if __name__ == "__main__":
    _main()
