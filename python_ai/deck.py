"""The trainee's deck, set with CLASH_DECK:

    CLASH_DECK="hog rider,musketeer,cannon,ice golem,skeletons,ice spirit,the log,fireball"
    CLASH_DECK="15,6,25,40,24,72,33,7"
    CLASH_DECK="evo:archers,knight,..."        # an Evolution, explicitly

Names match ignoring case, dots and spaces, over the playable ids. Evolutions
reuse their base card's name, so a bare name means the base card and an
Evolution needs `evo:`.

A leaf: it imports only the engine, so any layer may use it. Checks beyond
legality (a win condition, advisor coverage) are in
`envs.deck_contract.validate_deck`.
"""
import os
import re

import clash_royale_env as E

#: The 2.6 Hog Cycle: Hog Rider, Musketeer, Cannon, Ice Golem, Skeletons,
#: Ice Spirit, The Log, Fireball. What runs when CLASH_DECK is unset.
SHIPPED_DECK = (15, 6, 25, 40, 24, 72, 33, 7)

DECK_SIZE = 8
_EVOLUTION_IDS = range(123, 164)


def _norm(name):
    return re.sub(r"[\s.\-_']", "", name).lower()


def _lookup(token):
    token = token.strip()
    if re.fullmatch(r"-?\d+", token):
        return int(token)
    evo = token.lower().startswith("evo:")
    want = _norm(token[4:] if evo else token)
    pool = _EVOLUTION_IDS if evo else E.get_all_card_ids()
    for cid in pool:
        try:
            if _norm(E.get_card_info(cid)["name"]) == want:
                return int(cid)
        except ValueError:
            continue
    kind = "Evolution" if evo else "card"
    raise ValueError(f"CLASH_DECK: no {kind} named {(token[4:] if evo else token)!r}")


def parse_deck(spec):
    """A deck spec -> list of 8 validated card ids; empty means SHIPPED_DECK.

    Raises ValueError naming the problem.
    """
    if spec is None or not str(spec).strip():
        return list(SHIPPED_DECK)
    tokens = [t for t in str(spec).split(",") if t.strip()]
    deck = []
    for t in tokens:
        deck.append(_lookup(t))
    if len(deck) != DECK_SIZE:
        raise ValueError(f"CLASH_DECK: {len(deck)} cards, need {DECK_SIZE}")
    if len(set(deck)) != DECK_SIZE:
        raise ValueError(f"CLASH_DECK: duplicate cards in {deck}")
    refusal = E.validate_deck_slots(deck)
    if refusal:
        raise ValueError(f"CLASH_DECK: the engine refuses {deck}: {refusal}")
    return deck


DEFAULT_DECK = parse_deck(os.environ.get("CLASH_DECK"))


def describe(deck):
    """'Hog Rider(15,4) Musketeer(6,4) ...' -- for the run log."""
    return " ".join(f"{E.get_card_info(c)['name']}({c},{E.get_card_info(c)['cost']:g})"
                    for c in deck)
