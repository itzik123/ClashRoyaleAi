"""The trainee's deck. ONE definition, settable without editing code.

    CLASH_DECK="hog rider,musketeer,cannon,ice golem,skeletons,ice spirit,the log,fireball"
    CLASH_DECK="15,6,25,40,24,72,33,7"
    CLASH_DECK="evo:archers,knight,..."        # an Evolution, explicitly

Names are matched ignoring case, dots and spaces, over the PLAYABLE ids, so
"pekka" is P.E.K.K.A. and "minipekka" is Mini PEKKA. Evolutions reuse their base
card's name verbatim (ids 1 and 128 are both "Archers"), so a bare name always
means the base card and an Evolution needs `evo:`.

WHY A MODULE. The deck used to be a literal in `envs/gym_wrapper.py`: changing
it meant editing a read-only-by-policy file, nothing validated the result, and
the run's log did not record which deck produced it. A deck the engine refuses
(a Champion outside slots 1-2) surfaced as a ValueError deep inside module
import. `parse_deck` fails at parse time, with the engine's own reason.

This module is a LEAF -- it imports the engine and nothing else from python_ai
-- so any layer may import it without creating a cycle. What a deck must satisfy
beyond legality (a win condition, advisor coverage, ...) is
`envs.deck_contract.validate_deck`, which needs heavier machinery.

Every AsyncVectorEnv worker and the phase-2 subprocess inherit the environment,
so they parse the same deck.
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
    """A deck spec (see module docstring) -> list of 8 card ids, validated.

    None or empty returns SHIPPED_DECK. Raises ValueError naming the problem.
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
