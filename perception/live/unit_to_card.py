"""Map a detected unit name to an engine card id.

CRBAB's detector names units, the engine names cards, and the relation is
many-to-one. Many detector classes do not match an engine card by name,
including `archer` and `minion`.

The reverse map is generated at import from CRBAB's own `Cards` namespace
(which lists the units each card spawns), so a new detector class raises
instead of going silently unmapped. The only hand-authored part is
`PREFERRED_CARD`, for units several cards spawn (a `skeleton` can come from
Skeletons, Skeleton Army, Graveyard, Skeleton Barrel, Witch or Skeleton King).

The cost: the observation gets the preferred card's attributes. Every attribute
the observation carries (HP, damage, range, speed, air/ground) belongs to the
unit, so they are identical either way; only elixir cost differs, and the
spatial channels do not use it. Death-spawns (`golemite` -> Golem, `lava_pup`
-> Lava Hound) follow the same rule.
"""
from __future__ import annotations

import dataclasses
import re
from functools import lru_cache

import engine as _engine_build  # noqa: F401 -- see engine.py; must precede
                                # any `import clash_royale_env` in the process
from contracts import UNKNOWN_CARD_SIM_ID

# One unit, several parent cards: the card that spawns it as its primary
# payload, the commonest source in play.
PREFERRED_CARD: dict[str, str] = {
    # Spawned directly by their namesake card, not by a barrel, hut or spell.
    "barbarian": "barbarians",        # over barbarian_barrel, barbarian_hut
    "bat": "bats",                    # over night_witch
    "goblin": "goblins",              # over goblin_barrel, goblin_drill, goblin_gang
    "spear_goblin": "spear_goblins",  # over goblin_gang, goblin_giant, goblin_hut
    "skeleton": "skeletons",          # over skeleton_army, graveyard, witch, ...
    "minion": "minions",              # over minion_horde
    "musketeer": "musketeer",         # over three_musketeers
    "royal_recruit": "royal_recruits",  # over royal_delivery
    # Death-spawns and stage-splits -> parent card.
    "golemite": "golem",
    "lava_pup": "lava_hound",
    "elixir_golem_large": "elixir_golem",
    "elixir_golem_medium": "elixir_golem",
    "elixir_golem_small": "elixir_golem",
    "phoenix_large": "phoenix",
    "phoenix_egg": "phoenix",
    "phoenix_small": "phoenix",
    # Side-spawns with a single sensible parent.
    "royal_guardian": "little_prince",
    "brawler": "goblin_cage",
    "hog": "mother_witch",            # the hog a cursed troop becomes
}


# Detector classes that are projectiles, not board presence:
# extractObservationForTeam skips anything not isTargetable(). They still get a
# card id (a caller may want to know a snowball is in flight), but an adapter
# building the spatial channels must drop them.
PROJECTILE_CLASSES = frozenset({"giant_snowball"})


class UnitMappingError(KeyError):
    """A detected unit has no engine card. Never silently defaulted."""


def is_board_presence(unit_name: str) -> bool:
    """False for projectiles, which the observation's spatial channels omit."""
    return unit_name not in PROJECTILE_CLASSES


def _norm(name: str) -> str:
    """Compare names ignoring case, spaces, dots and hyphens.

    Needed both ways: the engine writes "Mini PEKKA", "P.E.K.K.A." and "X-Bow"
    while the detector writes "minipekka", "pekka" and "x_bow".
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


@lru_cache(maxsize=1)
def _unit_to_card_name() -> dict[str, str]:
    """Detector unit name -> CRBAB card name, from CRBAB's own tables."""
    from clashroyalebuildabot.namespaces.cards import Cards  # noqa: PLC0415

    spawned_by: dict[str, list[str]] = {}
    for field in dataclasses.fields(Cards):
        card = getattr(Cards, field.name)
        for unit in card.units:
            spawned_by.setdefault(unit.name, []).append(card.name)

    resolved = {}
    for unit, cards in spawned_by.items():
        if unit in PREFERRED_CARD:
            resolved[unit] = PREFERRED_CARD[unit]
        elif len(cards) == 1:
            resolved[unit] = cards[0]
        else:
            raise UnitMappingError(
                f"unit {unit!r} is spawned by {sorted(cards)} and has no entry "
                "in PREFERRED_CARD. Add one -- guessing here would silently "
                "give the observation the wrong card's attributes."
            )
    return resolved


@lru_cache(maxsize=1)
def unit_to_card_id() -> dict[str, int]:
    """Detector unit name -> engine card id, for every class the model emits."""
    import clash_royale_env as engine  # noqa: PLC0415

    by_norm = {_norm(engine.get_card_info(cid)["name"]): cid
               for cid in engine.get_all_card_ids()}

    mapping = {}
    unresolved = []
    for unit, card_name in _unit_to_card_name().items():
        cid = by_norm.get(_norm(card_name))
        if cid is None:
            unresolved.append((unit, card_name))
        else:
            mapping[unit] = cid

    # Detector classes that no card *spawns* -- spell projectiles, whose CRBAB
    # Card declares `units: []` so they never appear in the card->units table.
    # Resolved by direct name match instead.
    from clashroyalebuildabot.constants import DETECTOR_UNITS  # noqa: PLC0415

    for unit in sorted({u.name for u in DETECTOR_UNITS}):
        if unit in mapping:
            continue
        cid = by_norm.get(_norm(unit))
        if cid is not None:
            mapping[unit] = cid

    if unresolved:
        raise UnitMappingError(
            "no engine card matches these CRBAB card names: "
            f"{sorted(unresolved)}. Either the engine's registry lacks the card "
            "or the two spell it differently -- _norm() handles case, spaces, "
            "dots and hyphens, nothing else."
        )
    return mapping


@lru_cache(maxsize=1)
def card_name_to_id() -> dict[str, int]:
    """CRBAB card name -> engine card id.

    Separate from `unit_to_card_id`: the hand holds cards, the board holds
    units, and most names coincide, which is why using one table for the other
    fails quietly. Fireball spawns no unit, so it would resolve to nothing.
    """
    import clash_royale_env as engine  # noqa: PLC0415

    from clashroyalebuildabot.namespaces.cards import Cards  # noqa: PLC0415

    by_norm = {_norm(engine.get_card_info(cid)["name"]): cid
               for cid in engine.get_all_card_ids()}
    mapping = {}
    for field in dataclasses.fields(Cards):
        card = getattr(Cards, field.name)
        cid = by_norm.get(_norm(card.name))
        if cid is not None:
            mapping[card.name] = cid
    return mapping


def hand_card_id_for(card_name: str) -> int:
    """Engine card id for a card in hand, or UNKNOWN_CARD_SIM_ID. The sentinel
    rather than an exception: an unreadable hand slot is routine, while an
    unmappable unit on the board means the detector saw something the
    observation cannot represent.
    """
    if not card_name or card_name == "blank":
        return UNKNOWN_CARD_SIM_ID
    return card_name_to_id().get(card_name, UNKNOWN_CARD_SIM_ID)


def card_id_for(unit_name: str) -> int:
    """Engine card id for a detected unit. Raises rather than defaulting: a wrong
    id fills the attribute channels with another card's stats,
    indistinguishable downstream from a correct reading.
    """
    try:
        return unit_to_card_id()[unit_name]
    except KeyError as exc:
        raise UnitMappingError(
            f"no engine card for detected unit {unit_name!r}. If upstream "
            "retrained the detector with new classes, extend PREFERRED_CARD."
        ) from exc


def coverage() -> tuple[int, list[str]]:
    """(mapped count, detector classes still unmapped). For tests and CI."""
    from clashroyalebuildabot.constants import DETECTOR_UNITS  # noqa: PLC0415

    mapping = unit_to_card_id()
    classes = sorted({u.name for u in DETECTOR_UNITS})
    return len(mapping), [c for c in classes if c not in mapping]
