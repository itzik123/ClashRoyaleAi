"""Generate perception/mapping/card_map.json from the live CardRegistry.

Run this after any change to CardRegistry.h. It PRESERVES every hand-authored
field in the existing map (see MERGE POLICY) and only refreshes what it can
prove from the engine, so re-running it is safe and non-destructive.

    perception/.venv/Scripts/python.exe perception/tools/gen_card_map.py

WHY THIS IS GENERATED AND NOT HAND-WRITTEN
------------------------------------------
CardDefinition::name already holds real Clash Royale card names verbatim
("Hog Rider", "Elite Barbarians", "The Log"). So the bulk of the mapping is
mechanical and should never be typed by hand -- typing it by hand is how you
get a map that is 95% right and wrong in a way nothing detects.

WHAT THE GENERATOR CANNOT DO, AND LEAVES FOR A HUMAN
----------------------------------------------------
1. Evolutions share the base card's name exactly. Registry id 1 and id 128
   are both "Archers"; id 0 and id 135 are both "Knight". A name is therefore
   structurally incapable of identifying a card on its own, which is why the
   schema carries `sim_id` and `sim_evo_id` as separate fields and
   PlacementEvent carries an `is_evolution` flag rather than folding the
   evolved variant into the id. The generator pairs them by name, which is
   sound precisely BECAUSE addEvolution() copies the base name -- but the
   pairing is still marked for review, since a future card whose evolution is
   registered under a different name would pair silently wrong.

2. Cards that exist in the real game and NOT in the simulator cannot be
   enumerated from here at all -- the engine has no notion of what it is
   missing. Those rows have to be added by hand with `sim_id: null`, and
   until they are, detecting one of those cards raises rather than guessing.
   See detect/ and bridge/sim_driver.py for the defined behaviour.

3. Whether ids 165-175 ("Spirit Empress", "Hero Knight", "Hero Wizard", ...)
   correspond to anything in the build being recorded is not knowable from
   the registry. They are emitted with needs_review: true and
   real_verified: false, which keeps them out of the detector's label set
   until a human confirms them.

MERGE POLICY
------------
Regenerating never discards human input. For every entry already present:
  * `real_verified`, `real_only`, `aliases`, `notes` are preserved verbatim.
  * `sim_id`, `sim_evo_id`, `cost`, `is_spell`, `is_building`, `is_champion`,
    `is_hero` are refreshed from the engine.
  * `needs_review` is cleared only where the engine now agrees with a
    human-verified row; it is never set back to false automatically.
Rows present in the file but absent from the registry are kept and marked
`sim_id: null` with a note, rather than deleted -- a card disappearing from
the registry is a fact worth surfacing, not worth silently erasing.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_PERCEPTION_ROOT = Path(__file__).resolve().parent.parent
if str(_PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(_PERCEPTION_ROOT))

from geometry import _engine_module  # noqa: E402

MAP_PATH = _PERCEPTION_ROOT / "mapping" / "card_map.json"

# Registry ids whose real-game counterpart cannot be confirmed from the
# registry alone -- see point 3 in the module docstring. Kept as an explicit
# range check rather than a name prefix match so a card named "Heroic
# Something" that IS real does not get swept up by accident.
_UNVERIFIED_ID_MIN = 165

# CardRegistry.h registers Evolutions in this id band (addEvolution).
_EVOLUTION_ID_MIN = 123
_EVOLUTION_ID_MAX = 163


def slugify(name: str) -> str:
    """Canonical key for a card.

    Punctuation is stripped rather than normalised because the registry is
    internally inconsistent about it -- id 13 is "P.E.K.K.A." while id 5 is
    "Mini PEKKA" -- so any scheme that preserves dots would key the same real
    card two different ways depending on which row it came from.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


def build_rows(engine) -> dict[str, dict]:
    """Registry -> map rows, keyed by slug."""
    playable = set(engine.get_all_card_ids())

    # Evolution ids are excluded from get_all_card_ids() (ClashEnv.h's
    # getAllCardIds skips isEvolution), so they have to be probed directly.
    evolutions: dict[str, int] = {}
    for cid in range(_EVOLUTION_ID_MIN, _EVOLUTION_ID_MAX + 1):
        try:
            info = engine.get_card_info(cid)
        except Exception:
            continue
        evolutions[slugify(info["name"])] = cid

    rows: dict[str, dict] = {}
    for cid in sorted(playable):
        info = engine.get_card_info(cid)
        slug = slugify(info["name"])
        unverified = cid >= _UNVERIFIED_ID_MIN
        rows[slug] = {
            "real_name": info["name"],
            "sim_id": cid,
            "sim_evo_id": evolutions.get(slug),
            "cost": info["cost"],
            "is_spell": info["is_spell"],
            "is_building": info["is_building"],
            "is_champion": info["is_champion"],
            "is_hero": info["is_hero"],
            # False until a human confirms this card exists, under this name,
            # in the build actually being recorded. The detector's label set
            # is filtered on this -- see mapping/__init__.py.
            "real_verified": not unverified,
            "real_only": False,
            "aliases": [],
            "needs_review": unverified or slug in evolutions,
            "notes": (
                "Registry id >= 165 (Spirit Empress / Hero cards). Confirm this "
                "card exists in the recorded build before enabling."
                if unverified
                else "Evolution paired by shared name; confirm the pairing."
                if slug in evolutions
                else ""
            ),
        }
    return rows


def merge(existing: dict, generated: dict[str, dict]) -> dict:
    """Apply MERGE POLICY. Returns the full file body."""
    out_cards: dict[str, dict] = {}
    prior = existing.get("cards", {})

    for slug, gen in generated.items():
        old = prior.get(slug)
        if old is None:
            out_cards[slug] = gen
            continue
        merged = dict(gen)
        for human_field in ("real_verified", "real_only", "aliases", "notes"):
            if human_field in old:
                merged[human_field] = old[human_field]
        # Only ever clears review, never re-raises it.
        merged["needs_review"] = bool(old.get("needs_review", gen["needs_review"]))
        if old.get("real_verified") and old.get("sim_id") == gen["sim_id"]:
            merged["needs_review"] = False
        out_cards[slug] = merged

    # Rows the registry no longer knows about, and hand-added real-only rows.
    for slug, old in prior.items():
        if slug in out_cards:
            continue
        kept = dict(old)
        if not kept.get("real_only"):
            kept["sim_id"] = None
            kept["sim_evo_id"] = None
            kept["needs_review"] = True
            kept["notes"] = (
                (kept.get("notes", "") + " ").strip()
                + " No longer present in CardRegistry -- was it removed?"
            ).strip()
        out_cards[slug] = kept

    return {
        "_comment": (
            "Simulator <-> real-game card mapping. Generated by "
            "perception/tools/gen_card_map.py; hand-authored fields are "
            "preserved on regeneration. sim_id null means the card exists in "
            "the real game but not in this simulator -- placements of it are "
            "reported and NOT injected. See the tool's docstring."
        ),
        "_schema": {
            "real_name": "display name as it appears in the real game",
            "sim_id": "CardRegistry id, or null if unimplemented",
            "sim_evo_id": "CardRegistry id of the Evolution variant, or null",
            "real_verified": "a human confirmed this card exists in the recorded build",
            "real_only": "hand-added: exists in the real game, absent from the sim",
            "aliases": "alternative spellings the detector may emit",
            "needs_review": "generator could not fully determine this row",
        },
        "cards": dict(sorted(out_cards.items())),
    }


def main() -> int:
    engine = _engine_module()
    if engine is None:
        print(
            "gen_card_map: could not import clash_royale_env.\n"
            "  The .pyd is built for Python 3.11 and lives in python_ai/.\n"
            "  Run this with perception/.venv/Scripts/python.exe.",
            file=sys.stderr,
        )
        return 1

    existing = json.loads(MAP_PATH.read_text(encoding="utf-8")) if MAP_PATH.exists() else {}
    body = merge(existing, build_rows(engine))

    MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    MAP_PATH.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    cards = body["cards"]
    print(f"wrote {MAP_PATH.relative_to(_PERCEPTION_ROOT.parent)}")
    print(f"  {len(cards)} cards")
    print(f"  {sum(1 for c in cards.values() if c.get('sim_evo_id') is not None)} with an Evolution")
    print(f"  {sum(1 for c in cards.values() if c.get('sim_id') is None)} with no simulator id")
    print(f"  {sum(1 for c in cards.values() if c.get('needs_review'))} needing review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
