"""Report what a deck turns on and off in this training system.

Printed at the top of every training run and by `tools/validate_pipeline.py`;
changes nothing. Several mechanisms were once keyed to the 2.6 Hog Cycle and
silently contributed nothing under another deck; most now derive from the deck,
and whatever a deck still switches off is stated here.
"""
import clash_royale_env as E

from python_ai.deck import describe


def validate_deck(deck, *, strict=True):
    """[(level, message), ...] with level in ERROR / WARN / INFO.

    strict: raise ValueError listing every ERROR. Returns the report otherwise.
    """
    from python_ai.advisors import advisor_target, card_probes
    from python_ai.envs import scenarios
    from python_ai.opponents import teacher
    from python_ai.rewards import weights as W

    deck = [int(c) for c in deck]
    info = {c: E.get_card_info(c) for c in deck}
    name = lambda c: info[c]["name"] if c in info else E.get_card_info(c)["name"]  # noqa: E731
    out = [("INFO", f"deck: {describe(deck)}")]

    costs = [float(info[c]["cost"]) for c in deck]
    out.append(("INFO", f"average cost {sum(costs) / len(costs):.3f}, "
                        f"max {max(costs):g}, min {min(costs):g}"))

    champions = [c for c in deck if info[c]["is_champion"] or info[c]["is_hero"]]
    if champions:
        out.append(("WARN",
                    f"{', '.join(name(c) for c in champions)} is a Champion/Hero: its "
                    f"ability is trained since 2026-09-16 (rl/abilities.py), but "
                    f"that path is new, and the phase-1 mirror teacher uses the "
                    f"ability by a plain heuristic (ready + an enemy force on the "
                    f"board). Watch Policy/Entropy_Ability and the ability's use in "
                    f"replays early in the run."))

    roles = teacher.card_roles(deck)
    wincon = next((c for c, r in roles.items() if r == "wincon"), None)
    if wincon is None:
        out.append(("WARN",
                    "no win condition resolves (no building-targeter, deploy-anywhere "
                    "troop, siege building or spawning spell): W_WIN_CONDITION_DAMAGE "
                    "contributes nothing and the mirror teacher has no push to plan"))
    else:
        per = teacher.wincon_damage_per_elixir(wincon)
        level = "WARN" if per < teacher.WINCON_WEAK_DAMAGE_PER_ELIXIR else "INFO"
        out.append((level,
                    f"win condition: {name(wincon)} ({per:.0f} tower HP per elixir alone"
                    f"{' -- WEAK: below ' + str(int(teacher.WINCON_WEAK_DAMAGE_PER_ELIXIR)) if level == 'WARN' else ''})"))

    table = advisor_target.advisor_cards_for(deck)
    uncovered = [name(c) for c in deck if c not in table]
    msg = (f"advisor target speaks for {len(table)}/8 cards "
           f"({', '.join(f'{name(c)}={k}' for c, k in table.items()) or 'none'}); "
           f"entropy bonus only for: {', '.join(uncovered) or 'none'}")
    out.append(("WARN" if not table else "INFO", msg))

    # The spell terms follow the deck's damage spell.
    spell = card_probes.damage_spell(deck)
    if spell is None:
        out.append(("WARN",
                    f"no damaging area spell: the lethal-spell "
                    f"(W_LETHAL_SPELL={W.W_LETHAL_SPELL}) and value-spell "
                    f"(W_SPELL_VALUE_START={W.W_SPELL_VALUE_START}) reward terms "
                    f"contribute nothing for this deck"))
    else:
        sid, tower_dmg, cost = spell
        out.append(("INFO",
                    f"spell reward terms follow {name(sid)}: lethal window at enemy "
                    f"tower hp <= {tower_dmg:.0f} (its measured Crown Tower damage), "
                    f"trades priced at {cost:.0f} elixir"))
    spell_scen = getattr(scenarios, "spell_scenario_share", None)
    if spell_scen is not None:
        share = spell_scen(deck)
        if share == 0.0:
            out.append(("INFO", "Fireball scenarios disabled for this deck (no area "
                                "damage spell); bridge-push scenarios keep the full "
                                "injection budget"))

    damaging_spells = [c for c in deck if card_probes.spell_effect(c) is not None]
    if not damaging_spells:
        out.append(("INFO", "no area-damage spell in the deck"))
    if not any(card_probes.building_defends(c) for c in deck):
        out.append(("INFO", "no defensive building: the building advisor rule never fires"))

    if strict:
        errors = [m for lvl, m in out if lvl == "ERROR"]
        if errors:
            raise ValueError("validate_deck:\n  " + "\n  ".join(errors))
    return out


def print_report(report, file=None):
    for lvl, msg in report:
        print(f">>> [DECK {lvl}] {msg}", file=file)
