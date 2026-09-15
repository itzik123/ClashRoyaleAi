"""What a deck turns ON and OFF in this training system -- said out loud.

Run at the top of every training run (`trainers/train.py`) and by
`tools/validate_pipeline.py`. Nothing here changes behaviour; it reports.

WHY. The 2026-09-15 pre-launch audit found one bug four times: a mechanism keyed
to the 2.6 Hog Cycle that, under any other deck, quietly contributed nothing --
no exception, no warning, no log line, and a run that looks healthy:

  * the advisor-target term (10% of log(612) on the placement head) trained on
    zero cards for 5 of 8 plausible replacement decks;
  * the win-condition reward went dead for siege/spell/Miner decks;
  * the lethal-spell and value-spell terms are keyed to Fireball (id 7);
  * 60% of scenario injection builds boards whose answer is a Fireball.

Several of those are FIXED (the advisor and the win condition are derived now);
the rest are reported here, so a deck choice is an informed one and the log of a
run records what it trained with.
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

    # --- hard blockers ------------------------------------------------------
    champions = [c for c in deck if info[c]["is_champion"] or info[c]["is_hero"]]
    if champions:
        out.append(("WARN",
                    f"{', '.join(name(c) for c in champions)} is a Champion/Hero: its "
                    f"ability is trained since 2026-09-16 (rl/abilities.py), but "
                    f"that path is new, and the phase-1 mirror teacher uses the "
                    f"ability by a plain heuristic (ready + an enemy force on the "
                    f"board). Watch Policy/Entropy_Ability and the ability's use in "
                    f"replays early in the run."))

    # --- the win condition ---------------------------------------------------
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

    # --- the advisor target -------------------------------------------------
    table = advisor_target.advisor_cards_for(deck)
    uncovered = [name(c) for c in deck if c not in table]
    msg = (f"advisor target speaks for {len(table)}/8 cards "
           f"({', '.join(f'{name(c)}={k}' for c, k in table.items()) or 'none'}); "
           f"entropy bonus only for: {', '.join(uncovered) or 'none'}")
    out.append(("WARN" if not table else "INFO", msg))

    # --- Fireball-keyed reward terms and scenarios ----------------------------
    if W.FIREBALL_CARD_ID not in deck:
        out.append(("WARN",
                    f"no Fireball: the lethal-spell (W_LETHAL_SPELL={W.W_LETHAL_SPELL}) "
                    f"and value-spell (W_SPELL_VALUE_START={W.W_SPELL_VALUE_START}) "
                    f"reward terms are keyed to card id {W.FIREBALL_CARD_ID} and "
                    f"contribute nothing"))
    spell_scen = getattr(scenarios, "spell_scenario_share", None)
    if spell_scen is not None:
        share = spell_scen(deck)
        if share == 0.0:
            out.append(("INFO", "Fireball scenarios disabled for this deck (no area "
                                "damage spell); bridge-push scenarios keep the full "
                                "injection budget"))

    # --- other facts ----------------------------------------------------------
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
