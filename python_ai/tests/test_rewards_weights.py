"""Every reward weight, pinned to the value on record.

This test is meant to fail when a weight changes: a gameplay-affecting change
invalidates the win-rate history and must not happen unnoticed. If retuning
deliberately, change the number here too and say so in the commit.
"""
import pytest

from python_ai import engine_constants as EC
from python_ai.rewards import weights as W
from python_ai.rewards.elixir_shaping import SOLVENCY_RESERVE, W_SOLVENCY
from python_ai.rl.coverage import PLACEMENT_COVERAGE_COEF

#: name -> the value on record, grouped by what the term does.
RECORDED = {
    # the potential-based tower term
    "W_BLDG": 0.5,
    "W_TROOPS": 0.1,
    "W_ELIXIR_TRADE": 0.03,
    # the deliberately biasing terms
    "W_TOWER_DESTROYED": 0.6,
    "W_FLAWLESS_DEFENSE": 0.5,
    "FLAWLESS_REQUIRES_CROWN": True,
    "W_WIN_CONDITION_DAMAGE": 1.0,
    "DRAW_PENALTY": 1.0,
    # elixir
    "ELIXIR_OVERFLOW_THRESHOLD": 9.0,
    "W_ELIXIR_OVERFLOW": 0.1,
    "MAX_ELIXIR_PER_STEP": 10.0,
    "SOLVENCY_ENABLED": True,
    "SOLVENCY_COEF": 0.1,
    # the spell terms (the solvency reserve is the deck spell's own cost; see
    # test_damage_spell_is_deck_derived)
    "FIREBALL_CARD_ID": 7,
    "W_LETHAL_SPELL": 0.15,
    "W_SPELL_VALUE_START": 0.08,
    "W_SPELL_VALUE_FINAL": 0.0,
    "SPELL_VALUE_ANNEAL_EPISODES": 40000,
    "SPELL_VALUE_ANNEAL_START": 0,
}


@pytest.mark.parametrize("name,value", sorted(RECORDED.items()))
def test_weight_matches_the_value_the_recorded_results_were_earned_at(name, value):
    assert getattr(W, name) == value


def test_the_placement_coverage_coefficient_is_unchanged():
    assert PLACEMENT_COVERAGE_COEF == 0.02


def test_the_spell_reserve_is_the_spells_own_cost_not_a_constant():
    """"Enough elixir to answer with one more card" means the deck's own spell, so
    no Fireball literal. Behaviour is pinned in
    test_damage_spell_is_deck_derived.
    """
    assert not hasattr(W, "SPELL_SOLVENCY_RESERVE")


def test_the_solvency_reserve_and_its_weight_come_from_one_definition():
    """`elixir_shaping` owns them; `weights` re-exports the coefficient so the
    term can be ablated from one place.
    """
    assert W.SOLVENCY_COEF == W_SOLVENCY
    assert SOLVENCY_RESERVE == 4.0


def test_fireballs_damage_and_cost_are_read_from_the_registry():
    """Not copied: the 689 in the source is a fallback for a registry that does
    not expose `damage`.
    """
    import clash_royale_env
    info = clash_royale_env.get_card_info(W.FIREBALL_CARD_ID)
    assert W.FIREBALL_COST == float(info["cost"])
    if "damage" in info:
        assert W.FIREBALL_DAMAGE == float(info["damage"])


def test_the_own_tower_total_is_read_from_a_fresh_board_not_written_out():
    """2*2534 + 4008 = 9076 today, and tower HP is what a balance pass moves.
    """
    assert EC.OWN_TOWER_HP_TOTAL == pytest.approx(9076.0)


def test_the_env_overridable_weights_are_the_ones_meant_to_be_ablated():
    """The env-overridable weights are a short deliberate list: overriding lets an
    A/B run byte-identical code in both arms.
    """
    import pathlib

    import python_ai
    src = pathlib.Path(python_ai.PACKAGE_DIR, "rewards",
                       "weights.py").read_text(encoding="utf-8")
    import re
    # Every CLASH_* literal in the file, including one that wraps onto a
    # continuation line.
    knobs = set(re.findall(r'"(CLASH_[A-Z_]+)"', src))
    assert knobs == {"CLASH_W_WINCON_DAMAGE", "CLASH_SPELL_ANNEAL_EPISODES",
                     "CLASH_SPELL_ANNEAL_START", "CLASH_SOLVENCY",
                     "CLASH_SOLVENCY_COEF"}, knobs


def test_the_biasing_terms_are_documented_as_biasing():
    """Every non-potential-based term biases the optimum by construction and must
    be listed as such in the module docstring.
    """
    import pathlib

    import python_ai
    doc = pathlib.Path(python_ai.PACKAGE_DIR, "rewards",
                       "weights.py").read_text(encoding="utf-8")
    head = doc[:doc.index('"""', 3)]
    for name in ("W_TOWER_DESTROYED", "W_FLAWLESS_DEFENSE",
                 "W_WIN_CONDITION_DAMAGE", "W_SPELL_VALUE_START",
                 "DRAW_PENALTY"):
        assert name in head, f"{name} is biasing and must be listed as such"


def test_fireballs_damage_has_exactly_one_definition_in_the_package():
    """Fireball's damage has one definition. `get_card_info` exposes no damage, so
    nothing derives it, and two copies could drift with nothing to catch it:
    the shaping term and the placement advisor would disagree about the same
    fact.
    """
    import pathlib
    import re

    import python_ai
    root = pathlib.Path(python_ai.PACKAGE_DIR)
    defs = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(("tests/", "venv/", "archive")):
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            m = re.match(r"\s*FIREBALL_DAMAGE\s*=\s*(.+)$", code)
            if not m:
                continue
            rhs = m.group(1)
            # An alias (`= W.FIREBALL_DAMAGE`) re-exports the one definition; a
            # source is a numeric literal or a registry read.
            if re.search(r"\d", rhs) or "get_card_info" in rhs:
                defs.append(f"{rel}:{i}: {line.strip()}")
    assert len(defs) == 1, (
        "Fireball's damage is defined in more than one place:\n"
        + "\n".join(defs))


def test_the_advisor_and_the_shaping_term_agree_on_fireball():
    """The runtime property the single definition buys."""
    from python_ai.advisors import tactics
    from python_ai.rewards import weights
    assert tactics.FIREBALL_DAMAGE == weights.FIREBALL_DAMAGE
    assert tactics.FIREBALL_ID == weights.FIREBALL_CARD_ID
