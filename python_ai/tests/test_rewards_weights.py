"""Every reward weight, pinned to the value the recorded win rates were earned at.

READ THIS BEFORE "FIXING" A FAILURE HERE. This test is meant to fail when a
weight changes. That is its job: CLAUDE.md's standing rule is that a
gameplay-affecting change invalidates the win-rate history, and a weight that
can be edited without anything noticing is a silent invalidation. If you are
retuning deliberately, change the number here too and say so in the commit --
the failure is the paperwork, not an obstacle.

It exists because a weight moving unnoticed has already happened in a subtler
form: `spell_value_weight` was dead code for a whole training era because no
test ever varied its argument, so the anneal its own comment block described
never ran and every win rate in that period was earned at a constant 0.08.
"""
import pytest

from python_ai import engine_constants as EC
from python_ai.rewards import weights as W
from python_ai.rewards.elixir_shaping import SOLVENCY_RESERVE, W_SOLVENCY
from python_ai.rl.coverage import PLACEMENT_COVERAGE_COEF

#: name -> the value on record. Grouped by what the term does.
RECORDED = {
    # the potential-based tower term
    "W_BLDG": 0.5,
    "W_TROOPS": 0.1,
    "W_ELIXIR_TRADE": 0.03,
    # the DELIBERATELY biasing terms
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
    # the spell terms
    "FIREBALL_CARD_ID": 7,
    "W_LETHAL_SPELL": 0.15,
    "W_SPELL_VALUE_START": 0.08,
    "W_SPELL_VALUE_FINAL": 0.0,
    "SPELL_VALUE_ANNEAL_EPISODES": 40000,
    "SPELL_VALUE_ANNEAL_START": 0,
    # SPELL_SOLVENCY_RESERVE (4.0) was retired 2026-09-16: the reserve is now the
    # deck spell's own cost, which for the 2.6 deck's Fireball is still 4.0 --
    # `test_damage_spell_is_deck_derived` pins that the 2.6 reward is unchanged.
}


@pytest.mark.parametrize("name,value", sorted(RECORDED.items()))
def test_weight_matches_the_value_the_recorded_results_were_earned_at(name, value):
    assert getattr(W, name) == value


def test_the_placement_coverage_coefficient_is_unchanged():
    assert PLACEMENT_COVERAGE_COEF == 0.02


def test_the_spell_reserve_is_the_spells_own_cost_not_a_constant():
    """"Enough elixir to answer with one more card" is what makes the solvency
    gate on the spell term asymmetric rather than arbitrary -- and "one more
    card" means THE DECK'S spell, so it must not survive as a Fireball literal.
    Behaviour is pinned in test_damage_spell_is_deck_derived."""
    assert not hasattr(W, "SPELL_SOLVENCY_RESERVE")


def test_the_solvency_reserve_and_its_weight_come_from_one_definition():
    """`elixir_shaping` owns them; `weights` re-exports the coefficient so the
    term can be ablated from one place. Two literals would drift."""
    assert W.SOLVENCY_COEF == W_SOLVENCY
    assert SOLVENCY_RESERVE == 4.0


def test_fireballs_damage_and_cost_are_read_from_the_registry():
    """Not copied. A balance change must propagate, never leave this silently
    wrong -- the 689 in the source is a FALLBACK for a registry that does not
    expose `damage`, not a second copy of it."""
    import clash_royale_env
    info = clash_royale_env.get_card_info(W.FIREBALL_CARD_ID)
    assert W.FIREBALL_COST == float(info["cost"])
    if "damage" in info:
        assert W.FIREBALL_DAMAGE == float(info["damage"])


def test_the_own_tower_total_is_read_from_a_fresh_board_not_written_out():
    """2*2534 + 4008 = 9076 today, and the tower HPs are exactly the kind of
    number a balance pass moves."""
    assert EC.OWN_TOWER_HP_TOTAL == pytest.approx(9076.0)


def test_the_env_overridable_weights_are_the_ones_meant_to_be_ablated():
    """Overridability is a deliberate, short list: a term is env-overridable so
    an A/B can run BYTE-IDENTICAL code in both arms, which is the only way the
    comparison attributes the difference to the term rather than to two scripts.
    """
    import pathlib

    import python_ai
    src = pathlib.Path(python_ai.PACKAGE_DIR, "rewards",
                       "weights.py").read_text(encoding="utf-8")
    import re
    # Every CLASH_* literal in the file, not just the ones on the same line as
    # os.environ.get -- one of them wraps onto a continuation line, and a regex
    # that missed it would report a shorter list than reality.
    knobs = set(re.findall(r'"(CLASH_[A-Z_]+)"', src))
    assert knobs == {"CLASH_W_WINCON_DAMAGE", "CLASH_SPELL_ANNEAL_EPISODES",
                     "CLASH_SPELL_ANNEAL_START", "CLASH_SOLVENCY",
                     "CLASH_SOLVENCY_COEF"}, knobs


def test_the_biasing_terms_are_documented_as_biasing():
    """Every non-potential-based term biases the optimum by construction, and
    this project's rule is that such a term is stated rather than hidden. The
    module docstring carries the list; this checks it did not lose an entry."""
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
    """Two modules each carried `FIREBALL_DAMAGE = 689.0`.

    Neither could derive it -- `get_card_info` exposes cost, name, is_spell and
    placement_radius, but NOT damage -- so `weights.py` uses a
    derive-if-available-else-literal expression and `tactics.py` had a bare
    literal under a comment claiming it was "read from the registry", which it
    was not.

    That is the worst shape for a second copy: nothing derives it, so nothing
    catches the two drifting apart, and a balance change to Fireball would
    leave the shaping term and the placement advisor disagreeing about the same
    physical fact -- one deciding a tower is lethal, the other deciding the
    spell does not catch enough value.
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
            # An ALIAS (`= W.FIREBALL_DAMAGE`) re-exports the one definition
            # and is not a second copy. A SOURCE is a numeric literal or a
            # registry read -- those are what can drift apart.
            if re.search(r"\d", rhs) or "get_card_info" in rhs:
                defs.append(f"{rel}:{i}: {line.strip()}")
    assert len(defs) == 1, (
        "Fireball's damage is defined in more than one place:\n"
        + "\n".join(defs))


def test_the_advisor_and_the_shaping_term_agree_on_fireball():
    """The property the single definition buys. Kept as a separate assertion
    because it is the one that matters at runtime."""
    from python_ai.advisors import tactics
    from python_ai.rewards import weights
    assert tactics.FIREBALL_DAMAGE == weights.FIREBALL_DAMAGE
    assert tactics.FIREBALL_ID == weights.FIREBALL_CARD_ID
