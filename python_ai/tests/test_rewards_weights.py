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
    "SPELL_SOLVENCY_RESERVE": 4.0,
}


@pytest.mark.parametrize("name,value", sorted(RECORDED.items()))
def test_weight_matches_the_value_the_recorded_results_were_earned_at(name, value):
    assert getattr(W, name) == value


def test_the_placement_coverage_coefficient_is_unchanged():
    assert PLACEMENT_COVERAGE_COEF == 0.02


def test_the_spell_reserve_matches_fireballs_own_cost():
    """"Enough elixir to answer with one more card" is what makes the solvency
    gate on the spell term asymmetric rather than arbitrary."""
    assert W.SPELL_SOLVENCY_RESERVE == W.FIREBALL_COST


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
