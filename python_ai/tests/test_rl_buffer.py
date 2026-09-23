"""RolloutBuffer: every field grows together, or `add()` refuses the row. A field
appended on some steps and not others would stack into a misaligned batch and
silently corrupt the PPO ratio.
"""
import pytest
import torch

from python_ai.rl.buffer import (
    ADVISOR_FIELDS, CORE_FIELDS, TRUNCATION_FIELDS, RolloutBuffer,
)


def _row(buf, value):
    return {name: torch.full((2,), float(value)) for name in buf.fields}


def test_stack_returns_time_major_tensors():
    buf = RolloutBuffer(("a", "b"))
    for t in range(4):
        buf.add(a=torch.full((3,), float(t)), b=torch.zeros(3))
    stacked = buf.stack()
    assert stacked["a"].shape == (4, 3)
    assert torch.equal(stacked["a"][2], torch.full((3,), 2.0))


def test_a_partial_row_is_refused_rather_than_silently_misaligning():
    """The reason the class exists: a partial row would misalign every later row
    with no error anywhere.
    """
    buf = RolloutBuffer(("a", "b"))
    with pytest.raises(KeyError, match="missing"):
        buf.add(a=torch.zeros(3))


def test_an_unexpected_field_is_refused_too():
    """A typo'd field name would otherwise be accepted and never stacked."""
    buf = RolloutBuffer(("a",))
    with pytest.raises(KeyError, match="unexpected"):
        buf.add(a=torch.zeros(3), aa=torch.zeros(3))


def test_every_field_always_has_the_same_length():
    buf = RolloutBuffer(("a", "b", "c"))
    for i in range(7):
        buf.add(**_row(buf, i))
    lengths = {name: len(values) for name, values in buf._data.items()}
    assert set(lengths.values()) == {7}
    assert len(buf) == 7


def test_clear_empties_every_field():
    """A buffer that clears some lists and not others leaks until an OOM hours
    into a run.
    """
    buf = RolloutBuffer(("a", "b"))
    buf.add(**_row(buf, 1))
    buf.clear()
    assert len(buf) == 0
    assert all(v == [] for v in buf._data.values())


def test_stacking_an_empty_buffer_is_an_error_not_a_shape_surprise():
    buf = RolloutBuffer(("a",))
    with pytest.raises(RuntimeError):
        buf.stack()


def test_duplicate_field_names_are_rejected_at_construction():
    with pytest.raises(ValueError):
        RolloutBuffer(("a", "a"))


def test_the_two_pipelines_declare_the_field_sets_they_actually_use():
    """Pipeline 2 stores three extra quantities for bootstrapping through
    truncation; the advisor fields exist only when enabled. The difference
    stays one declaration.
    """
    assert set(TRUNCATION_FIELDS) == {"boot_nonterminal", "trunc_flag",
                                      "trunc_boot"}
    assert set(ADVISOR_FIELDS) == {"coverage_target", "coverage_has"}
    assert not set(CORE_FIELDS) & set(TRUNCATION_FIELDS)
    assert not set(CORE_FIELDS) & set(ADVISOR_FIELDS)
    # Everything the PPO update reads must be in CORE; see rl/ppo.py.
    for required in ("obs", "card_actions", "placement_actions", "decision",
                     "hx_in", "cx_in", "logprobs", "values", "rewards",
                     "masks", "valid", "aux_opp_played", "coverage_slot"):
        assert required in CORE_FIELDS


def test_membership_reports_declared_fields():
    buf = RolloutBuffer(CORE_FIELDS)
    assert "valid" in buf
    assert "trunc_flag" not in buf
