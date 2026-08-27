"""RolloutBuffer: the invariant that eighteen bare Python lists could not hold.

The buffer replaces per-field `obs_buffer.append(...)` / `torch.stack(...)` /
`.clear()` triples that appeared in three places each, in two files. The failure
mode that motivated the class is SILENT: a field appended on some steps and not
others stacks into a misaligned batch, and the misalignment surfaces as a
corrupted PPO ratio rather than as an error. `add()` refusing a partial row is
what makes that impossible.
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
    """THE reason this class exists.

    A field that is appended on some steps and not others produces a batch where
    row t of one tensor belongs to a different timestep than row t of another.
    Nothing downstream can detect that -- the shapes still broadcast, the update
    still runs, and the PPO ratio is quietly wrong.
    """
    buf = RolloutBuffer(("a", "b"))
    with pytest.raises(KeyError, match="missing"):
        buf.add(a=torch.zeros(3))


def test_an_unexpected_field_is_refused_too():
    """A typo'd field name would otherwise be accepted and then never stacked."""
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
    """A buffer that clears some lists and not others grows without bound, and
    the leak shows up as an OOM many hours into a run."""
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
    """Pipeline 2 stores three extra quantities for correct bootstrapping
    through a truncation; the advisor fields exist only when it is enabled.

    Pinned so the difference between the pipelines stays ONE declaration rather
    than being implied by which append calls happen to run.
    """
    assert set(TRUNCATION_FIELDS) == {"boot_nonterminal", "trunc_flag",
                                      "trunc_boot"}
    assert set(ADVISOR_FIELDS) == {"coverage_target", "coverage_has"}
    assert not set(CORE_FIELDS) & set(TRUNCATION_FIELDS)
    assert not set(CORE_FIELDS) & set(ADVISOR_FIELDS)
    # Everything the PPO update reads must be in CORE -- see rl/ppo.py.
    for required in ("obs", "card_actions", "placement_actions", "decision",
                     "hx_in", "cx_in", "logprobs", "values", "rewards",
                     "masks", "valid", "aux_opp_played", "coverage_slot"):
        assert required in CORE_FIELDS


def test_membership_reports_declared_fields():
    buf = RolloutBuffer(CORE_FIELDS)
    assert "valid" in buf
    assert "trunc_flag" not in buf
