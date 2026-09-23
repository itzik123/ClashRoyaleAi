"""Search proposes beyond a diffuse head, or it can only rank noise.

`build_candidates` expands the placement head's own top-k cells; when the head
is diffuse those are near-arbitrary. Below `WIDE_PROPOSAL_TOP1` search widens
to a spatially spread proposal set.

The threshold sits in the largest gap between the deck's sharp heads and its
diffuse ones. It is not a dead-card detector: cheap cycle cards can have
correctly flat maps, where widening costs compute but not correctness.
"""
import torch

from python_ai.search.search import WIDE_PROPOSAL_TOP1, propose_cells


def _logits(probs):
    return torch.log(torch.tensor(probs, dtype=torch.float32))


def _sharp(n=612, peak=0.80):
    rest = (1.0 - peak) / (n - 1)
    v = [rest] * n
    v[100] = peak
    v[101] = rest * 5          # a real runner-up for top-k to find
    return _logits(v)


def _diffuse(n=612):
    return _logits([1.0 / n] * n)


def test_a_sharp_head_is_left_alone():
    """Where the head is sharp, search over its own top-k is right and must not
    pay for a wide sweep.
    """
    cells = propose_cells(_sharp(), k_cells=2)
    assert len(cells) == 2
    assert 100 in cells, "the head's argmax must still be proposed"


def test_a_diffuse_head_is_widened():
    cells = propose_cells(_diffuse(), k_cells=2)
    assert len(cells) > 2, "a diffuse head must get more than its own top-k"


def test_the_widened_set_is_spatially_spread_not_clustered():
    """Proposals drawn by probability would be more of the same noise; they must
    span the board.
    """
    from python_ai.engine_constants import BOARD_W
    cells = propose_cells(_diffuse(), k_cells=2)
    xs = {c % BOARD_W for c in cells}
    ys = {c // BOARD_W for c in cells}
    assert len(xs) >= 4 and len(ys) >= 4, f"not spread: {len(xs)} x, {len(ys)} y"


def test_illegal_cells_are_never_proposed():
    """-inf marks a cell this card may not occupy; proposing it would have the
    engine silently drop the play while it looked like a real alternative.
    """
    n = 612
    v = torch.full((n,), 1.0 / n)
    lg = torch.log(v)
    lg[: n // 2] = float("-inf")          # first half illegal

    cells = propose_cells(lg, k_cells=2)
    assert cells, "some legal cell must survive"
    assert all(c >= n // 2 for c in cells)


def test_the_threshold_is_the_documented_one():
    """Pinned so the measured siting cannot drift silently."""
    assert WIDE_PROPOSAL_TOP1 == 0.25


def test_a_head_just_above_the_threshold_is_not_widened():
    """Bounds the rule from the other side: a function that widened everything
    would otherwise pass.
    """
    n = 612
    rest = (1.0 - 0.30) / (n - 1)
    v = [rest] * n
    v[7] = 0.30                            # above 0.25
    assert len(propose_cells(_logits(v), k_cells=2)) == 2
