"""Search must propose beyond a diffuse head, or it can only rank noise.

THE MEASUREMENT
---------------
14 threatened scenario states, Cannon in hand and affordable, elixir PAID
through env.step, scored by the engine (tower HP conceded, lower better):

    policy today                  5110
    Cannon @ head argmax          4932    +178   better in  6/14
    Cannon @ search top-k         4874    +236   better in  8/14
    Cannon @ search WIDE          4116    +994   better in 10/14
    Cannon @ engine oracle        3886   +1223   better in 14/14

`build_candidates` expands the top-k cells of the placement head's own
distribution, so when that head is diffuse its top-2 are near-arbitrary and
search gains almost nothing over the raw argmax (+58 HP). Widening the proposal
set is worth 4.2x more and captures 81% of the oracle ceiling.

The oracle arm also settles a question the earlier "+185, coin flip" result left
open: a WELL-PLACED Cannon is worth +1223 HP net of its 3 elixir and helps in
every single state (sign test p ~ 6e-5). The card is not marginal. Its placement
was.

WHERE THE THRESHOLD COMES FROM, AND WHAT IT DOES NOT MEAN
---------------------------------------------------------
Mean top-1 placement probability per deck card, ep-32,484 policy:

    Hog Rider  0.7654 | Musketeer 0.3939 | Skeletons 0.3527
    ------------------ largest gap in the diffuse region: 0.227 ------------
    Ice Golem  0.1259 | Fireball  0.1074 | Ice Spirit 0.0987
    Cannon     0.0970 | The Log   0.0288

`WIDE_PROPOSAL_TOP1 = 0.25` sits in that gap: 1.4x below Skeletons, 2.0x above
Ice Golem.

IT IS NOT A DEAD-CARD DETECTOR, and reading it as one would be wrong. Ice Golem
and Ice Spirit are among the most-played cards in the deck (P(play|in hand)
0.187 and 0.325) and their heads are just as diffuse as the Cannon's. For a
cheap cycle card a flat placement map may be CORRECT -- placement genuinely
matters less. So the threshold selects "search has nothing useful to rank here",
which is the question search actually needs answered, and widening a card whose
placement does not matter costs compute and not correctness.
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
    """The shipping behaviour must be unchanged where it works.

    Hog Rider sits at 0.7654 top-1 with 8 distinct cells: search over its own
    top-k is exactly right and must not start paying for a wide sweep.
    """
    cells = propose_cells(_sharp(), k_cells=2)
    assert len(cells) == 2
    assert 100 in cells, "the head's argmax must still be proposed"


def test_a_diffuse_head_is_widened():
    cells = propose_cells(_diffuse(), k_cells=2)
    assert len(cells) > 2, "a diffuse head must get more than its own top-k"


def test_the_widened_set_is_spatially_spread_not_clustered():
    """A wide set drawn by PROBABILITY would just be more of the same noise.

    The point is coverage of the board, so the proposals must span it. This is
    what separates the measured +994 arm from the +236 one.
    """
    from python_ai.engine_constants import BOARD_W
    cells = propose_cells(_diffuse(), k_cells=2)
    xs = {c % BOARD_W for c in cells}
    ys = {c // BOARD_W for c in cells}
    assert len(xs) >= 4 and len(ys) >= 4, f"not spread: {len(xs)} x, {len(ys)} y"


def test_illegal_cells_are_never_proposed():
    """-inf marks a cell this card may not occupy. Proposing one would have the
    engine silently drop the play, and the candidate would score as a no-op
    while looking like a real alternative."""
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
    """Bounds the rule from the other side -- without this the test suite would
    pass for a function that widened everything."""
    n = 612
    rest = (1.0 - 0.30) / (n - 1)
    v = [rest] * n
    v[7] = 0.30                            # above 0.25
    assert len(propose_cells(_logits(v), k_cells=2)) == 2
