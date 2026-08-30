"""Guards for the extracted human placement prior.

The prior is served into training as target logits, so a silently wrong one
would teach the placement head confidently incorrect cells. The checks that
matter are the ones that fail when the COORDINATE FRAME is wrong, because that
is the failure this corpus invites: their arena is 32 rows to our 34, their y
runs the opposite way, and their ego is our team 0.

A flipped frame would put the Fireball on our own half and the Hog behind our
king, and both of those are asserted here.
"""
from pathlib import Path

import numpy as np
import pytest

# Import the package FIRST: its __init__ is what puts the unpackaged .pyd on
# sys.path, so `clash_royale_env` is not importable before it.
from perception.replay_mining import prior as P
from perception.replay_mining.extract_prior import (BOARD_H, BOARD_W, N_CELLS,
                                                    cell_from_xy, geometry_stamp)

import clash_royale_env as E  # noqa: E402

# The artifact lives with its CONSUMER: python_ai serves it during training
# and must not import perception (the dependency runs the other way).
# perception is the producer, so this test reaches across to what it made.
ARTIFACT = (Path(__file__).resolve().parents[2] / "python_ai" / "advisors"
            / "data" / "hog26_placement_prior.npz")

HOG, CANNON, FIREBALL = 15, 25, 7


@pytest.fixture(scope="module")
def art():
    if not ARTIFACT.exists():
        pytest.skip(f"prior artifact not built: {ARTIFACT}")
    return np.load(ARTIFACT, allow_pickle=True)


@pytest.fixture(scope="module")
def masks(art):
    deck = [int(c) for c in art["deck"]]
    env = E.ClashRoyaleEnv(deck, deck, max_ticks=3600)
    env.seed(0)
    env.reset()
    out = {}
    for i, cid in enumerate(deck):
        out[i] = np.array([env.is_valid_placement(cid, float(c % BOARD_W),
                                                  float(c // BOARD_W), 0)
                           for c in range(N_CELLS)], dtype=bool)
    return out


def _slot(art, card_id):
    return [int(c) for c in art["deck"]].index(card_id)


def _surface(art, masks, card_id):
    i = _slot(art, card_id)
    p, _ = P.probabilities(art["counts"], i, 0, masks[i])
    return p.reshape(BOARD_H, BOARD_W)


def test_geometry_stamp_matches_the_live_engine(art):
    """A prior built against a different arena must be REFUSED, not reused.

    Same contract `bc_pretrain.load_dataset` enforces for observation_size.
    CLAUDE.md records the arena moving twice; a stale prior would keep serving
    plausible cells for a board that no longer exists.
    """
    keys = [str(k) for k in art["geometry_keys"]]
    vals = dict(zip(keys, [float(v) for v in art["geometry_vals"]]))
    live = geometry_stamp()
    assert set(vals) == set(live)
    for k, v in live.items():
        assert vals[k] == pytest.approx(float(v)), f"arena changed: {k}"


def test_no_probability_mass_on_illegal_cells(art, masks):
    for i in range(len(art["deck"])):
        p, _ = P.probabilities(art["counts"], i, 0, masks[i])
        assert p is not None
        assert float(p[~masks[i]].sum()) == 0.0
        assert float(p.sum()) == pytest.approx(1.0)


def test_fireball_goes_to_the_ENEMY_half(art, masks):
    """The sharpest frame check available: a spell is the only card allowed
    across the river, so a flipped y axis shows up here and nowhere else."""
    g = _surface(art, masks, FIREBALL)
    enemy = float(g[int(E.ARENA_BRIDGE_Y + 1.0):].sum())
    assert enemy > 0.5, f"Fireball mass on the enemy half is only {enemy:.3f}"


def test_hog_is_placed_just_below_the_river(art, masks):
    g = _surface(art, masks, HOG)
    near_bridge = float(g[12:16].sum())
    assert near_bridge > 0.35, f"Hog mass in rows 12-15 is only {near_bridge:.3f}"
    assert float(g[0:6].sum()) < 0.15, "Hog mass piled up behind our own king"


def test_cannon_is_a_defensive_building_on_our_own_half(art, masks):
    g = _surface(art, masks, CANNON)
    assert float(g[0:16].sum()) > 0.9
    row = int(np.argmax(g.sum(axis=1)))
    assert 4 <= row <= 12, f"Cannon modal row {row} is not mid-own-half"


def test_cell_from_xy_rejects_out_of_board_rather_than_truncating():
    """int(-0.87) == 0 silently relocates a deep placement to the back row.

    Measured: before this guard that artifact alone made (8, 0) the Musketeer's
    modal cell with 7.6% of its mass.
    """
    assert cell_from_xy(-0.87, 5.0) is None
    assert cell_from_xy(5.0, -0.87) is None
    assert cell_from_xy(float(BOARD_W), 5.0) is None
    assert cell_from_xy(3.0, 10.0) == 10 * BOARD_W + 3


def test_blur_matches_a_reference_convolution_in_the_INTERIOR():
    """The matmul blur replaced a per-row convolution for speed, so it must be
    the same operator where the two share a convention -- which is everywhere
    except the boundary.

    They differ AT the boundary on purpose. The matmul renormalises each output
    row, so a cell near the wall averages only over real cells; the reference
    pads by replicating the edge value, which duplicates mass that is not there.
    For a surface that is about to become a probability distribution the
    renormalising convention is the correct one, so the interior is what is
    pinned and the edges are asserted separately below.
    """
    rng = np.random.default_rng(0)
    flat = rng.random(N_CELLS)
    got = P._blur(flat, 1.0).reshape(BOARD_H, BOARD_W)

    g = np.arange(-3, 4)
    k = np.exp(-0.5 * (g / 1.0) ** 2)
    k /= k.sum()
    m = flat.reshape(BOARD_H, BOARD_W).copy()
    m = np.apply_along_axis(
        lambda r: np.convolve(np.pad(r, 3, mode="edge"), k, mode="valid"), 1, m)
    m = np.apply_along_axis(
        lambda c: np.convolve(np.pad(c, 3, mode="edge"), k, mode="valid"), 0, m)
    assert np.allclose(got[3:-3, 3:-3], m[3:-3, 3:-3], atol=1e-6)


def test_blur_is_a_weighted_average_and_invents_nothing():
    """Properties the serving path depends on, independent of any reference."""
    rng = np.random.default_rng(1)
    flat = rng.random(N_CELLS)
    out = P._blur(flat, 1.0)
    assert (out >= 0).all()
    # A weighted average cannot leave the range of its inputs -- in particular
    # it cannot manufacture mass at the wall, which is the edge bug above.
    assert out.min() >= flat.min() - 1e-9
    assert out.max() <= flat.max() + 1e-9


def test_blur_spreads_a_delta_without_moving_it():
    delta = np.zeros(N_CELLS)
    centre = 10 * BOARD_W + 9
    delta[centre] = 1.0
    for sigma in (1.0, 2.0):
        out = P._blur(delta, sigma)
        assert int(out.argmax()) == centre, "blur moved the mode"
    # Wider sigma must be strictly more spread out.
    assert P._blur(delta, 2.0).max() < P._blur(delta, 1.0).max()


def test_served_key_is_the_measured_one():
    """`evaluate_prior` found conditioning does not beat the marginal. If this
    changes, it must change with a new measurement attached."""
    assert P.SERVE_KEY == "marginal"
    assert P.BLUR_SIGMA == 1.0
