"""Tests for capture/window.py.

`find_game_rect` is tested and `WindowSource` is not, deliberately: the rect
finder is the part with logic and it is a pure function of an array, while the
source is a thin wrapper over an OS capture API that cannot run without a live
emulator. Synthetic surfaces let the failure modes that actually occurred be
reproduced exactly.
"""
from __future__ import annotations

import numpy as np
import pytest

from capture.window import (
    LETTERBOX_MAX,
    NATIVE_HEIGHT,
    NATIVE_WIDTH,
    GameRectError,
    find_game_rect,
)

# The real measured geometry: a 1920x1020 surface (physical pixels at 125% DPI,
# NOT the 1536x816 GetWindowRect reports) holding a 549-wide game.
SURFACE_W, SURFACE_H = 1920, 1020
GAME_X, GAME_W = 665, 549
TITLE_H = 42
TOOLBAR_X = 1879


def surface(*, game_x=GAME_X, game_w=GAME_W, title=True, toolbar=True,
            width=SURFACE_W, height=SURFACE_H) -> np.ndarray:
    """A window capture: black pillarbox, bright game, plus the chrome."""
    buf = np.zeros((height, width, 3), np.uint8)
    game_h = int(round(game_w * NATIVE_HEIGHT / NATIVE_WIDTH))
    y0 = max(0, height - game_h)
    buf[y0:, game_x:game_x + game_w] = 200
    if title:
        # Spans the FULL width, including the game's own columns. This is what
        # defeats a naive row scan.
        buf[:TITLE_H, :] = 60
    if toolbar:
        buf[:, TOOLBAR_X:] = 90
    return buf


def test_finds_the_measured_geometry():
    x, y, w, h = find_game_rect(surface())
    assert (x, w) == (GAME_X, GAME_W)
    assert h == int(round(GAME_W * NATIVE_HEIGHT / NATIVE_WIDTH))
    assert y + h <= SURFACE_H


def test_toolbar_is_not_mistaken_for_the_game():
    """The right-hand toolbar is a bright non-black column run too. It is
    narrower, which is the only thing separating them."""
    x, _y, w, _h = find_game_rect(surface())
    assert x + w <= TOOLBAR_X


def test_title_bar_does_not_stretch_the_rect():
    """The title bar spans the game's columns and is not black, so scanning
    rows for non-black content runs straight through it and reports a game
    ~1018 px tall inside a 976 px game. Height comes from the ASPECT instead."""
    _x, y, w, h = find_game_rect(surface())
    assert h == pytest.approx(w * NATIVE_HEIGHT / NATIVE_WIDTH, abs=1)
    assert y >= TITLE_H - 2


def test_aspect_matches_the_android_panel():
    _x, _y, w, h = find_game_rect(surface())
    assert w / h == pytest.approx(NATIVE_WIDTH / NATIVE_HEIGHT, abs=0.01)


@pytest.mark.parametrize("game_w", [400, 549, 700])
def test_tracks_a_resized_window(game_w):
    """The rect is found per frame rather than hardcoded, because the window
    can be moved or resized under a running loop."""
    tall = int(round(game_w * NATIVE_HEIGHT / NATIVE_WIDTH)) + TITLE_H
    x, _y, w, h = find_game_rect(
        surface(game_x=300, game_w=game_w, height=tall))
    assert (x, w) == (300, game_w)
    assert h == int(round(game_w * NATIVE_HEIGHT / NATIVE_WIDTH))


def test_height_is_clamped_to_the_surface():
    """A game wider than the surface can show at its own aspect must not
    produce a rect taller than the buffer -- the crop would silently come back
    short and every y-coordinate downstream would be measured against a height
    that does not exist."""
    buf = surface(game_x=300, game_w=700, height=1020)
    _x, y, _w, h = find_game_rect(buf)
    assert y >= 0
    assert y + h <= buf.shape[0]


def test_a_blank_surface_raises_rather_than_returning_nonsense():
    """A minimised or non-rendering window gives an all-black buffer. Returning
    an empty rect would feed the detector a zero-size image several stages
    later, where the cause is unrecoverable."""
    with pytest.raises(GameRectError):
        find_game_rect(np.zeros((SURFACE_H, SURFACE_W, 3), np.uint8))


def test_rejects_a_non_image():
    with pytest.raises(GameRectError):
        find_game_rect(np.zeros((10, 10), np.uint8))


def test_letterbox_threshold_is_above_pure_black():
    """Compression noise on the surface is not exactly zero, but the pillarbox
    must not be confused with dark GAME content either."""
    assert 0 < LETTERBOX_MAX < 64


def test_near_black_noise_is_still_letterbox():
    buf = surface()
    rng = np.random.default_rng(0)
    noise = rng.integers(0, LETTERBOX_MAX, size=buf.shape, dtype=np.uint8)
    buf = np.where(buf == 0, noise, buf).astype(np.uint8)
    x, _y, w, _h = find_game_rect(buf)
    assert (x, w) == (GAME_X, GAME_W)
