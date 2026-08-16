"""The adb daemon-start banner lands on STDOUT, ahead of the PNG.

Regression for a live failure on 2026-08-16: `mvp_loop --ensure-match` died
with `PIL.UnidentifiedImageError: cannot identify image file`, which reads like
a corrupt capture or a broken emulator and was neither. HD-Adb.exe prepends

    * daemon not running. starting it now on port 5037 *
    * daemon started successfully *

to STDOUT on the first call after the daemon is down -- exit code 0, stderr
EMPTY, 85 bytes ahead of the PNG signature.

It bites only the FIRST adb call of a session, which is exactly why it survived
so long: anyone debugging interactively runs `adb devices` first and warms the
daemon, so the failure never reproduces under investigation. These tests pin
the decode rather than the emulator, so they need no device.
"""
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from match_nav import decode_screencap  # noqa: E402

# The exact bytes observed, including the trailing newline layout.
BANNER = (b"* daemon not running. starting it now on port 5037 *\n"
          b"* daemon started successfully *\n")


def _png_bytes(size=(8, 4), colour=(10, 200, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


def test_banner_prefixed_capture_still_decodes():
    """The exact live failure: banner on stdout, empty stderr, exit 0."""
    png = _png_bytes()
    img = decode_screencap(BANNER + png, returncode=0, stderr=b"")
    assert img.size == (8, 4)
    assert img.mode == "RGB"


def test_banner_offset_matches_what_was_measured():
    """85 bytes is the number recorded in decode_screencap's docstring."""
    assert len(BANNER) == 85


def test_clean_capture_is_unaffected():
    """A warm daemon returns pure PNG; that path must not regress."""
    img = decode_screencap(_png_bytes(), returncode=0, stderr=b"")
    assert img.size == (8, 4)


def test_pixels_survive_the_strip():
    """Stripping must not shift the image -- decode the colour back out."""
    png = _png_bytes(colour=(7, 199, 33))
    assert decode_screencap(BANNER + png).getpixel((0, 0)) == (7, 199, 33)


def test_no_png_at_all_raises_something_actionable():
    """The old code raised UnidentifiedImageError, which named the wrong cause.

    'Is the emulator running' is the question the operator actually needs.
    """
    with pytest.raises(RuntimeError, match="no PNG signature"):
        decode_screencap(b"error: no devices/emulators found\n")


def test_nonzero_exit_reports_stderr_not_the_image():
    with pytest.raises(RuntimeError, match="device unauthorized"):
        decode_screencap(b"", returncode=1, stderr=b"device unauthorized")
