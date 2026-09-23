"""The adb daemon-start banner lands on stdout, ahead of the PNG.

On the first call after the daemon is down, HD-Adb.exe prepends

    * daemon not running. starting it now on port 5037 *
    * daemon started successfully *

to stdout (exit code 0, stderr empty, 85 bytes ahead of the PNG signature), and
PIL raises `UnidentifiedImageError`, which reads like a corrupt capture. It
bites only a session's first adb call, so interactive debugging never
reproduces it. These tests pin the decode, so they need no device.
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
    """85 bytes, the offset measured live."""
    assert len(BANNER) == 85


def test_clean_capture_is_unaffected():
    """A warm daemon returns pure PNG; that path must not regress."""
    img = decode_screencap(_png_bytes(), returncode=0, stderr=b"")
    assert img.size == (8, 4)


def test_pixels_survive_the_strip():
    """Stripping must not shift the image: decode the colour back out."""
    png = _png_bytes(colour=(7, 199, 33))
    assert decode_screencap(BANNER + png).getpixel((0, 0)) == (7, 199, 33)


def test_no_png_at_all_raises_something_actionable():
    """Name the real question, "Is the emulator running", rather than
    UnidentifiedImageError.
    """
    with pytest.raises(RuntimeError, match="no PNG signature"):
        decode_screencap(b"error: no devices/emulators found\n")


def test_nonzero_exit_reports_stderr_not_the_image():
    with pytest.raises(RuntimeError, match="device unauthorized"):
        decode_screencap(b"", returncode=1, stderr=b"device unauthorized")
