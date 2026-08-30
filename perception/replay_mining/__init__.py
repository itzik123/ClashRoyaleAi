"""Mine external Clash Royale replay corpora and measure whether our engine can
reconstruct them.

The corpus this targets is `wty-yy/Clash-Royale-Replay-Dataset`, whose
`fast_hog_2.6/` half is expert play on a deck that is card-for-card our
`DEFAULT_DECK`. Those episodes carry a LABELLED ego action stream and a
per-frame detected unit grid for BOTH sides -- but no observations in our
engine's layout, so behaviour cloning from them requires REPLAYING the
placements through our simulator and reading the observation back out.

That only works if our engine's board still resembles the real match after a
minute of divergence. Nothing here assumes it does; `divergence.py` measures it.
"""

# Every module here drives the simulator, and `clash_royale_env` is an
# unpackaged .pyd living in python_ai/. Importing python_ai is what puts it on
# sys.path (see python_ai/__init__.py), so doing it once here means each entry
# point does not need its own four-line bootstrap -- and `python -m
# perception.replay_mining.<x>` works the same as being called from a script
# that had already imported python_ai.
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))
import python_ai as _python_ai  # noqa: E402,F401
