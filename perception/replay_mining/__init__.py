"""Mine external Clash Royale replay corpora and measure whether our engine can
reconstruct them.

The target is `wty-yy/Clash-Royale-Replay-Dataset`, whose `fast_hog_2.6/` half
is expert play on our `DEFAULT_DECK`. Episodes carry a labelled ego action
stream and a per-frame detected unit grid for both sides, but no observations
in our layout, so behaviour cloning from them means replaying the placements
through our simulator and reading the observation back. That works only if the
engine's board still resembles the real match after a minute; `divergence.py`
measures it.
"""

# Every module here drives the simulator, and importing python_ai puts the
# unpackaged .pyd on sys.path, so `python -m perception.replay_mining.<x>`
# works without a per-entry bootstrap.
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))
import python_ai as _python_ai  # noqa: E402,F401
