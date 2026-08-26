# Exports for clashroyalebuildabot
#
# Vendored as a SENSOR only. Upstream's own agent -- the PyQt GUI, the
# rule-based `actions/`, the live-play `Bot` loop, the ADB `Emulator` and the
# Qt `Visualizer` -- was removed in the 2026-08-24 cleanup: this project brings
# its own agent, and that import chain pulled PyQt6 and `keyboard` into a
# module whose requirements.txt states it never opens a window.
from . import constants
from .detectors import (
    CardDetector,
    Detector,
    NumberDetector,
    OnnxDetector,
    ScreenDetector,
    UnitDetector,
)
from .namespaces import Cards, Screens, State, Units

__all__ = [
    "constants",
    "Cards",
    "Screens",
    "Units",
    "State",
    "Detector",
    "OnnxDetector",
    "ScreenDetector",
    "NumberDetector",
    "UnitDetector",
    "CardDetector",
]
