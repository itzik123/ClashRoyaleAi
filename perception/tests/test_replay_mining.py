"""Guards for the replay-mining probe.

Two things here can go wrong silently and did during development:

  - the KataCR card index. It is NOT `card_list.py`'s 126-card list; it is the
    sorted directory names of the 2.6-deck classification dataset. Reading it
    with the wrong table produces plausible card names ('mirror', 'fire-spirit')
    and a completely wrong event stream.
  - the frame timebase. `state['time']` resets on the victory screen, and
    deriving the frame rate from the endpoints reported 641 fps on one episode
    and 1.2e9 on another, which collapsed every timestamp to zero and silently
    emptied the metrics.
"""
import numpy as np
import pytest

from perception.replay_mining import katacr_format as kf


def test_katacr_card_classes_are_the_default_deck_plus_empty_and_evolutions():
    """The corpus's 2.6 deck must be exactly our DEFAULT_DECK.

    If this fails the corpus is a different deck and its value drops sharply --
    which is the step-0 check, pinned so a future dataset swap cannot pass
    unnoticed.
    """
    playable = {c for c in kf.KATACR_CARD_CLASSES
                if c != "empty" and not c.endswith("-evolution")}
    assert playable == {
        "cannon", "fireball", "hog-rider", "ice-golem",
        "ice-spirit", "musketeer", "skeletons", "the-log",
    }
    assert kf.KATACR_CARD_CLASSES[kf.EMPTY_CARD_INDEX] == "empty"
    # 'empty' must land where sorting puts it, which is what EMPTY_CARD_INDEX
    # pins upstream. Restating the index without this check would let the two
    # drift apart.
    assert kf.KATACR_CARD_CLASSES == sorted(kf.KATACR_CARD_CLASSES)


def _fake_state(times):
    return [{"time": t, "unit_infos": [], "cards": [1] * 5, "elixir": 5}
            for t in times]


def test_timebase_recovers_5fps_and_trims_the_victory_screen():
    clean = [1 + i // 5 for i in range(1000)]          # 5 fps, whole seconds
    spf, off, n_valid = kf._timebase(_fake_state(clean + [3]))
    assert spf == pytest.approx(0.2, abs=1e-3)
    assert n_valid == 1000                             # the reset frame is cut
    assert (off + spf * 999) == pytest.approx(clean[-1], abs=1.5)


def test_timebase_survives_an_isolated_ocr_blip():
    """A single misread second must not move the frame rate."""
    t = [1 + i // 5 for i in range(1000)]
    t[400] = 250                                       # one wild misread
    spf, _, n_valid = kf._timebase(_fake_state(t))
    assert spf == pytest.approx(0.2, abs=1e-3)
    assert n_valid >= 990


def test_timebase_never_returns_a_degenerate_rate():
    """The failure that started this: endpoints equal -> fps of 1.2e9."""
    spf, _, _ = kf._timebase(_fake_state([3] * 50))
    assert spf == pytest.approx(0.2)                   # documented dataset rate
