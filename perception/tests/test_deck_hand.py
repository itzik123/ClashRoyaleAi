"""The deck-restricted hand classifier, pinned against committed crops.

Run against both decks: a pool is deck-specific, so one passing proves only
that deck's templates. The two suites differ in deck, recording and emulator
window position:

  giant  the July batch's deck, from assets/live/match_practice_01
  hog26  gym_wrapper.DEFAULT_DECK (2.6 Hog Cycle), which the live loop plays,
         from assets/recordings/2026-09-02 19-33-22.mp4

The fixture is crops, not frames: `assets/live/` is gitignored, so a test bound
to it would skip on a fresh clone. Two dozen 61x87 crops are ~350 KB.

The four `kind`s are the four things a slot can be doing:

  clean   an affordable card, coloured, cost badge showing
  dimmed  unaffordable: the whole card, badge included, is rendered in true
          greyscale, so the magenta badge test fails on a slot that holds a
          card (15.1% of in-match slots on giant, 44.2% on hog26)
  lifted  selected: the art moves up ~11 px with a highlight bar below, and a
          rigid template's correlation collapses from ~0.99 to ~0.15 (on both
          recordings, so a property of the game's UI)
  empty   the blue card-back between a play and the next card. The only
          state that should read `blank`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_ASSETS = Path(__file__).resolve().parent / "assets" / "hand"
_TEMPLATES = _ROOT / "config" / "templates"

SUITES = {
    "giant": (_ASSETS / "crops", _TEMPLATES / "deck_pool_giant"),
    "hog26": (_ASSETS / "crops_hog26", _TEMPLATES / "deck_pool_hog26"),
}


def _available(name):
    crops, pool = SUITES[name]
    return (crops / "manifest.json").exists() and (pool / "icons.json").exists()


SUITE_PARAMS = [
    pytest.param(name, marks=pytest.mark.skipif(
        not _available(name), reason=f"{name} fixture or pool absent"))
    for name in sorted(SUITES)
]


@pytest.fixture(params=SUITE_PARAMS)
def suite(request):
    crops, pool = SUITES[request.param]
    return request.param, crops, pool


def _deck(pool):
    """The deck a pool was built for, derived from the pool.

    Not imported from `tools.audit_hand_id`: `perception/tools/` shares its
    name with the repo root's `tools/`, which conftest puts on sys.path, so
    under pytest that import resolves into the wrong package.
    """
    from clashroyalebuildabot.namespaces.cards import Cards
    from live.deck_hand import load_pool

    names = set(load_pool(pool))
    return [c for c in vars(Cards).values()
            if getattr(c, "name", None) in names]


def _detector(pool):
    from live.deck_hand import DeckHandDetector
    return DeckHandDetector(_deck(pool), pool)


def _manifest(crops):
    return json.loads((crops / "manifest.json").read_text(encoding="utf-8"))


def _window(crops, entry):
    """Rebuild the _Window the detector works in, from a committed crop."""
    import cv2

    from live.deck_hand import GREY_STD_THRESHOLD, _Window
    from readers.hand import has_cost_badge

    tall = cv2.imread(str(crops / entry["file"]), cv2.IMREAD_COLOR)
    assert tall is not None, entry["file"]
    base, height = entry["base"], entry["height"]
    crop = tall[base:base + height]
    badge = has_cost_badge(crop)
    colour = float(np.mean(np.std(crop.astype(np.float32), axis=2)))
    return _Window(tall, base, height, badge, colour,
                   badge or colour < GREY_STD_THRESHOLD)


def test_the_pool_holds_exactly_the_eight_cards_of_one_deck(suite):
    """A pool is deck-specific: eight templates, no more and no fewer. Asserted
    against the card registry, not `_deck()` (derived from the pool, so
    circular): every key resolves to a distinct known card and the count is a
    full deck.
    """
    from clashroyalebuildabot.namespaces.cards import Cards
    from live.deck_hand import load_pool

    _name, _crops, pool = suite
    templates = load_pool(pool)
    assert len(templates) == 8, f"expected 8 templates, got {sorted(templates)}"

    known = {getattr(c, "name", None) for c in vars(Cards).values()}
    assert set(templates) <= known, f"not in the registry: {set(templates) - known}"
    assert "blank" not in templates, "blank is an absence, not a template"
    for name, template in templates.items():
        assert isinstance(template, np.ndarray) and template.size, name


def test_the_live_pool_is_the_deck_the_agent_was_TRAINED_on():
    """The default pool must cover `gym_wrapper.DEFAULT_DECK` exactly. A pool for
    a different deck still loads, reads eight cards and looks healthy, while
    mapping every real card onto its nearest template.
    """
    from live.deck_hand import DEFAULT_POOL, load_pool
    from live.mvp_loop import _training_deck_ids
    from live.unit_to_card import hand_card_id_for

    if not (DEFAULT_POOL / "icons.json").exists():
        pytest.skip("default pool absent")

    pool_ids = {hand_card_id_for(n) for n in load_pool(DEFAULT_POOL)}
    assert pool_ids == set(_training_deck_ids()), (
        f"the default pool covers {sorted(pool_ids)} but DEFAULT_DECK is "
        f"{sorted(_training_deck_ids())} -- rebuild it from footage of the "
        f"deck the agent plays")


def test_a_pool_that_does_not_cover_the_deck_is_REFUSED(suite):
    """A pool built from another deck would map an unseen card onto its nearest
    template, confidently and every frame. Refusing at construction is the
    guard.
    """
    from clashroyalebuildabot.namespaces.cards import Cards
    from live.deck_hand import DeckHandDetector, DeckPoolMissing

    _name, _crops, pool = suite
    wrong = _deck(pool)[:-1] + [Cards.MEGA_KNIGHT]
    with pytest.raises(DeckPoolMissing, match="no template"):
        DeckHandDetector(wrong, pool)


@pytest.mark.parametrize("kind", ["clean", "dimmed", "lifted"])
def test_every_card_state_is_identified(suite, kind):
    name, crops, pool = suite
    det = _detector(pool)
    entries = [m for m in _manifest(crops) if m["kind"] == kind]
    assert entries, f"no {kind} crops in the {name} fixture"
    for entry in entries:
        read = det._read_window(_window(crops, entry), gate_presence=True)
        assert read.card.name == entry["truth"], (
            f"{entry['file']}: read {read.card.name} at score {read.score:.3f}")


def test_an_empty_slot_reads_blank_and_is_not_guessed(suite):
    name, crops, pool = suite
    det = _detector(pool)
    entries = [m for m in _manifest(crops) if m["kind"] == "empty"]
    assert entries, f"no empty crops in the {name} fixture"
    for entry in entries:
        read = det._read_window(_window(crops, entry), gate_presence=True)
        assert read.card.name == "blank", (
            f"{entry['file']}: invented {read.card.name}")


def test_an_unaffordable_card_is_read_but_reported_NOT_ready(suite):
    """Identity and affordability are different questions: the cost badge is
    greyscaled with the card, so gating identity on it made every unaffordable
    card vanish.
    """
    name, crops, pool = suite
    det = _detector(pool)
    manifest = _manifest(crops)

    dimmed = [m for m in manifest if m["kind"] == "dimmed"]
    assert dimmed, f"no dimmed crops in the {name} fixture"
    for entry in dimmed:
        window = _window(crops, entry)
        assert not window.badge, (
            f"{entry['file']}: a dimmed card should have NO magenta badge -- "
            "if this fails the premise of the presence rule has changed")
        assert window.present, f"{entry['file']}: dimmed card read as absent"
        assert det._detect_if_ready([window]) == [], (
            f"{entry['file']}: greyscale slot reported affordable")

    for entry in [m for m in manifest if m["kind"] == "clean"]:
        assert det._detect_if_ready([_window(crops, entry)]) == [0], (
            f"{entry['file']}: coloured slot reported unaffordable")


def test_the_lift_search_is_what_recovers_a_selected_card(suite):
    """Pins the mechanism, not just the outcome: without the vertical search a
    selected card scores below MIN_SCORE and reads `blank`, and an
    identity-only assertion would still pass if the search were deleted and
    MIN_SCORE lowered.
    """
    from live.deck_hand import MIN_SCORE

    name, crops, pool = suite
    det = _detector(pool)
    for entry in [m for m in _manifest(crops) if m["kind"] == "lifted"]:
        window = _window(crops, entry)
        crop = window.tall[window.base:window.base + window.height]

        fixed = det._read(crop, gate_presence=True)
        assert fixed.score < MIN_SCORE, (
            f"{entry['file']}: scores {fixed.score:.3f} unshifted, so it is no "
            "longer a lifted-card fixture and this test proves nothing")

        searched = det._read_window(window, gate_presence=True)
        assert searched.card.name == entry["truth"]
        assert searched.score > 0.8


def test_blank_is_never_produced_by_a_low_score_alone(suite):
    """Presence is decided before identity. An empty slot can correlate well with
    its best template (on giant the empty band tops out at 0.571 and the dimmed
    band reaches down to 0.513), so no score threshold separates them; presence
    is badge-or-greyscale, and score only ranks candidates. The overlap is
    asserted over the fixture as a whole, since individual empty crops can
    score low.
    """
    from live.deck_hand import MIN_SCORE

    _name, crops, pool = suite
    det = _detector(pool)
    empties = [m for m in _manifest(crops) if m["kind"] == "empty"]
    assert empties

    peak = 0.0
    for entry in empties:
        window = _window(crops, entry)
        scores, _dy = det._best_over_lift(window.tall, window.base, window.height)
        peak = max(peak, float(scores.max()))
        # The load-bearing half: whatever it scores, an empty slot is absent.
        assert not window.present, f"{entry['file']}: empty slot read as present"

    assert peak > MIN_SCORE, (
        f"no empty crop in this fixture scores above the {MIN_SCORE} floor "
        f"(best {peak:.3f}) -- the overlap this rule exists for is not "
        f"represented here, so the fixture no longer tests it")
