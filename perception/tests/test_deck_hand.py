"""The deck-restricted hand classifier, pinned against committed crops.

RUN AGAINST BOTH DECKS. A pool is deck-specific, so one pool passing proves
the templates for that deck are good and says nothing about the mechanism. The
two suites here are different decks, different recordings, and different
emulator window positions:

  giant  the July batch's deck, from assets/live/match_practice_01
  hog26  gym_wrapper.DEFAULT_DECK -- the 2.6 Hog Cycle, which the live loop
         actually plays -- from assets/recordings/2026-09-02 19-33-22.mp4

WHY THE FIXTURE IS CROPS AND NOT FRAMES
---------------------------------------
`assets/live/` is gitignored (~1.5 GB of PNGs), so a test bound to it SKIPS on
a fresh clone, which is the silent hole `perception/.gitignore`'s own
tower-digit exception was written to avoid. Two dozen 61x87 crops are ~350 KB
and make these assertions run everywhere.

The four `kind`s are the four things a hand slot can be doing, and three of
them were each, at some point, read wrong:

  clean   an affordable card, coloured, cost badge showing
  dimmed  UNAFFORDABLE: the game renders the whole card -- badge included --
          in true greyscale, so the magenta badge test fails on a slot that
          very much holds a card. 15.1% of in-match slots on the giant
          recording and 44.2% on hog26.
  lifted  SELECTED: the art translates up ~11 px with a highlight bar below,
          and a rigid template's correlation collapses from ~0.99 to ~0.15.
          Measured at dy = -10/-11 on BOTH recordings, so it is a property of
          the game's UI and not of one capture.
  empty   the blue crown card-back between a play and the next card sliding
          in. The only state that should read `blank`.
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
    """The deck a pool was built for, derived FROM the pool.

    NOT imported from `tools.audit_hand_id`, though a deck list lives there.
    `perception/tools/` shares its name with the repo root's `tools/`, and
    conftest puts the repo root on sys.path to reach the compiled engine -- so
    under pytest `tools.audit_hand_id` resolves into the wrong package and
    raises ModuleNotFoundError, while the identical import works when a script
    is run from perception/.
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
    """A pool is deck-specific: eight templates, no more and no fewer.

    Asserted against the CARD REGISTRY rather than against `_deck()`, which is
    itself derived from the pool -- comparing those two would be circular and
    could not fail. The content is that every key resolves to a distinct known
    card and the count is a full deck: a pool with a duplicate or a stray entry
    is one that will mismap at runtime.
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
    """The default pool must cover `gym_wrapper.DEFAULT_DECK`, exactly.

    This is the check the whole feature turns on. A pool that covers a
    DIFFERENT deck still loads, still reads eight cards, and still looks
    healthy -- it just maps every real card onto whichever of its eight
    templates it least mismatches. `mvp_loop` derives its deck from
    DEFAULT_DECK precisely so the live deck cannot drift from training, and
    this asserts the templates followed it.
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
    """The failure that must never be silent.

    A pool built from another deck has a template for every card it saw and
    none for the cards it did not, so an unseen card would be mapped onto
    whichever of the eight it least mismatches -- confidently, every frame,
    with nothing raising. Refusing at construction is the whole guard.
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
    """Identity and affordability are different questions.

    The bug this pins is reading them off one signal: the magenta cost badge is
    greyscaled along with the card, so gating identity on it made every
    unaffordable card vanish -- one in seven in-match slots on the giant
    recording, and nearly one in two on hog26.
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
    """Pins the MECHANISM, not just the outcome.

    Without the vertical search a selected card scores below MIN_SCORE and
    reads `blank`. Asserting only the identity would keep passing if the search
    were deleted and MIN_SCORE quietly lowered instead -- which would trade
    this bug for a worse one.
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
    """Presence is decided before identity, and the two must not be confused.

    A genuinely empty slot still correlates well with its best template -- on
    the giant recording the empty band tops out at 0.571 while the dimmed band
    reaches down to 0.513, so the two OVERLAP and no score threshold can
    separate them. That is why presence is decided on badge-or-greyscale first
    and score only ranks the candidates afterwards.

    The overlap is asserted over the fixture as a WHOLE, not per crop. An
    earlier version required every empty crop to clear MIN_SCORE, which was
    the giant recording's numbers written down as a universal: hog26 has an
    empty slot at 0.431, below the floor. One such crop does not weaken the
    argument -- the rule is unsafe as long as SOME empty slot scores like a
    card -- but it did make the test fail on a correct classifier.
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
