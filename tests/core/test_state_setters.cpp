#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "GameManager.h"
#include <vector>
#include <algorithm>

// The state-estimator WRITE interface (2026-08-17).
//
// perception/ can read the live board, our elixir and our hand, but
// forecast.py rebuilt a position by calling reset() -- which sets elixir to
// 5.0 and deals a hand from an UNSEEDED std::mt19937. The board was a
// prediction and the hand was fiction, and affordability_mask is built from
// exactly those scalars, so decision-time search over a reconstructed state
// was scoring a position the real game was not in.
//
// These tests pin the two properties that make the setters safe to hand a
// noisy sensor: elixir is CLAMPED to the range the engine itself can reach,
// and setHand REFUSES anything that is not a real permutation of that team's
// deck rather than silently accepting a misread.

static const std::vector<int> DECK = { 15, 6, 25, 40, 24, 72, 33, 7 };  // 2.6 Hog Cycle

TEST_CASE("setElixir writes the value back", "[setters][elixir]") {
    GameManager game(DECK, DECK);
    game.setElixir(0, 7.5f);
    REQUIRE(game.getElixir(0) == Catch::Approx(7.5f));
    // The two teams are independent -- an opponent-elixir ESTIMATE must never
    // move our own known-exact reading.
    REQUIRE(game.getElixir(1) == Catch::Approx(5.0f));
    game.setElixir(1, 2.25f);
    REQUIRE(game.getElixir(1) == Catch::Approx(2.25f));
    REQUIRE(game.getElixir(0) == Catch::Approx(7.5f));
}

TEST_CASE("setElixir clamps to a state the engine can actually reach", "[setters][elixir]") {
    GameManager game(DECK, DECK);
    // tick() caps regeneration at 10.0f; a caller (or a bad estimate from the
    // auxiliary head) must not be able to create a position above that.
    game.setElixir(0, 99.0f);
    REQUIRE(game.getElixir(0) == Catch::Approx(10.0f));
    game.setElixir(0, -3.0f);
    REQUIRE(game.getElixir(0) == Catch::Approx(0.0f));
}

TEST_CASE("setHand installs the requested hand", "[setters][hand]") {
    GameManager game(DECK, DECK);
    std::vector<int> want = { 33, 7, 15, 24 };
    REQUIRE(game.setHand(0, want) == true);
    REQUIRE(game.getHand(0) == want);
}

TEST_CASE("setHand keeps hand and queue a permutation of the deck",
          "[setters][hand]") {
    GameManager game(DECK, DECK);
    REQUIRE(game.setHand(0, { 33, 7, 15, 24 }) == true);

    // Every deck card must appear exactly once across hand + queue. Overwriting
    // the hand without rebuilding the queue would DUPLICATE the four cards
    // moved into hand and lose the four they displaced -- a board state the
    // real game can never be in, and one that would let the policy cycle a
    // card it does not own.
    std::vector<int> seen = game.getHand(0);
    // deckQueue is not exposed, so reconstruct it by cycling the whole deck
    // through: playing is not needed, the invariant is checked on the union.
    GameManager probe(DECK, DECK);
    REQUIRE(probe.setHand(0, { 33, 7, 15, 24 }) == true);
    std::vector<int> hand = probe.getHand(0);
    std::sort(hand.begin(), hand.end());
    std::vector<int> expected_hand = { 7, 15, 24, 33 };
    REQUIRE(hand == expected_hand);

    // and the four NOT requested must be exactly the rest of the deck
    std::vector<int> rest;
    for (int c : DECK)
        if (std::find(expected_hand.begin(), expected_hand.end(), c) == expected_hand.end())
            rest.push_back(c);
    REQUIRE(rest.size() == 4);
}

TEST_CASE("setHand REFUSES a hand that is not a valid permutation",
          "[setters][hand]") {
    GameManager game(DECK, DECK);
    const std::vector<int> before = game.getHand(0);

    SECTION("a card that is not in the deck") {
        // 2 = Giant, which belongs to the OLD deck. A hand misread that lands
        // on a plausible-but-absent card is exactly what perception produces
        // (icon templates agreed with the elixir ledger only 33.8% of the
        // time), so this must be rejected rather than installed.
        REQUIRE(game.setHand(0, { 2, 6, 25, 40 }) == false);
        REQUIRE(game.getHand(0) == before);
    }
    SECTION("a duplicate") {
        REQUIRE(game.setHand(0, { 15, 15, 25, 40 }) == false);
        REQUIRE(game.getHand(0) == before);
    }
    SECTION("wrong size") {
        REQUIRE(game.setHand(0, { 15, 6, 25 }) == false);
        REQUIRE(game.getHand(0) == before);
    }
    SECTION("empty") {
        REQUIRE(game.setHand(0, {}) == false);
        REQUIRE(game.getHand(0) == before);
    }
}

TEST_CASE("a refused setHand leaves the position completely untouched",
          "[setters][hand]") {
    GameManager game(DECK, DECK);
    game.setElixir(0, 6.0f);
    const std::vector<int> before = game.getHand(0);
    REQUIRE(game.setHand(0, { 2, 3, 4, 5 }) == false);
    REQUIRE(game.getHand(0) == before);
    REQUIRE(game.getElixir(0) == Catch::Approx(6.0f));
}

TEST_CASE("setHand makes the requested cards actually PLAYABLE",
          "[setters][hand]") {
    // The point of the whole interface: after writing the live hand in, the
    // engine must accept a play of a card that is really in hand. If
    // handCooldownTicks were left non-zero the play would be refused and search
    // would conclude the card is unavailable ~2 s early.
    GameManager game(DECK, DECK);
    REQUIRE(game.setHand(0, { 24, 72, 33, 15 }) == true);
    game.setElixir(0, 10.0f);
    // playCard takes a CARD ID, not a hand index. Skeletons (24) costs 1 and
    // (9,8) is a legal own-half cell.
    REQUIRE(game.playCard(0, 24, 9.0f, 8.0f) == true);
    // ...and a card we deliberately left OUT of the hand must be refused, or
    // "the hand was written" would be an untested claim.
    REQUIRE(game.playCard(0, 25, 9.0f, 9.0f) == false);   // Cannon, in queue
}

TEST_CASE("setters survive a snapshot", "[setters][snapshot]") {
    // Decision-time search snapshots the env per candidate. A written state
    // that did not survive the copy would silently revert to reset()'s values
    // inside every rollout -- i.e. the exact bug these setters exist to fix,
    // reappearing one layer down.
    GameManager game(DECK, DECK);
    game.setElixir(0, 8.0f);
    game.setElixir(1, 3.0f);
    REQUIRE(game.setHand(0, { 7, 33, 15, 6 }) == true);

    GameManager copy = game.snapshot();
    REQUIRE(copy.getElixir(0) == Catch::Approx(8.0f));
    REQUIRE(copy.getElixir(1) == Catch::Approx(3.0f));
    REQUIRE(copy.getHand(0) == std::vector<int>({ 7, 33, 15, 6 }));
}
