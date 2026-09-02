#include <catch_amalgamated.hpp>
#include "ClashEnv.h"
#include <vector>
#include <algorithm>
#include <random>

// Champion support (Mighty Miner, id 115) deliberately does NOT grow the
// flat observation vector -- see ClashEnv.h's own comment on
// isChampionAbilityReady for why (python_ai/model.py's scalar_size formula
// is fixed, independent of observation_size(), and would break on the very
// next forward pass if this vector grew). These tests lock that in.
//
// The vector grew several times since, though, all deliberate, lockstep
// changes (python_ai/model.py pulls these constants live from the compiled
// binding, no manual bump needed there), not a regression:
//   - NUM_CARD_IDS 120->175 (Evolutions/Mirror/Spirit Empress needed ids up
//     to 165), then 175->185 (Heroes need ids up to 175 -- see ClashEnv.h).
//   - NUM_CHANNELS 9->21: added 12 per-cell ATTRIBUTE channels (unit count,
//     flying, anti-air, DPS, range, speed -- ally+enemy each) so air/ground
//     counterplay and unit identity beyond raw HP fraction are actually
//     visible in the observation (see NUM_CHANNELS's own comment).
//   - NUM_EXTRA_SCALARS 0->9: appended scalars (time, elixir spent, tower HP),
//     then 9->10 on 2026-09-02 for the elixir-phase multiplier (1x/2x/3x,
//     normalised by GameManager::MAX_ELIXIR_MULTIPLIER). Note this one was
//     appended to the EXTRA SCALARS rather than to the end of the vector, so
//     unlike the cycle blocks it MOVED CYCLE_START -- by exactly one.
//   - CYCLE_BLOCK_SIZE 0->370 (2026-08-27, UPSTREAM_REQUESTS item 24): two
//     NUM_CARD_IDS-wide blocks carrying what the OPPONENT has played --
//     seen[] and an exponentially-decaying recency[]. The card identities are
//     observable to a human watching the screen (the encoder already gives the
//     agent the opponent's elixir SPEND on exactly that reasoning) and were
//     being reduced to a scalar sum of costs, which is the one summary that
//     destroys cycle information.
// 18*34*21 + 1 + 4 + 4*185 + 10 + 2*185 = 13977.
//
// Asserted BOTH ways on purpose, because the two catch different faults:
//   * the FORMULA catches observationSize() disagreeing with the constants it
//     is supposed to be built from -- an internal inconsistency that would
//     make model.py split the flat vector at the wrong offset and silently
//     misread every scalar, with no exception anywhere.
//   * the LITERAL catches the size changing at all. That is a real tripwire:
//     every checkpoint ever trained is invalidated by an observation resize,
//     so it should never happen by accident. When it IS intended, update the
//     literal deliberately -- that edit is the acknowledgement.

TEST_CASE("ClashEnv::observationSize matches its declared layout", "[clash_env]") {
    std::vector<int> deck = { 0, 1, 2, 3, 4, 5, 6, 7 };
    ClashEnv env(deck, deck, 100);

    const int expected =
        ClashEnv::BOARD_WIDTH * ClashEnv::BOARD_HEIGHT * ClashEnv::NUM_CHANNELS
        + 1                                              // own elixir
        + ClashEnv::HAND_SIZE                            // hand costs
        + ClashEnv::HAND_SIZE * ClashEnv::NUM_CARD_IDS   // hand identity one-hots
        + ClashEnv::NUM_EXTRA_SCALARS                    // time, elixir spent, tower HP
        + ClashEnv::CYCLE_BLOCK_SIZE;                    // opponent seen[] + recency[]

    REQUIRE(env.observationSize() == expected);
    REQUIRE(env.observationSize() == 13977);

    // CYCLE_START is 13607 since 2026-09-02, one higher than the 13606 it sat
    // at from the cycle blocks' own append. That is the elixir-phase scalar
    // being added to the EXTRA SCALARS block, which sits in front of the cycle
    // blocks and therefore shifts them.
    //
    // The literal is worth keeping even though it now moves: it is what
    // distinguishes "one scalar was appended to the extra block" (this, +1)
    // from "something was inserted in the middle of the spatial channels or
    // the one-hots" (a much larger jump). The relationship asserted below is
    // the invariant; this number is the tripwire.
    REQUIRE(ClashEnv::CYCLE_START == 13607);
    REQUIRE(ClashEnv::EXTRA_SCALARS_START + ClashEnv::NUM_EXTRA_SCALARS
            == ClashEnv::CYCLE_START);

    auto obs = env.reset();
    REQUIRE(obs.size() == static_cast<size_t>(expected));

    // Guards the split point model.py actually uses. If the spatial block and
    // the scalar tail ever disagree with observationSize(), every scalar the
    // network reads shifts by the difference -- with no exception anywhere.
    const int spatial = ClashEnv::BOARD_WIDTH * ClashEnv::BOARD_HEIGHT * ClashEnv::NUM_CHANNELS;
    REQUIRE(env.observationSize() - spatial
            == 1 + ClashEnv::HAND_SIZE + ClashEnv::HAND_SIZE * ClashEnv::NUM_CARD_IDS
               + ClashEnv::NUM_EXTRA_SCALARS + ClashEnv::CYCLE_BLOCK_SIZE);
    // ...and the spatial block itself is exactly where the scalar section
    // starts, which is the fact EXTRA_SCALARS_START is built on.
    REQUIRE(ClashEnv::EXTRA_SCALARS_START - spatial
            == 1 + ClashEnv::HAND_SIZE + ClashEnv::HAND_SIZE * ClashEnv::NUM_CARD_IDS);
}

TEST_CASE("isChampionAbilityReady/activateChampionAbility return false with nothing deployed", "[clash_env][champion]") {
    std::vector<int> deck = { 0, 1, 2, 3, 4, 5, 6, 7 }; // no Champion in this deck
    ClashEnv env(deck, deck, 100);
    env.reset();

    REQUIRE_FALSE(env.isChampionAbilityReady(0));
    REQUIRE_FALSE(env.activateChampionAbility(0));
}

TEST_CASE("step()'s activateAbility param defaults to false and only fires when explicitly true", "[clash_env][champion]") {
    // Mighty Miner in deck slot 1 (the Heroic slot -- Champions are only
    // legal in slot 1 or 2, see CardRegistry::validateDeckSlots). The
    // opening hand is now randomized (see PlayerState::initializeDeck's rng
    // overload), so deck order no longer guarantees hand order -- force him
    // into hand index 1 directly via the test-only debugGame() accessor
    // (ClashEnv wraps GameManager privately, so this is the only way in).
    std::vector<int> deck = { 1, 115, 2, 3, 4, 5, 6, 7 };
    ClashEnv env(deck, deck, 100);
    env.reset();
    env.debugGame().playerAI.hand[1] = 115;

    env.step(1, 9.0f, 10.0f, 1); // deploy Mighty Miner; activateAbility defaults to false
    REQUIRE(env.isChampionAbilityReady(0)); // deployed, off cooldown, affordable

    env.step(-1, 0.0f, 0.0f, 1); // no card, no ability -- default stays false
    REQUIRE(env.isChampionAbilityReady(0)); // still ready: nothing consumed it

    env.step(-1, 0.0f, 0.0f, 1, true); // explicitly activate slot 1's ability now
    REQUIRE_FALSE(env.isChampionAbilityReady(0)); // now on cooldown
}

TEST_CASE("sampleRandomDeck always produces a deck that passes validateDeckSlots", "[clash_env][random_deck]") {
    std::mt19937 rng(2024);
    for (int trial = 0; trial < 300; ++trial) {
        std::vector<int> deck = sampleRandomDeck(rng);
        REQUIRE(deck.size() == 8);
        REQUIRE(validateDeckSlots(deck).empty());

        // A real deck can't repeat a card -- confirm sampleRandomDeck never does either.
        std::vector<int> sorted = deck;
        std::sort(sorted.begin(), sorted.end());
        REQUIRE(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end());
    }
}


TEST_CASE("each hand slot's card-identity block is a true one-hot", "[clash_env][observation]") {
    // The one-hot tail is built by resize-and-set rather than by 185
    // push_backs per slot, and the two are only equivalent while the block is
    // zeroed first and exactly one index is written. An out-of-range card id
    // must leave the block empty rather than write past it.
    ClashEnv env({ 15, 6, 25, 40, 24, 72, 33, 7 }, { 15, 6, 25, 40, 24, 72, 33, 7 });
    env.reset();

    const std::vector<float> obs = env.getObservationForTeam(0);
    const std::vector<int> hand = env.getHand();

    constexpr int BOARD_W = 18, BOARD_H = 34, CHANNELS = 21, HAND = 4, CARD_IDS = 185;
    const size_t oneHotBase = static_cast<size_t>(BOARD_W) * BOARD_H * CHANNELS + 1 + HAND;

    REQUIRE(hand.size() == static_cast<size_t>(HAND));
    for (int slot = 0; slot < HAND; ++slot) {
        const size_t base = oneHotBase + static_cast<size_t>(slot) * CARD_IDS;
        int hotCount = 0, hotIndex = -1;
        for (int k = 0; k < CARD_IDS; ++k) {
            if (obs[base + k] != 0.0f) { hotCount++; hotIndex = k; }
        }
        INFO("hand slot " << slot << " holds card " << hand[slot]);
        REQUIRE(hotCount == 1);
        REQUIRE(hotIndex == hand[slot]);
        REQUIRE(obs[base + hotIndex] == Catch::Approx(1.0f));
    }
}
