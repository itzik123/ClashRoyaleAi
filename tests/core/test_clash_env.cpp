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
//   - NUM_EXTRA_SCALARS 0->9: appended scalars (time, elixir spent, tower HP).
// 18*34*21 + 1 + 4 + 4*185 + 9 = 13606.
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
        + ClashEnv::NUM_EXTRA_SCALARS;                   // time, elixir spent, tower HP

    REQUIRE(env.observationSize() == expected);
    REQUIRE(env.observationSize() == 13606);

    auto obs = env.reset();
    REQUIRE(obs.size() == static_cast<size_t>(expected));

    // Guards the split point model.py actually uses. If the spatial block and
    // the scalar tail ever disagree with observationSize(), every scalar the
    // network reads shifts by the difference -- with no exception anywhere.
    const int spatial = ClashEnv::BOARD_WIDTH * ClashEnv::BOARD_HEIGHT * ClashEnv::NUM_CHANNELS;
    REQUIRE(env.observationSize() - spatial
            == 1 + ClashEnv::HAND_SIZE + ClashEnv::HAND_SIZE * ClashEnv::NUM_CARD_IDS
               + ClashEnv::NUM_EXTRA_SCALARS);
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
