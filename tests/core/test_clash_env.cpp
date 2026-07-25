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
// The vector grew twice since, though: NUM_CARD_IDS 120->175 (Evolutions/
// Mirror/Spirit Empress needed ids up to 165), then 175->185 (Heroes need
// ids up to 175 -- see ClashEnv.h). Both are deliberate, lockstep changes
// (python_ai/model.py now pulls NUM_CARD_IDS live from the compiled binding,
// no manual bump needed there anymore), not a regression --
// 18*34*9 + 1 + 4 + 4*185 = 6253.

TEST_CASE("ClashEnv::observationSize matches NUM_CARD_IDS=185", "[clash_env]") {
    std::vector<int> deck = { 0, 1, 2, 3, 4, 5, 6, 7 };
    ClashEnv env(deck, deck, 100);

    REQUIRE(env.observationSize() == 6253);

    auto obs = env.reset();
    REQUIRE(obs.size() == 6253);
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
