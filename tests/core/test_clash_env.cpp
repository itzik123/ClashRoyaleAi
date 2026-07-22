#include <catch_amalgamated.hpp>
#include "ClashEnv.h"
#include <vector>

// Champion support (Mighty Miner, id 115) deliberately does NOT grow the
// flat observation vector -- see ClashEnv.h's own comment on
// isChampionAbilityReady for why (python_ai/model.py's scalar_size formula
// is fixed, independent of observation_size(), and would break on the very
// next forward pass if this vector grew). These tests lock that in.
//
// The vector DID grow once since, though: NUM_CARD_IDS 120->175 (Evolutions/
// Mirror/Spirit Empress needed ids up to 165 -- see ClashEnv.h). That's a
// deliberate, lockstep change (python_ai/model.py's num_card_ids default
// bumped the same way), not a regression -- 18*34*9 + 1 + 4 + 4*175 = 6213.

TEST_CASE("ClashEnv::observationSize matches NUM_CARD_IDS=175", "[clash_env]") {
    std::vector<int> deck = { 0, 1, 2, 3, 4, 5, 6, 7 };
    ClashEnv env(deck, deck, 100);

    REQUIRE(env.observationSize() == 6213);

    auto obs = env.reset();
    REQUIRE(obs.size() == 6213);
}

TEST_CASE("isChampionAbilityReady/activateChampionAbility return false with nothing deployed", "[clash_env][champion]") {
    std::vector<int> deck = { 0, 1, 2, 3, 4, 5, 6, 7 }; // no Champion in this deck
    ClashEnv env(deck, deck, 100);
    env.reset();

    REQUIRE_FALSE(env.isChampionAbilityReady(0));
    REQUIRE_FALSE(env.activateChampionAbility(0));
}

TEST_CASE("step()'s activateAbility param defaults to false and only fires when explicitly true", "[clash_env][champion]") {
    std::vector<int> deck = { 115, 1, 2, 3, 4, 5, 6, 7 }; // Mighty Miner in hand slot 0
    ClashEnv env(deck, deck, 100);
    env.reset();

    env.step(0, 9.0f, 10.0f, 1); // deploy Mighty Miner; activateAbility defaults to false
    REQUIRE(env.isChampionAbilityReady(0)); // deployed, off cooldown, affordable

    env.step(-1, 0.0f, 0.0f, 1); // no card, no ability -- default stays false
    REQUIRE(env.isChampionAbilityReady(0)); // still ready: nothing consumed it

    env.step(-1, 0.0f, 0.0f, 1, true); // explicitly activate now
    REQUIRE_FALSE(env.isChampionAbilityReady(0)); // now on cooldown
}
