#include <catch_amalgamated.hpp>
#include "CardRegistry.h"
#include "PlayerState.h"
#include "GameManager.h"
#include "ClashEnv.h"
#include <vector>
#include <algorithm>

// Framework pilot: Wall Breakers Evolution (id 123, base id 83). Proves the
// evolution-slot machinery end to end before the remaining 40 evolutions
// are batched in -- see CardRegistry.h's "=== Evolutions ===" section.

TEST_CASE("PlayerState::playCard resolves an Evolution slot: 2 un-evolved cycles, "
          "then 1 evolved, repeating for the rest of the match", "[player_state][evolution]") {
    PlayerState player;
    player.initializeDeck({ 123, 1, 2, 3, 4, 5, 6, 7 }); // Wall Breakers Evolution in hand slot 0
    player.elixir = 100.0f; // never the limiting factor here

    // Real cycling (playing the other 7 slots first) isn't needed to unit
    // test PlayerState's own resolution logic -- re-seeding hand[0] to 123
    // between calls simulates "this slot cycled back to the same card"
    // without needing 8 real plays per lap.
    auto r1 = player.playCard(0);
    REQUIRE(r1.cardId == 123);
    REQUIRE_FALSE(r1.useEvolvedForm);

    player.hand[0] = 123;
    auto r2 = player.playCard(0);
    REQUIRE_FALSE(r2.useEvolvedForm); // 2nd un-evolved cycle

    player.hand[0] = 123;
    auto r3 = player.playCard(0);
    REQUIRE(r3.useEvolvedForm); // 3rd play: evolved

    player.hand[0] = 123;
    auto r4 = player.playCard(0);
    REQUIRE_FALSE(r4.useEvolvedForm); // NOT a one-time charge -- back to un-evolved

    player.hand[0] = 123;
    auto r5 = player.playCard(0);
    REQUIRE_FALSE(r5.useEvolvedForm);

    player.hand[0] = 123;
    auto r6 = player.playCard(0);
    REQUIRE(r6.useEvolvedForm); // evolves again, confirming the repeat
}

TEST_CASE("A deck without an Evolution slot never touches evolutionState", "[player_state][evolution]") {
    PlayerState player;
    player.initializeDeck({ 83, 1, 2, 3, 4, 5, 6, 7 }); // plain Wall Breakers, not the Evolution
    REQUIRE(player.evolutionState.empty());

    auto r = player.playCard(0);
    REQUIRE(r.cardId == 83);
    REQUIRE_FALSE(r.useEvolvedForm);
}

TEST_CASE("initializeDeck resets Evolution progress -- no cross-match state leak", "[player_state][evolution]") {
    PlayerState player;
    player.initializeDeck({ 123, 1, 2, 3, 4, 5, 6, 7 });
    player.elixir = 100.0f;

    player.playCard(0);
    player.hand[0] = 123;
    player.playCard(0);
    player.hand[0] = 123;
    auto evolved = player.playCard(0);
    REQUIRE(evolved.useEvolvedForm); // fully cycled once

    player.initializeDeck({ 123, 1, 2, 3, 4, 5, 6, 7 }); // simulates GameManager::reset()
    auto fresh = player.playCard(0);
    REQUIRE_FALSE(fresh.useEvolvedForm); // back to un-evolved, not still mid-cycle
}

TEST_CASE("GameManager::playCard spawns the evolved entity only on an evolved play", "[game_manager][evolution]") {
    GameManager game({ 123, 1, 2, 3, 4, 5, 6, 7 }, { 0, 1, 2, 3, 4, 5, 6, 7 });
    game.playerAI.elixir = 100.0f;

    game.playCard(0, 123, 9.0f, 10.0f); // 1st play: un-evolved
    game.step();
    game.playerAI.hand[0] = 123;
    game.playCard(0, 123, 9.0f, 10.0f); // 2nd play: un-evolved
    game.step();
    game.playerAI.hand[0] = 123;
    game.playCard(0, 123, 9.0f, 10.0f); // 3rd play: evolved
    game.step();

    // The evolved spawn carries a deathEffect (Runners); the un-evolved
    // spawn does not -- kill everything and count how many total entities
    // this deploy round produced (2 base Wall Breakers each play, plus 2
    // Runners once, only from the evolved play's death).
    int wallBreakerCount = 0;
    for (const auto& e : game.getBoard().getEntities()) {
        if (e->name == "Wall Breakers") { wallBreakerCount++; e->takeDamage(e->hp); }
    }
    REQUIRE(wallBreakerCount == 6); // 3 plays x 2 Wall Breakers each

    game.getBoard().cleanDeadEntities();
    game.getBoard().commitPendingEntities();

    int runnerCount = 0;
    for (const auto& e : game.getBoard().getEntities()) {
        if (e->name == "Runner") runnerCount++;
    }
    REQUIRE(runnerCount == 2); // only the evolved play's pair died-and-spawned Runners
}

TEST_CASE("getAllCardIds excludes Evolution slots -- the RANDOM_DECK_POOL landmine guard",
        "[clash_env][evolution]") {
    auto ids = getAllCardIds();
    REQUIRE(std::find(ids.begin(), ids.end(), 123) == ids.end());
    REQUIRE(std::find(ids.begin(), ids.end(), 83) != ids.end()); // the base card is still a normal pick
}
