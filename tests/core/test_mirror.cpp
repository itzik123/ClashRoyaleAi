#include <catch_amalgamated.hpp>
#include "GameManager.h"
#include "CardRegistry.h"

TEST_CASE("Mirror fails to play when nothing has been played yet", "[game_manager][mirror]") {
    GameManager game({ 164, 1, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    // Mirror is now GUARANTEED excluded from the real opening hand (see
    // PlayerState::initializeDeck's rng overload) -- force it in directly so
    // this fails for the reason the test actually cares about (nothing
    // played yet), not just "card not in hand".
    game.playerAI.hand[0] = 164;
    game.playerAI.elixir = 10.0f;

    REQUIRE_FALSE(game.playCard(0, 164, 9.0f, 10.0f));
}

TEST_CASE("Mirror replays the last card played, at +1 elixir cost", "[game_manager][mirror]") {
    GameManager game({ 0, 164, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 }); // Knight (cost 3), then Mirror
    // Fully pin hand/deckQueue to the deterministic deck[0..3]/deck[4..7]
    // partition -- opening hand is now randomized (see PlayerState::
    // initializeDeck's rng overload), and Mirror is additionally guaranteed
    // excluded from it entirely. A partial single-slot force (hand[1]=164)
    // isn't enough here: Mirror could still be left sitting in deckQueue
    // from the random shuffle too, and Knight's OWN play draws from
    // deckQueue.front() -- if that residual Mirror got drawn into hand[0]
    // it would collide with the forced hand[1] copy and get found first by
    // GameManager::playCard's hand scan.
    game.playerAI.hand = { 0, 164, 2, 3 };
    game.playerAI.deckQueue = { 4, 5, 6, 7 };
    game.playerAI.elixir = 10.0f;

    REQUIRE(game.playCard(0, 0, 9.0f, 10.0f)); // Knight
    float elixirAfterKnight = game.getElixirAI();
    REQUIRE(elixirAfterKnight == Catch::Approx(7.0f)); // 10 - 3

    REQUIRE(game.playCard(0, 164, 9.0f, 10.0f)); // Mirror
    REQUIRE(game.getElixirAI() == Catch::Approx(3.0f)); // 7 - (3 + 1)

    game.step(); // commit both spawns
    int knightCount = 0;
    for (const auto& e : game.getBoard().getEntities()) {
        if (e->name == "Knight") knightCount++;
    }
    REQUIRE(knightCount == 2); // the original, plus Mirror's copy
}

TEST_CASE("Mirror fails when unaffordable (mirrored cost + 1), without deducting anything",
        "[game_manager][mirror]") {
    GameManager game({ 0, 164, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    // Fully pin hand/deckQueue -- see the previous test's own comment on why
    // a partial single-slot force isn't enough.
    game.playerAI.hand = { 0, 164, 2, 3 };
    game.playerAI.deckQueue = { 4, 5, 6, 7 };
    game.playerAI.elixir = 10.0f;
    game.playCard(0, 0, 9.0f, 10.0f); // Knight, cost 3
    game.playerAI.elixir = 3.5f; // enough for Knight (3) again, not Mirror's 4

    REQUIRE_FALSE(game.playCard(0, 164, 9.0f, 10.0f));
    REQUIRE(game.getElixirAI() == Catch::Approx(3.5f)); // untouched
}

TEST_CASE("A second Mirror replays what was played before the FIRST Mirror, not the Mirror itself",
        "[game_manager][mirror]") {
    // Deck: Knight (0), Mirror (164) in the first two slots; the queue
    // cycles Mirror back into hand slot 1 by the time we need to play it
    // again -- simpler to just re-seed hand[1] directly, same technique
    // already used for the Evolution cycling tests.
    GameManager game({ 0, 164, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    // Fully pin hand/deckQueue -- see the first Mirror test's own comment on
    // why a partial single-slot force isn't enough.
    game.playerAI.hand = { 0, 164, 2, 3 };
    game.playerAI.deckQueue = { 4, 5, 6, 7 };
    game.playerAI.elixir = 100.0f;

    REQUIRE(game.playCard(0, 0, 9.0f, 10.0f));   // Knight
    REQUIRE(game.playCard(0, 164, 9.0f, 10.0f)); // 1st Mirror -> mirrors Knight

    game.playerAI.hand[1] = 164; // simulate Mirror having cycled back to this slot
    game.playerAI.handCooldownTicks[1] = 0; // ...immediately playable, not still on the 1st Mirror's cycle-in delay
    REQUIRE(game.playCard(0, 164, 9.0f, 10.0f)); // 2nd Mirror -> should ALSO mirror Knight, not the 1st Mirror

    game.step();
    int knightCount = 0, mirrorEntityCount = 0;
    for (const auto& e : game.getBoard().getEntities()) {
        if (e->name == "Knight") knightCount++;
        if (e->name == "Mirror") mirrorEntityCount++;
    }
    REQUIRE(knightCount == 3);       // original + 2 mirrored copies
    REQUIRE(mirrorEntityCount == 0); // Mirror itself never spawns an entity of its own
}

TEST_CASE("Mirror respects the mirrored card's own placement rules (a spell may target the enemy half)",
        "[game_manager][mirror]") {
    GameManager game({ 7, 164, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 }); // Fireball (id 7, a spell), then Mirror
    // Fully pin hand/deckQueue -- see the first Mirror test's own comment on
    // why a partial single-slot force isn't enough.
    game.playerAI.hand = { 7, 164, 2, 3 };
    game.playerAI.deckQueue = { 4, 5, 6, 7 };
    game.playerAI.elixir = 100.0f;

    REQUIRE(game.playCard(0, 7, 9.0f, 10.0f)); // Fireball, own half

    // Mirroring a spell onto the ENEMY half must succeed -- spells aren't
    // restricted to the caster's own side the way troops are.
    REQUIRE(game.playCard(0, 164, 9.0f, 25.0f));
}
