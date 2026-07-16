#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "GameManager.h"
#include "BuildingTargeter.h"
#include <vector>

// ---------------- construction / reset ----------------

TEST_CASE("GameManager construction sets up exactly the 6 expected towers", "[game_manager][reset]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    const auto& entities = game.getBoard().getEntities();
    REQUIRE(entities.size() == 6);

    REQUIRE(entities[0]->symbol == 'R');
    REQUIRE(entities[0]->team == 0);
    REQUIRE(entities[0]->hp == 4008);
    REQUIRE(entities[0]->name == "King Tower");
    REQUIRE(entities[0]->position.x == Catch::Approx(9.0f));
    REQUIRE(entities[0]->position.y == Catch::Approx(2.0f));

    REQUIRE(entities[1]->symbol == 'R');
    REQUIRE(entities[1]->team == 1);
    REQUIRE(entities[1]->name == "King Tower");
    REQUIRE(entities[1]->position.y == Catch::Approx(30.0f));

    REQUIRE(entities[2]->symbol == 'P');
    REQUIRE(entities[2]->team == 0);
    REQUIRE(entities[2]->hp == 2534);
    REQUIRE(entities[2]->name == "Princess Tower");
    REQUIRE(entities[3]->symbol == 'P');
    REQUIRE(entities[3]->team == 0);

    REQUIRE(entities[4]->symbol == 'P');
    REQUIRE(entities[4]->team == 1);
    REQUIRE(entities[5]->symbol == 'P');
    REQUIRE(entities[5]->team == 1);
}

TEST_CASE("GameManager construction gives both players starting elixir and a 4-card hand", "[game_manager][reset]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 8,9,10,11,12,13,14,17 });
    REQUIRE(game.getElixirAI() == Catch::Approx(5.0f));
    REQUIRE(game.getElixirOpp() == Catch::Approx(5.0f));
    REQUIRE(game.getHand(0).size() == 4);
    REQUIRE(game.getHand(1).size() == 4);
    REQUIRE(game.getHand(1)[0] == 8);
}

// ---------------- isValidPlacement ----------------

TEST_CASE("isValidPlacement rejects out-of-bounds coordinates regardless of spell", "[game_manager][placement]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    REQUIRE_FALSE(game.isValidPlacement(0, -1.0f, 5.0f, false));
    REQUIRE_FALSE(game.isValidPlacement(0, 20.0f, 5.0f, false));
    REQUIRE_FALSE(game.isValidPlacement(0, 5.0f, -1.0f, true));
    REQUIRE_FALSE(game.isValidPlacement(0, 5.0f, 40.0f, true));
}

TEST_CASE("isValidPlacement restricts troop/building placement to the caller's own half", "[game_manager][placement]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });

    REQUIRE(game.isValidPlacement(0, 9.0f, 14.0f, false));
    REQUIRE_FALSE(game.isValidPlacement(0, 9.0f, 15.0f, false));

    REQUIRE(game.isValidPlacement(1, 9.0f, 18.0f, false));
    REQUIRE_FALSE(game.isValidPlacement(1, 9.0f, 17.0f, false));
}

TEST_CASE("isValidPlacement lets spells ignore the half restriction", "[game_manager][placement]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    REQUIRE(game.isValidPlacement(0, 9.0f, 25.0f, true)); // deep in the opponent's half
}

TEST_CASE("isValidPlacement rejects placement overlapping an existing building", "[game_manager][placement]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    // AI king tower sits at (9, 2) with a 2.0 collision radius -- requiredDist = 1.0 + 2.0 = 3.0
    REQUIRE_FALSE(game.isValidPlacement(0, 9.0f, 4.0f, false)); // dist 2.0 < 3.0
    REQUIRE(game.isValidPlacement(0, 9.0f, 9.0f, false));       // dist 7.0, clear
}

// ---------------- playCard ----------------

TEST_CASE("playCard: successful play deducts elixir, cycles the hand, and spawns the entity", "[game_manager][play_card]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 8,9,10,11,12,13,14,17 });
    REQUIRE(game.getHand(0) == std::vector<int>{0, 1, 2, 3});
    REQUIRE(game.getElixirAI() == Catch::Approx(5.0f));

    size_t countBefore = game.getBoard().getEntities().size();
    bool played = game.playCard(0, 0, 9.0f, 10.0f); // Knight, cost 3.0

    REQUIRE(played);
    REQUIRE(game.getElixirAI() == Catch::Approx(2.0f));
    REQUIRE(game.getHand(0)[0] == 4); // next queued card cycled in

    // spawnEntity only queues into Board's pendingEntities; it's not visible
    // via getEntities() until the next commit (normally done by step()).
    game.getBoard().commitPendingEntities();
    REQUIRE(game.getBoard().getEntities().size() == countBefore + 1);
}

TEST_CASE("playCard fails if the card id is not currently in hand", "[game_manager][play_card]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    bool played = game.playCard(0, 99, 9.0f, 10.0f);
    REQUIRE_FALSE(played);
    REQUIRE(game.getElixirAI() == Catch::Approx(5.0f));
}

TEST_CASE("playCard fails if elixir is insufficient, without spending any", "[game_manager][play_card]") {
    GameManager game({ 19,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 }); // Golem costs 8.0, starting elixir 5.0
    bool played = game.playCard(0, 19, 9.0f, 10.0f);
    REQUIRE_FALSE(played);
    REQUIRE(game.getElixirAI() == Catch::Approx(5.0f));
    REQUIRE(game.getHand(0)[0] == 19);
}

TEST_CASE("playCard fails for a placement in the opponent's half, without spending elixir", "[game_manager][play_card]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    bool played = game.playCard(0, 0, 9.0f, 20.0f); // team 0, deep in team 1's half
    REQUIRE_FALSE(played);
    REQUIRE(game.getElixirAI() == Catch::Approx(5.0f));
}

TEST_CASE("playCard fails once the game is over", "[game_manager][play_card]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    auto aiKing = game.getBoard().getEntities()[0];
    aiKing->takeDamage(aiKing->hp);
    game.step();

    bool played = game.playCard(0, 1, 9.0f, 10.0f);
    REQUIRE_FALSE(played);
}

// ---------------- step() ----------------

TEST_CASE("step() regenerates elixir at the fixed rate, capped at 10.0", "[game_manager][step]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    REQUIRE(game.getElixirAI() == Catch::Approx(5.0f));

    game.step();
    REQUIRE(game.getElixirAI() == Catch::Approx(5.035f));

    for (int i = 0; i < 1000; ++i) game.step();
    REQUIRE(game.getElixirAI() == Catch::Approx(10.0f));
    REQUIRE(game.getElixirOpp() == Catch::Approx(10.0f));
}

TEST_CASE("step() ends the game when a King tower dies, with the correct loser", "[game_manager][step]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    auto aiKing = game.getBoard().getEntities()[0];
    REQUIRE(aiKing->symbol == 'R');
    REQUIRE(aiKing->team == 0);

    aiKing->takeDamage(aiKing->hp);
    REQUIRE_FALSE(game.isGameOver());

    game.step();

    REQUIRE(game.isGameOver());
    REQUIRE(game.getLoserTeam() == 0);
}

TEST_CASE("step() is a no-op once the game is over", "[game_manager][step]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    auto aiKing = game.getBoard().getEntities()[0];
    aiKing->takeDamage(aiKing->hp);
    game.step();

    float elixirBefore = game.getElixirAI();
    game.step();
    REQUIRE(game.getElixirAI() == Catch::Approx(elixirBefore));
}

TEST_CASE("step() removes entities that died during the tick from the board", "[game_manager][step]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    Board& board = game.getBoard();
    auto decoy = std::make_shared<DummyEntity>(board.allocateId(), 0.0f, 0.0f, 100, 0);
    spawn(board, decoy);
    decoy->takeDamage(100);

    size_t countBefore = board.getEntities().size();
    game.step();

    REQUIRE(board.getEntities().size() == countBefore - 1);
}

// ---------------- fixed: step()'s post-collision clamp now respects riverIgnores ----------------

TEST_CASE("step()'s post-collision re-clamp respects riverIgnores (regression test)", "[game_manager][step]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    Board& board = game.getBoard();

    // Placed mid-river at a non-bridge column, exactly like a Hog Rider caught
    // mid-crossing. Used to get snapped back out by step()'s own separate
    // river-clamp pass even though its own clampPosition() call (inside
    // update()) correctly left it alone. Now both calls go through the same
    // Board::clampToBoard(), so the second call is a no-op for it too.
    auto hog = std::make_shared<BuildingTargeter>(board.allocateId(), 8.0f, 16.0f, 1408, 0, 0.8f, 1.0f, 264, 15, 'H');
    hog->setIgnoresRiver(true);
    board.addEntity(hog);
    board.commitPendingEntities();

    game.step();

    REQUIRE(hog->position.y > 15.0f);
    REQUIRE(hog->position.y < 17.0f);
}

// ---------------- fixed: PlayerState::playCard on an empty deckQueue ----------------

TEST_CASE("PlayerState::playCard does not spend elixir when the deck queue is empty", "[player_state]") {
    PlayerState player;
    player.initializeDeck({ 0, 1, 2, 3 }); // only 4 cards: deckQueue starts empty
    REQUIRE(player.elixir == Catch::Approx(5.0f));

    int result = player.playCard(0); // Knight, cost 3.0

    REQUIRE(result == -1);
    REQUIRE(player.elixir == Catch::Approx(5.0f)); // untouched
    REQUIRE(player.hand[0] == 0);                  // hand slot untouched
}
