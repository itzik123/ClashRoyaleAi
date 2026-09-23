#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "ArenaLayout.h"
#include "GameManager.h"
#include "CardRegistry.h"
#include "CardStats.h"
#include "BuildingTargeter.h"
#include "Building.h"
#include <vector>
#include <algorithm>
#include <random>

// ---------------- construction / reset ----------------

TEST_CASE("GameManager construction sets up exactly the 6 expected towers", "[game_manager][reset]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    const auto& entities = game.getBoard().getEntities();
    REQUIRE(entities.size() == 6);

    // Coordinates come from ArenaLayout, not restated. The case pins the
    // roster: six towers, their hp and names, in spawn order.
    REQUIRE(entities[0]->symbol == 'R');
    REQUIRE(entities[0]->team == 0);
    REQUIRE(entities[0]->hp == 4008);
    REQUIRE(entities[0]->name == "King Tower");
    REQUIRE(entities[0]->position.x == Catch::Approx(ArenaLayout::CENTER_X));
    REQUIRE(entities[0]->position.y == Catch::Approx(ArenaLayout::kingY(0)));

    REQUIRE(entities[1]->symbol == 'R');
    REQUIRE(entities[1]->team == 1);
    REQUIRE(entities[1]->name == "King Tower");
    REQUIRE(entities[1]->position.x == Catch::Approx(ArenaLayout::CENTER_X));
    // Mirrors the ally King via (height-1) - y = 33 - 2.5 = 30.5, as the team-1
    // observation does.
    REQUIRE(entities[1]->position.y == Catch::Approx(ArenaLayout::kingY(1)));

    REQUIRE(entities[2]->symbol == 'P');
    REQUIRE(entities[2]->team == 0);
    REQUIRE(entities[2]->hp == 2534);
    REQUIRE(entities[2]->name == "Princess Tower");
    REQUIRE(entities[2]->position.x == Catch::Approx(ArenaLayout::LEFT_LANE_X));
    REQUIRE(entities[2]->position.y == Catch::Approx(ArenaLayout::princessY(0)));
    REQUIRE(entities[3]->symbol == 'P');
    REQUIRE(entities[3]->team == 0);
    REQUIRE(entities[3]->position.x == Catch::Approx(ArenaLayout::RIGHT_LANE_X));
    REQUIRE(entities[3]->position.y == Catch::Approx(ArenaLayout::princessY(0)));

    REQUIRE(entities[4]->symbol == 'P');
    REQUIRE(entities[4]->team == 1);
    REQUIRE(entities[4]->position.x == Catch::Approx(ArenaLayout::LEFT_LANE_X));
    REQUIRE(entities[4]->position.y == Catch::Approx(ArenaLayout::princessY(1)));
    REQUIRE(entities[5]->symbol == 'P');
    REQUIRE(entities[5]->team == 1);
    REQUIRE(entities[5]->position.x == Catch::Approx(ArenaLayout::RIGHT_LANE_X));
    REQUIRE(entities[5]->position.y == Catch::Approx(ArenaLayout::princessY(1)));
}

TEST_CASE("GameManager construction gives both players starting elixir and a 4-card hand", "[game_manager][reset]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 8,9,10,11,12,13,14,17 });
    REQUIRE(game.getElixirAI() == Catch::Approx(5.0f));
    REQUIRE(game.getElixirOpp() == Catch::Approx(5.0f));
    REQUIRE(game.getHand(0).size() == 4);
    REQUIRE(game.getHand(1).size() == 4);
    // The opening hand is a random 4 of 8, so check it is a subset of the
    // opponent's deck.
    std::vector<int> oppDeck = { 8,9,10,11,12,13,14,17 };
    for (int cardId : game.getHand(1)) {
        REQUIRE(std::find(oppDeck.begin(), oppDeck.end(), cardId) != oppDeck.end());
    }
}

TEST_CASE("GameManager construction throws on a deck that fails validateDeckSlots", "[game_manager][reset][deck_slots]") {
    // Champion (115) in slot 0: legal only in slot 1 or 2.
    REQUIRE_THROWS_AS(
        GameManager({ 115, 1, 2, 3, 4, 5, 6, 7 }, { 0, 1, 2, 3, 4, 5, 6, 7 }),
        std::invalid_argument);
    // The opponent deck is checked too, not just the AI's.
    REQUIRE_THROWS_AS(
        GameManager({ 0, 1, 2, 3, 4, 5, 6, 7 }, { 115, 1, 2, 3, 4, 5, 6, 7 }),
        std::invalid_argument);
}

TEST_CASE("GameManager::setOpponentDeck throws on a deck that fails validateDeckSlots", "[game_manager][deck_slots]") {
    GameManager game({ 0, 1, 2, 3, 4, 5, 6, 7 }, { 0, 1, 2, 3, 4, 5, 6, 7 });
    REQUIRE_THROWS_AS(
        game.setOpponentDeck({ 115, 1, 2, 3, 4, 5, 6, 7 }),
        std::invalid_argument);
}

TEST_CASE("GameManager construction and setOpponentDeck accept a legal deck", "[game_manager][deck_slots]") {
    REQUIRE_NOTHROW(GameManager({ 1, 115, 2, 3, 4, 5, 6, 7 }, { 0, 1, 2, 3, 4, 5, 6, 7 }));
    GameManager game({ 0, 1, 2, 3, 4, 5, 6, 7 }, { 0, 1, 2, 3, 4, 5, 6, 7 });
    REQUIRE_NOTHROW(game.setOpponentDeck({ 1, 115, 118, 3, 4, 5, 6, 7 }));
}

// ---------------- isValidPlacement ----------------

TEST_CASE("isValidPlacement rejects out-of-bounds coordinates regardless of spell", "[game_manager][placement]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    // placedRadius is irrelevant: the bounds check fails first.
    REQUIRE_FALSE(game.isValidPlacement(0, -1.0f, 5.0f, false, Entity::IMPLICIT_TROOP_RADIUS));
    REQUIRE_FALSE(game.isValidPlacement(0, 20.0f, 5.0f, false, Entity::IMPLICIT_TROOP_RADIUS));
    REQUIRE_FALSE(game.isValidPlacement(0, 5.0f, -1.0f, true, Entity::IMPLICIT_TROOP_RADIUS));
    REQUIRE_FALSE(game.isValidPlacement(0, 5.0f, 40.0f, true, Entity::IMPLICIT_TROOP_RADIUS));
}

TEST_CASE("isValidPlacement rejects the back-row dead-zone corners but allows its center gap", "[game_manager][placement]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    // Spells skip the own-half rule but not the board's geometry; isSpell=true
    // isolates the dead-zone rule.
    REQUIRE_FALSE(game.isValidPlacement(0, 0.0f, 0.0f, true, 0.0f));   // bottom-row corner
    REQUIRE(game.isValidPlacement(0, 8.5f, 0.0f, true, 0.0f));         // bottom-row center gap
    REQUIRE_FALSE(game.isValidPlacement(1, 17.0f, 33.0f, true, 0.0f)); // top-row corner
    REQUIRE(game.isValidPlacement(1, 8.5f, 33.0f, true, 0.0f));        // top-row center gap
}

TEST_CASE("isValidPlacement restricts troop/building placement to the caller's own half", "[game_manager][placement]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });

    // The river is centred on 16.5: team 0's own half ends at y <= 15.0 and
    // team 1's starts at y >= 18.0, symmetric in each side's frame (33 - 18 =
    // 15).
    REQUIRE(game.isValidPlacement(0, 9.0f, 14.0f, false, Entity::IMPLICIT_TROOP_RADIUS));
    REQUIRE_FALSE(game.isValidPlacement(0, 9.0f, 16.0f, false, Entity::IMPLICIT_TROOP_RADIUS));

    REQUIRE(game.isValidPlacement(1, 9.0f, 19.0f, false, Entity::IMPLICIT_TROOP_RADIUS));
    REQUIRE_FALSE(game.isValidPlacement(1, 9.0f, 17.0f, false, Entity::IMPLICIT_TROOP_RADIUS));
}

TEST_CASE("isValidPlacement lets spells ignore the half restriction", "[game_manager][placement]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    REQUIRE(game.isValidPlacement(0, 9.0f, 25.0f, true, Entity::IMPLICIT_TROOP_RADIUS)); // deep in the opponent's half
}

TEST_CASE("isValidPlacement lets deployAnywhere troops (Miner, Goblin Drill) ignore the half restriction too", "[game_manager][placement]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    REQUIRE_FALSE(game.isValidPlacement(0, 9.0f, 25.0f, false, Entity::IMPLICIT_TROOP_RADIUS));       // ordinary troop: rejected
    REQUIRE(game.isValidPlacement(0, 9.0f, 25.0f, false, Entity::IMPLICIT_TROOP_RADIUS, true));       // deployAnywhere: allowed
}

TEST_CASE("Miner can be played deep in the opponent's half via GameManager::playCard, unlike an ordinary troop", "[game_manager][placement]") {
    GameManager game({ 52, 0, 1, 2, 3, 4, 5, 6 }, { 0,1,2,3,4,5,6,7 }); // Miner (52) first in the AI's deck
    game.playerAI.elixir = 10.0f;
    game.playerAI.hand[0] = 52; // force into hand: the opening hand is random

    REQUIRE(game.playCard(0, 52, 9.0f, 25.0f)); // deep in the opponent's half: succeeds for Miner
}

TEST_CASE("Elixir Collector passively grants its owner extra elixir beyond normal regen", "[game_manager][elixir]") {
    GameManager game({ 99, 0, 1, 2, 3, 4, 5, 6 }, { 0,1,2,3,4,5,6,7 }); // Elixir Collector (99) first in the AI's deck
    // Elixir Collector can never be in the opening hand, so it is forced in:
    // this test is about its mechanic once deployed.
    game.playerAI.hand[0] = 99;
    game.playerAI.elixir = 10.0f; // enough to afford its cost (6)
    REQUIRE(game.playCard(0, 99, 9.0f, 10.0f));

    game.playerAI.elixir = 0.0f; // reset low so the cap doesn't mask the effect
    // 80 ticks of interval plus the 10-tick deploy time before its first cycle.
    for (int i = 0; i < 80 + DEPLOY_TIME_TICKS; ++i) game.step();

    float pureRegenOnly = 0.035f * (80 + DEPLOY_TIME_TICKS); // regen alone, same window
    REQUIRE(game.getElixirAI() > pureRegenOnly + 0.5f); // meaningfully more than regen alone
}

TEST_CASE("isValidPlacement rejects placement overlapping an existing building", "[game_manager][placement]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    // The AI King Tower is at (8.5, 2.5) with radius 2.0; the test points at x
    // = 9.0 are ~2.06 and ~7.02 away.

    SECTION("placing a building-shaped card: requiredDist = Building::COLLISION_RADIUS (1.0) + 2.0 = 3.0") {
        REQUIRE_FALSE(game.isValidPlacement(0, 9.0f, 4.5f, false, Building::COLLISION_RADIUS)); // dist ~2.06 < 3.0
        REQUIRE(game.isValidPlacement(0, 9.0f, 9.5f, false, Building::COLLISION_RADIUS));        // dist ~7.02, clear
    }

    SECTION("placing a troop-shaped card: requiredDist = Entity::IMPLICIT_TROOP_RADIUS (0.4) + 2.0 = 2.4") {
        REQUIRE_FALSE(game.isValidPlacement(0, 9.0f, 4.5f, false, Entity::IMPLICIT_TROOP_RADIUS)); // dist ~2.06 < 2.4
        REQUIRE(game.isValidPlacement(0, 9.0f, 9.5f, false, Entity::IMPLICIT_TROOP_RADIUS));        // dist ~7.02, clear
    }
}

TEST_CASE("isValidPlacement's required gap tracks the placed card's own footprint, not a flat guess", "[game_manager][placement]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    // ~2.75 from the AI King Tower (8.5, 2.5): inside a flat 1.0 + 2.0 = 3.0
    // bound but outside the troop bound 0.4 + 2.0 = 2.4, a legal troop
    // placement a flat radius would reject.
    REQUIRE_FALSE(game.isValidPlacement(0, 9.0f, 5.2f, false, Building::COLLISION_RADIUS));
    REQUIRE(game.isValidPlacement(0, 9.0f, 5.2f, false, Entity::IMPLICIT_TROOP_RADIUS));
}

TEST_CASE("isValidPlacement rejects a body whose FOOTPRINT leaves the arena, and only its footprint",
          "[game_manager][placement][regression]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });

    // A body legal by its centre can still hang off the arena, so the footprint
    // is checked against the physical edge: cell i covers [i - 0.5, i + 0.5]
    // (Board::CELL_HALF_EXTENT), so an 18-wide board runs x in [-0.5, 17.5],
    // not [0, 17]. The troop section is the load-bearing half: checking
    // footprints against the index range would reject a 0.4-radius troop at
    // both edge columns and delete two columns from the action space.
    const float maxX = static_cast<float>(game.getBoard().getWidth() - 1);   // 17

    SECTION("a building may not hang off either side edge") {
        REQUIRE_FALSE(game.isValidPlacement(0, 0.0f, 10.0f, false, Building::COLLISION_RADIUS));
        REQUIRE_FALSE(game.isValidPlacement(0, maxX, 10.0f, false, Building::COLLISION_RADIUS));
        // One column in from each edge, [0.0, 2.0] and [15.0, 17.0], both
        // inside [-0.5, 17.5]: the rule costs buildings exactly one column a
        // side.
        REQUIRE(game.isValidPlacement(0, 1.0f, 10.0f, false, Building::COLLISION_RADIUS));
        REQUIRE(game.isValidPlacement(0, maxX - 1.0f, 10.0f, false, Building::COLLISION_RADIUS));
    }

    SECTION("a troop keeps BOTH edge columns") {
        REQUIRE(game.isValidPlacement(0, 0.0f, 10.0f, false, Entity::IMPLICIT_TROOP_RADIUS));
        REQUIRE(game.isValidPlacement(0, maxX, 10.0f, false, Entity::IMPLICIT_TROOP_RADIUS));
    }

    SECTION("a spell is exempt -- its radius is an area of effect, not a body") {
        REQUIRE(game.isValidPlacement(0, 0.0f, 10.0f, true, 3.0f));
        REQUIRE(game.isValidPlacement(0, maxX, 25.0f, true, 3.0f));
    }

    SECTION("the y axis is checked the same way") {
        // x = 5.6 is inside the back row's opening and 3.83 from the King Tower
        // (8.5, 2.5, radius 2.0), clearing both 2.4 and 3.0, so the overlap
        // rule cannot reject either arm and only the footprint check can fail
        // it.
        REQUIRE(game.isValidPlacement(0, 5.6f, 0.0f, false, Entity::IMPLICIT_TROOP_RADIUS));
        REQUIRE_FALSE(game.isValidPlacement(0, 5.6f, 0.0f, false, Building::COLLISION_RADIUS));
    }
}

// ---------------- playCard ----------------

TEST_CASE("playCard: successful play deducts elixir, cycles the hand, and spawns the entity", "[game_manager][play_card]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 8,9,10,11,12,13,14,17 });
    // Pin the deterministic hand/queue partition: this tests the cycling
    // mechanism (the played slot refills from deckQueue.front()).
    game.playerAI.hand = { 0, 1, 2, 3 };
    game.playerAI.deckQueue = { 4, 5, 6, 7 };
    REQUIRE(game.getHand(0) == std::vector<int>{0, 1, 2, 3});
    REQUIRE(game.getElixirAI() == Catch::Approx(5.0f));

    size_t countBefore = game.getBoard().getEntities().size();
    bool played = game.playCard(0, 0, 9.0f, 10.0f); // Knight, cost 3.0

    REQUIRE(played);
    REQUIRE(game.getElixirAI() == Catch::Approx(2.0f));
    REQUIRE(game.getHand(0)[0] == 4); // next queued card cycled in

    // spawnEntity only queues into pendingEntities; getEntities() shows it
    // after the next commit.
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
    game.playerAI.hand[0] = 19; // force into hand: the opening hand is random
    bool played = game.playCard(0, 19, 9.0f, 10.0f);
    REQUIRE_FALSE(played);
    REQUIRE(game.getElixirAI() == Catch::Approx(5.0f));
    REQUIRE(game.getHand(0)[0] == 19);
}

TEST_CASE("playCard fails for a placement in the opponent's half, without spending elixir", "[game_manager][play_card]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    game.playerAI.hand[0] = 0; // force into hand: the opening hand is random
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

TEST_CASE("step() ends the game as a draw when both King towers die the same tick", "[game_manager][step]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    auto aiKing = game.getBoard().getEntities()[0];
    auto oppKing = game.getBoard().getEntities()[1];
    REQUIRE(aiKing->team == 0);
    REQUIRE(oppKing->team == 1);

    aiKing->takeDamage(aiKing->hp);
    oppKing->takeDamage(oppKing->hp);
    game.step();

    REQUIRE(game.isGameOver());
    REQUIRE(game.getLoserTeam() == -1); // draw, not an arbitrary team-0 loss
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

// --- step()'s post-collision clamp respects riverIgnores ---

TEST_CASE("step()'s post-collision re-clamp respects riverIgnores (regression test)", "[game_manager][step]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    Board& board = game.getBoard();

    // Mid-river at a non-bridge column, like a Hog mid-crossing. Both clamps go
    // through Board::clampToBoard(), so neither snaps it back.
    auto hog = std::make_shared<BuildingTargeter>(board.allocateId(), 8.0f, 16.5f, 1408, 0, 0.8f, 1.0f, 264, 15, 'H');
    hog->setIgnoresRiver(true);
    board.addEntity(hog);
    board.commitPendingEntities();

    game.step();

    REQUIRE(hog->position.y > 15.5f);
    REQUIRE(hog->position.y < 17.5f);
}

// ---------------- statistics ----------------

TEST_CASE("A full scripted mini-match produces sane getStatistics() output", "[game_manager][statistics]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    Board& board = game.getBoard();

    game.playerAI.hand[0] = 0; // force into hand: the opening hand is random
    bool played = game.playCard(0, 0, 9.0f, 10.0f); // Knight, cost 3.0
    REQUIRE(played);

    // Next to the Knight's placement, inside its attack range.
    auto enemy = std::make_shared<DummyEntity>(board.allocateId(), 9.0f, 10.5f, 1000, 1);
    spawn(board, enemy);

    // 5 ticks of combat after the Knight's 10-tick deploy time.
    for (int i = 0; i < 5 + DEPLOY_TIME_TICKS; ++i) game.step();

    const auto& stats = game.getStatistics();
    REQUIRE(stats.elixirSpent(0) == Catch::Approx(3.0f));
    REQUIRE(stats.cardsPlayed(0).size() == 1);
    REQUIRE(stats.cardsPlayed(0)[0].cardId == 0);
    REQUIRE(stats.totalDamageDealt(0) > 0); // the Knight landed at least one hit
    REQUIRE(stats.damageDealtByCard(0, 0) > 0);
    REQUIRE(stats.loserTeam() == -1); // match still ongoing
    REQUIRE(stats.matchDurationTicks() == 0); // MatchEndedEvent hasn't fired yet
}

TEST_CASE("GameManager::reset() gives a fresh MatchStatistics, not stale numbers from the previous match", "[game_manager][statistics]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    game.playerAI.hand[0] = 0; // force into hand: the opening hand is random
    bool played = game.playCard(0, 0, 9.0f, 10.0f); // Knight, cost 3.0
    REQUIRE(played);
    game.step();

    REQUIRE(game.getStatistics().elixirSpent(0) == Catch::Approx(3.0f));
    REQUIRE(game.getStatistics().cardsPlayed(0).size() == 1);

    game.reset();
    game.playerAI.hand[0] = 0; // reset() reshuffles: force it again

    REQUIRE(game.getStatistics().elixirSpent(0) == Catch::Approx(0.0f));
    REQUIRE(game.getStatistics().cardsPlayed(0).empty());

    // The fresh stats object listens to the new board: a post-reset play is
    // picked up.
    bool playedAgain = game.playCard(0, 0, 9.0f, 10.0f);
    REQUIRE(playedAgain);
    REQUIRE(game.getStatistics().elixirSpent(0) == Catch::Approx(3.0f));
}

TEST_CASE("getStatistics() reports the match outcome once the game ends", "[game_manager][statistics]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    auto aiKing = game.getBoard().getEntities()[0];
    aiKing->takeDamage(aiKing->hp);

    game.step();

    REQUIRE(game.getStatistics().loserTeam() == 0);
    REQUIRE(game.getStatistics().matchDurationTicks() == 1);
}

// --- PlayerState::playCard on an empty deckQueue ---

TEST_CASE("PlayerState::playCard does not spend elixir when the deck queue is empty", "[player_state]") {
    PlayerState player;
    player.initializeDeck({ 0, 1, 2, 3 }); // only 4 cards: deckQueue starts empty
    REQUIRE(player.elixir == Catch::Approx(5.0f));

    PlayerState::PlayCardResult result = player.playCard(0); // Knight, cost 3.0

    REQUIRE(result.cardId == -1);
    REQUIRE_FALSE(result.useEvolvedForm);
    REQUIRE(player.elixir == Catch::Approx(5.0f)); // untouched
    REQUIRE(player.hand[0] == 0);                  // hand slot untouched
}

// --- random opening hand and the hand cycle delay ---

TEST_CASE("PlayerState::initializeDeck's random-hand overload excludes Elixir Collector/Mirror from the opening hand",
        "[player_state][hand_random]") {
    std::mt19937 rng(12345);
    std::vector<int> deck = { 99, 164, 0, 1, 2, 3, 4, 5 }; // Elixir Collector + Mirror both present
    for (int trial = 0; trial < 200; ++trial) {
        PlayerState player;
        player.initializeDeck(deck, rng);
        REQUIRE(player.hand.size() == 4);
        REQUIRE(std::find(player.hand.begin(), player.hand.end(), 99) == player.hand.end());
        REQUIRE(std::find(player.hand.begin(), player.hand.end(), 164) == player.hand.end());

        // hand + deckQueue remain a permutation of the original 8.
        std::vector<int> combined(player.hand.begin(), player.hand.end());
        combined.insert(combined.end(), player.deckQueue.begin(), player.deckQueue.end());
        std::sort(combined.begin(), combined.end());
        std::vector<int> expected = deck;
        std::sort(expected.begin(), expected.end());
        REQUIRE(combined == expected);
    }
}

TEST_CASE("Random-hand initializeDeck seeds championSlots by ORIGINAL deck index, not wherever the shuffle put the card",
        "[player_state][hand_random][champion]") {
    std::mt19937 rng(777);
    std::vector<int> deck = { 1, 115, 2, 3, 4, 5, 6, 7 }; // Mighty Miner (Champion) at ORIGINAL index 1
    for (int trial = 0; trial < 50; ++trial) {
        PlayerState player;
        player.initializeDeck(deck, rng);
        // Seeded by the original deck index (1), wherever the shuffle put card
        // 115: championSlots tracks deck position.
        REQUIRE(player.championSlots.count(1) == 1);
        REQUIRE(player.championSlots.count(2) == 0);
    }
}

TEST_CASE("A freshly-cycled-in hand slot is unplayable for 19 ticks, playable on the 20th -- the opening hand needs no such wait",
        "[player_state][cycle_delay]") {
    PlayerState player;
    player.initializeDeck({ 0, 1, 2, 3, 4, 5, 6, 7 }); // deterministic overload: hand={0,1,2,3}, queue={4,5,6,7}
    player.elixir = 100.0f;

    // Opening hand needs no wait at all.
    auto opening = player.playCard(0);
    REQUIRE(opening.cardId == 0);

    // hand[0] is now 4, cycled in from the queue with a fresh 20-tick cooldown:
    // unplayable for 19 ticks...
    REQUIRE(player.hand[0] == 4);
    for (int i = 0; i < 19; ++i) {
        auto blocked = player.playCard(0);
        REQUIRE(blocked.cardId == -1); // still cooling down
        REQUIRE(player.hand[0] == 4);  // nothing consumed, hand slot untouched
        player.tick();
    }
    auto stillBlocked = player.playCard(0); // 19 ticks elapsed -- one more needed
    REQUIRE(stillBlocked.cardId == -1);

    // ...and playable once the 20th tick lands.
    player.tick();
    auto afterCooldown = player.playCard(0);
    REQUIRE(afterCooldown.cardId == 4);
}

// --- activateChampionAbility ---
// Mighty Miner (115) sits at deck slot 1 (Champions are legal only in slots 1
// and 2). The opening hand is random, so each test forces hand[1] = 115; he is
// affordable (4.0) from the starting 5.0, and activateChampionAbility's slot
// defaults to 1.

TEST_CASE("activateChampionAbility fires, deducts elixir, and starts the cooldown", "[game_manager][champion]") {
    GameManager game({ 1, 115, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    game.playerAI.hand[1] = 115;
    game.playCard(0, 115, 9.0f, 10.0f);
    game.step(); // commit, so the Champion is visible
    float elixirBefore = game.getElixirAI();

    bool activated = game.activateChampionAbility(0);

    REQUIRE(activated);
    REQUIRE(game.getElixirAI() == Catch::Approx(elixirBefore - 1.0f)); // 1-elixir ability cost
}

TEST_CASE("activateChampionAbility feeds MatchStatistics' Champion ability tracking", "[game_manager][champion][stats]") {
    GameManager game({ 1, 115, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    game.playerAI.hand[1] = 115;
    game.playCard(0, 115, 9.0f, 10.0f);
    game.step(); // commit, so the Champion is visible

    REQUIRE(game.activateChampionAbility(0));

    REQUIRE(game.getStatistics().championAbilityElixirSpent(0) == Catch::Approx(1.0f));
    REQUIRE(game.getStatistics().championAbilityActivations(0) == 1);
    REQUIRE(game.getStatistics().championAbilityActivations(1) == 0);
    // Separate from card-play tracking: only deploying the Champion counts as
    // elixir spent on a card.
    REQUIRE(game.getStatistics().cardsPlayed(0).size() == 1);
}

TEST_CASE("activateChampionAbility fails when the team has no deployed Champion", "[game_manager][champion]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    REQUIRE_FALSE(game.activateChampionAbility(0));
}

TEST_CASE("activateChampionAbility fails when unaffordable, without deducting anything", "[game_manager][champion]") {
    GameManager game({ 1, 115, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    game.playerAI.hand[1] = 115;
    game.playCard(0, 115, 9.0f, 10.0f);
    game.step(); // commit, so the Champion is visible
    game.playerAI.elixir = 0.5f; // below the 1.0 ability cost

    bool activated = game.activateChampionAbility(0);

    REQUIRE_FALSE(activated);
    REQUIRE(game.getElixirAI() == Catch::Approx(0.5f)); // untouched
}

TEST_CASE("activateChampionAbility fails while still on cooldown", "[game_manager][champion]") {
    GameManager game({ 1, 115, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    game.playerAI.hand[1] = 115;
    game.playCard(0, 115, 9.0f, 10.0f);
    game.step(); // commit, so the Champion is visible
    game.playerAI.elixir = 10.0f; // plenty for a second attempt, if cooldown didn't block it

    REQUIRE(game.activateChampionAbility(0));
    REQUIRE_FALSE(game.activateChampionAbility(0));
}

TEST_CASE("activateChampionAbility is ready again once its full cooldown elapses via step()", "[game_manager][champion]") {
    GameManager game({ 1, 115, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    game.playerAI.hand[1] = 115;
    game.playCard(0, 115, 9.0f, 10.0f);
    game.step(); // commit, so the Champion is visible
    game.playerAI.elixir = 10.0f;
    REQUIRE(game.activateChampionAbility(0));

    // Fast-forward the cooldown directly: 130 real ticks would let the Miner
    // wander into combat and make the timing depend on it.
    std::shared_ptr<CombatEntity> champion;
    for (const auto& e : game.getBoard().getEntities()) {
        auto candidate = std::dynamic_pointer_cast<CombatEntity>(e);
        if (candidate && candidate->isChampion) { champion = candidate; break; }
    }
    REQUIRE(champion != nullptr);

    champion->abilityCooldownRemaining = 1;
    game.playerAI.elixir = 10.0f;
    REQUIRE_FALSE(game.activateChampionAbility(0)); // still 1 tick left

    game.step(); // decrements the last tick via CombatEntity::update()
    game.playerAI.elixir = 10.0f;
    REQUIRE(game.activateChampionAbility(0));
}

TEST_CASE("activateChampionAbility fails once the game is over", "[game_manager][champion]") {
    GameManager game({ 1, 115, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    game.playerAI.hand[1] = 115;
    game.playCard(0, 115, 9.0f, 10.0f);
    game.step(); // commit, so the Champion is visible
    auto aiKing = game.getBoard().getEntities()[0];
    aiKing->takeDamage(aiKing->hp);
    game.step();

    REQUIRE(game.isGameOver());
    REQUIRE_FALSE(game.activateChampionAbility(0));
}

// ---------------- multi-Champion, per-slot ability tracking ----------------

TEST_CASE("Two different Champions in slots 1 and 2 have fully independent ability readiness", "[game_manager][champion][multi]") {
    GameManager game({ 1, 115, 118, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    game.playerAI.elixir = 100.0f;
    // Pin hand and queue fully: playing 115 from hand[1] draws
    // deckQueue.front(), and a leftover 118 there would collide with the forced
    // hand[2] = 118 (see test_mirror.cpp).
    game.playerAI.hand = { 1, 115, 118, 3 };
    game.playerAI.deckQueue = { 4, 5, 6, 7 };
    game.playCard(0, 115, 9.0f, 10.0f);  // Mighty Miner -> slot 1
    game.playCard(0, 118, 12.0f, 10.0f); // Archer Queen -> slot 2
    game.step();

    REQUIRE(game.isChampionAbilityReady(0, 1));
    REQUIRE(game.isChampionAbilityReady(0, 2));

    REQUIRE(game.activateChampionAbility(0, 1));
    REQUIRE_FALSE(game.isChampionAbilityReady(0, 1)); // now on cooldown
    REQUIRE(game.isChampionAbilityReady(0, 2));        // slot 2 completely unaffected

    REQUIRE(game.activateChampionAbility(0, 2));
    REQUIRE_FALSE(game.isChampionAbilityReady(0, 2));
}

TEST_CASE("Redeploying the same Champion tracks the newest instance for ability activation", "[game_manager][champion][multi]") {
    GameManager game({ 1, 115, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    game.playerAI.elixir = 100.0f;
    game.playerAI.hand[1] = 115; // force into hand: the opening hand is random
    game.playCard(0, 115, 9.0f, 10.0f); // 1st deploy
    game.step();

    std::shared_ptr<CombatEntity> firstInstance;
    for (const auto& e : game.getBoard().getEntities()) {
        auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
        if (ce && ce->isChampion) { firstInstance = ce; break; }
    }
    REQUIRE(firstInstance != nullptr);

    game.playerAI.hand[1] = 115; // simulate it having cycled back to hand
    game.playerAI.handCooldownTicks[1] = 0; // skip the cycle-in delay
    game.playCard(0, 115, 12.0f, 10.0f); // 2nd deploy, while the first is still alive
    game.step();

    int championCount = 0;
    std::shared_ptr<CombatEntity> secondInstance;
    for (const auto& e : game.getBoard().getEntities()) {
        auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
        if (ce && ce->isChampion) {
            championCount++;
            if (ce->id != firstInstance->id) secondInstance = ce;
        }
    }
    REQUIRE(championCount == 2); // both alive simultaneously
    REQUIRE(secondInstance != nullptr);

    game.playerAI.elixir = 100.0f;
    REQUIRE(game.activateChampionAbility(0, 1));
    REQUIRE(secondInstance->abilityCooldownRemaining > 0); // the newest instance is the one that fired
    REQUIRE(firstInstance->abilityCooldownRemaining == 0); // the original, no-longer-tracked instance is untouched
}

TEST_CASE("A Champion's ability cooldown persists across death and redeploy of the same slot", "[game_manager][champion][multi]") {
    GameManager game({ 1, 115, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    game.playerAI.elixir = 100.0f;
    game.playerAI.hand[1] = 115; // force into hand: the opening hand is random
    game.playCard(0, 115, 9.0f, 10.0f);
    game.step();

    REQUIRE(game.activateChampionAbility(0, 1)); // cooldown set to 130 on the live entity
    game.step(); // synced into persistedCooldownRemaining (129) while still alive

    for (const auto& e : game.getBoard().getEntities()) {
        auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
        if (ce && ce->isChampion) { ce->takeDamage(ce->hp); break; }
    }
    game.step(); // this tick's sync sees it already dead, leaves the persisted value at 129; cleans the corpse up

    game.playerAI.hand[1] = 115;
    game.playerAI.handCooldownTicks[1] = 0; // skip the cycle-in delay
    game.playerAI.elixir = 100.0f;
    game.playCard(0, 115, 9.0f, 10.0f); // redeploy -- should seed from the persisted cooldown, not start at 0
    game.step();

    REQUIRE_FALSE(game.isChampionAbilityReady(0, 1)); // still cooling down, not freshly ready
}

TEST_CASE("A cloned Champion can never activate the ability, even after the original dies", "[game_manager][champion][multi]") {
    GameManager game({ 1, 115, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    game.playerAI.elixir = 100.0f;
    game.playerAI.hand[1] = 115; // force into hand: the opening hand is random
    game.playCard(0, 115, 9.0f, 10.0f);
    game.step();

    std::shared_ptr<CombatEntity> original;
    for (const auto& e : game.getBoard().getEntities()) {
        auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
        if (ce && ce->isChampion) { original = ce; break; }
    }
    REQUIRE(original != nullptr);

    auto clone = std::dynamic_pointer_cast<CombatEntity>(original->clone(game.getBoard().allocateId()));
    REQUIRE(clone != nullptr);
    REQUIRE(clone->isChampion); // the clone DOES carry isChampion=true -- it's the tracking that must exclude it
    game.getBoard().addEntity(clone);
    game.step();

    REQUIRE(game.isChampionAbilityReady(0, 1)); // still resolves to the original (still alive), not the clone

    original->takeDamage(original->hp); // kill the original
    game.step();

    REQUIRE_FALSE(game.isChampionAbilityReady(0, 1)); // the clone is alive but was never tracked -- unreachable
    REQUIRE_FALSE(game.activateChampionAbility(0, 1));
}

TEST_CASE("a rolling spell may be cast on its own half and the river, but no further",
          "[game_manager][placement][roll]") {
    // The Log and Barbarian Barrel may not be cast deep in enemy territory:
    // they damage by rolling forward, so such a cast rolls away from
    // everything.
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    game.reset();
    const float riverStart = game.getBoard().getRiverStart();
    const float riverEnd = game.getBoard().getRiverEnd();

    SECTION("team 0 keeps its own half and the river") {
        REQUIRE(game.isValidPlacement(0, 9.0f, 5.0f, true, 0.0f, false, true));
        REQUIRE(game.isValidPlacement(0, 9.0f, 15.0f, true, 0.0f, false, true));
        REQUIRE(game.isValidPlacement(0, 9.0f, riverStart, true, 0.0f, false, true));
        REQUIRE(game.isValidPlacement(0, 9.0f, riverEnd, true, 0.0f, false, true));
    }

    SECTION("team 0 is refused past the river") {
        REQUIRE_FALSE(game.isValidPlacement(0, 9.0f, 18.0f, true, 0.0f, false, true));
        REQUIRE_FALSE(game.isValidPlacement(0, 9.0f, 25.0f, true, 0.0f, false, true));
        REQUIRE_FALSE(game.isValidPlacement(0, 9.0f, 31.0f, true, 0.0f, false, true));
    }

    SECTION("team 1 is the exact mirror") {
        REQUIRE(game.isValidPlacement(1, 9.0f, 28.0f, true, 0.0f, false, true));
        REQUIRE(game.isValidPlacement(1, 9.0f, riverEnd, true, 0.0f, false, true));
        REQUIRE(game.isValidPlacement(1, 9.0f, riverStart, true, 0.0f, false, true));
        REQUIRE_FALSE(game.isValidPlacement(1, 9.0f, 15.0f, true, 0.0f, false, true));
        REQUIRE_FALSE(game.isValidPlacement(1, 9.0f, 5.0f, true, 0.0f, false, true));
    }

    SECTION("NON-rolling spells are unaffected and still go anywhere") {
        // Only rollers are restricted; a Fireball across the river stays legal.
        REQUIRE(game.isValidPlacement(0, 9.0f, 25.0f, true, 0.0f, false, false));
        REQUIRE(game.isValidPlacement(1, 9.0f, 5.0f, true, 0.0f, false, false));
    }
}

TEST_CASE("the rolling-spell zone still lets The Log reach the enemy Princess Tower",
          "[game_manager][placement][roll][bridge]") {
    // The load-bearing half: a strict own-half rule (y <= 15.0) would leave the
    // enemy tower out of reach of the 10.1 roll (25.1 against a tower edge at
    // 25.5). Including the river band preserves the interaction; this asserts
    // the arithmetic.
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    game.reset();
    const float riverEnd = game.getBoard().getRiverEnd();                 // 17.5
    const float ownHalfMax = game.getBoard().getRiverStart() - 0.5f;      // 15.0
    const float towerEdge = ArenaLayout::princessY(1) - 1.5f;             // 25.5
    const float logRange = CardRegistry::getInstance().getCard(33)->rollRange;

    REQUIRE(logRange == Catch::Approx(10.1f));
    REQUIRE(ownHalfMax + logRange < towerEdge);   // strict own half CANNOT reach
    REQUIRE(riverEnd + logRange > towerEdge);     // own half + river CAN

    // ...and the highest legal cell is legal.
    REQUIRE(game.isValidPlacement(0, ArenaLayout::LEFT_LANE_X, riverEnd, true, 0.0f, false, true));
}

TEST_CASE("the registry's two rolling spells are the only cards this rule catches",
          "[game_manager][placement][roll][registry]") {
    // The flag keys on rollRange > 0: exactly The Log, Barbarian Barrel and its
    // Hero variant.
    const auto& registry = CardRegistry::getInstance();
    std::vector<int> rollers;
    for (const auto& pair : registry.getAllCards()) {
        if (pair.second.rollRange > 0.0f) rollers.push_back(pair.first);
    }
    std::sort(rollers.begin(), rollers.end());
    // 33 The Log, 101 Barbarian Barrel, 174 its Hero variant, and nothing else.
    REQUIRE(rollers == std::vector<int>{33, 101, 174});
    REQUIRE(registry.getCard(7)->rollRange == 0.0f);    // Fireball unaffected
    REQUIRE(registry.getCard(100)->rollRange == 0.0f);  // Giant Snowball unaffected
}

// --- elixir phases ---
// In this existing file on purpose: a new test file needs the build run twice
// to be linked in.

TEST_CASE("elixir regen follows the real-game 1x/2x/3x phase schedule",
          "[game_manager][elixir_phase]") {
    // The schedule on both sides of each boundary; a one-sided check passes an
    // off-by-one.
    REQUIRE(GameManager::elixirMultiplierAtTick(0) == 1.0f);
    REQUIRE(GameManager::elixirMultiplierAtTick(GameManager::DOUBLE_ELIXIR_TICK - 1) == 1.0f);
    REQUIRE(GameManager::elixirMultiplierAtTick(GameManager::DOUBLE_ELIXIR_TICK) == 2.0f);
    REQUIRE(GameManager::elixirMultiplierAtTick(GameManager::TRIPLE_ELIXIR_TICK - 1) == 2.0f);
    REQUIRE(GameManager::elixirMultiplierAtTick(GameManager::TRIPLE_ELIXIR_TICK) == 3.0f);

    // The boundaries against the 10-ticks-per-second conversion.
    REQUIRE(GameManager::DOUBLE_ELIXIR_TICK == 120 * 10);   // 2:00
    REQUIRE(GameManager::TRIPLE_ELIXIR_TICK == 180 * 10);   // 3:00
    REQUIRE(GameManager::MAX_ELIXIR_MULTIPLIER
            == GameManager::elixirMultiplierAtTick(GameManager::TRIPLE_ELIXIR_TICK));
}

TEST_CASE("elixir income is actually multiplied, measured off the bar",
          "[game_manager][elixir_phase]") {
    std::vector<int> deck = { 15, 6, 25, 40, 24, 72, 33, 7 };

    // Reads the bar after stepping, so it catches step() failing to apply the
    // schedule.
    auto incomeOver = [&](int startTick, int ticks) {
        GameManager game(deck, deck);
        game.setCurrentTick(startTick);
        game.setElixir(0, 0.0f);
        const float before = game.playerAI.elixir;
        for (int i = 0; i < ticks; ++i) game.step();
        return game.playerAI.elixir - before;
    };

    // 20 ticks: the bar clamps at 10.0, so a long window would saturate and
    // every phase would read the same.
    REQUIRE(incomeOver(0, 20) == Catch::Approx(0.7f).margin(0.01f));
    REQUIRE(incomeOver(GameManager::DOUBLE_ELIXIR_TICK, 20) == Catch::Approx(1.4f).margin(0.01f));
    REQUIRE(incomeOver(GameManager::TRIPLE_ELIXIR_TICK, 20) == Catch::Approx(2.1f).margin(0.01f));

    // Crossing a boundary mid-window. step() increments the tick before reading
    // the phase, so stepping N from T covers T+1..T+N: from 1190, 9 ticks at 1x
    // (1191..1199) and 11 at 2x (1200..1210), 0.315 + 0.770. Pins per-tick
    // evaluation, which a schedule sampled once per step() would fail.
    REQUIRE(incomeOver(GameManager::DOUBLE_ELIXIR_TICK - 10, 20)
            == Catch::Approx(1.085f).margin(0.01f));
}

TEST_CASE("the elixir phase composes with the curriculum's opponent multiplier",
          "[game_manager][elixir_phase]") {
    std::vector<int> deck = { 15, 6, 25, 40, 24, 72, 33, 7 };

    auto oppIncomeOver = [&](int startTick, int ticks, float mult) {
        GameManager game(deck, deck);
        game.setOpponentElixirMultiplier(mult);
        game.setCurrentTick(startTick);
        game.setElixir(0, 0.0f);
        game.setElixir(1, 0.0f);
        const float before = game.playerOpponent.elixir;
        for (int i = 0; i < ticks; ++i) game.step();
        return std::make_pair(game.playerOpponent.elixir - before,
                              game.playerAI.elixir);
    };

    // The phase scales the base rate and oppElixirMultiplier scales that: a
    // 1.5x opponent in double elixir gets 3.0x, as intended.
    REQUIRE(oppIncomeOver(0, 20, 1.5f).first == Catch::Approx(1.05f).margin(0.01f));
    REQUIRE(oppIncomeOver(GameManager::DOUBLE_ELIXIR_TICK, 20, 1.5f).first
            == Catch::Approx(2.10f).margin(0.01f));

    // ...and the handicap never reaches the AI's own bar.
    REQUIRE(oppIncomeOver(GameManager::DOUBLE_ELIXIR_TICK, 20, 1.5f).second
            == Catch::Approx(1.40f).margin(0.01f));
}
