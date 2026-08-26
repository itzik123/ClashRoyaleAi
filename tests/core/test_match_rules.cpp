#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "MatchRules.h"
#include "Tower.h"
#include "CardRegistry.h"

namespace {
    // A real Tower, not a DummyEntity wearing the 'R' symbol.
    //
    // These four cases used a DummyEntity with symbol 'R' as a stand-in King,
    // which is EXACTLY the substitution card id 93 (Mortar, registered with
    // symbol 'R') performs at runtime -- so the suite could not distinguish a
    // King from anything else carrying that symbol, and neither could
    // MatchRules. Building the real type is what makes the case below
    // meaningful.
    void spawnKing(Board& board, int id, int team, bool alive) {
        auto king = std::make_shared<Tower>(id, 9.0f, team == 0 ? 2.5f : 30.5f, 100, team,
                                            7.0f, 90, 10, 'R');
        if (!alive) king->takeDamage(100);
        spawn(board, king);
    }
}

TEST_CASE("MatchRules::evaluate: both kings alive means the match is ongoing", "[match_rules]") {
    Board board;
    spawnKing(board, 1, 0, true);
    spawnKing(board, 2, 1, true);

    auto outcome = MatchRules::evaluate(board);
    REQUIRE_FALSE(outcome.over);
    REQUIRE(outcome.loserTeam == -1);
}

TEST_CASE("MatchRules::evaluate: team 0's king dead means team 0 lost", "[match_rules]") {
    Board board;
    spawnKing(board, 1, 0, false);
    spawnKing(board, 2, 1, true);

    auto outcome = MatchRules::evaluate(board);
    REQUIRE(outcome.over);
    REQUIRE(outcome.loserTeam == 0);
}

TEST_CASE("MatchRules::evaluate: team 1's king dead means team 1 lost", "[match_rules]") {
    Board board;
    spawnKing(board, 1, 0, true);
    spawnKing(board, 2, 1, false);

    auto outcome = MatchRules::evaluate(board);
    REQUIRE(outcome.over);
    REQUIRE(outcome.loserTeam == 1);
}

TEST_CASE("MatchRules::evaluate: both kings dead the same tick is a draw, not team 0 losing by default", "[match_rules]") {
    Board board;
    spawnKing(board, 1, 0, false);
    spawnKing(board, 2, 1, false);

    auto outcome = MatchRules::evaluate(board);
    REQUIRE(outcome.over);
    REQUIRE(outcome.loserTeam == -1); // draw -- not 0
}


TEST_CASE("MatchRules::evaluate: a Mortar does not stand in for a King Tower",
          "[match_rules][regression]") {
    // Card id 93 (Mortar) is registered with symbol 'R' -- the same character
    // GameManager::addTower gives the King and the only thing evaluate() used
    // to look at. A live Mortar therefore reported its owner's King as alive.
    //
    // The consequence is not cosmetic: with team 0's King destroyed and a
    // team 0 Mortar still standing, the match does not end. cleanDeadEntities
    // erases the dead King on the same tick, so from the next tick onward the
    // ONLY thing answering the "is team 0's King alive" question is the
    // Mortar -- and the win goes unrecorded until it expires or the episode
    // times out. Mortar is in the registry and phase 1's random_opponent
    // samples random decks, so this is reachable in ordinary training.
    //
    // Every other place in the engine that asks this question already guards
    // with the type (Tower.h's Princess count uses isTower() && symbol != 'R';
    // TimeoutRules::resolve uses dynamic_cast and says in its own comment that
    // a symbol check would be fragile). This was the one site that did not.
    Board board;
    spawnKing(board, 1, 0, false);   // team 0's King is destroyed
    spawnKing(board, 2, 1, true);

    // A team 0 Mortar, built the same way a real match builds it.
    CardRegistry::getInstance().getCard(93)->spawnEntity(9.0f, 8.0f, 0, board);
    board.commitPendingEntities();

    // Control: the Mortar really is on the board and really does carry 'R'.
    bool mortarPresent = false;
    for (const auto& e : board.getEntities())
        if (e->cardId == 93 && e->isAlive()) { mortarPresent = true; REQUIRE(e->symbol == 'R'); }
    REQUIRE(mortarPresent);

    auto outcome = MatchRules::evaluate(board);
    REQUIRE(outcome.over);
    REQUIRE(outcome.loserTeam == 0);
}
