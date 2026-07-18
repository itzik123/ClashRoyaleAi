#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "MatchRules.h"

namespace {
    void spawnKing(Board& board, int id, int team, bool alive) {
        auto king = std::make_shared<DummyEntity>(id, 9.0f, team == 0 ? 2.0f : 30.0f, 100, team, 'R');
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
