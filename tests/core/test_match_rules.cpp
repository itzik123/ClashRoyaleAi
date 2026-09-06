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

// ===========================================================================
// Regulation and overtime (2026-09-06). Real Clash Royale ends a match four
// ways and this engine modelled only two: a King falling, and the tower-hp
// tiebreak at the very end. The two in between -- a crown LEAD winning at 3:00,
// and the first crown in overtime winning immediately -- are what these pin.
// ===========================================================================
namespace {
    // A Princess Tower, so a census has something to count besides Kings.
    void spawnPrincess(Board& board, int id, int team, bool alive, int hp = 100) {
        auto t = std::make_shared<Tower>(id, team == 0 ? 3.0f : 14.0f,
                                         team == 0 ? 6.0f : 27.0f, hp, team,
                                         7.0f, 50, 10, 'P');
        if (!alive) t->takeDamage(hp);
        spawn(board, t);
    }
}

TEST_CASE("a crown lead does not end the match before regulation ends", "[match][overtime]") {
    Board board;
    spawnKing(board, 1, 0, true);
    spawnKing(board, 2, 1, true);
    spawnPrincess(board, 3, 0, true);
    spawnPrincess(board, 4, 1, false);      // team 0 is a crown up

    // One tick before the boundary the match is still running: taking a tower
    // early must not end it, or every match becomes first-crown-wins.
    auto before = MatchRules::evaluateAtTick(board, MatchRules::REGULATION_END_TICK - 1);
    REQUIRE_FALSE(before.over);
}

TEST_CASE("the side ahead on crowns wins when regulation ends", "[match][overtime]") {
    Board board;
    spawnKing(board, 1, 0, true);
    spawnKing(board, 2, 1, true);
    spawnPrincess(board, 3, 0, true);
    spawnPrincess(board, 4, 1, false);      // team 1 has fewer towers

    auto at = MatchRules::evaluateAtTick(board, MatchRules::REGULATION_END_TICK);
    REQUIRE(at.over);
    REQUIRE(at.loserTeam == 1);
}

TEST_CASE("level crowns at regulation send the match to overtime", "[match][overtime]") {
    Board board;
    spawnKing(board, 1, 0, true);
    spawnKing(board, 2, 1, true);
    spawnPrincess(board, 3, 0, true);
    spawnPrincess(board, 4, 1, true);

    // Equal towers -> nobody wins yet, however far past the boundary. The
    // weakest-tower tiebreak belongs to TimeoutRules at maxTicks, NOT here:
    // deciding it on hp at 3:00 would delete overtime entirely.
    REQUIRE_FALSE(MatchRules::evaluateAtTick(board, MatchRules::REGULATION_END_TICK).over);
    REQUIRE_FALSE(MatchRules::evaluateAtTick(board, MatchRules::REGULATION_END_TICK + 900).over);
}

TEST_CASE("the first crown taken in overtime ends the match at once", "[match][overtime]") {
    Board board;
    spawnKing(board, 1, 0, true);
    spawnKing(board, 2, 1, true);
    spawnPrincess(board, 3, 0, true);
    auto contested = std::make_shared<Tower>(4, 14.0f, 27.0f, 100, 1, 7.0f, 50, 10, 'P');
    spawn(board, contested);

    const int t = MatchRules::REGULATION_END_TICK + 300;
    REQUIRE_FALSE(MatchRules::evaluateAtTick(board, t).over);   // still level

    contested->takeDamage(100);                                  // team 0 takes it
    auto sudden = MatchRules::evaluateAtTick(board, t + 1);
    REQUIRE(sudden.over);
    REQUIRE(sudden.loserTeam == 1);
}

TEST_CASE("a King death still ends the match instantly in overtime", "[match][overtime]") {
    // The King rule outranks the crown rule: evaluateAtTick asks it FIRST, so a
    // side that is behind on crowns and then wins by King KO still wins.
    Board board;
    spawnKing(board, 1, 0, true);
    spawnKing(board, 2, 1, false);          // team 1's King is gone
    spawnPrincess(board, 3, 0, false);
    spawnPrincess(board, 4, 1, true);       // ...while team 0 is DOWN a crown

    auto o = MatchRules::evaluateAtTick(board, MatchRules::REGULATION_END_TICK + 50);
    REQUIRE(o.over);
    REQUIRE(o.loserTeam == 1);
}

TEST_CASE("the census counts towers only, not deployed buildings", "[match][overtime]") {
    // A Cannon is not a crown. If it counted, placing one would win an overtime
    // that was actually level -- the same class of error as the Mortar/'R'
    // clash above, arriving through the new rule instead of the old one.
    Board board;
    spawnKing(board, 1, 0, true);
    spawnKing(board, 2, 1, true);
    spawnPrincess(board, 3, 0, true);
    spawnPrincess(board, 4, 1, true);
    CardRegistry::getInstance().getCard(25)->spawnEntity(9.0f, 8.0f, 0, board);  // Cannon
    board.commitPendingEntities();

    auto c = MatchRules::census(board);
    REQUIRE(c.alive[0] == 2);
    REQUIRE(c.alive[1] == 2);
    REQUIRE_FALSE(MatchRules::evaluateAtTick(board, MatchRules::REGULATION_END_TICK).over);
}
