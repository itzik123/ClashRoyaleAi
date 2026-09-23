#include <catch_amalgamated.hpp>
#include "TimeoutRules.h"
#include "ClashEnv.h"
#include "GameManager.h"
#include "Tower.h"
#include "GameLogger.h"
#include <string>
#include <vector>

// TimeoutRules, and the ClashEnv accessor that exposes it to Python.
//
// The rules, in order:
//   1. fewer surviving towers loses
//   2. on equal counts, the lower weakest surviving tower loses
//   3. only an exact tie on both is a draw
//
// loserTeam is -1 for a draw, otherwise the team that lost.

namespace {

// TimeoutRules reads only team and hp from towers, so build the minimum that
// makes each rule legible.
void addTower(Board& board, int id, int team, int hp) {
    auto t = std::make_shared<Tower>(id, 9.0f, team == 0 ? 6.0f : 27.0f,
                                     hp, team, 7.0f, 90, 10, 'P');
    board.addEntity(t);
}

Board boardWith(const std::vector<int>& team0Hp, const std::vector<int>& team1Hp) {
    Board board;
    int id = 1;
    for (int hp : team0Hp) addTower(board, id++, 0, hp);
    for (int hp : team1Hp) addTower(board, id++, 1, hp);
    board.commitPendingEntities(0);
    return board;
}

} // namespace

TEST_CASE("TimeoutRules rule 1: fewer surviving towers loses", "[timeout_rules]") {
    // Team 1 has one tower fewer.
    Board b = boardWith({ 100, 100, 100 }, { 100, 100 });
    MatchRules::Outcome o = TimeoutRules::resolve(b);
    REQUIRE(o.over);
    REQUIRE(o.loserTeam == 1);

    Board b2 = boardWith({ 100, 100 }, { 100, 100, 100 });
    REQUIRE(TimeoutRules::resolve(b2).loserTeam == 0);
}

TEST_CASE("TimeoutRules rule 1 outranks rule 2: more towers wins on far worse HP",
          "[timeout_rules]") {
    // Team 0 is nearly dead on every tower but holds one more. Count is checked
    // first, so team 0 wins.
    Board b = boardWith({ 1, 1, 1 }, { 4000, 4000 });
    REQUIRE(TimeoutRules::resolve(b).loserTeam == 1);
}

TEST_CASE("TimeoutRules rule 2: on equal counts the lower weakest tower loses",
          "[timeout_rules]") {
    // Equal counts; team 1's weakest (90) is below team 0's (1200), so team 1
    // loses.
    Board b = boardWith({ 4000, 2000, 1200 }, { 4000, 2000, 90 });
    MatchRules::Outcome o = TimeoutRules::resolve(b);
    REQUIRE(o.over);
    REQUIRE(o.loserTeam == 1);

    Board b2 = boardWith({ 4000, 2000, 90 }, { 4000, 2000, 1200 });
    REQUIRE(TimeoutRules::resolve(b2).loserTeam == 0);
}

TEST_CASE("TimeoutRules rule 2 compares the WEAKEST tower, not the total",
          "[timeout_rules]") {
    // Team 0 has more total HP (7000 vs 3300) but the lower weakest tower:
    // separates "weakest" from "total".
    Board b = boardWith({ 3500, 3400, 100 }, { 1100, 1100, 1100 });
    REQUIRE(TimeoutRules::resolve(b).loserTeam == 0);
}

TEST_CASE("TimeoutRules rule 3: only an exact tie is a draw", "[timeout_rules]") {
    Board b = boardWith({ 4000, 2000, 1200 }, { 4000, 2000, 1200 });
    MatchRules::Outcome o = TimeoutRules::resolve(b);
    REQUIRE(o.over);
    REQUIRE(o.loserTeam == -1);

    // Same count and weakest, different elsewhere: still a draw.
    Board b2 = boardWith({ 4000, 4000, 500 }, { 900, 800, 500 });
    REQUIRE(TimeoutRules::resolve(b2).loserTeam == -1);
}

TEST_CASE("TimeoutRules counts only Towers, not player-placed buildings",
          "[timeout_rules]") {
    // A Cannon is a Building, not a Tower, and is not a crown.
    Board b = boardWith({ 100, 100 }, { 100, 100 });
    auto cannon = std::make_shared<Building>(99, 5.0f, 8.0f, 824, 0, 'C',
                                             5.5f, 202, 10);
    b.addEntity(cannon);
    b.commitPendingEntities(0);
    // Still an exact tie; if the Cannon counted, team 0 would win on count.
    REQUIRE(TimeoutRules::resolve(b).loserTeam == -1);
}

TEST_CASE("TimeoutRules ignores dead towers", "[timeout_rules]") {
    Board b = boardWith({ 100, 100, 100 }, { 100, 100, 100 });
    for (const auto& e : b.getEntities()) {
        if (e->team == 1) { e->takeDamage(1000); break; }   // kill one of team 1's
    }
    // Team 1 is now down to two live towers, so it loses on count.
    REQUIRE(TimeoutRules::resolve(b).loserTeam == 1);
}

// ---------------- the ClashEnv accessor ----------------

TEST_CASE("ClashEnv::resolveTimeoutOutcome returns TimeoutRules' own verdict",
          "[clash_env][timeout_rules]") {
    std::vector<int> deck = { 0, 1, 2, 3, 4, 5, 6, 7 };
    ClashEnv env(deck, deck, 100);
    env.reset();

    // A fresh board is six full-HP towers, symmetric -- an exact tie.
    REQUIRE(env.resolveTimeoutOutcome() == -1);

    // Wound one of team 1's Princesses: counts stay equal, so rule 2 decides
    // against team 1.
    for (const auto& e : env.debugGame().getBoard().getEntities()) {
        if (e->team == 1 && dynamic_cast<Tower*>(e.get()) != nullptr
            && e->cardId == GameManager::TOWER_PRINCESS_ID) {
            e->takeDamage(e->hp - 50);
            break;
        }
    }
    REQUIRE(env.getTowersAlive(0) == env.getTowersAlive(1));   // counts still tied
    REQUIRE(env.resolveTimeoutOutcome() == 1);                 // but team 1 is behind
}

// --- the verdict a replay reports ---
// GameLogger writes the engine's own verdict into the replay, and the viewer
// reads it. These pin that the written verdict is the engine's.

TEST_CASE("GameLogger writes a decisive verdict for a timed-out match on tower count",
          "[timeout][replay]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    Board& board = game.getBoard();
    // Team 1 loses a Princess with both Kings alive: a timeout decided on
    // count.
    for (const auto& e : board.getEntities()) {
        if (e->isTower() && e->team == 1 && e->symbol != 'R') { e->takeDamage(e->hp); break; }
    }
    board.cleanDeadEntities(0);

    GameLogger logger;
    logger.logTick(1, game);
    const std::string json = logger.resultJson();

    INFO(json);
    REQUIRE(json.find("\"loserTeam\": 1") != std::string::npos);
    REQUIRE(json.find("\"timedOut\": true") != std::string::npos);
    REQUIRE(json.find("surviving tower count") != std::string::npos);
}

TEST_CASE("GameLogger applies the weakest-tower tie-break, not king-alive",
          "[timeout][replay]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    Board& board = game.getBoard();
    // Equal counts, both Kings up; team 0's Princess is nearly dead, so team 0
    // loses.
    for (const auto& e : board.getEntities()) {
        if (e->isTower() && e->team == 0 && e->symbol != 'R') { e->takeDamage(e->hp - 90); break; }
    }

    GameLogger logger;
    logger.logTick(1, game);
    const std::string json = logger.resultJson();

    INFO(json);
    REQUIRE(json.find("\"loserTeam\": 0") != std::string::npos);
    REQUIRE(json.find("weakest tower") != std::string::npos);
    // The engine agrees with the replay: one rule, two data sources.
    REQUIRE(TimeoutRules::resolve(board).loserTeam == 0);
}

TEST_CASE("GameLogger reports a King KO as a King KO, not a timeout",
          "[timeout][replay]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    Board& board = game.getBoard();
    for (const auto& e : board.getEntities()) {
        if (e->isTower() && e->team == 1 && e->symbol == 'R') { e->takeDamage(e->hp); break; }
    }
    board.cleanDeadEntities(0);

    GameLogger logger;
    logger.logTick(1, game);
    const std::string json = logger.resultJson();

    INFO(json);
    REQUIRE(json.find("\"timedOut\": false") != std::string::npos);
    REQUIRE(json.find("\"loserTeam\": 1") != std::string::npos);
    REQUIRE(json.find("King Tower destroyed") != std::string::npos);
}

TEST_CASE("an untouched match is the one genuine draw", "[timeout][replay]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    GameLogger logger;
    logger.logTick(1, game);
    const std::string json = logger.resultJson();

    INFO(json);
    REQUIRE(json.find("\"loserTeam\": -1") != std::string::npos);
    REQUIRE(json.find("exact tie") != std::string::npos);
}

// decide() and resolve() must never disagree.
TEST_CASE("TimeoutRules::decide is the rule resolve() applies", "[timeout][replay]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    Board& board = game.getBoard();
    for (const auto& e : board.getEntities()) {
        if (e->isTower() && e->team == 1 && e->symbol != 'R') { e->takeDamage(e->hp - 5); break; }
    }
    int counts[2] = { 3, 3 };
    int weakest[2] = { 2534, 5 };
    REQUIRE(TimeoutRules::decide(counts, weakest).loserTeam
            == TimeoutRules::resolve(board).loserTeam);
}
