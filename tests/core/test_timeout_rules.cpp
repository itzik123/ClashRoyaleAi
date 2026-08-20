#include <catch_amalgamated.hpp>
#include "TimeoutRules.h"
#include "ClashEnv.h"
#include "GameManager.h"
#include "Tower.h"
#include <vector>

// TimeoutRules, and the ClashEnv accessor that exposes it to Python.
//
// WHY THIS FILE EXISTS
// --------------------
// TimeoutRules had NO tests at all before this, despite being what decides
// every match that reaches the tick limit with both Kings standing. Its own
// header records why it was added: until it existed every timeout scored 0.0,
// teaching the agent that running the clock out was free.
//
// It also had exactly one C++ call site (ClashEnv::calculateReward) and no
// binding, so Python could not ask the engine who won. Every evaluation script
// that wanted a verdict without going through `reward` re-derived one from
// get_towers_alive() and reproduced only the FIRST of the three rules --
// eight times, across three waves, before resolveTimeoutOutcome existed.
// See perception/UPSTREAM_REQUESTS.md item 16.
//
// The rules, in order:
//   1. fewer surviving towers loses
//   2. on equal counts, the lower weakest surviving tower loses
//   3. only an exact tie on both is a draw
//
// loserTeam is -1 for a draw, otherwise the team that LOST.

namespace {

// Tower ids/positions do not matter to TimeoutRules -- it filters on
// dynamic_cast<Tower*> and reads team + hp -- so build the minimum that makes
// each rule's intent legible.
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
    // Team 0 is nearly dead on every tower but holds one more of them. Count
    // is checked first, so team 0 still wins -- if the order were reversed
    // this would come out the other way, which is what makes it worth pinning.
    Board b = boardWith({ 1, 1, 1 }, { 4000, 4000 });
    REQUIRE(TimeoutRules::resolve(b).loserTeam == 1);
}

TEST_CASE("TimeoutRules rule 2: on equal counts the lower weakest tower loses",
          "[timeout_rules]") {
    // Equal counts. Team 1's weakest (90) is below team 0's weakest (1200),
    // so team 1 loses. This is the rule all five original Python scorers
    // dropped, and this exact shape -- 3v3 towers, 1200 against 90 -- is the
    // case they each reported as a draw.
    Board b = boardWith({ 4000, 2000, 1200 }, { 4000, 2000, 90 });
    MatchRules::Outcome o = TimeoutRules::resolve(b);
    REQUIRE(o.over);
    REQUIRE(o.loserTeam == 1);

    Board b2 = boardWith({ 4000, 2000, 90 }, { 4000, 2000, 1200 });
    REQUIRE(TimeoutRules::resolve(b2).loserTeam == 0);
}

TEST_CASE("TimeoutRules rule 2 compares the WEAKEST tower, not the total",
          "[timeout_rules]") {
    // Team 0 has far more total HP (7000 vs 3300) but the lower weakest tower.
    // A sum-based tie-break would give the opposite answer, so this separates
    // "weakest" from "total" rather than letting both readings pass.
    Board b = boardWith({ 3500, 3400, 100 }, { 1100, 1100, 1100 });
    REQUIRE(TimeoutRules::resolve(b).loserTeam == 0);
}

TEST_CASE("TimeoutRules rule 3: only an exact tie is a draw", "[timeout_rules]") {
    Board b = boardWith({ 4000, 2000, 1200 }, { 4000, 2000, 1200 });
    MatchRules::Outcome o = TimeoutRules::resolve(b);
    REQUIRE(o.over);
    REQUIRE(o.loserTeam == -1);

    // Same count AND same weakest, but different elsewhere -- still a draw,
    // because nothing past the weakest tower is consulted.
    Board b2 = boardWith({ 4000, 4000, 500 }, { 900, 800, 500 });
    REQUIRE(TimeoutRules::resolve(b2).loserTeam == -1);
}

TEST_CASE("TimeoutRules counts only Towers, not player-placed buildings",
          "[timeout_rules]") {
    // A Cannon is a Building, not a Tower, and must not count as a crown.
    // TimeoutRules uses dynamic_cast rather than a symbol check precisely so
    // this keeps holding if tower symbols change for the renderer.
    Board b = boardWith({ 100, 100 }, { 100, 100 });
    auto cannon = std::make_shared<Building>(99, 5.0f, 8.0f, 824, 0, 'C',
                                             5.5f, 202, 10);
    b.addEntity(cannon);
    b.commitPendingEntities(0);
    // Still an exact tie: the Cannon changed neither the count nor the weakest
    // TOWER. If it counted, team 0 would win on count and this would be 1.
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

    // Wound one of team 1's Princesses. Counts stay equal (nothing died), so
    // rule 2 decides and team 1 must lose. Under the old tower-count-only
    // Python scorers this same position read as a DRAW, which is the entire
    // reason this accessor exists.
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
