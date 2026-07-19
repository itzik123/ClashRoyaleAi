#include <catch_amalgamated.hpp>
#include "MatchStatistics.h"

TEST_CASE("MatchStatistics is safely queryable before attach() (unattached defaults)", "[match_statistics]") {
    MatchStatistics stats;
    REQUIRE(stats.totalDamageDealt(0) == 0);
    REQUIRE(stats.damageDealtByCard(5, 0) == 0);
    REQUIRE(stats.kills(0) == 0);
    REQUIRE(stats.elixirSpent(0) == Catch::Approx(0.0f));
    REQUIRE(stats.cardsPlayed(0).empty());
    REQUIRE(stats.loserTeam() == -1);
    REQUIRE(stats.matchDurationTicks() == 0);
}

TEST_CASE("MatchStatistics accumulates damage totals per team and per attacker card", "[match_statistics]") {
    Board board;
    MatchStatistics stats;
    stats.attach(board);

    board.statsEvents.notifyDamageDealt({ 1, 0, 5, 10, 20, 1, 100, 1 }); // team0/card5 hits team1's entity 10
    board.statsEvents.notifyDamageDealt({ 1, 0, 5, 11, 20, 1, 50, 2 });  // team0/card5 hits again
    board.statsEvents.notifyDamageDealt({ 2, 1, 6, 10, 20, 0, 30, 1 });  // team1/card6 hits team0's entity 10

    REQUIRE(stats.totalDamageDealt(0) == 150);
    REQUIRE(stats.damageDealtByCard(5, 0) == 150);
    REQUIRE(stats.totalDamageDealt(1) == 30);
    REQUIRE(stats.damageDealtByCard(6, 1) == 30);
    REQUIRE(stats.damageDealtByCard(999, 0) == 0); // never-dealt card: reports 0, not a crash
}

TEST_CASE("MatchStatistics splits damage dealt into troop vs building, per attacking team", "[match_statistics][damage_by_target_type]") {
    Board board;
    MatchStatistics stats;
    stats.attach(board);

    // team0/card5 (Knight, id 0 -- a troop) hits team1's Knight (targetCardId 0)
    board.statsEvents.notifyDamageDealt({ 1, 0, 5, 10, 0, 1, 100, 1 });
    // team0/card5 hits team1's Cannon (targetCardId 25 -- a building)
    board.statsEvents.notifyDamageDealt({ 1, 0, 5, 11, 25, 1, 60, 2 });
    // team1/card6 hits team0's King Tower (targetCardId -2, GameManager::TOWER_KING_ID --
    // not in CardRegistry at all, must still be classified as a building)
    board.statsEvents.notifyDamageDealt({ 2, 1, 6, 12, -2, 0, 40, 3 });

    REQUIRE(stats.troopDamageDealt(0) == 100);
    REQUIRE(stats.buildingDamageDealt(0) == 60);
    REQUIRE(stats.troopDamageDealt(1) == 0);
    REQUIRE(stats.buildingDamageDealt(1) == 40);
}

TEST_CASE("MatchStatistics ignores same-team damage when splitting troop vs building", "[match_statistics][damage_by_target_type]") {
    Board board;
    MatchStatistics stats;
    stats.attach(board);

    // attackerTeam == targetTeam: e.g. some future splash hitting your own troop.
    board.statsEvents.notifyDamageDealt({ 1, 0, 5, 10, 0, 0, 999, 1 });

    REQUIRE(stats.troopDamageDealt(0) == 0);
    REQUIRE(stats.buildingDamageDealt(0) == 0);
}

TEST_CASE("MatchStatistics::toJson includes the troop/building damage breakdown", "[match_statistics][damage_by_target_type][json]") {
    Board board;
    MatchStatistics stats;
    stats.attach(board);

    board.statsEvents.notifyDamageDealt({ 1, 0, 5, 10, 0, 1, 100, 1 });  // troop damage
    board.statsEvents.notifyDamageDealt({ 1, 0, 5, 11, 25, 1, 60, 2 }); // building damage

    std::string json = stats.toJson();
    REQUIRE(json.find("\"troopDamageDealt\":{\"team0\":100,\"team1\":0}") != std::string::npos);
    REQUIRE(json.find("\"buildingDamageDealt\":{\"team0\":60,\"team1\":0}") != std::string::npos);
}

TEST_CASE("MatchStatistics credits kills to the last attacker's team and card", "[match_statistics]") {
    Board board;
    MatchStatistics stats;
    stats.attach(board);

    board.statsEvents.notifyDamageDealt({ 1, 0, 5, 10, 20, 1, 100, 1 }); // team0/card5 hits entity 10
    board.statsEvents.notifyEntityDied({ 10, 20, 1, 2 });

    REQUIRE(stats.kills(0) == 1);
    REQUIRE(stats.killsByCard(5, 0) == 1);
    REQUIRE(stats.kills(1) == 0);
}

TEST_CASE("MatchStatistics reports no killer for an entity that dies without ever being hit", "[match_statistics]") {
    Board board;
    MatchStatistics stats;
    stats.attach(board);

    board.statsEvents.notifyEntityDied({ 10, 20, 1, 2 }); // no prior DamageDealtEvent for id 10

    REQUIRE(stats.kills(0) == 0);
    REQUIRE(stats.kills(1) == 0);
}

TEST_CASE("MatchStatistics: a building chip-damaged in combat that later dies of unrelated decay "
          "is not wrongly credited to the earlier attacker", "[match_statistics][regression]") {
    Board board;
    MatchStatistics stats;
    stats.attach(board);

    board.statsEvents.notifyDamageDealt({ 1, 0, 5, 10, 20, 1, 50, 1 }); // team0/card5 chip-damages the building (non-lethal)
    board.statsEvents.notifyAttributionCleared({ 10 });                 // Building::update()'s decay branch fires this
    board.statsEvents.notifyEntityDied({ 10, 20, 1, 300 });             // ...and it eventually dies of decay alone

    REQUIRE(stats.kills(0) == 0); // NOT credited to team0/card5
    REQUIRE(stats.kills(1) == 0);
}

TEST_CASE("MatchStatistics accumulates elixir spent per team and per card", "[match_statistics]") {
    Board board;
    MatchStatistics stats;
    stats.attach(board);

    board.statsEvents.notifyCardPlayed({ 0, 5, 3.0f, 9.0f, 10.0f, 1 });
    board.statsEvents.notifyCardPlayed({ 0, 5, 3.0f, 9.0f, 10.0f, 50 });
    board.statsEvents.notifyCardPlayed({ 1, 6, 4.0f, 9.0f, 20.0f, 2 });

    REQUIRE(stats.elixirSpent(0) == Catch::Approx(6.0f));
    REQUIRE(stats.elixirSpentByCard(5, 0) == Catch::Approx(6.0f));
    REQUIRE(stats.elixirSpent(1) == Catch::Approx(4.0f));
}

TEST_CASE("MatchStatistics records an ordered per-team card-play timeline", "[match_statistics]") {
    Board board;
    MatchStatistics stats;
    stats.attach(board);

    board.statsEvents.notifyCardPlayed({ 0, 5, 3.0f, 9.0f, 10.0f, 1 });
    board.statsEvents.notifyCardPlayed({ 0, 6, 4.0f, 3.0f, 5.0f, 20 });

    const auto& timeline = stats.cardsPlayed(0);
    REQUIRE(timeline.size() == 2);
    REQUIRE(timeline[0].cardId == 5);
    REQUIRE(timeline[0].tick == 1);
    REQUIRE(timeline[1].cardId == 6);
    REQUIRE(timeline[1].tick == 20);
    REQUIRE(stats.cardsPlayed(1).empty());
}

TEST_CASE("MatchStatistics reports the match outcome from MatchEndedEvent", "[match_statistics]") {
    Board board;
    MatchStatistics stats;
    stats.attach(board);

    board.statsEvents.notifyMatchEnded({ 0, 1800 });

    REQUIRE(stats.loserTeam() == 0);
    REQUIRE(stats.matchDurationTicks() == 1800);
}

TEST_CASE("MatchStatistics::toJson produces JSON containing the expected keys and values", "[match_statistics][json]") {
    Board board;
    MatchStatistics stats;
    stats.attach(board);

    board.statsEvents.notifyDamageDealt({ 1, 0, 5, 10, 20, 1, 100, 1 });
    board.statsEvents.notifyCardPlayed({ 0, 5, 3.0f, 9.0f, 10.0f, 1 });
    board.statsEvents.notifyMatchEnded({ 1, 50 });

    std::string json = stats.toJson();

    REQUIRE(json.find("\"loserTeam\":1") != std::string::npos);
    REQUIRE(json.find("\"matchDurationTicks\":50") != std::string::npos);
    REQUIRE(json.find("\"totalDamageDealt\"") != std::string::npos);
    REQUIRE(json.find("\"damageDealtByCard\"") != std::string::npos);
    REQUIRE(json.find("\"5\":100") != std::string::npos); // card 5 dealt 100 damage
    REQUIRE(json.find("\"cardsPlayed\"") != std::string::npos);
    REQUIRE(json.find("\"cardId\":5") != std::string::npos);
}

TEST_CASE("MatchStatistics::attach re-subscribes fresh collectors, discarding the previous match's numbers", "[match_statistics]") {
    Board boardA;
    MatchStatistics stats;
    stats.attach(boardA);
    boardA.statsEvents.notifyDamageDealt({ 1, 0, 5, 10, 20, 1, 100, 1 });
    REQUIRE(stats.totalDamageDealt(0) == 100);

    Board boardB; // simulates GameManager::reset()'s `board = Board();`
    stats.attach(boardB);
    REQUIRE(stats.totalDamageDealt(0) == 0); // fresh collectors, old numbers gone

    boardA.statsEvents.notifyDamageDealt({ 1, 0, 5, 10, 20, 1, 999, 2 }); // stale board's bus
    REQUIRE(stats.totalDamageDealt(0) == 0); // unaffected -- no longer subscribed to boardA
}
