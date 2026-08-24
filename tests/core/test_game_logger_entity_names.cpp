#include <catch_amalgamated.hpp>
#include "GameManager.h"
#include "GameLogger.h"
#include "MeleeTroop.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

// NAMING SPAWN-CHILDREN IN A REPLAY.
//
// perception/UPSTREAM_REQUESTS.md item 20 reported this as "spawned entities
// carry no cardId". That diagnosis was wrong, and the correction is the reason
// this file exists rather than a new Entity field:
//
//   * Spawn-children DO carry a cardId. CardRegistry assigns them deliberately
//     NEGATIVE ids -- Golemite is -1, and the comment above golemiteStats()
//     records that the scheme keeps them clear of GameManager's own
//     TOWER_KING_ID/TOWER_PRINCESS_ID (-2/-3).
//   * They also carry the RIGHT NAME already: CardFactories::applyCardMetadata
//     sets entity->name = stats.name, so a Golemite has been called "Golemite"
//     on the board the whole time.
//
// The actual gap was only that GameLogger never WROTE that name, while its
// "cardMeta" block is built from CardRegistry's registered (positive) ids and
// so can never resolve a negative one. The viewer was therefore pushed onto its
// symbol fallback, and 36 of the registry's 90 symbols (40%) are shared by more
// than one card -- so a spawned body could be shown under another card's name
// and HP maximum.
//
// Writing the name closes that without touching Entity, without threading a new
// field through the six spawn sites, and without going near cardId -- which
// stats attribution and therefore reward shaping both read.

namespace {

std::string readFile(const std::string& path) {
    std::ifstream in(path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Extract the single JSON object that starts at `needle`, so an assertion about
// one entity cannot be satisfied by text belonging to cardMeta or another row.
std::string objectContaining(const std::string& json, const std::string& needle) {
    const size_t at = json.find(needle);
    if (at == std::string::npos) return {};
    const size_t open = json.rfind('{', at);
    const size_t close = json.find('}', at);
    if (open == std::string::npos || close == std::string::npos) return {};
    return json.substr(open, close - open + 1);
}

} // namespace

TEST_CASE("a replay names an entity whose negative cardId cardMeta cannot resolve",
          "[replay][logger][names]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    Board& board = game.getBoard();

    // A death-spawn child, standing in for the Golemites a dying Golem leaves:
    // the negative cardId is the part cardMeta cannot look up.
    auto golemite = std::make_shared<MeleeTroop>(9001, 9.0f, 9.0f, 1039, 0,
                                                 0.0f, 0.25f, 84, 25, 'q');
    golemite->name = "Golemite";
    golemite->cardId = -1;
    board.addEntity(golemite);
    board.commitPendingEntities(0);

    GameLogger logger;
    logger.logTick(1, game);

    const std::string path = "test_replay_entity_names.json";
    REQUIRE(logger.save(path));
    const std::string json = readFile(path);
    std::remove(path.c_str());

    const std::string row = objectContaining(json, "\"id\":9001");

    // Non-vacuous: we must really have found the entity's own row.
    INFO("entity row: " << row);
    REQUIRE(!row.empty());
    REQUIRE(row.find("\"cardId\":-1") != std::string::npos);

    // The point of the change: the row carries the name, so the viewer never
    // has to guess from a symbol 40% of the roster shares.
    REQUIRE(row.find("\"name\":\"Golemite\"") != std::string::npos);
}

TEST_CASE("a replay still names an ordinary card played from hand",
          "[replay][logger][names]") {
    // Guard against a fix that only special-cases negative ids: the name must
    // be written for every entity, not just the ones cardMeta misses.
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    Board& board = game.getBoard();

    auto knight = std::make_shared<MeleeTroop>(9002, 9.0f, 9.0f, 1766, 0,
                                               0.0f, 1.2f, 202, 12, 'K');
    knight->name = "Knight";
    knight->cardId = 0;
    board.addEntity(knight);
    board.commitPendingEntities(0);

    GameLogger logger;
    logger.logTick(1, game);

    const std::string path = "test_replay_entity_names_hand.json";
    REQUIRE(logger.save(path));
    const std::string json = readFile(path);
    std::remove(path.c_str());

    const std::string row = objectContaining(json, "\"id\":9002");
    INFO("entity row: " << row);
    REQUIRE(!row.empty());
    REQUIRE(row.find("\"name\":\"Knight\"") != std::string::npos);
}

TEST_CASE("a name containing a JSON metacharacter does not corrupt the replay",
          "[replay][logger][names]") {
    // The symbol field already learned this lesson the hard way -- Ram Rider's
    // symbol is '"', which broke every replay containing one until escapeChar
    // was applied unconditionally. A name is attacker-free but far wider than a
    // char, so it goes through the same escaping rather than trusting the
    // roster to stay quote-free.
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    Board& board = game.getBoard();

    auto odd = std::make_shared<MeleeTroop>(9003, 9.0f, 9.0f, 100, 0,
                                            0.0f, 1.2f, 10, 12, 'z');
    odd->name = "Say \"hi\"\\now";
    odd->cardId = -44;
    board.addEntity(odd);
    board.commitPendingEntities(0);

    GameLogger logger;
    logger.logTick(1, game);

    const std::string path = "test_replay_entity_names_escape.json";
    REQUIRE(logger.save(path));
    const std::string json = readFile(path);
    std::remove(path.c_str());

    // Both metacharacters must arrive escaped, and no bare ones may survive.
    REQUIRE(json.find("Say \\\"hi\\\"\\\\now") != std::string::npos);
}
