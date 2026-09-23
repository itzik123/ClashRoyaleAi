#include <catch_amalgamated.hpp>
#include "GameManager.h"
#include "GameLogger.h"
#include "MeleeTroop.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

// Naming spawn-children in a replay. They carry negative cardIds (Golemite is
// -1), which cardMeta cannot resolve, and many symbols are shared by several
// cards. The entity already has the right name (applyCardMetadata sets it), so
// GameLogger writes it.

namespace {

std::string readFile(const std::string& path) {
    std::ifstream in(path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Extract the JSON object starting at `needle`, so an assertion cannot be
// satisfied by cardMeta or another row.
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

    // A death-spawn child, as a dying Golem leaves: its negative cardId is what
    // cardMeta cannot look up.
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

    // Non-vacuous: the entity's own row was found.
    INFO("entity row: " << row);
    REQUIRE(!row.empty());
    REQUIRE(row.find("\"cardId\":-1") != std::string::npos);

    // The row carries the name, so the viewer need not guess from a shared
    // symbol.
    REQUIRE(row.find("\"name\":\"Golemite\"") != std::string::npos);
}

TEST_CASE("a replay still names an ordinary card played from hand",
          "[replay][logger][names]") {
    // The name is written for every entity, not only the ones cardMeta misses.
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
    // Names go through the same escaping as symbols (Ram Rider's symbol is
    // '"').
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

    // Both metacharacters arrive escaped, with no bare ones left.
    REQUIRE(json.find("Say \\\"hi\\\"\\\\now") != std::string::npos);
}
