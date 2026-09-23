#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "GameManager.h"
#include "HeuristicOpponent.h"
#include "CardRegistry.h"
#include <random>
#include <vector>

// The built-in C++ opponent: phase 1's ClashEnv::step() opponent and the three
// builtin Elo anchors, so what it treats as a threat decides what those anchors
// measure.

namespace {

constexpr int HOG_RIDER = 15, MUSKETEER = 6, CANNON = 25, ICE_GOLEM = 40;
constexpr int SKELETONS = 24, ICE_SPIRIT = 72, THE_LOG = 33, FIREBALL = 7;

const std::vector<int>& deck() {
    static const std::vector<int> d = { HOG_RIDER, MUSKETEER, CANNON, ICE_GOLEM,
                                        SKELETONS, ICE_SPIRIT, THE_LOG, FIREBALL };
    return d;
}

// Did one act() call commit elixir? More direct than diffing the hand, and a
// cycle cannot fool it.
bool actsOn(int intruderCardId) {
    GameManager game(deck(), deck());
    game.reset();
    std::mt19937 rng(12345);
    HeuristicOpponent bot;
    bot.reset(rng);

    // Deep in team 1's half, where a real incursion demands an answer.
    CardRegistry::getInstance().getCard(intruderCardId)->spawnEntity(9.0f, 25.0f, 0, game.getBoard());
    game.getBoard().commitPendingEntities();

    const float before = game.getElixir(1);
    bot.act(game, rng);
    return game.getElixir(1) < before;
}

} // namespace

TEST_CASE("the built-in opponent answers a real incursion", "[heuristic]") {
    // The control for the case below: this bot holds elixir below 7 by design,
    // so "did not react to a spell" would also pass for a bot that never
    // reacts.
    REQUIRE(actsOn(HOG_RIDER));
}

TEST_CASE("the built-in opponent does not defend against a spell marker",
          "[heuristic][regression]") {
    // An AreaSpell sits on the board at its impact point for its whole fuse (hp
    // 1), and a Projectile while in flight; a threat scan must skip both, via
    // isTargetable(), or a Fireball at the bot's tower buys a full defensive
    // placement.
    REQUIRE_FALSE(actsOn(FIREBALL));
}
