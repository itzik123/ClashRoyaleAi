#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "GameManager.h"
#include "HeuristicOpponent.h"
#include "CardRegistry.h"
#include <random>
#include <vector>

// ============================================================================
// The built-in C++ opponent. It is phase 1's ClashEnv::step() opponent and all
// three BUILTIN_ANCHORS in the phase-2 Elo roster, so what it believes is a
// threat decides what those anchors measure.
//
// It had no coverage at all before this file.
// ============================================================================

namespace {

constexpr int HOG_RIDER = 15, MUSKETEER = 6, CANNON = 25, ICE_GOLEM = 40;
constexpr int SKELETONS = 24, ICE_SPIRIT = 72, THE_LOG = 33, FIREBALL = 7;

const std::vector<int>& deck() {
    static const std::vector<int> d = { HOG_RIDER, MUSKETEER, CANNON, ICE_GOLEM,
                                        SKELETONS, ICE_SPIRIT, THE_LOG, FIREBALL };
    return d;
}

// Did one act() call actually commit elixir? Cheaper and more direct than
// diffing the hand, and it cannot be fooled by a cycle.
bool actsOn(int intruderCardId) {
    GameManager game(deck(), deck());
    game.reset();
    std::mt19937 rng(12345);
    HeuristicOpponent bot;
    bot.reset(rng);

    // Deep in team 1's half, well past the river, where a real incursion would
    // demand an answer.
    CardRegistry::getInstance().getCard(intruderCardId)->spawnEntity(9.0f, 25.0f, 0, game.getBoard());
    game.getBoard().commitPendingEntities();

    const float before = game.getElixir(1);
    bot.act(game, rng);
    return game.getElixir(1) < before;
}

} // namespace

TEST_CASE("the built-in opponent answers a real incursion", "[heuristic]") {
    // The control for the case below. Without it, "did not react to a spell"
    // is satisfied by a bot that never reacts to anything -- and this bot
    // holds elixir below 7 by design, so that failure mode is live.
    REQUIRE(actsOn(HOG_RIDER));
}

TEST_CASE("the built-in opponent does not defend against a spell marker",
          "[heuristic][regression]") {
    // Its threat scan walked every living enemy entity and excluded only
    // Towers. An AreaSpell is a living entity (hp 1) parked at its impact
    // point for the length of its fuse, and a Projectile likewise -- so a
    // Fireball thrown at the opponent's own tower registered as the deepest
    // incursion on the board and bought a full defensive placement against
    // something that was never a unit. It is also the strongest affordable
    // card it holds, so the misread is expensive.
    //
    // isTargetable() is the discriminator the rest of the engine already uses
    // for exactly this ("Projectiles and pending spells are not board
    // presence" -- ClashEnv::extractObservationForTeam).
    REQUIRE_FALSE(actsOn(FIREBALL));
}
