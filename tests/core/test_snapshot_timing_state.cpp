#include <catch_amalgamated.hpp>
#include "Board.h"
#include "Building.h"
#include "MeleeTroop.h"
#include <vector>

// Internal timing state across Board::deepCopy(), tested behaviourally.
//
// test_board_deepcopy.cpp compares observable fields (id, hp, team, position),
// so a copy diverging only in timing (a reset cooldown, a rewound decay
// schedule) could pass it. Search exists to predict the next second or two,
// which is exactly where such a rollout goes wrong. The fields are not public,
// but the seams below make them observable; only ticksOnTarget got an accessor,
// because its behavioural proxy is lossy.

namespace {

// Board-level step order, as in test_board_deepcopy.cpp: deepCopy needs no
// GameManager.
void tickBoard(Board& board, int tick) {
    board.currentTick = tick;
    board.commitPendingEntities(tick);
    for (const auto& entity : board.getEntities()) {
        if (entity->isAlive()) entity->update(board);
    }
    board.pendingElixirGrant[0] = 0.0f;
    board.pendingElixirGrant[1] = 0.0f;
    board.commitPendingEntities(tick);
    board.resolveCollisions();
    board.cleanDeadEntities(tick);
}

std::vector<int> hpTrace(Board& board, int fromTick, int ticks) {
    std::vector<int> out;
    for (int i = 0; i < ticks; ++i) {
        tickBoard(board, fromTick + i);
        int hp = -1;
        for (const auto& e : board.getEntities()) {
            if (e->id == 1) { hp = e->hp; break; }
        }
        out.push_back(hp);
    }
    return out;
}

} // namespace

TEST_CASE("a copied Building keeps its position in the decay schedule, not just its hp",
          "[deepcopy][timing]") {
    // Buildings decay when ticksAlive % 10 == 0. Copied on a multiple of 10, a
    // copy with ticksAlive reset to 0 would decay on the same ticks, so the
    // copy is taken at step 7.
    Board original;
    auto cannon = std::make_shared<Building>(1, 9.0f, 8.0f, 824, 0, 'C',
                                             5.5f, 202, 10);   // lifetime 300, the default
    original.addEntity(cannon);
    original.commitPendingEntities(0);
    for (int t = 0; t < 7; ++t) tickBoard(original, t);

    Board copy = original.deepCopy();

    // 40 ticks: four decay events, enough to show a one-off or a persistent
    // phase error.
    std::vector<int> a = hpTrace(original, 7, 40);
    std::vector<int> b = hpTrace(copy, 7, 40);

    REQUIRE(a == b);

    // The trace must contain decay events, or two flat lines trivially match.
    REQUIRE(a.front() > a.back());
    int drops = 0;
    for (size_t i = 1; i < a.size(); ++i) if (a[i] < a[i - 1]) ++drops;
    REQUIRE(drops >= 3);
}

TEST_CASE("a copied Building's decay does not affect the original's",
          "[deepcopy][timing]") {
    Board original;
    original.addEntity(std::make_shared<Building>(1, 9.0f, 8.0f, 824, 0, 'C',
                                                  5.5f, 202, 10));
    original.commitPendingEntities(0);
    for (int t = 0; t < 7; ++t) tickBoard(original, t);

    int hpBefore = -1;
    for (const auto& e : original.getEntities()) if (e->id == 1) hpBefore = e->hp;

    Board copy = original.deepCopy();
    hpTrace(copy, 7, 60);   // decay the copy past several intervals

    int hpAfter = -1;
    for (const auto& e : original.getEntities()) if (e->id == 1) hpAfter = e->hp;

    // Shared state would decay the live match's Cannon from a rollout.
    REQUIRE(hpAfter == hpBefore);
}

TEST_CASE("a copied attacker re-arms on the same tick as the original",
          "[deepcopy][timing]") {
    // currentCooldown is protected; seedCooldown() is the public seam. A
    // deliberately odd cooldown makes a zeroed or rounded copy land its first
    // hit on a different tick.
    Board original;
    // (id, x, y, hp, team, speed, attackRange, damage, attackCooldown, symbol).
    // Speed 0 so movement adds no drift.
    auto attacker = std::make_shared<MeleeTroop>(1, 9.0f, 9.0f, 2000, 0,
                                                 0.0f, 1.2f, 100, 12, 'K');
    auto target = std::make_shared<MeleeTroop>(2, 9.0f, 9.5f, 5000, 1,
                                               0.0f, 1.2f, 0, 12, 'T');
    attacker->seedCooldown(7);
    original.addEntity(attacker);
    original.addEntity(target);
    original.commitPendingEntities(0);

    Board copy = original.deepCopy();

    // Trace the target's hp: its first drop marks when the cooldown reached
    // zero.
    auto traceTarget = [](Board& b) {
        std::vector<int> out;
        for (int t = 0; t < 30; ++t) {
            tickBoard(b, t);
            for (const auto& e : b.getEntities()) {
                if (e->id == 2) { out.push_back(e->hp); break; }
            }
        }
        return out;
    };

    std::vector<int> a = traceTarget(original);
    std::vector<int> b = traceTarget(copy);

    REQUIRE(a == b);

    // Non-vacuous: hits landed in the window.
    REQUIRE(a.front() > a.back());
}

TEST_CASE("a copied attacker keeps its position in the damage ramp, not just its cooldown",
          "[deepcopy][timing]") {
    // ticksOnTarget's only behavioural proxy, getDamagePerTick(), collapses it
    // into ramp buckets, so a within-bucket rewind is invisible to an hp trace.
    // It also drives the hit-speed ramp, so a desync changes when later hits
    // land.
    Board original;
    // Speed 0, as in the re-arm test.
    auto attacker = std::make_shared<MeleeTroop>(1, 9.0f, 9.0f, 2000, 0,
                                                 0.0f, 1.2f, 100, 12, 'K');
    auto target = std::make_shared<MeleeTroop>(2, 9.0f, 9.5f, 5000, 1,
                                               0.0f, 1.2f, 0, 12, 'T');
    original.addEntity(attacker);
    original.addEntity(target);
    original.commitPendingEntities(0);

    // A deliberately non-round ticksOnTarget, so a reset or rounded copy is
    // caught.
    for (int t = 0; t < 17; ++t) tickBoard(original, t);

    Board copy = original.deepCopy();

    auto ticksOnTargetOf = [](Board& b) {
        for (const auto& e : b.getEntities()) {
            if (e->id == 1) {
                auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
                REQUIRE(ce != nullptr);
                return ce->getTicksOnTarget();
            }
        }
        return -1;
    };

    // Non-vacuous: the attacker is engaged, or both sides read 0.
    REQUIRE(ticksOnTargetOf(original) > 0);
    REQUIRE(ticksOnTargetOf(copy) == ticksOnTargetOf(original));
}
