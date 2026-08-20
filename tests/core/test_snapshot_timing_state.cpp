#include <catch_amalgamated.hpp>
#include "Board.h"
#include "Building.h"
#include "MeleeTroop.h"
#include <vector>

// INTERNAL TIMING STATE across Board::deepCopy(), tested BEHAVIOURALLY.
//
// WHY THIS FILE EXISTS
// --------------------
// test_board_deepcopy.cpp's 120-tick acceptance test compares only externally
// observable fields -- id, hp, team, cardId, x, y, projectileTargetId. A copy
// that diverged ONLY in internal timing (a reset attack cooldown, a rewound
// decay schedule) passes it until the difference happens to surface as an hp or
// position change, which may be well outside the window.
//
// That is the defect class that matters most here: decision-time search exists
// to predict the next second or two accurately, and a rollout whose buildings
// decay on the wrong tick or whose troops re-arm at the wrong moment is wrong
// in exactly the way that is hardest to notice -- the numbers still look
// plausible.
//
// None of the relevant fields are publicly readable (currentCooldown is
// protected; Building::ticksAlive is private), and asking for accessors is
// filed as perception/UPSTREAM_REQUESTS.md item 17 -- deliberately as
// ergonomics, NOT as a coverage unblocker, because the seams below already make
// the state observable through behaviour. These tests are that argument, made
// concrete.

namespace {

// Same board-level step order as test_board_deepcopy.cpp's tickBoard, for the
// same reason: deepCopy is a Board operation and needs no GameManager.
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
    // Building decays on `ticksAlive % 10 == 0` (Building::update). The phase
    // of that counter is the thing under test, and it is INVISIBLE if you copy
    // on a multiple of 10: a copy that reset ticksAlive to 0 would still be
    // congruent mod 10 and would decay on exactly the same ticks.
    //
    // So step 7 -- deliberately NOT a multiple of 10. A reset copy would then
    // decay 3 ticks later than the original every single time.
    Board original;
    auto cannon = std::make_shared<Building>(1, 9.0f, 8.0f, 824, 0, 'C',
                                             5.5f, 202, 10);   // lifetime 300 (default)
    original.addEntity(cannon);
    original.commitPendingEntities(0);
    for (int t = 0; t < 7; ++t) tickBoard(original, t);

    Board copy = original.deepCopy();

    // 40 ticks is four decay events -- enough that a one-off phase error and a
    // persistent one both show up.
    std::vector<int> a = hpTrace(original, 7, 40);
    std::vector<int> b = hpTrace(copy, 7, 40);

    REQUIRE(a == b);

    // The trace must actually CONTAIN decay events, otherwise the comparison
    // above is vacuous: two flat lines are trivially equal.
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
    hpTrace(copy, 7, 60);   // decay the COPY well past several intervals

    int hpAfter = -1;
    for (const auto& e : original.getEntities()) if (e->id == 1) hpAfter = e->hp;

    // Shared state here would mean a search rollout decays the live match's
    // Cannon -- it would simply die early, mid-defence, for no visible reason.
    REQUIRE(hpAfter == hpBefore);
}

TEST_CASE("a copied attacker re-arms on the same tick as the original",
          "[deepcopy][timing]") {
    // CombatEntity::currentCooldown is protected, but seedCooldown() is public
    // and its own comment calls it "the one seam" for data-driven setup. Seed a
    // known, deliberately ODD cooldown so a copy that zeroed or rounded it
    // lands its first hit on a different tick.
    Board original;
    // (id, x, y, hp, team, speed, attackRange, damage, attackCooldown, symbol)
    // Speed 0 on both so neither walks: this test is about WHEN the attacker
    // swings, and movement would add a second, confounding source of drift.
    auto attacker = std::make_shared<MeleeTroop>(1, 9.0f, 9.0f, 2000, 0,
                                                 0.0f, 1.2f, 100, 12, 'K');
    auto target = std::make_shared<MeleeTroop>(2, 9.0f, 9.5f, 5000, 1,
                                               0.0f, 1.2f, 0, 12, 'T');
    attacker->seedCooldown(7);
    original.addEntity(attacker);
    original.addEntity(target);
    original.commitPendingEntities(0);

    Board copy = original.deepCopy();

    // Trace the TARGET's hp: the tick it first drops is the tick the attacker's
    // cooldown reached zero, which is the quantity under test.
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

    // Non-vacuous: the attacker must actually have landed hits in the window.
    REQUIRE(a.front() > a.back());
}
