#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "Projectile.h"
#include "FreezeOnHit.h"
#include "StatsEventBus.h"

namespace {
    class DamageRecorder : public IStatsObserver {
    public:
        std::vector<DamageDealtEvent> hits;
        void onDamageDealt(const DamageDealtEvent& e) override { hits.push_back(e); }
    };
}

TEST_CASE("Projectile is never itself a valid combat target", "[projectile]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 10.0f, 100, 1);
    spawn(board, target);
    Projectile p(2, 0.0f, 0.0f, 0, target, 1.5f, 50);
    REQUIRE_FALSE(p.isTargetable());
}

TEST_CASE("Projectile homes toward the target's current position each tick", "[projectile][movement]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 10.0f, 1000, 1);
    spawn(board, target);

    Projectile p(2, 0.0f, 0.0f, 0, target, 1.5f, 50);
    p.update(board);

    REQUIRE(p.position.x == Catch::Approx(0.0f));
    REQUIRE(p.position.y == Catch::Approx(1.5f));
}

TEST_CASE("Projectile re-homes when the target moves between ticks", "[projectile][movement]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 10.0f, 0.0f, 1000, 1);
    spawn(board, target);

    Projectile p(2, 0.0f, 0.0f, 0, target, 2.0f, 50);
    p.update(board); // heads toward (10, 0) -> lands at (2, 0)
    REQUIRE(p.position.x == Catch::Approx(2.0f));
    REQUIRE(p.position.y == Catch::Approx(0.0f));

    target->position = { 2.0f, 10.0f }; // target relocates
    p.update(board); // must now head toward the new position, not continue along x

    REQUIRE(p.position.x == Catch::Approx(2.0f));
    REQUIRE(p.position.y == Catch::Approx(2.0f));
}

TEST_CASE("Projectile deals damage and dies the instant it is within one tick's travel of the target", "[projectile][arrival]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 2.0f, 1000, 1);
    spawn(board, target);

    Projectile p(2, 0.0f, 0.0f, 0, target, 2.0f, 275); // dist(2.0) == speed(2.0)
    p.update(board);

    REQUIRE(target->hp == 725);
    REQUIRE_FALSE(p.isAlive());
    // Snaps to the impact point on the tick it lands, instead of dying
    // one step short of the target it just hit.
    REQUIRE(p.position.x == Catch::Approx(0.0f));
    REQUIRE(p.position.y == Catch::Approx(2.0f));
}

TEST_CASE("Projectile dies without dealing damage if its target already died first", "[projectile][arrival]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 100, 1);
    target->takeDamage(100); // dead before the projectile resolves
    spawn(board, target);

    Projectile p(2, 0.0f, 0.0f, 0, target, 5.0f, 50);
    p.update(board);

    REQUIRE_FALSE(p.isAlive());
    REQUIRE(target->hp == 0); // not driven further negative
}

TEST_CASE("Projectile dies without crashing if its target object no longer exists", "[projectile][arrival]") {
    Board board;
    std::weak_ptr<Entity> weakTarget;
    {
        auto target = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 100, 1);
        weakTarget = target;
        // target's shared_ptr goes out of scope here without ever being
        // added to the board, so the weak_ptr can no longer be locked.
    }

    Projectile p(2, 0.0f, 0.0f, 0, weakTarget, 5.0f, 50);
    REQUIRE_NOTHROW(p.update(board));
    REQUIRE_FALSE(p.isAlive());
}

TEST_CASE("Projectile applies on-hit effects on arrival, only when the target is a CombatEntity", "[projectile][arrival][on_hit]") {
    Board board;

    SECTION("CombatEntity target: effect fires") {
        auto target = std::make_shared<StationaryCombatant>(1, 0.0f, 2.0f, 1000, 1, 5.0f, 10, 10);
        spawn(board, target);

        Projectile p(2, 0.0f, 0.0f, 0, target, 2.0f, 50,
            std::vector<std::shared_ptr<IOnHitEffect>>{ std::make_shared<FreezeOnHit>(30, 0.65f) });
        p.update(board);

        REQUIRE(target->freezeTicks == 30);
        REQUIRE(target->freezeSlow == Catch::Approx(0.65f));
    }

    SECTION("plain Entity target: effect is silently skipped, damage still lands") {
        auto target = std::make_shared<DummyEntity>(1, 0.0f, 2.0f, 1000, 1);
        spawn(board, target);

        Projectile p(2, 0.0f, 0.0f, 0, target, 2.0f, 50,
            std::vector<std::shared_ptr<IOnHitEffect>>{ std::make_shared<FreezeOnHit>(30, 0.65f) });
        REQUIRE_NOTHROW(p.update(board));

        REQUIRE(target->hp == 950); // damage doesn't depend on the effect resolving
    }
}

// ---------------- boomerang (Executioner) ----------------

TEST_CASE("Boomerang projectile hits its target twice: once on arrival, once after the return delay", "[projectile][boomerang]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 2.0f, 1000, 1);
    spawn(board, target);

    Projectile p(2, 0.0f, 0.0f, 0, target, 2.0f, 100, {}, true, 3); // returnsToSender, 3-tick return delay
    p.update(board); // outbound hit lands (dist(2.0) == speed(2.0))
    REQUIRE(target->hp == 900);
    REQUIRE(p.isAlive()); // stays alive, waiting out the return trip

    p.update(board); // returnDelayTicks 3 -> 2
    p.update(board); // 2 -> 1
    p.update(board); // 1 -> 0
    REQUIRE(target->hp == 900); // still no second hit
    REQUIRE(p.isAlive());

    p.update(board); // returnDelayTicks == 0: return hit lands
    REQUIRE(target->hp == 800);
    REQUIRE_FALSE(p.isAlive());
}

TEST_CASE("A normal (non-boomerang) projectile still dies after a single hit", "[projectile][boomerang]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 2.0f, 1000, 1);
    spawn(board, target);

    Projectile p(2, 0.0f, 0.0f, 0, target, 2.0f, 100); // returnsToSender defaults false
    p.update(board);

    REQUIRE(target->hp == 900);
    REQUIRE_FALSE(p.isAlive());
}

TEST_CASE("Boomerang projectile skips the return hit if the target dies during the return delay", "[projectile][boomerang]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 2.0f, 150, 1);
    spawn(board, target);

    Projectile p(2, 0.0f, 0.0f, 0, target, 2.0f, 100, {}, true, 3);
    p.update(board); // outbound hit: 150 -> 50
    REQUIRE(target->hp == 50);

    target->takeDamage(100); // dies from something else while the axe is on its way back
    REQUIRE_FALSE(target->isAlive());

    for (int i = 0; i < 5; ++i) p.update(board); // well past the return delay
    REQUIRE_FALSE(p.isAlive());
    REQUIRE(target->hp == -50); // not driven further negative by a return hit that never lands
}

TEST_CASE("Boomerang projectile applies its on-hit effect on both the outbound and return hits", "[projectile][boomerang][on_hit]") {
    Board board;
    auto target = std::make_shared<StationaryCombatant>(1, 0.0f, 2.0f, 1000, 1, 5.0f, 10, 10);
    spawn(board, target);

    Projectile p(2, 0.0f, 0.0f, 0, target, 2.0f, 50,
        std::vector<std::shared_ptr<IOnHitEffect>>{ std::make_shared<FreezeOnHit>(5, 0.5f) }, true, 0);
    p.update(board); // outbound hit lands, applies freeze
    REQUIRE(target->freezeTicks == 5);

    target->freezeTicks = 1; // simulate some freeze already having ticked down
    p.update(board); // return trip (0-tick delay): second hit re-applies the freeze
    REQUIRE(target->freezeTicks == 5);
    REQUIRE_FALSE(p.isAlive());
}

// ---------------- attacker identity (DamageDealtEvent) ----------------

TEST_CASE("Projectile stamps DamageDealtEvent with the attacker identity it was constructed with", "[projectile][stats]") {
    Board board;
    auto recorder = std::make_shared<DamageRecorder>();
    board.statsEvents.subscribe(recorder);

    auto target = std::make_shared<DummyEntity>(1, 0.0f, 2.0f, 1000, 1);
    target->cardId = 99;
    spawn(board, target);

    // attackerId=7, team=0, attackerCardId=3
    Projectile p(2, 0.0f, 0.0f, 0, target, 2.0f, 50, {}, false, 0, 7, 3);
    p.update(board);

    REQUIRE(recorder->hits.size() == 1);
    REQUIRE(recorder->hits[0].attackerId == 7);
    REQUIRE(recorder->hits[0].attackerTeam == 0);
    REQUIRE(recorder->hits[0].attackerCardId == 3);
    REQUIRE(recorder->hits[0].targetId == 1);
    REQUIRE(recorder->hits[0].targetCardId == 99);
    REQUIRE(recorder->hits[0].targetTeam == 1);
    REQUIRE(recorder->hits[0].amount == 50);
}

TEST_CASE("Boomerang projectile stamps the same attacker identity on both the outbound and return hits", "[projectile][stats][boomerang]") {
    Board board;
    auto recorder = std::make_shared<DamageRecorder>();
    board.statsEvents.subscribe(recorder);

    auto target = std::make_shared<DummyEntity>(1, 0.0f, 2.0f, 1000, 1);
    spawn(board, target);

    Projectile p(2, 0.0f, 0.0f, 0, target, 2.0f, 100, {}, true, 0, 7, 3); // 0-tick return delay
    p.update(board); // outbound hit
    p.update(board); // return hit (immediate, 0-tick delay)

    REQUIRE(recorder->hits.size() == 2);
    for (const auto& hit : recorder->hits) {
        REQUIRE(hit.attackerId == 7);
        REQUIRE(hit.attackerCardId == 3);
    }
}
