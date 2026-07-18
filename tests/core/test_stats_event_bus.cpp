#include <catch_amalgamated.hpp>
#include "StatsEventBus.h"

namespace {
    // Records how many times each callback fired and the last event of each
    // kind, without needing a real collector to exercise pure bus/dispatch
    // behavior.
    class RecordingObserver : public IStatsObserver {
    public:
        int damageDealtCount = 0;
        int entityDiedCount = 0;
        int entitySpawnedCount = 0;
        int cardPlayedCount = 0;
        int matchEndedCount = 0;
        int attributionClearedCount = 0;
        DamageDealtEvent lastDamage{};

        void onDamageDealt(const DamageDealtEvent& e) override { damageDealtCount++; lastDamage = e; }
        void onEntityDied(const EntityDiedEvent&) override { entityDiedCount++; }
        void onEntitySpawned(const EntitySpawnedEvent&) override { entitySpawnedCount++; }
        void onCardPlayed(const CardPlayedEvent&) override { cardPlayedCount++; }
        void onMatchEnded(const MatchEndedEvent&) override { matchEndedCount++; }
        void onAttributionCleared(const AttributionClearedEvent&) override { attributionClearedCount++; }
    };

    // Only overrides onDamageDealt -- exercises that every other callback's
    // IStatsObserver default (no-op) is safe to invoke.
    class PartialObserver : public IStatsObserver {
    public:
        int damageDealtCount = 0;
        void onDamageDealt(const DamageDealtEvent&) override { damageDealtCount++; }
    };
}

TEST_CASE("StatsEventBus delivers each event type to a subscribed observer", "[stats_event_bus]") {
    StatsEventBus bus;
    auto observer = std::make_shared<RecordingObserver>();
    bus.subscribe(observer);

    bus.notifyDamageDealt({ 1, 0, 10, 2, 20, 1, 50, 5 });
    bus.notifyEntityDied({ 2, 20, 1, 5 });
    bus.notifyEntitySpawned({ 3, 30, 0, 1 });
    bus.notifyCardPlayed({ 0, 10, 3.0f, 9.0f, 10.0f, 1 });
    bus.notifyMatchEnded({ -1, 100 });
    bus.notifyAttributionCleared({ 2 });

    REQUIRE(observer->damageDealtCount == 1);
    REQUIRE(observer->entityDiedCount == 1);
    REQUIRE(observer->entitySpawnedCount == 1);
    REQUIRE(observer->cardPlayedCount == 1);
    REQUIRE(observer->matchEndedCount == 1);
    REQUIRE(observer->attributionClearedCount == 1);

    REQUIRE(observer->lastDamage.attackerId == 1);
    REQUIRE(observer->lastDamage.attackerTeam == 0);
    REQUIRE(observer->lastDamage.attackerCardId == 10);
    REQUIRE(observer->lastDamage.targetId == 2);
    REQUIRE(observer->lastDamage.targetCardId == 20);
    REQUIRE(observer->lastDamage.targetTeam == 1);
    REQUIRE(observer->lastDamage.amount == 50);
    REQUIRE(observer->lastDamage.tick == 5);
}

TEST_CASE("StatsEventBus fans out the same event to every subscribed observer", "[stats_event_bus]") {
    StatsEventBus bus;
    auto a = std::make_shared<RecordingObserver>();
    auto b = std::make_shared<RecordingObserver>();
    bus.subscribe(a);
    bus.subscribe(b);

    bus.notifyEntityDied({ 5, 50, 0, 3 });

    REQUIRE(a->entityDiedCount == 1);
    REQUIRE(b->entityDiedCount == 1);
}

TEST_CASE("An observer with no subscribers is simply a no-op, no crash", "[stats_event_bus]") {
    StatsEventBus bus; // nobody subscribed
    REQUIRE_NOTHROW(bus.notifyDamageDealt({ 1, 0, 10, 2, 20, 1, 50, 5 }));
}

TEST_CASE("An observer that only overrides one callback safely ignores the others firing", "[stats_event_bus]") {
    StatsEventBus bus;
    auto observer = std::make_shared<PartialObserver>();
    bus.subscribe(observer);

    REQUIRE_NOTHROW(bus.notifyEntityDied({ 1, 10, 0, 1 }));
    REQUIRE_NOTHROW(bus.notifyEntitySpawned({ 1, 10, 0, 1 }));
    REQUIRE_NOTHROW(bus.notifyCardPlayed({ 0, 10, 3.0f, 9.0f, 10.0f, 1 }));
    REQUIRE_NOTHROW(bus.notifyMatchEnded({ -1, 100 }));
    REQUIRE_NOTHROW(bus.notifyAttributionCleared({ 1 }));

    bus.notifyDamageDealt({ 1, 0, 10, 2, 20, 1, 50, 5 });
    REQUIRE(observer->damageDealtCount == 1);
}
