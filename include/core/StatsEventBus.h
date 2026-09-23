#pragma once
#include "StatsEvents.h"
#include <memory>
#include <vector>

// Read-only observer for "something happened this tick", unlike the
// gameplay-mutating effect interfaces. Every callback defaults to a no-op, so a
// collector overrides only what it needs.
class IStatsObserver {
public:
    virtual ~IStatsObserver() = default;
    virtual void onDamageDealt(const DamageDealtEvent&) {}
    virtual void onEntityDied(const EntityDiedEvent&) {}
    virtual void onEntitySpawned(const EntitySpawnedEvent&) {}
    virtual void onCardPlayed(const CardPlayedEvent&) {}
    virtual void onChampionAbilityActivated(const ChampionAbilityActivatedEvent&) {}
    virtual void onMatchEnded(const MatchEndedEvent&) {}
    virtual void onAttributionCleared(const AttributionClearedEvent&) {}
};

// Owned by Board, which every reporting call site already has. With no
// subscribers each notify is an empty loop.
class StatsEventBus {
    std::vector<std::shared_ptr<IStatsObserver>> observers;

public:
    void subscribe(std::shared_ptr<IStatsObserver> observer) {
        observers.push_back(std::move(observer));
    }

    void notifyDamageDealt(const DamageDealtEvent& e) const {
        for (const auto& o : observers) o->onDamageDealt(e);
    }
    void notifyEntityDied(const EntityDiedEvent& e) const {
        for (const auto& o : observers) o->onEntityDied(e);
    }
    void notifyEntitySpawned(const EntitySpawnedEvent& e) const {
        for (const auto& o : observers) o->onEntitySpawned(e);
    }
    void notifyCardPlayed(const CardPlayedEvent& e) const {
        for (const auto& o : observers) o->onCardPlayed(e);
    }
    void notifyChampionAbilityActivated(const ChampionAbilityActivatedEvent& e) const {
        for (const auto& o : observers) o->onChampionAbilityActivated(e);
    }
    void notifyMatchEnded(const MatchEndedEvent& e) const {
        for (const auto& o : observers) o->onMatchEnded(e);
    }
    void notifyAttributionCleared(const AttributionClearedEvent& e) const {
        for (const auto& o : observers) o->onAttributionCleared(e);
    }
};
