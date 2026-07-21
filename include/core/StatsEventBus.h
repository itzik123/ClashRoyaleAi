#pragma once
#include "StatsEvents.h"
#include <memory>
#include <vector>

// Read-only Observer channel for "something happened this tick" -- distinct
// from IOnHitEffect/IDeathEffect (single-method, no-default, gameplay-
// mutating Strategy interfaces composed onto individual CombatEntity
// instances). IStatsObserver is a genuinely new shape in this codebase: one
// interface, one callback per event category, default no-op (same idiom as
// Entity::onDeath/clampPosition) so a collector only overrides what it
// actually cares about, fanned out to any number of subscribers via
// StatsEventBus rather than being attached to a single entity.
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

// Owned by Board (not GameManager): Board& is already threaded through every
// call site that could ever need to report something (update(Board&),
// performAttack(Board&, ...), GameManager reaches it via its own board
// member), so this needs zero new parameters anywhere. Zero cost when
// nobody's subscribed -- each notify is just an empty-vector iteration.
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
