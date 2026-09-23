#pragma once
#include "../StatsEventBus.h"
#include "../CardRegistry.h"
#include <unordered_map>
#include <vector>

// Small single-purpose observers composed by MatchStatistics. A new stat is a
// new collector here, with no change to the code that fires the events.

// Total damage dealt per team, and per (team, attacker's cardId).
class DamageStatsCollector : public IStatsObserver {
    int totalByTeam[2] = { 0, 0 };
    std::unordered_map<int, int> byCardByTeam[2];

public:
    void onDamageDealt(const DamageDealtEvent& e) override {
        int t = (e.attackerTeam == 0) ? 0 : 1;
        totalByTeam[t] += e.amount;
        byCardByTeam[t][e.attackerCardId] += e.amount;
    }

    int total(int team) const { return totalByTeam[(team == 0) ? 0 : 1]; }
    int byCard(int cardId, int team) const {
        const auto& m = byCardByTeam[(team == 0) ? 0 : 1];
        auto it = m.find(cardId);
        return (it != m.end()) ? it->second : 0;
    }
    const std::unordered_map<int, int>& byCardMap(int team) const {
        return byCardByTeam[(team == 0) ? 0 : 1];
    }
};

// Damage dealt to the OPPONENT's troops vs buildings, per attacking team; the
// reward shaping reads these. Only cross-team hits count.
//
// Towers are tracked separately from deployed buildings. The tower potential
// must not charge the agent for damage to its own Cannon at the Princess Tower
// rate: that made a sacrificial Cannon negative anywhere useful and exactly
// zero parked out of reach (decay emits no DamageDealtEvent).
// buildingDamageDealt() still returns towers plus deployed buildings.
class DamageByTargetTypeCollector : public IStatsObserver {
    int troopDamageByTeam[2] = { 0, 0 };
    int buildingDamageByTeam[2] = { 0, 0 };
    int towerDamageByTeam[2] = { 0, 0 };

public:
    void onDamageDealt(const DamageDealtEvent& e) override {
        if (e.targetTeam == e.attackerTeam) return;
        int t = (e.attackerTeam == 0) ? 0 : 1;
        // A tower is identified by the target's own type
        // (DamageDealtEvent::targetIsTower), never by its id: spawned helper
        // bodies are unregistered too. An unregistered non-tower falls through
        // to troop damage.
        const CardDefinition* def = CardRegistry::getInstance().getCard(e.targetCardId);
        if (e.targetIsTower) {
            towerDamageByTeam[t] += e.amount;
            buildingDamageByTeam[t] += e.amount;
        } else if (def != nullptr && def->isBuilding) {
            // `def` is null for every spawned body.
            buildingDamageByTeam[t] += e.amount;
        } else {
            troopDamageByTeam[t] += e.amount;
        }
    }

    int troopDamageDealt(int team) const { return troopDamageByTeam[(team == 0) ? 0 : 1]; }
    int buildingDamageDealt(int team) const { return buildingDamageByTeam[(team == 0) ? 0 : 1]; }
    // Towers only; deployed buildings are the difference from
    // buildingDamageDealt().
    int towerDamageDealt(int team) const { return towerDamageByTeam[(team == 0) ? 0 : 1]; }
};

// Kills per team and per (team, killer's cardId). Attribution is the last
// hitter, credited on death and cleared by AttributionClearedEvent (Building
// decay).
class KillStatsCollector : public IStatsObserver {
    struct LastHit { int attackerTeam; int attackerCardId; };
    std::unordered_map<int, LastHit> lastHitByEntityId;
    int killsByTeam[2] = { 0, 0 };
    std::unordered_map<int, int> killsByCardByTeam[2];

public:
    void onDamageDealt(const DamageDealtEvent& e) override {
        lastHitByEntityId[e.targetId] = { e.attackerTeam, e.attackerCardId };
    }

    void onAttributionCleared(const AttributionClearedEvent& e) override {
        lastHitByEntityId.erase(e.entityId);
    }

    void onEntityDied(const EntityDiedEvent& e) override {
        // find, never operator[]: a victim with no recorded hit reports no
        // killer.
        auto it = lastHitByEntityId.find(e.entityId);
        if (it == lastHitByEntityId.end()) return;
        int t = (it->second.attackerTeam == 0) ? 0 : 1;
        killsByTeam[t]++;
        killsByCardByTeam[t][it->second.attackerCardId]++;
        lastHitByEntityId.erase(it);
    }

    int kills(int team) const { return killsByTeam[(team == 0) ? 0 : 1]; }
    int killsByCard(int cardId, int team) const {
        const auto& m = killsByCardByTeam[(team == 0) ? 0 : 1];
        auto it = m.find(cardId);
        return (it != m.end()) ? it->second : 0;
    }
    const std::unordered_map<int, int>& killsByCardMap(int team) const {
        return killsByCardByTeam[(team == 0) ? 0 : 1];
    }
};

// Elixir value destroyed, per (team, killer's cardId): the same attribution as
// KillStatsCollector, weighted by the victim's cost.
//
// Cost rather than hp: a trade is settled in elixir, damage is already rewarded
// elsewhere, and hp would pay for overkill. Towers are skipped (unregistered,
// no cost); their value is in the tower potential and W_TOWER_DESTROYED.
class ElixirValueKilledCollector : public IStatsObserver {
    struct LastHit { int attackerTeam; int attackerCardId; };
    std::unordered_map<int, LastHit> lastHitByEntityId;
    std::unordered_map<int, float> valueByCardByTeam[2];
    float valueByTeam[2] = { 0.0f, 0.0f };

public:
    void onDamageDealt(const DamageDealtEvent& e) override {
        if (e.targetTeam == e.attackerTeam) return;
        lastHitByEntityId[e.targetId] = { e.attackerTeam, e.attackerCardId };
    }

    void onAttributionCleared(const AttributionClearedEvent& e) override {
        lastHitByEntityId.erase(e.entityId);
    }

    void onEntityDied(const EntityDiedEvent& e) override {
        auto it = lastHitByEntityId.find(e.entityId);
        if (it == lastHitByEntityId.end()) return;
        const CardDefinition* victim = CardRegistry::getInstance().getCard(e.cardId);
        if (victim != nullptr) {
            int t = (it->second.attackerTeam == 0) ? 0 : 1;
            valueByCardByTeam[t][it->second.attackerCardId] += victim->cost;
            valueByTeam[t] += victim->cost;
        }
        lastHitByEntityId.erase(it);
    }

    float total(int team) const { return valueByTeam[(team == 0) ? 0 : 1]; }
    float byCard(int cardId, int team) const {
        const auto& m = valueByCardByTeam[(team == 0) ? 0 : 1];
        auto it = m.find(cardId);
        return (it != m.end()) ? it->second : 0.0f;
    }
};

// Elixir spent per team and per (team, cardId).
class ElixirStatsCollector : public IStatsObserver {
    float spentByTeam[2] = { 0.0f, 0.0f };
    std::unordered_map<int, float> spentByCardByTeam[2];

public:
    void onCardPlayed(const CardPlayedEvent& e) override {
        int t = (e.team == 0) ? 0 : 1;
        spentByTeam[t] += e.cost;
        spentByCardByTeam[t][e.cardId] += e.cost;
    }

    float total(int team) const { return spentByTeam[(team == 0) ? 0 : 1]; }
    float byCard(int cardId, int team) const {
        const auto& m = spentByCardByTeam[(team == 0) ? 0 : 1];
        auto it = m.find(cardId);
        return (it != m.end()) ? it->second : 0.0f;
    }
    const std::unordered_map<int, float>& byCardMap(int team) const {
        return spentByCardByTeam[(team == 0) ? 0 : 1];
    }
};

// Elixir spent and activation count on Champion abilities, kept apart from card
// plays.
class ChampionAbilityStatsCollector : public IStatsObserver {
    float spentByTeam[2] = { 0.0f, 0.0f };
    int activationsByTeam[2] = { 0, 0 };
    std::unordered_map<int, int> activationsByCardByTeam[2];

public:
    void onChampionAbilityActivated(const ChampionAbilityActivatedEvent& e) override {
        int t = (e.team == 0) ? 0 : 1;
        spentByTeam[t] += e.cost;
        activationsByTeam[t]++;
        activationsByCardByTeam[t][e.cardId]++;
    }

    float elixirSpent(int team) const { return spentByTeam[(team == 0) ? 0 : 1]; }
    int activations(int team) const { return activationsByTeam[(team == 0) ? 0 : 1]; }
    int activationsByCard(int cardId, int team) const {
        const auto& m = activationsByCardByTeam[(team == 0) ? 0 : 1];
        auto it = m.find(cardId);
        return (it != m.end()) ? it->second : 0;
    }
};

// Ordered per-team timeline of every card played.
class CardPlayStatsCollector : public IStatsObserver {
    std::vector<CardPlayedEvent> timelineByTeam[2];

public:
    void onCardPlayed(const CardPlayedEvent& e) override {
        timelineByTeam[(e.team == 0) ? 0 : 1].push_back(e);
    }

    const std::vector<CardPlayedEvent>& timeline(int team) const {
        return timelineByTeam[(team == 0) ? 0 : 1];
    }
};

// Match result and duration, from the MatchEndedEvent.
class MatchOutcomeCollector : public IStatsObserver {
    int loser = -1;
    int durationTicks = 0;

public:
    void onMatchEnded(const MatchEndedEvent& e) override {
        loser = e.loserTeam;
        durationTicks = e.tick;
    }

    int loserTeam() const { return loser; }
    int matchDurationTicks() const { return durationTicks; }
};
