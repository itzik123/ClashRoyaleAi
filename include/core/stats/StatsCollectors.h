#pragma once
#include "../StatsEventBus.h"
#include "../CardRegistry.h"
#include <unordered_map>
#include <vector>

// Six small, single-responsibility observers, each subscribing to just the
// event callback(s) it needs (the rest stay the IStatsObserver default
// no-op). Composed together by MatchStatistics -- a new stat category is a
// new collector class here, zero changes to the combat/economy code that
// fires the events these consume.

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

// Damage dealt to the OPPONENT's troops vs buildings, split by the
// attacking team -- exactly the breakdown python_ai/train.py's reward
// shaping needs (previously inferred by diffing noisy per-tick HP-channel
// sums in the observation; this reads the authoritative combat events
// instead). Only cross-team hits count (targetTeam != attackerTeam guard),
// so this stays correct even if some future card ever produced same-team
// splash. A targetCardId not found in CardRegistry is a Tower (Towers are
// built directly by GameManager with a negative sentinel id, never
// registered) -- Towers are buildings, so that's the "not found" branch
// below.
class DamageByTargetTypeCollector : public IStatsObserver {
    int troopDamageByTeam[2] = { 0, 0 };
    int buildingDamageByTeam[2] = { 0, 0 };

public:
    void onDamageDealt(const DamageDealtEvent& e) override {
        if (e.targetTeam == e.attackerTeam) return;
        int t = (e.attackerTeam == 0) ? 0 : 1;
        if (isBuildingCardId(e.targetCardId)) buildingDamageByTeam[t] += e.amount;
        else troopDamageByTeam[t] += e.amount;
    }

    int troopDamageDealt(int team) const { return troopDamageByTeam[(team == 0) ? 0 : 1]; }
    int buildingDamageDealt(int team) const { return buildingDamageByTeam[(team == 0) ? 0 : 1]; }

private:
    static bool isBuildingCardId(int cardId) {
        const CardDefinition* def = CardRegistry::getInstance().getCard(cardId);
        return (def == nullptr) ? true : def->isBuilding;
    }
};

// Kill count per team, per (team, killer's cardId). Attribution is "who
// last hit this entity" -- tracked from DamageDealtEvent, credited on
// EntityDiedEvent, invalidated by AttributionClearedEvent (see Building's
// decay branch: a building chip-damaged in combat that later dies of
// unrelated decay must not have that death credited to the earlier hitter).
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
        // find/count only, never operator[] -- a victim with no recorded
        // hit (e.g. one that only ever took excluded self-decay damage)
        // correctly reports "no killer" instead of inserting a spurious
        // entry.
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

// Elixir spent per team, per (team, cardId) -- spent == the card's cost,
// already on CardPlayedEvent.
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

// Ordered per-team timeline of every card played -- "what did each side
// play and when" reconstruction without re-diffing hands.
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

// Match result + duration, read off the single MatchEndedEvent -- event-
// driven like the other four collectors here, not a special-cased direct
// read off GameManager (MatchStatistics only ever holds a Board&, so that
// wouldn't even be reachable).
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
