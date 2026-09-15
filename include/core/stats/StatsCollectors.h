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
// TOWERS are tracked separately from deployed buildings, added 2026-08-06.
// Both are "buildings" and both used to land in one counter, which made
// train.py's potential function charge the agent for damage to its own Cannon
// at exactly the rate it charges damage to a Princess Tower. A Cannon is a
// sacrificial card -- its job is to absorb a push and die -- and at
// W_BLDG = 0.5 with MAX_BUILDING_HP = 4008, losing its 824 HP cost 0.1028 of
// shaping return while killing a troop paid only 0.1 * hp / 4256. It therefore
// had to kill 5.3x its own HP just to break even, so a Cannon placed anywhere
// useful was negative in expectation, while one parked in a back corner where
// nothing could reach it was exactly zero -- decay emits no DamageDealtEvent
// (see Building::update), so the engine never charged for it dying of old age.
// The measured policy did what that reward asked: 27.9% of its Cannons went to
// (11,2)/(11,3), behind its own King, mean placement y = 6.3 -- behind its own
// Princess Towers.
//
// buildingDamageDealt() deliberately still returns tower + deployed building,
// so GameLogger and every existing test keep their previous meaning; the split
// is exposed additively via towerDamageDealt().
class DamageByTargetTypeCollector : public IStatsObserver {
    int troopDamageByTeam[2] = { 0, 0 };
    int buildingDamageByTeam[2] = { 0, 0 };
    int towerDamageByTeam[2] = { 0, 0 };

public:
    void onDamageDealt(const DamageDealtEvent& e) override {
        if (e.targetTeam == e.attackerTeam) return;
        int t = (e.attackerTeam == 0) ? 0 : 1;
        // A tower is identified by the TARGET'S OWN TYPE, stamped on the event
        // (DamageDealtEvent::targetIsTower), never by its id. Until 2026-09-15
        // this read "targetCardId not in CardRegistry is a Tower" -- and spawned
        // helper bodies are not in the registry either, so a tower shooting a
        // Goblin Barrel's goblins booked TOWER damage for its owner (measured:
        // 810 for one Barrel, 1080 for a Graveyard, 1440 for a Battle Ram's
        // Barbarians, with the other side's towers untouched). Same fix as
        // MatchRules' King: identify by type, not by a discriminator that
        // something else can also wear. An unregistered NON-tower (a spawned
        // body) falls through to troop damage below.
        const CardDefinition* def = CardRegistry::getInstance().getCard(e.targetCardId);
        if (e.targetIsTower) {
            towerDamageByTeam[t] += e.amount;
            buildingDamageByTeam[t] += e.amount;
        } else if (def != nullptr && def->isBuilding) {
            // `def` IS null for every spawned body -- that is the whole bug
            // this block fixes -- so it must be checked, not assumed.
            buildingDamageByTeam[t] += e.amount;
        } else {
            troopDamageByTeam[t] += e.amount;
        }
    }

    int troopDamageDealt(int team) const { return troopDamageByTeam[(team == 0) ? 0 : 1]; }
    int buildingDamageDealt(int team) const { return buildingDamageByTeam[(team == 0) ? 0 : 1]; }
    // Towers only. Deployed buildings (Cannon, Tesla, ...) are the difference
    // between this and buildingDamageDealt().
    int towerDamageDealt(int team) const { return towerDamageByTeam[(team == 0) ? 0 : 1]; }
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

// Elixir VALUE destroyed, per (team, killer's cardId). Same attribution
// machinery as KillStatsCollector above -- last hitter, credited on death,
// invalidated by decay -- but weighted by the VICTIM's elixir cost instead of
// counted.
//
// Cost rather than HP on purpose. A trade is settled in elixir, and raw damage
// is already rewarded via train.py's W_TROOPS term; paying again on damage
// would double-count overkill, making a Fireball that deals 689 to a 230 HP
// Minion look three times better than one that deals exactly 230. Cost is also
// the only unit in which "did this spell earn its 4 elixir" is a well-posed
// question, which is what the shaping term consuming this actually asks.
//
// Towers are skipped: getCard() returns nullptr for them (negative sentinel
// id, never registered) and a tower has no elixir cost to credit. Tower value
// is already carried by the tower potential and W_TOWER_DESTROYED.
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

// Elixir spent (and activation count) on Champion abilities, per team and
// per (team, cardId) -- separate from ElixirStatsCollector/
// CardPlayStatsCollector since an ability activation isn't a card played
// from hand (see ChampionAbilityActivatedEvent). Previously this spend was
// simply invisible to MatchStatistics.
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
