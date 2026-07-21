#pragma once

// Plain data, no behavior -- what StatsEventBus fans out to IStatsObserver
// subscribers. Each struct carries exactly what its firing site already has
// in scope; see StatsEventBus.h for the interface these are delivered
// through, and MatchStatistics.h for the built-in collectors that consume
// them.

struct DamageDealtEvent {
    int attackerId;
    int attackerTeam;
    int attackerCardId;
    int targetId;
    int targetCardId;
    int targetTeam;
    int amount;
    int tick;
};

struct EntityDiedEvent {
    int entityId;
    int cardId;
    int team;
    int tick;
};

// No collector consumes this yet in this pass -- included because
// Board::commitPendingEntities() is a free, already-existing choke point for
// it, and it's likely needed soon (e.g. a future "peak board population"
// stat). Not a dangling half-finished feature: it's simply forward-looking.
struct EntitySpawnedEvent {
    int entityId;
    int cardId;
    int team;
    int tick;
};

struct CardPlayedEvent {
    int team;
    int cardId;
    float cost;
    float x;
    float y;
    int tick;
};

// Fired by GameManager::activateChampionAbility() on a successful
// activation -- deliberately NOT a CardPlayedEvent: activating an
// already-deployed Champion's ability doesn't touch the hand/deck the way
// playing a card does (see PlayerState::playCard vs.
// GameManager::activateChampionAbility), so folding it into
// CardPlayStatsCollector's "what was played and when" timeline would be
// misleading. No x/y -- an ability fires wherever the Champion currently
// is, not at a chosen placement point.
struct ChampionAbilityActivatedEvent {
    int team;
    int cardId;
    float cost;
    int tick;
};

struct MatchEndedEvent {
    int loserTeam; // -1 for a draw
    int tick;
};

// Small, single-purpose: tells KillStatsCollector to drop any stale "last
// hit" entry for an entity id, fired from Building::update()'s decay branch
// so a building that was chip-damaged in combat and later dies of unrelated
// decay doesn't have its death wrongly credited to whoever landed that
// earlier hit. Not a general-purpose event.
struct AttributionClearedEvent {
    int entityId;
};
