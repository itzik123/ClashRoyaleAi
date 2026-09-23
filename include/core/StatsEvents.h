#pragma once

// Plain data that StatsEventBus delivers to IStatsObserver subscribers
// (MatchStatistics.h has the built-in collectors). Each struct carries what its
// firing site has in scope.

struct DamageDealtEvent {
    int attackerId;
    int attackerTeam;
    int attackerCardId;
    int targetId;
    int targetCardId;
    int targetTeam;
    int amount;
    int tick;
    // The target's own isTower(), stamped at the emit site. targetCardId cannot
    // tell: towers and spawned helper bodies (Goblin Barrel's goblins,
    // Graveyard skeletons) both carry unregistered ids. Last and defaulted, so
    // an 8-value aggregate init means "not a tower".
    bool targetIsTower = false;
};

struct EntityDiedEvent {
    int entityId;
    int cardId;
    int team;
    int tick;
};

// No collector consumes this yet; Board::commitPendingEntities is its natural
// emit point.
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

// Not a CardPlayedEvent: activating an ability does not touch the hand, and it
// fires wherever the Champion is, so there is no x/y.
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

// Tells KillStatsCollector to forget the last hit on an entity. Fired by
// Building decay, so a building chipped in combat that later decays is not
// credited to the earlier attacker.
struct AttributionClearedEvent {
    int entityId;
};
