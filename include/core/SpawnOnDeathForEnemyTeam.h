#pragma once
#include "DeathEffect.h"
#include "CardStats.h"
#include "CardFactories.h"

// SpawnOnDeath for the OPPOSING team (Mother Witch's Cursed Hog fights for her,
// against the cursed unit's team).
class SpawnOnDeathForEnemyTeam : public IDeathEffect {
    CardStats childStats;

public:
    explicit SpawnOnDeathForEnemyTeam(CardStats childStats) : childStats(std::move(childStats)) {}

    void apply(Board& board, const Vector2D& position, int team) const override {
        CardFactories::spawn(childStats, position.x, position.y, 1 - team, board);
    }
};
