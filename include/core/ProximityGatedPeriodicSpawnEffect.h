#pragma once
#include "PeriodicEffect.h"
#include "CardStats.h"
#include "CardFactories.h"

// PeriodicSpawnEffect that spawns only while an enemy is within
// `detectionRadius` (Goblin Hut). The timer keeps running either way.
class ProximityGatedPeriodicSpawnEffect : public IPeriodicEffect {
    CardStats childStats;
    float detectionRadius;

public:
    ProximityGatedPeriodicSpawnEffect(CardStats childStats, float detectionRadius)
        : childStats(std::move(childStats)), detectionRadius(detectionRadius) {}

    void apply(Board& board, const Vector2D& position, int team) const override {
        for (const auto& entity : board.getEntities()) {
            if (entity->team == team || !entity->isAlive() || !entity->isTargetable()) continue;
            if (position.distanceTo(entity->position) <= detectionRadius) {
                CardFactories::spawn(childStats, position.x, position.y, team, board);
                return;
            }
        }
    }
};
