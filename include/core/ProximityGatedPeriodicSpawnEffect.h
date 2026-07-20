#pragma once
#include "PeriodicEffect.h"
#include "CardStats.h"
#include "CardFactories.h"

// Like PeriodicSpawnEffect, but only actually spawns if an enemy is within
// `detectionRadius` of this entity's position when the timer fires (Goblin
// Hut: real card only summons Spear Goblins while an enemy is in its
// 6-tile range, not on an unconditional timer). The interval keeps ticking
// either way -- this just no-ops the spawn itself when nothing's nearby,
// same "check every fire, skip if the condition isn't met" shape as every
// other conditional effect in this engine (e.g. shield/curse in takeDamage).
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
