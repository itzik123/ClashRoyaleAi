#pragma once
#include "PeriodicEffect.h"
#include "CardStats.h"
#include "CardFactories.h"
#include "Board.h"

// Spawns a copy of childStats at wherever the effect fires, unless
// maxAlive entities carrying childStats' own id are already alive on that
// team -- Evolved Skeletons' "Never-ending Horde" (spawns another Evolved
// Skeleton on every landed attack, capped at 8 total on the field).
// Reuses IPeriodicEffect's exact shape (Board&, position, team) since
// CombatEntity::onHitSpawnEffect fires it from the same "spawn something
// here" call site as periodicEffect, just on a landed hit instead of a
// tick interval -- see CombatEntity::onHitSpawnEffect's own comment.
class CappedSpawnOnHitEffect : public IPeriodicEffect {
    CardStats childStats;
    int maxAlive;

public:
    CappedSpawnOnHitEffect(CardStats childStats, int maxAlive)
        : childStats(std::move(childStats)), maxAlive(maxAlive) {}

    void apply(Board& board, const Vector2D& position, int team) const override {
        int aliveCount = 0;
        for (const auto& e : board.getEntities()) {
            if (e->isAlive() && e->team == team && e->cardId == childStats.id) aliveCount++;
        }
        if (aliveCount >= maxAlive) return;
        CardFactories::spawn(childStats, position.x, position.y, team, board);
    }
};
