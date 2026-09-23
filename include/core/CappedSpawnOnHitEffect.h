#pragma once
#include "PeriodicEffect.h"
#include "CardStats.h"
#include "CardFactories.h"
#include "Board.h"

// Evolved Skeletons' "Never-ending Horde": spawns another copy on each landed
// attack, unless maxAlive with that id are already alive on the team. Has
// IPeriodicEffect's shape because CombatEntity::onHitSpawnEffect fires it from
// the same call site.
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
