#pragma once
#include "PeriodicEffect.h"
#include "CombatEntity.h"
#include "Board.h"

// Periodically fully freezes (slowFactor 0.0 -- can't move or attack) the
// nearest enemy within radius (Hunter Evolution's net throw). Reuses the
// existing freeze machinery; only the "find nearest enemy and target it"
// part is new, since periodicEffect's normal use (Witch/Furnace) always
// spawns AT its own position rather than reaching out to a specific
// other entity.
class PeriodicFreezeNearestEffect : public IPeriodicEffect {
    float radius;
    int freezeTicks;

public:
    PeriodicFreezeNearestEffect(float radius, int freezeTicks)
        : radius(radius), freezeTicks(freezeTicks) {}

    void apply(Board& board, const Vector2D& position, int team) const override {
        std::shared_ptr<CombatEntity> nearest;
        float nearestDist = radius;
        for (const auto& e : board.getEntities()) {
            if (!e->isAlive() || e->team == team || !e->isTargetable()) continue;
            auto combatEntity = std::dynamic_pointer_cast<CombatEntity>(e);
            if (!combatEntity) continue;
            float dist = position.distanceTo(e->position);
            if (dist <= nearestDist) {
                nearest = combatEntity;
                nearestDist = dist;
            }
        }
        if (nearest) nearest->applyFreeze(freezeTicks, 0.0f);
    }
};
