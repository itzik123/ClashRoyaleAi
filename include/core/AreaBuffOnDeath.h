#pragma once
#include "DeathEffect.h"
#include "CombatEntity.h"

// Lumberjack's death potion: buffs every nearby ally's damage for a
// duration, the instant this entity dies. Reuses CombatEntity's own
// applyAreaBuff sweep -- excludeId is -1 since the dying entity is
// already gone by the time this fires, nothing to exclude.
class AreaBuffOnDeath : public IDeathEffect {
    float radius;
    float multiplier;
    int durationTicks;

public:
    AreaBuffOnDeath(float radius, float multiplier, int durationTicks)
        : radius(radius), multiplier(multiplier), durationTicks(durationTicks) {}

    void apply(Board& board, const Vector2D& position, int team) const override {
        applyAreaBuff(board, position, radius, -1, team, multiplier, durationTicks, 1000000);
    }
};
