#pragma once
#include "DeathEffect.h"
#include "CombatEntity.h"

// Lumberjack's death potion: buffs nearby allies' damage for a duration.
// excludeId is -1 because the dying entity is already gone.
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
