#pragma once
#include "PeriodicEffect.h"
#include "CombatEntity.h"

// Electro Giant's periodic shock: stuns every enemy CombatEntity within
// radius of this entity's own position, on a repeating timer (see
// CombatEntity::periodicEffect). Deals no damage -- the real shock is a
// pure stun, same "freeze with slowFactor 0" pattern as Electro Wizard's
// zap.
class AreaStunEffect : public IPeriodicEffect {
    float radius;
    int stunTicks;

public:
    AreaStunEffect(float radius, int stunTicks) : radius(radius), stunTicks(stunTicks) {}

    void apply(Board& board, const Vector2D& position, int team) const override {
        for (const auto& entity : board.getEntities()) {
            if (entity->team == team || !entity->isAlive() || !entity->isTargetable()) continue;
            if (position.distanceTo(entity->position) > radius) continue;
            if (auto combatEntity = std::dynamic_pointer_cast<CombatEntity>(entity)) {
                combatEntity->applyFreeze(stunTicks, 0.0f);
            }
        }
    }
};
