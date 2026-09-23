#pragma once
#include "PeriodicEffect.h"
#include "CombatEntity.h"
#include "Board.h"

// Royal Chef's periodic pancake: buffs the nearest ally's damage and tops up
// its hp by 10%, approximating the real "+1 level" (there are no levels). Its
// own scan instead of applyAreaBuff, which cannot exclude the tower at distance
// 0.
class RoyalChefBuffEffect : public IPeriodicEffect {
    float radius;
    float damageMultiplier;

public:
    RoyalChefBuffEffect(float radius, float damageMultiplier)
        : radius(radius), damageMultiplier(damageMultiplier) {}

    void apply(Board& board, const Vector2D& position, int team) const override {
        std::shared_ptr<CombatEntity> nearest;
        float nearestDist = radius;
        for (const auto& entity : board.getEntities()) {
            if (entity->team != team || !entity->isAlive()) continue;
            float dist = position.distanceTo(entity->position);
            if (dist <= 0.01f || dist > nearestDist) continue;
            auto combatEntity = std::dynamic_pointer_cast<CombatEntity>(entity);
            if (!combatEntity) continue;
            // Each ally is served once; repeat servings would compound without
            // bound. See CombatEntity::royalChefServed.
            if (combatEntity->royalChefServed) continue;
            nearest = combatEntity;
            nearestDist = dist;
        }
        if (!nearest) return;
        nearest->royalChefServed = true;
        nearest->applyBuff(damageMultiplier, 999999);
        nearest->hp += nearest->hp / 10;
    }
};
