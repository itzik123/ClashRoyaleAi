#pragma once
#include "PeriodicEffect.h"
#include "CombatEntity.h"
#include "Board.h"

// Royal Chef Tower Troop's periodic "pancake toss": buffs the nearest
// ally's damage and gives a flat hp top-up, approximating the real
// "+1 level" (no level system exists in this engine -- see CardStats.h's
// withPassiveDamageReduction-adjacent comment; this is the same
// approximation category as Mirror's own skipped level bump). "Permanent"
// is modeled as a very large tick count, not a new flag; a second pancake
// landing on an already-buffed ally overwrites rather than compounds
// (existing CombatEntity::applyBuff behavior). Does its own scan rather
// than reusing the shared applyAreaBuff free function, since
// IPeriodicEffect::apply has no id to exclude "self" with, and the
// casting Tower always sits at exactly `position` (distance 0) --
// skipped explicitly instead.
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
            // Already served. The chef moves on to a troop that has not eaten
            // rather than feeding the same one every interval -- see
            // CombatEntity::royalChefServed for the compounding this prevents.
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
