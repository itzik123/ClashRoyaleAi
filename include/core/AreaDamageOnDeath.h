#pragma once
#include "DeathEffect.h"
#include "Board.h"

// Area damage triggered on death (Giant Skeleton's bomb, Golem/Ice Golem/
// Balloon's explosion) -- everyone within radius of the death position,
// same team exclusion as applySplashDamage. No stats event fired for this
// damage (same "not a real attacker" treatment as Building's own decay
// damage -- the dying entity is already gone by the time this runs, so
// there's no attacker identity left to attribute a DamageDealtEvent to).
class AreaDamageOnDeath : public IDeathEffect {
    float radius;
    int damage;

public:
    AreaDamageOnDeath(float radius, int damage) : radius(radius), damage(damage) {}

    void apply(Board& board, const Vector2D& position, int team) const override {
        for (const auto& entity : board.getEntities()) {
            if (entity->team == team || !entity->isAlive() || !entity->isTargetable()) continue;
            if (position.distanceTo(entity->position) <= radius) {
                entity->takeDamage(damage);
            }
        }
    }
};
