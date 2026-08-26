#pragma once
#include "DeathEffect.h"
#include "Board.h"
#include "CombatEntity.h"
#include "OnHitEffect.h"
#include <memory>

// Area damage triggered on death (Giant Skeleton's bomb, Golem/Ice Golem/
// Balloon's explosion) -- everyone within radius of the death position,
// same team exclusion as applySplashDamage. No stats event fired for this
// damage (same "not a real attacker" treatment as Building's own decay
// damage -- the dying entity is already gone by the time this runs, so
// there's no attacker identity left to attribute a DamageDealtEvent to).
//
// `onHit` is the same optional slot AreaSpell already carries, for the same
// reason: Ice Golem's death explosion SLOWS what it damages. nullptr -- the
// default, and every caller that predates it -- leaves behaviour
// bit-identical; the loop below gains one null-guarded call after the damage
// lands and nothing else.
//
// It exists because without it the only place to hang Ice Golem's slow was its
// ATTACK, a mechanic the real card does not have -- see the registry entry for
// card id 40. On-hit effects only mean anything against a CombatEntity, so the
// narrowing cast happens here, the one place that needs it, exactly as
// AreaSpell::update and Projectile::applyHit already do.
class AreaDamageOnDeath : public IDeathEffect {
    float radius;
    int damage;
    std::shared_ptr<IOnHitEffect> onHit;

public:
    AreaDamageOnDeath(float radius, int damage, std::shared_ptr<IOnHitEffect> onHit = nullptr)
        : radius(radius), damage(damage), onHit(std::move(onHit)) {}

    void apply(Board& board, const Vector2D& position, int team) const override {
        for (const auto& entity : board.getEntities()) {
            if (entity->team == team || !entity->isAlive() || !entity->isTargetable()) continue;
            if (position.distanceTo(entity->position) > radius) continue;
            entity->takeDamage(damage);
            if (onHit) {
                if (auto combatTarget = std::dynamic_pointer_cast<CombatEntity>(entity)) {
                    onHit->apply(combatTarget);
                }
            }
        }
    }
};
