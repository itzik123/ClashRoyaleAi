#pragma once
#include "CardEntity.h"
#include "CombatEntity.h"
#include "OnHitEffect.h"
#include "Board.h"
#include <memory>

class AreaSpell : public CardEntity {
private:
    float radius;
    int damage;
    int delayTicks;
    std::shared_ptr<IOnHitEffect> onHit;
    // Most spells hit Air & Ground alike (Fireball, Zap, Poison, Rocket,
    // Lightning...); a handful (The Log, Barbarian Barrel) are ground-only
    // control spells that roll along the arena floor. Defaults to false
    // (hits everyone) to match the majority, with ground-only spells opting
    // in via CardStats::withGroundOnly.
    bool groundOnly;

    // Multi-tick spells (Poison's 8 ticks over 8s, Arrows' 3 rapid volleys):
    // remainingHits counts down one application at a time, re-evaluating who
    // is currently in radius on *each* application -- exactly like the real
    // game, a unit can walk out of a Poison cloud partway through and stop
    // taking damage. remainingHits == 1 (the default) is every other
    // spell's normal single-shot case; tickInterval is only relevant when
    // remainingHits > 1, and doubles delayTicks as the gap between hits.
    int remainingHits;
    int tickInterval;

public:
    AreaSpell(int id, float x, float y, int team, float radius, int damage, int delayTicks, char symbol = '*',
        std::shared_ptr<IOnHitEffect> onHit = nullptr, bool groundOnly = false,
        int remainingHits = 1, int tickInterval = 0)
        : CardEntity(id, x, y, 1, team, symbol), radius(radius), damage(damage), delayTicks(delayTicks),
        onHit(std::move(onHit)), groundOnly(groundOnly), remainingHits(remainingHits), tickInterval(tickInterval) {}

    bool isTargetable() const override { return false; }

    void update(Board& board) override {
        if (delayTicks > 0) {
            delayTicks--;
            return;
        }

        for (const auto& entity : board.getEntities()) {
            if (entity->isAlive() && entity->isTargetable() && entity->team != this->team && entity->id != this->id
                && (!groundOnly || !entity->isFlying)) {
                float dist = position.distanceTo(entity->position);
                if (dist <= radius) {
                    entity->takeDamage(damage);
                    // Same surgical cast as CombatEntity::applyOnHitEffects and
                    // Projectile's arrival handler -- on-hit effects only ever
                    // mean something against a CombatEntity, so this is the one
                    // place AreaSpell needs to know that, not its own type.
                    if (onHit) {
                        auto combatTarget = std::dynamic_pointer_cast<CombatEntity>(entity);
                        if (combatTarget) onHit->apply(combatTarget);
                    }
                }
            }
        }

        remainingHits--;
        if (remainingHits > 0) {
            delayTicks = tickInterval; // wait out the gap, then apply again
        } else {
            hp = 0;
        }
    }
};