#pragma once
#include "BuildingTargeter.h"
#include "Projectile.h"

// A building-targeting troop that attacks with a projectile. Targeting is
// BuildingTargeter's.
class RangedBuildingTargeter : public BuildingTargeter {
public:
    RangedBuildingTargeter(int id, float x, float y, int hp, int team,
        float speed, float attackRange, int damage, int attackCooldown, char symbol)
        : BuildingTargeter(id, x, y, hp, team, speed, attackRange, damage, attackCooldown, symbol) {}

    std::shared_ptr<Entity> clone(int newId) const override {
        auto copy = std::make_shared<RangedBuildingTargeter>(*this);
        copy->id = newId;
        copy->hp = 1; // Clone: full damage, 1 hp
        return copy;
    }

    // Its own override: BuildingTargeter's would slice this to the base class
    // and turn a ranged attacker melee inside every rollout.
    std::shared_ptr<Entity> snapshot() const override {
        return std::make_shared<RangedBuildingTargeter>(*this);
    }

protected:
    void performAttack(Board& board, std::shared_ptr<Entity> target) override {
        // lineSplash/lineSplashRange forwarded, as RangedTroop does; dropping
        // them silently turns a piercing shot into circular splash.
        auto proj = std::make_shared<Projectile>(
            board.allocateId(), position.x, position.y, team, target, 1.5f, getCurrentDamage(), onHitEffects,
            false, 0, id, cardId, splashRadius, lineSplash, lineSplashRange);
        board.addEntity(proj);
    }
};
