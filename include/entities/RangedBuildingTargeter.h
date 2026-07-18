#pragma once
#include "BuildingTargeter.h"
#include "Projectile.h"

// A building-only-targeting troop whose attack is a projectile instead of
// direct melee damage. Targeting (ignore troops, chase the closest enemy
// building) is inherited as-is from BuildingTargeter.
class RangedBuildingTargeter : public BuildingTargeter {
public:
    RangedBuildingTargeter(int id, float x, float y, int hp, int team,
        float speed, float attackRange, int damage, int attackCooldown, char symbol)
        : BuildingTargeter(id, x, y, hp, team, speed, attackRange, damage, attackCooldown, symbol) {}

protected:
    void performAttack(Board& board, std::shared_ptr<Entity> target) override {
        auto proj = std::make_shared<Projectile>(
            board.allocateId(), position.x, position.y, team, target, 1.5f, getCurrentDamage(), onHitEffects);
        board.addEntity(proj);
    }
};
