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

    std::shared_ptr<Entity> clone(int newId) const override {
        auto copy = std::make_shared<RangedBuildingTargeter>(*this);
        copy->id = newId;
        copy->hp = 1; // Clone: full damage, 1 hp
        return copy;
    }

    // Board::deepCopy. Its own override, not BuildingTargeter's inherited one:
    // make_shared<BuildingTargeter>(*this) would compile happily and SLICE
    // this back to its base, silently downgrading a ranged attacker into a
    // melee one inside every rollout. Same reason Tower needs one separately
    // from Building.
    std::shared_ptr<Entity> snapshot() const override {
        return std::make_shared<RangedBuildingTargeter>(*this);
    }

protected:
    void performAttack(Board& board, std::shared_ptr<Entity> target) override {
        auto proj = std::make_shared<Projectile>(
            board.allocateId(), position.x, position.y, team, target, 1.5f, getCurrentDamage(), onHitEffects,
            false, 0, id, cardId, splashRadius);
        board.addEntity(proj);
    }
};
