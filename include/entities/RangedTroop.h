#pragma once
#include "Troop.h"
#include "Projectile.h"

class RangedTroop : public Troop {
public:
    RangedTroop(int id, float x, float y, int hp, int team,
        float speed, float attackRange, int damage, int attackCooldown, char symbol = 'A')
        : Troop(id, x, y, hp, team, symbol, speed, attackRange, damage, attackCooldown) {}

protected:
    void performAttack(Board& board, std::shared_ptr<Entity> target) override {
        // On-hit effects ride along with the shot and land when it does,
        // instead of applying instantly at the moment of firing.
        auto arrow = std::make_shared<Projectile>(
            board.allocateId(), position.x, position.y, team, target, 1.5f, damage, onHitEffects);
        board.addEntity(arrow);
    }
};