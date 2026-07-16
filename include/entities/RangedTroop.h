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
        auto arrow = std::make_shared<Projectile>(
            board.allocateId(), position.x, position.y, team, target, 1.5f, damage);
        board.addEntity(arrow);
    }
};