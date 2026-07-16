#pragma once
#include "Building.h"
#include "Projectile.h"

class Tower : public Building {
public:
    Tower(int id, float x, float y, int hp, int team,
        float attackRange, int damage, int attackCooldown, char symbol)
        : Building(id, x, y, hp, team, symbol, attackRange, damage, attackCooldown, -1) {}

    float getCollisionRadius() const override {
        return (symbol == 'R') ? 2.0f : 1.5f;
    }

protected:
    void performAttack(Board& board, std::shared_ptr<Entity> target) override {
        auto arrow = std::make_shared<Projectile>(
            board.allocateId(), position.x, position.y, team, target, 2.0f, damage);
        board.addEntity(arrow);
    }
};