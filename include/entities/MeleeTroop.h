#pragma once
#include "Troop.h"

class MeleeTroop : public Troop {
public:
    MeleeTroop(int id, float x, float y, int hp, int team,
        float speed, float attackRange, int damage, int attackCooldown, char symbol)
        : Troop(id, x, y, hp, team, symbol, speed, attackRange, damage, attackCooldown) {}

protected:
    void performAttack(Board& board, std::shared_ptr<Entity> target) override {
        target->takeDamage(getCurrentDamage());
        applyOnHitEffects(target);
    }
};