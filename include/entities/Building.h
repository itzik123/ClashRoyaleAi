#pragma once
#include "CombatEntity.h"

class Building : public CombatEntity {
protected:
    int maxHp;
    int lifetimeTicks;
    int ticksAlive = 0;

public:
    Building(int id, float x, float y, int hp, int team, char symbol,
        float attackRange, int damage, int attackCooldown, int lifetime = 300)
        : CombatEntity(id, x, y, hp, team, symbol, attackRange, damage, attackCooldown), 
        maxHp(hp), lifetimeTicks(lifetime) {}

    float getCollisionRadius() const override { return 1.0f; }

    void update(Board& board) override {
        CombatEntity::update(board);
        if (lifetimeTicks > 0) {
            ticksAlive++;
            if (ticksAlive % 10 == 0) { // Decay every 1 second (10 ticks)
                int decayAmount = maxHp / (lifetimeTicks / 10);
                if (decayAmount <= 0) decayAmount = 1;
                takeDamage(decayAmount);
            }
        }
    }

protected:
    void performAttack(Board& board, std::shared_ptr<Entity> target) override {
        target->takeDamage(damage);
    }
};