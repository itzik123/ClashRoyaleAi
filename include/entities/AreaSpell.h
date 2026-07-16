#pragma once
#include "Entity.h"
#include "Board.h"

class AreaSpell : public Entity {
private:
    float radius;
    int damage;
    int delayTicks;

public:
    AreaSpell(int id, float x, float y, int team, float radius, int damage, int delayTicks, char symbol = '*')
        : Entity(id, x, y, 1, team, symbol), radius(radius), damage(damage), delayTicks(delayTicks) {}

    bool isTargetable() const override { return false; }

    void update(Board& board) override {
        if (delayTicks > 0) {
            delayTicks--;
            return;
        }

        for (const auto& entity : board.getEntities()) {
            if (entity->isAlive() && entity->isTargetable() && entity->team != this->team && entity->id != this->id) {
                float dist = position.distanceTo(entity->position);
                if (dist <= radius) {
                    entity->takeDamage(damage);
                }
            }
        }

        hp = 0;
    }
};