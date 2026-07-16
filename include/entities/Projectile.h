#pragma once
#include "Entity.h"
#include "Board.h"
#include <memory>

class Projectile : public Entity {
private:
    std::weak_ptr<Entity> target;
    float speed;
    int damage;

public:
    Projectile(int id, float x, float y, int team, std::weak_ptr<Entity> target, float speed, int damage)
        : Entity(id, x, y, 1, team, '-'), target(target), speed(speed), damage(damage) {}

    bool isTargetable() const override { return false; }

    void update(Board& board) override {
        if (auto t = target.lock()) {
            if (!t->isAlive()) {
                hp = 0;
                return;
            }

            float dist = position.distanceTo(t->position);
            if (dist <= speed) {
                t->takeDamage(damage);
                hp = 0;
            } else {
                float dx = t->position.x - position.x;
                float dy = t->position.y - position.y;
                position.x += (dx / dist) * speed;
                position.y += (dy / dist) * speed;
            }
        } else {
            hp = 0;
        }
    }
};