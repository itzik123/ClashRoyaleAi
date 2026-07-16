#pragma once
#include <cmath>

struct Vector2D {
    float x, y;

    float distanceTo(const Vector2D& other) const {
        return std::sqrt((other.x - x) * (other.x - x) + (other.y - y) * (other.y - y));
    }
};

class Board;

class Entity {
public:
    int id;
    Vector2D position;
    int hp;
    int team;
    char symbol;
    int freezeTicks = 0;
    float freezeSlow = 1.0f;

    Entity(int id, float x, float y, int hp, int team, char symbol = '?')
        : id(id), position{ x, y }, hp(hp), team(team), symbol(symbol) {}

    virtual ~Entity() = default;

    virtual void update(Board& board) = 0;

    virtual void takeDamage(int amount) { hp -= amount; }

    bool isAlive() const { return hp > 0; }

    virtual bool isTargetable() const { return true; }

    virtual float getCollisionRadius() const { return 0.0f; }

    void applyFreeze(int ticks, float slowFactor) {
        if (ticks > freezeTicks) {
            freezeTicks = ticks;
            freezeSlow = slowFactor;
        }
    }
};