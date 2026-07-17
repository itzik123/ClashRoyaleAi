#pragma once
#include <cmath>
#include <string>

struct Vector2D {
    float x, y;

    float distanceTo(const Vector2D& other) const {
        return std::sqrt((other.x - x) * (other.x - x) + (other.y - y) * (other.y - y));
    }
};

class Board;

class Entity {
public:
    // Fallback collision radius used everywhere a troop (getCollisionRadius()
    // <= 0, i.e. not a Building/Tower) needs to be treated as occupying some
    // physical space -- attack-range math, and push-apart physics in Board.
    // Named so it's asserted once instead of retyped as a bare 0.4f (or a
    // derived 0.8f for two troops) in each place that needs it.
    static constexpr float IMPLICIT_TROOP_RADIUS = 0.4f;

    int id;
    Vector2D position;
    int hp;
    int team;
    char symbol;
    // Human-readable display name (e.g. "Knight", "King Tower"). Empty by
    // default -- set by whoever actually knows it at construction time
    // (CardFactories, GameManager's tower setup), rather than threaded
    // through every derived class's constructor.
    std::string name;

    // Plain field, not a virtual/CombatEntity member: both
    // CombatEntity::findTarget() and Board::resolveCollisions() need to read
    // this on arbitrary Entity candidates in tight per-tick loops, and a
    // field read needs no cast or virtual dispatch. Buildings/AreaSpell/
    // Projectile simply never set it true.
    bool isFlying = false;

    Entity(int id, float x, float y, int hp, int team, char symbol = '?')
        : id(id), position{ x, y }, hp(hp), team(team), symbol(symbol) {}

    virtual ~Entity() = default;

    virtual void update(Board& board) = 0;

    virtual void takeDamage(int amount) { hp -= amount; }

    bool isAlive() const { return hp > 0; }

    virtual bool isTargetable() const { return true; }

    virtual float getCollisionRadius() const { return 0.0f; }

    // Re-applies board bounds/river constraints to this entity's position.
    // Default no-op: only Troop (the only thing that ever moves) overrides
    // it. Public and Board-aware so it can be called again, uniformly,
    // after collision resolution -- which itself doesn't respect those
    // constraints -- without the caller needing to know the concrete type.
    virtual void clampPosition(Board& board) { (void)board; }
};