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

    // Which CardRegistry card produced this entity (CardStats::id), or a
    // reserved negative sentinel for entities that aren't in CardRegistry at
    // all (Towers -- see GameManager::TOWER_KING_ID/TOWER_PRINCESS_ID). -1
    // (the default) means "never assigned", which nothing should treat as a
    // valid card. Exists so stats collectors can key by a reliable int
    // instead of the coincidental, unenforced `name` string.
    int cardId = -1;

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

    // Called once, by Board::cleanDeadEntities(), the instant this entity is
    // found dead and about to be removed (e.g. Golem spawning two Golemites).
    // Default no-op, same shape as clampPosition: lets Board trigger this
    // generically on every Entity without needing to know which concrete
    // type -- or even whether it's a CombatEntity, the only thing that ever
    // actually has a death effect to fire -- it's looking at.
    virtual void onDeath(Board& board) { (void)board; }

    // Clone spell: makes a full copy of this entity (every configured
    // combat field -- splash, charge, on-hit effects, all of it -- via
    // each concrete subclass's own implicit copy constructor) with a
    // fresh id and 1 hp, matching the real card's "duplicates all troops
    // in radius, clones have 1 hp but full damage" rule. Default returns
    // nullptr -- most entities (buildings, spells, projectiles) were never
    // valid Clone targets in the real game either.
    virtual std::shared_ptr<Entity> clone(int newId) const { (void)newId; return nullptr; }
};

// Repositioning helpers shared by every pull/push mechanic (Fisherman's
// hook, Tornado's pull, Bowler/Fireball/Rocket/Giant Snowball's knockback)
// -- free functions since they're just geometry, needed from entity
// classes and spell classes alike. Neither clamps to board bounds; callers
// that need that already call clampPosition() separately afterward (same
// as normal movement).

// Moves `entity` up to `distance` tiles toward `point`, never overshooting
// past it.
inline void pullToward(Entity& entity, const Vector2D& point, float distance) {
    float dist = entity.position.distanceTo(point);
    if (dist <= 0.01f) return; // already there (or coincident): no direction to move in
    float moveBy = (distance < dist) ? distance : dist;
    entity.position.x += (point.x - entity.position.x) / dist * moveBy;
    entity.position.y += (point.y - entity.position.y) / dist * moveBy;
}

// Moves `entity` exactly `distance` tiles directly away from `point`
// (knockback) -- no "overshoot" concept the other direction, so no clamp.
inline void pushAway(Entity& entity, const Vector2D& point, float distance) {
    float dist = entity.position.distanceTo(point);
    if (dist <= 0.01f) return; // coincident: no direction to push in
    entity.position.x += (entity.position.x - point.x) / dist * distance;
    entity.position.y += (entity.position.y - point.y) / dist * distance;
}