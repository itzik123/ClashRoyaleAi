#pragma once
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unordered_map>

struct Vector2D {
    float x, y;

    float distanceTo(const Vector2D& other) const {
        return std::sqrt((other.x - x) * (other.x - x) + (other.y - y) * (other.y - y));
    }
};

class Board;

class Entity {
public:
    // The radius a troop (getCollisionRadius() <= 0) occupies for attack-range
    // math and push-apart physics.
    static constexpr float IMPLICIT_TROOP_RADIUS = 0.4f;

    int id;
    Vector2D position;
    int hp;
    int team;
    char symbol;
    // Display name (e.g. "Knight", "King Tower"), set by whoever builds the
    // entity.
    std::string name;

    // The CardRegistry card that produced this entity, or a reserved negative
    // id for towers (GameManager::TOWER_KING_ID / TOWER_PRINCESS_ID). -1 means
    // never assigned.
    int cardId = -1;

    // A plain field, not a virtual: findTarget and resolveCollisions read it in
    // tight loops.
    bool isFlying = false;

    Entity(int id, float x, float y, int hp, int team, char symbol = '?')
        : id(id), position{ x, y }, hp(hp), team(team), symbol(symbol) {}

    virtual ~Entity() = default;

    virtual void update(Board& board) = 0;

    virtual void takeDamage(int amount) { hp -= amount; }

    bool isAlive() const { return hp > 0; }

    virtual bool isTargetable() const { return true; }

    // True only for Building and Tower. Lets code outside the Building chain
    // (knockback, pulls) ask without a dynamic_cast, which would need
    // Building.h and create an include cycle.
    virtual bool isBuilding() const { return false; }

    // True only for Tower; lets findTarget treat towers as an always-visible
    // fallback without a cast.
    virtual bool isTower() const { return false; }

    virtual float getCollisionRadius() const { return 0.0f; }

    // Footprint for target-selection distance only, never collision or attack
    // range. A Crown Tower's is larger than its body, since units close on its
    // edge (see Tower::getTargetingRadius).
    virtual float getTargetingRadius() const { return getCollisionRadius(); }

    // Re-applies board bounds and river constraints. Only Troop overrides it;
    // called again after collision resolution, which ignores them.
    virtual void clampPosition(Board& board) { (void)board; }

    // Called once by Board::cleanDeadEntities as a dead entity is removed (e.g.
    // Golem spawning Golemites).
    virtual void onDeath(Board& board) { (void)board; }

    // Called on every other living entity once per death, alongside onDeath
    // (Skeleton King's souls). deathPosition and deadTeam describe the entity
    // that died.
    virtual void onNearbyDeath(Board& board, const Vector2D& deathPosition, int deadTeam) {
        (void)board; (void)deathPosition; (void)deadTeam;
    }

    // Clone spell: a full copy with a fresh id and 1 hp, as the real card does.
    // nullptr for anything the real card cannot clone.
    virtual std::shared_ptr<Entity> clone(int newId) const { (void)newId; return nullptr; }

    // An exact copy for Board::deepCopy, used by decision-time search. Unlike
    // clone(), it keeps hp and the original id, which the remap table and
    // id-keyed members join on.
    //
    // Implementations are make_shared<T>(*this). Sharing effect pointers is
    // correct: every effect interface is const and stateless.
    //
    // Throws rather than returning nullptr, so a new concrete Entity without an
    // override fails by name instead of silently vanishing from every copied
    // board.
    virtual std::shared_ptr<Entity> snapshot() const {
        throw std::logic_error(
            std::string("Entity::snapshot() not implemented for concrete type '") +
            typeid(*this).name() +
            "'. Every concrete Entity must override snapshot() or Board::deepCopy() "
            "would silently drop it from the copied board.");
    }

    // Second pass of Board::deepCopy, run once every entity is copied and the
    // old-id map is complete. A virtual because Projectile.h includes Board.h,
    // so Board cannot cast to Projectile.
    //
    // Projectile::target is the only entity-pointer member in the hierarchy;
    // every other persistent reference is an int id, preserved by snapshot(). A
    // type that gains a pointer member must override this.
    virtual void remapSnapshotReferences(
        const std::unordered_map<int, std::shared_ptr<Entity>>& byOldId) {
        (void)byOldId;
    }
};

// Repositioning helpers shared by every pull and push (hooks, Tornado,
// knockback). None clamps to the board; callers call clampPosition()
// afterwards.
//
// All of them are no-ops on a building, enforced here once rather than at each
// call site. Nothing may move an entity by writing `position` directly.
inline bool exemptFromForcedMovement(const Entity& entity) {
    return entity.isBuilding();
}

// Moves `entity` up to `distance` tiles toward `point`, never past it. A
// non-positive distance is a no-op: callers compute "how far left to close"
// (e.g. dist - meleeRange), which goes negative against a close target and
// would otherwise pull backwards.
inline void pullToward(Entity& entity, const Vector2D& point, float distance) {
    if (exemptFromForcedMovement(entity)) return;
    if (distance <= 0.0f) return; // nothing to close
    float dist = entity.position.distanceTo(point);
    if (dist <= 0.01f) return; // coincident: no direction
    float moveBy = (distance < dist) ? distance : dist;
    entity.position.x += (point.x - entity.position.x) / dist * moveBy;
    entity.position.y += (point.y - entity.position.y) / dist * moveBy;
}

// Mirrors `entity` into the opposite lane (Mighty Miner's escape, Hero Giant's
// hurl), with the same building guard. Takes the width so a non-default test
// board mirrors about its own centre.
inline void mirrorToOppositeLane(Entity& entity, int boardWidth) {
    if (exemptFromForcedMovement(entity)) return;
    entity.position.x = static_cast<float>(boardWidth - 1) - entity.position.x;
}

// Moves `entity` exactly `distance` tiles directly away from `point`
// (knockback).
inline void pushAway(Entity& entity, const Vector2D& point, float distance) {
    if (exemptFromForcedMovement(entity)) return;
    if (distance <= 0.0f) return; // see pullToward
    float dist = entity.position.distanceTo(point);
    if (dist <= 0.01f) return; // coincident: no direction
    entity.position.x += (entity.position.x - point.x) / dist * distance;
    entity.position.y += (entity.position.y - point.y) / dist * distance;
}

// Moves `entity` `distance` tiles along an explicit direction, with the same
// guards. For the rolling spells' lateral throw: where a unit is caught across
// the corridor decides its direction, which a radial push from the log's centre
// cannot express. The direction need not be normalised; zero length is a no-op.
inline void pushAlong(Entity& entity, float dirX, float dirY, float distance) {
    if (exemptFromForcedMovement(entity)) return;
    if (distance <= 0.0f) return; // see pullToward
    float len = std::sqrt(dirX * dirX + dirY * dirY);
    if (len <= 0.0001f) return; // no direction
    entity.position.x += dirX / len * distance;
    entity.position.y += dirY / len * distance;
}