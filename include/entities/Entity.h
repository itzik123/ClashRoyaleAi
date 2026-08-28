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

    // True only for Building (and its Tower subclass) -- lets code outside
    // the Building/CombatEntity inheritance chain (e.g. knockback/pull
    // effects) check "is this a stationary building" without a
    // dynamic_cast<Building*>, which would need Building.h and create a
    // circular include from anywhere inside CombatEntity.h itself.
    // Overridden to return true in Building.h.
    virtual bool isBuilding() const { return false; }

    // True only for Tower (King/Princess) -- lets CombatEntity::findTarget
    // treat towers as an always-visible fallback destination (see
    // sightRange's own comment) without a dynamic_cast<Tower*>, same
    // circular-include reasoning as isBuilding() above. Overridden to
    // return true in Tower.h.
    virtual bool isTower() const { return false; }

    virtual float getCollisionRadius() const { return 0.0f; }

    // Footprint used for TARGET SELECTION distance only (never collision,
    // never attack range). Zero for everything except a Crown Tower: a tower
    // is a large structure and a unit closes on its edge, so comparing it to a
    // small building by centre distance overstates how far away it is. See
    // Tower::getTargetingRadius for the empirical fit.
    virtual float getTargetingRadius() const { return getCollisionRadius(); }

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

    // Called on every OTHER still-alive entity, once per death, right
    // alongside onDeath() above (see Board::cleanDeadEntities) -- lets a
    // "collects something whenever anything dies nearby" mechanic (Skeleton
    // King's souls) react without Board needing to know it's CombatEntity-
    // shaped enough to have one. Default no-op, same idiom as onDeath/
    // clampPosition. deathPosition/deadTeam describe whoever just died, not
    // the entity this is called on.
    virtual void onNearbyDeath(Board& board, const Vector2D& deathPosition, int deadTeam) {
        (void)board; (void)deathPosition; (void)deadTeam;
    }

    // Clone spell: makes a full copy of this entity (every configured
    // combat field -- splash, charge, on-hit effects, all of it -- via
    // each concrete subclass's own implicit copy constructor) with a
    // fresh id and 1 hp, matching the real card's "duplicates all troops
    // in radius, clones have 1 hp but full damage" rule. Default returns
    // nullptr -- most entities (buildings, spells, projectiles) were never
    // valid Clone targets in the real game either.
    virtual std::shared_ptr<Entity> clone(int newId) const { (void)newId; return nullptr; }

    // Exact copy of this entity for Board::deepCopy() -- decision-time search
    // rolls candidate actions forward on a copied board and keeps the best
    // (see CLAUDE.md open problem #2). Distinct from clone() directly above,
    // which CANNOT be reused for this despite looking almost identical:
    //
    //   * clone() sets hp = 1, because it implements the Clone *card's* rule
    //     ("duplicates have 1 hp but full damage"). A snapshot must preserve hp
    //     exactly, or the search reasons about a board that never existed.
    //   * clone() also assigns a FRESH id. A snapshot must keep the original
    //     id, since that is what Board::deepCopy's remap table and every
    //     id-keyed member (currentTargetId, forcedTargetEntityId) join on.
    //
    // Implementations are one line -- make_shared<T>(*this) -- relying on each
    // concrete type's implicit copy constructor, exactly as clone() already
    // does. Sharing the effect pointers (onHitEffects, deathEffect,
    // periodicEffect, ...) that the implicit copy carries over is CORRECT and
    // deliberate: every effect interface declares apply() const and none holds
    // mutable state, so they are stateless strategy objects and deep-copying
    // them would be wasted work. See UPSTREAM_REQUESTS.md item 13.
    //
    // THROWS rather than returning nullptr, which is the whole point of it
    // existing separately. clone()'s nullptr default is safe there -- a
    // building was never a legal Clone target anyway -- but the same default
    // here would let a copied board silently lose every tower, building, spell
    // and projectile and still look like a working snapshot. A search that
    // then distilled those futures back into the policy would be training on
    // positions that cannot occur. Loud beats silent: any concrete Entity
    // added later fails immediately and by name instead of quietly punching a
    // hole in every rollout.
    virtual std::shared_ptr<Entity> snapshot() const {
        throw std::logic_error(
            std::string("Entity::snapshot() not implemented for concrete type '") +
            typeid(*this).name() +
            "'. Every concrete Entity must override snapshot() or Board::deepCopy() "
            "would silently drop it from the copied board.");
    }

    // Second pass of Board::deepCopy, run on every copied entity once all of
    // them exist and the old-id -> new-entity map is complete (it cannot run
    // during the copy loop itself: a projectile may be homing on an entity
    // that has not been copied yet).
    //
    // Default no-op, same shape as onDeath/clampPosition/onNearbyDeath above
    // -- it lets Board repair cross-board references generically without
    // knowing which concrete type it is looking at, which here is not just
    // tidiness: Projectile.h includes Board.h, so Board CANNOT include
    // Projectile.h to dynamic_cast for it.
    //
    // Projectile is the only override, because Projectile::target is the only
    // entity-pointer MEMBER in the hierarchy. Every other shared_ptr<Entity>
    // (CombatEntity::update's local, findTarget/resolveCurrentTarget's
    // returns, the targeting helpers) is a per-tick local that never outlives
    // the tick that computed it, and every persistent reference to another
    // entity is stored as a plain int id (currentTargetId,
    // forcedTargetEntityId) which needs no remapping at all -- ids are
    // preserved exactly by snapshot(). If a future entity type gains a
    // pointer member, it must override this too.
    virtual void remapSnapshotReferences(
        const std::unordered_map<int, std::shared_ptr<Entity>>& byOldId) {
        (void)byOldId;
    }
};

// Repositioning helpers shared by every pull/push mechanic (Fisherman's
// hook, Tornado's pull, Bowler/Fireball/Rocket/Giant Snowball's knockback,
// Evolved Valkyrie's Whirlwind Axe) -- free functions since they're just
// geometry, needed from entity classes and spell classes alike. Neither
// clamps to board bounds; callers that need that already call
// clampPosition() separately afterward (same as normal movement).
//
// Both are a no-op on a Building (Entity::isBuilding) regardless of which
// mechanic is calling -- buildings are stationary, full stop, whether the
// pull/push comes from a spell's knockback, a hook, or anything else that
// exists now or gets added later. Enforced here, once, rather than at
// every individual call site, so nothing can reintroduce this bug by
// forgetting a per-site check. Self-directed calls (Mega Knight's jump,
// Golden Knight's dash both move `entity` == the attacker itself toward
// its target) are unaffected in practice -- no card that moves itself
// this way is ever a building.
inline bool exemptFromForcedMovement(const Entity& entity) {
    return entity.isBuilding();
}

// Moves `entity` up to `distance` tiles toward `point`, never overshooting
// past it.
//
// A NON-POSITIVE `distance` is a no-op, and that guard is load-bearing rather
// than defensive tidiness. Every caller computes it as "how far do I still
// have to close", e.g. `dist - meleeRange`, and when the target is ALREADY
// inside that range the subtraction goes negative -- at which point
// `moveBy = (distance < dist) ? distance : dist` picks the negative value and
// the pull runs backwards, shoving the entity AWAY. GoldenKnightDashEffect
// did exactly that against an adjacent enemy, and is registered with 10
// dashes. Enforced here, once, for the same reason exemptFromForcedMovement
// is: a per-call-site clamp is a check somebody will forget to add.
// Something that genuinely wants to move away calls pushAway.
inline void pullToward(Entity& entity, const Vector2D& point, float distance) {
    if (exemptFromForcedMovement(entity)) return;
    if (distance <= 0.0f) return; // already close enough: nothing to close
    float dist = entity.position.distanceTo(point);
    if (dist <= 0.01f) return; // already there (or coincident): no direction to move in
    float moveBy = (distance < dist) ? distance : dist;
    entity.position.x += (point.x - entity.position.x) / dist * moveBy;
    entity.position.y += (point.y - entity.position.y) / dist * moveBy;
}

// Mirrors `entity` into the opposite lane (Mighty Miner's Explosive Escape,
// Hero Giant's Hurl). Third member of this family, and it exists for the same
// reason as the other two: both callers previously wrote
// `victim->position.x = board.getWidth() - 1 - victim->position.x` DIRECTLY,
// which is a raw position write and therefore skipped
// exemptFromForcedMovement entirely -- so Hero Giant could hurl an enemy
// Cannon across the arena, the exact bug the guard on pullToward/pushAway
// exists to make impossible. It also removes two more restatements of the
// mirror formula, which on the standard 18-wide board is ArenaLayout::mirrorX.
//
// Takes the width rather than reading ArenaLayout so a Board constructed at a
// non-default size (tests do) mirrors about ITS OWN centre, exactly as the
// two call sites did before.
inline void mirrorToOppositeLane(Entity& entity, int boardWidth) {
    if (exemptFromForcedMovement(entity)) return;
    entity.position.x = static_cast<float>(boardWidth - 1) - entity.position.x;
}

// Moves `entity` exactly `distance` tiles directly away from `point`
// (knockback) -- no "overshoot" concept the other direction, so no clamp.
inline void pushAway(Entity& entity, const Vector2D& point, float distance) {
    if (exemptFromForcedMovement(entity)) return;
    if (distance <= 0.0f) return; // same guard as pullToward -- see there
    float dist = entity.position.distanceTo(point);
    if (dist <= 0.01f) return; // coincident: no direction to push in
    entity.position.x += (entity.position.x - point.x) / dist * distance;
    entity.position.y += (entity.position.y - point.y) / dist * distance;
}

// Moves `entity` exactly `distance` tiles along an EXPLICIT direction, rather
// than along the line from some point. Fourth member of the family, and it
// carries the same two guards for the same reasons -- see pullToward.
//
// WHY A DIRECTION AND NOT A POINT (2026-08-28). A rolling spell's knockback is
// not radial. The Log sweeps a rectangular corridor, and where it catches a
// unit ACROSS that corridor decides which way the unit is thrown: dead centre
// is shoved forward along the roll, at the left or right edge it is flung
// sideways. That lateral throw is the card's whole tactical point -- it is what
// splits a grouped push apart -- and `pushAway(entity, logCentre, d)` cannot
// express it. Pushing away from the log's centre POINT does produce some
// sideways motion, but its magnitude falls off with longitudinal distance
// rather than with lateral offset, so a unit level with the log and one at its
// nose get thrown the same way. The caller computes the blend and hands the
// unit vector here.
//
// `dirX`/`dirY` need not be normalised; a zero-length direction is a no-op
// rather than a division by zero.
inline void pushAlong(Entity& entity, float dirX, float dirY, float distance) {
    if (exemptFromForcedMovement(entity)) return;
    if (distance <= 0.0f) return; // same guard as pullToward -- see there
    float len = std::sqrt(dirX * dirX + dirY * dirY);
    if (len <= 0.0001f) return; // no direction to push in
    entity.position.x += dirX / len * distance;
    entity.position.y += dirY / len * distance;
}