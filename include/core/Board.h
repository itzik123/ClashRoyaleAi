#pragma once
#include <vector>
#include <memory>
#include <algorithm>
#include <unordered_map>
#include "ArenaLayout.h"
#include "Entity.h"
#include "StatsEventBus.h"

class Board {
private:
    int width, height;
    std::vector<std::shared_ptr<Entity>> activeEntities;
    std::vector<std::shared_ptr<Entity>> pendingEntities;
    int idCounter = 1000;

    // The river band [15.5, 17.5), centred on 16.5 so it is symmetric under the
    // tower layout's y -> 33 - y mirror. Both teams' own-half placement bound
    // then sits at the same mirrored row.
    float riverY_start = 15.5f;
    float riverY_end = 17.5f;
    // Bridge centres from ArenaLayout: 2.5 and 14.5, on the seam between each
    // bridge's two tiles (see ArenaLayout.h).
    Vector2D leftBridge{ ArenaLayout::LEFT_BRIDGE_X, ArenaLayout::BRIDGE_Y };
    Vector2D rightBridge{ ArenaLayout::RIGHT_BRIDGE_X, ArenaLayout::BRIDGE_Y };

    // Scratch for resolveCollisions and the colliders() cache; members so the
    // allocation happens once per Board. Neither holds meaning between calls.
    struct Body {
        Entity* entity;
        float radius;
        bool building;
        bool troop;
        bool flying;
        bool alive;
    };
    struct Collider { size_t index; float radius; };
    mutable std::vector<Body> bodyScratch;
    mutable std::vector<Collider> colliderCache;
    mutable bool collidersDirty = true;

    // The arena is 18x34: one row behind each King is dead space except a
    // centred opening of real ground (isBackRowDeadZone).
    static constexpr float BACK_ROW_OPENING_HALF_WIDTH = 3.0f;

public:
    // Half a bridge's two-tile width; correct only with seam-centred bridges.
    static constexpr float BRIDGE_HALF_WIDTH = 1.0f;

    // Cell i covers [i - 0.5, i + 0.5], so the board's physical extent is x in
    // [-0.5, width - 0.5] and y likewise. isValidPlacement's footprint check
    // needs this physical edge: checking footprints against the index range
    // would reject a troop at both edge columns.
    static constexpr float CELL_HALF_EXTENT = 0.5f;

private:

public:
    // Public: every fire site already holds a Board&.
    StatsEventBus statsEvents;

    // Set once per tick by GameManager::step, read deep in the
    // update/performAttack chain to stamp DamageDealtEvent without threading a
    // tick through every override.
    int currentTick = 0;

    // Elixir grants (Elixir Collector) dropped here from deep inside update();
    // GameManager::step drains them into the players' elixir each tick.
    float pendingElixirGrant[2] = { 0.0f, 0.0f };

    Board(int w = 18, int h = 34) : width(w), height(h) {}

    int allocateId() { return idCounter++; }

    // A fully independent copy for decision-time search: every entity is
    // duplicated via Entity::snapshot(), so the copy can be stepped without
    // touching the original. Copy-constructing a Board is still a shallow copy
    // that shares entities.
    //
    // Not carried across:
    //   * `statsEvents` starts empty. The collectors are stateful, and a rollout's hits would land in the live match's statistics, which feed the reward. GameManager::snapshot copies them properly.
    //   * GameManager's own state (tick, game over, elixir), which GameManager::snapshot handles.
    Board deepCopy() const {
        Board copy(width, height);

        // Geometry copied explicitly rather than trusting the in-class
        // initialisers to still agree.
        copy.riverY_start = riverY_start;
        copy.riverY_end = riverY_end;
        copy.leftBridge = leftBridge;
        copy.rightBridge = rightBridge;

        // idCounter carries across, or rollout spawns would reuse ids and
        // id-keyed lookups would join the wrong entity.
        copy.idCounter = idCounter;
        copy.currentTick = currentTick;
        copy.pendingElixirGrant[0] = pendingElixirGrant[0];
        copy.pendingElixirGrant[1] = pendingElixirGrant[1];

        std::unordered_map<int, std::shared_ptr<Entity>> byOldId;
        byOldId.reserve(activeEntities.size() + pendingEntities.size());

        // Pending entities too: a card played this tick lives there until the
        // next commit.
        copy.activeEntities.reserve(activeEntities.size());
        for (const auto& e : activeEntities) {
            auto c = e->snapshot();
            byOldId[e->id] = c;
            copy.activeEntities.push_back(std::move(c));
        }
        copy.pendingEntities.reserve(pendingEntities.size());
        for (const auto& e : pendingEntities) {
            auto c = e->snapshot();
            byOldId[e->id] = c;
            copy.pendingEntities.push_back(std::move(c));
        }

        // Must be a second pass: a projectile can home on an entity copied
        // after it. A no-op except for Projectile.
        for (const auto& e : copy.activeEntities) e->remapSnapshotReferences(byOldId);
        for (const auto& e : copy.pendingEntities) e->remapSnapshotReferences(byOldId);

        return copy;
    }

    int getWidth() const { return width; }
    int getHeight() const { return height; }

    // The river band, for callers such as the own-half placement check.
    float getRiverStart() const { return riverY_start; }
    float getRiverEnd() const { return riverY_end; }
    // Is this column a bridge? The single definition, shared by the physics
    // (clampToBoard) and the observation encoder so they cannot disagree.
    //
    // Takes a float to serve both: continuous positions, and integer cell
    // centres (cell i covers [i-0.5, i+0.5]).
    bool isOnBridge(float x) const {
        return (x >= leftBridge.x - BRIDGE_HALF_WIDTH && x <= leftBridge.x + BRIDGE_HALF_WIDTH)
            || (x >= rightBridge.x - BRIDGE_HALF_WIDTH && x <= rightBridge.x + BRIDGE_HALF_WIDTH);
    }

    const Vector2D& getLeftBridge() const { return leftBridge; }
    const Vector2D& getRightBridge() const { return rightBridge; }

    // True for the unplaceable corners of the two back rows, outside the
    // centred opening; false everywhere else.
    bool isBackRowDeadZone(float x, float y) const {
        bool inBackRow = (y < 1.0f) || (y > static_cast<float>(height) - 2.0f);
        if (!inBackRow) return false;
        float centerX = (static_cast<float>(width) - 1.0f) / 2.0f;
        return (x < centerX - BACK_ROW_OPENING_HALF_WIDTH) || (x > centerX + BACK_ROW_OPENING_HALF_WIDTH);
    }

    void addEntity(std::shared_ptr<Entity> entity) {
        pendingEntities.push_back(entity);
    }

    // Lets GameManager::playCard find the entity a deploy just spawned;
    // spawnEntity returns nothing.
    size_t pendingEntityCount() const { return pendingEntities.size(); }
    const std::shared_ptr<Entity>& getPendingEntity(size_t index) const { return pendingEntities[index]; }

    // GameManager::step passes the real tick; tests call it without one.
    void commitPendingEntities(int tick = 0) {
        if (!pendingEntities.empty()) {
            for (const auto& e : pendingEntities) {
                statsEvents.notifyEntitySpawned({ e->id, e->cardId, e->team, tick });
            }
            activeEntities.insert(activeEntities.end(), pendingEntities.begin(), pendingEntities.end());
            pendingEntities.clear();
            collidersDirty = true;   // membership changed; see colliders()
        }
    }

    const std::vector<std::shared_ptr<Entity>>& getEntities() const {
        return activeEntities;
    }

    void cleanDeadEntities(int tick = 0) {
        // Nothing died: skip the three passes below, which would all be no-ops.
        // Most ticks take this branch.
        bool anyDead = false;
        for (const auto& e : activeEntities) {
            if (!e->isAlive()) { anyDead = true; break; }
        }
        if (!anyDead) return;
        collidersDirty = true;   // membership is about to change

        // Death effects fire before the erase, via Entity::onDeath.
        for (const auto& e : activeEntities) {
            if (!e->isAlive()) {
                e->onDeath(*this);
                statsEvents.notifyEntityDied({ e->id, e->cardId, e->team, tick });
            }
        }

        // Every living entity hears of every death (Skeleton King's souls). A
        // separate pass, so it reaches only entities already on the board, not
        // anything a death effect just spawned.
        for (const auto& dying : activeEntities) {
            if (!dying->isAlive()) {
                for (const auto& other : activeEntities) {
                    if (other->isAlive()) other->onNearbyDeath(*this, dying->position, dying->team);
                }
            }
        }

        activeEntities.erase(
            std::remove_if(activeEntities.begin(), activeEntities.end(),
                [](const std::shared_ptr<Entity>& e) { return !e->isAlive(); }),
            activeEntities.end()
        );
    }

    // Push a point out of a circular obstacle, with a small perpendicular slide
    // so it travels around instead of sticking. Shared by movement and
    // collision resolution.
    static Vector2D pushAwayFrom(Vector2D pos, const Vector2D& obstacleCenter, float minDist) {
        float dx = pos.x - obstacleCenter.x;
        float dy = pos.y - obstacleCenter.y;
        float dist = std::sqrt(dx * dx + dy * dy);

        if (dist < minDist) {
            // Direction and distance are computed separately: a point exactly
            // on the centre is pushed the full minDist along +x. The ordinary
            // case is unchanged.
            float ux, uy;
            if (dist < 0.001f) {
                ux = 1.0f; uy = 0.0f;   // coincident: any direction
            } else {
                ux = dx / dist; uy = dy / dist;
            }
            float push = minDist - dist;
            pos.x += ux * push + uy * 0.05f;
            pos.y += uy * push - ux * 0.05f;
        }
        return pos;
    }

    // Indices of everything with a real collision radius (buildings and towers,
    // a handful of entities), so resolvePositionAgainstBuildings does not scan
    // the whole list per moving troop.
    //
    // Radii never change, so only membership invalidates the cache, and it
    // changes only in commitPendingEntities and cleanDeadEntities, which set
    // the flag. Liveness is checked live.
    const std::vector<Collider>& colliders() const {
        if (collidersDirty) {
            colliderCache.clear();
            for (size_t i = 0; i < activeEntities.size(); ++i) {
                const float r = activeEntities[i]->getCollisionRadius();
                if (r > 0.0f) colliderCache.push_back(Collider{ i, r });
            }
            collidersDirty = false;
        }
        return colliderCache;
    }

    Vector2D resolvePositionAgainstBuildings(const Vector2D& pos, int entityId) const {
        Vector2D resolved = pos;
        // Ascending index order: the pushes compose, so order is part of the
        // answer.
        for (const Collider& c : colliders()) {
            const auto& entity = activeEntities[c.index];
            if (entity->id == entityId || !entity->isAlive()) continue;

            resolved = pushAwayFrom(resolved, entity->position, c.radius + Entity::IMPLICIT_TROOP_RADIUS);
        }
        return resolved;
    }

    // Separates every overlapping pair for one tick, then re-clamps everyone to
    // the board and river.
    void resolveCollisions() {
        // Per-entity facts read once per entity instead of per pair; within
        // this call radius, targetability and liveness are constant. Positions
        // are not hoisted, since they are what this mutates.
        const size_t count = activeEntities.size();
        bodyScratch.clear();
        bodyScratch.reserve(count);
        for (const auto& e : activeEntities) {
            const float r = e->getCollisionRadius();
            bodyScratch.push_back(Body{
                e.get(), r, r > 0.0f, (r <= 0.0f && e->isTargetable()), e->isFlying, e->isAlive() });
        }

        for (size_t i = 0; i < count; ++i) {
            const Body& b1 = bodyScratch[i];
            if (!b1.alive) continue;
            for (size_t j = i + 1; j < count; ++j) {
                const Body& b2 = bodyScratch[j];

                if (!b2.alive) continue;

                Entity* e1 = b1.entity;
                Entity* e2 = b2.entity;

                const float r1 = b1.radius;
                const float r2 = b2.radius;

                const bool isBuilding1 = b1.building;
                const bool isBuilding2 = b2.building;
                const bool isTroop1 = b1.troop;
                const bool isTroop2 = b2.troop;

                // Only entities in the same plane collide; fliers pass through
                // everything.
                if (isTroop1 && isTroop2 && b1.flying == b2.flying) {
                    float dx = e1->position.x - e2->position.x;
                    float dy = e1->position.y - e2->position.y;
                    float dist = std::sqrt(dx * dx + dy * dy);
                    float minRadius = 2.0f * Entity::IMPLICIT_TROOP_RADIUS;

                    if (dist < minRadius) {
                        if (dist < 0.001f) { dx = 1.0f; dy = 0.0f; dist = 1.0f; }
                        float overlap = minRadius - dist;
                        float pushX = (dx / dist) * overlap * 0.5f;
                        float pushY = (dy / dist) * overlap * 0.5f;

                        // Tiny orthogonal nudge to prevent jitter lock.
                        float noise = 0.01f;
                        pushX += (dy / dist) * noise;
                        pushY -= (dx / dist) * noise;

                        e1->position.x += pushX;
                        e1->position.y += pushY;
                        e2->position.x -= pushX;
                        e2->position.y -= pushY;
                    }
                }

                if (isTroop1 && isBuilding2 && !b1.flying) {
                    e1->position = pushAwayFrom(e1->position, e2->position, r2 + Entity::IMPLICIT_TROOP_RADIUS);
                }
                if (isTroop2 && isBuilding1 && !b2.flying) {
                    e2->position = pushAwayFrom(e2->position, e1->position, r1 + Entity::IMPLICIT_TROOP_RADIUS);
                }
            }
        }

        // Re-clamp: collisions can push a troop into the river or off the
        // board. Each entity's clampPosition respects riverIgnores.
        for (auto& entity : activeEntities) {
            if (entity->isAlive()) {
                entity->clampPosition(*this);
            }
        }
    }

    // Keeps a position on the board and out of the river unless allowed in it.
    Vector2D clampToBoard(Vector2D pos, bool ignoresRiver) const {
        pos.x = std::max(0.0f, std::min(pos.x, static_cast<float>(width - 1)));
        pos.y = std::max(0.0f, std::min(pos.y, static_cast<float>(height - 1)));

        if (!ignoresRiver && pos.y > riverY_start && pos.y < riverY_end) {
            // isOnBridge(), so the observation encoder gets the same answer.
            if (!isOnBridge(pos.x)) {
                float riverMid = (riverY_start + riverY_end) / 2.0f;
                pos.y = (pos.y < riverMid) ? riverY_start : riverY_end;
            }
        }
        return pos;
    }

    // The single "standing on the waypoint" tolerance, shared with
    // Troop::moveTowards, which does not move closer than this. If
    // getNextWaypoint hands back a point the mover is already within this of,
    // the unit freezes for the rest of the match (tests/core/test_board.cpp).
    static constexpr float WAYPOINT_ARRIVAL_EPS = 0.01f;

    Vector2D getNextWaypoint(const Vector2D& currentPos, const Vector2D& targetPos) const {
        bool isCurrentBelow = currentPos.y <= riverY_start;
        bool isTargetBelow = targetPos.y <= riverY_start;
        bool isCurrentAbove = currentPos.y >= riverY_end;
        bool isTargetAbove = targetPos.y >= riverY_end;

        if ((isCurrentBelow && isTargetBelow) || (isCurrentAbove && isTargetAbove) ||
            (!isCurrentBelow && !isCurrentAbove && !isTargetBelow && !isTargetAbove)) {
            return targetPos;
        }

        float distToLeft = currentPos.distanceTo(leftBridge);
        float distToRight = currentPos.distanceTo(rightBridge);
        float bridgeX = (distToLeft < distToRight) ? leftBridge.x : rightBridge.x;

        if (isCurrentBelow) {
            // The bank tests are inclusive, so a unit already on the near bank
            // is still "below"; hand it the far bank, or it is routed to where
            // it stands.
            Vector2D nearBank{bridgeX, riverY_start};
            if (currentPos.distanceTo(nearBank) <= WAYPOINT_ARRIVAL_EPS) {
                return Vector2D{bridgeX, riverY_end};
            }
            return nearBank;
        } else if (isCurrentAbove) {
            Vector2D nearBank{bridgeX, riverY_end};
            if (currentPos.distanceTo(nearBank) <= WAYPOINT_ARRIVAL_EPS) {
                return Vector2D{bridgeX, riverY_start};
            }
            return nearBank;
        } else {
            // Inside the river band, heading for the exit bank. Within
            // WAYPOINT_ARRIVAL_EPS of it this leg is done: return targetPos,
            // which the first branch returns anyway once y reaches the bank.
            // Handing back the edge itself would freeze the unit just short of
            // dry land (tests/core/test_board.cpp, the bridge-exit block).
            if (isTargetAbove) {
                Vector2D exitBank{bridgeX, riverY_end};
                if (currentPos.distanceTo(exitBank) <= WAYPOINT_ARRIVAL_EPS) {
                    return targetPos;
                }
                return exitBank;
            } else if (isTargetBelow) {
                Vector2D exitBank{bridgeX, riverY_start};
                if (currentPos.distanceTo(exitBank) <= WAYPOINT_ARRIVAL_EPS) {
                    return targetPos;
                }
                return exitBank;
            } else {
                return targetPos;
            }
        }
    }
};