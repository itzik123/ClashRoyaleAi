#pragma once
#include <vector>
#include <memory>
#include <algorithm>
#include "Entity.h"
#include "StatsEventBus.h"

class Board {
private:
    int width, height;
    std::vector<std::shared_ptr<Entity>> activeEntities;
    std::vector<std::shared_ptr<Entity>> pendingEntities;
    int idCounter = 1000;

    float riverY_start = 16.0f;
    float riverY_end = 18.0f;
    Vector2D leftBridge{ 4.0f, 17.0f };
    Vector2D rightBridge{ 14.0f, 17.0f };

    // Real-map sync: the arena is 18x34, not 18x32 -- there's one extra row
    // behind each King Tower that this engine used to just not have. Most of
    // that row is dead space (matches the decorative rock/wall texture
    // flanking the real King Tower), except a BACK_ROW_OPENING_WIDTH-wide gap
    // centered on the board, which is real, placeable ground. See
    // isBackRowDeadZone() below.
    static constexpr float BACK_ROW_OPENING_HALF_WIDTH = 3.0f;

public:
    // Public, not a getter-wrapped private member: every fire site
    // (entities, GameManager) already holds a Board& and just calls
    // board.statsEvents.notifyX(...) directly, the same way they already
    // read board.getEntities()/call board.addEntity(...).
    StatsEventBus statsEvents;

    // Set once per tick by GameManager::step() (board.currentTick =
    // currentTick;), read by combat code deep in the update()/performAttack()
    // call chain (Projectile::applyHit, AreaSpell::update, Building/
    // MeleeTroop/BuildingTargeter::performAttack) to stamp DamageDealtEvent.
    // Unlike commitPendingEntities()/cleanDeadEntities() below -- which take
    // an explicit tick parameter because GameManager calls them directly --
    // there's no parameter-passing route all the way down through every
    // update() override without threading a tick argument through the whole
    // entity hierarchy, which would be a far bigger, more invasive change
    // than this one field. Board doesn't act on this value itself, purely a
    // read-only convenience for whoever's stamping an event.
    int currentTick = 0;

    // Elixir Collector: accumulated here by ElixirGrantEffect (a periodic
    // effect, like Board::currentTick above there's no clean route from
    // deep inside CombatEntity::update() to PlayerState's elixir, which
    // Board doesn't even know exists -- so this is a small drop-box
    // instead. GameManager::step() drains both entries into
    // playerAI/playerOpponent.elixir once per tick and resets them to 0;
    // Board doesn't act on this value itself.
    float pendingElixirGrant[2] = { 0.0f, 0.0f };

    Board(int w = 18, int h = 34) : width(w), height(h) {}

    int allocateId() { return idCounter++; }

    int getWidth() const { return width; }
    int getHeight() const { return height; }

    // Exposed so callers that need to reason about the river band (e.g.
    // GameManager's own-half placement check) derive it from here instead
    // of re-guessing the same two numbers as a second, driftable copy.
    float getRiverStart() const { return riverY_start; }
    float getRiverEnd() const { return riverY_end; }

    // True for the unplaceable corners of the two new back rows (y in
    // [0,1) or (height-2, height-1], outside the BACK_ROW_OPENING_HALF_WIDTH
    // gap centered on the board) -- everywhere else returns false, including
    // every row that existed before this back-row expansion. Exposed so
    // GameManager::isValidPlacement checks it alongside the plain width/
    // height bounds check, the same way it already reads getRiverStart()/
    // getRiverEnd() instead of re-deriving river geometry itself.
    bool isBackRowDeadZone(float x, float y) const {
        bool inBackRow = (y < 1.0f) || (y > static_cast<float>(height) - 2.0f);
        if (!inBackRow) return false;
        float centerX = (static_cast<float>(width) - 1.0f) / 2.0f;
        return (x < centerX - BACK_ROW_OPENING_HALF_WIDTH) || (x > centerX + BACK_ROW_OPENING_HALF_WIDTH);
    }

    void addEntity(std::shared_ptr<Entity> entity) {
        pendingEntities.push_back(entity);
    }

    // tick defaults to 0 (rather than Board storing its own tick mirror,
    // which would be a second, driftable copy of GameManager's currentTick
    // -- exactly the kind of duplication this codebase avoids elsewhere,
    // e.g. getRiverStart()/getWidth() existing so GameManager doesn't
    // re-guess them). GameManager::step() passes the real tick; the several
    // existing tickless test call sites keep compiling unchanged.
    void commitPendingEntities(int tick = 0) {
        if (!pendingEntities.empty()) {
            for (const auto& e : pendingEntities) {
                statsEvents.notifyEntitySpawned({ e->id, e->cardId, e->team, tick });
            }
            activeEntities.insert(activeEntities.end(), pendingEntities.begin(), pendingEntities.end());
            pendingEntities.clear();
        }
    }

    const std::vector<std::shared_ptr<Entity>>& getEntities() const {
        return activeEntities;
    }

    void cleanDeadEntities(int tick = 0) {
        // Fire death effects (e.g. Golem spawning two Golemites) before the
        // erase below, purely via Entity's own virtual onDeath() -- Board
        // never needs to know which entities are CombatEntity-shaped enough
        // to actually have one, same as clampPosition().
        for (const auto& e : activeEntities) {
            if (!e->isAlive()) {
                e->onDeath(*this);
                statsEvents.notifyEntityDied({ e->id, e->cardId, e->team, tick });
            }
        }

        // Notify every still-alive entity of every death that just
        // happened (Skeleton King's soul collection: "a troop dies in his
        // presence") -- a second pass, not folded into the loop above,
        // since a death effect firing (e.g. a spawn) could itself add
        // pending entities, and this only needs to reach entities already
        // on the board this tick, not anything a death effect just created.
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

    // Single source of truth for "push a point out of a circular obstacle,
    // with a small perpendicular slide so it can travel around it instead of
    // sticking". Used for normal movement (resolvePositionAgainstBuildings
    // below) and, identically, by GameManager's post-move collision pass --
    // previously copy-pasted between the two call sites (and a third time,
    // as a near-duplicate, for the mirrored troop2-vs-building1 case).
    static Vector2D pushAwayFrom(Vector2D pos, const Vector2D& obstacleCenter, float minDist) {
        float dx = pos.x - obstacleCenter.x;
        float dy = pos.y - obstacleCenter.y;
        float dist = std::sqrt(dx * dx + dy * dy);

        if (dist < minDist) {
            if (dist < 0.001f) {
                dx = 1.0f; dy = 0.0f; dist = 1.0f;
            }
            float push = minDist - dist;
            pos.x += (dx / dist) * push + (dy / dist) * 0.05f;
            pos.y += (dy / dist) * push - (dx / dist) * 0.05f;
        }
        return pos;
    }

    Vector2D resolvePositionAgainstBuildings(const Vector2D& pos, int entityId) const {
        Vector2D resolved = pos;
        for (const auto& entity : activeEntities) {
            float radius = entity->getCollisionRadius();
            if (radius <= 0.0f || entity->id == entityId || !entity->isAlive()) continue;

            resolved = pushAwayFrom(resolved, entity->position, radius + Entity::IMPLICIT_TROOP_RADIUS);
        }
        return resolved;
    }

    // Physically separates every overlapping pair of entities for one tick,
    // then re-clamps everyone to the board/river. This is "how entities on
    // this board physically interact" -- a Board concern, not a match-rules
    // one, so it lives here rather than in GameManager::step() alongside
    // elixir economy and win-condition checks.
    void resolveCollisions() {
        for (size_t i = 0; i < activeEntities.size(); ++i) {
            for (size_t j = i + 1; j < activeEntities.size(); ++j) {
                auto& e1 = activeEntities[i];
                auto& e2 = activeEntities[j];

                if (!e1->isAlive() || !e2->isAlive()) continue;

                float r1 = e1->getCollisionRadius();
                float r2 = e2->getCollisionRadius();

                bool isBuilding1 = r1 > 0.0f;
                bool isBuilding2 = r2 > 0.0f;
                bool isTroop1 = !isBuilding1 && e1->isTargetable();
                bool isTroop2 = !isBuilding2 && e2->isTargetable();

                // Flying units pass through everything -- ground and other
                // fliers alike -- so only entities sharing a plane collide.
                if (isTroop1 && isTroop2 && e1->isFlying == e2->isFlying) {
                    float dx = e1->position.x - e2->position.x;
                    float dy = e1->position.y - e2->position.y;
                    float dist = std::sqrt(dx * dx + dy * dy);
                    float minRadius = 2.0f * Entity::IMPLICIT_TROOP_RADIUS;

                    if (dist < minRadius) {
                        if (dist < 0.001f) { dx = 1.0f; dy = 0.0f; dist = 1.0f; }
                        float overlap = minRadius - dist;
                        float pushX = (dx / dist) * overlap * 0.5f;
                        float pushY = (dy / dist) * overlap * 0.5f;

                        // Add tiny orthogonal noise to prevent jitter locking
                        float noise = 0.01f;
                        pushX += (dy / dist) * noise;
                        pushY -= (dx / dist) * noise;

                        e1->position.x += pushX;
                        e1->position.y += pushY;
                        e2->position.x -= pushX;
                        e2->position.y -= pushY;
                    }
                }

                if (isTroop1 && isBuilding2 && !e1->isFlying) {
                    e1->position = pushAwayFrom(e1->position, e2->position, r2 + Entity::IMPLICIT_TROOP_RADIUS);
                }
                if (isTroop2 && isBuilding1 && !e2->isFlying) {
                    e2->position = pushAwayFrom(e2->position, e1->position, r1 + Entity::IMPLICIT_TROOP_RADIUS);
                }
            }
        }

        // Re-clamp after collision resolution, which can push a troop back
        // into the river or off the board edge. Delegates to each entity's
        // own clampPosition() (a no-op for anything that isn't a Troop) so
        // this respects riverIgnores instead of re-deriving the rule here.
        for (auto& entity : activeEntities) {
            if (entity->isAlive()) {
                entity->clampPosition(*this);
            }
        }
    }

    // Single source of truth for "keep a position on the board and out of the
    // river unless explicitly allowed in it" -- used both right after a troop
    // moves and again after collision resolution potentially nudges it.
    Vector2D clampToBoard(Vector2D pos, bool ignoresRiver) const {
        pos.x = std::max(0.0f, std::min(pos.x, static_cast<float>(width - 1)));
        pos.y = std::max(0.0f, std::min(pos.y, static_cast<float>(height - 1)));

        if (!ignoresRiver && pos.y > riverY_start && pos.y < riverY_end) {
            bool onLeftBridge = (pos.x >= leftBridge.x - 1.0f && pos.x <= leftBridge.x + 1.0f);
            bool onRightBridge = (pos.x >= rightBridge.x - 1.0f && pos.x <= rightBridge.x + 1.0f);
            if (!onLeftBridge && !onRightBridge) {
                float riverMid = (riverY_start + riverY_end) / 2.0f;
                pos.y = (pos.y < riverMid) ? riverY_start : riverY_end;
            }
        }
        return pos;
    }

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
            return Vector2D{bridgeX, riverY_start};
        } else if (isCurrentAbove) {
            return Vector2D{bridgeX, riverY_end};
        } else {
            if (isTargetAbove) {
                return Vector2D{bridgeX, riverY_end};
            } else if (isTargetBelow) {
                return Vector2D{bridgeX, riverY_start};
            } else {
                return targetPos;
            }
        }
    }
};