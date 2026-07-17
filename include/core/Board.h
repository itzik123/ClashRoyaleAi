#pragma once
#include <vector>
#include <memory>
#include <algorithm>
#include "Entity.h"

class Board {
private:
    int width, height;
    std::vector<std::shared_ptr<Entity>> activeEntities;
    std::vector<std::shared_ptr<Entity>> pendingEntities;
    int idCounter = 1000;

    float riverY_start = 15.0f;
    float riverY_end = 17.0f;
    Vector2D leftBridge{ 4.0f, 16.0f };
    Vector2D rightBridge{ 14.0f, 16.0f };

public:
    Board(int w = 18, int h = 32) : width(w), height(h) {}

    int allocateId() { return idCounter++; }

    int getWidth() const { return width; }
    int getHeight() const { return height; }

    void addEntity(std::shared_ptr<Entity> entity) {
        pendingEntities.push_back(entity);
    }

    void commitPendingEntities() {
        if (!pendingEntities.empty()) {
            activeEntities.insert(activeEntities.end(), pendingEntities.begin(), pendingEntities.end());
            pendingEntities.clear();
        }
    }

    const std::vector<std::shared_ptr<Entity>>& getEntities() const {
        return activeEntities;
    }

    void cleanDeadEntities() {
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

        float distToLeft = currentPos.distanceTo(Vector2D{leftBridge.x, 16.0f});
        float distToRight = currentPos.distanceTo(Vector2D{rightBridge.x, 16.0f});
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