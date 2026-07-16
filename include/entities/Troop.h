#pragma once
#include "CombatEntity.h"

class Troop : public CombatEntity {
protected:
    float speed;

public:
    bool riverIgnores = false;

    Troop(int id, float x, float y, int hp, int team, char symbol,
        float speed, float attackRange, int damage, int attackCooldown)
        : CombatEntity(id, x, y, hp, team, symbol, attackRange, damage, attackCooldown),
        speed(speed) {}

    void setIgnoresRiver(bool ignore) {
        riverIgnores = ignore;
    }

protected:
    void moveTowards(Board& board, const Vector2D& dest) override {
        Vector2D waypoint = riverIgnores ? dest : board.getNextWaypoint(position, dest);
        float dx = waypoint.x - position.x;
        float dy = waypoint.y - position.y;
        float distToWaypoint = position.distanceTo(waypoint);

        if (distToWaypoint > 0.01f) {
            float currentSpeed = (freezeTicks > 0) ? speed * freezeSlow : speed;
            Vector2D newPos;
            newPos.x = position.x + (dx / distToWaypoint) * currentSpeed;
            newPos.y = position.y + (dy / distToWaypoint) * currentSpeed;

            Vector2D resolved = board.resolvePositionAgainstBuildings(newPos, id);
            position = resolved;
        }
    }

    void clampPosition() override {
        if (position.x < 0.0f) position.x = 0.0f;
        if (position.x > 17.0f) position.x = 17.0f;
        if (position.y < 0.0f) position.y = 0.0f;
        if (position.y > 31.0f) position.y = 31.0f;

        if (!riverIgnores) {
            if (position.y > 15.0f && position.y < 17.0f) {
                bool onLeftBridge = (position.x >= 3.0f && position.x <= 5.0f);
                bool onRightBridge = (position.x >= 13.0f && position.x <= 15.0f);
                if (!onLeftBridge && !onRightBridge) {
                    if (position.y < 16.0f) {
                        position.y = 15.0f;
                    } else {
                        position.y = 17.0f;
                    }
                }
            }
        }
    }
};
