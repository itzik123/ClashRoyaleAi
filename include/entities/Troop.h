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

            // Flying troops fly over building footprints instead of routing
            // around them.
            position = isFlying ? newPos : board.resolvePositionAgainstBuildings(newPos, id);
        }
    }

    void clampPosition(Board& board) override {
        position = board.clampToBoard(position, riverIgnores);
    }
};
