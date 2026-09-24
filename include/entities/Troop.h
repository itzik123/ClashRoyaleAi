#pragma once
#include "CombatEntity.h"

class Troop : public CombatEntity {
protected:
    float speed;

public:
    bool riverIgnores = false;

    // Base speed for the observation encoder. Not freeze-adjusted: the
    // observation describes the unit, and a freeze shows up as it not moving.
    float getSpeed() const { return speed; }

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

        // Must be the same constant getNextWaypoint uses for arrival; two
        // different epsilons produce an absorbing state at the bridge mouths
        // (tests/core/test_board.cpp).
        if (distToWaypoint > Board::WAYPOINT_ARRIVAL_EPS) {
            // frozenThisTick, not freezeTicks: update() has already decremented
            // the counter by now.
            float currentSpeed = frozenThisTick ? speed * freezeSlow : speed;
            // Land on the waypoint, never past it: an overshoot along a bank
            // line keeps the unit "below" the river, so it is handed the same
            // bridge mouth back and orbits it forever (UPSTREAM_REQUESTS.md
            // item 31; tests/core/test_board.cpp, [orbit]).
            float step = std::min(currentSpeed, distToWaypoint);
            Vector2D newPos;
            newPos.x = position.x + (dx / distToWaypoint) * step;
            newPos.y = position.y + (dy / distToWaypoint) * step;

            // Flying troops pass over building footprints.
            position = isFlying ? newPos : board.resolvePositionAgainstBuildings(newPos, id);
        }
    }

    void clampPosition(Board& board) override {
        position = board.clampToBoard(position, riverIgnores);
    }
};
