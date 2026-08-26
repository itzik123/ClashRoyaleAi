#pragma once
#include "CombatEntity.h"

class Troop : public CombatEntity {
protected:
    float speed;

public:
    bool riverIgnores = false;

    // Read-only view of the protected movement speed, for the observation
    // encoder's attribute channels -- same rationale as CombatEntity's
    // getAttackRange/getAttackCooldown. Note this returns the BASE speed, not
    // the frozen-adjusted one used in update(): what the observation should
    // describe is what kind of unit this is, and freeze is already visible
    // through the unit simply not moving between consecutive frames.
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

        // Board::WAYPOINT_ARRIVAL_EPS, not a second literal 0.01f. This dead
        // zone and getNextWaypoint's notion of having reached a waypoint are
        // the same fact, and when they were two independent numbers they
        // disagreed at the bridge mouth and produced an absorbing state that
        // froze troops mid-crossing for 10+ seconds -- see that constant's
        // comment and the regression tests in tests/core/test_board.cpp.
        if (distToWaypoint > Board::WAYPOINT_ARRIVAL_EPS) {
            // frozenThisTick, not freezeTicks: this runs AFTER update() has
            // already decremented the counter, so the final tick of every
            // freeze read as thawed and moved at full speed. See
            // CombatEntity::frozenThisTick.
            float currentSpeed = frozenThisTick ? speed * freezeSlow : speed;
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
