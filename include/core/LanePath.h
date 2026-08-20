#pragma once
#include <memory>
#include "ArenaLayout.h"
#include "Board.h"
#include "Entity.h"

// The "dirt paths".
//
// A unit with nothing inside its own sight range is BLIND. Rule B of the
// 2026-08-21 player audit: a blind unit does not beeline for whatever tower
// happens to be nearest -- it acts "like a magnet to its predefined lane path"
// and walks its OWN lane toward the enemy base, ignoring globally closer things
// it cannot see.
//
// Before this, CombatEntity::findTarget's blind fallback was "closest enemy
// Tower by raw distance". With one enemy Princess destroyed that sends a unit
// diagonally across the arena to the OTHER lane's Princess. Measured on the
// corrected board with team 1's left Princess dead, a unit in the left lane:
//
//     at (2.5, 12.0)   right Princess 18.90   King 19.45   -> wrong tower
//     at (2.5, 14.0)   right Princess 17.36   King 17.56   -> wrong tower
//     at (2.5, 15.0)   right Princess 16.62   King 16.62   -> tie
//     at (2.5, 16.5)   right Princess 15.57   King 15.23   -> right, by accident
//
// Note the crossover at y ~ 15.0: from the bridge mouth the broken rule already
// answered correctly, which is why the regression tests anchor further back.
//
// Towers are identified through Entity::isTower()/symbol rather than by
// including Tower.h -- Tower.h pulls in Building.h and thence CombatEntity.h,
// and CombatEntity is one of this header's own consumers.
namespace LanePath {

// The tower a blind unit on `myTeam` should walk to from `from`: its OWN lane's
// enemy Princess Tower if that is still standing, else the enemy King.
//
// Lane is nearest-bridge, re-evaluated on every call -- the same rule
// Board::getNextWaypoint uses to choose a crossing, so the objective and the
// crossing agree by construction and a unit can never be routed over one bridge
// while aiming at the other lane's tower. Nothing is latched onto the entity,
// so Board::deepCopy and Tower::snapshot need no new field and a rollout cannot
// diverge from the live board on lane state.
//
// Returns nullptr only if the enemy has no towers left at all, which cannot
// happen in a live match (the King dying ends it).
inline std::shared_ptr<Entity> laneObjective(const Board& board, int myTeam,
                                             const Vector2D& from) {
    const bool wantLeft = ArenaLayout::isLeftLane(from.x);
    std::shared_ptr<Entity> lanePrincess, king;

    for (const auto& e : board.getEntities()) {
        if (!e->isAlive() || !e->isTower() || e->team == myTeam) continue;
        if (e->symbol == 'R') { king = e; continue; }
        if (ArenaLayout::isLeftLane(e->position.x) == wantLeft) lanePrincess = e;
    }
    return lanePrincess ? lanePrincess : king;
}

// Where to WALK toward while approaching `target`.
//
// For every target except an enemy KING this is just the target's own position,
// so the overwhelmingly common case -- a live lane Princess, or any troop or
// building already in sight -- is completely untouched.
//
// Heading for the King, a straight line from the bridge cuts diagonally inward
// across the arena immediately. The real dirt path runs UP THE LANE first and
// only then angles in, so the unit is routed via W1, the lane's Princess slot,
// which is empty by definition: a living Princess there would have BEEN the
// objective.
//
//     W0 = bridge(lane)                            (2.5 | 14.5, 16.5)
//     W1 = (laneX(lane), princessY(enemy team))     (3.0 | 14.0, 27.0 | 6.0)
//     W2 = the King itself                          (8.5, 30.5 | 2.5)
//
// ABSORBING-STATE DISCIPLINE. This introduces a new intermediate waypoint, and
// TWO absorbing states have already shipped in this engine from exactly that --
// a planner handing a mover the point it is already standing on, while
// Troop::moveTowards refuses to move inside Board::WAYPOINT_ARRIVAL_EPS, so the
// position never changes, so the waypoint never changes, forever. W1 is
// therefore released the moment the unit is within the epsilon of it OR has
// reached its row, and is never returned to somebody standing on it. Swept at
// finer-than-epsilon resolution in test_lane_pathing.cpp and analytically in
// tools/audit/waypoint_probe.cpp.
inline Vector2D approachPoint(const Board& board, int myTeam, const Vector2D& from,
                              const std::shared_ptr<Entity>& target) {
    (void)board;
    if (!target) return from;
    if (!target->isTower() || target->symbol != 'R') return target->position;

    const int enemyTeam = 1 - myTeam;
    const Vector2D w1{ ArenaLayout::laneXFor(from.x), ArenaLayout::princessY(enemyTeam) };

    // Advancing toward the enemy means increasing y for team 0 and decreasing
    // it for team 1. Once level with the Princess row, the lane leg is done and
    // the remaining leg is the angle in to the King.
    const bool advancingUp = (enemyTeam == 1);
    const bool reachedRow = advancingUp ? (from.y >= w1.y) : (from.y <= w1.y);
    if (reachedRow) return target->position;
    if (from.distanceTo(w1) <= Board::WAYPOINT_ARRIVAL_EPS) return target->position;
    return w1;
}

}  // namespace LanePath
