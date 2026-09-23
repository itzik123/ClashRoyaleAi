#pragma once
#include <memory>
#include "ArenaLayout.h"
#include "Board.h"
#include "Entity.h"

// Lane paths. A unit with nothing in sight walks its own lane toward the enemy
// base, rather than beelining for the globally nearest tower (which, with one
// Princess down, sends it diagonally to the other lane).
//
// Towers are identified via Entity::isTower()/symbol rather than by including
// Tower.h, which would pull in CombatEntity.h, one of this header's consumers.
namespace LanePath {

// The tower a blind unit on `myTeam` walks to from `from`: its own lane's enemy
// Princess if standing, else the enemy King.
//
// The lane is the nearest bridge, re-evaluated per call, the same rule
// Board::getNextWaypoint uses, so objective and crossing always agree and
// nothing is latched onto the entity. nullptr only if the enemy has no towers,
// which cannot happen in a live match.
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

// Where to walk while approaching `target`. For anything but an enemy King, the
// target's own position.
//
// Toward the King, the path runs up the lane first and then angles in, via W1,
// the lane's (necessarily empty) Princess slot:
//
//     W0 = bridge(lane)                            (2.5 | 14.5, 16.5)
//     W1 = (laneX(lane), princessY(enemy team))     (3.0 | 14.0, 27.0 | 6.0)
//     W2 = the King                                 (8.5, 30.5 | 2.5)
//
// W1 is released once the unit is within Board::WAYPOINT_ARRIVAL_EPS of it or
// has reached its row, and is never handed to a unit already standing on it;
// otherwise Troop::moveTowards refuses to move and the unit freezes. Swept in
// test_lane_pathing.cpp and tools/audit/waypoint_probe.cpp.
inline Vector2D approachPoint(const Board& board, int myTeam, const Vector2D& from,
                              const std::shared_ptr<Entity>& target) {
    (void)board;
    if (!target) return from;
    if (!target->isTower() || target->symbol != 'R') return target->position;

    const int enemyTeam = 1 - myTeam;
    const Vector2D w1{ ArenaLayout::laneXFor(from.x), ArenaLayout::princessY(enemyTeam) };

    // Advancing is +y for team 0 and -y for team 1. Level with the Princess
    // row, the lane leg is done.
    const bool advancingUp = (enemyTeam == 1);
    const bool reachedRow = advancingUp ? (from.y >= w1.y) : (from.y <= w1.y);
    if (reachedRow) return target->position;
    if (from.distanceTo(w1) <= Board::WAYPOINT_ARRIVAL_EPS) return target->position;
    return w1;
}

}  // namespace LanePath
