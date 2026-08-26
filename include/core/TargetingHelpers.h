#pragma once
#include "Board.h"
#include "Entity.h"
#include <limits>
#include <memory>

// Closest-in-spirit-to CombatEntity::findTarget()'s own eligibility filter,
// but selecting by HP extreme instead of distance -- used by Hero Giant's
// throw (highest-HP within a short range) and Hero Mega Minion's warp
// (lowest-HP anywhere on the board, maxRadius <= 0 meaning unbounded).
// Excludes every BUILDING -- deployed ones (Cannon, Tombstone, X-Bow) as well
// as Crown Towers -- because both sourced Hero abilities say "enemy TROOP".
// This used to exclude only `isTower()`, and a Cannon is the highest-HP thing
// inside Hero Giant's 3-tile grab radius far more often than a troop is, so
// Hurl spent most of its uses throwing a stationary building into the other
// lane. `isBuilding()` is strictly wider than `isTower()` (Tower derives from
// Building), so nothing that was excluded before is admitted now.
// Returns nullptr if nothing eligible is found.
inline std::shared_ptr<Entity> findHpExtremeEnemy(Board& board, const Vector2D& origin,
        float maxRadius, int myTeam, bool wantHighestHp) {
    std::shared_ptr<Entity> best;
    int bestHp = wantHighestHp ? -1 : std::numeric_limits<int>::max();
    for (const auto& entity : board.getEntities()) {
        if (entity->team == myTeam || !entity->isAlive() || !entity->isTargetable() || entity->isBuilding()) continue;
        if (maxRadius > 0.0f && origin.distanceTo(entity->position) > maxRadius) continue;
        if ((wantHighestHp && entity->hp > bestHp) || (!wantHighestHp && entity->hp < bestHp)) {
            bestHp = entity->hp;
            best = entity;
        }
    }
    return best;
}
