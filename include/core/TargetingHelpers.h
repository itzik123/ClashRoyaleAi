#pragma once
#include "Board.h"
#include "Entity.h"
#include <limits>
#include <memory>

// The enemy troop with the highest or lowest HP, for Hero Giant's throw
// (highest, within range) and Hero Mega Minion's warp (lowest; maxRadius <= 0
// means unbounded). Excludes every building, since both abilities target
// troops. nullptr if none.
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
