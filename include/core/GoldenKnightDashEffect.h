#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "Tower.h"
#include "Board.h"
#include <memory>

// Golden Knight's "Dashing Dash": chain-dashes to the nearest enemy within
// dashRange of his CURRENT position, dealing dashDamage each hop, up to
// maxDashes, stopping early with no target or after hitting a Crown Tower. The
// real ~1 s invulnerable chain resolves within one tick; nothing else acts in
// between.
class GoldenKnightDashEffect : public IAbilityEffect {
    int dashDamage;
    float dashRange;
    int maxDashes;

    std::shared_ptr<Entity> findNearestEnemy(Board& board, const CombatEntity& self) const {
        std::shared_ptr<Entity> closest;
        float minDist = dashRange;
        for (const auto& e : board.getEntities()) {
            if (e->team == self.team || !e->isAlive() || !e->isTargetable()) continue;
            if (e->isFlying) continue; // ground-only
            float d = self.position.distanceTo(e->position);
            if (d <= minDist) { minDist = d; closest = e; }
        }
        return closest;
    }

public:
    GoldenKnightDashEffect(int dashDamage, float dashRange, int maxDashes)
        : dashDamage(dashDamage), dashRange(dashRange), maxDashes(maxDashes) {}

    void apply(Board& board, CombatEntity& self) const override {
        for (int i = 0; i < maxDashes; ++i) {
            auto target = findNearestEnemy(board, self);
            if (!target) break;

            // Closes to melee adjacency (1.0 tile); pullToward never
            // overshoots, even when already closer.
            float dist = self.position.distanceTo(target->position);
            pullToward(self, target->position, dist - 1.0f);
            target->takeDamage(dashDamage);
            board.statsEvents.notifyDamageDealt(
                { self.id, self.team, self.cardId, target->id, target->cardId, target->team,
                  dashDamage, board.currentTick, target->isTower() });

            if (dynamic_cast<Tower*>(target.get()) != nullptr) break; // stops after a Crown Tower
        }
    }
};
