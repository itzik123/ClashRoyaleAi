#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "Tower.h"
#include "Board.h"
#include <memory>

// Golden Knight's "Dashing Dash": instantly chain-dashes from enemy to
// enemy within dashRange of his CURRENT position each hop (not his
// starting position), dealing dashDamage each time, up to maxDashes total,
// stopping early if no enemy is in range or the last one hit was a Crown
// Tower. Real dashes are invulnerable and spread over ~1s; this engine
// resolves the whole chain synchronously within one tick instead (a
// documented timing simplification, same category as this engine's other
// "collapse a multi-tick real-game animation into one instant" choices --
// e.g. Spirit troops detonating on launch, not arrival) -- since nothing
// else acts in between hops, the invulnerability is trivially true anyway.
class GoldenKnightDashEffect : public IAbilityEffect {
    int dashDamage;
    float dashRange;
    int maxDashes;

    std::shared_ptr<Entity> findNearestEnemy(Board& board, const CombatEntity& self) const {
        std::shared_ptr<Entity> closest;
        float minDist = dashRange;
        for (const auto& e : board.getEntities()) {
            if (e->team == self.team || !e->isAlive() || !e->isTargetable()) continue;
            if (e->isFlying) continue; // Golden Knight is ground-only
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

            // Closes to melee adjacency (1.0 tile), not exactly onto the
            // target's own position -- pullToward's own clamp (never
            // overshoots past the destination) keeps this safe even when
            // already closer than that.
            float dist = self.position.distanceTo(target->position);
            pullToward(self, target->position, dist - 1.0f);
            target->takeDamage(dashDamage);
            board.statsEvents.notifyDamageDealt(
                { self.id, self.team, self.cardId, target->id, target->cardId, target->team,
                  dashDamage, board.currentTick });

            if (dynamic_cast<Tower*>(target.get()) != nullptr) break; // stops after hitting a Crown Tower
        }
    }
};
