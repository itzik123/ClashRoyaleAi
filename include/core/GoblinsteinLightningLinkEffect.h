#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "AreaSpell.h"
#include "Board.h"

// Goblinstein's "Lightning Link" (fired by the Monster): a repeating shock zone
// anchored where the Monster stands, as a multi-hit AreaSpell. Anchored rather
// than following him, like the real card's receiver, which keeps working after
// the Monster dies.
class GoblinsteinLightningLinkEffect : public IAbilityEffect {
    float radius;
    int damagePerTick;
    int tickIntervalTicks;
    int totalTicks;

public:
    GoblinsteinLightningLinkEffect(float radius, int damagePerTick, int tickIntervalTicks, int totalTicks)
        : radius(radius), damagePerTick(damagePerTick),
          tickIntervalTicks(tickIntervalTicks), totalTicks(totalTicks) {}

    void apply(Board& board, CombatEntity& self) const override {
        int hits = tickIntervalTicks > 0 ? (totalTicks / tickIntervalTicks) : 1;
        if (hits < 1) hits = 1;
        auto shock = std::make_shared<AreaSpell>(
            board.allocateId(), self.position.x, self.position.y, self.team,
            radius, damagePerTick, /*delayTicks=*/0, /*symbol=*/'*', /*onHit=*/nullptr,
            /*groundOnly=*/false, /*remainingHits=*/hits, /*tickInterval=*/tickIntervalTicks);
        board.addEntity(shock);
    }
};
