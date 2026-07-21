#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "AreaSpell.h"
#include "Board.h"

// Goblinstein's "Lightning Link": anchors a repeating shock zone at the
// Monster's current position (self here IS the Monster -- see its
// CardRegistry entry) -- reuses AreaSpell's existing multi-tick support
// (remainingHits/tickInterval, the same mechanism Poison/Graveyard already
// use) rather than a new "periodic ability" primitive. This is actually a
// closer match to the real card than "follows the Monster around" would
// be: the sourced mechanic explicitly anchors to a fixed ground position
// (a "glowing receiver" stays behind and keeps working even after the
// Monster dies), which a stationary AreaSpell models directly.
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
