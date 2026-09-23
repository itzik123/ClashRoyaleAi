#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "AreaSpell.h"
#include "Board.h"

// Mighty Miner's "Explosive Escape": swaps to the mirrored lane (x only) and
// leaves a delayed bomb at his original position, hitting ground and air.
// bombRadius and bombKnockback are not from published data.
class MightyMinerEscapeEffect : public IAbilityEffect {
    float bombRadius;
    int bombDamage;
    int bombDelayTicks;
    float bombKnockback;

public:
    MightyMinerEscapeEffect(float bombRadius, int bombDamage, int bombDelayTicks, float bombKnockback)
        : bombRadius(bombRadius), bombDamage(bombDamage),
          bombDelayTicks(bombDelayTicks), bombKnockback(bombKnockback) {}

    void apply(Board& board, CombatEntity& self) const override {
        Vector2D originalPosition = self.position;

        mirrorToOppositeLane(self, board.getWidth());
        // y unchanged: a lane swap, not a top/bottom flip.

        auto bomb = std::make_shared<AreaSpell>(
            board.allocateId(), originalPosition.x, originalPosition.y, self.team,
            bombRadius, bombDamage, bombDelayTicks, /*symbol=*/'*', /*onHit=*/nullptr,
            /*groundOnly=*/false, /*remainingHits=*/1, /*tickInterval=*/0,
            /*buffsAllies=*/false, /*buffMultiplier=*/1.0f, /*buffDurationTicks=*/0,
            /*knockback=*/bombKnockback);
        board.addEntity(bomb);
    }
};
