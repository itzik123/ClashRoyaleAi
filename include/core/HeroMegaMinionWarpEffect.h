#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "TargetingHelpers.h"
#include "Board.h"
#include "StatsEvents.h"

// Hero Mega Minion's "Wounding Warp": teleports to the lowest-HP enemy anywhere
// and deals bonus damage. The once-per-deploy limit and the initial cooldown
// come from the registration.
class HeroMegaMinionWarpEffect : public IAbilityEffect {
    int bonusDamage;

public:
    explicit HeroMegaMinionWarpEffect(int bonusDamage) : bonusDamage(bonusDamage) {}

    void apply(Board& board, CombatEntity& self) const override {
        auto victim = findHpExtremeEnemy(board, self.position, 0.0f, self.team, /*wantHighestHp=*/false);
        if (!victim) return;
        self.position = victim->position;
        victim->takeDamage(bonusDamage);
        board.statsEvents.notifyDamageDealt(
            { self.id, self.team, self.cardId, victim->id, victim->cardId, victim->team, bonusDamage, board.currentTick, victim->isTower() });
    }
};
