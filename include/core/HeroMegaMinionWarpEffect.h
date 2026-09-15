#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "TargetingHelpers.h"
#include "Board.h"
#include "StatsEvents.h"

// Hero Mega Minion's "Wounding Warp": teleports (infinite range, maxRadius
// 0 to findHpExtremeEnemy) to the lowest-HP enemy anywhere on the board and
// deals bonus damage on arrival. Gated to fire only once per deployment via
// the registration's own usesLimit=1, and unusable for the first 1.5s after
// spawn via CardStats::initialAbilityCooldownTicks -- neither is this
// effect's own concern.
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
