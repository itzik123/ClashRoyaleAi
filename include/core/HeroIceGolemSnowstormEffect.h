#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "AreaSpell.h"
#include "FreezeOnHit.h"
#include "Board.h"

// Hero Ice Golem's "Snowstorm": 3 staggered blasts in a fixed radius --
// the first two apply pushback+damage+a partial slow, the third a full
// freeze -- reusing MightyMinerEscapeEffect's own pattern of constructing
// AreaSpells directly from within an ability effect (not through
// CardFactories::spawnSpell, since this fires mid-battle, not from a card
// play). Damage to Crown Towers is reduced via AreaSpell's own
// spellTowerDamageMultiplier (a new primitive -- see that field's own
// comment for why no existing mechanism covers this).
class HeroIceGolemSnowstormEffect : public IAbilityEffect {
    float radius;
    int blastDamage;
    float knockback;
    int slowTicks;
    float slowFactor;
    int freezeTicks;

public:
    HeroIceGolemSnowstormEffect(float radius, int blastDamage, float knockback,
            int slowTicks, float slowFactor, int freezeTicks)
        : radius(radius), blastDamage(blastDamage), knockback(knockback),
          slowTicks(slowTicks), slowFactor(slowFactor), freezeTicks(freezeTicks) {}

    void apply(Board& board, CombatEntity& self) const override {
        const int STAGE_GAP_TICKS = 5;
        for (int stage = 0; stage < 3; ++stage) {
            bool isFinalBlast = (stage == 2);
            auto onHit = isFinalBlast
                ? std::make_shared<FreezeOnHit>(freezeTicks, 0.0f)
                : std::make_shared<FreezeOnHit>(slowTicks, slowFactor);
            auto blast = std::make_shared<AreaSpell>(
                board.allocateId(), self.position.x, self.position.y, self.team,
                radius, blastDamage, stage * STAGE_GAP_TICKS, /*symbol=*/'*', onHit,
                /*groundOnly=*/false, /*remainingHits=*/1, /*tickInterval=*/0,
                /*buffsAllies=*/false, /*buffMultiplier=*/1.0f, /*buffDurationTicks=*/0,
                /*knockback=*/isFinalBlast ? 0.0f : knockback, /*spawnOnDetonate=*/nullptr,
                /*clonesAllies=*/false, /*targetTopHpCount=*/0,
                /*tieredDamage=*/false, 0, 0, 0,
                /*spellTowerDamageMultiplier=*/0.05f);
            board.addEntity(blast);
        }
    }
};
