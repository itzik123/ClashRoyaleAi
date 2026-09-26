#pragma once
#include "DeathEffect.h"
#include "Board.h"
#include "AreaSpell.h"
#include <memory>
#include <string>

// A bomb dropped where the unit dies and detonated after a fuse (Giant
// Skeleton). Built as an AreaSpell with delayTicks = the fuse, so it reuses
// AreaSpell's timing, knockback, hitbox test, Crown Tower rule, snapshot and
// DamageDealtEvent attribution; an AreaSpell is untargetable, as the real bomb
// is. Unlike AreaDamageOnDeath, which lands on the tick of death.
//
// `fuseTicks` is the AreaSpell delay: an AreaSpell applies on the update after
// its delay reaches 0, and the bomb first updates the tick after the death, so
// a fuse of N detonates N + 1 ticks after the death.
class DelayedAreaDamageOnDeath : public IDeathEffect {
    float radius;
    int damage;
    int fuseTicks;
    float knockback;
    int sourceCardId;
    std::string bombName;

public:
    DelayedAreaDamageOnDeath(float radius, int damage, int fuseTicks, float knockback, int sourceCardId,
            std::string bombName)
        : radius(radius), damage(damage), fuseTicks(fuseTicks), knockback(knockback),
          sourceCardId(sourceCardId), bombName(std::move(bombName)) {}

    void apply(Board& board, const Vector2D& position, int team) const override {
        auto bomb = std::make_shared<AreaSpell>(board.allocateId(), position.x, position.y, team, radius, damage,
            fuseTicks, '*', nullptr, false, 1, 0, false, 1.0f, 0, knockback);
        bomb->name = bombName;
        bomb->cardId = sourceCardId;
        board.addEntity(bomb);
    }
};
