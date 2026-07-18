#pragma once
#include "Entity.h"
#include "CombatEntity.h"
#include "Board.h"
#include "OnHitEffect.h"
#include <memory>
#include <vector>

class Projectile : public Entity {
private:
    // Entity-typed, matching CombatEntity::findTarget's own result type
    // (performAttack passes that same target straight through) -- keeps
    // Projectile decoupled from the CombatEntity-only on-hit-effect
    // machinery below, which does its own narrowing where it's needed.
    std::weak_ptr<Entity> target;
    float speed;
    int damage;
    std::vector<std::shared_ptr<IOnHitEffect>> onHitEffects;

    // Boomerang support (Executioner): after the outbound hit lands, instead
    // of dying immediately, wait returnDelayTicks and hit the same target
    // again (if it's still alive) before dying. The real axe pierces every
    // enemy along its path both ways; this engine has no line/path
    // collision primitive, so the simplification is "hits its one target
    // twice, with the real gap between throw and return" rather than
    // "hits everyone in a line twice".
    bool returnsToSender;
    int returnDelayTicks;
    bool outboundHitLanded = false;

public:
    Projectile(int id, float x, float y, int team, std::weak_ptr<Entity> target, float speed, int damage,
        std::vector<std::shared_ptr<IOnHitEffect>> onHitEffects = {},
        bool returnsToSender = false, int returnDelayTicks = 0)
        : Entity(id, x, y, 1, team, '-'), target(target), speed(speed), damage(damage),
        onHitEffects(std::move(onHitEffects)), returnsToSender(returnsToSender),
        returnDelayTicks(returnDelayTicks) {}

    bool isTargetable() const override { return false; }

    void update(Board& board) override {
        auto t = target.lock();
        if (!t || !t->isAlive()) {
            hp = 0;
            return;
        }

        if (!outboundHitLanded) {
            float dist = position.distanceTo(t->position);
            if (dist <= speed) {
                position = t->position; // snap to the impact point before dying
                applyHit(t);
                outboundHitLanded = true;
                if (!returnsToSender) hp = 0; // normal projectile: done after one hit
                // else: stay alive, waiting out the return trip below
            } else {
                float dx = t->position.x - position.x;
                float dy = t->position.y - position.y;
                position.x += (dx / dist) * speed;
                position.y += (dy / dist) * speed;
            }
        } else if (returnDelayTicks > 0) {
            returnDelayTicks--;
        } else {
            applyHit(t); // return trip complete: second hit, same target
            hp = 0;
        }
    }

private:
    void applyHit(const std::shared_ptr<Entity>& t) {
        t->takeDamage(damage);
        // On-hit effects (e.g. Ice Wizard's freeze) fire on arrival, not
        // when the shot was fired -- they ride along with the projectile
        // instead of applying instantly at the shooter. Only meaningful
        // against a CombatEntity (freeze etc.), so the cast happens right
        // here, the one place it's needed.
        if (onHitEffects.empty()) return;
        if (auto combatTarget = std::dynamic_pointer_cast<CombatEntity>(t)) {
            for (const auto& effect : onHitEffects) {
                effect->apply(combatTarget);
            }
        }
    }
};
