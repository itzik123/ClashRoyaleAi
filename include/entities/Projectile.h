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

public:
    Projectile(int id, float x, float y, int team, std::weak_ptr<Entity> target, float speed, int damage,
        std::vector<std::shared_ptr<IOnHitEffect>> onHitEffects = {})
        : Entity(id, x, y, 1, team, '-'), target(target), speed(speed), damage(damage),
        onHitEffects(std::move(onHitEffects)) {}

    bool isTargetable() const override { return false; }

    void update(Board& board) override {
        if (auto t = target.lock()) {
            if (!t->isAlive()) {
                hp = 0;
                return;
            }

            float dist = position.distanceTo(t->position);
            if (dist <= speed) {
                position = t->position; // snap to the impact point before dying
                t->takeDamage(damage);
                // On-hit effects (e.g. Ice Wizard's freeze) fire on arrival,
                // not when the shot was fired -- they ride along with the
                // projectile instead of applying instantly at the shooter.
                // Only meaningful against a CombatEntity (freeze etc.), so
                // the cast happens right here, the one place it's needed.
                if (!onHitEffects.empty()) {
                    if (auto combatTarget = std::dynamic_pointer_cast<CombatEntity>(t)) {
                        for (const auto& effect : onHitEffects) {
                            effect->apply(combatTarget);
                        }
                    }
                }
                hp = 0;
            } else {
                float dx = t->position.x - position.x;
                float dy = t->position.y - position.y;
                position.x += (dx / dist) * speed;
                position.y += (dy / dist) * speed;
            }
        } else {
            hp = 0;
        }
    }
};
