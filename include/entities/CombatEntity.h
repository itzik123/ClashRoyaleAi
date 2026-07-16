#pragma once
#include "CardEntity.h"
#include "Board.h"
#include "OnHitEffect.h"
#include <memory>
#include <limits>
#include <vector>

class CombatEntity : public CardEntity {
protected:
    float attackRange;
    int damage;
    int attackCooldown;
    float currentCooldown;
    std::vector<std::shared_ptr<IOnHitEffect>> onHitEffects;

public:
    CombatEntity(int id, float x, float y, int hp, int team, char symbol,
        float attackRange, int damage, int attackCooldown)
        : CardEntity(id, x, y, hp, team, symbol),
        attackRange(attackRange), damage(damage),
        attackCooldown(attackCooldown), currentCooldown(0.0f) {}

    // Composes extra behavior (e.g. freeze) onto every successful attack,
    // without needing a bespoke Entity subclass per effect combination.
    void addOnHitEffect(std::shared_ptr<IOnHitEffect> effect) {
        onHitEffects.push_back(std::move(effect));
    }

    void update(Board& board) override {
        if (freezeTicks > 0) {
            freezeTicks--;
            if (currentCooldown > 0.0f) {
                currentCooldown -= freezeSlow;
            }
        } else {
            if (currentCooldown > 0.0f) {
                currentCooldown -= 1.0f;
            }
        }
        
        if (currentCooldown < 0.0f) currentCooldown = 0.0f;

        auto target = findTarget(board);
        if (target) {
            float dist = position.distanceTo(target->position);
            
            float targetRadius = target->getCollisionRadius();
            if (targetRadius <= 0.0f) targetRadius = 0.4f; // Implicit radius for troops
            
            float myRadius = this->getCollisionRadius();
            if (myRadius <= 0.0f) myRadius = 0.4f; // Implicit radius for troops
            
            float effectiveAttackRange = attackRange + myRadius + targetRadius;

            if (dist <= effectiveAttackRange) {
                if (currentCooldown == 0.0f) {
                    // Effects are applied by performAttack itself, not here,
                    // because *when* they should fire depends on *when* the
                    // damage actually lands: instantly for a direct hit, but
                    // only on arrival for an attack that spawns a projectile.
                    performAttack(board, target);
                    currentCooldown = static_cast<float>(attackCooldown);
                }
            } else {
                moveTowards(board, target->position);
            }
        }
        
        clampPosition(board);
    }

protected:
    virtual std::shared_ptr<Entity> findTarget(Board& board) const {
        std::shared_ptr<Entity> closestTarget = nullptr;
        float minDistance = std::numeric_limits<float>::max();

        for (const auto& entity : board.getEntities()) {
            if (entity->team != this->team && entity->isAlive() && entity->isTargetable() && entity->id != this->id) {
                float dist = position.distanceTo(entity->position);
                if (dist < minDistance) {
                    minDistance = dist;
                    closestTarget = entity;
                }
            }
        }
        return closestTarget;
    }

    virtual void performAttack(Board& board, std::shared_ptr<Entity> target) = 0;

    // Called by a direct-damage performAttack override at the exact moment
    // its damage lands. Ranged attacks don't call this -- they hand
    // onHitEffects to the Projectile instead, so effects land with the hit.
    void applyOnHitEffects(const std::shared_ptr<Entity>& target) const {
        for (const auto& effect : onHitEffects) {
            effect->apply(target);
        }
    }

    virtual void moveTowards(Board& board, const Vector2D& dest) {
        // Default: stationary entities don't move
    }
};
