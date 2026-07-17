#pragma once
#include "CardEntity.h"
#include "Board.h"
#include "OnHitEffect.h"
#include <memory>
#include <limits>
#include <vector>
#include <algorithm>

class CombatEntity : public CardEntity {
protected:
    float attackRange;
    int damage;
    int attackCooldown;
    float currentCooldown;
    std::vector<std::shared_ptr<IOnHitEffect>> onHitEffects;

public:
    // Freeze only ever means anything to something that attacks or moves
    // (cooldown/speed slowdown below), so it lives here rather than on
    // Entity -- AreaSpell and Projectile can never be frozen, since neither
    // is ever a valid findTarget() result (both are isTargetable() == false).
    int freezeTicks = 0;
    float freezeSlow = 1.0f;

    // Whether this attacker's findTarget() may pick a flying candidate.
    // Lives here (not Entity) because only things that attack care --
    // Troop and Building alike (Inferno Tower/Tesla hit air, Cannon/Bomb
    // Tower don't) -- and it's only ever read on `this`, never cast off a
    // generic candidate the way isFlying is (see Entity.h).
    bool targetsAir = false;

    CombatEntity(int id, float x, float y, int hp, int team, char symbol,
        float attackRange, int damage, int attackCooldown)
        : CardEntity(id, x, y, hp, team, symbol),
        attackRange(attackRange), damage(damage),
        attackCooldown(attackCooldown), currentCooldown(0.0f) {}

    void applyFreeze(int ticks, float slowFactor) {
        // Duration and strength are judged independently so a new freeze can
        // never leave the target better off than it already was: a shorter
        // but stronger slow no longer gets silently dropped just because a
        // longer, weaker one is already active.
        freezeTicks = std::max(freezeTicks, ticks);
        freezeSlow = std::min(freezeSlow, slowFactor);
    }

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
            if (targetRadius <= 0.0f) targetRadius = Entity::IMPLICIT_TROOP_RADIUS;

            float myRadius = this->getCollisionRadius();
            if (myRadius <= 0.0f) myRadius = Entity::IMPLICIT_TROOP_RADIUS;

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
    // Entity, not CombatEntity: targeting itself doesn't care about freeze
    // or on-hit effects, and every other consumer of findTarget's result
    // (range math, movement) only ever needs Entity's own surface. Keeping
    // this Entity-typed means the only place that needs to know "is this
    // actually a CombatEntity" is applyOnHitEffects below, where it's
    // genuinely required -- not the whole targeting system.
    virtual std::shared_ptr<Entity> findTarget(Board& board) const {
        std::shared_ptr<Entity> closestTarget = nullptr;
        float minDistance = std::numeric_limits<float>::max();

        for (const auto& entity : board.getEntities()) {
            if (entity->team != this->team && entity->isAlive() && entity->isTargetable() && entity->id != this->id
                && (!entity->isFlying || targetsAir)) {
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
    // On-hit effects (freeze, etc.) only ever mean something against a
    // CombatEntity, so the cast happens here, once, rather than forcing
    // every target-typed signature in the codebase to narrow to
    // CombatEntity just to serve this one specific need.
    void applyOnHitEffects(const std::shared_ptr<Entity>& target) const {
        if (onHitEffects.empty()) return;
        auto combatTarget = std::dynamic_pointer_cast<CombatEntity>(target);
        if (!combatTarget) return; // not something on-hit effects can apply to
        for (const auto& effect : onHitEffects) {
            effect->apply(combatTarget);
        }
    }

    virtual void moveTowards(Board& board, const Vector2D& dest) {
        // Default: stationary entities don't move
    }
};
