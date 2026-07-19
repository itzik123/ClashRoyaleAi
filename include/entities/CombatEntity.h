#pragma once
#include "CardEntity.h"
#include "Board.h"
#include "OnHitEffect.h"
#include "DeathEffect.h"
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

    // Ramp bookkeeping: which target this attacker has been locked onto, and
    // for how many consecutive ticks. Tracked unconditionally (cheap, two
    // ints) even for the vast majority of entities that never ramp, so the
    // logic lives in exactly one place instead of being opt-in duplicated.
    int currentTargetId = -1;
    int ticksOnTarget = 0;

    // How many targets the attack actually landed on this time (1 normally,
    // up to maxSplitTargets otherwise) -- set right before performAttack()
    // is invoked, purely so getCurrentDamage() can divide by it.
    int currentHitCount = 1;

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

    // Fired once by onDeath() below (e.g. Golem spawning two Golemites).
    // Lives here, not Entity, for the same reason as onHitEffects: only
    // something that's a real combatant ever has one.
    std::shared_ptr<IDeathEffect> deathEffect;

    // Ramping damage (Inferno Tower): `damage` scales up the longer this
    // attacker stays locked onto the *same* target, reaching rampStartFraction
    // until rampMidTick, rampMidFraction until rampFullTick, and full damage
    // after -- reset by a target switch, losing the target, or being frozen
    // (matches the real "stun resets the charge" rule). rampFullTick == 0
    // (the default) disables ramping entirely: getCurrentDamage() is just
    // `damage`, unchanged, for every card that doesn't opt in.
    int rampMidTick = 0;
    int rampFullTick = 0;
    float rampStartFraction = 1.0f;
    float rampMidFraction = 1.0f;

    // Split-target attacks (Electro Wizard): instead of hitting only the
    // closest enemy, hits up to this many of the closest enemies at once,
    // each for damage / (however many were actually found this attack) --
    // full damage if only one target is in range, matching the real card.
    // 1 (the default) is the normal single-target case every other card uses.
    int maxSplitTargets = 1;

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
        // Captured before the decrement below so a freeze that's about to
        // expire this very tick still counts as "was frozen" for the ramp
        // reset -- matches the real "a stun resets the charge" rule for
        // every tick actually spent frozen, not all-but-the-last one.
        bool wasFrozen = freezeTicks > 0;

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

        // Target-lock: once committed to a target, stay on it -- attacking
        // or chasing -- instead of re-picking "whoever's closest" every
        // tick. Matches the real game: a unit mid-fight doesn't get
        // distracted just because something else wandered closer. Only
        // reacquires when there's no valid lock at all, or (stationary
        // attackers only, see canMove()) the lock has walked out of range
        // with no way to close the gap -- a mobile attacker never
        // force-drops on range alone, it just keeps chasing.
        auto target = resolveCurrentTarget(board);
        if (target && !canMove() && position.distanceTo(target->position) > effectiveRangeTo(target)) {
            target = nullptr; // out of reach, can't chase: drop the lock
        }
        if (!target) {
            target = findTarget(board);
        }

        if (target) {
            if (wasFrozen || target->id != currentTargetId) {
                currentTargetId = target->id;
                ticksOnTarget = 0;
            } else {
                ticksOnTarget++;
            }

            float dist = position.distanceTo(target->position);
            float effectiveAttackRange = effectiveRangeTo(target);

            if (dist <= effectiveAttackRange) {
                if (currentCooldown == 0.0f) {
                    // Effects are applied by performAttack itself, not here,
                    // because *when* they should fire depends on *when* the
                    // damage actually lands: instantly for a direct hit, but
                    // only on arrival for an attack that spawns a projectile.
                    if (maxSplitTargets <= 1) {
                        currentHitCount = 1;
                        performAttack(board, target);
                    } else {
                        auto targets = findSplitTargets(board, maxSplitTargets);
                        currentHitCount = static_cast<int>(targets.size());
                        for (const auto& t : targets) {
                            performAttack(board, t);
                        }
                    }
                    currentCooldown = static_cast<float>(attackCooldown);
                }
            } else {
                moveTowards(board, target->position);
            }
        } else {
            currentTargetId = -1;
            ticksOnTarget = 0;
        }

        clampPosition(board);
    }

    void onDeath(Board& board) override {
        if (deathEffect) deathEffect->apply(board, position, team);
    }

protected:
    // Whether this attacker can make progress toward a target that's out of
    // range this tick. True (the default) for every mobile troop; Building
    // overrides this to false, since its moveTowards is a no-op -- a locked
    // target that walks out of a stationary building's fixed range can
    // never be reached by chasing, so the lock has to be dropped instead of
    // held forever (matches the real game: a Cannon a troop walks past and
    // out of range re-targets immediately, but a Musketeer chasing someone
    // across the arena never gives up just because the gap grew).
    virtual bool canMove() const { return true; }

    // Entity, not CombatEntity: targeting itself doesn't care about freeze
    // or on-hit effects, and every other consumer of findTarget's result
    // (range math, movement) only ever needs Entity's own surface. Keeping
    // this Entity-typed means the only place that needs to know "is this
    // actually a CombatEntity" is applyOnHitEffects below, where it's
    // genuinely required -- not the whole targeting system.
    bool isValidTarget(const std::shared_ptr<Entity>& entity) const {
        return entity && entity->team != this->team && entity->isAlive() && entity->isTargetable()
            && entity->id != this->id && (!entity->isFlying || targetsAir);
    }

    float effectiveRangeTo(const std::shared_ptr<Entity>& target) const {
        float targetRadius = target->getCollisionRadius();
        if (targetRadius <= 0.0f) targetRadius = Entity::IMPLICIT_TROOP_RADIUS;
        float myRadius = this->getCollisionRadius();
        if (myRadius <= 0.0f) myRadius = Entity::IMPLICIT_TROOP_RADIUS;
        return attackRange + myRadius + targetRadius;
    }

    // Re-validates the currently-locked target (by id) rather than running
    // a full closest-enemy scan -- Board has no id index, so this is still
    // a linear pass, but it's the one that lets a locked-on attacker keep
    // its target instead of findTarget() picking a new "closest" every
    // tick. Returns nullptr if there's no lock, or the locked entity no
    // longer exists / is no longer a legal target (dead, no longer
    // targetable, etc).
    std::shared_ptr<Entity> resolveCurrentTarget(Board& board) const {
        if (currentTargetId < 0) return nullptr;
        for (const auto& entity : board.getEntities()) {
            if (entity->id == currentTargetId) {
                return isValidTarget(entity) ? entity : nullptr;
            }
        }
        return nullptr;
    }

    virtual std::shared_ptr<Entity> findTarget(Board& board) const {
        std::shared_ptr<Entity> closestTarget = nullptr;
        float minDistance = std::numeric_limits<float>::max();

        for (const auto& entity : board.getEntities()) {
            if (isValidTarget(entity)) {
                float dist = position.distanceTo(entity->position);
                if (dist < minDistance) {
                    minDistance = dist;
                    closestTarget = entity;
                }
            }
        }
        return closestTarget;
    }

    // Only called when maxSplitTargets > 1 (Electro Wizard). Reuses
    // findTarget's own eligibility filter, just keeping the N closest
    // instead of only the closest one.
    std::vector<std::shared_ptr<Entity>> findSplitTargets(Board& board, int maxCount) const {
        std::vector<std::shared_ptr<Entity>> candidates;
        for (const auto& entity : board.getEntities()) {
            if (isValidTarget(entity)) {
                candidates.push_back(entity);
            }
        }
        std::sort(candidates.begin(), candidates.end(),
            [this](const std::shared_ptr<Entity>& a, const std::shared_ptr<Entity>& b) {
                return position.distanceTo(a->position) < position.distanceTo(b->position);
            });
        if (static_cast<int>(candidates.size()) > maxCount) candidates.resize(maxCount);
        return candidates;
    }

    // Ramped, then split across however many targets this attack actually
    // landed on. Both default to no-ops (rampFullTick == 0, currentHitCount
    // == 1), so this returns `damage` unchanged for every card that doesn't
    // opt into either mechanic.
    int getCurrentDamage() const {
        int base = damage;
        if (rampFullTick > 0) {
            float fraction = (ticksOnTarget >= rampFullTick) ? 1.0f
                : (ticksOnTarget >= rampMidTick) ? rampMidFraction
                : rampStartFraction;
            base = static_cast<int>(damage * fraction);
        }
        return currentHitCount > 1 ? base / currentHitCount : base;
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
