#pragma once
#include "Entity.h"
#include "CombatEntity.h"
#include "Board.h"
#include "OnHitEffect.h"
#include "StatsEvents.h"
#include <memory>
#include <unordered_map>
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

    // Who fired this -- needed only to stamp DamageDealtEvent on arrival
    // (the shooter itself is long done with its own performAttack() call by
    // the time this lands, so the projectile has to carry the identity
    // forward itself).
    int attackerId;
    int attackerCardId;

    // Splash damage (see CombatEntity::applySplashDamage) -- the shooter's
    // own splashRadius, carried forward the same way attackerId/
    // attackerCardId are, since the shooter is done with its own
    // performAttack() by the time this arrives. 0.0f (the default) is
    // every non-splash card's normal single-target hit.
    float splashRadius;

    // Piercing-line hit (Bowler, Magic Archer) -- see
    // CombatEntity::applyLineSplashDamage. When lineSplash is set,
    // splashRadius above doubles as the line's half-width instead of a
    // circle radius. origin is the shooter's own position at the moment
    // it fired, captured in the constructor before this projectile starts
    // moving (position itself gets snapped to the impact point on
    // arrival, so it can't be reused for the line's start point by then).
    bool lineSplash;
    float lineSplashRange;
    Vector2D origin;

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
        bool returnsToSender = false, int returnDelayTicks = 0,
        int attackerId = -1, int attackerCardId = -1, float splashRadius = 0.0f,
        bool lineSplash = false, float lineSplashRange = 0.0f)
        : Entity(id, x, y, 1, team, '-'), target(target), speed(speed), damage(damage),
        onHitEffects(std::move(onHitEffects)), attackerId(attackerId), attackerCardId(attackerCardId),
        splashRadius(splashRadius), lineSplash(lineSplash), lineSplashRange(lineSplashRange), origin{ x, y },
        returnsToSender(returnsToSender), returnDelayTicks(returnDelayTicks) {}

    bool isTargetable() const override { return false; }

    // Board::deepCopy. UNSAFE ON ITS OWN -- the implicit copy constructor
    // carries `target` over verbatim, so the copy homes on, and deals damage
    // to, an entity belonging to whatever board the ORIGINAL lives on. That is
    // repaired by remapSnapshotReferences() below, which Board::deepCopy calls
    // on every copied entity; this is the one type in the hierarchy where that
    // second pass is not a no-op.
    //
    // Deliberately a plain copy rather than defensively clearing the target:
    // a projectile with no target dies on its next tick (update() sets hp = 0
    // when the lock fails), so a "safe" default would silently delete every
    // shot in flight from every rollout -- the same quiet wrongness this whole
    // mechanism exists to prevent. Better that the remap is mandatory and the
    // tests prove it is happening.
    std::shared_ptr<Entity> snapshot() const override {
        return std::make_shared<Projectile>(*this);
    }

    // Re-points this shot at the equivalent entity in the copied board. The
    // lookup lives here rather than in Board because `target` is private and
    // this is the only member in the hierarchy that needs repairing -- Board
    // stays ignorant of concrete entity types, exactly like onDeath() and
    // clampPosition() above it.
    //
    // A target that is missing from the map (already dead and erased, so the
    // weak_ptr has expired) is cleared rather than left pointing across
    // boards. Clearing costs one projectile that was about to die anyway;
    // leaving it would be a live cross-board write.
    void remapSnapshotReferences(
        const std::unordered_map<int, std::shared_ptr<Entity>>& byOldId) override {
        auto t = target.lock();
        if (!t) return; // already expired: nothing aliased, nothing to repair
        auto it = byOldId.find(t->id);
        target = (it != byOldId.end())
            ? std::weak_ptr<Entity>(it->second)
            : std::weak_ptr<Entity>();
    }

    // Which entity this shot is homing on, or -1 if that target is gone.
    // Read-only and id-valued: no caller can obtain a handle to another
    // board's entity through it. Exists so the deepCopy divergence tests can
    // assert the remap actually happened, instead of inferring it from
    // damage landing in the right place.
    int getTargetId() const {
        auto t = target.lock();
        return t ? t->id : -1;
    }

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
                applyHit(board, t);
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
            applyHit(board, t); // return trip complete: second hit, same target
            hp = 0;
        }
    }

private:
    void applyHit(Board& board, const std::shared_ptr<Entity>& t) {
        t->takeDamage(damage);
        board.statsEvents.notifyDamageDealt(
            { attackerId, team, attackerCardId, t->id, t->cardId, t->team, damage, board.currentTick, t->isTower() });
        // On-hit effects (e.g. Ice Wizard's freeze) fire on arrival, not
        // when the shot was fired -- they ride along with the projectile
        // instead of applying instantly at the shooter. Only meaningful
        // against a CombatEntity (freeze etc.), so the cast happens right
        // here, the one place it's needed.
        if (!onHitEffects.empty()) {
            if (auto combatTarget = std::dynamic_pointer_cast<CombatEntity>(t)) {
                for (const auto& effect : onHitEffects) {
                    effect->apply(combatTarget);
                }
            }
        }
        if (lineSplash) {
            applyLineSplashDamage(board, origin, t->position, lineSplashRange, splashRadius,
                t->id, attackerId, team, attackerCardId, damage);
        } else {
            applySplashDamage(board, t->position, splashRadius, t->id, attackerId, team, attackerCardId, damage);
        }
    }
};
