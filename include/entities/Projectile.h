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
    // Entity-typed, like findTarget's result; applyHit narrows to CombatEntity
    // where on-hit effects need it.
    std::weak_ptr<Entity> target;
    float speed;
    int damage;
    std::vector<std::shared_ptr<IOnHitEffect>> onHitEffects;

    // The shooter's identity, carried forward to stamp DamageDealtEvent on
    // arrival.
    int attackerId;
    int attackerCardId;

    // The shooter's splash radius (CombatEntity::applySplashDamage); 0 is a
    // single-target hit.
    float splashRadius;

    // Piercing line (Bowler, Magic Archer; see
    // CombatEntity::applyLineSplashDamage). With lineSplash set, splashRadius
    // is the line's half-width. `origin` is where the shot was fired from,
    // since `position` snaps to the impact point.
    bool lineSplash;
    float lineSplashRange;
    Vector2D origin;

    // Boomerang (Executioner): hits its target on arrival and again
    // returnDelayTicks later. The real axe pierces everything along its path
    // both ways; there is no path-collision primitive, so it hits its one
    // target twice.
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

    // For Board::deepCopy. The copy still points `target` at the original
    // board's entity; Board::deepCopy repairs that with
    // remapSnapshotReferences(), which is a no-op for every other type. Not
    // cleared defensively: a projectile with no target dies next tick, which
    // would silently delete every shot in flight from a rollout.
    std::shared_ptr<Entity> snapshot() const override {
        return std::make_shared<Projectile>(*this);
    }

    // Re-points this shot at the equivalent entity on the copied board. A
    // target missing from the map (already dead) is cleared rather than left
    // pointing across boards.
    void remapSnapshotReferences(
        const std::unordered_map<int, std::shared_ptr<Entity>>& byOldId) override {
        auto t = target.lock();
        if (!t) return; // expired: nothing to repair
        auto it = byOldId.find(t->id);
        target = (it != byOldId.end())
            ? std::weak_ptr<Entity>(it->second)
            : std::weak_ptr<Entity>();
    }

    // The id this shot is homing on, or -1. Lets the deepCopy tests assert the
    // remap directly.
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
                position = t->position; // snap to the impact point
                applyHit(board, t);
                outboundHitLanded = true;
                if (!returnsToSender) hp = 0; // a normal projectile is done after one hit
                // else stay alive for the return trip
            } else {
                float dx = t->position.x - position.x;
                float dy = t->position.y - position.y;
                position.x += (dx / dist) * speed;
                position.y += (dy / dist) * speed;
            }
        } else if (returnDelayTicks > 0) {
            returnDelayTicks--;
        } else {
            applyHit(board, t); // return trip: second hit, same target
            hp = 0;
        }
    }

private:
    void applyHit(Board& board, const std::shared_ptr<Entity>& t) {
        t->takeDamage(damage);
        board.statsEvents.notifyDamageDealt(
            { attackerId, team, attackerCardId, t->id, t->cardId, t->team, damage, board.currentTick, t->isTower() });
        // On-hit effects (e.g. Ice Wizard's freeze) land on arrival, and only
        // against a CombatEntity.
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
