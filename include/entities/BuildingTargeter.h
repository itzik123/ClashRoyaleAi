#pragma once
#include "Troop.h"
#include "Building.h"
#include "LanePath.h"
#include "StatsEvents.h"

class BuildingTargeter : public Troop {
public:
    BuildingTargeter(int id, float x, float y, int hp, int team,
        float speed, float attackRange, int damage, int attackCooldown, char symbol)
        : Troop(id, x, y, hp, team, symbol, speed, attackRange, damage, attackCooldown) {}

    std::shared_ptr<Entity> clone(int newId) const override {
        auto copy = std::make_shared<BuildingTargeter>(*this);
        copy->id = newId;
        copy->hp = 1; // Clone: full damage, 1 hp
        return copy;
    }

    // Board::deepCopy -- id and hp preserved exactly, unlike clone() above.
    std::shared_ptr<Entity> snapshot() const override {
        return std::make_shared<BuildingTargeter>(*this);
    }

protected:
    // Same two-tier sight-range/tower-fallback shape as
    // CombatEntity::findTarget's own (see its comment), just filtered to
    // buildings only -- a building-targeter never considers troops at
    // all. Since every enemy Tower is itself a Building, this is really
    // "closest non-tower defensive building within sightRange, else the
    // closest tower regardless of distance," matching a building-
    // targeter's actual real-game behavior: engage a visible defensive
    // building, otherwise beeline for the enemy tower. isBuilding()
    // (Entity.h) replaces this method's own dynamic_pointer_cast<Building>
    // that used to live here.
    std::shared_ptr<Entity> findTarget(Board& board) const override {
        std::shared_ptr<Entity> closestInSight = nullptr;
        float minSightDistance = std::numeric_limits<float>::max();
        std::shared_ptr<Entity> closestTower = nullptr;
        float minTowerDistance = std::numeric_limits<float>::max();

        const float myRadius = ownEffectiveRadius();   // loop invariant
        for (const auto& entity : board.getEntities()) {
            if (entity->team == this->team || !entity->isAlive() || !entity->isTargetable()) continue;
            if (!entity->isBuilding()) continue;
            float dist = position.distanceTo(entity->position);

            // The nearest TOWER is tracked separately, but only to serve the
            // out-of-sight fallback below -- never as a competing tier.
            if (entity->isTower() && dist < minTowerDistance) {
                minTowerDistance = dist;
                closestTower = entity;
            }

            // NEAREST BUILDING IN SIGHT WINS, AND TOWERS COMPETE ON EQUAL
            // TERMS. This used to be an `else if`, which made every non-tower
            // building beat every tower at ANY distance -- so a Cannon parked
            // far off-lane out-prioritised a Princess Tower the unit was
            // already closer to.
            //
            // Measured 2026-08-28 on replays/hog_test_1.json: at the tick it
            // turned, the Hog was 7.714 tiles from the Princess Tower and
            // 9.160 from the Cannon. It abandoned the nearer objective to walk
            // at the further one, and stayed wrong for seven more ticks until
            // the Cannon actually became closest at tick 22.
            //
            // A building-targeter in the real game walks at whichever BUILDING
            // is closest, and Crown Towers are buildings. That is one rule, not
            // two tiers, and the previous comment here stated the two-tier
            // behaviour as a deliberate model of the real game -- it was not.
            //
            // effectiveSightWith, not raw sightRange -- see CombatEntity's own
            // findTarget and the comment on effectiveSightTo for the measured
            // free-siege bug the mismatch caused.
            if (dist <= effectiveSightWith(myRadius, *entity) && dist < minSightDistance) {
                minSightDistance = dist;
                closestInSight = entity;
            }
        }
        if (closestInSight) return closestInSight;

        // Same lane rule as CombatEntity::findTarget, deliberately SHARED
        // rather than reimplemented: a building-targeter drifting out of
        // agreement with a troop about which tower its lane leads to would be
        // invisible until a Hog and its escort walked to different towers.
        // Every Tower is a Building, so the objective is always eligible here.
        auto laneTarget = LanePath::laneObjective(board, team, position);
        if (laneTarget && laneTarget->isBuilding()) return laneTarget;
        return closestTower;
    }

    void performAttack(Board& board, std::shared_ptr<Entity> target) override {
        int dealt = getCurrentDamage();
        target->takeDamage(dealt);
        board.statsEvents.notifyDamageDealt(
            { id, team, cardId, target->id, target->cardId, target->team, dealt, board.currentTick });
        applyOnHitEffects(target);
        applySplashDamage(board, target->position, splashRadius, target->id, id, team, cardId, dealt);
    }
};