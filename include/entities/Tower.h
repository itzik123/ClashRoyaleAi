#pragma once
#include "Building.h"
#include "Projectile.h"

class Tower : public Building {
public:
    Tower(int id, float x, float y, int hp, int team,
        float attackRange, int damage, int attackCooldown, char symbol)
        : Building(id, x, y, hp, team, symbol, attackRange, damage, attackCooldown, -1) {
        targetsAir = true; // every tower in Clash Royale defends against air
        // Default sightRange to attackRange -- never less than what it can
        // already hit, regardless of construction path. GameManager sets
        // the exact sourced values (King 7.0, Princess Tower variants 7.5
        // via towerTroopStats/applyCardMetadata) explicitly after this;
        // this is just a sane fallback for a Tower built any other way
        // (tests, etc).
        sightRange = attackRange;
    }

    float getCollisionRadius() const override {
        return (symbol == 'R') ? 2.0f : 1.5f;
    }

    // Target-selection footprint, larger than the collision radius and fitted
    // to observed play: a Hog Rider with a Cannon 8 or 7 tiles off-lane walks
    // at the Princess Tower, and only diverts at 6. Centre-to-centre made it
    // divert in all three; the collision radius (1.5) is too small to change
    // that. The band that reproduces the real behaviour is 2.92 < r < 3.65
    // against a Cannon's own 1.0, so 3.3 sits in the middle of it.
    //
    // The King's is scaled by the same ratio its collision radius carries
    // (2.0 / 1.5), keeping the two towers proportional rather than
    // independently tuned.
    float getTargetingRadius() const override {
        return (symbol == 'R') ? 4.4f : 3.3f;
    }

    bool isTower() const override { return true; }

    // Board::deepCopy. Its own override, not Building's inherited one, which
    // would slice a Tower down to a plain Building -- losing isTower() and
    // with it the findTarget fallback that makes towers always-visible
    // destinations, so troops in a rollout would wander instead of pushing.
    // The implicit copy constructor carries `awake` with it, so a snapshotted
    // King keeps its dormancy and a rollout cannot silently arm it.
    std::shared_ptr<Entity> snapshot() const override {
        return std::make_shared<Tower>(*this);
    }

    // ---- activation ----
    //
    // Real Clash Royale's King Tower is DORMANT at the start of a match: it
    // cannot acquire a target or fire until it is activated, permanently, by
    // either taking any damage or by losing a Princess Tower on its own team.
    // Until 2026-08-21 this engine's King fired from tick 0 -- long-standing,
    // recorded as perception/UPSTREAM_REQUESTS.md item 3, and made worse by the
    // 2026-08-20 sight fix, which widened its effective reach from 7.0 to 9.4
    // tiles and took its share of the damage against a lone Hog from 5.3% to
    // 36.8%. All of that came from a tower that should have been asleep.
    //
    // Only the King is ever built asleep; see GameManager::addTower. A Princess
    // Tower constructs awake and stays awake.
    bool isAwake() const { return awake; }
    void wake() { awake = true; }
    void sleep() { awake = false; }

    void update(Board& board) override {
        if (!awake) {
            // Trigger 1: ANY damage. Latched on the HP invariant rather than by
            // overriding a damage entry point -- damage reaches a tower through
            // Projectile::applyHit, AreaSpell::update, a direct performAttack,
            // splash and poison-over-time, and hooking one of those would
            // silently miss the rest. Because this only ever SETS the flag, a
            // later heal cannot put the King back to sleep.
            if (hp < maxHp) awake = true;

            // Trigger 2: a friendly Princess Tower destroyed.
            //
            // The count is RECORDED on the first update rather than compared
            // against a hardcoded 2. A King built on a bare board -- which
            // every unit test does -- has zero friendly Princesses, and a rule
            // phrased as "fewer than 2 alive" would wake it instantly on tick
            // one. Recording what it actually started with is what makes the
            // rule mean "one of mine died".
            int living = 0;
            for (const auto& e : board.getEntities()) {
                if (e->isAlive() && e->isTower() && e->team == team && e->symbol != 'R') ++living;
            }
            if (initialFriendlyPrincesses < 0) initialFriendlyPrincesses = living;
            else if (living < initialFriendlyPrincesses) awake = true;
        }
        Building::update(board);
    }

protected:
    // The single choke point for dormancy. No acquisition means no lock, so
    // resolveCurrentTarget finds nothing and performAttack is never reached --
    // Building::update needs no change at all. The tower stays targetable and
    // damageable, which is exactly what lets trigger 1 above ever fire.
    std::shared_ptr<Entity> findTarget(Board& board) const override {
        if (!awake) return nullptr;
        return CombatEntity::findTarget(board);
    }

    void performAttack(Board& board, std::shared_ptr<Entity> target) override {
        // lineSplash/lineSplashRange forwarded -- same omission, same reason,
        // as RangedBuildingTargeter::performAttack. A Tower Troop
        // (TowerTroops.h) is built through CardFactories::applyCardMetadata
        // like any other card and so can carry these fields.
        auto arrow = std::make_shared<Projectile>(
            board.allocateId(), position.x, position.y, team, target, 2.0f, getCurrentDamage(), onHitEffects,
            false, 0, id, cardId, splashRadius, lineSplash, lineSplashRange);
        board.addEntity(arrow);
    }

private:
    // Princess Towers; only the King is put to sleep, by GameManager::addTower.
    bool awake = true;
    // -1 until the first update() records what this team actually started with.
    int initialFriendlyPrincesses = -1;
};