#pragma once
#include "CardEntity.h"
#include "CombatEntity.h"
#include "OnHitEffect.h"
#include "PeriodicEffect.h"
#include "Board.h"
#include "StatsEvents.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

class AreaSpell : public CardEntity {
private:
    float radius;
    int damage;
    int delayTicks;
    std::shared_ptr<IOnHitEffect> onHit;
    // Ground-only spells (The Log, Barbarian Barrel) opt in via
    // CardStats::withGroundOnly; most spells hit air and ground.
    bool groundOnly;

    // Multi-hit spells (Poison, Arrows' volleys): each application re-checks
    // who is in radius, so a unit can walk out partway. tickInterval, the gap
    // between hits, matters only when remainingHits > 1.
    int remainingHits;
    int tickInterval;

    // Rage: buffs allies in radius instead of damaging enemies; `damage` is
    // unused.
    bool buffsAllies;
    float buffMultiplier;
    int buffDurationTicks;

    // Positive pushes away from the spell, negative pulls toward it (Entity.h
    // pushAway / pullToward). Buildings take the damage but are never moved, as
    // in the real game; pushAway / pullToward enforce that themselves.
    float knockback;

    // Spell-spawns-troops (Goblin Barrel, Royal Delivery, Graveyard): fires
    // once per application at the spell's position.
    std::shared_ptr<IPeriodicEffect> spawnOnDetonate;

    // Clone: duplicates every ally in radius; each clone has a fresh id and 1
    // hp (Entity::clone).
    bool clonesAllies;

    // Vines: only the N highest-HP targets in radius. 0 means everyone.
    int targetTopHpCount;

    // Void: damage per target falls with the number caught, in three sourced
    // tiers (1 / 2-4 / 5+). `damage` is unused.
    bool tieredDamage;
    int tierSingleDamage;
    int tierFewDamage;
    int tierManyDamage;

    // Multiplier on damage to Crown Towers. Only Hero Ice Golem's Snowstorm
    // sets it (0.05); 1.0 elsewhere.
    float spellTowerDamageMultiplier;

    // Applications in total, for the spawn-ring index (remainingHits counts
    // down).
    int totalHits;

public:
    // Damage one application deals to a Crown Tower, when the card publishes
    // one (CardStats::spellCrownTowerDamage; Poison: 21). -1 keeps
    // spellTowerDamageMultiplier. Set after construction by
    // CardFactories::spawnSpell, like configureRoll.
    int crownTowerDamage = -1;

    // With a ring (Graveyard: 3.3 tiles), application k spawns at the k-th of
    // seven fixed points, cycled: the cast point, then six on the ring 60
    // degrees apart, starting on the caster's forward axis. The real layout is
    // seven fixed points, cycled, oriented by side (RoyaleAPI on the 12 Jan
    // 2026 rework) and near the edge; their exact positions are unpublished,
    // so these are an approximation. The centre point is the skeleton that
    // "rises right on the Crown Tower" when the spell is centred on one (June
    // 2026 notes): with every point on the ring, a Graveyard centred on a
    // defended tower never connected. 0 spawns every body at the cast point.
    float spawnRingRadius = 0.0f;
    AreaSpell(int id, float x, float y, int team, float radius, int damage, int delayTicks, char symbol = '*',
        std::shared_ptr<IOnHitEffect> onHit = nullptr, bool groundOnly = false,
        int remainingHits = 1, int tickInterval = 0,
        bool buffsAllies = false, float buffMultiplier = 1.0f, int buffDurationTicks = 0,
        float knockback = 0.0f, std::shared_ptr<IPeriodicEffect> spawnOnDetonate = nullptr,
        bool clonesAllies = false, int targetTopHpCount = 0,
        bool tieredDamage = false, int tierSingleDamage = 0, int tierFewDamage = 0, int tierManyDamage = 0,
        float spellTowerDamageMultiplier = 1.0f)
        : CardEntity(id, x, y, 1, team, symbol), radius(radius), damage(damage), delayTicks(delayTicks),
        onHit(std::move(onHit)), groundOnly(groundOnly), remainingHits(remainingHits), tickInterval(tickInterval),
        buffsAllies(buffsAllies), buffMultiplier(buffMultiplier), buffDurationTicks(buffDurationTicks),
        knockback(knockback), spawnOnDetonate(std::move(spawnOnDetonate)), clonesAllies(clonesAllies),
        targetTopHpCount(targetTopHpCount), tieredDamage(tieredDamage), tierSingleDamage(tierSingleDamage),
        tierFewDamage(tierFewDamage), tierManyDamage(tierManyDamage),
        spellTowerDamageMultiplier(spellTowerDamageMultiplier), totalHits(remainingHits) {}

    bool isTargetable() const override { return false; }

    // For Board::deepCopy: a spell mid-fuse or mid-volley resumes where the
    // original is.
    std::shared_ptr<Entity> snapshot() const override {
        return std::make_shared<AreaSpell>(*this);
    }

    // --- rolling sweep (The Log, Barbarian Barrel) ---
    // These roll forward from where they land, sweeping a rectangular corridor:
    // a fixed width across the roll axis and a length covered over time. Each
    // target is damaged at most once per roll (`sweptIds`).
    //
    // `rollSpeed` is tiles per tick, stated directly, not through
    // MOVEMENT_SPEED_SCALE, which converts troop speed tiers.
    float rollRange = 0.0f;   // total distance; 0 => not a roller
    float rollWidth = 0.0f;   // FULL width, not a radius
    float rollSpeed = 0.0f;   // tiles per tick
    float rollKnockback = 0.0f;
    float rollTravelled = 0.0f;
    int rollDirY = 0;         // +1 for team 0, -1 for team 1
    Vector2D rollOrigin{0.0f, 0.0f};
    std::vector<int> sweptIds;

    bool isRolling() const { return rollRange > 0.0f; }

    // Set after construction; the constructor's other call sites never roll.
    void configureRoll(float range, float width, float speed, float knockback) {
        rollRange = range;
        rollWidth = width;
        rollSpeed = speed;
        rollKnockback = knockback;
        rollDirY = (team == 0) ? 1 : -1;
        rollOrigin = position;
    }

    void update(Board& board) override {
        if (delayTicks > 0) {
            delayTicks--;
            return;
        }
        if (isRolling()) {
            updateRoll(board);
            return;
        }

        bool alliesOnly = buffsAllies || clonesAllies;
        // Collect, then clone: cloning mid-loop would grow the vector being
        // iterated.
        std::vector<std::shared_ptr<Entity>> toClone;

        // Collected up front because Vines (top-HP only) and Void (damage
        // depends on the catch) need the full set.
        //
        // In the disc if the target's HITBOX overlaps it: centre distance up to
        // radius plus the target's radius, the surface convention the rolling
        // sweep below and effectiveRangeTo already use (UPSTREAM_REQUESTS.md
        // item 32l). A centre test missed a troop 3.2 tiles inside a Poison.
        std::vector<std::shared_ptr<Entity>> candidates;
        for (const auto& entity : board.getEntities()) {
            bool teamMatches = alliesOnly ? (entity->team == this->team) : (entity->team != this->team);
            if (entity->isAlive() && entity->isTargetable() && teamMatches && entity->id != this->id
                && (!groundOnly || !entity->isFlying)
                && position.distanceTo(entity->position) <= radius + CombatEntity::effectiveRadiusOf(*entity)) {
                candidates.push_back(entity);
            }
        }

        if (targetTopHpCount > 0 && static_cast<int>(candidates.size()) > targetTopHpCount) {
            std::sort(candidates.begin(), candidates.end(),
                [](const std::shared_ptr<Entity>& a, const std::shared_ptr<Entity>& b) {
                    return a->hp > b->hp;
                });
            candidates.resize(targetTopHpCount);
        }

        int effectiveDamage = damage;
        if (tieredDamage) {
            int count = static_cast<int>(candidates.size());
            effectiveDamage = (count <= 1) ? tierSingleDamage : (count <= 4) ? tierFewDamage : tierManyDamage;
        }

        for (const auto& entity : candidates) {
            if (buffsAllies) {
                if (auto combatTarget = std::dynamic_pointer_cast<CombatEntity>(entity)) {
                    combatTarget->applyBuff(buffMultiplier, buffDurationTicks);
                }
                continue;
            }
            if (clonesAllies) {
                toClone.push_back(entity);
                continue;
            }
            // Tower damage: the card's own Crown Tower value where it has one,
            // else scaled by spellTowerDamageMultiplier.
            int dealt = entity->isTower() ? towerDamage(effectiveDamage) : effectiveDamage;
            entity->takeDamage(dealt);
            if (knockback != 0.0f) {
                if (knockback > 0.0f) {
                    pushAway(*entity, position, knockback);
                } else {
                    pullToward(*entity, position, -knockback);
                }
            }
            // The spell itself is the attacker; whatever cast it may already be
            // gone.
            board.statsEvents.notifyDamageDealt(
                { id, team, cardId, entity->id, entity->cardId, entity->team, dealt, board.currentTick, entity->isTower() });
            // On-hit effects only apply to a CombatEntity.
            if (onHit) {
                auto combatTarget = std::dynamic_pointer_cast<CombatEntity>(entity);
                if (combatTarget) onHit->apply(combatTarget);
            }
        }

        for (const auto& original : toClone) {
            auto copy = original->clone(board.allocateId());
            if (copy) board.addEntity(copy);
        }

        if (spawnOnDetonate) spawnOnDetonate->apply(board, spawnPoint(board), team);

        remainingHits--;
        if (remainingHits > 0) {
            // tickInterval ticks from this application to the next. The
            // countdown returns on each tick it decrements and applies on the
            // tick it finds 0, so it starts one short: seeding it with
            // tickInterval ran every multi-hit spell a tick slow (Poison 1.1 s,
            // Graveyard 0.6 s; UPSTREAM_REQUESTS.md item 32i).
            delayTicks = (tickInterval > 0) ? tickInterval - 1 : 0;
        } else {
            hp = 0;
        }
    }

private:
    int towerDamage(int troopDamage) const {
        return (crownTowerDamage >= 0) ? crownTowerDamage
                                       : static_cast<int>(troopDamage * spellTowerDamageMultiplier);
    }

    // Where this application's spawn rises (see spawnRingRadius), kept on the
    // board.
    Vector2D spawnPoint(const Board& board) const {
        if (spawnRingRadius <= 0.0f) return position;
        constexpr int POINTS = 7;
        const int slot = (totalHits - remainingHits) % POINTS;
        if (slot == 0) return position;
        constexpr float PI = 3.14159265f;
        const float forward = (team == 0) ? PI / 2.0f : -PI / 2.0f; // toward the enemy side
        const float angle = forward + static_cast<float>(slot - 1) * (PI / 3.0f);
        Vector2D p{ position.x + spawnRingRadius * std::cos(angle),
                    position.y + spawnRingRadius * std::sin(angle) };
        const float maxX = static_cast<float>(board.getWidth() - 1);
        const float maxY = static_cast<float>(board.getHeight() - 1);
        p.x = std::max(0.0f, std::min(p.x, maxX));
        p.y = std::max(0.0f, std::min(p.y, maxY));
        return p;
    }
    bool alreadySwept(int entityId) const {
        return std::find(sweptIds.begin(), sweptIds.end(), entityId) != sweptIds.end();
    }

    // One tick of a rolling sweep: advance, damage whatever the corridor newly
    // reached, and throw each victim in a direction set by where across the
    // corridor it was caught.
    void updateRoll(Board& board) {
        rollTravelled += rollSpeed;
        if (rollTravelled > rollRange) rollTravelled = rollRange;
        // The position tracks the leading edge, which is what the replay
        // records and the viewer draws from.
        position.y = rollOrigin.y + static_cast<float>(rollDirY) * rollTravelled;

        const float halfWidth = rollWidth * 0.5f;

        for (const auto& entity : board.getEntities()) {
            if (!entity->isAlive() || !entity->isTargetable()) continue;
            if (entity->team == this->team || entity->id == this->id) continue;
            if (groundOnly && entity->isFlying) continue;
            if (alreadySwept(entity->id)) continue;

            const float r = CombatEntity::effectiveRadiusOf(*entity);
            const float dx = entity->position.x - rollOrigin.x;
            // Offset along the roll direction, so one set of comparisons serves
            // both teams.
            const float dy = (entity->position.y - rollOrigin.y) * static_cast<float>(rollDirY);

            if (std::fabs(dx) > halfWidth + r) continue;  // outside the corridor
            if (dy + r < 0.0f) continue;                  // entirely behind the spawn point
            // Reached when the leading edge touches the target's near surface,
            // not its centre (the effectiveRangeTo convention). From the
            // bridge, The Log's 10.1 reaches an enemy Princess Tower's near
            // edge (9.00 away) but never its centre (10.50); pinned in
            // tests/entities/test_area_spell.cpp.
            if (dy - r > rollTravelled) continue;

            sweptIds.push_back(entity->id);

            const int dealt = entity->isTower() ? towerDamage(damage) : damage;
            entity->takeDamage(dealt);
            board.statsEvents.notifyDamageDealt(
                { id, team, cardId, entity->id, entity->cardId, entity->team, dealt, board.currentTick, entity->isTower() });

            if (rollKnockback > 0.0f) {
                // The lateral throw: `lateral` in [-1, 1] is where across the
                // corridor the unit was caught. The centre is shoved along the
                // roll, the edges flung sideways, the rest blended. A radial
                // pushAway cannot do this (see pushAlong in Entity.h).
                float lateral = (halfWidth > 0.0f) ? (dx / halfWidth) : 0.0f;
                if (lateral > 1.0f) lateral = 1.0f;
                if (lateral < -1.0f) lateral = -1.0f;
                const float forward = 1.0f - std::fabs(lateral);
                pushAlong(*entity, lateral, forward * static_cast<float>(rollDirY), rollKnockback);
            }

            if (onHit) {
                auto combatTarget = std::dynamic_pointer_cast<CombatEntity>(entity);
                if (combatTarget) onHit->apply(combatTarget);
            }
        }

        if (rollTravelled >= rollRange) {
            // Barbarian Barrel drops its Barbarian where the barrel stops.
            if (spawnOnDetonate) spawnOnDetonate->apply(board, position, team);
            hp = 0;
        }
    }
};