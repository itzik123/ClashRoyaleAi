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
    // Most spells hit Air & Ground alike (Fireball, Zap, Poison, Rocket,
    // Lightning...); a handful (The Log, Barbarian Barrel) are ground-only
    // control spells that roll along the arena floor. Defaults to false
    // (hits everyone) to match the majority, with ground-only spells opting
    // in via CardStats::withGroundOnly.
    bool groundOnly;

    // Multi-tick spells (Poison's 8 ticks over 8s, Arrows' 3 rapid volleys):
    // remainingHits counts down one application at a time, re-evaluating who
    // is currently in radius on *each* application -- exactly like the real
    // game, a unit can walk out of a Poison cloud partway through and stop
    // taking damage. remainingHits == 1 (the default) is every other
    // spell's normal single-shot case; tickInterval is only relevant when
    // remainingHits > 1, and doubles delayTicks as the gap between hits.
    int remainingHits;
    int tickInterval;

    // Rage: instead of damaging enemies, buffs allies in radius. Flips the
    // team filter (== instead of !=) and applies a damage buff instead of
    // takeDamage()/onHit -- `damage` goes unused in this mode (should be 0
    // at the call site). false (the default) is every other spell here.
    bool buffsAllies;
    float buffMultiplier;
    int buffDurationTicks;

    // Knockback/pull (Fireball, Rocket, Giant Snowball push away; Tornado
    // pulls toward center): applied to everyone hit, alongside damage.
    // Positive pushes away from the spell's position, negative pulls
    // toward it -- see Entity.h's pushAway/pullToward. 0.0f (the default)
    // is every other spell here.
    //
    // Buildings (Building and its Tower subclass) are never physically
    // moved by this, though they still take damage same as any other
    // target -- confirmed real-game rule: Tornado has dealt damage to
    // buildings since a May 2020 balance update, but its pull has never
    // displaced them. Enforced inside pushAway/pullToward themselves
    // (Entity.h), not with a check here -- see their own comment.
    float knockback;

    // Spell-spawns-troops (Goblin Barrel, Royal Delivery, Graveyard):
    // fires once per application (same cadence as remainingHits/
    // tickInterval above -- a single application for the first two,
    // repeated for Graveyard's rain-of-skeletons), spawning at this
    // spell's own position/team. Reuses IPeriodicEffect/
    // PeriodicSpawnEffect from core/ -- same interface as
    // CombatEntity's own periodic spawns, just fired from a spell
    // instead of a tick timer. nullptr (the default) is every spell
    // that doesn't spawn anything.
    std::shared_ptr<IPeriodicEffect> spawnOnDetonate;

    // Clone: duplicates every ally troop in radius instead of damaging
    // enemies (same ally-team filter as buffsAllies). Each clone gets a
    // fresh board id and 1 hp (see Entity::clone) but keeps every other
    // configured field of the original. false (the default) is every
    // other spell here.
    bool clonesAllies;

    // Vines: real card only roots the highest-HP troops/buildings in
    // radius, not everyone caught in it. 0 (the default) means "everyone
    // in radius", every other spell here.
    int targetTopHpCount;

    // Void: real damage scales inversely with how many targets are caught
    // -- fewer targets means more damage each. Modeled as 3 discrete tiers
    // (1 / 2-4 / 5+ targets), matching Void's own sourced data rather than
    // a continuous formula. `damage` above goes unused in this mode.
    // tieredDamage == false (the default) is every other spell here.
    bool tieredDamage;
    int tierSingleDamage;
    int tierFewDamage;
    int tierManyDamage;

    // Hero Ice Golem's Snowstorm: damage to Crown Towers reduced to 5% of
    // the normal amount (this engine has no existing "reduced damage vs
    // buildings" spell mechanism -- confirmed by reading Earthquake's own
    // comment, which explicitly documents its real "3.5x vs buildings" as
    // NOT modeled here, so this is a genuinely new primitive, not a reuse).
    // 1.0f (the default) is every other spell, unaffected.
    float spellTowerDamageMultiplier;

public:
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
        spellTowerDamageMultiplier(spellTowerDamageMultiplier) {}

    bool isTargetable() const override { return false; }

    // Board::deepCopy. delayTicks and remainingHits ride along with the
    // implicit copy, so a spell mid-fuse or mid-volley resumes where the
    // original is rather than re-arming -- which is exactly the state a
    // rollout needs to reason about ("does my Fireball land before that
    // Musketeer walks out of it").
    std::shared_ptr<Entity> snapshot() const override {
        return std::make_shared<AreaSpell>(*this);
    }

    // --- Rolling sweep (The Log, Barbarian Barrel), 2026-08-28 -------------
    //
    // These two are not static circles. They are dynamic bodies that roll
    // forward from where they land, sweeping a RECTANGULAR corridor: a fixed
    // width across the roll axis, and a length they travel over time. Modelled
    // here rather than as a new Entity subclass because everything else about
    // them -- team filter, ground-only, damage, on-hit, the Barrel's spawn --
    // is already exactly AreaSpell's behaviour; only the shape and the motion
    // differ.
    //
    // UNITS. `rollSpeed` is tiles per TICK, stated directly and deliberately
    // NOT routed through CardStats::MOVEMENT_SPEED_SCALE. That scale exists to
    // convert the registry's troop SPEED_* tier literals into real-game
    // tiles/tick; a spell has no tier and borrowing the conversion would make
    // these two numbers mean something different from every other number in
    // this file. See CLAUDE.md's movement-speed section for why that confusion
    // is worth one comment.
    //
    // Each target is damaged AT MOST ONCE per roll (`sweptIds`), which is the
    // real-game behaviour -- the log rolls over you, it does not grind.
    float rollRange = 0.0f;   // total distance travelled; 0 => not a roller
    float rollWidth = 0.0f;   // FULL width across the roll axis, not a radius
    float rollSpeed = 0.0f;   // tiles per tick
    float rollKnockback = 0.0f;
    float rollTravelled = 0.0f;
    int rollDirY = 0;         // +1 for team 0 (attacks toward +y), -1 for team 1
    Vector2D rollOrigin{0.0f, 0.0f};
    std::vector<int> sweptIds;

    bool isRolling() const { return rollRange > 0.0f; }

    // Set post-construction rather than through the constructor: that
    // parameter list is already 20 wide and has five other call sites
    // (spawnDeployEffect, Goblinstein's link, Hero Ice Golem's snowstorm,
    // Mighty Miner's escape, and snapshot()), none of which roll.
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
        // Cloning while iterating board.getEntities() would add to the
        // same vector mid-loop -- collect targets first, clone after.
        std::vector<std::shared_ptr<Entity>> toClone;

        // Collected first (rather than acted on inline) because both
        // targetTopHpCount (Vines: only the highest-HP few) and
        // tieredDamage (Void: per-target damage depends on the total
        // catch) need the full candidate set up front, before anything
        // is actually hit.
        std::vector<std::shared_ptr<Entity>> candidates;
        for (const auto& entity : board.getEntities()) {
            bool teamMatches = alliesOnly ? (entity->team == this->team) : (entity->team != this->team);
            if (entity->isAlive() && entity->isTargetable() && teamMatches && entity->id != this->id
                && (!groundOnly || !entity->isFlying) && position.distanceTo(entity->position) <= radius) {
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
            // Hero Ice Golem's Snowstorm: Crown Towers take only 5% of the
            // normal amount -- entity->isTower() (Entity.h) instead of a
            // cast, matching how exemptFromForcedMovement already queries
            // building-ness cheaply elsewhere in this engine.
            int dealt = entity->isTower()
                ? static_cast<int>(effectiveDamage * spellTowerDamageMultiplier)
                : effectiveDamage;
            entity->takeDamage(dealt);
            if (knockback != 0.0f) {
                if (knockback > 0.0f) {
                    pushAway(*entity, position, knockback);
                } else {
                    pullToward(*entity, position, -knockback);
                }
            }
            // The spell entity itself is the "attacker" -- it never
            // has a separate caster once cast (the troop/tower that
            // played the card is already gone by the time this
            // fires, for spawn-effect zaps like Electro Wizard's).
            board.statsEvents.notifyDamageDealt(
                { id, team, cardId, entity->id, entity->cardId, entity->team, dealt, board.currentTick, entity->isTower() });
            // Same surgical cast as CombatEntity::applyOnHitEffects and
            // Projectile's arrival handler -- on-hit effects only ever
            // mean something against a CombatEntity, so this is the one
            // place AreaSpell needs to know that, not its own type.
            if (onHit) {
                auto combatTarget = std::dynamic_pointer_cast<CombatEntity>(entity);
                if (combatTarget) onHit->apply(combatTarget);
            }
        }

        for (const auto& original : toClone) {
            auto copy = original->clone(board.allocateId());
            if (copy) board.addEntity(copy);
        }

        if (spawnOnDetonate) spawnOnDetonate->apply(board, position, team);

        remainingHits--;
        if (remainingHits > 0) {
            delayTicks = tickInterval; // wait out the gap, then apply again
        } else {
            hp = 0;
        }
    }

private:
    bool alreadySwept(int entityId) const {
        return std::find(sweptIds.begin(), sweptIds.end(), entityId) != sweptIds.end();
    }

    // One tick of a rolling sweep. Advances the body, damages whatever the
    // corridor has newly reached, and throws each victim along a direction
    // that depends on WHERE ACROSS the corridor it was caught.
    void updateRoll(Board& board) {
        rollTravelled += rollSpeed;
        if (rollTravelled > rollRange) rollTravelled = rollRange;
        // The entity's own position tracks the leading edge. That is what the
        // replay records per tick and therefore what the viewer draws the
        // rectangle from -- no extra per-tick field needed in the log.
        position.y = rollOrigin.y + static_cast<float>(rollDirY) * rollTravelled;

        const float halfWidth = rollWidth * 0.5f;

        for (const auto& entity : board.getEntities()) {
            if (!entity->isAlive() || !entity->isTargetable()) continue;
            if (entity->team == this->team || entity->id == this->id) continue;
            if (groundOnly && entity->isFlying) continue;
            if (alreadySwept(entity->id)) continue;

            const float r = CombatEntity::effectiveRadiusOf(*entity);
            const float dx = entity->position.x - rollOrigin.x;
            // Longitudinal offset measured ALONG the roll direction, so one
            // set of comparisons serves both teams.
            const float dy = (entity->position.y - rollOrigin.y) * static_cast<float>(rollDirY);

            if (std::fabs(dx) > halfWidth + r) continue;  // outside the corridor
            if (dy + r < 0.0f) continue;                  // entirely behind the spawn point
            // Reached when the leading edge touches the target's near SURFACE,
            // not its centre -- the same convention effectiveRangeTo uses, and
            // the one the bridge interaction depends on: an enemy Princess
            // Tower's centre is 10.50 tiles from BRIDGE_Y but its near edge only
            // 9.00, so The Log's 10.1 roll reaches it from the bridge with 1.1
            // to spare while never reaching the centre. Pinned in
            // tests/entities/test_area_spell.cpp.
            if (dy - r > rollTravelled) continue;

            sweptIds.push_back(entity->id);

            const int dealt = entity->isTower()
                ? static_cast<int>(damage * spellTowerDamageMultiplier)
                : damage;
            entity->takeDamage(dealt);
            board.statsEvents.notifyDamageDealt(
                { id, team, cardId, entity->id, entity->cardId, entity->team, dealt, board.currentTick, entity->isTower() });

            if (rollKnockback > 0.0f) {
                // THE LATERAL THROW. `lateral` is where across the corridor
                // this unit was caught, in [-1, +1]; `forward` is what is left
                // over. Dead centre is shoved straight along the roll, the very
                // edge is flung purely sideways, and everything between blends.
                // Separating a grouped push is exactly this, and a radial
                // pushAway from the log's centre cannot produce it -- see
                // pushAlong's comment in Entity.h.
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
            // Barbarian Barrel drops its Barbarian where the barrel STOPS, not
            // where it was thrown -- so this fires at the final position.
            if (spawnOnDetonate) spawnOnDetonate->apply(board, position, team);
            hp = 0;
        }
    }
};