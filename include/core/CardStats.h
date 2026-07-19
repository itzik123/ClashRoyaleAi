#pragma once
#include "Entity.h"
#include "OnHitEffect.h"
#include "DeathEffect.h"
#include <memory>
#include <string>
#include <vector>

// The handful of ways a card actually gets built onto the board. Adding a
// new card of an existing shape is a data row (see CardRegistry.h); adding a
// genuinely new mechanic is a new archetype + one new factory function in
// CardFactories.h.
enum class Archetype {
    MeleeSquad,             // one or more direct-damage troops (Knight, Goblins, Barbarians...)
    RangedSquad,             // one or more projectile troops (Musketeer, Archers, Spear Goblins...)
    MeleeBuildingTargeter,   // ignores troops, melee-hits buildings (Giant, Hog Rider, Golem)
    RangedBuildingTargeter,  // ignores troops, projectile-hits buildings (Royal Giant)
    DefensiveBuilding,       // stationary, direct damage (Cannon, Tesla, ...)
    Spell                    // one-shot area effect (Fireball, Zap, ...)
};

struct CardStats {
    int id = 0;
    std::string name;
    float cost = 0.0f;
    Archetype archetype = Archetype::MeleeSquad;

    // Combat stats, used by every archetype except Spell.
    int hp = 0;
    float speed = 0.0f;
    float attackRange = 0.0f;
    int damage = 0;
    int attackCooldown = 0;
    char symbol = '?';
    bool ignoresRiver = false;
    bool isFlying = false;
    bool targetsAir = false;

    // Relative spawn positions for each unit in the card. A single-unit
    // card is just one offset of {0, 0} -- "squad of one".
    std::vector<Vector2D> spawnOffsets{ Vector2D{ 0.0f, 0.0f } };

    // Optional extra behavior applied on every successful attack landed by
    // every unit this card spawns (e.g. Ice Wizard/Ice Golem's freeze).
    std::shared_ptr<IOnHitEffect> onHit;

    // Spell archetype only. groundOnly defaults false (hits Air & Ground,
    // matching most spells -- Fireball, Zap, Poison, Rocket, Lightning);
    // The Log/Barbarian Barrel-style ground-control spells opt in.
    float spellRadius = 0.0f;
    int spellDelayTicks = 0;
    bool spellGroundOnly = false;
    // Multi-tick spells (Poison, Arrows) -- see AreaSpell::remainingHits.
    // 1 (the default) is every other spell's normal single-shot case.
    int spellRemainingHits = 1;
    int spellTickInterval = 0;
    // Optional on-hit effect the spell itself applies to whatever it
    // touches, independent of its (possibly zero) direct damage -- e.g.
    // Freeze's full stun with no damage component at all. nullptr (the
    // default) is every other spell's normal damage-only case.
    std::shared_ptr<IOnHitEffect> spellOnHit;

    // One-time area burst applied the instant a troop-shaped card deploys,
    // independent of its regular attacks (e.g. Electro Wizard's spawn zap).
    // Radius 0 (the default) means no spawn effect.
    float spawnEffectRadius = 0.0f;
    int spawnEffectDamage = 0;
    std::shared_ptr<IOnHitEffect> spawnEffectOnHit;

    // Fired once when a unit this card spawns dies (e.g. Golem's two
    // Golemites). Concrete effects live in core/ (e.g. SpawnOnDeath) since
    // they need CardFactories to know how to spawn anything.
    std::shared_ptr<IDeathEffect> deathEffect;

    // Ramping damage (Inferno Tower) -- see CombatEntity::getCurrentDamage
    // for the exact fraction schedule. rampFullTick == 0 (the default)
    // means no ramping.
    int rampMidTick = 0;
    int rampFullTick = 0;
    float rampStartFraction = 1.0f;
    float rampMidFraction = 1.0f;

    // Split-target attacks (Electro Wizard) -- see
    // CombatEntity::findSplitTargets/getCurrentDamage. 1 (the default)
    // means the normal single-target case.
    int maxSplitTargets = 1;

    // Boomerang projectiles (Executioner) -- RangedSquad archetype only. See
    // Projectile's returnsToSender. false (the default) is a normal
    // single-hit projectile.
    bool boomerang = false;
    int boomerangReturnDelayTicks = 0;

    // Splash damage on every regular attack (Wizard, Bowler, Valkyrie, ...)
    // -- see CombatEntity::applySplashDamage. 0.0f (the default) is every
    // card that doesn't opt in. Unlike hp/damage/cost, splash radius isn't
    // part of the sourced stats data (it's an engine-internal geometry
    // choice, same category as movement speed above) -- splash-flagged
    // cards use a single reasonable constant rather than per-card figures
    // that were never actually sourced.
    float splashRadius = 0.0f;

    // Small fluent setters so CardRegistry's data table can stay one card
    // per line/two, instead of spelling out every field for every card.
    CardStats& withOffsets(std::vector<Vector2D> offsets) {
        spawnOffsets = std::move(offsets);
        return *this;
    }
    CardStats& withIgnoresRiver(bool value = true) {
        ignoresRiver = value;
        return *this;
    }
    CardStats& withFlying(bool value = true) {
        isFlying = value;
        return *this;
    }
    CardStats& withTargetsAir(bool value = true) {
        targetsAir = value;
        return *this;
    }
    CardStats& withOnHit(std::shared_ptr<IOnHitEffect> effect) {
        onHit = std::move(effect);
        return *this;
    }
    CardStats& withSpawnEffect(float radius, int damage, std::shared_ptr<IOnHitEffect> effect = nullptr) {
        spawnEffectRadius = radius;
        spawnEffectDamage = damage;
        spawnEffectOnHit = std::move(effect);
        return *this;
    }
    CardStats& withDeathEffect(std::shared_ptr<IDeathEffect> effect) {
        deathEffect = std::move(effect);
        return *this;
    }
    CardStats& withGroundOnly(bool value = true) {
        spellGroundOnly = value;
        return *this;
    }
    CardStats& withDamageRamp(int midTick, int fullTick, float startFraction, float midFraction) {
        rampMidTick = midTick;
        rampFullTick = fullTick;
        rampStartFraction = startFraction;
        rampMidFraction = midFraction;
        return *this;
    }
    CardStats& withSplitTargets(int maxTargets) {
        maxSplitTargets = maxTargets;
        return *this;
    }
    CardStats& withBoomerang(int returnDelayTicks) {
        boomerang = true;
        boomerangReturnDelayTicks = returnDelayTicks;
        return *this;
    }
    CardStats& withSplash(float radius) {
        splashRadius = radius;
        return *this;
    }
    CardStats& withRepeats(int count, int intervalTicks) {
        spellRemainingHits = count;
        spellTickInterval = intervalTicks;
        return *this;
    }
    CardStats& withSpellOnHit(std::shared_ptr<IOnHitEffect> effect) {
        spellOnHit = std::move(effect);
        return *this;
    }
};
