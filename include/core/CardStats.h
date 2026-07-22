#pragma once
#include "Entity.h"
#include "OnHitEffect.h"
#include "OnDamageTakenEffect.h"
#include "DeathEffect.h"
#include "PeriodicEffect.h"
#include "AbilityEffect.h"
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

    // Rage: buffs allies in radius instead of damaging enemies -- see
    // AreaSpell's buffsAllies mode. false (the default) is every other
    // spell here.
    bool spellBuffsAllies = false;
    float spellBuffMultiplier = 1.0f;
    int spellBuffDurationTicks = 0;

    // Knockback/pull (Fireball, Rocket, Giant Snowball push; Tornado
    // pulls) -- see AreaSpell's knockback field. Positive pushes away,
    // negative pulls toward center. 0.0f (the default) is every other
    // spell here.
    float spellKnockback = 0.0f;

    // Spell-spawns-troops (Goblin Barrel, Royal Delivery, Graveyard) --
    // see AreaSpell::spawnOnDetonate. nullptr (the default) is every
    // spell that doesn't spawn anything.
    std::shared_ptr<IPeriodicEffect> spellSpawnEffect;

    // Clone -- see AreaSpell::clonesAllies. false (the default) is every
    // other spell here.
    bool spellClonesAllies = false;

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
    // Electro Dragon's chain -- see CombatEntity::splitTargetsFullDamage.
    bool splitTargetsFullDamage = false;

    // Boomerang projectiles (Executioner) -- RangedSquad archetype only. See
    // Projectile's returnsToSender. false (the default) is a normal
    // single-hit projectile.
    bool boomerang = false;
    int boomerangReturnDelayTicks = 0;

    // Kamikaze (Wall Breakers, the "Spirit" troops) -- see
    // CombatEntity::dieAfterFirstHit. false (the default) is every other
    // card here.
    bool dieAfterFirstHit = false;

    // Deploy anywhere on the board (Miner, Goblin Drill) -- see
    // GameManager::isValidPlacement. false (the default) is every other
    // troop/building, which must stay on this player's own half.
    bool deployAnywhere = false;

    // Splash damage on every regular attack (Wizard, Bowler, Valkyrie, ...)
    // -- see CombatEntity::applySplashDamage. 0.0f (the default) is every
    // card that doesn't opt in. Unlike hp/damage/cost, splash radius isn't
    // part of the sourced stats data (it's an engine-internal geometry
    // choice, same category as movement speed above) -- splash-flagged
    // cards use a single reasonable constant rather than per-card figures
    // that were never actually sourced.
    float splashRadius = 0.0f;

    // Shield HP (Guards, Royal Recruits, Dark Prince, Cannon Cart) -- see
    // CombatEntity::takeDamage. 0 (the default) is every card without one.
    // Same "not part of the sourced data" caveat as splashRadius above.
    int shieldHp = 0;

    // Charge/dash bonus damage (Prince, Battle Ram, Ram Rider, Royal Hogs,
    // Bandit) -- see CombatEntity::chargeThreshold/chargeMultiplier.
    // 0.0f threshold (the default) is every card without a charge. Same
    // "not part of the sourced data" caveat as splashRadius above.
    float chargeThreshold = 0.0f;
    float chargeMultiplier = 1.0f;

    // Enrage (Berserker) -- see CombatEntity::enrageMaxHp/enrageHealPerHit.
    // 0 (the default) is every card without it. Set via withEnrage below,
    // which reads back the `hp` field already set by troop(...).
    int enrageMaxHp = 0;
    int enrageHealPerHit = 0;

    // Parry (Ronin) -- see CombatEntity::parryIntervalTicks. 0 (the
    // default) is every card without one.
    int parryIntervalTicks = 0;

    // Hook (Fisherman) -- see CombatEntity::hookRange. 0.0f (the default)
    // is every card without one.
    float hookRange = 0.0f;

    // Invisibility (Royal Ghost, Suspicious Bush) -- see
    // CombatEntity::startsInvisible/isTargetable. Also makes the card
    // immune to splash/spell area damage while invisible, not just
    // individual targeting (applySplashDamage/AreaSpell both gate on
    // isTargetable() too) -- broader than the real game's "can't be
    // individually selected, but area effects still land" rule. false
    // (the default) is every card without it.
    bool startsInvisible = false;
    int revealTicksAfterAttack = 0;

    // Compound cards (Goblin Giant's carried Spear Goblins, Ram Rider's
    // independent crossbow, Goblin Machine's rocket turret, Goblin Gang,
    // Rascals): a second, independently-targeting unit spawned alongside
    // the primary one, at the same deploy point -- see
    // CardFactories::spawn, which spawns this recursively right after the
    // primary archetype. nullptr (the default) is every single-unit card.
    std::shared_ptr<CardStats> secondaryUnit;

    // Periodic spawning while alive (Witch, Night Witch, Furnace,
    // Barbarian Hut, Goblin Hut, Tombstone, Goblin Drill) -- see
    // CombatEntity::periodicEffect/periodicIntervalTicks. Fires the first
    // time after one full interval has passed, not immediately at deploy.
    // periodicIntervalTicks == 0 (the default) is every card without one.
    std::shared_ptr<IPeriodicEffect> periodicEffect;
    int periodicIntervalTicks = 0;

    // Ally aura on landed attacks (Rune Giant's every-Nth-attack buff,
    // Battle Healer's heal) -- see CombatEntity's own aura* fields.
    // auraEveryNAttacks == 0 disables the buff aura; healAllyAmount == 0
    // disables the heal aura -- independent opt-ins, both default off.
    float auraRadius = 0.0f;
    int auraMaxTargets = 1000000;
    int auraEveryNAttacks = 0;
    float auraBuffMultiplier = 1.0f;
    int auraBuffDurationTicks = 0;
    int healAllyAmount = 0;

    // Dash invulnerability (Bandit) -- see CombatEntity::chargeGrantsInvulnerability.
    bool chargeGrantsInvulnerability = false;
    // Stun fully resets (not just slows) the attack cooldown (Sparky) --
    // see CombatEntity::resetCooldownOnFreeze.
    bool resetCooldownOnFreeze = false;
    // Recoil after attacking (Firecracker) -- see CombatEntity::recoilDistance.
    float recoilDistance = 0.0f;
    // Minimum attack range / blind spot (Mortar) -- see CombatEntity::minAttackRange.
    float minAttackRange = 0.0f;
    // Deploy delay before the first attack is ready (X-Bow) -- see
    // CombatEntity::seedCooldown. 0 (the default) is every other card,
    // ready to fire as soon as a target's in range.
    int initialCooldownTicks = 0;
    // HP-threshold transform (Cannon Cart) -- see
    // CombatEntity::transformAtHpFraction/transformCheckMaxHp.
    float transformAtHpFraction = 0.0f;
    int transformLifetimeTicks = 0;
    bool transformBecomesStationary = false;
    // Archetype-swap transform (Goblin Demolisher) -- see
    // CombatEntity::transformKillsSelf/transformDeathEffect. Deliberately
    // separate from the ordinary `deathEffect` above -- see that field's
    // own comment for why.
    bool transformKillsSelf = false;
    std::shared_ptr<IDeathEffect> transformDeathEffect;

    // Periodic jump (Mega Knight) -- see CombatEntity::jumpMinRange/jumpMaxRange.
    float jumpMinRange = 0.0f;
    float jumpMaxRange = 0.0f;
    float jumpDamageMultiplier = 1.0f;
    float jumpSplashRadius = 0.0f;

    // Piercing-line hit (Bowler, Magic Archer) -- see CombatEntity::lineSplash.
    bool lineSplash = false;
    float lineSplashRange = 0.0f;

    // Range-based damage falloff (Hunter) -- NOT sourced data, see
    // CombatEntity::rangeFalloff's own comment for why.
    bool rangeFalloff = false;
    float rangeFalloffMinFraction = 1.0f;

    // AreaSpell-only: Vines' top-N-highest-HP targeting -- see
    // AreaSpell::targetTopHpCount. 0 (the default) is every other spell.
    int spellTargetTopHpCount = 0;
    // AreaSpell-only: Void's 3-tier target-count-based damage -- see
    // AreaSpell::tieredDamage. false (the default) is every other spell.
    bool spellTieredDamage = false;
    int spellTierSingleDamage = 0;
    int spellTierFewDamage = 0;
    int spellTierManyDamage = 0;

    // Champion marker + activated ability (Mighty Miner's "Explosive
    // Escape") -- see CombatEntity::isChampion/abilityElixirCost/
    // abilityCooldownTicks/abilityEffect. isChampion alone changes no
    // targeting/combat behavior -- a Champion stays whatever ordinary
    // Archetype it already is (Mighty Miner is plain MeleeSquad, same as
    // the regular Miner); this is purely a marker so
    // GameManager::activateChampionAbility can find "my deployed Champion"
    // on the board. false/0/nullptr (the defaults) are every non-Champion
    // card.
    bool isChampion = false;
    float abilityElixirCost = 0.0f;
    int abilityCooldownTicks = 0;
    std::shared_ptr<IAbilityEffect> abilityEffect;
    // -1 (the default) is unlimited activations -- see
    // CombatEntity::abilityUsesRemaining. Only Boss Bandit sets this.
    int abilityUsesLimit = -1;

    // Soul collection (Skeleton King) -- see CombatEntity::
    // soulCollectionRadius/maxSouls. 0.0f/0 (the defaults) are every
    // non-soul-collecting card.
    float soulCollectionRadius = 0.0f;
    int maxSouls = 0;

    // Hit-speed ramp (Little Prince) -- see CombatEntity::
    // hitSpeedRampMidTick's own comment for how this differs from the
    // damage ramp above.
    int hitSpeedRampMidTick = 0;
    int hitSpeedRampFullTick = 0;
    float hitSpeedRampMidFraction = 1.0f;
    float hitSpeedRampFullFraction = 1.0f;

    // Burst-on-Nth-attack (Dagger Duchess, some Evolutions) -- see
    // CombatEntity::burstEveryNAttacks's own comment. 0 (the default)
    // disables it.
    int burstEveryNAttacks = 0;
    float burstDamageMultiplier = 1.0f;

    // Self-buff (or other self effect) on taking damage (some Evolutions,
    // e.g. Barbarians) -- see CombatEntity::onDamageTakenEffect. nullptr
    // (the default) is every card without one.
    std::shared_ptr<IOnDamageTakenEffect> onDamageTaken;

    // Self-heal on landing a hit (Evolved Bats) -- see
    // CombatEntity::healOnHitAmount. 0 (the default) disables it.
    int healOnHitAmount = 0;
    int healOnHitMaxHp = 0;

    // Self-spawn on landing a hit (Evolved Skeletons) -- see
    // CombatEntity::onHitSpawnEffect. nullptr (the default) is every
    // card without one.
    std::shared_ptr<IPeriodicEffect> onHitSpawnEffect;

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
    CardStats& withDeployAnywhere(bool value = true) {
        deployAnywhere = value;
        return *this;
    }
    CardStats& withDieAfterFirstHit(bool value = true) {
        dieAfterFirstHit = value;
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
    CardStats& withSplitTargetsFullDamage() {
        splitTargetsFullDamage = true;
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
    CardStats& withShield(int amount) {
        shieldHp = amount;
        return *this;
    }
    CardStats& withCharge(float threshold, float multiplier) {
        chargeThreshold = threshold;
        chargeMultiplier = multiplier;
        return *this;
    }
    // Reads back `hp` (already set by troop(...) before this chains on).
    CardStats& withEnrage(int healPerHit) {
        enrageMaxHp = hp;
        enrageHealPerHit = healPerHit;
        return *this;
    }
    CardStats& withParry(int intervalTicks) {
        parryIntervalTicks = intervalTicks;
        return *this;
    }
    CardStats& withHook(float range) {
        hookRange = range;
        return *this;
    }
    CardStats& withInvisibility(int revealTicks) {
        startsInvisible = true;
        revealTicksAfterAttack = revealTicks;
        return *this;
    }
    CardStats& withPeriodicEffect(int intervalTicks, std::shared_ptr<IPeriodicEffect> effect) {
        periodicIntervalTicks = intervalTicks;
        periodicEffect = std::move(effect);
        return *this;
    }
    CardStats& withAllyBuffAura(float radius, int everyNAttacks, float multiplier, int durationTicks, int maxTargets) {
        auraRadius = radius;
        auraEveryNAttacks = everyNAttacks;
        auraBuffMultiplier = multiplier;
        auraBuffDurationTicks = durationTicks;
        auraMaxTargets = maxTargets;
        return *this;
    }
    CardStats& withHealAura(float radius, int amount) {
        auraRadius = radius; // shared with the buff aura's radius; a card only ever uses one of the two auras
        healAllyAmount = amount;
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
    CardStats& withSpellBuff(float multiplier, int durationTicks) {
        spellBuffsAllies = true;
        spellBuffMultiplier = multiplier;
        spellBuffDurationTicks = durationTicks;
        return *this;
    }
    CardStats& withKnockback(float distance) {
        spellKnockback = distance;
        return *this;
    }
    CardStats& withSpellSpawn(std::shared_ptr<IPeriodicEffect> effect) {
        spellSpawnEffect = std::move(effect);
        return *this;
    }
    CardStats& withSecondaryUnit(CardStats unit) {
        secondaryUnit = std::make_shared<CardStats>(std::move(unit));
        return *this;
    }
    CardStats& withSpellClone() {
        spellClonesAllies = true;
        return *this;
    }
    CardStats& withChargeInvulnerability() {
        chargeGrantsInvulnerability = true;
        return *this;
    }
    CardStats& withStunResetsCooldown() {
        resetCooldownOnFreeze = true;
        return *this;
    }
    CardStats& withRecoil(float distance) {
        recoilDistance = distance;
        return *this;
    }
    CardStats& withMinRange(float range) {
        minAttackRange = range;
        return *this;
    }
    CardStats& withDeployDelay(int ticks) {
        initialCooldownTicks = ticks;
        return *this;
    }
    // Reads back `hp` (already set by troop(...)/building(...) before this
    // chains on), same idiom as withEnrage.
    CardStats& withHpTransform(float atFraction, int lifetimeTicks, bool becomesStationary) {
        transformAtHpFraction = atFraction;
        transformLifetimeTicks = lifetimeTicks;
        transformBecomesStationary = becomesStationary;
        return *this;
    }
    CardStats& withSpellTopHpTargets(int count) {
        spellTargetTopHpCount = count;
        return *this;
    }
    CardStats& withSpellTieredDamage(int single, int few, int many) {
        spellTieredDamage = true;
        spellTierSingleDamage = single;
        spellTierFewDamage = few;
        spellTierManyDamage = many;
        return *this;
    }
    CardStats& withHpTransformIntoDeath(float atFraction, std::shared_ptr<IDeathEffect> effect) {
        transformAtHpFraction = atFraction;
        transformKillsSelf = true;
        transformDeathEffect = std::move(effect);
        return *this;
    }
    CardStats& withJump(float minRange, float maxRange, float damageMultiplier, float splashRadiusOnLand) {
        jumpMinRange = minRange;
        jumpMaxRange = maxRange;
        jumpDamageMultiplier = damageMultiplier;
        jumpSplashRadius = splashRadiusOnLand;
        return *this;
    }
    // splashRadius (see withSplash) doubles as the line's half-width.
    CardStats& withLineSplash(float range) {
        lineSplash = true;
        lineSplashRange = range;
        return *this;
    }
    // minFraction is an invented engine constant, not sourced data -- see
    // CombatEntity::rangeFalloff's comment.
    CardStats& withRangeFalloff(float minFraction) {
        rangeFalloff = true;
        rangeFalloffMinFraction = minFraction;
        return *this;
    }
    CardStats& withChampionAbility(float elixirCost, int cooldownTicks, std::shared_ptr<IAbilityEffect> effect,
            int usesLimit = -1) {
        isChampion = true;
        abilityElixirCost = elixirCost;
        abilityCooldownTicks = cooldownTicks;
        abilityEffect = std::move(effect);
        abilityUsesLimit = usesLimit;
        return *this;
    }
    CardStats& withSoulCollection(float radius, int maxSoulCount) {
        soulCollectionRadius = radius;
        maxSouls = maxSoulCount;
        return *this;
    }
    CardStats& withHitSpeedRamp(int midTick, int fullTick, float midFraction, float fullFraction) {
        hitSpeedRampMidTick = midTick;
        hitSpeedRampFullTick = fullTick;
        hitSpeedRampMidFraction = midFraction;
        hitSpeedRampFullFraction = fullFraction;
        return *this;
    }
    CardStats& withBurstAttack(int everyNAttacks, float multiplier) {
        burstEveryNAttacks = everyNAttacks;
        burstDamageMultiplier = multiplier;
        return *this;
    }
    CardStats& withOnDamageTaken(std::shared_ptr<IOnDamageTakenEffect> effect) {
        onDamageTaken = std::move(effect);
        return *this;
    }
    CardStats& withHealOnHit(int amount, int maxHp) {
        healOnHitAmount = amount;
        healOnHitMaxHp = maxHp;
        return *this;
    }
    CardStats& withOnHitSpawn(std::shared_ptr<IPeriodicEffect> effect) {
        onHitSpawnEffect = std::move(effect);
        return *this;
    }
    // Permanent damage-taken reduction from spawn (Evolved Knight's
    // shield) -- approximates a real mechanic that's conditional on
    // movement state (shield only while not actively attacking) as a
    // flat, permanent, smaller reduction instead, since this engine has
    // no "is this entity currently mid-attack vs. approaching" signal
    // exposed at the CardStats level. multiplier < 1.0 reduces damage
    // taken; 1.0 (the default) is every card without one. Applied via
    // CombatEntity::applyCurse at spawn with an effectively-infinite
    // duration -- reuses the existing curse machinery, just seeded once
    // instead of by a timed ability/spell.
    float passiveDamageReduction = 1.0f;
    CardStats& withPassiveDamageReduction(float multiplier) {
        passiveDamageReduction = multiplier;
        return *this;
    }
};
