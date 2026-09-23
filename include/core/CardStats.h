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

// The shapes a card can be built as. A new card of an existing shape is a data
// row in CardRegistry.h; a new mechanic is a new archetype plus a factory in
// CardFactories.h.
enum class Archetype {
    MeleeSquad,             // direct-damage troops (Knight, Goblins, Barbarians...)
    RangedSquad,             // projectile troops (Musketeer, Archers, Spear Goblins...)
    MeleeBuildingTargeter,   // ignores troops, melee-hits buildings (Giant, Hog Rider, Golem)
    RangedBuildingTargeter,  // ignores troops, projectile-hits buildings (Royal Giant)
    DefensiveBuilding,       // stationary, direct damage (Cannon, Tesla, ...)
    Spell                    // area effect (Fireball, Zap, ...)
};

// Converts the registry's speed values (the engine's original units) into
// real-game tiles/tick. Applied once, in CardRegistry's troop() and
// SpiritEmpressForms.h, at construction rather than in Troop::update, so
// Troop::getSpeed() (the observation's CH_SPEED) reports the speed actually
// moved at. Derived from real recordings (perception/UPSTREAM_REQUESTS.md item
// 9).
inline constexpr float MOVEMENT_SPEED_SCALE = 0.2f;

// Speed tiers. The real game publishes one speed per card, in tiles per minute,
// taking only five values: 30 / 45 / 60 / 90 / 120 (Very Slow .. Very Fast). A
// real tile is not an engine tile, so the conversion is measured, from two
// cards tracked frame by frame (Giant, real 45 -> 0.987 engine tiles/s; Mini
// P.E.K.K.A, real 90 -> 2.003), which agree to 1.5%.
//
// Named tiers, because cards sharing a real tier must share an engine speed,
// which a global scale cannot enforce.
inline constexpr float REAL_TILES_PER_MIN_TO_ENGINE = 0.011045f;
inline constexpr float SPEED_VERY_SLOW = 30.0f  * REAL_TILES_PER_MIN_TO_ENGINE;
inline constexpr float SPEED_SLOW      = 45.0f  * REAL_TILES_PER_MIN_TO_ENGINE;
inline constexpr float SPEED_MEDIUM    = 60.0f  * REAL_TILES_PER_MIN_TO_ENGINE;
inline constexpr float SPEED_FAST      = 90.0f  * REAL_TILES_PER_MIN_TO_ENGINE;
inline constexpr float SPEED_VERY_FAST = 120.0f * REAL_TILES_PER_MIN_TO_ENGINE;

// Ticks a freshly placed troop or building spends inert (targetable and
// damageable, but unable to move, target or attack): 1.0 s, as in the real
// game. Without it every defensive placement acts immediately, a systematic
// subsidy to defence (perception/UPSTREAM_REQUESTS.md item 14).
//
// Applied in CardFactories::applyCardMetadata, which spells, deploy effects,
// towers and projectiles bypass.
inline constexpr int DEPLOY_TIME_TICKS = 10;

struct CardStats {
    int id = 0;
    std::string name;
    float cost = 0.0f;
    Archetype archetype = Archetype::MeleeSquad;

    // Combat stats, for every archetype but Spell.
    int hp = 0;
    float speed = 0.0f;
    float attackRange = 0.0f;
    int damage = 0;
    int attackCooldown = 0;
    char symbol = '?';
    bool ignoresRiver = false;
    bool isFlying = false;
    bool targetsAir = false;

    // Spawn offsets for each unit; a single-unit card is one {0, 0}.
    std::vector<Vector2D> spawnOffsets{ Vector2D{ 0.0f, 0.0f } };

    // Applied on every landed attack by every unit of the card (e.g. Ice
    // Wizard's slow, Ice Spirit's stun). Not Ice Golem: its slow is on the
    // death explosion (card 40).
    std::shared_ptr<IOnHitEffect> onHit;

    // Spell archetype only. Ground-only control spells (The Log, Barbarian
    // Barrel) opt in; most spells hit air and ground.
    float spellRadius = 0.0f;
    int spellDelayTicks = 0;
    bool spellGroundOnly = false;
    // Multi-hit spells (Poison, Arrows); see AreaSpell::remainingHits.
    int spellRemainingHits = 1;
    int spellTickInterval = 0;
    // An on-hit effect the spell applies to whatever it touches, independent of
    // damage (e.g. Freeze's stun).
    std::shared_ptr<IOnHitEffect> spellOnHit;

    // Rage: buffs allies instead of damaging enemies (AreaSpell's buffsAllies).
    bool spellBuffsAllies = false;
    float spellBuffMultiplier = 1.0f;
    int spellBuffDurationTicks = 0;

    // Positive pushes away, negative pulls toward the centre (Fireball, Rocket,
    // Giant Snowball; Tornado).
    float spellKnockback = 0.0f;

    // --- rolling sweep (The Log, Barbarian Barrel) ---
    // See AreaSpell's rolling-sweep block. spellRollWidth is a FULL width (the
    // published Log 3.9 and Barrel 2.6 are widths). spellRollRange == 0 means
    // not a roller.
    float spellRollRange = 0.0f;
    float spellRollWidth = 0.0f;
    float spellRollSpeed = 0.0f;
    float spellRollKnockback = 0.0f;

    // Spell-spawns-troops (Goblin Barrel, Royal Delivery, Graveyard); see
    // AreaSpell::spawnOnDetonate.
    std::shared_ptr<IPeriodicEffect> spellSpawnEffect;

    // Clone; see AreaSpell::clonesAllies.
    bool spellClonesAllies = false;

    // A one-time area burst when a troop-shaped card deploys (e.g. Electro
    // Wizard's zap). Radius 0 means none.
    float spawnEffectRadius = 0.0f;
    int spawnEffectDamage = 0;
    std::shared_ptr<IOnHitEffect> spawnEffectOnHit;

    // Fired once when a unit of this card dies (e.g. Golem's Golemites).
    // Concrete effects live in core/.
    std::shared_ptr<IDeathEffect> deathEffect;

    // Ramping damage (Inferno Tower); see CombatEntity::getCurrentDamage.
    // rampFullTick == 0 means none.
    int rampMidTick = 0;
    int rampFullTick = 0;
    float rampStartFraction = 1.0f;
    float rampMidFraction = 1.0f;
    // Fourth ramp stage and reset grace period (Inferno Dragon Evolution only).
    int rampStage4Tick = 0;
    float rampStage4Fraction = 1.0f;
    int rampGracePeriodTicks = 0;

    // Split-target attacks (Electro Wizard); see
    // CombatEntity::findSplitTargets.
    int maxSplitTargets = 1;
    // Electro Dragon's chain.
    bool splitTargetsFullDamage = false;

    // Boomerang projectiles (Executioner), RangedSquad only.
    bool boomerang = false;
    int boomerangReturnDelayTicks = 0;

    // Kamikaze (Wall Breakers, the Spirits); see
    // CombatEntity::dieAfterFirstHit.
    bool dieAfterFirstHit = false;

    // Deploy anywhere (Miner, Goblin Drill); otherwise troops and buildings
    // stay on their own half.
    bool deployAnywhere = false;

    // Splash on every regular attack (Wizard, Bowler, Valkyrie...); see
    // CombatEntity::applySplashDamage. Not sourced data: flagged cards share
    // one engine constant.
    float splashRadius = 0.0f;

    // Shield hp (Guards, Royal Recruits, Dark Prince, Cannon Cart); see
    // CombatEntity::takeDamage. Not sourced data.
    int shieldHp = 0;

    // Charge bonus (Prince, Battle Ram, Ram Rider, Royal Hogs, Bandit). Not
    // sourced data.
    float chargeThreshold = 0.0f;
    float chargeMultiplier = 1.0f;
    // Sticky charge (Evolved Battle Ram only).
    bool chargeIsSticky = false;

    // Enrage (Berserker). Set via withEnrage, which reads back `hp`.
    int enrageMaxHp = 0;
    int enrageHealPerHit = 0;

    // Parry (Ronin).
    int parryIntervalTicks = 0;

    // Hook (Fisherman).
    float hookRange = 0.0f;

    // Invisibility (Royal Ghost, Suspicious Bush). Also blocks splash and spell
    // damage while invisible, since both gate on isTargetable(); the real game
    // still lets area effects land.
    bool startsInvisible = false;
    int revealTicksAfterAttack = 0;

    // Compound cards (Goblin Giant, Ram Rider, Goblin Machine, Goblin Gang,
    // Rascals): a second, independently targeting unit spawned at the same
    // point by CardFactories::spawn.
    std::shared_ptr<CardStats> secondaryUnit;

    // Periodic spawning while alive (Witch, Night Witch, Furnace, Barbarian
    // Hut, Goblin Hut, Tombstone, Goblin Drill). First fires after one full
    // interval.
    std::shared_ptr<IPeriodicEffect> periodicEffect;
    int periodicIntervalTicks = 0;

    // Ally auras on landed attacks: Rune Giant's every-Nth-attack buff, Battle
    // Healer's heal. Independent opt-ins.
    float auraRadius = 0.0f;
    int auraMaxTargets = 1000000;
    int auraEveryNAttacks = 0;
    float auraBuffMultiplier = 1.0f;
    int auraBuffDurationTicks = 0;
    int healAllyAmount = 0;

    // Dash invulnerability (Bandit).
    bool chargeGrantsInvulnerability = false;
    // A stun fully resets the attack cooldown (Sparky).
    bool resetCooldownOnFreeze = false;
    // Recoil after attacking (Firecracker).
    float recoilDistance = 0.0f;
    // Minimum range / blind spot (Mortar).
    float minAttackRange = 0.0f;
    // Sight / aggro range; see CombatEntity::sightRange. 5.5 is the sourced
    // standard for cards without their own value.
    float sightRange = 5.5f;
    // Deploy delay before the first attack (X-Bow); see
    // CombatEntity::seedCooldown.
    int initialCooldownTicks = 0;
    // HP-threshold transform (Cannon Cart).
    float transformAtHpFraction = 0.0f;
    int transformLifetimeTicks = 0;
    bool transformBecomesStationary = false;
    // Transform into a death effect (Goblin Demolisher); separate from
    // `deathEffect`.
    bool transformKillsSelf = false;
    std::shared_ptr<IDeathEffect> transformDeathEffect;

    // Periodic jump (Mega Knight).
    float jumpMinRange = 0.0f;
    float jumpMaxRange = 0.0f;
    float jumpDamageMultiplier = 1.0f;
    float jumpSplashRadius = 0.0f;

    // Piercing line (Bowler, Magic Archer).
    bool lineSplash = false;
    float lineSplashRange = 0.0f;

    // Range-based damage falloff (Hunter). Not sourced data.
    bool rangeFalloff = false;
    float rangeFalloffMinFraction = 1.0f;

    // Bonus damage within a distance band (Archers' Power Shot, Executioner's
    // Axe Smash). rangeBandMaxDist == 0 disables it.
    float rangeBandMinDist = 0.0f;
    float rangeBandMaxDist = 0.0f;
    float rangeBandDamageMultiplier = 1.0f;

    // Spell only: Vines' top-N-highest-HP targeting.
    int spellTargetTopHpCount = 0;
    // Spell only: Void's three-tier damage.
    bool spellTieredDamage = false;
    int spellTierSingleDamage = 0;
    int spellTierFewDamage = 0;
    int spellTierManyDamage = 0;

    // Champion marker and activated ability. The marker changes no combat
    // behaviour; it lets GameManager::activateChampionAbility find the deployed
    // Champion.
    bool isChampion = false;
    // Hero marker: an ordinary troop with its own id and an activated ability
    // (e.g. Hero Musketeer's "Trusty Turret"). A separate flag, but every
    // Champion-slot consumer checks `isChampion || isHero`.
    bool isHero = false;
    float abilityElixirCost = 0.0f;
    int abilityCooldownTicks = 0;
    std::shared_ptr<IAbilityEffect> abilityEffect;
    // -1 is unlimited.
    int abilityUsesLimit = -1;
    // Lockout before the FIRST use only (Hero Mega Minion: 1.5 s), distinct
    // from the between-uses cooldown.
    int initialAbilityCooldownTicks = 0;
    // Post-death squad reactivation (Hero Goblins' "Banner Brigade"): once
    // every live unit of this slot's card has died, activateChampionAbility has
    // this many ticks to fire postDeathAbilityEffect (tracked in
    // ChampionSlotState::lastSquadWipeTick). A card with this has no alive-path
    // ability; it reuses abilityElixirCost.
    int abilityUsableAfterDeathTicks = 0;
    std::shared_ptr<IPeriodicEffect> postDeathAbilityEffect;

    // Soul collection (Skeleton King).
    float soulCollectionRadius = 0.0f;
    int maxSouls = 0;

    // Hit-speed ramp (Little Prince); see CombatEntity::hitSpeedRampMidTick.
    int hitSpeedRampMidTick = 0;
    int hitSpeedRampFullTick = 0;
    float hitSpeedRampMidFraction = 1.0f;
    float hitSpeedRampFullFraction = 1.0f;

    // A burst every Nth attack (Dagger Duchess, some Evolutions).
    int burstEveryNAttacks = 0;
    float burstDamageMultiplier = 1.0f;

    // An effect on taking damage (some Evolutions, e.g. Barbarians).
    std::shared_ptr<IOnDamageTakenEffect> onDamageTaken;

    // Self-heal on landing a hit (Evolved Bats).
    int healOnHitAmount = 0;
    int healOnHitMaxHp = 0;

    // Self-spawn on landing a hit (Evolved Skeletons).
    std::shared_ptr<IPeriodicEffect> onHitSpawnEffect;

    // Pull nearby enemy troops on landing a hit (Evolved Valkyrie).
    float onHitPullRadius = 0.0f;
    float onHitPullDistance = 0.0f;
    int onHitPullDamage = 0;

    // Self-haste on each landed hit (Evolved Barbarians' Blade Rage).
    int selfHasteDurationTicks = 0;
    float selfHasteCooldownMultiplier = 1.0f;

    // Fluent setters, so a registry entry fits on a line or two.
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
    CardStats& withRampStage4(int stage4Tick, float stage4Fraction) {
        rampStage4Tick = stage4Tick;
        rampStage4Fraction = stage4Fraction;
        return *this;
    }
    CardStats& withRampGracePeriod(int graceTicks) {
        rampGracePeriodTicks = graceTicks;
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
    CardStats& withStickyCharge() {
        chargeIsSticky = true;
        return *this;
    }
    // Reads back `hp`, so chain it after troop(...).
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
        auraRadius = radius; // shared with the buff aura; a card uses one or the other
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
    // width is the FULL corridor width; speed is tiles per tick; knockback
    // direction depends on where across the corridor a unit is caught
    // (AreaSpell::updateRoll).
    CardStats& withRollingSweep(float range, float width, float speed, float knockback) {
        spellRollRange = range;
        spellRollWidth = width;
        spellRollSpeed = speed;
        spellRollKnockback = knockback;
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
    CardStats& withSightRange(float range) {
        sightRange = range;
        return *this;
    }
    CardStats& withDeployDelay(int ticks) {
        initialCooldownTicks = ticks;
        return *this;
    }
    // Reads back `hp`, like withEnrage.
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
    // splashRadius doubles as the line's half-width.
    CardStats& withLineSplash(float range) {
        lineSplash = true;
        lineSplashRange = range;
        return *this;
    }
    // minFraction is an engine constant, not sourced data.
    CardStats& withRangeFalloff(float minFraction) {
        rangeFalloff = true;
        rangeFalloffMinFraction = minFraction;
        return *this;
    }
    CardStats& withRangeBandBonus(float minDist, float maxDist, float multiplier) {
        rangeBandMinDist = minDist;
        rangeBandMaxDist = maxDist;
        rangeBandDamageMultiplier = multiplier;
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
    // withChampionAbility, setting isHero instead.
    CardStats& withHeroAbility(float elixirCost, int cooldownTicks, std::shared_ptr<IAbilityEffect> effect,
            int usesLimit = -1) {
        isHero = true;
        abilityElixirCost = elixirCost;
        abilityCooldownTicks = cooldownTicks;
        abilityEffect = std::move(effect);
        abilityUsesLimit = usesLimit;
        return *this;
    }
    CardStats& withInitialAbilityCooldown(int ticks) {
        initialAbilityCooldownTicks = ticks;
        return *this;
    }
    // No abilityEffect or cooldown: a post-death card has no alive-path
    // ability.
    CardStats& withPostDeathAbility(float elixirCost, int windowTicks, std::shared_ptr<IPeriodicEffect> effect) {
        isHero = true;
        abilityElixirCost = elixirCost;
        abilityUsableAfterDeathTicks = windowTicks;
        postDeathAbilityEffect = std::move(effect);
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
    CardStats& withOnHitPull(float radius, float distance, int damage = 0) {
        onHitPullRadius = radius;
        onHitPullDistance = distance;
        onHitPullDamage = damage;
        return *this;
    }
    CardStats& withSelfHasteOnHit(int durationTicks, float cooldownMultiplier) {
        selfHasteDurationTicks = durationTicks;
        selfHasteCooldownMultiplier = cooldownMultiplier;
        return *this;
    }
    // Permanent damage-taken reduction from spawn (Evolved Knight's shield),
    // applied via applyCurse. The real shield holds only while not attacking;
    // with no such signal here it is a smaller flat reduction. 1.0 is none.
    float passiveDamageReduction = 1.0f;
    CardStats& withPassiveDamageReduction(float multiplier) {
        passiveDamageReduction = multiplier;
        return *this;
    }
};
