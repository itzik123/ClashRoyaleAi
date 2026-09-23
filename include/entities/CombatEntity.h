#pragma once
#include "CardEntity.h"
#include "Board.h"
#include "LanePath.h"
#include "OnHitEffect.h"
#include "OnDamageTakenEffect.h"
#include "DeathEffect.h"
#include "PeriodicEffect.h"
#include "AbilityEffect.h"
#include <memory>
#include <limits>
#include <vector>
#include <algorithm>

class CombatEntity;

// Declared here, defined after the class, because update() calls them from
// inside the class body. applyAreaBuff returns the entities it buffed.
inline std::vector<std::shared_ptr<CombatEntity>> applyAreaBuff(Board& board, const Vector2D& origin, float radius, int excludeId,
    int team, float multiplier, int durationTicks, int maxTargets);
inline void applyAreaHeal(Board& board, const Vector2D& origin, float radius, int excludeId,
    int team, int amount);
// Declared early for the same reason (Mega Knight's jump in update()).
inline void applySplashDamage(Board& board, const Vector2D& origin, float radius, int excludeId,
    int attackerId, int attackerTeam, int attackerCardId, int dealt);
// Declared early for Evolved Valkyrie's on-hit pull in update().
inline void applyPullNearby(Board& board, const Vector2D& origin, float radius, float distance,
    int excludeId, int attackerTeam);
// Declared early for Hero Knight's taunt (HeroKnightTauntEffect).
inline void applyTauntNearby(Board& board, const Vector2D& origin, float radius, int ticks,
    int taunterId, int taunterTeam);

class CombatEntity : public CardEntity {
protected:
    float attackRange;
    int damage;
    int attackCooldown;
    float currentCooldown;
    std::vector<std::shared_ptr<IOnHitEffect>> onHitEffects;

    // Ramp bookkeeping: the locked target and how many consecutive ticks on it.
    // Tracked for every entity so the logic lives in one place.
    int currentTargetId = -1;
    int ticksOnTarget = 0;
    // Ticks since the last landed hit. Consulted only with rampGracePeriodTicks
    // > 0 (Inferno Dragon Evolution): a lost or switched target then keeps its
    // ramp stage for up to the grace period instead of resetting at once. A
    // stun still resets immediately.
    int ticksSinceLastHit = 0;

    // How many targets this attack landed on, set just before performAttack()
    // so getCurrentDamage() can divide by it.
    int currentHitCount = 1;

public:
    // Freeze lives here, not on Entity: only things that attack or move can be
    // frozen.
    int freezeTicks = 0;
    float freezeSlow = 1.0f;

    // Whether this unit was frozen when the tick began: set at the top of
    // update(), before freezeTicks is decremented, and read by
    // Troop::moveTowards later in the same call, so attacks and movement see
    // the same freeze. Recomputed every update().
    bool frozenThisTick = false;

    // Ticks left of deploy time (CardStats.h, DEPLOY_TIME_TICKS): on the board,
    // targetable and damageable, but not moving, targeting or attacking.
    //
    // Not freezeTicks: a freeze means "stunned" and resets the ramp and charge,
    // and a new unit has not been stunned. The implicit copy carries it, so
    // snapshots keep the exact remaining deploy time.
    int deployTicksRemaining = 0;

    // Damage-over-time mark (Dart Goblin / Firecracker Evolutions; see
    // applyDot, PoisonOnHit). Ticks down independently of freeze and curse.
    int dotDamagePerTick = 0;
    int dotTicksRemaining = 0;
    int dotTickInterval = 0;
    int dotTicksUntilNextDamage = 0;

    // Whether findTarget() may pick a flyer. Only read on `this`, unlike
    // isFlying.
    bool targetsAir = false;

    // Fired once by onDeath() (e.g. Golem spawning Golemites).
    std::shared_ptr<IDeathEffect> deathEffect;

    // Ramping damage (Inferno Tower): the longer the lock on the same target,
    // the more damage: rampStartFraction until rampMidTick, rampMidFraction
    // until rampFullTick, then full. Reset by a target switch, losing the
    // target, or a stun. rampFullTick == 0 disables it.
    int rampMidTick = 0;
    int rampFullTick = 0;
    float rampStartFraction = 1.0f;
    float rampMidFraction = 1.0f;

    // Fourth ramp stage beyond rampFullTick (Inferno Dragon Evolution only). 0
    // disables it.
    int rampStage4Tick = 0;
    float rampStage4Fraction = 1.0f;

    // Grace period before a lost or switched target resets the ramp (see
    // ticksSinceLastHit). 0 resets at once.
    int rampGracePeriodTicks = 0;

    // Split-target attacks (Electro Wizard): hits up to this many of the
    // closest enemies, each for damage / (targets actually hit).
    int maxSplitTargets = 1;
    // Temporary split-target window (Hero Magic Archer's Triple Threat), an
    // approximation of extra arrows via the split machinery.
    // baseMaxSplitTargets is restored when the window closes.
    int temporarySplitTargetsTicksRemaining = 0;
    int baseMaxSplitTargets = 1;

    // Electro Dragon's chain: every chained target takes full damage rather
    // than a share.
    bool splitTargetsFullDamage = false;

    // Splash (Wizard, Bowler, Valkyrie...): every regular attack also hits
    // every other enemy within this radius of the primary target, ground and
    // air alike, regardless of targetsAir. A free function (applySplashDamage)
    // because Projectile needs it too.
    float splashRadius = 0.0f;

    // Shield (Guards, Royal Recruits, Dark Prince, Cannon Cart): a second hp
    // pool that absorbs damage from any source first. Does not regenerate.
    int shieldHp = 0;
    // Timed shield expiry (Hero Knight's taunt: the shield lasts 5 s even if
    // not depleted). 0 leaves shieldHp permanent.
    int shieldExpiresTicksRemaining = 0;

    // Forced retarget (Hero Knight's taunt): while > 0, update() attacks
    // forcedTargetEntityId instead of its normal target (see applyTauntNearby).
    int forcedTargetEntityId = -1;
    int forcedTargetTicksRemaining = 0;

    // Charge (Prince, Battle Ram, Ram Rider, Royal Hogs, Bandit): after moving
    // chargeThreshold tiles without landing a hit, the next attack deals
    // chargeMultiplier x damage, and progress resets either way. 0 disables it.
    float chargeThreshold = 0.0f;
    float chargeMultiplier = 1.0f;
    float chargeProgress = 0.0f;

    // Sticky charge (Evolved Battle Ram): once reached, the charge bonus
    // applies to every hit for the rest of its life. chargeHasStuck is
    // internal.
    bool chargeIsSticky = false;
    bool chargeHasStuck = false;

    // Enrage (Berserker): the attack cooldown shortens (up to 2x speed at 0 hp)
    // and each hit heals, scaling with hp lost. enrageMaxHp == 0 disables it
    // (see CardStats::withEnrage).
    int enrageMaxHp = 0;
    int enrageHealPerHit = 0;

    // Parry (Ronin): every parryIntervalTicks, the next incoming hit is
    // negated. The real card also reflects damage and parries only ground
    // melee; neither is modelled (takeDamage has no attacker, and there is no
    // melee/ranged distinction). Ready at spawn.
    int parryIntervalTicks = 0;
    int parryTicksUntilReady = 0;

    // Hook (Fisherman): a target beyond attackRange but within hookRange is
    // pulled to just inside melee range. Consumes the cooldown; the damage
    // lands on a later hit.
    float hookRange = 0.0f;

    // Invisibility (Royal Ghost, Suspicious Bush): untargetable except briefly
    // after landing a hit.
    bool startsInvisible = false;
    int revealTicksAfterAttack = 0;
    int visibleTicksRemaining = 0;

    // Periodic effects while alive (Witch, Night Witch, Furnace, huts,
    // Tombstone, Goblin Drill): fires periodicEffect every
    // periodicIntervalTicks. 0 disables it.
    std::shared_ptr<IPeriodicEffect> periodicEffect;
    int periodicIntervalTicks = 0;
    int periodicTicksUntilNext = 0;

    // Temporary damage buff (Rage, Rune Giant's enchant). Real Rage also speeds
    // movement and attacks; only damage is modelled, since speed lives on
    // Troop.
    float buffDamageMultiplier = 1.0f;
    int buffTicksRemaining = 0;

    // Damage-taken debuff (Mother Witch's curse), applied in takeDamage()
    // before shield and parry.
    float curseDamageTakenMultiplier = 1.0f;
    int curseTicksRemaining = 0;
    // Latch: Mother Witch's hog spawn is armed once (see CursedHogOnHit); the
    // curse's duration still refreshes.
    bool curseDeathSpawnAttached = false;
    // Latch: the Royal Chef serves each ally once (see RoyalChefBuffEffect).
    bool royalChefServed = false;

    // Ally auras on landed attacks: Rune Giant's every-Nth buff, Battle
    // Healer's heal. Independent opt-ins.
    float auraRadius = 0.0f;
    int auraMaxTargets = 1000000; // everyone in radius unless capped
    int auraEveryNAttacks = 0;
    int attacksSinceAura = 0;
    float auraBuffMultiplier = 1.0f;
    int auraBuffDurationTicks = 0;
    int healAllyAmount = 0;

    // Every Nth landed hit deals burstDamageMultiplier x damage (Dagger
    // Duchess, some Evolutions). currentHitIsBurst is set just before
    // performAttack and read by getCurrentDamage().
    int burstEveryNAttacks = 0;
    float burstDamageMultiplier = 1.0f;
    int attacksSinceBurst = 0;
    bool currentHitIsBurst = false;

    // Fired when this entity actually takes damage; the receiving-end
    // counterpart of onHitEffects.
    std::shared_ptr<IOnDamageTakenEffect> onDamageTakenEffect;

    // Self-heal on landing a hit (Evolved Bats); healOnHitMaxHp is an absolute
    // cap.
    int healOnHitAmount = 0;
    int healOnHitMaxHp = 0;

    // Self-spawn on landing a hit (Evolved Skeletons); the cap lives in
    // CappedSpawnOnHitEffect.
    std::shared_ptr<IPeriodicEffect> onHitSpawnEffect;

    // Pull nearby enemies on landing a hit (Evolved Valkyrie). Buildings are
    // never moved (pullToward), but onHitPullDamage reaches them, towers
    // included, as sourced.
    float onHitPullRadius = 0.0f;
    float onHitPullDistance = 0.0f;
    int onHitPullDamage = 0;

    // Kamikaze (Wall Breakers, the Spirits): dies after its one hit. A ranged
    // unit dies when the shot is launched, slightly before the real on-arrival
    // timing.
    bool dieAfterFirstHit = false;

    // Dash invulnerability (Bandit). With no separate dash state, immune once
    // past half the charge distance: a longer window than the real ~0.8 s.
    bool chargeGrantsInvulnerability = false;

    // A stun resets, not just slows, the attack cooldown (Sparky restarts her
    // wind-up after a Zap).
    bool resetCooldownOnFreeze = false;

    // Recoil after attacking (Firecracker); see pushAway.
    float recoilDistance = 0.0f;

    // Minimum range / blind spot (Mortar). The real attacker idles against a
    // target inside it; here such a target is simply not valid, so the attacker
    // picks the next legal one.
    float minAttackRange = 0.0f;

    // Sight / aggro range (distinct from attackRange; see findTarget()): how
    // far this unit can detect an enemy to go fight instead of heading for a
    // tower. Sourced per card; 5.5 is the source's standard for cards without
    // their own value. Set per card for buildings too, not derived from
    // attackRange.
    float sightRange = 5.5f;

    // HP-threshold transform (Cannon Cart: at <= 50% hp it grounds itself and
    // self-destructs after a fixed lifespan). The fraction is of
    // transformCheckMaxHp, the spawn hp. Grounding reuses applyFreeze(ticks,
    // 0).
    float transformAtHpFraction = 0.0f;
    int transformCheckMaxHp = 0;
    int transformLifetimeTicks = 0;
    bool transformBecomesStationary = false;
    bool hasTransformed = false;
    int transformTicksRemaining = 0;

    // Transform into a different unit (Goblin Demolisher): an entity's C++ type
    // cannot change in place, so it dies at the threshold and
    // transformDeathEffect spawns the new form.
    bool transformKillsSelf = false;
    // A separate slot from `deathEffect`: a hit that kills outright, skipping
    // the threshold, must not spawn the transformed form.
    std::shared_ptr<IDeathEffect> transformDeathEffect;

    // Periodic jump (Mega Knight): a ground target between jumpMinRange and
    // jumpMaxRange is closed on instantly with a boosted, splashing hit.
    float jumpMinRange = 0.0f;
    float jumpMaxRange = 0.0f;
    float jumpDamageMultiplier = 1.0f;
    float jumpSplashRadius = 0.0f;

    // Piercing line (Bowler, Magic Archer): hits everyone within splashRadius
    // (reused as the half-width) of the line from this attacker toward the
    // target, out to lineSplashRange.
    bool lineSplash = false;
    float lineSplashRange = 0.0f;

    // Range falloff (Hunter). Not a sourced mechanic: the real card fires 10
    // randomly spread pellets and no curve is published. A deterministic linear
    // scale approximates "weaker at range", keeping combat reproducible.
    bool rangeFalloff = false;
    float rangeFalloffMinFraction = 1.0f; // damage fraction at max range

    // Bonus damage within a distance band (Archers' Power Shot: +50% at 4-6
    // tiles; Executioner's Axe Smash: +75% at <= 3.5), from lastAttackDistance.
    // rangeBandMaxDist == 0 disables it.
    float rangeBandMinDist = 0.0f;
    float rangeBandMaxDist = 0.0f;
    float rangeBandDamageMultiplier = 1.0f;

    // Distance to the target at the last attack, for getCurrentDamage(), which
    // has no target of its own.
    float lastAttackDistance = 0.0f;

    // Champion marker. Gates no combat behaviour; lets GameManager find the
    // deployed Champion.
    bool isChampion = false;
    // Hero marker; Champion-slot code checks `isChampion || isHero`, so Heroes
    // share the Champion path.
    bool isHero = false;
    // Elixir charged on each activation, on top of the card's deploy cost. 0
    // with a null abilityEffect means no ability.
    float abilityElixirCost = 0.0f;
    // Cooldown between activations. Starts at 0 (ready at deploy) and resets to
    // abilityCooldownTicks on each activation.
    int abilityCooldownTicks = 0;
    int abilityCooldownRemaining = 0;
    // What the ability does (e.g. MightyMinerEscapeEffect in core/). nullptr
    // makes activateAbility() a no-op.
    std::shared_ptr<IAbilityEffect> abilityEffect;
    // -1 is unlimited; otherwise a hard cap on activations (Boss Bandit: 2).
    int abilityUsesRemaining = -1;

    // Soul collection (Skeleton King): counts deaths within radius, ally or
    // enemy, via onNearbyDeath, up to maxSouls.
    float soulCollectionRadius = 0.0f;
    int soulCount = 0;
    int maxSouls = 0;

    // Timed invisibility plus attack-speed buff (Archer Queen's Cloaking Cape,
    // Boss Bandit's Getaway Grenade). Their movement-speed change is not
    // modelled.
    int temporaryInvisibilityTicksRemaining = 0;
    float temporaryHitSpeedMultiplier = 1.0f;

    // Self-haste on each landed hit, refreshed while attacking (Evolved
    // Barbarians' Blade Rage: +35% attack speed for 3 s). Separate from the
    // ability haste above, which carries invisibility. Movement speed not
    // modelled.
    int selfHasteDurationTicks = 0;
    float selfHasteCooldownMultiplier = 1.0f;
    int selfHasteTicksRemaining = 0;

    // Temporary flight (Hero Wizard's Fiery Flight): isFlying is read live
    // everywhere, so this timer only reverts it. ignoresRiver is fixed at spawn
    // and not toggled.
    int temporaryFlightTicksRemaining = 0;
    // On-hit tornado pulse during the flight window, centred on the TARGET
    // (unlike Evolved Valkyrie's pull, centred on self).
    int flightPulseTicksRemaining = 0;
    float flightPulseRadius = 0.0f;
    int flightPulseDamage = 0;
    float flightPulsePullDistance = 0.0f;

    // Hit-speed ramp while locked on the same target (Little Prince): the
    // cooldown shrinks, rather than the damage growing. 0 disables it.
    int hitSpeedRampMidTick = 0;
    int hitSpeedRampFullTick = 0;
    float hitSpeedRampMidFraction = 1.0f;
    float hitSpeedRampFullFraction = 1.0f;

    CombatEntity(int id, float x, float y, int hp, int team, char symbol,
        float attackRange, int damage, int attackCooldown)
        : CardEntity(id, x, y, hp, team, symbol),
        attackRange(attackRange), damage(damage),
        attackCooldown(attackCooldown), currentCooldown(0.0f) {}

    bool isTargetable() const override {
        return (!startsInvisible || visibleTicksRemaining > 0) && temporaryInvisibilityTicksRemaining <= 0;
    }

    // Soul collection: any death within radius, ally or enemy.
    void onNearbyDeath(Board&, const Vector2D& deathPosition, int) override {
        if (soulCollectionRadius <= 0.0f || soulCount >= maxSouls) return;
        if (position.distanceTo(deathPosition) <= soulCollectionRadius) {
            soulCount++;
        }
    }

    void applyBuff(float multiplier, int ticks) {
        buffDamageMultiplier = multiplier;
        buffTicksRemaining = ticks;
    }

    void applyCurse(float damageTakenMultiplier, int ticks) {
        curseDamageTakenMultiplier = damageTakenMultiplier;
        curseTicksRemaining = ticks;
    }

    // Fires abilityEffect and restarts the cooldown if configured, off cooldown
    // and not exhausted. Elixir is GameManager's concern and must be checked
    // and deducted before calling. Returns whether it fired.
    bool activateAbility(Board& board) {
        if (!abilityEffect || abilityCooldownRemaining > 0) return false;
        if (abilityUsesRemaining == 0) return false; // exhausted
        abilityEffect->apply(board, *this);
        abilityCooldownRemaining = abilityCooldownTicks;
        if (abilityUsesRemaining > 0) abilityUsesRemaining--;
        return true;
    }

    // Deploy delay before the first shot (X-Bow's slow lock-on). Called once
    // after spawn by CardFactories; the one seam onto the protected cooldown.
    void seedCooldown(int ticks) {
        currentCooldown = static_cast<float>(ticks);
    }

    // Parry is checked before shield, so a parried hit costs no shield. The
    // shield absorbs first and only the excess reaches hp.
    void takeDamage(int amount) override {
        if (chargeGrantsInvulnerability && chargeThreshold > 0.0f
            && chargeProgress >= chargeThreshold * 0.5f) {
            return; // mid-dash: immune
        }
        if (curseTicksRemaining > 0) {
            amount = static_cast<int>(amount * curseDamageTakenMultiplier);
        }
        if (parryIntervalTicks > 0 && parryTicksUntilReady <= 0) {
            parryTicksUntilReady = parryIntervalTicks;
            return; // negated
        }
        if (shieldHp > 0) {
            int absorbed = (shieldHp < amount) ? shieldHp : amount;
            shieldHp -= absorbed;
            amount -= absorbed;
        }
        if (amount > 0) {
            Entity::takeDamage(amount);
            if (onDamageTakenEffect) onDamageTakenEffect->apply(*this);
        }
    }

    // The latest application replaces the mark, like applyBuff / applyCurse.
    void applyDot(int damagePerTick, int totalTicks, int tickInterval) {
        dotDamagePerTick = damagePerTick;
        dotTicksRemaining = totalTicks;
        dotTickInterval = tickInterval;
        dotTicksUntilNextDamage = tickInterval;
    }

    void applyFreeze(int ticks, float slowFactor) {
        // Duration and strength are judged independently, so a new freeze never
        // leaves the target better off: a shorter but stronger slow is not
        // dropped for a longer, weaker one.
        freezeTicks = std::max(freezeTicks, ticks);
        freezeSlow = std::min(freezeSlow, slowFactor);
        if (resetCooldownOnFreeze && ticks > 0) {
            currentCooldown = static_cast<float>(attackCooldown);
        }
    }

    // Composes extra behaviour onto every successful attack.
    void addOnHitEffect(std::shared_ptr<IOnHitEffect> effect) {
        onHitEffects.push_back(std::move(effect));
    }

    void update(Board& board) override {
        // Deploy time: only the decrement happens here. The bail-out is below
        // the status timers, because a deploying unit is inert, not
        // time-stopped: poison, cooldowns and buffs keep ticking.
        const bool deploying = deployTicksRemaining > 0;
        if (deploying) deployTicksRemaining--;

        // Captured before the decrement, so the last tick of a freeze still
        // counts as frozen for the ramp reset.
        bool wasFrozen = freezeTicks > 0;
        // Published for Troop::moveTowards, which runs after the decrement.
        frozenThisTick = wasFrozen;
        ticksSinceLastHit++; // zeroed below when a hit lands

        if (transformAtHpFraction > 0.0f && !hasTransformed && transformCheckMaxHp > 0
            && static_cast<float>(hp) / static_cast<float>(transformCheckMaxHp) <= transformAtHpFraction) {
            hasTransformed = true;
            if (transformKillsSelf) {
                hp = 0;
                if (transformDeathEffect) transformDeathEffect->apply(board, position, team);
                // Dead this tick: skip targeting, movement and attack.
                // cleanDeadEntities fires nothing more, since deathEffect is
                // unset on this card.
                return;
            }
            transformTicksRemaining = transformLifetimeTicks;
            if (transformBecomesStationary) applyFreeze(transformLifetimeTicks, 0.0f);
        }
        if (hasTransformed && transformLifetimeTicks > 0) {
            transformTicksRemaining--;
            if (transformTicksRemaining <= 0) hp = 0;
        }

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

        if (parryIntervalTicks > 0 && parryTicksUntilReady > 0) parryTicksUntilReady--;
        if (startsInvisible && visibleTicksRemaining > 0) visibleTicksRemaining--;
        if (buffTicksRemaining > 0) buffTicksRemaining--;
        if (curseTicksRemaining > 0) curseTicksRemaining--;
        if (abilityCooldownRemaining > 0) abilityCooldownRemaining--;
        if (temporaryInvisibilityTicksRemaining > 0) temporaryInvisibilityTicksRemaining--;
        if (selfHasteTicksRemaining > 0) selfHasteTicksRemaining--;
        if (temporaryFlightTicksRemaining > 0) {
            temporaryFlightTicksRemaining--;
            if (temporaryFlightTicksRemaining == 0) isFlying = false;
        }
        if (flightPulseTicksRemaining > 0) flightPulseTicksRemaining--;
        if (temporarySplitTargetsTicksRemaining > 0) {
            temporarySplitTargetsTicksRemaining--;
            if (temporarySplitTargetsTicksRemaining == 0) maxSplitTargets = baseMaxSplitTargets;
        }
        if (forcedTargetTicksRemaining > 0) forcedTargetTicksRemaining--;
        if (shieldExpiresTicksRemaining > 0) {
            shieldExpiresTicksRemaining--;
            if (shieldExpiresTicksRemaining == 0) shieldHp = 0;
        }

        // The damage-over-time mark, through takeDamage so shield, parry and
        // curse apply.
        if (dotTicksRemaining > 0) {
            dotTicksRemaining--;
            dotTicksUntilNextDamage--;
            if (dotTicksUntilNextDamage <= 0 && dotDamagePerTick > 0) {
                takeDamage(dotDamagePerTick);
                dotTicksUntilNextDamage = dotTickInterval;
            }
        }

        // Above: bookkeeping that runs while deploying. Below: action, which a
        // deploying unit takes none of.
        if (deploying) return;

        if (periodicIntervalTicks > 0) {
            periodicTicksUntilNext--;
            if (periodicTicksUntilNext <= 0) {
                if (periodicEffect) periodicEffect->apply(board, position, team);
                periodicTicksUntilNext = periodicIntervalTicks;
            }
        }

        // Targeting. A taunt overrides everything. Otherwise the lock is
        // trusted only while the locked target is within attack range; out of
        // range (still approaching, knocked back, pulled away) there is no
        // lock, and findTarget() re-scans every tick for whatever is closest in
        // sight. A stun breaks the lock outright.
        std::shared_ptr<Entity> target;
        if (forcedTargetTicksRemaining > 0) {
            for (const auto& e : board.getEntities()) {
                if (e->id == forcedTargetEntityId && e->isAlive() && e->isTargetable()) { target = e; break; }
            }
        }
        if (!target) {
            target = wasFrozen ? nullptr : resolveCurrentTarget(board);
            if (target && position.distanceTo(target->position) > effectiveRangeTo(target)) {
                target = nullptr;
            }
            if (!target) {
                target = findTarget(board);
            }
        }

        if (target) {
            if (wasFrozen) {
                // A stun resets unconditionally, grace period or not.
                currentTargetId = target->id;
                ticksOnTarget = 0;
                ticksSinceLastHit = 0;
            } else if (target->id != currentTargetId) {
                bool withinGrace = rampGracePeriodTicks > 0 && ticksSinceLastHit < rampGracePeriodTicks;
                currentTargetId = target->id;
                if (!withinGrace) ticksOnTarget = 0;
                // else keep the ramp stage across the switch; this tick neither
                // resets nor increments it.
            } else {
                ticksOnTarget++;
            }

            float dist = position.distanceTo(target->position);
            float effectiveAttackRange = effectiveRangeTo(target);

            if (dist <= effectiveAttackRange) {
                if (currentCooldown == 0.0f) {
                    // Effects are applied by performAttack, since when they
                    // fire depends on when the damage lands (at once for a
                    // direct hit, on arrival for a projectile).
                    lastAttackDistance = dist;
                    // Reset here, not on projectile arrival: the only
                    // grace-period card (Inferno Dragon Evolution) attacks
                    // instantly.
                    ticksSinceLastHit = 0;
                    if (burstEveryNAttacks > 0) {
                        attacksSinceBurst++;
                        currentHitIsBurst = (attacksSinceBurst >= burstEveryNAttacks);
                        if (currentHitIsBurst) attacksSinceBurst = 0;
                    } else {
                        currentHitIsBurst = false;
                    }
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
                    if (chargeThreshold > 0.0f) {
                        if (chargeIsSticky && chargeProgress >= chargeThreshold) chargeHasStuck = true;
                        if (!chargeHasStuck) chargeProgress = 0.0f;
                    }
                    if (startsInvisible) visibleTicksRemaining = revealTicksAfterAttack;
                    if (enrageMaxHp > 0) {
                        float hpFraction = static_cast<float>(hp) / static_cast<float>(enrageMaxHp);
                        currentCooldown *= (0.5f + 0.5f * hpFraction); // up to 2x attack speed at 0 hp
                        if (enrageHealPerHit > 0 && hp < enrageMaxHp) {
                            hp = (hp + enrageHealPerHit < enrageMaxHp) ? hp + enrageHealPerHit : enrageMaxHp;
                        }
                    }
                    // Little Prince's hit-speed ramp; ticksOnTarget already
                    // includes this hit.
                    if (hitSpeedRampFullTick > 0) {
                        float cooldownFraction = (ticksOnTarget >= hitSpeedRampFullTick) ? hitSpeedRampFullFraction
                            : (ticksOnTarget >= hitSpeedRampMidTick) ? hitSpeedRampMidFraction
                            : 1.0f;
                        currentCooldown *= cooldownFraction;
                    }
                    // Temporary ability haste (Cloaking Cape, Getaway Grenade).
                    if (temporaryInvisibilityTicksRemaining > 0) currentCooldown *= temporaryHitSpeedMultiplier;
                    // Blade Rage: refresh the window, then haste the cooldown
                    // just set, so continued hits keep the haste alive.
                    if (selfHasteDurationTicks > 0) {
                        selfHasteTicksRemaining = selfHasteDurationTicks;
                        currentCooldown *= selfHasteCooldownMultiplier;
                    }
                    // Ally auras (Rune Giant's buff, Battle Healer's heal).
                    if (healAllyAmount > 0) {
                        applyAreaHeal(board, position, auraRadius, id, team, healAllyAmount);
                    }
                    if (auraEveryNAttacks > 0) {
                        attacksSinceAura++;
                        if (attacksSinceAura >= auraEveryNAttacks) {
                            attacksSinceAura = 0;
                            applyAreaBuff(board, position, auraRadius, id, team,
                                auraBuffMultiplier, auraBuffDurationTicks, auraMaxTargets);
                        }
                    }
                    if (recoilDistance > 0.0f) pushAway(*this, target->position, recoilDistance);
                    // Evolved Bats: heals past starting hp, up to the absolute
                    // healOnHitMaxHp.
                    if (healOnHitAmount > 0 && hp < healOnHitMaxHp) {
                        hp = std::min(hp + healOnHitAmount, healOnHitMaxHp);
                    }
                    // Evolved Skeletons: spawn on hit; the cap is inside
                    // CappedSpawnOnHitEffect.
                    if (onHitSpawnEffect) onHitSpawnEffect->apply(board, position, team);
                    // Whirlwind pull (Evolved Valkyrie): the pull damage
                    // reaches everyone in radius, including this attack's
                    // target and buildings; only the physical pull exempts
                    // buildings.
                    if (onHitPullRadius > 0.0f) {
                        if (onHitPullDamage > 0) {
                            applySplashDamage(board, position, onHitPullRadius, -1, id, team, cardId, onHitPullDamage);
                        }
                        applyPullNearby(board, position, onHitPullRadius, onHitPullDistance, id, team);
                    }
                    // Fiery Flight's tornado, centred on the target and
                    // additive: the primary target is not exempt.
                    if (flightPulseTicksRemaining > 0 && flightPulseRadius > 0.0f) {
                        applySplashDamage(board, target->position, flightPulseRadius, -1, id, team, cardId, flightPulseDamage);
                        applyPullNearby(board, target->position, flightPulseRadius, flightPulsePullDistance, -1, team);
                    }
                    if (dieAfterFirstHit) hp = 0;
                }
            } else if (jumpMaxRange > 0.0f && dist >= jumpMinRange && dist <= jumpMaxRange && currentCooldown == 0.0f) {
                // Jump (Mega Knight): close to just inside attack range and
                // land a boosted, splashing hit at once.
                pullToward(*this, target->position, dist - effectiveAttackRange + 0.1f);
                int jumpDamage = static_cast<int>(getCurrentDamage() * jumpDamageMultiplier);
                target->takeDamage(jumpDamage);
                board.statsEvents.notifyDamageDealt(
                    { id, team, cardId, target->id, target->cardId, target->team, jumpDamage, board.currentTick, target->isTower() });
                applySplashDamage(board, target->position, jumpSplashRadius, target->id, id, team, cardId, jumpDamage);
                currentCooldown = static_cast<float>(attackCooldown);
            } else if (hookRange > 0.0f && dist <= hookRange && currentCooldown == 0.0f) {
                // Hook: pull the target to just inside melee range; the damage
                // lands on a later hit. A no-op on a building.
                pullToward(*target, position, dist - effectiveAttackRange + 0.1f);
                currentCooldown = static_cast<float>(attackCooldown);
            } else {
                Vector2D beforeMove = position;
                // Lane-aware approach: a King objective is walked to up this
                // unit's own lane; any other target is its own position
                // (LanePath::approachPoint).
                moveTowards(board, LanePath::approachPoint(board, team, position, target));
                if (chargeThreshold > 0.0f) chargeProgress += beforeMove.distanceTo(position);
            }
        } else if (wasFrozen || rampGracePeriodTicks <= 0 || ticksSinceLastHit >= rampGracePeriodTicks) {
            // No target: reset fully if frozen this tick, with no grace period
            // configured, or once it has run out.
            currentTargetId = -1;
            ticksOnTarget = 0;
            ticksSinceLastHit = 0;
        }
        // else: within the grace period with no target, hold the ramp stage for
        // a new target.

        clampPosition(board);
    }

    void onDeath(Board& board) override {
        if (deathEffect) deathEffect->apply(board, position, team);
    }

    // The radius an entity occupies for range math; a troop uses
    // Entity::IMPLICIT_TROOP_RADIUS. Public and static because AreaSpell's
    // rolling sweep needs it too.
    static float effectiveRadiusOf(const Entity& e) {
        const float r = e.getCollisionRadius();
        return (r > 0.0f) ? r : Entity::IMPLICIT_TROOP_RADIUS;
    }

protected:
    // Entity-typed: targeting needs no CombatEntity surface; applyOnHitEffects
    // narrows where needed.
    bool isValidTarget(const std::shared_ptr<Entity>& entity) const {
        return entity && entity->team != this->team && entity->isAlive() && entity->isTargetable()
            && entity->id != this->id && (!entity->isFlying || targetsAir)
            && (minAttackRange <= 0.0f || position.distanceTo(entity->position) >= minAttackRange);
    }

    float ownEffectiveRadius() const { return effectiveRadiusOf(*this); }

    float effectiveRangeTo(const std::shared_ptr<Entity>& target) const {
        return attackRange + ownEffectiveRadius() + effectiveRadiusOf(*target);
    }

    // Sight to `target`; see effectiveSightWith.
    float effectiveSightTo(const std::shared_ptr<Entity>& target) const {
        return effectiveSightWith(ownEffectiveRadius(), *target);
    }

    // Sight is strict centre-to-centre: a Hog Rider's 9.5 means 9.5 tiles
    // centre to centre, as the published ranges do. Radii answer a hitbox
    // question, not a vision one.
    //
    // One floor: attack reach is surface-to-surface (effectiveRangeTo), so a
    // unit whose reach exceeds its sight could hit what it cannot acquire and
    // would stand idle. Sight is therefore floored at the unit's own attack
    // reach. The floor binds only where attackRange is close to sightRange
    // (towers, Musketeer-likes); for long-sight cards sightRange wins outright.
    // Without it, two Musketeers would stop out of each other's sight.
    //
    // Takes the attacker's radius as a parameter so findTarget computes it
    // once.
    float effectiveSightWith(float myRadius, const Entity& target) const {
        const float attackReach = attackRange + myRadius + effectiveRadiusOf(target);
        // Only for a card whose raw sightRange covers its raw attackRange,
        // which the catalogue guarantees. Otherwise the unit stays blind past
        // its sight, as test_combat_entity.cpp pins.
        if (sightRange >= attackRange && attackReach > sightRange) return attackReach;
        return sightRange;
    }

    // Re-validates the locked target by id rather than re-scanning for the
    // closest. nullptr if there is no lock or the target is no longer legal.
    std::shared_ptr<Entity> resolveCurrentTarget(Board& board) const {
        if (currentTargetId < 0) return nullptr;
        for (const auto& entity : board.getEntities()) {
            if (entity->id == currentTargetId) {
                return isValidTarget(entity) ? entity : nullptr;
            }
        }
        return nullptr;
    }

    // The closest valid enemy in sight, towers competing on equal terms; with
    // nothing in sight, this unit's lane objective. With update()'s "lock only
    // within attack range" rule this is the real game's model: locked on once
    // fighting, re-evaluating while chasing, heading for its lane's tower when
    // blind.
    virtual std::shared_ptr<Entity> findTarget(Board& board) const {
        std::shared_ptr<Entity> closestInSight = nullptr;
        float minSightDistance = std::numeric_limits<float>::max();
        std::shared_ptr<Entity> closestTower = nullptr;
        float minTowerDistance = std::numeric_limits<float>::max();

        const float myRadius = ownEffectiveRadius();   // loop invariant
        for (const auto& entity : board.getEntities()) {
            if (!isValidTarget(entity)) continue;
            float dist = position.distanceTo(entity->position);
            if (entity->isTower() && dist < minTowerDistance) {
                minTowerDistance = dist;
                closestTower = entity;
            }
            // The tower tracking above serves only the fallback. Ranked by
            // footprint distance (Entity::getTargetingRadius); the sight gate
            // uses centre distance.
            const float rank = dist - entity->getTargetingRadius();
            if (rank <= effectiveSightWith(myRadius, *entity) && rank < minSightDistance) {
                minSightDistance = rank;
                closestInSight = entity;
            }
        }
        if (closestInSight) return closestInSight;

        // Blind: walk this unit's own lane objective, not the nearest tower
        // (LanePath.h). Falls back to the nearest tower when the lane objective
        // is not a legal target for this attacker (a Mortar's blind spot,
        // air/ground).
        auto laneTarget = LanePath::laneObjective(board, team, position);
        if (laneTarget && isValidTarget(laneTarget)) return laneTarget;
        return closestTower;
    }

    // For maxSplitTargets > 1 (Electro Wizard): the N closest valid targets.
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

// Read-only views of protected combat stats, for the observation encoder's
// attribute channels.
public:
    float getAttackRange() const { return attackRange; }
    int getAttackCooldown() const { return attackCooldown; }

    // Ticks engaged with the current target, for the deepCopy divergence tests
    // (perception/UPSTREAM_REQUESTS.md item 17). Its only other proxy,
    // getDamagePerTick(), collapses it into ramp buckets, so a within-bucket
    // desync would be invisible.
    int getTicksOnTarget() const { return ticksOnTarget; }

    // Damage per tick, the comparable quantity, including ramp, charge and buff
    // state.
    float getDamagePerTick() const {
        int cooldown = attackCooldown > 0 ? attackCooldown : 1;
        return static_cast<float>(getCurrentDamage()) / static_cast<float>(cooldown);
    }

protected:

    int getCurrentDamage() const {
        int base = damage;
        if (rampFullTick > 0) {
            float fraction = (rampStage4Tick > 0 && ticksOnTarget >= rampStage4Tick) ? rampStage4Fraction
                : (ticksOnTarget >= rampFullTick) ? 1.0f
                : (ticksOnTarget >= rampMidTick) ? rampMidFraction
                : rampStartFraction;
            base = static_cast<int>(damage * fraction);
        }
        if (chargeThreshold > 0.0f && (chargeProgress >= chargeThreshold || chargeHasStuck)) {
            base = static_cast<int>(base * chargeMultiplier);
        }
        if (buffTicksRemaining > 0) {
            base = static_cast<int>(base * buffDamageMultiplier);
        }
        if (rangeFalloff && attackRange > 0.0f) {
            float distFraction = std::min(lastAttackDistance / attackRange, 1.0f);
            float rangeFactor = 1.0f - (1.0f - rangeFalloffMinFraction) * distFraction;
            base = static_cast<int>(base * rangeFactor);
        }
        if (rangeBandMaxDist > 0.0f && lastAttackDistance >= rangeBandMinDist && lastAttackDistance <= rangeBandMaxDist) {
            base = static_cast<int>(base * rangeBandDamageMultiplier);
        }
        if (currentHitIsBurst) {
            base = static_cast<int>(base * burstDamageMultiplier);
        }
        return (currentHitCount > 1 && !splitTargetsFullDamage) ? base / currentHitCount : base;
    }

    virtual void performAttack(Board& board, std::shared_ptr<Entity> target) = 0;

    // Called by direct-damage performAttack overrides as the damage lands;
    // ranged attacks hand onHitEffects to the Projectile instead. On-hit
    // effects apply only to a CombatEntity.
    void applyOnHitEffects(const std::shared_ptr<Entity>& target) const {
        if (onHitEffects.empty()) return;
        auto combatTarget = std::dynamic_pointer_cast<CombatEntity>(target);
        if (!combatTarget) return; // nothing to apply on-hit effects to
        for (const auto& effect : onHitEffects) {
            effect->apply(combatTarget);
        }
    }

    virtual void moveTowards(Board& board, const Vector2D& dest) {
        // Default: stationary.
        (void)board;
        (void)dest;
    }
};

// Splash: `dealt` to every other valid enemy within `radius` of `origin`, each
// with its own DamageDealtEvent. A free function because Projectile, not a
// CombatEntity, needs it too. No-op for radius <= 0.
inline void applySplashDamage(Board& board, const Vector2D& origin, float radius, int excludeId,
        int attackerId, int attackerTeam, int attackerCardId, int dealt) {
    if (radius <= 0.0f) return;
    for (const auto& entity : board.getEntities()) {
        if (entity->id == excludeId) continue; // already hit as the primary target
        if (entity->team == attackerTeam || !entity->isAlive() || !entity->isTargetable()) continue;
        if (origin.distanceTo(entity->position) > radius) continue;
        entity->takeDamage(dealt);
        board.statsEvents.notifyDamageDealt(
            { attackerId, attackerTeam, attackerCardId, entity->id, entity->cardId, entity->team, dealt, board.currentTick, entity->isTower() });
    }
}

// On-hit area pull (Evolved Valkyrie's Whirlwind Axe) toward `origin`.
// Buildings are never moved (pullToward). No-op for radius or distance <= 0.
inline void applyPullNearby(Board& board, const Vector2D& origin, float radius, float distance,
        int excludeId, int attackerTeam) {
    if (radius <= 0.0f || distance <= 0.0f) return;
    for (const auto& entity : board.getEntities()) {
        if (entity->id == excludeId) continue;
        if (entity->team == attackerTeam || !entity->isAlive() || !entity->isTargetable()) continue;
        if (origin.distanceTo(entity->position) > radius) continue;
        pullToward(*entity, origin, distance);
    }
}

// Taunt (Hero Knight): forces every valid enemy within `radius` onto
// `taunterId` for `ticks`. No-op for radius or ticks <= 0.
inline void applyTauntNearby(Board& board, const Vector2D& origin, float radius, int ticks,
        int taunterId, int taunterTeam) {
    if (radius <= 0.0f || ticks <= 0) return;
    for (const auto& entity : board.getEntities()) {
        if (entity->team == taunterTeam || !entity->isAlive() || !entity->isTargetable()) continue;
        if (origin.distanceTo(entity->position) > radius) continue;
        if (auto ce = std::dynamic_pointer_cast<CombatEntity>(entity)) {
            ce->forcedTargetEntityId = taunterId;
            ce->forcedTargetTicksRemaining = ticks;
        }
    }
}

// Piercing line (Bowler, Magic Archer): `dealt` to every valid enemy within
// `halfWidth` of the segment from `origin` (where the attacker fired) toward
// `aimPoint`, out to `range`. No-op for range or halfWidth <= 0.
inline void applyLineSplashDamage(Board& board, const Vector2D& origin, const Vector2D& aimPoint,
        float range, float halfWidth, int excludeId, int attackerId, int attackerTeam, int attackerCardId, int dealt) {
    if (range <= 0.0f || halfWidth <= 0.0f) return;
    float dx = aimPoint.x - origin.x;
    float dy = aimPoint.y - origin.y;
    float lineLen = std::sqrt(dx * dx + dy * dy);
    if (lineLen <= 0.01f) return; // no direction
    float ux = dx / lineLen, uy = dy / lineLen; // unit direction, origin -> aimPoint

    for (const auto& entity : board.getEntities()) {
        if (entity->id == excludeId) continue; // already hit as the primary target
        if (entity->team == attackerTeam || !entity->isAlive() || !entity->isTargetable()) continue;

        // Project onto the line, clamped to [0, range], then take the
        // perpendicular distance.
        float proj = (entity->position.x - origin.x) * ux + (entity->position.y - origin.y) * uy;
        if (proj < 0.0f) proj = 0.0f;
        if (proj > range) proj = range;
        float closestX = origin.x + ux * proj;
        float closestY = origin.y + uy * proj;
        float ddx = entity->position.x - closestX;
        float ddy = entity->position.y - closestY;
        if (std::sqrt(ddx * ddx + ddy * ddy) > halfWidth) continue;

        entity->takeDamage(dealt);
        board.statsEvents.notifyDamageDealt(
            { attackerId, attackerTeam, attackerCardId, entity->id, entity->cardId, entity->team, dealt, board.currentTick, entity->isTower() });
    }
}

// Damage buff on up to maxTargets of the closest same-team entities within
// radius (Rune Giant's enchant, Lumberjack's potion, Rage), excluding the
// source. No-op for radius or maxTargets <= 0.
inline std::vector<std::shared_ptr<CombatEntity>> applyAreaBuff(Board& board, const Vector2D& origin, float radius, int excludeId,
        int team, float multiplier, int durationTicks, int maxTargets) {
    if (radius <= 0.0f || maxTargets <= 0) return {};
    std::vector<std::shared_ptr<CombatEntity>> candidates;
    for (const auto& entity : board.getEntities()) {
        if (entity->id == excludeId) continue;
        if (entity->team != team || !entity->isAlive()) continue;
        if (origin.distanceTo(entity->position) > radius) continue;
        if (auto combatEntity = std::dynamic_pointer_cast<CombatEntity>(entity)) {
            candidates.push_back(combatEntity);
        }
    }
    std::sort(candidates.begin(), candidates.end(),
        [&origin](const std::shared_ptr<CombatEntity>& a, const std::shared_ptr<CombatEntity>& b) {
            return origin.distanceTo(a->position) < origin.distanceTo(b->position);
        });
    if (static_cast<int>(candidates.size()) > maxTargets) candidates.resize(maxTargets);
    for (const auto& c : candidates) c->applyBuff(multiplier, durationTicks);
    return candidates;
}

// Heal every same-team entity within radius by `amount` (Battle Healer),
// excluding the source. There is no generic max hp to cap against, so it can
// overheal. No-op for radius or amount <= 0.
inline void applyAreaHeal(Board& board, const Vector2D& origin, float radius, int excludeId,
        int team, int amount) {
    if (radius <= 0.0f || amount <= 0) return;
    for (const auto& entity : board.getEntities()) {
        if (entity->id == excludeId) continue;
        if (entity->team != team || !entity->isAlive()) continue;
        if (origin.distanceTo(entity->position) > radius) continue;
        entity->hp += amount;
    }
}
