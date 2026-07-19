#pragma once
#include "CardEntity.h"
#include "Board.h"
#include "OnHitEffect.h"
#include "DeathEffect.h"
#include "PeriodicEffect.h"
#include <memory>
#include <limits>
#include <vector>
#include <algorithm>

class CombatEntity;

// Forward-declared here, defined after the class (below) -- CombatEntity's
// own update() calls applyAreaBuff/applyAreaHeal directly (unlike
// applySplashDamage, which only ever gets called from leaf classes that
// include this whole header first), so the class body needs to see these
// signatures before they're fully defined.
inline void applyAreaBuff(Board& board, const Vector2D& origin, float radius, int excludeId,
    int team, float multiplier, int durationTicks, int maxTargets);
inline void applyAreaHeal(Board& board, const Vector2D& origin, float radius, int excludeId,
    int team, int amount);

class CombatEntity : public CardEntity {
protected:
    float attackRange;
    int damage;
    int attackCooldown;
    float currentCooldown;
    std::vector<std::shared_ptr<IOnHitEffect>> onHitEffects;

    // Ramp bookkeeping: which target this attacker has been locked onto, and
    // for how many consecutive ticks. Tracked unconditionally (cheap, two
    // ints) even for the vast majority of entities that never ramp, so the
    // logic lives in exactly one place instead of being opt-in duplicated.
    int currentTargetId = -1;
    int ticksOnTarget = 0;

    // How many targets the attack actually landed on this time (1 normally,
    // up to maxSplitTargets otherwise) -- set right before performAttack()
    // is invoked, purely so getCurrentDamage() can divide by it.
    int currentHitCount = 1;

public:
    // Freeze only ever means anything to something that attacks or moves
    // (cooldown/speed slowdown below), so it lives here rather than on
    // Entity -- AreaSpell and Projectile can never be frozen, since neither
    // is ever a valid findTarget() result (both are isTargetable() == false).
    int freezeTicks = 0;
    float freezeSlow = 1.0f;

    // Whether this attacker's findTarget() may pick a flying candidate.
    // Lives here (not Entity) because only things that attack care --
    // Troop and Building alike (Inferno Tower/Tesla hit air, Cannon/Bomb
    // Tower don't) -- and it's only ever read on `this`, never cast off a
    // generic candidate the way isFlying is (see Entity.h).
    bool targetsAir = false;

    // Fired once by onDeath() below (e.g. Golem spawning two Golemites).
    // Lives here, not Entity, for the same reason as onHitEffects: only
    // something that's a real combatant ever has one.
    std::shared_ptr<IDeathEffect> deathEffect;

    // Ramping damage (Inferno Tower): `damage` scales up the longer this
    // attacker stays locked onto the *same* target, reaching rampStartFraction
    // until rampMidTick, rampMidFraction until rampFullTick, and full damage
    // after -- reset by a target switch, losing the target, or being frozen
    // (matches the real "stun resets the charge" rule). rampFullTick == 0
    // (the default) disables ramping entirely: getCurrentDamage() is just
    // `damage`, unchanged, for every card that doesn't opt in.
    int rampMidTick = 0;
    int rampFullTick = 0;
    float rampStartFraction = 1.0f;
    float rampMidFraction = 1.0f;

    // Split-target attacks (Electro Wizard): instead of hitting only the
    // closest enemy, hits up to this many of the closest enemies at once,
    // each for damage / (however many were actually found this attack) --
    // full damage if only one target is in range, matching the real card.
    // 1 (the default) is the normal single-target case every other card uses.
    int maxSplitTargets = 1;

    // Splash damage (Wizard, Bowler, Valkyrie, ...): every regular attack
    // also hits every other enemy within this radius of the primary
    // target's position, for the same amount as the primary hit -- ground
    // and air alike, regardless of whether this attacker's own targetsAir
    // could normally reach flying units (an explosion doesn't care what
    // the thrower could aim at; matches AreaSpell's own non-ground-only
    // default). 0.0f (the default) is every card that doesn't opt in.
    // See applySplashDamage below -- a free function, not a method, since
    // Projectile needs it too and isn't a CombatEntity.
    float splashRadius = 0.0f;

    // Shield (Guards, Royal Recruits, Dark Prince, Cannon Cart): a second
    // HP pool that absorbs damage first, from any source (direct hit,
    // splash, spell) since it's implemented in takeDamage() itself, not
    // per-attacker. Doesn't regenerate. 0 (the default) is every card
    // without one.
    int shieldHp = 0;

    // Charge/dash bonus damage (Prince, Battle Ram, Ram Rider, Royal Hogs,
    // Bandit): once this attacker has moved at least chargeThreshold tiles
    // continuously without landing a hit, its next attack deals
    // chargeMultiplier x damage -- then chargeProgress resets to 0,
    // charged or not, same as the real game's "have to run it up again"
    // rule. chargeThreshold == 0.0f (the default) disables the mechanic
    // for every card that doesn't opt in.
    float chargeThreshold = 0.0f;
    float chargeMultiplier = 1.0f;
    float chargeProgress = 0.0f;

    // Enrage (Berserker): attack cooldown shortens (up to 2x speed at 0 hp)
    // and a small self-heal lands with every hit, scaling with how much of
    // enrageMaxHp is already gone. enrageMaxHp == 0 (the default) disables
    // the mechanic; set to the card's own starting hp to enable it (see
    // CardStats::withEnrage).
    int enrageMaxHp = 0;
    int enrageHealPerHit = 0;

    // Parry (Ronin): every parryIntervalTicks ticks, the next incoming hit
    // is fully negated instead of applying normally, then the timer
    // resets. The real card also reflects bonus damage back at the
    // attacker and only parries ground-melee hits (ranged/air/spells
    // bypass it) -- neither is modeled (takeDamage carries no attacker
    // identity to reflect at, and no melee-vs-ranged distinction exists
    // anywhere in this engine), so this is a simpler "occasionally shrugs
    // off an incoming hit entirely" approximation. parryIntervalTicks == 0
    // (the default) disables the mechanic; parryTicksUntilReady <= 0 means
    // ready to parry the next hit, right away at spawn.
    int parryIntervalTicks = 0;
    int parryTicksUntilReady = 0;

    // Hook (Fisherman): when the current target is beyond attackRange but
    // still within hookRange, instantly pulls it to just inside melee
    // range instead of walking toward it -- consumes this tick's attack
    // cooldown, so the actual damage lands on a later, normal-range hit.
    // hookRange == 0.0f (the default) disables the mechanic.
    float hookRange = 0.0f;

    // Invisibility (Royal Ghost, Suspicious Bush): untargetable except for
    // a brief window right after this attacker lands a hit, which reveals
    // it -- see isTargetable() and revealTicksAfterAttack below.
    // startsInvisible == false (the default) is every card without it.
    bool startsInvisible = false;
    int revealTicksAfterAttack = 0;
    int visibleTicksRemaining = 0;

    // Periodic spawning/effects while alive (Witch, Night Witch, Furnace,
    // Barbarian Hut, Goblin Hut, Tombstone, Goblin Drill): every
    // periodicIntervalTicks ticks, fires periodicEffect at this entity's
    // own position/team -- unrelated to deathEffect, which fires once, on
    // death, instead. periodicIntervalTicks == 0 (the default) disables
    // the mechanic; no timer runs and periodicEffect is never read.
    std::shared_ptr<IPeriodicEffect> periodicEffect;
    int periodicIntervalTicks = 0;
    int periodicTicksUntilNext = 0;

    // Temporary damage buff (Rage spell/potion, Rune Giant's ally
    // enchant): while buffTicksRemaining > 0, getCurrentDamage() is
    // multiplied by buffDamageMultiplier. Real Rage also boosts movement/
    // attack speed; only the damage part is modeled (speed lives on
    // Troop, a layer above CombatEntity -- not worth duplicating this
    // multiplier system there for one card's secondary effect).
    // buffTicksRemaining == 0 (the default) is every card unaffected.
    float buffDamageMultiplier = 1.0f;
    int buffTicksRemaining = 0;

    // Temporary damage-taken debuff (Mother Witch's curse): while
    // curseTicksRemaining > 0, incoming damage in takeDamage() is
    // multiplied by curseDamageTakenMultiplier before shield/parry see it.
    // curseTicksRemaining == 0 (the default) is every card unaffected.
    float curseDamageTakenMultiplier = 1.0f;
    int curseTicksRemaining = 0;

    // Ally aura on landed attacks (Rune Giant's every-Nth-attack buff,
    // Battle Healer's heal): fires the configured effect(s) at nearby
    // allies right after this attacker's own hit lands. auraEveryNAttacks
    // == 0 (the default) disables the buff aura; healAllyAmount == 0
    // disables the heal aura -- each opts in independently.
    float auraRadius = 0.0f;
    int auraMaxTargets = 1000000; // effectively "everyone in radius" unless a card sets a real cap
    int auraEveryNAttacks = 0;
    int attacksSinceAura = 0;
    float auraBuffMultiplier = 1.0f;
    int auraBuffDurationTicks = 0;
    int healAllyAmount = 0;

    // Kamikaze (Wall Breakers, the "Spirit" troops): dies immediately
    // after landing its one hit instead of surviving to attack
    // repeatedly. For ranged troops this fires the instant the shot is
    // launched (performAttack spawning the Projectile), not on the
    // projectile's later arrival -- the shooter vanishing slightly before
    // the real card's on-arrival timing, a minor simplification. false
    // (the default) is every other card here.
    bool dieAfterFirstHit = false;

    CombatEntity(int id, float x, float y, int hp, int team, char symbol,
        float attackRange, int damage, int attackCooldown)
        : CardEntity(id, x, y, hp, team, symbol),
        attackRange(attackRange), damage(damage),
        attackCooldown(attackCooldown), currentCooldown(0.0f) {}

    bool isTargetable() const override {
        return !startsInvisible || visibleTicksRemaining > 0;
    }

    void applyBuff(float multiplier, int ticks) {
        buffDamageMultiplier = multiplier;
        buffTicksRemaining = ticks;
    }

    void applyCurse(float damageTakenMultiplier, int ticks) {
        curseDamageTakenMultiplier = damageTakenMultiplier;
        curseTicksRemaining = ticks;
    }

    // Parry checked before shield: a parried hit is negated outright, not
    // absorbed by (and wasting) shield capacity. Shield absorbs first,
    // dollar-for-dollar, before any of this spills onto real hp -- matches
    // the real game's "shield breaks silently, no damage carries over"
    // rule (a hit bigger than the remaining shield only costs the excess,
    // not double-counted).
    void takeDamage(int amount) override {
        if (curseTicksRemaining > 0) {
            amount = static_cast<int>(amount * curseDamageTakenMultiplier);
        }
        if (parryIntervalTicks > 0 && parryTicksUntilReady <= 0) {
            parryTicksUntilReady = parryIntervalTicks;
            return; // fully negated
        }
        if (shieldHp > 0) {
            int absorbed = (shieldHp < amount) ? shieldHp : amount;
            shieldHp -= absorbed;
            amount -= absorbed;
        }
        if (amount > 0) Entity::takeDamage(amount);
    }

    void applyFreeze(int ticks, float slowFactor) {
        // Duration and strength are judged independently so a new freeze can
        // never leave the target better off than it already was: a shorter
        // but stronger slow no longer gets silently dropped just because a
        // longer, weaker one is already active.
        freezeTicks = std::max(freezeTicks, ticks);
        freezeSlow = std::min(freezeSlow, slowFactor);
    }

    // Composes extra behavior (e.g. freeze) onto every successful attack,
    // without needing a bespoke Entity subclass per effect combination.
    void addOnHitEffect(std::shared_ptr<IOnHitEffect> effect) {
        onHitEffects.push_back(std::move(effect));
    }

    void update(Board& board) override {
        // Captured before the decrement below so a freeze that's about to
        // expire this very tick still counts as "was frozen" for the ramp
        // reset -- matches the real "a stun resets the charge" rule for
        // every tick actually spent frozen, not all-but-the-last one.
        bool wasFrozen = freezeTicks > 0;

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

        if (periodicIntervalTicks > 0) {
            periodicTicksUntilNext--;
            if (periodicTicksUntilNext <= 0) {
                if (periodicEffect) periodicEffect->apply(board, position, team);
                periodicTicksUntilNext = periodicIntervalTicks;
            }
        }

        // Target-lock: once committed to a target, stay on it -- attacking
        // or chasing -- instead of re-picking "whoever's closest" every
        // tick. Matches the real game: a unit mid-fight doesn't get
        // distracted just because something else wandered closer. The lock
        // only breaks when the target itself becomes invalid (dies, etc.)
        // or leaves this attacker's effective range (e.g. pulled out by
        // Tornado, or knocked back) -- at that point a fresh closest-enemy
        // scan runs immediately, same tick, so nothing is left stuck
        // chasing a target it can no longer reach while ignoring whoever's
        // actually closest now.
        auto target = resolveCurrentTarget(board);
        if (target && position.distanceTo(target->position) > effectiveRangeTo(target)) {
            target = nullptr;
        }
        if (!target) {
            target = findTarget(board);
        }

        if (target) {
            if (wasFrozen || target->id != currentTargetId) {
                currentTargetId = target->id;
                ticksOnTarget = 0;
            } else {
                ticksOnTarget++;
            }

            float dist = position.distanceTo(target->position);
            float effectiveAttackRange = effectiveRangeTo(target);

            if (dist <= effectiveAttackRange) {
                if (currentCooldown == 0.0f) {
                    // Effects are applied by performAttack itself, not here,
                    // because *when* they should fire depends on *when* the
                    // damage actually lands: instantly for a direct hit, but
                    // only on arrival for an attack that spawns a projectile.
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
                    if (chargeThreshold > 0.0f) chargeProgress = 0.0f;
                    if (startsInvisible) visibleTicksRemaining = revealTicksAfterAttack;
                    if (enrageMaxHp > 0) {
                        float hpFraction = static_cast<float>(hp) / static_cast<float>(enrageMaxHp);
                        currentCooldown *= (0.5f + 0.5f * hpFraction); // up to 2x attack speed at 0 hp
                        if (enrageHealPerHit > 0 && hp < enrageMaxHp) {
                            hp = (hp + enrageHealPerHit < enrageMaxHp) ? hp + enrageHealPerHit : enrageMaxHp;
                        }
                    }
                    // Ally aura on landed attacks (Rune Giant's every-Nth
                    // buff, Battle Healer's heal) -- see CombatEntity's own
                    // aura* fields above.
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
                    if (dieAfterFirstHit) hp = 0;
                }
            } else if (hookRange > 0.0f && dist <= hookRange && currentCooldown == 0.0f) {
                // Hook: instantly pull the target to just inside melee
                // range instead of walking toward it. No damage lands this
                // tick -- the real hit happens on a later, normal-range
                // attack once it arrives, consistent with the real card
                // (hook first, melee second).
                pullToward(*target, position, dist - effectiveAttackRange + 0.1f);
                currentCooldown = static_cast<float>(attackCooldown);
            } else {
                Vector2D beforeMove = position;
                moveTowards(board, target->position);
                if (chargeThreshold > 0.0f) chargeProgress += beforeMove.distanceTo(position);
            }
        } else {
            currentTargetId = -1;
            ticksOnTarget = 0;
        }

        clampPosition(board);
    }

    void onDeath(Board& board) override {
        if (deathEffect) deathEffect->apply(board, position, team);
    }

protected:
    // Entity, not CombatEntity: targeting itself doesn't care about freeze
    // or on-hit effects, and every other consumer of findTarget's result
    // (range math, movement) only ever needs Entity's own surface. Keeping
    // this Entity-typed means the only place that needs to know "is this
    // actually a CombatEntity" is applyOnHitEffects below, where it's
    // genuinely required -- not the whole targeting system.
    bool isValidTarget(const std::shared_ptr<Entity>& entity) const {
        return entity && entity->team != this->team && entity->isAlive() && entity->isTargetable()
            && entity->id != this->id && (!entity->isFlying || targetsAir);
    }

    float effectiveRangeTo(const std::shared_ptr<Entity>& target) const {
        float targetRadius = target->getCollisionRadius();
        if (targetRadius <= 0.0f) targetRadius = Entity::IMPLICIT_TROOP_RADIUS;
        float myRadius = this->getCollisionRadius();
        if (myRadius <= 0.0f) myRadius = Entity::IMPLICIT_TROOP_RADIUS;
        return attackRange + myRadius + targetRadius;
    }

    // Re-validates the currently-locked target (by id) rather than running
    // a full closest-enemy scan -- Board has no id index, so this is still
    // a linear pass, but it's the one that lets a locked-on attacker keep
    // its target instead of findTarget() picking a new "closest" every
    // tick. Returns nullptr if there's no lock, or the locked entity no
    // longer exists / is no longer a legal target (dead, no longer
    // targetable, etc).
    std::shared_ptr<Entity> resolveCurrentTarget(Board& board) const {
        if (currentTargetId < 0) return nullptr;
        for (const auto& entity : board.getEntities()) {
            if (entity->id == currentTargetId) {
                return isValidTarget(entity) ? entity : nullptr;
            }
        }
        return nullptr;
    }

    virtual std::shared_ptr<Entity> findTarget(Board& board) const {
        std::shared_ptr<Entity> closestTarget = nullptr;
        float minDistance = std::numeric_limits<float>::max();

        for (const auto& entity : board.getEntities()) {
            if (isValidTarget(entity)) {
                float dist = position.distanceTo(entity->position);
                if (dist < minDistance) {
                    minDistance = dist;
                    closestTarget = entity;
                }
            }
        }
        return closestTarget;
    }

    // Only called when maxSplitTargets > 1 (Electro Wizard). Reuses
    // findTarget's own eligibility filter, just keeping the N closest
    // instead of only the closest one.
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

    // Ramped, charge-boosted, then split across however many targets this
    // attack actually landed on. All three default to no-ops
    // (rampFullTick == 0, chargeThreshold == 0.0f, currentHitCount == 1),
    // so this returns `damage` unchanged for every card that doesn't opt
    // into any of them.
    int getCurrentDamage() const {
        int base = damage;
        if (rampFullTick > 0) {
            float fraction = (ticksOnTarget >= rampFullTick) ? 1.0f
                : (ticksOnTarget >= rampMidTick) ? rampMidFraction
                : rampStartFraction;
            base = static_cast<int>(damage * fraction);
        }
        if (chargeThreshold > 0.0f && chargeProgress >= chargeThreshold) {
            base = static_cast<int>(base * chargeMultiplier);
        }
        if (buffTicksRemaining > 0) {
            base = static_cast<int>(base * buffDamageMultiplier);
        }
        return currentHitCount > 1 ? base / currentHitCount : base;
    }

    virtual void performAttack(Board& board, std::shared_ptr<Entity> target) = 0;

    // Called by a direct-damage performAttack override at the exact moment
    // its damage lands. Ranged attacks don't call this -- they hand
    // onHitEffects to the Projectile instead, so effects land with the hit.
    // On-hit effects (freeze, etc.) only ever mean something against a
    // CombatEntity, so the cast happens here, once, rather than forcing
    // every target-typed signature in the codebase to narrow to
    // CombatEntity just to serve this one specific need.
    void applyOnHitEffects(const std::shared_ptr<Entity>& target) const {
        if (onHitEffects.empty()) return;
        auto combatTarget = std::dynamic_pointer_cast<CombatEntity>(target);
        if (!combatTarget) return; // not something on-hit effects can apply to
        for (const auto& effect : onHitEffects) {
            effect->apply(combatTarget);
        }
    }

    virtual void moveTowards(Board& board, const Vector2D& dest) {
        // Default: stationary entities don't move
    }
};

// Splash damage: applies `dealt` to every other valid enemy within `radius`
// of `origin`, on `attackerTeam`'s behalf, each getting its own
// DamageDealtEvent. A free function rather than a CombatEntity method,
// since direct-damage attackers (MeleeTroop/BuildingTargeter/Building) and
// Projectile (ranged attacks, arriving after the shooter's own
// performAttack() already returned) both need it, and Projectile isn't a
// CombatEntity. No-op when radius <= 0 -- every card that doesn't opt into
// splash, the default.
inline void applySplashDamage(Board& board, const Vector2D& origin, float radius, int excludeId,
        int attackerId, int attackerTeam, int attackerCardId, int dealt) {
    if (radius <= 0.0f) return;
    for (const auto& entity : board.getEntities()) {
        if (entity->id == excludeId) continue; // already damaged as the primary target
        if (entity->team == attackerTeam || !entity->isAlive() || !entity->isTargetable()) continue;
        if (origin.distanceTo(entity->position) > radius) continue;
        entity->takeDamage(dealt);
        board.statsEvents.notifyDamageDealt(
            { attackerId, attackerTeam, attackerCardId, entity->id, entity->cardId, entity->team, dealt, board.currentTick });
    }
}

// Ally buff: applies a temporary damage buff to up to maxTargets of the
// closest same-team CombatEntity within radius of origin (Rune Giant's
// per-3rd-attack enchant, Lumberjack's death-potion, Rage). excludeId
// skips the source itself (a buffing unit doesn't buff itself). No-op
// when radius <= 0 or maxTargets <= 0.
inline void applyAreaBuff(Board& board, const Vector2D& origin, float radius, int excludeId,
        int team, float multiplier, int durationTicks, int maxTargets) {
    if (radius <= 0.0f || maxTargets <= 0) return;
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
}

// Ally heal (Battle Healer): heals every same-team Entity within radius of
// origin by `amount`. excludeId skips the source itself. Entity has no
// generic "max hp" concept to cap against (only Building tracks one, for
// decay), so this can overheal past an ally's original spawn hp -- not
// modeled, a minor simplification. No-op when radius <= 0 or amount <= 0.
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
