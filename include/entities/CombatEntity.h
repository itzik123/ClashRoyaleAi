#pragma once
#include "CardEntity.h"
#include "Board.h"
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

// Forward-declared here, defined after the class (below) -- CombatEntity's
// own update() calls applyAreaBuff/applyAreaHeal directly (unlike
// applySplashDamage, which only ever gets called from leaf classes that
// include this whole header first), so the class body needs to see these
// signatures before they're fully defined.
// Returns whichever entities actually got buffed -- callers that need to
// do something ELSE to the same targets (Royal Chef's Tower Troop also
// bumps hp on whoever it just buffed) can reuse the exact selection
// instead of re-scanning the board. Every existing caller ignores the
// return value, so this is backward compatible.
inline std::vector<std::shared_ptr<CombatEntity>> applyAreaBuff(Board& board, const Vector2D& origin, float radius, int excludeId,
    int team, float multiplier, int durationTicks, int maxTargets);
inline void applyAreaHeal(Board& board, const Vector2D& origin, float radius, int excludeId,
    int team, int amount);
// Forward-declared for the same reason as the two above: Mega Knight's
// jump (see update() below) calls this directly from inside the class
// body, before its own definition later in this file is visible.
inline void applySplashDamage(Board& board, const Vector2D& origin, float radius, int excludeId,
    int attackerId, int attackerTeam, int attackerCardId, int dealt);
// Forward-declared for the same reason: Evolved Valkyrie's on-hit pull
// (see update() below) calls this directly from inside the class body.
inline void applyPullNearby(Board& board, const Vector2D& origin, float radius, float distance,
    int excludeId, int attackerTeam);
// Forward-declared for the same reason: Hero Knight's Triumphant Taunt
// (see HeroKnightTauntEffect) calls this directly, and it's defined later
// in this file alongside applyPullNearby/applySplashDamage.
inline void applyTauntNearby(Board& board, const Vector2D& origin, float radius, int ticks,
    int taunterId, int taunterTeam);

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
    // Ticks elapsed since the last landed hit (any target) -- also tracked
    // unconditionally, same reasoning. Only consulted when
    // rampGracePeriodTicks > 0 (Inferno Dragon Evolution): normally
    // (rampGracePeriodTicks == 0) a target switch or losing the target
    // entirely resets ticksOnTarget straight to 0, same as always. With a
    // grace period configured, that reset is deferred -- the current ramp
    // stage is preserved across a target switch, or through a gap with no
    // target at all, for up to rampGracePeriodTicks, only actually
    // resetting once this counter crosses that threshold without a hit
    // landing. A freeze/stun still resets immediately regardless (see
    // wasFrozen below) -- the grace period only ever softens the "lost my
    // target" case, never the "got stunned" one.
    int ticksSinceLastHit = 0;

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

    // Poison-style damage-over-time mark (Dart Goblin/Firecracker
    // Evolutions) -- see applyDot/PoisonOnHit. dotTicksRemaining == 0
    // (the default) means no active mark; ticks down independently of
    // freeze/curse.
    int dotDamagePerTick = 0;
    int dotTicksRemaining = 0;
    int dotTickInterval = 0;
    int dotTicksUntilNextDamage = 0;

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

    // 4th ramp stage (Inferno Dragon Evolution only): a rarely-reached
    // stage beyond rampFullTick, dealing rampStage4Fraction * damage
    // instead of the usual full damage. rampStage4Tick == 0 (the default)
    // disables it, leaving getCurrentDamage's 3-stage schedule above
    // completely unchanged for every other ramping card (Inferno Tower,
    // regular Inferno Dragon, Mighty Miner).
    int rampStage4Tick = 0;
    float rampStage4Fraction = 1.0f;

    // Grace period before a lost/switched target resets the ramp (Inferno
    // Dragon Evolution only) -- see ticksSinceLastHit above for the full
    // explanation. 0 (the default) means an instant reset, exactly like
    // every other ramping card today.
    int rampGracePeriodTicks = 0;

    // Split-target attacks (Electro Wizard): instead of hitting only the
    // closest enemy, hits up to this many of the closest enemies at once,
    // each for damage / (however many were actually found this attack) --
    // full damage if only one target is in range, matching the real card.
    // 1 (the default) is the normal single-target case every other card uses.
    int maxSplitTargets = 1;
    // Temporary split-target window (Hero Magic Archer's Triple Threat):
    // bumps maxSplitTargets up for a fixed duration, then restores it --
    // reuses the Electro Wizard machinery above as a documented
    // approximation of "fires 2 extra arrows" (this engine divides damage
    // across split targets rather than firing genuinely independent
    // projectiles). baseMaxSplitTargets captures whatever maxSplitTargets
    // was at the moment the window opened (always 1 for every card that
    // uses this, since no card both split-targets permanently AND has this
    // ability), restored once temporarySplitTargetsTicksRemaining reaches 0.
    // 0 (the default) is every card without an active window.
    int temporarySplitTargetsTicksRemaining = 0;
    int baseMaxSplitTargets = 1;

    // Electro Dragon's chain: unlike Electro Wizard's split (which divides
    // `damage` across however many targets it hit), each chained target
    // takes the FULL damage value independently. false (the default)
    // keeps every other split-target card's existing divide-by-hitcount
    // behavior unchanged.
    bool splitTargetsFullDamage = false;

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
    // Fixed-duration shield expiry (Hero Knight's Triumphant Taunt: the
    // shield lasts exactly 5s even if never fully depleted by damage) --
    // distinct from shieldHp's own permanent variant above, which never
    // expires on its own. 0 (the default) is every card using the
    // permanent shieldHp above unaffected -- update() only zeroes shieldHp
    // once this counts down to 0, never touching it otherwise.
    int shieldExpiresTicksRemaining = 0;

    // Forced retarget (Hero Knight's Triumphant Taunt): while > 0, this
    // entity's update() is forced onto forcedTargetEntityId instead of its
    // own normal resolveCurrentTarget()/findTarget() resolution -- see
    // applyTauntNearby below and update()'s own targeting block. 0 (the
    // default) is every card without one, whose targeting is entirely
    // unaffected.
    int forcedTargetEntityId = -1;
    int forcedTargetTicksRemaining = 0;

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

    // Sticky charge (Evolved Battle Ram): unlike the "have to run it up
    // again" reset above, once this attacker has reached chargeThreshold
    // ONE time, chargeMultiplier keeps applying to every hit for the rest
    // of its life instead of resetting -- "constantly ramming into its
    // target... dealing double damage for every connection made."
    // chargeIsSticky == false (the default) is every other charging card
    // (Prince, base Battle Ram, Ram Rider, Royal Hogs, Bandit), completely
    // unaffected. chargeHasStuck is internal bookkeeping, not
    // CardStats-configurable.
    bool chargeIsSticky = false;
    bool chargeHasStuck = false;

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

    // Burst-on-Nth-attack (Dagger Duchess/Evolution burst mechanics): every
    // burstEveryNAttacks-th landed hit deals burstDamageMultiplier extra
    // damage to itself, instead of buffing allies the way the aura above
    // does. currentHitIsBurst is transient, set right before performAttack
    // fires and read back by getCurrentDamage() -- same idiom as
    // lastAttackDistance/currentHitCount below.
    int burstEveryNAttacks = 0;
    float burstDamageMultiplier = 1.0f;
    int attacksSinceBurst = 0;
    bool currentHitIsBurst = false;

    // Composable extra behavior when this entity actually TAKES damage
    // (mirrors onHitEffects, which fires for the attacker that LANDS a
    // hit) -- e.g. an evolution that buffs itself when struck. nullptr
    // (the default) is every card without one.
    std::shared_ptr<IOnDamageTakenEffect> onDamageTakenEffect;

    // Self-heal on landing a hit (Evolved Bats) -- healOnHitMaxHp is an
    // absolute cap, not a multiplier; 0 (the default) disables it.
    int healOnHitAmount = 0;
    int healOnHitMaxHp = 0;

    // Self-spawn on landing a hit, capped inside the effect itself
    // (Evolved Skeletons) -- see CappedSpawnOnHitEffect. nullptr (the
    // default) is every card without one.
    std::shared_ptr<IPeriodicEffect> onHitSpawnEffect;

    // Pull nearby enemies toward self on landing a hit (Evolved
    // Valkyrie's Whirlwind Axe) -- see applyPullNearby below.
    // onHitPullRadius == 0.0f (the default) disables it. Buildings/Towers
    // are never actually displaced by the pull (enforced inside
    // pullToward itself, Entity.h), but onHitPullDamage -- a separate,
    // typically-low amount applied via applySplashDamage -- still reaches
    // everyone in radius, matching the sourced "including Crown Towers"
    // wording; only the physical pull exempts them.
    float onHitPullRadius = 0.0f;
    float onHitPullDistance = 0.0f;
    int onHitPullDamage = 0;

    // Kamikaze (Wall Breakers, the "Spirit" troops): dies immediately
    // after landing its one hit instead of surviving to attack
    // repeatedly. For ranged troops this fires the instant the shot is
    // launched (performAttack spawning the Projectile), not on the
    // projectile's later arrival -- the shooter vanishing slightly before
    // the real card's on-arrival timing, a minor simplification. false
    // (the default) is every other card here.
    bool dieAfterFirstHit = false;

    // Dash invulnerability (Bandit): immune to all damage while charging
    // in for a hit. The real card is only invulnerable during a short
    // (~0.8s) dash burst once a target is in range; this engine has no
    // separate "dash" state from ordinary charge-buildup movement, so
    // it's approximated as invulnerable once past the halfway point of
    // closing the charge distance (chargeProgress >= half of
    // chargeThreshold) -- a longer window than the real 0.8s, but the
    // same spirit ("hard to punish mid-charge"). false (the default) is
    // every card without a charge, or with one but no invulnerability.
    bool chargeGrantsInvulnerability = false;

    // Reset (not just pause) the attack cooldown on being frozen/stunned
    // (Sparky: a Zap mid-charge makes her start her whole 4s wind-up over
    // instead of merely slowing it). false (the default) is every other
    // card, which only gets the normal freezeSlow cooldown-drain-rate
    // treatment in update() below.
    bool resetCooldownOnFreeze = false;

    // Recoil after landing an attack (Firecracker: kicks herself backward
    // away from her own target). recoilDistance == 0.0f (the default)
    // is every card without one. See Entity.h's pushAway.
    float recoilDistance = 0.0f;

    // Minimum attack range / "blind spot" (Mortar): can't hit anything
    // closer than this, on top of the normal `attackRange` maximum --
    // real game leaves the attacker just standing idle against a target
    // that's ducked inside the blind spot instead of retargeting, but
    // this engine's targeting has no notion of "valid target, but
    // currently unreachable" separate from "not a valid target at all",
    // so a candidate inside the blind spot is filtered out of
    // findTarget() entirely, same as if it were on the wrong team --
    // this attacker just finds the next-closest legal target instead of
    // freezing up. 0.0f (the default) is every card without one.
    float minAttackRange = 0.0f;

    // Sight/aggro range (distinct from attackRange -- see findTarget()
    // below): how far this attacker can detect an enemy troop/building to
    // go fight instead of just heading for the enemy tower. Sourced per-
    // card (a Clash Royale community stats breakdown of exactly this,
    // not shown on the in-game card info screens); 5.5 tiles is that
    // source's own stated "main standard for most cards", used here as
    // the default for every card not individually called out with its
    // own value. For buildings this is generally the same number as
    // their own attackRange (buildings can't chase, so sight beyond
    // attack range would never actually matter) but is still set
    // per-card from the same source rather than derived from
    // attackRange, since the two aren't always exactly equal in the
    // sourced data.
    float sightRange = 5.5f;

    // HP-threshold transform (Cannon Cart: at <=50% hp, permanently
    // grounds itself and gets a fixed lifespan before self-destructing).
    // transformAtHpFraction == 0.0f (the default) disables the mechanic;
    // transformCheckMaxHp is the hp this attacker spawned with (read back
    // by CardStats::withHpTransform, same "reads back hp already set"
    // idiom as withEnrage) -- the fraction is computed against that, not
    // against whatever hp happens to be at any later moment. Modeled by
    // reusing applyFreeze(ticks, 0.0f) to zero out movement for the
    // transform's duration (see update() below) rather than adding a
    // second, CombatEntity-level "speed" concept -- Troop already has its
    // own speed field one layer up, and a full 0.0 freeze already means
    // "can't move" exactly like the real transform's grounding.
    float transformAtHpFraction = 0.0f;
    int transformCheckMaxHp = 0;
    int transformLifetimeTicks = 0;
    bool transformBecomesStationary = false;
    bool hasTransformed = false;
    int transformTicksRemaining = 0;

    // HP-threshold transform, archetype-swap variant (Goblin Demolisher:
    // becomes a fundamentally different unit -- ranged squad turns into a
    // melee building-only kamikaze -- not just "the same stats, grounded"
    // like Cannon Cart above). This engine has no way to change an
    // already-spawned entity's C++ type in place, so it's modeled by
    // killing this entity outright the instant the threshold is crossed
    // (see update() below) and letting the ordinary deathEffect/
    // SpawnOnDeath machinery -- already used everywhere else for
    // Golem/Golemites, Lava Hound/Pups, etc. -- spawn the transformed
    // form as a normal, independent child entity. false (the default) is
    // every other transform-capable card, which uses
    // transformBecomesStationary instead.
    bool transformKillsSelf = false;
    // Fired only by the transformKillsSelf self-kill itself (see update()
    // below) -- deliberately a SEPARATE slot from the ordinary
    // `deathEffect` above, not a reuse of it. A genuine combat death (a
    // hit big enough to skip straight past the transform threshold to 0
    // hp in one blow, never triggering the transform check at all) must
    // NOT also spawn the transformed form -- only an actual
    // threshold-triggered transform should.
    std::shared_ptr<IDeathEffect> transformDeathEffect;

    // Periodic jump (Mega Knight): once a ground target is between
    // jumpMinRange and jumpMaxRange away (farther than normal
    // attackRange), instantly closes the distance -- reusing the exact
    // same pullToward primitive as Fisherman's hook, just pulling this
    // attacker toward the target instead of the other way around -- and
    // lands a boosted, splashy hit immediately instead of spending
    // several ticks walking in. jumpMaxRange == 0.0f (the default)
    // disables the mechanic for every card without one.
    float jumpMinRange = 0.0f;
    float jumpMaxRange = 0.0f;
    float jumpDamageMultiplier = 1.0f;
    float jumpSplashRadius = 0.0f;

    // Piercing-line hit (Bowler, Magic Archer): instead of a circular
    // splash around the primary target, hits everyone within splashRadius
    // of the straight line from this attacker's own position toward the
    // target, out to lineSplashRange total distance -- splashRadius is
    // reused as the line's half-width rather than adding a whole separate
    // field, matching this engine's existing habit of dual-purposing one
    // field across a card's single active mode (see auraRadius above).
    // lineSplash == false (the default) is every other card, which keeps
    // the ordinary circular applySplashDamage behavior.
    bool lineSplash = false;
    float lineSplashRange = 0.0f;

    // Range-based damage falloff (Hunter): unlike every other numeric
    // field on this class, this one is NOT modeling a sourced mechanic --
    // the wiki research explicitly found no documented falloff formula
    // (the real effect is 10 fixed-damage pellets in a random spread,
    // fewer of which statistically land at range; no angle/probability
    // curve is published anywhere). This is a deliberately invented,
    // clearly-labeled approximation of the *qualitative* behavior
    // ("weaker at range") via a deterministic linear scale instead --
    // kept deterministic on purpose, matching every other mechanic in
    // this engine (including the real game's OTHER randomized cards,
    // e.g. Bowler's knockback, all modeled without dice rolls), since
    // combat here needs to stay reproducible for both tests and RL
    // training. rangeFalloff == false (the default) is every other card.
    bool rangeFalloff = false;
    float rangeFalloffMinFraction = 1.0f; // damage fraction at max range

    // Bonus damage within a specific distance band (Archers' Power Shot:
    // +50% at 4-6 tiles; Executioner's Axe Smash: +75% at <= 3.5 tiles) --
    // distinct from rangeFalloff above (a continuous scale-down across the
    // whole range), this is a flat multiplier that only applies while
    // lastAttackDistance falls within [rangeBandMinDist, rangeBandMaxDist].
    // rangeBandMaxDist == 0.0f (the default) disables it for every card
    // that doesn't opt in.
    float rangeBandMinDist = 0.0f;
    float rangeBandMaxDist = 0.0f;
    float rangeBandDamageMultiplier = 1.0f;

    // Distance to the target at the moment of the most recent attack --
    // set right alongside currentHitCount, just before performAttack(),
    // purely so getCurrentDamage() (a const method with no target of its
    // own) has something to scale rangeFalloff against.
    float lastAttackDistance = 0.0f;

    // Champion-only activated ability (Mighty Miner's "Explosive Escape",
    // and any future Champion). isChampion itself gates no combat behavior
    // on its own -- Champions fight exactly like an ordinary troop of
    // their archetype (see CardStats::isChampion's own comment for why
    // this isn't a 7th Archetype); it exists purely so
    // GameManager::activateChampionAbility can find "my one deployed
    // Champion" on the board by scanning for (team, isAlive(), isChampion)
    // instead of a fragile cardId allowlist. false (the default) is every
    // non-Champion card.
    bool isChampion = false;
    // Hero marker (see CardStats::isHero's own comment) -- every existing
    // Champion-slot consumer (PlayerState::seedSlotState, GameManager::
    // playCard's tracking hook/Mirror-block) checks `isChampion || isHero`,
    // so a Hero shares the exact same per-slot ability-tracking/activation
    // path as a Champion without any of that machinery needing to change.
    // false (the default) is every non-Hero card, including all 8 Champions.
    bool isHero = false;
    // In-battle elixir cost of activating this ability -- separate from
    // CardStats::cost (the up-front deploy cost already spent placing this
    // entity on the board), charged again on every activation by
    // GameManager::activateChampionAbility. 0.0f (the default, alongside
    // abilityEffect == nullptr) means "no activated ability" -- every
    // non-Champion card.
    float abilityElixirCost = 0.0f;
    // Cooldown between activations, and how many ticks remain before the
    // next one is allowed. abilityCooldownRemaining starts at 0 -- ready
    // immediately at deploy, matching the real game's Champions (no forced
    // wait before the first use) -- and is reset to abilityCooldownTicks
    // every time activateAbility() actually fires.
    int abilityCooldownTicks = 0;
    int abilityCooldownRemaining = 0;
    // What the ability actually does (Mighty Miner: teleport + delayed bomb
    // -- see MightyMinerEscapeEffect in core/, following the same
    // "concrete effects live in core/" convention as SpawnOnDeath/
    // PeriodicSpawnEffect). nullptr (the default) is every card without
    // one -- activateAbility() is always a no-op in that case, regardless
    // of cooldown.
    std::shared_ptr<IAbilityEffect> abilityEffect;
    // -1 (the default) means unlimited activations, gated only by cooldown/
    // elixir like every other Champion. 0 or positive is a hard cap on how
    // many times activateAbility() can ever fire (Boss Bandit: 2 uses per
    // deployment, no more once exhausted regardless of cooldown/elixir).
    int abilityUsesRemaining = -1;

    // Soul collection (Skeleton King's Soul Summoning): counts nearby
    // deaths -- ally or enemy alike, matching the real card's "a troop
    // dies in his presence" wording -- via onNearbyDeath below, up to
    // maxSouls. soulCollectionRadius == 0.0f (the default) is every
    // non-soul-collecting card; onNearbyDeath is simply never relevant for
    // them (no filter needed there beyond the radius check itself).
    float soulCollectionRadius = 0.0f;
    int soulCount = 0;
    int maxSouls = 0;

    // Temporary invisibility + attack-speed buff (Archer Queen's Cloaking
    // Cape, Boss Bandit's Getaway Grenade) -- a fixed-duration window,
    // unlike startsInvisible/revealTicksAfterAttack above (which reveals
    // on landing a hit, not after a timer). The real cards' accompanying
    // movement-speed change isn't modeled -- speed lives on Troop, a layer
    // above CombatEntity, same documented gap as Rage's own movement-speed
    // component elsewhere in this codebase.
    int temporaryInvisibilityTicksRemaining = 0;
    float temporaryHitSpeedMultiplier = 1.0f;

    // Self-haste on landing a hit, refreshing on every subsequent hit
    // (Evolved Barbarians' Blade Rage: +35% attack speed for 3s, timer
    // resets while they keep attacking) -- deliberately separate from
    // temporaryInvisibilityTicksRemaining/temporaryHitSpeedMultiplier
    // above: those are champion-ability-activation-only (see
    // CardFactories::applyCardMetadata's own comment) and carry an
    // invisibility side effect this needs to avoid. selfHasteDurationTicks
    // == 0 (the default) disables it; the real card's accompanying
    // movement-speed component isn't modeled, same documented gap as
    // Rage/Baby Dragon Evolution elsewhere in this file.
    int selfHasteDurationTicks = 0;
    float selfHasteCooldownMultiplier = 1.0f;
    int selfHasteTicksRemaining = 0;

    // Temporary flight (Hero Wizard's Fiery Flight): isFlying itself lives
    // on Entity (see its own comment there) and is already read live every
    // tick by isValidTarget/Troop::moveTowards/Board's collision grouping,
    // so toggling it at runtime needs no new field of its own -- this timer
    // just reverts isFlying=false once the window ends. Does NOT also
    // toggle ignoresRiver (fixed at spawn, CardFactories::
    // shouldIgnoreRiver) -- an accepted gap, same as the real ability not
    // relocating her across the map either. 0 (the default) is every card
    // without a temporary flight window.
    int temporaryFlightTicksRemaining = 0;
    // On-hit tornado pulse while the flight window is active (Hero
    // Wizard's Fiery Flight: fireballs gain their own damaging, pulling
    // tornado) -- see update()'s attack-landing block. Centered on the
    // TARGET's position, unlike Evolved Valkyrie's onHitPullRadius above
    // (centered on self) -- this accompanies a ranged hit landing on the
    // target, not a melee spin around the caster. 0 (the default) is every
    // card without one.
    int flightPulseTicksRemaining = 0;
    float flightPulseRadius = 0.0f;
    int flightPulseDamage = 0;
    float flightPulsePullDistance = 0.0f;

    // Hit-speed ramp while locked onto the same target (Little Prince):
    // unlike rampMidTick/rampFullTick above (which ramp DAMAGE while
    // attackCooldown stays fixed, for Inferno Dragon/Tower/Mighty Miner),
    // this ramps the COOLDOWN itself down -- he fires faster, not harder,
    // the longer he stays locked on. hitSpeedRampFullTick == 0 (the
    // default) disables it for every other card, matching rampFullTick's
    // own "0 disables" convention.
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

    // Soul collection (Skeleton King) -- see soulCollectionRadius's own
    // comment. Counts ANY death within radius, ally or enemy alike,
    // matching the real card's wording.
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

    // Activated ability (Champion-only). Fires abilityEffect and resets the
    // cooldown, but only when one is actually configured and off cooldown
    // -- elixir affordability is GameManager's concern (mirrors playCard's
    // own split: CombatEntity/CardFactories never touch PlayerState), so
    // the caller (GameManager::activateChampionAbility) must confirm and
    // deduct elixir BEFORE calling this; this method only ever gates on
    // cooldown/configuration. Returns whether it actually fired.
    bool activateAbility(Board& board) {
        if (!abilityEffect || abilityCooldownRemaining > 0) return false;
        if (abilityUsesRemaining == 0) return false; // exhausted (Boss Bandit-style limited uses)
        abilityEffect->apply(board, *this);
        abilityCooldownRemaining = abilityCooldownTicks;
        if (abilityUsesRemaining > 0) abilityUsesRemaining--;
        return true;
    }

    // Deploy delay (X-Bow: ~3.5s slow lock-on before its first shot,
    // instead of every other card's immediately-ready-to-fire default).
    // Called once, right after spawn, by whichever CardFactories function
    // built this entity -- currentCooldown is otherwise protected, so
    // this is the one seam that lets data-driven setup seed it without
    // exposing the field itself.
    void seedCooldown(int ticks) {
        currentCooldown = static_cast<float>(ticks);
    }

    // Parry checked before shield: a parried hit is negated outright, not
    // absorbed by (and wasting) shield capacity. Shield absorbs first,
    // dollar-for-dollar, before any of this spills onto real hp -- matches
    // the real game's "shield breaks silently, no damage carries over"
    // rule (a hit bigger than the remaining shield only costs the excess,
    // not double-counted).
    void takeDamage(int amount) override {
        if (chargeGrantsInvulnerability && chargeThreshold > 0.0f
            && chargeProgress >= chargeThreshold * 0.5f) {
            return; // mid-dash: fully immune
        }
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
        if (amount > 0) {
            Entity::takeDamage(amount);
            if (onDamageTakenEffect) onDamageTakenEffect->apply(*this);
        }
    }

    // Refreshes the mark outright (unlike applyFreeze's independent-judge
    // idiom) -- matches applyBuff/applyCurse's simpler "latest application
    // wins" behavior, the more common idiom for this kind of timed mark
    // in this codebase.
    void applyDot(int damagePerTick, int totalTicks, int tickInterval) {
        dotDamagePerTick = damagePerTick;
        dotTicksRemaining = totalTicks;
        dotTickInterval = tickInterval;
        dotTicksUntilNextDamage = tickInterval;
    }

    void applyFreeze(int ticks, float slowFactor) {
        // Duration and strength are judged independently so a new freeze can
        // never leave the target better off than it already was: a shorter
        // but stronger slow no longer gets silently dropped just because a
        // longer, weaker one is already active.
        freezeTicks = std::max(freezeTicks, ticks);
        freezeSlow = std::min(freezeSlow, slowFactor);
        if (resetCooldownOnFreeze && ticks > 0) {
            currentCooldown = static_cast<float>(attackCooldown);
        }
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
        ticksSinceLastHit++; // zeroed below the moment a hit actually lands this tick

        if (transformAtHpFraction > 0.0f && !hasTransformed && transformCheckMaxHp > 0
            && static_cast<float>(hp) / static_cast<float>(transformCheckMaxHp) <= transformAtHpFraction) {
            hasTransformed = true;
            if (transformKillsSelf) {
                hp = 0;
                if (transformDeathEffect) transformDeathEffect->apply(board, position, team);
                // Dead this tick -- skip the rest of update() (targeting,
                // movement, attack) entirely. Board::cleanDeadEntities()
                // still runs its own normal dead-entity cleanup later this
                // tick, but with deathEffect (not transformDeathEffect)
                // left unset on this card, that pass fires nothing further.
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

        // Poison-style damage-over-time mark from PoisonOnHit (Dart
        // Goblin/Firecracker Evolutions) -- independent of freeze/curse,
        // ticks down even while otherwise idle. takeDamage (not a direct
        // hp -=) so shield/parry/curse-on-incoming-damage still apply,
        // same as every other damage source.
        if (dotTicksRemaining > 0) {
            dotTicksRemaining--;
            dotTicksUntilNextDamage--;
            if (dotTicksUntilNextDamage <= 0 && dotDamagePerTick > 0) {
                takeDamage(dotDamagePerTick);
                dotTicksUntilNextDamage = dotTickInterval;
            }
        }

        if (periodicIntervalTicks > 0) {
            periodicTicksUntilNext--;
            if (periodicTicksUntilNext <= 0) {
                if (periodicEffect) periodicEffect->apply(board, position, team);
                periodicTicksUntilNext = periodicIntervalTicks;
            }
        }

        // Target-lock only applies once actually in attack range -- not
        // while still chasing. If the currently-locked target is within
        // effectiveRangeTo (i.e. this attacker is genuinely fighting it
        // this tick, or about to), resolveCurrentTarget() is trusted as-is
        // and findTarget() never runs: an attacker mid-fight doesn't get
        // distracted just because something else wandered closer. But if
        // the lock target is out of attack range (still being approached,
        // never reached it, or just left it -- pulled out by Tornado,
        // knocked back, etc.), there is deliberately NO lock: a fresh
        // findTarget() scan runs this same tick and every tick after,
        // freely switching to whatever's actually closest now. This is
        // also where findTarget's own sightRange limit does the real
        // work -- an attacker only ever chases something it can actually
        // see, falling back to the nearest enemy tower once nothing else
        // is in sight (see findTarget's own comment).
        //
        // A stun breaks the lock outright, same as leaving effective
        // range: resolveCurrentTarget() is skipped entirely on a tick
        // spent frozen, forcing a fresh findTarget() scan -- if something
        // is already in attack range once the stun clears (or during a
        // partial slow that still lets this tick's cooldown reach 0),
        // this attacker goes straight for it rather than blindly resuming
        // whatever it was fighting before. Matches the ramp system's own
        // "stun resets the charge" rule (below) being a full reset, not
        // just a number going back to 0 while the old fight continues
        // uninterrupted.
        // Forced retarget (Hero Knight's Triumphant Taunt) overrides the
        // whole normal lock/findTarget chain below outright -- resolved
        // first, same linear id-scan idiom as resolveCurrentTarget itself.
        // Once forced, this becomes THE target for every purpose below
        // (movement, lock bookkeeping, attack) exactly like any normally-
        // resolved one; forcedTargetTicksRemaining <= 0 (every card without
        // an active taunt on it) leaves this whole block a no-op, falling
        // straight through to the unchanged original resolution.
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
                // Stun always resets immediately, grace period or not --
                // "or it is hit by a stun attack" is an unconditional
                // reset in the sourced Evolution text, same as the
                // baseline (non-evolved) rule this branch already covered.
                currentTargetId = target->id;
                ticksOnTarget = 0;
                ticksSinceLastHit = 0;
            } else if (target->id != currentTargetId) {
                bool withinGrace = rampGracePeriodTicks > 0 && ticksSinceLastHit < rampGracePeriodTicks;
                currentTargetId = target->id;
                if (!withinGrace) ticksOnTarget = 0;
                // else: keep the current ramp stage across the switch --
                // this tick neither resets nor increments it, matching the
                // existing "switch tick itself doesn't count" shape below.
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
                    lastAttackDistance = dist;
                    // Fired here (rather than on projectile arrival) is
                    // fine for ticksSinceLastHit specifically -- the only
                    // card that ever configures rampGracePeriodTicks
                    // (Inferno Dragon Evolution) attacks instantly, same
                    // "no projectile travel time" shaping as the base
                    // Inferno Dragon (see its own CardRegistry comment).
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
                    // Hit-speed ramp (Little Prince): fires faster, not
                    // harder, the longer he's locked onto the same target --
                    // ticksOnTarget already includes the hit that just
                    // landed (incremented earlier this same update() call).
                    if (hitSpeedRampFullTick > 0) {
                        float cooldownFraction = (ticksOnTarget >= hitSpeedRampFullTick) ? hitSpeedRampFullFraction
                            : (ticksOnTarget >= hitSpeedRampMidTick) ? hitSpeedRampMidFraction
                            : 1.0f;
                        currentCooldown *= cooldownFraction;
                    }
                    // Temporary haste (Archer Queen's Cloaking Cape, Boss
                    // Bandit's Getaway Grenade).
                    if (temporaryInvisibilityTicksRemaining > 0) currentCooldown *= temporaryHitSpeedMultiplier;
                    // Self-haste on hit (Evolved Barbarians' Blade Rage):
                    // refresh the window first, then apply it to the
                    // cooldown this same attack just set -- so the very
                    // next attack is the hastened one, and continuing to
                    // land hits keeps the window (and the haste) alive
                    // indefinitely ("timer resets if they keep attacking").
                    if (selfHasteDurationTicks > 0) {
                        selfHasteTicksRemaining = selfHasteDurationTicks;
                        currentCooldown *= selfHasteCooldownMultiplier;
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
                    if (recoilDistance > 0.0f) pushAway(*this, target->position, recoilDistance);
                    // Self-heal on landing a hit (Evolved Bats: heals past
                    // its own starting max hp, up to healOnHitMaxHp -- an
                    // absolute cap set at spawn time, not a live fraction
                    // of `hp` the way enrageMaxHp/enrageHealPerHit is).
                    if (healOnHitAmount > 0 && hp < healOnHitMaxHp) {
                        hp = std::min(hp + healOnHitAmount, healOnHitMaxHp);
                    }
                    // Self-spawn on landing a hit, capped at how many are
                    // already alive (Evolved Skeletons' "Never-ending
                    // Horde") -- reuses IPeriodicEffect's exact shape
                    // (Board&, position, team) since it's the same
                    // "spawn something at my position" need as
                    // periodicEffect above, just triggered by a landed
                    // hit instead of a tick interval. The cap check itself
                    // lives inside the effect (see CappedSpawnOnHitEffect),
                    // not here, since it needs to know which cardId to
                    // count.
                    if (onHitSpawnEffect) onHitSpawnEffect->apply(board, position, team);
                    // Whirlwind pull (Evolved Valkyrie): everyone in
                    // radius takes the (typically low) pull damage,
                    // including the entity already hit by this same
                    // attack (excludeId -1 so nothing is skipped) and
                    // including buildings/towers -- only the physical
                    // pull itself exempts buildings, inside
                    // applyPullNearby.
                    if (onHitPullRadius > 0.0f) {
                        if (onHitPullDamage > 0) {
                            applySplashDamage(board, position, onHitPullRadius, -1, id, team, cardId, onHitPullDamage);
                        }
                        applyPullNearby(board, position, onHitPullRadius, onHitPullDistance, id, team);
                    }
                    // Tornado pulse while flying (Hero Wizard's Fiery
                    // Flight) -- centered on the TARGET, not self (see
                    // flightPulseTicksRemaining's own comment for why this
                    // differs from onHitPullRadius above). excludeId -1
                    // (nothing skipped), same as Valkyrie's own onHitPullRadius
                    // splash above -- "does its own damage" reads as
                    // additive on top of the main hit, not a replacement,
                    // so the primary target isn't exempted from the pulse
                    // just because it was already hit this same attack.
                    if (flightPulseTicksRemaining > 0 && flightPulseRadius > 0.0f) {
                        applySplashDamage(board, target->position, flightPulseRadius, -1, id, team, cardId, flightPulseDamage);
                        applyPullNearby(board, target->position, flightPulseRadius, flightPulsePullDistance, -1, team);
                    }
                    if (dieAfterFirstHit) hp = 0;
                }
            } else if (jumpMaxRange > 0.0f && dist >= jumpMinRange && dist <= jumpMaxRange && currentCooldown == 0.0f) {
                // Jump: instantly close to just inside attack range instead
                // of walking in over several ticks, then land the boosted,
                // splashy hit immediately -- Mega Knight's leap. A simpler
                // self-contained special case than the main attack branch
                // above, same precedent as the hook branch below (no
                // enrage/aura/dieAfterFirstHit follow-up, none of Mega
                // Knight's cards need it here).
                pullToward(*this, target->position, dist - effectiveAttackRange + 0.1f);
                int jumpDamage = static_cast<int>(getCurrentDamage() * jumpDamageMultiplier);
                target->takeDamage(jumpDamage);
                board.statsEvents.notifyDamageDealt(
                    { id, team, cardId, target->id, target->cardId, target->team, jumpDamage, board.currentTick });
                applySplashDamage(board, target->position, jumpSplashRadius, target->id, id, team, cardId, jumpDamage);
                currentCooldown = static_cast<float>(attackCooldown);
            } else if (hookRange > 0.0f && dist <= hookRange && currentCooldown == 0.0f) {
                // Hook: instantly pull the target to just inside melee
                // range instead of walking toward it. No damage lands this
                // tick -- the real hit happens on a later, normal-range
                // attack once it arrives, consistent with the real card
                // (hook first, melee second). If `target` ever resolved to
                // a building this would be a no-op, same as every other
                // pull/push in this codebase -- see pullToward's own
                // comment.
                pullToward(*target, position, dist - effectiveAttackRange + 0.1f);
                currentCooldown = static_cast<float>(attackCooldown);
            } else {
                Vector2D beforeMove = position;
                moveTowards(board, target->position);
                if (chargeThreshold > 0.0f) chargeProgress += beforeMove.distanceTo(position);
            }
        } else if (wasFrozen || rampGracePeriodTicks <= 0 || ticksSinceLastHit >= rampGracePeriodTicks) {
            // No target at all -- reset fully if this tick was spent
            // frozen (a stun always resets, bypassing any grace period
            // entirely, same as the target-found branch above), or if
            // there's no grace period configured (the baseline, unchanged
            // rule), or the grace period already ran out.
            currentTargetId = -1;
            ticksOnTarget = 0;
            ticksSinceLastHit = 0;
        }
        // else: not frozen this tick, and still within the grace period
        // with no target to fight -- hold currentTargetId/ticksOnTarget
        // exactly where they are, so the ramp stage is still there if a
        // new target shows up before ticksSinceLastHit crosses
        // rampGracePeriodTicks.

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
            && entity->id != this->id && (!entity->isFlying || targetsAir)
            && (minAttackRange <= 0.0f || position.distanceTo(entity->position) >= minAttackRange);
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

    // Two-tier scan: prefer the closest valid enemy within sightRange (a
    // troop or non-tower building it can actually "see"); only if nothing
    // qualifies there, fall back to the closest enemy Tower regardless of
    // distance -- a tower is always the eventual objective, never
    // competing with something in-sight purely on raw distance (a closer
    // tower does NOT steal aggro from a farther-but-still-in-sight enemy).
    // Combined with update()'s existing "only trust the current lock while
    // it's within actual attack range" check above, this reproduces the
    // real game's targeting model: locked on and fighting once in attack
    // range; freely re-evaluating "what's closest in sight" every tick
    // while just chasing (no lock during the chase itself); and, with
    // nothing in sight at all, heading for the nearest tower.
    virtual std::shared_ptr<Entity> findTarget(Board& board) const {
        std::shared_ptr<Entity> closestInSight = nullptr;
        float minSightDistance = std::numeric_limits<float>::max();
        std::shared_ptr<Entity> closestTower = nullptr;
        float minTowerDistance = std::numeric_limits<float>::max();

        for (const auto& entity : board.getEntities()) {
            if (!isValidTarget(entity)) continue;
            float dist = position.distanceTo(entity->position);
            if (entity->isTower()) {
                if (dist < minTowerDistance) {
                    minTowerDistance = dist;
                    closestTower = entity;
                }
            } else if (dist <= sightRange && dist < minSightDistance) {
                minSightDistance = dist;
                closestInSight = entity;
            }
        }
        return closestInSight ? closestInSight : closestTower;
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
        (void)board;
        (void)dest;
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

// On-hit area pull (Evolved Valkyrie's Whirlwind Axe): pulls every valid
// enemy within `radius` of `origin` toward it by `distance` tiles.
// Buildings/Towers are never actually displaced by this -- enforced
// inside pullToward itself (Entity.h), not a check here. No-op when
// radius or distance <= 0.
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

// Taunt (Hero Knight's Triumphant Taunt): forces every valid enemy within
// `radius` of `origin` onto `taunterId` as their target for `ticks` --
// see CombatEntity::forcedTargetEntityId/forcedTargetTicksRemaining and
// update()'s own targeting block for how this actually overrides normal
// targeting. No-op when radius or ticks <= 0.
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

// Piercing-line splash (Bowler, Magic Archer): applies `dealt` to every
// valid enemy within `halfWidth` of the straight line segment from
// `origin` (the attacker's own position at the moment it fired) toward
// `aimPoint` (the primary target's position), extended out to `range`
// total distance from origin -- a real line/rectangle hit test instead of
// applySplashDamage's circle-around-the-primary-target. No-op when range
// or halfWidth <= 0 -- every card that doesn't opt into line splash.
inline void applyLineSplashDamage(Board& board, const Vector2D& origin, const Vector2D& aimPoint,
        float range, float halfWidth, int excludeId, int attackerId, int attackerTeam, int attackerCardId, int dealt) {
    if (range <= 0.0f || halfWidth <= 0.0f) return;
    float dx = aimPoint.x - origin.x;
    float dy = aimPoint.y - origin.y;
    float lineLen = std::sqrt(dx * dx + dy * dy);
    if (lineLen <= 0.01f) return; // no direction to fire along
    float ux = dx / lineLen, uy = dy / lineLen; // unit direction, origin -> aimPoint

    for (const auto& entity : board.getEntities()) {
        if (entity->id == excludeId) continue; // already damaged as the primary target
        if (entity->team == attackerTeam || !entity->isAlive() || !entity->isTargetable()) continue;

        // Project the candidate onto the line, clamped to [0, range] so
        // nothing behind the shooter or past the line's far end counts,
        // then measure perpendicular distance from that closest point.
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
            { attackerId, attackerTeam, attackerCardId, entity->id, entity->cardId, entity->team, dealt, board.currentTick });
    }
}

// Ally buff: applies a temporary damage buff to up to maxTargets of the
// closest same-team CombatEntity within radius of origin (Rune Giant's
// per-3rd-attack enchant, Lumberjack's death-potion, Rage). excludeId
// skips the source itself (a buffing unit doesn't buff itself). No-op
// when radius <= 0 or maxTargets <= 0.
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
