#pragma once
#include <string>
#include <functional>
#include <unordered_map>
#include <vector>
#include <memory>
#include "Board.h"
#include "CardStats.h"
#include "CardFactories.h"
#include "FreezeOnHit.h"
#include "SpawnOnDeath.h"
#include "AreaDamageOnDeath.h"
#include "CompositeDeathEffect.h"
#include "PeriodicSpawnEffect.h"
#include "CurseOnHit.h"
#include "AreaBuffOnDeath.h"
#include "AreaStunEffect.h"
#include "ElixirGrantEffect.h"
#include "EnemyElixirGrantOnDeath.h"
#include "SpawnOnDeathForEnemyTeam.h"
#include "ProximityGatedPeriodicSpawnEffect.h"
#include "CursedHogOnHit.h"
#include "MightyMinerEscapeEffect.h"
#include "GoldenKnightDashEffect.h"
#include "SkeletonKingSoulSummonEffect.h"
#include "ArcherQueenCloakEffect.h"
#include "MonkDeflectEffect.h"
#include "SpawnOnAbility.h"
#include "GoblinsteinLightningLinkEffect.h"
#include "BossBanditGetawayGrenadeEffect.h"
#include "CappedSpawnOnHitEffect.h"
#include "PoisonOnHit.h"
#include "PeriodicFreezeNearestEffect.h"
#include "DarkGuardOnDamageEffect.h"
#include "HeroMiniPekkaBoostEffect.h"
#include "HeroKnightTauntEffect.h"
#include "HeroWizardFieryFlightEffect.h"
#include "HeroGiantHurlEffect.h"
#include "HeroMegaMinionWarpEffect.h"
#include "HeroMagicArcherTripleThreatEffect.h"
#include "HeroIceGolemSnowstormEffect.h"
#include "HeroBarbarianBarrelRerollEffect.h"

// External-facing shape is unchanged on purpose: GameManager, ClashEnv,
// GameLogger, TerminalRenderer and main.cpp all consume CardDefinition as
// before. Internally, spawnEntity is now always one of the small archetype
// factories in CardFactories.h bound to this card's CardStats, instead of a
// bespoke lambda per card.
struct CardDefinition {
    int id;
    std::string name;
    float cost;
    bool isSpell;
    // True only for Archetype::DefensiveBuilding cards (Cannon/Tesla/Bomb
    // Tower/Inferno Tower) -- Towers aren't in this registry at all (built
    // directly by GameManager with a negative sentinel cardId), so stats
    // code classifying a DamageDealtEvent's targetCardId treats "not found
    // in the registry" as a Tower, itself also a building. See
    // stats/StatsCollectors.h's DamageByTargetTypeCollector.
    bool isBuilding;
    // The footprint GameManager::isValidPlacement keeps clear of an existing
    // building -- see CardFactories::placementRadius. Unused for spells.
    float placementRadius;
    // Miner, Goblin Drill: can be placed anywhere on the board, not just
    // this player's own half -- see GameManager::isValidPlacement. false
    // (the default) is every other troop/building.
    bool deployAnywhere;
    // Mirrors CombatEntity::isChampion/CardStats::isChampion -- surfaced
    // here (not just on the spawned entity) so deck contents can be
    // inspected before anything is placed, e.g. by countChampions() below.
    bool isChampion;
    // Mirrors CombatEntity::isHero/CardStats::isHero -- see that field's own
    // comment. validateDeckSlots/PlayerState::seedSlotState/GameManager::
    // playCard's tracking hook all check `isChampion || isHero` uniformly.
    bool isHero = false;
    // Mirrors CardStats::abilityElixirCost/abilityUsableAfterDeathTicks/
    // postDeathAbilityEffect -- surfaced at the REGISTRY level (not just on
    // the spawned entity) because Hero Goblins' post-death reactivation
    // (see PlayerState::ChampionSlotState::lastSquadWipeTick) fires when
    // NOTHING is alive to read these off of, so GameManager::
    // activateChampionAbility looks them up here instead. 0/nullptr (the
    // defaults) are every card without a post-death ability -- including
    // every OTHER Hero/Champion, which read cost off their own live entity
    // instead.
    float abilityElixirCost = 0.0f;
    int abilityUsableAfterDeathTicks = 0;
    std::shared_ptr<IPeriodicEffect> postDeathAbilityEffect;
    // Display/rendering metadata -- NOT used by any gameplay logic (that all
    // goes through the CardStats captured in spawnEntity's closure below).
    // Exists purely so GameLogger can embed an authoritative, per-replay
    // symbol/maxHp/isFlying table sourced live from this registry instead of
    // a hand-copied one (e.g. web/viewer.html's JS tables) drifting out of
    // sync the next time a card is added -- see GameLogger::save()'s
    // "cardMeta" block. hp=0 for spells (CardStats' own default -- spells
    // never set it, matching "no persistent HP" correctly). Defaults here
    // mirror CardStats' own, so a construction path that forgets to set
    // them (there shouldn't be one -- both add() and addEvolution() do)
    // fails safe instead of reading uninitialized memory.
    int hp = 0;
    char symbol = '?';
    bool isFlying = false;
    std::function<void(float x, float y, int team, Board& board)> spawnEntity;

    // Evolution slot (see addEvolution() below and PlayerState::playCard).
    // isEvolution=false (every ordinary card) means the rest of these are
    // unused. cost/isSpell/isBuilding/placementRadius/deployAnywhere above
    // are always derived from the BASE (un-evolved) stats -- real
    // Evolutions never change elixir cost or placement legality -- so
    // spawnEntity above already spawns the base form; spawnEvolvedEntity
    // is the one extra thing an evolution slot needs.
    bool isEvolution = false;
    int evolutionCycleThreshold = 0;
    int evolvedUsesGranted = 0;
    std::function<void(float x, float y, int team, Board& board)> spawnEvolvedEntity;
};

class CardRegistry {
private:
    std::unordered_map<int, CardDefinition> cards;

    static CardStats troop(int id, std::string name, float cost, Archetype archetype,
        int hp, float speed, float attackRange, int damage, int attackCooldown, char symbol) {
        CardStats s;
        s.id = id; s.name = std::move(name); s.cost = cost; s.archetype = archetype;
        s.hp = hp; s.speed = speed; s.attackRange = attackRange;
        s.damage = damage; s.attackCooldown = attackCooldown; s.symbol = symbol;
        return s;
    }

    static CardStats building(int id, std::string name, float cost,
        int hp, char symbol, float attackRange, int damage, int attackCooldown) {
        CardStats s;
        s.id = id; s.name = std::move(name); s.cost = cost; s.archetype = Archetype::DefensiveBuilding;
        s.hp = hp; s.symbol = symbol; s.attackRange = attackRange;
        s.damage = damage; s.attackCooldown = attackCooldown;
        return s;
    }

    static CardStats spell(int id, std::string name, float cost,
        float radius, int damage, int delayTicks, char symbol) {
        CardStats s;
        s.id = id; s.name = std::move(name); s.cost = cost; s.archetype = Archetype::Spell;
        s.spellRadius = radius; s.damage = damage; s.spellDelayTicks = delayTicks; s.symbol = symbol;
        return s;
    }

    // 4-wide, 0.6-spaced grid layout for Skeleton Army's 15 skeletons.
    static std::vector<Vector2D> skeletonArmyOffsets() {
        std::vector<Vector2D> offsets;
        for (int i = 0; i < 15; ++i) {
            float ox = (static_cast<float>(i % 4) - 1.5f) * 0.6f;
            float oy = (static_cast<float>(i / 4) - 1.0f) * 0.6f;
            offsets.push_back({ ox, oy });
        }
        return offsets;
    }
    // Skeleton Army Evolution's "+1 Skeleton" stat boost: the same 4-wide
    // grid, extended by the one remaining cell (row 3, col 3) the base
    // loop above deliberately stops short of -- 16 total, representing
    // the 15 ordinary Skeletons plus the Skeleton General.
    static std::vector<Vector2D> skeletonArmyEvolvedOffsets() {
        std::vector<Vector2D> offsets = skeletonArmyOffsets();
        offsets.push_back({ 0.9f, 1.2f });
        return offsets;
    }

    // Golemite: only ever spawned by a Golem's death, never itself a
    // playable card (no id in the registry, hence never add()-ed).
    static CardStats golemiteStats() {
        return troop(-1, "Golemite", 0.0f, Archetype::MeleeBuildingTargeter, 1039, 0.2f, 0.25f, 84, 25, 'q')
            .withOffsets({ {-0.3f, 0.0f}, {0.3f, 0.0f} }).withSightRange(7.0f);
    }

    // Death-spawn child units below reuse an already-registered card's own
    // sourced stats rather than inventing unsourced numbers for units this
    // data sync couldn't find independent stats for (e.g. "Bush Goblins",
    // "Elixir Blobs", "Lava Pups", "Goblin Brawler") -- those on-death spawns
    // are simply left unmodeled instead (see each card's own comment below).
    // Negative ids stay clear of -1 (Golemite, above) and of
    // GameManager::TOWER_KING_ID/TOWER_PRINCESS_ID (-2/-3).
    static CardStats battleRamBarbarianStats() {
        return troop(-10, "Barbarians", 0.0f, Archetype::MeleeSquad, 691, 0.5f, 0.7f, 192, 14, 'B')
            .withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f} });
    }
    static CardStats skeletonBarrelSkeletonStats() {
        return troop(-11, "Skeletons", 0.0f, Archetype::MeleeSquad, 81, 1.0f, 0.5f, 81, 11, 'k')
            .withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f} });
    }
    // Confirmed: death spawn is 1 Bat, not 3 -- default single offset.
    static CardStats nightWitchBatStats() {
        return troop(-12, "Bats", 0.0f, Archetype::MeleeSquad, 81, 0.85f, 0.5f, 81, 12, 't')
            .withFlying().withTargetsAir();
    }

    // Periodic-spawn child units below (Witch/Night Witch/Furnace/
    // Barbarian Hut/Goblin Hut/Goblin Drill) follow the exact same
    // reuse-sourced-stats-where-possible reasoning as the death-spawn
    // helpers above; Goblin Hut's per-spawn count and every interval
    // below aren't sourced either (see each card's own comment).
    static CardStats witchSkeletonStats() {
        return troop(-13, "Skeletons", 0.0f, Archetype::MeleeSquad, 81, 1.0f, 0.5f, 81, 11, 'k')
            .withOffsets({ {-0.4f, -0.4f}, {0.4f, -0.4f}, {-0.4f, 0.4f}, {0.4f, 0.4f} });
    }
    static CardStats nightWitchPeriodicBatStats() {
        return troop(-14, "Bats", 0.0f, Archetype::MeleeSquad, 81, 0.85f, 0.5f, 81, 12, 't')
            .withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f} })
            .withFlying().withTargetsAir();
    }
    static CardStats furnaceFireSpiritStats() {
        return troop(-15, "Fire Spirit", 0.0f, Archetype::RangedSquad, 230, 0.85f, 2.5f, 207, 10, '<')
            .withTargetsAir();
    }
    static CardStats barbarianHutBarbarianStats() {
        return troop(-16, "Barbarians", 0.0f, Archetype::MeleeSquad, 691, 0.5f, 0.7f, 192, 14, 'B')
            .withOffsets({ {0.0f, 0.0f}, {-0.5f, -0.5f}, {0.5f, -0.5f} });
    }
    static CardStats goblinHutSpearGoblinStats() {
        return troop(-17, "Spear Goblins", 0.0f, Archetype::RangedSquad, 133, 1.0f, 5.0f, 81, 17, 'S');
    }
    static CardStats goblinDrillGoblinStats() {
        return troop(-18, "Goblins", 0.0f, Archetype::MeleeSquad, 202, 1.0f, 0.5f, 120, 11, 'g');
    }
    // Phoenix's one-time revive: a second Phoenix with no deathEffect of
    // its own, so it only ever comes back once, not indefinitely. The
    // real card leaves a vulnerable egg for ~4s before hatching back to
    // full -- this spawns the reviving Phoenix immediately at full hp
    // instead, dropping that vulnerability window (a minor simplification,
    // no delayed/interruptible spawn primitive exists).
    static CardStats phoenixReviveStats() {
        return troop(-19, "Phoenix", 0.0f, Archetype::MeleeSquad, 1052, 0.5f, 1.0f, 217, 10, '7')
            .withFlying().withTargetsAir();
    }
    // Spell-spawn child units (Goblin Barrel, Royal Delivery, Graveyard) --
    // same reuse-sourced-stats reasoning as the death/periodic-spawn
    // helpers above.
    static CardStats goblinBarrelGoblinStats() {
        return troop(-20, "Goblins", 0.0f, Archetype::MeleeSquad, 202, 1.0f, 0.5f, 120, 11, 'g')
            .withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f}, {0.0f, 0.4f} });
    }
    static CardStats graveyardSkeletonStats() {
        return troop(-21, "Skeletons", 0.0f, Archetype::MeleeSquad, 81, 1.0f, 0.5f, 81, 11, 'k');
    }
    static CardStats royalDeliveryRecruitStats() {
        return troop(-22, "Royal Recruits", 0.0f, Archetype::MeleeSquad, 547, 0.5f, 1.0f, 133, 13, '@')
            .withShield(240); // matches Royal Recruits' own corrected shield value
    }
    static CardStats barbarianBarrelBarbarianStats() {
        return troop(-23, "Barbarians", 0.0f, Archetype::MeleeSquad, 691, 0.5f, 0.7f, 192, 14, 'B');
    }
    // Hero Barbarian Barrel Hero-ifies the SPAWNED Barbarian himself, not
    // the ephemeral one-tick barrel spell (which has no persistent entity
    // to carry an ability at all) -- same base stats as
    // barbarianBarrelBarbarianStats above, plus "Rowdy Reroll".
    static CardStats heroBarbarianBarrelBarbarianStats() {
        return troop(-47, "Hero Barbarian Barrel", 0.0f, Archetype::MeleeSquad, 691, 0.5f, 0.7f, 192, 14, 'B')
            .withHeroAbility(1.0f, 0, std::make_shared<HeroBarbarianBarrelRerollEffect>(3.0f, 0.7f, 233), 1);
    }
    // Compound-card secondary units (Goblin Machine's rocket turret, Ram
    // Rider's crossbow, Goblin Giant's carried Spear Goblins, Goblin
    // Gang's ranged half, Rascals' ranged half) -- spawned via
    // CardStats::secondaryUnit, riding along at the same deploy point as
    // the primary unit but targeting entirely independently. Stats reused
    // from their standalone registered cards where possible.
    // Confirmed real split: rocket launcher hits harder per-shot (304) but
    // fires much slower (DPS 86 -> ~3.5s/35-tick cooldown) than the melee
    // arms' own 212/12-tick attack -- corrected from an unsourced 50/50
    // guess (106 damage @ 12 ticks).
    static CardStats goblinMachineTurretStats() {
        return troop(-24, "Goblin Machine", 0.0f, Archetype::RangedSquad, 2150, 0.5f, 5.0f, 304, 35, '3')
            .withSplash(1.5f).withTargetsAir();
    }
    // Confirmed real split: crossbow deals less per-shot (104) but fires
    // faster (DPS 94 -> ~1.1s/11-tick cooldown) than the Ram's own 250/
    // 17-tick melee hit -- corrected from an unsourced 50/50 guess (125
    // damage @ 17 ticks). withIgnoresRiver matches the main Ram body's own
    // flag (see Ram Rider's registration) -- without it, this
    // independently-spawned secondary unit would desync from the ram at
    // the river, left stuck on the near bank while the ram itself crosses.
    static CardStats ramRiderCrossbowStats() {
        return troop(-25, "Ram Rider", 0.0f, Archetype::RangedSquad, 1766, 0.5f, 5.0f, 104, 11, '"')
            .withTargetsAir()
            .withIgnoresRiver();
    }
    static CardStats goblinGiantSpearGoblinsStats() {
        return troop(-26, "Spear Goblins", 0.0f, Archetype::RangedSquad, 133, 0.5f, 5.0f, 81, 17, 'S')
            .withOffsets({ {-0.4f, 0.3f}, {0.4f, 0.3f} }).withTargetsAir();
    }

    // Death-spawn child units researched after the initial 2026 sync (see
    // the wiki-research pass that closed the "no sourced stats" gaps this
    // file used to note for Lava Pups/Bush Goblins/Goblin Brawler/Elixir
    // Blobs/Cursed Hog). Same reuse-sourced-stats reasoning as every
    // helper above; ids continue from -29.
    static CardStats lavaHoundPupStats() {
        return troop(-29, "Lava Pups", 0.0f, Archetype::MeleeSquad, 217, 0.5f, 1.6f, 81, 17, 'y')
            .withOffsets({ {-0.8f, 0.0f}, {0.8f, 0.0f}, {-0.5f, 0.7f}, {0.5f, 0.7f}, {-0.5f, -0.7f}, {0.5f, -0.7f} })
            .withFlying().withTargetsAir();
    }
    static CardStats suspiciousBushGoblinStats() {
        return troop(-30, "Bush Goblins", 0.0f, Archetype::MeleeSquad, 304, 0.5f, 0.8f, 256, 14, 'g')
            .withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f} });
    }
    static CardStats goblinCageBrawlerStats() {
        return troop(-31, "Goblin Brawler", 0.0f, Archetype::MeleeSquad, 1080, 0.7f, 0.8f, 337, 11, 'o');
    }
    // Elixir Golem's split chain: Golem -> 2 Golemites -> 2 Blobs each (4
    // total). Each tier's own death both spawns the next tier down (via
    // the ordinary SpawnOnDeath already used everywhere else) AND hands
    // the OPPONENT a slice of elixir (EnemyElixirGrantOnDeath) -- 1.0 from
    // the Golem itself, 0.5 from each Golemite, 0.5 from each Blob; 4.0
    // total if the whole chain is killed. Blob is the base case: no
    // further spawn, just its own elixir grant.
    static CardStats elixirBlobStats() {
        return troop(-33, "Elixir Blob", 0.0f, Archetype::MeleeBuildingTargeter, 360, 0.7f, 0.5f, 64, 11, 'e')
            .withDeathEffect(std::make_shared<EnemyElixirGrantOnDeath>(0.5f)).withSightRange(7.5f);
    }
    static CardStats elixirGolemiteStats() {
        return troop(-32, "Elixir Golemite", 0.0f, Archetype::MeleeBuildingTargeter, 762, 0.5f, 1.0f, 128, 11, 'e')
            .withDeathEffect(std::make_shared<CompositeDeathEffect>(
                std::vector<std::shared_ptr<IDeathEffect>>{
                    std::make_shared<SpawnOnDeath>(
                        elixirBlobStats().withOffsets({ {-0.3f, 0.0f}, {0.3f, 0.0f} })),
                    std::make_shared<EnemyElixirGrantOnDeath>(0.5f)
                })).withSightRange(7.5f);
    }
    // Mother Witch's Cursed Hog: spawned via SpawnOnDeathForEnemyTeam (see
    // CursedHogOnHit), not the ordinary same-team SpawnOnDeath every other
    // helper here uses -- it fights FOR Mother Witch, against whichever
    // team the cursed unit that just died belonged to.
    static CardStats cursedHogStats() {
        return troop(-34, "Cursed Hog", 0.0f, Archetype::MeleeBuildingTargeter, 629, 1.0f, 0.75f, 53, 12, 'h');
    }
    // Reuses barbarianHutBarbarianStats' own combat numbers (a single
    // Barbarian released when the Hut itself dies, not the 3-at-once
    // periodic spawn).
    static CardStats barbarianHutDeathBarbarianStats() {
        return troop(-35, "Barbarians", 0.0f, Archetype::MeleeSquad, 691, 0.5f, 0.7f, 192, 14, 'B');
    }
    // Reuses goblinDrillGoblinStats' own combat numbers, just 2 offsets
    // instead of the periodic spawn's 1 (2021 balance patch reduced this
    // from 3 to 2).
    static CardStats goblinDrillDeathGoblinStats() {
        return troop(-36, "Goblins", 0.0f, Archetype::MeleeSquad, 202, 1.0f, 0.5f, 120, 11, 'g')
            .withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f} });
    }
    // Goblin Demolisher's transformed form: a short-range, Very Fast,
    // building-only kamikaze that detonates in a 2.5-radius, 404-damage
    // burst on its first hit -- spawned via SpawnOnDeath the instant the
    // ranged Goblin Demolisher above kills itself at <=50% hp (see
    // CombatEntity::transformKillsSelf). Reuses the ranged form's own
    // 1300 hp rather than inventing a separate transformed-form hp figure
    // (not part of the sourced data either way). The real card's 10s
    // post-transform lifetime cap isn't modeled -- dieAfterFirstHit
    // already bounds it in practice (buildings/towers are essentially
    // always present for a building-only target to detonate against).
    static CardStats goblinDemolisherKamikazeStats() {
        return troop(-37, "Kamikaze Goblin Demolisher", 0.0f, Archetype::MeleeBuildingTargeter,
            1300, 1.0f, 0.5f, 404, 10, '&')
            .withSplash(2.5f).withDieAfterFirstHit();
    }
    // Skeleton King's Soul Summoning children -- count is dynamic (base +
    // however many souls were collected), so unlike every other multi-unit
    // helper above, this is spawned in a loop by
    // SkeletonKingSoulSummonEffect itself rather than via static
    // spawnOffsets; this stats object is always spawned one at a time.
    static CardStats skeletonKingSummonedSkeletonStats() {
        return troop(-38, "Skeletons", 0.0f, Archetype::MeleeSquad, 81, 0.7f, 0.5f, 81, 11, 'k');
    }
    // Little Prince's Royal Rescue: the summoned Guardienne. Real card's
    // spawn-in dash+knockback isn't modeled (same "knockback on a spawn
    // burst not modeled" simplification already noted for Goblin Drill).
    static CardStats guardienneStats() {
        return troop(-39, "Guardienne", 0.0f, Archetype::MeleeSquad, 1600, 0.5f, 1.2f, 217, 12, 'u')
            .withCharge(3.0f, 2.0f);
    }
    // Hero Musketeer's "Trusty Turret": a short-range auto-turret spawned
    // in front of her, self-destructing after a fixed 10s (100-tick)
    // lifetime -- see withHpTransform(1.0f, ...)'s own comment for how a
    // fraction of EXACTLY 1.0 turns that HP-threshold mechanism into a
    // pure fixed-lifetime timer with zero hp loss required.
    // becomesStationary stays false: the turret is already stationary via
    // its own DefensiveBuilding archetype, and that branch's
    // applyFreeze(ticks, 0.0f) would additionally zero the turret's own
    // attack-cooldown drain rate, permanently silencing it. hp/damage
    // aren't part of any sourced data -- reasonable engine-internal
    // constants for a small, short-lived defensive structure, same
    // caveat as splashRadius/shieldHp elsewhere in this file.
    static CardStats heroMusketeerTurretStats() {
        return building(-45, "Trusty Turret", 0.0f, 200, 't', 4.0f, 90, 5)
            .withTargetsAir()
            .withHpTransform(1.0f, 100, false);
    }
    // Hero Magic Archer's Triple Threat: the decoy left behind at his old
    // position. Real card's decoy soaks hits/draws aggro but deals none of
    // its own -- modeled as a zero-damage, modest-hp stationary (speed 0)
    // unit. hp/archetype/lifetime aren't part of the sourced data --
    // reasonable engine-internal constants, same caveat as splashRadius/
    // shieldHp elsewhere in this file.
    static CardStats heroMagicArcherDecoyStats() {
        return troop(-46, "Decoy", 0.0f, Archetype::MeleeSquad, 100, 0.0f, 1.0f, 0, 100, 'd');
    }
    // Wall Breakers Evolution's "Runner": spawned on death (see
    // evolvedWallBreakersStats below). Only Level-6 data was found (103hp/
    // 113dmg) -- scaled to this file's level-11 convention via the ~10%/
    // level compounding growth already used elsewhere in this file for
    // similar level-gap approximations (103*1.1^5≈166, 113*1.1^5≈182),
    // not independently sourced at level 11. Symbol 'g' reused from the
    // Goblins family (harmless cosmetic reuse, same precedent as every
    // other reused symbol in this file) -- this roster has exhausted
    // nearly the entire printable-ASCII symbol space.
    // Lumberjack Evolution's death-spawn "ghost" -- real card's ghost is
    // invincible only while inside its own dropped Rage zone, for the
    // zone's duration; approximated as a plain invisible troop with no
    // conditional invincibility (this engine has no "invincible while
    // standing in a specific spell zone" mechanism).
    static CardStats lumberjackGhostStats() {
        return troop(-44, "Lumberjack Ghost", 0.0f, Archetype::MeleeSquad, 400, 1.0f, 0.7f, 256, 8, 'l')
            .withInvisibility(10);
    }
    static CardStats runnerStats() {
        // 175 = 50% of Wall Breakers' own 350 damage, matching the sourced
        // "continue to run to the nearest building... dealing 50% of the
        // original damage" figure precisely (corrected from an earlier,
        // unsourced 182). Still an approximation of the real one-shot-
        // then-gone barrel: this engine models it as an ordinary
        // MeleeBuildingTargeter that keeps attacking on a normal cooldown
        // once it arrives, rather than a single detonation that consumes
        // it, and doesn't reduce its damage against Crown Towers
        // specifically the way the sourced text does for the FIRST
        // (death) explosion.
        return troop(-41, "Runner", 0.0f, Archetype::MeleeBuildingTargeter, 166, 1.2f, 0.6f, 175, 8, 'g');
    }
    // Battle Ram Evolution's death-spawn: a stronger version of the
    // ordinary battleRamBarbarianStats() pair -- see that Evolution's own
    // registration comment for why this is a flat stat buff rather than
    // recursively invoking the Evolution framework itself.
    static CardStats battleRamEvolvedBarbarianStats() {
        return troop(-10, "Barbarians", 0.0f, Archetype::MeleeSquad, 830, 0.5f, 0.7f, 230, 14, 'B')
            .withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f} });
    }
    // Evolved Skeletons' "Never-ending Horde" spawn -- deliberately reuses
    // id 24 (the base Skeletons card's own id), not a fresh negative
    // sentinel like every other spawned-helper stats function in this
    // file: CappedSpawnOnHitEffect's cap check counts entities by cardId,
    // so sharing id 24 makes it count the 3 originally-deployed Evolved
    // Skeletons AND every child this effect has already spawned as one
    // combined population, capping the true on-field total at exactly 8
    // (see addEvolution(125,...) below) -- matching the sourced "up to
    // eight Evolved Skeletons on the field at once" precisely, instead of
    // a separate id needing its own approximate sub-cap. Doesn't itself
    // carry the spawn-on-hit trigger (only the 3 originals do), so growth
    // is bounded by 3 spawn sources, not an unbounded chain reaction --
    // the real card's exact spawn-chain depth isn't in the sourced data
    // either way.
    static CardStats evolvedSkeletonChildStats() {
        return troop(24, "Skeletons", 0.0f, Archetype::MeleeSquad, 81, 1.0f, 0.5f, 81, 11, 'k');
    }
    // Skeleton Army Evolution's death-spawn "shadow": real card's shadows
    // have unlimited hp and are untargetable by anything but spells --
    // approximated as a normal targetable unit with a large-but-finite hp
    // instead (500), avoiding an unremovable unit cluttering the board.
    // "General Gerry" (a named unit in the squad) isn't modeled -- no
    // described mechanical difference beyond cosmetic positioning.
    static CardStats skeletonArmyShadowStats() {
        return troop(-43, "Skeleton Shadow", 0.0f, Archetype::MeleeSquad, 500, 1.0f, 0.5f, 81, 11, 'k');
    }
    // Royal Ghost Evolution's on-hit spawn: 51% damage (261*0.51=133),
    // 6.7% hp (1210*0.067=81) of the Royal Ghost's own stats, per the
    // sourced table. Reuses her own invisibility duration/symbol -- "the
    // Souldiers' stats are identical to the Royal Ghost's" beyond the two
    // scaled-down numbers.
    static CardStats royalGhostSouldierStats() {
        return troop(-42, "Souldier", 0.0f, Archetype::MeleeSquad, 81, 0.7f, 1.2f, 133, 18, 'Q');
    }

    void add(const CardStats& stats) {
        CardDefinition def;
        def.id = stats.id;
        def.name = stats.name;
        def.cost = stats.cost;
        def.isSpell = (stats.archetype == Archetype::Spell);
        def.isBuilding = (stats.archetype == Archetype::DefensiveBuilding);
        def.placementRadius = CardFactories::placementRadius(stats.archetype);
        def.deployAnywhere = stats.deployAnywhere;
        def.isChampion = stats.isChampion;
        def.isHero = stats.isHero;
        def.abilityElixirCost = stats.abilityElixirCost;
        def.abilityUsableAfterDeathTicks = stats.abilityUsableAfterDeathTicks;
        def.postDeathAbilityEffect = stats.postDeathAbilityEffect;
        def.hp = stats.hp;
        def.symbol = stats.symbol;
        def.isFlying = stats.isFlying;
        def.spawnEntity = [stats](float x, float y, int team, Board& board) {
            CardFactories::spawn(stats, x, y, team, board);
        };
        cards[stats.id] = std::move(def);
    }

    // Registers one Evolution slot: `evolutionId` is what a deck puts in
    // that slot instead of the base card's own id (e.g. Knight stays 0;
    // "Evolution Knight" gets its own id here). baseStats/evolvedStats
    // must otherwise represent the same card -- cost/isSpell/isBuilding/
    // placementRadius/deployAnywhere are all derived from baseStats alone,
    // since a real Evolution never changes any of those, only combat
    // behavior. cycleThreshold is how many times this slot must be played
    // (un-evolved) before it unlocks; evolvedUses is how many of the
    // following plays use evolvedStats before the slot goes back to
    // counting un-evolved cycles again -- this repeats for the whole
    // match, it's not a one-time charge (confirmed: Wall Breakers
    // Evolution's "1 in every 3 deploys will be evolved" describes a
    // repeating 2-cycles-then-1-evolved pattern, not a fixed total) -- see
    // PlayerState::EvolutionSlotState/playCard for where that's tracked.
    void addEvolution(int evolutionId, CardStats baseStats, CardStats evolvedStats,
            int cycleThreshold, int evolvedUses) {
        CardDefinition def;
        def.id = evolutionId;
        def.name = baseStats.name;
        def.cost = baseStats.cost;
        def.isSpell = (baseStats.archetype == Archetype::Spell);
        def.isBuilding = (baseStats.archetype == Archetype::DefensiveBuilding);
        def.placementRadius = CardFactories::placementRadius(baseStats.archetype);
        def.deployAnywhere = baseStats.deployAnywhere;
        def.isChampion = baseStats.isChampion;
        def.isHero = baseStats.isHero;
        def.abilityElixirCost = baseStats.abilityElixirCost;
        def.abilityUsableAfterDeathTicks = baseStats.abilityUsableAfterDeathTicks;
        def.postDeathAbilityEffect = baseStats.postDeathAbilityEffect;
        def.hp = baseStats.hp;
        def.symbol = baseStats.symbol;
        def.isFlying = baseStats.isFlying;
        def.spawnEntity = [baseStats](float x, float y, int team, Board& board) {
            CardFactories::spawn(baseStats, x, y, team, board);
        };
        def.isEvolution = true;
        def.evolutionCycleThreshold = cycleThreshold;
        def.evolvedUsesGranted = evolvedUses;
        def.spawnEvolvedEntity = [evolvedStats](float x, float y, int team, Board& board) {
            CardFactories::spawn(evolvedStats, x, y, team, board);
        };
        cards[evolutionId] = std::move(def);
    }

    CardRegistry() {
        // Stats below are synced to current level-11 tournament-standard game
        // data (cost/hp/damage/hit-speed/range/count). Speed categories
        // aren't part of that data set, so existing speed values are left
        // as-is. Inferno Tower's ramp, Electro Wizard's split hit,
        // Executioner's boomerang return, and Poison/Arrows' multi-tick
        // damage are all modeled where they occur below (see each card's own
        // comment) -- not collapsed to a flat number. On-death burst damage
        // (Giant Skeleton, Golem, Ice Golem, Balloon) is modeled via
        // AreaDamageOnDeath below; Lumberjack's death effect is a Rage buff,
        // not damage -- see the buff/aura cards further down.
        // Electro Wizard's spawn zap is the one on-spawn effect that *is*
        // modeled, via CardStats::withSpawnEffect (see below).

        // === Melee Troops ===
        add(troop(0, "Knight", 3.0f, Archetype::MeleeSquad, 1766, 0.5f, 1.2f, 202, 12, 'K'));

        add(troop(4, "Goblins", 2.0f, Archetype::MeleeSquad, 202, 1.0f, 0.5f, 120, 11, 'g')
            .withOffsets({ {-0.5f, -0.5f}, {0.5f, -0.5f}, {-0.5f, 0.5f}, {0.5f, 0.5f} }));

        add(troop(5, "Mini PEKKA", 4.0f, Archetype::MeleeSquad, 1390, 0.8f, 0.8f, 755, 16, 'M'));

        add(troop(8, "Barbarians", 5.0f, Archetype::MeleeSquad, 691, 0.5f, 0.7f, 192, 14, 'B')
            .withOffsets({ {0.0f, 0.0f}, {-0.5f, -0.5f}, {0.5f, -0.5f}, {-0.5f, 0.5f}, {0.5f, 0.5f} }));

        add(troop(10, "Valkyrie", 4.0f, Archetype::MeleeSquad, 1907, 0.5f, 1.2f, 266, 15, 'V')
            .withSplash(1.5f)); // 360-degree swing

        add(troop(12, "Skeleton Army", 3.0f, Archetype::MeleeSquad, 81, 1.0f, 0.5f, 81, 11, 's')
            .withOffsets(skeletonArmyOffsets()));

        add(troop(13, "P.E.K.K.A.", 7.0f, Archetype::MeleeSquad, 3760, 0.4f, 1.2f, 842, 18, 'E').withSightRange(5.0f));

        // Charge threshold/multiplier below aren't part of the sourced
        // stats data -- reasonable engine-internal constants, same caveat
        // as splashRadius/shieldHp. Prince crosses the river directly
        // (confirmed real-game river-crossing troop, one of the "jumpers"),
        // not routed through a bridge -- see withIgnoresRiver.
        add(troop(14, "Prince", 5.0f, Archetype::MeleeSquad, 1920, 0.6f, 1.6f, 391, 14, 'p')
            .withCharge(3.0f, 2.0f)
            .withIgnoresRiver());

        add(troop(17, "Elite Barbarians", 6.0f, Archetype::MeleeSquad, 1341, 0.7f, 1.2f, 384, 14, 'e')
            .withOffsets({ {-0.5f, 0.0f}, {0.5f, 0.0f} }));

        // Lumberjack's death-potion radius/multiplier/duration below aren't
        // part of the sourced stats data -- reasonable engine-internal
        // constants, same caveat as splashRadius.
        add(troop(21, "Lumberjack", 4.0f, Archetype::MeleeSquad, 1282, 0.8f, 0.7f, 256, 8, 'l')
            .withDeathEffect(std::make_shared<AreaBuffOnDeath>(2.5f, 1.75f, 55)));

        add(troop(24, "Skeletons", 1.0f, Archetype::MeleeSquad, 81, 1.0f, 0.5f, 81, 11, 'k')
            .withOffsets({ {0.0f, 0.0f}, {0.6f, 0.0f}, {-0.6f, 0.0f} }));

        // Death-bomb radius/damage below aren't part of the sourced stats
        // data -- reasonable engine-internal constants, same caveat as
        // splashRadius/shieldHp.
        add(troop(39, "Giant Skeleton", 6.0f, Archetype::MeleeSquad, 3361, 0.4f, 0.8f, 276, 13, 'J')
            .withDeathEffect(std::make_shared<AreaDamageOnDeath>(2.0f, 300)).withSightRange(5.0f));

        // Electro Wizard: real attack is an instant zap with no projectile
        // travel time, so this is MeleeSquad-shaped (direct damage) despite
        // its 5.0 range -- archetype names describe *how* damage lands in
        // this engine, not the card's real-world range category. It splits
        // to the 2 closest enemies at once (each takes damage/2), or full
        // damage if only one is in range -- see
        // CombatEntity::findSplitTargets/getCurrentDamage. `damage` here is
        // the full single-target figure (118); getCurrentDamage() halves it
        // when a second target is found. Its 0.5s full stun (per hit *and*
        // on the spawn zap) is a freeze with slowFactor 0, same pattern as
        // Ice Wizard's slow. The spawn zap itself (radius 3, matching the
        // attack's own damage since the real card's deploy zap mirrors its
        // regular hit, same stun) rides withSpawnEffect, firing once at
        // deploy via a delay-0 AreaSpell.
        add(troop(35, "Electro Wizard", 4.0f, Archetype::MeleeSquad, 714, 0.5f, 5.0f, 118, 18, 'z')
            .withTargetsAir()
            .withSplitTargets(2)
            .withOnHit(std::make_shared<FreezeOnHit>(5, 0.0f))
            .withSpawnEffect(3.0f, 118, std::make_shared<FreezeOnHit>(5, 0.0f)));

        // First flying cards -- ground/air targeting mechanism (isFlying/
        // targetsAir) exists purely to support these. Minions/Minion Horde
        // are the same unit at two squad sizes/costs, direct-damage like the
        // game's other small swarms (Goblins, Skeletons); Mega Minion is a
        // single heavier melee flier. All three hit air and ground alike.
        add(troop(41, "Minions", 3.0f, Archetype::MeleeSquad, 230, 0.8f, 2.5f, 107, 12, 'm')
            .withOffsets({ {0.0f, 0.0f}, {0.6f, 0.0f}, {-0.6f, 0.0f} })
            .withFlying().withTargetsAir());

        add(troop(42, "Minion Horde", 5.0f, Archetype::MeleeSquad, 230, 0.8f, 2.5f, 107, 12, 'h')
            .withOffsets({ {-0.6f, -0.3f}, {0.0f, -0.3f}, {0.6f, -0.3f},
                           {-0.6f, 0.3f}, {0.0f, 0.3f}, {0.6f, 0.3f} })
            .withFlying().withTargetsAir());

        add(troop(43, "Mega Minion", 3.0f, Archetype::MeleeSquad, 837, 0.5f, 1.6f, 312, 15, 'F')
            .withFlying().withTargetsAir());

        // === Ranged Troops ===
        // Target: Air & Ground -- every ranged troop below needs its own
        // confirmed Target field before getting .withTargetsAir() (default
        // is ground-only); don't assume "ranged" implies it.
        add(troop(1, "Archers", 3.0f, Archetype::RangedSquad, 304, 0.5f, 5.0f, 112, 9, 'A')
            .withOffsets({ {0.0f, 0.0f}, {1.0f, 0.0f} })
            .withTargetsAir());

        add(troop(6, "Musketeer", 4.0f, Archetype::RangedSquad, 721, 0.5f, 6.0f, 217, 10, 'U')
            .withTargetsAir().withSightRange(6.0f));
        // Bomber deliberately has NO withTargetsAir: the real card is
        // ground-only. Same for Bowler/Sparky/Cannon Cart below.
        add(troop(9, "Bomber", 2.0f, Archetype::RangedSquad, 304, 0.5f, 4.5f, 225, 18, 'b'));
        add(troop(11, "Wizard", 5.0f, Archetype::RangedSquad, 755, 0.5f, 5.5f, 281, 14, 'W')
            .withTargetsAir().withSplash(1.5f));
        add(troop(20, "Dart Goblin", 3.0f, Archetype::RangedSquad, 261, 0.8f, 6.5f, 151, 8, 'd')
            .withTargetsAir().withSightRange(7.5f));
        // Now modeled as a real piercing line (applyLineSplashDamage) --
        // total travel 11.5 (4.0 attack range + 7.5 extra), half-width
        // 1.8 (splashRadius doubles as the line's half-width in line-
        // splash mode). Knockback on every hit target still isn't
        // modeled -- AreaSpell's knockback has no equivalent on the
        // troop-attack path this engine's splash/line-splash share.
        add(troop(22, "Bowler", 5.0f, Archetype::RangedSquad, 2081, 0.4f, 4.0f, 289, 25, 'w')
            .withSplash(1.8f).withLineSplash(11.5f).withSightRange(4.0f));

        add(troop(23, "Spear Goblins", 2.0f, Archetype::RangedSquad, 133, 1.0f, 5.0f, 81, 17, 'S')
            .withTargetsAir()
            .withOffsets({ {0.0f, 0.0f}, {0.7f, 0.0f}, {-0.7f, 0.0f} }));

        // Ice Wizard: genuinely ranged -- fires a projectile that applies the
        // freeze on arrival. (Originally this was mislabeled: it inherited
        // from RangedTroop but overrode performAttack to hit instantly,
        // bypassing the projectile entirely. Fixed now that on-hit effects
        // can ride along with a projectile instead of firing at launch.)
        add(troop(34, "Ice Wizard", 3.0f, Archetype::RangedSquad, 689, 0.5f, 5.5f, 90, 17, 'i')
            .withTargetsAir()
            .withOnHit(std::make_shared<FreezeOnHit>(30, 0.65f)));

        // Executioner's axe hits on arrival, then again 1.5s (15 ticks)
        // later on its "return" -- the real axe pierces everyone along its
        // path both ways; this engine has no line-collision primitive, so
        // it hits just the original target twice instead (see
        // Projectile's returnsToSender).
        add(troop(36, "Executioner", 5.0f, Archetype::RangedSquad, 1280, 0.4f, 4.5f, 179, 24, 'x')
            .withTargetsAir()
            .withBoomerang(15));

        add(troop(44, "Baby Dragon", 4.0f, Archetype::RangedSquad, 1152, 0.8f, 3.5f, 168, 15, 'y')
            .withFlying().withTargetsAir()
            .withSplash(1.5f));

        // === Building Targeters ===
        add(troop(2, "Giant", 5.0f, Archetype::MeleeBuildingTargeter, 3968, 0.3f, 1.2f, 253, 15, 'G').withSightRange(7.5f));

        add(troop(15, "Hog Rider", 4.0f, Archetype::MeleeBuildingTargeter, 1697, 0.8f, 0.8f, 317, 16, 'H')
            .withIgnoresRiver().withSightRange(9.5f));

        // Golem splits into two Golemites on death AND deals its own
        // death-explosion damage -- two death effects composed via
        // CompositeDeathEffect, since deathEffect is a single slot.
        add(troop(19, "Golem", 8.0f, Archetype::MeleeBuildingTargeter, 5120, 0.2f, 0.75f, 312, 25, 'L')
            .withDeathEffect(std::make_shared<CompositeDeathEffect>(
                std::vector<std::shared_ptr<IDeathEffect>>{
                    std::make_shared<SpawnOnDeath>(golemiteStats()),
                    std::make_shared<AreaDamageOnDeath>(2.5f, 200) }))
            .withSightRange(7.0f));

        // Ice Golem: same story as Ice Wizard -- BuildingTargeter's own
        // performAttack is already direct damage, so this is behavior-exact.
        // Explodes on death dealing area damage; the real explosion also
        // slows everyone it hits, but AreaDamageOnDeath is damage-only
        // (no onHit-style hook, unlike AreaSpell's spellOnHit) -- not
        // modeled.
        add(troop(40, "Ice Golem", 2.0f, Archetype::MeleeBuildingTargeter, 1315, 0.4f, 0.75f, 84, 25, 'c')
            .withOnHit(std::make_shared<FreezeOnHit>(30, 0.65f))
            .withDeathEffect(std::make_shared<AreaDamageOnDeath>(2.0f, 84)).withSightRange(7.0f));

        // Balloon: flying, buildings-only, real point-blank 0.1 attack range
        // (BuildingTargeter's own findTarget already never considers
        // isFlying/targetsAir, since Buildings never fly -- so no
        // .withTargetsAir() needed here, matching that Balloon can't hit air
        // in the real game either). Real death-explosion damage is
        // conditional (only if shot down before dropping its bomb) --
        // that condition isn't modeled, it always deals the death-explosion
        // here regardless of cause of death, a minor over-approximation.
        add(troop(45, "Balloon", 5.0f, Archetype::MeleeBuildingTargeter, 1676, 0.5f, 0.1f, 640, 20, 'a')
            .withFlying()
            .withDeathEffect(std::make_shared<AreaDamageOnDeath>(1.5f, 240)).withSightRange(7.7f));

        // === Ranged Building Targeter ===
        add(troop(18, "Royal Giant", 6.0f, Archetype::RangedBuildingTargeter, 3164, 0.3f, 5.0f, 307, 18, 'Y').withSightRange(7.5f));

        // === Defensive Structures ===
        add(building(25, "Cannon", 3.0f, 824, 'C', 5.5f, 202, 10));
        add(building(26, "Tesla", 4.0f, 1182, 'T', 5.5f, 220, 11).withTargetsAir());
        add(building(27, "Bomb Tower", 4.0f, 1356, 'D', 6.0f, 222, 18));
        // Inferno Tower: damage is 5% of max for the first 2 seconds (20
        // ticks), 18.75% for the next 2 (tick 20-40), then full damage --
        // reset by a target switch or a stun (Zap/Freeze/etc, already
        // covered generically since ramp resets on any freeze application).
        // `damage` itself is the fully-ramped max (847); getCurrentDamage()
        // scales it down early in the ramp.
        add(building(28, "Inferno Tower", 5.0f, 1748, 'I', 6.0f, 847, 4)
            .withTargetsAir()
            .withDamageRamp(20, 40, 0.05f, 0.1875f).withSightRange(6.0f));

        // === Spells ===
        // Arrows: 3 rapid volleys of 123 each (369 total), not one 369 lump.
        add(spell(3, "Arrows", 3.0f, 3.5f, 123, 10, '*').withRepeats(3, 2));
        add(spell(7, "Fireball", 4.0f, 2.5f, 689, 10, 'O')
            .withKnockback(1.0f));
        add(spell(29, "Zap", 2.0f, 2.5f, 192, 3, 'Z'));
        add(spell(30, "Rocket", 6.0f, 2.0f, 1485, 15, 'r')
            .withKnockback(1.0f));
        add(spell(31, "Lightning", 6.0f, 3.5f, 1057, 5, 'j'));
        // Poison: a real damage-over-time cloud -- 92 damage once per second
        // (10 ticks) for 8 seconds, not 736 dumped in one lump. Whoever's
        // actually standing in it gets hit each tick; walking out mid-cloud
        // stops further damage, same as the real spell.
        add(spell(32, "Poison", 4.0f, 3.5f, 92, 0, 'n').withRepeats(8, 10));
        // The Log is a ground-control spell -- unlike every other spell here,
        // it rolls along the arena floor and can't hit flying units.
        add(spell(33, "The Log", 2.0f, 3.9f, 269, 8, 'o').withGroundOnly());

        // ====================================================================
        // 2026 roster expansion (ids 46+): every remaining non-Champion,
        // non-Evolution, non-Tower-Troop card. Same real-game data sync as
        // above (cost/hp/damage/hit-speed/range/count; speed categories are
        // this engine's own approximation, same as every card before this
        // point). Mechanics with no existing hook in this engine (shields,
        // charge/dash bonus damage, periodic spawning while alive, curses,
        // buffs, invisibility, pull effects, splash on troop attacks,
        // multi-unit "compound" cards where a second independently-targeting
        // sub-unit rides along) are NOT modeled -- each card below notes
        // what's missing, same convention as Golem/Ice Golem/Balloon's
        // on-death splash above. Cards that are *entirely* built around a
        // mechanic this engine has no way to approximate at all (buff-only
        // spells, spell-spawns-troops, replay-last-card, duplicate-troops,
        // compound cards with no single "primary" unit) are skipped outright
        // -- see the excluded-cards note at the very end of this file.
        //
        // Single-character `symbol` is purely cosmetic (TerminalRenderer/
        // GameLogger display only -- combat logic keys everything off
        // cardId/id, see Entity::cardId). This engine has exactly 94
        // printable ASCII characters to draw from and 62 new cards to draw
        // with, so the last 14 cards below (ids 94-107) intentionally reuse
        // an earlier new card's symbol rather than leaving them undefined --
        // a harmless display-only overlap, not a data collision.

        // === New Melee Troops ===
        // Dark Prince crosses the river directly (one of the "jumpers"),
        // not routed through a bridge -- see withIgnoresRiver.
        add(troop(46, "Dark Prince", 4.0f, Archetype::MeleeSquad, 1200, 0.5f, 1.2f, 266, 14, 'N')
            .withShield(240) // corrected from an unsourced 200 guess
            .withCharge(3.0f, 2.0f) // confirmed: +100% (double) damage on a charging hit
            .withIgnoresRiver());
        // Royal Ghost crosses the river directly, not routed through a
        // bridge -- see withIgnoresRiver.
        add(troop(47, "Royal Ghost", 3.0f, Archetype::MeleeSquad, 1210, 0.7f, 1.2f, 261, 18, 'Q')
            .withInvisibility(5) // brief reveal window after attacking
            .withIgnoresRiver());
        // Deploy slam via the existing one-time spawn-effect burst (radius
        // 1.3, damage 430). Periodic jump now modeled too, reusing
        // Fisherman's pullToward primitive to instantly close from
        // 3.5-5 tiles away instead of walking in, landing a ~2x-damage,
        // 2.2-radius splash hit (537 confirmed vs. 268 normal -- ratio
        // ~2.003, rounds to the same 2.0 multiplier convention already
        // used for every other charge card). Confirmed: the jump can carry
        // him over the river onto a target on the other side. The jump
        // itself only travels dist-minus-effectiveAttackRange (stopping
        // just inside melee range of the target, not landing exactly on
        // it) -- close to but not always past the river's own 2-tile
        // width, and this engine's own end-of-update clamp would otherwise
        // shove him back to the near edge if he lands short of fully
        // clearing it. Same river-crossing approximation as Bandit/Boss
        // Bandit (this engine has no discrete "currently jumping" movement
        // state to gate river-ignoring on more precisely than "always") --
        // see Bandit's own registry comment.
        add(troop(48, "Mega Knight", 7.0f, Archetype::MeleeSquad, 3993, 0.5f, 1.2f, 268, 17, 'X')
            .withSplash(1.5f)
            .withSpawnEffect(1.3f, 430)
            .withJump(3.5f, 5.0f, 2.0f, 2.2f)
            .withIgnoresRiver());
        // Confirmed: real heal lands as 4 pulses of 25.5 (102 total) per
        // attack cycle -- collapsed here into this engine's single
        // heal-on-landed-hit model as one 102 lump, corrected from an
        // unsourced 60 guess. A separate, larger heal burst on deployment
        // (202 total) isn't modeled -- no heal-on-spawn primitive exists
        // (spawnEffect is damage-only).
        // Battle Healer crosses the river directly, not routed through a
        // bridge -- see withIgnoresRiver.
        add(troop(49, "Battle Healer", 4.0f, Archetype::MeleeSquad, 1717, 0.5f, 1.2f, 148, 15, 'f')
            .withHealAura(3.0f, 102)
            .withIgnoresRiver());
        // Bandit's river-crossing is only via her dash (confirmed: she
        // isn't a permanent river-crosser like Prince/Hog Rider -- her
        // charge can carry her straight across it toward a target, instead
        // of routing to a bridge). This engine has no separate "currently
        // mid-dash" movement state to gate that precisely on (charge here
        // only tracks bonus-damage progress, not a discrete dash phase),
        // so this collapses to always-ignoring the river as a documented
        // approximation -- same caveat category as splashRadius/shieldHp.
        add(troop(50, "Bandit", 3.0f, Archetype::MeleeSquad, 906, 0.8f, 1.0f, 194, 10, 'u')
            .withCharge(3.0f, 2.0f)
            .withChargeInvulnerability() // confirmed: fully invulnerable while charging in
            .withSightRange(6.0f)
            .withIgnoresRiver());
        // Confirmed: the base card has no self-heal at all -- "heal on
        // attack" only exists as an optional Epic modifier card, not part
        // of standard Berserker. enrageHealPerHit 0 keeps the attack-speed
        // ramp-up (real, confirmed) while dropping the heal this file used
        // to guess at.
        add(troop(51, "Berserker", 2.0f, Archetype::MeleeSquad, 896, 0.7f, 1.0f, 102, 6, 'v')
            .withEnrage(0));
        add(troop(52, "Miner", 3.0f, Archetype::MeleeSquad, 1210, 0.7f, 1.0f, 194, 13, '0')
            .withDeployAnywhere());
        add(troop(53, "Fisherman", 3.0f, Archetype::MeleeSquad, 870, 0.5f, 1.0f, 194, 13, '1')
            .withHook(6.5f).withSightRange(7.5f));
        // Confirmed: interval is exactly 3.5s (35 ticks, already correct).
        // Real parry reflects 200% of the attacker's own damage back at
        // them and only triggers against ground melee hits -- neither is
        // modeled: takeDamage() carries no attacker identity to reflect
        // at (would need a signature change touching every takeDamage
        // call site in the engine), and no melee-vs-ranged distinction
        // exists anywhere here. Stays a "negates the hit" approximation.
        add(troop(54, "Ronin", 5.0f, Archetype::MeleeSquad, 1779, 0.7f, 1.2f, 371, 14, '2')
            .withParry(35));
        add(troop(55, "Goblin Machine", 5.0f, Archetype::MeleeSquad, 2150, 0.5f, 1.2f, 212, 12, '3')
            .withSecondaryUnit(goblinMachineTurretStats())); // independently-targeting rocket turret

        // Inferno Dragon / Electro Dragon: same "instant zap, no projectile
        // travel time" shaping as Electro Wizard above -- MeleeSquad despite
        // real ranged-looking numbers, because archetype describes *how*
        // damage lands here, not the card's on-screen range.
        // Inferno Dragon's beam ramps 35 -> 120 -> 422 damage every 1.5s
        // (15 ticks) locked onto one target, same ramp mechanism as Inferno
        // Tower; `damage` stores the fully-ramped max.
        add(troop(56, "Inferno Dragon", 4.0f, Archetype::MeleeSquad, 1295, 0.5f, 5.0f, 422, 4, '4')
            .withFlying().withTargetsAir()
            .withDamageRamp(15, 30, 0.083f, 0.284f));
        // Confirmed: not splash -- a true chain hitting up to 3 total
        // distinct targets (original + 2 more, each hop up to 4 tiles),
        // each taking the FULL 192 damage independently (not divided).
        add(troop(57, "Electro Dragon", 5.0f, Archetype::MeleeSquad, 1049, 0.5f, 3.5f, 192, 21, '5')
            .withFlying().withTargetsAir()
            .withSplitTargets(3)
            .withSplitTargetsFullDamage()
            .withOnHit(std::make_shared<FreezeOnHit>(5, 0.0f)));

        // Night Witch: periodic bat-spawning while alive isn't modeled (no
        // such hook exists), but her on-death release of 3 Bats reuses the
        // same SpawnOnDeath mechanism as Golem/Battle Ram, with the newly-
        // registered Bats card's own stats (see nightWitchBatStats above).
        add(troop(58, "Night Witch", 4.0f, Archetype::MeleeSquad, 906, 0.5f, 1.0f, 314, 13, '6')
            .withDeathEffect(std::make_shared<SpawnOnDeath>(nightWitchBatStats()))
            .withPeriodicEffect(50, std::make_shared<PeriodicSpawnEffect>(nightWitchPeriodicBatStats())));
        // Phoenix: one-time revive on death -- see phoenixReviveStats above.
        add(troop(59, "Phoenix", 4.0f, Archetype::MeleeSquad, 1052, 0.5f, 1.0f, 217, 10, '7')
            .withFlying().withTargetsAir()
            .withDeathEffect(std::make_shared<SpawnOnDeath>(phoenixReviveStats())));

        // === New Ranged Troops ===
        add(troop(60, "Sparky", 6.0f, Archetype::RangedSquad, 1451, 0.3f, 5.0f, 1331, 40, '8')
            .withSplash(1.5f)
            .withStunResetsCooldown() // confirmed: any stun fully restarts her charge, doesn't just slow it
            .withSightRange(5.0f));
        add(troop(61, "Princess", 3.0f, Archetype::RangedSquad, 261, 0.5f, 9.0f, 168, 30, '9')
            .withTargetsAir()
            .withSplash(1.5f).withSightRange(9.5f));
        // No sourced falloff formula exists (confirmed: 10 fixed-damage
        // pellets in a random, unquantified spread -- "weaker at range" is
        // really "fewer pellets statistically land," not a per-pellet
        // damage curve). withRangeFalloff below is this project's own
        // invented approximation of that qualitative behavior, NOT a
        // sourced number -- deliberately deterministic (damage scales
        // linearly from full at point-blank to half at max range) rather
        // than simulating actual random pellets, matching every other
        // mechanic in this engine (reproducibility for tests/RL training
        // matters more here than faithfully modeling randomness the real
        // card has but no published curve for).
        add(troop(62, "Hunter", 4.0f, Archetype::RangedSquad, 885, 0.5f, 4.0f, 84, 22, '!')
            .withTargetsAir()
            .withSplash(1.5f)
            .withRangeFalloff(0.5f));
        // Now modeled as a real piercing line: total travel 11.0, half-
        // width 0.25 (0.5 total width) -- thin, not a cone.
        add(troop(63, "Magic Archer", 4.0f, Archetype::RangedSquad, 529, 0.5f, 7.0f, 143, 11, '#')
            .withTargetsAir()
            .withSplash(0.25f).withLineSplash(11.0f).withSightRange(7.5f));
        add(troop(64, "Firecracker", 3.0f, Archetype::RangedSquad, 304, 0.7f, 6.0f, 64, 30, '$')
            .withTargetsAir()
            .withSplash(1.5f)
            .withRecoil(1.0f).withSightRange(8.5f)); // confirmed: kicks back 1 tile after every attack
        add(troop(65, "Skeleton Dragons", 4.0f, Archetype::RangedSquad, 560, 0.7f, 3.5f, 151, 20, '%')
            .withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f} }).withFlying().withTargetsAir()
            .withSplash(1.5f));
        // Transform now modeled via transformKillsSelf + SpawnOnDeath
        // (see goblinDemolisherKamikazeStats above) -- at <=50% hp this
        // entity kills itself and the kamikaze form spawns in its place
        // at the same position, on the same team.
        add(troop(66, "Goblin Demolisher", 4.0f, Archetype::RangedSquad, 1300, 0.5f, 5.0f, 186, 11, '&')
            .withSplash(1.5f)
            .withHpTransformIntoDeath(0.5f, std::make_shared<SpawnOnDeath>(goblinDemolisherKamikazeStats())));
        add(troop(67, "Flying Machine", 4.0f, Archetype::RangedSquad, 614, 0.7f, 6.0f, 171, 11, '+')
            .withFlying().withTargetsAir().withSightRange(6.0f));
        // Confirmed: the spawned unit is a unique "Cursed Hog" (building-
        // targeter, see cursedHogStats), not a plain Goblin -- now modeled
        // via CursedHogOnHit, which arms a SpawnOnDeathForEnemyTeam on
        // whatever this curses.
        add(troop(68, "Mother Witch", 4.0f, Archetype::RangedSquad, 529, 0.5f, 5.5f, 133, 10, ',')
            .withTargetsAir()
            .withOnHit(std::make_shared<CursedHogOnHit>(1.3f, 60, cursedHogStats())));
        // Correction: no separate "shield" stat actually exists for this
        // card (the earlier 500 guess was wrong) -- it's a single 1809 hp
        // pool that triggers a transform at 50% instead. Confirmed the
        // transformed building form keeps identical damage/range/hit
        // speed to the troop form -- only mobility changes (grounds
        // itself for 15s, then self-destructs), modeled by reusing
        // applyFreeze(ticks, 0.0f) rather than adding a second speed
        // concept -- see CombatEntity::transformBecomesStationary.
        add(troop(69, "Cannon Cart", 5.0f, Archetype::RangedSquad, 1809, 0.5f, 5.5f, 212, 9, '?')
            .withHpTransform(0.5f, 150, true)
            // Sourced sight range differs slightly by form (6.0 mobile,
            // 5.5 once transformed stationary) -- this engine models both
            // forms as one entity that just stops moving, so the mobile
            // form's value is the one that actually matters for chasing.
            .withSightRange(6.0f));
        add(troop(70, "Furnace", 4.0f, Archetype::RangedSquad, 727, 0.5f, 5.5f, 179, 17, '/')
            .withTargetsAir() // reworked 2026 from Building to mobile Troop
            .withPeriodicEffect(70, std::make_shared<PeriodicSpawnEffect>(furnaceFireSpiritStats())));
        add(troop(71, "Witch", 5.0f, Archetype::RangedSquad, 839, 0.5f, 5.5f, 135, 11, ':')
            .withTargetsAir()
            .withPeriodicEffect(70, std::make_shared<PeriodicSpawnEffect>(witchSkeletonStats())));

        // "Spirit" troops: detonate once on arrival then vanish -- see
        // CombatEntity::dieAfterFirstHit (fires the instant the shot is
        // launched for these ranged troops, not on the projectile's later
        // arrival, a minor timing simplification -- see its own comment).
        add(troop(72, "Ice Spirit", 1.0f, Archetype::RangedSquad, 230, 0.85f, 2.5f, 110, 10, ';')
            .withTargetsAir()
            .withOnHit(std::make_shared<FreezeOnHit>(10, 0.5f))
            .withDieAfterFirstHit());
        add(troop(73, "Fire Spirit", 1.0f, Archetype::RangedSquad, 230, 0.85f, 2.5f, 207, 10, '<')
            .withTargetsAir()
            .withDieAfterFirstHit()); // splash not modeled
        // Confirmed: real heal is 4 pulses of 100.25 (401 total, not a
        // single 110 burst) -- collapsed into one lump on this kamikaze's
        // single landed hit, corrected from an unsourced 110 guess.
        add(troop(74, "Heal Spirit", 1.0f, Archetype::RangedSquad, 230, 0.85f, 2.5f, 110, 10, '=')
            .withTargetsAir()
            .withHealAura(2.5f, 401)
            .withDieAfterFirstHit());
        add(troop(75, "Electro Spirit", 1.0f, Archetype::RangedSquad, 230, 0.85f, 2.5f, 99, 10, '>')
            .withTargetsAir()
            .withSplitTargets(2)
            .withOnHit(std::make_shared<FreezeOnHit>(8, 0.0f))
            .withDieAfterFirstHit());

        // === New Swarms ===
        // Shield HP: Guards/Royal Recruits confirmed at 256/240 (corrected
        // from earlier unsourced 65/52 guesses); Cannon Cart's "shield" was
        // found to not be a real distinct stat at all (see its own entry).
        add(troop(76, "Guards", 3.0f, Archetype::MeleeSquad, 81, 0.7f, 1.0f, 117, 10, '?')
            .withOffsets({ {0.0f, 0.0f}, {0.6f, 0.0f}, {-0.6f, 0.0f} })
            .withShield(256));
        add(troop(77, "Royal Recruits", 7.0f, Archetype::MeleeSquad, 547, 0.5f, 1.0f, 133, 13, '@')
            .withOffsets({ {-2.5f, 0.0f}, {-1.5f, 0.0f}, {-0.5f, 0.0f}, {0.5f, 0.0f}, {1.5f, 0.0f}, {2.5f, 0.0f} })
            .withShield(240));
        add(troop(78, "Bats", 2.0f, Archetype::MeleeSquad, 81, 0.85f, 1.0f, 81, 12, 't')
            .withOffsets({ {0.0f, 0.0f}, {0.6f, 0.0f}, {-0.6f, 0.0f}, {0.3f, 0.5f}, {-0.3f, 0.5f} })
            .withFlying().withTargetsAir());

        add(troop(79, "Zappies", 4.0f, Archetype::RangedSquad, 529, 0.5f, 4.5f, 117, 21, '[')
            .withOffsets({ {0.0f, 0.0f}, {0.6f, 0.0f}, {-0.6f, 0.0f} })
            .withTargetsAir()
            .withOnHit(std::make_shared<FreezeOnHit>(3, 0.0f)).withSightRange(5.0f));
        add(troop(80, "Three Musketeers", 9.0f, Archetype::RangedSquad, 722, 0.5f, 6.0f, 218, 10, ']')
            .withOffsets({ {-2.0f, 0.0f}, {0.0f, 0.0f}, {2.0f, 0.0f} })
            .withTargetsAir());

        // === New Building Targeters ===
        // Battle Ram: releases 2 Barbarians on death (reuses the already-
        // sourced Barbarians card's own stats, see battleRamBarbarianStats).
        add(troop(81, "Battle Ram", 4.0f, Archetype::MeleeBuildingTargeter, 691, 0.6f, 1.0f, 192, 14, '^')
            .withDeathEffect(std::make_shared<SpawnOnDeath>(battleRamBarbarianStats()))
            .withCharge(3.0f, 2.0f));
        // Royal Hogs cross the river directly (one of the "jumpers"), not
        // routed through a bridge -- see withIgnoresRiver.
        add(troop(82, "Royal Hogs", 5.0f, Archetype::MeleeBuildingTargeter, 837, 0.85f, 1.0f, 74, 12, '_')
            .withOffsets({ {-1.0f, -0.3f}, {-0.3f, 0.3f}, {0.3f, -0.3f}, {1.0f, 0.3f} })
            .withCharge(3.0f, 2.0f).withSightRange(9.5f)
            .withIgnoresRiver());
        add(troop(83, "Wall Breakers", 2.0f, Archetype::MeleeBuildingTargeter, 330, 0.85f, 1.0f, 350, 12, '{')
            .withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f} })
            .withSplash(1.5f)
            .withDieAfterFirstHit().withSightRange(7.0f));
        add(troop(84, "Electro Giant", 7.0f, Archetype::MeleeBuildingTargeter, 3952, 0.3f, 1.0f, 163, 18, '|')
            .withSplash(1.5f)
            .withPeriodicEffect(50, std::make_shared<AreaStunEffect>(2.5f, 5)));
        add(troop(85, "Suspicious Bush", 2.0f, Archetype::MeleeBuildingTargeter, 81, 0.5f, 0.25f, 256, 14, '}')
            .withInvisibility(5)
            .withDeathEffect(std::make_shared<SpawnOnDeath>(suspiciousBushGoblinStats())));
        add(troop(86, "Rune Giant", 4.0f, Archetype::MeleeBuildingTargeter, 2662, 0.5f, 1.2f, 153, 15, '~')
            .withAllyBuffAura(3.0f, 3, 1.5f, 50, 2));
        // Ram Rider crosses the river directly (one of the "jumpers"), not
        // routed through a bridge -- see withIgnoresRiver.
        add(troop(87, "Ram Rider", 5.0f, Archetype::MeleeBuildingTargeter, 1766, 0.5f, 1.0f, 250, 17, '"')
            .withCharge(3.0f, 2.0f)
            .withSecondaryUnit(ramRiderCrossbowStats()) // rider's independently-targeting crossbow (5.5 sight, its own default)
            .withSightRange(7.5f) // the "ram" component itself
            .withIgnoresRiver());
        add(troop(88, "Goblin Giant", 6.0f, Archetype::MeleeBuildingTargeter, 3110, 0.5f, 1.2f, 176, 15, '`')
            .withSecondaryUnit(goblinGiantSpearGoblinsStats()).withSightRange(7.5f)); // carried Spear Goblins, independently-targeting
        // Skeleton Barrel: releases 2 Skeletons on death (reuses the
        // already-sourced Skeletons card's own stats).
        add(troop(89, "Skeleton Barrel", 3.0f, Archetype::MeleeBuildingTargeter, 532, 0.85f, 1.0f, 81, 10, ',')
            .withFlying()
            .withDeathEffect(std::make_shared<SpawnOnDeath>(skeletonBarrelSkeletonStats())).withSightRange(7.7f));
        // Full split chain now modeled: Golem -> 2 Golemites (each -> 2
        // Blobs) + 1.0 elixir to the opponent; see elixirGolemiteStats/
        // elixirBlobStats above for the rest of the chain and their own
        // elixir grants (1.0 + 1.0 + 2.0 = 4.0 total if the whole thing
        // is killed).
        add(troop(90, "Elixir Golem", 3.0f, Archetype::MeleeBuildingTargeter, 1569, 0.3f, 1.0f, 253, 11, '(')
            .withDeathEffect(std::make_shared<CompositeDeathEffect>(
                std::vector<std::shared_ptr<IDeathEffect>>{
                    std::make_shared<SpawnOnDeath>(
                        elixirGolemiteStats().withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f} })),
                    std::make_shared<EnemyElixirGrantOnDeath>(1.0f)
                }))
            .withSightRange(7.5f));

        // === New Ranged Building Targeter ===
        // Death spawn confirmed and now modeled: 6 Lava Pups, spread out
        // (no documented exact geometric pattern -- lavaHoundPupStats uses
        // a reasonable hexagonal spread as an engine-internal choice).
        add(troop(91, "Lava Hound", 7.0f, Archetype::RangedBuildingTargeter, 3581, 0.3f, 3.5f, 53, 13, ')')
            .withFlying()
            .withDeathEffect(std::make_shared<SpawnOnDeath>(lavaHoundPupStats())));

        // === New Defensive Structures ===
        // 30s lifespan: already correct for free -- every Building here
        // defaults to a 300-tick (30s) decay lifetime (see Building's own
        // constructor), which happens to match X-Bow's real lifespan
        // exactly. The slow ~3.5s lock-on before its first shot is new,
        // via withDeployDelay.
        add(building(92, "X-Bow", 6.0f, 1600, 'P', 11.5f, 43, 3)
            .withDeployDelay(35).withSightRange(11.5f));
        add(building(93, "Mortar", 4.0f, 1369, 'R', 11.5f, 266, 50)
            .withMinRange(3.5f).withSightRange(11.5f)); // confirmed blind spot

        // Spawner buildings: none of these attack in the real game (see
        // ClashStrategic's own data flagging their damage fields as
        // vestigial) -- modeled as inert 0-damage structures that
        // periodically spawn troops via PeriodicSpawnEffect (see the
        // periodic-spawn child-stats helpers above). Goblin Cage has no
        // periodic spawn in the real game either (its whole function is a
        // release-on-death burst) -- stays unmodeled, no sourced stats for
        // Goblin Brawler.
        // Confirmed: 3-per-pulse interval is 15s (150 ticks), not 14 --
        // plus a 1-Barbarian release on the Hut's own death.
        add(building(94, "Barbarian Hut", 6.0f, 1164, '2', 0.0f, 0, 100)
            .withPeriodicEffect(150, std::make_shared<PeriodicSpawnEffect>(barbarianHutBarbarianStats()))
            .withDeathEffect(std::make_shared<SpawnOnDeath>(barbarianHutDeathBarbarianStats())));
        // Mechanism correction (not just numbers): the real card doesn't
        // spawn on an unconditional timer -- it only fires while an enemy
        // is within its 6-tile range, every 2.2s (22 ticks), via
        // ProximityGatedPeriodicSpawnEffect. Also releases 1 Spear Goblin
        // on the Hut's own death. HP corrected 1180 -> 1228.
        add(building(95, "Goblin Hut", 4.0f, 1228, '3', 0.0f, 0, 100)
            .withPeriodicEffect(22, std::make_shared<ProximityGatedPeriodicSpawnEffect>(goblinHutSpearGoblinStats(), 6.0f))
            .withDeathEffect(std::make_shared<SpawnOnDeath>(goblinHutSpearGoblinStats())));
        // 2-per-pulse already correct (skeletonBarrelSkeletonStats has 2
        // offsets) -- adds a confirmed 4-Skeleton release on the
        // Tombstone's own death (reusing witchSkeletonStats' 4 offsets).
        add(building(96, "Tombstone", 3.0f, 529, '5', 0.0f, 0, 100)
            .withPeriodicEffect(35, std::make_shared<PeriodicSpawnEffect>(skeletonBarrelSkeletonStats()))
            .withDeathEffect(std::make_shared<SpawnOnDeath>(witchSkeletonStats())));
        // Goblin Cage's entire function is this release-on-death burst --
        // now modeled (1 Goblin Brawler).
        add(building(97, "Goblin Cage", 4.0f, 780, '6', 0.0f, 0, 100)
            .withDeathEffect(std::make_shared<SpawnOnDeath>(goblinCageBrawlerStats())));
        // Mechanism correction: the "emergence burst" isn't extra
        // Goblins -- it's a one-time 360 deploy-impact burst (modeled via
        // the existing spawn-effect mechanism, knockback component not
        // modeled). Death spawn corrected from 3 to 2 Goblins (2021
        // balance patch).
        add(building(98, "Goblin Drill", 4.0f, 1313, '7', 0.0f, 0, 100)
            .withPeriodicEffect(30, std::make_shared<PeriodicSpawnEffect>(goblinDrillGoblinStats()))
            .withDeployAnywhere()
            .withSpawnEffect(2.0f, 84)
            .withDeathEffect(std::make_shared<SpawnOnDeath>(goblinDrillDeathGoblinStats())));
        // Elixir Collector: no attack, but does passively generate elixir
        // via ElixirGrantEffect (a periodic effect that credits
        // Board::pendingElixirGrant instead of spawning a troop). Interval/
        // amount aren't part of the sourced stats data.
        add(building(99, "Elixir Collector", 6.0f, 1070, '9', 0.0f, 0, 100)
            .withPeriodicEffect(80, std::make_shared<ElixirGrantEffect>(1.0f)));

        // === New Spells ===
        add(spell(100, "Giant Snowball", 2.0f, 2.5f, 179, 8, '!')
            .withSpellOnHit(std::make_shared<FreezeOnHit>(15, 0.5f))
            .withKnockback(1.0f));
        // Barbarian Barrel: rolling damage plus a single Barbarian spawned
        // at the landing point via the same spell-spawn mechanism as
        // Goblin Barrel below.
        add(spell(101, "Barbarian Barrel", 2.0f, 2.5f, 233, 8, '#')
            .withGroundOnly()
            .withSpellSpawn(std::make_shared<PeriodicSpawnEffect>(barbarianBarrelBarbarianStats())));
        // Goblin Curse: damage-over-time portion modeled via withRepeats;
        // the damage-taken-amplification debuff and cursed-death-spawns-a-
        // Goblin bonus aren't (no debuff-modifier or on-cursed-death hook).
        add(spell(102, "Goblin Curse", 2.0f, 3.0f, 43, 8, '$').withRepeats(6, 10));
        // Earthquake: ground-only DoT, same shape as Poison, now with the
        // confirmed -50% movement slow (a 2020 patch removed the
        // attack-speed component entirely, so this is the complete real
        // effect, not a partial model). Bonus damage vs. buildings (3.5x)
        // still isn't modeled -- would need a target-type-conditional
        // damage split AreaSpell doesn't have.
        add(spell(103, "Earthquake", 3.0f, 3.5f, 82, 8, '%').withRepeats(3, 10).withGroundOnly()
            .withSpellOnHit(std::make_shared<FreezeOnHit>(30, 0.5f)));
        // Void: confirmed 3 discrete tiers (not a continuous formula) --
        // 340/hit at 1 target, 160/hit at 2-4, 76/hit at 5+, applied once
        // per wave via the existing withRepeats(3, ...) cadence.
        add(spell(104, "Void", 3.0f, 2.5f, 0, 8, '&').withRepeats(3, 10)
            .withSpellTieredDamage(340, 160, 76));
        // Vines: confirmed HP-based selection -- only the 3 highest-HP
        // enemies in radius are affected, not everyone caught in it.
        // Flyers-pulled-to-ground still isn't modeled (no equivalent to
        // Tornado/hook's pull toward a point, this would need "pull down"
        // as a distinct axis this engine doesn't have).
        add(spell(105, "Vines", 3.0f, 2.5f, 135, 8, '+').withRepeats(3, 7)
            .withSpellTopHpTargets(3));
        add(spell(106, "Tornado", 3.0f, 5.5f, 154, 8, '_')
            .withKnockback(-1.5f)); // negative: pulls toward center instead of pushing away
        // Freeze: the source data's "damage" field for this card is treated
        // as vestigial (same judgment call ClashStrategic's own data made
        // for the non-attacking spawner buildings above) -- the real card's
        // entire function is the stun, modeled here as 0 damage plus a full
        // (slowFactor 0.0) 4s freeze via the new spellOnHit hook.
        add(spell(107, "Freeze", 4.0f, 3.0f, 0, 8, '~')
            .withSpellOnHit(std::make_shared<FreezeOnHit>(40, 0.0f)));
        // Rage: now that AreaSpell supports a buffsAllies mode (see the
        // buff/aura cards above), this no longer needs to stay excluded --
        // buffs allied damage in radius instead of damaging enemies.
        // Duration/multiplier aren't part of the sourced data.
        add(spell(108, "Rage", 2.0f, 3.0f, 0, 8, '(')
            .withSpellBuff(1.35f, 45));
        // Goblin Barrel: drops 3 Goblins directly at the target via
        // AreaSpell::spawnOnDetonate (reusing PeriodicSpawnEffect), no
        // longer excluded now that spell-spawns-troops exists. Real card
        // deals no separate area damage of its own -- damage 0 here, the
        // Goblins' own combat stats are what actually hits.
        add(spell(109, "Goblin Barrel", 3.0f, 0.5f, 0, 8, '[')
            .withSpellSpawn(std::make_shared<PeriodicSpawnEffect>(goblinBarrelGoblinStats())));
        // Graveyard: rains Skeletons over 9s -- one small spawn per tick
        // via withRepeats, same cadence mechanism as Poison's DoT.
        add(spell(110, "Graveyard", 5.0f, 4.0f, 0, 8, ']')
            .withRepeats(9, 10)
            .withSpellSpawn(std::make_shared<PeriodicSpawnEffect>(graveyardSkeletonStats())));
        // Royal Delivery: drops a single shielded Royal-Recruit-shaped
        // defender at the target; also deals its own landing-impact damage.
        add(spell(111, "Royal Delivery", 3.0f, 2.0f, 438, 8, '^')
            .withSpellSpawn(std::make_shared<PeriodicSpawnEffect>(royalDeliveryRecruitStats())));

        // === New Compound Troops (two equal-weight sub-units) ===
        // Goblin Gang / Rascals: now that secondaryUnit exists (see the
        // Goblin Machine/Ram Rider/Goblin Giant cards above), these no
        // longer need one sub-unit to dominate -- both halves are simply
        // spawned via primary + secondaryUnit.
        add(troop(112, "Goblin Gang", 3.0f, Archetype::MeleeSquad, 202, 1.0f, 0.5f, 120, 11, 'g')
            .withOffsets({ {-0.5f, -0.5f}, {0.5f, -0.5f}, {0.0f, 0.5f} })
            .withSecondaryUnit(
                troop(-27, "Spear Goblins", 0.0f, Archetype::RangedSquad, 133, 1.0f, 5.0f, 81, 17, 'S')
                    .withOffsets({ {-0.5f, 0.5f}, {0.5f, 0.5f}, {0.0f, -0.5f} })));
        // Rascals: only the "Boy" half's stats are separately sourced; the
        // two "Girls" reuse Spear Goblins' own sourced ranged stats as a
        // reasonable stand-in rather than an invented split of the card's
        // one combined figure.
        add(troop(113, "Rascals", 5.0f, Archetype::MeleeSquad, 1832, 0.5f, 1.2f, 217, 15, 'X')
            .withSecondaryUnit(
                troop(-28, "Rascals", 0.0f, Archetype::RangedSquad, 133, 0.5f, 5.0f, 81, 17, 'r')
                    .withOffsets({ {-0.5f, 0.0f}, {0.5f, 0.0f} })));

        // Clone: now that Entity::clone() exists (see MeleeTroop/
        // RangedTroop/BuildingTargeter/RangedBuildingTargeter's own
        // overrides), this no longer needs to stay excluded -- duplicates
        // every allied troop in radius, each clone getting a fresh id and
        // 1 hp but every other configured combat field of the original.
        add(spell(114, "Clone", 3.0f, 3.0f, 0, 8, ')')
            .withSpellClone());

        // === Champions ===
        // First Champion implemented in this engine -- previously entirely
        // out of scope (see the "Excluded from this sync" note below,
        // updated accordingly). Mighty Miner is plain MeleeSquad, identical
        // archetype/targeting to the regular Miner (52) -- ground only, no
        // building restriction -- but, confirmed, does NOT deploy anywhere
        // like the regular Miner does: .withDeployAnywhere() is
        // deliberately omitted, so this stays on the caster's own half like
        // any ordinary troop.
        //
        // 3-stage damage ramp (40 -> 204 -> 409) while locked onto one
        // target, same ramp mechanism as Inferno Dragon/Inferno Tower --
        // `damage` stores the fully-ramped max (409). Stage-transition
        // timing is now sourced (Liquipedia version history): "Time
        // required to change stages" is 2 seconds per stage as of the
        // 2025-01-08 balance patch (down from 2.25s, itself up from an
        // original 2s in 2023-08-08) -- i.e. 2s to reach stage 2, another 2s
        // (4s total) to reach stage 3/max. At this engine's 10-ticks/second
        // rate that's mid=20 ticks, full=40 ticks -- previously a 15/30
        // placeholder borrowed from Inferno Dragon's own (different)
        // cadence, now corrected.
        //
        // Activated ability "Explosive Escape" (1 elixir, ~13s/130-tick
        // cooldown -- see CombatEntity::abilityElixirCost/
        // abilityCooldownTicks): teleports to the horizontally-mirrored
        // position across the board's center line (a lane swap, same Y)
        // and leaves a bomb at the ORIGINAL position that detonates after
        // ~1s (10 ticks) for 332 damage (tournament standard), hitting
        // ground and air alike, with a 1.8-tile knockback. Bomb radius
        // (2.5) isn't part of the sourced data -- a reasonable
        // engine-internal geometry constant, same caveat as splashRadius/
        // shieldHp elsewhere in this file. See MightyMinerEscapeEffect.
        add(troop(115, "Mighty Miner", 4.0f, Archetype::MeleeSquad, 2250, 0.5f, 1.6f, 409, 4, '\'')
            .withDamageRamp(20, 40, 40.0f / 409.0f, 204.0f / 409.0f)
            .withChampionAbility(1.0f, 130, std::make_shared<MightyMinerEscapeEffect>(2.5f, 332, 10, 1.8f)));

        // Golden Knight: "Dashing Dash" -- chain-dashes to the nearest
        // enemy within 5.5 tiles, up to 10 times, stopping at a Crown
        // Tower. See GoldenKnightDashEffect for the "resolved in one tick"
        // timing simplification.
        add(troop(116, "Golden Knight", 4.0f, Archetype::MeleeSquad, 1799, 0.5f, 1.2f, 161, 9, 'k')
            .withChampionAbility(1.0f, 120, std::make_shared<GoldenKnightDashEffect>(335, 5.5f, 10)));

        // Skeleton King: passively collects a "soul" (up to 10) whenever
        // ANY unit dies within soulCollectionRadius (not part of the
        // sourced data -- a reasonable engine-internal constant, same
        // caveat as splashRadius elsewhere); "Soul Summoning" spawns 6 + 1
        // per collected soul (6-16 total), consuming them.
        add(troop(117, "Skeleton King", 4.0f, Archetype::MeleeSquad, 2298, 0.5f, 1.2f, 180, 16, 'K')
            .withSplash(1.3f)
            .withSoulCollection(5.0f, 10)
            .withChampionAbility(2.0f, 200,
                std::make_shared<SkeletonKingSoulSummonEffect>(skeletonKingSummonedSkeletonStats(), 6, 3.5f)));

        // Archer Queen: "Cloaking Cape" -- untargetable + ~2.8x attack
        // speed (180% increase) for 3.5s. Movement-speed drop not modeled
        // -- see ArcherQueenCloakEffect's own comment.
        add(troop(118, "Archer Queen", 5.0f, Archetype::RangedSquad, 1000, 0.5f, 5.0f, 225, 12, 'Q')
            .withTargetsAir()
            .withChampionAbility(1.0f, 170, std::make_shared<ArcherQueenCloakEffect>(35, 1.0f / 2.8f)));

        // Monk: 3-hit-combo (normal hits + a bonus-damage/knockback 3rd
        // hit) isn't modeled -- flat per-hit damage instead, a documented
        // simplification given this is a base-attack nuance separate from
        // his actual Champion ability. "Pensive Protection" (65% damage
        // reduction for 4s) reuses applyCurse -- see MonkDeflectEffect's
        // own comment for why, and for what else isn't modeled (projectile
        // reflection, knockback immunity).
        add(troop(119, "Monk", 4.0f, Archetype::MeleeSquad, 2214, 0.5f, 1.2f, 140, 8, 'M')
            .withChampionAbility(1.0f, 170, std::make_shared<MonkDeflectEffect>(0.35f, 40)));

        // Little Prince: hit-speed ramps 1.2s -> 0.8s -> 0.4s while
        // locked onto the same target (3 attacks per stage, approximated
        // via tick-equivalent thresholds -- see CombatEntity::
        // hitSpeedRampMidTick's own comment for how this differs from the
        // damage-ramp mechanism). "Royal Rescue" summons the Guardienne
        // (see guardienneStats above).
        add(troop(120, "Little Prince", 3.0f, Archetype::RangedSquad, 698, 0.5f, 5.5f, 104, 12, 'p')
            .withTargetsAir()
            .withHitSpeedRamp(36, 60, 8.0f / 12.0f, 4.0f / 12.0f)
            .withChampionAbility(3.0f, 300, std::make_shared<SpawnOnAbility>(guardienneStats())));

        // Goblinstein: compound card, Monster (front, building-only tank,
        // carries the Champion ability) + Doctor (secondary, 3 tiles
        // behind -- fixed offset regardless of team, same simplification
        // already used for every other compound card's secondary unit
        // offset in this file) with a brief full-stun on hit. "Lightning
        // Link" anchors a repeating shock zone at the Monster's position
        // -- see GoblinsteinLightningLinkEffect's own comment for why a
        // fixed-position AreaSpell is actually the more accurate model
        // here, not just a simplification.
        add(troop(121, "Goblinstein", 5.0f, Archetype::MeleeBuildingTargeter, 2385, 0.5f, 1.2f, 128, 15, 'G')
            .withChampionAbility(2.0f, 170, std::make_shared<GoblinsteinLightningLinkEffect>(2.0f, 107, 5, 40))
            .withSecondaryUnit(
                troop(-40, "Goblinstein", 0.0f, Archetype::RangedSquad, 721, 0.5f, 5.5f, 92, 18, 'D')
                    .withOffsets({ {0.0f, 3.0f} })
                    .withTargetsAir()
                    .withOnHit(std::make_shared<FreezeOnHit>(5, 0.0f))));

        // Boss Bandit: same charge mechanism as the regular Bandit (double
        // damage, 3-6 tile trigger window) -- dash invulnerability
        // inferred from the base Bandit's own confirmed mechanic, not
        // independently sourced for this card. Same river-crossing
        // approximation as the regular Bandit -- see that card's own
        // withIgnoresRiver comment. "Getaway Grenade": brief invisibility +
        // an unconditional 6-tile teleport back toward her own side,
        // limited to 2 total uses per deployment (not an infinitely-
        // repeating cooldown like every other Champion here -- see
        // CombatEntity::abilityUsesRemaining); also crosses the river (see
        // BossBanditGetawayGrenadeEffect), since a "getaway" retreating
        // from enemy territory back to her own side routinely needs to.
        add(troop(122, "Boss Bandit", 6.0f, Archetype::MeleeSquad, 2624, 0.7f, 0.8f, 245, 11, 'x')
            .withCharge(3.0f, 2.0f)
            .withChargeInvulnerability()
            .withIgnoresRiver()
            .withChampionAbility(1.0f, 30, std::make_shared<BossBanditGetawayGrenadeEffect>(10, 6.0f), 2));

        // === Evolutions ===
        // Framework pilot: proves the evolution-slot machinery (see
        // CardDefinition::isEvolution/addEvolution and
        // PlayerState::EvolutionSlotState/playCard) end to end with one
        // card before the remaining 40 are batched in. A deck that wants
        // Wall Breakers evolved puts id 123 in a slot instead of 83 (the
        // regular Wall Breakers stays completely unaffected/unchanged).
        //
        // Confirmed (Liquipedia): 2 cycles to unlock, "1 in every 3
        // deploys will be evolved" (2 un-evolved cycles, then 1 evolved
        // play, repeating for the rest of the match -- see addEvolution's
        // own comment on why this isn't a one-time charge). Evolved form,
        // "Powder Barrels" (per a later, more precise source): if a Wall
        // Breaker is defeated, its barrel breaks -- a moderate-damage
        // (reduced vs. Crown Towers, not modeled) explosion in a 1.5-tile
        // radius, modeled via AreaDamageOnDeath -- then a barrel remnant
        // continues rolling to the nearest building at Very Fast speed,
        // dealing 50% of the original damage once it connects (see
        // runnerStats' own comment for that piece). Composed via
        // CompositeDeathEffect, same shape as Lumberjack Evolution above.
        // One Runner PER Wall Breaker unit that dies (there are 2 in the
        // squad), not a 2-Runner burst from a single death, so each of
        // the 2 Wall Breakers just carries its own copy of the same
        // deathEffect. Sources disagreed on whether the evolved Wall
        // Breakers' own walking stats also get a damage buff, so the more
        // conservative reading (unchanged combat stats, only the new
        // death behavior) is used here, not the contested number.
        addEvolution(123,
            troop(83, "Wall Breakers", 2.0f, Archetype::MeleeBuildingTargeter, 330, 0.85f, 1.0f, 350, 12, '{')
                .withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f} })
                .withSplash(1.5f)
                .withDieAfterFirstHit().withSightRange(7.0f),
            troop(83, "Wall Breakers", 2.0f, Archetype::MeleeBuildingTargeter, 330, 0.85f, 1.0f, 350, 12, '{')
                .withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f} })
                .withSplash(1.5f)
                .withDieAfterFirstHit().withSightRange(7.0f)
                .withDeathEffect(std::make_shared<CompositeDeathEffect>(
                    std::vector<std::shared_ptr<IDeathEffect>>{
                        std::make_shared<AreaDamageOnDeath>(1.5f, 150),
                        std::make_shared<SpawnOnDeath>(runnerStats())
                    })),
            2, 1);

        // Zap Evolution: 2 cycles, "1 in every 3 deploys will be evolved"
        // (same repeating pattern). "Triple Shock": zaps the area 3 times
        // instead of once, with the radius growing between zaps -- only
        // the repeat-hit count is modeled (via the same multi-tick
        // AreaSpell mechanism as Poison/Arrows, see withRepeats), not the
        // growing radius, which this engine's AreaSpell has no per-repeat
        // -radius field for. Base Zap has no stun modeled in this engine
        // at all (id 29 above never chains a stun effect), so there's no
        // stun-refresh interaction to model either -- confirmed real-game
        // nuance (resetting charge attacks like Battle Ram/Sparky twice)
        // that's simply inapplicable here. 3-tick interval between zaps
        // is an engine-internal timing choice, not sourced.
        addEvolution(124,
            spell(29, "Zap", 2.0f, 2.5f, 192, 3, 'Z'),
            spell(29, "Zap", 2.0f, 2.5f, 192, 3, 'Z').withRepeats(3, 3),
            2, 1);

        // Skeletons Evolution: 2 cycles ("at least twice a match" given
        // the 1-elixir cost, not a distinct evolved-uses number -- kept
        // at the standard 1). "Never-ending Horde": see
        // evolvedSkeletonChildStats above for the cap-counting design.
        addEvolution(125,
            troop(24, "Skeletons", 1.0f, Archetype::MeleeSquad, 81, 1.0f, 0.5f, 81, 11, 'k')
                .withOffsets({ {0.0f, 0.0f}, {0.6f, 0.0f}, {-0.6f, 0.0f} }),
            troop(24, "Skeletons", 1.0f, Archetype::MeleeSquad, 81, 1.0f, 0.5f, 81, 11, 'k')
                .withOffsets({ {0.0f, 0.0f}, {0.6f, 0.0f}, {-0.6f, 0.0f} })
                .withOnHitSpawn(std::make_shared<CappedSpawnOnHitEffect>(evolvedSkeletonChildStats(), 8)),
            2, 1);

        // Bats Evolution: 2 cycles, "1 in every 3 deploys will be
        // evolved" (same repeating pattern as the others above). +50% hp
        // (81 -> 121) and heals on every landed hit, up to double its OWN
        // (already-boosted) starting hp -- 121*2=242. Heal-per-hit amount
        // isn't part of the sourced data (only the hp figures are) -- a
        // reasonable ~10%-of-cap engine-internal constant, same caveat
        // category as splashRadius/shieldHp elsewhere in this file.
        addEvolution(126,
            troop(78, "Bats", 2.0f, Archetype::MeleeSquad, 81, 0.85f, 1.0f, 81, 12, 't')
                .withOffsets({ {0.0f, 0.0f}, {0.6f, 0.0f}, {-0.6f, 0.0f}, {0.3f, 0.5f}, {-0.3f, 0.5f} })
                .withFlying().withTargetsAir(),
            troop(78, "Bats", 2.0f, Archetype::MeleeSquad, 121, 0.85f, 1.0f, 81, 12, 't')
                .withOffsets({ {0.0f, 0.0f}, {0.6f, 0.0f}, {-0.6f, 0.0f}, {0.3f, 0.5f}, {-0.3f, 0.5f} })
                .withFlying().withTargetsAir()
                .withHealOnHit(24, 242),
            2, 1);

        // Bomber Evolution: 2 cycles (standard pattern). +25% hp
        // (304->380). "Bouncy Bomb": bounces 2 additional times, each
        // dealing full damage -- approximated via the existing split-
        // target mechanism (Electro Wizard/Dragon) at 3 total targets,
        // full damage each, rather than a bespoke sequential-bounce
        // primitive; the net effect (up to 3 nearby enemies take full
        // damage from one shot) is the same even though the real
        // animation chains instead of hitting simultaneously.
        addEvolution(127,
            troop(9, "Bomber", 2.0f, Archetype::RangedSquad, 304, 0.5f, 4.5f, 225, 18, 'b'),
            troop(9, "Bomber", 2.0f, Archetype::RangedSquad, 380, 0.5f, 4.5f, 225, 18, 'b')
                .withSplitTargets(3).withSplitTargetsFullDamage(),
            2, 1);

        // Archers Evolution: 2 cycles (standard pattern). Sourced stat
        // boost is "+1 tile Range" (5.0 -> 6.0, corrected from an earlier
        // guess of 6.5). "Power Shot": +50% damage against a target 4-6
        // tiles away -- new rangeBandBonus primitive (see CombatEntity/
        // CardStats), also reused by Executioner's Axe Smash below.
        addEvolution(128,
            troop(1, "Archers", 3.0f, Archetype::RangedSquad, 304, 0.5f, 5.0f, 112, 9, 'A')
                .withOffsets({ {0.0f, 0.0f}, {1.0f, 0.0f} }).withTargetsAir(),
            troop(1, "Archers", 3.0f, Archetype::RangedSquad, 304, 0.5f, 6.0f, 112, 9, 'A')
                .withOffsets({ {0.0f, 0.0f}, {1.0f, 0.0f} }).withTargetsAir()
                .withRangeBandBonus(4.0f, 6.0f, 1.5f),
            2, 1);

        // Cannon Evolution: 2 cycles (standard pattern). "Deploy Barrage":
        // a one-time 9-projectile burst the instant it deploys (2.5-tile
        // damage radius per sources) -- maps directly onto the existing
        // spawnEffect one-time-burst-on-deploy mechanism (already used
        // elsewhere, e.g. Electro Wizard). Knockback on the burst isn't
        // modeled (spawnEffect has no knockback param); the burst's own
        // damage figure isn't independently sourced, so it reuses the
        // Cannon's regular per-hit damage as a reasonable placeholder.
        addEvolution(129,
            building(25, "Cannon", 3.0f, 824, 'C', 5.5f, 202, 10),
            building(25, "Cannon", 3.0f, 824, 'C', 5.5f, 202, 10)
                .withSpawnEffect(2.5f, 202),
            2, 1);

        // Firecracker Evolution: 2 cycles (standard pattern). Real
        // mechanic is a center spark (2.5-tile radius) plus shrapnel
        // sparks (1.2-tile radius) dealing very low damage every 0.25s
        // for 3 seconds (sourced duration -- corrected from an earlier
        // guess of 4s/40 ticks), plus a 15% move-speed slow -- approximated
        // here as a flat DoT mark on whatever the main shot hits (new
        // PoisonOnHit/CombatEntity::applyDot), not modeling the shrapnel
        // spread or the slow (this engine's DoT has no accompanying
        // speed-reduction component). Damage-per-tick isn't independently
        // sourced (only "very low, every 0.25s" is), so it's a reasonable
        // engine-internal constant, same caveat category as splashRadius
        // elsewhere in this file; the 3-tick interval approximates 0.25s
        // (2.5 ticks) at this engine's 10-ticks/second rate.
        addEvolution(130,
            troop(64, "Firecracker", 3.0f, Archetype::RangedSquad, 304, 0.7f, 6.0f, 64, 30, '$')
                .withTargetsAir().withSplash(1.5f).withRecoil(1.0f).withSightRange(8.5f),
            troop(64, "Firecracker", 3.0f, Archetype::RangedSquad, 304, 0.7f, 6.0f, 64, 30, '$')
                .withTargetsAir().withSplash(1.5f).withRecoil(1.0f).withSightRange(8.5f)
                .withOnHit(std::make_shared<PoisonOnHit>(16, 30, 3)),
            2, 1);

        // Dart Goblin Evolution: 2 cycles (standard pattern). Real
        // mechanic is a 3-tier escalating poison (51/115/307 dmg per
        // consecutive dart) with a 1.25s initial delay -- approximated
        // as a flat DoT at the tier-1 rate throughout (51 dmg/tick, 4s,
        // 1 tick/second), not modeling the escalation or the initial
        // delay. The real card's "poison persists even if the Dart
        // Goblin dies" already falls out for free here -- the mark lives
        // on the victim, independent of the attacker's lifetime.
        // withTargetsAir on BOTH stat blocks: addEvolution inherits nothing
        // from the base card's registration (it copies isFlying explicitly and
        // nothing else -- targetsAir only reaches the entity via
        // CardFactories::spawn on whichever CardStats is passed here). Without
        // it the base Dart Goblin could hit air and its EVOLVED form could
        // not, i.e. evolving strictly downgraded the card.
        addEvolution(131,
            troop(20, "Dart Goblin", 3.0f, Archetype::RangedSquad, 261, 0.8f, 6.5f, 151, 8, 'd')
                .withTargetsAir().withSightRange(7.5f),
            troop(20, "Dart Goblin", 3.0f, Archetype::RangedSquad, 261, 0.8f, 6.5f, 151, 8, 'd')
                .withTargetsAir()
                .withOnHit(std::make_shared<PoisonOnHit>(51, 40, 10)).withSightRange(7.5f),
            2, 1);

        // Goblin Barrel Evolution: 2 cycles (standard pattern). Real
        // mechanic launches a 2nd barrel on the mirrored lane with 3
        // Decoy Goblins -- approximated as 6 Goblins at the same landing
        // point instead (double the base's 3), not modeling the mirrored
        // second location.
        addEvolution(132,
            spell(109, "Goblin Barrel", 3.0f, 0.5f, 0, 8, '[')
                .withSpellSpawn(std::make_shared<PeriodicSpawnEffect>(goblinBarrelGoblinStats())),
            spell(109, "Goblin Barrel", 3.0f, 0.5f, 0, 8, '[')
                .withSpellSpawn(std::make_shared<PeriodicSpawnEffect>(goblinBarrelGoblinStats()
                    .withOffsets({ {-0.4f,0.0f},{0.4f,0.0f},{0.0f,0.4f},{-0.4f,0.4f},{0.4f,0.4f},{0.0f,-0.4f} }))),
            2, 1);

        // Skeleton Army Evolution: 2 cycles (standard pattern). "+1
        // Skeleton" (16 total, corrected from an earlier version that
        // left the evolved form at the same 15 as the base card) --
        // see skeletonArmyEvolvedOffsets above, representing the Skeleton
        // General alongside skeletonArmyShadowStats for the shadow-spawn
        // design. The General's own distinct shield-hp isn't modeled
        // (every spawned unit in an .withOffsets() squad shares one
        // CardStats, no per-position stat variation).
        addEvolution(133,
            troop(12, "Skeleton Army", 3.0f, Archetype::MeleeSquad, 81, 1.0f, 0.5f, 81, 11, 's')
                .withOffsets(skeletonArmyOffsets()),
            troop(12, "Skeleton Army", 3.0f, Archetype::MeleeSquad, 81, 1.0f, 0.5f, 81, 11, 's')
                .withOffsets(skeletonArmyEvolvedOffsets())
                .withDeathEffect(std::make_shared<SpawnOnDeath>(skeletonArmyShadowStats())),
            2, 1);

        // Skeleton Barrel Evolution: 2 cycles (standard pattern). +25% hp
        // (532->665). Real card drops 2 waves of 7 skeletons (at 75% hp
        // and on death); this engine only models the on-death spawn (the
        // base card doesn't have an hp-threshold trigger mechanism wired
        // for a non-transforming side effect), scaled up to 7 skeletons
        // instead of the base's 2. The base card's own death-damage stat
        // referenced in some sources isn't modeled here either way (not
        // present on the un-evolved card in this engine).
        addEvolution(134,
            troop(89, "Skeleton Barrel", 3.0f, Archetype::MeleeBuildingTargeter, 532, 0.85f, 1.0f, 81, 10, ',')
                .withFlying()
                .withDeathEffect(std::make_shared<SpawnOnDeath>(skeletonBarrelSkeletonStats())).withSightRange(7.7f),
            troop(89, "Skeleton Barrel", 3.0f, Archetype::MeleeBuildingTargeter, 665, 0.85f, 1.0f, 81, 10, ',')
                .withFlying()
                .withDeathEffect(std::make_shared<SpawnOnDeath>(skeletonBarrelSkeletonStats().withOffsets({
                    {-0.6f,0.0f},{0.6f,0.0f},{-0.3f,0.3f},{0.3f,0.3f},{-0.3f,-0.3f},{0.3f,-0.3f},{0.0f,0.0f}
                }))).withSightRange(7.7f),
            2, 1);

        // Knight Evolution: 2 cycles (standard pattern). Real shield: 60%
        // less damage while approaching, drops once he starts attacking
        // -- approximated as a flat, permanent, smaller reduction (35%)
        // instead, since this engine has no "is mid-attack vs.
        // approaching" signal at the CardStats level (see
        // withPassiveDamageReduction's own comment).
        addEvolution(135,
            troop(0, "Knight", 3.0f, Archetype::MeleeSquad, 1766, 0.5f, 1.2f, 202, 12, 'K'),
            troop(0, "Knight", 3.0f, Archetype::MeleeSquad, 1766, 0.5f, 1.2f, 202, 12, 'K')
                .withPassiveDamageReduction(0.65f),
            2, 1);

        // Royal Ghost Evolution: 2 cycles (standard pattern). Longer
        // reveal window after attacking (18 ticks/1.8s vs the base's 5
        // ticks/0.5s) -- a pure stat change on the existing invisibility
        // mechanism. "Souldier Summoning": every landed hit spawns 2
        // Souldiers dealing spawn damage at the attack location -- new
        // royalGhostSouldierStats() (see its own comment), wired via the
        // existing onHitSpawnEffect mechanism (a clean fit, same shape as
        // Evolved Skeletons' self-spawn). The "only while invisible"
        // qualifier isn't separately gated -- the Royal Ghost is already
        // cloaked for virtually all of her own attacks by design, so this
        // fires unconditionally on every hit rather than adding an extra
        // invisibility check.
        addEvolution(136,
            troop(47, "Royal Ghost", 3.0f, Archetype::MeleeSquad, 1210, 0.7f, 1.2f, 261, 18, 'Q')
                .withInvisibility(5).withIgnoresRiver(),
            troop(47, "Royal Ghost", 3.0f, Archetype::MeleeSquad, 1210, 0.7f, 1.2f, 261, 18, 'Q')
                .withInvisibility(18)
                .withOnHitSpawn(std::make_shared<PeriodicSpawnEffect>(royalGhostSouldierStats()
                    .withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f} })))
                .withIgnoresRiver(),
            2, 1);

        // Baby Dragon Evolution: 2 cycles (standard pattern). Real aura
        // slows enemies 30% and speeds up allies 30% continuously in an
        // 8x9 area -- approximated via the existing ally-buff aura
        // (reinterpreting "speed buff" as a damage buff, since this
        // engine has no movement-speed-buff plumbing at all -- see
        // CardStats.h's own comment on Rage skipping the same thing) and
        // triggered per landed hit rather than continuously. Enemy-slow
        // half of the aura isn't modeled either, for the same reason.
        addEvolution(137,
            troop(44, "Baby Dragon", 4.0f, Archetype::RangedSquad, 1152, 0.8f, 3.5f, 168, 15, 'y')
                .withFlying().withTargetsAir(),
            troop(44, "Baby Dragon", 4.0f, Archetype::RangedSquad, 1152, 0.8f, 3.5f, 168, 15, 'y')
                .withFlying().withTargetsAir()
                .withAllyBuffAura(4.0f, 1, 1.3f, 20, 1000000),
            2, 1);

        // Furnace Evolution: 2 cycles (standard pattern). "Hot Spawning":
        // Fire Spirit spawn period drops to 2.4s (24 ticks, corrected from
        // an earlier halved-interval guess of 35 ticks/3.5s -- the sourced
        // number isn't simply half of the base's 7.0s). No Fire Spirit
        // damage change is sourced (an earlier version of this comment
        // incorrectly claimed +180% damage; not modeled, and shouldn't
        // be). Directional alternating-side spawn positioning isn't
        // modeled either (single spawn point, same as the base card).
        addEvolution(138,
            troop(70, "Furnace", 4.0f, Archetype::RangedSquad, 727, 0.5f, 5.5f, 179, 17, '/')
                .withTargetsAir()
                .withPeriodicEffect(70, std::make_shared<PeriodicSpawnEffect>(furnaceFireSpiritStats())),
            troop(70, "Furnace", 4.0f, Archetype::RangedSquad, 727, 0.5f, 5.5f, 179, 17, '/')
                .withTargetsAir()
                .withPeriodicEffect(24, std::make_shared<PeriodicSpawnEffect>(
                    furnaceFireSpiritStats().withOffsets({ {0.0f, 0.0f} }))),
            2, 1);

        // Goblin Cage Evolution: 2 cycles (standard pattern). Real
        // mechanic pulls enemies in and holds them, dealing DoT until the
        // cage expires -- approximated via the existing hook (pull) plus
        // the new PoisonOnHit DoT mark, not a true persistent hold/root
        // (this engine has no such lock mechanism).
        addEvolution(139,
            building(97, "Goblin Cage", 4.0f, 780, '6', 0.0f, 0, 100)
                .withDeathEffect(std::make_shared<SpawnOnDeath>(goblinCageBrawlerStats())),
            building(97, "Goblin Cage", 4.0f, 780, '6', 3.0f, 40, 20)
                .withHook(3.0f)
                .withOnHit(std::make_shared<PoisonOnHit>(30, 40, 10))
                .withDeathEffect(std::make_shared<SpawnOnDeath>(goblinCageBrawlerStats())),
            2, 1);

        // Musketeer Evolution: 2 cycles (standard pattern). "Sniper Shot":
        // 3 empowered long-range shots (+80% damage, corrected from an
        // earlier guess of +50%) -- the damage part reuses the burst-on-
        // Nth-attack primitive (a clean fit for "every 3rd shot"), though
        // that primitive repeats forever rather than the sourced "3
        // total, then never again"; the "vertically infinite / 2-tile
        // horizontal, can't target towers" range shape isn't modeled
        // either (burst only affects damage, not targeting).
        addEvolution(140,
            troop(6, "Musketeer", 4.0f, Archetype::RangedSquad, 721, 0.5f, 6.0f, 217, 10, 'U')
                .withTargetsAir().withSightRange(6.0f),
            troop(6, "Musketeer", 4.0f, Archetype::RangedSquad, 721, 0.5f, 6.0f, 217, 10, 'U')
                .withTargetsAir().withBurstAttack(3, 1.8f).withSightRange(6.0f),
            2, 1);

        // Wizard Evolution: 2 cycles (standard pattern). "Fire Shield" is
        // sourced as 25% of his own hp (755*0.25=189, corrected from an
        // earlier unsourced guess of 300) -- reuses the existing shield
        // mechanism directly. The shield-break knockback+damage explosion
        // isn't modeled (no "on shield depleted" trigger exists in this
        // engine).
        addEvolution(141,
            troop(11, "Wizard", 5.0f, Archetype::RangedSquad, 755, 0.5f, 5.5f, 281, 14, 'W')
                .withTargetsAir().withSplash(1.5f),
            troop(11, "Wizard", 5.0f, Archetype::RangedSquad, 755, 0.5f, 5.5f, 281, 14, 'W')
                .withTargetsAir().withSplash(1.5f).withShield(189),
            2, 1);

        // Witch Evolution: 2 cycles (standard pattern). Real mechanic
        // heals when a spawned Skeleton dies, overhealing up to 24% above
        // max hp -- approximated as healing on her own landed hits
        // instead (this engine has no "notify parent when spawned child
        // dies" hook), capped at 839*1.24=1040.
        addEvolution(142,
            troop(71, "Witch", 5.0f, Archetype::RangedSquad, 839, 0.5f, 5.5f, 135, 11, ':')
                .withTargetsAir()
                .withPeriodicEffect(70, std::make_shared<PeriodicSpawnEffect>(witchSkeletonStats())),
            troop(71, "Witch", 5.0f, Archetype::RangedSquad, 839, 0.5f, 5.5f, 135, 11, ':')
                .withTargetsAir()
                .withPeriodicEffect(70, std::make_shared<PeriodicSpawnEffect>(witchSkeletonStats()))
                .withHealOnHit(20, 1040),
            2, 1);

        // Royal Giant Evolution: 2 cycles (standard pattern). Identical
        // base stats; every attack (always against a building, since this
        // archetype already only targets buildings) also splashes a 2.5
        // radius -- a clean direct fit for the existing splash mechanism.
        // Knockback on the splash isn't modeled.
        addEvolution(143,
            troop(18, "Royal Giant", 6.0f, Archetype::RangedBuildingTargeter, 3164, 0.3f, 5.0f, 307, 18, 'Y')
                .withSightRange(7.5f),
            troop(18, "Royal Giant", 6.0f, Archetype::RangedBuildingTargeter, 3164, 0.3f, 5.0f, 307, 18, 'Y')
                .withSplash(2.5f).withSightRange(7.5f),
            2, 1);

        // Ice Spirit Evolution: 2 cycles (standard pattern). Sourced stat
        // boost is "+0.5 tiles Splash Radius" -- the base Ice Spirit (id
        // 72 above) has no splash modeled in this engine at all (a
        // pre-existing, separate simplification, single-target only), so
        // this is applied as a new absolute splash radius on the evolved
        // form alone (1.7 tiles, based on the real card's own ~1.2-tile
        // splash + the sourced 0.5 delta) rather than literally "base
        // + 0.5", to avoid quietly changing the un-evolved card's own
        // behavior as a side effect of this fix. Real mechanic also
        // re-applies the same stun 3s after the first (a delayed second
        // pulse) -- approximated as one longer freeze instead of two
        // separate pulses (10 ticks -> 51 ticks, covering roughly the
        // same total window: 1s initial + 3s delay + 1.1s repeat).
        addEvolution(144,
            troop(72, "Ice Spirit", 1.0f, Archetype::RangedSquad, 230, 0.85f, 2.5f, 110, 10, ';')
                .withTargetsAir()
                .withOnHit(std::make_shared<FreezeOnHit>(10, 0.5f))
                .withDieAfterFirstHit(),
            troop(72, "Ice Spirit", 1.0f, Archetype::RangedSquad, 230, 0.85f, 2.5f, 110, 10, ';')
                .withTargetsAir()
                .withSplash(1.7f)
                .withOnHit(std::make_shared<FreezeOnHit>(51, 0.5f))
                .withDieAfterFirstHit(),
            2, 1);

        // Princess Evolution: 2 cycles (standard pattern). Real mechanic
        // alternates: 1st/4th/7th... shot slows (30%, 7s, 3-tile), other
        // shots are normal, and death leaves a lingering slow zone --
        // approximated as every hit slowing (not just every 3rd) and no
        // death zone (this engine has no "spawn a lingering area effect
        // on death" primitive).
        addEvolution(145,
            troop(61, "Princess", 3.0f, Archetype::RangedSquad, 261, 0.5f, 9.0f, 168, 30, '9')
                .withTargetsAir().withSplash(1.5f).withSightRange(9.5f),
            troop(61, "Princess", 3.0f, Archetype::RangedSquad, 261, 0.5f, 9.0f, 168, 30, '9')
                .withTargetsAir().withSplash(1.5f).withSightRange(9.5f)
                .withOnHit(std::make_shared<FreezeOnHit>(70, 0.7f)),
            2, 1);

        // Hunter Evolution: 2 cycles (standard pattern). "Netting Trap":
        // nets the nearest enemy (can't move or attack) for 3s, recharging
        // 5s after that -- read as an 8s total cycle (30-tick net +
        // 50-tick recharge = 80 ticks between throws), corrected from an
        // earlier reading of the interval as a flat 5s/50 ticks. New
        // PeriodicFreezeNearestEffect, reusing the existing freeze
        // machinery -- "letting ground units pile on" falls out naturally
        // (a frozen unit just takes normal damage from whatever reaches
        // it).
        addEvolution(146,
            troop(62, "Hunter", 4.0f, Archetype::RangedSquad, 885, 0.5f, 4.0f, 84, 22, '!')
                .withTargetsAir().withSplash(1.5f).withRangeFalloff(0.5f),
            troop(62, "Hunter", 4.0f, Archetype::RangedSquad, 885, 0.5f, 4.0f, 84, 22, '!')
                .withTargetsAir().withSplash(1.5f).withRangeFalloff(0.5f)
                .withPeriodicEffect(80, std::make_shared<PeriodicFreezeNearestEffect>(4.0f, 30)),
            2, 1);

        // Valkyrie Evolution: 2 cycles (standard pattern). "Identical
        // Stats" per the sourced table (reverted an earlier, incorrect
        // splash-radius buff that had been invented to compensate for the
        // unmodeled pull -- now that a real pull primitive exists, there's
        // no need to compensate with a stat change at all). "Whirlwind
        // Axe": every landed hit pulls enemy troops within a 5.5-tile
        // radius toward her (new onHitPull primitive, excludes buildings/
        // towers same as Tornado's own knockback) plus low damage to
        // everyone in that radius, including Crown Towers (the pull
        // itself still exempts them). Pull distance-per-hit and the low
        // damage amount aren't independently sourced -- reasonable
        // engine-internal constants, same caveat category as
        // splashRadius elsewhere in this file. The lingering 0.5s zone
        // itself isn't modeled (this is an instant per-hit effect, not a
        // separate lingering hazard).
        addEvolution(147,
            troop(10, "Valkyrie", 4.0f, Archetype::MeleeSquad, 1907, 0.5f, 1.2f, 266, 15, 'V')
                .withSplash(1.5f),
            troop(10, "Valkyrie", 4.0f, Archetype::MeleeSquad, 1907, 0.5f, 1.2f, 266, 15, 'V')
                .withSplash(1.5f)
                .withOnHitPull(5.5f, 2.0f, 50),
            2, 1);

        // P.E.K.K.A. Evolution: 2 cycles (standard pattern). "Butter-Heal"
        // heals on the FINAL BLOW that defeats a troop/building (scaled to
        // the victim's hp), overhealing up to +66% max hp (3760*1.66=6242,
        // corrected from an earlier guess of +50%/5640) -- approximated as
        // a smaller heal on every landed HIT instead (this engine has no
        // "notify on kill" hook, only on-hit), tuned down since hits are
        // far more frequent than kills.
        addEvolution(148,
            troop(13, "P.E.K.K.A.", 7.0f, Archetype::MeleeSquad, 3760, 0.4f, 1.2f, 842, 18, 'E')
                .withSightRange(5.0f),
            troop(13, "P.E.K.K.A.", 7.0f, Archetype::MeleeSquad, 3760, 0.4f, 1.2f, 842, 18, 'E')
                .withHealOnHit(40, 6242).withSightRange(5.0f),
            2, 1);

        // Minion Horde Evolution: 2 cycles (standard pattern). "Dark
        // Guard": taking damage from a troop or spell turns the hit
        // member invisible (untargetable) for 3s, after which it's
        // vulnerable again -- corrected from an earlier reading (a flat
        // absorb-shield) that modeled a fundamentally different mechanic
        // (damage immunity, not evasion). New DarkGuardOnDamageEffect,
        // reusing temporaryInvisibilityTicksRemaining via the existing
        // onDamageTaken hook, same technique as Archer Queen's Cloaking
        // Cape/Boss Bandit's Getaway Grenade (an activated ability there,
        // a defensive trigger here). Still an approximation: real Dark
        // Guard specifically requires the FIRST hit taken to trigger it
        // (already-invisible members presumably don't re-trigger/extend
        // it), whereas this fires -- and refreshes the 3s window -- on
        // every hit taken, invisible or not.
        addEvolution(149,
            troop(42, "Minion Horde", 5.0f, Archetype::MeleeSquad, 230, 0.8f, 2.5f, 107, 12, 'h')
                .withOffsets({ {-0.6f, -0.3f}, {0.0f, -0.3f}, {0.6f, -0.3f},
                               {-0.6f, 0.3f}, {0.0f, 0.3f}, {0.6f, 0.3f} })
                .withFlying().withTargetsAir(),
            troop(42, "Minion Horde", 5.0f, Archetype::MeleeSquad, 230, 0.8f, 2.5f, 107, 12, 'h')
                .withOffsets({ {-0.6f, -0.3f}, {0.0f, -0.3f}, {0.6f, -0.3f},
                               {-0.6f, 0.3f}, {0.0f, 0.3f}, {0.6f, 0.3f} })
                .withFlying().withTargetsAir()
                .withOnDamageTaken(std::make_shared<DarkGuardOnDamageEffect>(30)),
            2, 1);

        // Royal Recruits Evolution: 2 cycles (standard pattern). Real
        // mechanic grants a charge specifically once the shield breaks,
        // requiring 2.5 tiles of travel for 2x damage -- approximated as
        // an unconditional charge bonus instead (this engine's charge
        // mechanism isn't gated on shield state), but now using the
        // sourced distance/multiplier (2.5, 2.0) rather than an earlier
        // guess (2.0, 1.5).
        addEvolution(150,
            troop(77, "Royal Recruits", 7.0f, Archetype::MeleeSquad, 547, 0.5f, 1.0f, 133, 13, '@')
                .withOffsets({ {-2.5f, 0.0f}, {-1.5f, 0.0f}, {-0.5f, 0.0f}, {0.5f, 0.0f}, {1.5f, 0.0f}, {2.5f, 0.0f} })
                .withShield(240),
            troop(77, "Royal Recruits", 7.0f, Archetype::MeleeSquad, 547, 0.5f, 1.0f, 133, 13, '@')
                .withOffsets({ {-2.5f, 0.0f}, {-1.5f, 0.0f}, {-0.5f, 0.0f}, {0.5f, 0.0f}, {1.5f, 0.0f}, {2.5f, 0.0f} })
                .withShield(240)
                .withCharge(2.5f, 2.0f),
            2, 1);

        // Electro Dragon Evolution: 2 cycles (standard pattern). "Infinite
        // Bolts" chains beyond the base's 3 targets (at reduced damage/
        // speed, no longer stunning past the 3rd) until only one enemy
        // remains -- approximated as a fixed higher split-target count (6)
        // at full damage throughout, not a true unbounded chain with
        // degrading effect past the 3rd target.
        addEvolution(151,
            troop(57, "Electro Dragon", 5.0f, Archetype::MeleeSquad, 1049, 0.5f, 3.5f, 192, 21, '5')
                .withFlying().withTargetsAir()
                .withSplitTargets(3).withSplitTargetsFullDamage()
                .withOnHit(std::make_shared<FreezeOnHit>(5, 0.0f)),
            troop(57, "Electro Dragon", 5.0f, Archetype::MeleeSquad, 1049, 0.5f, 3.5f, 192, 21, '5')
                .withFlying().withTargetsAir()
                .withSplitTargets(6).withSplitTargetsFullDamage()
                .withOnHit(std::make_shared<FreezeOnHit>(5, 0.0f)),
            2, 1);

        // Mortar Evolution: 2 cycles (standard pattern). Sourced stat
        // boost is "-1 second Attack Period" (50 ticks -> 40). "Green
        // Siege": a Goblin spawns with every landed shot -- modeled via
        // onHitSpawnEffect (fires exactly when a shot lands), not an
        // independent periodic timer, so the spawn can never drift out of
        // sync with actual shots the way a fixed-interval timer could.
        addEvolution(152,
            building(93, "Mortar", 4.0f, 1369, 'R', 11.5f, 266, 50)
                .withMinRange(3.5f).withSightRange(11.5f),
            building(93, "Mortar", 4.0f, 1369, 'R', 11.5f, 266, 40)
                .withMinRange(3.5f).withSightRange(11.5f)
                .withOnHitSpawn(std::make_shared<PeriodicSpawnEffect>(goblinDrillGoblinStats())),
            2, 1);

        // Goblin Drill Evolution: 2 cycles (standard pattern). Real
        // mechanic resurfaces at 66%/33% hp, leaving a Goblin behind each
        // time -- approximated as more Goblins in the single final
        // death-spawn instead (this engine has no multi-threshold
        // resurface-in-place mechanism), not modeling the actual
        // resurfacing/repositioning.
        addEvolution(153,
            building(98, "Goblin Drill", 4.0f, 1313, '7', 0.0f, 0, 100)
                .withPeriodicEffect(30, std::make_shared<PeriodicSpawnEffect>(goblinDrillGoblinStats()))
                .withDeployAnywhere()
                .withSpawnEffect(2.0f, 84)
                .withDeathEffect(std::make_shared<SpawnOnDeath>(goblinDrillDeathGoblinStats())),
            building(98, "Goblin Drill", 4.0f, 1313, '7', 0.0f, 0, 100)
                .withPeriodicEffect(30, std::make_shared<PeriodicSpawnEffect>(goblinDrillGoblinStats()))
                .withDeployAnywhere()
                .withSpawnEffect(2.0f, 84)
                .withDeathEffect(std::make_shared<SpawnOnDeath>(goblinDrillDeathGoblinStats()
                    .withOffsets({ {-0.4f,0.0f},{0.4f,0.0f},{0.0f,0.4f},{0.0f,-0.4f} }))),
            2, 1);

        // Tesla Evolution: 2 cycles (standard pattern). "Electro Pulse"
        // fires when it emerges from hiding, 6-tile radius (corrected from
        // an earlier guess of 3.0), low damage + 0.5s stun -- this
        // engine's Tesla has no invisibility/hiding mechanic to "emerge"
        // from at all, so this is approximated as a one-time stun burst
        // at deploy only (reusing the existing spawn-effect mechanism),
        // not a repeating per-emergence pulse.
        addEvolution(154,
            building(26, "Tesla", 4.0f, 1182, 'T', 5.5f, 220, 11).withTargetsAir(),
            building(26, "Tesla", 4.0f, 1182, 'T', 5.5f, 220, 11).withTargetsAir()
                .withSpawnEffect(6.0f, 100, std::make_shared<FreezeOnHit>(5, 0.0f)),
            2, 1);

        // Barbarians Evolution: 2 cycles (standard pattern). +10% hp
        // (691->760). "Blade Rage": +35% attack speed (corrected from an
        // earlier reading of +30%) for 3s on every attack, timer resets
        // while they keep attacking -- new selfHasteOnHit primitive
        // (0.74 approximates 1/1.35). The accompanying +35% movement
        // speed isn't modeled (this engine has no movement-speed-buff
        // plumbing at all, same documented gap as Baby Dragon Evolution).
        addEvolution(155,
            troop(8, "Barbarians", 5.0f, Archetype::MeleeSquad, 691, 0.5f, 0.7f, 192, 14, 'B')
                .withOffsets({ {0.0f, 0.0f}, {-0.5f, -0.5f}, {0.5f, -0.5f}, {-0.5f, 0.5f}, {0.5f, 0.5f} }),
            troop(8, "Barbarians", 5.0f, Archetype::MeleeSquad, 760, 0.5f, 0.7f, 192, 14, 'B')
                .withOffsets({ {0.0f, 0.0f}, {-0.5f, -0.5f}, {0.5f, -0.5f}, {-0.5f, 0.5f}, {0.5f, 0.5f} })
                .withSelfHasteOnHit(30, 0.74f),
            2, 1);

        // Lumberjack Evolution: 2 cycles (standard pattern). See
        // lumberjackGhostStats above for the ghost-spawn design; composed
        // with the existing Rage-drop-on-death via CompositeDeathEffect.
        addEvolution(156,
            troop(21, "Lumberjack", 4.0f, Archetype::MeleeSquad, 1282, 0.8f, 0.7f, 256, 8, 'l')
                .withDeathEffect(std::make_shared<AreaBuffOnDeath>(2.5f, 1.75f, 55)),
            troop(21, "Lumberjack", 4.0f, Archetype::MeleeSquad, 1282, 0.8f, 0.7f, 256, 8, 'l')
                .withDeathEffect(std::make_shared<CompositeDeathEffect>(
                    std::vector<std::shared_ptr<IDeathEffect>>{
                        std::make_shared<AreaBuffOnDeath>(2.5f, 1.75f, 55),
                        std::make_shared<SpawnOnDeath>(lumberjackGhostStats())
                    })),
            2, 1);

        // Executioner Evolution: 1 cycle (confirmed different from the
        // standard 2 -- "Axe Smash" only needs 1 cycle to unlock). Real
        // mechanic deals +75% damage (corrected from an earlier "doubles"
        // reading of a less precise source) plus 1.5-tile knockback when
        // the target is within 3.5 tiles -- now uses the same
        // rangeBandBonus primitive as Archers' Power Shot (0-3.5 tiles,
        // 1.75x), replacing the earlier flat +50% unconditional buff.
        // Knockback on the outgoing hit (but not the return, and not
        // against heavy units) isn't modeled (no "push the target, not
        // the attacker" primitive at this specific call site).
        addEvolution(157,
            troop(36, "Executioner", 5.0f, Archetype::RangedSquad, 1280, 0.4f, 4.5f, 179, 24, 'x')
                .withTargetsAir().withBoomerang(15),
            troop(36, "Executioner", 5.0f, Archetype::RangedSquad, 1280, 0.4f, 4.5f, 179, 24, 'x')
                .withTargetsAir().withBoomerang(15)
                .withRangeBandBonus(0.0f, 3.5f, 1.75f),
            1, 1);

        // Giant Snowball Evolution: 2 cycles (standard pattern). "Snow
        // Roll" sacrifices the base card's knockback (push) for a 4.5-tile
        // PULL instead -- gathering enemies in its path rather than
        // scattering them, the opposite direction from the base card and
        // from an earlier (incorrectly-signed) version of this file that
        // used a bigger push. "Untargetable while trapped" isn't modeled
        // (this engine has no such mechanism).
        addEvolution(158,
            spell(100, "Giant Snowball", 2.0f, 2.5f, 179, 8, '!')
                .withSpellOnHit(std::make_shared<FreezeOnHit>(15, 0.5f))
                .withKnockback(1.0f),
            spell(100, "Giant Snowball", 2.0f, 2.5f, 179, 8, '!')
                .withSpellOnHit(std::make_shared<FreezeOnHit>(40, 0.7f))
                .withKnockback(-4.5f),
            2, 1);

        // Goblin Giant Evolution: 2 cycles (standard pattern). "Sack-
        // trick" spawns knife Goblins every 2.2s once below 50% hp --
        // approximated as an unconditional periodic spawn from deploy
        // (this engine has no "enable a periodic effect only below an hp
        // threshold" mechanism, only full-transform hp thresholds).
        addEvolution(159,
            troop(88, "Goblin Giant", 6.0f, Archetype::MeleeBuildingTargeter, 3110, 0.5f, 1.2f, 176, 15, '`')
                .withSecondaryUnit(goblinGiantSpearGoblinsStats()).withSightRange(7.5f),
            troop(88, "Goblin Giant", 6.0f, Archetype::MeleeBuildingTargeter, 3110, 0.5f, 1.2f, 176, 15, '`')
                .withSecondaryUnit(goblinGiantSpearGoblinsStats()).withSightRange(7.5f)
                .withPeriodicEffect(22, std::make_shared<PeriodicSpawnEffect>(goblinDrillGoblinStats())),
            2, 1);

        // Mega Knight Evolution: 2 cycles (standard pattern). "Identical
        // Stats" per the sourced table -- reverted an earlier, incorrect
        // +20% flat damage buff that had been invented to compensate for
        // the unmodeled ability. "Mega Uppercut" launches every hit target
        // back 4 tiles toward the enemy Crown Tower -- not modeled (this
        // engine has no "knock the target toward a specific board-
        // relative point" primitive, only recoilDistance which pushes the
        // attacker, not the target, and no existing "find the nearest
        // enemy tower's position" board query to aim it at).
        addEvolution(160,
            troop(48, "Mega Knight", 7.0f, Archetype::MeleeSquad, 3993, 0.5f, 1.2f, 268, 17, 'X')
                .withSplash(1.5f).withSpawnEffect(1.3f, 430).withJump(3.5f, 5.0f, 2.0f, 2.2f).withIgnoresRiver(),
            troop(48, "Mega Knight", 7.0f, Archetype::MeleeSquad, 3993, 0.5f, 1.2f, 268, 17, 'X')
                .withSplash(1.5f).withSpawnEffect(1.3f, 430).withJump(3.5f, 5.0f, 2.0f, 2.2f).withIgnoresRiver(),
            2, 1);

        // Battle Ram Evolution: 2 cycles (standard pattern). Real card is
        // the only evolution that spawns another EVOLVED troop (its
        // death-spawned Barbarians are themselves upgraded) --
        // approximated as a flat stat buff on the spawned Barbarians
        // instead of recursively invoking the Evolution framework for a
        // spawned child (that machinery is keyed by deck-slot cycling,
        // which a death-spawned child has no equivalent of). "Head-First
        // Ram": once its charge connects, the Battle Ram keeps dealing
        // double damage on every subsequent hit against that target
        // instead of just the first ("constantly ramming... for every
        // connection made") -- new withStickyCharge() (see
        // CombatEntity::chargeIsSticky), layered on the existing charge
        // mechanism already used by the base card. The contact damage +
        // 2-tile knockback dealt to OTHER small/medium troops while still
        // approaching isn't modeled (this engine's charge only affects
        // the Ram's own damage on arrival, not a moving hitbox along the
        // way).
        addEvolution(161,
            troop(81, "Battle Ram", 4.0f, Archetype::MeleeBuildingTargeter, 691, 0.6f, 1.0f, 192, 14, '^')
                .withDeathEffect(std::make_shared<SpawnOnDeath>(battleRamBarbarianStats()))
                .withCharge(3.0f, 2.0f),
            troop(81, "Battle Ram", 4.0f, Archetype::MeleeBuildingTargeter, 691, 0.6f, 1.0f, 192, 14, '^')
                .withDeathEffect(std::make_shared<SpawnOnDeath>(battleRamEvolvedBarbarianStats()))
                .withCharge(3.0f, 2.0f).withStickyCharge(),
            2, 1);

        // Royal Hogs Evolution: 2 cycles (standard pattern). "Identical
        // Stats" per the sourced table -- reverted an earlier hp buff
        // that had been invented under a since-resolved low-confidence
        // reading of the mechanic. "Hog Flight": each Hog spawns flying,
        // then falls to the ground on its first attack, dealing fall
        // damage equal to 155% of its normal damage -- not modeled. This
        // engine's charge bonus (already on both forms, unchanged) is
        // also framed as a one-time bonus on an early hit, and how the
        // two would actually stack in the real game isn't clear from the
        // sourced text (which doesn't mention charge at all here), so
        // this is left as a genuine follow-up rather than guessing at an
        // interaction.
        addEvolution(162,
            troop(82, "Royal Hogs", 5.0f, Archetype::MeleeBuildingTargeter, 837, 0.85f, 1.0f, 74, 12, '_')
                .withOffsets({ {-1.0f, -0.3f}, {-0.3f, 0.3f}, {0.3f, -0.3f}, {1.0f, 0.3f} })
                .withCharge(3.0f, 2.0f).withSightRange(9.5f).withIgnoresRiver(),
            troop(82, "Royal Hogs", 5.0f, Archetype::MeleeBuildingTargeter, 837, 0.85f, 1.0f, 74, 12, '_')
                .withOffsets({ {-1.0f, -0.3f}, {-0.3f, 0.3f}, {0.3f, -0.3f}, {1.0f, 0.3f} })
                .withCharge(3.0f, 2.0f).withSightRange(9.5f).withIgnoresRiver(),
            2, 1);

        // Inferno Dragon Evolution: 2 cycles, 12 shards, 4 elixir (matches
        // the base card) -- "Identical Stats" per the sourced evolution
        // table, i.e. hp/damage/speed/range/attackCooldown are unchanged
        // from the regular Inferno Dragon (id 56) below; the entire
        // difference is the "Damage Charge-up" behavior itself:
        //   1) Losing its target (or having none at all) no longer resets
        //      the ramp instantly -- the current stage is held for a 9s
        //      (90-tick) grace period, so a fresh target picked up within
        //      that window inherits whatever stage the dragon was already
        //      at ("keeps that stage on the following troops"). Only a
        //      stun still resets immediately, same as the base card --
        //      see CombatEntity's rampGracePeriodTicks/ticksSinceLastHit.
        //   2) A 4th ramp stage after 20s (200 ticks) of continuous
        //      attacking, dealing double the 3rd-stage (max) damage --
        //      see rampStage4Tick/rampStage4Fraction. This resolves the
        //      previous internal inconsistency in the sourced numbers
        //      (49 ticks vs. 20 seconds) simply by trusting the
        //      seconds-based figure and this engine's own 10-ticks/second
        //      rate throughout, rather than a raw tick count from a
        //      differently-timed source.
        addEvolution(163,
            troop(56, "Inferno Dragon", 4.0f, Archetype::MeleeSquad, 1295, 0.5f, 5.0f, 422, 4, '4')
                .withFlying().withTargetsAir()
                .withDamageRamp(15, 30, 0.083f, 0.284f),
            troop(56, "Inferno Dragon", 4.0f, Archetype::MeleeSquad, 1295, 0.5f, 5.0f, 422, 4, '4')
                .withFlying().withTargetsAir()
                .withDamageRamp(15, 30, 0.083f, 0.284f)
                .withRampStage4(200, 2.0f)
                .withRampGracePeriod(90),
            2, 1);

        // === Mirror ===
        // Registered mainly so it's a real, playable hand/deck slot --
        // its own cost/isSpell/placementRadius/deployAnywhere here are
        // never actually read at play time (GameManager::playCard
        // special-cases MIRROR_CARD_ID and substitutes whatever card was
        // last played instead, including for the elixir cost, which is
        // always the mirrored card's own cost + 1). Symbol '=' reused
        // from Heal Spirit (harmless cosmetic reuse, same precedent as
        // every other reused symbol in this file) -- this roster has
        // exhausted nearly the entire printable-ASCII symbol space.
        add(spell(164, "Mirror", 3.0f, 0.0f, 0, 0, '='));

        // === Spirit Empress ===
        // Registered as a Champion-shaped MeleeSquad at the ground form's
        // floor cost (3.0) -- like Mirror, this static registration is
        // mostly a placeholder for hand-display purposes (the observation
        // feature showing hand-slot cost) and for isValidPlacement, since
        // GameManager::playCard's SPIRIT_EMPRESS_CARD_ID branch always
        // spawns via SpiritEmpressForms.h's two dedicated CardStats
        // instead of this entry's own (unused) spawnEntity. isChampion is
        // NOT set here on purpose: Champion-ness only matters for
        // findChampion()/activateChampionAbility, and Spirit Empress has
        // no activated ability to speak of in the sourced data -- see
        // SpiritEmpressForms.h for the full mechanic and its caveats.
        add(troop(165, "Spirit Empress", 3.0f, Archetype::MeleeSquad, 926, 0.85f, 1.2f, 249, 12, '<'));

        // === Heroes ===
        // Clash Royale's "Hero" mechanic (added ~Dec 2025/2026 real-game
        // updates, researched from public sources -- no in-repo sourced
        // text for this one). A Hero takes an EXISTING ordinary troop and
        // gives it a second, ability-carrying form -- mechanically the
        // same isChampion-style machinery as the 8 Champions above (see
        // CardStats::isHero/withHeroAbility's own comments for why this is
        // a parallel flag, not a rename), just layered onto an
        // already-registered base card instead of a wholly new character.
        // Base combat stats below are copied VERBATIM from this engine's
        // own existing base-card registration (never the possibly-
        // different real-game numbers found online) -- only the new
        // ability layer is added on top. Staged implementation (see this
        // project's own plan file): this pilot pair (Hero Mini P.E.K.K.A,
        // Hero Musketeer) uses zero new engine primitives, proving the
        // generalized isHero/championSlots path end-to-end before later
        // Heroes introduce genuinely new mechanics (taunt, flight, etc.).

        // Hero Mini P.E.K.K.A. Base stats copied from card id 5 (Mini
        // PEKKA), see that registration above. "Breakfast Boost":
        // simplified per this feature's own plan (no meter-fills-via-
        // attacking simulation) to a flat, one-time (usesLimit=1, no
        // repeating cooldown) hp+damage boost -- see
        // HeroMiniPekkaBoostEffect.
        add(troop(170, "Hero Mini P.E.K.K.A.", 4.0f, Archetype::MeleeSquad, 1390, 0.8f, 0.8f, 755, 16, 'M')
            .withHeroAbility(1.0f, 0, std::make_shared<HeroMiniPekkaBoostEffect>(210, 1.15f), 1));

        // Hero Musketeer. Base stats copied from card id 6 (Musketeer), see
        // that registration above. "Trusty Turret": spawns a short-range
        // auto-turret in front of her (see heroMusketeerTurretStats above)
        // with a fixed 10s lifetime, targeting air+ground.
        add(troop(168, "Hero Musketeer", 4.0f, Archetype::RangedSquad, 721, 0.5f, 6.0f, 217, 10, 'U')
            .withTargetsAir().withSightRange(6.0f)
            .withHeroAbility(3.0f, 220, std::make_shared<SpawnOnAbility>(heroMusketeerTurretStats())));

        // Hero Goblins. Base stats copied from card id 4 (Goblins), see
        // that registration above -- including the 4-unit squad offsets.
        // "Banner Brigade" (1 elixir, ONE USE, only activatable within a
        // 70-tick/7s window after the LAST goblin of the squad dies):
        // reactivates a fresh 4-unit squad at the death position. Uses
        // withPostDeathAbility (not withHeroAbility) since this card has no
        // alive-path ability at all -- see GameManager::
        // syncChampionCooldowns/isPostDeathAbilityReady/
        // activateChampionAbility for how the window is tracked and
        // consumed. The reactivated squad spawns via the PLAIN (non-Hero)
        // base Goblins CardStats, not this Hero variant, so a second Banner
        // Brigade can never chain off a reactivated squad.
        add(troop(172, "Hero Goblins", 2.0f, Archetype::MeleeSquad, 202, 1.0f, 0.5f, 120, 11, 'g')
            .withOffsets({ {-0.5f, -0.5f}, {0.5f, -0.5f}, {-0.5f, 0.5f}, {0.5f, 0.5f} })
            .withPostDeathAbility(1.0f, 70, std::make_shared<PeriodicSpawnEffect>(
                troop(-48, "Goblins", 0.0f, Archetype::MeleeSquad, 202, 1.0f, 0.5f, 120, 11, 'g')
                    .withOffsets({ {-0.5f, -0.5f}, {0.5f, -0.5f}, {-0.5f, 0.5f}, {0.5f, 0.5f} }))));

        // Hero Knight. Base stats copied from card id 0 (Knight), see that
        // registration above. "Triumphant Taunt" (2 elixir, 250-tick/25s
        // cooldown): gains a shield and forces enemies within 6.5 tiles to
        // attack him for 50 ticks/5s -- see HeroKnightTauntEffect,
        // CombatEntity::forcedTargetEntityId/shieldExpiresTicksRemaining.
        // Shield amount isn't part of the sourced data -- a reasonable
        // engine-internal constant (roughly half his own hp), same caveat
        // as splashRadius/shieldHp elsewhere in this file.
        add(troop(166, "Hero Knight", 3.0f, Archetype::MeleeSquad, 1766, 0.5f, 1.2f, 202, 12, 'K')
            .withHeroAbility(2.0f, 250, std::make_shared<HeroKnightTauntEffect>(880, 50, 6.5f)));

        // Hero Wizard. Base stats copied from card id 11 (Wizard), see that
        // registration above. "Fiery Flight" (1 elixir, 200-tick/20s
        // cooldown): takes flight for 50 ticks/5s, during which every
        // landed attack also pulses a damaging, pulling tornado on the
        // target -- see HeroWizardFieryFlightEffect. Pulse pull distance
        // (0.5 tiles) approximates the sourced "~50% pull strength" --
        // not an exact sourced tile value, same caveat category as
        // splashRadius/shieldHp elsewhere in this file.
        add(troop(167, "Hero Wizard", 5.0f, Archetype::RangedSquad, 755, 0.5f, 5.5f, 281, 14, 'W')
            .withTargetsAir().withSplash(1.5f)
            .withHeroAbility(1.0f, 200, std::make_shared<HeroWizardFieryFlightEffect>(50, 4.0f, 20, 0.5f)));

        // Hero Giant. Base stats copied from card id 2 (Giant), see that
        // registration above. "Heroic Hurl" (2 elixir, 140-tick/14s
        // cooldown): grabs the highest-HP enemy troop within short range
        // (3.0 tiles -- not part of the sourced data, a reasonable
        // engine-internal constant) and throws it to the opposite lane,
        // stunning it 20 ticks/2s on landing -- see HeroGiantHurlEffect.
        add(troop(169, "Hero Giant", 5.0f, Archetype::MeleeBuildingTargeter, 3968, 0.3f, 1.2f, 253, 15, 'G')
            .withSightRange(7.5f)
            .withHeroAbility(2.0f, 140, std::make_shared<HeroGiantHurlEffect>(3.0f, 20)));

        // Hero Mega Minion. Base stats copied from card id 43 (Mega
        // Minion), see that registration above. "Wounding Warp" (2 elixir,
        // ONE USE per deployment, unusable for the first 15 ticks/1.5s
        // after spawn -- see CardStats::withInitialAbilityCooldown):
        // teleports (infinite range) to the lowest-HP enemy on the board
        // and deals bonus damage on arrival -- see HeroMegaMinionWarpEffect.
        // Bonus damage isn't part of the sourced data -- a reasonable
        // engine-internal constant roughly matching her own per-hit
        // damage, same caveat as splashRadius/shieldHp elsewhere.
        add(troop(173, "Hero Mega Minion", 3.0f, Archetype::MeleeSquad, 837, 0.5f, 1.6f, 312, 15, 'F')
            .withFlying().withTargetsAir()
            .withHeroAbility(2.0f, 0, std::make_shared<HeroMegaMinionWarpEffect>(300), 1)
            .withInitialAbilityCooldown(15));

        // Hero Magic Archer. Base stats copied from card id 63 (Magic
        // Archer), see that registration above. "Triple Threat" (2 elixir,
        // 250-tick/25s cooldown): dashes back 5 tiles, spawns a decoy at
        // his old position (see heroMagicArcherDecoyStats above), and gains
        // a 70-tick/7s multi-shot window -- see
        // HeroMagicArcherTripleThreatEffect.
        add(troop(171, "Hero Magic Archer", 4.0f, Archetype::RangedSquad, 529, 0.5f, 7.0f, 143, 11, '#')
            .withTargetsAir()
            .withSplash(0.25f).withLineSplash(11.0f).withSightRange(7.5f)
            .withHeroAbility(2.0f, 250, std::make_shared<HeroMagicArcherTripleThreatEffect>(
                5.0f, heroMagicArcherDecoyStats(), 70)));

        // Hero Ice Golem. Base stats copied from card id 40 (Ice Golem),
        // see that registration above. "Snowstorm" (2 elixir, 170-tick/17s
        // cooldown): 3 staggered blasts in a 4-tile radius -- the first two
        // push+damage+slow, the third a full 15-tick/1.5s freeze -- damage
        // to Crown Towers reduced via AreaSpell's new
        // spellTowerDamageMultiplier. See HeroIceGolemSnowstormEffect.
        // Per-blast damage/knockback/slow strength aren't part of the
        // sourced data -- reasonable engine-internal constants, same
        // caveat as splashRadius/shieldHp elsewhere in this file.
        add(troop(175, "Hero Ice Golem", 2.0f, Archetype::MeleeBuildingTargeter, 1315, 0.4f, 0.75f, 84, 25, 'c')
            .withOnHit(std::make_shared<FreezeOnHit>(30, 0.65f))
            .withDeathEffect(std::make_shared<AreaDamageOnDeath>(2.0f, 84)).withSightRange(7.0f)
            .withHeroAbility(2.0f, 170, std::make_shared<HeroIceGolemSnowstormEffect>(4.0f, 80, 1.0f, 20, 0.6f, 15)));

        // Hero Barbarian Barrel. Base stats copied from card id 101
        // (Barbarian Barrel), see that registration above -- but the
        // ability itself lives on the SPAWNED Barbarian's own CardStats
        // (heroBarbarianBarrelBarbarianStats above), not this ephemeral
        // one-tick spell's, since the spell entity dies the same tick it
        // spawns and never has a live turn to activate anything. isHero is
        // set directly on this spell's own CardStats (not via
        // withHeroAbility, which would also wire up ability fields this
        // entity never uses) purely so validateDeckSlots/seedSlotState
        // recognize deck id 174 itself as Hero-eligible --
        // CardFactories::spawnSpell never reads isHero at all, so this has
        // no runtime effect on the spawned AreaSpell. "Rowdy Reroll" (1
        // elixir, ONE USE): see HeroBarbarianBarrelRerollEffect.
        {
            CardStats heroBarbarianBarrelSpellStats = spell(174, "Hero Barbarian Barrel", 2.0f, 2.5f, 233, 8, '#')
                .withGroundOnly()
                .withSpellSpawn(std::make_shared<PeriodicSpawnEffect>(heroBarbarianBarrelBarbarianStats()));
            heroBarbarianBarrelSpellStats.isHero = true;
            add(heroBarbarianBarrelSpellStats);
        }

        // === Status of the full-refactor initiative (Champions/Evolutions/
        // === Tower Troops/Mirror/Spirit Empress) ===
        // All 8 Champions, all 4 Tower Troops, Mirror, and Spirit Empress
        // are now fully implemented above, along with all 41 real
        // Evolutions (see each addEvolution(...) call's own comment for
        // what's approximated and why) -- Inferno Dragon Evolution was
        // the last holdout, needing the shared ramp system itself
        // extended with a reset grace period and a 4th stage (see
        // CombatEntity's rampGracePeriodTicks/rampStage4Tick/
        // ticksSinceLastHit and this card's own comment, a few cards
        // above this one) rather than a per-card addition.
    }

public:
    static CardRegistry& getInstance() {
        static CardRegistry instance;
        return instance;
    }

    const CardDefinition* getCard(int id) const {
        auto it = cards.find(id);
        if (it != cards.end()) {
            return &(it->second);
        }
        return nullptr;
    }

    const std::unordered_map<int, CardDefinition>& getAllCards() const {
        return cards;
    }
};

// How many Champion cards appear in a deck config. The real game limits a
// deck to at most one Champion via the deck-builder UI; this engine has no
// equivalent gate -- PlayerState::initializeDeck (and GameManager::reset(),
// which calls it) accept whatever card ids they're given with zero
// validation of any kind, by design, matching this project's convention of
// leaving deck legality to the caller (see GameManager::findChampion's own
// comment on the same point). This is opt-in for a caller (tests, Python)
// that wants to check a deck before using it -- not called from anywhere
// in this engine itself.
inline int countChampions(const std::vector<int>& deck) {
    int count = 0;
    for (int cardId : deck) {
        const CardDefinition* def = CardRegistry::getInstance().getCard(cardId);
        if (def && def->isChampion) count++;
    }
    return count;
}

// Same shape as countChampions above, but for Heroes (Clash Royale's
// separate "Hero" card mechanic -- see CardDefinition::isHero's own
// comment). Kept as its own function rather than folded into
// countChampions so each stays an accurate count of exactly what its name
// says.
inline int countHeroes(const std::vector<int>& deck) {
    int count = 0;
    for (int cardId : deck) {
        const CardDefinition* def = CardRegistry::getInstance().getCard(cardId);
        if (def && def->isHero) count++;
    }
    return count;
}

// Deck slot-position legality: slot 0 (Evolution slot) may hold an
// Evolution-flagged card or a plain card; slot 1 (Heroic slot) may hold a
// Champion, a Hero, or a plain card; slot 2 (Wild Card slot) may hold a
// Champion, a Hero, an Evolution, or a plain card; slots 3-7 must be plain
// (no Champion, no Hero, no Evolution). Champion and Hero share the same
// two slots -- Clash Royale's real "Heroes and Champions now share deck
// slots" rule -- so this gives "at most 2 special units total (Champion or
// Hero, one per slot)" for free, same as the pre-Hero "at most 2
// Champions" rule did. Unlike countChampions/countHeroes above, this IS
// meant to be called as a real gate -- see GameManager::reset()/
// setOpponentDeck(), which throw std::invalid_argument on a non-empty
// result. Returns "" for a legal deck, otherwise a human-readable reason
// naming the offending slot/card.
inline std::string validateDeckSlots(const std::vector<int>& deck) {
    if (deck.size() != 8) {
        return "deck must have exactly 8 cards (got " + std::to_string(deck.size()) + ")";
    }
    for (int i = 0; i < 8; ++i) {
        const CardDefinition* def = CardRegistry::getInstance().getCard(deck[i]);
        if (!def) {
            return "slot " + std::to_string(i) + ": card id " + std::to_string(deck[i]) + " is not a registered card";
        }
        bool specialUnitAllowed = (i == 1 || i == 2); // Heroic / Wild Card -- Champion OR Hero
        bool evolutionAllowed = (i == 0 || i == 2);   // Evolution slot / Wild Card
        if ((def->isChampion || def->isHero) && !specialUnitAllowed) {
            return "slot " + std::to_string(i) + ": " + def->name +
                " is a Champion/Hero, only allowed in slot 1 (Heroic) or slot 2 (Wild Card)";
        }
        if (def->isEvolution && !evolutionAllowed) {
            return "slot " + std::to_string(i) + ": " + def->name +
                " is an Evolution, only allowed in slot 0 or slot 2 (Wild Card)";
        }
    }
    return "";
}
