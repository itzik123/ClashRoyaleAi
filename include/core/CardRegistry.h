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
    std::function<void(float x, float y, int team, Board& board)> spawnEntity;
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

    // Golemite: only ever spawned by a Golem's death, never itself a
    // playable card (no id in the registry, hence never add()-ed).
    static CardStats golemiteStats() {
        return troop(-1, "Golemite", 0.0f, Archetype::MeleeBuildingTargeter, 1039, 0.2f, 0.25f, 84, 25, 'q')
            .withOffsets({ {-0.3f, 0.0f}, {0.3f, 0.0f} });
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
    // damage @ 17 ticks).
    static CardStats ramRiderCrossbowStats() {
        return troop(-25, "Ram Rider", 0.0f, Archetype::RangedSquad, 1766, 0.5f, 5.0f, 104, 11, '"')
            .withTargetsAir();
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
            .withDeathEffect(std::make_shared<EnemyElixirGrantOnDeath>(0.5f));
    }
    static CardStats elixirGolemiteStats() {
        return troop(-32, "Elixir Golemite", 0.0f, Archetype::MeleeBuildingTargeter, 762, 0.5f, 1.0f, 128, 11, 'e')
            .withDeathEffect(std::make_shared<CompositeDeathEffect>(
                std::vector<std::shared_ptr<IDeathEffect>>{
                    std::make_shared<SpawnOnDeath>(
                        elixirBlobStats().withOffsets({ {-0.3f, 0.0f}, {0.3f, 0.0f} })),
                    std::make_shared<EnemyElixirGrantOnDeath>(0.5f)
                }));
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

    void add(const CardStats& stats) {
        CardDefinition def;
        def.id = stats.id;
        def.name = stats.name;
        def.cost = stats.cost;
        def.isSpell = (stats.archetype == Archetype::Spell);
        def.isBuilding = (stats.archetype == Archetype::DefensiveBuilding);
        def.placementRadius = CardFactories::placementRadius(stats.archetype);
        def.deployAnywhere = stats.deployAnywhere;
        def.spawnEntity = [stats](float x, float y, int team, Board& board) {
            CardFactories::spawn(stats, x, y, team, board);
        };
        cards[stats.id] = std::move(def);
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

        add(troop(13, "P.E.K.K.A.", 7.0f, Archetype::MeleeSquad, 3760, 0.4f, 1.2f, 842, 18, 'E'));

        // Charge threshold/multiplier below aren't part of the sourced
        // stats data -- reasonable engine-internal constants, same caveat
        // as splashRadius/shieldHp.
        add(troop(14, "Prince", 5.0f, Archetype::MeleeSquad, 1920, 0.6f, 1.6f, 391, 14, 'p')
            .withCharge(3.0f, 2.0f));

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
            .withDeathEffect(std::make_shared<AreaDamageOnDeath>(2.0f, 300)));

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

        add(troop(6, "Musketeer", 4.0f, Archetype::RangedSquad, 721, 0.5f, 6.0f, 217, 10, 'U'));
        add(troop(9, "Bomber", 2.0f, Archetype::RangedSquad, 304, 0.5f, 4.5f, 225, 18, 'b'));
        add(troop(11, "Wizard", 5.0f, Archetype::RangedSquad, 755, 0.5f, 5.5f, 281, 14, 'W')
            .withSplash(1.5f));
        add(troop(20, "Dart Goblin", 3.0f, Archetype::RangedSquad, 261, 0.8f, 6.5f, 151, 8, 'd')
            .withTargetsAir());
        // Now modeled as a real piercing line (applyLineSplashDamage) --
        // total travel 11.5 (4.0 attack range + 7.5 extra), half-width
        // 1.8 (splashRadius doubles as the line's half-width in line-
        // splash mode). Knockback on every hit target still isn't
        // modeled -- AreaSpell's knockback has no equivalent on the
        // troop-attack path this engine's splash/line-splash share.
        add(troop(22, "Bowler", 5.0f, Archetype::RangedSquad, 2081, 0.4f, 4.0f, 289, 25, 'w')
            .withSplash(1.8f).withLineSplash(11.5f));

        add(troop(23, "Spear Goblins", 2.0f, Archetype::RangedSquad, 133, 1.0f, 5.0f, 81, 17, 'S')
            .withOffsets({ {0.0f, 0.0f}, {0.7f, 0.0f}, {-0.7f, 0.0f} }));

        // Ice Wizard: genuinely ranged -- fires a projectile that applies the
        // freeze on arrival. (Originally this was mislabeled: it inherited
        // from RangedTroop but overrode performAttack to hit instantly,
        // bypassing the projectile entirely. Fixed now that on-hit effects
        // can ride along with a projectile instead of firing at launch.)
        add(troop(34, "Ice Wizard", 3.0f, Archetype::RangedSquad, 689, 0.5f, 5.5f, 90, 17, 'i')
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
        add(troop(2, "Giant", 5.0f, Archetype::MeleeBuildingTargeter, 3968, 0.3f, 1.2f, 253, 15, 'G'));

        add(troop(15, "Hog Rider", 4.0f, Archetype::MeleeBuildingTargeter, 1697, 0.8f, 0.8f, 317, 16, 'H')
            .withIgnoresRiver());

        // Golem splits into two Golemites on death AND deals its own
        // death-explosion damage -- two death effects composed via
        // CompositeDeathEffect, since deathEffect is a single slot.
        add(troop(19, "Golem", 8.0f, Archetype::MeleeBuildingTargeter, 5120, 0.2f, 0.75f, 312, 25, 'L')
            .withDeathEffect(std::make_shared<CompositeDeathEffect>(
                std::vector<std::shared_ptr<IDeathEffect>>{
                    std::make_shared<SpawnOnDeath>(golemiteStats()),
                    std::make_shared<AreaDamageOnDeath>(2.5f, 200) })));

        // Ice Golem: same story as Ice Wizard -- BuildingTargeter's own
        // performAttack is already direct damage, so this is behavior-exact.
        // Explodes on death dealing area damage; the real explosion also
        // slows everyone it hits, but AreaDamageOnDeath is damage-only
        // (no onHit-style hook, unlike AreaSpell's spellOnHit) -- not
        // modeled.
        add(troop(40, "Ice Golem", 2.0f, Archetype::MeleeBuildingTargeter, 1315, 0.4f, 0.75f, 84, 25, 'c')
            .withOnHit(std::make_shared<FreezeOnHit>(30, 0.65f))
            .withDeathEffect(std::make_shared<AreaDamageOnDeath>(2.0f, 84)));

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
            .withDeathEffect(std::make_shared<AreaDamageOnDeath>(1.5f, 240)));

        // === Ranged Building Targeter ===
        add(troop(18, "Royal Giant", 6.0f, Archetype::RangedBuildingTargeter, 3164, 0.3f, 5.0f, 307, 18, 'Y'));

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
            .withDamageRamp(20, 40, 0.05f, 0.1875f));

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
        add(troop(46, "Dark Prince", 4.0f, Archetype::MeleeSquad, 1200, 0.5f, 1.2f, 266, 14, 'N')
            .withShield(240) // corrected from an unsourced 200 guess
            .withCharge(3.0f, 2.0f)); // confirmed: +100% (double) damage on a charging hit
        add(troop(47, "Royal Ghost", 3.0f, Archetype::MeleeSquad, 1210, 0.7f, 1.2f, 261, 18, 'Q')
            .withInvisibility(5)); // brief reveal window after attacking
        // Deploy slam via the existing one-time spawn-effect burst (radius
        // 1.3, damage 430). Periodic jump now modeled too, reusing
        // Fisherman's pullToward primitive to instantly close from
        // 3.5-5 tiles away instead of walking in, landing a ~2x-damage,
        // 2.2-radius splash hit (537 confirmed vs. 268 normal -- ratio
        // ~2.003, rounds to the same 2.0 multiplier convention already
        // used for every other charge card).
        add(troop(48, "Mega Knight", 7.0f, Archetype::MeleeSquad, 3993, 0.5f, 1.2f, 268, 17, 'X')
            .withSplash(1.5f)
            .withSpawnEffect(1.3f, 430)
            .withJump(3.5f, 5.0f, 2.0f, 2.2f));
        // Confirmed: real heal lands as 4 pulses of 25.5 (102 total) per
        // attack cycle -- collapsed here into this engine's single
        // heal-on-landed-hit model as one 102 lump, corrected from an
        // unsourced 60 guess. A separate, larger heal burst on deployment
        // (202 total) isn't modeled -- no heal-on-spawn primitive exists
        // (spawnEffect is damage-only).
        add(troop(49, "Battle Healer", 4.0f, Archetype::MeleeSquad, 1717, 0.5f, 1.2f, 148, 15, 'f')
            .withHealAura(3.0f, 102));
        add(troop(50, "Bandit", 3.0f, Archetype::MeleeSquad, 906, 0.8f, 1.0f, 194, 10, 'u')
            .withCharge(3.0f, 2.0f)
            .withChargeInvulnerability()); // confirmed: fully invulnerable while charging in
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
            .withHook(6.5f));
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
            .withStunResetsCooldown()); // confirmed: any stun fully restarts her charge, doesn't just slow it
        add(troop(61, "Princess", 3.0f, Archetype::RangedSquad, 261, 0.5f, 9.0f, 168, 30, '9')
            .withTargetsAir()
            .withSplash(1.5f));
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
            .withSplash(0.25f).withLineSplash(11.0f));
        add(troop(64, "Firecracker", 3.0f, Archetype::RangedSquad, 304, 0.7f, 6.0f, 64, 30, '$')
            .withTargetsAir()
            .withSplash(1.5f)
            .withRecoil(1.0f)); // confirmed: kicks back 1 tile after every attack
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
            .withFlying().withTargetsAir());
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
            .withHpTransform(0.5f, 150, true));
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
            .withOnHit(std::make_shared<FreezeOnHit>(3, 0.0f)));
        add(troop(80, "Three Musketeers", 9.0f, Archetype::RangedSquad, 722, 0.5f, 6.0f, 218, 10, ']')
            .withOffsets({ {-2.0f, 0.0f}, {0.0f, 0.0f}, {2.0f, 0.0f} })
            .withTargetsAir());

        // === New Building Targeters ===
        // Battle Ram: releases 2 Barbarians on death (reuses the already-
        // sourced Barbarians card's own stats, see battleRamBarbarianStats).
        add(troop(81, "Battle Ram", 4.0f, Archetype::MeleeBuildingTargeter, 691, 0.6f, 1.0f, 192, 14, '^')
            .withDeathEffect(std::make_shared<SpawnOnDeath>(battleRamBarbarianStats()))
            .withCharge(3.0f, 2.0f));
        add(troop(82, "Royal Hogs", 5.0f, Archetype::MeleeBuildingTargeter, 837, 0.85f, 1.0f, 74, 12, '_')
            .withOffsets({ {-1.0f, -0.3f}, {-0.3f, 0.3f}, {0.3f, -0.3f}, {1.0f, 0.3f} })
            .withCharge(3.0f, 2.0f));
        add(troop(83, "Wall Breakers", 2.0f, Archetype::MeleeBuildingTargeter, 330, 0.85f, 1.0f, 350, 12, '{')
            .withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f} })
            .withSplash(1.5f)
            .withDieAfterFirstHit());
        add(troop(84, "Electro Giant", 7.0f, Archetype::MeleeBuildingTargeter, 3952, 0.3f, 1.0f, 163, 18, '|')
            .withSplash(1.5f)
            .withPeriodicEffect(50, std::make_shared<AreaStunEffect>(2.5f, 5)));
        add(troop(85, "Suspicious Bush", 2.0f, Archetype::MeleeBuildingTargeter, 81, 0.5f, 0.25f, 256, 14, '}')
            .withInvisibility(5)
            .withDeathEffect(std::make_shared<SpawnOnDeath>(suspiciousBushGoblinStats())));
        add(troop(86, "Rune Giant", 4.0f, Archetype::MeleeBuildingTargeter, 2662, 0.5f, 1.2f, 153, 15, '~')
            .withAllyBuffAura(3.0f, 3, 1.5f, 50, 2));
        add(troop(87, "Ram Rider", 5.0f, Archetype::MeleeBuildingTargeter, 1766, 0.5f, 1.0f, 250, 17, '"')
            .withCharge(3.0f, 2.0f)
            .withSecondaryUnit(ramRiderCrossbowStats())); // rider's independently-targeting crossbow
        add(troop(88, "Goblin Giant", 6.0f, Archetype::MeleeBuildingTargeter, 3110, 0.5f, 1.2f, 176, 15, '`')
            .withSecondaryUnit(goblinGiantSpearGoblinsStats())); // carried Spear Goblins, independently-targeting
        // Skeleton Barrel: releases 2 Skeletons on death (reuses the
        // already-sourced Skeletons card's own stats).
        add(troop(89, "Skeleton Barrel", 3.0f, Archetype::MeleeBuildingTargeter, 532, 0.85f, 1.0f, 81, 10, ',')
            .withFlying()
            .withDeathEffect(std::make_shared<SpawnOnDeath>(skeletonBarrelSkeletonStats())));
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
                })));

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
            .withDeployDelay(35));
        add(building(93, "Mortar", 4.0f, 1369, 'R', 11.5f, 266, 50)
            .withMinRange(3.5f)); // confirmed blind spot

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

        // === Excluded from this sync (no supporting mechanism in this engine) ===
        // Champions (Golden Knight, Skeleton King, Archer Queen, Monk,
        // Mighty Miner, Little Prince, Goblinstein, Boss Bandit), Evolutions,
        // and Tower Troops (Tower Princess, Cannoneer, Dagger Duchess, Royal
        // Chef) were out of scope per the sync request and never researched.
        // Also excluded, for lack of any matching mechanism even
        // approximately:
        //   - Mirror: replays the last card played, at +1 elixir cost and
        //     +1 level -- needs "what was the last card played, by whom"
        //     state that lives in GameManager/PlayerState, a layer entirely
        //     above CardFactories/AreaSpell (which only ever see a single
        //     spawn point, not match history). No other card in this pass
        //     needed cross-layer state like this.
        //   - Spirit Empress: stateful dual-form (ground vs. flying)
        //     auto-switching has no equivalent and the exact switching rule
        //     couldn't be confirmed from sourced data.
        //     weight sub-unit types with independent HP pools, unlike e.g.
        //     Goblin Giant/Ram Rider/Goblin Machine above which have one
        //     clear primary body) -- no single-unit approximation fits
        //     without fabricating unsourced numbers.
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
