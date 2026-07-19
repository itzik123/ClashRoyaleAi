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
    static CardStats nightWitchBatStats() {
        return troop(-12, "Bats", 0.0f, Archetype::MeleeSquad, 81, 0.85f, 0.5f, 81, 12, 't')
            .withOffsets({ {-0.5f, 0.0f}, {0.0f, 0.0f}, {0.5f, 0.0f} })
            .withFlying().withTargetsAir();
    }

    void add(const CardStats& stats) {
        CardDefinition def;
        def.id = stats.id;
        def.name = stats.name;
        def.cost = stats.cost;
        def.isSpell = (stats.archetype == Archetype::Spell);
        def.isBuilding = (stats.archetype == Archetype::DefensiveBuilding);
        def.placementRadius = CardFactories::placementRadius(stats.archetype);
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
        // (Giant Skeleton, Lumberjack, Golem, Ice Golem, Balloon) isn't
        // modeled since there's no death-trigger system in this engine yet.
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

        add(troop(14, "Prince", 5.0f, Archetype::MeleeSquad, 1920, 0.6f, 1.6f, 391, 14, 'p'));

        add(troop(17, "Elite Barbarians", 6.0f, Archetype::MeleeSquad, 1341, 0.7f, 1.2f, 384, 14, 'e')
            .withOffsets({ {-0.5f, 0.0f}, {0.5f, 0.0f} }));

        add(troop(21, "Lumberjack", 4.0f, Archetype::MeleeSquad, 1282, 0.8f, 0.7f, 256, 8, 'l'));

        add(troop(24, "Skeletons", 1.0f, Archetype::MeleeSquad, 81, 1.0f, 0.5f, 81, 11, 'k')
            .withOffsets({ {0.0f, 0.0f}, {0.6f, 0.0f}, {-0.6f, 0.0f} }));

        add(troop(39, "Giant Skeleton", 6.0f, Archetype::MeleeSquad, 3361, 0.4f, 0.8f, 276, 13, 'J'));

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
        add(troop(22, "Bowler", 5.0f, Archetype::RangedSquad, 2081, 0.4f, 4.0f, 289, 25, 'w')
            .withSplash(1.5f)); // pierce-through-a-line-with-knockback not modeled, plain radius splash instead

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

        // Golem splits into two Golemites on death (its separate death-damage
        // splash isn't modeled, same as every other card's on-death damage).
        add(troop(19, "Golem", 8.0f, Archetype::MeleeBuildingTargeter, 5120, 0.2f, 0.75f, 312, 25, 'L')
            .withDeathEffect(std::make_shared<SpawnOnDeath>(golemiteStats())));

        // Ice Golem: same story as Ice Wizard -- BuildingTargeter's own
        // performAttack is already direct damage, so this is behavior-exact.
        add(troop(40, "Ice Golem", 2.0f, Archetype::MeleeBuildingTargeter, 1315, 0.4f, 0.75f, 84, 25, 'c')
            .withOnHit(std::make_shared<FreezeOnHit>(30, 0.65f)));

        // Balloon: flying, buildings-only, real point-blank 0.1 attack range
        // (BuildingTargeter's own findTarget already never considers
        // isFlying/targetsAir, since Buildings never fly -- so no
        // .withTargetsAir() needed here, matching that Balloon can't hit air
        // in the real game either). Its on-death area damage isn't modeled
        // (no death-triggered effects exist in this engine yet).
        add(troop(45, "Balloon", 5.0f, Archetype::MeleeBuildingTargeter, 1676, 0.5f, 0.1f, 640, 20, 'a')
            .withFlying());

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
        add(spell(7, "Fireball", 4.0f, 2.5f, 689, 10, 'O'));
        add(spell(29, "Zap", 2.0f, 2.5f, 192, 3, 'Z'));
        add(spell(30, "Rocket", 6.0f, 2.0f, 1485, 15, 'r'));
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
        add(troop(46, "Dark Prince", 4.0f, Archetype::MeleeSquad, 1200, 0.5f, 1.2f, 266, 14, 'N')); // shield + charge bonus not modeled
        add(troop(47, "Royal Ghost", 3.0f, Archetype::MeleeSquad, 1210, 0.7f, 1.2f, 261, 18, 'Q')); // invisibility-until-attack not modeled
        add(troop(48, "Mega Knight", 7.0f, Archetype::MeleeSquad, 3993, 0.5f, 1.2f, 268, 17, 'X')
            .withSplash(1.5f)); // deploy slam + periodic dash not modeled
        add(troop(49, "Battle Healer", 4.0f, Archetype::MeleeSquad, 1717, 0.5f, 1.2f, 148, 15, 'f')); // heal aura not modeled
        add(troop(50, "Bandit", 3.0f, Archetype::MeleeSquad, 906, 0.8f, 1.0f, 194, 10, 'u')); // dash + brief invuln not modeled
        add(troop(51, "Berserker", 2.0f, Archetype::MeleeSquad, 896, 0.7f, 1.0f, 102, 6, 'v')); // enrage-as-damaged not modeled
        add(troop(52, "Miner", 3.0f, Archetype::MeleeSquad, 1210, 0.7f, 1.0f, 194, 13, '0')); // deploy-anywhere (burrow) not modeled
        add(troop(53, "Fisherman", 3.0f, Archetype::MeleeSquad, 870, 0.5f, 1.0f, 194, 13, '1')); // hook-pull not modeled
        add(troop(54, "Ronin", 5.0f, Archetype::MeleeSquad, 1779, 0.7f, 1.2f, 371, 14, '2')); // periodic melee parry not modeled
        add(troop(55, "Goblin Machine", 5.0f, Archetype::MeleeSquad, 2150, 0.5f, 1.2f, 212, 12, '3')); // independent rocket launcher sub-unit not modeled

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
        add(troop(57, "Electro Dragon", 5.0f, Archetype::MeleeSquad, 1049, 0.5f, 3.5f, 192, 21, '5')
            .withFlying().withTargetsAir()
            .withSplitTargets(2)
            .withOnHit(std::make_shared<FreezeOnHit>(5, 0.0f))); // splash not modeled (split-target approximates the chain)

        // Night Witch: periodic bat-spawning while alive isn't modeled (no
        // such hook exists), but her on-death release of 3 Bats reuses the
        // same SpawnOnDeath mechanism as Golem/Battle Ram, with the newly-
        // registered Bats card's own stats (see nightWitchBatStats above).
        add(troop(58, "Night Witch", 4.0f, Archetype::MeleeSquad, 906, 0.5f, 1.0f, 314, 13, '6')
            .withDeathEffect(std::make_shared<SpawnOnDeath>(nightWitchBatStats())));
        // Phoenix: one-time revive-from-egg on death isn't modeled (no
        // multi-life system exists).
        add(troop(59, "Phoenix", 4.0f, Archetype::MeleeSquad, 1052, 0.5f, 1.0f, 217, 10, '7')
            .withFlying().withTargetsAir());

        // === New Ranged Troops ===
        add(troop(60, "Sparky", 6.0f, Archetype::RangedSquad, 1451, 0.3f, 5.0f, 1331, 40, '8')
            .withSplash(1.5f)); // charge-up-that-resets-on-stun not modeled
        add(troop(61, "Princess", 3.0f, Archetype::RangedSquad, 261, 0.5f, 9.0f, 168, 30, '9')
            .withTargetsAir()
            .withSplash(1.5f));
        add(troop(62, "Hunter", 4.0f, Archetype::RangedSquad, 885, 0.5f, 4.0f, 84, 22, '!')
            .withTargetsAir()
            .withSplash(1.5f)); // shotgun falloff-with-range not modeled, plain radius splash instead
        add(troop(63, "Magic Archer", 4.0f, Archetype::RangedSquad, 529, 0.5f, 7.0f, 143, 11, '#')
            .withTargetsAir()
            .withSplash(1.5f)); // real hit is a piercing line, not a radius -- approximated as splash
        add(troop(64, "Firecracker", 3.0f, Archetype::RangedSquad, 304, 0.7f, 6.0f, 64, 30, '$')
            .withTargetsAir()
            .withSplash(1.5f)); // recoil-kiting not modeled
        add(troop(65, "Skeleton Dragons", 4.0f, Archetype::RangedSquad, 560, 0.7f, 3.5f, 151, 20, '%')
            .withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f} }).withFlying().withTargetsAir()
            .withSplash(1.5f));
        add(troop(66, "Goblin Demolisher", 4.0f, Archetype::RangedSquad, 1300, 0.5f, 5.0f, 186, 11, '&')
            .withSplash(1.5f)); // below-50%-HP melee-bomber transform not modeled
        add(troop(67, "Flying Machine", 4.0f, Archetype::RangedSquad, 614, 0.7f, 6.0f, 171, 11, '+')
            .withFlying().withTargetsAir());
        add(troop(68, "Mother Witch", 4.0f, Archetype::RangedSquad, 529, 0.5f, 5.5f, 133, 10, ',')
            .withTargetsAir()); // curse-on-hit + cursed-death-spawns-goblin not modeled
        add(troop(69, "Cannon Cart", 5.0f, Archetype::RangedSquad, 1809, 0.5f, 5.5f, 212, 9, '?')); // mobile shield + post-shield transform-to-building not modeled
        add(troop(70, "Furnace", 4.0f, Archetype::RangedSquad, 727, 0.5f, 5.5f, 179, 17, '/')
            .withTargetsAir()); // reworked 2026 from Building to mobile Troop; periodic Fire Spirit spawn not modeled
        add(troop(71, "Witch", 5.0f, Archetype::RangedSquad, 839, 0.5f, 5.5f, 135, 11, ':')
            .withTargetsAir()); // periodic Skeleton spawn not modeled

        // "Spirit" troops: real game has them detonate once on arrival then
        // vanish; this engine has no kamikaze/one-shot-then-die primitive,
        // so they're modeled as small, cheap, ordinary ranged units instead
        // (they'll survive and keep re-attacking rather than vanishing).
        add(troop(72, "Ice Spirit", 1.0f, Archetype::RangedSquad, 230, 0.85f, 2.5f, 110, 10, ';')
            .withTargetsAir()
            .withOnHit(std::make_shared<FreezeOnHit>(10, 0.5f)));
        add(troop(73, "Fire Spirit", 1.0f, Archetype::RangedSquad, 230, 0.85f, 2.5f, 207, 10, '<')
            .withTargetsAir()); // splash not modeled
        add(troop(74, "Heal Spirit", 1.0f, Archetype::RangedSquad, 230, 0.85f, 2.5f, 110, 10, '=')
            .withTargetsAir()); // ally heal-on-hit not modeled
        add(troop(75, "Electro Spirit", 1.0f, Archetype::RangedSquad, 230, 0.85f, 2.5f, 99, 10, '>')
            .withTargetsAir()
            .withSplitTargets(2)
            .withOnHit(std::make_shared<FreezeOnHit>(8, 0.0f)));

        // === New Swarms ===
        add(troop(76, "Guards", 3.0f, Archetype::MeleeSquad, 81, 0.7f, 1.0f, 117, 10, '?')
            .withOffsets({ {0.0f, 0.0f}, {0.6f, 0.0f}, {-0.6f, 0.0f} })); // personal shields not modeled
        add(troop(77, "Royal Recruits", 7.0f, Archetype::MeleeSquad, 547, 0.5f, 1.0f, 133, 13, '@')
            .withOffsets({ {-2.5f, 0.0f}, {-1.5f, 0.0f}, {-0.5f, 0.0f}, {0.5f, 0.0f}, {1.5f, 0.0f}, {2.5f, 0.0f} })); // personal shields not modeled
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
            .withDeathEffect(std::make_shared<SpawnOnDeath>(battleRamBarbarianStats()))); // charge bonus damage not modeled
        add(troop(82, "Royal Hogs", 5.0f, Archetype::MeleeBuildingTargeter, 837, 0.85f, 1.0f, 74, 12, '_')
            .withOffsets({ {-1.0f, -0.3f}, {-0.3f, 0.3f}, {0.3f, -0.3f}, {1.0f, 0.3f} }));
        add(troop(83, "Wall Breakers", 2.0f, Archetype::MeleeBuildingTargeter, 330, 0.85f, 1.0f, 350, 12, '{')
            .withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f} })
            .withSplash(1.5f)); // kamikaze one-shot (dies after its single hit) not modeled
        add(troop(84, "Electro Giant", 7.0f, Archetype::MeleeBuildingTargeter, 3952, 0.3f, 1.0f, 163, 18, '|')
            .withSplash(1.5f)); // periodic shock aura not modeled
        add(troop(85, "Suspicious Bush", 2.0f, Archetype::MeleeBuildingTargeter, 81, 0.5f, 0.25f, 256, 14, '}')); // disguise/invisibility + on-death Bush Goblins not modeled (no sourced stats for the split)
        add(troop(86, "Rune Giant", 4.0f, Archetype::MeleeBuildingTargeter, 2662, 0.5f, 1.2f, 153, 15, '~')); // ally-buff-on-attack aura not modeled
        add(troop(87, "Ram Rider", 5.0f, Archetype::MeleeBuildingTargeter, 1766, 0.5f, 1.0f, 250, 17, '"')); // rider's independent crossbow + charge not modeled
        add(troop(88, "Goblin Giant", 6.0f, Archetype::MeleeBuildingTargeter, 3110, 0.5f, 1.2f, 176, 15, '`')); // carried Spear Goblins sub-unit not modeled
        // Skeleton Barrel: releases 2 Skeletons on death (reuses the
        // already-sourced Skeletons card's own stats).
        add(troop(89, "Skeleton Barrel", 3.0f, Archetype::MeleeBuildingTargeter, 532, 0.85f, 1.0f, 81, 10, ',')
            .withFlying()
            .withDeathEffect(std::make_shared<SpawnOnDeath>(skeletonBarrelSkeletonStats())));
        add(troop(90, "Elixir Golem", 3.0f, Archetype::MeleeBuildingTargeter, 1569, 0.3f, 1.0f, 253, 11, '(')); // split-into-smaller-golems-on-death chain + opponent-elixir-grant not modeled (no sourced stats for the split tiers)

        // === New Ranged Building Targeter ===
        add(troop(91, "Lava Hound", 7.0f, Archetype::RangedBuildingTargeter, 3581, 0.3f, 3.5f, 53, 13, ')')
            .withFlying()); // split-into-Lava-Pups-on-death not modeled (no sourced pup stats)

        // === New Defensive Structures ===
        add(building(92, "X-Bow", 6.0f, 1600, 'P', 11.5f, 43, 3)); // 30s lifespan + slow deploy not modeled (same simplification as every building's implicit indefinite lifetime already)
        add(building(93, "Mortar", 4.0f, 1369, 'R', 11.5f, 266, 50)); // minimum-range blind spot not modeled

        // Spawner buildings: none of these attack in the real game (see
        // ClashStrategic's own data flagging their damage fields as
        // vestigial) -- modeled as inert 0-damage structures. Their entire
        // real function (periodic troop spawning, or a release-on-death
        // burst) isn't modeled since no periodic-spawn-while-alive hook
        // exists and none of these have sourced stats for their spawned
        // child units.
        add(building(94, "Barbarian Hut", 6.0f, 1164, '2', 0.0f, 0, 100));
        add(building(95, "Goblin Hut", 4.0f, 1180, '3', 0.0f, 0, 100));
        add(building(96, "Tombstone", 3.0f, 529, '5', 0.0f, 0, 100));
        add(building(97, "Goblin Cage", 4.0f, 780, '6', 0.0f, 0, 100));
        add(building(98, "Goblin Drill", 4.0f, 1313, '7', 0.0f, 0, 100)); // burrow-to-target + emergence burst + periodic Goblin spawn not modeled
        // Elixir Collector: no attack, and passive elixir generation isn't
        // modeled (no per-entity-drives-player-elixir hook exists) -- this
        // engine's version is just inert HP for the opponent to decide
        // whether to punish, same "not modeled" simplification as above.
        add(building(99, "Elixir Collector", 6.0f, 1070, '9', 0.0f, 0, 100));

        // === New Spells ===
        add(spell(100, "Giant Snowball", 2.0f, 2.5f, 179, 8, '!')
            .withSpellOnHit(std::make_shared<FreezeOnHit>(15, 0.5f))); // pushback not modeled; slow approximated via the new spellOnHit hook
        // Barbarian Barrel: the rolling damage is modeled; spawning a
        // Barbarian at the landing point isn't (no sourced stats gap here --
        // it's the same "spell spawns a troop" mechanism Goblin Barrel/
        // Graveyard/Royal Delivery need and this engine doesn't have).
        add(spell(101, "Barbarian Barrel", 2.0f, 2.5f, 233, 8, '#').withGroundOnly());
        // Goblin Curse: damage-over-time portion modeled via withRepeats;
        // the damage-taken-amplification debuff and cursed-death-spawns-a-
        // Goblin bonus aren't (no debuff-modifier or on-cursed-death hook).
        add(spell(102, "Goblin Curse", 2.0f, 3.0f, 43, 8, '$').withRepeats(6, 10));
        // Earthquake: ground-only DoT, same shape as Poison. Bonus damage
        // vs. buildings and the ground move/attack-speed slow aren't
        // modeled (flat DoT to everyone in radius instead).
        add(spell(103, "Earthquake", 3.0f, 3.5f, 82, 8, '%').withRepeats(3, 10).withGroundOnly());
        // Void: real damage scales inversely with troops caught (fewer
        // troops = more damage each); not modeled -- flat per-wave damage
        // regardless of how many targets are in range.
        add(spell(104, "Void", 3.0f, 2.5f, 85, 8, '&').withRepeats(3, 10));
        // Vines: real card roots only the top-3-HP enemies and pulls flyers
        // to the ground; this engine's version just hits everyone in radius
        // each tick like every other multi-tick spell (root/pull/highest-HP
        // targeting not modeled).
        add(spell(105, "Vines", 3.0f, 2.5f, 135, 8, '+').withRepeats(3, 7));
        add(spell(106, "Tornado", 3.0f, 5.5f, 154, 8, '_')); // pull-to-center not modeled
        // Freeze: the source data's "damage" field for this card is treated
        // as vestigial (same judgment call ClashStrategic's own data made
        // for the non-attacking spawner buildings above) -- the real card's
        // entire function is the stun, modeled here as 0 damage plus a full
        // (slowFactor 0.0) 4s freeze via the new spellOnHit hook.
        add(spell(107, "Freeze", 4.0f, 3.0f, 0, 8, '~')
            .withSpellOnHit(std::make_shared<FreezeOnHit>(40, 0.0f)));

        // === Excluded from this sync (no supporting mechanism in this engine) ===
        // Champions (Golden Knight, Skeleton King, Archer Queen, Monk,
        // Mighty Miner, Little Prince, Goblinstein, Boss Bandit), Evolutions,
        // and Tower Troops (Tower Princess, Cannoneer, Dagger Duchess, Royal
        // Chef) were out of scope per the sync request and never researched.
        // Also excluded, for lack of any matching mechanism even
        // approximately:
        //   - Clone, Mirror, Rage: buff/duplicate/replay-last-card spells --
        //     no buff-zone, troop-duplication, or play-history hook exists.
        //   - Goblin Barrel, Graveyard, Royal Delivery: "spell spawns
        //     troops" is a different mechanism from every other spell here
        //     (which all apply direct area damage) and doesn't exist.
        //   - Spirit Empress: stateful dual-form (ground vs. flying)
        //     auto-switching has no equivalent and the exact switching rule
        //     couldn't be confirmed from sourced data.
        //   - Goblin Gang, Rascals: genuinely compound cards (two equal-
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
