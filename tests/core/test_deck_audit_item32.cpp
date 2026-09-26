#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "Board.h"
#include "GameManager.h"
#include "CardRegistry.h"
#include "CombatEntity.h"
#include "MeleeTroop.h"
#include "Troop.h"
#include "Building.h"
#include "Tower.h"
#include "StatsEventBus.h"
#include "HeroBarbarianBarrelRerollEffect.h"
#include <cmath>
#include <memory>
#include <set>
#include <vector>

// perception/UPSTREAM_REQUESTS.md item 32: a real ladder deck audited card by
// card against the live game (Evo Furnace, Hero Barbarian Barrel, Evo Bats,
// Poison, Giant Skeleton, Graveyard, Berserker, Goblin Hut), then fixed. Each
// case pins the corrected BEHAVIOUR through the real pipeline, and each would
// fail on the pre-fix engine; the measured "before" is in the item. The
// instrument that found them is tools/audit/gy_deck_audit.cpp.

namespace {

constexpr int FURNACE = 70;
constexpr int EVO_FURNACE = 138;
constexpr int HERO_BARBARIAN_BARREL = 174;
constexpr int BARBARIAN_BARREL = 101;
constexpr int EVO_BATS = 126;
constexpr int POISON = 32;
constexpr int GIANT_SKELETON = 39;
constexpr int GRAVEYARD = 110;
constexpr int BERSERKER = 51;
constexpr int GOBLIN_HUT = 95;
constexpr int BARBARIANS = 8;
constexpr int KNIGHT = 0;
constexpr int THE_LOG = 33;

// The audited deck, in slot order: Evolution, Hero, Wild, then plain.
const std::vector<int>& deck() {
    static const std::vector<int> d = { EVO_FURNACE, HERO_BARBARIAN_BARREL, EVO_BATS, POISON,
                                        GIANT_SKELETON, GRAVEYARD, BERSERKER, GOBLIN_HUT };
    return d;
}

struct Recorder : IStatsObserver {
    std::vector<DamageDealtEvent> damage;
    void onDamageDealt(const DamageDealtEvent& e) override { damage.push_back(e); }
};

// A real match with every tower asleep (Tower::sleep: acquires nothing until
// damaged), so only the units a case places act. Not applyFreeze: a stunned
// tower would still fire once if its cooldown were ready -- see the stun cases.
struct Match {
    GameManager game;
    std::shared_ptr<Recorder> rec = std::make_shared<Recorder>();

    Match() : game(deck(), deck()) {
        game.seed(1);
        game.reset();
        game.getBoard().statsEvents.subscribe(rec);
        for (const auto& e : game.getBoard().getEntities())
            if (auto t = std::dynamic_pointer_cast<Tower>(e)) t->sleep();
    }
    Board& board() { return game.getBoard(); }
    void step(int n = 1) { for (int i = 0; i < n; ++i) game.step(); }

    std::vector<std::shared_ptr<Entity>> spawnCard(int cardId, float x, float y, int team, bool evolved = false) {
        std::set<int> before;
        for (const auto& e : board().getEntities()) before.insert(e->id);
        const CardDefinition* def = CardRegistry::getInstance().getCard(cardId);
        if (evolved) def->spawnEvolvedEntity(x, y, team, board());
        else def->spawnEntity(x, y, team, board());
        board().commitPendingEntities();
        std::vector<std::shared_ptr<Entity>> out;
        for (const auto& e : board().getEntities())
            if (!before.count(e->id)) out.push_back(e);
        return out;
    }
    // A harmless, immobile target: StationaryCombatant with 0 damage.
    std::shared_ptr<StationaryCombatant> dummy(float x, float y, int team, int hp, bool flying = false) {
        auto d = std::make_shared<StationaryCombatant>(board().allocateId(), x, y, hp, team, 0.5f, 0, 1000);
        d->isFlying = flying;
        spawn(board(), d);
        return d;
    }
    // The first live CombatEntity of this name on `team`, stepping until one
    // exists. Skips spells: Hero Barbarian Barrel's spell shares its Hero's
    // name.
    std::shared_ptr<CombatEntity> stepUntil(const std::string& name, int team, int maxTicks) {
        for (int i = 0; i <= maxTicks; ++i) {
            for (const auto& e : board().getEntities())
                if (e->isAlive() && e->team == team && e->name == name)
                    if (auto c = std::dynamic_pointer_cast<CombatEntity>(e)) return c;
            step();
        }
        return nullptr;
    }
    std::vector<int> hitTicksBy(int attackerId) const {
        std::vector<int> t;
        for (const auto& d : rec->damage) if (d.attackerId == attackerId) t.push_back(d.tick);
        return t;
    }
    std::shared_ptr<Entity> princessTower(int team, float x) {
        for (const auto& e : board().getEntities())
            if (e->isTower() && e->team == team && e->symbol != 'R' && std::fabs(e->position.x - x) < 0.1f) return e;
        return nullptr;
    }
};

float tierSpeed(float tier) { return tier * MOVEMENT_SPEED_SCALE; }

} // namespace

// ---------------- 32a: Hero Barbarian Barrel's ability is reachable ----------------

TEST_CASE("Hero Barbarian Barrel's Rowdy Reroll can be activated through the real play path",
          "[item32][hero_barbarian_barrel][regression]") {
    // The Barbarian lands ~17 ticks after the play, as a -47 helper. It used to
    // be tracked only at play time and by the deck card's id, so the ability
    // was never ready and activating it did nothing, all match.
    Match m;
    REQUIRE(m.game.setHand(0, { HERO_BARBARIAN_BARREL, POISON, GIANT_SKELETON, GRAVEYARD }));
    m.game.setElixir(0, 10.0f);
    REQUIRE(m.game.playCard(0, HERO_BARBARIAN_BARREL, 8.5f, 6.0f));
    REQUIRE_FALSE(m.game.isChampionAbilityReady(0, 1)); // the barrel is still rolling

    m.step(25);
    REQUIRE(m.game.isChampionAbilityReady(0, 1));
    const float before = m.game.getElixir(0);
    REQUIRE(m.game.activateChampionAbility(0, 1));
    REQUIRE(m.game.getElixir(0) == Catch::Approx(before - 1.0f));

    // Single use (every Hero ability since the official August 2026 notes).
    REQUIRE_FALSE(m.game.isChampionAbilityReady(0, 1));
    REQUIRE_FALSE(m.game.activateChampionAbility(0, 1));
}

TEST_CASE("A cloned Hero Barbarian never gets the ability, even after the original dies",
          "[item32][hero_barbarian_barrel][regression]") {
    // The late-spawn adoption that fixed the Barbarian must not adopt a Clone
    // copy, which carries isHero; the rule the Champion Clone test pins.
    Match m;
    REQUIRE(m.game.setHand(0, { HERO_BARBARIAN_BARREL, POISON, GIANT_SKELETON, GRAVEYARD }));
    m.game.setElixir(0, 10.0f);
    REQUIRE(m.game.playCard(0, HERO_BARBARIAN_BARREL, 8.5f, 6.0f));
    auto original = m.stepUntil("Hero Barbarian Barrel", 0, 40);
    REQUIRE(original);
    m.step();
    REQUIRE(m.game.isChampionAbilityReady(0, 1));

    auto copy = std::dynamic_pointer_cast<CombatEntity>(original->clone(m.board().allocateId()));
    REQUIRE(copy);
    REQUIRE(copy->isHero);
    m.board().addEntity(copy);
    m.step();
    REQUIRE(m.game.isChampionAbilityReady(0, 1)); // still the original

    original->takeDamage(original->hp);
    m.step();
    REQUIRE_FALSE(m.game.isChampionAbilityReady(0, 1));
    REQUIRE_FALSE(m.game.activateChampionAbility(0, 1));
}

// ---------------- 32h: what the reroll does ----------------

TEST_CASE("Rowdy Reroll sweeps the barrel's 2.6-wide corridor, ground only, and heals half the damage",
          "[item32][hero_barbarian_barrel]") {
    Board board;
    auto barbarian = std::make_shared<StationaryCombatant>(1, 8.5f, 5.0f, 716, 0, 0.7f, 192, 14);
    barbarian->hp = 300;
    // Surface test: a 0.4-radius troop is inside while its centre is within
    // 1.3 + 0.4 = 1.7 of the line.
    auto inCorridor = std::make_shared<StationaryCombatant>(2, 8.5f + 1.6f, 7.0f, 10000, 1, 1.0f, 0, 10);
    auto outside    = std::make_shared<StationaryCombatant>(3, 8.5f - 1.8f, 7.0f, 10000, 1, 1.0f, 0, 10);
    auto flyer      = std::make_shared<StationaryCombatant>(4, 8.5f, 6.5f, 10000, 1, 1.0f, 0, 10);
    flyer->isFlying = true;
    auto tower = std::make_shared<Tower>(5, 8.5f, 8.5f, 5000, 1, 7.0f, 90, 10, 'P');
    for (auto e : std::vector<std::shared_ptr<Entity>>{ barbarian, inCorridor, outside, flyer, tower }) spawn(board, e);

    HeroBarbarianBarrelRerollEffect effect(3.0f, 1.3f, 232, 716);
    effect.apply(board, *barbarian);

    REQUIRE(inCorridor->hp == 10000 - 232);
    REQUIRE(outside->hp == 10000);
    REQUIRE(flyer->hp == 10000);          // the barrel rolls along the ground
    REQUIRE(tower->hp == 5000 - 116);     // halved against a Crown Tower
    REQUIRE(barbarian->hp == 300 + (232 + 116) / 2);

    SECTION("the heal never overheals past the Barbarian's hp") {
        barbarian->hp = 700;
        barbarian->position = { 8.5f, 5.0f };
        effect.apply(board, *barbarian);
        REQUIRE(barbarian->hp == 716);
    }
}

// ---------------- 32b: the Furnace's Fire Spirits ----------------

TEST_CASE("A Furnace's Fire Spirit hits once, for 215 area damage, and dies",
          "[item32][furnace][regression]") {
    // It used to lack both the kamikaze flag and the splash: one spirit hit a
    // clump 7 times in 8 s and was still alive.
    Match m;
    auto furnace = std::dynamic_pointer_cast<CombatEntity>(m.spawnCard(FURNACE, 8.5f, 6.0f, 0).front());
    furnace->applyFreeze(100000, 0.0f); // held in place; its spawn timer still runs
    auto spirit = m.stepUntil("Fire Spirit", 0, 200);
    REQUIRE(spirit);
    furnace->hp = 0; // keep only the spirit
    m.step();

    auto a = m.dummy(8.0f, 10.0f, 1, 1000);
    auto b = m.dummy(9.0f, 10.0f, 1, 1000);
    auto c = m.dummy(8.5f, 10.8f, 1, 1000);
    m.step(60);

    REQUIRE(a->hp == 1000 - 215);
    REQUIRE(b->hp == 1000 - 215);
    REQUIRE(c->hp == 1000 - 215);
    REQUIRE(m.hitTicksBy(spirit->id).size() == 3); // one shot: the target plus its splash
    REQUIRE_FALSE(spirit->isAlive());
}

// ---------------- 32c: Giant Skeleton's bomb ----------------

TEST_CASE("Giant Skeleton's bomb explodes 3.0 s after he dies, 886 in a 3-tile radius, and throws",
          "[item32][giant_skeleton][regression]") {
    Match m;
    auto gs = m.spawnCard(GIANT_SKELETON, 8.5f, 10.0f, 0).front();
    REQUIRE(gs->hp == 3126); // 3361 x 0.93 (May 2026)
    m.step(10);
    // Surface test: radius 3 plus the 0.4 troop radius.
    auto near = m.dummy(8.5f + 2.0f, 10.0f, 1, 5000);
    auto edge = m.dummy(8.5f - 3.3f, 10.0f, 1, 5000);
    auto away = m.dummy(8.5f, 10.0f + 3.5f, 1, 5000);
    const Vector2D nearBefore = near->position;
    gs->hp = 0;

    m.step(30); // the death tick, then 29 more: still fused
    REQUIRE(near->hp == 5000);
    REQUIRE(edge->hp == 5000);

    m.step(); // 3.0 s after the death
    REQUIRE(near->hp == 5000 - 886);
    REQUIRE(edge->hp == 5000 - 886);
    REQUIRE(away->hp == 5000);
    REQUIRE(near->position.x > nearBefore.x + 0.5f); // knocked away from the bomb
}

TEST_CASE("A Giant Skeleton dying at a Princess Tower bombs it for the full 886",
          "[item32][giant_skeleton][regression]") {
    // It used to deal the tower 0: an instant 2.0-radius centre test could not
    // reach a tower centre 2.7 tiles away.
    Match m;
    auto tower = m.princessTower(1, ArenaLayout::LEFT_LANE_X);
    REQUIRE(tower);
    // Where he stands to swing at it: 1.5 (tower) + 0.4 (troop) + 0.8 (reach).
    auto gs = m.spawnCard(GIANT_SKELETON, ArenaLayout::LEFT_LANE_X, tower->position.y - 2.7f, 0).front();
    gs->hp = 0;
    const int before = tower->hp;
    m.step(31);
    REQUIRE(before - tower->hp == 886);
}

// ---------------- 32d: Berserker ----------------

TEST_CASE("Berserker hits every 0.6 s at any hp, moves Fast, and reaches 0.8",
          "[item32][berserker][regression]") {
    // An invented "enrage" sped her up to 0.4 s as she lost hp.
    for (int hp : { 896, 448, 90 }) {
        Match m;
        auto berserker = m.spawnCard(BERSERKER, 8.5f, 10.0f, 0).front();
        berserker->hp = hp;
        m.dummy(8.5f, 11.5f, 1, 1000000);
        m.step(80);
        auto ticks = m.hitTicksBy(berserker->id);
        INFO("berserker at " << hp << " hp");
        REQUIRE(ticks.size() >= 5);
        for (size_t i = 1; i < ticks.size(); ++i) REQUIRE(ticks[i] - ticks[i - 1] == 6);
    }
    Match m;
    auto berserker = std::dynamic_pointer_cast<Troop>(m.spawnCard(BERSERKER, 8.5f, 10.0f, 0).front());
    REQUIRE(berserker->getSpeed() == Catch::Approx(tierSpeed(SPEED_FAST)));
    REQUIRE(berserker->getAttackRange() == Catch::Approx(0.8f));
}

// ---------------- 32e: speed tiers ----------------

TEST_CASE("Furnace and the Hero Barbarian move at Medium, like the Barbarians card",
          "[item32][speed][regression]") {
    Match m;
    const float medium = tierSpeed(SPEED_MEDIUM);
    auto barbarian = std::dynamic_pointer_cast<Troop>(m.spawnCard(BARBARIANS, 4.0f, 8.0f, 0).front());
    REQUIRE(barbarian->getSpeed() == Catch::Approx(medium));

    auto furnace = std::dynamic_pointer_cast<Troop>(m.spawnCard(FURNACE, 8.5f, 8.0f, 0).front());
    REQUIRE(furnace->getSpeed() == Catch::Approx(medium));
    auto evoFurnace = std::dynamic_pointer_cast<Troop>(m.spawnCard(EVO_FURNACE, 12.0f, 8.0f, 0, true).front());
    REQUIRE(evoFurnace->getSpeed() == Catch::Approx(medium));

    m.spawnCard(HERO_BARBARIAN_BARREL, 11.0f, 3.0f, 0);
    auto hero = std::dynamic_pointer_cast<Troop>(m.stepUntil("Hero Barbarian Barrel", 0, 40));
    REQUIRE(hero);
    REQUIRE(hero->getSpeed() == Catch::Approx(barbarian->getSpeed()));
    REQUIRE(hero->hp == 716);
}

// ---------------- 32f: Evolved Furnace's Hot Spawning ----------------

namespace {
struct SpiritLog { std::vector<int> ticks; std::vector<float> dx; };
SpiritLog spiritSpawns(Match& m, const std::shared_ptr<Entity>& furnace, int ticks) {
    SpiritLog log;
    std::set<int> seen;
    for (int i = 0; i < ticks; ++i) {
        m.step();
        for (const auto& e : m.board().getEntities())
            if (e->name == "Fire Spirit" && e->team == 0 && !seen.count(e->id)) {
                seen.insert(e->id);
                log.ticks.push_back(m.game.getCurrentTick());
                log.dx.push_back(e->position.x - furnace->position.x);
            }
    }
    return log;
}
}

TEST_CASE("Evolved Furnace spawns every 2.4 s only while attacking, to alternating sides",
          "[item32][furnace][evolution][regression]") {
    SECTION("idle, it spawns at the base card's 7 s") {
        Match m;
        auto f = m.spawnCard(EVO_FURNACE, 8.5f, 6.0f, 0, true).front();
        std::dynamic_pointer_cast<CombatEntity>(f)->applyFreeze(1000, 0.0f); // held, nothing in sight
        auto log = spiritSpawns(m, f, 160);
        REQUIRE(log.ticks == std::vector<int>{ 80, 150 });
    }
    SECTION("attacking, every 2.4 s, left then right") {
        Match m;
        auto f = m.spawnCard(EVO_FURNACE, 8.5f, 6.0f, 0, true).front();
        m.dummy(8.5f, 10.5f, 1, 100000000); // in its 5.5 range from the start
        auto log = spiritSpawns(m, f, 110);
        // First attack on tick 11; hot from tick 12, which pulls the timer in.
        REQUIRE(log.ticks == std::vector<int>{ 35, 59, 83, 107 });
        REQUIRE(log.dx[0] == Catch::Approx(-1.0f).margin(0.05f)); // team 0: left is -x
        REQUIRE(log.dx[1] == Catch::Approx(1.0f).margin(0.05f));
        REQUIRE(log.dx[2] == Catch::Approx(-1.0f).margin(0.05f));
    }
}

// ---------------- 32g: Evolved Bats ----------------

TEST_CASE("Evolved Bats spawn at 122 and heal 38 twice per attack, 0.5 s apart, up to 244",
          "[item32][bats][evolution][regression]") {
    Match m;
    auto bats = m.spawnCard(EVO_BATS, 8.5f, 10.0f, 0, true);
    REQUIRE(bats.size() == 5);
    for (const auto& b : bats) REQUIRE(b->hp == 122);
    auto bat = std::dynamic_pointer_cast<CombatEntity>(bats.front());
    m.dummy(bat->position.x, bat->position.y + 1.2f, 1, 1000000);
    m.step(10);
    bat->hp = 40;

    int hitTick = -1;
    for (int i = 0; i < 40 && hitTick < 0; ++i) {
        m.step();
        auto hits = m.hitTicksBy(bat->id);
        if (!hits.empty()) hitTick = hits.front();
    }
    REQUIRE(hitTick > 0);
    REQUIRE(bat->hp == 40 + 38);
    m.step(4);
    REQUIRE(bat->hp == 40 + 38);
    m.step();                         // 5 ticks after the hit
    REQUIRE(bat->hp == 40 + 76);

    bat->hp = 230;
    m.step(20);                       // at least one more hit
    REQUIRE(bat->hp == 244);          // the overheal cap
}

// ---------------- 32i + 32k + 32l + item 29: Poison ----------------

TEST_CASE("Poison pulses exactly 1 s apart, 21 to a Crown Tower, and reaches a unit overlapping its edge",
          "[item32][poison][regression]") {
    SECTION("cadence and troop damage") {
        Match m;
        auto target = m.dummy(8.5f, 20.0f, 1, 100000);
        auto spell = m.spawnCard(POISON, 8.5f, 20.0f, 0).front();
        m.step(100);
        auto ticks = m.hitTicksBy(spell->id);
        REQUIRE(ticks.size() == 8);
        for (size_t i = 1; i < ticks.size(); ++i) REQUIRE(ticks[i] - ticks[i - 1] == 10); // was 11
        REQUIRE(target->hp == 100000 - 8 * 92);
    }
    SECTION("a Crown Tower takes 21 a pulse") {
        Match m;
        auto tower = m.princessTower(1, ArenaLayout::LEFT_LANE_X);
        const int before = tower->hp;
        m.spawnCard(POISON, tower->position.x, tower->position.y, 0);
        m.step(100);
        REQUIRE(before - tower->hp == 8 * 21);
    }
    SECTION("hitbox overlap: a troop centred 3.6 out is inside, one at 3.95 is not") {
        Match m;
        auto inside = m.dummy(8.5f + 3.6f, 20.0f, 1, 100000);
        auto outside = m.dummy(8.5f - 3.95f, 20.0f, 1, 100000);
        m.spawnCard(POISON, 8.5f, 20.0f, 0);
        m.step();
        REQUIRE(inside->hp == 100000 - 92);
        REQUIRE(outside->hp == 100000);
    }
}

TEST_CASE("Poison slows enemy troop movement 15%, not their attacks, and not buildings",
          "[item32][poison][regression]") {
    Match m;
    auto knight = std::dynamic_pointer_cast<Troop>(m.spawnCard(KNIGHT, 8.5f, 24.0f, 1).front());
    m.step(12);
    m.spawnCard(POISON, knight->position.x, knight->position.y - 1.5f, 0);
    m.step(2); // the first pulse lands; from the next tick it walks slowed
    Vector2D prev = knight->position;
    m.step();
    const float stepLen = std::hypot(knight->position.x - prev.x, knight->position.y - prev.y);
    REQUIRE(stepLen == Catch::Approx(knight->getSpeed() * 0.85f).epsilon(0.02));
    REQUIRE(knight->freezeTicks == 0); // movement-only: no freeze, so no slower cooldown

    auto hut = std::dynamic_pointer_cast<CombatEntity>(m.spawnCard(GOBLIN_HUT, 8.5f, 21.0f, 1).front());
    m.spawnCard(POISON, 8.5f, 21.0f, 0);
    m.step(2);
    REQUIRE(hut->moveSlowTicks == 0); // buildings are not slowed
}

// ---------------- 32i + 32j: Graveyard ----------------

TEST_CASE("Graveyard raises 12 Skeletons on seven fixed points, the first at 2.2 s, then every 0.5 s",
          "[item32][graveyard][regression]") {
    Match m;
    const Vector2D cast{ 8.5f, 22.0f };
    m.spawnCard(GRAVEYARD, cast.x, cast.y, 0);
    const int t0 = m.game.getCurrentTick();
    std::vector<int> ticks;
    std::vector<float> dist;
    std::set<int> seen;
    for (int i = 0; i < 100; ++i) {
        m.step();
        for (const auto& e : m.board().getEntities())
            if (e->name == "Skeletons" && e->team == 0 && !seen.count(e->id)) {
                seen.insert(e->id);
                ticks.push_back(m.game.getCurrentTick() - t0);
                dist.push_back(e->position.distanceTo(cast));
            }
    }
    REQUIRE(ticks.size() == 12);
    REQUIRE(ticks.front() == 22);                                               // was 9
    for (size_t i = 1; i < ticks.size(); ++i) REQUIRE(ticks[i] - ticks[i - 1] == 5); // was 6
    // Seven points, cycled: the cast point, then six on the 3.3 ring. Before,
    // every one rose on the cast point, where one small splash caught them all.
    for (size_t k = 0; k < dist.size(); ++k) {
        INFO("skeleton " << k);
        if (k % 7 == 0) REQUIRE(dist[k] == Catch::Approx(0.0f).margin(0.35f));
        else REQUIRE(dist[k] == Catch::Approx(3.3f).margin(0.35f));
    }
}

// ---------------- 32m: stun and slow ----------------

TEST_CASE("A slow that lands after an earlier stun has ended is a slow, not a stun",
          "[item32][freeze][regression]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 12.0f, 100, 1);
    spawn(board, target);
    auto troop = std::make_shared<MeleeTroop>(2, 0.0f, 0.0f, 100, 0, 0.1f, 1.0f, 10, 10, 'K');
    troop->sightRange = 20.0f;
    spawn(board, troop);

    troop->applyFreeze(5, 0.0f);
    for (int i = 0; i < 5; ++i) troop->update(board);
    REQUIRE(troop->freezeTicks == 0);
    REQUIRE(troop->freezeSlow == Catch::Approx(1.0f)); // the stun's factor ends with it
    for (int i = 0; i < 10; ++i) troop->update(board);

    troop->applyFreeze(10, 0.5f);
    const float y0 = troop->position.y;
    troop->update(board);
    REQUIRE(troop->position.y - y0 == Catch::Approx(0.05f)); // was 0: the old stun's factor stuck
}

TEST_CASE("A short stun during a long slow is the stun, then the rest of the slow",
          "[item32][freeze][regression]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 12.0f, 100, 1);
    spawn(board, target);
    auto troop = std::make_shared<MeleeTroop>(2, 0.0f, 0.0f, 100, 0, 0.1f, 1.0f, 10, 10, 'K');
    troop->sightRange = 20.0f;
    spawn(board, troop);

    troop->applyFreeze(30, 0.5f);
    troop->update(board);
    troop->update(board);
    troop->applyFreeze(3, 0.0f);
    const float yStun = troop->position.y;
    for (int i = 0; i < 3; ++i) troop->update(board);
    REQUIRE(troop->position.y == Catch::Approx(yStun)); // stunned for its 3 ticks

    const float ySlow = troop->position.y;
    troop->update(board);
    REQUIRE(troop->position.y - ySlow == Catch::Approx(0.05f)); // the slow resumes, not the stun
    for (int i = 0; i < 30; ++i) troop->update(board);
    const float yFree = troop->position.y;
    troop->update(board);
    REQUIRE(troop->position.y - yFree == Catch::Approx(0.1f)); // and ends on its own clock
}

TEST_CASE("A stunned unit whose attack is ready does not swing until the stun ends",
          "[item32][freeze][regression]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 100000, 1);
    spawn(board, target);
    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 1000, 0, 5.0f, 50, 10);
    spawn(board, attacker);

    attacker->applyFreeze(20, 0.0f); // idle: its cooldown is already 0
    for (int i = 0; i < 20; ++i) attacker->update(board);
    REQUIRE(attacker->attackCount == 0); // was 1: it swung through the stun
    attacker->update(board);
    REQUIRE(attacker->attackCount == 1);

    SECTION("a slowed unit still attacks, only slower") {
        auto slowed = std::make_shared<StationaryCombatant>(3, 1.0f, 0.0f, 1000, 0, 5.0f, 50, 10);
        spawn(board, slowed);
        slowed->applyFreeze(20, 0.5f);
        slowed->update(board);
        REQUIRE(slowed->attackCount == 1);
    }
}

// ---------------- 32n + item 30: data ----------------

TEST_CASE("Goblin Hut is 1180 hp and its Spear Goblins shoot air, every 1.6 s",
          "[item32][goblin_hut][regression]") {
    Match m;
    auto hut = m.spawnCard(GOBLIN_HUT, 3.0f, 10.0f, 0).front();
    REQUIRE(hut->hp == 1180);
    auto flyer = m.dummy(3.0f, 14.0f, 1, 1000000, true);
    auto goblin = m.stepUntil("Spear Goblins", 0, 60);
    REQUIRE(goblin);
    REQUIRE(goblin->targetsAir);
    REQUIRE(goblin->getAttackCooldown() == 16);
    REQUIRE(goblin->deployTicksRemaining == 5); // 0.5 s, not the default 1 s (August 2025)
    m.step(100);
    REQUIRE(flyer->hp < 1000000); // was untouched: the Hut's goblins could not target air
}

TEST_CASE("Every Barbarian is 716 hp, and Barbarian Barrel rolls for 232",
          "[item32][barbarians][regression]") {
    {
        Match cardMatch;
        for (const auto& b : cardMatch.spawnCard(BARBARIANS, 4.0f, 8.0f, 0)) REQUIRE(b->hp == 716);
    }
    Match m;
    auto target = m.dummy(8.5f, 8.0f, 1, 10000);
    m.spawnCard(BARBARIAN_BARREL, 8.5f, 6.0f, 0);
    auto barbarian = m.stepUntil("Barbarians", 0, 40);
    REQUIRE(barbarian);
    REQUIRE(barbarian->hp == 716);
    REQUIRE(target->hp <= 10000 - 232);
    REQUIRE(target->hp > 10000 - 2 * 232); // one roll hit (the Barbarian may have swung too)
}

TEST_CASE("Barbarian Barrel is cast on its own side only; The Log keeps the river",
          "[item32][barbarian_barrel][placement][regression]") {
    GameManager game(deck(), deck());
    const float ownHalfMax = game.getOwnHalfMaxY();
    const float riverMid = (game.getBoard().getRiverStart() + game.getBoard().getRiverEnd()) * 0.5f;
    auto legal = [&](int cardId, int team, float y) {
        const CardDefinition* d = CardRegistry::getInstance().getCard(cardId);
        return game.isValidPlacement(team, 8.5f, y, d->isSpell, d->placementRadius, d->deployAnywhere,
                                     d->rollRange > 0.0f, d->castOwnSideOnly);
    };
    for (int card : { BARBARIAN_BARREL, HERO_BARBARIAN_BARREL }) {
        INFO("card " << card);
        REQUIRE(legal(card, 0, ownHalfMax));
        REQUIRE_FALSE(legal(card, 0, riverMid));           // was legal
        REQUIRE_FALSE(legal(card, 1, riverMid));
        REQUIRE(legal(card, 1, 33.0f - ownHalfMax));       // the mirror of the team-0 limit
    }
    REQUIRE(legal(THE_LOG, 0, riverMid));
}
