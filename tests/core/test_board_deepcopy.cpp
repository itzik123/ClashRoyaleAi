#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "Board.h"
#include "CardRegistry.h"
#include "StatsEventBus.h"
#include "AreaSpell.h"
#include "Building.h"
#include "BuildingTargeter.h"
#include "MeleeTroop.h"
#include "Projectile.h"
#include "RangedBuildingTargeter.h"
#include "RangedTroop.h"
#include "Tower.h"
#include <algorithm>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <vector>

// Board::deepCopy() and Entity::snapshot(), the basis of decision-time search:
// roll candidate actions forward on a copy, score them, keep the best.
//
// The failure mode is silent: a copy sharing entities with the live board
// produces wrong futures, which search would distill into the policy as expert
// labels. So most tests here check that it fails as it must when wrong; the
// negative cases at the end reproduce the cross-board corruption and show
// deepCopy closing it.

namespace {

// The board-level half of GameManager::step(), in the same order (commit ->
// update -> commit -> resolveCollisions -> cleanDeadEntities). A Board can be
// stepped without a GameManager.
void tickBoard(Board& board, int tick) {
    board.currentTick = tick;
    board.commitPendingEntities(tick);
    for (const auto& entity : board.getEntities()) {
        if (entity->isAlive()) entity->update(board);
    }
    board.pendingElixirGrant[0] = 0.0f;
    board.pendingElixirGrant[1] = 0.0f;
    board.commitPendingEntities(tick);
    board.resolveCollisions();
    board.cleanDeadEntities(tick);
}

// Everything a rollout could observe about one entity, compared field by field
// so a divergence names the entity.
struct EntityRow {
    int id;
    int hp;
    int team;
    int cardId;
    float x;
    float y;
    // -1 except for a Projectile, whose target is the one per-entity state a
    // naive copy gets wrong.
    int projectileTargetId;

    bool operator==(const EntityRow& other) const {
        return id == other.id && hp == other.hp && team == other.team &&
            cardId == other.cardId && x == other.x && y == other.y &&
            projectileTargetId == other.projectileTargetId;
    }
};

// Without this Catch2 prints "{?}"; the diff must name the entity and field
// that drifted.
std::ostream& operator<<(std::ostream& os, const EntityRow& row) {
    os << "{id=" << row.id << " hp=" << row.hp << " team=" << row.team
        << " card=" << row.cardId
        << " pos=(" << row.x << ", " << row.y << ")";
    if (row.projectileTargetId != -1) os << " target=" << row.projectileTargetId;
    return os << "}";
}

// Not sorted by id: the order is state. resolveCollisions walks activeEntities
// as an ordered i<j loop, so two boards with the same entities in a different
// order drift apart.
std::vector<EntityRow> describe(const Board& board) {
    std::vector<EntityRow> rows;
    rows.reserve(board.getEntities().size());
    for (const auto& e : board.getEntities()) {
        int targetId = -1;
        if (auto projectile = std::dynamic_pointer_cast<Projectile>(e)) {
            targetId = projectile->getTargetId();
        }
        rows.push_back({ e->id, e->hp, e->team, e->cardId, e->position.x, e->position.y, targetId });
    }
    return rows;
}

// Counts events, enough to prove the copy's bus is disconnected without
// depending on a collector's schema.
class CountingStatsObserver : public IStatsObserver {
public:
    int damageEvents = 0;
    int spawnEvents = 0;
    int deathEvents = 0;

    void onDamageDealt(const DamageDealtEvent&) override { damageEvents++; }
    void onEntitySpawned(const EntitySpawnedEvent&) override { spawnEvents++; }
    void onEntityDied(const EntityDiedEvent&) override { deathEvents++; }
};

void spawnCard(Board& board, int cardId, float x, float y, int team) {
    const CardDefinition* def = CardRegistry::getInstance().getCard(cardId);
    REQUIRE(def != nullptr);
    def->spawnEntity(x, y, team, board);
    board.commitPendingEntities();
}

// Towers at their real coordinates, built directly rather than through
// GameManager, so the fixture involves no RNG.
void addTowers(Board& board) {
    auto tower = [&](float x, float y, int hp, int team, float range, int damage, int cooldown, char symbol) {
        auto t = std::make_shared<Tower>(board.allocateId(), x, y, hp, team, range, damage, cooldown, symbol);
        t->cardId = (symbol == 'R') ? -2 : -3;
        board.addEntity(t);
    };
    tower(9.0f, 2.5f, 4008, 0, 7.0f, 90, 10, 'R');
    tower(9.0f, 30.5f, 4008, 1, 7.0f, 90, 10, 'R');
    tower(4.0f, 6.0f, 2534, 0, 7.5f, 109, 8, 'P');
    tower(14.0f, 6.0f, 2534, 0, 7.5f, 109, 8, 'P');
    tower(4.0f, 27.0f, 2534, 1, 7.5f, 109, 8, 'P');
    tower(14.0f, 27.0f, 2534, 1, 7.5f, 109, 8, 'P');
    board.commitPendingEntities();
}

bool boardHasProjectileInFlight(const Board& board) {
    for (const auto& e : board.getEntities()) {
        if (std::dynamic_pointer_cast<Projectile>(e)) return true;
    }
    return false;
}

// Ticks the fixture advances; tests resume past this.
constexpr int WARMUP_TICKS = 60;
constexpr int FIRST_TEST_TICK = 200;

// A genuine mid-game position: troops across the river, towers engaged,
// projectiles in flight. The warm-up runs until a projectile exists rather than
// assuming one at a fixed tick, so the aliasing tests cannot quietly lose their
// subject.
Board buildMidGameBoard() {
    Board board;
    addTowers(board);

    spawnCard(board, 2, 4.0f, 14.0f, 0);    // Giant, left lane
    spawnCard(board, 6, 5.0f, 11.0f, 0);    // Musketeer behind it
    spawnCard(board, 10, 14.0f, 13.0f, 0);  // Valkyrie, right lane
    spawnCard(board, 25, 9.0f, 8.0f, 0);    // Cannon, defensive

    spawnCard(board, 5, 4.0f, 19.0f, 1);    // Mini P.E.K.K.A, contesting left
    spawnCard(board, 1, 5.0f, 22.0f, 1);    // Archers behind it
    spawnCard(board, 41, 14.0f, 20.0f, 1);  // Minions, air, right lane
    spawnCard(board, 6, 13.0f, 23.0f, 1);   // Musketeer

    int tick = 1;
    for (; tick <= WARMUP_TICKS; ++tick) tickBoard(board, tick);
    for (; tick < FIRST_TEST_TICK && !boardHasProjectileInFlight(board); ++tick) {
        tickBoard(board, tick);
    }
    return board;
}

} // namespace

// --- Entity::snapshot() contract ---

TEST_CASE("snapshot preserves id and hp where clone deliberately does not", "[board][deepcopy][snapshot]") {
    MeleeTroop original(42, 3.0f, 4.0f, 777, 0, 0.5f, 1.2f, 120, 12, 'K');

    auto snap = original.snapshot();
    REQUIRE(snap->id == 42);
    REQUIRE(snap->hp == 777);
    REQUIRE(snap->position.x == Catch::Approx(3.0f));
    REQUIRE(snap->position.y == Catch::Approx(4.0f));

    // Why snapshot() exists beside clone(): clone implements the Clone card
    // (fresh id, 1 hp).
    auto cloned = original.clone(99);
    REQUIRE(cloned->id == 99);
    REQUIRE(cloned->hp == 1);
}

TEST_CASE("snapshot on a concrete type that forgot to override it throws", "[board][deepcopy][snapshot]") {
    // DummyEntity has no snapshot() override, the shape of a future entity type
    // added without knowing about deepCopy. The base must throw by name; a
    // nullptr default would drop it from every rollout.
    DummyEntity orphan(7, 1.0f, 1.0f, 100, 0);
    REQUIRE_THROWS_AS(orphan.snapshot(), std::logic_error);
}

TEST_CASE("snapshot does not slice derived types back to their bases", "[board][deepcopy][snapshot]") {
    // Tower -> Building and RangedBuildingTargeter -> BuildingTargeter:
    // inheriting the base snapshot() would compile and slice the entity; a
    // sliced Tower stops answering isTower().
    Tower tower(1, 9.0f, 2.5f, 4008, 0, 7.0f, 90, 10, 'R');
    auto towerSnap = tower.snapshot();
    REQUIRE(std::dynamic_pointer_cast<Tower>(towerSnap) != nullptr);
    REQUIRE(towerSnap->isTower());
    REQUIRE(towerSnap->isBuilding());

    RangedBuildingTargeter ranged(2, 5.0f, 5.0f, 800, 0, 0.06f, 6.0f, 100, 16, 'H');
    auto rangedSnap = ranged.snapshot();
    REQUIRE(std::dynamic_pointer_cast<RangedBuildingTargeter>(rangedSnap) != nullptr);

    // Every remaining concrete type returns its own type without throwing.
    RangedTroop rangedTroop(3, 5.0f, 5.0f, 340, 0, 0.05f, 6.0f, 100, 10, 'A');
    BuildingTargeter buildingTargeter(4, 5.0f, 5.0f, 800, 0, 0.06f, 1.2f, 100, 16, 'H');
    Building building(5, 9.0f, 8.0f, 824, 0, 'C', 5.5f, 83, 8, 300);
    AreaSpell spell(6, 9.0f, 20.0f, 0, 2.5f, 689, 10, '*');
    REQUIRE(std::dynamic_pointer_cast<RangedTroop>(rangedTroop.snapshot()) != nullptr);
    REQUIRE(std::dynamic_pointer_cast<BuildingTargeter>(buildingTargeter.snapshot()) != nullptr);
    REQUIRE(std::dynamic_pointer_cast<Building>(building.snapshot()) != nullptr);
    REQUIRE(std::dynamic_pointer_cast<AreaSpell>(spell.snapshot()) != nullptr);

    auto victim = std::make_shared<MeleeTroop>(8, 5.0f, 12.0f, 500, 1, 0.0f, 1.0f, 50, 10, 'M');
    Projectile projectile(7, 5.0f, 8.0f, 0, victim, 1.5f, 100);
    REQUIRE(std::dynamic_pointer_cast<Projectile>(projectile.snapshot()) != nullptr);
}

// --- Board::deepCopy() structure ---

TEST_CASE("deepCopy duplicates entities instead of sharing them", "[board][deepcopy]") {
    Board original;
    auto troop = std::make_shared<MeleeTroop>(1, 5.0f, 5.0f, 500, 0, 0.05f, 1.2f, 100, 12, 'K');
    spawn(original, troop);

    Board copy = original.deepCopy();
    REQUIRE(copy.getEntities().size() == 1);

    // Same values, different objects; a plain `Board b = a;` fails the second
    // REQUIRE.
    REQUIRE(copy.getEntities()[0]->id == troop->id);
    REQUIRE(copy.getEntities()[0]->hp == troop->hp);
    REQUIRE(copy.getEntities()[0].get() != troop.get());

    copy.getEntities()[0]->takeDamage(200);
    REQUIRE(copy.getEntities()[0]->hp == 300);
    REQUIRE(troop->hp == 500); // original untouched
}

TEST_CASE("deepCopy carries idCounter so the copy cannot reissue live ids", "[board][deepcopy]") {
    Board original;
    original.allocateId();
    original.allocateId();
    int nextInOriginal = original.allocateId();

    Board copy = original.deepCopy();
    // A restarted counter would reuse ids during a rollout, and id-keyed
    // lookups would join the wrong entity.
    REQUIRE(copy.allocateId() == nextInOriginal + 1);
}

TEST_CASE("deepCopy carries pending entities, not just committed ones", "[board][deepcopy]") {
    Board original;
    auto committed = std::make_shared<MeleeTroop>(1, 5.0f, 5.0f, 500, 0, 0.05f, 1.2f, 100, 12, 'K');
    spawn(original, committed);
    // A card played this tick lives in pendingEntities until the next commit; a
    // snapshot must keep it.
    auto pending = std::make_shared<MeleeTroop>(2, 6.0f, 6.0f, 400, 0, 0.05f, 1.2f, 100, 12, 'K');
    original.addEntity(pending);

    Board copy = original.deepCopy();
    REQUIRE(copy.getEntities().size() == 1);
    copy.commitPendingEntities();
    REQUIRE(copy.getEntities().size() == 2);
    REQUIRE(copy.getEntities()[1]->id == 2);
    REQUIRE(copy.getEntities()[1].get() != pending.get());
}

TEST_CASE("deepCopy does not carry the stats subscriber list", "[board][deepcopy][stats]") {
    Board original;
    auto observer = std::make_shared<CountingStatsObserver>();
    original.statsEvents.subscribe(observer);

    auto victim = std::make_shared<MeleeTroop>(1, 5.0f, 12.0f, 5000, 1, 0.0f, 1.0f, 50, 10, 'M');
    spawn(original, victim);
    auto shot = std::make_shared<Projectile>(2, 5.0f, 9.0f, 0, victim, 1.5f, 250);
    spawn(original, shot);
    observer->damageEvents = 0;
    observer->spawnEvents = 0;

    Board copy = original.deepCopy();
    for (int tick = 1; tick <= 6; ++tick) tickBoard(copy, tick);

    // The bus holds stateful collectors that feed the reward; a copied
    // subscriber list would post every rollout's hits into the live statistics.
    REQUIRE(observer->damageEvents == 0);
    REQUIRE(observer->spawnEvents == 0);
    REQUIRE(observer->deathEvents == 0);

    // ...and the original's bus is still connected: a copy property, not a
    // broken fixture.
    original.statsEvents.notifyDamageDealt({ 0, 0, 0, 1, 0, 1, 10, 0 });
    REQUIRE(observer->damageEvents == 1);
}

// --- the acceptance test: zero divergence ---

TEST_CASE("a deep-copied mid-game board and its original stay bit-identical for 120 ticks",
    "[board][deepcopy][divergence]") {
    Board original = buildMidGameBoard();

    // The fixture must be a real fight.
    REQUIRE(original.getEntities().size() > 8);
    REQUIRE(boardHasProjectileInFlight(original));
    std::vector<EntityRow> atCopyTime = describe(original);

    Board copy = original.deepCopy();
    REQUIRE(describe(copy) == atCopyTime);

    // Identical inputs and no RNG in play, so every field must match exactly,
    // every tick.
    for (int tick = FIRST_TEST_TICK; tick < FIRST_TEST_TICK + 120; ++tick) {
        tickBoard(original, tick);
        tickBoard(copy, tick);

        std::vector<EntityRow> originalRows = describe(original);
        std::vector<EntityRow> copyRows = describe(copy);
        INFO("divergence first seen at tick " << tick);
        REQUIRE(copyRows.size() == originalRows.size());
        REQUIRE(copyRows == originalRows);
    }

    // The board genuinely moved, or two idle boards would trivially match.
    REQUIRE(describe(original) != atCopyTime);
}

TEST_CASE("stepping a deep copy leaves the original board completely untouched",
    "[board][deepcopy][divergence]") {
    Board original = buildMidGameBoard();
    std::vector<EntityRow> before = describe(original);

    Board copy = original.deepCopy();
    for (int tick = FIRST_TEST_TICK; tick < FIRST_TEST_TICK + 120; ++tick) tickBoard(copy, tick);

    // Rolling a candidate forward must not disturb the live game.
    REQUIRE(describe(original) == before);
    // ...and the copy really moved on.
    REQUIRE(describe(copy) != before);
}

// --- negative cases: the corruption deepCopy prevents ---

TEST_CASE("a snapshot-only copy lets a projectile in flight damage the ORIGINAL board",
    "[board][deepcopy][projectile][negative]") {
    // Reproduces deepCopy without the remap pass, proving the remap is
    // load-bearing; written to fail if remapSnapshotReferences is removed.
    Board original;
    auto victim = std::make_shared<MeleeTroop>(1, 5.0f, 12.0f, 5000, 1, 0.0f, 1.0f, 50, 10, 'M');
    spawn(original, victim);
    // No shooter and a stationary victim, so only this one shot can change any
    // hp.
    auto shot = std::make_shared<Projectile>(2, 5.0f, 9.0f, 0, victim, 1.5f, 250);
    spawn(original, shot);

    // A naive copy: everything duplicated, nothing remapped.
    Board naive;
    for (const auto& e : original.getEntities()) naive.addEntity(e->snapshot());
    naive.commitPendingEntities();
    REQUIRE(naive.getEntities().size() == 2);

    auto naiveVictim = naive.getEntities()[0];
    REQUIRE(naiveVictim->id == 1);
    REQUIRE(naiveVictim.get() != victim.get()); // genuinely a different object

    // Step only the naive copy.
    for (int tick = 1; tick <= 6; ++tick) tickBoard(naive, tick);

    // ...and the original's troop is the one that bled: the copy's projectile
    // held a weak_ptr into the live board and damaged it. Nothing crashes; the
    // numbers are just wrong.
    REQUIRE(victim->hp == 4750);
    REQUIRE(naiveVictim->hp == 5000); // the copy's own troop was never hit
}

TEST_CASE("deepCopy remaps a projectile in flight onto the copy's own target",
    "[board][deepcopy][projectile]") {
    Board original;
    auto victim = std::make_shared<MeleeTroop>(1, 5.0f, 12.0f, 5000, 1, 0.0f, 1.0f, 50, 10, 'M');
    spawn(original, victim);
    auto shot = std::make_shared<Projectile>(2, 5.0f, 9.0f, 0, victim, 1.5f, 250);
    spawn(original, shot);

    Board copy = original.deepCopy();
    auto copyVictim = copy.getEntities()[0];
    REQUIRE(copyVictim.get() != victim.get());

    // The target id is identical either way (snapshot preserves ids); only
    // where the damage lands tells them apart.
    auto copyShot = std::dynamic_pointer_cast<Projectile>(copy.getEntities()[1]);
    REQUIRE(copyShot != nullptr);
    REQUIRE(copyShot->getTargetId() == 1);

    for (int tick = 1; tick <= 6; ++tick) tickBoard(copy, tick);

    REQUIRE(copyVictim->hp == 4750); // the copy's troop took the hit
    REQUIRE(victim->hp == 5000);     // the live board is untouched
}

TEST_CASE("deepCopy clears a projectile whose target is not in the copied board",
    "[board][deepcopy][projectile]") {
    // A target missing from the map: the pointer is cleared rather than left
    // crossing boards, costing a projectile that was about to die anyway.
    Board original;
    auto offBoardVictim = std::make_shared<MeleeTroop>(99, 5.0f, 12.0f, 5000, 1, 0.0f, 1.0f, 50, 10, 'M');
    auto shot = std::make_shared<Projectile>(2, 5.0f, 9.0f, 0, offBoardVictim, 1.5f, 250);
    spawn(original, shot);

    Board copy = original.deepCopy();
    auto copyShot = std::dynamic_pointer_cast<Projectile>(copy.getEntities()[0]);
    REQUIRE(copyShot != nullptr);
    REQUIRE(copyShot->getTargetId() == -1); // cleared, not left aliasing

    tickBoard(copy, 1);
    REQUIRE(offBoardVictim->hp == 5000); // never reached across
    REQUIRE(copy.getEntities().empty()); // and the shot expired harmlessly
}

TEST_CASE("a deep copy taken mid-fight can be stepped repeatedly from the same position",
    "[board][deepcopy][divergence]") {
    // Search tries several futures from one position: each branch starts from
    // the same state and none contaminates another.
    Board position = buildMidGameBoard();
    std::vector<EntityRow> start = describe(position);

    std::vector<EntityRow> firstBranchResult;
    for (int branch = 0; branch < 3; ++branch) {
        Board rollout = position.deepCopy();
        REQUIRE(describe(rollout) == start);

        for (int tick = FIRST_TEST_TICK; tick < FIRST_TEST_TICK + 60; ++tick) tickBoard(rollout, tick);

        if (branch == 0) {
            firstBranchResult = describe(rollout);
        } else {
            // Deterministic engine, identical inputs: every branch lands on the
            // same result.
            REQUIRE(describe(rollout) == firstBranchResult);
        }
        REQUIRE(describe(position) == start); // the shared position never moves
    }
}
