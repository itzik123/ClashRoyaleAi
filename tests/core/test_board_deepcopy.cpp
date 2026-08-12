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

// Board::deepCopy() and the Entity::snapshot() mechanism under it -- the
// prerequisite for decision-time search (CLAUDE.md open problem #2): roll
// candidate actions forward on a copied board, score them, keep the best.
//
// The failure mode this file exists to catch is SILENT. A copy that shares
// entities with the live board produces wrong simulated futures, and those
// futures would then be distilled back into the policy as if they were expert
// labels. Nothing crashes and no metric moves. So the tests below are mostly
// not "does deepCopy work" but "does it fail the way it must when it's wrong"
// -- see the two negative cases at the bottom, which reproduce the actual
// cross-board corruption and then show deepCopy closing it.

namespace {

// The board-level half of GameManager::step(), in the same order (commit ->
// update -> commit -> resolveCollisions -> cleanDeadEntities). Deliberately
// mirrors that function rather than calling it: deepCopy is a Board operation,
// and a Board can be stepped without a GameManager wrapped around it. The
// elixir/PlayerState/MatchRules parts of step() are GameManager's own members,
// not this board's, and are out of scope for a Board-level copy.
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

// Everything about one entity that a rollout could observe. Compared field by
// field rather than via the observation encoder so a divergence points at the
// entity that drifted instead of at a 13606-long float vector.
struct EntityRow {
    int id;
    int hp;
    int team;
    int cardId;
    float x;
    float y;
    // -1 for everything that is not a Projectile. Included because a
    // projectile's target is the one piece of per-entity state that a naive
    // copy gets wrong, so leaving it out would hide exactly the bug this file
    // is about.
    int projectileTargetId;

    bool operator==(const EntityRow& other) const {
        return id == other.id && hp == other.hp && team == other.team &&
            cardId == other.cardId && x == other.x && y == other.y &&
            projectileTargetId == other.projectileTargetId;
    }
};

// Without this Catch2 prints a row of "{?}" on failure, which tells you a
// divergence happened but not where. Since the whole point of these tests is
// to localise a silent corruption, the diff has to name the entity that
// drifted and the field that did it.
std::ostream& operator<<(std::ostream& os, const EntityRow& row) {
    os << "{id=" << row.id << " hp=" << row.hp << " team=" << row.team
        << " card=" << row.cardId
        << " pos=(" << row.x << ", " << row.y << ")";
    if (row.projectileTargetId != -1) os << " target=" << row.projectileTargetId;
    return os << "}";
}

// NOT sorted by id: the vector order is itself load-bearing state. Board::
// resolveCollisions walks activeEntities as an ordered i<j double loop, so two
// boards holding the same entities in a different order resolve overlaps
// differently and drift apart within a few ticks. Comparing in order catches
// that; sorting first would hide it.
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

// Counts events instead of accumulating damage totals -- enough to prove the
// copied board's bus is disconnected, without depending on any real
// collector's schema.
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

// Four towers a side at their real coordinates (GameManager::reset's own
// layout), built directly rather than through GameManager so the fixture
// carries no RNG: GameManager's constructor seeds from std::random_device and
// shuffles both opening hands, which a divergence test should not have to
// reason about.
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

// Ticks the fixture advances before handing the board over. Tests resume well
// past this so their own tick numbers never overlap the warmup's.
constexpr int WARMUP_TICKS = 60;
constexpr int FIRST_TEST_TICK = 200;

// A genuine mid-game position: both sides' towers up, DEFAULT_DECK cards
// pushing on both lanes, stepped far enough that troops have crossed, towers
// have acquired targets and projectiles are in flight. Everything after this
// is measured against a board that is actually doing something -- a snapshot
// of an empty board would pass every test here while proving nothing.
//
// The warmup runs on past WARMUP_TICKS until a projectile actually exists
// rather than assuming one does at a fixed tick: whether a shot happens to be
// mid-air on tick 60 depends on attack cooldowns lining up, and a fixture that
// silently stopped containing projectiles would quietly retire the only tests
// that cover the aliasing bug.
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

// ---------------- Entity::snapshot() contract ----------------

TEST_CASE("snapshot preserves id and hp where clone deliberately does not", "[board][deepcopy][snapshot]") {
    MeleeTroop original(42, 3.0f, 4.0f, 777, 0, 0.5f, 1.2f, 120, 12, 'K');

    auto snap = original.snapshot();
    REQUIRE(snap->id == 42);
    REQUIRE(snap->hp == 777);
    REQUIRE(snap->position.x == Catch::Approx(3.0f));
    REQUIRE(snap->position.y == Catch::Approx(4.0f));

    // The reason snapshot() had to be added at all rather than reusing
    // clone(): clone implements the Clone CARD (fresh id, 1 hp), which would
    // silently hand a search a board full of 1-hp units.
    auto cloned = original.clone(99);
    REQUIRE(cloned->id == 99);
    REQUIRE(cloned->hp == 1);
}

TEST_CASE("snapshot on a concrete type that forgot to override it throws", "[board][deepcopy][snapshot]") {
    // DummyEntity (tests/entities/test_helpers.h) is a concrete Entity with no
    // snapshot() override -- exactly the shape of a future entity type someone
    // adds without knowing about deepCopy. The base must fail loudly and by
    // name, because the alternative default (nullptr, as clone() uses) would
    // let Board::deepCopy quietly drop it from every rollout.
    DummyEntity orphan(7, 1.0f, 1.0f, 100, 0);
    REQUIRE_THROWS_AS(orphan.snapshot(), std::logic_error);
}

TEST_CASE("snapshot does not slice derived types back to their bases", "[board][deepcopy][snapshot]") {
    // Tower -> Building and RangedBuildingTargeter -> BuildingTargeter are the
    // two places where inheriting the base's snapshot() would compile fine and
    // silently downgrade the entity: a sliced Tower stops answering isTower(),
    // which is what makes towers always-visible fallback targets in
    // CombatEntity::findTarget.
    Tower tower(1, 9.0f, 2.5f, 4008, 0, 7.0f, 90, 10, 'R');
    auto towerSnap = tower.snapshot();
    REQUIRE(std::dynamic_pointer_cast<Tower>(towerSnap) != nullptr);
    REQUIRE(towerSnap->isTower());
    REQUIRE(towerSnap->isBuilding());

    RangedBuildingTargeter ranged(2, 5.0f, 5.0f, 800, 0, 0.06f, 6.0f, 100, 16, 'H');
    auto rangedSnap = ranged.snapshot();
    REQUIRE(std::dynamic_pointer_cast<RangedBuildingTargeter>(rangedSnap) != nullptr);

    // Every remaining concrete type, so "all 8 are covered" is asserted rather
    // than assumed. Each must return its own type and must not throw.
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

// ---------------- Board::deepCopy() structure ----------------

TEST_CASE("deepCopy duplicates entities instead of sharing them", "[board][deepcopy]") {
    Board original;
    auto troop = std::make_shared<MeleeTroop>(1, 5.0f, 5.0f, 500, 0, 0.05f, 1.2f, 100, 12, 'K');
    spawn(original, troop);

    Board copy = original.deepCopy();
    REQUIRE(copy.getEntities().size() == 1);

    // Same values, different objects. The second REQUIRE is the whole point:
    // a plain `Board b = a;` passes the first and fails this one.
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
    // A copy that restarted its counter would hand the next projectile spawned
    // during a rollout an id an existing entity already holds, and every
    // id-keyed lookup would then join on the wrong entity.
    REQUIRE(copy.allocateId() == nextInOriginal + 1);
}

TEST_CASE("deepCopy carries pending entities, not just committed ones", "[board][deepcopy]") {
    Board original;
    auto committed = std::make_shared<MeleeTroop>(1, 5.0f, 5.0f, 500, 0, 0.05f, 1.2f, 100, 12, 'K');
    spawn(original, committed);
    // A card played this tick lives in pendingEntities until the next commit.
    // Dropping those would make a snapshot taken mid-tick lose the very
    // placement a search is trying to evaluate.
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

    // StatsEventBus holds shared_ptr to STATEFUL collectors (running damage
    // totals, kill attribution, match outcome), and those feed the reward
    // shaping. Copying the subscriber list would post every hypothetical hit
    // in every rollout into the real match's statistics -- a search silently
    // rewriting the returns it is being scored against. Same failure shape as
    // the Projectile alias below, one level up.
    REQUIRE(observer->damageEvents == 0);
    REQUIRE(observer->spawnEvents == 0);
    REQUIRE(observer->deathEvents == 0);

    // ...and the original's own bus is still connected, so this is a copy
    // property and not a broken fixture.
    original.statsEvents.notifyDamageDealt({ 0, 0, 0, 1, 0, 1, 10, 0 });
    REQUIRE(observer->damageEvents == 1);
}

// ---------------- The acceptance test: zero divergence ----------------

TEST_CASE("a deep-copied mid-game board and its original stay bit-identical for 120 ticks",
    "[board][deepcopy][divergence]") {
    Board original = buildMidGameBoard();

    // The fixture has to be a real fight or this test proves nothing.
    REQUIRE(original.getEntities().size() > 8);
    REQUIRE(boardHasProjectileInFlight(original));
    std::vector<EntityRow> atCopyTime = describe(original);

    Board copy = original.deepCopy();
    REQUIRE(describe(copy) == atCopyTime);

    // Stepped with identical inputs: the engine's only RNG is the opening-hand
    // shuffle and HeuristicOpponent, neither of which is in play here, so
    // combat is fully deterministic and "close enough" is not the bar --
    // every field must match exactly, every tick.
    for (int tick = FIRST_TEST_TICK; tick < FIRST_TEST_TICK + 120; ++tick) {
        tickBoard(original, tick);
        tickBoard(copy, tick);

        std::vector<EntityRow> originalRows = describe(original);
        std::vector<EntityRow> copyRows = describe(copy);
        INFO("divergence first seen at tick " << tick);
        REQUIRE(copyRows.size() == originalRows.size());
        REQUIRE(copyRows == originalRows);
    }

    // The board genuinely moved over those 120 ticks -- otherwise the loop
    // above compared two boards that were merely sitting still, which would
    // pass whether or not deepCopy did anything.
    REQUIRE(describe(original) != atCopyTime);
}

TEST_CASE("stepping a deep copy leaves the original board completely untouched",
    "[board][deepcopy][divergence]") {
    Board original = buildMidGameBoard();
    std::vector<EntityRow> before = describe(original);

    Board copy = original.deepCopy();
    for (int tick = FIRST_TEST_TICK; tick < FIRST_TEST_TICK + 120; ++tick) tickBoard(copy, tick);

    // The property search actually depends on: rolling a candidate forward
    // must not advance, damage or disturb the live game in any way.
    REQUIRE(describe(original) == before);
    // ...and the copy really did move on, so the assertion above is not
    // passing because nothing happened anywhere.
    REQUIRE(describe(copy) != before);
}

// ---------------- Negative cases: the corruption deepCopy prevents ----------------

TEST_CASE("a snapshot-only copy lets a projectile in flight damage the ORIGINAL board",
    "[board][deepcopy][projectile][negative]") {
    // The deliberate negative case. This reproduces what deepCopy would do if
    // it stopped after snapshot() and skipped the remap pass -- i.e. it proves
    // the remap is load-bearing rather than decorative, and it is written to
    // FAIL if someone deletes Projectile::remapSnapshotReferences and
    // "simplifies" deepCopy into a plain snapshot loop.
    Board original;
    auto victim = std::make_shared<MeleeTroop>(1, 5.0f, 12.0f, 5000, 1, 0.0f, 1.0f, 50, 10, 'M');
    spawn(original, victim);
    // Constructed directly, with no shooter on the board and a stationary
    // victim, so the only thing that can change any hp is this one shot.
    auto shot = std::make_shared<Projectile>(2, 5.0f, 9.0f, 0, victim, 1.5f, 250);
    spawn(original, shot);

    // A naive copy: every entity duplicated, nothing remapped.
    Board naive;
    for (const auto& e : original.getEntities()) naive.addEntity(e->snapshot());
    naive.commitPendingEntities();
    REQUIRE(naive.getEntities().size() == 2);

    auto naiveVictim = naive.getEntities()[0];
    REQUIRE(naiveVictim->id == 1);
    REQUIRE(naiveVictim.get() != victim.get()); // genuinely a different object

    // Step ONLY the naive copy. The original is never stepped.
    for (int tick = 1; tick <= 6; ++tick) tickBoard(naive, tick);

    // ...and yet the original's troop is the one that bled. The copy's
    // projectile carried a weak_ptr straight back into the live board, homed
    // on an entity belonging to another simulation, and dealt real damage to
    // it. Being a weak_ptr, nothing leaks and nothing crashes -- the symptom
    // is wrong numbers in a game nobody was stepping.
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

    // The target id is IDENTICAL either way -- snapshot() preserves ids on
    // purpose -- so the id alone can never tell a remapped projectile from an
    // aliased one. Only where the damage lands can.
    auto copyShot = std::dynamic_pointer_cast<Projectile>(copy.getEntities()[1]);
    REQUIRE(copyShot != nullptr);
    REQUIRE(copyShot->getTargetId() == 1);

    for (int tick = 1; tick <= 6; ++tick) tickBoard(copy, tick);

    REQUIRE(copyVictim->hp == 4750); // the copy's troop took the hit
    REQUIRE(victim->hp == 5000);     // the live board is untouched
}

TEST_CASE("deepCopy clears a projectile whose target is not in the copied board",
    "[board][deepcopy][projectile]") {
    // The edge the remap must not fall through: a target the map has no row
    // for. Leaving the weak_ptr as-is would be a live cross-board write, so it
    // is cleared instead -- costing one projectile that was about to die
    // anyway (update() sets hp = 0 the moment its target cannot be locked).
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
    // What search actually does: take one position and try several candidate
    // futures from it. Each branch must start from the same state and none may
    // contaminate another, which is a stronger requirement than a single copy
    // being correct.
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
            // Deterministic engine, identical inputs: every branch must land
            // on exactly the same result. A difference here means an earlier
            // branch left state behind somewhere.
            REQUIRE(describe(rollout) == firstBranchResult);
        }
        REQUIRE(describe(position) == start); // the shared position never moves
    }
}
