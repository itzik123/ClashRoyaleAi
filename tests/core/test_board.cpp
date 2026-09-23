#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "Board.h"
#include "Building.h"
#include "MeleeTroop.h"

// --- clampToBoard ---
// Used after a troop moves and again after collision resolution.

TEST_CASE("clampToBoard clamps out-of-bounds positions to the board edges", "[board][clamp]") {
    Board board;
    REQUIRE(board.clampToBoard(Vector2D{ -5.0f, 10.0f }, false).x == Catch::Approx(0.0f));
    REQUIRE(board.clampToBoard(Vector2D{ 25.0f, 10.0f }, false).x == Catch::Approx(17.0f));
    REQUIRE(board.clampToBoard(Vector2D{ 10.0f, -5.0f }, false).y == Catch::Approx(0.0f));
    REQUIRE(board.clampToBoard(Vector2D{ 10.0f, 40.0f }, false).y == Catch::Approx(33.0f));
}

TEST_CASE("clampToBoard pushes non-bridge river-band positions to the nearest bank", "[board][clamp]") {
    Board board;
    REQUIRE(board.clampToBoard(Vector2D{ 10.0f, 16.0f }, false).y == Catch::Approx(15.5f));
    REQUIRE(board.clampToBoard(Vector2D{ 10.0f, 17.0f }, false).y == Catch::Approx(17.5f));
}

TEST_CASE("clampToBoard does not snap positions sitting on a bridge column", "[board][clamp]") {
    Board board;
    // Read off the board: a literal becomes an assertion about open water when
    // a bridge moves.
    REQUIRE(board.clampToBoard(Vector2D{ board.getLeftBridge().x, 17.0f }, false).y == Catch::Approx(17.0f));
    REQUIRE(board.clampToBoard(Vector2D{ board.getRightBridge().x, 17.0f }, false).y == Catch::Approx(17.0f));
}

TEST_CASE("clampToBoard skips the river snap entirely when ignoresRiver is true", "[board][clamp]") {
    Board board;
    Vector2D result = board.clampToBoard(Vector2D{ 10.0f, 17.0f }, true); // off-bridge, mid-river
    REQUIRE(result.y == Catch::Approx(17.0f)); // left untouched
}

TEST_CASE("Board::allocateId returns increasing, unique ids", "[board][id]") {
    Board board;
    int a = board.allocateId();
    int b = board.allocateId();
    int c = board.allocateId();
    REQUIRE(b == a + 1);
    REQUIRE(c == b + 1);
}

TEST_CASE("Board::addEntity is invisible until commitPendingEntities", "[board][lifecycle]") {
    Board board;
    auto e = std::make_shared<DummyEntity>(1, 0.0f, 0.0f, 100, 0);
    board.addEntity(e);
    REQUIRE(board.getEntities().empty());

    board.commitPendingEntities();
    REQUIRE(board.getEntities().size() == 1);
}

TEST_CASE("Board::commitPendingEntities with nothing pending is a no-op", "[board][lifecycle]") {
    Board board;
    board.commitPendingEntities();
    REQUIRE(board.getEntities().empty());
}

TEST_CASE("Board::cleanDeadEntities removes only the dead", "[board][lifecycle]") {
    Board board;
    auto alive = std::make_shared<DummyEntity>(1, 0.0f, 0.0f, 100, 0);
    auto dead = std::make_shared<DummyEntity>(2, 0.0f, 0.0f, 100, 0);
    dead->takeDamage(100);
    spawn(board, alive);
    spawn(board, dead);

    board.cleanDeadEntities();

    REQUIRE(board.getEntities().size() == 1);
    REQUIRE(board.getEntities()[0]->id == 1);
}

TEST_CASE("Board::cleanDeadEntities fires a dying entity's death effect before removing it", "[board][lifecycle][death]") {
    Board board;
    auto dying = std::make_shared<StationaryCombatant>(1, 3.0f, 4.0f, 100, 1, 5.0f, 10, 10);
    auto effect = std::make_shared<RecordingDeathEffect>();
    dying->deathEffect = effect;
    dying->takeDamage(100);
    spawn(board, dying);

    board.cleanDeadEntities();

    REQUIRE(effect->applied);
    REQUIRE(effect->lastPosition.x == Catch::Approx(3.0f));
    REQUIRE(effect->lastTeam == 1);
    REQUIRE(board.getEntities().empty()); // still removed as normal
}

TEST_CASE("Board::cleanDeadEntities does not fire a death effect for entities that are still alive", "[board][lifecycle][death]") {
    Board board;
    auto alive = std::make_shared<StationaryCombatant>(1, 3.0f, 4.0f, 100, 1, 5.0f, 10, 10);
    auto effect = std::make_shared<RecordingDeathEffect>();
    alive->deathEffect = effect;
    spawn(board, alive);

    board.cleanDeadEntities();

    REQUIRE_FALSE(effect->applied);
    REQUIRE(board.getEntities().size() == 1);
}

// ---------------- resolvePositionAgainstBuildings ----------------

TEST_CASE("resolvePositionAgainstBuildings leaves a position untouched when far from any building", "[board][collision]") {
    Board board;
    auto building = std::make_shared<Building>(1, 10.0f, 10.0f, 1000, 1, 'C', 5.0f, 10, 10);
    spawn(board, building);

    Vector2D resolved = board.resolvePositionAgainstBuildings(Vector2D{ 0.0f, 0.0f }, 999);
    REQUIRE(resolved.x == Catch::Approx(0.0f));
    REQUIRE(resolved.y == Catch::Approx(0.0f));
}

TEST_CASE("resolvePositionAgainstBuildings pushes a position out of a building's radius", "[board][collision]") {
    Board board;
    // radius 1.0, minDist 1.4. Input 0.5 above the centre: push 1.4 - 0.5 =
    // 0.9, plus the 0.05 perpendicular slide.
    auto building = std::make_shared<Building>(1, 10.0f, 10.0f, 1000, 1, 'C', 5.0f, 10, 10);
    spawn(board, building);

    Vector2D resolved = board.resolvePositionAgainstBuildings(Vector2D{ 10.0f, 10.5f }, 999);
    REQUIRE(resolved.x == Catch::Approx(10.05f));
    REQUIRE(resolved.y == Catch::Approx(11.4f));
}

TEST_CASE("resolvePositionAgainstBuildings ignores dead buildings", "[board][collision]") {
    Board board;
    auto building = std::make_shared<Building>(1, 10.0f, 10.0f, 100, 1, 'C', 5.0f, 10, 10);
    building->takeDamage(100); // dead
    spawn(board, building);

    Vector2D resolved = board.resolvePositionAgainstBuildings(Vector2D{ 10.0f, 10.5f }, 999);
    REQUIRE(resolved.x == Catch::Approx(10.0f));
    REQUIRE(resolved.y == Catch::Approx(10.5f));
}

TEST_CASE("resolvePositionAgainstBuildings excludes the entity's own id", "[board][collision]") {
    Board board;
    auto building = std::make_shared<Building>(42, 10.0f, 10.0f, 1000, 1, 'C', 5.0f, 10, 10);
    spawn(board, building);

    // Resolving against itself must not push against its own radius.
    Vector2D resolved = board.resolvePositionAgainstBuildings(Vector2D{ 10.0f, 10.5f }, 42);
    REQUIRE(resolved.x == Catch::Approx(10.0f));
    REQUIRE(resolved.y == Catch::Approx(10.5f));
}

TEST_CASE("resolvePositionAgainstBuildings ignores non-building entities (radius 0)", "[board][collision]") {
    Board board;
    auto troop = std::make_shared<DummyEntity>(1, 10.0f, 10.0f, 100, 1); // radius 0.0 by default
    spawn(board, troop);

    Vector2D resolved = board.resolvePositionAgainstBuildings(Vector2D{ 10.0f, 10.1f }, 999);
    REQUIRE(resolved.x == Catch::Approx(10.0f));
    REQUIRE(resolved.y == Catch::Approx(10.1f));
}

// --- resolveCollisions ---

TEST_CASE("resolveCollisions pushes two overlapping troops apart symmetrically", "[board][collision]") {
    Board board;
    auto troop1 = std::make_shared<DummyEntity>(1, 10.0f, 10.0f, 100, 0);
    auto troop2 = std::make_shared<DummyEntity>(2, 10.0f, 10.3f, 100, 1); // dist 0.3, inside minRadius 0.8
    spawn(board, troop1);
    spawn(board, troop2);

    board.resolveCollisions();

    float dist = troop1->position.distanceTo(troop2->position);
    REQUIRE(dist > 0.79f);
}

TEST_CASE("resolveCollisions pushes a troop out of an overlapping building without moving the building", "[board][collision]") {
    Board board;
    auto building = std::make_shared<Building>(1, 10.0f, 10.0f, 1000, 1, 'C', 5.0f, 10, 10); // radius 1.0
    auto troop = std::make_shared<DummyEntity>(2, 10.0f, 10.3f, 100, 0); // dist 0.3, inside minDist 1.4
    spawn(board, building);
    spawn(board, troop);

    board.resolveCollisions();

    REQUIRE(building->position.x == Catch::Approx(10.0f));
    REQUIRE(building->position.y == Catch::Approx(10.0f));
    float dist = troop->position.distanceTo(building->position);
    REQUIRE(dist > 1.39f);
}

TEST_CASE("resolveCollisions leaves entities untouched when far apart", "[board][collision]") {
    Board board;
    auto troop1 = std::make_shared<DummyEntity>(1, 0.0f, 0.0f, 100, 0);
    auto troop2 = std::make_shared<DummyEntity>(2, 20.0f, 20.0f, 100, 1);
    spawn(board, troop1);
    spawn(board, troop2);

    board.resolveCollisions();

    REQUIRE(troop1->position.x == Catch::Approx(0.0f));
    REQUIRE(troop1->position.y == Catch::Approx(0.0f));
    REQUIRE(troop2->position.x == Catch::Approx(20.0f));
    REQUIRE(troop2->position.y == Catch::Approx(20.0f));
}

TEST_CASE("resolveCollisions ignores dead entities without crashing", "[board][collision]") {
    Board board;
    auto troop1 = std::make_shared<DummyEntity>(1, 10.0f, 10.0f, 100, 0);
    auto troop2 = std::make_shared<DummyEntity>(2, 10.0f, 10.1f, 100, 1);
    troop2->takeDamage(100); // dead
    spawn(board, troop1);
    spawn(board, troop2);

    REQUIRE_NOTHROW(board.resolveCollisions());
    REQUIRE(troop1->position.x == Catch::Approx(10.0f));
    REQUIRE(troop1->position.y == Catch::Approx(10.0f));
}

TEST_CASE("resolveCollisions re-clamps entities to the board bounds afterward", "[board][collision]") {
    Board board;
    // DummyEntity has no clampPosition; only a real Troop shows that
    // resolveCollisions re-clamps.
    auto troop = std::make_shared<MeleeTroop>(1, 25.0f, 10.0f, 100, 0, 1.0f, 1.0f, 10, 10, 'K');
    spawn(board, troop);

    board.resolveCollisions();

    REQUIRE(troop->position.x == Catch::Approx(17.0f));
}

// --- flying ---
// Fliers pass through everything; only entities sharing a plane collide.

TEST_CASE("resolveCollisions does not push a flying troop out of an overlapping building", "[board][collision][flying]") {
    Board board;
    auto building = std::make_shared<Building>(1, 10.0f, 10.0f, 1000, 1, 'C', 5.0f, 10, 10); // radius 1.0
    auto troop = std::make_shared<DummyEntity>(2, 10.0f, 10.3f, 100, 0); // dist 0.3, would be inside minDist 1.4
    troop->isFlying = true;
    spawn(board, building);
    spawn(board, troop);

    board.resolveCollisions();

    REQUIRE(troop->position.x == Catch::Approx(10.0f));
    REQUIRE(troop->position.y == Catch::Approx(10.3f));
}

TEST_CASE("resolveCollisions does not push a flying troop away from an overlapping grounded troop", "[board][collision][flying]") {
    Board board;
    auto flying = std::make_shared<DummyEntity>(1, 10.0f, 10.0f, 100, 0);
    flying->isFlying = true;
    auto grounded = std::make_shared<DummyEntity>(2, 10.0f, 10.3f, 100, 1); // dist 0.3, would be inside minRadius 0.8
    spawn(board, flying);
    spawn(board, grounded);

    board.resolveCollisions();

    REQUIRE(flying->position.x == Catch::Approx(10.0f));
    REQUIRE(flying->position.y == Catch::Approx(10.0f));
    REQUIRE(grounded->position.x == Catch::Approx(10.0f));
    REQUIRE(grounded->position.y == Catch::Approx(10.3f));
}

TEST_CASE("resolveCollisions still pushes two overlapping flying troops apart from each other", "[board][collision][flying]") {
    Board board;
    auto troop1 = std::make_shared<DummyEntity>(1, 10.0f, 10.0f, 100, 0);
    auto troop2 = std::make_shared<DummyEntity>(2, 10.0f, 10.3f, 100, 1); // dist 0.3, inside minRadius 0.8
    troop1->isFlying = true;
    troop2->isFlying = true;
    spawn(board, troop1);
    spawn(board, troop2);

    board.resolveCollisions();

    float dist = troop1->position.distanceTo(troop2->position);
    REQUIRE(dist > 0.79f);
}

// ---------------- getNextWaypoint ----------------

TEST_CASE("getNextWaypoint returns the target directly when both points are below the river", "[board][waypoint]") {
    Board board;
    Vector2D wp = board.getNextWaypoint(Vector2D{ 5.0f, 5.0f }, Vector2D{ 5.0f, 10.0f });
    REQUIRE(wp.x == Catch::Approx(5.0f));
    REQUIRE(wp.y == Catch::Approx(10.0f));
}

TEST_CASE("getNextWaypoint returns the target directly when both points are above the river", "[board][waypoint]") {
    Board board;
    Vector2D wp = board.getNextWaypoint(Vector2D{ 5.0f, 20.0f }, Vector2D{ 5.0f, 25.0f });
    REQUIRE(wp.x == Catch::Approx(5.0f));
    REQUIRE(wp.y == Catch::Approx(25.0f));
}

TEST_CASE("getNextWaypoint returns the target directly when both points are inside the river band", "[board][waypoint]") {
    Board board;
    // River band (15.5, 17.5): strictly inside, not at an edge.
    Vector2D wp = board.getNextWaypoint(Vector2D{ 4.0f, 16.0f }, Vector2D{ 14.0f, 17.0f });
    REQUIRE(wp.x == Catch::Approx(14.0f));
    REQUIRE(wp.y == Catch::Approx(17.0f));
}

TEST_CASE("getNextWaypoint routes below-to-above via the nearest bridge's start edge", "[board][waypoint]") {
    Board board;
    // Closer to the right bridge than the left.
    Vector2D wp = board.getNextWaypoint(Vector2D{ 10.0f, 5.0f }, Vector2D{ 10.0f, 25.0f });
    REQUIRE(wp.x == Catch::Approx(board.getRightBridge().x));
    REQUIRE(wp.y == Catch::Approx(15.5f)); // riverY_start
}

TEST_CASE("getNextWaypoint routes above-to-below via the nearest bridge's end edge", "[board][waypoint]") {
    Board board;
    Vector2D wp = board.getNextWaypoint(Vector2D{ 10.0f, 25.0f }, Vector2D{ 10.0f, 5.0f });
    REQUIRE(wp.x == Catch::Approx(board.getRightBridge().x));
    REQUIRE(wp.y == Catch::Approx(17.5f)); // riverY_end
}

TEST_CASE("getNextWaypoint picks the left bridge when it is nearer", "[board][waypoint]") {
    Board board;
    Vector2D wp = board.getNextWaypoint(Vector2D{ 2.0f, 5.0f }, Vector2D{ 2.0f, 25.0f });
    REQUIRE(wp.x == Catch::Approx(board.getLeftBridge().x));
}

// --- the bridge-mouth absorbing state ---
// getNextWaypoint's bank tests are inclusive (`y <= riverY_start`), so a unit
// standing exactly on the near bank is still "below" and would be handed the
// point it occupies, while Troop::moveTowards does not move within 0.01 of a
// waypoint: the unit never crosses. A 0.01-radius trap disc at each bridge
// mouth.

TEST_CASE("getNextWaypoint does not strand a unit standing on the near bank", "[board][waypoint][regression]") {
    Board board;
    // The measured position: on the left bridge, at riverY_start.
    Vector2D here{ board.getLeftBridge().x, 15.5f };
    Vector2D wp = board.getNextWaypoint(here, Vector2D{ board.getLeftBridge().x, 27.0f });
    // The unit still has to cross, so the waypoint must be somewhere it is not
    // standing.
    REQUIRE(here.distanceTo(wp) > Board::WAYPOINT_ARRIVAL_EPS);
    // The far bank.
    REQUIRE(wp.y == Catch::Approx(17.5f));
}

TEST_CASE("getNextWaypoint does not strand a unit standing on the far bank", "[board][waypoint][regression]") {
    Board board;
    // The same trap mirrored, heading south.
    Vector2D here{ board.getRightBridge().x, 17.5f };
    Vector2D wp = board.getNextWaypoint(here, Vector2D{ board.getRightBridge().x, 6.0f });
    REQUIRE(here.distanceTo(wp) > Board::WAYPOINT_ARRIVAL_EPS);
    REQUIRE(wp.y == Catch::Approx(15.5f));
}

TEST_CASE("getNextWaypoint never returns the caller's own position while it still has to cross",
          "[board][waypoint][regression]") {
    Board board;
    // Sweep the trap disc around both bridge mouths finer than the arrival
    // epsilon, both directions; any point returning itself is absorbing. Bridge
    // positions are read off the board, so the sweep follows the bridges if
    // they move.
    const float bridges[] = { board.getLeftBridge().x, board.getRightBridge().x };
    const float banks[] = { 15.5f, 17.5f };
    for (float bx : bridges) {
        for (float by : banks) {
            for (int dx = -2; dx <= 2; ++dx) {
                for (int dy = -2; dy <= 2; ++dy) {
                    Vector2D here{ bx + dx * 0.005f, by + dy * 0.005f };
                    // Target across the river from this bank.
                    Vector2D target = (by < 16.5f) ? Vector2D{ bx, 27.0f }
                                                   : Vector2D{ bx, 6.0f };
                    Vector2D wp = board.getNextWaypoint(here, target);
                    INFO("stuck at x=" << here.x << " y=" << here.y);
                    REQUIRE(here.distanceTo(wp) > Board::WAYPOINT_ARRIVAL_EPS);
                }
            }
        }
    }
}

// --- the bridge-exit absorbing state ---
// The same defect one branch over: inside the river band, within epsilon of the
// bank it is arriving at, the unit was handed that bank's edge and froze just
// short of dry land. Four exit traps mirroring the four entry traps. The entry
// sweep above pairs each bank only with the direction in which it is the entry,
// so this sweep is the full cross product: every bank, both directions.

TEST_CASE("getNextWaypoint does not strand a unit arriving at the far bank", "[board][waypoint][regression]") {
    Board board;
    // A hair short of the north bank, still crossing north: the measured
    // freeze.
    Vector2D here{ board.getLeftBridge().x, 17.4995f };
    Vector2D wp = board.getNextWaypoint(here, Vector2D{ board.getLeftBridge().x, 27.0f });
    REQUIRE(here.distanceTo(wp) > Board::WAYPOINT_ARRIVAL_EPS);
}

TEST_CASE("getNextWaypoint does not strand a unit arriving at the near bank", "[board][waypoint][regression]") {
    Board board;
    // Mirrored: a hair past the south bank, still heading south.
    Vector2D here{ board.getRightBridge().x, 15.5005f };
    Vector2D wp = board.getNextWaypoint(here, Vector2D{ board.getRightBridge().x, 6.0f });
    REQUIRE(here.distanceTo(wp) > Board::WAYPOINT_ARRIVAL_EPS);
}

TEST_CASE("getNextWaypoint never returns the caller's own position, in EITHER crossing direction",
          "[board][waypoint][regression]") {
    Board board;
    // Bridge positions are read off the board, so the sweep follows the bridges
    // if they move.
    const float bridges[] = { board.getLeftBridge().x, board.getRightBridge().x };
    const float banks[] = { 15.5f, 17.5f };
    // Both destinations at both banks.
    const float targets[] = { 27.0f, 6.0f };

    for (float bx : bridges) {
        for (float by : banks) {
            for (float ty : targets) {
                for (int dx = -2; dx <= 2; ++dx) {
                    for (int dy = -3; dy <= 3; ++dy) {
                        Vector2D here{ bx + dx * 0.005f, by + dy * 0.005f };
                        Vector2D wp = board.getNextWaypoint(here, Vector2D{ bx, ty });
                        INFO("stuck at x=" << here.x << " y=" << here.y << " target y=" << ty);
                        REQUIRE(here.distanceTo(wp) > Board::WAYPOINT_ARRIVAL_EPS);
                    }
                }
            }
        }
    }
}

TEST_CASE("getNextWaypoint from inside the river band heads to the exit edge toward the target's side", "[board][waypoint]") {
    Board board;

    SECTION("target is above -> heads to the river end edge") {
        Vector2D wp = board.getNextWaypoint(Vector2D{ board.getLeftBridge().x, 17.0f }, Vector2D{ board.getLeftBridge().x, 25.0f });
        REQUIRE(wp.y == Catch::Approx(17.5f));
    }

    SECTION("target is below -> heads to the river start edge") {
        Vector2D wp = board.getNextWaypoint(Vector2D{ board.getLeftBridge().x, 17.0f }, Vector2D{ board.getLeftBridge().x, 5.0f });
        REQUIRE(wp.y == Catch::Approx(15.5f));
    }
}

// --- isBackRowDeadZone ---
// One row behind each King, dead except a centred opening (Board.h,
// BACK_ROW_OPENING_HALF_WIDTH).

TEST_CASE("isBackRowDeadZone is false everywhere outside the two new back rows", "[board][deadzone]") {
    Board board;
    REQUIRE_FALSE(board.isBackRowDeadZone(0.0f, 5.0f));   // ordinary row, corner x
    REQUIRE_FALSE(board.isBackRowDeadZone(17.0f, 5.0f));  // ordinary row, other corner x
    REQUIRE_FALSE(board.isBackRowDeadZone(8.5f, 16.5f));  // mid-board
}

TEST_CASE("isBackRowDeadZone blocks the corners of the bottom back row but not its center gap", "[board][deadzone]") {
    Board board;
    REQUIRE(board.isBackRowDeadZone(0.0f, 0.0f));
    REQUIRE(board.isBackRowDeadZone(17.0f, 0.0f));
    REQUIRE_FALSE(board.isBackRowDeadZone(8.5f, 0.0f));   // board's own horizontal center
    REQUIRE_FALSE(board.isBackRowDeadZone(5.5f, 0.0f));   // left edge of the opening
    REQUIRE_FALSE(board.isBackRowDeadZone(11.5f, 0.0f));  // right edge of the opening
    REQUIRE(board.isBackRowDeadZone(5.4f, 0.0f));         // just outside the opening
    REQUIRE(board.isBackRowDeadZone(11.6f, 0.0f));        // just outside the opening
}

TEST_CASE("isBackRowDeadZone mirrors the same opening on the top back row", "[board][deadzone]") {
    Board board;
    float maxY = static_cast<float>(board.getHeight() - 1); // 33
    REQUIRE(board.isBackRowDeadZone(0.0f, maxY));
    REQUIRE_FALSE(board.isBackRowDeadZone(8.5f, maxY));
}


// --- collision: what does not participate ---

TEST_CASE("resolveCollisions ignores untargetable entities (spells, projectiles)",
          "[board][collision]") {
    // resolveCollisions treats "radius 0 and isTargetable()" as a troop.
    // AreaSpell and Projectile return false from isTargetable(), which is the
    // only thing stopping a Fireball's marker from shoving the troops it lands
    // on.
    Board board;
    auto troop = std::make_shared<DummyEntity>(1, 5.0f, 5.0f, 100, 0);
    auto ghost = std::make_shared<DummyEntity>(2, 5.0f, 5.0f, 100, 1);
    ghost->targetable = false;   // stands in for AreaSpell/Projectile
    spawn(board, troop);
    spawn(board, ghost);

    board.resolveCollisions();

    REQUIRE(troop->position.x == Catch::Approx(5.0f));
    REQUIRE(troop->position.y == Catch::Approx(5.0f));
    REQUIRE(ghost->position.x == Catch::Approx(5.0f));
    REQUIRE(ghost->position.y == Catch::Approx(5.0f));
}

TEST_CASE("resolvePositionAgainstBuildings tracks buildings appearing and disappearing",
          "[board][collision][lifecycle]") {
    // The collider set changes only through commitPendingEntities and
    // cleanDeadEntities; this walks a position through both, so any cache must
    // be invalidated by both.
    Board board;
    const Vector2D probe{ 5.0f, 5.0f };

    // 1. Empty board: untouched.
    REQUIRE(board.resolvePositionAgainstBuildings(probe, 99).x == Catch::Approx(5.0f));

    // 2. A building commits: pushed out.
    class Blocker : public Entity {
    public:
        Blocker(int id, float x, float y) : Entity(id, x, y, 100, 1, 'B') {}
        void update(Board&) override {}
        float getCollisionRadius() const override { return 1.0f; }
    };
    auto blocker = std::make_shared<Blocker>(7, 5.0f, 5.0f);
    board.addEntity(blocker);
    board.commitPendingEntities();
    const Vector2D pushed = board.resolvePositionAgainstBuildings(probe, 99);
    REQUIRE(pushed.distanceTo(blocker->position) > 1.0f);

    // 3. It dies and is erased: untouched again.
    blocker->takeDamage(100);
    board.cleanDeadEntities();
    const Vector2D afterDeath = board.resolvePositionAgainstBuildings(probe, 99);
    REQUIRE(afterDeath.x == Catch::Approx(5.0f));
    REQUIRE(afterDeath.y == Catch::Approx(5.0f));
}


TEST_CASE("pushAwayFrom clears the full radius even from dead centre",
          "[board][collision][regression]") {
    // A point exactly on the obstacle's centre is pushed the full minDist (see
    // Board::pushAwayFrom).
    const Vector2D centre{ 5.0f, 5.0f };

    SECTION("dead centre") {
        const Vector2D out = Board::pushAwayFrom(centre, centre, 1.4f);
        REQUIRE(out.distanceTo(centre) >= Catch::Approx(1.4f).margin(0.01f));
    }

    SECTION("a King Tower's larger footprint, same rule") {
        const Vector2D out = Board::pushAwayFrom(centre, centre, 2.4f);
        REQUIRE(out.distanceTo(centre) >= Catch::Approx(2.4f).margin(0.01f));
    }

    SECTION("the ordinary off-centre case is unchanged") {
        // 0.5 away, needs 1.4: lands on the boundary plus the perpendicular
        // slide.
        const Vector2D near{ 5.5f, 5.0f };
        const Vector2D out = Board::pushAwayFrom(near, centre, 1.4f);
        REQUIRE(out.x == Catch::Approx(6.4f));
        REQUIRE(out.y == Catch::Approx(4.95f));
    }
}
