#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "Board.h"
#include "GameManager.h"
#include "CardRegistry.h"
#include "CombatEntity.h"
#include "Troop.h"
#include "Building.h"
#include "Tower.h"
#include <vector>
#include <memory>

// DEFAULT_DECK behavioural QA (the 2.6 Hog Cycle). test_card_registry.cpp pins
// the deck's identity; this pins what the eight cards do: attack, defence,
// crossing the river, tower targeting, deploy time, air handling. Each card is
// spawned through the real pipeline into a real match, which the mechanism
// tests with hand-built entities do not cover. Figures were measured with
// tools/audit/deck_audit.cpp; where one is an engine artefact, the comment says
// so.

namespace {

constexpr int HOG_RIDER = 15;
constexpr int MUSKETEER = 6;
constexpr int CANNON    = 25;
constexpr int ICE_GOLEM = 40;
constexpr int SKELETONS = 24;
constexpr int ICE_SPIRIT = 72;
constexpr int THE_LOG   = 33;
constexpr int FIREBALL  = 7;

const std::vector<int>& deck() {
    static const std::vector<int> d = { HOG_RIDER, MUSKETEER, CANNON, ICE_GOLEM,
                                        SKELETONS, ICE_SPIRIT, THE_LOG, FIREBALL };
    return d;
}

struct Match {
    GameManager game;
    Match() : game(deck(), deck()) { game.reset(); }

    std::shared_ptr<Entity> play(int cardId, float x, float y, int team) {
        CardRegistry::getInstance().getCard(cardId)->spawnEntity(x, y, team, game.getBoard());
        game.getBoard().commitPendingEntities();
        std::shared_ptr<Entity> found;
        for (const auto& e : game.getBoard().getEntities())
            if (e->cardId == cardId && e->team == team && e->isAlive()) found = e;
        return found;
    }
    void step(int n) { for (int i = 0; i < n; ++i) game.step(); }

    int towerHp(int team) const {
        int t = 0;
        for (const auto& e : game.getBoard().getEntities())
            if (e->isAlive() && e->isTower() && e->team == team) t += e->hp;
        return t;
    }
    int aliveOf(int cardId, int team) const {
        int n = 0;
        for (const auto& e : game.getBoard().getEntities())
            if (e->isAlive() && e->cardId == cardId && e->team == team) n++;
        return n;
    }
};

// A flying punching bag with no behaviour, so an air test measures only the
// attacker.
class FlyingDummy : public Entity {
public:
    FlyingDummy(int id, float x, float y, int team)
        : Entity(id, x, y, 2000, team, 'F') { isFlying = true; }
    void update(Board&) override {}
    std::shared_ptr<Entity> snapshot() const override {
        return std::make_shared<FlyingDummy>(*this);
    }
};

// Does `cardId` damage a flying target parked next to it? Bare Board, nothing
// else that can deal damage.
int damageDealtToAir(int cardId) {
    Board board;
    auto flyer = std::make_shared<FlyingDummy>(board.allocateId(), 9.0f, 10.6f, 1);
    board.addEntity(flyer);
    CardRegistry::getInstance().getCard(cardId)->spawnEntity(9.0f, 10.0f, 0, board);
    board.commitPendingEntities();

    const int before = flyer->hp;
    for (int i = 0; i < 60; ++i) {
        // GameManager::step()'s order, including both commits: a RangedTroop's
        // projectile is queued into pendingEntities and never enters the world
        // without them.
        board.currentTick = i + 1;
        board.commitPendingEntities(board.currentTick);
        for (const auto& e : board.getEntities()) if (e->isAlive()) e->update(board);
        board.commitPendingEntities(board.currentTick);
        board.resolveCollisions();
        board.cleanDeadEntities();
    }
    return before - flyer->hp;
}

} // namespace

// --- identity ---

TEST_CASE("every DEFAULT_DECK card spawns something through the real pipeline",
          "[deck_qa]") {
    for (int id : deck()) {
        Match m;
        auto e = m.play(id, 9.0f, 10.0f, 0);
        INFO("card id " << id << " (" << CardRegistry::getInstance().getCard(id)->name << ")");
        REQUIRE(CardRegistry::getInstance().getCard(id) != nullptr);
        REQUIRE(e != nullptr);
    }
}

TEST_CASE("Skeletons spawn exactly three bodies", "[deck_qa]") {
    // The deck's only multi-body card.
    Match m;
    m.play(SKELETONS, 9.0f, 10.0f, 0);
    REQUIRE(m.aliveOf(SKELETONS, 0) == 3);
}

// --- air handling ---

TEST_CASE("air targeting matches each card's role", "[deck_qa][air]") {
    // Measured against a stationary flying dummy. Musketeer once could not
    // shoot air.
    SECTION("Musketeer shoots air") { REQUIRE(damageDealtToAir(MUSKETEER) > 0); }
    SECTION("Ice Spirit shoots air") { REQUIRE(damageDealtToAir(ICE_SPIRIT) > 0); }
    SECTION("Fireball hits air") { REQUIRE(damageDealtToAir(FIREBALL) > 0); }

    SECTION("Hog Rider cannot touch air") { REQUIRE(damageDealtToAir(HOG_RIDER) == 0); }
    SECTION("Skeletons cannot touch air") { REQUIRE(damageDealtToAir(SKELETONS) == 0); }
    SECTION("Ice Golem cannot touch air") { REQUIRE(damageDealtToAir(ICE_GOLEM) == 0); }
    SECTION("Cannon cannot touch air") { REQUIRE(damageDealtToAir(CANNON) == 0); }
    SECTION("The Log cannot touch air") {
        // The Log rolls along the ground: flying units are not hit.
        REQUIRE(damageDealtToAir(THE_LOG) == 0);
    }
}

// --- crossing ---

TEST_CASE("every ground troop in the deck crosses the river unaided", "[deck_qa][crossing]") {
    // Can each card in this deck actually reach the other side? (See the
    // bridge-exit trap in test_board.cpp.)
    for (int id : { HOG_RIDER, MUSKETEER, ICE_GOLEM, SKELETONS, ICE_SPIRIT }) {
        Match m;
        auto unit = m.play(id, 4.0f, 12.0f, 0);
        REQUIRE(unit != nullptr);

        bool crossed = false;
        for (int i = 0; i < 900 && !crossed; ++i) {
            m.game.step();
            for (const auto& e : m.game.getBoard().getEntities()) {
                if (e->cardId == id && e->team == 0 && e->isAlive() && e->position.y >= 17.5f) {
                    crossed = true;
                    break;
                }
            }
        }
        INFO("card " << CardRegistry::getInstance().getCard(id)->name << " never reached y >= 17.5");
        REQUIRE(crossed);
    }
}

TEST_CASE("a Cannon is a building and never crosses anything", "[deck_qa][crossing]") {
    Match m;
    auto cannon = m.play(CANNON, 9.0f, 10.0f, 0);
    REQUIRE(cannon != nullptr);
    REQUIRE(cannon->isBuilding());
    const Vector2D where = cannon->position;
    m.step(100);
    REQUIRE(where.distanceTo(cannon->position) == Catch::Approx(0.0f).margin(1e-4));
}

// --- tower targeting ---

TEST_CASE("the Hog Rider ignores troops entirely and goes for the tower",
          "[deck_qa][targeting]") {
    // A win condition stops for nothing: Skeletons and a Musketeer stop a Hog
    // only by killing it. Asserted as "deals no damage to the Skeletons", not
    // survival: with a tower in range the Hog dies within three seconds anyway.
    Match m;
    auto hog = m.play(HOG_RIDER, 4.0f, 16.0f, 0);
    REQUIRE(hog != nullptr);
    m.play(SKELETONS, 4.0f, 18.5f, 1);   // directly in its path

    int skeletonHpBefore = 0;
    for (const auto& e : m.game.getBoard().getEntities())
        if (e->cardId == SKELETONS && e->team == 1) skeletonHpBefore += e->hp;
    REQUIRE(skeletonHpBefore > 0);

    m.step(60);

    // Whatever happens to the Hog, it never swings at them. (Its total damage
    // is not the measure: hitting the enemy tower is what it is for.)
    int skeletonHpAfter = 0;
    for (const auto& e : m.game.getBoard().getEntities())
        if (e->cardId == SKELETONS && e->team == 1) skeletonHpAfter += e->hp;
    REQUIRE(skeletonHpAfter == skeletonHpBefore);
    REQUIRE(m.game.getStatistics().killsByCard(HOG_RIDER, 0) == 0);
}

TEST_CASE("a Musketeer cannot siege a Princess Tower for free", "[deck_qa][targeting]") {
    // The sight/attack mismatch let her destroy a tower from 8 tiles taking
    // zero damage; pinned at deck level, the form it took in a match.
    Match m;
    auto musket = m.play(MUSKETEER, 4.0f, 19.0f, 0);
    REQUIRE(musket != nullptr);
    const int startHp = musket->hp;
    m.step(300);

    int survivorHp = 0;
    for (const auto& e : m.game.getBoard().getEntities())
        if (e->id == musket->id && e->isAlive()) survivorHp = e->hp;
    REQUIRE(survivorHp < startHp);
}

// --- deploy time ---

TEST_CASE("every deployable card in the deck is inert for its deploy time",
          "[deck_qa][deploy]") {
    for (int id : { HOG_RIDER, MUSKETEER, CANNON, ICE_GOLEM, ICE_SPIRIT }) {
        Match m;
        auto e = m.play(id, 9.0f, 10.0f, 0);
        REQUIRE(e != nullptr);
        auto combat = std::dynamic_pointer_cast<CombatEntity>(e);
        REQUIRE(combat != nullptr);

        INFO("card " << CardRegistry::getInstance().getCard(id)->name);
        REQUIRE(combat->deployTicksRemaining == DEPLOY_TIME_TICKS);

        const Vector2D placed = e->position;
        m.step(DEPLOY_TIME_TICKS);
        // Not one tile of movement while deploying.
        REQUIRE(placed.distanceTo(e->position) == Catch::Approx(0.0f).margin(1e-4));
    }
}

TEST_CASE("Skeletons are separated by collision while deploying, not moved by choice",
          "[deck_qa][deploy]") {
    // Skeletons move slightly during deploy (~0.2 tiles): resolveCollisions
    // separating three bodies, not the units acting. Bounded so it can never
    // become real movement.
    Match m;
    m.play(SKELETONS, 9.0f, 10.0f, 0);
    std::vector<Vector2D> placed;
    for (const auto& e : m.game.getBoard().getEntities())
        if (e->cardId == SKELETONS && e->team == 0) placed.push_back(e->position);
    REQUIRE(placed.size() == 3);

    m.step(DEPLOY_TIME_TICKS);

    size_t i = 0;
    for (const auto& e : m.game.getBoard().getEntities()) {
        if (e->cardId != SKELETONS || e->team != 0) continue;
        float moved = placed[i++].distanceTo(e->position);
        INFO("skeleton " << i << " moved " << moved << " during deploy");
        REQUIRE(moved < 0.5f);   // separation, not a walk
    }
}

// --- defence ---

TEST_CASE("the Cannon fully answers a lone Hog Rider", "[deck_qa][defence]") {
    // An unanswered Hog takes 2534 tower hp; with a Cannon down it lands one
    // swing (317). The deck's whole defensive premise, a 3-elixir building
    // neutralising a 4-elixir win condition. The assertions are relative, so
    // they survive engine changes that move the absolute figures.
    int unanswered = 0;
    {
        Match m;
        const int before = m.towerHp(0);
        m.play(HOG_RIDER, 4.0f, 18.0f, 1);
        m.step(400);
        unanswered = before - m.towerHp(0);
    }
    REQUIRE(unanswered > 0);

    Match m;
    const int before = m.towerHp(0);
    m.play(HOG_RIDER, 4.0f, 18.0f, 1);
    m.step(20);
    m.play(CANNON, 4.0f, 11.0f, 0);
    m.step(380);
    const int withCannon = before - m.towerHp(0);

    INFO("unanswered " << unanswered << " hp lost, with a Cannon " << withCannon);
    REQUIRE(withCannon < unanswered);
    // The Cannon absorbs the overwhelming majority of the push: the Hog, Very
    // Fast in the real game, reaches the tower for one hit before it dies.
    REQUIRE(withCannon * 5 < unanswered);
}

TEST_CASE("Skeletons and a Musketeer both blunt a Hog, a Hog does not",
          "[deck_qa][defence]") {
    auto hpLostAnsweringHog = [](int answerCard) {
        Match m;
        const int before = m.towerHp(0);
        m.play(HOG_RIDER, 4.0f, 18.0f, 1);
        m.step(20);
        if (answerCard >= 0) m.play(answerCard, 4.0f, 11.0f, 0);
        m.step(380);
        return before - m.towerHp(0);
    };

    const int noAnswer = hpLostAnsweringHog(-1);
    REQUIRE(hpLostAnsweringHog(SKELETONS) < noAnswer);
    REQUIRE(hpLostAnsweringHog(MUSKETEER) < noAnswer);

    // The control: a Hog Rider is not a defensive card; it ignores troops. The
    // little it "prevents" is our own tower shooting.
    REQUIRE(hpLostAnsweringHog(HOG_RIDER) > hpLostAnsweringHog(SKELETONS));
}

// --- spells ---

TEST_CASE("Fireball and The Log both clear a Skeleton clump", "[deck_qa][spells]") {
    auto skeletonsKilledBy = [](int spell) {
        Match m;
        m.play(SKELETONS, 9.0f, 20.0f, 1);
        m.play(SKELETONS, 9.2f, 20.2f, 1);
        m.play(SKELETONS, 8.8f, 19.8f, 1);
        const int before = m.aliveOf(SKELETONS, 1);
        if (spell >= 0) m.play(spell, 9.0f, 20.0f, 0);
        m.step(40);
        return before - m.aliveOf(SKELETONS, 1);
    };
    // Skeletons walk toward our side and can die to a tower, so "some died"
    // alone does not prove the spell landed.
    const int control = skeletonsKilledBy(-1);
    REQUIRE(skeletonsKilledBy(FIREBALL) > control);
    REQUIRE(skeletonsKilledBy(THE_LOG) > control);
}

TEST_CASE("a spell placed nowhere near anything kills nothing", "[deck_qa][spells]") {
    // A Fireball out of range does nothing; pinned so a radius change cannot
    // make a spell a board wipe.
    Match m;
    m.play(SKELETONS, 4.0f, 20.0f, 1);
    const int before = m.aliveOf(SKELETONS, 1);
    m.play(FIREBALL, 15.0f, 8.0f, 0);   // far corner, other side of the board
    m.step(30);
    REQUIRE(m.aliveOf(SKELETONS, 1) == before);
}

// --- status effects and lifetime ---
// The state each card leaves behind on something else: freeze, slow, expiry.

namespace {

// One GameManager-order tick on a bare Board, both commits included (see
// damageDealtToAir).
void tickBoard(Board& board, int tick) {
    board.currentTick = tick;
    board.commitPendingEntities(tick);
    for (const auto& e : board.getEntities()) if (e->isAlive()) e->update(board);
    board.commitPendingEntities(tick);
    board.resolveCollisions();
    board.cleanDeadEntities(tick);
}

std::shared_ptr<Entity> findByCard(Board& board, int cardId, int team) {
    for (const auto& e : board.getEntities())
        if (e->cardId == cardId && e->team == team) return e;
    return nullptr;
}

} // namespace

TEST_CASE("an Ice Spirit's freeze is a full stun, not a half-speed slow",
          "[deck_qa][status]") {
    // Ice Spirit stuns: its victim stops dead for a second (FreezeOnHit with
    // 0.0f, as Electro Spirit and Freeze use). Measured by what the victim
    // does, not by reading freezeSlow.
    Board board;
    // A destination: with no tower, findTarget returns nothing and "did not
    // move" would pass for the wrong reason.
    auto goal = std::make_shared<Tower>(board.allocateId(), 9.0f, 2.0f, 2534, 0, 7.5f, 109, 8, 'P');
    board.addEntity(goal);
    CardRegistry::getInstance().getCard(MUSKETEER)->spawnEntity(9.0f, 14.0f, 1, board);
    CardRegistry::getInstance().getCard(ICE_SPIRIT)->spawnEntity(9.0f, 13.0f, 0, board);
    board.commitPendingEntities();

    auto victimEntity = findByCard(board, MUSKETEER, 1);
    REQUIRE(victimEntity);
    auto victim = std::dynamic_pointer_cast<CombatEntity>(victimEntity);
    REQUIRE(victim);

    // Both spawn through the real pipeline, so both owe a deploy second.
    int tick = 0;
    while (victim->freezeTicks == 0 && tick < 60) tickBoard(board, ++tick);

    // Control 1: the freeze landed; every failure of the displacement check
    // otherwise reads as "did not move".
    REQUIRE(victim->freezeTicks > 0);
    REQUIRE(victim->isAlive());

    const Vector2D frozenAt = victim->position;
    const int freezeWindow = victim->freezeTicks;
    for (int i = 0; i < freezeWindow; ++i) tickBoard(board, ++tick);
    const float movedWhileFrozen = frozenAt.distanceTo(victim->position);

    // Control 2: the same unit, board and tick count after the stun expires
    // moves, so the zero above is the stun.
    REQUIRE(victim->freezeTicks == 0);
    const Vector2D thawedAt = victim->position;
    for (int i = 0; i < freezeWindow; ++i) tickBoard(board, ++tick);
    const float movedAfterThaw = thawedAt.distanceTo(victim->position);

    INFO("moved while frozen = " << movedWhileFrozen << ", after thaw = " << movedAfterThaw);
    REQUIRE(movedAfterThaw > 0.1f);
    REQUIRE(movedWhileFrozen == Catch::Approx(0.0f).margin(1e-4f));
}

TEST_CASE("an Ice Golem does not slow what it attacks", "[deck_qa][status]") {
    // Ice Golem's slow is on its death explosion; it has no on-attack slow. A
    // tower it tanks must not be slowed.
    Board board;
    auto tower = std::make_shared<Tower>(board.allocateId(), 3.0f, 27.0f, 2534, 1, 7.5f, 109, 8, 'P');
    board.addEntity(tower);
    CardRegistry::getInstance().getCard(ICE_GOLEM)->spawnEntity(3.0f, 25.2f, 0, board);
    board.commitPendingEntities();

    const int towerBefore = tower->hp;
    for (int t = 1; t <= 60; ++t) tickBoard(board, t);

    // Control: the Golem connected with the tower, its target.
    REQUIRE(tower->hp < towerBefore);
    REQUIRE(tower->freezeTicks == 0);
    REQUIRE(tower->freezeSlow == Catch::Approx(1.0f));
}

TEST_CASE("an Ice Golem's death explosion slows nearby enemies", "[deck_qa][status]") {
    // The other half: the death explosion slows.
    Board board;
    CardRegistry::getInstance().getCard(ICE_GOLEM)->spawnEntity(9.0f, 10.0f, 0, board);
    CardRegistry::getInstance().getCard(MUSKETEER)->spawnEntity(9.0f, 11.0f, 1, board);
    board.commitPendingEntities();

    auto golem = findByCard(board, ICE_GOLEM, 0);
    auto victimEntity = findByCard(board, MUSKETEER, 1);
    REQUIRE(golem);
    REQUIRE(victimEntity);
    auto victim = std::dynamic_pointer_cast<CombatEntity>(victimEntity);
    REQUIRE(victim);
    REQUIRE(victim->freezeTicks == 0); // nothing has touched it yet

    const int hpBefore = victim->hp;
    golem->hp = 0;
    board.cleanDeadEntities(1);

    // Control: the explosion's damage half works, so this measures the slow.
    REQUIRE(victim->hp < hpBefore);
    REQUIRE(victim->freezeTicks > 0);
    REQUIRE(victim->freezeSlow < 1.0f);
}

TEST_CASE("a Cannon expires at exactly its lifetime, not one second late",
          "[deck_qa][lifetime]") {
    // A building dies exactly at its lifetime, not whenever integer decay
    // happens to finish it.
    Board board;
    CardRegistry::getInstance().getCard(CANNON)->spawnEntity(9.0f, 8.0f, 0, board);
    board.commitPendingEntities();
    auto cannon = findByCard(board, CANNON, 0);
    REQUIRE(cannon);

    constexpr int LIFETIME_TICKS = 300; // Building's own default, 30s at 10 ticks/s
    for (int t = 1; t < LIFETIME_TICKS; ++t) tickBoard(board, t);
    REQUIRE(cannon->isAlive()); // still standing one tick short of expiry

    tickBoard(board, LIFETIME_TICKS);
    REQUIRE_FALSE(cannon->isAlive());
}
