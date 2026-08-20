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

// ============================================================================
// DEFAULT_DECK behavioural QA.
//
// python_ai/envs/gym_wrapper.py's DEFAULT_DECK is the 2.6 Hog Cycle and it is
// the deck every current checkpoint was trained on. `test_card_registry.cpp`
// already pins the deck's IDENTITY (which eight cards, and that none costs more
// than 4). This file pins what those eight cards DO -- attack, defence, pathing
// across the river, tower targeting, deploy time and air handling.
//
// Written 2026-08-20 from tools/audit/deck_audit.cpp, which measured each of
// these against the engine first. Every number here was observed, not assumed;
// where a figure is an engine artifact rather than a real-game fact, the
// comment says so.
//
// The gap this closes: the C++ suite covered movement, targeting and combat as
// MECHANISMS, in isolation, with hand-built entities. It had almost nothing
// asserting that a specific registry card, spawned through the real pipeline
// into a real match, behaves the way its role requires. That is exactly the
// blind spot CLAUDE.md already records for the movement-speed bug -- "the suite
// covers the movement MECHANISM and is blind to the DATA REGISTRY".
// ============================================================================

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

// A flying punching bag with no behaviour of its own, so an air test measures
// the ATTACKER and nothing else. The scenario probe in deck_audit.cpp could not
// do this -- real Minions fly off toward the towers and aggro onto whatever
// ground unit is nearby, which contaminated every reading it took.
class FlyingDummy : public Entity {
public:
    FlyingDummy(int id, float x, float y, int team)
        : Entity(id, x, y, 2000, team, 'F') { isFlying = true; }
    void update(Board&) override {}
    std::shared_ptr<Entity> snapshot() const override {
        return std::make_shared<FlyingDummy>(*this);
    }
};

// Does `cardId` damage a flying target parked next to it? Bare Board, no
// towers, nothing else alive: the only thing that can deal damage is the card.
int damageDealtToAir(int cardId) {
    Board board;
    auto flyer = std::make_shared<FlyingDummy>(board.allocateId(), 9.0f, 10.6f, 1);
    board.addEntity(flyer);
    CardRegistry::getInstance().getCard(cardId)->spawnEntity(9.0f, 10.0f, 0, board);
    board.commitPendingEntities();

    const int before = flyer->hp;
    for (int i = 0; i < 60; ++i) {
        // GameManager::step()'s order, including BOTH commits. Leaving the
        // commits out is not a shortcut: Board::addEntity queues into
        // pendingEntities, so a RangedTroop's projectile is created and then
        // never enters the world, and the card reads as dealing no damage at
        // all. That is what the first version of this helper did, and it
        // reported the Ice Spirit -- which certainly does shoot air -- as
        // unable to touch it.
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

// ---------------------------------------------------------------- identity --

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
    // The deck's only multi-body card. A silent drop to one body would look
    // like a balance change rather than a bug.
    Match m;
    m.play(SKELETONS, 9.0f, 10.0f, 0);
    REQUIRE(m.aliveOf(SKELETONS, 0) == 3);
}

// ------------------------------------------------------------ air handling --

TEST_CASE("air targeting matches each card's role", "[deck_qa][air]") {
    // Measured in isolation against a stationary flying dummy. CLAUDE.md
    // records a real regression of exactly this shape -- a CardRegistry audit
    // on 2026-07-29 found Musketeer among four cards that COULD NOT SHOOT AIR,
    // and Musketeer is in this deck.
    SECTION("Musketeer shoots air") { REQUIRE(damageDealtToAir(MUSKETEER) > 0); }
    SECTION("Ice Spirit shoots air") { REQUIRE(damageDealtToAir(ICE_SPIRIT) > 0); }
    SECTION("Fireball hits air") { REQUIRE(damageDealtToAir(FIREBALL) > 0); }

    SECTION("Hog Rider cannot touch air") { REQUIRE(damageDealtToAir(HOG_RIDER) == 0); }
    SECTION("Skeletons cannot touch air") { REQUIRE(damageDealtToAir(SKELETONS) == 0); }
    SECTION("Ice Golem cannot touch air") { REQUIRE(damageDealtToAir(ICE_GOLEM) == 0); }
    SECTION("Cannon cannot touch air") { REQUIRE(damageDealtToAir(CANNON) == 0); }
    SECTION("The Log cannot touch air") {
        // The one that is a rule rather than a stat: The Log rolls along the
        // ground and flying units are simply not hit by it.
        REQUIRE(damageDealtToAir(THE_LOG) == 0);
    }
}

// ---------------------------------------------------------------- crossing --

TEST_CASE("every ground troop in the deck crosses the river unaided", "[deck_qa][crossing]") {
    // The bridge-EXIT absorbing state (see test_board.cpp) stopped 6-8 of every
    // 34 lone ground units a few thousandths of a tile short of dry land, and
    // the slower the card the likelier it was -- Giant 27/34, Ice Golem 28/34.
    // This is the same check at the level a training run cares about: can this
    // card, from this deck, actually get to the other side.
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

// -------------------------------------------------------- tower targeting --

TEST_CASE("the Hog Rider ignores troops entirely and goes for the tower",
          "[deck_qa][targeting]") {
    // The defining property of a win condition, and the reason the defence
    // table below reads the way it does: Skeletons and a Musketeer stop a Hog
    // by KILLING it, never by distracting it.
    //
    // Asserted as "deals no damage to the Skeletons", not as "survives and
    // walks past them". The first version required survival over 120 ticks and
    // failed for a reason that has nothing to do with targeting: three
    // Skeletons are ~220 dps and a Princess Tower in range is another 382, so a
    // 1697 hp Hog is dead in under three seconds. Surviving was never the
    // property worth pinning.
    Match m;
    auto hog = m.play(HOG_RIDER, 4.0f, 16.0f, 0);
    REQUIRE(hog != nullptr);
    m.play(SKELETONS, 4.0f, 18.5f, 1);   // directly in its path

    int skeletonHpBefore = 0;
    for (const auto& e : m.game.getBoard().getEntities())
        if (e->cardId == SKELETONS && e->team == 1) skeletonHpBefore += e->hp;
    REQUIRE(skeletonHpBefore > 0);

    m.step(60);

    // Whatever happened to the Hog, it must never have swung at them. Damage
    // dealt is read by card, so our own towers' shooting cannot be credited to
    // the Hog by accident.
    REQUIRE(m.game.getStatistics().damageDealtByCard(HOG_RIDER, 0) == 0);
    REQUIRE(m.game.getStatistics().killsByCard(HOG_RIDER, 0) == 0);
}

TEST_CASE("a Musketeer cannot siege a Princess Tower for free", "[deck_qa][targeting]") {
    // The sight/attack measurement mismatch (test_sight_range.cpp) let her
    // destroy a 3204 hp tower from 8 tiles taking ZERO damage: 5355 damage
    // dealt, 0 received. Pinned here at deck level too, because this is the
    // form the bug actually took in a match.
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

// ------------------------------------------------------------- deploy time --

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
    // Skeletons are the one card that is NOT motionless during deploy: measured
    // 0.204 tiles while every other troop moved exactly 0.0000. That is
    // Board::resolveCollisions pushing three bodies off one another, not the
    // units acting -- deploy time forbids movement, targeting and attacking,
    // and physical separation is none of those. Pinned so the distinction stays
    // deliberate, and bounded so it can never grow into real movement.
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

// ----------------------------------------------------------------- defence --

TEST_CASE("the Cannon fully answers a lone Hog Rider", "[deck_qa][defence]") {
    // Measured: an unanswered Hog takes 2219 tower hp; with a Cannon down it
    // takes ZERO. This is the deck's whole defensive premise -- a 3-elixir
    // building neutralising a 4-elixir win condition -- and it is the single
    // most load-bearing interaction in the 2.6 matchup.
    //
    // That figure was 1268 until King dormancy landed on 2026-08-21
    // (tools/audit/king_activation_audit.cpp): with the defending King asleep
    // the Hog survives to tick 170 instead of 122 and deals 75% more. The
    // ASSERTIONS below are deliberately relative, so they held across that
    // change -- only this comment needed re-measuring.
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
    REQUIRE(withCannon == 0);
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

    // ...and the control that makes those two mean something: a Hog Rider is
    // NOT a defensive card, because it ignores troops entirely and walks the
    // other way. Measured 315 hp prevented against the Cannon's full answer --
    // and that 315 is our own tower shooting, not the Hog defending. (The
    // absolute figures moved when the King went dormant on 2026-08-21; the
    // ordering these assertions check did not.)
    REQUIRE(hpLostAnsweringHog(HOG_RIDER) > hpLostAnsweringHog(SKELETONS));
}

// ------------------------------------------------------------------ spells --

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
    // The control matters: Skeletons walk toward our side and can die to a
    // tower, so "some died" is not by itself evidence the spell landed. This is
    // the same attribution trap CLAUDE.md records for get_troop_damage_dealt.
    const int control = skeletonsKilledBy(-1);
    REQUIRE(skeletonsKilledBy(FIREBALL) > control);
    REQUIRE(skeletonsKilledBy(THE_LOG) > control);
}

TEST_CASE("a spell placed nowhere near anything kills nothing", "[deck_qa][spells]") {
    // CLAUDE.md: "Fireball injected out of range does nothing, and that is not
    // a bug." Pinned so a future splash-radius change cannot quietly turn a
    // spell into a board wipe.
    Match m;
    m.play(SKELETONS, 4.0f, 20.0f, 1);
    const int before = m.aliveOf(SKELETONS, 1);
    m.play(FIREBALL, 15.0f, 8.0f, 0);   // far corner, other side of the board
    m.step(30);
    REQUIRE(m.aliveOf(SKELETONS, 1) == before);
}
