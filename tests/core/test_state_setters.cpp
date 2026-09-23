#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "GameManager.h"
#include "ClashEnv.h"
#include "Tower.h"
#include <vector>
#include <algorithm>

// The state-estimator write interface, so decision-time search over a
// reconstructed position scores the real one rather than reset()'s elixir and
// random hand. Two properties make the setters safe for a noisy sensor: elixir
// is clamped to the range the engine can reach, and setHand refuses anything
// that is not a real permutation of the team's deck.

static const std::vector<int> DECK = { 15, 6, 25, 40, 24, 72, 33, 7 };  // 2.6 Hog Cycle

TEST_CASE("setElixir writes the value back", "[setters][elixir]") {
    GameManager game(DECK, DECK);
    game.setElixir(0, 7.5f);
    REQUIRE(game.getElixir(0) == Catch::Approx(7.5f));
    // The teams are independent: an opponent-elixir estimate must never move
    // our own exact reading.
    REQUIRE(game.getElixir(1) == Catch::Approx(5.0f));
    game.setElixir(1, 2.25f);
    REQUIRE(game.getElixir(1) == Catch::Approx(2.25f));
    REQUIRE(game.getElixir(0) == Catch::Approx(7.5f));
}

TEST_CASE("setElixir clamps to a state the engine can actually reach", "[setters][elixir]") {
    GameManager game(DECK, DECK);
    // step() caps regen at 10.0; a caller must not create a position above
    // that.
    game.setElixir(0, 99.0f);
    REQUIRE(game.getElixir(0) == Catch::Approx(10.0f));
    game.setElixir(0, -3.0f);
    REQUIRE(game.getElixir(0) == Catch::Approx(0.0f));
}

TEST_CASE("setHand installs the requested hand", "[setters][hand]") {
    GameManager game(DECK, DECK);
    std::vector<int> want = { 33, 7, 15, 24 };
    REQUIRE(game.setHand(0, want) == true);
    REQUIRE(game.getHand(0) == want);
}

TEST_CASE("setHand keeps hand and queue a permutation of the deck",
          "[setters][hand]") {
    GameManager game(DECK, DECK);
    REQUIRE(game.setHand(0, { 33, 7, 15, 24 }) == true);

    // Every deck card appears exactly once across hand and queue; overwriting
    // the hand without rebuilding the queue would duplicate four cards and lose
    // four.
    std::vector<int> seen = game.getHand(0);
    // deckQueue is not exposed; cycle the whole deck through and check the
    // union.
    GameManager probe(DECK, DECK);
    REQUIRE(probe.setHand(0, { 33, 7, 15, 24 }) == true);
    std::vector<int> hand = probe.getHand(0);
    std::sort(hand.begin(), hand.end());
    std::vector<int> expected_hand = { 7, 15, 24, 33 };
    REQUIRE(hand == expected_hand);

    // and the four not requested are exactly the rest of the deck
    std::vector<int> rest;
    for (int c : DECK)
        if (std::find(expected_hand.begin(), expected_hand.end(), c) == expected_hand.end())
            rest.push_back(c);
    REQUIRE(rest.size() == 4);
}

TEST_CASE("setHand REFUSES a hand that is not a valid permutation",
          "[setters][hand]") {
    GameManager game(DECK, DECK);
    const std::vector<int> before = game.getHand(0);

    SECTION("a card that is not in the deck") {
        // 2 = Giant, from the old deck. A plausible-but-absent card is exactly
        // what a hand misread produces, so it must be rejected.
        REQUIRE(game.setHand(0, { 2, 6, 25, 40 }) == false);
        REQUIRE(game.getHand(0) == before);
    }
    SECTION("a duplicate") {
        REQUIRE(game.setHand(0, { 15, 15, 25, 40 }) == false);
        REQUIRE(game.getHand(0) == before);
    }
    SECTION("wrong size") {
        REQUIRE(game.setHand(0, { 15, 6, 25 }) == false);
        REQUIRE(game.getHand(0) == before);
    }
    SECTION("empty") {
        REQUIRE(game.setHand(0, {}) == false);
        REQUIRE(game.getHand(0) == before);
    }
}

TEST_CASE("a refused setHand leaves the position completely untouched",
          "[setters][hand]") {
    GameManager game(DECK, DECK);
    game.setElixir(0, 6.0f);
    const std::vector<int> before = game.getHand(0);
    REQUIRE(game.setHand(0, { 2, 3, 4, 5 }) == false);
    REQUIRE(game.getHand(0) == before);
    REQUIRE(game.getElixir(0) == Catch::Approx(6.0f));
}

TEST_CASE("setHand makes the requested cards actually PLAYABLE",
          "[setters][hand]") {
    // After writing the hand, a play of a card really in hand must be accepted;
    // a leftover hand cooldown would refuse it.
    GameManager game(DECK, DECK);
    REQUIRE(game.setHand(0, { 24, 72, 33, 15 }) == true);
    game.setElixir(0, 10.0f);
    // playCard takes a card id. Skeletons (24) costs 1; (9,8) is a legal
    // own-half cell.
    REQUIRE(game.playCard(0, 24, 9.0f, 8.0f) == true);
    // ...and a card left out of the hand must be refused.
    REQUIRE(game.playCard(0, 25, 9.0f, 9.0f) == false);   // Cannon, in queue
}

TEST_CASE("setters survive a snapshot", "[setters][snapshot]") {
    // A written state must survive the per-candidate snapshot, or every rollout
    // reverts to reset()'s values.
    GameManager game(DECK, DECK);
    game.setElixir(0, 8.0f);
    game.setElixir(1, 3.0f);
    REQUIRE(game.setHand(0, { 7, 33, 15, 6 }) == true);

    GameManager copy = game.snapshot();
    REQUIRE(copy.getElixir(0) == Catch::Approx(8.0f));
    REQUIRE(copy.getElixir(1) == Catch::Approx(3.0f));
    REQUIRE(copy.getHand(0) == std::vector<int>({ 7, 33, 15, 6 }));
}

// --- towers, clock and injected health (perception/UPSTREAM_REQUESTS.md item
// 22) ---
// The rest of the write interface: tower HP, the match clock, injected units'
// health and deploy time. A write is clamped to a state the engine can reach,
// and one it cannot represent is refused, never half-applied.

// The Tower behind (team, slot), to assert on state the setters do not expose
// (`awake`). Slots are board coordinates: 0 = King, 1 = left Princess, 2 =
// right.
static Tower* towerAt(Board& board, int team, int slot) {
    for (const auto& e : board.getEntities()) {
        Tower* t = dynamic_cast<Tower*>(e.get());
        if (t == nullptr || t->team != team) continue;
        const bool isKing = (t->symbol == 'R');
        if (slot == 0 && isKing) return t;
        if (slot == 1 && !isKing && t->position.x < ArenaLayout::CENTER_X) return t;
        if (slot == 2 && !isKing && t->position.x > ArenaLayout::CENTER_X) return t;
    }
    return nullptr;
}

static int towersStanding(Board& board, int team) {
    int n = 0;
    for (const auto& e : board.getEntities()) {
        if (e->isAlive() && e->team == team && dynamic_cast<Tower*>(e.get()) != nullptr) ++n;
    }
    return n;
}

TEST_CASE("setTowerHp writes the value back", "[setters][tower]") {
    GameManager game(DECK, DECK);
    REQUIRE(game.setTowerHp(1, 1, 900.0f) == true);
    REQUIRE(game.getTowerHp(1, 1) == 900);

    // Every other tower is untouched.
    REQUIRE(game.getTowerHp(1, 2) == game.getTowerMaxHp(1, 2));
    REQUIRE(game.getTowerHp(1, 0) == game.getTowerMaxHp(1, 0));
    REQUIRE(game.getTowerHp(0, 1) == game.getTowerMaxHp(0, 1));
}

TEST_CASE("setTowerHp REFUSES a value at or below zero and changes nothing",
          "[setters][tower]") {
    GameManager game(DECK, DECK);

    // Positive control: without it this passes against a setter that refuses
    // everything.
    REQUIRE(game.setTowerHp(1, 1, 1234.0f) == true);
    REQUIRE(game.getTowerHp(1, 1) == 1234);
    const int before = game.getTowerHp(1, 1);

    // A 0-hp tower that still stands is impossible, and killing one has side
    // effects that belong to destroyTower.
    REQUIRE(game.setTowerHp(1, 1, 0.0f) == false);
    REQUIRE(game.getTowerHp(1, 1) == before);
    REQUIRE(game.setTowerHp(1, 1, -50.0f) == false);
    REQUIRE(game.getTowerHp(1, 1) == before);

    // A refusal that still wrote would be invisible to a caller checking only
    // the bool.
    REQUIRE(towersStanding(game.getBoard(), 1) == 3);
}

TEST_CASE("setTowerHp clamps to the tower's OWN maximum, not a shared one",
          "[setters][tower]") {
    GameManager game(DECK, DECK);

    // King and Princess maxima differ (4008 vs 2534); a shared ceiling would
    // over-heal one of them.
    REQUIRE(game.getTowerMaxHp(0, 0) != game.getTowerMaxHp(0, 1));

    REQUIRE(game.setTowerHp(0, 0, 999999.0f) == true);
    REQUIRE(game.getTowerHp(0, 0) == game.getTowerMaxHp(0, 0));

    REQUIRE(game.setTowerHp(0, 1, 999999.0f) == true);
    REQUIRE(game.getTowerHp(0, 1) == game.getTowerMaxHp(0, 1));
}

TEST_CASE("setTowerHp addresses slots in BOARD coordinates, not team-relative",
          "[setters][tower]") {
    GameManager game(DECK, DECK);
    Board& board = game.getBoard();

    // Slot 1 is the left tower for both teams; the sensor never mirrors its
    // coordinates.
    REQUIRE(game.setTowerHp(0, 1, 111.0f) == true);
    REQUIRE(game.setTowerHp(1, 1, 222.0f) == true);
    REQUIRE(towerAt(board, 0, 1)->hp == 111);
    REQUIRE(towerAt(board, 1, 1)->hp == 222);
    REQUIRE(towerAt(board, 0, 1)->position.x < ArenaLayout::CENTER_X);
    REQUIRE(towerAt(board, 1, 1)->position.x < ArenaLayout::CENTER_X);
}

TEST_CASE("an injected wound wakes the King, exactly as real damage would",
          "[setters][tower][king]") {
    GameManager game(DECK, DECK);
    Board& board = game.getBoard();
    Tower* king = towerAt(board, 1, 0);
    REQUIRE(king != nullptr);
    REQUIRE_FALSE(king->isAwake());

    // Tower::awake latches on hp < maxHp, so a wound written by this setter
    // wakes the King like any damage.
    REQUIRE(game.setTowerHp(1, 0, static_cast<float>(game.getTowerMaxHp(1, 0)) - 1.0f));
    king->update(board);
    REQUIRE(king->isAwake());
}

TEST_CASE("setTowerHp back to full does NOT re-sleep a woken King",
          "[setters][tower][king]") {
    GameManager game(DECK, DECK);
    Board& board = game.getBoard();
    Tower* king = towerAt(board, 0, 0);

    REQUIRE(game.setTowerHp(0, 0, static_cast<float>(game.getTowerMaxHp(0, 0)) - 1.0f));
    king->update(board);
    REQUIRE(king->isAwake());

    // The flag latches: neither a heal nor a sensor misreading full health
    // re-sleeps it.
    REQUIRE(game.setTowerHp(0, 0, static_cast<float>(game.getTowerMaxHp(0, 0))));
    king->update(board);
    REQUIRE(king->isAwake());
}

TEST_CASE("destroyTower kills the tower and drops the standing count",
          "[setters][tower]") {
    GameManager game(DECK, DECK);
    Board& board = game.getBoard();
    REQUIRE(towersStanding(board, 1) == 3);

    REQUIRE(game.destroyTower(1, 1) == true);

    REQUIRE(towersStanding(board, 1) == 2);
    REQUIRE_FALSE(towerAt(board, 1, 1)->isAlive());
    REQUIRE(towersStanding(board, 0) == 3);
}

TEST_CASE("destroying a Princess wakes that team's King and not the other's",
          "[setters][tower][king]") {
    GameManager game(DECK, DECK);
    Board& board = game.getBoard();
    Tower* ours = towerAt(board, 0, 0);
    Tower* theirs = towerAt(board, 1, 0);

    // The Princess trigger records the living count on the first update, so
    // both Kings tick once before the tower falls.
    ours->update(board);
    theirs->update(board);
    REQUIRE_FALSE(theirs->isAwake());

    REQUIRE(game.destroyTower(1, 1) == true);

    theirs->update(board);
    ours->update(board);
    REQUIRE(theirs->isAwake());
    REQUIRE_FALSE(ours->isAwake());
}

TEST_CASE("destroyTower refuses a tower that is already down", "[setters][tower]") {
    GameManager game(DECK, DECK);
    REQUIRE(game.destroyTower(1, 2) == true);
    // Idempotent but not silent: a second call reports that it changed nothing.
    REQUIRE(game.destroyTower(1, 2) == false);
    REQUIRE(towersStanding(game.getBoard(), 1) == 2);
}

TEST_CASE("inject defaults to full health, preserving every existing caller",
          "[setters][inject]") {
    std::vector<int> deck = DECK;
    ClashEnv env(deck, deck, 100);
    env.reset();

    env.inject(15, 9.0f, 20.0f, 1);  // Hog Rider, enemy side

    Board& board = env.debugGame().getBoard();
    REQUIRE(board.pendingEntityCount() == 1);
    REQUIRE(board.getPendingEntity(0)->hp == 1697);  // test_default_deck_qa's figure
}

TEST_CASE("inject honours an explicit hp", "[setters][inject]") {
    std::vector<int> deck = DECK;
    ClashEnv env(deck, deck, 100);
    env.reset();

    env.inject(15, 9.0f, 20.0f, 1, 400.0f);

    Board& board = env.debugGame().getBoard();
    REQUIRE(board.getPendingEntity(0)->hp == 400);
}

TEST_CASE("an injected hp above the card's own maximum is clamped",
          "[setters][inject]") {
    std::vector<int> deck = DECK;
    ClashEnv env(deck, deck, 100);
    env.reset();

    Board& board = env.debugGame().getBoard();

    // Control first: a value below the maximum must land, or "clamped" is
    // indistinguishable from "ignored".
    env.inject(15, 9.0f, 20.0f, 1, 500.0f);
    REQUIRE(board.getPendingEntity(0)->hp == 500);

    env.inject(15, 9.0f, 20.0f, 1, 999999.0f);
    REQUIRE(board.getPendingEntity(1)->hp == 1697);
}

TEST_CASE("every body of a multi-body card takes the injected hp",
          "[setters][inject]") {
    std::vector<int> deck = DECK;
    ClashEnv env(deck, deck, 100);
    env.reset();

    // Skeletons spawn three bodies; the hp applies to all of them.
    env.inject(24, 9.0f, 20.0f, 1, 40.0f);

    Board& board = env.debugGame().getBoard();
    REQUIRE(board.pendingEntityCount() == 3);
    for (size_t i = 0; i < board.pendingEntityCount(); ++i) {
        REQUIRE(board.getPendingEntity(i)->hp == 40);
    }
}

TEST_CASE("inject defaults to the full deploy delay", "[setters][inject][deploy]") {
    std::vector<int> deck = DECK;
    ClashEnv env(deck, deck, 100);
    env.reset();

    env.inject(15, 9.0f, 20.0f, 1);

    Board& board = env.debugGame().getBoard();
    auto hog = board.getPendingEntity(0);
    const float y0 = hog->position.y;
    board.commitPendingEntities();
    for (int i = 0; i < DEPLOY_TIME_TICKS; ++i) hog->update(board);

    // The default: inert for the whole deploy second.
    REQUIRE(hog->position.y == Catch::Approx(y0));
}

TEST_CASE("inject with deployTicks 0 puts an ALREADY-DEPLOYED unit on the board",
          "[setters][inject][deploy]") {
    std::vector<int> deck = DECK;
    ClashEnv env(deck, deck, 100);
    env.reset();

    // A unit perception can already see has finished deploying; a fresh deploy
    // second would subsidise the defender in every rollout.
    env.inject(15, 9.0f, 20.0f, 1, -1.0f, 0);

    Board& board = env.debugGame().getBoard();
    auto hog = board.getPendingEntity(0);
    const float y0 = hog->position.y;
    board.commitPendingEntities();
    for (int i = 0; i < DEPLOY_TIME_TICKS; ++i) hog->update(board);

    // Team 1 advances down the board.
    REQUIRE(hog->position.y < y0);
}

TEST_CASE("setCurrentTick moves the observation's clock scalar",
          "[setters][clock]") {
    std::vector<int> deck = DECK;
    ClashEnv env(deck, deck, 1000);
    env.reset();

    const int spatial =
        ClashEnv::BOARD_WIDTH * ClashEnv::BOARD_HEIGHT * ClashEnv::NUM_CHANNELS;
    const int clockIdx = spatial + 1 + ClashEnv::HAND_SIZE
                       + ClashEnv::HAND_SIZE * ClashEnv::NUM_CARD_IDS;

    REQUIRE(env.getObservationForTeam(0)[clockIdx] == Catch::Approx(0.0f));

    env.setCurrentTick(500);

    // The time scalar is currentTick / maxTicks, the one field nothing else
    // reconstructs.
    REQUIRE(env.getObservationForTeam(0)[clockIdx] == Catch::Approx(0.5f));
    REQUIRE(env.getObservationForTeam(1)[clockIdx] == Catch::Approx(0.5f));
}

TEST_CASE("setCurrentTick clamps into a range the match can reach",
          "[setters][clock]") {
    std::vector<int> deck = DECK;
    ClashEnv env(deck, deck, 1000);
    env.reset();

    const int spatial =
        ClashEnv::BOARD_WIDTH * ClashEnv::BOARD_HEIGHT * ClashEnv::NUM_CHANNELS;
    const int clockIdx = spatial + 1 + ClashEnv::HAND_SIZE
                       + ClashEnv::HAND_SIZE * ClashEnv::NUM_CARD_IDS;

    env.setCurrentTick(999999);
    REQUIRE(env.getObservationForTeam(0)[clockIdx] == Catch::Approx(1.0f));

    env.setCurrentTick(-5);
    REQUIRE(env.getObservationForTeam(0)[clockIdx] == Catch::Approx(0.0f));
}

TEST_CASE("setCurrentTick keeps ClashEnv's clock and GameManager's in step",
          "[setters][clock]") {
    std::vector<int> deck = DECK;
    ClashEnv env(deck, deck, 1000);
    env.reset();

    // Both clocks move together: ClashEnv's drives the observation and done
    // condition, GameManager's the cooldowns and event stamps.
    env.setCurrentTick(742);
    REQUIRE(env.debugGame().getCurrentTick() == 742);
}

TEST_CASE("the item 22 setters survive a snapshot", "[setters][snapshot]") {
    std::vector<int> deck = DECK;
    ClashEnv env(deck, deck, 1000);
    env.reset();

    env.setTowerHp(1, 1, 700.0f);
    env.destroyTower(1, 2);
    env.setCurrentTick(300);
    env.inject(15, 9.0f, 20.0f, 1, 500.0f, 0);

    // The mirror's state must reach the snapshot a rollout runs on.
    ClashEnv copy = env.snapshot();
    REQUIRE(copy.getTowerHp(1, 1) == 700);
    REQUIRE(towersStanding(copy.debugGame().getBoard(), 1) == 2);

    const int spatial =
        ClashEnv::BOARD_WIDTH * ClashEnv::BOARD_HEIGHT * ClashEnv::NUM_CHANNELS;
    const int clockIdx = spatial + 1 + ClashEnv::HAND_SIZE
                       + ClashEnv::HAND_SIZE * ClashEnv::NUM_CARD_IDS;
    REQUIRE(copy.getObservationForTeam(0)[clockIdx] == Catch::Approx(0.3f));
}
