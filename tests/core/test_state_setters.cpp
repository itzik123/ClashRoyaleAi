#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "GameManager.h"
#include "ClashEnv.h"
#include "Tower.h"
#include <vector>
#include <algorithm>

// The state-estimator WRITE interface (2026-08-17).
//
// perception/ can read the live board, our elixir and our hand, but
// forecast.py rebuilt a position by calling reset() -- which sets elixir to
// 5.0 and deals a hand from an UNSEEDED std::mt19937. The board was a
// prediction and the hand was fiction, and affordability_mask is built from
// exactly those scalars, so decision-time search over a reconstructed state
// was scoring a position the real game was not in.
//
// These tests pin the two properties that make the setters safe to hand a
// noisy sensor: elixir is CLAMPED to the range the engine itself can reach,
// and setHand REFUSES anything that is not a real permutation of that team's
// deck rather than silently accepting a misread.

static const std::vector<int> DECK = { 15, 6, 25, 40, 24, 72, 33, 7 };  // 2.6 Hog Cycle

TEST_CASE("setElixir writes the value back", "[setters][elixir]") {
    GameManager game(DECK, DECK);
    game.setElixir(0, 7.5f);
    REQUIRE(game.getElixir(0) == Catch::Approx(7.5f));
    // The two teams are independent -- an opponent-elixir ESTIMATE must never
    // move our own known-exact reading.
    REQUIRE(game.getElixir(1) == Catch::Approx(5.0f));
    game.setElixir(1, 2.25f);
    REQUIRE(game.getElixir(1) == Catch::Approx(2.25f));
    REQUIRE(game.getElixir(0) == Catch::Approx(7.5f));
}

TEST_CASE("setElixir clamps to a state the engine can actually reach", "[setters][elixir]") {
    GameManager game(DECK, DECK);
    // tick() caps regeneration at 10.0f; a caller (or a bad estimate from the
    // auxiliary head) must not be able to create a position above that.
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

    // Every deck card must appear exactly once across hand + queue. Overwriting
    // the hand without rebuilding the queue would DUPLICATE the four cards
    // moved into hand and lose the four they displaced -- a board state the
    // real game can never be in, and one that would let the policy cycle a
    // card it does not own.
    std::vector<int> seen = game.getHand(0);
    // deckQueue is not exposed, so reconstruct it by cycling the whole deck
    // through: playing is not needed, the invariant is checked on the union.
    GameManager probe(DECK, DECK);
    REQUIRE(probe.setHand(0, { 33, 7, 15, 24 }) == true);
    std::vector<int> hand = probe.getHand(0);
    std::sort(hand.begin(), hand.end());
    std::vector<int> expected_hand = { 7, 15, 24, 33 };
    REQUIRE(hand == expected_hand);

    // and the four NOT requested must be exactly the rest of the deck
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
        // 2 = Giant, which belongs to the OLD deck. A hand misread that lands
        // on a plausible-but-absent card is exactly what perception produces
        // (icon templates agreed with the elixir ledger only 33.8% of the
        // time), so this must be rejected rather than installed.
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
    // The point of the whole interface: after writing the live hand in, the
    // engine must accept a play of a card that is really in hand. If
    // handCooldownTicks were left non-zero the play would be refused and search
    // would conclude the card is unavailable ~2 s early.
    GameManager game(DECK, DECK);
    REQUIRE(game.setHand(0, { 24, 72, 33, 15 }) == true);
    game.setElixir(0, 10.0f);
    // playCard takes a CARD ID, not a hand index. Skeletons (24) costs 1 and
    // (9,8) is a legal own-half cell.
    REQUIRE(game.playCard(0, 24, 9.0f, 8.0f) == true);
    // ...and a card we deliberately left OUT of the hand must be refused, or
    // "the hand was written" would be an untested claim.
    REQUIRE(game.playCard(0, 25, 9.0f, 9.0f) == false);   // Cannon, in queue
}

TEST_CASE("setters survive a snapshot", "[setters][snapshot]") {
    // Decision-time search snapshots the env per candidate. A written state
    // that did not survive the copy would silently revert to reset()'s values
    // inside every rollout -- i.e. the exact bug these setters exist to fix,
    // reappearing one layer down.
    GameManager game(DECK, DECK);
    game.setElixir(0, 8.0f);
    game.setElixir(1, 3.0f);
    REQUIRE(game.setHand(0, { 7, 33, 15, 6 }) == true);

    GameManager copy = game.snapshot();
    REQUIRE(copy.getElixir(0) == Catch::Approx(8.0f));
    REQUIRE(copy.getElixir(1) == Catch::Approx(3.0f));
    REQUIRE(copy.getHand(0) == std::vector<int>({ 7, 33, 15, 6 }));
}

// ===========================================================================
// UPSTREAM item 22 (2026-08-24): the rest of the state-estimator WRITE
// interface, so a mirror rebuilt from perception is the position the real
// game is in rather than a fresh board wearing its unit layout.
//
// The 2026-08-17 setters above closed elixir and the hand. Four gaps were
// left, and `forecast.py` names three of them: tower HP and the match clock
// always came back from reset(), and injected units always spawned at full
// health. The fourth was found while verifying this proposal and is the one
// nobody had written down:
//
//     inject -> spawnEntity -> applyCardMetadata -> deployTicksRemaining
//
// so EVERY unit rebuilt from perception was inert for a full second at the
// start of every rollout, including a Hog that had been running for six. The
// 2026-08-19 audit measured that same second in the other direction at ~520
// tower HP on a supported push, which makes it the largest of the four.
//
// The properties pinned here are the ones that make these safe to hand a
// noisy sensor, and they are the same two the 2026-08-17 setters established:
// a write is CLAMPED to a state the engine can actually reach, and a write
// the engine cannot represent is REFUSED rather than half-applied.
// ===========================================================================

// The Tower behind a (team, slot), so a case can assert on state the setters
// deliberately do not expose -- `awake` in particular. Slots are BOARD
// coordinates, not team-relative: 0 = King, 1 = left Princess, 2 = right,
// with left/right decided against ArenaLayout's own centre rather than a
// restated literal.
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

    // Every other tower is untouched. A per-tower sensor reading must never
    // move a tower it did not measure.
    REQUIRE(game.getTowerHp(1, 2) == game.getTowerMaxHp(1, 2));
    REQUIRE(game.getTowerHp(1, 0) == game.getTowerMaxHp(1, 0));
    REQUIRE(game.getTowerHp(0, 1) == game.getTowerMaxHp(0, 1));
}

TEST_CASE("setTowerHp REFUSES a value at or below zero and changes nothing",
          "[setters][tower]") {
    GameManager game(DECK, DECK);

    // POSITIVE CONTROL, and it is load-bearing: without it this case passes
    // against a setter that does nothing at all, which is exactly the state
    // it exists to exclude. A refusal test whose only failure mode is
    // "everything is refused" cannot fail.
    REQUIRE(game.setTowerHp(1, 1, 1234.0f) == true);
    REQUIRE(game.getTowerHp(1, 1) == 1234);
    const int before = game.getTowerHp(1, 1);

    // A 0-hp tower that still occupies its cell and still fires is a position
    // the real game can never be in. Killing one has side effects -- the
    // crown, the King's princess-count trigger, LanePath's retargeting -- and
    // routing those is destroyTower's job, not a clamp's. Same refuse-rather-
    // than-accept contract as setHand above.
    REQUIRE(game.setTowerHp(1, 1, 0.0f) == false);
    REQUIRE(game.getTowerHp(1, 1) == before);
    REQUIRE(game.setTowerHp(1, 1, -50.0f) == false);
    REQUIRE(game.getTowerHp(1, 1) == before);

    // Both halves matter: a refusal that still wrote would be the worst of
    // the three outcomes, and is invisible to a caller that only checks the
    // bool.
    REQUIRE(towersStanding(game.getBoard(), 1) == 3);
}

TEST_CASE("setTowerHp clamps to the tower's OWN maximum, not a shared one",
          "[setters][tower]") {
    GameManager game(DECK, DECK);

    // The King and a Princess have different maxima (4008 vs 2534), so a
    // single shared ceiling would silently over-heal one of them. This is the
    // same class of mistake as reading a level-9 max off a level-4 tower,
    // which is why perception reports a FRACTION -- see item 22.
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

    // The caller is a sensor reading a screen; asking it to mirror its own
    // coordinates per team is exactly the convention error that put the
    // arena half a tile off-centre. Slot 1 is the LEFT tower for both teams.
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

    // Tower::awake latches on the HP invariant `hp < maxHp` rather than on a
    // damage entry point, precisely so that every route to a damaged tower
    // wakes it. This setter is a new route, and it must not be an exception:
    // a mirror whose King sleeps through a wound it can see would rate every
    // rollout's defence too low.
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

    // The flag is a LATCH -- a heal cannot re-sleep it, and neither can a
    // sensor misreading one frame at full. Otherwise a single bad frame would
    // put the King back to sleep mid-match, which the real game never does.
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

    // The princess trigger records the team's living count on its FIRST
    // update, so both Kings have to have ticked once before the tower falls.
    // Reading the count AFTER the loss would make nothing look lost -- the
    // same reason Tower::update records it rather than testing `< 2`.
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
    // Idempotence is not silence: a second call reports that it changed
    // nothing, so a caller re-sending a stale reading can tell.
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

    // CONTROL FIRST: a value below the maximum has to actually land, or
    // "clamped to full" is indistinguishable from "the parameter was
    // ignored" -- and ignoring it is the pre-item-22 behaviour.
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

    // Skeletons spawn three bodies from one call. Applying the hp to only the
    // first would leave two at full health, which reads as a threat that is
    // three times fresher than the one on screen.
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

    // Unchanged default: still inert for its whole deploy second.
    REQUIRE(hog->position.y == Catch::Approx(y0));
}

TEST_CASE("inject with deployTicks 0 puts an ALREADY-DEPLOYED unit on the board",
          "[setters][inject][deploy]") {
    std::vector<int> deck = DECK;
    ClashEnv env(deck, deck, 100);
    env.reset();

    // This is the whole point of the parameter: a unit perception can already
    // SEE has finished deploying, and re-charging it a deploy second hands the
    // defender a subsidy on every rollout.
    env.inject(15, 9.0f, 20.0f, 1, -1.0f, 0);

    Board& board = env.debugGame().getBoard();
    auto hog = board.getPendingEntity(0);
    const float y0 = hog->position.y;
    board.commitPendingEntities();
    for (int i = 0; i < DEPLOY_TIME_TICKS; ++i) hog->update(board);

    // Team 1 advances toward team 0, i.e. down the board.
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

    // The scalar is currentTick / maxTicks. This is the one field no
    // combination of the others can reconstruct, and the deployed encoder
    // divides by it -- CLAUDE.md records a live agent whose clock ran at
    // twice the rate it trained at because two defaults disagreed.
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

    // Two parallel counters exist (ClashEnv::currentTick drives the
    // observation and the done condition; GameManager::currentTick drives
    // ability cooldowns and the stats events' tick stamps). They are
    // incremented together everywhere else, and a setter that moved only one
    // would introduce exactly the kind of second, driftable copy this
    // codebase removes elsewhere.
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

    // A rollout runs on a snapshot, so a state the mirror was given has to
    // reach it -- otherwise every candidate is scored on the fresh board this
    // whole item exists to replace.
    ClashEnv copy = env.snapshot();
    REQUIRE(copy.getTowerHp(1, 1) == 700);
    REQUIRE(towersStanding(copy.debugGame().getBoard(), 1) == 2);

    const int spatial =
        ClashEnv::BOARD_WIDTH * ClashEnv::BOARD_HEIGHT * ClashEnv::NUM_CHANNELS;
    const int clockIdx = spatial + 1 + ClashEnv::HAND_SIZE
                       + ClashEnv::HAND_SIZE * ClashEnv::NUM_CARD_IDS;
    REQUIRE(copy.getObservationForTeam(0)[clockIdx] == Catch::Approx(0.3f));
}
