#include <catch_amalgamated.hpp>
#include "ClashEnv.h"
#include <vector>
#include <cmath>

// OPPONENT CARD-CYCLE OBSERVATION (UPSTREAM_REQUESTS.md item 24, 2026-08-27).
//
// The encoder gave the agent the opponent's cumulative elixir SPEND and nothing
// else about what they had played -- a scalar sum of costs, which is precisely
// the summary that destroys cycle information. Two NUM_CARD_IDS-wide blocks now
// carry what a human watching the screen already knows:
//
//   seen[c]     1.0 once the opponent has played card c this match
//   recency[c]  exp(-(now - lastPlayed[c]) / CYCLE_RECENCY_TAU_TICKS)
//
// These tests pin the four things that can go wrong SILENTLY, in the sense this
// project keeps being bitten by -- each of them would leave a
// plausible-looking observation that is simply describing the wrong thing:
//
//   1. the blocks landing at the wrong offset (every downstream index shifts)
//   2. reading the OWN cycle instead of the opponent's (trains on information
//      the agent already has, leaves the gap unfilled, and looks identical in
//      any aggregate statistic)
//   3. plays made through a path other than the one hooked (the heuristic
//      opponent reaches playCard directly, not through ClashEnv::step)
//   4. a snapshot losing the cycle, so every search candidate believes the
//      opponent has played nothing all match

namespace {

// Where the first cycle block starts. Derived the same way the encoder builds
// it -- spatial, then elixir, costs, one-hots, extra scalars -- rather than
// hardcoded, so a change to any earlier section moves this with it instead of
// silently pointing at the wrong floats.
constexpr int cycleBase() {
    return ClashEnv::BOARD_WIDTH * ClashEnv::BOARD_HEIGHT * ClashEnv::NUM_CHANNELS
         + 1
         + ClashEnv::HAND_SIZE
         + ClashEnv::HAND_SIZE * ClashEnv::NUM_CARD_IDS
         + ClashEnv::NUM_EXTRA_SCALARS;
}

float seenOf(const std::vector<float>& obs, int cardId) {
    return obs[cycleBase() + cardId];
}
float recencyOf(const std::vector<float>& obs, int cardId) {
    return obs[cycleBase() + ClashEnv::NUM_CARD_IDS + cardId];
}

// The 2.6 Hog Cycle -- DEFAULT_DECK, so these ids are real registered cards.
const std::vector<int> DECK = { 15, 6, 25, 40, 24, 72, 33, 7 };


// Plays `cardId` as team 1 through the SAME path a live match uses --
// stepSelfPlay -> GameManager::playCard -- rather than through inject(), which
// deliberately bypasses playCard and therefore records no cycle. Pinning the
// hand first is what makes the play deterministic: the opening hand is dealt
// by an unseeded shuffle, so slot 0 is otherwise an unknown card.
bool playAsOpponent(ClashEnv& env, int cardId) {
    std::vector<int> hand = { cardId, DECK[1], DECK[2], DECK[3] };
    if (!env.setHandForTeam(1, hand)) return false;
    env.setElixirForTeam(1, 10.0f);
    // TEAM 1'S Y IS IN TEAM 1'S OWN MIRRORED FRAME, not board coordinates:
    // runSelfPlayTicks converts it as `realY1 = (BOARD_HEIGHT - 1) - targetY1`,
    // which is what lets one network drive either side. So y=8 here lands at
    // board y=25 -- inside team 1's own half. Passing the board coordinate 25
    // directly maps to board y=8, which is team 0's half, and the play is
    // correctly REFUSED; that is what these tests did on first run.
    const float ownHalfY = 8.0f;
    // cardIndex 4 == the no-op arm for team 0; team 1 plays slot 0.
    env.stepSelfPlay(4, 0.0f, 0.0f, 0, 9.0f, ownHalfY, 1);
    return env.getObservationForTeam(0)[cycleBase() + cardId] > 0.5f;
}

}  // namespace

TEST_CASE("the observation carries two card-cycle blocks after the extra scalars", "[cycle]") {
    ClashEnv env(DECK, DECK, 3600);
    env.reset();
    std::vector<float> obs = env.getObservationForTeam(0);

    REQUIRE(static_cast<int>(obs.size()) == env.observationSize());
    // The blocks are the LAST thing in the vector, so the base plus both
    // blocks must land exactly on the end. This is what fails if anything is
    // ever inserted rather than appended.
    REQUIRE(cycleBase() + ClashEnv::CYCLE_BLOCK_SIZE == env.observationSize());
    REQUIRE(ClashEnv::CYCLE_BLOCK_SIZE
            == ClashEnv::NUM_CYCLE_BLOCKS * ClashEnv::NUM_CARD_IDS);
}

TEST_CASE("on a fresh match nothing is seen and nothing is recent", "[cycle]") {
    ClashEnv env(DECK, DECK, 3600);
    env.reset();
    std::vector<float> obs = env.getObservationForTeam(0);

    for (int c = 0; c < ClashEnv::NUM_CARD_IDS; ++c) {
        REQUIRE(seenOf(obs, c) == 0.0f);
        REQUIRE(recencyOf(obs, c) == 0.0f);
    }
}

TEST_CASE("an opponent play shows up in the observing team's cycle blocks", "[cycle]") {
    ClashEnv env(DECK, DECK, 3600);
    env.reset();

    // Play as team 1 through the same choke point every real play uses.
    const int card = DECK[0];
    REQUIRE(playAsOpponent(env, card));

    // stepSelfPlay advances the clock AFTER the play lands, so "now" is a
    // tick or two past it. Pin the clock to the recorded play tick to make the
    // zero-age case exact rather than approximately-one.
    const int playedAt = env.getLastPlayedTick(1, card);
    REQUIRE(playedAt >= 0);
    env.setCurrentTick(playedAt);

    std::vector<float> team0 = env.getObservationForTeam(0);
    REQUIRE(seenOf(team0, card) == 1.0f);
    REQUIRE(recencyOf(team0, card) == Catch::Approx(1.0f));

    // ...and it must NOT appear in team 1's own view. Team 1 played it, so for
    // team 1 this is own-cycle information, which these blocks deliberately do
    // not carry. Getting this backwards is the failure that would look correct
    // in every aggregate metric.
    std::vector<float> team1 = env.getObservationForTeam(1);
    REQUIRE(seenOf(team1, card) == 0.0f);
    REQUIRE(recencyOf(team1, card) == 0.0f);
}

TEST_CASE("recency decays with the documented time constant", "[cycle]") {
    ClashEnv env(DECK, DECK, 3600);
    env.reset();
    const int card = DECK[0];
    REQUIRE(playAsOpponent(env, card));

    // Measured from the RECORDED play tick, not from "now" -- stepSelfPlay
    // has already advanced the clock past it, and starting the interval in the
    // wrong place is how a decay test ends up asserting against its own
    // rounding.
    const int playedAt = env.getLastPlayedTick(1, card);
    REQUIRE(playedAt >= 0);
    // One full TAU later the channel must read exp(-1) ~ 0.3679.
    env.setCurrentTick(playedAt + static_cast<int>(ClashEnv::CYCLE_RECENCY_TAU_TICKS));
    std::vector<float> obs = env.getObservationForTeam(0);

    REQUIRE(seenOf(obs, card) == 1.0f);          // seen never decays
    REQUIRE(recencyOf(obs, card) == Catch::Approx(std::exp(-1.0f)).margin(1e-4));
}

TEST_CASE("seen and recency answer different questions", "[cycle]") {
    // The reason there are TWO blocks rather than one. A card played long ago
    // reads seen=1, recency~0; a card never played reads seen=0, recency=0.
    // A single channel could not tell "they have it and it is nearly back"
    // from "they have never shown it", which is the distinction the whole
    // feature exists for.
    ClashEnv env(DECK, DECK, 3600);
    env.reset();
    const int played = DECK[0];
    const int never = DECK[1];
    REQUIRE(playAsOpponent(env, played));

    env.setCurrentTick(env.getLastPlayedTick(1, played)
                       + static_cast<int>(10.0f * ClashEnv::CYCLE_RECENCY_TAU_TICKS));
    std::vector<float> obs = env.getObservationForTeam(0);

    REQUIRE(seenOf(obs, played) == 1.0f);
    REQUIRE(recencyOf(obs, played) < 0.01f);     // long gone
    REQUIRE(seenOf(obs, never) == 0.0f);
    REQUIRE(recencyOf(obs, never) == 0.0f);
    // The pair is what distinguishes them, not either channel alone.
    REQUIRE(seenOf(obs, played) != seenOf(obs, never));
}

TEST_CASE("a rewound clock cannot push recency above 1", "[cycle]") {
    // set_current_tick() lets a state estimator move the clock BACKWARDS, and
    // a negative age would make exp(-age/TAU) exceed 1.0 and break the [0,1]
    // contract every other channel keeps.
    ClashEnv env(DECK, DECK, 3600);
    env.reset();
    env.setCurrentTick(500);
    const int card = DECK[0];
    REQUIRE(playAsOpponent(env, card));

    env.setCurrentTick(100);                     // rewind, behind the play
    std::vector<float> obs = env.getObservationForTeam(0);
    REQUIRE(recencyOf(obs, card) <= 1.0f);
    REQUIRE(recencyOf(obs, card) == Catch::Approx(1.0f));
}

TEST_CASE("a snapshot inherits the cycle it is searching from", "[cycle]") {
    // Decision-time search rolls a snapshot forward. If the cycle did not ride
    // along, every candidate would be scored against an opponent who had
    // apparently played nothing all match -- a wrong answer that looks like a
    // working search, which is this project's most-repeated failure shape.
    ClashEnv env(DECK, DECK, 3600);
    env.reset();
    const int card = DECK[0];
    REQUIRE(playAsOpponent(env, card));

    // Asserted as EQUALITY WITH THE ORIGINAL rather than against a literal.
    // The property under test is "the snapshot inherited the cycle", and that
    // holds whatever the current tick happens to be -- pinning a literal here
    // instead just re-tests the decay formula, and gets 0.99501 rather than
    // 1.0 because stepSelfPlay advances the clock one tick past the play.
    ClashEnv copy = env.snapshot();
    std::vector<float> before = env.getObservationForTeam(0);
    std::vector<float> obs = copy.getObservationForTeam(0);
    REQUIRE(seenOf(obs, card) == 1.0f);
    REQUIRE(seenOf(obs, card) == seenOf(before, card));
    REQUIRE(recencyOf(obs, card) == recencyOf(before, card));
    REQUIRE(copy.getLastPlayedTick(1, card) == env.getLastPlayedTick(1, card));

    // ...and the snapshot must be INDEPENDENT: a play inside the rollout must
    // not appear in the live match's cycle. Sharing that state would let a
    // search write its hypotheticals back into the position it is searching.
    const int other = DECK[1];
    copy.notePlayedCard(1, other);
    REQUIRE(copy.getLastPlayedTick(1, other) >= 0);
    REQUIRE(env.getLastPlayedTick(1, other) == -1);
}

TEST_CASE("reset clears the cycle so a new match starts blind", "[cycle]") {
    ClashEnv env(DECK, DECK, 3600);
    env.reset();
    const int card = DECK[0];
    REQUIRE(playAsOpponent(env, card));
    REQUIRE(seenOf(env.getObservationForTeam(0), card) == 1.0f);

    env.reset();
    std::vector<float> obs = env.getObservationForTeam(0);
    REQUIRE(seenOf(obs, card) == 0.0f);
    REQUIRE(recencyOf(obs, card) == 0.0f);
}

TEST_CASE("every cycle value stays inside the [0,1] range the encoder promises", "[cycle]") {
    ClashEnv env(DECK, DECK, 3600);
    env.reset();
    for (int i = 0; i < 40; ++i) {
        playAsOpponent(env, DECK[i % DECK.size()]);
    }
    std::vector<float> obs = env.getObservationForTeam(0);
    for (int i = cycleBase(); i < cycleBase() + ClashEnv::CYCLE_BLOCK_SIZE; ++i) {
        REQUIRE(obs[i] >= 0.0f);
        REQUIRE(obs[i] <= 1.0f);
        REQUIRE(std::isfinite(obs[i]));
    }
}
