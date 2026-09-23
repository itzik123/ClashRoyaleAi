#include <catch_amalgamated.hpp>
#include "ClashEnv.h"
#include <vector>
#include <cmath>

// Opponent card-cycle observation (perception/UPSTREAM_REQUESTS.md item 24):
// two NUM_CARD_IDS-wide blocks carrying what the opponent has played.
//
//   seen[c]     1.0 once the opponent has played card c this match
//   recency[c]  exp(-(now - lastPlayed[c]) / CYCLE_RECENCY_TAU_TICKS)
//
// These pin the four silent failures: the blocks at the wrong offset; reading
// the own cycle instead of the opponent's; plays made through a path other than
// the hooked one (the heuristic opponent calls playCard directly); and a
// snapshot losing the cycle.

namespace {

// Where the first cycle block starts, derived the way the encoder builds it, so
// earlier changes move it too.
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

// The 2.6 Hog Cycle (DEFAULT_DECK): real registered cards.
const std::vector<int> DECK = { 15, 6, 25, 40, 24, 72, 33, 7 };


// Plays `cardId` as team 1 through the path a live match uses (stepSelfPlay ->
// GameManager::playCard), not inject(), which records no cycle. The hand is
// pinned first, since the opening hand is random.
bool playAsOpponent(ClashEnv& env, int cardId) {
    std::vector<int> hand = { cardId, DECK[1], DECK[2], DECK[3] };
    if (!env.setHandForTeam(1, hand)) return false;
    env.setElixirForTeam(1, 10.0f);
    // Team 1's y is in its own mirrored frame: runSelfPlayTicks converts
    // `realY1 = (BOARD_HEIGHT - 1) - targetY1`, so y=8 lands at board y=25, in
    // team 1's half. Board y=25 passed directly would land in team 0's half and
    // be refused.
    const float ownHalfY = 8.0f;
    // cardIndex 4 is team 0's no-op; team 1 plays slot 0.
    env.stepSelfPlay(4, 0.0f, 0.0f, 0, 9.0f, ownHalfY, 1);
    return env.getObservationForTeam(0)[cycleBase() + cardId] > 0.5f;
}

}  // namespace

TEST_CASE("the observation carries two card-cycle blocks after the extra scalars", "[cycle]") {
    ClashEnv env(DECK, DECK, 3600);
    env.reset();
    std::vector<float> obs = env.getObservationForTeam(0);

    REQUIRE(static_cast<int>(obs.size()) == env.observationSize());
    // The blocks are last, so the base plus both blocks ends exactly at the
    // vector's end; this fails if anything is inserted rather than appended.
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

    // Play as team 1 through the choke point every real play uses.
    const int card = DECK[0];
    REQUIRE(playAsOpponent(env, card));

    // stepSelfPlay advances the clock after the play, so pin the clock to the
    // recorded play tick to make the zero-age case exact.
    const int playedAt = env.getLastPlayedTick(1, card);
    REQUIRE(playedAt >= 0);
    env.setCurrentTick(playedAt);

    std::vector<float> team0 = env.getObservationForTeam(0);
    REQUIRE(seenOf(team0, card) == 1.0f);
    REQUIRE(recencyOf(team0, card) == Catch::Approx(1.0f));

    // ...and not in team 1's own view: its own cycle is not what these blocks
    // carry. Getting this backwards would look correct in every aggregate.
    std::vector<float> team1 = env.getObservationForTeam(1);
    REQUIRE(seenOf(team1, card) == 0.0f);
    REQUIRE(recencyOf(team1, card) == 0.0f);
}

TEST_CASE("recency decays with the documented time constant", "[cycle]") {
    ClashEnv env(DECK, DECK, 3600);
    env.reset();
    const int card = DECK[0];
    REQUIRE(playAsOpponent(env, card));

    // Measured from the recorded play tick, not "now", which stepSelfPlay has
    // already advanced.
    const int playedAt = env.getLastPlayedTick(1, card);
    REQUIRE(playedAt >= 0);
    // One TAU later: exp(-1) ~ 0.3679.
    env.setCurrentTick(playedAt + static_cast<int>(ClashEnv::CYCLE_RECENCY_TAU_TICKS));
    std::vector<float> obs = env.getObservationForTeam(0);

    REQUIRE(seenOf(obs, card) == 1.0f);          // seen never decays
    REQUIRE(recencyOf(obs, card) == Catch::Approx(std::exp(-1.0f)).margin(1e-4));
}

TEST_CASE("seen and recency answer different questions", "[cycle]") {
    // Why there are two blocks: played long ago reads seen=1, recency~0; never
    // played reads 0, 0. One channel could not tell those apart.
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
    // The pair distinguishes them, not either channel alone.
    REQUIRE(seenOf(obs, played) != seenOf(obs, never));
}

TEST_CASE("a rewound clock cannot push recency above 1", "[cycle]") {
    // set_current_tick() can move the clock backwards; a negative age would
    // push exp() above 1.0.
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
    // The cycle rides along with a snapshot, or every search candidate faces an
    // opponent who has played nothing.
    ClashEnv env(DECK, DECK, 3600);
    env.reset();
    const int card = DECK[0];
    REQUIRE(playAsOpponent(env, card));

    // Equality with the original, not a literal: the property is inheritance,
    // whatever the tick.
    ClashEnv copy = env.snapshot();
    std::vector<float> before = env.getObservationForTeam(0);
    std::vector<float> obs = copy.getObservationForTeam(0);
    REQUIRE(seenOf(obs, card) == 1.0f);
    REQUIRE(seenOf(obs, card) == seenOf(before, card));
    REQUIRE(recencyOf(obs, card) == recencyOf(before, card));
    REQUIRE(copy.getLastPlayedTick(1, card) == env.getLastPlayedTick(1, card));

    // ...and the snapshot is independent: a play inside the rollout must not
    // reach the live match's cycle.
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
