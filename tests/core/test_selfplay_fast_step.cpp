#include <catch_amalgamated.hpp>
#include "ClashEnv.h"
#include <vector>

// stepSelfPlayFast: the same advance as stepSelfPlay, without building the two
// observations a rollout discards.
//
// The two share one tick loop; these cases pin that they leave the environment
// in the same state, checked on the full observation (tick, elixir, hands,
// every entity). Comparing only reward and done would miss a skipped tick.

namespace {

// The shipped 2.6 Hog Cycle deck.
const std::vector<int> DECK = {15, 6, 25, 40, 24, 72, 33, 7};

// Advance to a mid-fight board: on an empty arena both paths trivially agree.
void warmUp(ClashEnv& env, int decisions) {
    for (int d = 0; d < decisions; ++d) {
        float x = (d % 2 == 0) ? 2.5f : 14.5f;
        env.stepSelfPlay(0, x, 14.0f, 0, x, 14.0f, 10);
        if (env.isGameOver()) break;
    }
}

}  // namespace

TEST_CASE("stepSelfPlayFast advances the match identically to stepSelfPlay",
          "[clash_env][selfplay][performance]") {
    ClashEnv root(DECK, DECK, 3600);
    root.seed(4242);
    root.reset();
    warmUp(root, 25);
    REQUIRE_FALSE(root.isGameOver());

    SECTION("same reward, same done, same resulting board -- no card played") {
        ClashEnv slow = root.snapshot();
        ClashEnv fast = root.snapshot();

        auto slowResult = slow.stepSelfPlay(-1, 0.0f, 0.0f, -1, 0.0f, 0.0f, 10);
        auto fastResult = fast.stepSelfPlayFast(-1, 0.0f, 0.0f, -1, 0.0f, 0.0f, 10);

        REQUIRE(fastResult.reward0 == slowResult.reward0);
        REQUIRE(fastResult.done == slowResult.done);

        // Bit-identical: the engine is deterministic and both paths run one
        // loop.
        REQUIRE(fast.getObservationForTeam(0) == slow.getObservationForTeam(0));
        REQUIRE(fast.getObservationForTeam(1) == slow.getObservationForTeam(1));
    }

    SECTION("same when BOTH sides play a card -- placement must still happen") {
        ClashEnv slow = root.snapshot();
        ClashEnv fast = root.snapshot();

        // Fund the bar first: after the warm-up team 0 has ~0.75 elixir, the
        // play would be refused, and both arms would agree because neither
        // placed anything.
        slow.setElixirForTeam(0, 10.0f);
        slow.setElixirForTeam(1, 10.0f);
        fast.setElixirForTeam(0, 10.0f);
        fast.setElixirForTeam(1, 10.0f);

        auto slowResult = slow.stepSelfPlay(0, 5.0f, 10.0f, 0, 12.0f, 10.0f, 10);
        auto fastResult = fast.stepSelfPlayFast(0, 5.0f, 10.0f, 0, 12.0f, 10.0f, 10);

        REQUIRE(fastResult.reward0 == slowResult.reward0);
        REQUIRE(fastResult.done == slowResult.done);
        REQUIRE(fast.getObservationForTeam(0) == slow.getObservationForTeam(0));
        REQUIRE(fast.getObservationForTeam(1) == slow.getObservationForTeam(1));

        // The elixir bar proves a card was played: 10.0 minus its cost plus ten
        // ticks of regen stays below 10.0, since every card here costs more
        // than 0.35.
        REQUIRE(fast.getElixirForTeam(0) == slow.getElixirForTeam(0));
        REQUIRE(fast.getElixirForTeam(0) < 10.0f);
        REQUIRE(fast.getElixirForTeam(1) == slow.getElixirForTeam(1));
        REQUIRE(fast.getElixirForTeam(1) < 10.0f);
    }

    SECTION("stays identical over a full rollout horizon, chunked as the "
            "teacher chunks it") {
        ClashEnv slow = root.snapshot();
        ClashEnv fast = root.snapshot();

        // 10 chunks of 10 ticks: the top teacher rung's horizon.
        for (int chunk = 0; chunk < 10; ++chunk) {
            auto s = slow.stepSelfPlay(-1, 0.0f, 0.0f, -1, 0.0f, 0.0f, 10);
            auto f = fast.stepSelfPlayFast(-1, 0.0f, 0.0f, -1, 0.0f, 0.0f, 10);
            REQUIRE(f.reward0 == s.reward0);
            REQUIRE(f.done == s.done);
            if (s.done) break;
        }
        REQUIRE(fast.getObservationForTeam(0) == slow.getObservationForTeam(0));
        REQUIRE(fast.getObservationForTeam(1) == slow.getObservationForTeam(1));
    }

    SECTION("a zero-tick fast call places nothing, exactly like the slow one") {
        // Placement happens inside the tick loop, so a 0-tick call places
        // nothing; teacher combos depend on it (test_teacher_combos.py pins the
        // same on the Python side).
        ClashEnv slow = root.snapshot();
        ClashEnv fast = root.snapshot();
        float before = root.getElixirForTeam(0);

        slow.stepSelfPlay(0, 5.0f, 10.0f, -1, 0.0f, 0.0f, 0);
        fast.stepSelfPlayFast(0, 5.0f, 10.0f, -1, 0.0f, 0.0f, 0);

        REQUIRE(slow.getElixirForTeam(0) == before);
        REQUIRE(fast.getElixirForTeam(0) == before);
        REQUIRE(fast.getObservationForTeam(0) == slow.getObservationForTeam(0));
    }
}

TEST_CASE("stepSelfPlay still returns both observations after the refactor",
          "[clash_env][selfplay]") {
    // stepSelfPlay still returns both observations, the right size, each in its
    // own team's frame.
    ClashEnv env(DECK, DECK, 3600);
    env.seed(99);
    env.reset();
    warmUp(env, 20);

    auto r = env.stepSelfPlay(-1, 0.0f, 0.0f, -1, 0.0f, 0.0f, 10);
    REQUIRE(r.observation0.size() == static_cast<size_t>(env.observationSize()));
    REQUIRE(r.observation1.size() == static_cast<size_t>(env.observationSize()));
    REQUIRE(r.observation0 == env.getObservationForTeam(0));
    REQUIRE(r.observation1 == env.getObservationForTeam(1));
    REQUIRE(r.observation0 != r.observation1);
}
