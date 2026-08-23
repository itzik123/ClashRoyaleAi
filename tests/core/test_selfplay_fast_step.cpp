#include <catch_amalgamated.hpp>
#include "ClashEnv.h"
#include <vector>

// stepSelfPlayFast: the same match advance as stepSelfPlay, without building
// the two observation vectors the caller is going to throw away.
//
// WHY IT EXISTS. `UtilityTeacher.execute_steps` rolls every candidate forward in
// 10-tick chunks and discards the returned object each time. Measured on the
// training box (profile_training.py --mode sync, 48 episodes, teacher stage 5):
// 500,076 chunk calls building 1,000,152 observation vectors, against 51,566
// ever read -- a 19.4 : 1 build-to-read ratio, and 14.9% of wall clock. See
// perception/UPSTREAM_REQUESTS.md item 21.
//
// WHAT THESE CASES PIN, and it is the only thing that makes the refactor safe:
// the two entry points share ONE tick loop, so they cannot diverge in what they
// simulate. The assertion is therefore not "fast is correct" in isolation but
// "fast and slow leave the environment in the SAME state" -- which is checked
// against the full observation vector, because that encodes the tick counter,
// both elixir bars, both hands and every entity's position and HP. A test that
// compared only reward and done would pass even if the fast path silently
// skipped a tick.

namespace {

// python_ai/envs/gym_wrapper.py DEFAULT_DECK -- the 2.6 Hog Cycle.
const std::vector<int> DECK = {15, 6, 25, 40, 24, 72, 33, 7};

// Advance to a mid-fight board. On an empty arena both paths trivially agree,
// so an equivalence test anchored there proves almost nothing -- the same
// "cross-check anchored where the error is zero" trap CLAUDE.md records for the
// 2026-08-05 tile-grid refit and the lane-pathing bridge mouth.
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

        // The full state, both frames. Bit-identical, not approximately equal:
        // the engine is deterministic and the two paths run the same loop, so
        // any difference at all is a real divergence.
        REQUIRE(fast.getObservationForTeam(0) == slow.getObservationForTeam(0));
        REQUIRE(fast.getObservationForTeam(1) == slow.getObservationForTeam(1));
    }

    SECTION("same when BOTH sides play a card -- placement must still happen") {
        ClashEnv slow = root.snapshot();
        ClashEnv fast = root.snapshot();

        // THE BAR MUST BE FUNDED FIRST. Without this the section silently
        // degenerates: 25 decisions of warm-up leave team 0 on ~0.75 elixir, so
        // playCard is refused, ten ticks of regen run instead, and the two arms
        // agree because NEITHER placed anything. The first version of this test
        // asserted the bar had fallen and failed with `1.10 < 0.75` -- elixir had
        // gone UP. That failure is the only reason the section is not still a
        // no-op comparison dressed up as a placement test.
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

        // The elixir bar is what proves a card was actually PLAYED rather than
        // the call being a no-op that happens to match another no-op. 10.0 minus
        // a card's cost, plus ten ticks of regen, must still be below 10.0 --
        // every card in this deck costs more than 0.35.
        REQUIRE(fast.getElixirForTeam(0) == slow.getElixirForTeam(0));
        REQUIRE(fast.getElixirForTeam(0) < 10.0f);
        REQUIRE(fast.getElixirForTeam(1) == slow.getElixirForTeam(1));
        REQUIRE(fast.getElixirForTeam(1) < 10.0f);
    }

    SECTION("stays identical over a full rollout horizon, chunked as the "
            "teacher chunks it") {
        ClashEnv slow = root.snapshot();
        ClashEnv fast = root.snapshot();

        // 10 chunks of 10 ticks == teacher stage 5's horizon_ticks = 100.
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
        // Pinned because teacher combos depend on it: placement is processed
        // INSIDE the tick loop, so a 0-tick call spends no elixir and puts no
        // unit on the board. test_teacher_combos.py pins the same property on
        // the Python side; the fast path must not quietly change it.
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
    // The refactor moves stepSelfPlay's loop body into a shared helper. This is
    // the guard that its RETURN did not change with it: the observations must
    // still be there, still be the right size, and still be each team's own
    // mirrored frame rather than two copies of one.
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
