// Engine-side cost breakdown for the training loop. A measurement harness built
// standalone (tools/audit/build.ps1, or build.sh under wsl g++).
//
// From Python a step times three fused things: (a) the physics tick,
// GameManager::step(); (b) building the observation,
// extractObservationForTeam(); (c) copying it across pybind. This measures (a)
// and (b) with no interpreter, so the Python profiler's numbers minus these
// give (c).
//
// pybind's def_readonly converts a vector to a Python list on attribute access,
// so an observation that is returned but never read costs its C++ construction
// and no marshalling. Everything here is construction cost; the boundary is
// what profile_training.py --mode boundary measures.

// Reports the minimum of N repeats with the median beside it: deterministic
// cost, additive noise.

#include "ClashEnv.h"
#include "GameManager.h"
#include "CardRegistry.h"
#include "Board.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

// DEFAULT_DECK, the 2.6 Hog Cycle.
const std::vector<int> DEFAULT_DECK = {15, 6, 25, 40, 24, 72, 33, 7};

// One decision is one skip_frames block, as in gym_wrapper's step().
constexpr int SKIP_FRAMES = 10;

struct Sample {
    double minMs = 0.0;
    double medianMs = 0.0;
    int reps = 0;
};

Sample summarize(std::vector<double> ms) {
    Sample s;
    s.reps = static_cast<int>(ms.size());
    if (ms.empty()) return s;
    std::sort(ms.begin(), ms.end());
    s.minMs = ms.front();
    s.medianMs = ms[ms.size() / 2];
    return s;
}

void row(const char* label, const Sample& s, const char* note = "") {
    std::printf("  %-48s %9.4f %9.4f  %5d  %s\n",
                label, s.minMs, s.medianMs, s.reps, note);
}

void header(const char* title) {
    std::printf("\n%s\n", title);
    std::printf("  %-48s %9s %9s  %5s\n", "", "min ms", "med ms", "reps");
}

double elapsedMs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int liveEntities(const GameManager& g) {
    int n = 0;
    for (const auto& e : g.getBoard().getEntities()) {
        if (e->isAlive()) ++n;
    }
    return n;
}

// Drive a match to a realistic mid-fight entity count; an empty board would
// flatter every number.
void warmUp(ClashEnv& env, int decisions) {
    for (int d = 0; d < decisions; ++d) {
        float x = (d % 2 == 0) ? 2.5f : 14.5f;
        env.stepSelfPlay(0, x, 14.0f, 0, x, 14.0f, SKIP_FRAMES);
        if (env.isGameOver()) break;
    }
}

}  // namespace

int main(int argc, char** argv) {
    int reps = (argc > 1) ? std::atoi(argv[1]) : 200;
    if (reps < 5) reps = 5;

    std::printf("engine_profile -- C++ cost of the phase-1 training loop\n");
    std::printf("reps=%d  skip_frames=%d  deck=2.6 Hog Cycle\n", reps, SKIP_FRAMES);

    ClashEnv env(DEFAULT_DECK, DEFAULT_DECK, 3600);
    env.seed(12345);
    env.reset();
    warmUp(env, 40);
    std::printf("observation_size=%d\n", env.observationSize());

    double obsMs = 0.0;
    double physics10Ms = 0.0;

    // ---------------------------------------------------------------- (b)
    header("OBSERVATION CONSTRUCTION (no interpreter, no pybind)");
    {
        std::vector<double> ms;
        for (int i = 0; i < reps; ++i) {
            auto t0 = Clock::now();
            auto obs = env.getObservationForTeam(0);
            auto t1 = Clock::now();
            if (obs.empty()) std::printf("!");   // defeat dead-store removal
            ms.push_back(elapsedMs(t0, t1));
        }
        Sample s = summarize(ms);
        obsMs = s.minMs;
        row("extractObservationForTeam(0)", s, "one 13,606-float vector");
    }

    // ---------------------------------------------------------------- (a)
    header("PHYSICS (GameManager::step -- no observation, no logger)");
    {
        GameManager g(DEFAULT_DECK, DEFAULT_DECK);
        g.reset();
        for (int d = 0; d < 40; ++d) {
            float x = (d % 2 == 0) ? 2.5f : 14.5f;
            const auto& h0 = g.getHand(0);
            if (!h0.empty()) g.playCard(0, h0[0], x, 14.0f);
            for (int t = 0; t < SKIP_FRAMES; ++t) g.step();
            if (g.isGameOver()) g.reset();
        }
        std::printf("  (live entities on the measured board: %d)\n", liveEntities(g));

        std::vector<double> ms;
        for (int i = 0; i < reps; ++i) {
            auto t0 = Clock::now();
            for (int t = 0; t < SKIP_FRAMES; ++t) g.step();
            auto t1 = Clock::now();
            ms.push_back(elapsedMs(t0, t1));
            if (g.isGameOver()) g.reset();
        }
        Sample s = summarize(ms);
        physics10Ms = s.minMs;
        row("10 ticks (== one decision of physics)", s);
        Sample per = s;
        per.minMs /= SKIP_FRAMES;
        per.medianMs /= SKIP_FRAMES;
        row("  ...per tick", per);
    }

    // ------------------------------------------------------- (a)+(b)+logger
    header("ClashEnv::stepSelfPlay -- what the .pyd actually calls");
    {
        std::vector<double> ms;
        for (int i = 0; i < reps; ++i) {
            ClashEnv s = env.snapshot();
            auto t0 = Clock::now();
            auto r = s.stepSelfPlay(-1, 0.0f, 0.0f, -1, 0.0f, 0.0f, SKIP_FRAMES);
            auto t1 = Clock::now();
            if (r.observation0.empty()) std::printf("!");
            ms.push_back(elapsedMs(t0, t1));
        }
        Sample s = summarize(ms);
        row("stepSelfPlay(skip=10)", s, "10 physics + 10 logTick + TWO observations");
        std::printf("  accounted: 10 physics %.4f + 2 obs %.4f = %.4f ms"
                    "   -> logger+overhead %.4f ms\n",
                    physics10Ms, 2 * obsMs, physics10Ms + 2 * obsMs,
                    s.minMs - physics10Ms - 2 * obsMs);
        std::printf("  observations are %.1f%% of this call, and gym_wrapper.step\n"
                    "  reads only observation0 -- observation1 is built and dropped.\n",
                    100.0 * (2 * obsMs) / s.minMs);
    }

    // Measure the logger directly rather than inferring it from a residual: it
    // is enabled by default, and a snapshot's logger records every tick of a
    // rollout and then discards it.
    header("GameLogger::logTick -- measured directly, not by subtraction");
    {
        GameManager g(DEFAULT_DECK, DEFAULT_DECK);
        g.reset();
        for (int d = 0; d < 40; ++d) {
            float x = (d % 2 == 0) ? 2.5f : 14.5f;
            const auto& h0 = g.getHand(0);
            if (!h0.empty()) g.playCard(0, h0[0], x, 14.0f);
            for (int t = 0; t < SKIP_FRAMES; ++t) g.step();
            if (g.isGameOver()) g.reset();
        }

        std::vector<double> on, off;
        for (int i = 0; i < reps; ++i) {
            {
                GameLogger lg;                       // enabled(true) by default
                auto t0 = Clock::now();
                for (int t = 0; t < SKIP_FRAMES; ++t) lg.logTick(t, g);
                auto t1 = Clock::now();
                on.push_back(elapsedMs(t0, t1));
            }
            {
                GameLogger lg;
                lg.setEnabled(false);
                auto t0 = Clock::now();
                for (int t = 0; t < SKIP_FRAMES; ++t) lg.logTick(t, g);
                auto t1 = Clock::now();
                off.push_back(elapsedMs(t0, t1));
            }
        }
        Sample sOn = summarize(on);
        Sample sOff = summarize(off);
        row("10 logTick calls, enabled (the default)", sOn);
        row("10 logTick calls, disabled", sOff, "the early-return guard");
        std::printf("  logging costs %.4f ms per decision, against %.4f ms of physics"
                    "  -- %.1fx the physics it is recording.\n",
                    sOn.minMs - sOff.minMs, physics10Ms,
                    (sOn.minMs - sOff.minMs) / physics10Ms);
    }

    // A residual is not an explanation. Sweep skipFrames: the slope is the
    // per-tick cost (physics, logTick, calculateReward, isGameOver) and the
    // intercept the per-call cost, dominated by the two observations returned.
    header("skipFrames SWEEP -- separates per-tick cost from per-call cost");
    {
        double x1 = 0.0, x10 = 0.0;
        for (int frames : {1, 2, 5, 10}) {
            std::vector<double> ms;
            for (int i = 0; i < reps; ++i) {
                ClashEnv s = env.snapshot();
                auto t0 = Clock::now();
                auto r = s.stepSelfPlay(-1, 0.0f, 0.0f, -1, 0.0f, 0.0f, frames);
                auto t1 = Clock::now();
                if (r.observation0.empty()) std::printf("!");
                ms.push_back(elapsedMs(t0, t1));
            }
            char label[64];
            std::snprintf(label, sizeof(label), "stepSelfPlay(skip=%2d)", frames);
            Sample s = summarize(ms);
            row(label, s);
            if (frames == 1) x1 = s.minMs;
            if (frames == 10) x10 = s.minMs;
        }
        double perTick = (x10 - x1) / 9.0;
        double perCall = x1 - perTick;
        std::printf("  slope  = %.4f ms per TICK  (physics %.4f + logTick %.4f"
                    " + reward/gameover)\n", perTick, physics10Ms / 10.0, 0.0);
        std::printf("  intercept = %.4f ms per CALL, against %.4f ms for the two\n"
                    "              observations it returns (%.0f%% of the intercept).\n",
                    perCall, 2 * obsMs, 100.0 * (2 * obsMs) / perCall);
    }

    header("SNAPSHOT (what every teacher candidate rollout starts with)");
    {
        std::vector<double> ms;
        for (int i = 0; i < reps; ++i) {
            auto t0 = Clock::now();
            ClashEnv s = env.snapshot();
            auto t1 = Clock::now();
            if (s.observationSize() == 0) std::printf("!");
            ms.push_back(elapsedMs(t0, t1));
        }
        row("ClashEnv::snapshot()", summarize(ms), "deep copy, empty logger");
    }

    // --- one teacher candidate, shaped like UtilityTeacher.rollout_stats ---
    // snapshot -> execute_steps(horizon) in 10-tick chunks -> stats -> one
    // observation. With stepSelfPlay every chunk builds two observations the
    // rollout never reads.
    header("ONE TEACHER CANDIDATE ROLLOUT (teacher.rollout_stats shape)");
    {
        const int horizons[] = {30, 50, 70, 100};
        const char* stages[] = {"2", "3", "4", "5"};
        for (int hi = 0; hi < 4; ++hi) {
            int horizon = horizons[hi];
            std::vector<double> ms;
            for (int i = 0; i < reps; ++i) {
                auto t0 = Clock::now();
                ClashEnv s = env.snapshot();
                for (int t = 0; t < horizon; t += SKIP_FRAMES) {
                    int chunk = std::min(SKIP_FRAMES, horizon - t);
                    s.stepSelfPlay(-1, 0.0f, 0.0f, -1, 0.0f, 0.0f, chunk);
                    if (s.isGameOver()) break;
                }
                auto obs = s.getObservationForTeam(1);
                auto t1 = Clock::now();
                if (obs.empty()) std::printf("!");
                ms.push_back(elapsedMs(t0, t1));
            }
            char label[96];
            std::snprintf(label, sizeof(label),
                          "horizon %3d ticks  (teacher stage %s)", horizon, stages[hi]);
            Sample s = summarize(ms);
            row(label, s);
            int chunks = horizon / SKIP_FRAMES;
            double wasted = 2.0 * chunks * obsMs;
            std::printf("      of which %2d discarded observations = %.4f ms"
                        "  (%.1f%% of the rollout)\n",
                        2 * chunks, wasted, 100.0 * wasted / s.minMs);
        }
    }

    // --- what an observation-free step saves ---
    // The same 100 ticks through GameManager directly, which builds no
    // observation: the difference against the block above. Same physics,
    // population and snapshot depth.
    header("WHAT A ROLLOUT COSTS WITHOUT THE DISCARDED OBSERVATIONS");
    {
        GameManager g(DEFAULT_DECK, DEFAULT_DECK);
        g.reset();
        for (int d = 0; d < 40; ++d) {
            float x = (d % 2 == 0) ? 2.5f : 14.5f;
            const auto& h0 = g.getHand(0);
            if (!h0.empty()) g.playCard(0, h0[0], x, 14.0f);
            for (int t = 0; t < SKIP_FRAMES; ++t) g.step();
            if (g.isGameOver()) g.reset();
        }

        std::vector<double> ms;
        for (int i = 0; i < reps; ++i) {
            auto t0 = Clock::now();
            GameManager c = g.snapshot();
            for (int t = 0; t < 100; ++t) {
                c.step();
                if (c.isGameOver()) break;
            }
            auto t1 = Clock::now();
            ms.push_back(elapsedMs(t0, t1));
        }
        Sample s = summarize(ms);
        row("snapshot + 100 ticks, NO observations built", s,
            "the irreducible work");
        std::printf("  add ONE observation for positional_advantage: %.4f ms\n",
                    s.minMs + obsMs);
        std::printf("  against the horizon-100 rollout measured above.\n");
        std::printf("  NOTE: GameManager::snapshot is a shallower copy than\n"
                    "  ClashEnv::snapshot (no env wrapper, no logger), so read\n"
                    "  this as the FLOOR of an observation-free rollout, not as\n"
                    "  a drop-in substitute.\n");
    }

    std::printf("\nRead the last block against the ONE observation a rollout\n"
                "actually reads: everything else it builds is thrown away.\n");
    std::printf("done.\n");
    return 0;
}
