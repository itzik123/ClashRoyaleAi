// Measures the elixir phase schedule. Standalone like every instrument here,
// compiling straight against the header-only engine with cl.exe. Four questions
// the source cannot answer:
//
//   1. does income run 1x / 2x / 3x across the two boundaries
//   2. does the phase compose with oppElixirMultiplier rather than replace it
//   3. does the observation carry the phase at the right index, normalised to 3.0
//   4. did CYCLE_START and observationSize() move by exactly one
//
// Build:  powershell -File tools/audit/build.ps1 elixir_phase_audit
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>

#include "ClashEnv.h"
#include "GameLogger.h"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

void checkNear(float got, float want, float tol, const char* what) {
    const bool ok = std::fabs(got - want) <= tol;
    std::printf("  [%s] %-52s got %8.4f  want %8.4f\n",
                ok ? "PASS" : "FAIL", what, got, want);
    if (!ok) ++failures;
}

// A no-op action. ClashEnv::step treats any cardIndex outside [0, HAND_SIZE)
// as "play nothing", so this advances the clock and regen and nothing else.
constexpr int NOOP = ClashEnv::HAND_SIZE;

// Elixir gained by `team` over `ticks` from `startTick`, measured by setting
// the clock and stepping, so it reads the bar rather than the schedule. Short
// windows, since the bar clamps at 10.0 and a long one saturates.
// stepSelfPlayFast, never step(): the HeuristicOpponent would spend team 1's
// elixir.
float incomeOver(int startTick, int ticks, int team, float oppMultiplier) {
    std::vector<int> deck = { 15, 6, 25, 40, 24, 72, 33, 7 };   // DEFAULT_DECK
    ClashEnv env(deck, deck);
    env.reset();
    env.setOpponentElixirMultiplier(oppMultiplier);
    env.setCurrentTick(startTick);
    env.setElixirForTeam(0, 0.0f);
    env.setElixirForTeam(1, 0.0f);

    const float before = env.getElixirForTeam(team);
    env.stepSelfPlayFast(NOOP, 0.0f, 0.0f, NOOP, 0.0f, 0.0f, ticks);
    return env.getElixirForTeam(team) - before;
}

}  // namespace

int main() {
    std::printf("elixir phase audit -- boundaries %d (2:00) / %d (3:00), 10 ticks = 1 s\n\n",
                GameManager::DOUBLE_ELIXIR_TICK, GameManager::TRIPLE_ELIXIR_TICK);

    // ---- 1. the schedule itself, as a pure function -----------------------
    std::printf("schedule (pure function of the tick)\n");
    checkNear(GameManager::elixirMultiplierAtTick(0),    1.0f, 1e-6f, "tick 0 -> 1x");
    checkNear(GameManager::elixirMultiplierAtTick(1199), 1.0f, 1e-6f, "tick 1199 (1:59.9) -> 1x");
    checkNear(GameManager::elixirMultiplierAtTick(1200), 2.0f, 1e-6f, "tick 1200 (2:00) -> 2x");
    checkNear(GameManager::elixirMultiplierAtTick(1799), 2.0f, 1e-6f, "tick 1799 (2:59.9) -> 2x");
    checkNear(GameManager::elixirMultiplierAtTick(1800), 3.0f, 1e-6f, "tick 1800 (3:00) -> 3x");
    checkNear(GameManager::elixirMultiplierAtTick(3600), 3.0f, 1e-6f, "tick 3600 (6:00) -> 3x");

    // --- 2. measured income, the half that can fail ---
    // 20 ticks = 2 s; at 0.035/tick that is 0.7 elixir at 1x.
    std::printf("\nmeasured income over 20 ticks (2 s), read off team 0's bar\n");
    checkNear(incomeOver(0,    20, 0, 1.0f), 0.7f, 0.01f, "single elixir  (from 0:00)");
    checkNear(incomeOver(1200, 20, 0, 1.0f), 1.4f, 0.01f, "double elixir  (from 2:00)");
    checkNear(incomeOver(1800, 20, 0, 1.0f), 2.1f, 0.01f, "triple elixir  (from 3:00)");

    // The boundary crossed mid-window, which pins per-tick evaluation.
    // GameManager::step increments the tick before reading the phase, so
    // stepping N from T covers T+1..T+N:
    //
    //     from 1190, 20 ticks -> 1191..1210
    //       1191..1199 =  9 ticks @ 1x = 0.315
    //       1200..1210 = 11 ticks @ 2x = 0.770
    //                                    -----
    //                                    1.085
    //
    // Tick 1200 is 2:00 and pays double; the naive 1.05 would assert it pays
    // single.
    checkNear(incomeOver(1190, 20, 0, 1.0f), 1.085f, 0.01f, "crossing 2:00 mid-window (9 @ 1x + 11 @ 2x)");

    // --- 3. composition with the curriculum handicap ---
    // The phase scales the base rate and oppElixirMultiplier scales that: a
    // 1.5x opponent in double elixir gets 3x.
    std::printf("\ncomposition with the curriculum's oppElixirMultiplier (team 1)\n");
    checkNear(incomeOver(0,    20, 1, 1.5f), 1.05f, 0.01f, "1.5x opponent, single -> 1.5x");
    checkNear(incomeOver(1200, 20, 1, 1.5f), 2.10f, 0.01f, "1.5x opponent, double -> 3.0x");
    // ...and team 0 is unaffected by the handicap in every phase.
    checkNear(incomeOver(1200, 20, 0, 1.5f), 1.40f, 0.01f, "1.5x opponent leaves team 0 at 2x");

    // ---- 4. the observation ----------------------------------------------
    std::vector<int> deck = { 15, 6, 25, 40, 24, 72, 33, 7 };
    ClashEnv env(deck, deck);
    env.reset();

    const int phaseIdx = ClashEnv::EXTRA_SCALARS_START + 9;
    std::printf("\nobservation layout\n");
    check(ClashEnv::NUM_EXTRA_SCALARS == 10, "NUM_EXTRA_SCALARS == 10");
    check(ClashEnv::CYCLE_START == ClashEnv::EXTRA_SCALARS_START + 10,
          "CYCLE_START == EXTRA_SCALARS_START + 10 (moved by exactly one)");
    std::printf("       observation_size() = %d,  CYCLE_START = %d\n",
                env.observationSize(), ClashEnv::CYCLE_START);

    std::printf("\nphase scalar at index EXTRA_SCALARS_START + 9\n");
    for (int tick : { 0, 1200, 1800 }) {
        env.setCurrentTick(tick);
        const std::vector<float> obs0 = env.getObservationForTeam(0);
        const std::vector<float> obs1 = env.getObservationForTeam(1);
        const float want = GameManager::elixirMultiplierAtTick(tick) / 3.0f;
        char label[96];
        std::snprintf(label, sizeof(label), "tick %4d, team 0", tick);
        checkNear(obs0[phaseIdx], want, 1e-5f, label);
        // Symmetric: team 1 reads the identical value. The tower block before
        // it is mirrored; this one must not be.
        std::snprintf(label, sizeof(label), "tick %4d, team 1 (must MATCH team 0)", tick);
        checkNear(obs1[phaseIdx], want, 1e-5f, label);
    }

    // The size equals the vector's actual length.
    check(static_cast<int>(env.getObservationForTeam(0).size()) == env.observationSize(),
          "built vector length == observationSize()");

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASS",
                failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
