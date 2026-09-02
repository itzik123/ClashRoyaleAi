// Measures the real-game elixir phase schedule added 2026-09-02.
//
// Standalone for the reason every instrument in this directory is: it compiles
// straight against the header-only engine with cl.exe and cannot force a
// reconfigure of the solution the .pyd and the Catch2 suite build from. That
// matters more than usual here -- this change was authored while a training run
// held clash_royale_env.pyd mapped, so the ordinary build was unavailable and
// this file was the only way to compile the edited headers at all.
//
// It answers four questions the source cannot:
//
//   1. does income actually run 1x / 2x / 3x across the two boundaries
//   2. does the phase COMPOSE with oppElixirMultiplier rather than replace it
//   3. does the observation carry the phase, at the index NUM_EXTRA_SCALARS
//      says it does, normalised to 3.0
//   4. did CYCLE_START move by exactly one, and observationSize() with it
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

// Elixir gained by `team` over `ticks` ticks starting from `startTick`,
// measured by setting the clock and stepping. Reads the BAR, not the schedule,
// so it cannot pass by agreeing with the thing it is testing.
//
// Windows are short (20 ticks = 2 s, at most 2.1 elixir at 3x) for a reason
// that is easy to get wrong: PlayerState clamps the bar at 10.0, so a longer
// window saturates and every phase reports the same number. A measurement whose
// failure mode is "all arms agree" would look like the phases doing nothing.
//
// stepSelfPlayFast, never step: step() runs the HeuristicOpponent, which would
// SPEND team 1's elixir and make the income read low for reasons unrelated to
// regen. Same trap verify_pyd.py documents for navigation.
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

    // ---- 2. MEASURED income, which is the half that can actually fail -----
    // 20 ticks = 2 s. At the base 0.035/tick that is 0.7 elixir at 1x.
    std::printf("\nmeasured income over 20 ticks (2 s), read off team 0's bar\n");
    checkNear(incomeOver(0,    20, 0, 1.0f), 0.7f, 0.01f, "single elixir  (from 0:00)");
    checkNear(incomeOver(1200, 20, 0, 1.0f), 1.4f, 0.01f, "double elixir  (from 2:00)");
    checkNear(incomeOver(1800, 20, 0, 1.0f), 2.1f, 0.01f, "triple elixir  (from 3:00)");

    // The boundary crossed MID-WINDOW. A schedule sampled once per step()
    // rather than once per tick would report a flat 0.7 or 1.4 here and pass
    // every test above, so this is the case that pins per-tick evaluation.
    //
    // The expected value is 1.085, NOT the 1.05 an even 10/10 split suggests,
    // and the difference is worth stating because it is the kind of off-by-one
    // this codebase has been bitten by twice (the freeze decrement, the bridge
    // waypoint epsilon). GameManager::step does `currentTick++` BEFORE reading
    // the phase, so stepping N ticks from T processes ticks T+1 .. T+N:
    //
    //     from 1190, 20 ticks -> 1191..1210
    //       1191..1199 =  9 ticks @ 1x = 0.315
    //       1200..1210 = 11 ticks @ 2x = 0.770
    //                                    -----
    //                                    1.085
    //
    // Tick 1200 IS 2:00 and correctly pays double, which is what makes the
    // split 9/11 rather than 10/10. Asserting the naive 1.05 here would be
    // asserting that the first tick of double elixir pays single.
    checkNear(incomeOver(1190, 20, 0, 1.0f), 1.085f, 0.01f, "crossing 2:00 mid-window (9 @ 1x + 11 @ 2x)");

    // ---- 3. composition with the curriculum handicap ----------------------
    // The phase scales the BASE rate and oppElixirMultiplier scales that, so a
    // 1.5x opponent in double elixir must get 3x, not 2x and not 1.5x.
    std::printf("\ncomposition with the curriculum's oppElixirMultiplier (team 1)\n");
    checkNear(incomeOver(0,    20, 1, 1.5f), 1.05f, 0.01f, "1.5x opponent, single -> 1.5x");
    checkNear(incomeOver(1200, 20, 1, 1.5f), 2.10f, 0.01f, "1.5x opponent, double -> 3.0x");
    // ...and team 0 must be UNAFFECTED by the opponent handicap in every phase.
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
        // Symmetric: the phase is a property of the clock, so team 1 must read
        // the identical value. The tower block two slots earlier is mirrored;
        // this one deliberately is not, and a copy-paste of that mirroring
        // would be silent.
        std::snprintf(label, sizeof(label), "tick %4d, team 1 (must MATCH team 0)", tick);
        checkNear(obs1[phaseIdx], want, 1e-5f, label);
    }

    // Size must equal what the vector actually is, not what the constant says.
    check(static_cast<int>(env.getObservationForTeam(0).size()) == env.observationSize(),
          "built vector length == observationSize()");

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASS",
                failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
