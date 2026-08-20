// Does a match that reaches the tick limit actually resolve? Runs matches to
// maxTicks and prints the surviving-tower state alongside the verdict the
// engine hands back, so "the rule is wrong" and "the rule is right but nothing
// reads it" can be told apart.
//
// Build:  powershell -File tools/audit/build.ps1 timeout_audit
#include <cstdio>
#include <vector>
#include <map>
#include <string>
#include <limits>
#include "ClashEnv.h"

static void describe(const Board& b, int* count, int* weakest) {
    count[0] = count[1] = 0;
    weakest[0] = weakest[1] = std::numeric_limits<int>::max();
    for (const auto& e : b.getEntities()) {
        if (!e->isAlive() || !e->isTower()) continue;
        count[e->team]++;
        if (e->hp < weakest[e->team]) weakest[e->team] = e->hp;
    }
}

int main(int argc, char** argv) {
    int matches = (argc > 1) ? std::atoi(argv[1]) : 12;
    bool passive = (argc > 2 && std::string(argv[2]) == "passive");
    std::vector<int> deck = { 15, 6, 25, 40, 24, 72, 33, 7 };

    std::map<std::string, int> verdicts;
    int reachedLimit = 0;

    for (int m = 0; m < matches; ++m) {
        ClashEnv env(deck, deck, 3600);
        env.reset();
        float lastReward = 0.0f;
        bool done = false;
        int ticks = 0;
        while (!done) {
            // slot 4 == no-op. In `passive` mode BOTH sides no-op (stepSelfPlay
            // never runs the heuristic), which is the only reliable way to
            // force the tick limit; otherwise team 1's heuristic plays and the
            // match usually ends on a King.
            if (passive) {
                auto r = env.stepSelfPlay(4, 0.0f, 0.0f, 4, 0.0f, 0.0f, 10);
                lastReward = r.reward0; done = r.done;
            } else {
                auto r = env.step(4, 0.0f, 0.0f, 10);
                lastReward = r.reward; done = r.done;
            }
            ticks += 10;
        }

        int count[2], weakest[2];
        describe(env.debugGame().getBoard(), count, weakest);
        int loser = env.resolveTimeoutOutcome();
        bool timedOut = ticks >= 3600;
        if (timedOut) reachedLimit++;

        const char* verdict = (loser == -1) ? "DRAW" : (loser == 1 ? "team0 WINS" : "team1 WINS");
        std::printf("match %2d  ticks=%4d %-9s towers %d-%d  weakest %5d vs %5d  "
                    "resolveTimeoutOutcome=%-10s reward0=%+.1f\n",
                    m, ticks, timedOut ? "(TIMEOUT)" : "(king)", count[0], count[1],
                    weakest[0] == std::numeric_limits<int>::max() ? 0 : weakest[0],
                    weakest[1] == std::numeric_limits<int>::max() ? 0 : weakest[1],
                    verdict, lastReward);
        verdicts[std::string(verdict) + (timedOut ? " @timeout" : " @king")]++;
    }

    std::printf("\nreached the 3600-tick limit: %d of %d\n", reachedLimit, matches);
    for (const auto& kv : verdicts) std::printf("  %-24s %d\n", kv.first.c_str(), kv.second);
    return 0;
}
