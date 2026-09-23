#pragma once
#include <algorithm>
#include <limits>

#include "Board.h"

class MatchRules {
public:
    struct Outcome {
        bool over;
        // -1 while ongoing, or for a draw (both Kings down the same tick).
        int loserTeam;
    };

    // The end of regulation: 1800 ticks = 3:00, the real schedule. The same
    // instant as GameManager::TRIPLE_ELIXIR_TICK, one fact with two
    // consequences; GameManager static_asserts they agree, since this header
    // knows nothing about elixir.
    static constexpr int REGULATION_END_TICK = 1800;

    // Surviving towers per side, and the weakest one's hp. Uses isTower(),
    // never the 'R' symbol, which Mortar (card 93) shares.
    struct TowerCensus {
        int alive[2] = { 0, 0 };
        int weakestHp[2] = { std::numeric_limits<int>::max(),
                             std::numeric_limits<int>::max() };
    };

    static TowerCensus census(const Board& board) {
        TowerCensus c;
        for (const auto& entity : board.getEntities()) {
            if (!entity->isAlive() || !entity->isTower()) continue;
            const int team = entity->team;
            if (team != 0 && team != 1) continue;
            c.alive[team]++;
            c.weakestHp[team] = std::min(c.weakestHp[team], entity->hp);
        }
        return c;
    }

    // A King dying ends the match; both dying the same tick is a draw.
    static Outcome evaluate(const Board& board) {
        bool team0KingAlive = false;
        bool team1KingAlive = false;

        for (const auto& entity : board.getEntities()) {
            // isTower() first: 'R' is a renderer symbol that card 93 (Mortar)
            // also uses, so a living Mortar would report its owner's King as
            // alive.
            if (entity->isTower() && entity->symbol == 'R' && entity->isAlive()) {
                if (entity->team == 0) team0KingAlive = true;
                if (entity->team == 1) team1KingAlive = true;
            }
        }

        if (team0KingAlive && team1KingAlive) return { false, -1 };
        if (!team0KingAlive && !team1KingAlive) return { true, -1 }; // draw
        return { true, team0KingAlive ? 1 : 0 };
    }

    // The per-tick rule, which GameManager::step calls. The real game ends a
    // match four ways:
    //
    //   1. a King Tower falls                   -> immediate win
    //   2. regulation ends with a crown lead    -> the leader wins at 3:00
    //   3. overtime, first crown taken          -> immediate win
    //   4. overtime expires                     -> tower-hp tiebreak
    //
    // 2 and 3 are one predicate: from REGULATION_END_TICK on, any difference in
    // surviving tower count ends the match. Rule 4 is TimeoutRules, called once
    // at maxTicks from ClashEnv::calculateReward.
    static Outcome evaluateAtTick(const Board& board, int tick) {
        const Outcome king = evaluate(board);
        if (king.over) return king;
        if (tick < REGULATION_END_TICK) return { false, -1 };

        const TowerCensus c = census(board);
        if (c.alive[0] != c.alive[1]) {
            return { true, c.alive[0] < c.alive[1] ? 0 : 1 };
        }
        return { false, -1 };
    }
};
