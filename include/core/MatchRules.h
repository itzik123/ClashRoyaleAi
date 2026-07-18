#pragma once
#include "Board.h"

class MatchRules {
public:
    struct Outcome {
        bool over;
        // -1 while the match is ongoing, or if it ended in a draw
        // (simultaneous double King Tower KO) -- ClashEnv::calculateReward
        // already treats loserTeam == -1 as "no winner" for reward
        // purposes, so this is the only state that needs distinguishing.
        int loserTeam;
    };

    // A single King Tower dying ends the match for the other side. Both
    // dying the same tick (each already critically low, both hit lethal
    // damage this tick) is a draw, not an arbitrary loss for team 0 --
    // the earlier version always returned 0 first and never noticed team 1
    // was also dead.
    static Outcome evaluate(const Board& board) {
        bool team0KingAlive = false;
        bool team1KingAlive = false;

        for (const auto& entity : board.getEntities()) {
            if (entity->symbol == 'R' && entity->isAlive()) {
                if (entity->team == 0) team0KingAlive = true;
                if (entity->team == 1) team1KingAlive = true;
            }
        }

        if (team0KingAlive && team1KingAlive) return { false, -1 };
        if (!team0KingAlive && !team1KingAlive) return { true, -1 }; // draw
        return { true, team0KingAlive ? 1 : 0 };
    }
};
