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
            // isTower() FIRST, and that guard is the whole point. 'R' is a
            // RENDERER symbol, not an identity: card id 93 (Mortar) is
            // registered with it too, so a living Mortar reported its owner's
            // King as alive and a match whose King had just fallen simply did
            // not end. Every other site asking this question already guards by
            // type -- Tower::update's Princess count uses
            // isTower() && symbol != 'R', and TimeoutRules::resolve uses a
            // dynamic_cast whose own comment says a symbol check would be
            // fragile. This was the one that did not.
            if (entity->isTower() && entity->symbol == 'R' && entity->isAlive()) {
                if (entity->team == 0) team0KingAlive = true;
                if (entity->team == 1) team1KingAlive = true;
            }
        }

        if (team0KingAlive && team1KingAlive) return { false, -1 };
        if (!team0KingAlive && !team1KingAlive) return { true, -1 }; // draw
        return { true, team0KingAlive ? 1 : 0 };
    }
};
