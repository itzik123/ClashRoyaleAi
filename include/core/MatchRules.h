#pragma once
#include "Board.h"

class MatchRules {
public:
    // Returns the team ID that lost due to their King Tower dying, or -1 if neither is dead.
    static int getDeadKingTeam(const Board& board) {
        bool team0KingAlive = false;
        bool team1KingAlive = false;
        
        for (const auto& entity : board.getEntities()) {
            if (entity->symbol == 'R' && entity->isAlive()) {
                if (entity->team == 0) team0KingAlive = true;
                if (entity->team == 1) team1KingAlive = true;
            }
        }
        
        if (!team0KingAlive) return 0; // Team 0 lost
        if (!team1KingAlive) return 1; // Team 1 lost
        return -1;
    }
};
