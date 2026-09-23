#pragma once
#include <algorithm>
#include <limits>

#include "Board.h"
#include "MatchRules.h"
#include "Tower.h"

// How a match that reaches the tick limit with both Kings standing is decided:
//
//   1. The side with fewer surviving towers loses.
//   2. On equal counts, the side whose weakest surviving tower has less hp loses.
//   3. Only an exact tie on both is a draw.
//
// A separate class from MatchRules::evaluate, which answers "has a King died?"
// every tick. The integration point is ClashEnv::calculateReward.
class TimeoutRules {
public:
    // The rule, separated from where the numbers come from, so consumers
    // without a Board (GameLogger's tick snapshots; web/viewer.html via the
    // replay JSON) share it instead of re-deriving it.
    //
    // weakestHp is int max for a side with no surviving towers, unreachable in
    // a live match; more towers still wins.
    static MatchRules::Outcome decide(const int aliveCount[2], const int weakestHp[2]) {
        // 1. Fewer surviving towers loses.
        if (aliveCount[0] != aliveCount[1]) {
            return { true, aliveCount[0] < aliveCount[1] ? 0 : 1 };
        }
        // 2. Equal counts: the lower weakest tower loses, by ABSOLUTE hp. The
        //    real game compares percentages, and with King 4008 vs Princess
        //    2534 the two often disagree; absolute is a deliberate divergence.
        if (weakestHp[0] != weakestHp[1]) {
            return { true, weakestHp[0] < weakestHp[1] ? 0 : 1 };
        }
        // 3. The only way to a draw.
        return { true, -1 };
    }

    static MatchRules::Outcome resolve(const Board& board) {
        // The tower census is shared with MatchRules' overtime rule.
        const MatchRules::TowerCensus c = MatchRules::census(board);
        return decide(c.alive, c.weakestHp);
    }
};
