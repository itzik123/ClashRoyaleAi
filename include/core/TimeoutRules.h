#pragma once
#include <algorithm>
#include <limits>

#include "Board.h"
#include "MatchRules.h"
#include "Tower.h"

// How a match that reaches the tick limit with BOTH King Towers still standing
// gets decided.
//
// Until this existed, every timed-out match was scored as a draw: ClashEnv::
// calculateReward() returned 0.0 whenever GameManager::isGameOver() was false,
// and hitting maxTicks never sets that flag. Real Clash Royale never does this
// -- a timed-out match is decided on towers, and only an exact tie on every
// criterion is a genuine draw.
//
// The rules applied here, in order:
//   1. Tower count. The side with FEWER surviving towers has lost more of them
//      and loses the match. (Equivalently: more crowns taken wins.)
//   2. Weakest tower. On equal counts, the side whose lowest-HP surviving tower
//      is lower loses -- the real game's tiebreak, where being closer to losing
//      another tower decides it.
//   3. Only if both are exactly equal is the result a true draw.
//
// Deliberately a SEPARATE rule class rather than another branch inside
// MatchRules::evaluate(): that function answers a different question ("has a
// King died yet?"), is called every single tick from GameManager::step(), and
// is correct as-is. Extending it would mean touching the hot path and the one
// piece of match logic every existing test already pins down. This adds the new
// behavior alongside it instead -- MatchRules, GameManager, Board and Tower are
// all unmodified; the only integration point is the single call site in
// ClashEnv::calculateReward().
class TimeoutRules {
public:
    // Board is const& and only read -- this decides an outcome, it never
    // mutates match state.
    static MatchRules::Outcome resolve(const Board& board) {
        int aliveCount[2] = { 0, 0 };
        int weakestHp[2] = { std::numeric_limits<int>::max(),
                             std::numeric_limits<int>::max() };

        for (const auto& entity : board.getEntities()) {
            if (!entity->isAlive()) continue;
            // Towers only -- Cannon/Tombstone and other player-placed
            // buildings are not crowns and must not count toward either
            // criterion. dynamic_cast rather than a symbol check so this
            // keeps working if tower symbols are ever changed for the
            // renderer (King is 'R', Princess 'P' today).
            if (dynamic_cast<const Tower*>(entity.get()) == nullptr) continue;

            int team = entity->team;
            if (team != 0 && team != 1) continue;

            aliveCount[team]++;
            weakestHp[team] = std::min(weakestHp[team], entity->hp);
        }

        // 1. Fewer surviving towers loses.
        if (aliveCount[0] != aliveCount[1]) {
            return { true, aliveCount[0] < aliveCount[1] ? 0 : 1 };
        }
        // 2. Equal counts -> the lower weakest tower loses.
        if (weakestHp[0] != weakestHp[1]) {
            return { true, weakestHp[0] < weakestHp[1] ? 0 : 1 };
        }
        // 3. Genuine draw -- the only way to get one.
        return { true, -1 };
    }
};
