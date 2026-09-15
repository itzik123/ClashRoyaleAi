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
    // THE RULE, separated from where the numbers came from.
    //
    // Exists because GameLogger has to reach the same verdict from its own tick
    // SNAPSHOTS (it is const and holds no Board), and web/viewer.html from the
    // replay JSON. Every previous consumer that could not call resolve() below
    // re-derived a verdict instead and got it wrong -- eight times across three
    // waves on the Python side, and the replay viewer was still doing it on
    // 2026-08-21, reporting "Draw" for a match with a Princess Tower at 90 hp.
    // Splitting the decision from the gathering is what lets a caller with a
    // different data source share the rule rather than copy it.
    //
    // weakestHp entries are int max for a side with no surviving towers, which
    // is unreachable in a live match (a King dying ends it) and is handled
    // consistently anyway: more towers wins.
    static MatchRules::Outcome decide(const int aliveCount[2], const int weakestHp[2]) {
        // 1. Fewer surviving towers loses.
        if (aliveCount[0] != aliveCount[1]) {
            return { true, aliveCount[0] < aliveCount[1] ? 0 : 1 };
        }
        // 2. Equal counts -> the lower weakest tower loses. ABSOLUTE hp, not a
        //    fraction of max: the real game breaks this tie on percentage, and
        //    King 4008 vs Princess 2534 means the two disagree often. Absolute
        //    is what the 2026-08-21 audit specified, recorded as a deliberate,
        //    known divergence rather than an oversight.
        if (weakestHp[0] != weakestHp[1]) {
            return { true, weakestHp[0] < weakestHp[1] ? 0 : 1 };
        }
        // 3. Genuine draw -- the only way to get one.
        return { true, -1 };
    }

    static MatchRules::Outcome resolve(const Board& board) {
        // The census moved to MatchRules on 2026-09-06, because the overtime
        // rule added there asks the same question every tick and two copies of
        // "count the surviving towers" is the duplication this repo keeps
        // paying for. `isTower()` replaced the dynamic_cast: it is the same
        // virtual dispatch, equally immune to the renderer symbols the old
        // comment here was guarding against, and it is what MatchRules::
        // evaluate already trusted one function away.
        const MatchRules::TowerCensus c = MatchRules::census(board);
        return decide(c.alive, c.weakestHp);
    }
};
