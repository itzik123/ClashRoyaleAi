#pragma once
#include <algorithm>
#include <limits>

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

    // The end of REGULATION, in ticks. 1800 = 3:00, the real game's schedule.
    //
    // The same instant as GameManager::TRIPLE_ELIXIR_TICK, and that is not a
    // coincidence to be factored away: in the real game overtime and triple
    // elixir both begin when regulation ends, so it is ONE fact with two
    // consequences. GameManager static_asserts the two agree rather than
    // deriving one from the other, because the dependency only runs one way
    // (this header knows nothing about elixir) and a static_assert gives the
    // no-second-copies guarantee without the coupling.
    static constexpr int REGULATION_END_TICK = 1800;

    // Surviving towers per side, and the weakest one's hp.
    //
    // `isTower()` and not a dynamic_cast: it is a virtual on Entity, so it is
    // exactly as robust as the cast TimeoutRules used to do for itself, needs
    // no Tower.h here, and is what evaluate() below already trusts. It is
    // emphatically not a symbol check -- 'R' is a RENDERER symbol shared with
    // card 93 (Mortar), which is the bug the comment in evaluate() records.
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

    // The per-tick rule, tick-aware. THIS is what GameManager::step calls.
    //
    // Real Clash Royale ends a match four ways, and until 2026-09-06 this
    // engine modelled only the first and the last:
    //
    //   1. a King Tower falls                      -> immediate win
    //   2. regulation ends with a crown lead       -> the leader wins AT 3:00
    //   3. overtime, first crown taken             -> immediate win
    //   4. overtime expires                        -> tower-hp tiebreak
    //
    // 2 and 3 are ONE line of code, and seeing that is the whole design: from
    // REGULATION_END_TICK onward, any difference in surviving tower count ends
    // the match. Before 3:00 that difference is a lead being held and the match
    // continues; after 3:00 it is either a lead carried across the boundary
    // (rule 2) or a crown just taken in sudden death (rule 3). One predicate,
    // both rules, no clock-edge special case to get wrong.
    //
    // Rule 4 stays in TimeoutRules, which owns the weakest-tower tiebreak and
    // is called once at maxTicks from ClashEnv::calculateReward.
    //
    // GAMEPLAY-AFFECTING, and substantially: mean match end was tick 1758, so a
    // large share of matches previously ran past 3:00 to the 3600 limit and are
    // now decided at 1800. Episodes get shorter (more of them per hour) and
    // holding a one-crown lead to 3:00 becomes a WIN rather than a position to
    // keep defending -- which is the real game's incentive and was not this
    // engine's. No win rate is comparable across this date.
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
