#pragma once
#include <algorithm>
#include <random>
#include <vector>

#include "ArenaLayout.h"
#include "Board.h"
#include "CardRegistry.h"
#include "GameManager.h"
#include "Tower.h"

// A competent replacement for the original built-in opponent.
//
// What it replaces and why. The previous opponentTurn() ran EVERY tick and, the
// moment it held 4 elixir, played a uniformly random affordable card at a
// uniformly random x in [2,15] with y hardcoded to 25 -- never at a bridge,
// never in response to anything, never holding elixir. Measured consequences
// after ~47,000 training episodes against it:
//
//   * the trained policy beat it 100% at 1.0x elixir using SIX of its eight
//     cards, never once playing its win condition or its spell
//   * an isolated probe put it at 13-0-2 against an ATTACKING scripted bot
//     while still never playing the win condition
//
// The reason is that an opponent which walks units into a defended lane on a
// timer makes pure defence genuinely optimal -- there is nothing to break
// through, so a win condition has no value. No reward shaping can hide that;
// it is a property of the opponent.
//
// This version does three things the old one did not: it DEFENDS (answers the
// deepest incursion into its own half), it ACCUMULATES (holds elixir instead of
// dumping at 4), and it ATTACKS AT A BRIDGE in a committed lane instead of a
// random column. It is still a heuristic, not a strong player -- the point is
// only that beating it should require actually taking a tower.
//
// Kept as its own class with GameManager passed in, so ClashEnv::opponentTurn()
// becomes a one-line delegate and GameManager/Board/CardRegistry are untouched.
class HeuristicOpponent {
public:
    // Re-rolled per match so the opponent does not always threaten the same
    // side. reset() is called from ClashEnv::reset().
    void reset(std::mt19937& rng) {
        std::uniform_int_distribution<int> laneDist(0, 1);
        lane = laneDist(rng);
        ticksSincePlay = ACTION_COOLDOWN_TICKS;   // may act immediately
    }

    void act(GameManager& game, std::mt19937& rng) {
        (void)rng; // not currently used by this heuristic's decision logic
        ++ticksSincePlay;
        if (ticksSincePlay < ACTION_COOLDOWN_TICKS) return;

        const std::vector<int>& hand = game.getHand(1);
        float elixir = game.getElixir(1);

        // (hand index, cost) for everything affordable right now.
        std::vector<std::pair<int, float>> playable;
        for (int i = 0; i < static_cast<int>(hand.size()); ++i) {
            const CardDefinition* card = CardRegistry::getInstance().getCard(hand[i]);
            if (card && elixir >= card->cost) playable.emplace_back(i, card->cost);
        }
        if (playable.empty()) return;

        const Board& board = game.getBoard();
        // Team 0 attacks toward HIGH y (its towers sit low, team 1's high), so
        // the most advanced enemy unit is the one with the largest y.
        bool threatFound = false;
        float threatX = 0.0f, threatY = 0.0f;
        for (const auto& entity : board.getEntities()) {
            if (!entity->isAlive() || entity->team != 0) continue;
            if (dynamic_cast<const Tower*>(entity.get()) != nullptr) continue;
            if (entity->position.y > threatY || !threatFound) {
                threatFound = true;
                threatX = entity->position.x;
                threatY = entity->position.y;
            }
        }

        // Own half only for non-spell placement, with a margin so the play is
        // never rejected by isValidPlacement for landing a hair over the line.
        const float ownHalfMinY = board.getRiverEnd() + 1.0f;

        int handIndex = -1;
        float targetX = 0.0f, targetY = 0.0f;

        if (threatFound && threatY > board.getRiverStart()) {
            // DEFEND: strongest affordable answer, met as far forward as legal.
            handIndex = std::max_element(playable.begin(), playable.end(),
                [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                    return a.second < b.second;
                })->first;
            targetX = threatX;
            targetY = std::max(threatY, ownHalfMinY);
        } else if (elixir >= PUSH_ELIXIR_THRESHOLD) {
            // ATTACK: commit to this match's lane, at the bridge rather than a
            // random column deep in its own half.
            handIndex = std::max_element(playable.begin(), playable.end(),
                [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                    return a.second < b.second;
                })->first;
            targetX = (lane == 0) ? LEFT_BRIDGE_X : RIGHT_BRIDGE_X;
            targetY = ownHalfMinY;
        } else {
            // HOLD -- accumulating is a real move, and the old bot never made it.
            return;
        }

        int cardId = hand[handIndex];
        const CardDefinition* card = CardRegistry::getInstance().getCard(cardId);
        if (!card) return;
        // Spells are exempt from the own-half restriction, so aim them at the
        // threat wherever it actually is.
        if (card->isSpell && threatFound) targetY = threatY;

        if (game.playCard(1, cardId, targetX, targetY)) {
            ticksSincePlay = 0;
        }
    }

private:
    // Bridge columns, read from ArenaLayout rather than restated.
    //
    // These were 3.5f / 13.5f and had ALREADY drifted before the 2026-08-21
    // arena correction: Board's own bridges were at 4.0 / 14.0 at the time, so
    // this bot had been aiming half a tile off its own engine's bridges for as
    // long as the constants existed. Nothing caught it because nothing compared
    // the two numbers -- which is the whole argument for not having two.
    static constexpr float LEFT_BRIDGE_X = ArenaLayout::LEFT_BRIDGE_X;
    static constexpr float RIGHT_BRIDGE_X = ArenaLayout::RIGHT_BRIDGE_X;
    // Elixir held before starting a push of its own. Above a single card's
    // cost on purpose: the old bot dumped at 4 and could never follow up.
    static constexpr float PUSH_ELIXIR_THRESHOLD = 7.0f;
    // Minimum ticks between plays. The old bot was invoked every tick and
    // played whenever it could afford anything, which is what produced the
    // endless stream of units walking into a defended lane.
    static constexpr int ACTION_COOLDOWN_TICKS = 25;

    int lane = 0;
    int ticksSincePlay = 0;
};
