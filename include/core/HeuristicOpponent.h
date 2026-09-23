#pragma once
#include <algorithm>
#include <random>
#include <vector>

#include "ArenaLayout.h"
#include "Board.h"
#include "CardRegistry.h"
#include "GameManager.h"
#include "Tower.h"

// The C++ built-in opponent. It defends (answers the deepest incursion into its
// half), accumulates elixir, and attacks at a bridge in a committed lane. Still
// a heuristic, but beating it requires taking a tower. ClashEnv::opponentTurn
// delegates here.
class HeuristicOpponent {
public:
    // Re-rolled per match so the threatened lane varies; called from
    // ClashEnv::reset().
    void reset(std::mt19937& rng) {
        std::uniform_int_distribution<int> laneDist(0, 1);
        lane = laneDist(rng);
        ticksSincePlay = ACTION_COOLDOWN_TICKS;   // may act immediately
    }

    void act(GameManager& game, std::mt19937& rng) {
        (void)rng; // unused
        ++ticksSincePlay;
        if (ticksSincePlay < ACTION_COOLDOWN_TICKS) return;

        const std::vector<int>& hand = game.getHand(1);
        float elixir = game.getElixir(1);

        // (hand index, cost) for everything affordable.
        std::vector<std::pair<int, float>> playable;
        for (int i = 0; i < static_cast<int>(hand.size()); ++i) {
            const CardDefinition* card = CardRegistry::getInstance().getCard(hand[i]);
            if (card && elixir >= card->cost) playable.emplace_back(i, card->cost);
        }
        if (playable.empty()) return;

        const Board& board = game.getBoard();
        // Team 0 attacks toward high y, so the most advanced enemy unit has the
        // largest y.
        bool threatFound = false;
        float threatX = 0.0f, threatY = 0.0f;
        for (const auto& entity : board.getEntities()) {
            if (!entity->isAlive() || entity->team != 0) continue;
            // Only targetable units count: a pending spell or a projectile in
            // flight is not an incursion, and a cloaked unit is invisible to
            // the bot too.
            if (!entity->isTargetable()) continue;
            if (dynamic_cast<const Tower*>(entity.get()) != nullptr) continue;
            if (entity->position.y > threatY || !threatFound) {
                threatFound = true;
                threatX = entity->position.x;
                threatY = entity->position.y;
            }
        }

        // Own half only for non-spells, with a margin so isValidPlacement never
        // rejects the play.
        const float ownHalfMinY = board.getRiverEnd() + 1.0f;

        int handIndex = -1;
        float targetX = 0.0f, targetY = 0.0f;

        if (threatFound && threatY > board.getRiverStart()) {
            // DEFEND: the strongest affordable answer, as far forward as legal.
            handIndex = std::max_element(playable.begin(), playable.end(),
                [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                    return a.second < b.second;
                })->first;
            targetX = threatX;
            targetY = std::max(threatY, ownHalfMinY);
        } else if (elixir >= PUSH_ELIXIR_THRESHOLD) {
            // ATTACK: this match's lane, at the bridge.
            handIndex = std::max_element(playable.begin(), playable.end(),
                [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                    return a.second < b.second;
                })->first;
            targetX = (lane == 0) ? LEFT_BRIDGE_X : RIGHT_BRIDGE_X;
            targetY = ownHalfMinY;
        } else {
            // HOLD.
            return;
        }

        int cardId = hand[handIndex];
        const CardDefinition* card = CardRegistry::getInstance().getCard(cardId);
        if (!card) return;
        // Spells may go anywhere, so aim them at the threat.
        if (card->isSpell && threatFound) targetY = threatY;

        if (game.playCard(1, cardId, targetX, targetY)) {
            ticksSincePlay = 0;
        }
    }

private:
    // Bridge columns, from ArenaLayout.
    static constexpr float LEFT_BRIDGE_X = ArenaLayout::LEFT_BRIDGE_X;
    static constexpr float RIGHT_BRIDGE_X = ArenaLayout::RIGHT_BRIDGE_X;
    // Elixir held before starting a push; above one card's cost so the push has
    // a follow-up.
    static constexpr float PUSH_ELIXIR_THRESHOLD = 7.0f;
    // Minimum ticks between plays, so the bot does not feed a steady stream of
    // units into a defended lane.
    static constexpr int ACTION_COOLDOWN_TICKS = 25;

    int lane = 0;
    int ticksSincePlay = 0;
};
