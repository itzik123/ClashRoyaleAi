#pragma once
#include "Board.h"
#include "PlayerState.h" 
#include "Tower.h"
#include "MatchRules.h"
#include <algorithm>
#include <vector>
#include <string>

class GameManager {
private:
    Board board;
    int currentTick;
    bool gameOver;
    int loserTeam;
    const float ELIXIR_REGEN_RATE = 0.035f;
    // Curriculum hook: scales the opponent's elixir regen relative to the base rate.
    // 1.0 = normal opponent, >1.0 = faster-elixir opponent for later training stages.
    float oppElixirMultiplier = 1.0f;
    // How far short of the river a non-spell placement must stay on the
    // caller's own side (Board itself only enforces the river during
    // movement/clamping, not placement).
    static constexpr float OWN_HALF_RIVER_BUFFER = 0.5f;

    std::vector<int> aiDeckConfig = { 0, 1, 2, 3, 4, 5, 6, 7 };
    std::vector<int> oppDeckConfig = { 0, 1, 2, 3, 4, 5, 6, 7 };

    void addTower(float x, float y, int hp, int team, float attackRange, int damage, int attackCooldown,
        char symbol, const std::string& towerName) {
        auto tower = std::make_shared<Tower>(board.allocateId(), x, y, hp, team, attackRange, damage, attackCooldown, symbol);
        tower->name = towerName;
        board.addEntity(tower);
    }

public:
    PlayerState playerAI;
    PlayerState playerOpponent;

    GameManager(const std::vector<int>& aiDeck, const std::vector<int>& opponentDeck)
        : gameOver(false), loserTeam(-1) {
        aiDeckConfig = aiDeck;
        oppDeckConfig = opponentDeck;
        reset();
    }

    void setOpponentDeck(const std::vector<int>& deck) {
        oppDeckConfig = deck;
    }

    void setOpponentElixirMultiplier(float multiplier) {
        oppElixirMultiplier = std::max(0.0f, multiplier);
    }

    Board& getBoard() { return board; }
    const Board& getBoard() const { return board; }

    float getElixirAI() const { return playerAI.elixir; }
    float getElixirOpp() const { return playerOpponent.elixir; }

    bool isGameOver() const { return gameOver; }
    int getLoserTeam() const { return loserTeam; }

    const std::vector<int>& getHand(int team) const {
        return (team == 0) ? playerAI.hand : playerOpponent.hand;
    }

    float getElixir(int team) const {
        return (team == 0) ? playerAI.elixir : playerOpponent.elixir;
    }

    bool isValidPlacement(int team, float x, float y, bool isSpell, float placedRadius) const {
        float maxX = static_cast<float>(board.getWidth() - 1);
        float maxY = static_cast<float>(board.getHeight() - 1);
        if (x < 0.0f || x > maxX || y < 0.0f || y > maxY) return false;

        if (!isSpell) {
            if (team == 0 && y > board.getRiverStart() - OWN_HALF_RIVER_BUFFER) return false;
            if (team == 1 && y < board.getRiverEnd() + OWN_HALF_RIVER_BUFFER) return false;

            // Prevent placing on top of an existing building -- Clash Royale
            // forbids this outright regardless of what's being placed, so the
            // required gap is the building's own radius plus whatever
            // footprint the new card will actually spawn with.
            for (const auto& entity : board.getEntities()) {
                float r = entity->getCollisionRadius();
                if (entity->isAlive() && r > 0.0f) {
                    float dx = x - entity->position.x;
                    float dy = y - entity->position.y;
                    float requiredDist = placedRadius + r;
                    if (dx*dx + dy*dy < requiredDist*requiredDist) return false;
                }
            }
        }
        return true;
    }

    bool playCard(int team, int targetCardId, float x, float y) {
        if (gameOver) return false;

        PlayerState& player = (team == 0) ? playerAI : playerOpponent;

        int handIndex = -1;
        for (int i = 0; i < static_cast<int>(player.hand.size()); ++i) {
            if (player.hand[i] == targetCardId) {
                handIndex = i;
                break;
            }
        }

        if (handIndex == -1) return false;

        const CardDefinition* cardDef = CardRegistry::getInstance().getCard(targetCardId);
        if (!cardDef) return false;

        if (!isValidPlacement(team, x, y, cardDef->isSpell, cardDef->placementRadius)) return false;

        int cardId = player.playCard(handIndex);
        if (cardId != -1) {
            cardDef->spawnEntity(x, y, team, board);
            return true;
        }
        return false;
    }

    void reset() {
        board = Board();
        currentTick = 0;
        gameOver = false;
        loserTeam = -1;

        playerAI.initializeDeck(aiDeckConfig);
        playerOpponent.initializeDeck(oppDeckConfig);

        addTower(9.0f, 2.0f, 4008, 0, 7.0f, 90, 10, 'R', "King Tower");
        addTower(9.0f, 30.0f, 4008, 1, 7.0f, 90, 10, 'R', "King Tower");

        addTower(3.0f, 5.0f, 2534, 0, 7.5f, 90, 8, 'P', "Princess Tower");
        addTower(14.0f, 5.0f, 2534, 0, 7.5f, 90, 8, 'P', "Princess Tower");
        addTower(3.0f, 27.0f, 2534, 1, 7.5f, 90, 8, 'P', "Princess Tower");
        addTower(14.0f, 27.0f, 2534, 1, 7.5f, 90, 8, 'P', "Princess Tower");

        board.commitPendingEntities();
    }

    void step() {
        if (gameOver) return;

        currentTick++;

        playerAI.elixir = std::min(playerAI.elixir + ELIXIR_REGEN_RATE, 10.0f);
        playerOpponent.elixir = std::min(playerOpponent.elixir + ELIXIR_REGEN_RATE * oppElixirMultiplier, 10.0f);

        board.commitPendingEntities();

        for (auto& entity : board.getEntities()) {
            if (entity->isAlive()) entity->update(board);
        }

        board.commitPendingEntities();
        board.resolveCollisions();

        MatchRules::Outcome outcome = MatchRules::evaluate(board);
        if (outcome.over) {
            gameOver = true;
            loserTeam = outcome.loserTeam;
        }

        board.cleanDeadEntities();
    }
};