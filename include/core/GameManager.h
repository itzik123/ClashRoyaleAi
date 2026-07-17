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
    const float BOARD_MAX_X = 17.0f;
    const float BOARD_MAX_Y = 31.0f;

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

    bool isValidPlacement(int team, float x, float y, bool isSpell) const {
        if (x < 0.0f || x > BOARD_MAX_X || y < 0.0f || y > BOARD_MAX_Y) return false;

        if (!isSpell) {
            if (team == 0 && y > 14.5f) return false;
            if (team == 1 && y < 17.5f) return false;
            
            // Prevent overlapping buildings
            for (const auto& entity : board.getEntities()) {
                float r = entity->getCollisionRadius();
                if (entity->isAlive() && r > 0.0f) {
                    float dx = x - entity->position.x;
                    float dy = y - entity->position.y;
                    float requiredDist = 1.0f + r; // 1.0f is default building placement radius
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

        if (!isValidPlacement(team, x, y, cardDef->isSpell)) return false;

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

        int deadKing = MatchRules::getDeadKingTeam(board);
        if (deadKing != -1) {
            gameOver = true;
            loserTeam = deadKing;
        }

        board.cleanDeadEntities();
    }
};