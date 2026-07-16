#pragma once
#include "Board.h"
#include "PlayerState.h" 
#include "Tower.h"
#include "MatchRules.h"
#include <algorithm>
#include <vector>

class GameManager {
private:
    Board board;
    int currentTick;
    bool gameOver;
    int loserTeam;
    const float ELIXIR_REGEN_RATE = 0.035f;
    const float BOARD_MAX_X = 17.0f;
    const float BOARD_MAX_Y = 31.0f;

    std::vector<int> aiDeckConfig = { 0, 1, 2, 3, 4, 5, 6, 7 };
    std::vector<int> oppDeckConfig = { 0, 1, 2, 3, 4, 5, 6, 7 };

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

        board.addEntity(std::make_shared<Tower>(board.allocateId(), 9.0f, 2.0f, 4008, 0, 7.0f, 90, 10, 'R'));
        board.addEntity(std::make_shared<Tower>(board.allocateId(), 9.0f, 30.0f, 4008, 1, 7.0f, 90, 10, 'R'));

        board.addEntity(std::make_shared<Tower>(board.allocateId(), 3.0f, 5.0f, 2534, 0, 7.5f, 90, 8, 'P'));
        board.addEntity(std::make_shared<Tower>(board.allocateId(), 14.0f, 5.0f, 2534, 0, 7.5f, 90, 8, 'P'));
        board.addEntity(std::make_shared<Tower>(board.allocateId(), 3.0f, 27.0f, 2534, 1, 7.5f, 90, 8, 'P'));
        board.addEntity(std::make_shared<Tower>(board.allocateId(), 14.0f, 27.0f, 2534, 1, 7.5f, 90, 8, 'P'));

        board.commitPendingEntities();
    }

    void step() {
        if (gameOver) return;

        currentTick++;

        playerAI.elixir = std::min(playerAI.elixir + ELIXIR_REGEN_RATE, 10.0f);
        playerOpponent.elixir = std::min(playerOpponent.elixir + ELIXIR_REGEN_RATE, 10.0f);

        board.commitPendingEntities();

        for (auto& entity : board.getEntities()) {
            if (entity->isAlive()) entity->update(board);
        }

        board.commitPendingEntities();

        const auto& ents = board.getEntities();
        for (size_t i = 0; i < ents.size(); ++i) {
            for (size_t j = i + 1; j < ents.size(); ++j) {
                auto& e1 = ents[i];
                auto& e2 = ents[j];

                if (!e1->isAlive() || !e2->isAlive()) continue;

                float r1 = e1->getCollisionRadius();
                float r2 = e2->getCollisionRadius();

                bool isBuilding1 = r1 > 0.0f;
                bool isBuilding2 = r2 > 0.0f;
                bool isTroop1 = !isBuilding1 && e1->isTargetable();
                bool isTroop2 = !isBuilding2 && e2->isTargetable();

                if (isTroop1 && isTroop2) {
                    float dx = e1->position.x - e2->position.x;
                    float dy = e1->position.y - e2->position.y;
                    float dist = std::sqrt(dx * dx + dy * dy);
                    float minRadius = 0.8f;

                    if (dist < minRadius) {
                        if (dist < 0.001f) { dx = 1.0f; dy = 0.0f; dist = 1.0f; }
                        float overlap = minRadius - dist;
                        float pushX = (dx / dist) * overlap * 0.5f;
                        float pushY = (dy / dist) * overlap * 0.5f;
                        
                        // Add tiny orthogonal noise to prevent jitter locking
                        float noise = 0.01f;
                        pushX += (dy / dist) * noise;
                        pushY -= (dx / dist) * noise;

                        e1->position.x += pushX;
                        e1->position.y += pushY;
                        e2->position.x -= pushX;
                        e2->position.y -= pushY;
                    }
                }

                if (isTroop1 && isBuilding2) {
                    float dx = e1->position.x - e2->position.x;
                    float dy = e1->position.y - e2->position.y;
                    float dist = std::sqrt(dx * dx + dy * dy);
                    float minDist = r2 + 0.4f;

                    if (dist < minDist) {
                        if (dist < 0.001f) { dx = 1.0f; dy = 0.0f; dist = 1.0f; }
                        float push = minDist - dist;
                        e1->position.x += (dx / dist) * push + (dy / dist) * 0.05f;
                        e1->position.y += (dy / dist) * push - (dx / dist) * 0.05f;
                    }
                }
                if (isTroop2 && isBuilding1) {
                    float dx = e2->position.x - e1->position.x;
                    float dy = e2->position.y - e1->position.y;
                    float dist = std::sqrt(dx * dx + dy * dy);
                    float minDist = r1 + 0.4f;

                    if (dist < minDist) {
                        if (dist < 0.001f) { dx = 1.0f; dy = 0.0f; dist = 1.0f; }
                        float push = minDist - dist;
                        e2->position.x += (dx / dist) * push + (dy / dist) * 0.05f;
                        e2->position.y += (dy / dist) * push - (dx / dist) * 0.05f;
                    }
                }
            }
        }

        // Re-clamp after collision resolution, which can push a troop back
        // into the river or off the board edge. Delegates to each entity's
        // own clampPosition() (a no-op for anything that isn't a Troop) so
        // this respects riverIgnores instead of re-deriving the rule here.
        for (auto& entity : board.getEntities()) {
            if (entity->isAlive()) {
                entity->clampPosition(board);
            }
        }

        int deadKing = MatchRules::getDeadKingTeam(board);
        if (deadKing != -1) {
            gameOver = true;
            loserTeam = deadKing;
        }

        board.cleanDeadEntities();
    }
};