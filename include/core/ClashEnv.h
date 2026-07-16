#pragma once
#include "GameManager.h"
#include "Building.h"
#include "GameLogger.h"
#include <vector>
#include <random>
#include <tuple>

struct StepResult {
    std::vector<float> observation;
    float reward;
    bool done;
};

class ClashEnv {
private:
    GameManager game;
    int maxTicks;
    int currentTick;
    std::mt19937 rng;
    GameLogger logger;

    static constexpr int BOARD_WIDTH = 18;
    static constexpr int BOARD_HEIGHT = 32;
    static constexpr int NUM_CHANNELS = 5;
    static constexpr float MAX_TROOP_HP = 4256.0f;
    static constexpr float MAX_BUILDING_HP = 4008.0f;

    std::vector<float> extractObservation() {
        int spatialSize = BOARD_WIDTH * BOARD_HEIGHT * NUM_CHANNELS;
        std::vector<float> obs(spatialSize, 0.0f);

        auto getIndex = [&](int channel, int y, int x) {
            return channel * (BOARD_HEIGHT * BOARD_WIDTH) + y * BOARD_WIDTH + x;
        };

        for (int x = 0; x < BOARD_WIDTH; ++x) {
            if ((x >= 3 && x <= 4) || (x >= 13 && x <= 14)) {
                obs[getIndex(4, 16, x)] = 1.0f;
            } else {
                obs[getIndex(4, 16, x)] = -1.0f;
            }
        }

        for (const auto& entity : game.getBoard().getEntities()) {
            if (!entity->isAlive()) continue;

            int x = static_cast<int>(entity->position.x);
            int y = static_cast<int>(entity->position.y);

            if (x < 0 || x >= BOARD_WIDTH || y < 0 || y >= BOARD_HEIGHT) continue;

            bool isBuilding = (dynamic_cast<Building*>(entity.get()) != nullptr);
            float maxHp = isBuilding ? MAX_BUILDING_HP : MAX_TROOP_HP;
            float normalizedHp = std::min(static_cast<float>(entity->hp) / maxHp, 1.0f);
            int channel = (entity->team == 0) ? (isBuilding ? 2 : 0) : (isBuilding ? 3 : 1);

            obs[getIndex(channel, y, x)] = normalizedHp;
        }

        obs.push_back(game.getElixir(0) / 10.0f);

        for (int cardId : game.getHand(0)) {
            const auto* card = CardRegistry::getInstance().getCard(cardId);
            obs.push_back(card ? card->cost / 10.0f : 0.0f);
        }

        return obs;
    }

    float calculateReward() {
        if (!game.isGameOver()) return 0.0f;
        int loser = game.getLoserTeam();
        if (loser == 1) return 1.0f;
        if (loser == 0) return -1.0f;
        return 0.0f;
    }

    void opponentTurn() {
        auto& hand = game.playerOpponent.hand;
        float elixir = game.playerOpponent.elixir;

        if (elixir < 4.0f) return;

        std::vector<int> playableIndices;
        for (int i = 0; i < static_cast<int>(hand.size()); ++i) {
            const auto* card = CardRegistry::getInstance().getCard(hand[i]);
            if (card && elixir >= card->cost) {
                playableIndices.push_back(i);
            }
        }

        if (playableIndices.empty()) return;

        std::uniform_int_distribution<int> indexDist(0, static_cast<int>(playableIndices.size()) - 1);
        int chosen = playableIndices[indexDist(rng)];
        int cardId = hand[chosen];

        const auto* card = CardRegistry::getInstance().getCard(cardId);
        if (!card) return;

        std::uniform_real_distribution<float> xDist(2.0f, 15.0f);
        float spawnX = xDist(rng);
        float spawnY = card->isSpell ? 10.0f : 25.0f;

        game.playCard(1, cardId, spawnX, spawnY);
    }

public:
    ClashEnv(const std::vector<int>& aiDeck, const std::vector<int>& oppDeck, int maxTicks = 3600)
        : game(aiDeck, oppDeck), maxTicks(maxTicks), currentTick(0),
          rng(std::random_device{}()) {}

    int observationSize() const {
        return BOARD_WIDTH * BOARD_HEIGHT * NUM_CHANNELS + 1 + 4;
    }

    std::vector<float> reset() {
        logger.clear();
        game.reset();
        logger.logTick(0, game);
        currentTick = 0;
        return extractObservation();
    }

    std::vector<int> getHand() const {
        return game.getHand(0);
    }

    float getElixir() const {
        return game.getElixir(0);
    }

    bool isGameOver() const {
        return game.isGameOver() || currentTick >= maxTicks;
    }

    StepResult step(int cardIndex, float targetX, float targetY, int skipFrames = 10) {
        float totalReward = 0.0f;
        bool isDone = false;

        for (int i = 0; i < skipFrames; ++i) {
            if (i == 0 && cardIndex >= 0 && cardIndex < 4) {
                const auto& hand = game.getHand(0);
                if (cardIndex < static_cast<int>(hand.size())) {
                    int cardId = hand[cardIndex];
                    game.playCard(0, cardId, targetX, targetY);
                }
            }

            opponentTurn();

            game.step();
            currentTick++;

            isDone = (currentTick >= maxTicks) || game.isGameOver();
            totalReward += calculateReward();

            logger.logTick(currentTick, game);

            if (isDone) break;
        }

        return { extractObservation(), totalReward, isDone };
    }

    void injectEnemy(int cardId, float x, float y) {
        const auto* card = CardRegistry::getInstance().getCard(cardId);
        if (card) {
            card->spawnEntity(x, y, 1, game.getBoard());
        }
    }

    void setOpponentDeck(const std::vector<int>& deck) {
        game.setOpponentDeck(deck);
    }

    void setOpponentElixirMultiplier(float multiplier) {
        game.setOpponentElixirMultiplier(multiplier);
    }

    void saveLog(const std::string& filepath) {
        logger.save(filepath);
    }
};