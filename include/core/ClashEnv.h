#pragma once
#include "GameManager.h"
#include "Building.h"
#include "BuildingTargeter.h"
#include "RangedTroop.h"
#include "GameLogger.h"
#include <vector>
#include <random>
#include <tuple>

struct StepResult {
    std::vector<float> observation;
    float reward;
    bool done;
};

// Historical/true self-play: both sides act each step instead of team 1 being
// driven by the built-in random opponentTurn(). observation1 is already from
// team 1's OWN point of view (see extractObservationForTeam) so the exact
// same network that plays team 0 elsewhere can also drive team 1 here.
// reward0 is from team 0's perspective (+1/-1/0, same convention as
// StepResult::reward); this is a zero-sum win/loss, so team 1's reward is
// just -reward0 -- not worth a redundant field.
struct SelfPlayStepResult {
    std::vector<float> observation0;
    std::vector<float> observation1;
    float reward0;
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
    // Spatial channels, per team: melee troops / ranged troops / building-targeters
    // (win-conditions like Hog, Giant, Golem) / buildings (towers + defensive).
    // Unit-TYPE visibility is what lets the net answer "what is attacking me and
    // with what should I respond" -- with HP-only channels a wounded PEKKA and a
    // Skeleton looked identical.
    // 0-3: ally melee, ranged, tank, buildings | 4-7: enemy same | 8: river/bridges
    static constexpr int NUM_CHANNELS = 9;
    static constexpr int HAND_SIZE = 4;
    // One-hot size for card identity in hand. Card ids run 0..40 in CardRegistry.
    static constexpr int NUM_CARD_IDS = 41;
    static constexpr float MAX_TROOP_HP = 4256.0f;
    static constexpr float MAX_BUILDING_HP = 4008.0f;

    // Generalized over which team the observation is FOR, so the same
    // network -- always trained believing it's "team 0" (self near low y,
    // enemy near high y, self always channels 0-3) -- can also drive team 1
    // in self-play by getting an observation from team 1's own point of view.
    // team==0 reproduces the exact previous behavior byte-for-byte.
    std::vector<float> extractObservationForTeam(int team) {
        int spatialSize = BOARD_WIDTH * BOARD_HEIGHT * NUM_CHANNELS;
        std::vector<float> obs(spatialSize, 0.0f);

        auto getIndex = [&](int channel, int y, int x) {
            return channel * (BOARD_HEIGHT * BOARD_WIDTH) + y * BOARD_WIDTH + x;
        };

        // River/bridge marker row -- x is already left/right symmetric (both
        // teams' towers and the bridge gaps sit at the same x coordinates),
        // so only the row itself needs mirroring for team 1.
        int riverRow = (team == 0) ? 16 : (BOARD_HEIGHT - 1 - 16);
        for (int x = 0; x < BOARD_WIDTH; ++x) {
            if ((x >= 3 && x <= 4) || (x >= 13 && x <= 14)) {
                obs[getIndex(8, riverRow, x)] = 1.0f;
            } else {
                obs[getIndex(8, riverRow, x)] = -1.0f;
            }
        }

        for (const auto& entity : game.getBoard().getEntities()) {
            if (!entity->isAlive()) continue;
            // Projectiles and pending spells are not board presence
            if (!entity->isTargetable()) continue;

            int x = static_cast<int>(entity->position.x);
            int rawY = static_cast<int>(entity->position.y);
            // Mirrored for team 1: physically team 1 sits near high y, but its
            // own network needs to see itself near low y (same layout it was
            // trained on as "team 0"), so flip before placing into the grid.
            int y = (team == 0) ? rawY : (BOARD_HEIGHT - 1 - rawY);

            if (x < 0 || x >= BOARD_WIDTH || y < 0 || y >= BOARD_HEIGHT) continue;

            // Type category: building / building-targeter (tank) / ranged / melee.
            // BuildingTargeter* also matches RangedBuildingTargeter (inheritance).
            int typeOffset;
            bool isBuilding = (dynamic_cast<Building*>(entity.get()) != nullptr);
            if (isBuilding) typeOffset = 3;
            else if (dynamic_cast<BuildingTargeter*>(entity.get()) != nullptr) typeOffset = 2;
            else if (dynamic_cast<RangedTroop*>(entity.get()) != nullptr) typeOffset = 1;
            else typeOffset = 0;

            float maxHp = isBuilding ? MAX_BUILDING_HP : MAX_TROOP_HP;
            float normalizedHp = std::min(static_cast<float>(entity->hp) / maxHp, 1.0f);
            // "Ally" = whichever team this observation is FOR, always channels
            // 0-3 -- not hardcoded to raw team 0 anymore.
            bool isAlly = (entity->team == team);
            int channel = (isAlly ? 0 : 4) + typeOffset;

            obs[getIndex(channel, y, x)] = normalizedHp;
        }

        obs.push_back(game.getElixir(team) / 10.0f);

        for (int cardId : game.getHand(team)) {
            const auto* card = CardRegistry::getInstance().getCard(cardId);
            obs.push_back(card ? card->cost / 10.0f : 0.0f);
        }

        // Card IDENTITY per hand slot (one-hot). Costs alone made a Hog Rider and a
        // Musketeer indistinguishable (both 0.4), so no card-specific strategy could
        // ever be learned.
        for (int cardId : game.getHand(team)) {
            for (int k = 0; k < NUM_CARD_IDS; ++k) {
                obs.push_back(k == cardId ? 1.0f : 0.0f);
            }
        }

        return obs;
    }

    std::vector<float> extractObservation() { return extractObservationForTeam(0); }

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
        return BOARD_WIDTH * BOARD_HEIGHT * NUM_CHANNELS   // spatial type/HP channels
             + 1                                            // elixir
             + HAND_SIZE                                    // card costs
             + HAND_SIZE * NUM_CARD_IDS;                    // card identity one-hots
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

    // team 1's own point of view (mirrored) -- see extractObservationForTeam.
    // Used by self-play: after reset()/step(), the trainer also needs an
    // observation to feed whatever policy (frozen historical snapshot, or a
    // second live network) is driving team 1 this step.
    std::vector<float> getObservationForTeam(int team) {
        return extractObservationForTeam(team);
    }

    // Self-play stepping: BOTH sides' actions are supplied externally instead
    // of team 1 being driven by the built-in random opponentTurn() (which
    // this does NOT call at all). targetY1 arrives in team 1's own local
    // frame -- the same mirrored frame its observation came in -- and is
    // converted back to real board Y before actually placing anything;
    // mirroring twice is the identity, so the same formula that produced the
    // observation also inverts it here.
    SelfPlayStepResult stepSelfPlay(int cardIndex0, float targetX0, float targetY0,
                                     int cardIndex1, float targetX1, float targetY1,
                                     int skipFrames = 10) {
        float totalReward = 0.0f;
        bool isDone = false;

        float realY1 = static_cast<float>(BOARD_HEIGHT - 1) - targetY1;

        for (int i = 0; i < skipFrames; ++i) {
            if (i == 0) {
                if (cardIndex0 >= 0 && cardIndex0 < 4) {
                    const auto& hand0 = game.getHand(0);
                    if (cardIndex0 < static_cast<int>(hand0.size())) {
                        game.playCard(0, hand0[cardIndex0], targetX0, targetY0);
                    }
                }
                if (cardIndex1 >= 0 && cardIndex1 < 4) {
                    const auto& hand1 = game.getHand(1);
                    if (cardIndex1 < static_cast<int>(hand1.size())) {
                        game.playCard(1, hand1[cardIndex1], targetX1, realY1);
                    }
                }
            }

            game.step();
            currentTick++;

            isDone = (currentTick >= maxTicks) || game.isGameOver();
            totalReward += calculateReward();

            logger.logTick(currentTick, game);

            if (isDone) break;
        }

        return { extractObservationForTeam(0), extractObservationForTeam(1), totalReward, isDone };
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

    // Thin pass-throughs to GameManager::getStatistics() -- plain ints (not
    // the JSON string) since these are read every training step across
    // several vectorized envs; avoiding a JSON round-trip on that hot path.
    // See MatchStatistics.h for what these actually measure.
    int getTroopDamageDealt(int team) const { return game.getStatistics().troopDamageDealt(team); }
    int getBuildingDamageDealt(int team) const { return game.getStatistics().buildingDamageDealt(team); }
    float getElixirSpent(int team) const { return game.getStatistics().elixirSpent(team); }
};