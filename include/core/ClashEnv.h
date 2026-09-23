#pragma once
#include "GameManager.h"
#include "TimeoutRules.h"
#include "HeuristicOpponent.h"
#include "Building.h"
#include "BuildingTargeter.h"
#include "RangedTroop.h"
// Included explicitly: the attribute channels read Troop and CombatEntity
// fields directly.
#include "Troop.h"
#include "CombatEntity.h"
#include "GameLogger.h"
#include <vector>
#include <random>
#include <tuple>
#include <unordered_set>
#include <cmath>

struct StepResult {
    std::vector<float> observation;
    float reward;
    bool done;
};

// Self-play: both sides act each step. observation1 is from team 1's own
// mirrored point of view, so one network can drive either side. reward0 is team
// 0's (+1/-1/0); the game is zero-sum, so team 1's is -reward0.
struct SelfPlayStepResult {
    std::vector<float> observation0;
    std::vector<float> observation1;
    float reward0;
    bool done;
};

// The same advance for a caller that discards both observations. Rollouts step
// a snapshot many times and read one observation at the end, and building each
// unused observation costs far more than the physics tick
// (perception/UPSTREAM_REQUESTS.md item 21). Both entry points share
// runSelfPlayTicks, so they cannot diverge; only the return differs.
struct SelfPlayFastResult {
    float reward0;
    bool done;
};

class ClashEnv {
public:
    // Structural constants of the observation and action encoding, bound
    // read-only so Python queries them instead of keeping copies.
    static constexpr int BOARD_WIDTH = 18;
    // 34 rows: one mostly-dead row behind each King (see
    // Board::isBackRowDeadZone).
    static constexpr int BOARD_HEIGHT = 34;
    // Spatial channels:
    //
    //   0-3: ally melee, ranged, building-targeter (win conditions), buildings | 4-7: enemy, same | 8: river / bridges
    //
    // 9-20 are attribute channels, for what 0-8 cannot express: air vs ground,
    // how many units share a cell (0-7 store one HP fraction by assignment),
    // and what a unit does (damage, reach, speed). Per-cell attributes rather
    // than a card-id one-hot stack, which would be ~113k floats. Ally first,
    // enemy second, as in 0-3 / 4-7.
    static constexpr int CH_COUNT = 9;    // 9 ally / 10 enemy: units per cell
    static constexpr int CH_FLYING = 11;  // 11 / 12: any flyer here
    static constexpr int CH_ANTIAIR = 13; // 13 / 14: can hit air
    static constexpr int CH_DPS = 15;     // 15 / 16: damage per tick
    static constexpr int CH_RANGE = 17;   // 17 / 18: attack range
    static constexpr int CH_SPEED = 19;   // 19 / 20: move speed
    static constexpr int NUM_CHANNELS = 21;
    static constexpr int HAND_SIZE = 4;
    // One-hot width for card identity. An alias: the value lives in
    // CardRegistry.h (CARD_ID_COUNT) so GameManager, which cannot include this
    // header, sizes its cycle tracking from it too. Kept a few ids ahead of the
    // highest registered id.
    static constexpr int NUM_CARD_IDS = CARD_ID_COUNT;
    static constexpr float MAX_TROOP_HP = 4256.0f;
    static constexpr float MAX_BUILDING_HP = 4008.0f;

    // Normalisers for the attribute channels, from the registry's real spread.
    // Every attribute is clamped to 1.0, so an outlier card saturates rather
    // than blowing up the input scale.
    static constexpr float MAX_UNIT_DPS = 60.0f;
    static constexpr float MAX_ATTACK_RANGE = 12.0f;
    static constexpr float MAX_UNIT_SPEED = 1.5f;
    static constexpr float MAX_CELL_UNITS = 5.0f;

    // Scalars after the [elixir, costs, one-hots] block:
    //
    //   0 time fraction | 1 own elixir spent | 2 opp elixir spent
    //   3-5 own king/left/right tower HP | 6-8 enemy king/left/right tower HP
    //   9 elixir phase multiplier / 3.0
    //
    // The phase is observable (the "2x ELIXIR" banner and the clock), and
    // without it the state is not Markovian: the value of holding elixir
    // depends on the rate it will arrive at.
    static constexpr int NUM_EXTRA_SCALARS = 10;

    // --- opponent card-cycle blocks (perception/UPSTREAM_REQUESTS.md item 24)
    // ---
    // Two NUM_CARD_IDS-wide blocks after the extra scalars, describing what the
    // OPPONENT has shown this match:
    //
    //   block 0  seen[c]     1.0 once they have played card c
    //   block 1  recency[c]  exp(-(now - lastPlayed[c]) / TAU), else 0.0
    //
    // Not cheating: a human watching sees every card played, just as the
    // encoder already exposes the opponent's spend but not their elixir. The
    // hand stays hidden. Recency decays rather than being a timestamp, so it
    // needs no normalisation and answers "can it be back yet?" directly.
    static constexpr int NUM_CYCLE_BLOCKS = 2;
    static constexpr int CYCLE_BLOCK_SIZE = NUM_CYCLE_BLOCKS * NUM_CARD_IDS;
    // 200 ticks = 20 s, about one 8-card rotation: half a cycle ago reads
    // ~0.61, a full cycle ~0.37.
    static constexpr float CYCLE_RECENCY_TAU_TICKS = 200.0f;

    // --- named offsets into the observation ---
    // Where each appended section starts, measured forward from index 0. The
    // layout only ever grows by appending, so a forward offset cannot be
    // invalidated; subtracting from the end breaks on the next append. Bound to
    // Python.
    static constexpr int EXTRA_SCALARS_START =
          BOARD_WIDTH * BOARD_HEIGHT * NUM_CHANNELS   // spatial channels
        + 1                                            // own elixir
        + HAND_SIZE                                    // hand costs
        + HAND_SIZE * NUM_CARD_IDS;                    // hand identity one-hots
    static constexpr int CYCLE_START = EXTRA_SCALARS_START + NUM_EXTRA_SCALARS;
    // Ceiling on cumulative per-match elixir spend, the normaliser of scalars
    // 1-2. Sized for a full-length match under the elixir phases (~278
    // including the starting 5), so the spend scalars never saturate mid-match.
    static constexpr float MAX_MATCH_ELIXIR = 280.0f;

private:
    GameManager game;
    int maxTicks;
    int currentTick;
    std::mt19937 rng;
    HeuristicOpponent heuristicOpponent;
    GameLogger logger;

    // Distinguishes the snapshot constructor from the ordinary (shallow) copy
    // constructor, which pybind11 relies on.
    struct SnapshotTag {};

    // A constructor rather than copy-then-fix: GameManager's copy assignment is
    // deleted (it holds a const member), but copy-initialising it here works.
    // `logger` is left out and starts empty (see snapshot()).
    ClashEnv(const ClashEnv& other, SnapshotTag)
        : game(other.game.snapshot()),
          maxTicks(other.maxTicks),
          currentTick(other.currentTick),
          rng(other.rng),
          heuristicOpponent(other.heuristicOpponent) {}

    // The observation from `team`'s own point of view: the network always sees
    // itself as team 0 (near low y, channels 0-3), so it can drive either side.
    // team 0's output is unmirrored.
    std::vector<float> extractObservationForTeam(int team) {
        int spatialSize = BOARD_WIDTH * BOARD_HEIGHT * NUM_CHANNELS;
        // Reserve the full observation before laying down the spatial block, so
        // the scalar push_backs never reallocate and copy it. This is the
        // hottest function in the C++ layer.
        std::vector<float> obs;
        obs.reserve(observationSize());
        obs.resize(spatialSize, 0.0f);

        auto getIndex = [&](int channel, int y, int x) {
            return channel * (BOARD_HEIGHT * BOARD_WIDTH) + y * BOARD_WIDTH + x;
        };

        // River/bridge marker row. The same row for both teams, since each
        // observation is its own mirrored frame. Bridge columns come from
        // Board::isOnBridge, the rule the physics uses.
        constexpr int riverRow = 17;
        const Board& board = game.getBoard();
        for (int x = 0; x < BOARD_WIDTH; ++x) {
            obs[getIndex(8, riverRow, x)] =
                board.isOnBridge(static_cast<float>(x)) ? 1.0f : -1.0f;
        }

        for (const auto& entity : game.getBoard().getEntities()) {
            if (!entity->isAlive()) continue;
            // Projectiles and pending spells are not board presence.
            if (!entity->isTargetable()) continue;

            int x = static_cast<int>(entity->position.x);
            // Mirrored for team 1, which physically sits near high y. Mirror
            // the position, then truncate: `33 - int(y)` equals `int(33 - y)`
            // only for integer y, and every troop sits at a fractional y.
            // std::floor so an entity behind the back row yields -1 and is
            // dropped.
            int y = (team == 0)
                ? static_cast<int>(entity->position.y)
                : static_cast<int>(std::floor((BOARD_HEIGHT - 1) - entity->position.y));

            if (x < 0 || x >= BOARD_WIDTH || y < 0 || y >= BOARD_HEIGHT) continue;

            // Type category: building / building-targeter / ranged / melee.
            // BuildingTargeter* also matches RangedBuildingTargeter.
            int typeOffset;
            // isBuilding() instead of a dynamic_cast, on a loop over every
            // entity.
            bool isBuilding = entity->isBuilding();
            if (isBuilding) typeOffset = 3;
            else if (dynamic_cast<BuildingTargeter*>(entity.get()) != nullptr) typeOffset = 2;
            else if (dynamic_cast<RangedTroop*>(entity.get()) != nullptr) typeOffset = 1;
            else typeOffset = 0;

            float maxHp = isBuilding ? MAX_BUILDING_HP : MAX_TROOP_HP;
            float normalizedHp = std::min(static_cast<float>(entity->hp) / maxHp, 1.0f);
            // "Ally" is whichever team this observation is for.
            bool isAlly = (entity->team == team);
            int channel = (isAlly ? 0 : 4) + typeOffset;

            obs[getIndex(channel, y, x)] = normalizedHp;

            // --- attribute channels (9-20); side 0 = ally, 1 = enemy ---
            int side = isAlly ? 0 : 1;

            // COUNT accumulates where the HP channels assign, so a Skeleton
            // Army stops looking like one skeleton.
            float& cell = obs[getIndex(CH_COUNT + side, y, x)];
            cell = std::min(cell + 1.0f / MAX_CELL_UNITS, 1.0f);

            // The other attributes take the max over a cell's occupants: the
            // decision is "what is the most dangerous thing here, and can I hit
            // it". A sum would make three skeletons read like a P.E.K.K.A.
            auto putMax = [&](int channelBase, float value) {
                float& slot = obs[getIndex(channelBase + side, y, x)];
                slot = std::max(slot, std::min(value, 1.0f));
            };

            if (entity->isFlying) putMax(CH_FLYING, 1.0f);

            if (const auto* combat = dynamic_cast<const CombatEntity*>(entity.get())) {
                if (combat->targetsAir) putMax(CH_ANTIAIR, 1.0f);
                // Per tick, not per attack.
                putMax(CH_DPS, combat->getDamagePerTick() / MAX_UNIT_DPS);
                putMax(CH_RANGE, combat->getAttackRange() / MAX_ATTACK_RANGE);
            }
            if (const auto* troop = dynamic_cast<const Troop*>(entity.get())) {
                putMax(CH_SPEED, troop->getSpeed() / MAX_UNIT_SPEED);
            }
        }

        obs.push_back(game.getElixir(team) / 10.0f);

        for (int cardId : game.getHand(team)) {
            const auto* card = CardRegistry::getInstance().getCard(cardId);
            obs.push_back(card ? card->cost / 10.0f : 0.0f);
        }

        // Card identity per hand slot; costs alone cannot tell a Hog Rider from
        // a Musketeer.
        for (int cardId : game.getHand(team)) {
            // Zero the block and set one bit. An out-of-range or -1 id leaves
            // it all zeros.
            const size_t base = obs.size();
            obs.resize(base + NUM_CARD_IDS, 0.0f);
            if (cardId >= 0 && cardId < NUM_CARD_IDS) obs[base + cardId] = 1.0f;
        }

        // --- appended scalars (NUM_EXTRA_SCALARS) ---
        // Time. Without it the state is not Markovian: TimeoutRules decides a
        // timed-out match on towers, and the critic predicts a time-dependent
        // return.
        obs.push_back(static_cast<float>(currentTick) / static_cast<float>(maxTicks));

        // Cumulative elixir spent, both sides. The spend, not the opponent's
        // current elixir: a human sees every card played and knows its cost,
        // but not the bar.
        obs.push_back(std::min(game.getStatistics().elixirSpent(team) / MAX_MATCH_ELIXIR, 1.0f));
        obs.push_back(std::min(game.getStatistics().elixirSpent(1 - team) / MAX_MATCH_ELIXIR, 1.0f));

        // Tower HP as explicit scalars: the win condition is defined on them,
        // and in the building channel each is one cell that must survive two
        // MaxPools. A destroyed tower reads 0.0. Left/right are by x, which is
        // not mirrored for team 1 (its view is a y-flip), and judged against
        // ArenaLayout::CENTER_X, as GameManager::findTower does. One pass for
        // all six.
        float towerHp[2][3] = { { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f } };
        for (const auto& entity : game.getBoard().getEntities()) {
            if (!entity->isAlive() || !entity->isTower()) continue;
            const int side = (entity->team == team) ? 0 : 1;
            const bool isKing = (entity->cardId == GameManager::TOWER_KING_ID);
            const int which = isKing ? 0
                : (entity->position.x < ArenaLayout::CENTER_X ? 1 : 2);
            towerHp[side][which] = entity->hp / MAX_BUILDING_HP;
        }
        for (int which = 0; which < 3; ++which) obs.push_back(towerHp[0][which]);
        for (int which = 0; which < 3; ++which) obs.push_back(towerHp[1][which]);

        // Elixir phase (scalar 9). The same for both teams. Read from
        // GameManager's schedule rather than recomputed from scalar 0, and not
        // redundant with it: scalar 0 is a smooth ramp, the phase a step, and
        // crossing 2:00 changes the reward discontinuously.
        obs.push_back(game.getElixirMultiplier() / GameManager::MAX_ELIXIR_MULTIPLIER);

        // --- opponent card cycle (item 24) ---
        // `1 - team`: what the observer has watched the OPPONENT play; its own
        // cycle is already visible in its hand.
        const int opponent = 1 - team;
        const int now = currentTick;
        for (int block = 0; block < NUM_CYCLE_BLOCKS; ++block) {
            for (int cardId = 0; cardId < NUM_CARD_IDS; ++cardId) {
                const int last = game.getLastPlayedTick(opponent, cardId);
                if (last < 0) {
                    // Never played: both blocks 0. seen=0 means "no
                    // information"; seen=1 with recency ~0 means "they have it,
                    // played long ago".
                    obs.push_back(0.0f);
                    continue;
                }
                if (block == 0) {
                    obs.push_back(1.0f);
                } else {
                    // set_current_tick() can rewind the clock; a negative age
                    // would push exp() past 1.0.
                    const float age = static_cast<float>(std::max(0, now - last));
                    obs.push_back(std::exp(-age / CYCLE_RECENCY_TAU_TICKS));
                }
            }
        }

        return obs;
    }

    std::vector<float> extractObservation() { return extractObservationForTeam(0); }

    float calculateReward() {
        if (!game.isGameOver()) {
            // The tick limit is not a draw: both Kings are up, so TimeoutRules
            // decides on towers.
            if (currentTick >= maxTicks) {
                MatchRules::Outcome timeoutOutcome = TimeoutRules::resolve(game.getBoard());
                if (timeoutOutcome.loserTeam == 1) return 1.0f;
                if (timeoutOutcome.loserTeam == 0) return -1.0f;
                return 0.0f;   // exact tie on towers and weakest-tower HP
            }
            return 0.0f;
        }
        int loser = game.getLoserTeam();
        if (loser == 1) return 1.0f;
        if (loser == 0) return -1.0f;
        return 0.0f;
    }

    // See HeuristicOpponent.
    void opponentTurn() {
        heuristicOpponent.act(game, rng);
    }

public:
    ClashEnv(const std::vector<int>& aiDeck, const std::vector<int>& oppDeck, int maxTicks = 3600,
            TowerTroopType aiTowerTroop = TowerTroopType::None,
            TowerTroopType oppTowerTroop = TowerTroopType::None)
        : game(aiDeck, oppDeck, aiTowerTroop, oppTowerTroop), maxTicks(maxTicks), currentTick(0),
          rng(std::random_device{}()) { heuristicOpponent.reset(rng); }

    // An independent copy for decision-time search (see GameManager::snapshot
    // and Board::deepCopy underneath). The replay logger starts empty: copying
    // a match's history per candidate is the largest cost, and a rollout's
    // ticks do not belong in the real replay. `rng` and `heuristicOpponent` are
    // copied, so the opponent behaves in the rollout as it would in the real
    // match.
    ClashEnv snapshot() const { return ClashEnv(*this, SnapshotTag{}); }

    // Built from the named offsets, so size and offsets cannot disagree.
    int observationSize() const { return CYCLE_START + CYCLE_BLOCK_SIZE; }

    // How many of `team`'s towers stand (King and Princesses only). The reward
    // puts a deliberately non-potential-based bonus on destroying a tower: the
    // policy-invariant tower term left pure defence optimal.
    int getTowersAlive(int team) const {
        int count = 0;
        for (const auto& entity : game.getBoard().getEntities()) {
            if (!entity->isAlive() || entity->team != team) continue;
            if (dynamic_cast<const Tower*>(entity.get()) != nullptr) ++count;
        }
        return count;
    }

    // The winner by TimeoutRules' full rule (tower count, then weakest tower
    // hp, then draw), as MatchRules::Outcome::loserTeam: -1 draw, else the
    // losing team. Bound so evaluation scripts read the verdict instead of
    // re-deriving it from tower counts and dropping the tie-break
    // (python_ai/eval/match_outcome.py).
    int resolveTimeoutOutcome() const {
        return TimeoutRules::resolve(game.getBoard()).loserTeam;
    }

    float getMaxPlacementX() const { return game.getMaxPlacementX(); }
    float getOwnHalfMaxY() const { return game.getOwnHalfMaxY(); }

    // Would playCard accept this card here? A pure query, bound so the Python
    // placement mask uses the engine's own predicate (bounds, back-row dead
    // zone, footprint, own half, tower clearance) rather than a copy of it.
    bool isValidPlacementForCard(int cardId, float x, float y, int team) const {
        const CardDefinition* def = CardRegistry::getInstance().getCard(cardId);
        if (!def) return false;
        return game.isValidPlacement(team, x, y, def->isSpell,
                                     def->placementRadius, def->deployAnywhere,
                                     def->rollRange > 0.0f);
    }

    std::vector<float> reset() {
        logger.clear();
        game.reset();
        // New lane per match (HeuristicOpponent::reset).
        heuristicOpponent.reset(rng);
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

    // Either team's current elixir. Not in the observation: the opponent's bar
    // is hidden on a real screen. For tests, probes and the teacher.
    float getElixirForTeam(int team) const {
        return game.getElixir(team);
    }

    bool isGameOver() const {
        return game.isGameOver() || currentTick >= maxTicks;
    }

    // Whether `team`'s Champion in `slot` (1 = Heroic, 2 = Wild Card) can
    // activate now. An accessor, not an observation field.
    bool isChampionAbilityReady(int team, int slot = 1) const {
        return game.isChampionAbilityReady(team, slot);
    }

    // Activates the ability directly, for tests and scripts that do not go
    // through step().
    bool activateChampionAbility(int team, int slot = 1) {
        return game.activateChampionAbility(team, slot);
    }

    // activateAbilitySlot1/2: deck slots 1 (Heroic) and 2 (Wild Card),
    // independent.
    StepResult step(int cardIndex, float targetX, float targetY, int skipFrames = 10,
            bool activateAbilitySlot1 = false, bool activateAbilitySlot2 = false) {
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
            if (i == 0 && activateAbilitySlot1) {
                game.activateChampionAbility(0, 1);
            }
            if (i == 0 && activateAbilitySlot2) {
                game.activateChampionAbility(0, 2);
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

    // The observation from `team`'s own point of view, for whatever policy
    // drives team 1 in self-play.
    std::vector<float> getObservationForTeam(int team) {
        return extractObservationForTeam(team);
    }

    // Test-only access to the wrapped GameManager.
    GameManager& debugGame() { return game; }

    // Self-play stepping: both sides' actions come from outside and
    // opponentTurn() is not called. targetY1 arrives in team 1's mirrored frame
    // and is converted back; mirroring twice is the identity.
private:
    // What the self-play tick loop produces, before deciding whether to build
    // observations.
    struct SelfPlayTickOutcome {
        float reward;
        bool done;
    };

    // The one self-play tick loop, shared by stepSelfPlay and stepSelfPlayFast
    // so the two cannot drift; the fast path runs only inside rollouts, where a
    // divergence would go unnoticed.
    SelfPlayTickOutcome runSelfPlayTicks(int cardIndex0, float targetX0, float targetY0,
                                          int cardIndex1, float targetX1, float targetY1,
                                          int skipFrames,
                                          bool activateAbility0Slot1, bool activateAbility0Slot2,
                                          bool activateAbility1Slot1, bool activateAbility1Slot2) {
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
                if (activateAbility0Slot1) game.activateChampionAbility(0, 1);
                if (activateAbility0Slot2) game.activateChampionAbility(0, 2);
                if (activateAbility1Slot1) game.activateChampionAbility(1, 1);
                if (activateAbility1Slot2) game.activateChampionAbility(1, 2);
            }

            game.step();
            currentTick++;

            isDone = (currentTick >= maxTicks) || game.isGameOver();
            totalReward += calculateReward();

            logger.logTick(currentTick, game);

            if (isDone) break;
        }

        return { totalReward, isDone };
    }

public:
    // Advance, then both observations.
    SelfPlayStepResult stepSelfPlay(int cardIndex0, float targetX0, float targetY0,
                                     int cardIndex1, float targetX1, float targetY1,
                                     int skipFrames = 10,
                                     bool activateAbility0Slot1 = false, bool activateAbility0Slot2 = false,
                                     bool activateAbility1Slot1 = false, bool activateAbility1Slot2 = false) {
        SelfPlayTickOutcome out = runSelfPlayTicks(
            cardIndex0, targetX0, targetY0, cardIndex1, targetX1, targetY1, skipFrames,
            activateAbility0Slot1, activateAbility0Slot2,
            activateAbility1Slot1, activateAbility1Slot2);
        return { extractObservationForTeam(0), extractObservationForTeam(1),
                 out.reward, out.done };
    }

    // The same advance with no observations, for search and teacher rollouts
    // that read the board once at the end, if at all.
    SelfPlayFastResult stepSelfPlayFast(int cardIndex0, float targetX0, float targetY0,
                                         int cardIndex1, float targetX1, float targetY1,
                                         int skipFrames = 10,
                                         bool activateAbility0Slot1 = false, bool activateAbility0Slot2 = false,
                                         bool activateAbility1Slot1 = false, bool activateAbility1Slot2 = false) {
        SelfPlayTickOutcome out = runSelfPlayTicks(
            cardIndex0, targetX0, targetY0, cardIndex1, targetX1, targetY1, skipFrames,
            activateAbility0Slot1, activateAbility0Slot2,
            activateAbility1Slot1, activateAbility1Slot2);
        return { out.reward, out.done };
    }

    void injectEnemy(int cardId, float x, float y) {
        const auto* card = CardRegistry::getInstance().getCard(cardId);
        if (card) {
            card->spawnEntity(x, y, 1, game.getBoard());
        }
    }

    // The general form of injectEnemy, for state estimation: spawns directly,
    // bypassing hand, elixir and placement legality, since the real game
    // already validated the placement. hp < 0 keeps full health; deployTicks <
    // 0 keeps DEPLOY_TIME_TICKS.
    //
    // Pass deployTicks = 0 for a unit perception can already see:
    // applyCardMetadata otherwise hands every injected unit a fresh deploy
    // second, including one that has been walking for six.
    void inject(int cardId, float x, float y, int team,
                float hp = -1.0f, int deployTicks = -1) {
        const auto* card = CardRegistry::getInstance().getCard(cardId);
        if (!card) return;
        Board& board = game.getBoard();
        // The pending range is the handle back to what spawnEntity created; the
        // whole range, so a multi-body card (Skeletons: three) is overridden
        // for every body.
        const size_t before = board.pendingEntityCount();
        card->spawnEntity(x, y, team, board);
        if (hp < 0.0f && deployTicks < 0) return;   // nothing to override
        for (size_t i = before; i < board.pendingEntityCount(); ++i) {
            const std::shared_ptr<Entity>& e = board.getPendingEntity(i);
            if (hp >= 0.0f) {
                // e->hp is still the card's full health here, the natural
                // ceiling.
                int want = static_cast<int>(hp + 0.5f);
                if (want > e->hp) want = e->hp;
                if (want < 1) want = 1;
                e->hp = want;
            }
            if (deployTicks >= 0) {
                if (auto* combat = dynamic_cast<CombatEntity*>(e.get())) {
                    combat->deployTicksRemaining = deployTicks;
                }
            }
        }
    }

    // getHand() is team 0 only; this serves either team.
    std::vector<int> getHandForTeam(int team) const {
        return game.getHand(team);
    }

    // The write half of the estimator interface; see GameManager::setElixir /
    // setHand, including why setHand can refuse.
    void setElixirForTeam(int team, float value) { game.setElixir(team, value); }
    bool setHandForTeam(int team, const std::vector<int>& cards) {
        return game.setHand(team, cards);
    }

    // --- tower HP and the match clock (item 22) ---
    // Thin wrappers; the contracts live on GameManager.
    bool setTowerHp(int team, int slot, float hp) { return game.setTowerHp(team, slot, hp); }
    bool destroyTower(int team, int slot) { return game.destroyTower(team, slot); }
    int getTowerHp(int team, int slot) const { return game.getTowerHp(team, slot); }
    int getTowerMaxHp(int team, int slot) const { return game.getTowerMaxHp(team, slot); }

    // The one field nothing else reconstructs. Clamped to [0, maxTicks]. Sets
    // both clocks: this one drives the time scalar and the done condition,
    // GameManager's the event stamps and ability cooldowns.
    void setCurrentTick(int tick) {
        currentTick = std::max(0, std::min(tick, maxTicks));
        game.setCurrentTick(currentTick);
    }

    // The matching read.
    int getCurrentTick() const { return currentTick; }

    // The elixir phase in force (1.0 / 2.0 / 3.0), so readers need not know the
    // boundary ticks.
    float getElixirMultiplier() const { return game.getElixirMultiplier(); }

    // Records that `team` played `cardId` now, without spawning anything.
    //
    // For the live mirror in perception/, and not redundant with inject():
    // inject() puts a BODY on the board without going through playCard (so it
    // costs no elixir and no hand slot), and playCard is where cycle tracking
    // is hooked. Without this, every card the estimator saw would be missing
    // from the cycle blocks in deployment. inject() does not call it: scenario
    // setup and tests inject bodies that were never played.
    void notePlayedCard(int team, int cardId) {
        game.notePlayedCard(team, cardId);
    }

    // The raw tick behind the recency channel, so a test can check what was
    // recorded separately from how it is encoded.
    int getLastPlayedTick(int team, int cardId) const {
        return game.getLastPlayedTick(team, cardId);
    }

    // Reproducibility: same seed, same opening hands, cycle order and heuristic
    // rolls. Seeds both generators (this one drives HeuristicOpponent, `game`'s
    // deals the hands), offset by an XOR so the streams are not identical. The
    // reset() re-deals; without it the seed would only apply from the next
    // episode.
    void seed(unsigned int s) {
        rng.seed(s);
        heuristicOpponent.reset(rng);
        game.seed(s ^ 0x9E3779B9u);
        reset();
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

    // Plain ints, read every training step (no JSON round trip). See
    // MatchStatistics.h.
    int getTroopDamageDealt(int team) const { return game.getStatistics().troopDamageDealt(team); }
    int getBuildingDamageDealt(int team) const { return game.getStatistics().buildingDamageDealt(team); }
    int getTowerDamageDealt(int team) const { return game.getStatistics().towerDamageDealt(team); }
    // Elixir value a card has destroyed, and the elixir spent on it. Both are
    // needed: rewarding only value destroyed would make a whiffed spell free.
    float getElixirValueKilledBy(int cardId, int team) const {
        return game.getStatistics().elixirValueKilledBy(cardId, team);
    }
    float getElixirSpentOnCard(int cardId, int team) const {
        return game.getStatistics().elixirSpentByCard(cardId, team);
    }
    // Total damage one card has dealt this match, to anything. For a
    // building-targeting win condition this is nearly all tower damage (plus
    // any deployed building in its path); for a troop-targeting card it
    // measures something else.
    int getDamageDealtByCard(int cardId, int team) const {
        return game.getStatistics().damageDealtByCard(cardId, team);
    }
    float getElixirSpent(int team) const { return game.getStatistics().elixirSpent(team); }
};

// Every registered playable card id. Child units (Golemite, Ram Rider's
// crossbow) have negative ids and are never registered; Evolutions are excluded
// because they are upgrades equipped onto a base card, not standalone cards.
// Lets Python sample random decks from what is actually registered.
inline std::vector<int> getAllCardIds() {
    std::vector<int> ids;
    for (const auto& [id, def] : CardRegistry::getInstance().getAllCards()) {
        if (def.isEvolution) continue;
        ids.push_back(id);
    }
    return ids;
}

// A random 8-card deck that is legal by construction: each slot draws only from
// the pools CardRegistry::validateDeckSlots allows there (slot 0: plain +
// Evolution; slot 1: plain + Champion/Hero; slot 2: all three; slots 3-7:
// plain), with a moderate chance per eligible slot of a special card. The
// chance is a training choice, not a game rule. No card repeats.
inline std::vector<int> sampleRandomDeck(std::mt19937& rng) {
    std::vector<int> plainPool, evolutionPool, specialUnitPool;
    for (const auto& [id, def] : CardRegistry::getInstance().getAllCards()) {
        if (def.isEvolution) evolutionPool.push_back(id);
        else if (def.isChampion || def.isHero) specialUnitPool.push_back(id);
        else plainPool.push_back(id);
    }

    std::unordered_set<int> used;
    std::uniform_real_distribution<float> chance01(0.0f, 1.0f);
    const float SPECIAL_SLOT_CHANCE = 0.4f;

    auto pickFrom = [&rng, &used](const std::vector<int>& pool) {
        std::vector<int> eligible;
        for (int id : pool) if (!used.count(id)) eligible.push_back(id);
        std::uniform_int_distribution<size_t> dist(0, eligible.size() - 1);
        int chosen = eligible[dist(rng)];
        used.insert(chosen);
        return chosen;
    };

    std::vector<int> deck(8, -1);
    // Slot 0: Evolution slot.
    deck[0] = (!evolutionPool.empty() && chance01(rng) < SPECIAL_SLOT_CHANCE)
        ? pickFrom(evolutionPool) : pickFrom(plainPool);
    // Slot 1: Heroic slot (Champion / Hero).
    deck[1] = (!specialUnitPool.empty() && chance01(rng) < SPECIAL_SLOT_CHANCE)
        ? pickFrom(specialUnitPool) : pickFrom(plainPool);
    // Slot 2: Wild Card slot (Champion / Hero or Evolution): roll for special,
    // then for which kind.
    bool slot2Special = (!evolutionPool.empty() || !specialUnitPool.empty())
        && chance01(rng) < SPECIAL_SLOT_CHANCE;
    if (slot2Special) {
        bool useEvolution = evolutionPool.empty() ? false
            : specialUnitPool.empty() ? true : (chance01(rng) < 0.5f);
        deck[2] = useEvolution ? pickFrom(evolutionPool) : pickFrom(specialUnitPool);
    } else {
        deck[2] = pickFrom(plainPool);
    }
    // Slots 3-7: plain only.
    for (int i = 3; i < 8; ++i) deck[i] = pickFrom(plainPool);

    return deck;
}