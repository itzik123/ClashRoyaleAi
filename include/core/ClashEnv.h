#pragma once
#include "GameManager.h"
#include "TimeoutRules.h"
#include "HeuristicOpponent.h"
#include "Building.h"
#include "BuildingTargeter.h"
#include "RangedTroop.h"
// Explicit, not relied on transitively through RangedTroop.h: the attribute
// channels read Troop::speed and CombatEntity::damage/attackRange/targetsAir
// directly.
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
public:
    // Structural constants the observation/action encoding is actually built
    // from -- public (and bound read-only in bindings.cpp) so the Python side
    // can query them directly at runtime instead of hardcoding a matching copy
    // that has to be remembered and hand-updated every time one of these
    // changes on the C++ side. Confirmed painful in practice: silent drift
    // here (a card-roster bump, a board-geometry change) has crashed training
    // more than once this project's history before this was queryable.
    static constexpr int BOARD_WIDTH = 18;
    // Real-map sync: 34, not 32 -- one extra row behind each King Tower
    // (mostly dead space, a narrow center gap is real ground). See Board.h's
    // BACK_ROW_OPENING_HALF_WIDTH / isBackRowDeadZone for the placement side
    // of this; this constant only affects the observation tensor's shape.
    static constexpr int BOARD_HEIGHT = 34;
    // Spatial channels, per team: melee troops / ranged troops / building-targeters
    // (win-conditions like Hog, Giant, Golem) / buildings (towers + defensive).
    // Unit-TYPE visibility is what lets the net answer "what is attacking me and
    // with what should I respond" -- with HP-only channels a wounded PEKKA and a
    // Skeleton looked identical.
    //
    // 0-3: ally melee, ranged, tank, buildings | 4-7: enemy same | 8: river/bridges
    //
    // 9-20 are ATTRIBUTE channels, added because 0-8 could not express three
    // things the game is actually decided by:
    //
    //  (a) AIR vs GROUND. isFlying/targetsAir have existed on Entity/
    //      CombatEntity since flying cards were added, and were visible
    //      NOWHERE in the observation. A Balloon and a Hog Rider were both
    //      just "building-targeter"; a Minion and a Knight were both just
    //      "melee". "Can the thing I am about to play even hit that" was
    //      unanswerable from the observation, so no air/ground counterplay
    //      could ever be learned no matter how long training ran.
    //  (b) HOW MANY units are in a cell. Channels 0-7 store an HP fraction by
    //      ASSIGNMENT, so a Skeleton Army collapsed to a handful of cells
    //      indistinguishable from a handful of single skeletons.
    //  (c) WHAT a unit does. Two ranged units with equal HP fractions were
    //      identical regardless of damage, reach or speed.
    //
    // Encoded as per-cell attributes rather than a NUM_CARD_IDS-deep one-hot
    // stack: identity per se is not what the decision needs, the attributes
    // are, and a 185-channel board would be ~113k floats per observation.
    // ALLY channel first, ENEMY second, so every pair is (base + isAlly?0:1)
    // exactly like the 0-3/4-7 blocks above.
    static constexpr int CH_COUNT = 9;    // 9 ally / 10 enemy -- units per cell
    static constexpr int CH_FLYING = 11;  // 11 ally / 12 enemy -- any flyer here
    static constexpr int CH_ANTIAIR = 13; // 13 ally / 14 enemy -- can hit air
    static constexpr int CH_DPS = 15;     // 15 ally / 16 enemy -- damage per tick
    static constexpr int CH_RANGE = 17;   // 17 ally / 18 enemy -- attack range
    static constexpr int CH_SPEED = 19;   // 19 ally / 20 enemy -- move speed
    static constexpr int NUM_CHANNELS = 21;
    static constexpr int HAND_SIZE = 4;
    // One-hot size for card identity in hand. Registered ids currently run up
    // to 175 (Evolutions 123-163, Mirror 164, Spirit Empress 165, Heroes
    // 166-175) -- kept a few slots ahead of that max so future card
    // additions don't silently go blind again the way ids 120-122 did
    // between the last bump and this one (see CardRegistry.h for the actual
    // registered range). No longer needs a matching manual bump in
    // python_ai/model.py -- see this constant's binding in bindings.cpp.
    static constexpr int NUM_CARD_IDS = 185;
    static constexpr float MAX_TROOP_HP = 4256.0f;
    static constexpr float MAX_BUILDING_HP = 4008.0f;

    // Normalizers for the attribute channels above. Picked from the real
    // spread in CardRegistry rather than guessed: troop speeds run 0.5
    // (Knight/Musketeer/Valkyrie) to 1.0+ (Goblins/Skeletons); attack ranges
    // run 0.5 (Skeletons) to 6.0 (Musketeer) for troops and up to ~9 for
    // Princess Towers; damage/attackCooldown runs ~7/tick (Skeletons) to
    // ~47/tick (Mini PEKKA). Every attribute channel is clamped to 1.0, so a
    // future outlier card saturates instead of blowing the input scale.
    static constexpr float MAX_UNIT_DPS = 60.0f;
    static constexpr float MAX_ATTACK_RANGE = 12.0f;
    static constexpr float MAX_UNIT_SPEED = 1.5f;
    static constexpr float MAX_CELL_UNITS = 5.0f;

    // Scalars appended AFTER the existing [elixir, costs, one-hots] block --
    // see extractObservationForTeam's tail. Appended, never inserted, so
    // every existing offset into the scalar section (model.py's onehot_start,
    // train_selfplay.py's scripted opponents) stays valid unchanged.
    //   0 time fraction | 1 own elixir spent | 2 opp elixir spent
    //   3-5 own king/left/right tower HP | 6-8 enemy king/left/right tower HP
    static constexpr int NUM_EXTRA_SCALARS = 9;
    // Ceiling on cumulative per-match elixir spend used to normalize scalars
    // 1-2: maxTicks(3600) * ELIXIR_REGEN_RATE(0.035) + 5 starting = 131, so
    // 140 leaves headroom for the curriculum's opponent elixir multiplier
    // without ever exceeding 1.0 in practice. Clamped anyway.
    static constexpr float MAX_MATCH_ELIXIR = 140.0f;

private:
    GameManager game;
    int maxTicks;
    int currentTick;
    std::mt19937 rng;
    HeuristicOpponent heuristicOpponent;
    GameLogger logger;

    // Disambiguates the snapshot constructor from the implicit copy
    // constructor, which must keep its ordinary shallow meaning -- pybind11
    // and anything else that copies a ClashEnv by value relies on it.
    struct SnapshotTag {};

    // A constructor rather than copy-then-fix, because GameManager's implicit
    // copy ASSIGNMENT is deleted: it holds `const float ELIXIR_REGEN_RATE`, so
    // `game = other.game.snapshot()` does not compile. Copy-INITIALISING it in
    // the member list works, and in C++17 the prvalue is elided straight into
    // place rather than moved.
    //
    // Initialiser order matches declaration order exactly (game, maxTicks,
    // currentTick, rng, heuristicOpponent) -- `logger` is omitted on purpose
    // and default-constructs empty, see snapshot() below.
    ClashEnv(const ClashEnv& other, SnapshotTag)
        : game(other.game.snapshot()),
          maxTicks(other.maxTicks),
          currentTick(other.currentTick),
          rng(other.rng),
          heuristicOpponent(other.heuristicOpponent) {}

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
        // teams' towers and the bridge gaps sit at the same x coordinates).
        //
        // The SAME row for both teams, deliberately. Each team's observation is
        // its own mirrored frame, so for the two to be interchangeable -- which
        // is the whole premise of driving team 1 with a network trained as
        // team 0 -- the marker has to land on the same row index in both. This
        // used to be (team == 0) ? 17 : BOARD_HEIGHT - 1 - 17, i.e. row 17 for
        // team 0 and row 16 for team 1, so a net that learned where the bridges
        // are as team 0 saw them one row nearer when driving team 1: 36
        // differing cells in this channel at reset, on an empty board.
        // See perception/UPSTREAM_REQUESTS.md item 6.
        constexpr int riverRow = 17;
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
            // Mirrored for team 1: physically team 1 sits near high y, but its
            // own network needs to see itself near low y (same layout it was
            // trained on as "team 0"), so flip before placing into the grid.
            //
            // Mirror the POSITION, then truncate -- not the other way round.
            // This was `BOARD_HEIGHT - 1 - static_cast<int>(position.y)`, and
            // `33 - int(y)` equals `int(33 - y)` only when y is an integer.
            // Every troop in play sits at a fractional y, so team 1's entire
            // observation was displaced one row, every tick, for both its own
            // and enemy units -- while team 0's was correct. Measured: a policy
            // played against a bit-exact copy of itself scored 0.598 as team 0
            // over 400 episodes (95% CI [0.548, 0.647]).
            //
            // It survived the 2026-07-30 geometry audit because the Princess
            // towers sit at y = 27.0 (integer, mirrors correctly) while the
            // Kings sit at y = 30.5 (fractional, off by one) -- the positions
            // themselves are symmetric, so auditing coordinates finds nothing.
            // See perception/UPSTREAM_REQUESTS.md item 5.
            //
            // std::floor rather than a bare cast so an entity behind the back
            // row yields -1 and is rejected by the bounds check below, instead
            // of truncating toward zero into row 0.
            int y = (team == 0)
                ? static_cast<int>(entity->position.y)
                : static_cast<int>(std::floor((BOARD_HEIGHT - 1) - entity->position.y));

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

            // --- attribute channels (9-20) ---------------------------------
            // Side offset is 0 for ally / 1 for enemy, matching the 0-3 vs
            // 4-7 split above.
            int side = isAlly ? 0 : 1;

            // COUNT accumulates (+=) where the HP channels assign (=). This
            // is the whole point of the channel: it is the only place a
            // 15-skeleton Skeleton Army stops looking like one skeleton.
            float& cell = obs[getIndex(CH_COUNT + side, y, x)];
            cell = std::min(cell + 1.0f / MAX_CELL_UNITS, 1.0f);

            // The remaining attributes take the MAX over units sharing a
            // cell, not the last writer and not the sum: the decision these
            // feed is "what is the most dangerous thing standing here and can
            // I hit it", so the strongest occupant is the right summary. A
            // sum would make three skeletons read like a PEKKA.
            auto putMax = [&](int channelBase, float value) {
                float& slot = obs[getIndex(channelBase + side, y, x)];
                slot = std::max(slot, std::min(value, 1.0f));
            };

            if (entity->isFlying) putMax(CH_FLYING, 1.0f);

            if (const auto* combat = dynamic_cast<const CombatEntity*>(entity.get())) {
                if (combat->targetsAir) putMax(CH_ANTIAIR, 1.0f);
                // Per-TICK damage, not per-attack: attackCooldown is in ticks
                // and a 755-damage Mini PEKKA swinging every 16 ticks is not
                // 3.7x a 202-damage Knight swinging every 12.
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

        // Card IDENTITY per hand slot (one-hot). Costs alone made a Hog Rider and a
        // Musketeer indistinguishable (both 0.4), so no card-specific strategy could
        // ever be learned.
        for (int cardId : game.getHand(team)) {
            for (int k = 0; k < NUM_CARD_IDS; ++k) {
                obs.push_back(k == cardId ? 1.0f : 0.0f);
            }
        }

        // --- appended scalars (NUM_EXTRA_SCALARS) --------------------------
        // TIME. Until this was added the observation had NO temporal
        // component at all: tick 100 and tick 3500 with the same board were
        // literally the same input vector. That is not a missing convenience,
        // it made the state non-Markovian for two decisions that now exist --
        // TimeoutRules decides a timed-out match on towers (so "I am ahead,
        // run the clock down" is a real strategy the agent could not even
        // represent), and the critic was being asked to predict a
        // time-dependent return from a time-free input, making part of its
        // residual variance structurally unlearnable rather than undertrained.
        obs.push_back(static_cast<float>(currentTick) / static_cast<float>(maxTicks));

        // ELIXIR SPENT, both sides, cumulative this match. Deliberately the
        // SPEND and not the opponent's current elixir: spend is what a human
        // can actually observe (you see every card they play and you know
        // what it costs), current elixir is hidden information. Handing the
        // agent the hidden value would train a policy that cannot be deployed
        // against a real opponent through perception/. Estimating the hidden
        // value from these is exactly what the network's auxiliary elixir
        // head is asked to learn instead.
        obs.push_back(std::min(game.getStatistics().elixirSpent(team) / MAX_MATCH_ELIXIR, 1.0f));
        obs.push_back(std::min(game.getStatistics().elixirSpent(1 - team) / MAX_MATCH_ELIXIR, 1.0f));

        // TOWER HP as explicit scalars. It is technically present in the
        // building channel already, but only as one cell's value that has to
        // survive two MaxPools -- while being the single number the win
        // condition is defined on. A destroyed tower reads 0.0 (it is no
        // longer a live entity), which is exactly the right encoding.
        // Left/right are by x, and x is NOT mirrored for team 1 -- same
        // convention the spatial channels above already use (team 1's view is
        // a y-flip, not a 180-degree rotation), so this stays consistent with
        // them rather than inventing a second frame.
        auto towerHp = [&](int forTeam, int which) {   // which: 0=king, 1=left, 2=right
            for (const auto& entity : game.getBoard().getEntities()) {
                if (!entity->isAlive() || entity->team != forTeam) continue;
                if (dynamic_cast<const Tower*>(entity.get()) == nullptr) continue;
                bool isKing = (entity->cardId == GameManager::TOWER_KING_ID);
                if (which == 0) {
                    if (isKing) return entity->hp / MAX_BUILDING_HP;
                } else if (!isKing) {
                    bool isLeft = entity->position.x < BOARD_WIDTH / 2.0f;
                    if ((which == 1) == isLeft) return entity->hp / MAX_BUILDING_HP;
                }
            }
            return 0.0f;
        };
        for (int which = 0; which < 3; ++which) obs.push_back(towerHp(team, which));
        for (int which = 0; which < 3; ++which) obs.push_back(towerHp(1 - team, which));

        return obs;
    }

    std::vector<float> extractObservation() { return extractObservationForTeam(0); }

    float calculateReward() {
        if (!game.isGameOver()) {
            // Reaching the tick limit is NOT a draw. Both Kings are still up
            // (or GameManager would have flagged game-over), so the match is
            // decided on towers -- see TimeoutRules for the exact ordering.
            // Before this, every timed-out match scored 0.0, which taught the
            // agent that running the clock out was a neutral outcome rather
            // than a loss it should have been trying to avoid.
            if (currentTick >= maxTicks) {
                MatchRules::Outcome timeoutOutcome = TimeoutRules::resolve(game.getBoard());
                if (timeoutOutcome.loserTeam == 1) return 1.0f;
                if (timeoutOutcome.loserTeam == 0) return -1.0f;
                return 0.0f;   // exact tie on towers AND weakest-tower HP
            }
            return 0.0f;
        }
        int loser = game.getLoserTeam();
        if (loser == 1) return 1.0f;
        if (loser == 0) return -1.0f;
        return 0.0f;
    }

    // Delegates to HeuristicOpponent -- see that header for what the previous
    // inline implementation was and the measurements that motivated replacing
    // it. Kept as a method so every existing call site in step() is unchanged.
    void opponentTurn() {
        heuristicOpponent.act(game, rng);
    }

public:
    ClashEnv(const std::vector<int>& aiDeck, const std::vector<int>& oppDeck, int maxTicks = 3600,
            TowerTroopType aiTowerTroop = TowerTroopType::None,
            TowerTroopType oppTowerTroop = TowerTroopType::None)
        : game(aiDeck, oppDeck, aiTowerTroop, oppTowerTroop), maxTicks(maxTicks), currentTick(0),
          rng(std::random_device{}()) { heuristicOpponent.reset(rng); }

    // Independent copy of this environment, for decision-time search: try a
    // candidate action on the copy, roll it forward, score it, throw it away.
    // Nothing done to the copy can reach this env. See GameManager::snapshot
    // and Board::deepCopy for the two layers underneath.
    //
    // Everything is carried across except the replay logger, which starts
    // EMPTY. That is deliberate on both counts:
    //   * cost -- GameLogger accumulates a TickSnapshot per tick with an
    //     EntitySnapshot per entity, so by mid-match it is the largest thing
    //     in the object. Search copies an env once per candidate per decision,
    //     and copying a thousand ticks of history to simulate twenty is the
    //     kind of overhead that makes lookahead look infeasible when it isn't.
    //   * meaning -- a rollout is a hypothetical, not a match. Its ticks do
    //     not belong in a replay of the real one, and save_log() on a snapshot
    //     writing out the parent's real history followed by imagined ticks
    //     would be worse than either.
    // `rng` and `heuristicOpponent` ARE copied, so the opponent plays the same
    // way in the rollout as it would have in the real match -- a search whose
    // opponent behaved differently from the real one would be scoring the
    // wrong game.
    ClashEnv snapshot() const { return ClashEnv(*this, SnapshotTag{}); }

    int observationSize() const {
        return BOARD_WIDTH * BOARD_HEIGHT * NUM_CHANNELS   // spatial type/HP/attribute channels
             + 1                                            // elixir
             + HAND_SIZE                                    // card costs
             + HAND_SIZE * NUM_CARD_IDS                     // card identity one-hots
             + NUM_EXTRA_SCALARS;                           // time, elixir spent, tower HP
    }

    // Thin delegates to GameManager's own placement-bound queries (see that
    // class's comment) -- exposed here since ClashEnv, not GameManager, is
    // what's actually bound to Python. Lets the training scripts scale their
    // action space from the engine's real enforced bounds instead of a
    // hardcoded copy of the same numbers.
    // How many of `team`'s TOWERS are still standing (King + Princesses only --
    // Cannon/Tombstone and other placed buildings are not crowns). Exposed so
    // the Python reward can put a deliberate, NON-potential-based bonus on the
    // discrete act of destroying a tower. That bias is the point: the
    // potential-based tower term is policy-invariant by construction, which
    // measurably left "pure defence" as the true optimum against this engine's
    // opponent -- the agent stopped playing its win condition entirely.
    int getTowersAlive(int team) const {
        int count = 0;
        for (const auto& entity : game.getBoard().getEntities()) {
            if (!entity->isAlive() || entity->team != team) continue;
            if (dynamic_cast<const Tower*>(entity.get()) != nullptr) ++count;
        }
        return count;
    }

    float getMaxPlacementX() const { return game.getMaxPlacementX(); }
    float getOwnHalfMaxY() const { return game.getOwnHalfMaxY(); }

    // Would playCard accept this card here? READ-ONLY: a pure query that
    // touches no state and changes no gameplay path.
    //
    // Exposed because the Python action space was building its placement mask
    // from its own idea of the legal area -- the 16 own-half rows -- while the
    // engine additionally rejects Board::isBackRowDeadZone and the tower
    // footprints. Measured on the ep~45,800 checkpoint over 1,340 decision
    // steps, 58.7% of the policy's card choices were refused here and returned
    // false silently, which is indistinguishable from a no-op and puts pure
    // noise into the gradient. See perception/UPSTREAM_REQUESTS.md item 12.
    //
    // Deliberately not re-derived on the Python side: this predicate combines
    // board bounds, the back-row dead zone, per-card placementRadius/isSpell/
    // deployAnywhere and the tower footprint clearance. A second copy of that
    // geometry is exactly the drift this project has already paid for twice.
    bool isValidPlacementForCard(int cardId, float x, float y, int team) const {
        const CardDefinition* def = CardRegistry::getInstance().getCard(cardId);
        if (!def) return false;
        return game.isValidPlacement(team, x, y, def->isSpell,
                                     def->placementRadius, def->deployAnywhere);
    }

    std::vector<float> reset() {
        logger.clear();
        game.reset();
        // New lane choice per match -- see HeuristicOpponent::reset.
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

    // Either team's CURRENT elixir. Deliberately NOT part of the observation
    // vector -- the opponent's elixir is hidden information a human cannot
    // read off the screen, so a policy conditioned on it could never be
    // deployed through perception/. It is exposed here only as the SUPERVISION
    // TARGET for the network's auxiliary elixir-estimation head: the net sees
    // elapsed time and both sides' cumulative spend (see the appended scalars
    // in extractObservationForTeam) and is trained to infer this from them,
    // which is exactly the elixir counting strong human players do by hand.
    float getElixirForTeam(int team) const {
        return game.getElixir(team);
    }

    bool isGameOver() const {
        return game.isGameOver() || currentTick >= maxTicks;
    }

    // Champion ability: true iff `team` currently has a living, deployed
    // Champion whose ability is off cooldown AND affordable right now.
    // Exposed as a plain accessor rather than folded into the flat
    // observation vector returned by step()/reset() -- growing that vector
    // would silently break python_ai/model.py's fixed scalar_size formula
    // (self.scalar_size = 1 + hand_size + hand_size*num_card_ids, computed
    // independently of observationSize() -- see MatchStatistics-adjacent
    // discussion in the Champion plan for why this stays out of the
    // vector).
    // slot: 1 = Heroic, 2 = Wild Card (see CardRegistry::validateDeckSlots) --
    // defaults to 1 so any single-Champion-in-slot-1 caller keeps compiling
    // and behaving unchanged. The two slots are fully independent.
    bool isChampionAbilityReady(int team, int slot = 1) const {
        return game.isChampionAbilityReady(team, slot);
    }

    // Activates `team`'s deployed Champion's ability in the given slot, if
    // any -- see GameManager::activateChampionAbility. Exposed directly
    // (not just via step()'s activateAbilitySlot1/2 params below) so tests /
    // ad hoc scripts can trigger it without going through a full
    // skip_frames window.
    bool activateChampionAbility(int team, int slot = 1) {
        return game.activateChampionAbility(team, slot);
    }

    // activateAbilitySlot1/2 correspond to deck slots 1 (Heroic) and 2
    // (Wild Card) -- see GameManager::activateChampionAbility's own slot
    // parameter. Independent triggers: either, both, or neither can fire
    // the same step.
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

    // team 1's own point of view (mirrored) -- see extractObservationForTeam.
    // Used by self-play: after reset()/step(), the trainer also needs an
    // observation to feed whatever policy (frozen historical snapshot, or a
    // second live network) is driving team 1 this step.
    std::vector<float> getObservationForTeam(int team) {
        return extractObservationForTeam(team);
    }

    // Test-only escape hatch: ClashEnv wraps GameManager privately (Python
    // only ever drives it through step()/reset()), but tests that need to
    // force a specific hand slot -- e.g. after hand randomization made the
    // opening hand's exact contents non-deterministic -- have no other way
    // to reach playerAI/playerOpponent the way GameManager-level tests
    // already do directly. Not meant for anything but tests.
    GameManager& debugGame() { return game; }

    // Self-play stepping: BOTH sides' actions are supplied externally instead
    // of team 1 being driven by the built-in random opponentTurn() (which
    // this does NOT call at all). targetY1 arrives in team 1's own local
    // frame -- the same mirrored frame its observation came in -- and is
    // converted back to real board Y before actually placing anything;
    // mirroring twice is the identity, so the same formula that produced the
    // observation also inverts it here.
    SelfPlayStepResult stepSelfPlay(int cardIndex0, float targetX0, float targetY0,
                                     int cardIndex1, float targetX1, float targetY1,
                                     int skipFrames = 10,
                                     bool activateAbility0Slot1 = false, bool activateAbility0Slot2 = false,
                                     bool activateAbility1Slot1 = false, bool activateAbility1Slot2 = false) {
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

        return { extractObservationForTeam(0), extractObservationForTeam(1), totalReward, isDone };
    }

    void injectEnemy(int cardId, float x, float y) {
        const auto* card = CardRegistry::getInstance().getCard(cardId);
        if (card) {
            card->spawnEntity(x, y, 1, game.getBoard());
        }
    }

    // General form of injectEnemy above -- kept as a separate method (not a
    // refactor of injectEnemy into a team=1 call) so no existing caller
    // changes. Requested by perception/ (see its own UPSTREAM_REQUESTS.md)
    // as a state-estimator primitive: spawns directly, bypassing hand/
    // elixir/placement legality entirely, which is correct for an
    // estimator replaying placements the real game already validated.
    void inject(int cardId, float x, float y, int team) {
        const auto* card = CardRegistry::getInstance().getCard(cardId);
        if (card) {
            card->spawnEntity(x, y, team, game.getBoard());
        }
    }

    // getHand() above is team-0-only; this is the general form, requested
    // alongside inject() so an estimator can read either side's hand
    // without a second, parallel accessor per team.
    std::vector<int> getHandForTeam(int team) const {
        return game.getHand(team);
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
    int getTowerDamageDealt(int team) const { return game.getStatistics().towerDamageDealt(team); }
    // Heuristic-1 inputs: elixir value a given card has destroyed, and the
    // elixir it has been spent on (casts = spend / cost). Both are needed --
    // rewarding only the value destroyed makes a whiffed spell FREE, which is
    // the same guaranteed-zero trap that put the Cannon in a back corner.
    float getElixirValueKilledBy(int cardId, int team) const {
        return game.getStatistics().elixirValueKilledBy(cardId, team);
    }
    float getElixirSpentOnCard(int cardId, int team) const {
        return game.getStatistics().elixirSpentByCard(cardId, team);
    }
    float getElixirSpent(int team) const { return game.getStatistics().elixirSpent(team); }
};

// Every id CardRegistry actually has registered right now (real, playable
// cards only -- death/periodic/secondary child-unit stats like Golemite or
// Ram Rider's crossbow use negative sentinel ids and are never add()-ed, so
// they never appear here). A free function, not a ClashEnv method, since
// CardRegistry is a singleton independent of any particular env instance.
//
// Exists so Python-side random-deck sampling (gym_wrapper.py, train.py) can
// derive its card pool from whatever's actually registered instead of a
// hardcoded id range + exclusion list that silently drifts out of sync the
// next time a card is added to (or removed from) CardRegistry.h -- exactly
// what happened here: a hardcoded range(46) pool went stale the moment the
// roster grew to 114 registered cards, and a hand-maintained exclusion list
// is exactly the kind of thing that's easy to get wrong in the other
// direction too (mistaking real registered ids for gaps).
// Evolution-slot ids (see CardRegistry::addEvolution) are deliberately
// excluded -- they're not an independently-playable 8th-of-a-deck card,
// they're an upgrade equipped onto whichever base card already occupies a
// slot. Without this filter, random-deck sampling (train.py's
// RANDOM_DECK_POOL, gym_wrapper.py's randomize_opp_deck) would start
// picking them as if they were ordinary standalone cards.
inline std::vector<int> getAllCardIds() {
    std::vector<int> ids;
    for (const auto& [id, def] : CardRegistry::getInstance().getAllCards()) {
        if (def.isEvolution) continue;
        ids.push_back(id);
    }
    return ids;
}

// Builds a random 8-card deck that's ALWAYS validateDeckSlots-legal by
// construction, not by rejection-sampling and retrying -- buckets every
// registered card into plain/Evolution/special-unit pools once, then fills
// each deck slot only from whichever pools CardRegistry::validateDeckSlots
// actually allows there (slot 0: plain+Evolution, slot 1: plain+special
// unit, slot 2: plain+Evolution+special unit, slots 3-7: plain only),
// independently rolling a moderate chance per eligible slot of using a
// special unit/Evolution instead of defaulting to plain. "Special unit"
// means Champion OR Hero (see CardDefinition::isHero's own comment --
// Heroes and Champions share the same two deck slots), bucketed together
// since they're equally eligible everywhere validateDeckSlots allows one.
// That chance is a training-curriculum judgment call (not sourced from the
// game itself) picked so random decks aren't artificially special-unit/
// Evolution-heavy compared to a real deck. Never repeats a card within one
// deck (a real deck can't either) -- always possible here since every pool
// comfortably exceeds the at-most-2 special cards any single deck could
// ever need.
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
    // Slot 1: Heroic slot (Champion/Hero-eligible).
    deck[1] = (!specialUnitPool.empty() && chance01(rng) < SPECIAL_SLOT_CHANCE)
        ? pickFrom(specialUnitPool) : pickFrom(plainPool);
    // Slot 2: Wild Card slot (Champion/Hero OR Evolution-eligible) -- roll
    // once for "special or not", then once more for which kind if so.
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