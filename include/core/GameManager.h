#pragma once
#include "Board.h"
#include "PlayerState.h"
#include "Tower.h"
#include "CombatEntity.h"
#include "MatchRules.h"
#include "MatchStatistics.h"
#include "TowerTroops.h"
#include "CardFactories.h"
#include <algorithm>
#include <vector>
#include <string>

class GameManager {
public:
    // Reserved cardId sentinels for Towers -- they're built directly here,
    // never through CardRegistry, so they need a stable, non-clashing id of
    // their own for stats collectors to key on instead of string-matching
    // `name`. Negative so they can never collide with a real CardRegistry id.
    static constexpr int TOWER_KING_ID = -2;
    static constexpr int TOWER_PRINCESS_ID = -3;

private:
    Board board;
    MatchStatistics stats;
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
    // Per-match config, like the deck itself -- not a step() action (see
    // TowerTroops.h). None (the default) reproduces this engine's
    // original hardcoded Princess Tower exactly.
    TowerTroopType aiTowerTroop = TowerTroopType::None;
    TowerTroopType oppTowerTroop = TowerTroopType::None;

    void addTower(float x, float y, int hp, int team, float attackRange, int damage, int attackCooldown,
        char symbol, const std::string& towerName) {
        auto tower = std::make_shared<Tower>(board.allocateId(), x, y, hp, team, attackRange, damage, attackCooldown, symbol);
        tower->name = towerName;
        tower->cardId = (symbol == 'R') ? TOWER_KING_ID : TOWER_PRINCESS_ID;
        board.addEntity(tower);
    }

    // Tower Troops overload: builds a Princess Tower from a CardStats
    // (see TowerTroops.h) instead of individual hp/range/damage/cooldown
    // args, then reuses CardFactories::applyCardMetadata to wire whatever
    // extra fields that troop's stats carry (Dagger Duchess's burst,
    // Royal Chef's periodic buff) -- the same generic metadata-copy every
    // CardRegistry-spawned entity already gets, even though Towers aren't
    // registered in CardRegistry. Symbol stays 'P' for every variant:
    // web/viewer.html sizes an entity's footprint by symbol, and
    // MatchRules only checks 'R' for win conditions, so varying the
    // symbol per troop would shrink 3 of the 4 to the wrong footprint for
    // no benefit -- only name/stats vary.
    void addTower(float x, float y, int team, const std::string& towerName, const CardStats& stats) {
        auto tower = std::make_shared<Tower>(board.allocateId(), x, y, stats.hp, team, stats.attackRange, stats.damage, stats.attackCooldown, 'P');
        // Order matters: applyCardMetadata sets name/cardId from `stats`
        // too (both left at their CardStats defaults, empty/-1), so the
        // real values are assigned after, not before.
        CardFactories::applyCardMetadata(tower, stats);
        tower->name = towerName;
        tower->cardId = TOWER_PRINCESS_ID;
        board.addEntity(tower);
    }

    // Shared scan for both activateChampionAbility and
    // isChampionAbilityReady: "find team's one deployed, living Champion,
    // if any." This engine doesn't enforce the real game's "only one
    // Champion in your deck" rule at all -- see countChampions() in
    // CardRegistry.h, which a caller can use to check a deck before
    // handing it to GameManager. If more than one Champion is somehow
    // alive at once, board-scan order is NOT arbitrary: Board::activeEntities
    // is append-only and cleanDeadEntities()'s erase-remove preserves
    // relative order, so this deterministically returns whichever surviving
    // Champion was deployed earliest.
    std::shared_ptr<CombatEntity> findChampion(int team) const {
        for (const auto& entity : board.getEntities()) {
            if (entity->team != team || !entity->isAlive()) continue;
            auto combatEntity = std::dynamic_pointer_cast<CombatEntity>(entity);
            if (combatEntity && combatEntity->isChampion) return combatEntity;
        }
        return nullptr;
    }

public:
    PlayerState playerAI;
    PlayerState playerOpponent;

    GameManager(const std::vector<int>& aiDeck, const std::vector<int>& opponentDeck,
            TowerTroopType aiTowerTroopType = TowerTroopType::None,
            TowerTroopType oppTowerTroopType = TowerTroopType::None)
        : gameOver(false), loserTeam(-1) {
        aiDeckConfig = aiDeck;
        oppDeckConfig = opponentDeck;
        aiTowerTroop = aiTowerTroopType;
        oppTowerTroop = oppTowerTroopType;
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
    const MatchStatistics& getStatistics() const { return stats; }

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

    bool isValidPlacement(int team, float x, float y, bool isSpell, float placedRadius, bool deployAnywhere = false) const {
        float maxX = static_cast<float>(board.getWidth() - 1);
        float maxY = static_cast<float>(board.getHeight() - 1);
        if (x < 0.0f || x > maxX || y < 0.0f || y > maxY) return false;
        if (board.isBackRowDeadZone(x, y)) return false;

        if (!isSpell) {
            // Miner/Goblin Drill skip the own-half restriction (they can
            // deploy anywhere on the board) but still can't overlap an
            // existing building -- that check runs unconditionally below.
            if (!deployAnywhere) {
                if (team == 0 && y > board.getRiverStart() - OWN_HALF_RIVER_BUFFER) return false;
                if (team == 1 && y < board.getRiverEnd() + OWN_HALF_RIVER_BUFFER) return false;
            }

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

        if (!isValidPlacement(team, x, y, cardDef->isSpell, cardDef->placementRadius, cardDef->deployAnywhere)) return false;

        PlayerState::PlayCardResult result = player.playCard(handIndex);
        if (result.cardId != -1) {
            if (result.useEvolvedForm && cardDef->spawnEvolvedEntity) {
                cardDef->spawnEvolvedEntity(x, y, team, board);
            } else {
                cardDef->spawnEntity(x, y, team, board);
            }
            board.statsEvents.notifyCardPlayed({ team, result.cardId, cardDef->cost, x, y, currentTick });
            player.lastPlayedCardId = result.cardId;
            return true;
        }
        return false;
    }

    // Read-only: whether `team` could successfully activate its Champion's
    // ability right now (deployed, off cooldown, affordable) without
    // actually doing so -- exposed for ClashEnv to surface without
    // mutating state.
    bool isChampionAbilityReady(int team) const {
        auto champion = findChampion(team);
        if (!champion || !champion->abilityEffect) return false;
        if (champion->abilityCooldownRemaining > 0) return false;
        if (champion->abilityUsesRemaining == 0) return false;
        const PlayerState& player = (team == 0) ? playerAI : playerOpponent;
        return player.elixir >= champion->abilityElixirCost;
    }

    // Activates `team`'s deployed Champion's ability (e.g. Mighty Miner's
    // "Explosive Escape") -- distinct from playCard, which places a NEW
    // card from hand onto an empty spot. Mirrors playCard's own shape:
    // returns bool, never throws, deducts elixir only once success is
    // already guaranteed (find the champion, confirm cooldown/elixir, THEN
    // deduct and fire) so a failed call never partially spends resources.
    bool activateChampionAbility(int team) {
        if (gameOver) return false;

        auto champion = findChampion(team);
        if (!champion || !champion->abilityEffect) return false;
        if (champion->abilityCooldownRemaining > 0) return false;

        PlayerState& player = (team == 0) ? playerAI : playerOpponent;
        if (player.elixir < champion->abilityElixirCost) return false;

        player.elixir -= champion->abilityElixirCost;
        champion->activateAbility(board);
        board.statsEvents.notifyChampionAbilityActivated({ team, champion->cardId, champion->abilityElixirCost, currentTick });
        return true;
    }

    void reset() {
        board = Board();
        // First line after replacing board -- board = Board() destroys the
        // old Board's statsEvents subscriber list along with it, so stats
        // needs fresh collectors subscribed to the *new* bus before
        // anything below (the addTower() spawns, the final
        // commitPendingEntities()) fires a single event.
        stats.attach(board);
        currentTick = 0;
        gameOver = false;
        loserTeam = -1;

        playerAI.initializeDeck(aiDeckConfig);
        playerOpponent.initializeDeck(oppDeckConfig);

        // King Tower is rendered as a 4x4-tile footprint (see web/viewer.html's
        // sizeInTiles), which only sits flush on whole tile boundaries when
        // centered on a half-integer coordinate (a 4-wide span covering tiles
        // i..i+3 runs from i-0.5 to i+3.5, so its center is always X.5) --
        // these were previously on whole-integer coordinates, straddling
        // tile boundaries. Corrected per-team by the actual visual offset
        // needed (the two sides weren't symmetric to begin with), not a
        // shared mirror formula. Princess Tower positions are unaffected by
        // the alignment fix (3-wide footprint, already correctly aligned);
        // the Red-side princesses moved slightly only to match the King's
        // corrected position, preserving the two teams' visual symmetry.
        //
        // Real-map sync: the board grew by one back row per side (see
        // Board.h's BACK_ROW_OPENING_HALF_WIDTH), and the King Tower moved
        // back exactly one tile into that new space -- Blue's Y is
        // numerically unchanged (2.5) only because the board's own new row
        // 0 already accounts for the other tile of growth; Red mirrors it
        // via (height-1) - y = 33 - 2.5 = 30.5, same convention as every
        // other team-mirroring formula in this engine (e.g.
        // ClashEnv::extractObservationForTeam). Princess Towers didn't move
        // independently -- their Y values below are just the same +1 board-
        // growth carry-along every pre-existing coordinate got.
        addTower(8.5f, 2.5f, 4008, 0, 7.0f, 90, 10, 'R', "King Tower");
        addTower(8.5f, 30.5f, 4008, 1, 7.0f, 90, 10, 'R', "King Tower");

        addTower(3.0f, 6.0f, 0, "Princess Tower", towerTroopStats(aiTowerTroop));
        addTower(14.0f, 6.0f, 0, "Princess Tower", towerTroopStats(aiTowerTroop));
        addTower(3.0f, 27.0f, 1, "Princess Tower", towerTroopStats(oppTowerTroop));
        addTower(14.0f, 27.0f, 1, "Princess Tower", towerTroopStats(oppTowerTroop));

        board.commitPendingEntities(currentTick);
    }

    void step() {
        if (gameOver) return;

        currentTick++;
        board.currentTick = currentTick;

        playerAI.elixir = std::min(playerAI.elixir + ELIXIR_REGEN_RATE, 10.0f);
        playerOpponent.elixir = std::min(playerOpponent.elixir + ELIXIR_REGEN_RATE * oppElixirMultiplier, 10.0f);

        board.commitPendingEntities(currentTick);

        for (auto& entity : board.getEntities()) {
            if (entity->isAlive()) entity->update(board);
        }

        // Elixir Collector: drain whatever ElixirGrantEffect accumulated
        // this tick into the real PlayerState, then reset -- same cap as
        // normal regen above.
        playerAI.elixir = std::min(playerAI.elixir + board.pendingElixirGrant[0], 10.0f);
        playerOpponent.elixir = std::min(playerOpponent.elixir + board.pendingElixirGrant[1], 10.0f);
        board.pendingElixirGrant[0] = 0.0f;
        board.pendingElixirGrant[1] = 0.0f;

        board.commitPendingEntities(currentTick);
        board.resolveCollisions();

        MatchRules::Outcome outcome = MatchRules::evaluate(board);
        if (outcome.over) {
            gameOver = true;
            loserTeam = outcome.loserTeam;
            board.statsEvents.notifyMatchEnded({ loserTeam, currentTick });
        }

        board.cleanDeadEntities(currentTick);
    }
};