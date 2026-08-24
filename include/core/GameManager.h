#pragma once
#include "ArenaLayout.h"
#include "Board.h"
#include "PlayerState.h"
#include "Tower.h"
#include "CombatEntity.h"
#include "MatchRules.h"
#include "MatchStatistics.h"
#include "TowerTroops.h"
#include "CardFactories.h"
#include "SpiritEmpressForms.h"
#include <algorithm>
#include <vector>
#include <string>
#include <stdexcept>
#include <random>
// std::as_const, used by the non-const findTower below. It was reaching this
// header transitively and compiled inside the Catch2 suite by luck; any
// translation unit that does not pull <utility> in some other way failed with
// "'as_const' is not a member of 'std'" -- tools/audit/*.cpp did, immediately.
#include <utility>

class GameManager {
public:
    // Reserved cardId sentinels for Towers -- they're built directly here,
    // never through CardRegistry, so they need a stable, non-clashing id of
    // their own for stats collectors to key on instead of string-matching
    // `name`. Negative so they can never collide with a real CardRegistry id.
    static constexpr int TOWER_KING_ID = -2;
    static constexpr int TOWER_PRINCESS_ID = -3;
    // Mirror's own registered CardRegistry id (see CardRegistry.h) --
    // named here since playCard special-cases it directly.
    static constexpr int MIRROR_CARD_ID = 164;
    // Spirit Empress's own registered CardRegistry id -- see
    // SpiritEmpressForms.h and playCard's own dynamic-cost branch.
    static constexpr int SPIRIT_EMPRESS_CARD_ID = 165;

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
    // Seeded once (not on every reset() -- same pattern as ClashEnv::rng),
    // so std::shuffle draws a genuinely fresh permutation each episode
    // instead of the same one every reset. Feeds PlayerState::initializeDeck's
    // random-hand overload.
    std::mt19937 rng;

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
        // This overload bypasses CardStats/applyCardMetadata entirely (raw
        // hp/range/damage args), so sightRange needs setting directly --
        // only ever called for the King Tower (symbol 'R'), sourced at
        // 7.0 tiles. The Tower Troops overload below sets its own from
        // towerTroopStats() instead.
        if (symbol == 'R') {
            tower->sightRange = 7.0f;
            // The King starts DORMANT -- see Tower::isAwake. This is the only
            // construction path that ever builds a King, so it is the only
            // place the flag needs clearing; every other Tower is a Princess
            // and stays awake.
            tower->sleep();
        }
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

    // Resolves a specific Champion slot (1 = Heroic, 2 = Wild Card -- see
    // CardRegistry::validateDeckSlots/PlayerState::ChampionSlotState) to its
    // currently-tracked live entity, if any. Deliberately NOT a generic
    // "find any isChampion entity" board scan (that was the old
    // findChampion(team), now removed) -- resolving strictly through
    // championSlots[slot].trackedEntityId is what makes a Clone-spell
    // duplicate permanently unable to activate an ability (it's never
    // created via playCard, so it can never become a tracked id) and what
    // makes "whoever was created last" the one that answers, with zero
    // extra special-casing needed for either rule.
    std::shared_ptr<CombatEntity> findChampionInSlot(int team, int slot) const {
        const PlayerState& player = (team == 0) ? playerAI : playerOpponent;
        auto it = player.championSlots.find(slot);
        if (it == player.championSlots.end() || it->second.trackedEntityId == -1) return nullptr;
        for (const auto& entity : board.getEntities()) {
            if (entity->id == it->second.trackedEntityId) {
                if (!entity->isAlive()) return nullptr;
                return std::dynamic_pointer_cast<CombatEntity>(entity);
            }
        }
        return nullptr;
    }

    // Called once per tick (see step()) for each team: mirrors the tracked
    // Champion's own live abilityCooldownRemaining into
    // persistedCooldownRemaining, so that value is always fresh right up
    // until the entity dies -- at which point it simply stops being
    // updated, and that last-known value is what seeds a fresh redeploy of
    // the same slot's Champion (see playCard's own tracking hook), giving
    // "redeploying while on cooldown just continues the timer" for free.
    void syncChampionCooldowns(int team) {
        PlayerState& player = (team == 0) ? playerAI : playerOpponent;
        const std::vector<int>& deckConfig = (team == 0) ? aiDeckConfig : oppDeckConfig;
        for (auto& entry : player.championSlots) {
            int slot = entry.first;
            PlayerState::ChampionSlotState& state = entry.second;
            if (state.trackedEntityId == -1) continue;
            for (const auto& entity : board.getEntities()) {
                if (entity->id != state.trackedEntityId) continue;
                if (entity->isAlive()) {
                    if (auto ce = std::dynamic_pointer_cast<CombatEntity>(entity)) {
                        state.persistedCooldownRemaining = ce->abilityCooldownRemaining;
                    }
                }
                break;
            }

            // Post-death squad reactivation (Hero Goblins' "Banner
            // Brigade"): independent of the single tracked-instance scan
            // above -- a multi-unit squad has several live entities
            // sharing this slot's own cardId+team, only one of which is
            // ever "tracked", so this scans for ANY of them, opening the
            // window only once the WHOLE squad is confirmed gone. Called
            // BEFORE board.cleanDeadEntities() (see step()'s own comment),
            // so an entity that died THIS tick is still present here with
            // isAlive()==false and its death position intact.
            const CardDefinition* slotDef = CardRegistry::getInstance().getCard(deckConfig[slot]);
            if (!slotDef || slotDef->abilityUsableAfterDeathTicks <= 0) continue;
            bool anyFound = false; // at least one entity matching this slot's cardId+team seen THIS tick
            bool anyAlive = false;
            Vector2D deathPosition = state.lastSquadWipePosition;
            for (const auto& entity : board.getEntities()) {
                if (entity->cardId != deckConfig[slot] || entity->team != team) continue;
                anyFound = true;
                if (entity->isAlive()) { anyAlive = true; break; }
                deathPosition = entity->position;
            }
            if (anyAlive) {
                state.lastSquadWipeTick = -1; // a fresh deploy invalidates any earlier, unconsumed window
            } else if (anyFound && state.lastSquadWipeTick < 0) {
                // anyFound guards against re-arming the window forever: once
                // the dead squad's own entities are erased by this same
                // step()'s later cleanDeadEntities() call, later ticks see
                // NO entity of this cardId+team at all (anyFound == false)
                // -- without this guard, that "nothing found" state would
                // be indistinguishable from "just wiped" and would keep
                // resetting lastSquadWipeTick to the current tick forever,
                // letting Banner Brigade be spammed indefinitely for 1
                // elixir a pop instead of firing exactly once per real wipe.
                state.lastSquadWipeTick = currentTick; // first tick nothing's left alive
                state.lastSquadWipePosition = deathPosition;
            }
        }
    }

    // Post-death squad reactivation readiness (Hero Goblins-style) -- shared
    // by isChampionAbilityReady/activateChampionAbility, both of which fall
    // back to this once findChampionInSlot finds nothing alive in the slot.
    bool isPostDeathAbilityReady(int team, int slot) const {
        const PlayerState& player = (team == 0) ? playerAI : playerOpponent;
        auto it = player.championSlots.find(slot);
        if (it == player.championSlots.end() || it->second.lastSquadWipeTick < 0) return false;
        const std::vector<int>& deckConfig = (team == 0) ? aiDeckConfig : oppDeckConfig;
        const CardDefinition* slotDef = CardRegistry::getInstance().getCard(deckConfig[slot]);
        if (!slotDef || slotDef->abilityUsableAfterDeathTicks <= 0 || !slotDef->postDeathAbilityEffect) return false;
        if (currentTick - it->second.lastSquadWipeTick > slotDef->abilityUsableAfterDeathTicks) return false;
        return player.elixir >= slotDef->abilityElixirCost;
    }

public:
    PlayerState playerAI;
    PlayerState playerOpponent;

    GameManager(const std::vector<int>& aiDeck, const std::vector<int>& opponentDeck,
            TowerTroopType aiTowerTroopType = TowerTroopType::None,
            TowerTroopType oppTowerTroopType = TowerTroopType::None)
        : gameOver(false), loserTeam(-1), rng(std::random_device{}()) {
        aiDeckConfig = aiDeck;
        oppDeckConfig = opponentDeck;
        aiTowerTroop = aiTowerTroopType;
        oppTowerTroop = oppTowerTroopType;
        reset();
    }

    // Pin the generator that deals the OPENING HAND.
    //
    // This is the one that matters and it is NOT ClashEnv::rng: that member
    // feeds HeuristicOpponent only, while reset() below deals both players'
    // hands -- and their starting deckQueue ORDER -- from this one via
    // PlayerState::initializeDeck. Seeding the other generator alone leaves
    // the hand exactly as random as before, which reads as "seeding does not
    // work" rather than "the wrong generator was seeded".
    //
    // Takes effect on the NEXT reset(); ClashEnv::seed calls one for you.
    void seed(unsigned int s) { rng.seed(s); }

    void setOpponentDeck(const std::vector<int>& deck) {
        std::string err = validateDeckSlots(deck);
        if (!err.empty()) throw std::invalid_argument("GameManager::setOpponentDeck: invalid deck -- " + err);
        oppDeckConfig = deck;
    }

    void setOpponentElixirMultiplier(float multiplier) {
        oppElixirMultiplier = std::max(0.0f, multiplier);
    }

    Board& getBoard() { return board; }
    const Board& getBoard() const { return board; }
    const MatchStatistics& getStatistics() const { return stats; }

    // Fully independent copy of this match, for decision-time search: step the
    // result as far as you like and nothing about `this` changes. The board
    // half is Board::deepCopy(); everything else here is already a value type
    // and copies correctly on its own.
    //
    // What rides along on the implicit copy, and why each is right:
    //   currentTick / gameOver / loserTeam   plain scalars
    //   oppElixirMultiplier                  plain scalar (curriculum setting)
    //   aiDeckConfig / oppDeckConfig         vector<int>
    //   aiTowerTroop / oppTowerTroop         enums
    //   playerAI / playerOpponent            elixir, hand, handCooldownTicks,
    //                                        deckQueue, evolutionState,
    //                                        championSlots -- all values, so
    //                                        a rollout cycles its own deck and
    //                                        spends its own elixir
    //   rng                                  COPIED, not reseeded: two
    //                                        snapshots of the same position
    //                                        must roll out identically, or a
    //                                        search would be comparing
    //                                        candidates across different
    //                                        futures and scoring noise
    //
    // Only `board` and `stats` need fixing up, and both for the same reason --
    // they are the only members holding shared_ptr, so the implicit copy
    // aliases rather than duplicates them.
    //
    // Implemented as copy-then-replace rather than a member-by-member
    // constructor deliberately: a new GameManager field then joins the
    // snapshot automatically, whereas an explicit list would silently omit it.
    // The intermediate shallow board exists only between these two statements,
    // and nothing is stepped in that window.
    GameManager snapshot() const {
        GameManager copy(*this);
        copy.board = board.deepCopy();
        copy.stats = stats.snapshotFor(copy.board);
        return copy;
    }

    // Exact bounds isValidPlacement enforces, exposed so callers (the Python
    // binding layer) query the real boundary instead of re-deriving it from
    // separate board/river/buffer constants that could silently drift out of
    // sync (confirmed painful in practice -- this project's own map-geometry
    // and NUM_CARD_IDS incidents were both exactly this kind of drift, just
    // for other constants).
    float getMaxPlacementX() const {
        return static_cast<float>(board.getWidth() - 1);
    }

    // "How far into your own half can a non-spell, non-deploy-anywhere card
    // be aimed" -- symmetric for both teams from each one's own point of view
    // (extractObservationForTeam mirrors team 1's board so it sees itself the
    // same way team 0 does), so one value covers both sides' action-space
    // scaling.
    float getOwnHalfMaxY() const {
        return board.getRiverStart() - OWN_HALF_RIVER_BUFFER;
    }

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

    // --- STATE-ESTIMATOR SETTERS (2026-08-17) ------------------------------
    // Write a reconstructed live state into the simulator, so decision-time
    // search evaluates the REAL position rather than a reset one.
    //
    // Why these have to exist. perception/ already reads our elixir accurately
    // (587 samples, mean confidence 0.990) and reports the hand, but
    // forecast.py rebuilds a board by calling reset() and injecting units --
    // and reset() sets elixir to 5.0 and deals a hand from an UNSEEDED
    // std::mt19937. So the reconstructed position had the right units and a
    // fabricated hand/elixir, and `affordability_mask` is built from exactly
    // those scalars. Search over it would score fiction. The existing
    // workaround (perception/bridge/sim_driver.py) reverse-engineers the
    // shuffle by drawing resets until the hand matches, which is expensive and
    // only ever recovers the OPENING hand.
    //
    // Deliberately on GameManager and not on PlayerState alone: the hand and
    // the deck queue are one invariant (together they are a permutation of the
    // 8-card deck), and letting a caller set one without the other is how that
    // invariant rots.
    void setElixir(int team, float value) {
        PlayerState& player = (team == 0) ? playerAI : playerOpponent;
        // Same [0, 10] clamp tick() enforces -- a caller handing us 11 elixir
        // (or a negative from a bad estimate) must not create a state the
        // engine itself can never reach.
        player.elixir = std::max(0.0f, std::min(value, 10.0f));
    }

    // Returns FALSE and changes nothing if `cards` is not a valid hand for
    // this team's deck. Loudly rejecting is the point: perception's card
    // identity is the least reliable reading it produces, and a silently
    // accepted misread would put the policy in a position the real game is not
    // in -- the exact failure mode this project keeps paying for.
    //
    // Rejects: wrong size, a duplicate, or a card that is not in this team's
    // deck at all.
    bool setHand(int team, const std::vector<int>& cards) {
        PlayerState& player = (team == 0) ? playerAI : playerOpponent;
        if (cards.size() != player.hand.size()) return false;

        // The deck is whatever is currently in hand plus whatever is queued --
        // read from the live state rather than from the original deck list, so
        // this stays correct mid-match after any amount of cycling.
        std::vector<int> pool(player.hand.begin(), player.hand.end());
        pool.insert(pool.end(), player.deckQueue.begin(), player.deckQueue.end());

        std::vector<int> remaining = pool;
        for (int card : cards) {
            auto it = std::find(remaining.begin(), remaining.end(), card);
            if (it == remaining.end()) return false;   // not in deck, or duplicate
            remaining.erase(it);
        }

        player.hand.assign(cards.begin(), cards.end());
        // Queue keeps the relative order the remaining cards already had, so a
        // caller that sets the hand to what it already was is a no-op on the
        // cycle rather than a silent reshuffle.
        player.deckQueue.assign(remaining.begin(), remaining.end());
        // Cooldowns cleared, and this is a judgement call worth stating: a
        // freshly-cycled slot is unplayable for 20 ticks, but perception cannot
        // observe that. Zero is the permissive reading -- it can make search
        // believe a card is playable ~2 s early, whereas a non-zero guess would
        // make it refuse a play the real game allows. Refusing a legal play is
        // the worse error for a policy that is already too passive.
        player.handCooldownTicks.assign(player.hand.size(), 0);
        return true;
    }

    // --- The tower and clock half of the estimator write interface --------
    // perception/UPSTREAM_REQUESTS.md item 22 (2026-08-24). setElixir/setHand
    // above closed elixir and the hand; a mirror rebuilt from a real screen
    // still came back with every tower at full health and the clock at zero.
    //
    // SLOTS ARE BOARD COORDINATES, NEVER TEAM-RELATIVE: 0 = King, 1 = left
    // Princess, 2 = right Princess, left/right decided against ArenaLayout's
    // own centre rather than a restated literal. The caller is a sensor
    // reading a screen, and asking it to mirror its own coordinates per team
    // is the convention error that put the arena half a tile off-centre.
    const Tower* findTower(int team, int slot) const {
        for (const auto& e : board.getEntities()) {
            const Tower* t = dynamic_cast<const Tower*>(e.get());
            if (t == nullptr || t->team != team) continue;
            const bool isKing = (t->symbol == 'R');
            if (slot == 0 && isKing) return t;
            if (slot == 1 && !isKing && t->position.x < ArenaLayout::CENTER_X) return t;
            if (slot == 2 && !isKing && t->position.x > ArenaLayout::CENTER_X) return t;
        }
        return nullptr;
    }

    Tower* findTower(int team, int slot) {
        return const_cast<Tower*>(std::as_const(*this).findTower(team, slot));
    }

    // Returns FALSE and changes nothing on a value the engine cannot hold.
    //
    // hp <= 0 is REFUSED rather than clamped, and that is the whole contract:
    // a 0-hp tower that still occupies its cell and still fires is a position
    // the real game can never be in, and killing one has side effects -- the
    // crown, the King's princess-count trigger, LanePath's retargeting --
    // that belong to destroyTower() below. Same refuse-rather-than-accept
    // rule setHand uses, and for the same reason: a silently accepted misread
    // is worse than none.
    //
    // Takes ENGINE-ABSOLUTE hp. The caller converts, because tower levels do
    // not match -- this engine's towers are level 9 (2534/4008) while a real
    // account's may be level 4-5 (1750/1890), i.e. wrong by a DIFFERENT factor
    // per player. perception/ reports a fraction and multiplies by
    // getTowerMaxHp() below, keeping the level knowledge on the side that
    // already owns it.
    bool setTowerHp(int team, int slot, float hp) {
        Tower* t = findTower(team, slot);
        if (t == nullptr || !t->isAlive()) return false;
        if (hp <= 0.0f) return false;
        int want = static_cast<int>(hp + 0.5f);
        const int ceiling = t->getMaxHp();
        if (want > ceiling) want = ceiling;
        if (want < 1) want = 1;
        t->hp = want;
        // `awake` is deliberately NOT set here. Tower::update latches it on
        // the `hp < maxHp` invariant, so a wound written this way wakes the
        // King through exactly the path real damage uses -- and because the
        // flag is a latch, writing full health back cannot re-sleep it.
        return true;
    }

    // The destruction destroyTower exists to route: setTowerHp refuses hp <= 0,
    // so without this a fallen tower would be inexpressible in a mirror -- and
    // a rollout that still has the tower standing is wrong from the moment it
    // falls, i.e. exactly when the position matters most.
    bool destroyTower(int team, int slot) {
        Tower* t = findTower(team, slot);
        if (t == nullptr || !t->isAlive()) return false;   // idempotent, and says so
        t->takeDamage(t->hp);
        // takeDamage is the right entry point -- it is what a real killing
        // blow uses and it fires onDamageTakenEffect. But CombatEntity's
        // override can ABSORB (shield, parry, mid-dash invulnerability). No
        // Tower carries any of those today; pinning the postcondition means
        // this stays a destruction if one ever does, instead of silently
        // leaving the tower standing.
        if (t->isAlive()) t->hp = 0;
        return true;
    }

    // -1 for a slot that does not exist, so a caller cannot mistake a missing
    // tower for a tower at zero.
    int getTowerHp(int team, int slot) const {
        const Tower* t = findTower(team, slot);
        return t == nullptr ? -1 : t->hp;
    }

    int getTowerMaxHp(int team, int slot) const {
        const Tower* t = findTower(team, slot);
        return t == nullptr ? -1 : t->getMaxHp();
    }

    // Clamped at zero only: the UPPER bound is maxTicks, which lives in
    // ClashEnv, and ClashEnv::setCurrentTick applies it before calling here.
    // Board's mirror moves with it so the two never drift -- board.currentTick
    // stamps every spawn event and drives ability cooldown arithmetic.
    void setCurrentTick(int tick) {
        currentTick = std::max(0, tick);
        board.currentTick = currentTick;
    }

    int getCurrentTick() const { return currentTick; }

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

        // Mirror: placement legality, spawn, and cost all come from
        // whatever this team last played (+1 elixir), not from Mirror's
        // own (otherwise-unused) registration -- reuses the real
        // mirrored card's own rules verbatim (a mirrored Fireball must
        // target the enemy half, a mirrored Knight must not) instead of
        // a bespoke Mirror spawn closure. Fails outright with nothing
        // played yet (lastPlayedCardId == -1), same as any other
        // unaffordable/invalid play.
        bool isMirror = (targetCardId == MIRROR_CARD_ID);
        // Spirit Empress: form (and elixir cost) is deduced fresh from
        // CURRENT elixir at the moment of play, not sticky/ratcheted --
        // documented approximation, see SpiritEmpressForms.h's own
        // comment (the real switching rule isn't clearly sourced even
        // from this project's usual trusted sources). Unlike Mirror, both
        // forms share the same placement footprint (see that header's
        // comment), so no effectiveDef substitution is needed for
        // isValidPlacement -- only cost and which spawn function runs.
        bool isSpiritEmpress = (targetCardId == SPIRIT_EMPRESS_CARD_ID);
        const CardDefinition* effectiveDef = cardDef;
        float costOverride = -1.0f;
        if (isMirror) {
            effectiveDef = CardRegistry::getInstance().getCard(player.lastPlayedCardId);
            if (!effectiveDef) return false;
            // Mirror CAN duplicate a Champion/Hero -- the ability always
            // belongs to whichever instance of that slot's troop was most
            // recently deployed, whether that deployment came from playing
            // the original card or from Mirror (see the tracking hook
            // below, which resolves this off the spawned entity's own
            // cardId rather than off result.cardId, so a Mirror-spawned
            // instance is just as trackable as the original).
            costOverride = effectiveDef->cost + 1.0f;
        } else if (isSpiritEmpress) {
            costOverride = (player.elixir >= 6.0f) ? 6.0f : 3.0f;
        }

        if (!isValidPlacement(team, x, y, effectiveDef->isSpell, effectiveDef->placementRadius, effectiveDef->deployAnywhere)) return false;

        PlayerState::PlayCardResult result = player.playCard(handIndex, costOverride);
        if (result.cardId != -1) {
            size_t pendingBefore = board.pendingEntityCount();
            if (isSpiritEmpress) {
                bool flying = (costOverride >= 6.0f);
                CardFactories::spawn(flying ? spiritEmpressFlyingStats() : spiritEmpressGroundStats(), x, y, team, board);
            } else if (result.useEvolvedForm && effectiveDef->spawnEvolvedEntity) {
                effectiveDef->spawnEvolvedEntity(x, y, team, board);
            } else {
                effectiveDef->spawnEntity(x, y, team, board);
            }
            float reportedCost = (costOverride >= 0.0f) ? costOverride : cardDef->cost;
            board.statsEvents.notifyCardPlayed({ team, result.cardId, reportedCost, x, y, currentTick });
            // A second Mirror replays whatever was played before the
            // FIRST Mirror, not the first Mirror itself -- so a Mirror
            // play must not overwrite lastPlayedCardId with its own id.
            if (!isMirror) player.lastPlayedCardId = result.cardId;

            // Champion/Hero per-slot tracking (see PlayerState::
            // ChampionSlotState): resolved off the freshly-spawned
            // entity's OWN cardId, not result.cardId -- a Mirror play's
            // result.cardId is always Mirror's own id (164), but the
            // entity it actually spawns carries the mirrored (effectiveDef)
            // card's real id (see applyCardMetadata), so checking the
            // entity itself tracks a Mirror-duplicated Champion/Hero just
            // as well as an original play. This gives "the ability always
            // belongs to whichever instance was deployed last" uniformly,
            // matching real-game fidelity -- Mirror CAN duplicate a
            // Champion/Hero, and activating the ability targets whichever
            // copy (original or mirrored) was created most recently. Hero
            // shares this exact same tracking as Champion -- both occupy
            // the same two slots.
            const std::vector<int>& deckConfig = (team == 0) ? aiDeckConfig : oppDeckConfig;
            for (int slot : { 1, 2 }) {
                for (size_t i = pendingBefore; i < board.pendingEntityCount(); ++i) {
                    auto ce = std::dynamic_pointer_cast<CombatEntity>(board.getPendingEntity(i));
                    if (ce && (ce->isChampion || ce->isHero) && ce->cardId == deckConfig[slot]) {
                        PlayerState::ChampionSlotState& slotState = player.championSlots[slot];
                        ce->abilityCooldownRemaining = slotState.persistedCooldownRemaining;
                        slotState.trackedEntityId = ce->id;
                        break;
                    }
                }
            }

            return true;
        }
        return false;
    }

    // Read-only: whether `team`'s Champion in the given slot (1 = Heroic,
    // 2 = Wild Card; defaults to 1 so any single-Champion-in-slot-1 caller
    // keeps compiling and behaving unchanged) could successfully activate
    // its ability right now (deployed, off cooldown, affordable) without
    // actually doing so -- exposed for ClashEnv to surface without
    // mutating state. Two different Champions (slots 1 and 2) are fully
    // independent -- checking/activating one never touches the other's
    // cooldown or elixir cost.
    bool isChampionAbilityReady(int team, int slot = 1) const {
        auto champion = findChampionInSlot(team, slot);
        if (champion && champion->abilityEffect) {
            if (champion->abilityCooldownRemaining > 0) return false;
            if (champion->abilityUsesRemaining == 0) return false;
            const PlayerState& player = (team == 0) ? playerAI : playerOpponent;
            return player.elixir >= champion->abilityElixirCost;
        }
        // Nothing alive in this slot -- fall back to the post-death path
        // (Hero Goblins-style; a no-op false for every other card, since
        // isPostDeathAbilityReady itself requires abilityUsableAfterDeathTicks > 0).
        return isPostDeathAbilityReady(team, slot);
    }

    // Activates `team`'s deployed Champion's ability in the given slot
    // (e.g. Mighty Miner's "Explosive Escape") -- distinct from playCard,
    // which places a NEW card from hand onto an empty spot. Mirrors
    // playCard's own shape: returns bool, never throws, deducts elixir only
    // once success is already guaranteed (find the champion, confirm
    // cooldown/elixir, THEN deduct and fire) so a failed call never
    // partially spends resources.
    bool activateChampionAbility(int team, int slot = 1) {
        if (gameOver) return false;

        auto champion = findChampionInSlot(team, slot);
        if (champion && champion->abilityEffect) {
            if (champion->abilityCooldownRemaining > 0) return false;
            // Matches isChampionAbilityReady's own check -- without this, a
            // uses-limited ability (Boss Bandit; now also Hero Mini P.E.K.K.A's
            // one-use Breakfast Boost) whose abilityCooldownTicks happens to be
            // 0 would deduct elixir and return true on every call even after
            // CombatEntity::activateAbility's own internal uses check makes the
            // activation itself a no-op -- previously masked for Boss Bandit
            // only because his cooldown (30 ticks) is nonzero, so a repeat call
            // was always caught by the cooldown check above first.
            if (champion->abilityUsesRemaining == 0) return false;

            PlayerState& player = (team == 0) ? playerAI : playerOpponent;
            if (player.elixir < champion->abilityElixirCost) return false;

            player.elixir -= champion->abilityElixirCost;
            champion->activateAbility(board);
            board.statsEvents.notifyChampionAbilityActivated({ team, champion->cardId, champion->abilityElixirCost, currentTick });
            return true;
        }

        // Post-death squad reactivation (Hero Goblins-style): only
        // reachable when nothing is currently alive in this slot. Mirrors
        // the alive-path's own shape -- confirm readiness fully (via the
        // shared isPostDeathAbilityReady helper) BEFORE deducting anything,
        // so a failed call never partially spends resources.
        if (!isPostDeathAbilityReady(team, slot)) return false;

        PlayerState& player = (team == 0) ? playerAI : playerOpponent;
        PlayerState::ChampionSlotState& slotState = player.championSlots[slot];
        const std::vector<int>& deckConfig = (team == 0) ? aiDeckConfig : oppDeckConfig;
        const CardDefinition* slotDef = CardRegistry::getInstance().getCard(deckConfig[slot]);

        player.elixir -= slotDef->abilityElixirCost;
        slotDef->postDeathAbilityEffect->apply(board, slotState.lastSquadWipePosition, team);
        slotState.lastSquadWipeTick = -1; // consumed -- until redeployed and wiped again
        board.statsEvents.notifyChampionAbilityActivated({ team, deckConfig[slot], slotDef->abilityElixirCost, currentTick });
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

        std::string aiErr = validateDeckSlots(aiDeckConfig);
        if (!aiErr.empty()) throw std::invalid_argument("GameManager: invalid AI deck -- " + aiErr);
        std::string oppErr = validateDeckSlots(oppDeckConfig);
        if (!oppErr.empty()) throw std::invalid_argument("GameManager: invalid opponent deck -- " + oppErr);

        playerAI.initializeDeck(aiDeckConfig, rng);
        playerOpponent.initializeDeck(oppDeckConfig, rng);

        // X-coordinates below corrected 2026-07-30 per perception/'s
        // UPSTREAM_REQUESTS.md items 1-2, fitted from real-recording
        // homography (screen->tile, 8 landmarks, aggregated over 8 matches):
        //
        //   - Left Princess x: 3.0 -> 4.0. It was a full tile off its own
        //     bridge (also x=4.0) while the right side already agreed with
        //     ITS bridge (both 14.0) -- an internal asymmetry, not a
        //     convention choice. Measured: left tower centre 819px, left
        //     bridge centre 821px, same lane.
        //   - King x: 8.5 -> 9.0. Supersedes this engine's own previous
        //     flush-footprint reasoning (a 4-wide footprint centered on a
        //     half-integer lands on whole tile-boundary lines) now that real
        //     footage gives a directly measured value instead: king centre
        //     955.75px resolves to 8.79 in bridge-calibrated tile
        //     coordinates, closer to 9.0. Moving the King alone is a wash
        //     (it fixes one landmark while the left Princess is still wrong);
        //     combined with the Princess fix above, held-out calibration
        //     error dropped max 0.63 -> 0.31 tiles, rms 0.33 -> 0.21.
        //   - Right side (14.0/14.0) was already correct and is unchanged.
        //
        // Y-coordinates are untouched here (separate history -- see the
        // river re-centring commit): King 2.5<->30.5, Princess 6.0<->27.0,
        // still symmetric under (height-1) - y = 33 - y, same convention as
        // every other team-mirroring formula in this engine (e.g.
        // ClashEnv::extractObservationForTeam).
        // 8.5, not 9.0: x is a cell index in [0, 17], so the board's centre
        // -- and the fixed point of the mirror 17 - x -- is 8.5. A King at
        // 9.0 sat half a tile right of centre on both teams, which is also
        // why the left Princess and the left bridge were each a half tile
        // out. See Board::leftBridge for the same correction.
        addTower(ArenaLayout::CENTER_X, ArenaLayout::kingY(0), 4008, 0, 7.0f, 90, 10, 'R', "King Tower");
        addTower(ArenaLayout::CENTER_X, ArenaLayout::kingY(1), 4008, 1, 7.0f, 90, 10, 'R', "King Tower");

        // 3.0 / 14.0: mirror images under 17 - x, and each flush with its own
        // bridge column (Board::leftBridge / rightBridge).
        addTower(ArenaLayout::LEFT_LANE_X,  ArenaLayout::princessY(0), 0, "Princess Tower", towerTroopStats(aiTowerTroop));
        addTower(ArenaLayout::RIGHT_LANE_X, ArenaLayout::princessY(0), 0, "Princess Tower", towerTroopStats(aiTowerTroop));
        addTower(ArenaLayout::LEFT_LANE_X,  ArenaLayout::princessY(1), 1, "Princess Tower", towerTroopStats(oppTowerTroop));
        addTower(ArenaLayout::RIGHT_LANE_X, ArenaLayout::princessY(1), 1, "Princess Tower", towerTroopStats(oppTowerTroop));

        board.commitPendingEntities(currentTick);
    }

    void step() {
        if (gameOver) return;

        currentTick++;
        board.currentTick = currentTick;

        playerAI.elixir = std::min(playerAI.elixir + ELIXIR_REGEN_RATE, 10.0f);
        playerOpponent.elixir = std::min(playerOpponent.elixir + ELIXIR_REGEN_RATE * oppElixirMultiplier, 10.0f);
        playerAI.tick();
        playerOpponent.tick();

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

        // After this tick's entity updates (so this tick's cooldown
        // decrement is captured) but before dead entities are erased (so a
        // Champion that died this very tick still gets its final cooldown
        // value persisted) -- see PlayerState::ChampionSlotState's comment.
        syncChampionCooldowns(0);
        syncChampionCooldowns(1);

        board.cleanDeadEntities(currentTick);
    }
};