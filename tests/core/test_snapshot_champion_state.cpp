#include <catch_amalgamated.hpp>
#include "GameManager.h"
#include "CardRegistry.h"
#include "CombatEntity.h"
#include <vector>

// Champion/Hero slot state through GameManager::snapshot().
//
// WHY THIS FILE EXISTS
// --------------------
// The existing snapshot suites (test_board_deepcopy.cpp,
// test_game_manager_snapshot.cpp) and the existing Champion suites
// (test_champion_abilities.cpp, test_hero_abilities.cpp) have ZERO overlap:
// every snapshot fixture uses an all-plain deck, so PlayerState::seedSlotState
// leaves `championSlots` EMPTY in all of them, and every Champion test builds a
// GameManager it never copies.
//
// That leaves the most copy-hostile structure in PlayerState untested across a
// copy. `championSlots` holds `trackedEntityId`, an id that means nothing on
// its own -- `findChampionInSlot` resolves it by scanning `board.getEntities()`
// for a matching `entity->id`. So it is a cross-structure invariant: an int
// living in PlayerState that must keep pointing at the right object in Board,
// across a deepCopy that rebuilds every one of those objects.
//
// Board::deepCopy preserves ids (it copies them rather than re-allocating), so
// the invariant holds today -- these are regression guards, not a bug report.
// The failure they exist to catch is silent in the worst way: a search rollout
// whose Champion is simply unresolvable would decline every ability activation
// and score "using the ability is never worth it", which reads as a strategic
// finding rather than a copy defect.

namespace {

// Mighty Miner (115) in slot 1, the Heroic slot. validateDeckSlots only permits
// a Champion/Hero in slot 1 or 2, so the position is load-bearing, not cosmetic.
const std::vector<int> CHAMPION_DECK = { 10, 115, 41, 25, 7, 2, 6, 5 };

// Resolve this team's live Champion by CARD ID rather than by "first entity
// with isChampion", so a deck carrying two special units can never make the
// helper ambiguous.
std::shared_ptr<CombatEntity> championOnBoard(const GameManager& game, int cardId, int team) {
    for (const auto& e : game.getBoard().getEntities()) {
        if (!e->isAlive() || e->team != team || e->cardId != cardId) continue;
        if (auto ce = std::dynamic_pointer_cast<CombatEntity>(e)) {
            if (ce->isChampion || ce->isHero) return ce;
        }
    }
    return nullptr;
}

// Deploy the slot-1 Champion and settle it onto the board.
//
// Two gotchas both worth stating, because each silently no-ops the setup:
//   * the opening hand is randomised (PlayerState::initializeDeck's rng
//     overload), so the card must be forced into a hand slot directly;
//   * elixir is clamped to 10.0 in step(), so setting it higher than that
//     achieves nothing -- 10.0 is the whole budget available.
void deployChampion(GameManager& game, int cardId) {
    game.playerAI.hand[0] = cardId;
    game.playerAI.elixir = 10.0f;
    REQUIRE(game.playCard(0, cardId, 9.0f, 10.0f));
    game.step();   // commit pendingEntities so the entity is really on the board
}

} // namespace

TEST_CASE("snapshot carries championSlots and the copy can still resolve its Champion",
          "[snapshot][champion]") {
    GameManager original(CHAMPION_DECK, CHAMPION_DECK);
    deployChampion(original, 115);

    auto live = championOnBoard(original, 115, 0);
    REQUIRE(live != nullptr);
    const int trackedId = original.playerAI.championSlots.at(1).trackedEntityId;
    REQUIRE(trackedId == live->id);

    GameManager copy = original.snapshot();

    // The slot state itself survived...
    REQUIRE(copy.playerAI.championSlots.count(1) == 1);
    REQUIRE(copy.playerAI.championSlots.at(1).trackedEntityId == trackedId);

    // ...and, the part that actually matters, the id still RESOLVES against the
    // copy's own rebuilt board. This is the assertion that would fail if
    // deepCopy ever re-allocated ids instead of preserving them.
    auto copied = championOnBoard(copy, 115, 0);
    REQUIRE(copied != nullptr);
    REQUIRE(copied->id == trackedId);

    // Distinct objects, not a shared pointer -- otherwise a rollout's damage
    // would land on the live match's Champion.
    REQUIRE(copied.get() != live.get());
}

TEST_CASE("a snapshot's Champion ability is independent of the original's",
          "[snapshot][champion]") {
    GameManager original(CHAMPION_DECK, CHAMPION_DECK);
    deployChampion(original, 115);
    original.playerAI.elixir = 10.0f;
    REQUIRE(original.isChampionAbilityReady(0, 1));

    GameManager copy = original.snapshot();
    copy.playerAI.elixir = 10.0f;
    REQUIRE(copy.isChampionAbilityReady(0, 1));

    // Spend it on the COPY only.
    REQUIRE(copy.activateChampionAbility(0, 1));
    REQUIRE_FALSE(copy.isChampionAbilityReady(0, 1));

    // The original must be untouched -- this is the "a rollout put the real
    // Champion on cooldown" failure, which would make the ability look far
    // rarer than it is and is invisible without an explicit check.
    REQUIRE(original.isChampionAbilityReady(0, 1));

    // And symmetrically: spending on the original must not disturb the copy's
    // now-spent state (i.e. they are not aliasing one cooldown counter).
    REQUIRE(original.activateChampionAbility(0, 1));
    REQUIRE_FALSE(copy.isChampionAbilityReady(0, 1));
}

TEST_CASE("persistedCooldownRemaining survives a snapshot rather than resetting",
          "[snapshot][champion]") {
    GameManager original(CHAMPION_DECK, CHAMPION_DECK);
    deployChampion(original, 115);
    original.playerAI.elixir = 10.0f;
    REQUIRE(original.activateChampionAbility(0, 1));

    // syncChampionCooldowns mirrors the live entity's cooldown into the slot
    // once per tick, so one step is enough to populate it.
    original.step();
    const int persisted = original.playerAI.championSlots.at(1).persistedCooldownRemaining;
    REQUIRE(persisted > 0);

    GameManager copy = original.snapshot();
    REQUIRE(copy.playerAI.championSlots.at(1).persistedCooldownRemaining == persisted);

    // Draining the copy's cooldown must not drain the original's. A shared
    // counter here would let a search rollout "pay off" the real cooldown.
    for (int i = 0; i < 5; ++i) copy.step();
    REQUIRE(original.playerAI.championSlots.at(1).persistedCooldownRemaining == persisted);
    REQUIRE(copy.playerAI.championSlots.at(1).persistedCooldownRemaining < persisted);
}

TEST_CASE("an all-plain deck snapshots with championSlots empty on both sides",
          "[snapshot][champion]") {
    // The control: this is the state every PRE-EXISTING snapshot test runs in,
    // pinned explicitly so "the Champion tests pass" can never be a side effect
    // of the slots being empty in the fixtures that exercise them.
    const std::vector<int> plain = { 10, 1, 41, 25, 7, 2, 6, 5 };
    GameManager original(plain, plain);
    REQUIRE(original.playerAI.championSlots.empty());

    GameManager copy = original.snapshot();
    REQUIRE(copy.playerAI.championSlots.empty());
    REQUIRE(copy.playerOpponent.championSlots.empty());
}
