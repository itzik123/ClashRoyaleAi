#include <catch_amalgamated.hpp>
#include "GameManager.h"
#include "CardRegistry.h"
#include "CombatEntity.h"
#include <vector>

// Champion / Hero slot state through GameManager::snapshot().
//
// championSlots holds trackedEntityId, an int resolved by scanning the board
// for a matching entity id: a cross-structure invariant that must survive a
// deepCopy rebuilding every entity. Board::deepCopy preserves ids, so these are
// regression guards. The failure they catch is silent: a rollout whose Champion
// is unresolvable declines every activation, which reads as "the ability is
// never worth it".

namespace {

// Mighty Miner (115) in slot 1: validateDeckSlots allows a Champion only in
// slot 1 or 2.
const std::vector<int> CHAMPION_DECK = { 10, 115, 41, 25, 7, 2, 6, 5 };

// Resolve by card id, not "first entity with isChampion", so a deck with two
// special units cannot make it ambiguous.
std::shared_ptr<CombatEntity> championOnBoard(const GameManager& game, int cardId, int team) {
    for (const auto& e : game.getBoard().getEntities()) {
        if (!e->isAlive() || e->team != team || e->cardId != cardId) continue;
        if (auto ce = std::dynamic_pointer_cast<CombatEntity>(e)) {
            if (ce->isChampion || ce->isHero) return ce;
        }
    }
    return nullptr;
}

// Deploy the slot-1 Champion. The opening hand is random, so the card is forced
// into a slot; elixir is clamped to 10.0, so 10.0 is the whole budget.
void deployChampion(GameManager& game, int cardId) {
    game.playerAI.hand[0] = cardId;
    game.playerAI.elixir = 10.0f;
    REQUIRE(game.playCard(0, cardId, 9.0f, 10.0f));
    game.step();   // commit, so the entity is on the board
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

    // ...and the id still resolves against the copy's rebuilt board.
    auto copied = championOnBoard(copy, 115, 0);
    REQUIRE(copied != nullptr);
    REQUIRE(copied->id == trackedId);

    // Distinct objects, or a rollout's damage would land on the live Champion.
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

    // The original is untouched: a rollout must not put the real Champion on
    // cooldown.
    REQUIRE(original.isChampionAbilityReady(0, 1));

    // And spending on the original leaves the copy alone: no shared cooldown
    // counter.
    REQUIRE(original.activateChampionAbility(0, 1));
    REQUIRE_FALSE(copy.isChampionAbilityReady(0, 1));
}

TEST_CASE("persistedCooldownRemaining survives a snapshot rather than resetting",
          "[snapshot][champion]") {
    GameManager original(CHAMPION_DECK, CHAMPION_DECK);
    deployChampion(original, 115);
    original.playerAI.elixir = 10.0f;
    REQUIRE(original.activateChampionAbility(0, 1));

    // syncChampionCooldowns copies the live cooldown into the slot once per
    // tick.
    original.step();
    const int persisted = original.playerAI.championSlots.at(1).persistedCooldownRemaining;
    REQUIRE(persisted > 0);

    GameManager copy = original.snapshot();
    REQUIRE(copy.playerAI.championSlots.at(1).persistedCooldownRemaining == persisted);

    // Draining the copy's cooldown must not drain the original's.
    for (int i = 0; i < 5; ++i) copy.step();
    REQUIRE(original.playerAI.championSlots.at(1).persistedCooldownRemaining == persisted);
    REQUIRE(copy.playerAI.championSlots.at(1).persistedCooldownRemaining < persisted);
}

TEST_CASE("an all-plain deck snapshots with championSlots empty on both sides",
          "[snapshot][champion]") {
    // The control: the pre-existing snapshot fixtures run with empty slots,
    // pinned explicitly.
    const std::vector<int> plain = { 10, 1, 41, 25, 7, 2, 6, 5 };
    GameManager original(plain, plain);
    REQUIRE(original.playerAI.championSlots.empty());

    GameManager copy = original.snapshot();
    REQUIRE(copy.playerAI.championSlots.empty());
    REQUIRE(copy.playerOpponent.championSlots.empty());
}
