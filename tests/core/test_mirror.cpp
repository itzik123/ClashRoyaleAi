#include <catch_amalgamated.hpp>
#include "GameManager.h"
#include "CardRegistry.h"
#include "CombatEntity.h"

TEST_CASE("Mirror fails to play when nothing has been played yet", "[game_manager][mirror]") {
    GameManager game({ 164, 1, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    // Mirror is now GUARANTEED excluded from the real opening hand (see
    // PlayerState::initializeDeck's rng overload) -- force it in directly so
    // this fails for the reason the test actually cares about (nothing
    // played yet), not just "card not in hand".
    game.playerAI.hand[0] = 164;
    game.playerAI.elixir = 10.0f;

    REQUIRE_FALSE(game.playCard(0, 164, 9.0f, 10.0f));
}

TEST_CASE("Mirror replays the last card played, at +1 elixir cost", "[game_manager][mirror]") {
    GameManager game({ 0, 164, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 }); // Knight (cost 3), then Mirror
    // Fully pin hand/deckQueue to the deterministic deck[0..3]/deck[4..7]
    // partition -- opening hand is now randomized (see PlayerState::
    // initializeDeck's rng overload), and Mirror is additionally guaranteed
    // excluded from it entirely. A partial single-slot force (hand[1]=164)
    // isn't enough here: Mirror could still be left sitting in deckQueue
    // from the random shuffle too, and Knight's OWN play draws from
    // deckQueue.front() -- if that residual Mirror got drawn into hand[0]
    // it would collide with the forced hand[1] copy and get found first by
    // GameManager::playCard's hand scan.
    game.playerAI.hand = { 0, 164, 2, 3 };
    game.playerAI.deckQueue = { 4, 5, 6, 7 };
    game.playerAI.elixir = 10.0f;

    REQUIRE(game.playCard(0, 0, 9.0f, 10.0f)); // Knight
    float elixirAfterKnight = game.getElixirAI();
    REQUIRE(elixirAfterKnight == Catch::Approx(7.0f)); // 10 - 3

    REQUIRE(game.playCard(0, 164, 9.0f, 10.0f)); // Mirror
    REQUIRE(game.getElixirAI() == Catch::Approx(3.0f)); // 7 - (3 + 1)

    game.step(); // commit both spawns
    int knightCount = 0;
    for (const auto& e : game.getBoard().getEntities()) {
        if (e->name == "Knight") knightCount++;
    }
    REQUIRE(knightCount == 2); // the original, plus Mirror's copy
}

TEST_CASE("Mirror fails when unaffordable (mirrored cost + 1), without deducting anything",
        "[game_manager][mirror]") {
    GameManager game({ 0, 164, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    // Fully pin hand/deckQueue -- see the previous test's own comment on why
    // a partial single-slot force isn't enough.
    game.playerAI.hand = { 0, 164, 2, 3 };
    game.playerAI.deckQueue = { 4, 5, 6, 7 };
    game.playerAI.elixir = 10.0f;
    game.playCard(0, 0, 9.0f, 10.0f); // Knight, cost 3
    game.playerAI.elixir = 3.5f; // enough for Knight (3) again, not Mirror's 4

    REQUIRE_FALSE(game.playCard(0, 164, 9.0f, 10.0f));
    REQUIRE(game.getElixirAI() == Catch::Approx(3.5f)); // untouched
}

TEST_CASE("A second Mirror replays what was played before the FIRST Mirror, not the Mirror itself",
        "[game_manager][mirror]") {
    // Deck: Knight (0), Mirror (164) in the first two slots; the queue
    // cycles Mirror back into hand slot 1 by the time we need to play it
    // again -- simpler to just re-seed hand[1] directly, same technique
    // already used for the Evolution cycling tests.
    GameManager game({ 0, 164, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    // Fully pin hand/deckQueue -- see the first Mirror test's own comment on
    // why a partial single-slot force isn't enough.
    game.playerAI.hand = { 0, 164, 2, 3 };
    game.playerAI.deckQueue = { 4, 5, 6, 7 };
    game.playerAI.elixir = 100.0f;

    REQUIRE(game.playCard(0, 0, 9.0f, 10.0f));   // Knight
    REQUIRE(game.playCard(0, 164, 9.0f, 10.0f)); // 1st Mirror -> mirrors Knight

    game.playerAI.hand[1] = 164; // simulate Mirror having cycled back to this slot
    game.playerAI.handCooldownTicks[1] = 0; // ...immediately playable, not still on the 1st Mirror's cycle-in delay
    REQUIRE(game.playCard(0, 164, 9.0f, 10.0f)); // 2nd Mirror -> should ALSO mirror Knight, not the 1st Mirror

    game.step();
    int knightCount = 0, mirrorEntityCount = 0;
    for (const auto& e : game.getBoard().getEntities()) {
        if (e->name == "Knight") knightCount++;
        if (e->name == "Mirror") mirrorEntityCount++;
    }
    REQUIRE(knightCount == 3);       // original + 2 mirrored copies
    REQUIRE(mirrorEntityCount == 0); // Mirror itself never spawns an entity of its own
}

TEST_CASE("Mirror respects the mirrored card's own placement rules (a spell may target the enemy half)",
        "[game_manager][mirror]") {
    GameManager game({ 7, 164, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 }); // Fireball (id 7, a spell), then Mirror
    // Fully pin hand/deckQueue -- see the first Mirror test's own comment on
    // why a partial single-slot force isn't enough.
    game.playerAI.hand = { 7, 164, 2, 3 };
    game.playerAI.deckQueue = { 4, 5, 6, 7 };
    game.playerAI.elixir = 100.0f;

    REQUIRE(game.playCard(0, 7, 9.0f, 10.0f)); // Fireball, own half

    // Mirroring a spell onto the ENEMY half must succeed -- spells aren't
    // restricted to the caster's own side the way troops are.
    REQUIRE(game.playCard(0, 164, 9.0f, 25.0f));
}

// ---------------- Mirror + Champion/Hero ----------------
// Real-game fidelity: Mirror CAN duplicate a Champion/Hero (previously
// blocked outright in this engine -- corrected per direct game-design
// feedback). The ability always belongs to whichever instance of that
// slot's troop was deployed most recently, whether the deployment came
// from the original card or from Mirror (see GameManager::playCard's
// tracking hook, which resolves off the spawned entity's own cardId).

TEST_CASE("Mirror can duplicate a Champion, and the ability targets whichever instance was deployed last",
        "[game_manager][mirror][champion]") {
    GameManager game({ 1, 115, 164, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 }); // Mighty Miner in slot 1, Mirror in slot 2
    game.playerAI.hand = { 1, 115, 164, 3 };
    game.playerAI.deckQueue = { 4, 5, 6, 7 };
    game.playerAI.elixir = 100.0f;

    REQUIRE(game.playCard(0, 115, 9.0f, 10.0f)); // original Mighty Miner
    game.step();

    int originalId = -1;
    for (const auto& e : game.getBoard().getEntities()) {
        auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
        if (ce && ce->isChampion) originalId = ce->id;
    }
    REQUIRE(originalId != -1);
    REQUIRE(game.playerAI.championSlots[1].trackedEntityId == originalId);

    REQUIRE(game.playCard(0, 164, 9.0f, 10.0f)); // Mirror duplicates the Champion
    game.step();

    int mightyMinerCount = 0;
    for (const auto& e : game.getBoard().getEntities()) {
        if (e->name == "Mighty Miner") mightyMinerCount++;
    }
    REQUIRE(mightyMinerCount == 2); // original + Mirror's copy

    // Tracking now points at the MIRRORED copy, not the original --
    // "the ability belongs to whoever was created last" applies
    // uniformly, Mirror included.
    int trackedId = game.playerAI.championSlots[1].trackedEntityId;
    REQUIRE(trackedId != originalId);

    REQUIRE(game.isChampionAbilityReady(0, 1));
    REQUIRE(game.activateChampionAbility(0, 1));

    for (const auto& e : game.getBoard().getEntities()) {
        auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
        if (!ce || !ce->isChampion) continue;
        if (ce->id == originalId) {
            REQUIRE(ce->abilityCooldownRemaining == 0); // original untouched
        } else {
            REQUIRE(ce->abilityCooldownRemaining == 130); // the mirrored copy actually activated
        }
    }
}

TEST_CASE("Mirror can duplicate a Hero, and the mirrored copy has its own independent one-use ability",
        "[game_manager][mirror][hero]") {
    GameManager game({ 1, 170, 164, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 }); // Hero Mini P.E.K.K.A. in slot 1, Mirror in slot 2
    game.playerAI.hand = { 1, 170, 164, 3 };
    game.playerAI.deckQueue = { 4, 5, 6, 7 };
    game.playerAI.elixir = 100.0f;

    REQUIRE(game.playCard(0, 170, 9.0f, 10.0f)); // original Hero Mini P.E.K.K.A.
    game.step();
    REQUIRE(game.activateChampionAbility(0, 1)); // consume the original's one use
    REQUIRE_FALSE(game.isChampionAbilityReady(0, 1)); // original is now spent

    REQUIRE(game.playCard(0, 164, 9.0f, 10.0f)); // Mirror duplicates the Hero
    game.step();

    // Tracking now points at the mirrored copy, which carries its OWN
    // fresh one-use ability -- independent of the original's already-
    // spent use.
    REQUIRE(game.isChampionAbilityReady(0, 1));
    REQUIRE(game.activateChampionAbility(0, 1));
    REQUIRE_FALSE(game.isChampionAbilityReady(0, 1)); // now the mirrored copy is spent too
}

// ---------------- Mirror + Evolution ----------------
// Real-game fidelity: mirroring an Evolution-slot card always plays its
// BASE (non-evolved) form, and does not count toward that Evolution's own
// cycle progress -- confirmed as already-correct engine behavior
// (PlayerState::playCard resolves useEvolvedForm/evolutionState off
// Mirror's OWN hand slot/cardId, never the mirrored card's), just
// previously untested.

TEST_CASE("Mirror plays the base (non-evolved) form when mirroring an Evolution, without consuming its cycle",
        "[game_manager][mirror][evolution]") {
    GameManager game({ 123, 164, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 }); // Wall Breakers Evolution, then Mirror
    game.playerAI.hand = { 123, 164, 2, 3 };
    game.playerAI.deckQueue = { 4, 5, 6, 7 };
    game.playerAI.elixir = 100.0f;

    REQUIRE(game.playCard(0, 123, 9.0f, 10.0f)); // 1st real play: un-evolved (1 of 2 cycles used)
    game.step();

    auto it = game.playerAI.evolutionState.find(123);
    REQUIRE(it != game.playerAI.evolutionState.end());
    REQUIRE(it->second.cyclesUntilEvolved == 1); // 1 real cycle consumed

    REQUIRE(game.playCard(0, 164, 9.0f, 10.0f)); // Mirror -- does NOT count as a cycle
    game.step();

    REQUIRE(it->second.cyclesUntilEvolved == 1); // untouched by the Mirror play

    // The mirrored copy is the BASE form -- no evolved-only deathEffect
    // (Runner spawn). Kill every Wall Breakers unit from both plays and
    // confirm no Runner appears.
    int wallBreakerCount = 0;
    for (const auto& e : game.getBoard().getEntities()) {
        if (e->name == "Wall Breakers") { wallBreakerCount++; e->takeDamage(e->hp); }
    }
    REQUIRE(wallBreakerCount == 4); // 2 plays x 2 Wall Breakers each (both un-evolved)

    game.getBoard().cleanDeadEntities();
    game.getBoard().commitPendingEntities();

    int runnerCount = 0;
    for (const auto& e : game.getBoard().getEntities()) {
        if (e->name == "Runner") runnerCount++;
    }
    REQUIRE(runnerCount == 0); // neither play was evolved -- no Runners at all
}
