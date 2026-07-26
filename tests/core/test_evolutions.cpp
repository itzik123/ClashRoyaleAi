#include <catch_amalgamated.hpp>
#include "CardRegistry.h"
#include "PlayerState.h"
#include "GameManager.h"
#include "ClashEnv.h"
#include "CombatEntity.h"
#include "test_helpers.h"
#include <vector>
#include <algorithm>

// Framework pilot: Wall Breakers Evolution (id 123, base id 83). Proves the
// evolution-slot machinery end to end before the remaining 40 evolutions
// are batched in -- see CardRegistry.h's "=== Evolutions ===" section.

TEST_CASE("PlayerState::playCard resolves an Evolution slot: 2 un-evolved cycles, "
          "then 1 evolved, repeating for the rest of the match", "[player_state][evolution]") {
    PlayerState player;
    player.initializeDeck({ 123, 1, 2, 3, 4, 5, 6, 7 }); // Wall Breakers Evolution in hand slot 0
    player.elixir = 100.0f; // never the limiting factor here

    // Real cycling (playing the other 7 slots first) isn't needed to unit
    // test PlayerState's own resolution logic -- re-seeding hand[0] to 123
    // between calls simulates "this slot cycled back to the same card"
    // without needing 8 real plays per lap.
    auto r1 = player.playCard(0);
    REQUIRE(r1.cardId == 123);
    REQUIRE_FALSE(r1.useEvolvedForm);

    player.hand[0] = 123;
    player.handCooldownTicks[0] = 0; // immediately playable, not still on the previous play's cycle-in delay
    auto r2 = player.playCard(0);
    REQUIRE_FALSE(r2.useEvolvedForm); // 2nd un-evolved cycle

    player.hand[0] = 123;
    player.handCooldownTicks[0] = 0; // immediately playable, not still on the previous play's cycle-in delay
    auto r3 = player.playCard(0);
    REQUIRE(r3.useEvolvedForm); // 3rd play: evolved

    player.hand[0] = 123;
    player.handCooldownTicks[0] = 0; // immediately playable, not still on the previous play's cycle-in delay
    auto r4 = player.playCard(0);
    REQUIRE_FALSE(r4.useEvolvedForm); // NOT a one-time charge -- back to un-evolved

    player.hand[0] = 123;
    player.handCooldownTicks[0] = 0; // immediately playable, not still on the previous play's cycle-in delay
    auto r5 = player.playCard(0);
    REQUIRE_FALSE(r5.useEvolvedForm);

    player.hand[0] = 123;
    player.handCooldownTicks[0] = 0; // immediately playable, not still on the previous play's cycle-in delay
    auto r6 = player.playCard(0);
    REQUIRE(r6.useEvolvedForm); // evolves again, confirming the repeat
}

TEST_CASE("A deck without an Evolution slot never touches evolutionState", "[player_state][evolution]") {
    PlayerState player;
    player.initializeDeck({ 83, 1, 2, 3, 4, 5, 6, 7 }); // plain Wall Breakers, not the Evolution
    REQUIRE(player.evolutionState.empty());

    auto r = player.playCard(0);
    REQUIRE(r.cardId == 83);
    REQUIRE_FALSE(r.useEvolvedForm);
}

TEST_CASE("initializeDeck resets Evolution progress -- no cross-match state leak", "[player_state][evolution]") {
    PlayerState player;
    player.initializeDeck({ 123, 1, 2, 3, 4, 5, 6, 7 });
    player.elixir = 100.0f;

    player.playCard(0);
    player.hand[0] = 123;
    player.handCooldownTicks[0] = 0; // immediately playable, not still on the previous play's cycle-in delay
    player.playCard(0);
    player.hand[0] = 123;
    player.handCooldownTicks[0] = 0; // immediately playable, not still on the previous play's cycle-in delay
    auto evolved = player.playCard(0);
    REQUIRE(evolved.useEvolvedForm); // fully cycled once

    player.initializeDeck({ 123, 1, 2, 3, 4, 5, 6, 7 }); // simulates GameManager::reset()
    auto fresh = player.playCard(0);
    REQUIRE_FALSE(fresh.useEvolvedForm); // back to un-evolved, not still mid-cycle
}

TEST_CASE("GameManager::playCard spawns the evolved entity only on an evolved play", "[game_manager][evolution]") {
    GameManager game({ 123, 1, 2, 3, 4, 5, 6, 7 }, { 0, 1, 2, 3, 4, 5, 6, 7 });
    game.playerAI.elixir = 100.0f;
    game.playerAI.hand[0] = 123; // force into hand -- opening hand is now randomized

    game.playCard(0, 123, 9.0f, 10.0f); // 1st play: un-evolved
    game.step();
    game.playerAI.hand[0] = 123;
    game.playerAI.handCooldownTicks[0] = 0; // immediately playable, not still on the previous play's cycle-in delay
    game.playCard(0, 123, 9.0f, 10.0f); // 2nd play: un-evolved
    game.step();
    game.playerAI.hand[0] = 123;
    game.playerAI.handCooldownTicks[0] = 0; // immediately playable, not still on the previous play's cycle-in delay
    game.playCard(0, 123, 9.0f, 10.0f); // 3rd play: evolved
    game.step();

    // The evolved spawn carries a deathEffect (Runners); the un-evolved
    // spawn does not -- kill everything and count how many total entities
    // this deploy round produced (2 base Wall Breakers each play, plus 2
    // Runners once, only from the evolved play's death).
    int wallBreakerCount = 0;
    for (const auto& e : game.getBoard().getEntities()) {
        if (e->name == "Wall Breakers") { wallBreakerCount++; e->takeDamage(e->hp); }
    }
    REQUIRE(wallBreakerCount == 6); // 3 plays x 2 Wall Breakers each

    game.getBoard().cleanDeadEntities();
    game.getBoard().commitPendingEntities();

    int runnerCount = 0;
    for (const auto& e : game.getBoard().getEntities()) {
        if (e->name == "Runner") runnerCount++;
    }
    REQUIRE(runnerCount == 2); // only the evolved play's pair died-and-spawned Runners
}

TEST_CASE("Inferno Dragon Evolution's evolved spawn carries the ramp grace period and 4th stage",
        "[game_manager][evolution][inferno_dragon]") {
    GameManager game({ 163, 1, 2, 3, 4, 5, 6, 7 }, { 0, 1, 2, 3, 4, 5, 6, 7 });
    game.playerAI.elixir = 100.0f;
    game.playerAI.hand[0] = 163; // force into hand -- opening hand is now randomized

    game.playCard(0, 163, 9.0f, 10.0f); // 1st play: un-evolved
    game.step();
    game.playerAI.hand[0] = 163;
    game.playerAI.handCooldownTicks[0] = 0; // immediately playable, not still on the previous play's cycle-in delay
    game.playCard(0, 163, 9.0f, 10.0f); // 2nd play: un-evolved
    game.step();
    game.playerAI.hand[0] = 163;
    game.playerAI.handCooldownTicks[0] = 0; // immediately playable, not still on the previous play's cycle-in delay
    game.playCard(0, 163, 9.0f, 10.0f); // 3rd play: evolved
    game.step();

    std::shared_ptr<CombatEntity> evolvedDragon;
    int dragonCount = 0;
    for (const auto& e : game.getBoard().getEntities()) {
        if (e->name == "Inferno Dragon") {
            dragonCount++;
            auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
            if (ce && ce->rampGracePeriodTicks > 0) evolvedDragon = ce;
        }
    }
    REQUIRE(dragonCount == 3); // 3 plays, 1 Inferno Dragon each
    REQUIRE(evolvedDragon != nullptr); // exactly one of the 3 is the evolved play

    REQUIRE(evolvedDragon->rampGracePeriodTicks == 90); // 9s at this engine's 10-ticks/second rate
    REQUIRE(evolvedDragon->rampStage4Tick == 200); // 20s
    REQUIRE(evolvedDragon->rampStage4Fraction == Catch::Approx(2.0f));
    // Base ramp schedule (3-stage 35/120/422) is untouched -- "Identical
    // Stats" per the sourced evolution table, only the charge-up behavior
    // itself differs.
    REQUIRE(evolvedDragon->rampMidTick == 15);
    REQUIRE(evolvedDragon->rampFullTick == 30);
}

TEST_CASE("Minion Horde Evolution's Dark Guard turns a member untargetable after it takes damage",
        "[game_manager][evolution][minion_horde]") {
    GameManager game({ 149, 1, 2, 3, 4, 5, 6, 7 }, { 0, 1, 2, 3, 4, 5, 6, 7 });
    game.playerAI.elixir = 100.0f;
    game.playerAI.hand[0] = 149; // force into hand -- opening hand is now randomized

    game.playCard(0, 149, 9.0f, 10.0f); // 1st play: un-evolved
    game.step();
    game.playerAI.hand[0] = 149;
    game.playerAI.handCooldownTicks[0] = 0; // immediately playable, not still on the previous play's cycle-in delay
    game.playCard(0, 149, 9.0f, 10.0f); // 2nd play: un-evolved
    game.step();
    game.playerAI.hand[0] = 149;
    game.playerAI.handCooldownTicks[0] = 0; // immediately playable, not still on the previous play's cycle-in delay
    game.playCard(0, 149, 9.0f, 10.0f); // 3rd play: evolved
    game.step();

    std::shared_ptr<CombatEntity> evolvedMinion;
    for (const auto& e : game.getBoard().getEntities()) {
        if (e->name != "Minion Horde") continue;
        auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
        if (ce && ce->onDamageTakenEffect) { evolvedMinion = ce; break; }
    }
    REQUIRE(evolvedMinion != nullptr); // found one of the 6 members from the evolved play

    REQUIRE(evolvedMinion->isTargetable()); // untouched so far
    evolvedMinion->takeDamage(1);
    REQUIRE_FALSE(evolvedMinion->isTargetable()); // Dark Guard triggered: briefly untargetable
    REQUIRE(evolvedMinion->temporaryInvisibilityTicksRemaining == 30); // 3s at this engine's 10-ticks/second rate
}

TEST_CASE("Skeleton Army Evolution's evolved play deploys 16 skeletons (+1, the Skeleton General), not 15",
        "[game_manager][evolution][skeleton_army]") {
    GameManager game({ 133, 1, 2, 3, 4, 5, 6, 7 }, { 0, 1, 2, 3, 4, 5, 6, 7 });
    game.playerAI.elixir = 100.0f;
    game.playerAI.hand[0] = 133; // force into hand -- opening hand is now randomized

    game.playCard(0, 133, 9.0f, 10.0f); // 1st play: un-evolved (15)
    game.step();
    int unevolvedCount = 0;
    for (const auto& e : game.getBoard().getEntities()) {
        if (e->name == "Skeleton Army") unevolvedCount++;
    }
    REQUIRE(unevolvedCount == 15);

    game.playerAI.hand[0] = 133;
    game.playerAI.handCooldownTicks[0] = 0; // immediately playable, not still on the previous play's cycle-in delay
    game.playCard(0, 133, 9.0f, 10.0f); // 2nd play: un-evolved
    game.step();
    game.playerAI.hand[0] = 133;
    game.playerAI.handCooldownTicks[0] = 0; // immediately playable, not still on the previous play's cycle-in delay
    game.playCard(0, 133, 9.0f, 10.0f); // 3rd play: evolved (16)
    game.step();

    int totalCount = 0;
    for (const auto& e : game.getBoard().getEntities()) {
        if (e->name == "Skeleton Army") totalCount++;
    }
    REQUIRE(totalCount == 15 + 15 + 16); // the 3rd play's evolved squad has one more than the first two
}

TEST_CASE("Wall Breakers Evolution's death effect both explodes (moderate AoE) and spawns a Runner",
        "[game_manager][evolution][wall_breakers]") {
    GameManager game({ 123, 1, 2, 3, 4, 5, 6, 7 }, { 0, 1, 2, 3, 4, 5, 6, 7 });
    game.playerAI.elixir = 100.0f;
    game.playerAI.hand[0] = 123; // force into hand -- opening hand is now randomized

    game.playCard(0, 123, 9.0f, 10.0f); // 1st: un-evolved
    game.step();
    game.playerAI.hand[0] = 123;
    game.playerAI.handCooldownTicks[0] = 0; // immediately playable, not still on the previous play's cycle-in delay
    game.playCard(0, 123, 9.0f, 10.0f); // 2nd: un-evolved
    game.step();
    game.playerAI.hand[0] = 123;
    game.playerAI.handCooldownTicks[0] = 0; // immediately playable, not still on the previous play's cycle-in delay
    game.playCard(0, 123, 9.0f, 10.0f); // 3rd: evolved
    game.step();

    // Only the evolved play's 2 copies carry a deathEffect at all (the
    // un-evolved copies from the first 2 plays don't) -- board iteration
    // order isn't guaranteed to put the 3rd play's entities first, so
    // this is the reliable way to grab one of the evolved copies
    // specifically, not just "any Wall Breakers".
    std::shared_ptr<CombatEntity> evolvedWallBreaker;
    for (const auto& e : game.getBoard().getEntities()) {
        if (e->name != "Wall Breakers") continue;
        auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
        if (ce && ce->deathEffect) { evolvedWallBreaker = ce; break; }
    }
    REQUIRE(evolvedWallBreaker != nullptr);

    // A bystander right next to it, well within the new 1.5-tile death
    // explosion but far enough that it was never hit by anything else this
    // test does -- confirms AreaDamageOnDeath actually fires now, not just
    // the pre-existing Runner spawn.
    auto bystander = std::make_shared<DummyEntity>(9999, evolvedWallBreaker->position.x + 0.5f,
        evolvedWallBreaker->position.y, 100000, 1);
    game.getBoard().addEntity(bystander);
    game.getBoard().commitPendingEntities();

    evolvedWallBreaker->takeDamage(evolvedWallBreaker->hp); // kill it directly
    game.getBoard().cleanDeadEntities();
    game.getBoard().commitPendingEntities();

    REQUIRE(bystander->hp == 100000 - 150); // AreaDamageOnDeath(1.5, 150)

    int runnerCount = 0;
    for (const auto& e : game.getBoard().getEntities()) {
        if (e->name == "Runner") runnerCount++;
    }
    REQUIRE(runnerCount == 1); // the same death also still spawns its Runner
}

TEST_CASE("getAllCardIds excludes Evolution slots -- the RANDOM_DECK_POOL landmine guard",
        "[clash_env][evolution]") {
    auto ids = getAllCardIds();
    REQUIRE(std::find(ids.begin(), ids.end(), 123) == ids.end());
    REQUIRE(std::find(ids.begin(), ids.end(), 83) != ids.end()); // the base card is still a normal pick
}
