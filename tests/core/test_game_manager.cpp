#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "GameManager.h"
#include "BuildingTargeter.h"
#include "Building.h"
#include <vector>

// ---------------- construction / reset ----------------

TEST_CASE("GameManager construction sets up exactly the 6 expected towers", "[game_manager][reset]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    const auto& entities = game.getBoard().getEntities();
    REQUIRE(entities.size() == 6);

    REQUIRE(entities[0]->symbol == 'R');
    REQUIRE(entities[0]->team == 0);
    REQUIRE(entities[0]->hp == 4008);
    REQUIRE(entities[0]->name == "King Tower");
    // Centered on a half-integer coordinate so its 4x4-tile footprint (see
    // web/viewer.html) sits flush on whole tile boundaries instead of
    // straddling them -- see the addTower() calls in GameManager::reset().
    REQUIRE(entities[0]->position.x == Catch::Approx(8.5f));
    REQUIRE(entities[0]->position.y == Catch::Approx(2.5f));

    REQUIRE(entities[1]->symbol == 'R');
    REQUIRE(entities[1]->team == 1);
    REQUIRE(entities[1]->name == "King Tower");
    REQUIRE(entities[1]->position.x == Catch::Approx(8.5f));
    // Mirrors the ally king via (height-1) - y = 33 - 2.5 = 30.5, same
    // convention as ClashEnv::extractObservationForTeam's team-1 mirroring.
    REQUIRE(entities[1]->position.y == Catch::Approx(30.5f));

    REQUIRE(entities[2]->symbol == 'P');
    REQUIRE(entities[2]->team == 0);
    REQUIRE(entities[2]->hp == 2534);
    REQUIRE(entities[2]->name == "Princess Tower");
    REQUIRE(entities[2]->position.y == Catch::Approx(6.0f));
    REQUIRE(entities[3]->symbol == 'P');
    REQUIRE(entities[3]->team == 0);
    REQUIRE(entities[3]->position.y == Catch::Approx(6.0f));

    REQUIRE(entities[4]->symbol == 'P');
    REQUIRE(entities[4]->team == 1);
    REQUIRE(entities[4]->position.y == Catch::Approx(27.0f));
    REQUIRE(entities[5]->symbol == 'P');
    REQUIRE(entities[5]->team == 1);
    REQUIRE(entities[5]->position.y == Catch::Approx(27.0f));
}

TEST_CASE("GameManager construction gives both players starting elixir and a 4-card hand", "[game_manager][reset]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 8,9,10,11,12,13,14,17 });
    REQUIRE(game.getElixirAI() == Catch::Approx(5.0f));
    REQUIRE(game.getElixirOpp() == Catch::Approx(5.0f));
    REQUIRE(game.getHand(0).size() == 4);
    REQUIRE(game.getHand(1).size() == 4);
    REQUIRE(game.getHand(1)[0] == 8);
}

// ---------------- isValidPlacement ----------------

TEST_CASE("isValidPlacement rejects out-of-bounds coordinates regardless of spell", "[game_manager][placement]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    // placedRadius is irrelevant here -- the bounds check short-circuits first.
    REQUIRE_FALSE(game.isValidPlacement(0, -1.0f, 5.0f, false, Entity::IMPLICIT_TROOP_RADIUS));
    REQUIRE_FALSE(game.isValidPlacement(0, 20.0f, 5.0f, false, Entity::IMPLICIT_TROOP_RADIUS));
    REQUIRE_FALSE(game.isValidPlacement(0, 5.0f, -1.0f, true, Entity::IMPLICIT_TROOP_RADIUS));
    REQUIRE_FALSE(game.isValidPlacement(0, 5.0f, 40.0f, true, Entity::IMPLICIT_TROOP_RADIUS));
}

TEST_CASE("isValidPlacement rejects the back-row dead-zone corners but allows its center gap", "[game_manager][placement]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    // Spells skip the own-half check but must still respect real board
    // geometry -- true=isSpell keeps this focused on the dead-zone rule.
    REQUIRE_FALSE(game.isValidPlacement(0, 0.0f, 0.0f, true, 0.0f));   // bottom-row corner
    REQUIRE(game.isValidPlacement(0, 8.5f, 0.0f, true, 0.0f));         // bottom-row center gap
    REQUIRE_FALSE(game.isValidPlacement(1, 17.0f, 33.0f, true, 0.0f)); // top-row corner
    REQUIRE(game.isValidPlacement(1, 8.5f, 33.0f, true, 0.0f));        // top-row center gap
}

TEST_CASE("isValidPlacement restricts troop/building placement to the caller's own half", "[game_manager][placement]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });

    REQUIRE(game.isValidPlacement(0, 9.0f, 15.0f, false, Entity::IMPLICIT_TROOP_RADIUS));
    REQUIRE_FALSE(game.isValidPlacement(0, 9.0f, 16.0f, false, Entity::IMPLICIT_TROOP_RADIUS));

    REQUIRE(game.isValidPlacement(1, 9.0f, 19.0f, false, Entity::IMPLICIT_TROOP_RADIUS));
    REQUIRE_FALSE(game.isValidPlacement(1, 9.0f, 18.0f, false, Entity::IMPLICIT_TROOP_RADIUS));
}

TEST_CASE("isValidPlacement lets spells ignore the half restriction", "[game_manager][placement]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    REQUIRE(game.isValidPlacement(0, 9.0f, 25.0f, true, Entity::IMPLICIT_TROOP_RADIUS)); // deep in the opponent's half
}

TEST_CASE("isValidPlacement lets deployAnywhere troops (Miner, Goblin Drill) ignore the half restriction too", "[game_manager][placement]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    REQUIRE_FALSE(game.isValidPlacement(0, 9.0f, 25.0f, false, Entity::IMPLICIT_TROOP_RADIUS));       // ordinary troop: rejected
    REQUIRE(game.isValidPlacement(0, 9.0f, 25.0f, false, Entity::IMPLICIT_TROOP_RADIUS, true));       // deployAnywhere: allowed
}

TEST_CASE("Miner can be played deep in the opponent's half via GameManager::playCard, unlike an ordinary troop", "[game_manager][placement]") {
    GameManager game({ 52, 0, 1, 2, 3, 4, 5, 6 }, { 0,1,2,3,4,5,6,7 }); // Miner (52) first in the AI's deck
    game.playerAI.elixir = 10.0f;

    REQUIRE(game.playCard(0, 52, 9.0f, 25.0f)); // deep in the opponent's half: succeeds for Miner
}

TEST_CASE("Elixir Collector passively grants its owner extra elixir beyond normal regen", "[game_manager][elixir]") {
    GameManager game({ 99, 0, 1, 2, 3, 4, 5, 6 }, { 0,1,2,3,4,5,6,7 }); // Elixir Collector (99) first in the AI's deck
    game.playerAI.elixir = 10.0f; // enough to afford its cost (6)
    REQUIRE(game.playCard(0, 99, 9.0f, 10.0f));

    game.playerAI.elixir = 0.0f; // reset low so the cap doesn't mask the effect
    for (int i = 0; i < 80; ++i) game.step(); // its periodic interval

    float pureRegenOnly = 0.035f * 80; // ELIXIR_REGEN_RATE * ticks, no collector
    REQUIRE(game.getElixirAI() > pureRegenOnly + 0.5f); // meaningfully more than regen alone
}

TEST_CASE("isValidPlacement rejects placement overlapping an existing building", "[game_manager][placement]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    // AI king tower sits at (8.5, 2.5) with a 2.0 collision radius.

    SECTION("placing a building-shaped card: requiredDist = Building::COLLISION_RADIUS (1.0) + 2.0 = 3.0") {
        REQUIRE_FALSE(game.isValidPlacement(0, 8.5f, 4.5f, false, Building::COLLISION_RADIUS)); // dist 2.0 < 3.0
        REQUIRE(game.isValidPlacement(0, 8.5f, 9.5f, false, Building::COLLISION_RADIUS));        // dist 7.0, clear
    }

    SECTION("placing a troop-shaped card: requiredDist = Entity::IMPLICIT_TROOP_RADIUS (0.4) + 2.0 = 2.4") {
        REQUIRE_FALSE(game.isValidPlacement(0, 8.5f, 4.5f, false, Entity::IMPLICIT_TROOP_RADIUS)); // dist 2.0 < 2.4
        REQUIRE(game.isValidPlacement(0, 8.5f, 9.5f, false, Entity::IMPLICIT_TROOP_RADIUS));        // dist 7.0, clear
    }
}

TEST_CASE("isValidPlacement's required gap tracks the placed card's own footprint, not a flat guess", "[game_manager][placement]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    // dist 2.7 from the AI king tower (8.5, 2.5): inside the old
    // one-size-fits-all bound (1.0 + 2.0 = 3.0) but outside the troop-shaped
    // bound (0.4 + 2.0 = 2.4) -- a legal troop placement a flat radius would
    // have wrongly rejected.
    REQUIRE_FALSE(game.isValidPlacement(0, 8.5f, 5.2f, false, Building::COLLISION_RADIUS));
    REQUIRE(game.isValidPlacement(0, 8.5f, 5.2f, false, Entity::IMPLICIT_TROOP_RADIUS));
}

// ---------------- playCard ----------------

TEST_CASE("playCard: successful play deducts elixir, cycles the hand, and spawns the entity", "[game_manager][play_card]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 8,9,10,11,12,13,14,17 });
    REQUIRE(game.getHand(0) == std::vector<int>{0, 1, 2, 3});
    REQUIRE(game.getElixirAI() == Catch::Approx(5.0f));

    size_t countBefore = game.getBoard().getEntities().size();
    bool played = game.playCard(0, 0, 9.0f, 10.0f); // Knight, cost 3.0

    REQUIRE(played);
    REQUIRE(game.getElixirAI() == Catch::Approx(2.0f));
    REQUIRE(game.getHand(0)[0] == 4); // next queued card cycled in

    // spawnEntity only queues into Board's pendingEntities; it's not visible
    // via getEntities() until the next commit (normally done by step()).
    game.getBoard().commitPendingEntities();
    REQUIRE(game.getBoard().getEntities().size() == countBefore + 1);
}

TEST_CASE("playCard fails if the card id is not currently in hand", "[game_manager][play_card]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    bool played = game.playCard(0, 99, 9.0f, 10.0f);
    REQUIRE_FALSE(played);
    REQUIRE(game.getElixirAI() == Catch::Approx(5.0f));
}

TEST_CASE("playCard fails if elixir is insufficient, without spending any", "[game_manager][play_card]") {
    GameManager game({ 19,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 }); // Golem costs 8.0, starting elixir 5.0
    bool played = game.playCard(0, 19, 9.0f, 10.0f);
    REQUIRE_FALSE(played);
    REQUIRE(game.getElixirAI() == Catch::Approx(5.0f));
    REQUIRE(game.getHand(0)[0] == 19);
}

TEST_CASE("playCard fails for a placement in the opponent's half, without spending elixir", "[game_manager][play_card]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    bool played = game.playCard(0, 0, 9.0f, 20.0f); // team 0, deep in team 1's half
    REQUIRE_FALSE(played);
    REQUIRE(game.getElixirAI() == Catch::Approx(5.0f));
}

TEST_CASE("playCard fails once the game is over", "[game_manager][play_card]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    auto aiKing = game.getBoard().getEntities()[0];
    aiKing->takeDamage(aiKing->hp);
    game.step();

    bool played = game.playCard(0, 1, 9.0f, 10.0f);
    REQUIRE_FALSE(played);
}

// ---------------- step() ----------------

TEST_CASE("step() regenerates elixir at the fixed rate, capped at 10.0", "[game_manager][step]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    REQUIRE(game.getElixirAI() == Catch::Approx(5.0f));

    game.step();
    REQUIRE(game.getElixirAI() == Catch::Approx(5.035f));

    for (int i = 0; i < 1000; ++i) game.step();
    REQUIRE(game.getElixirAI() == Catch::Approx(10.0f));
    REQUIRE(game.getElixirOpp() == Catch::Approx(10.0f));
}

TEST_CASE("step() ends the game when a King tower dies, with the correct loser", "[game_manager][step]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    auto aiKing = game.getBoard().getEntities()[0];
    REQUIRE(aiKing->symbol == 'R');
    REQUIRE(aiKing->team == 0);

    aiKing->takeDamage(aiKing->hp);
    REQUIRE_FALSE(game.isGameOver());

    game.step();

    REQUIRE(game.isGameOver());
    REQUIRE(game.getLoserTeam() == 0);
}

TEST_CASE("step() ends the game as a draw when both King towers die the same tick", "[game_manager][step]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    auto aiKing = game.getBoard().getEntities()[0];
    auto oppKing = game.getBoard().getEntities()[1];
    REQUIRE(aiKing->team == 0);
    REQUIRE(oppKing->team == 1);

    aiKing->takeDamage(aiKing->hp);
    oppKing->takeDamage(oppKing->hp);
    game.step();

    REQUIRE(game.isGameOver());
    REQUIRE(game.getLoserTeam() == -1); // draw, not an arbitrary team-0 loss
}

TEST_CASE("step() is a no-op once the game is over", "[game_manager][step]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    auto aiKing = game.getBoard().getEntities()[0];
    aiKing->takeDamage(aiKing->hp);
    game.step();

    float elixirBefore = game.getElixirAI();
    game.step();
    REQUIRE(game.getElixirAI() == Catch::Approx(elixirBefore));
}

TEST_CASE("step() removes entities that died during the tick from the board", "[game_manager][step]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    Board& board = game.getBoard();
    auto decoy = std::make_shared<DummyEntity>(board.allocateId(), 0.0f, 0.0f, 100, 0);
    spawn(board, decoy);
    decoy->takeDamage(100);

    size_t countBefore = board.getEntities().size();
    game.step();

    REQUIRE(board.getEntities().size() == countBefore - 1);
}

// ---------------- fixed: step()'s post-collision clamp now respects riverIgnores ----------------

TEST_CASE("step()'s post-collision re-clamp respects riverIgnores (regression test)", "[game_manager][step]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    Board& board = game.getBoard();

    // Placed mid-river at a non-bridge column, exactly like a Hog Rider caught
    // mid-crossing. Used to get snapped back out by step()'s own separate
    // river-clamp pass even though its own clampPosition() call (inside
    // update()) correctly left it alone. Now both calls go through the same
    // Board::clampToBoard(), so the second call is a no-op for it too.
    auto hog = std::make_shared<BuildingTargeter>(board.allocateId(), 8.0f, 17.0f, 1408, 0, 0.8f, 1.0f, 264, 15, 'H');
    hog->setIgnoresRiver(true);
    board.addEntity(hog);
    board.commitPendingEntities();

    game.step();

    REQUIRE(hog->position.y > 16.0f);
    REQUIRE(hog->position.y < 18.0f);
}

// ---------------- statistics ----------------

TEST_CASE("A full scripted mini-match produces sane getStatistics() output", "[game_manager][statistics]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    Board& board = game.getBoard();

    bool played = game.playCard(0, 0, 9.0f, 10.0f); // Knight, cost 3.0
    REQUIRE(played);

    // Right next to the Knight's placement, well within its effective
    // attack range -- combat should start on the very first step().
    auto enemy = std::make_shared<DummyEntity>(board.allocateId(), 9.0f, 10.5f, 1000, 1);
    spawn(board, enemy);

    for (int i = 0; i < 5; ++i) game.step();

    const auto& stats = game.getStatistics();
    REQUIRE(stats.elixirSpent(0) == Catch::Approx(3.0f));
    REQUIRE(stats.cardsPlayed(0).size() == 1);
    REQUIRE(stats.cardsPlayed(0)[0].cardId == 0);
    REQUIRE(stats.totalDamageDealt(0) > 0); // the Knight landed at least one hit
    REQUIRE(stats.damageDealtByCard(0, 0) > 0);
    REQUIRE(stats.loserTeam() == -1); // match still ongoing
    REQUIRE(stats.matchDurationTicks() == 0); // MatchEndedEvent hasn't fired yet
}

TEST_CASE("GameManager::reset() gives a fresh MatchStatistics, not stale numbers from the previous match", "[game_manager][statistics]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    bool played = game.playCard(0, 0, 9.0f, 10.0f); // Knight, cost 3.0
    REQUIRE(played);
    game.step();

    REQUIRE(game.getStatistics().elixirSpent(0) == Catch::Approx(3.0f));
    REQUIRE(game.getStatistics().cardsPlayed(0).size() == 1);

    game.reset();

    REQUIRE(game.getStatistics().elixirSpent(0) == Catch::Approx(0.0f));
    REQUIRE(game.getStatistics().cardsPlayed(0).empty());

    // The fresh stats object must actually be listening to the NEW board,
    // not silently detached -- play another card post-reset and confirm
    // it's picked up.
    bool playedAgain = game.playCard(0, 0, 9.0f, 10.0f);
    REQUIRE(playedAgain);
    REQUIRE(game.getStatistics().elixirSpent(0) == Catch::Approx(3.0f));
}

TEST_CASE("getStatistics() reports the match outcome once the game ends", "[game_manager][statistics]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    auto aiKing = game.getBoard().getEntities()[0];
    aiKing->takeDamage(aiKing->hp);

    game.step();

    REQUIRE(game.getStatistics().loserTeam() == 0);
    REQUIRE(game.getStatistics().matchDurationTicks() == 1);
}

// ---------------- fixed: PlayerState::playCard on an empty deckQueue ----------------

TEST_CASE("PlayerState::playCard does not spend elixir when the deck queue is empty", "[player_state]") {
    PlayerState player;
    player.initializeDeck({ 0, 1, 2, 3 }); // only 4 cards: deckQueue starts empty
    REQUIRE(player.elixir == Catch::Approx(5.0f));

    PlayerState::PlayCardResult result = player.playCard(0); // Knight, cost 3.0

    REQUIRE(result.cardId == -1);
    REQUIRE_FALSE(result.useEvolvedForm);
    REQUIRE(player.elixir == Catch::Approx(5.0f)); // untouched
    REQUIRE(player.hand[0] == 0);                  // hand slot untouched
}

// ---------------- activateChampionAbility ----------------
// Deck seeds card 115 (Mighty Miner) as the first hand slot -- affordable
// (4.0 elixir) from the starting 5.0, so game.playCard(0, 115, 9.0f, 10.0f)
// (team 0's own half, same spot playCard's own "fails once game over" test
// above already uses) reliably deploys him before each ability test.

TEST_CASE("activateChampionAbility fires, deducts elixir, and starts the cooldown", "[game_manager][champion]") {
    GameManager game({ 115, 1, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    game.playCard(0, 115, 9.0f, 10.0f);
    game.step(); // commits the pending entity so findChampion can see it
    float elixirBefore = game.getElixirAI();

    bool activated = game.activateChampionAbility(0);

    REQUIRE(activated);
    REQUIRE(game.getElixirAI() == Catch::Approx(elixirBefore - 1.0f)); // 1-elixir ability cost
}

TEST_CASE("activateChampionAbility feeds MatchStatistics' Champion ability tracking", "[game_manager][champion][stats]") {
    GameManager game({ 115, 1, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    game.playCard(0, 115, 9.0f, 10.0f);
    game.step(); // commits the pending entity so findChampion can see it

    REQUIRE(game.activateChampionAbility(0));

    REQUIRE(game.getStatistics().championAbilityElixirSpent(0) == Catch::Approx(1.0f));
    REQUIRE(game.getStatistics().championAbilityActivations(0) == 1);
    REQUIRE(game.getStatistics().championAbilityActivations(1) == 0);
    // Distinct from playCard's own tracking -- deploying the Champion
    // itself is the only thing that shows up as elixirSpent/cardsPlayed.
    REQUIRE(game.getStatistics().cardsPlayed(0).size() == 1);
}

TEST_CASE("activateChampionAbility fails when the team has no deployed Champion", "[game_manager][champion]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    REQUIRE_FALSE(game.activateChampionAbility(0));
}

TEST_CASE("activateChampionAbility fails when unaffordable, without deducting anything", "[game_manager][champion]") {
    GameManager game({ 115, 1, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    game.playCard(0, 115, 9.0f, 10.0f);
    game.step(); // commits the pending entity so findChampion can see it
    game.playerAI.elixir = 0.5f; // below the 1.0 ability cost

    bool activated = game.activateChampionAbility(0);

    REQUIRE_FALSE(activated);
    REQUIRE(game.getElixirAI() == Catch::Approx(0.5f)); // untouched
}

TEST_CASE("activateChampionAbility fails while still on cooldown", "[game_manager][champion]") {
    GameManager game({ 115, 1, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    game.playCard(0, 115, 9.0f, 10.0f);
    game.step(); // commits the pending entity so findChampion can see it
    game.playerAI.elixir = 10.0f; // plenty for a second attempt, if cooldown didn't block it

    REQUIRE(game.activateChampionAbility(0));
    REQUIRE_FALSE(game.activateChampionAbility(0));
}

TEST_CASE("activateChampionAbility is ready again once its full cooldown elapses via step()", "[game_manager][champion]") {
    GameManager game({ 115, 1, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    game.playCard(0, 115, 9.0f, 10.0f);
    game.step(); // commits the pending entity so findChampion can see it
    game.playerAI.elixir = 10.0f;
    REQUIRE(game.activateChampionAbility(0));

    // Fast-forward the cooldown directly instead of driving 130 real
    // game.step() ticks -- the Miner has no restriction to buildings only
    // (plain MeleeSquad), so 130 ticks of real simulation would let him
    // wander into and fight enemy towers along the way, making this
    // test's timing depend on unrelated combat outcomes instead of just
    // the cooldown mechanism itself.
    std::shared_ptr<CombatEntity> champion;
    for (const auto& e : game.getBoard().getEntities()) {
        auto candidate = std::dynamic_pointer_cast<CombatEntity>(e);
        if (candidate && candidate->isChampion) { champion = candidate; break; }
    }
    REQUIRE(champion != nullptr);

    champion->abilityCooldownRemaining = 1;
    game.playerAI.elixir = 10.0f;
    REQUIRE_FALSE(game.activateChampionAbility(0)); // still 1 tick left

    game.step(); // decrements the last tick via CombatEntity::update()
    game.playerAI.elixir = 10.0f;
    REQUIRE(game.activateChampionAbility(0));
}

TEST_CASE("activateChampionAbility fails once the game is over", "[game_manager][champion]") {
    GameManager game({ 115, 1, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    game.playCard(0, 115, 9.0f, 10.0f);
    game.step(); // commits the pending entity so findChampion can see it
    auto aiKing = game.getBoard().getEntities()[0];
    aiKing->takeDamage(aiKing->hp);
    game.step();

    REQUIRE(game.isGameOver());
    REQUIRE_FALSE(game.activateChampionAbility(0));
}
