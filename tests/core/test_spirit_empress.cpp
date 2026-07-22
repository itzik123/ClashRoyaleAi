#include <catch_amalgamated.hpp>
#include "GameManager.h"
#include "SpiritEmpressForms.h"
#include <algorithm>

TEST_CASE("Spirit Empress plays the ground form (cost 3) when elixir is below 6", "[game_manager][spirit_empress]") {
    GameManager game({ 165, 1, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    game.playerAI.elixir = 5.0f;

    REQUIRE(game.playCard(0, 165, 9.0f, 10.0f));
    REQUIRE(game.getElixirAI() == Catch::Approx(2.0f)); // 5 - 3

    game.step();
    auto entities = game.getBoard().getEntities();
    auto it = std::find_if(entities.begin(), entities.end(),
        [](const auto& e) { return e->name == "Spirit Empress"; });
    REQUIRE(it != entities.end());
    REQUIRE((*it)->getCollisionRadius() >= 0.0f); // sanity: a real entity, not null
    REQUIRE_FALSE((*it)->isFlying);
}

TEST_CASE("Spirit Empress plays the flying form (cost 6) when elixir is at or above 6", "[game_manager][spirit_empress]") {
    GameManager game({ 165, 1, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    game.playerAI.elixir = 10.0f;

    REQUIRE(game.playCard(0, 165, 9.0f, 10.0f));
    REQUIRE(game.getElixirAI() == Catch::Approx(4.0f)); // 10 - 6

    game.step();
    auto entities = game.getBoard().getEntities();
    auto it = std::find_if(entities.begin(), entities.end(),
        [](const auto& e) { return e->name == "Spirit Empress"; });
    REQUIRE(it != entities.end());
    REQUIRE((*it)->isFlying);
}

TEST_CASE("Spirit Empress fails when even the cheaper ground form is unaffordable", "[game_manager][spirit_empress]") {
    GameManager game({ 165, 1, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    game.playerAI.elixir = 2.0f;

    REQUIRE_FALSE(game.playCard(0, 165, 9.0f, 10.0f));
    REQUIRE(game.getElixirAI() == Catch::Approx(2.0f)); // untouched
}

TEST_CASE("Both Spirit Empress forms carry the same registered cardId for stats attribution",
        "[spirit_empress]") {
    REQUIRE(spiritEmpressGroundStats().id == 165);
    REQUIRE(spiritEmpressFlyingStats().id == 165);
}
