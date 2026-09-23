#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <optional>
#include "ArenaLayout.h"
#include "ClashEnv.h"

namespace py = pybind11;

PYBIND11_MODULE(clash_royale_env, m) {
    m.doc() = "Clash Royale RL Environment with Pybind11";

    // --- arena geometry (ArenaLayout.h), so Python derives rather than copies
    // ---
    // Module-level: the arena belongs to the game, not to an env instance.
    m.attr("ARENA_WIDTH") = ArenaLayout::WIDTH;
    m.attr("ARENA_HEIGHT") = ArenaLayout::HEIGHT;
    m.attr("ARENA_CENTER_X") = ArenaLayout::CENTER_X;
    m.attr("ARENA_LEFT_LANE_X") = ArenaLayout::LEFT_LANE_X;
    m.attr("ARENA_RIGHT_LANE_X") = ArenaLayout::RIGHT_LANE_X;
    m.attr("ARENA_LEFT_BRIDGE_X") = ArenaLayout::LEFT_BRIDGE_X;
    m.attr("ARENA_RIGHT_BRIDGE_X") = ArenaLayout::RIGHT_BRIDGE_X;
    m.attr("ARENA_BRIDGE_Y") = ArenaLayout::BRIDGE_Y;
    m.def("arena_king_y", &ArenaLayout::kingY, py::arg("team"),
          "Row of `team`'s King Tower.");
    m.def("arena_princess_y", &ArenaLayout::princessY, py::arg("team"),
          "Row of `team`'s two Princess Towers.");

    // --- elixir phase schedule ---
    // Module-level because its main consumers have no env: perception's
    // observation encoder, which must write the same scalar the training
    // encoder does, and the replay viewer.
    m.attr("MAX_ELIXIR_MULTIPLIER") = GameManager::MAX_ELIXIR_MULTIPLIER;
    m.def("elixir_multiplier_at_tick", &GameManager::elixirMultiplierAtTick,
          py::arg("tick"),
          "Elixir regeneration multiplier (1.0 / 2.0 / 3.0) in force at `tick`. "
          "10 ticks = 1 second: double from 2:00, triple from 3:00.");

    py::class_<StepResult>(m, "StepResult")
        .def_readonly("observation", &StepResult::observation)
        .def_readonly("reward", &StepResult::reward)
        .def_readonly("done", &StepResult::done);

    py::class_<SelfPlayStepResult>(m, "SelfPlayStepResult")
        .def_readonly("observation0", &SelfPlayStepResult::observation0)
        .def_readonly("observation1", &SelfPlayStepResult::observation1)
        .def_readonly("reward0", &SelfPlayStepResult::reward0)
        .def_readonly("done", &SelfPlayStepResult::done);

    // No observation fields: a rollout gets this back, and neither vector is
    // built.
    py::class_<SelfPlayFastResult>(m, "SelfPlayFastResult")
        .def_readonly("reward0", &SelfPlayFastResult::reward0)
        .def_readonly("done", &SelfPlayFastResult::done);

    py::enum_<TowerTroopType>(m, "TowerTroopType")
        .value("NONE", TowerTroopType::None)
        .value("TOWER_PRINCESS", TowerTroopType::TowerPrincess)
        .value("CANNONEER", TowerTroopType::Cannoneer)
        .value("DAGGER_DUCHESS", TowerTroopType::DaggerDuchess)
        .value("ROYAL_CHEF", TowerTroopType::RoyalChef);

    py::class_<ClashEnv>(m, "ClashRoyaleEnv")
        .def(py::init<const std::vector<int>&, const std::vector<int>&, int, TowerTroopType, TowerTroopType>(),
            py::arg("ai_deck"), py::arg("opp_deck"), py::arg("max_ticks") = 1800,
            py::arg("ai_tower_troop") = TowerTroopType::None, py::arg("opp_tower_troop") = TowerTroopType::None)
        .def("reset", &ClashEnv::reset)
        .def("step", &ClashEnv::step, py::arg("card_index"), py::arg("target_x"), py::arg("target_y"),
            py::arg("skip_frames") = 10,
            py::arg("activate_ability_slot1") = false, py::arg("activate_ability_slot2") = false)
        .def("step_self_play", &ClashEnv::stepSelfPlay,
            py::arg("card_index0"), py::arg("target_x0"), py::arg("target_y0"),
            py::arg("card_index1"), py::arg("target_x1"), py::arg("target_y1"),
            py::arg("skip_frames") = 10,
            py::arg("activate_ability0_slot1") = false, py::arg("activate_ability0_slot2") = false,
            py::arg("activate_ability1_slot1") = false, py::arg("activate_ability1_slot2") = false)
        // The same advance with no observations built, for rollouts (see
        // SelfPlayFastResult in ClashEnv.h).
        .def("step_self_play_fast", &ClashEnv::stepSelfPlayFast,
            py::arg("card_index0"), py::arg("target_x0"), py::arg("target_y0"),
            py::arg("card_index1"), py::arg("target_x1"), py::arg("target_y1"),
            py::arg("skip_frames") = 10,
            py::arg("activate_ability0_slot1") = false, py::arg("activate_ability0_slot2") = false,
            py::arg("activate_ability1_slot1") = false, py::arg("activate_ability1_slot2") = false)
        // slot: 1 = Heroic, 2 = Wild Card (CardRegistry::validateDeckSlots).
        .def("is_champion_ability_ready", &ClashEnv::isChampionAbilityReady, py::arg("team"), py::arg("slot") = 1)
        .def("activate_champion_ability", &ClashEnv::activateChampionAbility, py::arg("team"), py::arg("slot") = 1)
        .def("get_hand", &ClashEnv::getHand)
        .def("get_hand_for_team", &ClashEnv::getHandForTeam, py::arg("team"))
        .def("get_elixir", &ClashEnv::getElixir)
        .def("get_elixir_for_team", &ClashEnv::getElixirForTeam, py::arg("team"))
        .def("get_observation_for_team", &ClashEnv::getObservationForTeam, py::arg("team"))
        .def("is_game_over", &ClashEnv::isGameOver)
        .def("observation_size", &ClashEnv::observationSize)
        .def("inject_enemy", &ClashEnv::injectEnemy, py::arg("card_id"), py::arg("x"), py::arg("y"))
        // Defaults keep full health and a full deploy delay. Pass
        // deploy_ticks=0 for a unit perception can already see, or every
        // rollout grants it an extra second.
        .def("inject", &ClashEnv::inject, py::arg("card_id"), py::arg("x"), py::arg("y"),
             py::arg("team"), py::arg("hp") = -1.0f, py::arg("deploy_ticks") = -1)
        .def("set_opponent_deck", &ClashEnv::setOpponentDeck, py::arg("deck"))
        .def("set_opponent_elixir_multiplier", &ClashEnv::setOpponentElixirMultiplier, py::arg("multiplier"))
        .def("save_log", &ClashEnv::saveLog, py::arg("filepath"))
        .def("get_troop_damage_dealt", &ClashEnv::getTroopDamageDealt, py::arg("team"))
        .def("get_building_damage_dealt", &ClashEnv::getBuildingDamageDealt, py::arg("team"))
        .def("get_tower_damage_dealt", &ClashEnv::getTowerDamageDealt, py::arg("team"))
        .def("get_elixir_value_killed_by", &ClashEnv::getElixirValueKilledBy,
             py::arg("card_id"), py::arg("team"))
        .def("get_elixir_spent_on_card", &ClashEnv::getElixirSpentOnCard,
             py::arg("card_id"), py::arg("team"))
        // Cumulative damage dealt by one card, for the win-condition damage
        // term; see ClashEnv::getDamageDealtByCard.
        .def("get_damage_dealt_by_card", &ClashEnv::getDamageDealtByCard,
             py::arg("card_id"), py::arg("team"))
        // State-estimator write interface, so search evaluates the real
        // position rather than reset()'s. set_hand_for_team returns False and
        // changes nothing on an invalid hand: check it.
        .def("set_elixir_for_team", &ClashEnv::setElixirForTeam,
             py::arg("team"), py::arg("value"))
        .def("set_hand_for_team", &ClashEnv::setHandForTeam,
             py::arg("team"), py::arg("cards"))
        // Tower HP, the match clock and destruction
        // (perception/UPSTREAM_REQUESTS.md item 22).
        //
        // slot is 0 = King, 1 = left Princess, 2 = right Princess, in board
        // coordinates for both teams.
        //
        // set_tower_hp takes engine-absolute hp and refuses hp <= 0. Tower
        // levels differ between the engine and a real account, by a different
        // factor per player, so perception reports a fraction and multiplies by
        // get_tower_max_hp. Destroying a tower is destroy_tower's job (crown,
        // King wake-up, lane retargeting).
        .def("set_tower_hp", &ClashEnv::setTowerHp,
             py::arg("team"), py::arg("slot"), py::arg("hp"))
        .def("destroy_tower", &ClashEnv::destroyTower,
             py::arg("team"), py::arg("slot"))
        .def("get_tower_hp", &ClashEnv::getTowerHp,
             py::arg("team"), py::arg("slot"))
        .def("get_tower_max_hp", &ClashEnv::getTowerMaxHp,
             py::arg("team"), py::arg("slot"))
        // Clamped to [0, max_ticks]; sets both engine clocks.
        .def("set_current_tick", &ClashEnv::setCurrentTick, py::arg("tick"))
        .def("get_current_tick", &ClashEnv::getCurrentTick)
        // The elixir phase (1.0 / 2.0 / 3.0) now.
        .def("get_elixir_multiplier", &ClashEnv::getElixirMultiplier)
        // Card-cycle tracking (item 24). `inject` bypasses playCard, where the
        // cycle is recorded, so the live mirror reports plays through
        // note_played_card.
        .def("note_played_card", &ClashEnv::notePlayedCard,
             py::arg("team"), py::arg("card_id"))
        .def("get_last_played_tick", &ClashEnv::getLastPlayedTick,
             py::arg("team"), py::arg("card_id"))
        // Reproducible episodes: seeds both engine generators and re-deals
        // (ClashEnv::seed).
        .def("seed", &ClashEnv::seed, py::arg("seed"))
        .def("get_elixir_spent", &ClashEnv::getElixirSpent, py::arg("team"))
        // Surviving towers (King and Princesses) for one team; see
        // ClashEnv::getTowersAlive.
        .def("get_towers_alive", &ClashEnv::getTowersAlive, py::arg("team"))
        // The full timeout verdict: tower count, then weakest surviving tower,
        // then draw. Returns loserTeam: -1 draw, else the losing team. See
        // ClashEnv::resolveTimeoutOutcome.
        .def("resolve_timeout_outcome", &ClashEnv::resolveTimeoutOutcome)
        // The engine's enforced placement bounds, for scaling the action space.
        .def("get_max_placement_x", &ClashEnv::getMaxPlacementX)
        .def("get_own_half_max_y", &ClashEnv::getOwnHalfMaxY)
        .def("is_valid_placement", &ClashEnv::isValidPlacementForCard,
             py::arg("card_id"), py::arg("x"), py::arg("y"), py::arg("team") = 0,
             "Would playCard accept this card at this point? The exact "
             "predicate playCard uses, so the Python placement mask derives "
             "from the engine's legality rule. Read-only.")
        // A deep copy for decision-time search. Moved into the Python object
        // rather than copied a second time.
        .def("snapshot", &ClashEnv::snapshot, py::return_value_policy::move,
             "Independent deep copy of this environment, for decision-time "
             "search. Stepping the copy cannot affect the original: entities "
             "and stats collectors are duplicated, and projectiles in flight "
             "are re-pointed at the copy's own targets. Cumulative statistics "
             "carry over; the replay log does not.")
        // Structural constants, as read-only class attributes, so Python
        // derives its dimensions from them.
        .def_readonly_static("BOARD_WIDTH", &ClashEnv::BOARD_WIDTH)
        .def_readonly_static("BOARD_HEIGHT", &ClashEnv::BOARD_HEIGHT)
        .def_readonly_static("NUM_CHANNELS", &ClashEnv::NUM_CHANNELS)
        .def_readonly_static("HAND_SIZE", &ClashEnv::HAND_SIZE)
        .def_readonly_static("NUM_CARD_IDS", &ClashEnv::NUM_CARD_IDS)
        .def_readonly_static("MAX_TROOP_HP", &ClashEnv::MAX_TROOP_HP)
        .def_readonly_static("MAX_BUILDING_HP", &ClashEnv::MAX_BUILDING_HP)
        // Attribute-channel indices; the Python side indexes the raw
        // observation with them.
        .def_readonly_static("CH_COUNT", &ClashEnv::CH_COUNT)
        .def_readonly_static("CH_FLYING", &ClashEnv::CH_FLYING)
        .def_readonly_static("CH_ANTIAIR", &ClashEnv::CH_ANTIAIR)
        .def_readonly_static("CH_DPS", &ClashEnv::CH_DPS)
        .def_readonly_static("CH_RANGE", &ClashEnv::CH_RANGE)
        .def_readonly_static("CH_SPEED", &ClashEnv::CH_SPEED)
        // Opponent card-cycle block sizes (item 24); MicroRoyaleNet sizes its
        // scalar input from them.
        .def_readonly_static("NUM_CYCLE_BLOCKS", &ClashEnv::NUM_CYCLE_BLOCKS)
        .def_readonly_static("CYCLE_BLOCK_SIZE", &ClashEnv::CYCLE_BLOCK_SIZE)
        .def_readonly_static("CYCLE_RECENCY_TAU_TICKS", &ClashEnv::CYCLE_RECENCY_TAU_TICKS)
        // Forward offsets into the observation: an append cannot move them.
        .def_readonly_static("EXTRA_SCALARS_START", &ClashEnv::EXTRA_SCALARS_START)
        .def_readonly_static("CYCLE_START", &ClashEnv::CYCLE_START)
        .def_readonly_static("NUM_EXTRA_SCALARS", &ClashEnv::NUM_EXTRA_SCALARS)
        .def_readonly_static("MAX_MATCH_ELIXIR", &ClashEnv::MAX_MATCH_ELIXIR)
        // Elixir phase boundaries in ticks, from GameManager.
        .def_readonly_static("DOUBLE_ELIXIR_TICK", &GameManager::DOUBLE_ELIXIR_TICK)
        .def_readonly_static("TRIPLE_ELIXIR_TICK", &GameManager::TRIPLE_ELIXIR_TICK);

    m.def("get_all_card_ids", &getAllCardIds,
        "All ids CardRegistry currently has registered (real, playable cards only).");

    // Per-card registry facts: whether a card is a spell (spells are exempt
    // from the own-half rule), its cost (for the affordability mask), and
    // Champion / Hero flags. Read-only.
    m.def("get_card_info", [](int cardId) {
        const CardDefinition* def = CardRegistry::getInstance().getCard(cardId);
        if (!def) {
            throw std::invalid_argument("get_card_info: unknown card id " + std::to_string(cardId));
        }
        py::dict info;
        info["id"] = def->id;
        info["name"] = def->name;
        info["cost"] = def->cost;
        info["is_spell"] = def->isSpell;
        info["is_building"] = def->isBuilding;
        info["placement_radius"] = def->placementRadius;
        info["deploy_anywhere"] = def->deployAnywhere;
        info["is_champion"] = def->isChampion;
        info["is_hero"] = def->isHero;
        return info;
    }, py::arg("card_id"),
        "Registry facts for one card id: name, cost, is_spell, is_building, "
        "placement_radius, deploy_anywhere, is_champion, is_hero.");

    // The slot-legality check GameManager::reset and setOpponentDeck enforce,
    // so Python validates decks without restating the rules. "" for a legal
    // deck, else the reason.
    m.def("validate_deck_slots", &validateDeckSlots,
        "Returns \"\" if `deck` (8 card ids) satisfies Evolution/Champion slot-position rules, else an error string.");

    // A random deck that is legal by construction. With a seed, drawn from a
    // generator private to the call: the process-global stream is shared by
    // every env in the process, so seeding it would only be reproducible for a
    // fixed call order. Without one, the global stream as before. This
    // generator is the one ClashEnv::seed cannot reach (item 23C).
    m.def("sample_random_deck", [](std::optional<unsigned int> seed) {
        if (seed.has_value()) {
            std::mt19937 local(seed.value());
            return sampleRandomDeck(local);
        }
        static std::mt19937 rng(std::random_device{}());
        return sampleRandomDeck(rng);
    }, py::arg("seed") = py::none(),
       "Builds a random 8-card deck that always satisfies Evolution/Champion slot-position rules by construction. "
       "Pass seed= for a reproducible deck drawn from a generator private to this call; omit it for the shared "
       "process-global stream.");
}
