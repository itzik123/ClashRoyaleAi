#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ClashEnv.h"

namespace py = pybind11;

PYBIND11_MODULE(clash_royale_env, m) {
    m.doc() = "Clash Royale RL Environment with Pybind11";

    py::class_<StepResult>(m, "StepResult")
        .def_readonly("observation", &StepResult::observation)
        .def_readonly("reward", &StepResult::reward)
        .def_readonly("done", &StepResult::done);

    py::class_<SelfPlayStepResult>(m, "SelfPlayStepResult")
        .def_readonly("observation0", &SelfPlayStepResult::observation0)
        .def_readonly("observation1", &SelfPlayStepResult::observation1)
        .def_readonly("reward0", &SelfPlayStepResult::reward0)
        .def_readonly("done", &SelfPlayStepResult::done);

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
        // slot: 1 = Heroic, 2 = Wild Card (see CardRegistry::validateDeckSlots) --
        // up to 2 independently-tracked Champions per deck.
        .def("is_champion_ability_ready", &ClashEnv::isChampionAbilityReady, py::arg("team"), py::arg("slot") = 1)
        .def("activate_champion_ability", &ClashEnv::activateChampionAbility, py::arg("team"), py::arg("slot") = 1)
        .def("get_hand", &ClashEnv::getHand)
        .def("get_elixir", &ClashEnv::getElixir)
        .def("get_observation_for_team", &ClashEnv::getObservationForTeam, py::arg("team"))
        .def("is_game_over", &ClashEnv::isGameOver)
        .def("observation_size", &ClashEnv::observationSize)
        .def("inject_enemy", &ClashEnv::injectEnemy, py::arg("card_id"), py::arg("x"), py::arg("y"))
        .def("set_opponent_deck", &ClashEnv::setOpponentDeck, py::arg("deck"))
        .def("set_opponent_elixir_multiplier", &ClashEnv::setOpponentElixirMultiplier, py::arg("multiplier"))
        .def("save_log", &ClashEnv::saveLog, py::arg("filepath"))
        .def("get_troop_damage_dealt", &ClashEnv::getTroopDamageDealt, py::arg("team"))
        .def("get_building_damage_dealt", &ClashEnv::getBuildingDamageDealt, py::arg("team"))
        .def("get_elixir_spent", &ClashEnv::getElixirSpent, py::arg("team"))
        // Real enforced placement bounds -- see GameManager::getMaxPlacementX/
        // getOwnHalfMaxY's own comments. Lets the Python side scale its action
        // space from the engine's actual numbers instead of a hardcoded copy.
        .def("get_max_placement_x", &ClashEnv::getMaxPlacementX)
        .def("get_own_half_max_y", &ClashEnv::getOwnHalfMaxY)
        // Structural constants the observation/action encoding is built from --
        // read-only class attributes (ClashRoyaleEnv.NUM_CARD_IDS etc, no
        // instance needed) so model.py/train.py/train_selfplay.py/
        // gym_wrapper.py can derive their own dimensions from these instead of
        // hardcoding a matching copy that has to be remembered and updated by
        // hand every time one of these changes on the C++ side.
        .def_readonly_static("BOARD_WIDTH", &ClashEnv::BOARD_WIDTH)
        .def_readonly_static("BOARD_HEIGHT", &ClashEnv::BOARD_HEIGHT)
        .def_readonly_static("NUM_CHANNELS", &ClashEnv::NUM_CHANNELS)
        .def_readonly_static("HAND_SIZE", &ClashEnv::HAND_SIZE)
        .def_readonly_static("NUM_CARD_IDS", &ClashEnv::NUM_CARD_IDS)
        .def_readonly_static("MAX_TROOP_HP", &ClashEnv::MAX_TROOP_HP)
        .def_readonly_static("MAX_BUILDING_HP", &ClashEnv::MAX_BUILDING_HP);

    m.def("get_all_card_ids", &getAllCardIds,
        "All ids CardRegistry currently has registered (real, playable cards only).");
}
