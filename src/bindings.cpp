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

    py::class_<ClashEnv>(m, "ClashRoyaleEnv")
        .def(py::init<const std::vector<int>&, const std::vector<int>&, int>(),
            py::arg("ai_deck"), py::arg("opp_deck"), py::arg("max_ticks") = 1800)
        .def("reset", &ClashEnv::reset)
        .def("step", &ClashEnv::step, py::arg("card_index"), py::arg("target_x"), py::arg("target_y"), py::arg("skip_frames") = 10)
        .def("step_self_play", &ClashEnv::stepSelfPlay,
            py::arg("card_index0"), py::arg("target_x0"), py::arg("target_y0"),
            py::arg("card_index1"), py::arg("target_x1"), py::arg("target_y1"),
            py::arg("skip_frames") = 10)
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
        .def("get_elixir_spent", &ClashEnv::getElixirSpent, py::arg("team"));

    m.def("get_all_card_ids", &getAllCardIds,
        "All ids CardRegistry currently has registered (real, playable cards only).");
}
