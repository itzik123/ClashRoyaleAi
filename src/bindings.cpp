#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <optional>
#include "ArenaLayout.h"
#include "ClashEnv.h"

namespace py = pybind11;

PYBIND11_MODULE(clash_royale_env, m) {
    m.doc() = "Clash Royale RL Environment with Pybind11";

    // ---- arena geometry, so nothing has to keep its own copy ----
    //
    // Until 2026-08-21 no binding exposed a board entity's position, so every
    // Python consumer that needed a tower or bridge column hardcoded one with a
    // comment naming the header. THREE of them went stale the moment the arena
    // was corrected -- advisors/tactics.py (OWN_KING 9.0, OWN_PRINCESS 4.0,
    // BRIDGE_XS (4,14)), perception/geometry.py, and on the C++ side
    // HeuristicOpponent's own LEFT_BRIDGE_X/RIGHT_BRIDGE_X, which had already
    // been wrong before that. Same failure CLAUDE.md records for the river in
    // 2026-07-29 and the towers in 2026-07-30.
    //
    // These are the ArenaLayout.h constants, exposed read-only so Python can
    // DERIVE instead. Module-level rather than class-static because the arena
    // is a property of the game, not of an env instance.
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

    // --- ELIXIR PHASE SCHEDULE (2026-09-02) --------------------------------
    // Module level, not on the env class, because the consumers that most need
    // it have no env: perception/'s hand-built observation encoder (which must
    // write the identical scalar the training encoder writes, or the deployed
    // agent reads a phase it was never trained on) and the replay viewer.
    // Exposing the FUNCTION as well as the boundaries means neither has to
    // restate the "two thresholds, three values" shape either.
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

    // No observation fields, deliberately: this is what a rollout gets back, and
    // the whole point is that neither vector was built. `def_readonly` on the
    // pair above converts to a Python list ON ATTRIBUTE ACCESS, so a caller
    // ignoring them already skipped the marshalling -- what it could not skip,
    // until this type existed, was the C++ construction.
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
        // Same advance, no observations built. For rollouts that step a
        // snapshot and throw the result away -- see SelfPlayFastResult in
        // ClashEnv.h and perception/UPSTREAM_REQUESTS.md item 21.
        .def("step_self_play_fast", &ClashEnv::stepSelfPlayFast,
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
        .def("get_hand_for_team", &ClashEnv::getHandForTeam, py::arg("team"))
        .def("get_elixir", &ClashEnv::getElixir)
        .def("get_elixir_for_team", &ClashEnv::getElixirForTeam, py::arg("team"))
        .def("get_observation_for_team", &ClashEnv::getObservationForTeam, py::arg("team"))
        .def("is_game_over", &ClashEnv::isGameOver)
        .def("observation_size", &ClashEnv::observationSize)
        .def("inject_enemy", &ClashEnv::injectEnemy, py::arg("card_id"), py::arg("x"), py::arg("y"))
        // hp and deploy_ticks default to the pre-item-22 behaviour (full
        // health, a full DEPLOY_TIME_TICKS delay), so every existing caller is
        // unchanged. Pass deploy_ticks=0 for a unit perception can already SEE
        // on the board -- otherwise the mirror re-charges it a deploy second
        // and every rollout believes it has an extra second to react.
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
        // Cumulative damage dealt BY one card -- read by train.py's
        // win-condition damage term. Pure accessor over a counter the engine
        // already maintains; see ClashEnv::getDamageDealtByCard for what it
        // does and does not measure.
        .def("get_damage_dealt_by_card", &ClashEnv::getDamageDealtByCard,
             py::arg("card_id"), py::arg("team"))
        // State-estimator WRITE interface -- lets perception/ push a
        // reconstructed live state in, so decision-time search evaluates the
        // real position instead of reset()'s 5.0 elixir and unseeded hand
        // shuffle. set_hand returns False (and changes nothing) on a hand that
        // is not a valid permutation of that team's deck; check it, because a
        // silently accepted misread is worse than no update at all.
        .def("set_elixir_for_team", &ClashEnv::setElixirForTeam,
             py::arg("team"), py::arg("value"))
        .def("set_hand_for_team", &ClashEnv::setHandForTeam,
             py::arg("team"), py::arg("cards"))
        // Tower HP, the match clock, and destruction -- item 22 (2026-08-24),
        // the rest of the same interface.
        //
        // slot is 0 = King, 1 = LEFT Princess, 2 = RIGHT Princess, in BOARD
        // coordinates for both teams (never team-relative -- the caller is a
        // sensor reading a screen, and asking it to mirror its own coordinates
        // is the convention error that put the arena half a tile off-centre).
        //
        // set_tower_hp takes ENGINE-ABSOLUTE hp and returns False, changing
        // nothing, on hp <= 0. Tower levels do not match between this engine
        // (level 9: 2534/4008) and a real account (often level 4-5:
        // 1750/1890), i.e. wrong by a DIFFERENT factor per player -- so
        // perception reports a FRACTION and multiplies by get_tower_max_hp.
        // Destroying a tower is destroy_tower's job, because it has side
        // effects a clamp cannot express: the crown, the King's
        // princess-count wake trigger, and LanePath's retargeting.
        .def("set_tower_hp", &ClashEnv::setTowerHp,
             py::arg("team"), py::arg("slot"), py::arg("hp"))
        .def("destroy_tower", &ClashEnv::destroyTower,
             py::arg("team"), py::arg("slot"))
        .def("get_tower_hp", &ClashEnv::getTowerHp,
             py::arg("team"), py::arg("slot"))
        .def("get_tower_max_hp", &ClashEnv::getTowerMaxHp,
             py::arg("team"), py::arg("slot"))
        // Clamped to [0, max_ticks]. Sets BOTH engine clocks so they cannot
        // drift; see ClashEnv::setCurrentTick.
        .def("set_current_tick", &ClashEnv::setCurrentTick, py::arg("tick"))
        .def("get_current_tick", &ClashEnv::getCurrentTick)
        // Elixir phase (1.0 / 2.0 / 3.0) at the current tick. Bound so that
        // probes, perception/ and the tests read the schedule from the engine
        // instead of restating DOUBLE_ELIXIR_TICK/TRIPLE_ELIXIR_TICK.
        .def("get_elixir_multiplier", &ClashEnv::getElixirMultiplier)
        // Card-cycle tracking (item 24). `note_played_card` is for the live
        // mirror in perception/: `inject` places a BODY and deliberately
        // bypasses GameManager::playCard, which is where the cycle is hooked,
        // so an estimator has to report the PLAY separately or the cycle
        // blocks stay empty in deployment while working fine in training.
        .def("note_played_card", &ClashEnv::notePlayedCard,
             py::arg("team"), py::arg("card_id"))
        .def("get_last_played_tick", &ClashEnv::getLastPlayedTick,
             py::arg("team"), py::arg("card_id"))
        // Reproducible episodes. Seeds BOTH engine generators and re-deals,
        // so two envs given the same seed agree on the opening hand, the
        // cycle order and the heuristic's rolls. See ClashEnv::seed, and
        // perception/UPSTREAM_REQUESTS.md item 7 for what it is worth.
        .def("seed", &ClashEnv::seed, py::arg("seed"))
        .def("get_elixir_spent", &ClashEnv::getElixirSpent, py::arg("team"))
        // Surviving TOWER count (King + Princesses) for one team -- see
        // ClashEnv::getTowersAlive for why the Python reward needs this.
        .def("get_towers_alive", &ClashEnv::getTowersAlive, py::arg("team"))
        // The FULL outcome verdict -- tower count, then weakest surviving
        // tower, then draw. get_towers_alive above gives only the first of
        // those three rules, and eight separate scripts re-derived the rest
        // wrongly from it before this existed. Returns loserTeam: -1 draw,
        // 0 team 0 lost, 1 team 1 lost. See ClashEnv::resolveTimeoutOutcome.
        .def("resolve_timeout_outcome", &ClashEnv::resolveTimeoutOutcome)
        // Real enforced placement bounds -- see GameManager::getMaxPlacementX/
        // getOwnHalfMaxY's own comments. Lets the Python side scale its action
        // space from the engine's actual numbers instead of a hardcoded copy.
        .def("get_max_placement_x", &ClashEnv::getMaxPlacementX)
        .def("get_own_half_max_y", &ClashEnv::getOwnHalfMaxY)
        .def("is_valid_placement", &ClashEnv::isValidPlacementForCard,
             py::arg("card_id"), py::arg("x"), py::arg("y"), py::arg("team") = 0,
             "Would playCard accept this card at this point? The exact "
             "predicate playCard uses, exposed so the Python placement mask is "
             "derived from the engine's legality rule instead of a second copy "
             "of the board geometry. Read-only. See "
             "perception/UPSTREAM_REQUESTS.md item 12 -- 58.7% of the policy's "
             "card choices were being refused here, silently.")
        // Decision-time search. Returns a genuinely independent environment --
        // entities deep-copied, stats collectors deep-copied, projectile
        // targets remapped -- so a caller can try a candidate action, roll it
        // forward and throw it away without touching the live match.
        //
        // return_value_policy::move so the freshly built ClashEnv is moved
        // into the Python object rather than copied again; copying it would be
        // correct but would redo the whole deep copy a second time.
        .def("snapshot", &ClashEnv::snapshot, py::return_value_policy::move,
             "Independent deep copy of this environment, for decision-time "
             "search or what-if analysis. Stepping the copy cannot affect the "
             "original: entities, and the stats collectors behind "
             "get_tower_damage_dealt()/get_elixir_spent(), are all duplicated, "
             "and a projectile in flight is re-pointed at the copy's own "
             "target rather than the original's. Cumulative statistics carry "
             "over, so a rollout continues the match's totals instead of "
             "restarting them. The replay log does NOT carry over (a rollout "
             "is a hypothetical, not part of the match). Roughly 150x cheaper "
             "than one network forward -- see perception/UPSTREAM_REQUESTS.md "
             "item 13.")
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
        .def_readonly_static("MAX_BUILDING_HP", &ClashEnv::MAX_BUILDING_HP)
        // Attribute-channel indices and the appended-scalar count, bound for
        // the same reason every other structural constant here is: the Python
        // side derives its layout math from the engine instead of keeping a
        // hand-synced copy. model.py and train_selfplay.py's scripted
        // opponents both index the raw observation directly.
        .def_readonly_static("CH_COUNT", &ClashEnv::CH_COUNT)
        .def_readonly_static("CH_FLYING", &ClashEnv::CH_FLYING)
        .def_readonly_static("CH_ANTIAIR", &ClashEnv::CH_ANTIAIR)
        .def_readonly_static("CH_DPS", &ClashEnv::CH_DPS)
        .def_readonly_static("CH_RANGE", &ClashEnv::CH_RANGE)
        .def_readonly_static("CH_SPEED", &ClashEnv::CH_SPEED)
        // Opponent card-cycle blocks (item 24, 2026-08-27). Bound for exactly
        // the reason the block above is: `MicroRoyaleNet` sizes its scalar
        // input from these, and a hand-synced copy of 2*185 in Python is the
        // defect this project has now hit six times.
        .def_readonly_static("NUM_CYCLE_BLOCKS", &ClashEnv::NUM_CYCLE_BLOCKS)
        .def_readonly_static("CYCLE_BLOCK_SIZE", &ClashEnv::CYCLE_BLOCK_SIZE)
        .def_readonly_static("CYCLE_RECENCY_TAU_TICKS", &ClashEnv::CYCLE_RECENCY_TAU_TICKS)
        // FORWARD offsets into the observation. Bound because locating a
        // section by subtracting from the end silently re-points the moment
        // anything is appended behind it -- which is exactly what the cycle
        // blocks did to five call sites that used to read the tower HPs.
        .def_readonly_static("EXTRA_SCALARS_START", &ClashEnv::EXTRA_SCALARS_START)
        .def_readonly_static("CYCLE_START", &ClashEnv::CYCLE_START)
        .def_readonly_static("NUM_EXTRA_SCALARS", &ClashEnv::NUM_EXTRA_SCALARS)
        .def_readonly_static("MAX_MATCH_ELIXIR", &ClashEnv::MAX_MATCH_ELIXIR)
        // Elixir phase boundaries, in TICKS (10 ticks = 1 s). Bound from
        // GameManager, which is where the schedule lives -- the viewer and
        // perception/ both need the boundaries themselves and not just the
        // current value, and neither may restate them.
        .def_readonly_static("DOUBLE_ELIXIR_TICK", &GameManager::DOUBLE_ELIXIR_TICK)
        .def_readonly_static("TRIPLE_ELIXIR_TICK", &GameManager::TRIPLE_ELIXIR_TICK);

    m.def("get_all_card_ids", &getAllCardIds,
        "All ids CardRegistry currently has registered (real, playable cards only).");

    // Per-card registry facts the Python side cannot otherwise see. Added
    // because three separate Python-side problems all reduced to "the trainer
    // has no way to ask what a card IS":
    //
    //  * is_spell: the action space caps target_y at getOwnHalfMaxY() for every
    //    card, but isValidPlacement deliberately exempts spells from the
    //    own-half restriction. Without this flag the agent could never aim a
    //    Fireball past the river -- an entire card in the deck was unusable for
    //    its actual purpose, and no amount of training could fix it.
    //  * cost: lets the affordability action mask use the registry's own number
    //    instead of re-deriving it from the observation's scaled copy.
    //  * is_champion: python_ai/gym_wrapper.py had to hand-maintain a
    //    DEFAULT_DECK_ABILITY_SLOTS constant next to the deck literal purely
    //    because this was not queryable; it can now be derived.
    //
    // Read-only accessor over data CardRegistry already holds -- no engine
    // behavior changes.
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

    // Exposes the exact same slot-legality check GameManager::reset()/
    // setOpponentDeck() already enforce (throwing on a non-empty result) --
    // lets a caller pre-validate (or rejection-sample) a random 8-card deck
    // from Python without duplicating the Evolution/Champion slot rules on
    // that side. Returns "" for a legal deck, otherwise a human-readable
    // reason naming the offending slot/card.
    m.def("validate_deck_slots", &validateDeckSlots,
        "Returns \"\" if `deck` (8 card ids) satisfies Evolution/Champion slot-position rules, else an error string.");

    // Correct-by-construction replacement for the various Python-side
    // "random.sample(get_all_card_ids(), 8), retry until validate_deck_slots
    // passes" patterns (gym_wrapper.py, train.py, train_selfplay.py) --
    // sampleRandomDeck itself takes an std::mt19937& (reusable from other
    // C++ callers), which pybind11 can't expose directly; this lambda wraps
    // it with its own internally-seeded generator (seeded once, like
    // ClashEnv::rng/GameManager::rng) so the Python-facing call takes no
    // arguments.
    // The THIRD generator in this project, and the one ClashEnv::seed cannot
    // reach: it is not a member of anything. Item 7 seeded ClashEnv::rng (the
    // HeuristicOpponent) and GameManager::rng (the opening hand and cycle
    // order), which left a run with randomize_opp_deck=True reproducible in
    // every respect EXCEPT the opponent's deck -- the largest single remaining
    // source of episode-to-episode variance.
    //
    // WHY A SEEDED CALL GETS ITS OWN GENERATOR rather than seeding the static.
    // The static is process-global, so at num_envs=8 every env interleaves
    // draws from one stream: seeding it would make a result reproducible only
    // for a fixed construction and call ORDER, which is not a property a
    // vectorised trainer can offer. A local generator is order-independent by
    // construction.
    //
    // Passing no seed keeps the previous behaviour bit-for-bit -- same static,
    // same stream -- so every existing zero-argument call site is unaffected.
    // See perception/UPSTREAM_REQUESTS.md item 23C.
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
       "process-global stream, which is what every pre-2026-08-24 call site used.");
}
