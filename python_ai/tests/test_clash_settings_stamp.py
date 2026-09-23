"""Every CLASH_* setting is stamped into the checkpoint and compared on resume.

TODO 00.9. Each CLASH_* variable is read at IMPORT -- the reward weights, gamma,
the spell anneal, the scenario mix, the deck pool -- so the process that resumes
a run is configured by whatever the relaunching shell happens to hold. Only the
deck was recorded, so a crash-resume under a different CLASH_GAMMA (or a missing
one) continued silently under a different objective.
"""
from python_ai.rl import checkpointing as C


def test_the_stamp_is_every_clash_variable_and_nothing_else():
    env = {"CLASH_GAMMA": "0.99", "CLASH_DECK": "hog rider", "PATH": "x",
           "clash_lower": "y", "CLASH_SEED": "3"}
    assert C.clash_settings(env) == {"CLASH_DECK": "hog rider",
                                     "CLASH_GAMMA": "0.99", "CLASH_SEED": "3"}


def test_identical_settings_report_nothing():
    s = {"CLASH_GAMMA": "0.99", "CLASH_LOGDIR": "runs/a"}
    assert C.settings_drift(s, dict(s)) == ([], [])


def test_a_changed_setting_that_alters_the_run_is_reported_as_such():
    changed, operational = C.settings_drift({"CLASH_GAMMA": "0.999"},
                                            {"CLASH_GAMMA": "0.99"})
    assert changed == [("CLASH_GAMMA", "0.999", "0.99")] and operational == []


def test_adding_or_removing_a_setting_counts_as_a_change():
    """The failure this exists for is usually a variable the relaunching shell
    simply does not have -- unset must compare unequal to set."""
    changed, _ = C.settings_drift({"CLASH_SOLVENCY": "0"}, {})
    assert changed == [("CLASH_SOLVENCY", "0", None)]
    changed, _ = C.settings_drift({}, {"CLASH_W_WINCON_DAMAGE": "0.2"})
    assert changed == [("CLASH_W_WINCON_DAMAGE", None, "0.2")]


def test_paths_cadence_workers_and_seed_are_operational_not_alarming():
    changed, operational = C.settings_drift(
        {"CLASH_LOGDIR": "runs/a", "CLASH_SAVE_EVERY": "250", "CLASH_SEED": "1"},
        {"CLASH_LOGDIR": "runs/b", "CLASH_SAVE_EVERY": "100"})
    assert changed == []
    assert [k for k, _a, _b in operational] == ["CLASH_LOGDIR", "CLASH_SAVE_EVERY",
                                                "CLASH_SEED"]


def test_the_deck_is_left_to_the_resolved_deck_check():
    """`CLASH_DECK` takes names or ids, so two different strings can be one deck;
    restore_common compares the RESOLVED ids and warns on its own."""
    assert C.settings_drift({"CLASH_DECK": "hog rider,musketeer"},
                            {"CLASH_DECK": "15,6"}) == ([], [])
