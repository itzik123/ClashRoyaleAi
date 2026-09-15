"""The curriculum at a FROM-SCRATCH start (audit 04, 2026-09-15).

A random-init policy begins at rung 0. Four findings, each measured:

C1  At rung 0 every valve is structurally silent. `maybe_demote_stage` returns
    None at stage 0 (there is nowhere to go) and the advance/plateau valves need
    a win rate the run does not have. Driven through the real manager, 4,000
    episodes at rung 0 / 0.00 win rate fired NO event, and every OTHER failure
    path demotes down to rung 0 and then goes quiet too. `monitor_run.py` reads
    no win rate at all. A dead from-scratch run was indistinguishable from
    healthy early training.

C5  `load_state_dict` re-ran the one-off 6 -> 11 rung migration on ANY
    table-size mismatch. Adding a single rung sent a rung-3 checkpoint to rung 7
    while printing "(same lookahead horizon)", which was false.

C6  The plateau detector's progress signal rides on the deck read-out, which
    `on_episode_end` skipped for scenario episodes -- about 30% of calls lost,
    taking `Decks/WinRate_Min` with them.

C4  `setup_ab_arm.py` wrote `curriculum_stage` with no `teacher_table_size`
    stamp, so `--stage 5` resumed at rung 10.
"""
import types

import numpy as np
import pytest

from python_ai.rl import curriculum as C
from python_ai.rl.curriculum import CurriculumManager, new_outcome_window


def manager():
    return CurriculumManager(entry_win_rate=0.60, min_stage_for_phase2=8,
                             phase2_win_rate_gate=0.80,
                             max_episodes_per_deck=1250,
                             random_opponent_budget=5000)


def losing_window():
    w = new_outcome_window()
    w.extend([-1] * w.maxlen)
    return w


# --- C1: the rung-0 floor alarm --------------------------------------------

def test_a_run_that_never_wins_at_rung_0_raises_the_floor_alarm():
    m = manager()
    assert m.stage == 0
    patience = C.FLOOR_ALARM_PATIENCE_EPISODES
    assert m.floor_alarm(losing_window(), patience - 1) is None
    msg = m.floor_alarm(losing_window(), patience)
    assert msg and "rung" in msg.lower()


def test_the_floor_alarm_re_arms_rather_than_firing_every_episode():
    m = manager()
    p = C.FLOOR_ALARM_PATIENCE_EPISODES
    assert m.floor_alarm(losing_window(), p)
    assert m.floor_alarm(losing_window(), p + 1) is None
    assert m.floor_alarm(losing_window(), 2 * p)


def test_the_floor_alarm_is_silent_for_a_run_that_is_winning():
    """CONTROL: must NOT fire, or the alarm is just a timer."""
    m = manager()
    w = new_outcome_window()
    w.extend([1] * 20 + [-1] * (w.maxlen - 20))
    assert m.floor_alarm(w, 10 * C.FLOOR_ALARM_PATIENCE_EPISODES) is None


def test_the_floor_alarm_is_silent_above_rung_0():
    """Above rung 0 the stall valve owns this case and CAN act on it."""
    m = manager()
    m.stage = 3
    assert m.floor_alarm(losing_window(), 10 * C.FLOOR_ALARM_PATIENCE_EPISODES) is None


# --- C5: only the legacy migration remaps ------------------------------------

def test_an_unstamped_legacy_checkpoint_is_still_remapped_by_horizon():
    m = manager()
    m.load_state_dict({"curriculum_stage": 3, "stage_start_episode": 0})
    assert m.stage == 6, "old six-rung stage 3 (5 s) is new rung 6"


def test_a_checkpoint_stamped_with_a_different_table_size_is_NOT_remapped(capsys):
    m = manager()
    m.load_state_dict({"curriculum_stage": 3, "stage_start_episode": 0,
                       "teacher_table_size": len(C.CURRICULUM_STAGES) + 1})
    assert m.stage == 3
    assert "NOT remapping" in capsys.readouterr().out


def test_a_stamped_index_beyond_the_table_is_clamped():
    m = manager()
    m.load_state_dict({"curriculum_stage": 99, "stage_start_episode": 0,
                       "teacher_table_size": 99})
    assert m.stage == len(C.CURRICULUM_STAGES) - 1


# --- C6: a scenario episode still feeds the deck read-out ----------------------

def test_a_scenario_episode_still_runs_the_deck_read_out():
    from python_ai.trainers.train import Phase1Trainer
    calls = []

    class Stub:
        cfg = types.SimpleNamespace(num_envs=1)
        metrics = types.SimpleNamespace(reset_env=lambda i: None)

        def _print_deck_pool(self):
            calls.append(True)

    ctx = types.SimpleNamespace(infos={"is_scenario": np.array([1.0], np.float32)})
    Phase1Trainer.on_episode_end(Stub(), 0, ctx)
    assert calls, "a scenario episode skipped the deck read-out and note_progress"


# --- C4: the A/B arm tool stamps the table ---------------------------------------

def test_an_ab_arm_resumes_at_the_rung_it_was_built_for(tmp_path, monkeypatch):
    """Build a real arm with --stage 5 and load it back through the manager.
    Unstamped, rung 5 (40 ticks) silently became rung 10 (100 ticks)."""
    import sys
    import torch
    from python_ai.models.net import MicroRoyaleNet
    from python_ai.tools import setup_ab_arm

    src, dst = tmp_path / "src.pth", tmp_path / "arm.pth"
    torch.save(MicroRoyaleNet(num_ability_slots=0).state_dict(), src)
    monkeypatch.setattr(sys, "argv", ["setup_ab_arm", "--src", str(src),
                                      "--dst", str(dst), "--stage", "5",
                                      "--episodes", "0"])
    setup_ab_arm.main()
    ck = torch.load(dst, map_location="cpu", weights_only=False)
    m = manager()
    m.load_state_dict(ck)
    assert m.stage == 5
