"""`tools/run_watchdog.py` across the phase-1 -> phase-2 handoff: both phases are
recognised in every launch form, and the phase furthest along is the one
relaunched.
"""
import pytest

from python_ai.tools import run_watchdog as W


@pytest.mark.parametrize("cmd,phase", [
    (r"C:\py\python.exe -u -m python_ai.trainers.train", "phase1"),
    (r"C:\py\python.exe -u python_ai\trainers\train.py", "phase1"),
    (r"C:\py\python.exe -u python_ai/trainers/train.py", "phase1"),
    (r"C:\py\python.exe -u C:\repo\python_ai\trainers\train_selfplay.py", "phase2"),
    (r"C:\py\python.exe -u -m python_ai.trainers.train_selfplay", "phase2"),
    (r"C:\py\python.exe -m pytest python_ai/tests", None),
    (r"C:\py\python.exe -m python_ai.tools.monitor_run --dir .", None),
])
def test_both_phases_are_recognised_in_every_launch_form(cmd, phase):
    assert W.classify(cmd) == phase


def test_after_the_handoff_the_watchdog_relaunches_phase_2(tmp_path, monkeypatch):
    weights = tmp_path / "model_weights.pth"
    monkeypatch.setenv("CLASH_WEIGHTS", str(weights))
    assert W.phase_to_resume() == "phase1"
    (tmp_path / "model_weights_selfplay.pth").write_bytes(b"x")
    assert W.phase_to_resume() == "phase2"


def test_there_is_no_default_checkpoint_from_an_old_run():
    parser = W.build_parser()
    assert parser.get_default("weights") is None
