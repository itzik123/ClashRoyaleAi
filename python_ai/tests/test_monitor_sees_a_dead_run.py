"""`tools/monitor_run.py` must tell a dead run from a slow one."""
import sys

import pytest

from python_ai.tools import monitor_run


def _write_run(root, win_rate, floor_alarm=False, episodes=2400, stage=0):
    from torch.utils.tensorboard import SummaryWriter
    w = SummaryWriter(str(root / "runs" / "exp"))
    for ep in range(0, episodes + 1, 100):
        w.add_scalar("Loss/Actor", 0.01, ep)
        w.add_scalar("Loss/Critic", 0.05, ep)
        w.add_scalar("Loss/Entropy", 1.0, ep)
        w.add_scalar("Training/Win_Rate_100", win_rate(ep), ep)
        w.add_scalar("Training/Avg_Reward_50", -1.0 + 2.0 * win_rate(ep), ep)
        w.add_scalar("Training/Curriculum_Stage", stage, ep)
    if floor_alarm:
        w.add_scalar("Training/Curriculum_FloorAlarm", 1.0, episodes)
    w.close()


def _run(monkeypatch, root, capsys):
    monkeypatch.setattr(sys, "argv", ["monitor_run", "--dir", str(root),
                                      "--skip-placement"])
    code = monitor_run.main()
    return code, capsys.readouterr().out


def test_a_run_pinned_at_zero_wins_is_an_alarm(tmp_path, monkeypatch, capsys):
    _write_run(tmp_path, lambda ep: 0.0, floor_alarm=True)
    code, out = _run(monkeypatch, tmp_path, capsys)
    assert code == 1
    assert "[ALARM] win rate" in out
    assert "[ALARM] floor alarm" in out


def test_a_learning_run_is_not_an_alarm(tmp_path, monkeypatch, capsys):
    """Control: the check must not fire on a run that is getting better."""
    _write_run(tmp_path, lambda ep: min(0.6, ep / 4000.0), stage=2)
    code, out = _run(monkeypatch, tmp_path, capsys)
    assert "[ALARM] win rate" not in out
    assert code == 0, out
