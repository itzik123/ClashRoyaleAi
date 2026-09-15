"""Two preflight gates that could not tell the truth (2026-09-15).

1. "the test binary post-dates the engine source" compared MTIMES. Tooling that
   rewrote six headers byte-for-byte made it FAIL on a build that was verified
   current behaviourally -- the failure mode CLAUDE.md calls the worst a gate
   has: it teaches you to ignore it. A source file now counts as changed after
   the build only if its CONTENT differs from what was committed before it.

2. "side null" needed `model_weights_selfplay.pth`, deleted in the 2026-08-19
   cleanup, so the one diagnostic that catches an observation-shaped side
   asymmetry had been permanently unrunnable. It does not need a TRAINED net: a
   seeded random-init net against a bit-exact copy of itself is a sharper
   subject, because an untrained policy has no side-specific skill to confound a
   structural asymmetry.
"""
import os
import subprocess
import time

import pytest

from python_ai.tools import validate_pipeline as VP


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "include").mkdir()
    src = tmp_path / "include" / "A.h"
    src.write_text("int a = 1;\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path, src


def test_a_touched_but_unchanged_source_is_not_stale(repo):
    root, src = repo
    built = time.time()
    os.utime(src, (built + 100, built + 100))          # touched after the build
    assert VP._sources_changed_since(built, root=str(root)) == []


def test_an_edited_source_is_stale(repo):
    """CONTROL that must fire: a real edit after the build."""
    root, src = repo
    built = time.time()
    src.write_text("int a = 2;\n")
    os.utime(src, (built + 100, built + 100))
    assert VP._sources_changed_since(built, root=str(root)) == ["include/A.h"]


def test_the_side_null_runs_without_a_checkpoint(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(VP, "check", lambda name, ok, detail="": seen.append((name, ok, detail)))
    VP.validate_side_null(str(tmp_path / "does_not_exist.pth"), episodes=2)
    names = [n for n, _, _ in seen]
    assert "side null" not in names or all(ok for n, ok, _ in seen if n == "side null"), seen
    assert any("side advantage" in n for n in names), seen
