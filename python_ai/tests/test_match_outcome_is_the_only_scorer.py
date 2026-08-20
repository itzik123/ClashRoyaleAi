"""Nothing outside match_outcome.py may score a match from tower counts.

WHY THIS TEST EXISTS
--------------------
This is the only test in the suite that greps source instead of running code,
and it earns that because the defect it guards has now appeared SEVEN times in
three waves.

`include/core/TimeoutRules.h` decides a finished match in three ordered steps:
fewer surviving towers loses; on equal counts the side whose weakest surviving
tower has lower HP loses; only an exact tie is a draw. It has exactly one call
site in C++ (`ClashEnv::calculateReward`) and is not reachable from Python at
all.

So every script wanting an outcome without going through `reward` re-derives
one, and every single time it re-derives only the FIRST rule:

    a, b = env.get_towers_alive(0), env.get_towers_alive(1)
    return 1.0 if a > b else (0.5 if a == b else 0.0)

A match ending 3-3 on towers but 1200 HP against 90 HP on the weakest is a
clear win by the engine's own rules, and that expression calls it a draw -- in
the scripts whose entire output is a win rate.

Wave 1 (2026-08-19) fixed four: net_ab, net_h2h, net_h2h_search x2.
Wave 2, same day, found a fifth in tools/validate_pipeline.py, where the
  count-only version was ABSORBING the very signal that test exists to detect
  (equal-count finishes fell into `draws`, worth 0.5 either way).
Wave 3 (2026-08-20) found three MORE in files written after the fix landed:
  eval/prove_combos.py, eval/prove_teacher.py, eval/prove_environment.py.

Fixing instances plainly does not hold, because the expression is short,
obvious, and easy to retype from scratch. So this test fails on the PATTERN.

THE REAL FIX, AND WHY THIS IS A STOPGAP
---------------------------------------
`perception/UPSTREAM_REQUESTS.md` item 16 proposes binding
`TimeoutRules::resolve` directly, which would delete the hand-written Python
mirror entirely and make this test unnecessary. Until that lands, this is what
keeps the mirror singular.

IF THIS TEST FAILS
------------------
Do not add your file to the allowlist. Call
`python_ai.eval.match_outcome.score_from_towers(env, team)` instead -- it
applies all three rules and needs no new binding, reading the six tower-HP
scalars the observation already carries.
"""
import os
import re

import python_ai

# Files allowed to read tower counts next to an outcome-shaped literal.
#
#   match_outcome.py  -- IS the implementation.
#   this file         -- quotes the bad pattern in its own docstring.
ALLOWED = {
    os.path.join("eval", "match_outcome.py"),
    os.path.join("tests", "test_match_outcome_is_the_only_scorer.py"),
}

# The tell is a 1.0/0.5/0.0 (or 1.0/0.0/-1.0) ladder in the same statement as a
# tower-count comparison. Deliberately narrow: plenty of legitimate code reads
# get_towers_alive for crown counts or as a feature (opponents/teacher.py's
# "crowns" differential, eval/probe_perfect_defense.py's crowns_on_win), and
# those must not trip it.
SCORE_LADDER = re.compile(
    r"get_towers_alive.*\n?.*?(?:1\.0\s+if.*else|if\s+a\s*>\s*b)",
    re.MULTILINE,
)


def _python_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in {"__pycache__", "venv", ".venv", "archive"}
                       and not d.startswith("archive_")]
        for name in filenames:
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def test_only_match_outcome_scores_from_tower_counts():
    root = os.path.join(python_ai.REPO_ROOT, "python_ai")
    offenders = []
    for path in _python_files(root):
        rel = os.path.relpath(path, root)
        if rel in ALLOWED:
            continue
        text = open(path, encoding="utf-8", errors="replace").read()
        if "get_towers_alive" not in text:
            continue
        if SCORE_LADDER.search(text):
            offenders.append(rel)

    assert not offenders, (
        "These files score a match outcome from tower counts alone, which "
        "ignores TimeoutRules' weakest-tower tie-break and reports a draw for "
        "matches the engine calls a win:\n  "
        + "\n  ".join(sorted(offenders))
        + "\n\nUse python_ai.eval.match_outcome.score_from_towers(env, team). "
          "Do not add the file to ALLOWED -- see this module's docstring."
    )


# ---------------------------------------------------------------------------
# The same pattern, outside Python.
#
# The guard above walks .py files ONLY, and that is exactly why it never saw
# web/viewer.html -- which scored every timed-out match from "are both King
# Towers alive?" and called the rest a draw, right up until 2026-08-21. A
# replay ending 3-3 on towers with a Princess at 90 hp displayed as
# "Draw. Timeout - both King Towers still standing".
#
# The defect is language-independent, so the guard has to be too.

def test_the_replay_viewer_reads_the_engines_verdict():
    viewer = os.path.join(python_ai.REPO_ROOT, "web", "viewer.html")
    text = open(viewer, encoding="utf-8", errors="replace").read()

    assert "gameData.result" in text, (
        "web/viewer.html must consume the engine's own `result` object "
        "(GameLogger::resultJson) rather than re-deriving a verdict of its own."
    )

    # The specific shortcut that caused the bug: concluding 'draw' straight off
    # both Kings being alive, with no tower-count or weakest-hp tie-break.
    king_only_draw = re.compile(
        r"kingAlive\[0\]\s*&&\s*kingAlive\[1\][^\n]*\n?[^\n]*winner:\s*['\"]draw['\"]"
    )
    assert not king_only_draw.search(text), (
        "web/viewer.html calls a match a draw purely because both King Towers "
        "are alive. That is MatchRules::evaluate's question ('has a King died "
        "yet?'), not TimeoutRules' ('who won at the limit?'). Apply tower count, "
        "then the weakest tower's ABSOLUTE hp, before concluding a draw."
    )

    # And the fallback for pre-`result` replays must actually implement the
    # tie-breaks rather than bailing out to a draw.
    for needed in ("count[0] !== count[1]", "weakest[0] !== weakest[1]"):
        assert needed in text, (
            f"web/viewer.html's fallback for older replays is missing {needed!r}. "
            "It must be a port of TimeoutRules, not the king-alive shortcut."
        )
