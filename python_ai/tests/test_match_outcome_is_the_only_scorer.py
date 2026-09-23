"""Nothing outside match_outcome.py may score a match from tower counts.

`include/core/TimeoutRules.h` decides a finished match in three ordered steps
(fewer towers loses; on equal counts the lower weakest tower loses; only an
exact tie is a draw), but scripts keep re-deriving only the first:

    a, b = env.get_towers_alive(0), env.get_towers_alive(1)
    return 1.0 if a > b else (0.5 if a == b else 0.0)

which calls a 3-3 finish at 1200 HP vs 90 HP a draw. The expression is short
and easy to retype, so this test fails on the pattern.

If it fails, do not add your file to the allowlist: call
`python_ai.eval.match_outcome.score_from_towers(env, team)`.
"""
import os
import re

import python_ai

# Files allowed to read tower counts next to an outcome-shaped literal:
#   match_outcome.py  the implementation
#   this file         quotes the pattern in its docstring
ALLOWED = {
    os.path.join("eval", "match_outcome.py"),
    os.path.join("tests", "test_match_outcome_is_the_only_scorer.py"),
}

# The tell is a 1.0/0.5/0.0 (or 1.0/0.0/-1.0) ladder beside a tower-count
# comparison. Narrow on purpose: code that reads get_towers_alive for crown
# counts or as a feature must not trip it.
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


# --- the same pattern, outside Python ---
# The guard above walks .py files only; the viewer must consume the engine's
# verdict too.

def test_the_replay_viewer_reads_the_engines_verdict():
    viewer = os.path.join(python_ai.REPO_ROOT, "web", "viewer.html")
    text = open(viewer, encoding="utf-8", errors="replace").read()

    assert "gameData.result" in text, (
        "web/viewer.html must consume the engine's own `result` object "
        "(GameLogger::resultJson) rather than re-deriving a verdict of its own."
    )

    # The shortcut to forbid: "draw" straight off both Kings being alive, with
    # no tie-break.
    king_only_draw = re.compile(
        r"kingAlive\[0\]\s*&&\s*kingAlive\[1\][^\n]*\n?[^\n]*winner:\s*['\"]draw['\"]"
    )
    assert not king_only_draw.search(text), (
        "web/viewer.html calls a match a draw purely because both King Towers "
        "are alive. That is MatchRules::evaluate's question ('has a King died "
        "yet?'), not TimeoutRules' ('who won at the limit?'). Apply tower count, "
        "then the weakest tower's ABSOLUTE hp, before concluding a draw."
    )

    # The fallback for pre-`result` replays must implement the tie-breaks.
    for needed in ("count[0] !== count[1]", "weakest[0] !== weakest[1]"):
        assert needed in text, (
            f"web/viewer.html's fallback for older replays is missing {needed!r}. "
            "It must be a port of TimeoutRules, not the king-alive shortcut."
        )
