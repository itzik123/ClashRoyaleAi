"""The package structure, pinned: without these checks the tree drifts back into
cross-layer imports.
"""
import ast
import importlib
import pathlib
import pkgutil

import pytest

import python_ai

PKG = pathlib.Path(python_ai.PACKAGE_DIR)
PACKAGES = ("models", "envs", "opponents", "advisors", "rewards", "rl",
            "trainers", "search", "eval", "tools", "tests")


def _modules(package):
    return [f"python_ai.{package}.{m.name}"
            for m in pkgutil.iter_modules([str(PKG / package)])]


def _first_party_imports(path):
    """Every `python_ai.*` module this file imports at module scope."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for node in tree.body:                       # module scope only
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.startswith("python_ai"):
                base = node.module
                out.add(base)
                for alias in node.names:
                    out.add(f"{base}.{alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("python_ai"):
                    out.add(alias.name)
    return out


# --- the package ---
def test_every_package_exists_and_is_importable():
    for name in PACKAGES:
        assert (PKG / name).is_dir(), name
        # `tests/` is a namespace portion, not a regular package: an
        # __init__.py there changes how pytest names collected modules, and
        # `python_ai.tests.helpers` resolves either way.
        if name != "tests":
            assert (PKG / name / "__init__.py").is_file(), name
        importlib.import_module(f"python_ai.{name}")


def test_the_shared_test_helpers_are_importable_by_their_package_path():
    from python_ai.tests import helpers
    assert callable(helpers.chunk_fixture)
    assert callable(helpers.shaping_stats)


def test_no_python_module_is_left_loose_at_the_package_root():
    """Only the package marker, the pytest bootstrap, the shipping config, the
    engine constants and the trainee's deck live at the root. The last two are
    leaves every layer reads and that import nothing from python_ai, so any
    subpackage owning them would point an import edge the wrong way.
    """
    loose = {p.name for p in PKG.glob("*.py")}
    assert loose == {"__init__.py", "conftest.py", "shipping.py",
                     "engine_constants.py", "deck.py"}, loose


def test_importing_the_package_makes_the_compiled_engine_importable():
    """The .pyd is unpackaged in python_ai/; `__init__` puts it on the path once.
    """
    assert python_ai.PACKAGE_DIR in __import__("sys").path
    importlib.import_module("clash_royale_env")


def test_repo_root_is_one_level_above_the_package():
    assert (pathlib.Path(python_ai.REPO_ROOT) / "python_ai").is_dir()
    assert (pathlib.Path(python_ai.REPO_ROOT) / "include").is_dir()


# --- dependency direction ---
def test_the_model_layer_never_imports_a_trainer_or_an_environment():
    """`policy_io` turns a .pth into a ready net without importing an experiment
    script or a trainer.
    """
    for name in _modules("models"):
        path = PKG / "models" / f"{name.rsplit('.', 1)[1]}.py"
        for imported in _first_party_imports(path):
            assert ".trainers" not in imported, f"{name} imports {imported}"
            assert ".envs" not in imported, f"{name} imports {imported}"
            assert ".eval" not in imported, f"{name} imports {imported}"


def test_the_rl_package_never_imports_a_trainer():
    """The direction is trainers -> rl, never back."""
    for name in _modules("rl"):
        path = PKG / "rl" / f"{name.rsplit('.', 1)[1]}.py"
        for imported in _first_party_imports(path):
            assert ".trainers" not in imported, f"{name} imports {imported}"


def test_the_reward_layer_imports_no_torch_and_no_trainer():
    """The shaping terms are pure functions of engine statistics: no torch
    (testable with plain arrays, cannot touch the policy) and no trainer or rl
    import.
    """
    for module in ("shaping", "weights", "elixir_shaping"):
        src = (PKG / "rewards" / f"{module}.py").read_text(encoding="utf-8")
        assert "import torch" not in src, module
        for imported in _first_party_imports(PKG / "rewards" / f"{module}.py"):
            assert ".trainers" not in imported
            assert ".rl" not in imported


def test_the_shipping_config_stays_cheap_to_import():
    """The deployable configuration must be the cheapest thing in the tree to
    import.
    """
    imports = _first_party_imports(PKG / "shipping.py")
    assert not any(".trainers" in i for i in imports), imports
    assert not any(".eval" in i for i in imports), imports


def test_the_search_core_is_not_inside_an_experiment_script():
    """The search core is public in search/search.py, not private names inside an
    A/B harness.
    """
    from python_ai.search import search
    for name in ("policy_head", "greedy_from_logits", "build_candidates",
                 "search_action", "play_episode", "outcome_score"):
        assert hasattr(search, name), name
        assert not name.startswith("_")


def test_nobody_imports_the_search_internals_from_the_ab_harness_any_more():
    offenders = []
    for path in PKG.rglob("*.py"):
        # This file names the forbidden import; a scan matching its own source
        # could only fail.
        if ("__pycache__" in path.parts or path.name == "search_ab_test.py"
                or path.resolve() == pathlib.Path(__file__).resolve()):
            continue
        src = path.read_text(encoding="utf-8")
        if "search_ab_test import" in src:
            offenders.append(path.relative_to(PKG).as_posix())
    assert not offenders, offenders


# --- entry points ---
def test_every_runnable_script_can_find_the_package_when_run_as_a_file():
    """`python python_ai/eval/prove_hog.py` must keep working: run as a file the
    repo root is not on sys.path.
    """
    missing = []
    for path in sorted(PKG.rglob("*.py")):
        if "__pycache__" in path.parts or "venv" in path.parts:
            continue
        if path.parts[-2:] == ("python_ai", "conftest.py"):
            continue
        src = path.read_text(encoding="utf-8")
        if '__main__' not in src:
            continue
        if "sys.path.insert" not in src:
            missing.append(path.relative_to(PKG).as_posix())
    assert not missing, f"runnable but cannot import python_ai: {missing}"


def test_no_module_still_carries_the_old_flat_sys_path_bootstrap():
    """`sys.path.insert(0, dirname(__file__))` would let a module be imported
    twice under two names.
    """
    needle = "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))"
    offenders = []
    for path in PKG.rglob("*.py"):
        if ("__pycache__" in path.parts
                or path.resolve() == pathlib.Path(__file__).resolve()):
            continue
        if needle in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(PKG).as_posix())
    assert not offenders, offenders


@pytest.mark.parametrize("package", PACKAGES)
def test_every_module_in_every_package_imports(package):
    """The cheapest regression on a rename: an unreachable module is invisible
    until someone runs it.
    """
    for name in _modules(package):
        if package == "tests":
            continue
        importlib.import_module(name)


# --- derive, do not restate ---
# Arena geometry is bound as `clash_royale_env.ARENA_*` and surfaced through
# `engine_constants`, so every consumer here can derive it. A restated constant
# is correct the day it is written and wrong the day the arena moves.

def test_no_module_restates_a_geometry_constant_it_could_derive():
    """A module-level literal equal to a bound arena value is a second copy.

    Matched on the value, and to avoid false hits (`FIREBALL_RADIUS = 2.5`
    collides with `LEFT_BRIDGE_X = 2.5`) only under a positional-looking name.
    That misses a coordinate hidden behind an unrelated name.
    """
    import pathlib
    import re

    from python_ai import engine_constants as EC

    derivable = {
        float(EC.BRIDGE_Y): "engine_constants.BRIDGE_Y",
        float(EC.LEFT_BRIDGE_X): "engine_constants.LEFT_BRIDGE_X",
        float(EC.RIGHT_BRIDGE_X): "engine_constants.RIGHT_BRIDGE_X",
    }
    root = pathlib.Path(EC.__file__).parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(("tests/", "venv/", "archive")) or rel == "engine_constants.py":
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            m = re.match(r"\s*([A-Z_][A-Z0-9_]*)\s*=\s*(-?\d+\.\d+)\s*$", code)
            positional = m and re.search(
                r"(^|_)(X|Y|RIVER|BRIDGE|LANE|TOWER|KING|ARENA|BOARD|ROW|COL)(_|$)",
                m.group(1))
            if m and positional and float(m.group(2)) in derivable:
                offenders.append(
                    f"{rel}:{i}: {m.group(1)} = {m.group(2)} "
                    f"-- derive from {derivable[float(m.group(2))]}")
    assert not offenders, (
        "arena geometry restated instead of derived:\n" + "\n".join(offenders))


def test_the_offensive_scenario_reads_its_geometry_from_the_engine():
    """The offensive scenario reads its bridges and river from the engine; pinning
    the equality fails the next time the arena moves.
    """
    from python_ai import engine_constants as EC
    from python_ai.envs import scenario_offense
    assert scenario_offense.RIVER_Y == EC.BRIDGE_Y
    assert scenario_offense.BRIDGE_XS == (EC.LEFT_BRIDGE_X, EC.RIGHT_BRIDGE_X)
