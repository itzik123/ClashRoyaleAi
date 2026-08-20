"""The structure itself, pinned. Layout decays silently unless something checks.

Every assertion here corresponds to a specific coupling this refactor removed.
Without them the tree drifts straight back: someone needs a constant, imports
the module that happens to hold it, and three months later a probe that wanted
to read one .pth again pulls in two trainers, a card-registry probe and torch.
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


# ------------------------------------------------------------- the package --
def test_every_package_exists_and_is_importable():
    for name in PACKAGES:
        assert (PKG / name).is_dir(), name
        # `tests/` is deliberately a namespace portion rather than a regular
        # package: adding __init__.py there changes how pytest names collected
        # modules, and nothing needs it -- `python_ai.tests.helpers` resolves
        # either way because `python_ai` itself is a regular package.
        if name != "tests":
            assert (PKG / name / "__init__.py").is_file(), name
        importlib.import_module(f"python_ai.{name}")


def test_the_shared_test_helpers_are_importable_by_their_package_path():
    from python_ai.tests import helpers
    assert callable(helpers.chunk_fixture)
    assert callable(helpers.shaping_stats)


def test_no_python_module_is_left_loose_at_the_package_root():
    """Only the package marker, the pytest bootstrap, the shipping config and
    the engine constants belong here. Everything else has a home."""
    loose = {p.name for p in PKG.glob("*.py")}
    assert loose == {"__init__.py", "conftest.py", "shipping.py",
                     "engine_constants.py"}, loose


def test_importing_the_package_makes_the_compiled_engine_importable():
    """The .pyd is unpackaged and lives in python_ai/. Doing this once in
    __init__ is what replaced ~30 copies of a sys.path.insert line."""
    assert python_ai.PACKAGE_DIR in __import__("sys").path
    importlib.import_module("clash_royale_env")


def test_repo_root_is_one_level_above_the_package():
    assert (pathlib.Path(python_ai.REPO_ROOT) / "python_ai").is_dir()
    assert (pathlib.Path(python_ai.REPO_ROOT) / "include").is_dir()


# ------------------------------------------------------- dependency direction --
def test_the_model_layer_never_imports_a_trainer_or_an_environment():
    """`policy_io` exists precisely so a probe can turn a .pth into a ready net
    WITHOUT importing an experiment script and, through it, both trainers."""
    for name in _modules("models"):
        path = PKG / "models" / f"{name.rsplit('.', 1)[1]}.py"
        for imported in _first_party_imports(path):
            assert ".trainers" not in imported, f"{name} imports {imported}"
            assert ".envs" not in imported, f"{name} imports {imported}"
            assert ".eval" not in imported, f"{name} imports {imported}"


def test_the_rl_package_never_imports_a_trainer():
    """The direction is trainers -> rl, never back. A cycle here would make the
    algorithm depend on which pipeline is using it, which is the exact fusion
    this refactor undid."""
    for name in _modules("rl"):
        path = PKG / "rl" / f"{name.rsplit('.', 1)[1]}.py"
        for imported in _first_party_imports(path):
            assert ".trainers" not in imported, f"{name} imports {imported}"


def test_the_reward_layer_imports_no_torch_and_no_trainer():
    """The shaping terms are pure functions of the engine's own statistics.
    Keeping torch out of them is what makes them testable with plain arrays --
    and what makes it obvious that they cannot accidentally touch the policy."""
    for module in ("shaping", "weights", "elixir_shaping"):
        src = (PKG / "rewards" / f"{module}.py").read_text(encoding="utf-8")
        assert "import torch" not in src, module
        for imported in _first_party_imports(PKG / "rewards" / f"{module}.py"):
            assert ".trainers" not in imported
            assert ".rl" not in imported


def test_the_shipping_config_stays_cheap_to_import():
    """It was importing a 1,152-line experiment harness -- and through it
    bc_pretrain, torch and a card-registry probe -- to reach one dataclass.
    A file that names the deployable configuration should be the cheapest thing
    in the tree to read, not the most expensive."""
    imports = _first_party_imports(PKG / "shipping.py")
    assert not any(".trainers" in i for i in imports), imports
    assert not any(".eval" in i for i in imports), imports


def test_the_search_core_is_not_inside_an_experiment_script():
    """Five modules used to import `_build_candidates` and `_search_action` --
    underscore-private names -- out of an A/B harness whose main() runs a whole
    experiment."""
    from python_ai.search import search
    for name in ("policy_head", "greedy_from_logits", "build_candidates",
                 "search_action", "play_episode", "outcome_score"):
        assert hasattr(search, name), name
        assert not name.startswith("_")


def test_nobody_imports_the_search_internals_from_the_ab_harness_any_more():
    offenders = []
    for path in PKG.rglob("*.py"):
        # This file names the forbidden import in order to forbid it; a scan
        # that could match its own source can only ever fail.
        if ("__pycache__" in path.parts or path.name == "search_ab_test.py"
                or path.resolve() == pathlib.Path(__file__).resolve()):
            continue
        src = path.read_text(encoding="utf-8")
        if "search_ab_test import" in src:
            offenders.append(path.relative_to(PKG).as_posix())
    assert not offenders, offenders


# ------------------------------------------------------------ entry points --
def test_every_runnable_script_can_find_the_package_when_run_as_a_file():
    """`python python_ai/eval/prove_hog.py` must keep working: the repo root is
    not on sys.path there, so `python_ai.*` would not resolve. bc_pretrain.py
    was missing this bootstrap entirely and could not be run at all."""
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
    """`sys.path.insert(0, dirname(__file__))` put the module's OWN directory on
    the path -- which under the package layout would let `python_ai/eval/foo.py`
    be imported twice under two different names."""
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
    """The cheapest possible regression on a rename: an unreachable module is
    invisible until the day someone runs it."""
    for name in _modules(package):
        if package == "tests":
            continue
        importlib.import_module(name)
