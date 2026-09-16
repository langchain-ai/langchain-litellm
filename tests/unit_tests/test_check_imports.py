"""Test the check_imports script."""

# stdlib
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_imports.py"
PACKAGE = REPO_ROOT / "langchain_litellm"


def _run(
    *paths: str,
    warnings_as_errors: bool = False,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    argv = [sys.executable]
    if warnings_as_errors:
        argv += ["-W", "error::DeprecationWarning"]

    env = dict(os.environ)
    if cwd is not None:
        # `python script.py` puts the script's own directory on sys.path, never
        # the working directory, so a tree that is not the installed package
        # has to be put on the path explicitly for its names to resolve.
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = f"{cwd}{os.pathsep}{existing}" if existing else str(cwd)

    return subprocess.run(  # noqa: S603
        [*argv, str(SCRIPT), *paths],
        capture_output=True,
        cwd=cwd or REPO_ROOT,
        env=env,
        text=True,
        check=False,
    )


# ── the contract: every shipped module imports cleanly ───────────────────────


def test_sibling_modules_of_the_same_name_do_not_collide(tmp_path: Path) -> None:
    """Each file must be checked under its own module name.

    Two subpackages here each own a module called ``thing``. Loading every file
    under one shared name leaves the first ``thing`` in the import cache, so the
    second subpackage's relative import resolves against its sibling's file and
    cannot find its own symbol.

    The tree is built here rather than read off the shipped package, because
    detecting the collision needs two subpackages with a same-named module and
    nothing guarantees ``langchain_litellm`` keeps a pair like that.
    """
    root = tmp_path / "fixture_pkg"
    root.mkdir()
    (root / "__init__.py").write_text("", encoding="utf-8")
    for name, symbol in (("alpha", "ALPHA"), ("beta", "BETA")):
        sub = root / name
        sub.mkdir()
        (sub / "__init__.py").write_text(
            f"from .thing import {symbol}\n\n__all__ = [{symbol!r}]\n",
            encoding="utf-8",
        )
        (sub / "thing.py").write_text(f'{symbol} = "{name}"\n', encoding="utf-8")

    # alpha's `thing` has to be loaded before beta's __init__ runs, or there is
    # nothing stale in the cache for beta to collide with.
    result = _run(
        "fixture_pkg/__init__.py",
        "fixture_pkg/alpha/__init__.py",
        "fixture_pkg/alpha/thing.py",
        "fixture_pkg/beta/__init__.py",
        "fixture_pkg/beta/thing.py",
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_whole_package_imports_in_a_single_pass() -> None:
    """The real package must survive the same treatment, not just a fixture."""
    modules = sorted(str(p.relative_to(REPO_ROOT)) for p in PACKAGE.rglob("*.py"))
    assert modules, "expected package modules to check"

    result = _run(*modules)

    assert result.returncode == 0, result.stdout + result.stderr


def test_loading_a_module_raises_no_deprecation_warning(tmp_path: Path) -> None:
    """The loader API must be one that survives Python 3.15."""
    module = tmp_path / "plain.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")

    result = _run(str(module), warnings_as_errors=True)

    assert result.returncode == 0, result.stdout + result.stderr


# ── the guard: it must still fail on a genuinely broken module ───────────────


def test_unimportable_module_is_reported_and_exits_non_zero(tmp_path: Path) -> None:
    broken = tmp_path / "broken.py"
    broken.write_text("import a_module_that_does_not_exist\n", encoding="utf-8")

    result = _run(str(broken))

    assert result.returncode == 1
    assert "broken.py" in result.stdout
    assert "ModuleNotFoundError" in result.stderr
