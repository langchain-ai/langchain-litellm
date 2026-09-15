"""Test the check_imports script."""

# stdlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_imports.py"
PACKAGE = REPO_ROOT / "langchain_litellm"


def _run(*paths: str, warnings_as_errors: bool = False) -> subprocess.CompletedProcess:
    argv = [sys.executable]
    if warnings_as_errors:
        argv += ["-W", "error::DeprecationWarning"]
    return subprocess.run(  # noqa: S603
        [*argv, str(SCRIPT), *paths],
        capture_output=True,
        cwd=REPO_ROOT,
        text=True,
        check=False,
    )


# ── the contract: every shipped module imports cleanly ───────────────────────


def test_whole_package_imports_in_a_single_pass() -> None:
    """Each file must be checked against itself, not against a previous one.

    Loading every file under one module name makes a relative import resolve
    against whichever file was loaded last, so `embeddings/__init__.py` looks
    for its symbols in `chat_models/litellm.py`.
    """
    modules = sorted(str(p.relative_to(REPO_ROOT)) for p in PACKAGE.rglob("*.py"))
    assert modules, "expected package modules to check"

    result = _run(*modules)

    assert result.returncode == 0, result.stdout + result.stderr


def test_loading_a_module_raises_no_deprecation_warning(tmp_path: Path) -> None:
    """The loader API must be one that survives Python 3.15."""
    module = tmp_path / "plain.py"
    module.write_text("VALUE = 1\n")

    result = _run(str(module), warnings_as_errors=True)

    assert result.returncode == 0, result.stdout + result.stderr


# ── the guard: it must still fail on a genuinely broken module ───────────────


def test_unimportable_module_is_reported_and_exits_non_zero(tmp_path: Path) -> None:
    broken = tmp_path / "broken.py"
    broken.write_text("import a_module_that_does_not_exist\n")

    result = _run(str(broken))

    assert result.returncode == 1
    assert "broken.py" in result.stdout
    assert "ModuleNotFoundError" in result.stderr
