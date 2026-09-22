"""Test the declared runtime dependencies."""

# stdlib
import ast
import re
import sys
from importlib.metadata import packages_distributions
from pathlib import Path

# third-party
import toml

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "langchain_litellm"


def _canonical(name: str) -> str:
    """Normalise a distribution name the way PEP 503 does, so spellings compare."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _runtime_dependency_names() -> list[str]:
    """Distribution names from [project].dependencies, without their specifiers."""
    pyproject = toml.load(REPO_ROOT / "pyproject.toml")
    names = []
    for spec in pyproject["project"]["dependencies"]:
        match = re.match(r"[A-Za-z0-9._-]+", spec)
        if match is None:
            raise ValueError(f"cannot read a distribution name from {spec!r}")
        names.append(match.group(0))
    return names


def _modules_by_distribution() -> dict[str, set[str]]:
    """Top-level modules each installed distribution provides, from its metadata.

    A distribution name is not its import name: PyYAML imports as ``yaml``,
    python-dateutil as ``dateutil``, and one distribution may install several
    top-level modules. Invert the installed metadata rather than guess.
    """
    modules: dict[str, set[str]] = {}
    for module, distributions in packages_distributions().items():
        for distribution in distributions:
            modules.setdefault(_canonical(distribution), set()).add(module)
    return modules


def _imported_modules() -> set[str]:
    """Top-level modules the package imports, parsed rather than matched.

    Searching the source as text would count a name inside a docstring example
    or a comment, and would miss every name after the first in ``import a, b``.
    Relative imports are skipped: they are the package's own modules, not
    dependencies.
    """
    modules: set[str] = set()
    for path in PACKAGE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                modules.add(node.module.split(".")[0])
    return modules


def test_every_runtime_dependency_is_imported_by_the_package() -> None:
    """A runtime dependency nothing imports still constrains every downstream user.

    It offers them nothing in return, and its upper bound forces them to counter-pin.
    """
    imported = _imported_modules()
    installed = _modules_by_distribution()

    unused = []
    for name in _runtime_dependency_names():
        # A distribution absent from this environment has no metadata to invert,
        # so fall back to its name. That is only a guess, but a wrong guess in a
        # half-installed environment is better than reporting a used dependency
        # as unused.
        modules = installed.get(_canonical(name)) or {name.replace("-", "_")}
        if not modules & imported:
            unused.append(name)

    assert not unused, (
        f"declared in [project].dependencies but never imported: {unused}"
    )


def test_every_imported_distribution_is_declared() -> None:
    """A third-party import that nothing declares installs only by luck.

    It resolves today because another dependency happens to pull it in, so the
    day that dependency drops it, this package breaks with no gate having fired.
    """
    declared = {_canonical(name) for name in _runtime_dependency_names()}
    installed = _modules_by_distribution()
    stdlib = set(sys.stdlib_module_names)

    undeclared = set()
    for module in _imported_modules():
        if module in stdlib or module == PACKAGE.name:
            continue
        providers = {
            distribution
            for distribution, modules in installed.items()
            if module in modules
        }
        if providers and not providers & declared:
            undeclared.add(f"{module} (from {', '.join(sorted(providers))})")

    assert not undeclared, (
        f"imported by the package but not in [project].dependencies: "
        f"{sorted(undeclared)}"
    )
