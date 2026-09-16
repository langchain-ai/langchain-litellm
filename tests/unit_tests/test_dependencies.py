"""Test the declared runtime dependencies."""

# stdlib
import re
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


def test_every_runtime_dependency_is_imported_by_the_package() -> None:
    """A runtime dependency nothing imports still constrains every downstream user.

    It offers them nothing in return, and its upper bound forces them to counter-pin.
    """
    sources = "\n".join(
        path.read_text(encoding="utf-8") for path in PACKAGE.rglob("*.py")
    )
    installed = _modules_by_distribution()

    unused = []
    for name in _runtime_dependency_names():
        # A distribution absent from this environment has no metadata to invert,
        # so fall back to its name. That is only a guess, but a wrong guess in a
        # half-installed environment is better than reporting a used dependency
        # as unused.
        modules = installed.get(_canonical(name)) or {name.replace("-", "_")}
        if not any(
            re.search(
                rf"^\s*(?:import|from)\s+{re.escape(module)}\b",
                sources,
                re.MULTILINE,
            )
            for module in modules
        ):
            unused.append(name)

    assert not unused, (
        f"declared in [project].dependencies but never imported: {unused}"
    )
