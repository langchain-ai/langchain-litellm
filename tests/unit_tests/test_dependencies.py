"""Test the declared runtime dependencies."""

# stdlib
import re
from pathlib import Path

# third-party
import toml

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "langchain_litellm"


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


def test_every_runtime_dependency_is_imported_by_the_package() -> None:
    """A runtime dependency nothing imports still constrains every downstream user.

    It offers them nothing in return, and its upper bound forces them to counter-pin.
    """
    sources = "\n".join(path.read_text() for path in PACKAGE.rglob("*.py"))

    unused = [
        name
        for name in _runtime_dependency_names()
        if not re.search(
            rf"^\s*(?:import|from)\s+{re.escape(name.replace('-', '_'))}\b",
            sources,
            re.MULTILINE,
        )
    ]

    assert not unused, (
        f"declared in [project].dependencies but never imported: {unused}"
    )
