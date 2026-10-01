"""Pin each runtime dependency to the lower bound pyproject.toml declares.

CI installs these pins over the locked test environment and runs the unit
suite, so a floor that cannot install or import together with the others fails
there instead of on a user's machine. ``--check`` then confirms the environment
really holds the pins: an install that missed it would test the lock and pass.
"""

import argparse
import sys
import tomllib
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version


def floors(pyproject: str) -> list[str]:
    """``name==lower`` for each ``project.dependencies`` entry, markers kept."""
    pins = []
    for entry in tomllib.loads(pyproject)["project"]["dependencies"]:
        requirement = Requirement(entry)
        lower = [
            spec.version for spec in requirement.specifier if spec.operator == ">="
        ]
        if len(lower) != 1:
            msg = f"{entry!r} needs exactly one '>=' lower bound for CI to test"
            raise ValueError(msg)
        marker = f"; {requirement.marker}" if requirement.marker else ""
        pins.append(f"{requirement.name}=={lower[0]}{marker}")
    return pins


def mismatches(pins: list[str], installed: Callable[[str], str]) -> list[str]:
    """Each pin that applies here and that the environment does not hold."""
    found = []
    for pin in pins:
        requirement = Requirement(pin)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        wanted = next(iter(requirement.specifier)).version
        have = installed(requirement.name)
        if Version(have) != Version(wanted):
            found.append(
                f"{requirement.name} {have} is installed, the floor is {wanted}"
            )
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pyproject", nargs="?", default="pyproject.toml", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail unless the environment holds the pins",
    )
    args = parser.parse_args()
    pins = floors(args.pyproject.read_text())
    if not args.check:
        print("\n".join(pins))
        return 0
    found = mismatches(pins, version)
    for line in found:
        print(line, file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
