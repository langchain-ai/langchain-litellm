"""The supported Python versions, read from pyproject.toml's classifiers.

`--matrix` prints them as JSON for a CI matrix and `--list` as words for make.
`--check` fails unless `.python-version`, the lowest classifier and the
`requires-python` floor name the same version.
"""

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLASSIFIER = re.compile(r"Programming Language :: Python :: (3\.\d+)")
FLOOR = re.compile(r">=\s*(\d+\.\d+)")


def _key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def _project(root: Path) -> dict:
    return tomllib.loads((root / "pyproject.toml").read_text())["project"]


def supported(root: Path = ROOT) -> list[str]:
    """Every `Programming Language :: Python :: 3.X` classifier, lowest first."""
    found = (CLASSIFIER.fullmatch(c) for c in _project(root).get("classifiers", []))
    return sorted((m.group(1) for m in found if m), key=_key)


def problems(root: Path = ROOT) -> list[str]:
    """Where `.python-version`, the classifiers and `requires-python` disagree."""
    versions = supported(root)
    if not versions:
        return ["pyproject.toml lists no `Programming Language :: Python :: 3.X`"]
    lowest = versions[0]
    found = []
    pinned = (root / ".python-version").read_text().strip()
    if pinned != lowest:
        found.append(f".python-version is {pinned}, the lowest classifier {lowest}")
    requires = _project(root)["requires-python"]
    floor = FLOOR.search(requires)
    if floor is None or floor.group(1) != lowest:
        found.append(f"requires-python is {requires!r}, the lowest classifier {lowest}")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--matrix", action="store_true", help="print JSON for CI")
    mode.add_argument("--list", action="store_true", help="print words for make")
    mode.add_argument("--check", action="store_true", help="fail on disagreement")
    args = parser.parse_args(argv)
    if args.check:
        found = problems()
        for problem in found:
            print(problem, file=sys.stderr)
        return 1 if found else 0
    versions = supported()
    print(json.dumps(versions) if args.matrix else " ".join(versions))
    return 0


if __name__ == "__main__":
    sys.exit(main())
