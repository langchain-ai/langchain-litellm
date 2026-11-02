import json
from pathlib import Path

import pytest
from python_versions import main, problems, supported


def _write(
    root: Path,
    classifiers: list[str],
    requires: str = ">=3.11,<4.0",
    pinned: str = "3.11",
) -> Path:
    lines = "".join(f'    "{c}",\n' for c in classifiers)
    (root / "pyproject.toml").write_text(
        f'[project]\nrequires-python = "{requires}"\nclassifiers = [\n{lines}]\n'
    )
    (root / ".python-version").write_text(f"{pinned}\n")
    return root


def _py(*versions: str) -> list[str]:
    return [f"Programming Language :: Python :: {v}" for v in versions]


def test_supported_sorts_numerically_and_skips_other_classifiers(
    tmp_path: Path,
) -> None:
    classifiers = [
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3",
        *_py("3.12", "3.10", "3.9"),
    ]
    _write(tmp_path, classifiers)
    assert supported(tmp_path) == ["3.9", "3.10", "3.12"]


def test_problems_empty_when_every_source_agrees(tmp_path: Path) -> None:
    _write(tmp_path, _py("3.11", "3.12"))
    assert problems(tmp_path) == []


def test_problems_flags_python_version_off_the_lowest_classifier(
    tmp_path: Path,
) -> None:
    _write(tmp_path, _py("3.11", "3.12"), pinned="3.12")
    assert problems(tmp_path) == [".python-version is 3.12, the lowest classifier 3.11"]


def test_problems_flags_requires_python_off_the_lowest_classifier(
    tmp_path: Path,
) -> None:
    _write(tmp_path, _py("3.12", "3.13"), requires=">=3.11", pinned="3.12")
    assert problems(tmp_path) == [
        "requires-python is '>=3.11', the lowest classifier 3.12"
    ]


def test_problems_flags_a_missing_version_classifier(tmp_path: Path) -> None:
    _write(tmp_path, ["Programming Language :: Python :: 3"])
    assert problems(tmp_path) == [
        "pyproject.toml lists no `Programming Language :: Python :: 3.X`"
    ]


@pytest.mark.parametrize("flag", ["--matrix", "--list"])
def test_main_prints_the_repo_versions_in_both_formats(
    flag: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([flag]) == 0
    out = capsys.readouterr().out.strip()
    printed = json.loads(out) if flag == "--matrix" else out.split()
    assert printed == supported()
