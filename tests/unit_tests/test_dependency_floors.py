"""Test the pins CI installs to run the unit suite at the declared floors."""

# stdlib
import re
import tomllib
from pathlib import Path

# third-party
import pytest

# first-party
from dependency_floors import floors, mismatches

REPO_ROOT = Path(__file__).resolve().parents[2]


def _name(requirement: str) -> str:
    return re.split(r"[<>=!~;\[\s]", requirement, maxsplit=1)[0]


def _pyproject(*dependencies: str) -> str:
    listed = ", ".join(f'"{dependency}"' for dependency in dependencies)
    return f"[project]\ndependencies = [{listed}]\n"


def test_each_dependency_is_pinned_to_its_lower_bound() -> None:
    pyproject = _pyproject("litellm>=1.101.0,<2.0.0", "httpx>=0.28.1,<0.29.0")

    assert floors(pyproject) == ["litellm==1.101.0", "httpx==0.28.1"]


def test_an_exclusion_does_not_move_the_floor() -> None:
    assert floors(_pyproject("litellm>=1.83.14,<2.0.0,!=1.82.7")) == [
        "litellm==1.83.14"
    ]


def test_a_marker_stays_with_its_pin() -> None:
    """pip then skips the pin wherever the dependency does not apply."""
    pyproject = _pyproject("cffi>=2.0.0; python_version >= '3.10'")

    assert floors(pyproject) == ['cffi==2.0.0; python_version >= "3.10"']


@pytest.mark.parametrize("dependency", ["httpx<0.29.0", "httpx", "httpx==0.28.1"])
def test_a_dependency_without_a_lower_bound_is_refused(dependency: str) -> None:
    """CI cannot test a floor the package never declared."""
    with pytest.raises(ValueError, match="lower bound"):
        floors(_pyproject(dependency))


def test_an_environment_still_on_the_lock_is_reported() -> None:
    """An install that missed the environment would test the lock and pass."""
    installed = {"litellm": "1.102.1", "httpx": "0.28.1"}

    assert mismatches(["litellm==1.101.0", "httpx==0.28.1"], installed.__getitem__) == [
        "litellm 1.102.1 is installed, the floor is 1.101.0"
    ]


def test_a_pin_whose_marker_does_not_apply_is_not_checked() -> None:
    def version(name: str) -> str:
        raise AssertionError(name)

    assert mismatches(['cffi==2.0.0; python_version < "3.0"'], version) == []


def test_every_runtime_dependency_of_this_package_gets_a_pin() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    declared = tomllib.loads(pyproject)["project"]["dependencies"]

    pins = floors(pyproject)

    # One pin per declared entry, in order, so a marker split pins both halves.
    assert [_name(pin) for pin in pins] == [_name(entry) for entry in declared]
